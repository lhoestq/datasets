import os
import re
from functools import partial
from glob import has_magic
from pathlib import Path, PurePath
from typing import Callable, Iterable, Optional, Union

import huggingface_hub
from fsspec.core import url_to_fs
from huggingface_hub import HfFileSystem


try:
    # `glob_translate` is what `fsspec` uses to translate glob patterns to regex patterns.
    # It may be missing from old `fsspec` versions, in which case patterns are resolved with `fs.glob`.
    from fsspec.utils import glob_translate as _glob_translate
except ImportError:
    _glob_translate = None
from packaging import version
from tqdm.contrib.concurrent import thread_map

from . import config
from .download import DownloadConfig
from .naming import _split_re
from .splits import Split
from .utils import logging
from .utils import tqdm as hf_tqdm
from .utils.file_utils import _prepare_path_and_storage_options, is_local_path, is_relative_path, xbasename, xjoin
from .utils.py_utils import string_to_dict


SingleOriginMetadata = Union[tuple[str, str], tuple[str], tuple[()]]


SANITIZED_DEFAULT_SPLIT = str(Split.TRAIN)


logger = logging.get_logger(__name__)


class Url(str):
    pass


class EmptyDatasetError(FileNotFoundError):
    pass


SPLIT_PATTERN_SHARDED = "data/{split}-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9]*.*"

SPLIT_KEYWORDS = {
    Split.TRAIN: ["train", "training"],
    Split.VALIDATION: ["validation", "valid", "dev", "val"],
    Split.TEST: ["test", "testing", "eval", "evaluation"],
}
NON_WORDS_CHARS = "-._ 0-9"
if config.FSSPEC_VERSION < version.parse("2023.9.0"):
    KEYWORDS_IN_FILENAME_BASE_PATTERNS = ["**[{sep}/]{keyword}[{sep}]*", "{keyword}[{sep}]*"]
    KEYWORDS_IN_DIR_NAME_BASE_PATTERNS = [
        "{keyword}/**",
        "{keyword}[{sep}]*/**",
        "**[{sep}/]{keyword}/**",
        "**[{sep}/]{keyword}[{sep}]*/**",
    ]
elif config.FSSPEC_VERSION < version.parse("2023.12.0"):
    KEYWORDS_IN_FILENAME_BASE_PATTERNS = ["**/*[{sep}/]{keyword}[{sep}]*", "{keyword}[{sep}]*"]
    KEYWORDS_IN_DIR_NAME_BASE_PATTERNS = [
        "{keyword}/**/*",
        "{keyword}[{sep}]*/**/*",
        "**/*[{sep}/]{keyword}/**/*",
        "**/*[{sep}/]{keyword}[{sep}]*/**/*",
    ]
else:
    KEYWORDS_IN_FILENAME_BASE_PATTERNS = ["**/{keyword}[{sep}]*", "**/*[{sep}]{keyword}[{sep}]*"]
    KEYWORDS_IN_DIR_NAME_BASE_PATTERNS = [
        "**/{keyword}/**",
        "**/{keyword}[{sep}]*/**",
        "**/*[{sep}]{keyword}/**",
        "**/*[{sep}]{keyword}[{sep}]*/**",
    ]

DEFAULT_SPLITS = [Split.TRAIN, Split.VALIDATION, Split.TEST]
DEFAULT_PATTERNS_SPLIT_IN_FILENAME = {
    split: [
        pattern.format(keyword=keyword, sep=NON_WORDS_CHARS)
        for keyword in SPLIT_KEYWORDS[split]
        for pattern in KEYWORDS_IN_FILENAME_BASE_PATTERNS
    ]
    for split in DEFAULT_SPLITS
}
DEFAULT_PATTERNS_SPLIT_IN_DIR_NAME = {
    split: [
        pattern.format(keyword=keyword, sep=NON_WORDS_CHARS)
        for keyword in SPLIT_KEYWORDS[split]
        for pattern in KEYWORDS_IN_DIR_NAME_BASE_PATTERNS
    ]
    for split in DEFAULT_SPLITS
}


DEFAULT_PATTERNS_ALL = {
    Split.TRAIN: ["**"],
}

DEFAULT_PATTERNS_LOGS = {"logs": ["**/*.eval"]}
DEFAULT_PATTERNS_HARBOR = {"test": ["**/task.toml", "**/instruction.md"]}

ALL_SPLIT_PATTERNS = [SPLIT_PATTERN_SHARDED]
ALL_DEFAULT_PATTERNS = [
    DEFAULT_PATTERNS_LOGS,
    DEFAULT_PATTERNS_HARBOR,
    DEFAULT_PATTERNS_SPLIT_IN_DIR_NAME,
    DEFAULT_PATTERNS_SPLIT_IN_FILENAME,
    DEFAULT_PATTERNS_ALL,
]
WILDCARD_CHARACTERS = "*[]"
FILES_TO_IGNORE = [
    "README.md",
    "config.json",
    "dataset_info.json",
    "dataset_infos.json",
    "dummy_data.zip",
    "dataset_dict.json",
]


def contains_wildcards(pattern: str) -> bool:
    return any(wildcard_character in pattern for wildcard_character in WILDCARD_CHARACTERS)


def sanitize_patterns(patterns: Union[dict, list, str]) -> dict[str, Union[list[str], "DataFilesList"]]:
    """
    Take the data_files patterns from the user, and format them into a dictionary.
    Each key is the name of the split, and each value is a list of data files patterns (paths or urls).
    The default split is "train".

    Returns:
        patterns: dictionary of split_name -> list of patterns
    """
    if isinstance(patterns, dict):
        return {str(key): value if isinstance(value, list) else [value] for key, value in patterns.items()}
    elif isinstance(patterns, str):
        return {SANITIZED_DEFAULT_SPLIT: [patterns]}
    elif isinstance(patterns, list):
        if any(isinstance(pattern, dict) for pattern in patterns):
            for pattern in patterns:
                if not (
                    isinstance(pattern, dict)
                    and len(pattern) == 2
                    and "split" in pattern
                    and isinstance(pattern.get("path"), (str, list))
                ):
                    raise ValueError(
                        "Invalid format for data_files entry. "
                        "Each item must be a dictionary with the structure "
                        "{'split': <split_name>, 'path': <path_or_list_of_paths>}.\n"
                        f"Received: {pattern}"
                    )
            splits = [pattern["split"] for pattern in patterns]
            if len(set(splits)) != len(splits):
                raise ValueError(f"Some splits are duplicated in data_files: {splits}")
            return {
                str(pattern["split"]): pattern["path"] if isinstance(pattern["path"], list) else [pattern["path"]]
                for pattern in patterns
            }
        else:
            return {SANITIZED_DEFAULT_SPLIT: patterns}
    else:
        return sanitize_patterns(list(patterns))


def _is_inside_unrequested_special_dir(matched_rel_path: str, pattern: str) -> bool:
    """
    When a path matches a pattern, we additionally check if it's inside a special directory
    we ignore by default (if it starts with a double underscore).

    Users can still explicitly request a filepath inside such a directory if "__pycache__" is
    mentioned explicitly in the requested pattern.

    Some examples:

    base directory:

        ./
        └── __pycache__
            └── b.txt

    >>> _is_inside_unrequested_special_dir("__pycache__/b.txt", "**")
    True
    >>> _is_inside_unrequested_special_dir("__pycache__/b.txt", "*/b.txt")
    True
    >>> _is_inside_unrequested_special_dir("__pycache__/b.txt", "__pycache__/*")
    False
    >>> _is_inside_unrequested_special_dir("__pycache__/b.txt", "__*/*")
    False
    """
    # We just need to check if every special directory from the path is present explicitly in the pattern.
    # Since we assume that the path matches the pattern, it's equivalent to counting that both
    # the parent path and the parent pattern have the same number of special directories.
    data_dirs_to_ignore_in_path = _num_special_dirs(PurePath(matched_rel_path).parent.parts)
    data_dirs_to_ignore_in_pattern = _num_special_dirs(PurePath(pattern).parent.parts)
    return data_dirs_to_ignore_in_path != data_dirs_to_ignore_in_pattern


def _is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir(matched_rel_path: str, pattern: str) -> bool:
    """
    When a path matches a pattern, we additionally check if it's a hidden file or if it's inside
    a hidden directory we ignore by default, i.e. if the file name or a parent directory name starts with a dot.

    Users can still explicitly request a filepath that is hidden or is inside a hidden directory
    if the hidden part is mentioned explicitly in the requested pattern.

    Some examples:

    base directory:

        ./
        └── .hidden_file.txt

    >>> _is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir(".hidden_file.txt", "**")
    True
    >>> _is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir(".hidden_file.txt", ".*")
    False

    base directory:

        ./
        └── .hidden_dir
            └── a.txt

    >>> _is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir(".hidden_dir/a.txt", "**")
    True
    >>> _is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir(".hidden_dir/a.txt", ".*/*")
    False
    >>> _is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir(".hidden_dir/a.txt", ".hidden_dir/*")
    False

    base directory:

        ./
        └── .hidden_dir
            └── .hidden_file.txt

    >>> _is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir(".hidden_dir/.hidden_file.txt", "**")
    True
    >>> _is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir(".hidden_dir/.hidden_file.txt", ".*/*")
    True
    >>> _is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir(".hidden_dir/.hidden_file.txt", ".*/.*")
    False
    >>> _is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir(".hidden_dir/.hidden_file.txt", ".hidden_dir/*")
    True
    >>> _is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir(".hidden_dir/.hidden_file.txt", ".hidden_dir/.*")
    False
    """
    # We just need to check if every hidden part from the path is present explicitly in the pattern.
    # Since we assume that the path matches the pattern, it's equivalent to counting that both
    # the path and the pattern have the same number of hidden parts.
    hidden_directories_in_path = _num_hidden_parts(PurePath(matched_rel_path).parts)
    hidden_directories_in_pattern = _num_hidden_parts(PurePath(pattern).parts)
    return hidden_directories_in_path != hidden_directories_in_pattern


def _num_special_dirs(parts: Iterable[str]) -> int:
    """Number of special directories in `parts`, i.e. of directories whose name starts with "__" like "__pycache__"."""
    return sum(1 for part in parts if part.startswith("__"))


def _num_hidden_parts(parts: Iterable[str]) -> int:
    """Number of hidden parts in `parts`, i.e. of parts that start with "." and that are not made only of dots."""
    return sum(1 for part in parts if part.startswith(".") and not set(part) == {"."})


def _path_parts(path: str) -> list[str]:
    """
    Same as `PurePath(path).parts` for the posix-like paths fsspec works with, but much faster.

    It is used instead of `PurePath` since it is called once per resolved file, and a repository can contain
    hundreds of thousands of files.
    """
    return [part for part in path.split("/") if part not in ("", ".")]


def _get_data_files_patterns(pattern_resolver: Callable[[str], list[str]]) -> dict[str, list[str]]:
    """
    Get the default pattern from a directory or repository by testing all the supported patterns.
    The first patterns to return a non-empty list of data files is returned.

    In order, it first tests if SPLIT_PATTERN_SHARDED works, otherwise it tests the patterns in ALL_DEFAULT_PATTERNS.
    """
    # first check the split patterns like data/{split}-00000-of-00001.parquet
    for split_pattern in ALL_SPLIT_PATTERNS:
        pattern = split_pattern.replace("{split}", "*")
        try:
            data_files = pattern_resolver(pattern)
        except FileNotFoundError:
            continue
        if len(data_files) > 0:
            splits: set[str] = set()
            for p in data_files:
                p_parts = string_to_dict(xbasename(p), xbasename(split_pattern))
                assert p_parts is not None
                splits.add(p_parts["split"])

            if any(not re.match(_split_re, split) for split in splits):
                raise ValueError(f"Split name should match '{_split_re}'' but got '{splits}'.")
            sorted_splits = [str(split) for split in DEFAULT_SPLITS if split in splits] + sorted(
                splits - {str(split) for split in DEFAULT_SPLITS}
            )
            return {split: [split_pattern.format(split=split)] for split in sorted_splits}
    # then check the default patterns based on train/valid/test splits
    for patterns_dict in ALL_DEFAULT_PATTERNS:
        non_empty_splits = []
        for split, patterns in patterns_dict.items():
            for pattern in patterns:
                try:
                    data_files = pattern_resolver(pattern)
                except FileNotFoundError:
                    continue
                if len(data_files) > 0:
                    non_empty_splits.append(split)
                    break
        if non_empty_splits:
            return {split: patterns_dict[split] for split in non_empty_splits}
    raise FileNotFoundError(f"Couldn't resolve pattern {pattern} with resolver {pattern_resolver}")


# Protocols for which the listing of the root of a pattern is reused when resolving several patterns one after the
# other (see `_glob_matches`). Only `hf://` is enabled by default since `HfFileSystem` already caches the listings it
# does, so that the behavior of the other remote filesystems (that list the data again at every call) doesn't change.
_REUSE_LISTING_PROTOCOLS = ("hf",)

# Attribute of a filesystem object used to store its last listing (see `_find_with_cached_listing`).
_LISTING_CACHE_ATTR = "_datasets_last_listing"

# Wildcard tokens of a glob pattern: "*" (and "**"), "?" and "[...]" character classes.
_WILDCARD_TOKENS_RE = re.compile(r"\*+|\?|\[[^\]]*\]")


def _required_literals(pattern: str) -> tuple[str, ...]:
    """
    The literal parts of the glob `pattern` that any matching path must contain, e.g. ("train",) for "**/*train*".

    They are used to skip the paths of a listing that cannot match the pattern before matching them against its
    regex (see `_glob_matches`), since matching a regex with "**" in it is orders of magnitude slower than
    checking that a string contains a literal.

    The "/" at the edges of a literal are dropped since a "**" in the pattern can match them: "**/*train*" matches
    both "train.txt" and "data/train.txt", so only "train" is required in the path.
    """
    return tuple(token for token in (part.strip("/") for part in _WILDCARD_TOKENS_RE.split(pattern)) if token)


def _filter_by_literals(listing: list[tuple[str, dict]], literals: tuple[str, ...]) -> list[tuple[str, dict]]:
    """Keep the `(path, info)` pairs of `listing` whose path contains every one of `literals`."""
    for literal in literals:
        listing = [item for item in listing if literal in item[0]]
    return listing


def _find_with_cached_listing(
    fs, root: str, maxdepth: Optional[int], withdirs: bool, **kwargs
) -> list[tuple[str, dict]]:
    """
    Same as `sorted(fs.find(root, maxdepth=maxdepth, withdirs=withdirs, detail=True, **kwargs).items())`, but the
    result is reused by the next call with the same arguments on the same filesystem object.

    Listing a large remote repository takes tens of seconds (the Hugging Face Hub serves 1000 entries per HTTP
    request, and a repository can contain hundreds of thousands of files). Since resolving the data files of a
    repository lists the same root again for every pattern (see `_glob_matches` and `_get_data_files_patterns`),
    the listing of the last root is kept on the filesystem object to be reused.

    Only one listing is kept at a time, so that the memory it uses doesn't grow with the number of resolved roots.
    Like the cache of `HfFileSystem`, this makes the result of a listing independent from the changes made to the
    repository since the previous listing with the same arguments.
    """
    key = (root, maxdepth, withdirs, tuple(sorted(kwargs.items())))
    cached_listing = getattr(fs, _LISTING_CACHE_ATTR, None)
    if cached_listing is not None and cached_listing[0] == key:
        return cached_listing[1]
    listing = sorted(fs.find(root, maxdepth=maxdepth, withdirs=withdirs, detail=True, **kwargs).items())
    setattr(fs, _LISTING_CACHE_ATTR, (key, listing))
    return listing


def _glob_matches(fs, path: str, **kwargs) -> list[tuple[str, dict]]:
    """
    Same as `fs.glob(path, detail=True, **kwargs).items()`, but reusing the listing of the root of `path` between
    calls (see `_find_with_cached_listing`).

    The algorithm is the one of `fsspec.spec.AbstractFileSystem.glob`: the root of the pattern is listed, then every
    path of the listing is matched against the regex of the pattern. The two differences are that:

    - the listing is done by `_find_with_cached_listing`, so that resolving several patterns that share the same
      root lists it only once
    - the paths that don't contain a literal of the pattern are skipped before being matched against its regex, for
      the same result at a fraction of the cost (see `_required_literals`)

    It is used to resolve the data files patterns of a remote repository (see `resolve_pattern`): auto-inferring the
    patterns of a dataset resolves dozens of patterns that all share the same root, and listing a large repository
    again for each of them takes tens of seconds.

    Returns:
        The sorted list of `(path, info)` pairs that match `path`, like `fs.glob(path, detail=True).items()`.
    """
    if _glob_translate is None:
        # `glob_translate` is not available in this version of `fsspec`: stick to `fs.glob`
        return list(fs.glob(path, detail=True, **kwargs).items())
    if isinstance(fs, HfFileSystem):
        # Same as `HfFileSystem.glob`: the path is resolved before being listed, so that e.g. a repository that
        # doesn't exist raises an error mentioning the whole pattern, like `fs.glob` does.
        path = fs.resolve_path(path, revision=kwargs.get("revision")).unresolve()
    seps = (os.path.sep, os.path.altsep) if os.path.altsep else (os.path.sep,)
    ends_with_sep = path.endswith(seps)
    path = fs._strip_protocol(path)
    append_slash_to_dirname = ends_with_sep or path.endswith(tuple(sep + "**" for sep in seps))
    first_magic_idx = min(path.find(char) if char in path else len(path) for char in ("*", "?", "["))
    if not has_magic(path):
        return [(path, fs.info(path, **kwargs))] if fs.exists(path, **kwargs) else []
    # Split off the part of the pattern that is a root directory, and count the depth of the remaining part: only
    # the files up to that depth have to be listed (like `fs.glob` does).
    if "/" in path[:first_magic_idx]:
        first_magic_idx = path[:first_magic_idx].rindex("/")
        root = path[: first_magic_idx + 1]
    else:
        root = ""
    depth = path[first_magic_idx + 1 :].count("/") + 1
    if "**" in path:
        depth = None
    pattern = re.compile(_glob_translate(path + ("/" if ends_with_sep else "")))
    listing = _filter_by_literals(
        _find_with_cached_listing(fs, root, maxdepth=depth, withdirs=True, **kwargs), _required_literals(path)
    )
    return [
        (matched_path, info)
        for matched_path, info in listing
        if pattern.match(
            matched_path + "/" if append_slash_to_dirname and info["type"] == "directory" else matched_path
        )
    ]


def resolve_pattern(
    pattern: str,
    base_path: str,
    allowed_extensions: Optional[list[str]] = None,
    download_config: Optional[DownloadConfig] = None,
) -> list[str]:
    """
    Resolve the paths and URLs of the data files from the pattern passed by the user.

    You can use patterns to resolve multiple local files. Here are a few examples:
    - *.csv to match all the CSV files at the first level
    - **.csv to match all the CSV files at any level
    - data/* to match all the files inside "data"
    - data/** to match all the files inside "data" and its subdirectories

    The patterns are resolved using the fsspec glob. In fsspec>=2023.12.0 this is equivalent to
    Python's glob.glob, Path.glob, Path.match and fnmatch where ** is unsupported with a prefix/suffix
    other than a forward slash /.

    More generally:
    - '*' matches any character except a forward-slash (to match just the file or directory name)
    - '**' matches any character including a forward-slash /

    Hidden files and directories (i.e. whose names start with a dot) are ignored, unless they are explicitly requested.
    The same applies to special directories that start with a double underscore like "__pycache__".
    You can still include one if the pattern explicitly mentions it:
    - to include a hidden file: "*/.hidden.txt" or "*/.*"
    - to include a hidden directory: ".hidden/*" or ".*/*"
    - to include a special directory: "__special__/*" or "__*/*"

    Example::

        >>> from datasets.data_files import resolve_pattern
        >>> base_path = "."
        >>> resolve_pattern("docs/**/*.py", base_path)
        [/Users/mariosasko/Desktop/projects/datasets/docs/source/_config.py']

    Args:
        pattern (str): Unix pattern or paths or URLs of the data files to resolve.
            The paths can be absolute or relative to base_path.
            Remote filesystems using fsspec are supported, e.g. with the hf:// protocol.
        base_path (str): Base path to use when resolving relative paths.
        allowed_extensions (Optional[list], optional): White-list of file extensions to use. Defaults to None (all extensions).
            For example: allowed_extensions=[".csv", ".json", ".txt", ".parquet"]
        download_config ([`DownloadConfig`], *optional*): Specific download configuration parameters.
    Returns:
        List[str]: List of paths or URLs to the local or remote files that match the patterns.
    """
    if is_relative_path(pattern):
        pattern = xjoin(base_path, pattern)
    elif is_local_path(pattern):
        base_path = os.path.splitdrive(pattern)[0] + os.sep
    else:
        base_path = ""
    pattern, storage_options = _prepare_path_and_storage_options(pattern, download_config=download_config)
    fs, fs_pattern = url_to_fs(pattern, **storage_options)
    files_to_ignore = set(FILES_TO_IGNORE) - {xbasename(pattern)}
    protocol = (
        pattern.split("://")[0]
        if "://" in pattern
        else (fs.protocol if isinstance(fs.protocol, str) else fs.protocol[0])
    )
    protocol_prefix = protocol + "://" if protocol != "file" else ""
    glob_kwargs = {}
    if protocol == "hf":
        # 10 times faster glob with detail=True (ignores costly info like lastCommit)
        glob_kwargs["expand_info"] = False

    # if the pattern contains hops like "zip://csv/*.csv::data.zip", we need to keep them after globbing
    _, *rest_hops = pattern.split("::")
    matched_paths = []
    if protocol in _REUSE_LISTING_PROTOCOLS and _glob_translate is not None and not rest_hops:
        # Reuse the listing of the root of the pattern: the patterns of a same resolution share the same root, and
        # listing a large remote repository again for each pattern takes minutes (see `_glob_matches`).
        matches = _glob_matches(fs, fs_pattern, **glob_kwargs)
    else:
        matches = list(fs.glob(fs_pattern, detail=True, **glob_kwargs).items())
    # The two checks below ignore the files in a special or hidden directory that the pattern doesn't request
    # explicitly: they do the same as `_is_inside_unrequested_special_dir` and
    # `_is_unrequested_hidden_file_or_is_inside_unrequested_hidden_dir`, but with the parts of the pattern computed
    # once for all the matched files, and without `PurePath` (see `_path_parts`).
    num_special_dirs_in_pattern = _num_special_dirs(_path_parts(fs_pattern)[:-1])
    num_hidden_parts_in_pattern = _num_hidden_parts(_path_parts(fs_pattern))
    for filepath, info in matches:
        if not (info["type"] == "file" or (info.get("islink") and os.path.isfile(os.path.realpath(filepath)))) or (
            xbasename(filepath) in files_to_ignore
        ):
            continue
        path_parts = _path_parts(filepath)
        if _num_special_dirs(path_parts[:-1]) != num_special_dirs_in_pattern:
            continue
        if _num_hidden_parts(path_parts) != num_hidden_parts_in_pattern:
            continue
        filepath = filepath if "://" in filepath else protocol_prefix + filepath
        if rest_hops:
            filepath = "::".join([filepath] + rest_hops)
        matched_paths.append(filepath)
    # ignore .ipynb and __pycache__, but keep /../
    if allowed_extensions is not None:
        out = [
            filepath
            for filepath in matched_paths
            if any("." + suffix in allowed_extensions for suffix in xbasename(filepath).split(".")[1:])
        ]
        if len(out) < len(matched_paths):
            invalid_matched_files = list(set(matched_paths) - set(out))
            logger.info(
                f"Some files matched the pattern '{pattern}' but don't have valid data file extensions: {invalid_matched_files}"
            )
    else:
        out = matched_paths
    if not out:
        error_msg = f"Unable to find '{pattern}'"
        if allowed_extensions is not None:
            error_msg += f" with any supported extension {list(allowed_extensions)}"
        raise FileNotFoundError(error_msg)
    return out


def get_data_patterns(base_path: str, download_config: Optional[DownloadConfig] = None) -> dict[str, list[str]]:
    """
    Get the default pattern from a directory testing all the supported patterns.
    The first patterns to return a non-empty list of data files is returned.

    Some examples of supported patterns:

    Input:

        my_dataset_repository/
        ├── README.md
        └── dataset.csv

    Output:

        {'train': ['**']}

    Input:

        my_dataset_repository/
        ├── README.md
        ├── train.csv
        └── test.csv

        my_dataset_repository/
        ├── README.md
        └── data/
            ├── train.csv
            └── test.csv

        my_dataset_repository/
        ├── README.md
        ├── train_0.csv
        ├── train_1.csv
        ├── train_2.csv
        ├── train_3.csv
        ├── test_0.csv
        └── test_1.csv

    Output:

        {'train': ['**/train[-._ 0-9]*', '**/*[-._ 0-9]train[-._ 0-9]*', '**/training[-._ 0-9]*', '**/*[-._ 0-9]training[-._ 0-9]*'],
         'test': ['**/test[-._ 0-9]*', '**/*[-._ 0-9]test[-._ 0-9]*', '**/testing[-._ 0-9]*', '**/*[-._ 0-9]testing[-._ 0-9]*', ...]}

    Input:

        my_dataset_repository/
        ├── README.md
        └── data/
            ├── train/
            │   ├── shard_0.csv
            │   ├── shard_1.csv
            │   ├── shard_2.csv
            │   └── shard_3.csv
            └── test/
                ├── shard_0.csv
                └── shard_1.csv

    Output:

        {'train': ['**/train/**', '**/train[-._ 0-9]*/**', '**/*[-._ 0-9]train/**', '**/*[-._ 0-9]train[-._ 0-9]*/**', ...],
         'test': ['**/test/**', '**/test[-._ 0-9]*/**', '**/*[-._ 0-9]test/**', '**/*[-._ 0-9]test[-._ 0-9]*/**', ...]}

    Input:

        my_dataset_repository/
        ├── README.md
        └── data/
            ├── train-00000-of-00003.csv
            ├── train-00001-of-00003.csv
            ├── train-00002-of-00003.csv
            ├── test-00000-of-00001.csv
            ├── random-00000-of-00003.csv
            ├── random-00001-of-00003.csv
            └── random-00002-of-00003.csv

    Output:

        {'train': ['data/train-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9]*.*'],
         'test': ['data/test-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9]*.*'],
         'random': ['data/random-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9]*.*']}

    In order, it first tests if SPLIT_PATTERN_SHARDED works, otherwise it tests the patterns in ALL_DEFAULT_PATTERNS.
    """
    resolver = partial(resolve_pattern, base_path=base_path, download_config=download_config)
    try:
        return _get_data_files_patterns(resolver)
    except FileNotFoundError:
        raise EmptyDatasetError(f"The directory at {base_path} doesn't contain any data files") from None


def _get_single_origin_metadata(
    data_file: str,
    download_config: Optional[DownloadConfig] = None,
) -> SingleOriginMetadata:
    if data_file.startswith(config.HF_ENDPOINT):
        fs = HfFileSystem(endpoint=config.HF_ENDPOINT, token=download_config.token)
        data_file = "hf://" + data_file[len(config.HF_ENDPOINT) + 1 :]
        data_file = data_file.replace("/resolve/", "/" if data_file.startswith("hf://buckets/") else "@", 1)
        fs_path = data_file
    else:
        data_file, storage_options = _prepare_path_and_storage_options(data_file, download_config=download_config)
        fs, fs_path = url_to_fs(data_file, **storage_options)
    if isinstance(fs, HfFileSystem):
        resolved_path = fs.resolve_path(fs_path)
        if hasattr(resolved_path, "revision"):  # no revision for buckets
            return resolved_path.repo_id, resolved_path.revision
    info = fs.info(fs_path)
    # s3fs uses "ETag", gcsfs uses "etag", and for local we simply check mtime
    for key in ["ETag", "etag", "mtime"]:
        if key in info:
            return (str(info[key]),)
    return ()


def _get_origin_metadata(
    data_files: list[str],
    download_config: Optional[DownloadConfig] = None,
    max_workers: Optional[int] = None,
) -> list[SingleOriginMetadata]:
    max_workers = max_workers if max_workers is not None else config.HF_DATASETS_MULTITHREADING_MAX_WORKERS
    if all("hf://" in data_file for data_file in data_files):
        # No need for multithreading here since the origin metadata of HF files
        # is (repo_id, revision) and is cached after first .info() call.
        return [
            _get_single_origin_metadata(data_file, download_config=download_config)
            for data_file in hf_tqdm(
                data_files,
                desc="Resolving data files",
                # set `disable=None` rather than `disable=False` by default to disable progress bar when no TTY attached
                disable=len(data_files) <= 16 or None,
            )
        ]
    return thread_map(
        partial(_get_single_origin_metadata, download_config=download_config),
        data_files,
        max_workers=max_workers,
        tqdm_class=hf_tqdm,
        desc="Resolving data files",
        # set `disable=None` rather than `disable=False` by default to disable progress bar when no TTY attached
        disable=len(data_files) <= 16 or None,
    )


class DataFilesList(list[str]):
    """
    List of data files (absolute local paths or URLs).
    It has two construction methods given the user's data files patterns:
    - ``from_hf_repo``: resolve patterns inside a dataset repository
    - ``from_local_or_remote``: resolve patterns from a local path

    Moreover, DataFilesList has an additional attribute ``origin_metadata``.
    It can store:
    - the last modified time of local files
    - ETag of remote files
    - commit sha of a dataset repository

    Thanks to this additional attribute, it is possible to hash the list
    and get a different hash if and only if at least one file changed.
    This is useful for caching Dataset objects that are obtained from a list of data files.
    """

    def __init__(self, data_files: list[str], origin_metadata: list[SingleOriginMetadata]) -> None:
        super().__init__(data_files)
        self.origin_metadata = origin_metadata

    def __add__(self, other: "DataFilesList") -> "DataFilesList":
        return DataFilesList([*self, *other], self.origin_metadata + other.origin_metadata)

    @classmethod
    def from_hf_repo(
        cls,
        patterns: list[str],
        dataset_info: huggingface_hub.hf_api.DatasetInfo,
        base_path: Optional[str] = None,
        allowed_extensions: Optional[list[str]] = None,
        download_config: Optional[DownloadConfig] = None,
    ) -> "DataFilesList":
        base_path = f"hf://datasets/{dataset_info.id}@{dataset_info.sha}/{base_path or ''}".rstrip("/")
        return cls.from_patterns(
            patterns, base_path=base_path, allowed_extensions=allowed_extensions, download_config=download_config
        )

    @classmethod
    def from_local_or_remote(
        cls,
        patterns: list[str],
        base_path: Optional[str] = None,
        allowed_extensions: Optional[list[str]] = None,
        download_config: Optional[DownloadConfig] = None,
    ) -> "DataFilesList":
        base_path = base_path if base_path is not None else Path().resolve().as_posix()
        return cls.from_patterns(
            patterns, base_path=base_path, allowed_extensions=allowed_extensions, download_config=download_config
        )

    @classmethod
    def from_patterns(
        cls,
        patterns: list[str],
        base_path: Optional[str] = None,
        allowed_extensions: Optional[list[str]] = None,
        download_config: Optional[DownloadConfig] = None,
    ) -> "DataFilesList":
        base_path = base_path if base_path is not None else Path().resolve().as_posix()
        data_files = []
        for pattern in patterns:
            try:
                data_files.extend(
                    resolve_pattern(
                        pattern,
                        base_path=base_path,
                        allowed_extensions=allowed_extensions,
                        download_config=download_config,
                    )
                )
            except FileNotFoundError:
                if not has_magic(pattern):
                    raise
        origin_metadata = _get_origin_metadata(data_files, download_config=download_config)
        return cls(data_files, origin_metadata)

    def filter(
        self, *, extensions: Optional[list[str]] = None, file_names: Optional[list[str]] = None
    ) -> "DataFilesList":
        patterns = []
        if extensions:
            ext_pattern = "|".join(re.escape(ext) for ext in extensions)
            patterns.append(re.compile(f".*({ext_pattern})(\\..+)?$"))
        if file_names:
            fn_pattern = "|".join(re.escape(fn) for fn in file_names)
            patterns.append(re.compile(rf".*[\/]?({fn_pattern})$"))
        if patterns:
            return DataFilesList(
                [data_file for data_file in self if any(pattern.match(data_file) for pattern in patterns)],
                origin_metadata=self.origin_metadata,
            )
        else:
            return DataFilesList(list(self), origin_metadata=self.origin_metadata)


class DataFilesDict(dict[str, DataFilesList]):
    """
    Dict of split_name -> list of data files (absolute local paths or URLs).
    It has two construction methods given the user's data files patterns :
    - ``from_hf_repo``: resolve patterns inside a dataset repository
    - ``from_local_or_remote``: resolve patterns from a local path

    Moreover, each list is a DataFilesList. It is possible to hash the dictionary
    and get a different hash if and only if at least one file changed.
    For more info, see [`DataFilesList`].

    This is useful for caching Dataset objects that are obtained from a list of data files.

    Changing the order of the keys of this dictionary also doesn't change its hash.
    """

    @classmethod
    def from_local_or_remote(
        cls,
        patterns: dict[str, Union[list[str], DataFilesList]],
        base_path: Optional[str] = None,
        allowed_extensions: Optional[list[str]] = None,
        download_config: Optional[DownloadConfig] = None,
    ) -> "DataFilesDict":
        out = cls()
        for key, patterns_for_key in patterns.items():
            out[key] = (
                patterns_for_key
                if isinstance(patterns_for_key, DataFilesList)
                else DataFilesList.from_local_or_remote(
                    patterns_for_key,
                    base_path=base_path,
                    allowed_extensions=allowed_extensions,
                    download_config=download_config,
                )
            )
        return out

    @classmethod
    def from_hf_repo(
        cls,
        patterns: dict[str, Union[list[str], DataFilesList]],
        dataset_info: huggingface_hub.hf_api.DatasetInfo,
        base_path: Optional[str] = None,
        allowed_extensions: Optional[list[str]] = None,
        download_config: Optional[DownloadConfig] = None,
    ) -> "DataFilesDict":
        out = cls()
        for key, patterns_for_key in patterns.items():
            out[key] = (
                patterns_for_key
                if isinstance(patterns_for_key, DataFilesList)
                else DataFilesList.from_hf_repo(
                    patterns_for_key,
                    dataset_info=dataset_info,
                    base_path=base_path,
                    allowed_extensions=allowed_extensions,
                    download_config=download_config,
                )
            )
        return out

    @classmethod
    def from_patterns(
        cls,
        patterns: dict[str, Union[list[str], DataFilesList]],
        base_path: Optional[str] = None,
        allowed_extensions: Optional[list[str]] = None,
        download_config: Optional[DownloadConfig] = None,
    ) -> "DataFilesDict":
        out = cls()
        for key, patterns_for_key in patterns.items():
            out[key] = (
                patterns_for_key
                if isinstance(patterns_for_key, DataFilesList)
                else DataFilesList.from_patterns(
                    patterns_for_key,
                    base_path=base_path,
                    allowed_extensions=allowed_extensions,
                    download_config=download_config,
                )
            )
        return out

    def filter(
        self, *, extensions: Optional[list[str]] = None, file_names: Optional[list[str]] = None
    ) -> "DataFilesDict":
        out = type(self)()
        for key, data_files_list in self.items():
            out[key] = data_files_list.filter(extensions=extensions, file_names=file_names)
        return out


class DataFilesPatternsList(list[str]):
    """
    List of data files patterns (absolute local paths or URLs).
    For each pattern there should also be a list of allowed extensions
    to keep, or a None ot keep all the files for the pattern.
    """

    def __init__(
        self,
        patterns: list[str],
        allowed_extensions: list[Optional[list[str]]],
    ):
        super().__init__(patterns)
        self.allowed_extensions = allowed_extensions

    def __add__(self, other):
        return DataFilesList([*self, *other], self.allowed_extensions + other.allowed_extensions)

    @classmethod
    def from_patterns(
        cls, patterns: list[str], allowed_extensions: Optional[list[str]] = None
    ) -> "DataFilesPatternsList":
        return cls(patterns, [allowed_extensions] * len(patterns))

    def resolve(
        self,
        base_path: str,
        download_config: Optional[DownloadConfig] = None,
    ) -> "DataFilesList":
        base_path = base_path if base_path is not None else Path().resolve().as_posix()
        data_files = []
        for pattern, allowed_extensions in zip(self, self.allowed_extensions):
            try:
                data_files.extend(
                    resolve_pattern(
                        pattern,
                        base_path=base_path,
                        allowed_extensions=allowed_extensions,
                        download_config=download_config,
                    )
                )
            except FileNotFoundError:
                if not has_magic(pattern):
                    raise
        origin_metadata = _get_origin_metadata(data_files, download_config=download_config)
        return DataFilesList(data_files, origin_metadata)

    def filter_extensions(self, extensions: list[str]) -> "DataFilesPatternsList":
        return DataFilesPatternsList(
            self, [allowed_extensions + extensions for allowed_extensions in self.allowed_extensions]
        )


class DataFilesPatternsDict(dict[str, DataFilesPatternsList]):
    """
    Dict of split_name -> list of data files patterns (absolute local paths or URLs).
    """

    @classmethod
    def from_patterns(
        cls, patterns: dict[str, list[str]], allowed_extensions: Optional[list[str]] = None
    ) -> "DataFilesPatternsDict":
        out = cls()
        for key, patterns_for_key in patterns.items():
            out[key] = (
                patterns_for_key
                if isinstance(patterns_for_key, DataFilesPatternsList)
                else DataFilesPatternsList.from_patterns(
                    patterns_for_key,
                    allowed_extensions=allowed_extensions,
                )
            )
        return out

    def resolve(
        self,
        base_path: str,
        download_config: Optional[DownloadConfig] = None,
    ) -> "DataFilesDict":
        out = DataFilesDict()
        for key, data_files_patterns_list in self.items():
            out[key] = data_files_patterns_list.resolve(base_path, download_config)
        return out

    def filter_extensions(self, extensions: list[str]) -> "DataFilesPatternsDict":
        out = type(self)()
        for key, data_files_patterns_list in self.items():
            out[key] = data_files_patterns_list.filter_extensions(extensions)
        return out
