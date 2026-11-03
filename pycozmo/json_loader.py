"""

JSON reading functions for files containing non-standard comments

"""

import json
import os
import threading
from typing import Dict, List, Optional, Tuple


def load_json_file(filename: str) -> Dict:
    with open(filename, 'r') as f:
        # Use join to avoid O(n^2) string concatenation and strip // comments per line.
        filtered_json = ''.join(line.split('//')[0] for line in f)
        parsed: Dict = json.loads(filtered_json)
        return parsed


def get_json_files(resource_dir: str, base_names: List[str]) -> List[str]:
    file_addr = []

    for name in base_names:
        if name[0] == '/':
            name = name[1:]
        addr = os.path.join(resource_dir, name)
        if os.path.isdir(addr):
            for root, _, files in os.walk(addr):
                for name in files:
                    if name.endswith('.json'):
                        file_addr.append(os.path.join(root, name))

        elif addr.endswith('.json'):
            file_addr.append(addr)

    return file_addr


# The files under a directory, by name: where the first of them is, as walking the tree finds it. Finding a file used to
# walk the whole tree, and the animation groups' 573 triggers walked it 573 times - 7 s of the 8 s load_anims() takes,
# 75 000 directories visited.
_FILE_INDEXES: Dict[Tuple[str, float], Dict[str, str]] = {}
_FILE_INDEXES_LOCK = threading.Lock()
_FILE_INDEXES_MAX = 32  # LRU cap for distinct (directory, mtime) indexes


def _index(directory: str) -> Dict[str, str]:
    # A tree changed since is walked again: the directory's own modification time is part of the key.
    try:
        key = (os.path.abspath(directory), os.stat(directory).st_mtime)
    except OSError:
        return {}
    with _FILE_INDEXES_LOCK:
        index = _FILE_INDEXES.get(key)
        if index is None:
            index = {}
            for root, _, files in os.walk(directory):
                for name in files:
                    index.setdefault(name, os.path.join(root, name))
            # LRU: move to end on hit, evict oldest on overflow
            _FILE_INDEXES[key] = index
            if len(_FILE_INDEXES) > _FILE_INDEXES_MAX:
                # pop oldest inserted key (dict preserves insertion order, Python 3.7+)
                oldest = next(iter(_FILE_INDEXES))
                if oldest != key:
                    _FILE_INDEXES.pop(oldest, None)
                else:
                    # if the newest is also the oldest (size 1 exceeds max impossible), keep it
                    pass
        else:
            # refresh LRU order
            _FILE_INDEXES.pop(key, None)
            _FILE_INDEXES[key] = index
        return index


def find_file(directory: str, name: str) -> Optional[str]:
    result = _index(directory).get(name)
    if result is None:
        # Cache hit but file not found may be stale (mtime reused, e.g. after utime).
        # Force a fresh walk for the current mtime.
        try:
            key = (os.path.abspath(directory), os.stat(directory).st_mtime)
        except OSError:
            return None
        with _FILE_INDEXES_LOCK:
            # Drop the potentially stale entry for this mtime and re-walk.
            _FILE_INDEXES.pop(key, None)
        # Re-index; if still not found, it truly does not exist.
        result = _index(directory).get(name)
    return result
