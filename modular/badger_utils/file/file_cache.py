import pickle
from pathlib import Path
from typing import TypeVar, Callable

T = TypeVar('T')


class FileCache:
    SUFFIX = '.pkl'

    def __init__(self, root_dir: Path = None):
        if root_dir is None:
            root_dir = Path('data') / 'cache'
        root_dir.mkdir(parents=True, exist_ok=True)
        self.root_dir = root_dir

    def _file(self, key: str) -> Path:
        return self.root_dir / f'{key}{self.SUFFIX}'

    def cached(self, key: str, fn: Callable[[], T], force_compute: bool = False) -> T:
        """
        Return result of fn(). The result is cached with key.
        When key is present in cache, the value is retrieved from the cache.
        Args:
            key: unique key that will be used as a filename
            fn: function with no parameters to be cached
            force_compute: overwrite cache even when the key is present

        Returns:
             Result of fn() or cached value.
        """
        file = self._file(key)
        if file.exists() and not force_compute:
            with file.open('rb') as h:
                data = pickle.load(h)
        else:
            data = fn()
            with file.open('wb') as h:
                pickle.dump(data, h)
        return data

    def clear(self, key: str):
        """Clear cache for given key. Key doesn't have to exist"""
        file = self._file(key)
        if file.exists():
            file.unlink()

    def clear_all(self, prefix: str = ''):
        for file in self.root_dir.iterdir():
            if file.is_file() and file.name.startswith(prefix):
                if file.name.endswith(self.SUFFIX):  # Just safety check
                    file.unlink()
