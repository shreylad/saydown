"""Stable bundled entry point for local Saydown updates; no network or credential access."""
import hashlib
import json
from pathlib import Path
import re
import runpy


def runtime_path(resources, data):
    fallback = resources / 'dictate_bundled.py'
    root = data / 'Runtime'
    try:
        pointer = json.loads((root / 'current.json').read_text())
        version = pointer['version']
        if not isinstance(version, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', version):
            raise ValueError('invalid runtime version')
        directory = root / version
        if directory.is_symlink() or not directory.resolve().is_relative_to(root.resolve()):
            raise ValueError('invalid runtime directory')
        manifest = json.loads((directory / 'manifest.json').read_text())
        if not isinstance(manifest, dict) or not isinstance(manifest.get('files'), dict):
            raise ValueError('invalid runtime manifest')
        base = hashlib.sha256(fallback.read_bytes()).hexdigest()
        if manifest.get('protocol') != 1 or manifest.get('base_sha256') != base:
            raise ValueError('incompatible runtime')
        if set(manifest['files']) != {'dictate.py', 'ui.html'}:
            raise ValueError('incomplete runtime')
        for name, digest in manifest['files'].items():
            path = directory / name
            if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError('incomplete runtime')
        return directory / 'dictate.py'
    except (OSError, ValueError, KeyError, TypeError):
        return fallback


if __name__ == '__main__':
    resources = Path(__file__).resolve().parent
    data = Path.home() / 'Library/Application Support/Saydown'
    runpy.run_path(str(runtime_path(resources, data)), run_name='__main__')
