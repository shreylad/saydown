"""Update the existing personal Saydown app without changing its signature on routine updates.

Run: the app's bundled Python update_app.py (then restart Saydown).
The first migration installs a stable bootstrap and requires one final permission renewal.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import uuid


def atomic_write(path, data):
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def update(app, stage, data):
    with (app / 'Contents/Info.plist').open('rb') as file:
        if plistlib.load(file).get('CFBundleIdentifier') != 'io.github.saydown':
            raise ValueError('This updater only supports the existing Saydown app.')
    subprocess.run(['codesign', '--verify', '--strict', str(app)], check=True)
    resources = app / 'Contents/Resources'
    bootstrap = (stage / 'runtime_bootstrap.py').read_bytes()
    compile(bootstrap, str(stage / 'runtime_bootstrap.py'), 'exec')
    migrate = (resources / 'dictate.py').read_bytes() != bootstrap
    runtime = {name: (stage / name).read_bytes() for name in ('dictate.py', 'ui.html')}
    compile(runtime['dictate.py'], str(stage / 'dictate.py'), 'exec')
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8]
    backup = data / 'Backups' / ('runtime-' + stamp)
    backup.mkdir(parents=True)
    root = data / 'Runtime'
    root.mkdir(exist_ok=True)
    pointer = root / 'current.json'
    old_pointer = pointer.read_bytes() if pointer.exists() else None
    if old_pointer is not None:
        (backup / 'current.json').write_bytes(old_pointer)
    changed = ('Contents/Resources/dictate.py', 'Contents/Resources/dictate_bundled.py',
               'Contents/Resources/ui.html', 'Contents/MacOS/Saydown', 'Contents/_CodeSignature/CodeResources')
    saved = {}
    if migrate:
        for name in changed:
            path = app / name
            saved[name] = path.read_bytes() if path.exists() else None
            if saved[name] is not None:
                destination = backup / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
    generation = root / stamp
    generation.mkdir()
    try:
        if migrate:
            atomic_write(resources / 'dictate_bundled.py', runtime['dictate.py'])
            atomic_write(resources / 'ui.html', runtime['ui.html'])
            atomic_write(resources / 'dictate.py', bootstrap)
            subprocess.run(['codesign', '--force', '--sign', '-', str(app)], check=True)
            subprocess.run(['codesign', '--verify', '--strict', str(app)], check=True)
        base = hashlib.sha256((resources / 'dictate_bundled.py').read_bytes()).hexdigest()
        for name, contents in runtime.items():
            (generation / name).write_bytes(contents)
        manifest = {'protocol': 1, 'base_sha256': base,
                    'files': {name: hashlib.sha256(contents).hexdigest() for name, contents in runtime.items()}}
        (generation / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        atomic_write(pointer, (json.dumps({'version': stamp}) + '\n').encode())
        subprocess.run(['codesign', '--verify', '--strict', str(app)], check=True)
    except BaseException:
        if old_pointer is None:
            pointer.unlink(missing_ok=True)
        else:
            atomic_write(pointer, old_pointer)
        if migrate:
            for name, contents in saved.items():
                if contents is None:
                    (app / name).unlink(missing_ok=True)
                else:
                    atomic_write(app / name, contents)
                    if name == 'Contents/MacOS/Saydown':
                        os.chmod(app / name, (backup / name).stat().st_mode)
        shutil.rmtree(generation)
        raise
    receipt = {'app': str(app), 'runtime': str(generation), 'bundle_changed': migrate,
               'backup': str(backup)}
    (backup / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--app', type=Path, default=Path('/Applications/Saydown.app'))
    args = parser.parse_args()
    receipt = update(args.app, Path(__file__).resolve().parent,
                     Path.home() / 'Library/Application Support/Saydown')
    print(json.dumps(receipt, indent=2))
    print('Restart Saydown to use the update.')
