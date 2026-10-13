"""Update the existing personal Saydown app without changing its signature.

Run: the app's bundled Python update_app.py (then restart Saydown).
The new code goes beside the user's data, where the app's bundled bootstrap finds it; the app itself is not touched.
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
    if (resources / 'dictate.py').read_bytes() != (stage / 'runtime_bootstrap.py').read_bytes():
        # the bootstrap is inside the signed app, so replacing it means a new build and renewed permissions
        raise ValueError('runtime_bootstrap.py differs from the one inside the app: rebuild it with ./install-app.sh --release.')
    runtime = {name: (stage / name).read_bytes() for name in ('dictate.py', 'ui.html')}
    compile(runtime['dictate.py'], str(stage / 'dictate.py'), 'exec')
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8]
    root = data / 'Runtime'
    root.mkdir(parents=True, exist_ok=True)
    pointer = root / 'current.json'
    old_pointer = pointer.read_bytes() if pointer.exists() else None
    generation = root / stamp  # earlier generations stay, so pointing current.json back at one undoes an update
    generation.mkdir()
    try:
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
        shutil.rmtree(generation)
        raise
    return {'app': str(app), 'runtime': str(generation)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--app', type=Path, default=Path('/Applications/Saydown.app'))
    args = parser.parse_args()
    receipt = update(args.app, Path(__file__).resolve().parent,
                     Path.home() / 'Library/Application Support/Saydown')
    print(json.dumps(receipt, indent=2))
    print('Restart Saydown to use the update.')
