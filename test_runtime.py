"""Isolated update, rollback, and bootstrap tests; never touches a real app or permissions."""
import hashlib
import json
from pathlib import Path
import plistlib
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import runtime_bootstrap as boot
import update_app as updater


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.app = self.root / 'Saydown.app'
        self.resources = self.app / 'Contents/Resources'
        self.resources.mkdir(parents=True)
        (self.app / 'Contents/MacOS').mkdir()
        (self.app / 'Contents/_CodeSignature').mkdir()
        (self.app / 'Contents/Info.plist').write_bytes(plistlib.dumps({'CFBundleIdentifier': 'io.github.saydown'}))
        (self.resources / 'dictate.py').write_text('print("old")\n')
        (self.resources / 'ui.html').write_text('old UI')
        (self.app / 'Contents/MacOS/Saydown').write_bytes(b'old executable')
        (self.app / 'Contents/MacOS/Saydown').chmod(0o755)
        (self.app / 'Contents/_CodeSignature/CodeResources').write_bytes(b'old signature')
        self.stage = self.root / 'stage'
        self.stage.mkdir()
        (self.stage / 'runtime_bootstrap.py').write_bytes(Path(boot.__file__).read_bytes())
        (self.stage / 'dictate.py').write_text('print("new")\n')
        (self.stage / 'ui.html').write_text('new UI')
        self.data = self.root / 'data'

    def install(self):
        with patch.object(updater.subprocess, 'run') as run:
            receipt = updater.update(self.app, self.stage, self.data)
        return receipt, run

    def snapshot(self):
        return {str(p.relative_to(self.app)): (p.read_bytes(), p.stat().st_mode)
                for p in self.app.rglob('*') if p.is_file()}

    def test_migration_then_routine_update_leaves_every_bundle_file_unchanged(self):
        first, run = self.install()
        self.assertTrue(first['bundle_changed'])
        self.assertEqual(sum('--sign' in c.args[0] for c in run.call_args_list), 1)
        before = self.snapshot()
        (self.stage / 'dictate.py').write_text('print("next")\n')
        second, run = self.install()
        self.assertFalse(second['bundle_changed'])
        self.assertEqual(before, self.snapshot())
        self.assertFalse(any('--sign' in c.args[0] for c in run.call_args_list))
        self.assertEqual(boot.runtime_path(self.resources, self.data).read_text(), 'print("next")\n')

    def test_valid_runtime_is_selected_and_older_generation_is_retained(self):
        first, _ = self.install()
        selected = boot.runtime_path(self.resources, self.data)
        self.assertEqual(selected.parent, Path(first['runtime']))
        second, _ = self.install()
        self.assertTrue(selected.exists())
        self.assertEqual(boot.runtime_path(self.resources, self.data).parent, Path(second['runtime']))

    def test_corrupt_or_missing_runtime_falls_back_to_bundled_code(self):
        receipt, _ = self.install()
        selected = boot.runtime_path(self.resources, self.data)
        selected.write_text('corrupted')
        self.assertEqual(boot.runtime_path(self.resources, self.data), self.resources / 'dictate_bundled.py')
        (self.data / 'Runtime/current.json').unlink()
        self.assertEqual(boot.runtime_path(self.resources, self.data), self.resources / 'dictate_bundled.py')

    def test_incomplete_manifest_and_wrong_base_fall_back(self):
        receipt, _ = self.install()
        manifest_path = Path(receipt['runtime']) / 'manifest.json'
        original = json.loads(manifest_path.read_text())
        for bad in [[], {}, original | {'protocol': 9}, original | {'base_sha256': 'wrong'},
                    original | {'files': {}}, original | {'files': ['dictate.py', 'ui.html']}]:
            manifest_path.write_text(json.dumps(bad))
            self.assertEqual(boot.runtime_path(self.resources, self.data), self.resources / 'dictate_bundled.py')

    def test_pointer_cannot_escape_runtime_directory(self):
        self.install()
        for name in ['../stage', '/tmp', '../../bad', 5]:
            (self.data / 'Runtime/current.json').write_text(json.dumps({'version': name}))
            self.assertEqual(boot.runtime_path(self.resources, self.data), self.resources / 'dictate_bundled.py')

    def test_symlinked_generation_or_code_is_rejected(self):
        receipt, _ = self.install()
        root = self.data / 'Runtime'
        (root / 'link').symlink_to(self.stage, target_is_directory=True)
        (root / 'current.json').write_text('{"version":"link"}')
        self.assertEqual(boot.runtime_path(self.resources, self.data), self.resources / 'dictate_bundled.py')
        (root / 'current.json').write_text(json.dumps({'version': Path(receipt['runtime']).name}))
        code = Path(receipt['runtime']) / 'dictate.py'
        code.unlink()
        code.symlink_to(self.stage / 'dictate.py')
        self.assertEqual(boot.runtime_path(self.resources, self.data), self.resources / 'dictate_bundled.py')

    def test_first_migration_failure_restores_app_bytes_modes_and_pointer(self):
        before = self.snapshot()
        with patch.object(updater.subprocess, 'run', side_effect=[None, subprocess.CalledProcessError(1, 'codesign')]):
            with self.assertRaises(subprocess.CalledProcessError):
                updater.update(self.app, self.stage, self.data)
        self.assertEqual(self.snapshot(), before)
        self.assertFalse((self.data / 'Runtime/current.json').exists())

    def test_failed_routine_update_restores_previous_pointer_and_code(self):
        self.install()
        old = (self.data / 'Runtime/current.json').read_bytes()
        selected = boot.runtime_path(self.resources, self.data)
        before = self.snapshot()
        (self.stage / 'dictate.py').write_text('print("next")\n')
        with patch.object(updater.subprocess, 'run', side_effect=[None, subprocess.CalledProcessError(1, 'codesign')]):
            with self.assertRaises(subprocess.CalledProcessError):
                updater.update(self.app, self.stage, self.data)
        self.assertEqual((self.data / 'Runtime/current.json').read_bytes(), old)
        self.assertEqual(boot.runtime_path(self.resources, self.data), selected)
        self.assertEqual(self.snapshot(), before)

    def test_bootstrap_actually_executes_selected_runtime(self):
        self.install()
        bootstrap = self.resources / 'dictate.py'
        code = "from pathlib import Path; import runpy; Path.home = lambda: Path(%r); runpy.run_path(%r, run_name='__main__')" % (str(self.root / 'home'), str(bootstrap))
        # Put the test data at the path the real bootstrap resolves for this fake home.
        import shutil
        home_data = self.root / 'home/Library/Application Support/Saydown'
        shutil.copytree(self.data, home_data)
        import sys
        result = subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)
        self.assertEqual(result.stdout.strip(), 'new')

    def test_wrong_bundle_or_invalid_python_is_rejected_before_changes(self):
        before = self.snapshot()
        (self.stage / 'dictate.py').write_text('this is not valid python !')
        with patch.object(updater.subprocess, 'run'):
            with self.assertRaises(SyntaxError):
                updater.update(self.app, self.stage, self.data)
        self.assertEqual(self.snapshot(), before)
        (self.app / 'Contents/Info.plist').write_bytes(plistlib.dumps({'CFBundleIdentifier': 'other.app'}))
        with patch.object(updater.subprocess, 'run') as run:
            with self.assertRaises(ValueError):
                updater.update(self.app, self.stage, self.data)
            run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
