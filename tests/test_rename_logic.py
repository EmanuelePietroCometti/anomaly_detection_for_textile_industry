import unittest
import os
import shutil
import tempfile
from pathlib import Path
from src.utils import rename_run_and_update_symlink

class TestUtils(unittest.TestCase):
    def setUp(self):
        """
        Create an isolated temporary directory for the test.
        This ensures a completely clean environment every time.
        """
        # Create a real temporary directory managed by the OS
        self.test_dir = tempfile.mkdtemp()
        self.test_root = Path(self.test_dir)
        
        self.run_dir = self.test_root / "run_123"
        self.symlink_path = self.test_root / "latest"
        
        self.run_dir.mkdir()
        (self.run_dir / "log.txt").write_text("test data content")

        # Use absolute paths for the symlink creation to avoid OS path resolution issues
        os.symlink(str(self.run_dir), str(self.symlink_path))

    def tearDown(self):
        """
        Force destruction of the temporary directory and all its contents
        after the test finishes, regardless of success or failure.
        """
        
        if self.test_root.exists():
            shutil.rmtree(self.test_root)

    def test_rename_and_update_symlink_success(self):
        """
        Main test: verifies physical directory renaming and symlink target update.
        """
        config = {
            "global_timestamp": "20260403",
            "dataset_pipeline": {"dataset_version": "v9"}
        }
        backbone = "resnet18"
        layers = ["layer3", "layer4"]
        
        expected_name = "20260403_resnet18_layer3_layer4_dv9"
        expected_path = self.test_root / expected_name

        # Execute target function
        rename_run_and_update_symlink(self.symlink_path, backbone, layers, config)

        # Assertions mapping the file system state
        self.assertFalse(
            self.run_dir.exists(), 
            "Original directory should no longer exist"
        )
        self.assertTrue(
            expected_path.exists(), 
            f"Renamed directory {expected_name} should exist"
        )
        
        # Resolve the symlink to check its actual absolute target
        current_target = self.symlink_path.resolve()
        self.assertEqual(
            current_target, 
            expected_path.resolve(), 
            "Symlink does not point to the renamed directory"
        )

    def test_invalid_symlink_path(self):
        """
        Verifies that the function handles non-existent paths gracefully
        without raising unhandled exceptions.
        """
        config = {"global_timestamp": "000", "dataset_pipeline": {}}
        invalid_path = self.test_root / "invalid_path"
        
        # It should process it without crashing
        rename_run_and_update_symlink(invalid_path, "none", [], config)
        
        self.assertFalse(invalid_path.exists())

if __name__ == "__main__":
    unittest.main()