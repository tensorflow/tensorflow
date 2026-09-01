# Copyright 2026 The OpenXLA Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Integration test for ZipWriter using Python's standard zipfile library."""

import os
import subprocess
import tempfile
import zipfile

from absl.testing import absltest


class ZipWriterIntegrationTest(absltest.TestCase):

  def test_generated_zip_validity(self):
    generator_path = os.path.join(
        os.path.dirname(__file__),
        "zip_generator",
    )

    with tempfile.TemporaryDirectory() as temp_dir:
      zip_path = os.path.join(temp_dir, "output.zip")

      test_files = {
          "empty.txt": "",
          "hello.txt": "Hello, world!",
          "large.txt": "A" * 100000,
      }

      args = [generator_path, zip_path]
      for name, content in test_files.items():
        args.extend([name, content])

      # Run the generator; exit status is checked automatically.
      subprocess.run(args, capture_output=True, text=True, check=True)

      with zipfile.ZipFile(zip_path, "r") as z:
        bad_file = z.testzip()
        self.assertIsNone(bad_file, f"Corrupted file in ZIP: {bad_file}")

        self.assertCountEqual(z.namelist(), test_files.keys())

        for name, expected_content in test_files.items():
          with z.open(name) as f:
            decompressed_content = f.read().decode("utf-8")
            self.assertEqual(decompressed_content, expected_content)


if __name__ == "__main__":
  absltest.main()
