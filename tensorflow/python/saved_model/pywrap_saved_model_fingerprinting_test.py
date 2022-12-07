# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for pywrap_saved_model_fingerprinting."""

import os

from tensorflow.core.protobuf import fingerprint_pb2
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import test
from tensorflow.python.saved_model.pywrap_saved_model import fingerprinting


class FingerprintingTest(test.TestCase):

  # Checks that the fingerprint values are preserved when passed from C++ to
  # Python.
  def test_fingerprint_def_is_deserialized_correctly(self):
    export_dir = test.test_src_dir_path(
        "cc/saved_model/testdata/VarsAndArithmeticObjectGraph")
    with file_io.FileIO(os.path.join(export_dir, "saved_model.pb"), "rb") as f:
      file_content = f.read()

    fingerprint_def = fingerprint_pb2.FingerprintDef()
    fingerprint_def.ParseFromString(
        fingerprinting.CreateFingerprintDef(file_content, export_dir))
    # We cannot check the value of the saved_model_checksum due to
    # non-determinism in serialization.
    self.assertGreater(fingerprint_def.saved_model_checksum, 0)
    self.assertEqual(fingerprint_def.graph_def_program_hash,
                     10127142238652115842)
    self.assertEqual(fingerprint_def.signature_def_hash, 5693392539583495303)
    self.assertEqual(fingerprint_def.saved_object_graph_hash,
                     3678101440349108924)
    # TODO(b/242348400): The checkpoint hash is non-deterministic, so we cannot
    # check its value here.
    self.assertGreater(fingerprint_def.checkpoint_hash, 0)

  def test_read_fingerprint_from_file(self):
    export_dir = test.test_src_dir_path(
        "cc/saved_model/testdata/VarsAndArithmeticObjectGraph")
    self.assertEqual(
        fingerprinting.MaybeReadSavedModelChecksum(export_dir),
        15788619162413586750)

  def test_read_nonexistent_fingerprint_from_file(self):
    export_dir = test.test_src_dir_path("cc/saved_model/testdata/AssetModule")
    self.assertEqual(fingerprinting.MaybeReadSavedModelChecksum(export_dir), 0)

if __name__ == "__main__":
  test.main()
