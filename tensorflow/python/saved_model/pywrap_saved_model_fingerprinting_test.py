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
from tensorflow.python.saved_model.pywrap_saved_model import fingerprinting as pywrap_fingerprinting


class FingerprintingTest(test.TestCase):
  def test_create_fingerprint_def(self):
    export_dir = test.test_src_dir_path(
        "cc/saved_model/testdata/VarsAndArithmeticObjectGraph")
    with file_io.FileIO(os.path.join(export_dir, "saved_model.pb"), "rb") as f:
      file_content = f.read()

    fingerprint = fingerprint_pb2.FingerprintDef().FromString(
        pywrap_fingerprinting.CreateFingerprintDef(file_content, export_dir))
    # We cannot check the value of the saved_model_checksum due to
    # non-determinism in serialization.
    self.assertGreater(fingerprint.saved_model_checksum, 0)
    self.assertEqual(fingerprint.graph_def_program_hash, 10127142238652115842)
    self.assertEqual(fingerprint.signature_def_hash, 5693392539583495303)
    self.assertEqual(fingerprint.saved_object_graph_hash, 3678101440349108924)
    # TODO(b/242348400): The checkpoint hash is non-deterministic, so we cannot
    # check its value here.
    self.assertGreater(fingerprint.checkpoint_hash, 0)

  def test_read_saved_model_fingerprint(self):
    export_dir = test.test_src_dir_path(
        "cc/saved_model/testdata/VarsAndArithmeticObjectGraph")
    fingerprint = fingerprint_pb2.FingerprintDef().FromString(
        pywrap_fingerprinting.ReadSavedModelFingerprint(export_dir))
    self.assertGreater(fingerprint.saved_model_checksum, 0)
    self.assertEqual(fingerprint.graph_def_program_hash, 706963557435316516)
    self.assertEqual(fingerprint.signature_def_hash, 5693392539583495303)
    self.assertEqual(fingerprint.saved_object_graph_hash, 12074714563970609759)
    self.assertGreater(fingerprint.checkpoint_hash, 0)
    self.assertEqual(fingerprint.version.producer, 1)

  def test_read_nonexistent_fingerprint(self):
    export_dir = test.test_src_dir_path("cc/saved_model/testdata/AssetModule")
    with self.assertRaises(
        pywrap_fingerprinting.FingerprintException) as excinfo:
      pywrap_fingerprinting.ReadSavedModelFingerprint(export_dir)
    self.assertRegex(str(excinfo.exception), "Could not read fingerprint.")

  def test_read_saved_model_singleprint(self):
    export_dir = test.test_src_dir_path(
        "cc/saved_model/testdata/VarsAndArithmeticObjectGraph")
    fingerprint = fingerprint_pb2.FingerprintDef().FromString(
        pywrap_fingerprinting.ReadSavedModelFingerprint(export_dir))
    singleprint = pywrap_fingerprinting.Singleprint(
        fingerprint.graph_def_program_hash,
        fingerprint.signature_def_hash,
        fingerprint.saved_object_graph_hash,
        fingerprint.checkpoint_hash)
    # checkpoint_hash is non-deterministic and not included
    self.assertRegex(singleprint,
                     "/".join([
                         "706963557435316516",  # graph_def_program_hash
                         "5693392539583495303",  # signature_def_hash
                         "12074714563970609759",  # saved_object_graph_hash
                         ]))


if __name__ == "__main__":
  test.main()
