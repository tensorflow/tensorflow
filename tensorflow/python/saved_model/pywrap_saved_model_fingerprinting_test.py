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

from tensorflow.core.protobuf import fingerprint_pb2
from tensorflow.python.platform import test
from tensorflow.python.saved_model.pywrap_saved_model import fingerprinting as pywrap_fingerprinting

is_oss = True  # Updated by copybara.


class FingerprintingTest(test.TestCase):
  def test_create_fingerprint_def(self):
    export_dir = test.test_src_dir_path(
        "cc/saved_model/testdata/VarsAndArithmeticObjectGraph")

    fingerprint = fingerprint_pb2.FingerprintDef().FromString(
        pywrap_fingerprinting.CreateFingerprintDef(export_dir))

    # We cannot check the value of the saved_model_checksum due to
    # non-determinism in saving.
    self.assertGreater(fingerprint.saved_model_checksum, 0)
    self.assertEqual(fingerprint.graph_def_program_hash, 10127142238652115842)
    self.assertEqual(fingerprint.signature_def_hash, 15570736222402453744)
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
        pywrap_fingerprinting.FileNotFoundException) as excinfo:
      pywrap_fingerprinting.ReadSavedModelFingerprint(export_dir)
    self.assertRegex(str(excinfo.exception), "Could not find fingerprint.")

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

  def test_read_saved_model_singleprint_from_fp(self):
    export_dir = test.test_src_dir_path(
        "cc/saved_model/testdata/VarsAndArithmeticObjectGraph")
    singleprint = pywrap_fingerprinting.SingleprintFromFP(export_dir)
    # checkpoint_hash is non-deterministic and not included
    self.assertRegex(singleprint,
                     "/".join([
                         "706963557435316516",  # graph_def_program_hash
                         "5693392539583495303",  # signature_def_hash
                         "12074714563970609759",  # saved_object_graph_hash
                         ]))

  def test_read_saved_model_singleprint_from_sm(self):
    export_dir = test.test_src_dir_path(
        "cc/saved_model/testdata/AssetModule")
    singleprint = pywrap_fingerprinting.SingleprintFromSM(export_dir)
    # checkpoint_hash is non-deterministic and not included
    self.assertRegex(singleprint,
                     "/".join([
                         "14732473038199296573",  # graph_def_program_hash
                         "11983586671997178523",  # signature_def_hash
                         "14640180866165615446",  # saved_object_graph_hash
                         ]))

  def test_read_chunked_saved_model_fingerprint(self):
    if is_oss:
      self.skipTest("Experimental image format disabled in OSS.")
    export_dir = test.test_src_dir_path(
        "cc/saved_model/testdata/chunked_saved_model/chunked_model")
    fingerprint = fingerprint_pb2.FingerprintDef().FromString(
        pywrap_fingerprinting.CreateFingerprintDef(export_dir))
    self.assertGreater(fingerprint.saved_model_checksum, 0)
    # We test for multiple fingerprints due to non-determinism when building
    # with different compilation_mode flag options.
    self.assertIn(fingerprint.graph_def_program_hash,
                  [906548630859202535, 9562420523583756263])
    self.assertEqual(fingerprint.signature_def_hash, 1043582354059066488)
    self.assertIn(fingerprint.saved_object_graph_hash,
                  [11894619660760763927, 2766043449526180728])
    self.assertEqual(fingerprint.checkpoint_hash, 0)


if __name__ == "__main__":
  test.main()
