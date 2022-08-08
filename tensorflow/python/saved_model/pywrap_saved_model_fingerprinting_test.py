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
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import test
from tensorflow.python.saved_model.pywrap_saved_model import fingerprinting


class FingerprintingTest(test.TestCase):

  def test_module_basic(self):
    sm_pb_file = test.test_src_dir_path(
        "cc/saved_model/testdata/VarsAndArithmeticObjectGraph/saved_model.pb")
    with file_io.FileIO(sm_pb_file, "rb") as f:
      file_content = f.read()

    fingerprint_def = fingerprint_pb2.FingerprintDef()
    fingerprint_def.ParseFromString(
        fingerprinting.CreateFingerprintDef(file_content))
    # We cannot check the value of the graph_def_checksum due to non-determinism
    # in serialization.
    self.assertGreater(fingerprint_def.graph_def_checksum, 0)
    self.assertEqual(fingerprint_def.graph_def_program_hash,
                     13188891313422428336)
    self.assertEqual(fingerprint_def.signature_def_hash, 5693392539583495303)
    self.assertEqual(fingerprint_def.saved_object_graph_hash,
                     3678101440349108924)


if __name__ == "__main__":
  test.main()
