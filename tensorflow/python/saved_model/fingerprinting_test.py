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
"""Tests for SavedModel fingerprinting.

These tests verify that FingerprintDef protobuf is written correctly in
`tf.saved_model.save`.
"""
import os
import shutil

from tensorflow.core.config import flags
from tensorflow.core.protobuf import fingerprint_pb2
from tensorflow.python.eager import test
from tensorflow.python.lib.io import file_io
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model.pywrap_saved_model import constants
from tensorflow.python.trackable import autotrackable


class FingerprintingTest(test.TestCase):

  def _create_saved_model(self):
    root = autotrackable.AutoTrackable()
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(root, save_dir)
    self.addCleanup(shutil.rmtree, save_dir)
    return save_dir

  def _read_fingerprint(self, filename):
    fingerprint_def = fingerprint_pb2.FingerprintDef()
    with file_io.FileIO(filename, "rb") as f:
      fingerprint_def.ParseFromString(f.read())
    return fingerprint_def

  def test_basic_module(self):
    flags.config().saved_model_fingerprinting.reset(True)
    save_dir = self._create_saved_model()
    files = file_io.list_directory_v2(save_dir)

    self.assertLen(files, 4)
    self.assertIn(constants.FINGERPRINT_FILENAME, files)

    fingerprint_def = self._read_fingerprint(
        file_io.join(save_dir, constants.FINGERPRINT_FILENAME))
    # We cannot check the value due to non-determinism in serialization.
    self.assertGreater(fingerprint_def.graph_def_hash, 0)


if __name__ == "__main__":
  test.main()
