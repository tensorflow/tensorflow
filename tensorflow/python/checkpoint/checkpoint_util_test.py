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
"""Tests for checkpoint_util."""

from tensorflow.python.checkpoint import checkpoint_util
from tensorflow.python.checkpoint import saveable_compat
from tensorflow.python.eager import test
from tensorflow.python.trackable import base


class TrackableWithLegacyName(base.Trackable):

  @saveable_compat.legacy_saveable_name("legacy")
  def _serialize_to_tensors(self):
    return {}


class ExtractSaveablenameTest(test.TestCase):

  def test_standard_saveable_name(self):
    self.assertEqual(
        "object_path/.ATTRIBUTES/",
        checkpoint_util.extract_saveable_name(
            base.Trackable(), "object_path/.ATTRIBUTES/123"))
    self.assertEqual(
        "object/path/ATTRIBUTES/.ATTRIBUTES/",
        checkpoint_util.extract_saveable_name(
            base.Trackable(), "object/path/ATTRIBUTES/.ATTRIBUTES/"))

  def test_legacy_saveable_name(self):
    self.assertEqual(
        "object_path/.ATTRIBUTES/123",
        checkpoint_util.extract_saveable_name(
            TrackableWithLegacyName(), "object_path/.ATTRIBUTES/123"))
    self.assertEqual(
        "object/path/ATTRIBUTES/.ATTRIBUTES/",
        checkpoint_util.extract_saveable_name(
            TrackableWithLegacyName(), "object/path/ATTRIBUTES/.ATTRIBUTES/"))


if __name__ == "__main__":
  test.main()
