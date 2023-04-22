# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for SavedModel instrumentation for Python reading/writing code.

These tests verify that the counters are incremented correctly after SavedModel
API calls.
"""

import os
import shutil

from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import builder_impl
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import load_v1_in_v2
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model.experimental.pywrap_libexport import metrics
from tensorflow.python.training.tracking import tracking


class MetricsTests(test.TestCase):

  def _create_save_v2_model(self):
    root = tracking.AutoTrackable()
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(root, save_dir)
    self.addCleanup(shutil.rmtree, save_dir)
    return save_dir

  def _create_save_v1_model(self):
    save_dir = os.path.join(self.get_temp_dir(), "builder")
    builder_ = builder.SavedModelBuilder(save_dir)

    with ops.Graph().as_default():
      with self.session(graph=ops.Graph()) as sess:
        constant_op.constant(5.0)
        builder_.add_meta_graph_and_variables(sess, ["foo"])
      builder_.save()
    self.addCleanup(shutil.rmtree, save_dir)
    return save_dir

  def test_python_save(self):
    write_count = metrics.GetWrite()
    save_api_count = metrics.GetWriteApi(save._SAVE_V2_LABEL, write_version="2")
    _ = self._create_save_v2_model()

    self.assertEqual(
        metrics.GetWriteApi(save._SAVE_V2_LABEL, write_version="2"),
        save_api_count + 1)
    self.assertEqual(metrics.GetWrite(), write_count + 1)

  def test_builder_save(self):
    write_count = metrics.GetWrite()
    save_builder_count = metrics.GetWriteApi(
        builder_impl._SAVE_BUILDER_LABEL, write_version="1")
    _ = self._create_save_v1_model()

    self.assertEqual(
        metrics.GetWriteApi(
            builder_impl._SAVE_BUILDER_LABEL, write_version="1"),
        save_builder_count + 1)
    self.assertEqual(metrics.GetWrite(), write_count + 1)

  def test_load_v2(self):
    read_count = metrics.GetRead()
    load_v2_count = metrics.GetReadApi(load._LOAD_V2_LABEL, write_version="2")

    save_dir = self._create_save_v2_model()
    load.load(save_dir)

    self.assertEqual(
        metrics.GetReadApi(load._LOAD_V2_LABEL, write_version="2"),
        load_v2_count + 1)
    self.assertEqual(metrics.GetRead(), read_count + 1)

  def test_load_v1_in_v2(self):
    read_count = metrics.GetRead()
    load_v2_count = metrics.GetReadApi(load._LOAD_V2_LABEL, write_version="2")
    load_v1_v2_count = metrics.GetReadApi(
        load_v1_in_v2._LOAD_V1_V2_LABEL, write_version="1")

    save_dir = self._create_save_v1_model()
    load.load(save_dir)

    # Check that `load_v2` was *not* incremented.
    self.assertEqual(
        metrics.GetReadApi(load._LOAD_V2_LABEL, write_version="2"),
        load_v2_count)
    self.assertEqual(
        metrics.GetReadApi(load_v1_in_v2._LOAD_V1_V2_LABEL, write_version="1"),
        load_v1_v2_count + 1)
    self.assertEqual(metrics.GetRead(), read_count + 1)

  def test_loader_v1(self):
    read_count = metrics.GetRead()
    ops.disable_eager_execution()
    save_dir = self._create_save_v1_model()
    loader = loader_impl.SavedModelLoader(save_dir)
    with self.session(graph=ops.Graph()) as sess:
      loader.load(sess, ["foo"])
    ops.enable_eager_execution()

    self.assertEqual(
        metrics.GetReadApi(loader_impl._LOADER_LABEL, write_version="1"), 1)
    self.assertEqual(metrics.GetRead(), read_count + 1)


if __name__ == "__main__":
  test.main()
