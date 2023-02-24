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
"""Tests for the buffer and DTensor conversion helpers."""

import numpy as np

# pylint: disable=g-direct-tensorflow-import
from tensorflow.dtensor.python import layout
from tensorflow.dtensor.python import numpy_util
from tensorflow.dtensor.python.tests import test_util
from tensorflow.python.platform import test as tf_test
# pylint: enable=g-direct-tensorflow-import

_MESH_DIM_X = 'x'
_MESH_DIM_Y = 'y'


class NumpyUtilTest(test_util.DTensorBaseTest):

  def setUp(self):
    super().setUp()
    global_ids = test_util.create_device_ids_array((2, 2))
    local_ids = np.ravel(global_ids).tolist()
    mesh_dict = {
        device: layout.Mesh([_MESH_DIM_X, _MESH_DIM_Y], global_ids, local_ids,
                            test_util.create_device_list((2, 2), device))
        for device in ('CPU', 'GPU', 'TPU')
    }
    self.mesh = self.configTestMesh(mesh_dict)

  def test_tensor_from_replicated(self):
    tensors = [np.arange(4) for i in range(self.mesh.size)]
    replicated_layout = layout.Layout([layout.UNSHARDED, layout.UNSHARDED],
                                      mesh=self.mesh)

    self.assertAllClose(
        np.arange(4), numpy_util.unpacked_to_numpy(tensors, replicated_layout))

  def test_tensor_x_sharded(self):
    t00 = np.arange(8).reshape(2, 4)
    t01 = np.arange(8).reshape(2, 4)
    t10 = np.arange(8, 16).reshape(2, 4)
    t11 = np.arange(8, 16).reshape(2, 4)
    tensors = [t00, t01, t10, t11]
    sharded_on_x = layout.Layout([_MESH_DIM_X, layout.UNSHARDED],
                                 mesh=self.mesh)
    self.assertAllClose(
        np.arange(16).reshape(4, 4),
        numpy_util.unpacked_to_numpy(tensors, sharded_on_x))

  def test_tensor_y_sharded(self):
    # [[0,1], [4,5], [8,9], [12,13]]
    t00 = np.arange(16).reshape(4, 4)[:, :-2]
    # [[2,3], [6,7], [10,11], [14,15]]
    t01 = np.arange(16).reshape(4, 4)[:, 2:4]
    t10 = np.arange(16).reshape(4, 4)[:, :-2]
    t11 = np.arange(16).reshape(4, 4)[:, 2:4]
    tensors = [t00, t01, t10, t11]
    sharded_on_y = layout.Layout([layout.UNSHARDED, _MESH_DIM_Y],
                                 mesh=self.mesh)
    self.assertAllClose(
        numpy_util.unpacked_to_numpy(tensors, sharded_on_y),
        np.arange(16).reshape(4, 4))

  def test_tensor_x_sharded_on_mesh_y(self):
    t00 = np.arange(8).reshape(2, 4)
    t01 = np.arange(8, 16).reshape(2, 4)
    t10 = np.arange(8).reshape(2, 4)
    t11 = np.arange(8, 16).reshape(2, 4)
    tensors = [t00, t01, t10, t11]
    sharded_on_y = layout.Layout([_MESH_DIM_Y, layout.UNSHARDED],
                                 mesh=self.mesh)
    self.assertAllClose(
        numpy_util.unpacked_to_numpy(tensors, sharded_on_y),
        np.arange(16).reshape(4, 4))

  def test_tensor_y_sharded_on_mesh_x(self):
    # [[0,1], [4,5], [8,9], [12,13]]
    t00 = np.arange(16).reshape(4, 4)[:, :-2]
    t01 = np.arange(16).reshape(4, 4)[:, :-2]
    # [[2,3], [6,7], [10,11], [14,15]]
    t10 = np.arange(16).reshape(4, 4)[:, 2:4]
    t11 = np.arange(16).reshape(4, 4)[:, 2:4]
    tensors = [t00, t01, t10, t11]
    sharded_on_x = layout.Layout([layout.UNSHARDED, _MESH_DIM_X],
                                 mesh=self.mesh)
    self.assertAllClose(
        numpy_util.unpacked_to_numpy(tensors, sharded_on_x),
        np.arange(16).reshape(4, 4))

  def test_tensor_x_y_sharded_x_y(self):
    t00 = np.array([[0, 1], [4, 5]])
    t01 = np.array([[2, 3], [6, 7]])
    t10 = np.array([[8, 9], [12, 13]])
    t11 = np.array([[10, 11], [14, 15]])
    tensors = [t00, t01, t10, t11]
    sharded_on_x_y = layout.Layout([_MESH_DIM_X, _MESH_DIM_Y], mesh=self.mesh)
    self.assertAllClose(
        numpy_util.unpacked_to_numpy(tensors, sharded_on_x_y),
        np.arange(16).reshape(4, 4))

  def test_tensor_x_y_sharded_y_x(self):
    t00 = np.array([[0, 1], [4, 5]])
    t01 = np.array([[8, 9], [12, 13]])
    t10 = np.array([[2, 3], [6, 7]])
    t11 = np.array([[10, 11], [14, 15]])
    tensors = [t00, t01, t10, t11]
    sharded_on_y_x = layout.Layout([_MESH_DIM_Y, _MESH_DIM_X], mesh=self.mesh)
    self.assertAllClose(
        numpy_util.unpacked_to_numpy(tensors, sharded_on_y_x),
        np.arange(16).reshape(4, 4))


if __name__ == '__main__':
  tf_test.main()
