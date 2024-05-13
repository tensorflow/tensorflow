# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

from absl.testing import parameterized

# pylint: disable=g-direct-tensorflow-import
from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import d_variable
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.dtensor.python import numpy_util
from tensorflow.dtensor.python.tests import test_util
from tensorflow.dtensor.python.tests import test_util_ops
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.distribute.cluster_resolver.tpu import tpu_cluster_resolver
from tensorflow.python.eager import remote
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_bitwise_ops
from tensorflow.python.ops import gen_stateful_random_ops
from tensorflow.python.ops import gen_stateless_random_ops_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.tpu import device_assignment as device_assignment_lib

# pylint: enable=g-direct-tensorflow-import

# Makes a 2-D mesh with dimensions as, X(2) and Y(4).
_MESH_DIM_X = 'x'
_MESH_DIM_Y = 'y'
_MESH_DIMS = [_MESH_DIM_X, _MESH_DIM_Y]

Layout = layout_lib.Layout
Mesh = layout_lib.Mesh

# Create a random local IDs to make tests more challenging.
_LOCAL_IDS = [7, 3, 1, 4, 2, 0, 6, 5]
# The row and col indices for each local id, e.g., 7 is (row=1, col=3)
_ROW_INDEX = [i / 4 for i in _LOCAL_IDS]
_COL_INDEX = [i % 4 for i in _LOCAL_IDS]

# The index of local id for the row head.
#
# For example, local id 7 is on row 1, the head is local id 4, whose index in
# _LOCAL_IDS is 3, i.e., _LOCAL_IDS[3] == 4
_ROW_0_HEAD = 3
_ROW_1_HEAD = 5
_ROW_HEAD = [3, 5, 5, 3, 5, 5, 3, 3]

# The index of local id for the col head. Similar to row id before.
_COL_0_HEAD = 5
_COL_1_HEAD = 2
_COL_2_HEAD = 4
_COL_3_HEAD = 1
_COL_HEAD = [1, 1, 2, 5, 4, 5, 4, 2]

_tpu_strategy = None


def _call_op(op, seed, shape, dtype, key, counter, alg, minval, maxval,
             op_version):
  if op_version == 'V1':
    return op(shape=shape, seed=seed, dtype=dtype)
  elif op_version == 'V2':
    return op(shape=shape, key=key, counter=counter, alg=alg, dtype=dtype)
  elif op_version == 'V2_RANGE':
    return op(
        shape=shape,
        key=key,
        counter=counter,
        alg=alg,
        minval=minval,
        maxval=maxval)
  else:
    raise ValueError('op_version argument was invalid.')


def _call_dtensor_op(op, seed, shape, dtype, key, counter, alg, minval, maxval,
                     op_version, mesh):
  if op_version == 'V1':
    return op(shape=shape, seed=seed, dtype=dtype)

  shape = numpy_util.pack_numpy(
      constant_op.constant(shape), Layout.replicated(mesh, 1)
  )
  key = numpy_util.pack_numpy(key, Layout.replicated(mesh, 1))
  counter = numpy_util.pack_numpy(counter, Layout.replicated(mesh, 1))

  if op_version == 'V2':
    return op(shape=shape, key=key, counter=counter, alg=alg, dtype=dtype)
  elif op_version == 'V2_RANGE':
    return op(
        shape=shape,
        key=key,
        counter=counter,
        alg=alg,
        minval=minval,
        maxval=maxval)
  else:
    raise ValueError('op_version argument was invalid.')


def get_tpu_strategy():
  """Returns a single-core TPUStrategy."""
  global _tpu_strategy
  if _tpu_strategy is not None:
    return _tpu_strategy

  resolver = tpu_cluster_resolver.TPUClusterResolver(tpu='')
  remote.connect_to_cluster(resolver)
  topology = tpu_cluster_resolver.initialize_tpu_system(resolver)
  device_assignment = device_assignment_lib.DeviceAssignment.build(
      topology, num_replicas=1
  )
  strategy = tpu_strategy.TPUStrategyV2(
      resolver, experimental_device_assignment=device_assignment
  )
  _tpu_strategy = strategy
  return strategy


def rng_op_spmd(op,
                device_id,
                seed,
                shape,
                dtype,
                key,
                counter,
                alg,
                minval,
                maxval,
                op_version,
                device_index_fn,
                full_replicated=False,
                is_tpu=False):

  if not is_tpu:
    return rng_op_spmd_fn(
        op,
        device_id,
        seed,
        shape,
        dtype,
        key,
        counter,
        alg,
        minval,
        maxval,
        op_version,
        device_index_fn,
        full_replicated=full_replicated)

  # As of 2021-April, TPU eager and multi-device function produce different
  # stateless rng results compared with bridge compiled function. As DTensor
  # uses bridge to lower TPU function by default, we need to create a
  # TPUStrategy for single core and invoke `run` on it.
  @polymorphic_function.function
  def tpu_fn(device_id, seed):
    return rng_op_spmd_fn(
        op,
        device_id,
        seed,
        shape,
        dtype,
        key,
        counter,
        alg,
        minval,
        maxval,
        op_version,
        device_index_fn,
        full_replicated=full_replicated)

  return get_tpu_strategy().run(tpu_fn, args=(device_id, seed))


def rng_op_spmd_fn(op,
                   device_id,
                   seed,
                   shape,
                   dtype,
                   key,
                   counter,
                   alg,
                   minval,
                   maxval,
                   op_version,
                   device_index_fn,
                   full_replicated=False):
  if full_replicated:
    # TODO(bfontain,xiejw): Consider to make this consistent with non-replicated
    # case. Seems very confusing.
    new_seed, new_key = seed, key
  else:
    # Runs on TF2 non-DTensor pure eager. This code should align the same
    # logic in RandomOpSPMDExpander.
    x_cord = device_id // 4
    y_cord = device_id % 4
    device_index = device_index_fn(x_cord, y_cord)
    device_id_seed = device_index * 65536 + 65521
    new_seed = gen_bitwise_ops.bitwise_xor(seed, device_id_seed)
    new_key = gen_bitwise_ops.bitwise_xor(
        key, math_ops.cast(device_id_seed, dtype=dtypes.uint64)
    )
  return _call_op(
      op=op,
      seed=new_seed,
      shape=shape,
      dtype=dtype,
      key=new_key,
      counter=counter,
      alg=alg,
      minval=minval,
      maxval=maxval,
      op_version=op_version)


class DTensorRNGTest(test_util.DTensorBaseTest):

  def setUp(self):
    super(DTensorRNGTest, self).setUp()
    global_ids = test_util.create_device_ids_array((2, 4))
    local_ids = _LOCAL_IDS
    mesh_dict = {
        device: Mesh(
            [_MESH_DIM_X, _MESH_DIM_Y],
            global_ids,
            local_ids,
            test_util.create_device_list((2, 4), device),
        )
        for device in ('CPU', 'GPU', 'TPU')
    }
    self.mesh = self.configTestMesh(mesh_dict)

    # Creates a bunch of common layouts used by tests later.
    self.replicated_layout_2d = Layout.replicated(self.mesh, rank=2)
    self.shardings = {
        'batch': Layout.batch_sharded,
        'inner': Layout.inner_sharded
    }
    # Creates a bunch of parameters for rng V2 ops
    self.key = constant_op.constant([123], dtype=dtypes.uint64)
    self.counter = constant_op.constant([1, 1], dtype=dtypes.uint64)
    self.alg = 1
    self.minval = 1
    self.maxval = 100

  @parameterized.named_parameters(test_util_ops.RANDOM_OPS)
  def testStatelessRNGWithFullyReplicated(self, op, dtype, op_version):
    layout = self.replicated_layout_2d
    shape = [16, 16]
    seed = [123, 321]

    with ops.device_v2(api.device_name()):
      with api._dtensor_device()._default_layout(layout):
        b = _call_dtensor_op(
            op=op,
            seed=seed,
            shape=shape,
            dtype=dtype,
            key=self.key,
            counter=self.counter,
            alg=self.alg,
            minval=self.minval,
            maxval=self.maxval,
            op_version=op_version,
            mesh=self.mesh)

    api.check_layout(b, layout)
    self.assertListEqual(shape, list(b.shape))

    b = [tensor.numpy() for tensor in api.unpack(b)]
    for i in range(self.mesh.num_local_devices() - 1):
      self.assertAllEqual(b[i], b[i + 1])

  @parameterized.named_parameters(test_util_ops.RANDOM_OPS)
  def testStatelessRNGWithFullyReplicatedComparingWithNonDTensor(
      self, op, dtype, op_version):

    layout = self.replicated_layout_2d
    shape = [16, 16]
    seed = [123, 321]

    with ops.device_v2(api.device_name()):
      with api._dtensor_device()._default_layout(layout):
        b = _call_dtensor_op(
            op=op,
            seed=seed,
            shape=shape,
            dtype=dtype,
            key=self.key,
            counter=self.counter,
            alg=self.alg,
            minval=self.minval,
            maxval=self.maxval,
            op_version=op_version,
            mesh=self.mesh)

    api.check_layout(b, layout)
    self.assertListEqual(shape, list(b.shape))

    b = [tensor.numpy() for tensor in api.unpack(b)]

    local_shape = shape
    for index, device_id in enumerate(_LOCAL_IDS):
      self.assertAllEqual(
          b[index],
          rng_op_spmd(
              op,
              device_id,
              seed,
              local_shape,
              dtype,
              key=self.key,
              counter=self.counter,
              alg=self.alg,
              minval=self.minval,
              maxval=self.maxval,
              op_version=op_version,
              device_index_fn=None,  # not needed
              full_replicated=True,
              is_tpu=self.mesh.device_type().upper() == 'TPU'))

  @parameterized.named_parameters(
      test_util_ops.expand_test_config(
          test_util_ops.RANDOM_OPS,
          [
              {
                  'dim': _MESH_DIM_X,
                  'shard_type': 'batch',
              },
              {
                  'dim': _MESH_DIM_Y,
                  'shard_type': 'batch',
              },
              {
                  'dim': _MESH_DIM_X,
                  'shard_type': 'inner',
              },
              {'dim': _MESH_DIM_Y, 'shard_type': 'inner'},
          ],
      )
  )
  def testStatelessRNGOpsWithSingleDimensionSharded(self, op, dtype, op_version,
                                                    dim, shard_type):
    shape = [128, 128]
    seed = [123, 321]
    sharding = self.shardings[shard_type]
    layout = sharding(self.mesh, dim, rank=2)

    # Raw rng Ops do not have inputs, so we need to place the Op DTensor device
    # explicitly.
    with ops.device_v2(api.device_name()):
      with api._dtensor_device()._default_layout(layout):
        b = _call_dtensor_op(
            op=op,
            seed=seed,
            shape=shape,
            dtype=dtype,
            key=self.key,
            counter=self.counter,
            alg=self.alg,
            minval=self.minval,
            maxval=self.maxval,
            op_version=op_version,
            mesh=self.mesh)

    api.check_layout(b, layout)
    b = [tensor.numpy() for tensor in api.unpack(b)]

    if dim == _MESH_DIM_X:
      if shard_type == 'batch':
        self.assertAllEqual(b[0].shape, [64, 128])
      else:
        assert shard_type == 'inner'
        self.assertAllEqual(b[0].shape, [128, 64])

      # first check that each component is same as the row header.
      for i in range(self.mesh.num_local_devices()):
        self.assertAllEqual(b[i], b[_ROW_HEAD[i]])
      # then check the row header are NOT identital.
      self.assertNotAllEqual(b[_ROW_0_HEAD], b[_ROW_1_HEAD])

    elif dim == _MESH_DIM_Y:
      if shard_type == 'batch':
        self.assertAllEqual(b[0].shape, [32, 128])
      else:
        assert shard_type == 'inner'
        self.assertAllEqual(b[0].shape, [128, 32])

      # first check elements in same columns are identical
      for i in range(self.mesh.num_local_devices()):
        self.assertAllEqual(b[i], b[_COL_HEAD[i]])

      col_heads = [_COL_0_HEAD, _COL_1_HEAD, _COL_2_HEAD, _COL_3_HEAD]
      # then check the column header are not identital (mutually)
      for i in range(self.mesh.num_local_devices() - 1):
        for j in range(self.mesh.num_local_devices()):
          if i == j:
            continue
          if i in col_heads and j in col_heads:
            self.assertNotAllEqual(b[i], b[j])

    else:
      self.fail('should not reach here.')

  @parameterized.named_parameters(
      test_util_ops.expand_test_config(
          test_util_ops.RANDOM_OPS,
          [
              {
                  'dim': _MESH_DIM_X,
                  'shard_type': 'batch',
              },
              {
                  'dim': _MESH_DIM_Y,
                  'shard_type': 'batch',
              },
              {
                  'dim': _MESH_DIM_X,
                  'shard_type': 'inner',
              },
              {'dim': _MESH_DIM_Y, 'shard_type': 'inner'},
          ],
      )
  )
  def testStatelessRNGOpsWithSingleDimensionShardedComparingWithNonDTensor(
      self, op, dtype, op_version, dim, shard_type):

    shape = [128, 128]
    seed = [123, 321]
    sharding = self.shardings[shard_type]
    layout = sharding(self.mesh, dim, rank=2)

    # Raw rng Ops do not have inputs, so we need to place the Op DTensor device
    # explicitly.
    with ops.device_v2(api.device_name()):
      with api._dtensor_device()._default_layout(layout):
        b = _call_dtensor_op(
            op=op,
            seed=seed,
            shape=shape,
            dtype=dtype,
            key=self.key,
            counter=self.counter,
            alg=self.alg,
            minval=self.minval,
            maxval=self.maxval,
            op_version=op_version,
            mesh=self.mesh)

    api.check_layout(b, layout)
    b = [tensor.numpy() for tensor in api.unpack(b)]

    if dim == _MESH_DIM_X:
      if shard_type == 'batch':
        local_shape = [64, 128]
      else:
        local_shape = [128, 64]

      def device_index_fn(x_cord, y_cord):
        # See todo of device_index_fn in 2d sharding case.
        del y_cord
        return x_cord

      for index, device_id in enumerate(_LOCAL_IDS):
        self.assertAllEqual(
            b[index],
            rng_op_spmd(
                op,
                device_id,
                seed,
                local_shape,
                dtype,
                key=self.key,
                counter=self.counter,
                alg=self.alg,
                minval=self.minval,
                maxval=self.maxval,
                op_version=op_version,
                device_index_fn=device_index_fn,
                is_tpu=self.mesh.device_type().upper() == 'TPU'))
    elif dim == _MESH_DIM_Y:
      if shard_type == 'batch':
        local_shape = [32, 128]
      else:
        local_shape = [128, 32]

      def device_index_fn(x_cord, y_cord):
        # See todo of device_index_fn in 2d sharding case. note this case is
        # particulary interesting as 2*y_cord is more natual.
        del x_cord
        return y_cord

      for index, device_id in enumerate(_LOCAL_IDS):
        self.assertAllEqual(
            b[index],
            rng_op_spmd(
                op,
                device_id,
                seed,
                local_shape,
                dtype,
                key=self.key,
                counter=self.counter,
                alg=self.alg,
                minval=self.minval,
                maxval=self.maxval,
                op_version=op_version,
                device_index_fn=device_index_fn,
                is_tpu=self.mesh.device_type().upper() == 'TPU'))

    else:
      self.fail('should not reach here.')

  @parameterized.named_parameters(test_util_ops.RANDOM_OPS)
  def testStatelessRNGOpsWith2DSharding(self, op, dtype, op_version):
    shape = [128, 128]
    seed = [123, 321]
    layout = Layout([_MESH_DIM_Y, _MESH_DIM_X], self.mesh)

    # Raw rng Ops do not have inputs, so we need to place the Op DTensor device
    # explicitly.
    with ops.device_v2(api.device_name()):
      with api._dtensor_device()._default_layout(layout):
        b = _call_dtensor_op(
            op=op,
            seed=seed,
            shape=shape,
            dtype=dtype,
            key=self.key,
            counter=self.counter,
            alg=self.alg,
            minval=self.minval,
            maxval=self.maxval,
            op_version=op_version,
            mesh=self.mesh)

    api.check_layout(b, layout)
    b = [tensor.numpy() for tensor in api.unpack(b)]

    # check all raw components are not identital (mutually)
    for i in range(self.mesh.num_local_devices() - 1):
      for j in range(self.mesh.num_local_devices()):
        if i == j:
          continue
        self.assertNotAllEqual(b[i], b[j])

  @parameterized.named_parameters(test_util_ops.RANDOM_OPS)
  def testStatelessRNGOpsWith2DShardingComparingWithNonDTensor(
      self, op, dtype, op_version):
    shape = [128, 128]
    seed = [123, 321]
    layout = Layout([_MESH_DIM_Y, _MESH_DIM_X], self.mesh)
    local_shape = [128 // 4, 128 // 2]

    # Raw rng Ops do not have inputs, so we need to place the Op DTensor device
    # explicitly.
    with ops.device_v2(api.device_name()):
      with api._dtensor_device()._default_layout(layout):
        b = _call_dtensor_op(
            op=op,
            seed=seed,
            shape=shape,
            dtype=dtype,
            key=self.key,
            counter=self.counter,
            alg=self.alg,
            minval=self.minval,
            maxval=self.maxval,
            op_version=op_version,
            mesh=self.mesh)

    api.check_layout(b, layout)
    b = [tensor.numpy() for tensor in api.unpack(b)]

    def device_index_fn(x_cord, y_cord):
      # TODO(bfontain,xiejw): Currently, the device index is x+2y. But it is
      # more natual to use 4x+y for a mesh<x=2, y=4>. Consider to change this
      # once all correctness tests are done.
      return x_cord + 2 * y_cord

    for index, device_id in enumerate(_LOCAL_IDS):
      self.assertAllEqual(
          b[index],
          rng_op_spmd(
              op,
              device_id,
              seed,
              local_shape,
              dtype,
              key=self.key,
              counter=self.counter,
              alg=self.alg,
              minval=self.minval,
              maxval=self.maxval,
              op_version=op_version,
              device_index_fn=device_index_fn,
              is_tpu=self.mesh.device_type().upper() == 'TPU'))

  def testRNGReadAndSkip(self):
    replicated_layout = Layout.replicated(self.mesh, 1)
    a = constant_op.constant([1, 2, 3], dtype=dtypes.int64)
    v = variables.Variable(a)
    expected = gen_stateful_random_ops.rng_read_and_skip(
        resource=v.handle,
        alg=1,
        delta=constant_op.constant(1, dtype=dtypes.uint64),
    )

    a = numpy_util.pack_numpy(a, replicated_layout)
    v = d_variable.DVariable(a)
    got = gen_stateful_random_ops.rng_read_and_skip(
        resource=v.handle,
        alg=1,
        delta=constant_op.constant(1, dtype=dtypes.uint64),
    )

    self.assertDTensorEqual(expected, replicated_layout, got)

  def testStatelessRandomGetKeyCounter(self):
    seed = constant_op.constant([7, 17], dtypes.int32)

    # TPU computation result is different from CPU computation.
    # We force it to run on the TPU using tpu_strategy for TPU mesh
    # so that we compare equal values.
    @polymorphic_function.function
    def tpu_fn():
      return gen_stateless_random_ops_v2.stateless_random_get_key_counter(
          seed=seed
      )

    if self.mesh.device_type().upper() == 'TPU':
      expected = get_tpu_strategy().run(tpu_fn)
    else:
      expected = gen_stateless_random_ops_v2.stateless_random_get_key_counter(
          seed=seed
      )

    replicated_1d_layout = Layout.replicated(self.mesh, 1)
    seed = numpy_util.pack_numpy(seed, replicated_1d_layout)

    got = gen_stateless_random_ops_v2.stateless_random_get_key_counter(
        seed=seed
    )
    self.assertDTensorEqual(expected[0], replicated_1d_layout, got[0])
    self.assertDTensorEqual(expected[1], replicated_1d_layout, got[1])


if __name__ == '__main__':
  test.main()
