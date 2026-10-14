# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for TPU ResizeNearestNeighbor op gather lowering and numerics."""

import glob
import os
import re
import tempfile
from typing import Any

from absl import flags
from absl import logging
from absl.testing import absltest as googletest
import numpy as np

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.eager import def_function
from tensorflow.python.eager import remote
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.tpu import device_assignment as device_assignment_lib
from tensorflow.python.tpu import tpu_strategy_util
# pylint: enable=g-direct-tensorflow-import

ops.enable_eager_execution()

# Configure XLA_FLAGS so XLA dumps all compiled HLO modules as text.
_OUTPUT_DIR = os.environ.get(
    'TEST_UNDECLARED_OUTPUTS_DIR', tempfile.gettempdir()
)
_BASE_DUMP_DIR = os.path.join(_OUTPUT_DIR, 'xla_hlo_dump')
os.makedirs(_BASE_DUMP_DIR, exist_ok=True)
_EXISTING_XLA_FLAGS = os.environ.get('XLA_FLAGS', '')
if '--xla_dump_to=' not in _EXISTING_XLA_FLAGS:
  os.environ['XLA_FLAGS'] = (
      f'{_EXISTING_XLA_FLAGS} '
      f'--xla_dump_to={_BASE_DUMP_DIR} '
      '--xla_dump_hlo_as_text'
  ).strip()


_BENCHMARK_DEVICE = flags.DEFINE_enum(
    'benchmark_device',
    'tpu',
    ['tpu', 'cpu'],
    'Device to run tests on: `tpu` for TPU hardware, `cpu` for local CPU'
    ' smoke.',
)


def _snapshot_hlo_files() -> set[str]:
  return set(
      f
      for f in glob.glob(os.path.join(_BASE_DUMP_DIR, '*.txt'))
      if f.endswith('after_optimizations.txt')
      and 'host' not in os.path.basename(f)
  )


def _count_hlo_instructions(before_files: set[str]) -> dict[str, Any]:
  """Counts ResizeNearest custom calls and gathers in newly dumped HLOs."""
  after_files = _snapshot_hlo_files()
  new_files = sorted(after_files - before_files)
  counts = {
      'num_files': len(new_files),
      'resize_nearest_custom_calls': 0,
      'gather_ops': 0,
      'files': [os.path.basename(f) for f in new_files],
  }
  inst_re = re.compile(r'^\s*(?:ROOT\s+)?%[a-zA-Z0-9_.-]+\s*=')
  gather_re = re.compile(r'=\s*[a-z0-9]+\[[^\]]*\](?:\{[^}]*\})?\s*gather\(')
  for fpath in new_files:
    with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
      for line in f:
        if not inst_re.match(line):
          continue
        if 'custom_call_target="ResizeNearest"' in line:
          counts['resize_nearest_custom_calls'] += 1
        elif gather_re.search(line):
          counts['gather_ops'] += 1
  return counts


class ImageResizeOpsTpuTest(googletest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    config.set_soft_device_placement(True)
    if _BENCHMARK_DEVICE.value == 'tpu':
      resolver = tpu_cluster_resolver.TPUClusterResolver(tpu='')
      remote.connect_to_cluster(resolver)
      topology = tpu_strategy_util.initialize_tpu_system_impl(
          resolver, tpu_cluster_resolver.TPUClusterResolver
      )
      device_assignment = device_assignment_lib.DeviceAssignment.build(
          topology, num_replicas=1
      )
      cls.strategy = tpu_strategy.TPUStrategyV2(
          resolver, experimental_device_assignment=device_assignment
      )
      cls.tpu_devices = config.list_logical_devices('TPU')
      logging.info('Initialized TPU devices: %s', cls.tpu_devices)
    else:
      cls.strategy = distribute_lib.get_strategy()
      cls.tpu_devices = []
      logging.info('Initialized CPU strategy for smoke testing.')

  def _run_resize(
      self,
      images: Any,
      size: list[int],
      align_corners: bool,
      half_pixel_centers: bool,
  ) -> tuple[np.ndarray, np.ndarray]:
    """Runs ResizeNearestNeighbor on CPU and TPU, returning both arrays."""
    # Compute CPU reference.
    with ops.device('/CPU:0'):
      cpu_images = array_ops.identity(images)
      cpu_out = image_ops.resize_nearest_neighbor(
          images=cpu_images,
          size=size,
          align_corners=align_corners,
          half_pixel_centers=half_pixel_centers,
      )

    # Compute TPU output.
    @def_function.function
    def _step(img):
      return image_ops.resize_nearest_neighbor(
          images=img,
          size=size,
          align_corners=align_corners,
          half_pixel_centers=half_pixel_centers,
      )

    tpu_out = self.strategy.run(_step, args=(images,))
    # Unwrap per-replica value if distributed.
    if hasattr(tpu_out, 'values'):
      tpu_out = tpu_out.values[0]

    return cpu_out.numpy(), tpu_out.numpy()

  def test_bit_exactness_all_modes_and_dtypes(self):
    """Verifies max abs diff == 0 vs CPU across all modes and dtypes."""
    rng = np.random.RandomState(42)

    modes = [
        # (align_corners, half_pixel_centers)
        (False, False),
        (True, False),
        (False, True),
    ]

    shapes_and_sizes = [
        # (input_shape, output_size, description)
        ([8, 688, 1520, 3], [459, 1013], 'flare_non_integer_downsample'),
        ([2, 100, 150, 4], [150, 225], 'non_integer_upsample_1.5x'),
        ([2, 100, 100, 4], [200, 150], 'mixed_scale_h2x_w1.5x'),
        ([2, 50, 50, 4], [100, 100], 'integer_upsample_2x'),
        ([2, 100, 100, 4], [50, 50], 'integer_downsample_2x'),
        ([2, 64, 64, 4], [64, 64], 'identity'),
        ([2, 1, 1, 4], [32, 32], '1x1_input'),
    ]

    dtypes_list = [
        (dtypes.float32, np.float32),
        (dtypes.bfloat16, dtypes.bfloat16.as_numpy_dtype),
        (dtypes.int32, np.int32),
    ]

    for align_corners, half_pixel_centers in modes:
      for in_shape, out_size, desc in shapes_and_sizes:
        for tf_dtype, np_dtype in dtypes_list:
          with self.subTest(
              align_corners=align_corners,
              half_pixel_centers=half_pixel_centers,
              shape=in_shape,
              out_size=out_size,
              desc=desc,
              dtype=tf_dtype.name,
          ):
            if tf_dtype == dtypes.int32:
              np_data = rng.randint(0, 1000, size=in_shape).astype(np_dtype)
            else:
              np_data = rng.uniform(0.0, 255.0, size=in_shape).astype(np_dtype)

            images = constant_op.constant(np_data, dtype=tf_dtype)
            cpu_arr, tpu_arr = self._run_resize(
                images, out_size, align_corners, half_pixel_centers
            )

            cpu_f64 = cpu_arr.astype(np.float64)
            tpu_f64 = tpu_arr.astype(np.float64)
            max_abs_diff = float(np.max(np.abs(cpu_f64 - tpu_f64)))
            self.assertEqual(
                max_abs_diff,
                0.0,
                f'Mismatch for mode=({align_corners}, {half_pixel_centers}), '
                f'shape={in_shape}->{out_size}, dtype={tf_dtype.name}: '
                f'max_abs_diff={max_abs_diff}',
            )

  def _test_hlo_lowering_helper(self, enable_mlir: bool):
    if _BENCHMARK_DEVICE.value != 'tpu':
      self.skipTest('HLO checks require TPU device.')

    if enable_mlir:
      config.enable_mlir_bridge()
    else:
      config.disable_mlir_bridge()

    # Case 1: Non-integer scale ([688, 1520] -> [459, 1013]).
    # Must lower to gathers: 0 ResizeNearest custom calls.
    before_files = _snapshot_hlo_files()

    @def_function.function
    def _step_non_integer(img):
      return image_ops.resize_nearest_neighbor(
          images=img,
          size=[459, 1013],
          align_corners=False,
          half_pixel_centers=False,
      )

    x_non_int = array_ops.ones([2, 688, 1520, 3], dtype=dtypes.float32)
    _ = self.strategy.run(_step_non_integer, args=(x_non_int,))
    counts_non_int = _count_hlo_instructions(before_files)
    logging.info(
        'HLO counts for non-integer (mlir=%s): %s',
        enable_mlir,
        counts_non_int,
    )
    self.assertEqual(
        counts_non_int['resize_nearest_custom_calls'],
        0,
        f'Non-integer scale had {counts_non_int["resize_nearest_custom_calls"]}'
        f' ResizeNearest custom calls (expected 0) with mlir={enable_mlir}.',
    )
    self.assertGreater(
        counts_non_int['gather_ops'],
        0,
        f'Non-integer scale had 0 gather ops with mlir={enable_mlir}.',
    )

    # Case 2: Integer ratio ([50, 50] -> [100, 100]).
    # Must keep ResizeNearest custom call: >= 1 ResizeNearest custom calls.
    before_files = _snapshot_hlo_files()

    @def_function.function
    def _step_integer(img):
      return image_ops.resize_nearest_neighbor(
          images=img,
          size=[100, 100],
          align_corners=False,
          half_pixel_centers=False,
      )

    x_int = array_ops.ones([2, 50, 50, 4], dtype=dtypes.float32)
    _ = self.strategy.run(_step_integer, args=(x_int,))
    counts_int = _count_hlo_instructions(before_files)
    logging.info(
        'HLO counts for integer ratio (mlir=%s): %s', enable_mlir, counts_int
    )
    self.assertGreater(
        counts_int['resize_nearest_custom_calls'],
        0,
        'Integer ratio had 0 ResizeNearest custom calls with'
        f' mlir={enable_mlir}.',
    )

  def test_hlo_mlir_bridge_off(self):
    """Verifies HLO lowering with MLIR bridge disabled."""
    self._test_hlo_lowering_helper(enable_mlir=False)

  def test_hlo_mlir_bridge_on(self):
    """Verifies HLO lowering with MLIR bridge enabled."""
    self._test_hlo_lowering_helper(enable_mlir=True)


if __name__ == '__main__':
  googletest.main()
