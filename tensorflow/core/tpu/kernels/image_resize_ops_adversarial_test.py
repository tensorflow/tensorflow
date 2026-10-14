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
"""Adversarial stress tests for TPU ResizeNearestNeighbor op gather lowering."""

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
    'Device to run tests on: `tpu` for TPU hardware, `cpu` for local CPU.',
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


class ImageResizeOpsAdversarialTest(googletest.TestCase):

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
      logging.info('Initialized CPU strategy.')

  def _run_resize(
      self,
      images: Any,
      size: list[int],
      align_corners: bool,
      half_pixel_centers: bool,
  ) -> tuple[np.ndarray, np.ndarray]:
    with ops.device('/CPU:0'):
      cpu_images = array_ops.identity(images)
      cpu_out = image_ops.resize_nearest_neighbor(
          images=cpu_images,
          size=size,
          align_corners=align_corners,
          half_pixel_centers=half_pixel_centers,
      )

    @def_function.function
    def _step(img):
      return image_ops.resize_nearest_neighbor(
          images=img,
          size=size,
          align_corners=align_corners,
          half_pixel_centers=half_pixel_centers,
      )

    tpu_out = self.strategy.run(_step, args=(images,))
    if hasattr(tpu_out, 'values'):
      tpu_out = tpu_out.values[0]

    return cpu_out.numpy(), tpu_out.numpy()

  def test_adversarial_shapes_and_modes_bit_exactness(self):
    """Stress tests bit-exactness across boundary shapes and dtypes."""
    rng = np.random.RandomState(1337)

    modes = [
        (False, False),
        (True, False),
        (False, True),
    ]

    shapes_and_sizes = [
        # Flare production resize:
        ([2, 688, 1520, 3], [459, 1013], 'flare_exact'),
        # Extreme aspect ratios:
        ([2, 1000, 3, 2], [537, 7], 'tall_skinny'),
        ([2, 3, 1000, 2], [7, 537], 'wide_short'),
        # Single spatial dimension input:
        ([2, 1, 100, 2], [50, 150], 'in_h_1_out_h_50'),
        ([2, 100, 1, 2], [150, 50], 'in_w_1_out_w_50'),
        # Odd and even downsamplings:
        ([2, 7, 7, 2], [3, 3], 'odd_to_odd_down'),
        ([2, 8, 8, 2], [6, 6], 'even_to_even_nonint_down'),
        ([2, 9, 9, 2], [4, 4], 'odd_to_even_down'),
        ([2, 10, 10, 2], [7, 7], 'even_to_odd_down'),
        # Prime dimensions:
        ([2, 17, 19, 2], [23, 29], 'primes_up'),
        ([2, 31, 37, 2], [11, 13], 'primes_down'),
        # Extreme scaling factors:
        ([2, 256, 256, 2], [3, 3], 'extreme_downsample'),
        ([2, 3, 3, 2], [127, 127], 'extreme_upsample'),
        # Mixed scale (one axis integer ratio, one axis non-integer):
        ([2, 50, 688, 2], [100, 459], 'mixed_int2x_h_nonint_w'),
        ([2, 688, 50, 2], [459, 100], 'mixed_nonint_h_int2x_w'),
        ([2, 100, 688, 2], [50, 459], 'mixed_int_half_h_nonint_w'),
        # Mixed scale with identity along one axis:
        ([2, 100, 688, 2], [100, 459], 'mixed_identity_h_nonint_w'),
        ([2, 688, 100, 2], [459, 100], 'mixed_nonint_h_identity_w'),
        # Fast paths:
        ([2, 1, 1, 3], [32, 32], '1x1_input'),
        ([2, 64, 64, 3], [64, 64], 'identity'),
        # Integer scales (custom call preservation):
        ([2, 50, 50, 3], [100, 100], 'int_upsample_2x'),
        ([2, 100, 100, 3], [50, 50], 'int_downsample_2x'),
        ([2, 30, 30, 3], [90, 90], 'int_upsample_3x'),
        ([2, 90, 90, 3], [30, 30], 'int_downsample_3x'),
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
              mode=f'ac={align_corners},hpc={half_pixel_centers}',
              shape=f'{in_shape}->{out_size}',
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
                f'Mismatch for mode=(ac={align_corners},'
                f' hpc={half_pixel_centers}), shape={in_shape}->{out_size}'
                f' ({desc}), dtype={tf_dtype.name}:'
                f' max_abs_diff={max_abs_diff}',
            )

  def _make_hlo_step_fn(self, size: list[int]):
    @def_function.function
    def _step(img):
      return image_ops.resize_nearest_neighbor(
          images=img,
          size=size,
          align_corners=False,
          half_pixel_centers=False,
      )

    return _step

  def _run_hlo_sweep(self, enable_mlir: bool):
    if _BENCHMARK_DEVICE.value != 'tpu':
      self.skipTest('HLO checks require TPU device.')

    if enable_mlir:
      config.enable_mlir_bridge()
    else:
      config.disable_mlir_bridge()

    test_specs = [
        # (in_shape, out_size, expect_custom_call, min_gathers, desc)
        ([2, 688, 1520, 3], [459, 1013], False, 2, 'flare_non_integer'),
        ([2, 50, 688, 2], [100, 459], False, 2, 'mixed_int2x_h_nonint_w'),
        ([2, 100, 688, 2], [100, 459], False, 1, 'mixed_identity_h_nonint_w'),
        ([2, 50, 50, 4], [100, 100], True, 0, 'integer_upsample_2x'),
        ([2, 100, 100, 4], [50, 50], True, 0, 'integer_downsample_2x'),
        ([2, 64, 64, 4], [64, 64], False, 0, 'identity'),
        ([2, 1, 1, 4], [32, 32], False, 0, '1x1_input'),
    ]

    for in_shape, out_size, expect_custom_call, min_gathers, desc in test_specs:
      with self.subTest(
          desc=desc,
          shape=f'{in_shape}->{out_size}',
          expect_custom_call=expect_custom_call,
          mlir=enable_mlir,
      ):
        before_files = _snapshot_hlo_files()
        _step = self._make_hlo_step_fn(out_size)

        x = array_ops.ones(in_shape, dtype=dtypes.float32)
        _ = self.strategy.run(_step, args=(x,))
        counts = _count_hlo_instructions(before_files)
        logging.info(
            'HLO counts for %s (mlir=%s): %s', desc, enable_mlir, counts
        )

        if expect_custom_call:
          self.assertGreater(
              counts['resize_nearest_custom_calls'],
              0,
              f'{desc}: Expected >= 1 ResizeNearest custom calls with'
              f' mlir={enable_mlir}, got {counts}',
          )
        else:
          self.assertEqual(
              counts['resize_nearest_custom_calls'],
              0,
              f'{desc}: Expected 0 ResizeNearest custom calls with'
              f' mlir={enable_mlir}, got {counts}',
          )

        if min_gathers > 0:
          self.assertGreaterEqual(
              counts['gather_ops'],
              min_gathers,
              f'{desc}: Expected >= {min_gathers} gather ops with'
              f' mlir={enable_mlir}, got {counts}',
          )

  def test_hlo_lowering_deep_sweep_mlir_bridge_off(self):
    self._run_hlo_sweep(enable_mlir=False)

  def test_hlo_lowering_deep_sweep_mlir_bridge_on(self):
    self._run_hlo_sweep(enable_mlir=True)


if __name__ == '__main__':
  googletest.main()
