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
"""Tests for configure.py CUDA / Blackwell helpers."""

import unittest
from unittest import mock

import configure


class CudaComputeCapabilityTest(unittest.TestCase):

  def test_parse_decimal_form(self):
    self.assertEqual(configure.cuda_compute_capability_version('7.5'), (7, 5))
    self.assertEqual(configure.cuda_compute_capability_version('9.0'), (9, 0))
    self.assertEqual(configure.cuda_compute_capability_version('10.0'), (10, 0))
    self.assertEqual(configure.cuda_compute_capability_version('12.0'), (12, 0))

  def test_parse_strips_whitespace(self):
    self.assertEqual(
        configure.cuda_compute_capability_version('  sm_80  '), (8, 0)
    )

  def test_parse_sm_and_compute_form(self):
    self.assertEqual(configure.cuda_compute_capability_version('sm_80'), (8, 0))
    self.assertEqual(configure.cuda_compute_capability_version('sm_89'), (8, 9))
    self.assertEqual(
        configure.cuda_compute_capability_version('compute_90'), (9, 0)
    )
    self.assertEqual(
        configure.cuda_compute_capability_version('sm_100'), (10, 0)
    )
    self.assertEqual(
        configure.cuda_compute_capability_version('sm_120'), (12, 0)
    )
    self.assertEqual(
        configure.cuda_compute_capability_version('compute_120'), (12, 0)
    )
    self.assertEqual(
        configure.cuda_compute_capability_version('sm_100a'), (10, 0)
    )
    self.assertEqual(
        configure.cuda_compute_capability_version('sm_120a'), (12, 0)
    )
    self.assertEqual(
        configure.cuda_compute_capability_version('sm_100f'), (10, 0)
    )

  def test_parse_invalid(self):
    self.assertIsNone(configure.cuda_compute_capability_version(''))
    self.assertIsNone(configure.cuda_compute_capability_version(None))
    self.assertIsNone(configure.cuda_compute_capability_version('gpu'))
    self.assertIsNone(configure.cuda_compute_capability_version('sm_'))
    self.assertIsNone(configure.cuda_compute_capability_version('compute'))

  def test_require_nvcc_for_blackwell(self):
    self.assertTrue(
        configure.compute_capabilities_require_nvcc('sm_120,compute_120')
    )
    self.assertTrue(configure.compute_capabilities_require_nvcc('sm_100'))
    self.assertTrue(
        configure.compute_capabilities_require_nvcc(
            'sm_75,sm_80,sm_90,sm_100,compute_120'
        )
    )
    self.assertTrue(configure.compute_capabilities_require_nvcc('12.0'))
    self.assertTrue(configure.compute_capabilities_require_nvcc('10.0'))
    # Whitespace and empty tokens
    self.assertTrue(
        configure.compute_capabilities_require_nvcc(' sm_80 , sm_120 ')
    )
    self.assertTrue(configure.compute_capabilities_require_nvcc('sm_120,'))

  def test_no_nvcc_required_for_pre_blackwell(self):
    self.assertFalse(
        configure.compute_capabilities_require_nvcc('sm_60,sm_70,sm_80,sm_89')
    )
    self.assertFalse(
        configure.compute_capabilities_require_nvcc('7.5,8.0,9.0')
    )
    self.assertFalse(configure.compute_capabilities_require_nvcc('sm_90'))
    self.assertFalse(configure.compute_capabilities_require_nvcc('compute_90'))
    self.assertFalse(configure.compute_capabilities_require_nvcc(''))
    self.assertFalse(configure.compute_capabilities_require_nvcc(None))
    # Invalid tokens alone do not force nvcc
    self.assertFalse(configure.compute_capabilities_require_nvcc('gpu,sm_'))

  def test_clang_cuda_compiler_default(self):
    self.assertTrue(configure.clang_cuda_compiler_default('sm_80,sm_90'))
    self.assertTrue(configure.clang_cuda_compiler_default(''))
    self.assertFalse(configure.clang_cuda_compiler_default('sm_120'))
    self.assertFalse(
        configure.clang_cuda_compiler_default('sm_75,sm_100,compute_120')
    )


class MaybeEnsureCuda13ForBlackwellTest(unittest.TestCase):

  def test_no_op_for_pre_blackwell(self):
    environ = {
        'HERMETIC_CUDA_COMPUTE_CAPABILITIES': 'sm_80,sm_90',
    }
    with mock.patch.object(configure, 'write_repo_env_to_bazelrc') as write_env:
      pinned = configure.maybe_ensure_cuda13_for_blackwell(environ)
    self.assertFalse(pinned)
    self.assertNotIn('HERMETIC_CUDA_VERSION', environ)
    write_env.assert_not_called()

  def test_pins_cuda13_when_blackwell_and_version_unset(self):
    environ = {
        'HERMETIC_CUDA_COMPUTE_CAPABILITIES': 'sm_120,compute_120',
    }
    with mock.patch.object(configure, 'write_repo_env_to_bazelrc') as write_env:
      pinned = configure.maybe_ensure_cuda13_for_blackwell(environ)
    self.assertTrue(pinned)
    self.assertEqual(
        environ['HERMETIC_CUDA_VERSION'],
        configure._BLACKWELL_HERMETIC_CUDA_VERSION,
    )
    self.assertEqual(
        environ['HERMETIC_CUDNN_VERSION'],
        configure._BLACKWELL_HERMETIC_CUDNN_VERSION,
    )
    write_env.assert_any_call(
        'cuda',
        'HERMETIC_CUDA_VERSION',
        configure._BLACKWELL_HERMETIC_CUDA_VERSION,
    )
    write_env.assert_any_call(
        'cuda',
        'HERMETIC_CUDNN_VERSION',
        configure._BLACKWELL_HERMETIC_CUDNN_VERSION,
    )

  def test_does_not_override_explicit_cuda_version(self):
    environ = {
        'HERMETIC_CUDA_COMPUTE_CAPABILITIES': 'sm_120',
        'HERMETIC_CUDA_VERSION': '12.5.1',
        'HERMETIC_CUDNN_VERSION': '9.3.0',
    }
    with mock.patch.object(configure, 'write_repo_env_to_bazelrc') as write_env:
      pinned = configure.maybe_ensure_cuda13_for_blackwell(environ)
    self.assertFalse(pinned)
    self.assertEqual(environ['HERMETIC_CUDA_VERSION'], '12.5.1')
    self.assertEqual(environ['HERMETIC_CUDNN_VERSION'], '9.3.0')
    write_env.assert_not_called()

  def test_pins_only_missing_cudnn_when_cuda_set(self):
    environ = {
        'HERMETIC_CUDA_COMPUTE_CAPABILITIES': 'sm_100',
        'HERMETIC_CUDA_VERSION': '13.0.0',
    }
    with mock.patch.object(configure, 'write_repo_env_to_bazelrc') as write_env:
      pinned = configure.maybe_ensure_cuda13_for_blackwell(environ)
    self.assertTrue(pinned)
    self.assertEqual(environ['HERMETIC_CUDA_VERSION'], '13.0.0')
    self.assertEqual(
        environ['HERMETIC_CUDNN_VERSION'],
        configure._BLACKWELL_HERMETIC_CUDNN_VERSION,
    )
    write_env.assert_called_once_with(
        'cuda',
        'HERMETIC_CUDNN_VERSION',
        configure._BLACKWELL_HERMETIC_CUDNN_VERSION,
    )


class SetOtherCudaVarsTest(unittest.TestCase):

  def test_clang_writes_cuda_clang(self):
    environ = {
        'TF_CUDA_CLANG': '1',
        'HERMETIC_CUDA_COMPUTE_CAPABILITIES': 'sm_80',
    }
    with mock.patch.object(configure, 'write_to_bazelrc') as write_rc:
      configure.set_other_cuda_vars(environ)
    write_rc.assert_called_once_with('build --config=cuda_clang')

  def test_pre_blackwell_nvcc_path_preserves_config_cuda_only(self):
    environ = {
        'TF_CUDA_CLANG': '0',
        'HERMETIC_CUDA_COMPUTE_CAPABILITIES': 'sm_80,sm_90',
    }
    with mock.patch.object(configure, 'write_to_bazelrc') as write_rc:
      configure.set_other_cuda_vars(environ)
    write_rc.assert_called_once_with('build --config=cuda')

  def test_blackwell_writes_cuda_nvcc(self):
    environ = {
        'TF_CUDA_CLANG': '0',
        'HERMETIC_CUDA_COMPUTE_CAPABILITIES': 'sm_120,compute_120',
    }
    with mock.patch.object(configure, 'write_to_bazelrc') as write_rc:
      configure.set_other_cuda_vars(environ)
    write_rc.assert_called_once_with('build --config=cuda_nvcc')


class RecommendedGpuWheelFlagsTest(unittest.TestCase):

  def test_blackwell_recommends_cuda13_and_nvcc(self):
    environ = {
        'HERMETIC_CUDA_COMPUTE_CAPABILITIES': 'sm_120',
        'TF_CUDA_CLANG': '0',
    }
    flags = configure.recommended_gpu_wheel_bazel_flags(environ)
    self.assertEqual(
        flags,
        [
            '--config=cuda',
            '--config=cuda_wheel',
            '--config=cuda13_version',
            '--config=cuda_nvcc',
        ],
    )
    self.assertNotIn('--config=cuda13_nvcc', flags)

  def test_pre_blackwell_clang(self):
    environ = {
        'HERMETIC_CUDA_COMPUTE_CAPABILITIES': 'sm_80',
        'TF_CUDA_CLANG': '1',
    }
    flags = configure.recommended_gpu_wheel_bazel_flags(environ)
    self.assertEqual(
        flags,
        ['--config=cuda', '--config=cuda_wheel', '--config=cuda_clang'],
    )

  def test_pre_blackwell_nvcc(self):
    environ = {
        'HERMETIC_CUDA_COMPUTE_CAPABILITIES': 'sm_80',
        'TF_CUDA_CLANG': '0',
    }
    flags = configure.recommended_gpu_wheel_bazel_flags(environ)
    self.assertEqual(
        flags,
        ['--config=cuda', '--config=cuda_wheel', '--config=cuda_nvcc'],
    )


class BlackwellDoesNotBreakCuda12DefaultsTest(unittest.TestCase):
  """Guards that CUDA 12 defaults stay intact for non-Blackwell configures."""

  def test_cuda12_capability_list_does_not_require_nvcc(self):
    # Mirrors .bazelrc common:cuda12_version / common:cuda_clang SM lists.
    cuda12_caps = 'sm_60,sm_70,sm_80,sm_89,compute_90'
    self.assertFalse(configure.compute_capabilities_require_nvcc(cuda12_caps))
    self.assertTrue(configure.clang_cuda_compiler_default(cuda12_caps))

  def test_cuda13_default_list_requires_nvcc(self):
    # Mirrors .bazelrc common:cuda13_version SM list (includes Blackwell).
    cuda13_caps = 'sm_75,sm_80,sm_90,sm_100,compute_120'
    self.assertTrue(configure.compute_capabilities_require_nvcc(cuda13_caps))
    self.assertFalse(configure.clang_cuda_compiler_default(cuda13_caps))

  def test_blackwell_pin_versions_match_cuda13_version_config(self):
    self.assertEqual(configure._BLACKWELL_HERMETIC_CUDA_VERSION, '13.0.0')
    self.assertEqual(configure._BLACKWELL_HERMETIC_CUDNN_VERSION, '9.12.0')


if __name__ == '__main__':
  unittest.main()
