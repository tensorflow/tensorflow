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
"""Tests for configure.py CUDA compute capability helpers."""

import unittest

import configure


class CudaComputeCapabilityTest(unittest.TestCase):

  def test_parse_decimal_form(self):
    self.assertEqual(configure.cuda_compute_capability_version('7.5'), (7, 5))
    self.assertEqual(configure.cuda_compute_capability_version('12.0'), (12, 0))

  def test_parse_sm_and_compute_form(self):
    self.assertEqual(configure.cuda_compute_capability_version('sm_80'), (8, 0))
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

  def test_parse_invalid(self):
    self.assertIsNone(configure.cuda_compute_capability_version(''))
    self.assertIsNone(configure.cuda_compute_capability_version('gpu'))
    self.assertIsNone(configure.cuda_compute_capability_version('sm_'))

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

  def test_no_nvcc_required_for_pre_blackwell(self):
    self.assertFalse(
        configure.compute_capabilities_require_nvcc('sm_60,sm_70,sm_80,sm_89')
    )
    self.assertFalse(
        configure.compute_capabilities_require_nvcc('7.5,8.0,9.0')
    )
    self.assertFalse(configure.compute_capabilities_require_nvcc(''))
    self.assertFalse(configure.compute_capabilities_require_nvcc(None))


if __name__ == '__main__':
  unittest.main()
