# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""RISC operation gradient."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops


@ops.RegisterGradient("RiscAdd")
def _RiscAddGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/171294012): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscBroadcast")
def _RiscBroadcastGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/171294012): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscConcat")
def _RiscConcatGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/171294012): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscConv")
def _RiscConvGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/171294012): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscDot")
def _RiscDotGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/171294012): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscMax")
def _RiscMaxGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/171294012): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscPad")
def _RiscPadGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/171294012): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscPool")
def _RiscPoolGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/171294012): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscReshape")
def _RiscReshapeGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/171294012): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscShape")
def _RiscShapeGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/171294012): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscSlice")
def _RiscSliceGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/171294012): Implement gradient of RISC with RISC ops.
  return None, None
