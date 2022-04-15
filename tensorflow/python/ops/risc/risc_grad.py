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

from tensorflow.python.framework import ops


@ops.RegisterGradient("RiscAbs")
def _RiscAbsGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/178234771): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscAdd")
def _RiscAddGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/178234771): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscBinaryArithmetic")
def _RiscBinaryArithmeticGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/178234771): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscBinaryComparison")
def _RiscBinaryComparisonGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/178234771): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscBitcast")
def _RiscBitcastGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/178234771): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscBroadcast")
def _RiscBroadcastGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/178234771): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscCast")
def _RiscCastGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/178234771): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscCholesky")
def _RiscCholeskyGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/178234771): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscCeil")
def _RiscCeilGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/178234771): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscConcat")
def _RiscConcatGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/178234771): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscCondition")
def _RiscConditionGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/178234771): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscConv")
def _RiscConvGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/178234771): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscCos")
def _RiscCosGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/178234771): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscDiv")
def _RiscDivGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/178234771): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscDot")
def _RiscDotGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/178234771): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscExp")
def _RiscExpGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/178234771): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscFft")
def _RiscFftGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/178234771): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscFloor")
def _RiscFloorGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/178234771): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscGather")
def _RiscGatherGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/178234771): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscImag")
def _RiscImagGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/178234771): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscIsFinite")
def _RiscIsFiniteGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/178234771): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscLog")
def _RiscLogGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/178234771): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscLogicalAnd")
def _RiscLogicalAndGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/178234771): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscLogicalNot")
def _RiscLogicalNotGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/178234771): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscLogicalOr")
def _RiscLogicalOrGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/178234771): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscMax")
def _RiscMaxGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/178234771): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscMin")
def _RiscMinGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/178234771): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscMul")
def _RiscMulGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/178234771): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscNeg")
def _RiscNegGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/178234771): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscPad")
def _RiscPadGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/178234771): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscPool")
def _RiscPoolGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/178234771): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscPow")
def _RiscPowGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/178234771): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscRandomUniform")
def _RiscRandomUniformGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/178234771): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscReal")
def _RiscRealGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/178234771): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscReduce")
def _RiscReduceGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/178234771): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscRem")
def _RiscRemGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/178234771): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscReshape")
def _RiscReshapeGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/178234771): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscReverse")
def _RiscReverseGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/178234771): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscScatter")
def _RiscScatterGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/178234771): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscShape")
def _RiscShapeGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/178234771): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscSign")
def _RiscSignGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/178234771): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscSlice")
def _RiscSliceGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/178234771): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscSort")
def _RiscSortGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/178234771): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscSqueeze")
def _RiscSqueezeGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/178234771): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscSub")
def _RiscSubGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/178234771): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscTranspose")
def _RiscTransposeGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/178234771): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscTriangularSolve")
def _RiscTriangularSolvesGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/178234771): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscUnary")
def _RiscUnaryGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/178234771): Implement gradient of RISC with RISC ops.
  return None, None


@ops.RegisterGradient("RiscWhile")
def _RiscWhileGrad(_, grad):
  # pylint: disable=unused-argument
  # TODO(b/178234771): Implement gradient of RISC with RISC ops.
  return None, None
