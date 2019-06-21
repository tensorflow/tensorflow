/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_MATCHER_PREDICATES_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_MATCHER_PREDICATES_H_

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"

namespace xla {

class HloInstruction;

namespace poplarplugin {

bool IsRandomNormal(const HloInstruction*);
bool IsRandomUniform(const HloInstruction*);
bool IsCompareEqual(const HloInstruction*);
bool IsConstantZero(const HloInstruction*);
bool IsExternalPadding(const HloInstruction*);
bool IsScalar(const HloInstruction*);
bool IsScalarConstant(const HloInstruction*);
bool IsScalarIntegerConstant(const HloInstruction*);
bool IsConvFilterTranspose(const HloInstruction*);
bool IsBiasReduce(const HloInstruction*);
bool IsOutputFeed(const HloInstruction*);
bool Is1DVector(const HloInstruction*);
bool IsExpandingReshape(const HloInstruction*);
bool IsF16(const HloInstruction*);
bool IsF32(const HloInstruction*);
bool IsF32ToF16Convert(const HloInstruction*);
bool IsF16ToF32Convert(const HloInstruction*);
bool IsPopOpsConvolution(const HloInstruction*);
bool IsPopOpsConvolutionWithReverse(const HloInstruction*);
bool IsOpWithWindowNoBaseDilation(const HloInstruction*);
bool IsOpWithWindowNoStride(const HloInstruction*);
bool IsPaddingReduceWindow(const HloInstruction*);
bool IsAddOrSubtract(const HloInstruction*);
bool IsBiasAdd(const HloInstruction*);
bool IsPopOpsBiasAdd(const xla::HloInstruction*);
bool IsPopOpsElementwise(const xla::HloInstruction*);
bool IsPopOpsElementwiseBinary(const xla::HloInstruction*);
bool IsNormInference(const xla::HloInstruction*);
bool IsNormTraining(const xla::HloInstruction*);
bool IsNormInferenceOrTraining(const xla::HloInstruction*);
bool IsNormGradient(const xla::HloInstruction*);
bool IsNonLinearityGradient(const xla::HloInstruction*);
bool IsNonLinearity(const xla::HloInstruction*);
bool IsSupportedAllReduce(const HloInstruction*);
template <typename T>
bool IsInstructionType(const HloInstruction* inst) {
  return DynCast<T>(inst);
}
}  // namespace poplarplugin
}  // namespace xla

#endif
