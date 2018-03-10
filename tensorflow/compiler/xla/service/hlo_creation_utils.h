/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_CREATION_UTILS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_CREATION_UTILS_H_

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

// Some lightweight utilities intended to make HLO instruction creation more
// ergonomic.  We don't have a complete set of helpers yet -- I expect we'll
// expand this interface as needed on an ad-hoc basis.

// Creates a binary HLO instruction and adds it to the computation containing
// `lhs` and `rhs` (`lhs` and `rhs` must be in the same computation).
StatusOr<HloInstruction*> CreateBinaryHlo(HloOpcode opcode, HloInstruction* lhs,
                                          HloInstruction* rhs);

// Creates a pad HLO instruction and adds it to the computation containing
// `operand` and `padding_value` (`operand` and `padding_value` must be in the
// same computation).
StatusOr<HloInstruction*> CreatePadHlo(HloInstruction* operand,
                                       HloInstruction* padding_value,
                                       const PaddingConfig& padding_config);

// Creates a slice HLO instruction and adds it to the computation containing
// `operand`.
StatusOr<HloInstruction*> CreateSliceHlo(
    HloInstruction* operand, tensorflow::gtl::ArraySlice<int64> start_indices,
    tensorflow::gtl::ArraySlice<int64> limit_indices,
    tensorflow::gtl::ArraySlice<int64> strides);

// Creates a convolution HLO instruction and adds it to the computation
// containing `lhs` and `rhs` (`lhs` and `rhs` must be in the same computation).
StatusOr<HloInstruction*> CreateConvolveHlo(
    HloInstruction* lhs, HloInstruction* rhs, const Window& window,
    const ConvolutionDimensionNumbers& dimension_numbers);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_CREATION_UTILS_H_
