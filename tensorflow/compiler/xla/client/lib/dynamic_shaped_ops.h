/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_CLIENT_LIB_DYNAMIC_SHAPED_OPS_H_
#define TENSORFLOW_COMPILER_XLA_CLIENT_LIB_DYNAMIC_SHAPED_OPS_H_

#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/value_inference.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

// Similar to static shaped conditional, but allows true_computation and
// false_computation to have different dimension sizes (ranks still have to be
// the same). Fall back to static conditional if dynamism is not presented.
XlaOp DynamicConditional(XlaBuilder* builder, XlaOp predicate,
                         XlaOp true_operand,
                         const XlaComputation& true_computation,
                         XlaOp false_operand,
                         const XlaComputation& false_computation);

// Similar to DynamicConditional, but support multiple branches.
XlaOp DynamicConditional(
    XlaBuilder* builder, XlaOp branch_index,
    absl::Span<const XlaComputation* const> branch_computations,
    absl::Span<const XlaOp> branch_operands);

// Similar to SetDimensionSize, but automatically adjust the bound of output if
// a tighter one can be inferred by `value_inference`.
StatusOr<XlaOp> SetDimensionSizeWithRebound(ValueInference* value_inference,
                                            XlaOp operand, XlaOp dimension_size,
                                            int64_t dimension);

// Take a `operand` tensor and a R1 tensor `size_vector` representing the sizes
// of `operand`, Call SetDimensionSize if for each dimension whose size is
// dynamic.
StatusOr<XlaOp> SetAllDimensionSizes(ValueInference* value_inference,
                                     XlaOp operand, XlaOp size_vector);
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_CLIENT_LIB_DYNAMIC_SHAPED_OPS_H_
