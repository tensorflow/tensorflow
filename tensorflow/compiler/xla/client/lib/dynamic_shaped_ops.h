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
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

// Similar to static shaped conditional, but allows true_computation and
// false_computation to have different dimension sizes (ranks still have to be
// the same).
XlaOp DynamicConditional(XlaBuilder* builder, XlaOp predicate,
                         XlaOp true_operand,
                         const XlaComputation& true_computation,
                         XlaOp false_operand,
                         const XlaComputation& false_computation);
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_CLIENT_LIB_DYNAMIC_SHAPED_OPS_H_
