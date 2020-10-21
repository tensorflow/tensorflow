/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TFR_IR_TFR_UTILS_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_TFR_IR_TFR_UTILS_UTILS_H_

#include <string>

#include "mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace TFR {

// This is a hardcoded rule for mapping a TF op name to the corresponding
// TFR function name. Examples:
//   tf.Pack => tf__pack
//   tf.ConcatV2 => tf__concat_v2
// TODO(fengliuai): move to an util file.
std::string GetComposeFuncName(StringRef tf_op_name);

// This is a hardcoded rule for mapping a TFR function op name to the
// corresponding TF opname. Examples:
//   tf__pack -> tf.Pack
//   tf__concat_v2 => tf.ConcatV2
std::string GetTFOpName(StringRef compose_func_name);

}  // namespace TFR
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TFR_IR_TFR_UTILS_UTILS_H_
