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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_TEST_PASSES_DETAIL_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_TEST_PASSES_DETAIL_H_

#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace tf_test {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tensorflow/transforms/test_passes.h.inc"

}  // namespace tf_test
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_TEST_PASSES_DETAIL_H_
