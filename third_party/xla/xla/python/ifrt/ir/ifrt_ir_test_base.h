/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_IFRT_IR_IFRT_IR_TEST_BASE_H_
#define XLA_PYTHON_IFRT_IR_IFRT_IR_TEST_BASE_H_

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace ifrt {
namespace test_util {

// Base class to help create pure-MLIR tests for IFRT IR.
class IfrtIrTestBase : public testing::Test {
 public:
  IfrtIrTestBase();

 protected:
  // Loads mlir from source string.
  absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> LoadFromSource(
      absl::string_view source);

  mlir::MLIRContext mlir_context_;
};

}  // namespace test_util
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_IR_IFRT_IR_TEST_BASE_H_
