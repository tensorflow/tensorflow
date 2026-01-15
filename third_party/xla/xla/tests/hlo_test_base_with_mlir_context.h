/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_TESTS_HLO_TEST_BASE_WITH_MLIR_CONTEXT_H_
#define XLA_TESTS_HLO_TEST_BASE_WITH_MLIR_CONTEXT_H_

#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/tests/hlo_test_base.h"

namespace xla {

class HloTestBaseWithMLIRContext : public HloTestBase {
 public:
  HloTestBaseWithMLIRContext() { RegisterSymbolicExprStorage(&mlir_context_); }
  mlir::MLIRContext* mlir_context() { return &mlir_context_; }

 private:
  mlir::MLIRContext mlir_context_;
};

}  // namespace xla

#endif  // XLA_TESTS_HLO_TEST_BASE_WITH_MLIR_CONTEXT_H_
