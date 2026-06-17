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

#ifndef XLA_CODEGEN_TRACE_PASS_INSTRUMENTATION_H_
#define XLA_CODEGEN_TRACE_PASS_INSTRUMENTATION_H_

#include <utility>

#include "absl/container/flat_hash_map.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {

// Simple instrumentation that creates a TraceMe for each pass.
// MLIR takes care of the thread safety so we don't have to.
class TraceInstrumentation : public mlir::PassInstrumentation {
 public:
  TraceInstrumentation();

  void runBeforePass(mlir::Pass* pass, mlir::Operation* op) override;
  void runAfterPass(mlir::Pass* pass, mlir::Operation* op) override;

 private:
  bool is_active_;
  absl::flat_hash_map<std::pair<mlir::Pass*, mlir::Operation*>,
                      tsl::profiler::TraceMe>
      active_tracers_;
};

}  // namespace xla

#endif  // XLA_CODEGEN_TRACE_PASS_INSTRUMENTATION_H_
