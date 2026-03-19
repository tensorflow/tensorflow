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

#include "xla/codegen/trace_pass_instrumentation.h"

#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {

TraceInstrumentation::TraceInstrumentation()
    : is_active_(tsl::profiler::TraceMe::Active()) {}

void TraceInstrumentation::runBeforePass(mlir::Pass* pass,
                                         mlir::Operation* op) {
  if (!is_active_) {
    return;
  }

  active_tracers_.emplace(std::make_pair(pass, op),
                          absl::string_view(pass->getName()));
}

void TraceInstrumentation::runAfterPass(mlir::Pass* pass, mlir::Operation* op) {
  if (!is_active_) {
    return;
  }

  active_tracers_.erase(std::make_pair(pass, op));
}

}  // namespace xla
