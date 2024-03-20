/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_MODEL_INDEXING_CONTEXT_H_
#define XLA_SERVICE_GPU_MODEL_INDEXING_CONTEXT_H_

#include <cstdint>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/model/indexing_map.h"

namespace xla {
namespace gpu {

class IndexingContext {
 public:
  explicit IndexingContext(mlir::MLIRContext* mlir_context)
      : mlir_context_(mlir_context) {}

  mlir::MLIRContext* GetMLIRContext() const { return mlir_context_; }

  // TBD: This method should behave like a thread-safe counter. It will register
  // a new RTSymbol by adding it to `rt_vals_registry_` with the newly generated
  // ID.
  RTVar RegisterRTVar(RTVarData rt_var_data);

  RTVarData& GetRTVarData(RTVarID id);

  static void ResetRTVarStateForTests();

 private:
  mlir::MLIRContext* mlir_context_;
  absl::flat_hash_map<RTVarID, RTVarData> rt_vars_registry_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_INDEXING_CONTEXT_H_
