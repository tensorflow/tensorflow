/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/ir_emitter_context.h"

#include <utility>

#include "tensorflow/compiler/xla/service/gpu/gpu_constants.h"

namespace xla {
namespace gpu {

void IrEmitterContext::emit_constant(int64_t num_elements,
                                     int64_t bytes_per_element,
                                     absl::string_view symbol_name,
                                     int allocation_idx,
                                     llvm::ArrayRef<uint8_t> content,
                                     llvm::IRBuilder<>* b) {
  // LLVM and PTXAS don't deal well with large constants, so we only emit very
  // small constants directly in LLVM IR.  Larger constants are emitted with
  // zero initializers in LLVM IR and are later overwritten when the PTX/CUBIN
  // is loaded.
  bool should_emit_initializer = num_elements <= 1;
  llvm::ArrayType* global_type =
      llvm::ArrayType::get(b->getInt8Ty(), num_elements * bytes_per_element);
  GpuExecutable::ConstantInfo info;
  llvm::Constant* initializer;
  if (should_emit_initializer) {
    initializer = llvm::ConstantDataArray::get<uint8_t>(
        llvm_module_->getContext(), content);
  } else {
    info.content = content;
    initializer = llvm::ConstantAggregateZero::get(global_type);
  }

  // These globals will be looked up by name by GpuExecutable so we need to
  // give them an external linkage.  Not all of their uses are visible in
  // the LLVM IR so we can't give then a linkage that merely preserves their
  // names (like available_externally), we also need to ensure that they stick
  // around even if they're "unused".
  //
  // We may have to be more clever here in the future if we notice that we're
  // keeping around too many globals because of their linkage.
  llvm::GlobalVariable* global_for_const = new llvm::GlobalVariable(
      global_type, /*isConstant=*/should_emit_initializer,
      llvm::GlobalValue::ExternalLinkage,
      /*Initializer=*/initializer, symbol_name,
      /*TLMode=*/llvm::GlobalValue::NotThreadLocal,
      /*AddressSpace=*/0,
      /*isExternallyInitialized=*/false);
  global_for_const->setAlignment(llvm::Align(kConstantBufferAlignBytes));
  llvm_module_->getGlobalList().push_back(global_for_const);

  info.symbol_name.assign(symbol_name);
  info.allocation_index = allocation_idx;
  constants_.push_back(std::move(info));
}

}  // namespace gpu
}  // namespace xla
