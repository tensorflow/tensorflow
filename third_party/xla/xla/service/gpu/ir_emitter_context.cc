/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/service/gpu/ir_emitter_context.h"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/Alignment.h"
#include "llvm/TargetParser/Triple.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/gpu/ir_emission_utils.h"

namespace xla {
namespace gpu {

void IrEmitterContext::emit_constant(int64_t num_elements,
                                     int64_t bytes_per_element,
                                     absl::string_view symbol_name,
                                     int allocation_idx,
                                     DenseDataIntermediate content,
                                     llvm::IRBuilder<>* b) {
  // LLVM and PTXAS don't deal well with large constants, so we only emit very
  // small constants directly in LLVM IR.  Larger constants are emitted with
  // zero initializers in LLVM IR and are later overwritten when the PTX/CUBIN
  // is loaded.
  bool should_emit_initializer = num_elements <= 1;

  // Ptxas has issues if the constant allocation is smaller than 64 bytes.
  // TODO(b/253259975): Remove when fixed ptxas version is submitted.
  constexpr int64_t kMinConstAllocationInBytes = 64;
  bool needs_padding =
      num_elements * bytes_per_element < kMinConstAllocationInBytes;

  llvm::ArrayType* global_type = llvm::ArrayType::get(
      b->getInt8Ty(),
      std::max(num_elements * bytes_per_element, kMinConstAllocationInBytes));

  GpuExecutable::ConstantInfo info;
  llvm::Constant* initializer = [&]() -> llvm::Constant* {
    if (!should_emit_initializer) {
      info.content = std::move(content);
      return llvm::ConstantAggregateZero::get(global_type);
    }

    std::vector<uint8_t> padded(kMinConstAllocationInBytes, 0);
    absl::c_copy(content.span(), padded.begin());
    return llvm::ConstantDataArray::get<uint8_t>(
        llvm_module_->getContext(),
        needs_padding ? llvm::ArrayRef<uint8_t>(padded)
                      : llvm::ArrayRef<uint8_t>(content.span().data(),
                                                content.span().size()));
  }();

  // Explicitly set global addrspace for SPIR backend.
  int addrspace =
      llvm::Triple(llvm_module_->getTargetTriple()).isSPIR() ? 1 : 0;
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
      /*AddressSpace=*/addrspace,
      /*isExternallyInitialized=*/false);
  global_for_const->setAlignment(llvm::Align(kConstantBufferAlignBytes));
  llvm_module_->insertGlobalVariable(global_for_const);

  info.symbol_name.assign(symbol_name);
  info.allocation_index = allocation_idx;
  constants_.push_back(std::move(info));
}

}  // namespace gpu
}  // namespace xla
