/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/mlir_gpu/hlo_dialect_emitter.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // TF:local_config_mlir
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/compiler/xla/service/gpu/thunk_emitter.h"

namespace xla {
namespace gpu {

using ::mlir::LLVM::LLVMDialect;

HloDialectEmitter::HloDialectEmitter(const HloModule& hlo_module,
                                     const BufferAssignment& assignment,
                                     const se::Platform* platform,
                                     ::mlir::ModuleOp mlir_module)
    : mlir_module_(mlir_module),
      builder_(mlir_module_.getContext()),
      buffer_assignment_(assignment),
      platform_(platform),
      thunk_sequence_(new ThunkSequence()) {
  LLVMDialect* llvmDialect =
      mlir_module.getContext()->getRegisteredDialect<LLVMDialect>();
  pointer_size_ = llvmDialect->getLLVMModule().getDataLayout().getPointerSize();
}

void HloDialectEmitter::AddThunkToThunkSequence(std::unique_ptr<Thunk> thunk) {
  thunk_sequence_->push_back(std::move(thunk));
}

StatusOr<BufferAllocation::Slice> HloDialectEmitter::MaybeGetAllocationSlice(
    const HloInstruction& hlo, const ShapeIndex& index) const {
  return buffer_assignment_.GetUniqueSlice(&hlo, index);
}

int64 HloDialectEmitter::ByteSizeOf(const Shape& shape) const {
  return ShapeUtil::ByteSizeOf(shape, pointer_size_);
}

const se::Platform* HloDialectEmitter::platform() const { return platform_; }

Status HloDialectEmitter::DefaultAction(HloInstruction* hlo) {
  LOG(FATAL) << "Not implemented yet.";
}

Status HloDialectEmitter::HandleFusion(HloInstruction* fusion) {
  LOG(FATAL) << "Not implemented yet.";
}

Status HloDialectEmitter::HandleCustomCall(HloInstruction* custom_call) {
  return ThunkEmitter(this).HandleCustomCall(custom_call);
}

Status HloDialectEmitter::FinishVisit(HloInstruction* root) {
  LOG(FATAL) << "Not implemented yet.";
}
}  // namespace gpu
}  // namespace xla
