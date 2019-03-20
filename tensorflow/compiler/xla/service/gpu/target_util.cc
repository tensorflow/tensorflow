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

// Provide helper routine for obtaining  gpu target information useful for llvm IR 
// contruction 
#include "target_util.h"

#include "llvm/IR/MDBuilder.h"

#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace gpu {
// Obtain target specific llvm intrinsic 
struct TargetIntrinsics GetIntrinsic(TargetIntrinsicID intrin) {
  switch (intrin) {
    case TargetIntrinsicID::kShfldownf32: {
      return {llvm::Intrinsic::nvvm_shfl_sync_down_f32,
              llvm::Intrinsic::not_intrinsic};
    }
    case TargetIntrinsicID::kShfldowni32: {
      return {llvm::Intrinsic::nvvm_shfl_sync_down_i32,
              llvm::Intrinsic::not_intrinsic};
    }
    case TargetIntrinsicID::kThreadidx: {
      return {llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x,
              llvm::Intrinsic::amdgcn_workitem_id_x};
    }
    case TargetIntrinsicID::kThreadidy: {
      return {llvm::Intrinsic::nvvm_read_ptx_sreg_tid_y,
              llvm::Intrinsic::amdgcn_workitem_id_y};
    }
    case TargetIntrinsicID::kThreadidz: {
      return {llvm::Intrinsic::nvvm_read_ptx_sreg_tid_z,
              llvm::Intrinsic::amdgcn_workitem_id_z};
    }
    case TargetIntrinsicID::kBlockidx: {
      return {llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_x,
              llvm::Intrinsic::amdgcn_workgroup_id_x};
    }
    case TargetIntrinsicID::kBlockidy: {
      return {llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_y,
              llvm::Intrinsic::amdgcn_workgroup_id_y};
    }
    case TargetIntrinsicID::kBlockidz: {
      return {llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_z,
              llvm::Intrinsic::amdgcn_workgroup_id_z};
    }
    case TargetIntrinsicID::kBarrierid: {
      return {llvm::Intrinsic::nvvm_barrier0,
              llvm::Intrinsic::amdgcn_s_barrier};
    }
  }

  return {llvm::Intrinsic::not_intrinsic, llvm::Intrinsic::not_intrinsic};
}

llvm::CallInst* EmitCallToTargetIntrinsic(
    TargetIntrinsicID intrinsic_id, absl::Span<llvm::Value* const> operands,
    absl::Span<llvm::Type* const> overloaded_types, llvm::IRBuilder<>* b) {
  llvm::Module* module = b->GetInsertBlock()->getModule();
  struct TargetIntrinsics gpu_intrinsic_id = GetIntrinsic(intrinsic_id);
  llvm::Triple target_triple = llvm::Triple(module->getTargetTriple());
  llvm::Intrinsic::ID llvm_intrinsic_id = llvm::Intrinsic::not_intrinsic;
  if ((target_triple.getArch() == llvm::Triple::nvptx) ||
      (target_triple.getArch() == llvm::Triple::nvptx64)) {
    llvm_intrinsic_id = gpu_intrinsic_id.nvptx_intrinsic;
  } else if (target_triple.getArch() == llvm::Triple::amdgcn) {
    llvm_intrinsic_id = gpu_intrinsic_id.amdgpu_intrinsic;
  } else {
    LOG(FATAL) << "Invalid triple " << target_triple.str();
  }
  CHECK_NE(llvm_intrinsic_id, llvm::Intrinsic::not_intrinsic);
  llvm::Function* intrinsic = llvm::Intrinsic::getDeclaration(
      module, llvm_intrinsic_id, llvm_ir::AsArrayRef(overloaded_types));
  return b->CreateCall(intrinsic, llvm_ir::AsArrayRef(operands));
}

}  // namespace gpu
}  // namespace xla
