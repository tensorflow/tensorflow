/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_LLVM_TARGET_FEATURES_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_LLVM_TARGET_FEATURES_H_

#include <string>
#include <tuple>
#include "llvm/IR/Intrinsics.h"

namespace xla {
namespace llvm_ir {
// Enmerations to get target specific intrinsics or function calls
typedef enum  TargetIntrinsicEnum {
  kSHFL_DOWN_F32 = 0,
  kSHFL_DOWN_I32 = 1,
  kTHREAD_ID_X = 2,
  kTHREAD_ID_Y = 3,
  kTHREAD_ID_Z = 4,
  kBLOCK_ID_X = 5,
  kBLOCK_ID_Y = 6,
  kBLOCK_ID_Z = 7,
  kBARRIER_ID = 8,
  kLAST = 9
} TargetIntrinsicEnum;

// Abstract interface for classes providing information about the gpu target
// we're compiling for.
class LLVMTargetFeatures {
  const llvm::Intrinsic::ID* tgt_intrinsic;

 protected:
  LLVMTargetFeatures(const llvm::Intrinsic::ID* intrin)
      : tgt_intrinsic(intrin) {}

 public:
  // Return  llvm intrinsic for target
  llvm::Intrinsic::ID GetIntrinsicID(TargetIntrinsicEnum tgt_id) {
    return tgt_intrinsic[tgt_id];
  };

  virtual ~LLVMTargetFeatures() = default;
};

class AMDGPUMachineFeatures : public LLVMTargetFeatures {
  // Please modify both both PTX and AMDGPU consistently
  // If either targets use a function call instead of intrinsic
  // please return llvm::Intrinsic::not_intrinsic
  static const constexpr llvm::Intrinsic::ID amdgpu_intrins[] = {
      llvm::Intrinsic::not_intrinsic,
      llvm::Intrinsic::not_intrinsic,
      llvm::Intrinsic::amdgcn_workitem_id_x,
      llvm::Intrinsic::amdgcn_workitem_id_y,
      llvm::Intrinsic::amdgcn_workitem_id_z,
      llvm::Intrinsic::amdgcn_workgroup_id_x,
      llvm::Intrinsic::amdgcn_workgroup_id_y,
      llvm::Intrinsic::amdgcn_workgroup_id_z,
      llvm::Intrinsic::amdgcn_s_barrier,
      llvm::Intrinsic::not_intrinsic};
  // Return  llvm intrinsic for AMD target
 public:
  AMDGPUMachineFeatures() : LLVMTargetFeatures(amdgpu_intrins){};
  ~AMDGPUMachineFeatures(){};
  static AMDGPUMachineFeatures& Singleton(){
    static AMDGPUMachineFeatures instance;
    return instance;
  }
      
};

class NVPTXMachineFeatures : public LLVMTargetFeatures {
  // Please modify both both PTX and AMDGPU consistently
  // If either targets use a function call instead of intrinsic
  // please return llvm::Intrinsic::not_intrinsic
  static const constexpr llvm::Intrinsic::ID nvptx_intrins[] = {
      llvm::Intrinsic::nvvm_shfl_sync_down_f32,
      llvm::Intrinsic::nvvm_shfl_sync_down_i32,
      llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x,
      llvm::Intrinsic::nvvm_read_ptx_sreg_tid_y,
      llvm::Intrinsic::nvvm_read_ptx_sreg_tid_z,
      llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_x,
      llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_y,
      llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_z,
      llvm::Intrinsic::nvvm_barrier0,
      llvm::Intrinsic::not_intrinsic};

 public:
  NVPTXMachineFeatures() : LLVMTargetFeatures(nvptx_intrins){};
  // Return  llvm intrinsic for NVIDIA target
  ~NVPTXMachineFeatures(){};

  static NVPTXMachineFeatures& Singleton(){
    static NVPTXMachineFeatures instance;
    return instance;
  }
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_LLVM_TARGET_FEATURES_H_
