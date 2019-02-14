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

#include "tensorflow/compiler/xla/service/llvm_ir/llvm_target_features.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace xla {
namespace llvm_ir {
llvm::Intrinsic::ID AMDGPUMachineFeatures::GetIntrinsicID(
    const std::string &name) {
  if (tensorflow::str_util::EndsWith(name, "__thread_id_x")) {
    return llvm::Intrinsic::amdgcn_workitem_id_x;
  } else if (tensorflow::str_util::EndsWith(name, "__thread_id_y")) {
    return llvm::Intrinsic::amdgcn_workitem_id_y;
  } else if (tensorflow::str_util::EndsWith(name, "__thread_id_z")) {
    return llvm::Intrinsic::amdgcn_workitem_id_z;
  } else if (tensorflow::str_util::EndsWith(name, "__block_id_x")) {
    return llvm::Intrinsic::amdgcn_workgroup_id_x;
  } else if (tensorflow::str_util::EndsWith(name, "__block_id_y")) {
    return llvm::Intrinsic::amdgcn_workgroup_id_y;
  } else if (tensorflow::str_util::EndsWith(name, "__block_id_z")) {
    return llvm::Intrinsic::amdgcn_workgroup_id_z;
  } else if (tensorflow::str_util::EndsWith(name, "__barrier")) {
    return llvm::Intrinsic::amdgcn_s_barrier;
  }
  return llvm::Intrinsic::not_intrinsic;
}

llvm::Intrinsic::ID NVPTXMachineFeatures::GetIntrinsicID(
    const std::string &name) {
  if (tensorflow::str_util::EndsWith(name, "__thread_id_x")) {
    return llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x;
  } else if (tensorflow::str_util::EndsWith(name, "__thread_id_y")) {
    return llvm::Intrinsic::nvvm_read_ptx_sreg_tid_y;
  } else if (tensorflow::str_util::EndsWith(name, "__thread_id_z")) {
    return llvm::Intrinsic::nvvm_read_ptx_sreg_tid_z;
  } else if (tensorflow::str_util::EndsWith(name, "__block_id_x")) {
    return llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_x;
  } else if (tensorflow::str_util::EndsWith(name, "__block_id_y")) {
    return llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_y;
  } else if (tensorflow::str_util::EndsWith(name, "__block_id_z")) {
    return llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_z;
  } else if (tensorflow::str_util::EndsWith(name, "__barrier")) {
    return llvm::Intrinsic::nvvm_barrier0;
  } else if (tensorflow::str_util::EndsWith(name, "__shfl_down")) {
    return llvm::Intrinsic::nvvm_shfl_sync_down_f32;
  }
  return llvm::Intrinsic::not_intrinsic;
}
}  // namespace llvm_ir 
}  // namespace xla
