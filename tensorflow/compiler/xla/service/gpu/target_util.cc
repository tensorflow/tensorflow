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
// Provide helper routine for obtaining  gpu target information useful
// for llvm IR contruction.

#include "tensorflow/compiler/xla/service/gpu/target_util.h"

#include "absl/strings/str_cat.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/MDBuilder.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_type_conversion_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/tsl/platform/logging.h"

namespace xla {
namespace gpu {
namespace {
// Utility functions to obtain NVPTX/AMDGPU specific information.
using absl::StrCat;

// Wrapper structure for carrying llvm intrinsic ids for NVPTX/AMDGPU platforms.
// On AMDGPU, some of these operations are made as device functions instead of
// intrinsics. Therefore a variant type is used to wrap the lambda to call
// those device functions.
struct TargetIntrinsics {
  llvm::Intrinsic::ID nvptx_intrinsic;
  std::variant<llvm::Intrinsic::ID,
               std::function<llvm::CallInst*(llvm::IRBuilder<>*)>>
      amdgpu_intrinsic_or_function;
};

// Gets the llvm intrinsic ids on different platforms (NVPTX, AMDGPU)
// corresponding to the give TargetIntrinsicID.
struct TargetIntrinsics GetIntrinsic(TargetIntrinsicID intrin) {
  switch (intrin) {
    case TargetIntrinsicID::kThreadIdx: {
      return {llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x,
              llvm::Intrinsic::amdgcn_workitem_id_x};
    }
    case TargetIntrinsicID::kThreadIdy: {
      return {llvm::Intrinsic::nvvm_read_ptx_sreg_tid_y,
              llvm::Intrinsic::amdgcn_workitem_id_y};
    }
    case TargetIntrinsicID::kThreadIdz: {
      return {llvm::Intrinsic::nvvm_read_ptx_sreg_tid_z,
              llvm::Intrinsic::amdgcn_workitem_id_z};
    }
    case TargetIntrinsicID::kBlockIdx: {
      return {llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_x,
              llvm::Intrinsic::amdgcn_workgroup_id_x};
    }
    case TargetIntrinsicID::kBlockIdy: {
      return {llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_y,
              llvm::Intrinsic::amdgcn_workgroup_id_y};
    }
    case TargetIntrinsicID::kBlockIdz: {
      return {llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_z,
              llvm::Intrinsic::amdgcn_workgroup_id_z};
    }
    case TargetIntrinsicID::kBarrierId: {
      return {llvm::Intrinsic::nvvm_barrier0,
              llvm::Intrinsic::amdgcn_s_barrier};
    }
    case TargetIntrinsicID::kBlockDimx: {
      return {llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_x,
              [](llvm::IRBuilder<>* b_) -> llvm::CallInst* {
                return EmitDeviceFunctionCall("__ockl_get_local_size",
                                              {b_->getInt32(0)}, {U32}, U64,
                                              {b_->getContext()}, b_);
              }};
    }
    case TargetIntrinsicID::kBlockDimy: {
      return {llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_y,
              [](llvm::IRBuilder<>* b_) -> llvm::CallInst* {
                return EmitDeviceFunctionCall("__ockl_get_local_size",
                                              {b_->getInt32(1)}, {U32}, U64,
                                              {b_->getContext()}, b_);
              }};
    }
    case TargetIntrinsicID::kBlockDimz: {
      return {llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_z,
              [](llvm::IRBuilder<>* b_) -> llvm::CallInst* {
                return EmitDeviceFunctionCall("__ockl_get_local_size",
                                              {b_->getInt32(2)}, {U32}, U64,
                                              {b_->getContext()}, b_);
              }};
    }
    case TargetIntrinsicID::kGroupBarrierId: {
      return {llvm::Intrinsic::nvvm_bar_warp_sync,
              llvm::Intrinsic::amdgcn_wave_barrier};
    }
  }
}

// Wrapper structure for carrying math functions for NVPTX/AMDGPU platforms.
struct TargetDeviceFunction {
  const std::string nvptx_root;
  const std::string amdgpu_root;
};

// Gets the device function name on different platforms (NVPTX, AMDGPU)
// corresponding to the given TargetDeviceFunctionID.
struct TargetDeviceFunction GetDeviceFunctionRoot(
    TargetDeviceFunctionID func_id) {
  switch (func_id) {
    case TargetDeviceFunctionID::kAtan2: {
      return {"__nv_atan2", "__ocml_atan2"};
    }
    case TargetDeviceFunctionID::kCos: {
      return {"__nv_cos", "__ocml_cos"};
    }
    case TargetDeviceFunctionID::kErfcinv: {
      return {"__nv_erfcinv", "__ocml_erfcinv"};
    }
    case TargetDeviceFunctionID::kExp: {
      return {"__nv_exp", "__ocml_exp"};
    }
    case TargetDeviceFunctionID::kExpm1: {
      return {"__nv_expm1", "__ocml_expm1"};
    }
    case TargetDeviceFunctionID::kFmod: {
      return {"__nv_fmod", "__ocml_fmod"};
    }
    case TargetDeviceFunctionID::kHypot: {
      return {"__nv_hypot", "__ocml_hypot"};
    }
    case TargetDeviceFunctionID::kLog: {
      return {"__nv_log", "__ocml_log"};
    }
    case TargetDeviceFunctionID::kLog1p: {
      return {"__nv_log1p", "__ocml_log1p"};
    }
    case TargetDeviceFunctionID::kPow: {
      return {"__nv_pow", "__ocml_pow"};
    }
    case TargetDeviceFunctionID::kRound: {
      return {"__nv_round", "__ocml_round"};
    }
    case TargetDeviceFunctionID::kRsqrt: {
      return {"__nv_rsqrt", "__ocml_rsqrt"};
    }
    case TargetDeviceFunctionID::kSin: {
      return {"__nv_sin", "__ocml_sin"};
    }
    case TargetDeviceFunctionID::kSqrt: {
      return {"__nv_sqrt", "__ocml_sqrt"};
    }
    case TargetDeviceFunctionID::kTanh: {
      return {"__nv_tanh", "__ocml_tanh"};
    }
  }
}
}  // namespace

std::string ObtainDeviceFunctionName(TargetDeviceFunctionID func_id,
                                     PrimitiveType output_type,
                                     llvm::IRBuilder<>* b) {
  // The device math functions differentiate between "double" and "float" by
  // appending a double or float specific suffix to a root name. The suffix and
  // the root name are specific to the target.
  llvm::Triple target_triple =
      llvm::Triple(b->GetInsertBlock()->getModule()->getTargetTriple());
  struct TargetDeviceFunction gpu_root_names = GetDeviceFunctionRoot(func_id);
  if (target_triple.isNVPTX()) {
    if (output_type == F32) {
      return StrCat(gpu_root_names.nvptx_root, "f");
    } else if (output_type == F64) {
      return gpu_root_names.nvptx_root;
    } else {
      LOG(FATAL) << "Unexpected type while getting device function name.";
    }
  } else if (target_triple.getArch() == llvm::Triple::amdgcn) {
    if (output_type == F32) {
      return StrCat(gpu_root_names.amdgpu_root, "_f32");
    } else if (output_type == F64) {
      return StrCat(gpu_root_names.amdgpu_root, "_f64");
    } else {
      LOG(FATAL) << "Unexpected type while getting device function name.";
    }
  } else {
    LOG(FATAL) << "Invalid triple " << target_triple.str();
  }
}

llvm::CallInst* EmitDeviceFunctionCall(
    const std::string& callee_name, absl::Span<llvm::Value* const> operands,
    absl::Span<const PrimitiveType> input_types, PrimitiveType output_type,
    const llvm::AttrBuilder& attributes, llvm::IRBuilder<>* b,
    absl::string_view name) {
  std::vector<llvm::Type*> ir_input_types;
  llvm::Module* module = b->GetInsertBlock()->getModule();
  for (PrimitiveType input_type : input_types) {
    ir_input_types.push_back(
        llvm_ir::PrimitiveTypeToIrType(input_type, module));
  }
  llvm::FunctionType* callee_type = llvm::FunctionType::get(
      llvm_ir::PrimitiveTypeToIrType(output_type, module),  // Return type.
      ir_input_types,                                       // Parameter types.
      false);  // No variadic arguments.

  // Declares the callee if it is not declared already.
  llvm::Function* callee = llvm::dyn_cast<llvm::Function>(
      b->GetInsertBlock()
          ->getModule()
          ->getOrInsertFunction(callee_name, callee_type)
          .getCallee());

  callee->addFnAttrs(attributes);

  return b->CreateCall(callee, llvm_ir::AsArrayRef(operands), name.data());
}

llvm::CallInst* EmitCallToTargetIntrinsic(
    TargetIntrinsicID intrinsic_id, absl::Span<llvm::Value* const> operands,
    absl::Span<llvm::Type* const> overloaded_types, llvm::IRBuilder<>* b) {
  llvm::Module* module = b->GetInsertBlock()->getModule();
  struct TargetIntrinsics gpu_intrinsic_id = GetIntrinsic(intrinsic_id);
  llvm::Triple target_triple = llvm::Triple(module->getTargetTriple());
  llvm::Intrinsic::ID llvm_intrinsic_id = llvm::Intrinsic::not_intrinsic;
  if (target_triple.isNVPTX()) {
    llvm_intrinsic_id = gpu_intrinsic_id.nvptx_intrinsic;
  } else if (target_triple.getArch() == llvm::Triple::amdgcn) {
    llvm::Intrinsic::ID* llvm_intrinsic_id_ptr =
        std::get_if<llvm::Intrinsic::ID>(
            &gpu_intrinsic_id.amdgpu_intrinsic_or_function);
    if (llvm_intrinsic_id_ptr) {
      llvm_intrinsic_id = *llvm_intrinsic_id_ptr;
    } else {
      std::function<llvm::CallInst*(llvm::IRBuilder<>*)>* builder_func =
          std::get_if<std::function<llvm::CallInst*(llvm::IRBuilder<>*)>>(
              &gpu_intrinsic_id.amdgpu_intrinsic_or_function);
      return (*builder_func)(b);
    }
  } else {
    LOG(FATAL) << "Invalid triple " << target_triple.str();
  }

  llvm::Function* intrinsic = llvm::Intrinsic::getDeclaration(
      module, llvm_intrinsic_id, llvm_ir::AsArrayRef(overloaded_types));
  return b->CreateCall(intrinsic, llvm_ir::AsArrayRef(operands));
}

void AnnotateFunctionAsGpuKernel(llvm::Module* module, llvm::Function* func,
                                 llvm::IRBuilder<>* b) {
  llvm::Triple target_triple = llvm::Triple(module->getTargetTriple());
  if (target_triple.isNVPTX()) {
    // Add the declaration of this kernel to llvm.nvvm.annotations so that NVPTX
    // treats function as a CUDA kernel.
    llvm::LLVMContext& context = module->getContext();
    llvm::NamedMDNode* nvvm_annotations_node =
        module->getOrInsertNamedMetadata("nvvm.annotations");
    nvvm_annotations_node->addOperand(llvm::MDNode::get(
        context, {llvm::ConstantAsMetadata::get(func),
                  llvm::MDString::get(context, "kernel"),
                  llvm::ConstantAsMetadata::get(b->getInt32(1))}));

  } else if (target_triple.getArch() == llvm::Triple::amdgcn) {
    // Attach information so AMDGPU can recognize function as a AMDGPU kernel.
    func->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
    func->addFnAttr("amdgpu-flat-work-group-size", "1, 1024");
  } else {
    LOG(FATAL) << "Invalid triple " << target_triple.str();
  }
}

}  // namespace gpu
}  // namespace xla
