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
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_builder_mixin.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

using absl::StrAppend;

namespace gpu {
namespace {

// Utility functions to obtain NVPTX/AMDGPU specific information.

// Wrapper structure for  intrinsic information for NVPTX/AMDGPU.
struct TargetIntrinsicInfo {
  TargetIntrinsicInfo(llvm::Intrinsic::ID x) : intrinsic(x) {}
  llvm::Intrinsic::ID intrinsic;
};

// Wrapper structure for  device functionc information for NVPTX/AMDGPU.
struct TargetFunctionInfo {
  // No signature mismatch.
  TargetFunctionInfo(const string callee_name_, bool check_signature_)
      : callee_name(callee_name_),
        check_signature(check_signature_),
        input_types({PRIMITIVE_TYPE_INVALID}),
        result_type(PRIMITIVE_TYPE_INVALID) {}
  // Potential signature mismatch - generate casts if needed.
  TargetFunctionInfo(const string callee_name_, bool check_signature_,
                     absl::Span<const PrimitiveType> input_types_,
                     const PrimitiveType output_type_)
      : callee_name(callee_name_),
        check_signature(check_signature_),
        input_types(input_types_),
        result_type(output_type_) {}
  // Potential signature mistamatch - use bitcast if needed.
  TargetFunctionInfo(const string callee_name_, bool check_signature_,
                     absl::Span<const PrimitiveType> input_types_,
                     const PrimitiveType output_type_, bool use_bitcast_)
      : callee_name(callee_name_),
        check_signature(check_signature_),
        input_types(input_types_),
        result_type(output_type_),
        use_bitcast(use_bitcast_) {}
  // Device function name.
  const string callee_name;
  // Check signature of the device function and if needed generate casts.
  bool check_signature;
  // Inpute types accespted by the device function.
  absl::Span<const PrimitiveType> input_types;
  // Result type of the device function.
  PrimitiveType result_type;
  // Use bitcast to generate casts if the desired signature at the call site
  // does not match the signature of the device function.
  absl::optional<bool> use_bitcast;
};

// Wrapper structure to carry either information about the intrinsic
// or device function for NVPTX/AMDGPU.
struct TargetInfo {
  absl::optional<struct TargetIntrinsicInfo> target_intrinsic_info;
  absl::optional<struct TargetFunctionInfo> target_function_info;
  TargetInfo(struct TargetIntrinsicInfo x)
      : target_intrinsic_info(x){}
  TargetInfo(struct TargetFunctionInfo y):target_function_info(y) {}
};

// Populates the function information for different platforms (NVPTX, AMDGPU)
// corresponding to the given TargetFunctionID.
struct TargetInfo GetTargetInfo(TargetFunctionID function_id,
                                llvm::Triple& target_triple) {

  if (!(target_triple.getArch() == llvm::Triple::nvptx ||
        target_triple.getArch() == llvm::Triple::nvptx64 ||
        target_triple.getArch() == llvm::Triple::amdgcn)) {
    LOG(FATAL) << "Invalid triple " << target_triple.str();
  }

  switch (function_id) {
    case TargetFunctionID::kShflDownF32:{ 
      TargetIntrinsicInfo nvptx_intrinsic_info(llvm::Intrinsic::nvvm_shfl_sync_down_f32);
      TargetFunctionInfo amdgpu_function_info(
          "__ockl_readuplane", true,
          {PRIMITIVE_TYPE_INVALID, S32, S32, PRIMITIVE_TYPE_INVALID}, S32,
          true);
      if (target_triple.getArch() == llvm::Triple::nvptx ||
          target_triple.getArch() == llvm::Triple::nvptx64) {
         return TargetInfo(nvptx_intrinsic_info);
      } else {
         return TargetInfo(amdgpu_function_info);
      }
    }
    case TargetFunctionID::kShflDownI32:{ 
      TargetIntrinsicInfo nvptx_intrinsic_info(llvm::Intrinsic::nvvm_shfl_sync_down_i32);
      TargetFunctionInfo amdgpu_function_info(
          "__ockl_readuplane", true,
          {PRIMITIVE_TYPE_INVALID, S32, S32, PRIMITIVE_TYPE_INVALID}, S32,
          true);

      if (target_triple.getArch() == llvm::Triple::nvptx ||
          target_triple.getArch() == llvm::Triple::nvptx64) {
        return TargetInfo(nvptx_intrinsic_info);
      } else {
        return TargetInfo(amdgpu_function_info);
      }
    }
    case TargetFunctionID::kThreadIdx: {
      TargetIntrinsicInfo nvptx_intrinsic_info(llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x);
      TargetIntrinsicInfo amdgpu_intrinsic_info(llvm::Intrinsic::amdgcn_workitem_id_x);
      if (target_triple.getArch() == llvm::Triple::nvptx ||
          target_triple.getArch() == llvm::Triple::nvptx64) {
        return TargetInfo(nvptx_intrinsic_info);
      } else {
        return TargetInfo(amdgpu_intrinsic_info);
      }
    }

    case TargetFunctionID::kThreadIdy: {
      TargetIntrinsicInfo nvptx_intrinsic_info(llvm::Intrinsic::nvvm_read_ptx_sreg_tid_y);

      TargetIntrinsicInfo amdgpu_intrinsic_info(llvm::Intrinsic::amdgcn_workitem_id_y);
      if (target_triple.getArch() == llvm::Triple::nvptx ||
          target_triple.getArch() == llvm::Triple::nvptx64) {
        return TargetInfo(nvptx_intrinsic_info);
      } else {
        return TargetInfo(amdgpu_intrinsic_info);
      }
    }

    case TargetFunctionID::kThreadIdz: {
      TargetIntrinsicInfo nvptx_intrinsic_info(llvm::Intrinsic::nvvm_read_ptx_sreg_tid_z);
      TargetIntrinsicInfo amdgpu_intrinsic_info(llvm::Intrinsic::amdgcn_workitem_id_z);
      if (target_triple.getArch() == llvm::Triple::nvptx ||
          target_triple.getArch() == llvm::Triple::nvptx64) {
        return TargetInfo(nvptx_intrinsic_info);
      } else {
        return TargetInfo(amdgpu_intrinsic_info);
      }
    }

    case TargetFunctionID::kBlockIdx: {
      TargetIntrinsicInfo nvptx_intrinsic_info(llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_x);
      TargetIntrinsicInfo amdgpu_intrinsic_info(llvm::Intrinsic::amdgcn_workgroup_id_x);
      if (target_triple.getArch() == llvm::Triple::nvptx ||
          target_triple.getArch() == llvm::Triple::nvptx64) {
        return TargetInfo(nvptx_intrinsic_info);
      } else {
        return TargetInfo(amdgpu_intrinsic_info);
      }
    }
    case TargetFunctionID::kBlockIdy: {
      TargetIntrinsicInfo nvptx_intrinsic_info(llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_y);
      TargetIntrinsicInfo amdgpu_intrinsic_info(llvm::Intrinsic::amdgcn_workgroup_id_y);
      if (target_triple.getArch() == llvm::Triple::nvptx ||
          target_triple.getArch() == llvm::Triple::nvptx64) {
        return TargetInfo(nvptx_intrinsic_info);
      } else {
        return TargetInfo(amdgpu_intrinsic_info);
      }
    }
    case TargetFunctionID::kBlockIdz: {
      TargetIntrinsicInfo nvptx_intrinsic_info(llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_z);
      TargetIntrinsicInfo amdgpu_intrinsic_info(llvm::Intrinsic::amdgcn_workgroup_id_z);
      if (target_triple.getArch() == llvm::Triple::nvptx ||
          target_triple.getArch() == llvm::Triple::nvptx64) {
        return TargetInfo(nvptx_intrinsic_info);
      } else {
        return TargetInfo(amdgpu_intrinsic_info);
      }
    }
    case TargetFunctionID::kBarrierId: {
      TargetIntrinsicInfo nvptx_intrinsic_info(llvm::Intrinsic::nvvm_barrier0);
      TargetIntrinsicInfo amdgpu_intrinsic_info(llvm::Intrinsic::amdgcn_s_barrier);
      if (target_triple.getArch() == llvm::Triple::nvptx ||
          target_triple.getArch() == llvm::Triple::nvptx64) {
        return TargetInfo(nvptx_intrinsic_info);
      } else {
        return TargetInfo(amdgpu_intrinsic_info);
      }
    }
    case TargetFunctionID::kPow: {
      TargetFunctionInfo amdgpu_function_info("__ocml_pow", false);
      TargetFunctionInfo nvptx_function_info("__nv_pow", false);
      if (target_triple.getArch() == llvm::Triple::nvptx ||
          target_triple.getArch() == llvm::Triple::nvptx64) {
        return TargetInfo(nvptx_function_info);
      } else {
        return TargetInfo(amdgpu_function_info);
      }
    }
    case TargetFunctionID::kErfcinv: {
      TargetFunctionInfo amdgpu_function_info("__ocml_erfcinv", false);
      TargetFunctionInfo nvptx_function_info("__nv_erfcinv", false);
      if (target_triple.getArch() == llvm::Triple::nvptx ||
          target_triple.getArch() == llvm::Triple::nvptx64) {
        return TargetInfo(nvptx_function_info);
      } else {
        return TargetInfo(amdgpu_function_info);
      }
    }
    case TargetFunctionID::kLog: {
      TargetFunctionInfo amdgpu_function_info("__ocml_log", false);
      TargetFunctionInfo nvptx_function_info("__nv_log", false);
      if (target_triple.getArch() == llvm::Triple::nvptx ||
          target_triple.getArch() == llvm::Triple::nvptx64) {
        return TargetInfo(nvptx_function_info);
      } else {
        return TargetInfo(amdgpu_function_info);
      }
    }
    case TargetFunctionID::kLog1p: {
      TargetFunctionInfo amdgpu_function_info("__ocml_log1p", false);
      TargetFunctionInfo nvptx_function_info("__nv_log1p", false);
      if (target_triple.getArch() == llvm::Triple::nvptx ||
          target_triple.getArch() == llvm::Triple::nvptx64) {
        return TargetInfo(nvptx_function_info);
      } else {
        return TargetInfo(amdgpu_function_info);
      }
    }
    case TargetFunctionID::kSin: {
      TargetFunctionInfo amdgpu_function_info("__ocml_sin", false);
      TargetFunctionInfo nvptx_function_info("__nv_sin", false);
      if (target_triple.getArch() == llvm::Triple::nvptx ||
          target_triple.getArch() == llvm::Triple::nvptx64) {
        return TargetInfo(nvptx_function_info);
      } else {
        return TargetInfo(amdgpu_function_info);
      }
    }
    case TargetFunctionID::kCos: {
      TargetFunctionInfo amdgpu_function_info("__ocml_cos", false);
      TargetFunctionInfo nvptx_function_info("__nv_cos", false);
      if (target_triple.getArch() == llvm::Triple::nvptx ||
          target_triple.getArch() == llvm::Triple::nvptx64) {
        return TargetInfo(nvptx_function_info);
      } else {
        return TargetInfo(amdgpu_function_info);
      }
    }
    case TargetFunctionID::kExp: {
      TargetFunctionInfo amdgpu_function_info("__ocml_exp", false);
      TargetFunctionInfo nvptx_function_info("__nv_exp", false);
      if (target_triple.getArch() == llvm::Triple::nvptx ||
          target_triple.getArch() == llvm::Triple::nvptx64) {
        return TargetInfo(nvptx_function_info);
      } else {
        return TargetInfo(amdgpu_function_info);
      }
    }
    case TargetFunctionID::kExpm1: {
      TargetFunctionInfo amdgpu_function_info("__ocml_expm1", false);
      TargetFunctionInfo nvptx_function_info("__nv_expm1", false);
      if (target_triple.getArch() == llvm::Triple::nvptx ||
          target_triple.getArch() == llvm::Triple::nvptx64) {
        return TargetInfo(nvptx_function_info);
      } else {
        return TargetInfo(amdgpu_function_info);
      }
    }
    case TargetFunctionID::kSqrt: {
      TargetFunctionInfo amdgpu_function_info("__ocml_sqrt", false);
      TargetFunctionInfo nvptx_function_info("__nv_sqrt", false);
      if (target_triple.getArch() == llvm::Triple::nvptx ||
          target_triple.getArch() == llvm::Triple::nvptx64) {
        return TargetInfo(nvptx_function_info);
      } else {
        return TargetInfo(amdgpu_function_info);
      }
    }
    case TargetFunctionID::kRsqrt: {
      TargetFunctionInfo amdgpu_function_info("__ocml_rsqrt", false);
      TargetFunctionInfo nvptx_function_info("__nv_rsqrt", false);
      if (target_triple.getArch() == llvm::Triple::nvptx ||
          target_triple.getArch() == llvm::Triple::nvptx64) {
        return TargetInfo(nvptx_function_info);
      } else {
        return TargetInfo(amdgpu_function_info);
      }
    }
    case TargetFunctionID::kAtan2: {
      TargetFunctionInfo amdgpu_function_info("__ocml_atan2", false);
      TargetFunctionInfo nvptx_function_info("__nv_atan2", false);
      if (target_triple.getArch() == llvm::Triple::nvptx ||
          target_triple.getArch() == llvm::Triple::nvptx64) {
        return TargetInfo(nvptx_function_info);
      } else {
        return TargetInfo(amdgpu_function_info);
      }
    }
    case TargetFunctionID::kFmod: {
      TargetFunctionInfo amdgpu_function_info("__ocml_fmod", false);
      TargetFunctionInfo nvptx_function_info("__nv_fmod", false);
      if (target_triple.getArch() == llvm::Triple::nvptx ||
          target_triple.getArch() == llvm::Triple::nvptx64) {
        return TargetInfo(nvptx_function_info);
      } else {
        return TargetInfo(amdgpu_function_info);
      }
    }
    case TargetFunctionID::kRound: {
      TargetFunctionInfo amdgpu_function_info("__ocml_round", false);
      TargetFunctionInfo nvptx_function_info("__nv_round", false);
      if (target_triple.getArch() == llvm::Triple::nvptx ||
          target_triple.getArch() == llvm::Triple::nvptx64) {
        return TargetInfo(nvptx_function_info);
      } else {
        return TargetInfo(amdgpu_function_info);
      }
    }
  }
}
}  // namespace

unsigned GetGlobalMemoryAddressSpace(const llvm::Module& module) {
  llvm::Triple target_triple = llvm::Triple(module.getTargetTriple());
  if (target_triple.getArch() == llvm::Triple::amdgcn){
    return kAMDGPUGlobalMemoryAddrSpace;
  }
  return 0;
}

unsigned GetSharedMemoryAddressSpace(const llvm::Module& module) {
  llvm::Triple target_triple = llvm::Triple(module.getTargetTriple());
  if (target_triple.getArch() == llvm::Triple::nvptx ||
      target_triple.getArch() == llvm::Triple::nvptx64) {
    return kNVPTXSharedMemoryAddrSpace;
  } else if (target_triple.getArch() == llvm::Triple::amdgcn) {
    return kAMDGPUSharedMemoryAddrSpace;
  }
  return 0;
}

void AnnotateFunctionAsGpuKernel(llvm::Module* module, llvm::Function* func,
                           llvm::IRBuilder<>* b) {
  llvm::Triple target_triple = llvm::Triple(module->getTargetTriple());
  if (target_triple.getArch() == llvm::Triple::nvptx ||
      target_triple.getArch() == llvm::Triple::nvptx64) {
    // Add the declaration of this kernel to llvm.nvvm.annotations so that NVPTX
    // treats it as a CUDA kernel.
    llvm::LLVMContext& context = module->getContext();
    llvm::NamedMDNode* nvvm_annotations_node =
        module->getOrInsertNamedMetadata("nvvm.annotations");
    nvvm_annotations_node->addOperand(llvm::MDNode::get(
        context, {llvm::ConstantAsMetadata::get(func),
                  llvm::MDString::get(context, "kernel"),
                  llvm::ConstantAsMetadata::get(b->getInt32(1))}));

  } else if (target_triple.getArch() == llvm::Triple::amdgcn) {
    func->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
    func->addFnAttr("amdgpu-flat-work-group-size", "1, 1024");
  }
}

llvm::Value* EmitCallToTargetFunction(
    TargetFunctionID function_id, absl::Span<llvm::Value* const> operands,
    absl::Span<const PrimitiveType> input_types, PrimitiveType output_type,
    absl::Span<const llvm::Attribute::AttrKind> attributes,
    absl::Span<llvm::Type* const> overloaded_types,
    llvm::IRBuilder<>* b) {
  llvm::Module* module = b->GetInsertBlock()->getModule();
  llvm::Triple target_triple = llvm::Triple(module->getTargetTriple());
  struct TargetInfo gpu_info = GetTargetInfo(function_id, target_triple);

  if (gpu_info.target_intrinsic_info) {
    llvm::Function* intrinsic = llvm::Intrinsic::getDeclaration(
        module, gpu_info.target_intrinsic_info->intrinsic,
        llvm_ir::AsArrayRef(overloaded_types));
    return b->CreateCall(intrinsic, llvm_ir::AsArrayRef(operands));
  } else {
    std::vector<llvm::Value*> converted_operands;
    std::vector<llvm::Type*> ir_input_types;
    PrimitiveType from_type, to_type;
    PrimitiveType callee_result_type;
    if (gpu_info.target_function_info->check_signature) {
      CHECK_EQ(input_types.size(),
               gpu_info.target_function_info->input_types.size());
      CHECK_EQ(input_types.size(), operands.size());
      for (unsigned int index = 0; index < operands.size(); ++index) {
        to_type = gpu_info.target_function_info->input_types[index];
        from_type = input_types[index];
        if (to_type == PRIMITIVE_TYPE_INVALID) continue;
        if (from_type == to_type) {
          converted_operands.push_back(
              const_cast<llvm::Value*>(operands[index]));
        } else if (gpu_info.target_function_info->use_bitcast) {
          converted_operands.push_back(b->CreateBitCast(
              operands[index],
              llvm_ir::PrimitiveTypeToIrType(to_type, module)));
        } else if (primitive_util::IsFloatingPointType(from_type) &&
                   primitive_util::IsSignedIntegralType(to_type)) {
          converted_operands.push_back(
              b->CreateFPToSI(operands[index],
                              llvm_ir::PrimitiveTypeToIrType(to_type, module)));
        } else {
          LOG(FATAL) << "unhandled conversion operation from "
                     << PrimitiveType_Name(from_type) << "to"
                     << PrimitiveType_Name(to_type);
        }
        ir_input_types.push_back(
            llvm_ir::PrimitiveTypeToIrType(to_type, module));
      }
      callee_result_type = gpu_info.target_function_info->result_type;
    } else {
      converted_operands.assign(operands.begin(), operands.end());
      for (PrimitiveType input_type : input_types) {
        ir_input_types.push_back(
            llvm_ir::PrimitiveTypeToIrType(input_type, module));
      }
      callee_result_type = output_type;
    }
    llvm::FunctionType* callee_type = llvm::FunctionType::get(
        llvm_ir::PrimitiveTypeToIrType(callee_result_type,
                                       module),  // Return type.
        ir_input_types,                          // Parameter types.
        false);                                  // No variadic arguments.

    string munged_callee = gpu_info.target_function_info->callee_name;
    switch (callee_result_type) {
      case S32:
        StrAppend(&munged_callee, "_i32");
        break;
      case S64:
        StrAppend(&munged_callee, "_i64");
        break;
      case F32:
        StrAppend(&munged_callee, "_f32");
        break;
      case F64:
        StrAppend(&munged_callee, "_f64");
        break;
      default:
        LOG(FATAL) << "Bad Type " << PrimitiveType_Name(callee_result_type)
                   << "\n";
    }
    // Declares the callee if it is not declared already.
    llvm::Function* callee = llvm::dyn_cast<llvm::Function>(
            b->GetInsertBlock()->getModule()->getOrInsertFunction(
            llvm_ir::AsStringRef(munged_callee), callee_type).getCallee());
    for (auto attribute : attributes) {
      callee->addFnAttr(attribute);
    }
    llvm::Value* result =  b->CreateCall(callee, llvm_ir::AsArrayRef(converted_operands));

    from_type = callee_result_type;
    to_type = output_type;
    if (from_type == to_type){
      return result;
    } else if (gpu_info.target_function_info->use_bitcast) {
      llvm::Value* converted_result = b->CreateBitCast(
          result, llvm_ir::PrimitiveTypeToIrType(to_type, module));
      return converted_result;
    } else if (primitive_util::IsFloatingPointType(to_type) &&
               primitive_util::IsSignedIntegralType(from_type)) {
      llvm::Value* converted_result = b->CreateSIToFP(
          result, llvm_ir::PrimitiveTypeToIrType(to_type, module));
      return converted_result;
    }
  }
  }

}  // namespace gpu
}  // namespace xla
