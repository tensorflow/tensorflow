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

#include "llvm/IR/MDBuilder.h"
#include "absl/strings/str_cat.h"
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

//Wrapper structure for  intrinsic information for NVPTX/AMDGPU. 
  struct TargetIntrinsicInfo {
    TargetIntrinsicInfo() : intrinsic(llvm::Intrinsic::not_intrinsic) {}
    TargetIntrinsicInfo(llvm::Intrinsic::ID x) : intrinsic(x) {}
    llvm::Intrinsic::ID intrinsic;
  };

//Wrapper structure for  device functionc information for NVPTX/AMDGPU. 
  struct TargetFunctionInfo {
    TargetFunctionInfo() : callee_name(""), 
                  input_types({ PRIMITIVE_TYPE_INVALID }),
                  result_type(PRIMITIVE_TYPE_INVALID),
                  use_bitcast(false) {}
    TargetFunctionInfo(const string callee_name_) : callee_name(callee_name_), 
                  input_types({ PRIMITIVE_TYPE_INVALID }),
                  result_type(PRIMITIVE_TYPE_INVALID),
                  use_bitcast(false) {}
    TargetFunctionInfo(const string callee_name_, 
                       absl::Span<const PrimitiveType> input_types_, 
                       const PrimitiveType output_type_,
                       bool use_bitcast_) : 
                  callee_name(callee_name_), 
                  input_types(input_types_),
                  result_type(output_type_),
                  use_bitcast(use_bitcast_) {}
    // Device function name. 
    const string callee_name;
    // Inpute types accespted by the device function.
    absl::Span<const PrimitiveType> input_types;
    // Result type of the device function.
    PrimitiveType result_type;
    // Use bitcast to generate casts if the desired signature at the call site 
    // does not match the signature of the device function.
    bool use_bitcast;
  } target_function_info; 

// Wrapper structure to carry either information about the intrinsic
// or device function for NVPTX/AMDGPU.
struct TargetInfo {
  struct TargetIntrinsicInfo target_intrinsic_info;
  struct TargetFunctionInfo target_function_info; 
  TargetInfo( struct TargetIntrinsicInfo x, struct TargetFunctionInfo y): 
                     target_intrinsic_info(x), target_function_info(y) {}
};


// Wrapper structure for carrying function information for NVPTX/AMDGPU platforms.
struct MultipleTargetInfo {
  struct TargetInfo nvptx_info;
  struct TargetInfo amdgpu_info;
  MultipleTargetInfo( struct TargetInfo nvptx_info_,
                      struct TargetInfo amdgpu_info_): 
                     nvptx_info(nvptx_info_), amdgpu_info(amdgpu_info_) {}
};

// Populates the function information for different platforms (NVPTX, AMDGPU)
// corresponding to the given TargetFunctionID.
struct MultipleTargetInfo GetTargetInfo(TargetFunctionID function_id) {
  TargetFunctionInfo default_nvptx_function_info, default_amdgpu_function_info;
  TargetIntrinsicInfo default_nvptx_intrinsic_info, default_amdgpu_intrinsic_info;
  switch (function_id) {
    case TargetFunctionID::kShflDownF32:{ 
      TargetIntrinsicInfo nvptx_intrinsic_info(llvm::Intrinsic::nvvm_shfl_sync_down_f32);
      TargetFunctionInfo amdgpu_function_info(
           "__ockl_readuplane" ,
           { PRIMITIVE_TYPE_INVALID, S32, S32, PRIMITIVE_TYPE_INVALID}, 
                  S32, true);
      TargetInfo nvptx_info(nvptx_intrinsic_info, default_nvptx_function_info);
      TargetInfo amdgpu_info(default_amdgpu_intrinsic_info, amdgpu_function_info);
      return MultipleTargetInfo(nvptx_info, amdgpu_info);
    }
    case TargetFunctionID::kShflDownI32:{ 
      TargetIntrinsicInfo nvptx_intrinsic_info(llvm::Intrinsic::nvvm_shfl_sync_down_i32);
      TargetFunctionInfo amdgpu_function_info(
           "__ockl_readuplane" ,
           { PRIMITIVE_TYPE_INVALID, S32, S32, PRIMITIVE_TYPE_INVALID}, 
                  S32, true);

      TargetInfo nvptx_info(nvptx_intrinsic_info, default_nvptx_function_info);
      TargetInfo amdgpu_info(default_amdgpu_intrinsic_info, amdgpu_function_info);
      return MultipleTargetInfo(nvptx_info, amdgpu_info);
    }
    case TargetFunctionID::kThreadIdx: {
      TargetIntrinsicInfo nvptx_intrinsic_info(llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x);
      TargetIntrinsicInfo amdgpu_intrinsic_info(llvm::Intrinsic::amdgcn_workitem_id_x);
      TargetInfo nvptx_info(nvptx_intrinsic_info, default_nvptx_function_info);
      TargetInfo amdgpu_info(amdgpu_intrinsic_info, default_amdgpu_function_info);
      return MultipleTargetInfo(nvptx_info, amdgpu_info);
    }

    case TargetFunctionID::kThreadIdy: {
      TargetIntrinsicInfo nvptx_intrinsic_info(llvm::Intrinsic::nvvm_read_ptx_sreg_tid_y);

      TargetIntrinsicInfo amdgpu_intrinsic_info(llvm::Intrinsic::amdgcn_workitem_id_y);
      TargetInfo nvptx_info(nvptx_intrinsic_info, default_nvptx_function_info);
      TargetInfo amdgpu_info(amdgpu_intrinsic_info, default_amdgpu_function_info);
      return MultipleTargetInfo(nvptx_info, amdgpu_info);
    }

    case TargetFunctionID::kThreadIdz: {
      TargetIntrinsicInfo nvptx_intrinsic_info(llvm::Intrinsic::nvvm_read_ptx_sreg_tid_z);
      TargetIntrinsicInfo amdgpu_intrinsic_info(llvm::Intrinsic::amdgcn_workitem_id_z);
      TargetInfo nvptx_info(nvptx_intrinsic_info, default_nvptx_function_info);
      TargetInfo amdgpu_info(amdgpu_intrinsic_info, default_amdgpu_function_info);
      return MultipleTargetInfo(nvptx_info, amdgpu_info);
    }

    case TargetFunctionID::kBlockIdx: {
      TargetIntrinsicInfo nvptx_intrinsic_info(llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_x);
      TargetIntrinsicInfo amdgpu_intrinsic_info(llvm::Intrinsic::amdgcn_workgroup_id_x);
      TargetInfo nvptx_info(nvptx_intrinsic_info, default_nvptx_function_info);
      TargetInfo amdgpu_info(amdgpu_intrinsic_info, default_amdgpu_function_info);
      return MultipleTargetInfo(nvptx_info, amdgpu_info);
    }
    case TargetFunctionID::kBlockIdy: {
      TargetIntrinsicInfo nvptx_intrinsic_info(llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_y);
      TargetIntrinsicInfo amdgpu_intrinsic_info(llvm::Intrinsic::amdgcn_workgroup_id_y);
      TargetInfo nvptx_info(nvptx_intrinsic_info, default_nvptx_function_info);
      TargetInfo amdgpu_info(amdgpu_intrinsic_info, default_amdgpu_function_info);
      return MultipleTargetInfo(nvptx_info, amdgpu_info);
    }
    case TargetFunctionID::kBlockIdz: {
      TargetIntrinsicInfo nvptx_intrinsic_info(llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_z);
      TargetIntrinsicInfo amdgpu_intrinsic_info(llvm::Intrinsic::amdgcn_workgroup_id_z);
      TargetInfo nvptx_info(nvptx_intrinsic_info, default_nvptx_function_info);
      TargetInfo amdgpu_info(amdgpu_intrinsic_info, default_amdgpu_function_info);
      return MultipleTargetInfo(nvptx_info, amdgpu_info);
    }
    case TargetFunctionID::kBarrierId: {
      TargetIntrinsicInfo nvptx_intrinsic_info(llvm::Intrinsic::nvvm_barrier0);
      TargetIntrinsicInfo amdgpu_intrinsic_info(llvm::Intrinsic::amdgcn_s_barrier);
      TargetInfo nvptx_info(nvptx_intrinsic_info, default_nvptx_function_info);
      TargetInfo amdgpu_info(amdgpu_intrinsic_info, default_amdgpu_function_info);
      return MultipleTargetInfo(nvptx_info, amdgpu_info);
    }
  }
}
}  // namespace

llvm::Value* EmitCallToTargetFunction(
    TargetFunctionID function_id, absl::Span<llvm::Value* const> operands,
    absl::Span<const PrimitiveType> input_types, PrimitiveType output_type,
    absl::Span<const llvm::Attribute::AttrKind> attributes,
    absl::Span<llvm::Type* const> overloaded_types,
    llvm::IRBuilder<>* b) {
  llvm::Module* module = b->GetInsertBlock()->getModule();
  struct MultipleTargetInfo all_gpu_info = GetTargetInfo(function_id);
  llvm::Triple target_triple = llvm::Triple(module->getTargetTriple());
  struct TargetInfo* gpu_info;

  if ((target_triple.getArch() == llvm::Triple::nvptx) ||
      (target_triple.getArch() == llvm::Triple::nvptx64)) {
    gpu_info  = &(all_gpu_info.nvptx_info);
  } else if (target_triple.getArch() == llvm::Triple::amdgcn) {
    gpu_info  = &(all_gpu_info.amdgpu_info);
  } else {
    LOG(FATAL) << "Invalid triple " << target_triple.str();
  }

  if (gpu_info->target_intrinsic_info.intrinsic
      != llvm::Intrinsic::not_intrinsic){
    llvm::Function* intrinsic = llvm::Intrinsic::getDeclaration(
      module, gpu_info->target_intrinsic_info.intrinsic, llvm_ir::AsArrayRef(overloaded_types));
    return b->CreateCall(intrinsic, llvm_ir::AsArrayRef(operands));
  }
  else { 
    std::vector<llvm::Value*> converted_operands;
    std::vector<llvm::Type*> ir_input_types;
    auto indices = gpu_info->target_function_info.input_types.size();
    PrimitiveType from_type, to_type;
    CHECK_EQ(input_types.size(), 
       gpu_info->target_function_info.input_types.size());
    CHECK_EQ(input_types.size(), operands.size());
    for (unsigned int index = 0; index < operands.size(); ++index){
     to_type = gpu_info->target_function_info.input_types[index];
     from_type = input_types[index];
    if (to_type == PRIMITIVE_TYPE_INVALID)
      continue;
     if (from_type == to_type){
	converted_operands.push_back(const_cast<llvm::Value*>(operands[index]));
     }
     else if (gpu_info->target_function_info.use_bitcast){
        converted_operands.push_back(b->CreateBitCast(operands[index],
                      llvm_ir::PrimitiveTypeToIrType(to_type, module)));
     }
     else if( primitive_util::IsFloatingPointType(from_type) && 
             primitive_util::IsSignedIntegralType(to_type) ) {
        converted_operands.push_back(b->CreateFPToSI(operands[index],
                      llvm_ir::PrimitiveTypeToIrType(to_type, module)));
      }
      else {
       LOG(FATAL) << "unhandled conversion operation from " << PrimitiveType_Name(from_type) << "to" << PrimitiveType_Name(to_type);
      }
      ir_input_types.push_back(
          llvm_ir::PrimitiveTypeToIrType(to_type, module));
    }
    llvm::FunctionType* callee_type = llvm::FunctionType::get(
        llvm_ir::PrimitiveTypeToIrType(gpu_info->target_function_info.result_type, module),  // Return type.
       ir_input_types,                                       // Parameter types.
        false);  // No variadic arguments.

   string munged_callee = gpu_info->target_function_info.callee_name;
   switch (gpu_info->target_function_info.result_type) {
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
       LOG(FATAL) << "Bad Type " << PrimitiveType_Name(output_type) << "\n";
   }
    // Declares the callee if it is not declared already.
    llvm::Function* callee = llvm::dyn_cast<llvm::Function>(
            b->GetInsertBlock()->getModule()->getOrInsertFunction(
            llvm_ir::AsStringRef(munged_callee), callee_type).getCallee());
    for (auto attribute : attributes) {
      callee->addFnAttr(attribute);
    }
    llvm::Value* result =  b->CreateCall(callee, llvm_ir::AsArrayRef(converted_operands));

    from_type = gpu_info->target_function_info.result_type;
    to_type = output_type;
    if (from_type == to_type){
      return result;
    }
    else if (gpu_info->target_function_info.use_bitcast){
        int bit_width = result->getType()->getPrimitiveSizeInBits();
        llvm::Value* converted_result= b->CreateBitCast(result,
                      llvm_ir::PrimitiveTypeToIrType(to_type, module));
        return converted_result;
    }
    else if( primitive_util::IsFloatingPointType(to_type) && 
             primitive_util::IsSignedIntegralType(from_type) ) {
        llvm::Value* converted_result= b->CreateSIToFP(result,
                      llvm_ir::PrimitiveTypeToIrType(to_type, module));
        return converted_result;
    }
   }
  }

}  // namespace gpu
}  // namespace xla
