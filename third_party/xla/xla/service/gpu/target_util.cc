/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/service/gpu/target_util.h"

#include <functional>
#include <string>
#include <variant>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/FPEnv.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/TargetParser/Triple.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/service/llvm_ir/llvm_type_conversion_util.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/status.h"
#include "xla/util.h"
#include "tsl/platform/logging.h"

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
  std::variant<llvm::Intrinsic::ID,
               std::function<llvm::CallInst*(llvm::IRBuilder<>*)>>
      spir_intrinsic_or_function;
};

// Gets the llvm intrinsic ids on different platforms (NVPTX, AMDGPU)
// corresponding to the give TargetIntrinsicID.
struct TargetIntrinsics GetIntrinsic(TargetIntrinsicID intrin) {
  switch (intrin) {
    case TargetIntrinsicID::kThreadIdx: {
      return {
          llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x,
          llvm::Intrinsic::amdgcn_workitem_id_x,
          [](llvm::IRBuilder<>* b_) -> llvm::CallInst* {
            return EmitDeviceFunctionCall(
                "_Z32__spirv_BuiltInLocalInvocationIdi", {b_->getInt32(0)},
                {U32}, U64, {b_->getContext()}, b_);
          },
      };
    }
    case TargetIntrinsicID::kThreadIdy: {
      return {
          llvm::Intrinsic::nvvm_read_ptx_sreg_tid_y,
          llvm::Intrinsic::amdgcn_workitem_id_y,
          [](llvm::IRBuilder<>* b_) -> llvm::CallInst* {
            return EmitDeviceFunctionCall(
                "_Z32__spirv_BuiltInLocalInvocationIdi", {b_->getInt32(1)},
                {U32}, U64, {b_->getContext()}, b_);
          },
      };
    }
    case TargetIntrinsicID::kThreadIdz: {
      return {
          llvm::Intrinsic::nvvm_read_ptx_sreg_tid_z,
          llvm::Intrinsic::amdgcn_workitem_id_z,
          [](llvm::IRBuilder<>* b_) -> llvm::CallInst* {
            return EmitDeviceFunctionCall(
                "_Z32__spirv_BuiltInLocalInvocationIdi", {b_->getInt32(2)},
                {U32}, U64, {b_->getContext()}, b_);
          },
      };
    }
    case TargetIntrinsicID::kBlockIdx: {
      return {
          llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_x,
          llvm::Intrinsic::amdgcn_workgroup_id_x,
          [](llvm::IRBuilder<>* b_) -> llvm::CallInst* {
            return EmitDeviceFunctionCall("_Z26__spirv_BuiltInWorkgroupIdi",
                                          {b_->getInt32(0)}, {U32}, U64,
                                          {b_->getContext()}, b_);
          },
      };
    }
    case TargetIntrinsicID::kBlockIdy: {
      return {
          llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_y,
          llvm::Intrinsic::amdgcn_workgroup_id_y,
          [](llvm::IRBuilder<>* b_) -> llvm::CallInst* {
            return EmitDeviceFunctionCall("_Z26__spirv_BuiltInWorkgroupIdi",
                                          {b_->getInt32(1)}, {U32}, U64,
                                          {b_->getContext()}, b_);
          },
      };
    }
    case TargetIntrinsicID::kBlockIdz: {
      return {
          llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_z,
          llvm::Intrinsic::amdgcn_workgroup_id_z,
          [](llvm::IRBuilder<>* b_) -> llvm::CallInst* {
            return EmitDeviceFunctionCall("_Z26__spirv_BuiltInWorkgroupIdi",
                                          {b_->getInt32(2)}, {U32}, U64,
                                          {b_->getContext()}, b_);
          },
      };
    }
    case TargetIntrinsicID::kBarrierId: {
      return {llvm::Intrinsic::nvvm_barrier0, llvm::Intrinsic::amdgcn_s_barrier,
              [](llvm::IRBuilder<>* b_) -> llvm::CallInst* {
                return EmitDeviceFunctionCall(
                    "_Z22__spirv_ControlBarrierjjj",
                    {b_->getInt32(2), b_->getInt32(2), b_->getInt32(272)},
                    {U32, U32, U32}, U32,
                    llvm::AttrBuilder(b_->getContext())
                        .addAttribute(llvm::Attribute::Convergent),
                    b_);
              }};
    }
    case TargetIntrinsicID::kBlockDimx: {
      return {llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_x,
              [](llvm::IRBuilder<>* b_) -> llvm::CallInst* {
                return EmitDeviceFunctionCall("__ockl_get_local_size",
                                              {b_->getInt32(0)}, {U32}, U64,
                                              {b_->getContext()}, b_);
              },
              [](llvm::IRBuilder<>* b_) -> llvm::CallInst* {
                return EmitDeviceFunctionCall(
                    "_Z28__spirv_BuiltInWorkgroupSizei", {b_->getInt32(0)},
                    {U32}, U64, {b_->getContext()}, b_);
              }};
    }
    case TargetIntrinsicID::kBlockDimy: {
      return {llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_y,
              [](llvm::IRBuilder<>* b_) -> llvm::CallInst* {
                return EmitDeviceFunctionCall("__ockl_get_local_size",
                                              {b_->getInt32(1)}, {U32}, U64,
                                              {b_->getContext()}, b_);
              },
              [](llvm::IRBuilder<>* b_) -> llvm::CallInst* {
                return EmitDeviceFunctionCall(
                    "_Z28__spirv_BuiltInWorkgroupSizei", {b_->getInt32(1)},
                    {U32}, U64, {b_->getContext()}, b_);
              }};
    }
    case TargetIntrinsicID::kBlockDimz: {
      return {llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_z,
              [](llvm::IRBuilder<>* b_) -> llvm::CallInst* {
                return EmitDeviceFunctionCall("__ockl_get_local_size",
                                              {b_->getInt32(2)}, {U32}, U64,
                                              {b_->getContext()}, b_);
              },
              [](llvm::IRBuilder<>* b_) -> llvm::CallInst* {
                return EmitDeviceFunctionCall(
                    "_Z28__spirv_BuiltInWorkgroupSizei", {b_->getInt32(2)},
                    {U32}, U64, {b_->getContext()}, b_);
              }};
    }
    case TargetIntrinsicID::kGroupBarrierId: {
      return {llvm::Intrinsic::nvvm_bar_warp_sync,
              llvm::Intrinsic::amdgcn_wave_barrier,
              [](llvm::IRBuilder<>* b_) -> llvm::CallInst* {
                return EmitDeviceFunctionCall(
                    "_Z22__spirv_ControlBarrierjjj",
                    {b_->getInt32(2), b_->getInt32(2), b_->getInt32(272)},
                    {U32, U32, U32}, U32,
                    llvm::AttrBuilder(b_->getContext())
                        .addAttribute(llvm::Attribute::Convergent),
                    b_);
              }};
    }
  }
}

// Wrapper structure for carrying math functions for NVPTX/AMDGPU platforms.
struct TargetDeviceFunction {
  const std::string nvptx_root;
  const std::string amdgpu_root;
  const std::string spir_root;
};

// Gets the device function name on different platforms (NVPTX, AMDGPU)
// corresponding to the given TargetDeviceFunctionID.
struct TargetDeviceFunction GetDeviceFunctionRoot(
    TargetDeviceFunctionID func_id) {
  switch (func_id) {
    case TargetDeviceFunctionID::kAtan2: {
      return {"__nv_atan2", "__ocml_atan2", "_Z17__spirv_ocl_atan2"};
    }
    case TargetDeviceFunctionID::kCos: {
      return {"__nv_cos", "__ocml_cos", "_Z15__spirv_ocl_cos"};
    }
    case TargetDeviceFunctionID::kErf: {
      return {"__nv_erf", "__ocml_erf", "_Z15__spirv_ocl_erf"};
    }
    case TargetDeviceFunctionID::kExp: {
      return {"__nv_exp", "__ocml_exp", "_Z15__spirv_ocl_exp"};
    }
    case TargetDeviceFunctionID::kExpm1: {
      return {"__nv_expm1", "__ocml_expm1", "_Z17__spirv_ocl_expm1"};
    }
    case TargetDeviceFunctionID::kFmod: {
      return {"__nv_fmod", "__ocml_fmod", "_Z16__spirv_ocl_fmod"};
    }
    case TargetDeviceFunctionID::kHypot: {
      return {"__nv_hypot", "__ocml_hypot", "_Z17__spirv_ocl_hypot"};
    }
    case TargetDeviceFunctionID::kLog: {
      return {"__nv_log", "__ocml_log", "_Z15__spirv_ocl_log"};
    }
    case TargetDeviceFunctionID::kLog1p: {
      return {"__nv_log1p", "__ocml_log1p", "_Z17__spirv_ocl_log1p"};
    }
    case TargetDeviceFunctionID::kPow: {
      return {"__nv_pow", "__ocml_pow", "_Z15__spirv_ocl_pow"};
    }
    case TargetDeviceFunctionID::kRsqrt: {
      return {"__nv_rsqrt", "__ocml_rsqrt", "_Z17__spirv_ocl_rsqrt"};
    }
    case TargetDeviceFunctionID::kSin: {
      return {"__nv_sin", "__ocml_sin", "_Z15__spirv_ocl_sin"};
    }
    case TargetDeviceFunctionID::kSqrt: {
      return {"__nv_sqrt", "__ocml_sqrt", "_Z16__spirv_ocl_sqrt"};
    }
    case TargetDeviceFunctionID::kTan: {
      return {"__nv_tan", "__ocml_tan", "_Z15__spirv_ocl_tan"};
    }
    case TargetDeviceFunctionID::kTanh: {
      return {"__nv_tanh", "__ocml_tanh", "_Z16__spirv_ocl_tanh"};
    }
    case TargetDeviceFunctionID::kCbrt: {
      return {"__nv_cbrt", "__ocml_cbrt", "_Z16__spirv_ocl_cbrt"};
    }
  }
}
}  // namespace

absl::StatusOr<TargetDeviceFunctionID> GetTargetDeviceFunctionID(HloOpcode op) {
  switch (op) {
    case HloOpcode::kAtan2:
      return TargetDeviceFunctionID::kAtan2;
    case HloOpcode::kCos:
      return TargetDeviceFunctionID::kCos;
    case HloOpcode::kExp:
      return TargetDeviceFunctionID::kExp;
    case HloOpcode::kErf:
      return TargetDeviceFunctionID::kErf;
    case HloOpcode::kExpm1:
      return TargetDeviceFunctionID::kExpm1;
    case HloOpcode::kLog:
      return TargetDeviceFunctionID::kLog;
    case HloOpcode::kLog1p:
      return TargetDeviceFunctionID::kLog1p;
    case HloOpcode::kPower:
      return TargetDeviceFunctionID::kPow;
    case HloOpcode::kRemainder:
      return TargetDeviceFunctionID::kFmod;
    case HloOpcode::kRsqrt:
      return TargetDeviceFunctionID::kRsqrt;
    case HloOpcode::kSin:
      return TargetDeviceFunctionID::kSin;
    case HloOpcode::kSqrt:
      return TargetDeviceFunctionID::kSqrt;
    case HloOpcode::kTan:
      return TargetDeviceFunctionID::kTan;
    case HloOpcode::kTanh:
      return TargetDeviceFunctionID::kTanh;
    case HloOpcode::kCbrt:
      return TargetDeviceFunctionID::kCbrt;
    default:
      break;
  }
  return NotFound("The HLO opcode %s is not mapped to a device function",
                  HloOpcodeString(op));
}

std::string ObtainDeviceFunctionName(TargetDeviceFunctionID func_id,
                                     PrimitiveType output_type,
                                     llvm::Triple target_triple) {
  // The device math functions differentiate between "double" and "float" by
  // appending a double or float specific suffix to a root name. The suffix and
  // the root name are specific to the target.
  struct TargetDeviceFunction gpu_root_names = GetDeviceFunctionRoot(func_id);
  if (target_triple.isNVPTX()) {
    if (output_type == F32) {
      return StrCat(gpu_root_names.nvptx_root, "f");
    } else if (output_type == F64) {
      return gpu_root_names.nvptx_root;
    } else {
      LOG(FATAL) << "Unexpected type while getting device function name: "
                 << primitive_util::LowercasePrimitiveTypeName(output_type);
    }
  } else if (target_triple.getArch() == llvm::Triple::amdgcn) {
    if (output_type == F32) {
      return StrCat(gpu_root_names.amdgpu_root, "_f32");
    } else if (output_type == F64) {
      return StrCat(gpu_root_names.amdgpu_root, "_f64");
    } else {
      LOG(FATAL) << "Unexpected type while getting device function name.";
    }
  } else if (target_triple.isSPIR()) {
    if (output_type == F32) {
      if (gpu_root_names.spir_root == "_Z17__spirv_ocl_hypot" ||
          gpu_root_names.spir_root == "_Z15__spirv_ocl_pow" ||
          gpu_root_names.spir_root == "_Z17__spirv_ocl_atan2" ||
          gpu_root_names.spir_root == "_Z16__spirv_ocl_fmod") {
        return StrCat(gpu_root_names.spir_root, "ff");
      } else {
        return StrCat(gpu_root_names.spir_root, "f");
      }
    } else if (output_type == F64) {
      if (gpu_root_names.spir_root == "_Z17__spirv_ocl_hypot" ||
          gpu_root_names.spir_root == "_Z15__spirv_ocl_pow" ||
          gpu_root_names.spir_root == "_Z17__spirv_ocl_atan2" ||
          gpu_root_names.spir_root == "_Z16__spirv_ocl_fmod") {
        return StrCat(gpu_root_names.spir_root, "dd");
      } else {
        return StrCat(gpu_root_names.spir_root, "d");
      }
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
  llvm::Triple target_triple = llvm::Triple(module->getTargetTriple());
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
  if (target_triple.isSPIR())
    callee->setCallingConv(llvm::CallingConv::SPIR_FUNC);

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
  } else if (target_triple.isSPIR()) {
    llvm::Intrinsic::ID* llvm_intrinsic_id_ptr =
        std::get_if<llvm::Intrinsic::ID>(
            &gpu_intrinsic_id.spir_intrinsic_or_function);
    if (llvm_intrinsic_id_ptr) {
      llvm_intrinsic_id = *llvm_intrinsic_id_ptr;
    } else {
      std::function<llvm::CallInst*(llvm::IRBuilder<>*)>* builder_func =
          std::get_if<std::function<llvm::CallInst*(llvm::IRBuilder<>*)>>(
              &gpu_intrinsic_id.spir_intrinsic_or_function);
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
  } else if (target_triple.isSPIR()) {
    // Attach information so that it can be recognized as a SPIR kernel.
    func->setCallingConv(llvm::CallingConv::SPIR_KERNEL);
  } else {
    LOG(FATAL) << "Invalid triple " << target_triple.str();
  }
}

}  // namespace gpu
}  // namespace xla
