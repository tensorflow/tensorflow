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

#include <cstring>
#include <string>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/TargetParser/Triple.h"
#include "xla/primitive_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {
// Utility functions to obtain NVPTX/AMDGPU specific information.
using absl::StrCat;

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
    case TargetDeviceFunctionID::kAcos: {
      return {"__nv_acos", "__ocml_acos", "_Z16__spirv_ocl_acos"};
    }
    case TargetDeviceFunctionID::kAcosh: {
      return {"__nv_acosh", "__ocml_acosh", "_Z17__spirv_ocl_acosh"};
    }
    case TargetDeviceFunctionID::kAtan2: {
      return {"__nv_atan2", "__ocml_atan2", "_Z17__spirv_ocl_atan2"};
    }
    case TargetDeviceFunctionID::kAtan: {
      return {"__nv_atan", "__ocml_atan", "_Z16__spirv_ocl_atan"};
    }
    case TargetDeviceFunctionID::kAsin: {
      return {"__nv_asin", "__ocml_asin", "_Z16__spirv_ocl_asin"};
    }
    case TargetDeviceFunctionID::kAsinh: {
      return {"__nv_asinh", "__ocml_asinh", "_Z17__spirv_ocl_asinh"};
    }
    case TargetDeviceFunctionID::kAtanh: {
      return {"__nv_atanh", "__ocml_atanh", "_Z17__spirv_ocl_atanh"};
    }
    case TargetDeviceFunctionID::kCos: {
      return {"__nv_cos", "__ocml_cos", "_Z15__spirv_ocl_cos"};
    }
    case TargetDeviceFunctionID::kCosh: {
      return {"__nv_cosh", "__ocml_cosh", "_Z16__spirv_ocl_cosh"};
    }
    case TargetDeviceFunctionID::kErf: {
      return {"__nv_erf", "__ocml_erf", "_Z15__spirv_ocl_erf"};
    }
    case TargetDeviceFunctionID::kExp: {
      return {"__nv_exp", "__ocml_exp", "_Z15__spirv_ocl_exp"};
    }
    case TargetDeviceFunctionID::kExp2: {
      return {"__nv_exp2", "__ocml_exp2", "_Z16__spirv_ocl_exp2"};
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
    case TargetDeviceFunctionID::kLog2: {
      return {"__nv_log2", "__ocml_log2", "_Z16__spirv_ocl_log2"};
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
    case TargetDeviceFunctionID::kSinh: {
      return {"__nv_sinh", "__ocml_sinh", "_Z16__spirv_ocl_sinh"};
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
    case TargetDeviceFunctionID::kRint: {
      return {"__nv_rint", "__ocml_rint", "_Z16__spirv_ocl_rint"};
    }
  }
  LOG(FATAL) << "Invalid TargetDeviceFunctionID: " << static_cast<int>(func_id);
}
}  // namespace

bool HasF16Implementation(TargetDeviceFunctionID func_id,
                          llvm::Triple target_triple) {
  return target_triple.isAMDGPU() &&
         (func_id == TargetDeviceFunctionID::kAtan2 ||
          func_id == TargetDeviceFunctionID::kAtan ||
          func_id == TargetDeviceFunctionID::kCbrt ||
          func_id == TargetDeviceFunctionID::kCos ||
          func_id == TargetDeviceFunctionID::kExp ||
          func_id == TargetDeviceFunctionID::kExp2 ||
          func_id == TargetDeviceFunctionID::kExpm1 ||
          func_id == TargetDeviceFunctionID::kFmod ||
          func_id == TargetDeviceFunctionID::kHypot ||
          func_id == TargetDeviceFunctionID::kLog ||
          func_id == TargetDeviceFunctionID::kLog1p ||
          func_id == TargetDeviceFunctionID::kLog2 ||
          func_id == TargetDeviceFunctionID::kPow ||
          func_id == TargetDeviceFunctionID::kRsqrt ||
          func_id == TargetDeviceFunctionID::kSin ||
          func_id == TargetDeviceFunctionID::kSqrt ||
          func_id == TargetDeviceFunctionID::kTan ||
          func_id == TargetDeviceFunctionID::kTanh ||
          func_id == TargetDeviceFunctionID::kErf ||
          func_id == TargetDeviceFunctionID::kAcosh ||
          func_id == TargetDeviceFunctionID::kAcos ||
          func_id == TargetDeviceFunctionID::kSinh ||
          func_id == TargetDeviceFunctionID::kAsin ||
          func_id == TargetDeviceFunctionID::kAsinh ||
          func_id == TargetDeviceFunctionID::kCosh ||
          func_id == TargetDeviceFunctionID::kAtanh);
}

namespace {
// TODO(b/370452608): Add more functions that have a fast approximation for f32
// that we can use for f16 types.
bool HasFastF32Approximation(TargetDeviceFunctionID func_id) {
  return func_id == TargetDeviceFunctionID::kExp ||
         func_id == TargetDeviceFunctionID::kLog;
}
}  // namespace

std::string ObtainDeviceFunctionName(TargetDeviceFunctionID func_id,
                                     PrimitiveType output_type,
                                     llvm::Triple target_triple) {
  // The device math functions differentiate between "double" and "float" by
  // appending a double or float specific suffix to a root name. The suffix and
  // the root name are specific to the target.
  struct TargetDeviceFunction gpu_root_names = GetDeviceFunctionRoot(func_id);
  if (target_triple.isNVPTX()) {
    bool is_supported_output_type =
        output_type == BF16 || output_type == F16 || output_type == F32;
    if (is_supported_output_type) {
      std::string function_name = StrCat(gpu_root_names.nvptx_root, "f");
      if (HasFastF32Approximation(func_id) &&
          (output_type == BF16 || output_type == F16)) {
        // All function names start with "__nv". The approximate version of the
        // function names continues with "_fast".
        return function_name.insert(strlen("__nv"), "_fast");
      }
      return function_name;
    } else if (output_type == F64) {
      return gpu_root_names.nvptx_root;
    } else {
      LOG(FATAL) << "Unexpected type while getting device function name: "
                 << primitive_util::LowercasePrimitiveTypeName(output_type);
    }
  } else if (target_triple.getArch() == llvm::Triple::amdgpu) {
    // TODO(b/370452608): Are there approximate functions we can use for BF16
    // and F16 types?
    if (output_type == F16 && HasF16Implementation(func_id, target_triple)) {
      // All these functions have f16 implementation - no need for conversion
      return StrCat(gpu_root_names.amdgpu_root, "_f16");
    }
    if (output_type == BF16 || output_type == F16 || output_type == F32) {
      return StrCat(gpu_root_names.amdgpu_root, "_f32");
    } else if (output_type == F64) {
      return StrCat(gpu_root_names.amdgpu_root, "_f64");
    } else {
      LOG(FATAL) << "Unexpected type while getting device function name.";
    }
  } else if (target_triple.isSPIROrSPIRV()) {
    // TODO(b/370452608): Are there approximate functions we can use for BF16
    // and F16 types?
    if (output_type == BF16 || output_type == F16 || output_type == F32) {
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

void AnnotateFunctionAsGpuKernel(llvm::Module* module, llvm::Function* func,
                                 llvm::IRBuilderBase* b) {
  llvm::Triple target_triple = llvm::Triple(module->getTargetTriple());
  if (target_triple.isNVPTX()) {
    // Attach information so NVPTX can recognize function as a CUDA kernel.
    func->setCallingConv(llvm::CallingConv::PTX_Kernel);

  } else if (target_triple.getArch() == llvm::Triple::amdgpu) {
    // Attach information so AMDGPU can recognize function as a AMDGPU kernel.
    func->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
    func->addFnAttr("uniform-work-group-size", "true");
  } else if (target_triple.isSPIROrSPIRV()) {
    // Attach information so that it can be recognized as a SPIR kernel.
    func->setCallingConv(llvm::CallingConv::SPIR_KERNEL);
  } else {
    LOG(FATAL) << "Invalid triple " << target_triple.str();
  }
}

}  // namespace gpu
}  // namespace xla
