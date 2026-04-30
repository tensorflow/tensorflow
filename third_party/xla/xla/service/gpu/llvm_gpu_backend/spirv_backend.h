/* Copyright 2025 The OpenXLA Authors.

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

// LLVM-based compiler backend.
#ifndef XLA_SERVICE_GPU_LLVM_GPU_BACKEND_SPIRV_BACKEND_H_
#define XLA_SERVICE_GPU_LLVM_GPU_BACKEND_SPIRV_BACKEND_H_

#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "llvm/IR/Module.h"
#include "llvm/lib/Target/SPIRV/MCTargetDesc/SPIRVBaseInfo.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla.pb.h"

namespace xla::gpu::spirv {

// SPIR-V extension list to be used by Intel GPU runtime. The runtime
// invokes Intel Graphics Compiler (IGC) to compile SPIR-V to native GPU binary.
// This is a common list. Based on the GpuComputeCapability, an extended list
// can be constructed dynamically by adding more extensions.
// TODO(intel-tf): The current list is based on Intel Arc B580 (Battlemage) GPU.
// Adjust the list to support other Intel GPU architectures.
inline const llvm::ExtensionSet common_spirv_extensions{
    llvm::SPIRV::Extension::SPV_EXT_optnone,
    llvm::SPIRV::Extension::SPV_EXT_shader_atomic_float_add,
    llvm::SPIRV::Extension::SPV_EXT_shader_atomic_float_min_max,
    llvm::SPIRV::Extension::SPV_INTEL_2d_block_io,
    llvm::SPIRV::Extension::SPV_INTEL_bfloat16_arithmetic,
    llvm::SPIRV::Extension::SPV_INTEL_bfloat16_conversion,
    llvm::SPIRV::Extension::SPV_INTEL_bindless_images,
    llvm::SPIRV::Extension::SPV_INTEL_cache_controls,
    llvm::SPIRV::Extension::SPV_INTEL_fp_max_error,
    llvm::SPIRV::Extension::SPV_INTEL_function_pointers,
    llvm::SPIRV::Extension::SPV_INTEL_global_variable_host_access,
    llvm::SPIRV::Extension::SPV_INTEL_inline_assembly,
    llvm::SPIRV::Extension::SPV_INTEL_joint_matrix,
    llvm::SPIRV::Extension::SPV_INTEL_kernel_attributes,
    llvm::SPIRV::Extension::SPV_INTEL_long_composites,
    llvm::SPIRV::Extension::SPV_INTEL_memory_access_aliasing,
    llvm::SPIRV::Extension::SPV_INTEL_optnone,
    llvm::SPIRV::Extension::SPV_INTEL_predicated_io,
    llvm::SPIRV::Extension::SPV_INTEL_split_barrier,
    llvm::SPIRV::Extension::SPV_INTEL_subgroup_matrix_multiply_accumulate,
    llvm::SPIRV::Extension::SPV_INTEL_subgroups,
    llvm::SPIRV::Extension::SPV_INTEL_tensor_float32_conversion,
    llvm::SPIRV::Extension::SPV_INTEL_variable_length_array,
    llvm::SPIRV::Extension::SPV_KHR_bfloat16,
    llvm::SPIRV::Extension::SPV_KHR_cooperative_matrix,
    llvm::SPIRV::Extension::SPV_KHR_expect_assume,
    llvm::SPIRV::Extension::SPV_KHR_integer_dot_product,
    llvm::SPIRV::Extension::SPV_KHR_linkonce_odr,
    llvm::SPIRV::Extension::SPV_KHR_no_integer_wrap_decoration,
    llvm::SPIRV::Extension::SPV_KHR_non_semantic_info,
    llvm::SPIRV::Extension::SPV_KHR_shader_clock,
    llvm::SPIRV::Extension::SPV_KHR_uniform_group_instructions};

absl::StatusOr<std::string> CompileToSPIRV(
    llvm::Module* module, stream_executor::GpuComputeCapability gpu_version,
    const DebugOptions& debug_options);

std::vector<std::string> SPIRVExtensionsEnumToString(
    const llvm::ExtensionSet& enum_extensions);

// Returns the LLVM command line flags that we use for compilation.
std::vector<std::string> GetSPIRVBackendOptions(
    const DebugOptions& debug_options);

}  // namespace xla::gpu::spirv

#endif  // XLA_SERVICE_GPU_LLVM_GPU_BACKEND_SPIRV_BACKEND_H_
