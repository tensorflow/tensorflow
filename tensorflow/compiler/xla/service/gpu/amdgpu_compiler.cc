/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/amdgpu_compiler.h"

#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_conv_algorithm_picker.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_conv_padding_legalization.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_conv_rewriter.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_layout_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/gemm_rewriter.h"
#include "tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.h"
#include "tensorflow/compiler/xla/service/gpu/target_constants.h"
#include "tensorflow/compiler/xla/service/hlo_constant_folding.h"
#include "tensorflow/compiler/xla/service/hlo_cse.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/service/tuple_simplifier.h"
#include "tensorflow/core/platform/rocm_rocdl_path.h"

namespace xla {
namespace gpu {

namespace {

// Returns the directory containing ROCm-Device-Libs files. This function is
// called in AMDGPUCompiler's constructor, so can't return an error. But
// AMDGPUCompiler::Compile will return an error when the wanted rocdl file
// doesn't exist in the folder this function returns.
string GetROCDLDir(const HloModuleConfig& config) {
  std::vector<string> potential_rocdl_dirs;
  const string datadir = config.debug_options().xla_gpu_cuda_data_dir();
  if (!datadir.empty()) {
    potential_rocdl_dirs.push_back(datadir);
  }
  potential_rocdl_dirs.push_back(tensorflow::RocdlRoot());

  // Tries all potential ROCDL directories in the order they are inserted.
  // Returns the first directory that exists in the file system.
  for (const string& potential_rocdl_dir : potential_rocdl_dirs) {
    if (tensorflow::Env::Default()->IsDirectory(potential_rocdl_dir).ok()) {
      VLOG(2) << "Found ROCm-Device-Libs dir " << potential_rocdl_dir;
      return potential_rocdl_dir;
    }
    VLOG(2) << "Unable to find potential ROCm-Device-Libs dir "
            << potential_rocdl_dir;
  }

  // Last resort: maybe in the current folder.
  return ".";
}

}  // namespace

Status AMDGPUCompiler::OptimizeHloConvolutionCanonicalization(
    HloModule* hlo_module, se::StreamExecutor* stream_exec,
    se::DeviceMemoryAllocator* device_allocator) {
  // Convert convolutions into CustomCalls to MIOpen, then canonicalize them
  // (PadInsertion).
  HloPassPipeline pipeline("conv_canonicalization");
  pipeline.AddInvariantChecker<HloVerifier>(/*layout_sensitive=*/false,
                                            /*allow_mixed_precision=*/false);
  pipeline.AddPass<GpuConvRewriter>();
  pipeline.AddPass<GpuConvPaddingLegalization>();

  pipeline.AddPass<HloConstantFolding>();
  TF_RETURN_IF_ERROR(pipeline.Run(hlo_module).status());

  return Status::OK();
}

Status AMDGPUCompiler::OptimizeHloPostLayoutAssignment(
    HloModule* hlo_module, se::StreamExecutor* stream_exec,
    se::DeviceMemoryAllocator* device_allocator) {
  HloPassPipeline pipeline("post-layout_assignment");
  pipeline.AddInvariantChecker<HloVerifier>(
      /*layout_sensitive=*/true,
      /*allow_mixed_precision=*/false,
      LayoutAssignment::InstructionCanChangeLayout);

  // The LayoutAssignment pass may leave behind kCopy instructions which are
  // duplicate or NOPs, so remove them with algebraic simplification and CSE.
  AlgebraicSimplifierOptions options;
  options.set_is_layout_sensitive(true);
  pipeline.AddPass<HloPassFix<AlgebraicSimplifier>>(options);

  // Rewrite GEMMs into custom calls.
  pipeline.AddPass<GemmRewriter>();

  pipeline.AddPass<GpuConvAlgorithmPicker>(stream_exec, device_allocator);

  // Clean up new_tuple described above.
  pipeline.AddPass<TupleSimplifier>();

  pipeline.AddPass<HloCSE>(/*is_layout_sensitive=*/true);
  TF_RETURN_IF_ERROR(pipeline.Run(hlo_module).status());

  return Status::OK();
}

AMDGPUCompiler::AMDGPUCompiler()
    : GpuCompiler(stream_executor::rocm::kROCmPlatformId, amdgpu::kTargetTriple,
                  amdgpu::kDataLayout) {}

GpuVersion AMDGPUCompiler::GetGpuVersion(se::StreamExecutor* stream_exec) {
  int isa_version = 0;
  if (!stream_exec->GetDeviceDescription().rocm_amdgpu_isa_version(
          &isa_version)) {
    LOG(WARNING)
        << "Couldn't get AMDGPU ISA version for device; assuming gfx803.";
    isa_version = 803;
  }

  return isa_version;
}

StatusOr<std::pair<std::string, std::vector<uint8>>>
AMDGPUCompiler::CompileTargetBinary(const HloModule* module,
                                    llvm::Module* llvm_module,
                                    GpuVersion gpu_version,
                                    se::StreamExecutor* stream_exec) {
  if (rocdl_dir_.empty()) {
    // Compute rocdl_dir_ just once and cache it in this member.
    rocdl_dir_ = GetROCDLDir(module->config());
  }

  std::vector<uint8> hsaco;
  {
    XLA_SCOPED_LOGGING_TIMER(
        "AMDGPUCompiler::CompileTargetBinary - CompileToHsaco");
    TF_ASSIGN_OR_RETURN(hsaco,
                        amdgpu::CompileToHsaco(llvm_module, gpu_version,
                                               module->config(), rocdl_dir_));
  }

  llvm_ir::DumpIrIfEnabled(*module, *llvm_module, /*optimized=*/false);

  if (user_post_optimization_hook_) {
    user_post_optimization_hook_(*llvm_module);
  }

  return std::pair<std::string, std::vector<uint8>>("", std::move(hsaco));
}

}  // namespace gpu
}  // namespace xla
