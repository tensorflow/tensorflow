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

<<<<<<< HEAD
#include <stdlib.h>

#include <atomic>
#include <functional>
#include <mutex>  // NOLINT(build/c++11): only using std::call_once, not mutex.
#include <utility>

#include "absl/memory/memory.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "tensorflow/compiler/xla/protobuf_util.h"
#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
#include "tensorflow/compiler/xla/service/batchnorm_expander.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/call_inliner.h"
#include "tensorflow/compiler/xla/service/conditional_simplifier.h"
#include "tensorflow/compiler/xla/service/convolution_group_converter.h"
#include "tensorflow/compiler/xla/service/dot_decomposer.h"
#include "tensorflow/compiler/xla/service/dump.h"
#include "tensorflow/compiler/xla/service/dynamic_index_splitter.h"
#include "tensorflow/compiler/xla/service/flatten_call_graph.h"
#include "tensorflow/compiler/xla/service/gpu/cudnn_batchnorm_rewriter.h"
#include "tensorflow/compiler/xla/service/gpu/cudnn_conv_padding_legalization.h"
#include "tensorflow/compiler/xla/service/gpu/cudnn_conv_rewriter.h"
#include "tensorflow/compiler/xla/service/gpu/fusion_merger.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_constants.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_conv_algorithm_picker.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_copy_insertion.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_hlo_schedule.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_hlo_support_checker.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_layout_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_sanitize_constant_names.h"
#include "tensorflow/compiler/xla/service/gpu/instruction_fusion.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emitter_context.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emitter_unnested.h"
#include "tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.h"
#include "tensorflow/compiler/xla/service/gpu/multi_output_fusion.h"
#include "tensorflow/compiler/xla/service/gpu/partition_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/stream_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
#include "tensorflow/compiler/xla/service/gpu/target_constants.h"
#include "tensorflow/compiler/xla/service/gpu/thunk_schedule.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_constant_folding.h"
#include "tensorflow/compiler/xla/service/hlo_cse.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_element_type_converter.h"
#include "tensorflow/compiler/xla/service/hlo_get_dimension_size_rewriter.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/hlo_proto_util.h"
#include "tensorflow/compiler/xla/service/hlo_subcomputation_unification.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/service/reduce_precision_insertion.h"
#include "tensorflow/compiler/xla/service/reshape_mover.h"
#include "tensorflow/compiler/xla/service/rng_expander.h"
#include "tensorflow/compiler/xla/service/sort_simplifier.h"
#include "tensorflow/compiler/xla/service/stable_sort_expander.h"
#include "tensorflow/compiler/xla/service/transpose_folding.h"
#include "tensorflow/compiler/xla/service/tuple_simplifier.h"
#include "tensorflow/compiler/xla/service/while_loop_constant_sinking.h"
#include "tensorflow/compiler/xla/service/while_loop_simplifier.h"
#include "tensorflow/compiler/xla/service/while_loop_trip_count_annotator.h"
#include "tensorflow/compiler/xla/service/zero_sized_hlo_elimination.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/regexp.h"
#include "tensorflow/core/platform/rocm_rocdl_path.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/subprocess.h"
#include "tensorflow/core/profiler/lib/traceme.h"
=======
#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
#include "tensorflow/compiler/xla/service/gpu/cudnn_conv_padding_legalization.h"
#include "tensorflow/compiler/xla/service/gpu/cudnn_conv_rewriter.h"
// TODO(whchung@gmail.com): Add gpu_conv_algorithm_picker after its PR merged.
#include "tensorflow/compiler/xla/service/gpu/gpu_layout_assignment.h"
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
>>>>>>> google_upstream/master

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
  pipeline.AddPass<CudnnConvRewriter>();
  pipeline.AddPass<CudnnConvPaddingLegalization>();

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
<<<<<<< HEAD
  pipeline.AddPass<GpuConvAlgorithmPicker>(stream_exec, device_allocator);
=======

  // TODO(whchung@gmail.com): Add gpu_conv_algorithm_picker after its PR merged.

>>>>>>> google_upstream/master
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
<<<<<<< HEAD
  if (!stream_exec->GetDeviceDescription().
                    rocm_amdgpu_isa_version(&isa_version)) {
=======
  if (!stream_exec->GetDeviceDescription().rocm_amdgpu_isa_version(
          &isa_version)) {
>>>>>>> google_upstream/master
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
