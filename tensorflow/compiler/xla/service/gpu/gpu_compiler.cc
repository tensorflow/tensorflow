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

#include "tensorflow/compiler/xla/service/gpu/gpu_compiler.h"

#include <stdlib.h>
#include <functional>
#include <utility>

#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "tensorflow/compiler/xla/protobuf_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
#include "tensorflow/compiler/xla/service/batchnorm_rewriter.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/buffer_liveness.h"
#include "tensorflow/compiler/xla/service/call_inliner.h"
#include "tensorflow/compiler/xla/service/flatten_call_graph.h"
#include "tensorflow/compiler/xla/service/gpu/convolution_folding.h"
#include "tensorflow/compiler/xla/service/gpu/copy_insertion.h"
#include "tensorflow/compiler/xla/service/gpu/fusion_merger.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable.h"
#include "tensorflow/compiler/xla/service/gpu/hlo_schedule.h"
#include "tensorflow/compiler/xla/service/gpu/instruction_fusion.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emitter.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emitter_context.h"
#include "tensorflow/compiler/xla/service/gpu/layout_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.h"
#include "tensorflow/compiler/xla/service/gpu/pad_insertion.h"
#include "tensorflow/compiler/xla/service/gpu/partition_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/stream_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/thunk_schedule.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_constant_folding.h"
#include "tensorflow/compiler/xla/service/hlo_cse.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/hlo_proto_util.h"
#include "tensorflow/compiler/xla/service/hlo_subcomputation_unification.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/service/reduce_precision_insertion.h"
#include "tensorflow/compiler/xla/service/reshape_mover.h"
#include "tensorflow/compiler/xla/service/transpose_folding.h"
#include "tensorflow/compiler/xla/service/tuple_simplifier.h"
#include "tensorflow/compiler/xla/service/while_loop_simplifier.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/cuda_libdevice_path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/subprocess.h"

namespace se = ::perftools::gputools;

namespace xla {
namespace gpu {

/* static */ const char* GpuCompiler::kTargetTriple = "nvptx64-nvidia-cuda";
/* static */ const char* GpuCompiler::kDataLayout =
    "e-i64:64-i128:128-v16:16-v32:32-n16:32:64";

namespace {

using tensorflow::strings::StrCat;

// Any address of a variable residing in global memory or returned by one of the
// memory allocation routines from the driver or runtime API is always aligned
// to at least 256 bytes.
//
// http://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory-accesses
constexpr int64 kMemoryAlignment = 256;

// Returns the directory containing nvvm libdevice files.  config_cuda_data_dir
// should be equal to config().debug_options().xla_gpu_cuda_data_dir() of the
// HloModule being compiled.
string GetLibdeviceDir(const string& config_cuda_data_dir) {
  std::vector<string> potential_libdevice_dirs;
  if (!config_cuda_data_dir.empty()) {
    potential_libdevice_dirs.push_back(config_cuda_data_dir);
  }
  potential_libdevice_dirs.push_back(tensorflow::LibdeviceRoot());

  // Tries all potential libdevice directories in the order they are inserted.
  // Returns the first directory that exists in the file system.
  for (const string& potential_libdevice_dir : potential_libdevice_dirs) {
    if (tensorflow::Env::Default()->IsDirectory(potential_libdevice_dir).ok()) {
      VLOG(2) << "Found libdevice dir " << potential_libdevice_dir;
      return potential_libdevice_dir;
    }
    VLOG(2) << "Unable to find potential libdevice dir "
            << potential_libdevice_dir;
  }

  // Last resort: maybe in the current folder.
  return ".";
}

// Runs optimization passes on the given HLO module.
tensorflow::Status OptimizeHloModule(
    HloModule* hlo_module, const se::DeviceDescription& device_desc,
    const HloCostAnalysis::ShapeSizeFunction& shape_size_function) {
  {
    HloPassPipeline pipeline("optimization");
    pipeline.AddInvariantChecker<HloVerifier>(shape_size_function);
    ReducePrecisionInsertion::AddPasses(
        &pipeline, hlo_module->config().debug_options(),
        ReducePrecisionInsertion::PassTiming::BEFORE_OPTIMIZATION);

    // TODO(b/64094172): make Call work on GPU instead of inlining.
    pipeline.AddPass<CallInliner>();

    {
      auto& pass =
          pipeline.AddPass<HloPassFix<HloPassPipeline>>("simplification");
      pass.AddInvariantChecker<HloVerifier>(shape_size_function);

      // TODO(b/62764704): Do not rewrite on GPU, use cuDNN's BatchNorm APIs
      // instead.
      pass.AddPass<BatchNormRewriter>(
          /*rewrite_training_op=*/true,
          /*rewrite_inference_op=*/true,
          /*rewrite_grad_op=*/true,
          /*use_fusion=*/false);
      pass.AddPass<AlgebraicSimplifier>(
          /*is_layout_sensitive=*/false,
          [](const Shape&, const Shape&) { return false; });
      pass.AddPass<TupleSimplifier>();
      pass.AddPass<WhileLoopSimplifier>();
      pass.AddPass<HloDCE>();
      pass.AddPass<ReshapeMover>();
      pass.AddPass<HloConstantFolding>();
    }
    pipeline.AddPass<ConvolutionFolding>();
    pipeline.AddPass<TransposeFolding>(
        [](const HloInstruction& dot,
           const TransposeFolding::OperandIndices& candidate_operands) {
          return ImplementedAsGemm(dot) ? candidate_operands
                                        : TransposeFolding::OperandIndices{};
        },
        TransposeFolding::NeverFoldTranspose);
    pipeline.AddPass<HloCSE>(/*is_layout_sensitive=*/false);
    pipeline.AddPass<HloDCE>();
    TF_RETURN_IF_ERROR(pipeline.Run(hlo_module).status());
  }
  {
    HloPassFix<HloPassPipeline> fusion("fusion");
    fusion.AddInvariantChecker<HloVerifier>(shape_size_function);
    fusion.AddPass<GpuInstructionFusion>(/*may_duplicate=*/false);
    fusion.AddPass<GpuInstructionFusion>(/*may_duplicate=*/true);
    fusion.AddPass<FusionMerger>();
    TF_RETURN_IF_ERROR(fusion.Run(hlo_module).status());

    HloPassPipeline reduce_pipeline("reduce-precision");
    reduce_pipeline.AddInvariantChecker<HloVerifier>(shape_size_function);
    ReducePrecisionInsertion::AddPasses(
        &reduce_pipeline, hlo_module->config().debug_options(),
        ReducePrecisionInsertion::PassTiming::AFTER_FUSION);
    StatusOr<bool> reduce_result = reduce_pipeline.Run(hlo_module);
    TF_RETURN_IF_ERROR(reduce_result.status());

    if (reduce_result.ValueOrDie()) {
      // Do another fusion pass, with the expectation that we may be able to
      // fuse the new ReducePrecision operations.
      TF_RETURN_IF_ERROR(fusion.Run(hlo_module).status());
    }
  }
  return tensorflow::Status::OK();
}

// Modifies the given HLO module so that it will be accepted by IrEmitter.
// Unlike optimization passes, the passes are necessary for correctness.
tensorflow::Status PrepareHloModuleForIrEmitting(
    HloModule* hlo_module,
    const HloCostAnalysis::ShapeSizeFunction& shape_size_function) {
  // In some cases, we have to place the result of an instruction in a temporary
  // buffer. For instance, the buffer that holds an external parameter is
  // assumed immutable at this point, and should not be reused for output
  // (b/27180329). Therefore, in that case, we set the output to be a copy of
  // the parameter.
  HloPassPipeline pipeline("GPU-ir-emit-prepare");
  pipeline.AddInvariantChecker<HloVerifier>(shape_size_function);
  pipeline.AddPass<PadInsertion>();
  pipeline.AddPass<GpuLayoutAssignment>(
      hlo_module->mutable_entry_computation_layout());
  // The LayoutAssignment pass may leave behind kCopy instructions which are
  // duplicate or NOPs, so remove them with algebraic simplification and CSE.
  pipeline.AddPass<HloPassFix<AlgebraicSimplifier>>(
      /*is_layout_sensitive=*/true,
      [](const Shape&, const Shape&) { return true; });
  pipeline.AddPass<HloCSE>(/*is_layout_sensitive=*/true);
  // Copy insertion should be performed immediately before IR emission to avoid
  // inserting unnecessary copies (later pass adds an instruction which
  // materializes the value) or missing a necessary copy (later pass removes an
  // instruction which materializes a value). DCE must be run immediately before
  // (and sometime after) copy insertion, to avoid dead code from interfering
  // with the rewrites.
  pipeline.AddPass<HloDCE>();
  pipeline.AddPass<GpuCopyInsertion>();
  pipeline.AddPass<HloDCE>();
  pipeline.AddPass<FlattenCallGraph>();
  return pipeline.Run(hlo_module).status();
}

// Invokes the ptxas tool on the given PTX string, and dumps its output.
void DumpPtxasInfo(const string& ptx, int cc_major, int cc_minor) {
  const string ptxas_path =
      tensorflow::io::JoinPath(tensorflow::CudaRoot(), "bin/ptxas");
  // Do not log PTX stats if ptxas is not found at the given path.
  if (!tensorflow::Env::Default()->FileExists(ptxas_path).ok()) {
    LOG(WARNING)
        << "Failed to dump PTX stats because ptxas is not found at path \""
        << ptxas_path << "\".";
    return;
  }

  // Write `ptx` into a temporary file.
  char tempdir_template[] = "/tmp/ptxXXXXXX";
  char* tempdir_name = mkdtemp(tempdir_template);
  CHECK_NOTNULL(tempdir_name);
  string ptx_path = tensorflow::io::JoinPath(tempdir_name, "ptx");
  TF_CHECK_OK(
      tensorflow::WriteStringToFile(tensorflow::Env::Default(), ptx_path, ptx));
  LOG(INFO) << "ptx file written to: " << ptx_path;

  // Invoke ptxas and collect its output.
  tensorflow::SubProcess ptxas_info_dumper;
  ptxas_info_dumper.SetProgram(ptxas_path,
                               {ptxas_path, ptx_path, "-o", "/dev/null", "-v",
                                StrCat("-arch=sm_", cc_major, cc_minor)});
  ptxas_info_dumper.SetChannelAction(tensorflow::CHAN_STDERR,
                                     tensorflow::ACTION_PIPE);
  if (!ptxas_info_dumper.Start()) {
    LOG(ERROR) << "Failed to launch ptxas.";
    return;
  }
  string stderr_output;
  int exit_status = ptxas_info_dumper.Communicate(
      /*stdin_input=*/nullptr, /*stdout_output=*/nullptr, &stderr_output);
  XLA_LOG_LINES(tensorflow::INFO, stderr_output);
  if (exit_status != 0) {
    LOG(ERROR) << "ptxas exited with non-zero error code " << exit_status
               << ".";
  }
}

}  // namespace

GpuCompiler::GpuCompiler()
    : pointer_size_(llvm::DataLayout(kDataLayout).getPointerSize()) {}

StatusOr<std::unique_ptr<Executable>> GpuCompiler::Compile(
    std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec) {
  TF_RET_CHECK(stream_exec != nullptr);

  TF_RETURN_IF_ERROR(OptimizeHloModule(module.get(),
                                       stream_exec->GetDeviceDescription(),
                                       ShapeSizeBytesFunction()));
  TF_RETURN_IF_ERROR(
      PrepareHloModuleForIrEmitting(module.get(), ShapeSizeBytesFunction()));

  llvm::LLVMContext llvm_context;
  std::string buffer;
  llvm::raw_string_ostream error(buffer);
  llvm::DiagnosticPrinterRawOStream printer(error);
  auto DiagnosticHandler = [](const llvm::DiagnosticInfo& diag_info,
                              void* Context) {
    auto printer = static_cast<llvm::DiagnosticPrinterRawOStream*>(Context);
    diag_info.print(*printer);
  };
  llvm_context.setDiagnosticHandlerCallBack(DiagnosticHandler, &printer);

  llvm::Module llvm_module(module->name().c_str(), llvm_context);
  // Set the target triple and the data layout.
  llvm_module.setTargetTriple(kTargetTriple);
  llvm_module.setDataLayout(kDataLayout);

  // Determine the HLO schedule, which is an ordering of HLO instructions.  This
  // is used by buffer assignment to enable buffer reuse, and the same ordering
  // must also be used to determine the thunk launch schedule.
  std::unique_ptr<StreamAssignment> stream_assignment = AssignStreams(*module);
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloSchedule> hlo_schedule,
      HloSchedule::Build(*module, *stream_assignment, pointer_size_));

  // Run buffer analysis on the HLO graph. This analysis figures out which
  // temporary buffers are required to run the computation.
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<BufferAssignment> buffer_assignment,
      BufferAssigner::Run(module.get(), hlo_schedule->ConsumeHloOrdering(),
                          BufferSizeBytesFunction(), [](LogicalBuffer::Color) {
                            return kMemoryAlignment;
                          }));
  // BufferAssignment::ToString() includes a header, so no need for us to
  // print one ourselves.
  XLA_VLOG_LINES(2, buffer_assignment->ToString());

  const string xla_dump_hlo_proto_to =
      module->config().debug_options().xla_dump_hlo_proto_to();
  if (!xla_dump_hlo_proto_to.empty()) {
    HloProto proto = MakeHloProto(*module, *buffer_assignment);
    TF_RETURN_IF_ERROR(protobuf_util::DumpProtoToDirectory(
        proto, xla_dump_hlo_proto_to, module->name()));
  }

  IrEmitterContext ir_emitter_context(module.get(), buffer_assignment.get(),
                                      &stream_exec->GetDeviceDescription(),
                                      &llvm_module);

  HloComputation* entry_computation = module->entry_computation();
  IrEmitterUnnested ir_emitter(module->config(), entry_computation,
                               &ir_emitter_context);
  TF_RETURN_IF_ERROR(
      entry_computation->root_instruction()->Accept(&ir_emitter));

  if (user_pre_optimization_hook_) {
    TF_CHECK_OK(user_pre_optimization_hook_(llvm_module));
  }
  string ir_module_string_before_opt;
  const bool embed_ir_in_executable =
      module->config().debug_options().xla_embed_ir_in_executable();
  if (VLOG_IS_ON(2) || embed_ir_in_executable) {
    ir_module_string_before_opt = llvm_ir::DumpModuleToString(llvm_module);
    VLOG(2) << "LLVM module before optimizations:";
    XLA_VLOG_LINES(2, ir_module_string_before_opt);
  }

  const string& ir_dump_directory =
      module->config().debug_options().xla_dump_ir_to();

  if (!ir_dump_directory.empty()) {
    TF_RETURN_IF_ERROR(llvm_ir::DumpIRToDirectory(
        /*directory_name=*/ir_dump_directory,
        /*hlo_module_name=*/module->name(), llvm_module,
        /*optimized=*/false));
  }

  string* ptx;
  string libdevice_dir;
  {
    tensorflow::mutex_lock lock(mutex_);

    // Reserve space for the PTX to be generated for this module.
    generated_ptxes_.emplace_back(MakeUnique<string>());
    ptx = generated_ptxes_.back().get();

    // Find the directory containing libdevice.  To avoid searching for it every
    // time, we have a one-element cache, keyed on the module's config's
    // cuda_data_dir.
    const auto& config_cuda_data_dir =
        module->config().debug_options().xla_gpu_cuda_data_dir();
    if (cached_libdevice_dir_.empty() ||
        cached_cuda_data_dir_ != config_cuda_data_dir) {
      cached_cuda_data_dir_ = config_cuda_data_dir;
      cached_libdevice_dir_ = GetLibdeviceDir(config_cuda_data_dir);
    }
    libdevice_dir = cached_libdevice_dir_;
  }
  int cc_major, cc_minor;
  if (!stream_exec->GetDeviceDescription().cuda_compute_capability(&cc_major,
                                                                   &cc_minor)) {
    LOG(WARNING)
        << "Couldn't get compute capability for device; assuming sm_20.";
    cc_major = 2;
    cc_minor = 0;
  }

  TF_ASSIGN_OR_RETURN(*ptx, CompileToPtx(&llvm_module, {cc_major, cc_minor},
                                         module->config(), libdevice_dir));

  if (!ir_dump_directory.empty()) {
    TF_RETURN_IF_ERROR(llvm_ir::DumpIRToDirectory(
        /*directory_name=*/ir_dump_directory,
        /*hlo_module_name=*/module->name(), llvm_module,
        /*optimized=*/true));
  }

  if (user_post_optimization_hook_) {
    TF_CHECK_OK(user_post_optimization_hook_(llvm_module));
  }
  VLOG(2) << "LLVM module after optimizations:";
  XLA_VLOG_LINES(2, llvm_ir::DumpModuleToString(llvm_module));
  VLOG(2) << "PTX:";
  XLA_VLOG_LINES(2, *ptx);
  if (VLOG_IS_ON(2)) {
    DumpPtxasInfo(*ptx, cc_major, cc_minor);
  }

  auto thunk_schedule = MakeUnique<ThunkSchedule>(
      ir_emitter.ConsumeThunkSequence(), std::move(stream_assignment),
      hlo_schedule->ThunkLaunchOrder());
  VLOG(2) << "Printing the thunk schedule...";
  XLA_VLOG_LINES(2, thunk_schedule->ToString());

  auto* gpu_executable =
      new GpuExecutable(*ptx, std::move(thunk_schedule), std::move(module),
                        std::move(buffer_assignment), ShapeSizeBytesFunction());
  if (embed_ir_in_executable) {
    DCHECK_NE("", ir_module_string_before_opt);
    gpu_executable->set_ir_module_string(ir_module_string_before_opt);
  }
  return std::unique_ptr<Executable>(gpu_executable);
}

StatusOr<std::vector<std::unique_ptr<Executable>>> GpuCompiler::Compile(
    std::vector<std::unique_ptr<HloModule>> modules,
    std::vector<std::vector<se::StreamExecutor*>> stream_execs) {
  return Unimplemented(
      "Compilation of multiple HLO modules is not yet supported on GPU.");
}

StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
GpuCompiler::CompileAheadOfTime(std::vector<std::unique_ptr<HloModule>> module,
                                const AotCompilationOptions& options) {
  return Unimplemented("not yet implemented: GpuCompiler::CompileAheadOfTime");
}

se::Platform::Id GpuCompiler::PlatformId() const {
  return se::cuda::kCudaPlatformId;
}

}  // namespace gpu
}  // namespace xla

static bool InitModule() {
  xla::Compiler::RegisterCompilerFactory(se::cuda::kCudaPlatformId, []() {
    return xla::MakeUnique<xla::gpu::GpuCompiler>();
  });
  return true;
}
static bool module_initialized = InitModule();
