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
#include <atomic>
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
#include "tensorflow/compiler/xla/service/gpu/fusion_merger.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_copy_insertion.h"
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
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/cuda_libdevice_path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/subprocess.h"
#include "tensorflow/core/platform/tracing.h"

namespace se = ::perftools::gputools;

namespace xla {
namespace gpu {

/* static */ const char* GpuCompiler::kTargetTriple = "nvptx64-nvidia-cuda";
/* static */ const char* GpuCompiler::kDataLayout =
    "e-i64:64-i128:128-v16:16-v32:32-n16:32:64";

namespace {

using tensorflow::port::Tracing;
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
    HloModule* hlo_module,
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
  pipeline.AddPass<FlattenCallGraph>();
  pipeline.AddPass<GpuCopyInsertion>();
  return pipeline.Run(hlo_module).status();
}

// Compiles the given PTX string using ptxas and returns the resulting machine
// code (i.e. a cubin) as a byte array.
StatusOr<std::vector<uint8>> CompilePtx(const string& ptx, int cc_major,
                                        int cc_minor) {
  Tracing::TraceMe annotation("Compile PTX", /*is_expensive=*/true);
  const string ptxas_path =
      tensorflow::io::JoinPath(tensorflow::CudaRoot(), "bin", "ptxas");
  VLOG(2) << "Using ptxas at " << ptxas_path;
  auto env = tensorflow::Env::Default();
  TF_RETURN_IF_ERROR(env->FileExists(ptxas_path));

  // Write ptx into a temporary file.
  string ptx_path;
  if (!env->LocalTempFilename(&ptx_path)) {
    return InternalError("couldn't get temp PTX file name");
  }
  auto ptx_cleaner = tensorflow::gtl::MakeCleanup([&ptx_path] {
    TF_CHECK_OK(tensorflow::Env::Default()->DeleteFile(ptx_path));
  });

  TF_RETURN_IF_ERROR(tensorflow::WriteStringToFile(env, ptx_path, ptx));
  VLOG(2) << "ptx written to: " << ptx_path;

  // Invoke ptxas and collect its output.
  string cubin_path;
  if (!env->LocalTempFilename(&cubin_path)) {
    return InternalError("couldn't get temp CUBIN file name");
  }
  auto cubin_cleaner = tensorflow::gtl::MakeCleanup([&cubin_path] {
    // CUBIN file may never be created, so the failure to delete it should not
    // produce TF error.
    tensorflow::Env::Default()->DeleteFile(cubin_path).IgnoreError();
  });
  tensorflow::SubProcess ptxas_info_dumper;
  std::vector<string> ptxas_args = {ptxas_path, ptx_path, "-o", cubin_path,
                                    StrCat("-arch=sm_", cc_major, cc_minor)};
  if (VLOG_IS_ON(2)) {
    ptxas_args.push_back("-v");
  }
  ptxas_info_dumper.SetProgram(ptxas_path, ptxas_args);
  ptxas_info_dumper.SetChannelAction(tensorflow::CHAN_STDERR,
                                     tensorflow::ACTION_PIPE);
  if (!ptxas_info_dumper.Start()) {
    return InternalError("Failed to launch ptxas");
  }
  string stderr_output;
  int exit_status = ptxas_info_dumper.Communicate(
      /*stdin_input=*/nullptr, /*stdout_output=*/nullptr, &stderr_output);
  XLA_LOG_LINES(tensorflow::INFO, stderr_output);
  if (exit_status != 0) {
    return InternalError("ptxas exited with non-zero error code %d",
                         exit_status);
  }

  // Read in the result of compilation and return it as a byte vector.
  string cubin;
  TF_RETURN_IF_ERROR(tensorflow::ReadFileToString(tensorflow::Env::Default(),
                                                  cubin_path, &cubin));
  std::vector<uint8> cubin_vector(cubin.begin(), cubin.end());
  return cubin_vector;
}

}  // namespace

GpuCompiler::GpuCompiler()
    : pointer_size_(llvm::DataLayout(kDataLayout).getPointerSize()) {}

StatusOr<std::unique_ptr<HloModule>> GpuCompiler::RunHloPasses(
    std::unique_ptr<HloModule> module, se::StreamExecutor* /*stream_exec*/) {
  XLA_SCOPED_LOGGING_TIMER("GpuCompiler::RunHloPasses");
  Tracing::TraceMe annotation("HLO Transforms", module->name(),
                              /*is_expensive=*/true);
  TF_RETURN_IF_ERROR(OptimizeHloModule(module.get(), ShapeSizeBytesFunction()));
  return std::move(module);
}

StatusOr<std::unique_ptr<Executable>> GpuCompiler::RunBackend(
    std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec) {
  XLA_SCOPED_LOGGING_TIMER("GpuCompiler::RunBackend");

  TF_RET_CHECK(stream_exec != nullptr);

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
  XLA_VLOG_LINES(2, module->ToString());
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
  {
    XLA_SCOPED_LOGGING_TIMER("GpuCompiler::RunBackend - IR emission");
    TF_RETURN_IF_ERROR(
        entry_computation->root_instruction()->Accept(&ir_emitter));
  }

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

  string libdevice_dir;
  {
    tensorflow::mutex_lock lock(mutex_);

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

  string ptx;
  {
    XLA_SCOPED_LOGGING_TIMER("GpuCompiler::RunBackend - CompileToPtx");
    TF_ASSIGN_OR_RETURN(ptx, CompileToPtx(&llvm_module, {cc_major, cc_minor},
                                          module->config(), libdevice_dir));
  }

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
  XLA_VLOG_LINES(2, ptx);

  // Write PTX to IR dump directory, if IR dumping was requested.
  if (!ir_dump_directory.empty()) {
    const string ptx_outfile = tensorflow::io::JoinPath(
        ir_dump_directory, StrCat(module->name(), ".ptx"));
    auto status = [&] {
      auto* env = tensorflow::Env::Default();
      TF_RETURN_IF_ERROR(env->RecursivelyCreateDir(ir_dump_directory));
      TF_RETURN_IF_ERROR(tensorflow::WriteStringToFile(env, ptx_outfile, ptx));
      return Status::OK();
    }();
    if (!status.ok()) {
      LOG(WARNING) << "Couldn't dump PTX for module " << module->name()
                   << " to " << ptx_outfile << ": " << status;
    }
  }

  const std::vector<uint8> cubin =
      CompilePtxOrGetCachedResult(ptx, cc_major, cc_minor);

  auto thunk_schedule = MakeUnique<ThunkSchedule>(
      ir_emitter.ConsumeThunkSequence(), std::move(stream_assignment),
      hlo_schedule->ThunkLaunchOrder());
  VLOG(2) << "Printing the thunk schedule...";
  XLA_VLOG_LINES(2, thunk_schedule->ToString());

  std::unique_ptr<HloProfileIndexMap> profile_index_map;
  std::unique_ptr<HloProfilePrinter> profile_printer;

  if (module->config().hlo_profiling_enabled()) {
    HloCostAnalysis cost_analysis(ShapeSizeBytesFunction());
    profile_index_map = MakeUnique<HloProfileIndexMap>(*module);
    profile_printer =
        CreateHloProfilePrinter(*profile_index_map, cost_analysis);
  }

  auto* gpu_executable = new GpuExecutable(
      ptx, cubin, {cc_major, cc_minor}, std::move(thunk_schedule),
      std::move(module), std::move(buffer_assignment),
      std::move(profile_printer), std::move(profile_index_map));
  if (embed_ir_in_executable) {
    DCHECK_NE("", ir_module_string_before_opt);
    gpu_executable->set_ir_module_string(ir_module_string_before_opt);
  }
  return std::unique_ptr<Executable>(gpu_executable);
}

std::vector<uint8> GpuCompiler::CompilePtxOrGetCachedResult(const string& ptx,
                                                            int cc_major,
                                                            int cc_minor) {
  XLA_SCOPED_LOGGING_TIMER("GpuCompiler::CompilePtxOrGetCachedResult");
  Tracing::TraceMe annotation("PTX->CUBIN", /*is_expensive=*/true);
  bool inserted;
  decltype(compilation_cache_.begin()) iter;
  // Pointers into compilation_cache_ where the ptx and (optional) cubin are
  // stored.
  const string* cache_ptx = nullptr;
  CompilationCacheValue* cache_value = nullptr;

  {
    tensorflow::mutex_lock lock(mutex_);
    std::tie(iter, inserted) = compilation_cache_.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(ptx, cc_major, cc_minor),
        std::forward_as_tuple());
    cache_ptx = &iter->first.ptx;
    cache_value = &iter->second;
  }

  // Compile the ptx if it wasn't in the cache before we called this function.
  // Other threads asking for the same compilation key will block on
  // cache_value->mutex_ until compilation is done.
  {
    tensorflow::mutex_lock lock(cache_value->mutex_);
    if (inserted) {
      CHECK(!cache_value->compilation_done);
      if (!ptx.empty()) {
        StatusOr<std::vector<uint8>> maybe_cubin =
            CompilePtx(*cache_ptx, cc_major, cc_minor);
        if (maybe_cubin.ok()) {
          cache_value->cubin_data = std::move(maybe_cubin).ValueOrDie();
          VLOG(2) << "Compiled PTX size:" << ptx.size()
                  << " CUBIN size: " << cache_value->cubin_data.size();
        } else {
          bool log_warning = true;
          if (maybe_cubin.status().code() ==
              tensorflow::error::Code::NOT_FOUND) {
            // Missing ptxas is expected in some environments where CUDA SDK
            // binaries are not available. We don't want to spam logs with
            // identical warnings in this case.

            // TODO(zhengxq): we should implement a LOG_FIRST_N and LOG_EVERY_N
            // for more general usage.
            static std::atomic<bool> warning_done(false);
            log_warning = !warning_done.exchange(true);
          }
          if (log_warning) {
            LOG(WARNING)
                << "Failed to compile ptx to cubin.  Will attempt to let "
                   "GPU driver compile the ptx. "
                << maybe_cubin.status();
          }
        }
      }
      cache_value->compilation_done = true;
      cache_value->compilation_done_cv_.notify_all();
    } else {
      while (!cache_value->compilation_done) {
        cache_value->compilation_done_cv_.wait(lock);
      }
    }
  }

  CHECK(cache_value != nullptr);
  CHECK(cache_value->compilation_done);
  return cache_value->cubin_data;
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
