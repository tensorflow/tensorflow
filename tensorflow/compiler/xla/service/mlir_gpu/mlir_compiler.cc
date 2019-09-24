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

#include "tensorflow/compiler/xla/service/mlir_gpu/mlir_compiler.h"

#include <memory>

#include "mlir/Dialect/GPU/GPUDialect.h"  // TF:local_config_mlir
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // TF:local_config_mlir
#include "mlir/IR/Function.h"  // TF:local_config_mlir
#include "mlir/IR/Location.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/dump.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_constants.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_hlo_schedule.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/nvptx_compiler.h"
#include "tensorflow/compiler/xla/service/gpu/stream_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/target_constants.h"
#include "tensorflow/compiler/xla/service/gpu/thunk_schedule.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/mlir_gpu/failover_compiler.h"
#include "tensorflow/compiler/xla/service/mlir_gpu/kernel_lowering.h"
#include "tensorflow/compiler/xla/service/mlir_gpu/lhlo_dialect_emitter.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace mlir_gpu {
namespace {

using ::mlir::MLIRContext;
using ::mlir::ModuleOp;
using ::mlir::OwningModuleRef;
using ::mlir::UnknownLoc;
using ::mlir::LLVM::LLVMDialect;
using ::xla::gpu::GpuExecutable;
using ::xla::gpu::GpuHloSchedule;
using ::xla::gpu::GpuVersion;
using ::xla::gpu::StreamAssignment;
using ::xla::gpu::ThunkSchedule;

int64 ConfigureLLVMModuleAndGetPointerSize(MLIRContext* context) {
  LLVMDialect* dialect = context->getRegisteredDialect<LLVMDialect>();
  llvm::Module& module = dialect->getLLVMModule();
  module.setTargetTriple(gpu::nvptx::kTargetTriple);
  module.setDataLayout(gpu::nvptx::kDataLayout);
  return module.getDataLayout().getPointerSize();
}

}  // namespace

MlirCompiler::MlirCompiler()
    : pointer_size_(ConfigureLLVMModuleAndGetPointerSize(&context_)) {}

se::Platform::Id MlirCompiler::PlatformId() const {
  return stream_executor::cuda::kCudaPlatformId;
}

StatusOr<std::unique_ptr<HloModule>> MlirCompiler::RunHloPasses(
    std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
    se::DeviceMemoryAllocator* device_allocator) {
  // Until we find a reason to do something different, run the same passes
  // that the normal GPU backend runs.
  gpu::NVPTXCompiler xla_compiler;
  TF_RETURN_IF_ERROR(xla_compiler.OptimizeHloModule(module.get(), stream_exec,
                                                    device_allocator));
  TF_RETURN_IF_ERROR(xla_compiler.PrepareHloModuleForIrEmitting(module.get()));

  return std::move(module);
}

namespace {

// TODO(b/137624192): Move this to custom call handling and share.
absl::optional<bool> CanShareBufferHint(const HloInstruction* user,
                                        const HloInstruction* operand,
                                        const ShapeIndex& user_index) {
  if (user->opcode() == HloOpcode::kCustomCall) {
    // Share the bias buffer with the parent instruction.
    if (user->custom_call_target() == xla::gpu::kGemmCallTarget) {
      if (user->operand_count() == 3 && user->operand(2) == operand) {
        return true;
      }
    }
    // The operand of cholesky can be shared with the first output.
    if (user->custom_call_target() == xla::gpu::kCusolverCholeskyCallTarget) {
      return user_index.size() == 1 && user_index[0] == 0;
    }
  }
  return absl::nullopt;
}

// TODO(b/137624192): Share this with nvptx backend.
GpuVersion GetGpuVersion(se::StreamExecutor* stream_exec) {
  int cc_major, cc_minor;
  const auto& device_description = stream_exec->GetDeviceDescription();
  if (!device_description.cuda_compute_capability(&cc_major, &cc_minor)) {
    LOG(WARNING)
        << "Couldn't get compute capability for device; assuming sm_20.";
    cc_major = 2;
    cc_minor = 0;
  }
  return std::make_pair(cc_major, cc_minor);
}

}  //  namespace

StatusOr<std::unique_ptr<Executable>> MlirCompiler::RunBackend(
    std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
    se::DeviceMemoryAllocator* device_allocator) {
  // Determine the HLO schedule, which is an ordering of HLO instructions. This
  // is used by buffer assignment to enable buffer reuse, and the same ordering
  // must also be used to determine the thunk launch schedule.
  std::unique_ptr<StreamAssignment> stream_assignment =
      xla::gpu::AssignStreams(*module);
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<GpuHloSchedule> hlo_schedule,
      GpuHloSchedule::Build(*module, *stream_assignment, pointer_size_));

  // Run buffer analysis on the HLO graph. This analysis figures out which
  // temporary buffers are required to run the computation.
  TF_ASSIGN_OR_RETURN(std::unique_ptr<BufferAssignment> buffer_assignment,
                      BufferAssigner::Run(
                          module.get(), hlo_schedule->ConsumeHloOrdering(),
                          BufferSizeBytesFunction(),
                          /*color_alignment=*/
                          [](LogicalBuffer::Color) {
                            return xla::gpu::kXlaAllocatedBufferAlignBytes;
                          },
                          /*allocate_buffers_for_constants=*/true,
                          /*colorer=*/BufferAssigner::DefaultColorer(),
                          /*must_not_live_out=*/{}, &CanShareBufferHint));
  DumpHloModuleIfEnabled(*module, *buffer_assignment, "after_optimizations");

  MLIRContext mlir_context;
  OwningModuleRef mlir_module =
      ModuleOp::create(UnknownLoc::get(&mlir_context));
  LhloDialectEmitter lhlo_emitter(*module, *buffer_assignment,
                                  stream_exec->platform(), *mlir_module);

  TF_RETURN_IF_ERROR(
      lhlo_emitter.EmitComputation(*module->entry_computation()));

  if (module_hook_.callback &&
      module_hook_.stage == IRHook::LoweringStage::LHLO) {
    module_hook_.callback(*mlir_module);
  }

  TF_RETURN_IF_ERROR(LowerLHLOToGPU(*mlir_module));

  if (module_hook_.callback &&
      module_hook_.stage == IRHook::LoweringStage::GPU) {
    module_hook_.callback(*mlir_module);
  }

  TF_RETURN_IF_ERROR(LowerKernelBodiesToNVVM(*mlir_module));

  if (module_hook_.callback &&
      module_hook_.stage == IRHook::LoweringStage::LLVM) {
    module_hook_.callback(*mlir_module);
  }

  // TODO(b/137624192): Emit function per hlo and turn into ptx string and blob.
  std::string ptx;
  std::vector<uint8> cubin;

  auto thunk_schedule = absl::make_unique<ThunkSchedule>(
      lhlo_emitter.ConsumeThunkSequence(), std::move(stream_assignment),
      hlo_schedule->ThunkLaunchOrder());

  if (DumpingEnabledForHloModule(*module)) {
    DumpToFileInDirOrStdout(*module, "thunk_schedule",
                            thunk_schedule->ToString());
  }

  // TODO(b/137624192): Add profiling support.
  return {absl::make_unique<GpuExecutable>(
      ptx, cubin, GetGpuVersion(stream_exec), std::move(thunk_schedule),
      std::move(module), std::move(buffer_assignment), nullptr, nullptr)};
}

StatusOr<std::vector<std::unique_ptr<Executable>>> MlirCompiler::Compile(
    std::unique_ptr<HloModuleGroup> module_group,
    std::vector<std::vector<se::StreamExecutor*>> stream_execs,
    se::DeviceMemoryAllocator* device_allocator) {
  return Unimplemented("Not yet implemented in MLIR compiler");
}

StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
MlirCompiler::CompileAheadOfTime(std::unique_ptr<HloModuleGroup> module_group,
                                 const AotCompilationOptions& options) {
  return Unimplemented("Not yet implemented in MLIR compiler");
}

void MlirCompiler::SetModuleHook(IRHook module_hook) {
  module_hook_ = module_hook;
}

void MlirCompiler::RemoveModuleHook() {
  module_hook_ = {nullptr, IRHook::LoweringStage::LHLO};
}

}  // namespace mlir_gpu
}  // namespace xla

static bool InitModule() {
  xla::Compiler::RegisterCompilerFactory(
      stream_executor::cuda::kCudaPlatformId, []() {
        return absl::make_unique<xla::FailoverCompiler>(
            absl::make_unique<xla::mlir_gpu::MlirCompiler>(),
            absl::make_unique<xla::gpu::NVPTXCompiler>());
      });
  return true;
}
static bool module_initialized = InitModule();
