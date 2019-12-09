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

#include "absl/container/flat_hash_map.h"
#include "mlir/Dialect/GPU/GPUDialect.h"  // TF:local_config_mlir
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // TF:local_config_mlir
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:local_config_mlir
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Function.h"  // TF:local_config_mlir
#include "mlir/IR/Location.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "mlir/IR/Value.h"  // TF:local_config_mlir
#include "mlir/Support/LLVM.h"  // TF:local_config_mlir
#include "mlir/Target/NVVMIR.h"  // TF:local_config_mlir
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/dump.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_constants.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_hlo_schedule.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_types.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/kernel_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.h"
#include "tensorflow/compiler/xla/service/gpu/nvptx_compiler.h"
#include "tensorflow/compiler/xla/service/gpu/partition_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/stream_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
#include "tensorflow/compiler/xla/service/gpu/target_constants.h"
#include "tensorflow/compiler/xla/service/gpu/thunk_schedule.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/mlir_gpu/emission_context.h"
#include "tensorflow/compiler/xla/service/mlir_gpu/failover_compiler.h"
#include "tensorflow/compiler/xla/service/mlir_gpu/kernel_lowering.h"
#include "tensorflow/compiler/xla/service/mlir_gpu/lhlo_dialect_emitter.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/cuda_libdevice_path.h"
#include "tensorflow/stream_executor/gpu/asm_compiler.h"

namespace xla {
namespace mlir_gpu {
namespace {

using ::mlir::BlockArgument;
using ::mlir::dyn_cast;
using ::mlir::FuncOp;
using ::mlir::MLIRContext;
using ::mlir::ModuleOp;
using ::mlir::OwningModuleRef;
using ::mlir::UnknownLoc;
using ::mlir::Value;
using ::mlir::gpu::LaunchFuncOp;
using ::mlir::LLVM::LLVMDialect;
using ::mlir::LLVM::LLVMFuncOp;
using ::mlir::LLVM::LLVMType;
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

// TODO(b/137624192) Share with NVPTX compiler
static std::vector<std::string> CandidateCudaRoots(
    const HloModuleConfig& config) {
  return tensorflow::CandidateCudaRoots(
      config.debug_options().xla_gpu_cuda_data_dir());
}

void PrintCantFindCudaMessage(absl::string_view msg,
                              const HloModuleConfig& hlo_module_config) {
  LOG(WARNING) << msg;
  LOG(WARNING) << "Searched for CUDA in the following directories:";

  for (const auto& dir : CandidateCudaRoots(hlo_module_config)) {
    LOG(WARNING) << "  " << dir;
  }
  LOG(WARNING)
      << "You can choose the search directory by setting xla_gpu_cuda_data_dir "
         "in HloModule's DebugOptions.  For most apps, setting the environment "
         "variable XLA_FLAGS=--xla_gpu_cuda_data_dir=/path/to/cuda will work.";
}

// Returns the directory containing nvvm libdevice files.
string GetLibdeviceDir(const HloModuleConfig& hlo_module_config) {
  for (const string& cuda_root : CandidateCudaRoots(hlo_module_config)) {
    const string libdevice_dir =
        tensorflow::io::JoinPath(cuda_root, "nvvm", "libdevice");
    VLOG(2) << "Looking for libdevice at " << libdevice_dir;
    if (tensorflow::Env::Default()->IsDirectory(libdevice_dir).ok()) {
      VLOG(2) << "Found libdevice dir " << libdevice_dir;
      return libdevice_dir;
    }
  }
  PrintCantFindCudaMessage(
      "Can't find libdevice directory ${CUDA_DIR}/nvvm/libdevice. This may "
      "result in compilation or runtime failures, if the program we try to run "
      "uses routines from libdevice.",
      hlo_module_config);

  // GetCudaRootCandidates always includes ".", but if everything fails, we
  // return it anyway.  Better than returning the empty string.
  return ".";
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

// Return the constant launch bound along the "x" dimension in "dim" if all the
// other dimensions are 1.  Return nullopt otherwise or when any of the bounds
// is not constant.
static absl::optional<int64> getLaunchBound(const mlir::gpu::KernelDim3& dim) {
  auto get_constant = [](mlir::Operation* op,
                         mlir::StringRef name) -> absl::optional<int64> {
    if (auto constant = llvm::dyn_cast_or_null<mlir::ConstantOp>(op)) {
      return constant.value().cast<mlir::IntegerAttr>().getInt();
    }
    op->emitError() << "bound " << name << " is not constant";
    return absl::nullopt;
  };
  auto y_op = dim.y->getDefiningOp();
  auto dim_y = get_constant(y_op, "y");
  if (!dim_y.has_value() || dim_y.value() != 1) {
    y_op->emitError() << "bound 'y' is not constant 1";
    return absl::nullopt;
  }
  auto z_op = dim.z->getDefiningOp();
  auto dim_z = get_constant(z_op, "z");
  if (!dim_z.has_value() || dim_z.value() != 1) {
    z_op->emitError() << "bound 'z' is not constant 1";
    return absl::nullopt;
  }
  return get_constant(dim.x->getDefiningOp(), "x");
}

using OperandToValueMap =
    absl::flat_hash_map<const HloInstruction*, std::vector<BlockArgument*>>;

static StatusOr<std::vector<const HloInstruction*>> ComputeOperandToValueMap(
    OperandToValueMap* operand_to_value_map, const HloInstruction* instr,
    LaunchFuncOp launchOp, LLVMFuncOp kernel) {
  auto operands = instr->operands();
  std::vector<const HloInstruction*> ordered_operands;
  bool has_failed = false;
  for (int kernel_index = 0; kernel_index < launchOp.getNumKernelOperands();
       ++kernel_index) {
    auto launchop_operand =
        dyn_cast<BlockArgument>(launchOp.getKernelOperand(kernel_index));
    if (!launchop_operand) {
      launchOp.emitError("argument to kernel is not a function input");
      has_failed = true;
      continue;
    }
    // host_index is the argument position to the surrounding function that
    // contains the launch. This index corresponds to HLO operand indices
    // by construction.
    auto host_index = launchop_operand->getArgNumber();
    // The trailing argument to the outer function are the results.
    auto operand =
        (host_index < operands.size()) ? operands[host_index] : instr;
    if (!operand_to_value_map->count(operand)) {
      ordered_operands.push_back(operand);
    }
    // Associate the HLO operand with the argument value of the kernel
    // function.
    (*operand_to_value_map)[operand].push_back(
        kernel.getArgument(kernel_index));
  }
  if (has_failed) {
    return InternalError("Mapping operands to kernel arguments has failed.");
  }
  return ordered_operands;
}

Status InsertBufferLoadPreduleIntoKernel(
    LLVMFuncOp kernel, const OperandToValueMap& operand_to_value_map,
    const std::vector<const HloInstruction*>& ordered_operands,
    BufferAssignment* assignment,
    const std::vector<const BufferAllocation*>& buffers) {
  mlir::OpBuilder builder(kernel.getBody());
  auto llvm_dialect = kernel.getContext()->getRegisteredDialect<LLVMDialect>();
  auto offset_type = LLVMType::getInt64Ty(llvm_dialect);
  auto ptr_type = LLVMType::getInt8PtrTy(llvm_dialect);
  auto void_type = LLVMType::getVoidTy(llvm_dialect);
  auto loc = kernel.getLoc();

  auto num_original_args = kernel.getNumArguments();
  std::vector<LLVMType> new_arg_types(buffers.size(), ptr_type);
  kernel.setAttr(kernel.getTypeAttrName(),
                 mlir::TypeAttr::get(LLVMType::getFunctionTy(
                     void_type, new_arg_types, /*isVarArg=*/false)));

  std::vector<mlir::Type> as_mlir_types(new_arg_types.begin(),
                                        new_arg_types.end());
  auto new_args = kernel.front().addArguments(as_mlir_types);
  std::vector<Value*> buffer_args(new_args.begin(), new_args.end());

  auto zero = builder.create<mlir::LLVM::ConstantOp>(
      loc, offset_type, builder.getI64IntegerAttr(0));
  auto one = builder.create<mlir::LLVM::ConstantOp>(
      loc, offset_type, builder.getI64IntegerAttr(1));
  auto baseIndex = builder.create<mlir::LLVM::ConstantOp>(
      loc, LLVMType::getInt32Ty(llvm_dialect), builder.getI32IntegerAttr(0));
  auto dataIndex = builder.create<mlir::LLVM::ConstantOp>(
      loc, LLVMType::getInt32Ty(llvm_dialect), builder.getI32IntegerAttr(1));
  auto offsetIndex = builder.create<mlir::LLVM::ConstantOp>(
      loc, LLVMType::getInt32Ty(llvm_dialect), builder.getI32IntegerAttr(2));
  auto shapeIndex = builder.create<mlir::LLVM::ConstantOp>(
      loc, LLVMType::getInt32Ty(llvm_dialect), builder.getI32IntegerAttr(3));
  auto strideIndex = builder.create<mlir::LLVM::ConstantOp>(
      loc, LLVMType::getInt32Ty(llvm_dialect), builder.getI32IntegerAttr(4));
  // Inject code to map from buffers to input/result values.
  for (auto operand : ordered_operands) {
    TF_ASSIGN_OR_RETURN(auto slice,
                        assignment->GetUniqueTopLevelSlice(operand));
    auto buffer = std::find(buffers.begin(), buffers.end(), slice.allocation());
    auto index = buffer - buffers.begin();
    auto offset = builder.create<mlir::LLVM::ConstantOp>(
        loc, offset_type, builder.getI64IntegerAttr(slice.offset()));
    auto ptr = buffer_args[index];
    // TODO(b/137624192) Add support for indices into tuples.
    for (auto value : operand_to_value_map.at(operand)) {
      // Allocate space for a descriptor. We use the type of the value here,
      // which is expected to be a pointer to a struct of the form
      //   { baseptr, dataptr, offset, shape_vect, stride_vect }
      // where shape_vect and stride_vect are integer vectors with length
      // matching the rank of the tensor.
      auto target_type = value->getType().cast<LLVMType>();
      auto struct_type = target_type.getPointerElementTy();
      auto descPtr =
          builder.create<mlir::LLVM::AllocaOp>(loc, target_type, one, 0);
      // Fill the base and aligned pointers.
      auto casted = builder.create<mlir::LLVM::BitcastOp>(
          loc, struct_type.getStructElementType(0),
          llvm::ArrayRef<Value*>{ptr});
      auto structPtrAddr = builder.create<mlir::LLVM::GEPOp>(
          loc, struct_type.getStructElementType(0), descPtr,
          llvm::ArrayRef<Value*>{zero, baseIndex});
      builder.create<mlir::LLVM::StoreOp>(loc, casted, structPtrAddr);
      casted = builder.create<mlir::LLVM::BitcastOp>(
          loc, struct_type.getStructElementType(1),
          llvm::ArrayRef<Value*>{ptr});
      structPtrAddr = builder.create<mlir::LLVM::GEPOp>(
          loc, struct_type.getStructElementType(1), descPtr,
          llvm::ArrayRef<Value*>{zero, dataIndex});
      builder.create<mlir::LLVM::StoreOp>(loc, casted, structPtrAddr);
      // Fill the offset value.
      auto structOffsetAddr = builder.create<mlir::LLVM::GEPOp>(
          loc, struct_type.getStructElementType(1), descPtr,
          llvm::ArrayRef<Value*>{zero, offsetIndex});
      builder.create<mlir::LLVM::StoreOp>(loc, offset, structOffsetAddr);
      // Fill the shape.
      auto shape = operand->shape();
      auto entry_type =
          struct_type.getStructElementType(3).getArrayElementType();
      // TODO(b/137624192) Pass in the descriptor to allow for dynamic shapes.
      assert(shape.IsArray() && shape.is_static());
      for (auto extent : llvm::enumerate(shape.dimensions())) {
        auto index = builder.create<mlir::LLVM::ConstantOp>(
            loc, offset_type, builder.getI64IntegerAttr(extent.index()));
        auto shapeEntryPtr = builder.create<mlir::LLVM::GEPOp>(
            loc, entry_type, descPtr,
            llvm::ArrayRef<Value*>{zero, shapeIndex, index});
        auto extentValue = builder.create<mlir::LLVM::ConstantOp>(
            loc, entry_type, builder.getI64IntegerAttr(extent.value()));
        builder.create<mlir::LLVM::StoreOp>(loc, extentValue, shapeEntryPtr);
      }
      // Finally, fill the strides.
      // TODO(b/137624192): Take assigned layout into account.
      entry_type = struct_type.getStructElementType(4).getArrayElementType();
      Value* accumulator = nullptr;
      for (int64 idx = shape.rank() - 1; idx >= 0; --idx) {
        auto indexValue = builder.create<mlir::LLVM::ConstantOp>(
            loc, offset_type, builder.getI64IntegerAttr(idx));
        auto strideEntryPtr = builder.create<mlir::LLVM::GEPOp>(
            loc, entry_type, descPtr,
            llvm::ArrayRef<Value*>{zero, strideIndex, indexValue});
        if (accumulator) {
          auto strideValue = builder.create<mlir::LLVM::ConstantOp>(
              loc, entry_type,
              builder.getI64IntegerAttr(shape.dimensions(idx + 1)));
          accumulator = builder.create<mlir::LLVM::MulOp>(
              loc, entry_type, accumulator, strideValue);
        } else {
          accumulator = one;
        }
        builder.create<mlir::LLVM::StoreOp>(loc, accumulator, strideEntryPtr);
      }
      // Now we can use the descriptor instead of the original argument.
      value->replaceAllUsesWith(descPtr);
    }
  }

  // Now we can remove the original arguments, as they should have no more
  // users.
  for (int i = 0; i < num_original_args; ++i) {
    kernel.front().eraseArgument(0);
  }

  return Status::OK();
}

StatusOr<std::unique_ptr<gpu::KernelThunk>> TransformKernelToXlaThunk(
    FuncOp func, const HloInstruction* const instr, ModuleOp kernel_module,
    BufferAssignment* assignment) {
  // Find the single LaunchFuncOp and compute a mapping from operands of
  // the hlo instruction to the corresponding values of the kernel
  // function in the target module;
  LaunchFuncOp launchOp;
  auto walkResult = func.walk([&launchOp](LaunchFuncOp op) {
    if (launchOp) {
      op.emitError("multiple kernels for single top-level HLO");
      return mlir::WalkResult::interrupt();
    }
    launchOp = op;
    return mlir::WalkResult::advance();
  });
  if (walkResult.wasInterrupted()) {
    return InternalError("Multiple kernels for single top-level HLO");
  }
  if (!launchOp) {
    // If there was no launchOp, then no kernel was generated, so the lowering
    // from the LHLO ops to the GPU dialect is not implemented yet.
    return Unimplemented("No kernel was generated.");
  }

  auto kernel = kernel_module.lookupSymbol<LLVMFuncOp>(launchOp.kernel());

  // Store the assignment of operands to block arguments. Note that an operand
  // might be used in multiple argument positions, hence the vector.
  OperandToValueMap operand_to_value_map;
  TF_ASSIGN_OR_RETURN(
      auto ordered_operands,
      ComputeOperandToValueMap(&operand_to_value_map, instr, launchOp, kernel));

  // Get the required buffers to support the inputs. Use a set and vector here
  // to keep the order fixed. This is mostly useful for testing.
  std::unordered_set<const BufferAllocation*> buffers_needed;
  std::vector<const BufferAllocation*> buffers;
  // TODO(b/137624192) Add support for tuples.
  for (auto operand : ordered_operands) {
    TF_ASSIGN_OR_RETURN(auto buffer,
                        assignment->GetUniqueTopLevelSlice(operand));
    if (buffers_needed.insert(buffer.allocation()).second) {
      buffers.push_back(buffer.allocation());
    }
  }

  // TODO(b/137624192) Add support for temp buffer.
  // TODO(b/137624192) Add support for constant buffers.

  // Change the signature to match what the XLA runtime expects from the
  // kernel.
  TF_RETURN_IF_ERROR(InsertBufferLoadPreduleIntoKernel(
      kernel, operand_to_value_map, ordered_operands, assignment, buffers));

  // Finally, create the thunk and set the launch dimensions.
  auto thunk = absl::make_unique<gpu::KernelThunk>(
      buffers, kernel.getName().str(), instr,
      /*unroll_factor=*/1);

  // Set launch bounds.
  mlir::gpu::KernelDim3 block = launchOp.getBlockSizeOperandValues();
  mlir::gpu::KernelDim3 grid = launchOp.getGridSizeOperandValues();
  absl::optional<int64> num_threads = getLaunchBound(block);
  absl::optional<int64> num_blocks = getLaunchBound(grid);
  if (!num_threads || !num_blocks) {
    return Unimplemented("Unsupported launch bounds");
  }
  thunk->SetLaunchDimensions(gpu::LaunchDimensions(*num_blocks, *num_threads));
  return std::move(thunk);
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

  EmissionContext emission_context(std::move(module));
  if (error_handler_) {
    emission_context.setErrorHandler(error_handler_);
  }

  OwningModuleRef mlir_module =
      ModuleOp::create(UnknownLoc::get(emission_context.getContext()));
  LhloDialectEmitter lhlo_emitter(&emission_context, *buffer_assignment,
                                  stream_exec->platform(), *mlir_module);

  TF_RETURN_IF_ERROR(lhlo_emitter.EmitComputation(
      *emission_context.getHloModule()->entry_computation()));

  TF_RETURN_IF_ERROR(
      module_hook_.invoke(IRHook::LoweringStage::LHLO, *mlir_module));

  TF_RETURN_IF_ERROR(LowerLHLOToGPU(*mlir_module));

  TF_RETURN_IF_ERROR(
      module_hook_.invoke(IRHook::LoweringStage::GPU, *mlir_module));

  TF_RETURN_IF_ERROR(LowerKernelBodiesToNVVM(*mlir_module));

  TF_RETURN_IF_ERROR(
      module_hook_.invoke(IRHook::LoweringStage::LLVM, *mlir_module));

  TF_ASSIGN_OR_RETURN(OwningModuleRef kernel_module,
                      ExtractKernelModule(*mlir_module));

  auto thunk_sequence = lhlo_emitter.ConsumeThunkSequence();
  for (auto entry : lhlo_emitter.InstructionToFunctionMap()) {
    TF_ASSIGN_OR_RETURN(
        auto thunk,
        TransformKernelToXlaThunk(entry.second, entry.first, *kernel_module,
                                  buffer_assignment.get()));
    thunk_sequence->push_back(std::move(thunk));
  }

  TF_RETURN_IF_ERROR(
      module_hook_.invoke(IRHook::LoweringStage::KERNEL, *kernel_module));

  auto llvmModule = mlir::translateModuleToNVVMIR(*kernel_module);

  if (!llvmModule) {
    return InternalError("Translation to LLVM failed");
  }

  llvmModule->setModuleIdentifier(emission_context.getHloModule()->name());
  // TODO(herhut): Why is this needed and does not come from the template?
  llvmModule->setDataLayout(gpu::nvptx::kDataLayout);

  const auto& config = emission_context.getHloModule()->config();
  TF_ASSIGN_OR_RETURN(
      auto ptx, xla::gpu::nvptx::CompileToPtx(llvmModule.get(),
                                              GetGpuVersion(stream_exec),
                                              config, GetLibdeviceDir(config)));
  TF_ASSIGN_OR_RETURN(
      auto cubin, se::CompileGpuAsm(stream_exec->device_ordinal(), ptx.c_str(),
                                    gpu::PtxOptsFromConfig(config)));

  auto thunk_schedule = absl::make_unique<ThunkSchedule>(
      std::move(thunk_sequence), std::move(stream_assignment),
      hlo_schedule->ThunkLaunchOrder());

  if (DumpingEnabledForHloModule(*emission_context.getHloModule())) {
    DumpToFileInDirOrStdout(*emission_context.getHloModule(), "thunk_schedule",
                            thunk_schedule->ToString());
  }

  // TODO(b/137624192): Add profiling support.
  return {absl::make_unique<GpuExecutable>(
      ptx, cubin, GetGpuVersion(stream_exec), std::move(thunk_schedule),
      emission_context.releaseHloModule(), std::move(buffer_assignment),
      nullptr, nullptr)};
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

void MlirCompiler::SetErrorHandler(ErrorHandler error_handler) {
  error_handler_ = error_handler;
}

void MlirCompiler::RemoveErrorHandler() { error_handler_ = nullptr; }

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
