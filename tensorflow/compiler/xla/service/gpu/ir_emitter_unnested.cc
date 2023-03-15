/*Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/ir_emitter_unnested.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Linker/Linker.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/GPU/IR/GPUDialect.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Export.h"  // from @llvm-project
#include "tensorflow/compiler/xla/hlo/ir/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/mlir_hlo/lhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/xla/mlir_hlo/lhlo_gpu/IR/lhlo_gpu_ops.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/xla/mlir_hlo/transforms/gpu_passes.h"
#include "tensorflow/compiler/xla/permutation_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/collective_ops_utils.h"
#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/conditional_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/convolution_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/copy_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/custom_call_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/fft_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/for_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/gemm_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_asm_opts_util.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_constants.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_conv_runner.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_fusible.h"
#include "tensorflow/compiler/xla/service/gpu/hlo_to_ir_bindings.h"
#include "tensorflow/compiler/xla/service/gpu/infeed_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emitter_context.h"
#include "tensorflow/compiler/xla/service/gpu/kernel_mapping_scheme.h"
#include "tensorflow/compiler/xla/service/gpu/kernel_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/launch_dimensions.h"
#include "tensorflow/compiler/xla/service/gpu/matmul_utils.h"
#include "tensorflow/compiler/xla/service/gpu/memset_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_all_gather_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_all_reduce_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_all_to_all_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_collective_permute_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_collective_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/outfeed_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/parallel_loop_emitter.h"
#include "tensorflow/compiler/xla/service/gpu/replica_id_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/reusable_kernel_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/sequential_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/target_util.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/compiler/xla/service/gpu/while_thunk.h"
#include "tensorflow/compiler/xla/service/llvm_ir/buffer_assignment_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/dynamic_update_slice_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/fused_ir_emitter.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_array.h"
#include "tensorflow/compiler/xla/service/llvm_ir/kernel_support_library.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/sort_util.h"
#include "tensorflow/compiler/xla/service/name_uniquer.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/translate/hlo_to_mhlo/hlo_utils.h"
#include "tensorflow/compiler/xla/translate/mhlo_to_hlo/attribute_exporter.h"
#include "tensorflow/compiler/xla/translate/mhlo_to_hlo/location_exporter.h"
#include "tensorflow/compiler/xla/translate/mhlo_to_hlo/mlir_hlo_to_hlo.h"
#include "tensorflow/compiler/xla/translate/mhlo_to_lhlo_with_xla/mhlo_to_lhlo_with_xla.h"
#include "tensorflow/compiler/xla/union_find.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/human_readable_json.h"
#include "tensorflow/tsl/platform/logging.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/statusor.h"
#include "tensorflow/tsl/protobuf/dnn.pb.h"

#if GOOGLE_CUDA
#include "tensorflow/compiler/xla/service/gpu/cublas_lt_matmul_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emitter_triton.h"
#endif  // GOOGLE_CUDA

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/compiler/xla/service/gpu/cholesky_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/triangular_solve_thunk.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace xla {
namespace gpu {

namespace {

using absl::InlinedVector;
using absl::StrCat;
using llvm_ir::IrArray;
using llvm_ir::IrName;
using std::optional;

const auto kDimX = TilingScheme::DimX;
const auto kDimY = TilingScheme::DimY;
const auto kDimZ = TilingScheme::DimZ;
const auto kDimTot = TilingScheme::DimTot;

const auto kLinearIndexingX = TilingScheme::LinearIndexingX;
const auto kStridedIndexingX = TilingScheme::StridedIndexingX;

// Some HLO operations are not implemented as Thunks, and only available when
// XLA:GPU compiled for XLA runtime. However we still depend on emitting thunk
// sequence during compilation, and for unsupported operations we emit
// unreachable thunk, which is not supposed to be executed, and exists only
// during compilation as we transition from thunks to XLA runtime.
//
// Examples: Point-to-point communication operations (Send and Recv) are only
// available as XLA runtime custom calls. API_VERSION_TYPED_FFI custom calls
// are only implemented when executing with XLA runtime.
class UnreachableThunk : public Thunk {
 public:
  UnreachableThunk(mlir::Operation* op, std::string error_message)
      : Thunk(Kind::kKernel, ThunkInfo(op)),
        error_message_(std::move(error_message)) {}

  UnreachableThunk(const UnreachableThunk&) = delete;
  UnreachableThunk& operator=(const UnreachableThunk&) = delete;

  Status Initialize(const GpuExecutable& executable,
                    se::StreamExecutor* executor) final {
    return tsl::errors::Internal(error_message_);
  }

  Status ExecuteOnStream(const ExecuteParams& params) final {
    return tsl::errors::Internal(error_message_);
  }

 private:
  std::string error_message_;
};

void AnnotateWithInt32Value(std::string name, int64_t value,
                            const std::string& kernel_name,
                            llvm::Module* llvm_module) {
  llvm::NamedMDNode* nvvm_annotations_node =
      llvm_module->getOrInsertNamedMetadata("nvvm.annotations");
  llvm::Function* ir_kernel = llvm_module->getFunction(kernel_name.c_str());
  llvm::LLVMContext& llvm_context = llvm_module->getContext();

  nvvm_annotations_node->addOperand(llvm::MDNode::get(
      llvm_context,
      {llvm::ConstantAsMetadata::get(ir_kernel),
       llvm::MDString::get(llvm_context, name),
       llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
           llvm::IntegerType::get(llvm_context, /*NumBits=*/32), value))}));
}

// Annotates the launch dimensions of the corresponding IR kernel in
// `llvm_module`.
void AnnotateKernelLaunchDimensions(const LaunchDimensions& launch_dims,
                                    const std::string& kernel_name,
                                    llvm::Module* llvm_module) {
  // Add __launch_bounds__ to metadata. This limits registers per thread to
  // avoid out-of-resources launching errors.

  // Our launch bounds are exact, so we can specify them as
  // reqntid[xyz] rather than maxntid[xyz].
  AnnotateWithInt32Value("reqntidx", launch_dims.thread_counts_per_block().x,
                         kernel_name, llvm_module);
  if (launch_dims.thread_counts_per_block().y > 1) {
    AnnotateWithInt32Value("reqntidy", launch_dims.thread_counts_per_block().y,
                           kernel_name, llvm_module);
  }
  if (launch_dims.thread_counts_per_block().z > 1) {
    AnnotateWithInt32Value("reqntidz", launch_dims.thread_counts_per_block().z,
                           kernel_name, llvm_module);
  }
}

bool IsSingleInstructionFusion(mlir::lmhlo::FusionOp fusion) {
  int instruction_count = 0;
  for (mlir::Operation& instr : fusion.getRegion().front()) {
    if (mlir::isa<mlir::lmhlo::TerminatorOp, mlir::mhlo::ReturnOp,
                  mlir::bufferization::ToTensorOp, mlir::memref::TensorStoreOp>(
            &instr)) {
      continue;
    }
    instruction_count++;
  }
  return instruction_count == 1;
}

bool MayPreventVectorization(mlir::Operation* op) {
  // An empirically chosen constant: unrolling concat with a large amount of
  // arguments causes excessive register spilling.
  static constexpr int kMaxConcatArgumentsForUnrolling = 10;

  auto fusion = mlir::cast<mlir::lmhlo::FusionOp>(op);

  for (mlir::Operation& instr : fusion.getRegion().front()) {
    if (mlir::isa<mlir::lmhlo::TerminatorOp, mlir::mhlo::ReturnOp,
                  mlir::bufferization::ToTensorOp, mlir::memref::TensorStoreOp>(
            &instr)) {
      continue;
    }

    CHECK(instr.getDialect() ==
          instr.getContext()->getLoadedDialect<mlir::mhlo::MhloDialect>())
        << llvm_ir::DumpToString(op);
    switch (*MhloToHloOpcode(&instr)) {
      case HloOpcode::kReduceWindow:
      case HloOpcode::kSort:
      case HloOpcode::kDot:
      case HloOpcode::kSin:
      case HloOpcode::kCos:
      case HloOpcode::kTan:
      case HloOpcode::kPower:
      case HloOpcode::kAtan2:
        return true;
      case HloOpcode::kConcatenate:
        if (instr.getOperands().size() > kMaxConcatArgumentsForUnrolling) {
          return true;
        }
        break;
      case HloOpcode::kReduce:
        if (instr.getNumResults() > 1) {
          return true;
        }
        break;
      default:
        break;
    }
  }
  return false;
}

// Computes the maximum valid unroll factor for a given instruction.
int ComputeMaxUnrollFactor(mlir::Type type,
                           const HloModuleConfig& hlo_module_config) {
  constexpr int kMaxUnrollFactor = 4;

  // Find the largest possible power of two to unroll by.
  // TODO(kramerb): Make this smarter.

  auto shaped_type = type.cast<mlir::ShapedType>();
  int64_t num_elements = std::accumulate(
      shaped_type.getShape().begin(), shaped_type.getShape().end(), int64_t{1},
      std::multiplies<int64_t>());
  for (int i = kMaxUnrollFactor; i > 1; i /= 2) {
    if (num_elements % i == 0) {
      return i;
    }
  }

  // Cannot unroll.
  return 1;
}

// Computes the maximum valid unroll factor for a given instruction.
int ComputeMaxUnrollFactor(mlir::Operation* op,
                           const HloModuleConfig& hlo_module_config) {
  mlir::Type element_shape = [&] {
    if (auto fusion = mlir::dyn_cast<mlir::lmhlo::FusionOp>(op)) {
      return fusion.getFusionRoots()[0]->getResult(0).getType();
    }
    return GetHloOutputs(op)[0].getType();
  }();
  return ComputeMaxUnrollFactor(element_shape, hlo_module_config);
}

// Returns the llvm type for the indices used in the kernel that contains the
// hlo instruction. Such indices include the index for the parallel loop and
// the indices for the tensors accessed by the kernel. The return type is i32
// iff the following conditions are met:
//  . The launch_size of the kernel is within the range of i32.
//  . The sizes of all the tensors accessed within the kernel are within the
//    range of i32.
// Otherwise, the return type is i64.
llvm::Type* GetIndexTypeForKernel(const HloInstruction* hlo,
                                  int64_t launch_size, llvm::IRBuilder<>* b) {
  // Find the unnested hlo instruction for which the kernel is generated for.
  const HloInstruction* unnested_hlo = hlo;
  const HloComputation* computation = hlo->parent();
  if (computation->IsFusionComputation()) {
    unnested_hlo = computation->FusionInstruction();
  }

  auto shape_in_range = [&](const Shape& s) {
    bool in_range = true;
    ShapeUtil::ForEachSubshape(s, [&](const Shape& sub_shape,
                                      const ShapeIndex& /*index*/) {
      if (sub_shape.IsArray() && !IsInt32(ShapeUtil::ElementsIn(sub_shape))) {
        in_range = false;
      }
    });

    return in_range;
  };

  llvm::Type* i64_ty = b->getInt64Ty();
  // Check launch dimension
  if (!IsInt32(launch_size)) {
    return i64_ty;
  }

  // Check the size of result tensors
  if (!shape_in_range(unnested_hlo->shape())) {
    return i64_ty;
  }

  auto hlo_shape_in_range = [&](const HloInstruction* operand) -> bool {
    return shape_in_range(operand->shape());
  };

  // Check the size of input tensors
  if (!absl::c_all_of(unnested_hlo->operands(), hlo_shape_in_range)) {
    return i64_ty;
  }

  // Check the size of the internal result tensors
  if (unnested_hlo->opcode() == HloOpcode::kFusion) {
    if (!absl::c_all_of(
            unnested_hlo->fused_instructions_computation()->instructions(),
            hlo_shape_in_range)) {
      return i64_ty;
    }
  }

  return b->getInt32Ty();
}

// The same as GetIndexTypeForKernel, but works with MLIR ops.
llvm::Type* GetIndexTypeForKernel(mlir::Operation* op, int64_t launch_size,
                                  llvm::IRBuilder<>* b) {
  auto shape_in_range = [&](const Shape& s) {
    bool in_range = true;
    ShapeUtil::ForEachSubshape(s, [&](const Shape& sub_shape,
                                      const ShapeIndex& /*index*/) {
      if (sub_shape.IsArray() && !IsInt32(ShapeUtil::ElementsIn(sub_shape))) {
        in_range = false;
      }
    });

    return in_range;
  };

  llvm::Type* i64_ty = b->getInt64Ty();
  // Check launch dimension
  if (!IsInt32(launch_size)) {
    return i64_ty;
  }

  // Check the size of result tensors
  for (auto result : GetHloOutputs(op)) {
    if (!shape_in_range(GetShape(result))) {
      return i64_ty;
    }
  }

  auto hlo_shape_in_range = [&](mlir::Value operand) -> bool {
    return shape_in_range(GetShape(operand));
  };

  // Check the size of input tensors
  if (!absl::c_all_of(op->getOperands(), hlo_shape_in_range)) {
    return i64_ty;
  }

  // Check the size of the internal result tensors
  if (auto fusion = mlir::dyn_cast<mlir::lmhlo::FusionOp>(op)) {
    auto result = fusion.getRegion().walk([&](mlir::Operation* op) {
      for (mlir::Value result : op->getResults()) {
        if (!hlo_shape_in_range(result)) {
          return mlir::WalkResult::interrupt();
        }
      }
      return mlir::WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      return i64_ty;
    }
  }

  return b->getInt32Ty();
}

// Gets the input shape of the ROOT slices, which will be used as the kernel
// launch dims. The slice input fusion requires the input shapes of the ROOT
// slices to be the same although the (slice) output shapes can be different.
//
// Returns the input shape of the ROOT slices if all the input shapes of ROOT
// slices are the same and the slices are non-strided. Otherwise, returns
// FailedPrecondition.
StatusOr<Shape> GetConsistentInputShapeForRootSlices(
    const HloComputation* fused_computation) {
  const HloInstruction& root = *fused_computation->root_instruction();
  if (root.opcode() == HloOpcode::kSlice) {
    return root.operands()[0]->shape();
  }

  CHECK_EQ(root.opcode(), HloOpcode::kTuple);
  const Shape& first_slice_operand_shape =
      root.operands()[0]->operands()[0]->shape();
  for (size_t i = 1; i < root.operands().size(); ++i) {
    const HloInstruction* slice = root.operands()[i];
    const Shape& operand_shape = slice->operands()[0]->shape();
    if (!ShapeUtil::EqualIgnoringElementType(first_slice_operand_shape,
                                             operand_shape)) {
      return FailedPrecondition(
          "Fused slices do not have the same input shape, fused computation = "
          "%s.",
          root.parent()->name());
    }
  }

  return first_slice_operand_shape;
}

// Returns a sanitized (doesn't need quoting) identifier name from a location.
std::string GetIrNameFromLoc(mlir::Location loc) {
  return llvm_ir::SanitizeConstantName(
      mlir::mhlo::GetDebugNameFromLocation(loc));
}

// For a row reduction, returns the number of rows we can process in parallel
// per warp.
int RowReductionGetRowsPerWarp(int reduced_dimension_size) {
  if (WarpSize() % reduced_dimension_size != 0 ||
      reduced_dimension_size >= WarpSize()) {
    return 1;
  }
  return WarpSize() / reduced_dimension_size;
}

}  // namespace

IrEmitterUnnested::IrEmitterUnnested(const HloModuleConfig& hlo_module_config,
                                     IrEmitterContext* ir_emitter_context)
    : IrEmitter(hlo_module_config, ir_emitter_context, /*is_nested=*/false),
      elemental_emitter_(hlo_module_config_, module_, &b_,
                         GetNestedComputer()) {}

StatusOr<std::unique_ptr<IrEmitterUnnested>> IrEmitterUnnested::Create(
    const HloModuleConfig& hlo_module_config,
    IrEmitterContext* ir_emitter_context) {
  return std::unique_ptr<IrEmitterUnnested>(
      new IrEmitterUnnested(hlo_module_config, ir_emitter_context));
}

llvm::Function* IrEmitterUnnested::BuildKernelPrototype(
    absl::string_view name, absl::Span<const BufferAllocation* const> args) {
  // Compute the kernel name. The opcode string may contain "-" which cannot be
  // in a PTX function name, so sanitize the name before uniquifying it.
  std::string kernel_name = ir_emitter_context_->name_uniquer()->GetUniqueName(
      llvm_ir::SanitizeFunctionName(std::string(name)));

  // Create the kernel and add it to the module.
  llvm::LLVMContext& context = module_->getContext();
  llvm::FunctionType* kernel_type = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getVoidTy(context),
      std::vector<llvm::Type*>(args.size(), b_.getInt8PtrTy()),
      /*isVarArg=*/false);
  llvm::Function* kernel =
      llvm::Function::Create(kernel_type, llvm::GlobalValue::ExternalLinkage,
                             kernel_name.c_str(), module_);

  // Add dereferenceable and alignment information to each of the kernel's
  // parameters.
  auto arg_it = kernel->arg_begin();
  for (size_t arg_no = 0; arg_no < args.size(); ++arg_no) {
    const BufferAllocation* alloc = args[arg_no];
    llvm::Argument& fn_arg = *arg_it;
    ++arg_it;

    kernel->addDereferenceableParamAttr(arg_no, alloc->size());

    const int64_t alignment = [&] {
      if (alloc->is_entry_computation_parameter()) {
        return kEntryParameterAlignBytes;
      } else if (alloc->is_constant()) {
        return kConstantBufferAlignBytes;
      } else {
        return kXlaAllocatedBufferAlignBytes;
      }
    }();

    kernel->addParamAttr(
        arg_no,
        llvm::Attribute::get(context, llvm::Attribute::Alignment, alignment));

    if (alloc->IsPreallocatedTempBuffer()) {
      fn_arg.setName("temp_buf");
    } else {
      fn_arg.setName(StrCat("alloc", alloc->index()));
    }
  }

  AnnotateFunctionAsGpuKernel(module_, kernel, &b_);

  // TODO(b/65380986): Investigate if adding fast math flags for generated
  // kernels makes sense.

  // Update the insert point to the entry basic block.
  llvm::BasicBlock* entry_bb =
      llvm::BasicBlock::Create(context, /*Name=*/"entry", /*Parent=*/kernel);

  // Emit a "return void" at entry_bb's end, and set the insert point before
  // that return instruction.
  b_.SetInsertPoint(llvm::ReturnInst::Create(context, entry_bb));

  return kernel;
}

IrEmitterUnnested::KernelAndIrArrays
IrEmitterUnnested::BuildReusableKernelPrototype(
    absl::string_view suggested_name,
    absl::Span<const ReusableKernelArgument> arguments,
    const LaunchDimensions& launch_dimensions) {
  // If some arguments have the same buffer, we will pass them only once.
  llvm::SmallVector<int> to_llvm_arg_no(arguments.size());
  llvm::SmallVector<int> to_arg_no;
  to_arg_no.reserve(arguments.size());
  for (const auto& [arg_no, argument] : llvm::enumerate(arguments)) {
    if (argument.first_with_same_slice.has_value()) {
      to_llvm_arg_no[arg_no] =
          to_llvm_arg_no[argument.first_with_same_slice.value()];
      continue;
    }

    to_llvm_arg_no[arg_no] = to_arg_no.size();
    to_arg_no.push_back(arg_no);
  }
  const int kNumLlvmArgs = to_arg_no.size();

  // Compute the kernel name. The opcode string may contain "-" which cannot be
  // in a PTX function name, so sanitize the name before uniquifying it.
  std::string kernel_name = ir_emitter_context_->name_uniquer()->GetUniqueName(
      llvm_ir::SanitizeFunctionName(std::string(suggested_name)));

  // Create the kernel and add it to the module.
  llvm::LLVMContext& context = module_->getContext();
  llvm::FunctionType* kernel_type = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getVoidTy(context),
      std::vector<llvm::Type*>(kNumLlvmArgs, b_.getInt8PtrTy()),
      /*isVarArg=*/false);
  llvm::Function* kernel = llvm::Function::Create(
      kernel_type, llvm::GlobalValue::ExternalLinkage, kernel_name, module_);

  AnnotateFunctionAsGpuKernel(module_, kernel, &b_);
  AnnotateKernelLaunchDimensions(launch_dimensions, kernel_name, module_);

  // TODO(b/65380986): Investigate if adding fast math flags for generated
  // kernels makes sense.

  // Update the insert point to the entry basic block.
  llvm::BasicBlock* entry_bb =
      llvm::BasicBlock::Create(context, /*Name=*/"entry", /*Parent=*/kernel);

  // Emit a "return void" at entry_bb's end, and set the insert point before
  // that return instruction.
  b_.SetInsertPoint(llvm::ReturnInst::Create(context, entry_bb));

  for (size_t llvm_arg_no = 0; llvm_arg_no < kernel->arg_size();
       ++llvm_arg_no) {
    const ReusableKernelArgument& kernel_argument =
        arguments[to_arg_no[llvm_arg_no]];
    llvm::Argument& llvm_arg = *kernel->getArg(llvm_arg_no);

    llvm_arg.setName(StrCat("arg", llvm_arg_no));

    kernel->addDereferenceableParamAttr(llvm_arg_no,
                                        kernel_argument.slice.size());

    kernel->addParamAttr(
        llvm_arg_no,
        llvm::Attribute::get(llvm_arg.getContext(), llvm::Attribute::Alignment,
                             kernel_argument.alignment));

    if (!kernel_argument.aliased) {
      kernel->addParamAttr(llvm_arg_no,
                           llvm::Attribute::get(llvm_arg.getContext(),
                                                llvm::Attribute::NoAlias));
    }
  }

  std::vector<llvm_ir::IrArray> ir_arrays;
  for (size_t arg_no = 0; arg_no < arguments.size(); ++arg_no) {
    const ReusableKernelArgument& kernel_argument = arguments[arg_no];
    llvm::Argument& llvm_arg = *kernel->getArg(to_llvm_arg_no[arg_no]);

    llvm::Type* ir_type =
        llvm_ir::ShapeToIrType(kernel_argument.shape, module_);
    llvm_ir::IrArray ir_array(
        CastToTypedValue(kernel_argument.shape, &llvm_arg, &b_), ir_type,
        kernel_argument.shape);

    if (!kernel_argument.written) {
      ir_array.MarkInvariantOverWholeProgram(&llvm_arg.getContext());
    }

    ir_arrays.push_back(ir_array);
  }

  return {kernel, std::move(ir_arrays)};
}

StatusOr<BufferAllocation::Slice> IrEmitterUnnested::GetAllocationSlice(
    mlir::Value v, std::string* constant_name) {
  return xla::gpu::GetAllocationSlice(v, ir_emitter_context_->allocations(),
                                      constant_name);
}

Status IrEmitterUnnested::EmitUnreachable(mlir::Operation* op,
                                          std::string error_message) {
  AddThunkToThunkSequence(std::unique_ptr<Thunk>(
      new UnreachableThunk(op, std::move(error_message))));
  return OkStatus();
}

Status IrEmitterUnnested::EmitConstant(mlir::Operation* op) {
  auto get_global = mlir::cast<mlir::memref::GetGlobalOp>(op);
  auto module = get_global->getParentOfType<mlir::ModuleOp>();
  auto global = mlir::cast<mlir::memref::GlobalOp>(
      module.lookupSymbol(get_global.getName()));
  auto literal = global.getInitialValue()->dyn_cast<mlir::DenseElementsAttr>();
  TF_RET_CHECK(literal);
  TF_ASSIGN_OR_RETURN(int element_bytes,
                      GetElementTypeBytes(literal.getType().getElementType()));
  std::vector<uint8_t> content;
  TF_RETURN_IF_ERROR(CopyDenseElementsDataToXlaFormat(literal, &content));
  ir_emitter_context_->emit_constant(
      literal.getType().getNumElements(), element_bytes, global.getSymName(),
      global->getAttrOfType<mlir::IntegerAttr>("lmhlo.alloc").getInt(), content,
      &b_);
  return OkStatus();
}

static ConditionalThunkConfig GetConditionalThunkConfig(
    mlir::lmhlo::CaseOp op, std::vector<ThunkSequence> branch_thunk_sequences) {
  ConditionalThunkConfig config;
  config.branch_index_is_bool = op.getIndex()
                                    .getType()
                                    .cast<mlir::ShapedType>()
                                    .getElementType()
                                    .isInteger(
                                        /*width=*/1);
  config.branch_count = op.getBranches().size();
  // Pass nullptr as the HloInstruction* to the branch_thunks
  // constructors because these SequentialThunks are logically "part of"
  // this ConditionalThunk, and shouldn't be profiled separately from it.
  config.branch_thunks.reserve(branch_thunk_sequences.size());
  for (auto& branch_thunk_sequence : branch_thunk_sequences) {
    config.branch_thunks.emplace_back(new SequentialThunk(
        Thunk::ThunkInfo(op), std::move(branch_thunk_sequence)));
  }
  return config;
}

Status IrEmitterUnnested::EmitConditional(mlir::Operation* op) {
  auto conditional = mlir::cast<mlir::lmhlo::CaseOp>(op);

  std::vector<ThunkSequence> branch_thunks;

  int branch_count = conditional.getBranches().size();
  branch_thunks.reserve(branch_count);

  for (int j = 0; j < branch_count; ++j) {
    mlir::Region* branch_computation = &conditional.getBranches()[j];
    TF_ASSIGN_OR_RETURN(
        auto ir_emitter,
        IrEmitterUnnested::Create(hlo_module_config_, ir_emitter_context_));
    TF_RETURN_IF_ERROR(ir_emitter->EmitLmhloRegion(branch_computation));
    branch_thunks.push_back(std::move(*ir_emitter->ConsumeThunkSequence()));
  }

  ConditionalThunkConfig config =
      GetConditionalThunkConfig(conditional, std::move(branch_thunks));

  TF_ASSIGN_OR_RETURN(auto slice, GetAllocationSlice(conditional.getIndex()));
  AddThunkToThunkSequence(std::unique_ptr<Thunk>(
      new ConditionalThunk(GetThunkInfo(op), std::move(config), slice)));
  return OkStatus();
}

llvm::Value* IrEmitterUnnested::CreateLoad(llvm::Value* address,
                                           llvm::Type* data_type,
                                           int alignment_bytes) {
  int data_bytes = data_type->getPrimitiveSizeInBits() /
                   primitive_util::BitWidth(PrimitiveType::U8);
  if (alignment_bytes == 0) {
    return b_.CreateLoad(data_type,
                         b_.CreateBitCast(address, data_type->getPointerTo()));
  }

  int alignment_bitwidth =
      alignment_bytes * primitive_util::BitWidth(PrimitiveType::U8);

  llvm::Value* output = llvm::ConstantInt::get(data_type, 0);
  for (int offset_bytes = 0; offset_bytes < data_bytes;
       offset_bytes += alignment_bytes) {
    llvm::Value* offset_address = b_.CreateConstInBoundsGEP1_32(
        b_.getInt8Ty(), address, offset_bytes, "offset_address");
    llvm::Value* partial_value = b_.CreateLoad(b_.getIntNTy(alignment_bitwidth),
                                               offset_address, "partial_value");
    llvm::Value* zextd =
        b_.CreateZExt(partial_value, output->getType(), "partial_value_zextd");
    llvm::Value* shifted = b_.CreateShl(
        zextd, llvm::ConstantInt::get(b_.getInt32Ty(), offset_bytes),
        "partial_input_shifted");
    output = b_.CreateAdd(output, shifted, "output_updated");
  }
  return output;
}

void IrEmitterUnnested::CreateStore(llvm::Value* data, llvm::Value* address,
                                    int alignment_bytes) {
  int data_bytes = data->getType()->getPrimitiveSizeInBits() /
                   primitive_util::BitWidth(PrimitiveType::U8);
  CHECK_GE(data_bytes, alignment_bytes);
  if (alignment_bytes == 0) {
    b_.CreateStore(data,
                   b_.CreateBitCast(address, data->getType()->getPointerTo()));
    return;
  }

  int alignment_bitwidth =
      alignment_bytes * primitive_util::BitWidth(PrimitiveType::U8);

  for (int offset_bytes = 0; offset_bytes < data_bytes;
       offset_bytes += alignment_bytes) {
    llvm::Value* offset_address = b_.CreateConstInBoundsGEP1_32(
        b_.getInt8Ty(), address, offset_bytes, "offset_address");
    llvm::Value* shifted_partial = b_.CreateTrunc(
        b_.CreateLShr(data,
                      llvm::ConstantInt::get(b_.getInt32Ty(), offset_bytes)),
        b_.getIntNTy(alignment_bitwidth), "truncated_value");
    b_.CreateStore(
        shifted_partial,
        b_.CreateBitCast(offset_address,
                         b_.getIntNTy(alignment_bitwidth)->getPointerTo()));
  }
}

// Input = {dynamic array(with dynamic dimension meta data at the end)}
// Output = {static array, dynamic_dim0, dynamic_dim1}
Status IrEmitterUnnested::EmitPadToStatic(mlir::Operation* op) {
  // TODO(jurahul): Create an op to represent PadToStatic.
  auto pad_to_static = mlir::cast<mlir::lmhlo::CustomCallOp>(op);
  int unroll_factor = 1;
  std::string ir_name = GetIrNameFromLoc(pad_to_static.getLoc());

  const Shape& input_shape = GetShape(pad_to_static.getArgs().front());
  TF_ASSIGN_OR_RETURN(LaunchDimensions launch_dimensions,
                      CalculateLaunchDimensions(
                          input_shape, ir_emitter_context_->gpu_device_info(),
                          {unroll_factor}));
  TF_ASSIGN_OR_RETURN(std::vector<llvm_ir::IrArray> ir_arrays,
                      BuildKernelThunk(pad_to_static, launch_dimensions));

  const llvm_ir::IrArray source_array = ir_arrays[0];
  const llvm_ir::IrArray output_array = ir_arrays[1];
  auto output_dim_arrays =
      absl::Span<const llvm_ir::IrArray>(ir_arrays).subspan(2);

  llvm::Type* index_ty = GetIndexTypeForKernel(
      pad_to_static, launch_dimensions.launch_bound(), &b_);

  // pseudo code for PadToStatic on a 2d array
  //   int* source_array = input[0];
  //   int* dest_array = output[0];
  llvm::Value* source_buffer = source_array.GetBasePointer();
  llvm::Value* raw_buffer =
      b_.CreateBitCast(source_buffer, b_.getInt8Ty()->getPointerTo());

  // TODO(jurahul): input_shape here is the static shape of the input (which has
  // a dynamic shape in XLA). Currently, we are mapping that to a static shaped
  // memref. When we change that to a more appropriate representation in MLIR,
  // fix this code to correctly deduce the static shape backing the dynamically
  // shaped memref.
  int64_t raw_data_size = ShapeUtil::ByteSizeOf(input_shape);

  //   int* dyn_dim0_size = source_array + meta_data_offset;
  //   int* dyn_dim1_size = source_array + meta_data_offset + sizeof(int);
  std::vector<llvm::Value*> dynamic_dims;
  int alignment = raw_data_size % sizeof(int32_t);
  for (int64_t i = 1; i < pad_to_static.getOutput().size(); ++i) {
    // Dynamic size of each dimension is attached at the end of the source
    // array(operand(0)). We need to extract these value.
    const Shape& dim_shape = GetShape(pad_to_static.getOutput()[i]);
    TF_RET_CHECK(Shape::Equal()(dim_shape, ShapeUtil::MakeScalarShape(S32)));

    const int64_t dim_index = i - 1;
    llvm::Value* metadata = b_.CreateConstInBoundsGEP1_32(
        b_.getInt8Ty(), raw_buffer,
        raw_data_size + dim_index * sizeof(int32_t));
    llvm::Value* dyn_dim_size =
        CreateLoad(metadata, b_.getInt32Ty(), alignment);
    dynamic_dims.push_back(dyn_dim_size);
  }

  // only one thread need to store the dynamic index
  //   int thread_id = GetThreadId();
  //   int block_id = GetBlockId();
  //   if (thread_id == 0 && block_id == 0) {
  //     *output[1] = *dyn_dim0_size;
  //     *output[2] = *dyn_dim1_size;
  //   }
  KernelSupportLibrary{&b_}.If("is_thread_0", IsBlock0Thread0(&b_), [&] {
    for (int64_t i = 1; i < pad_to_static.getOutput().size(); ++i) {
      const int64_t dim_index = i - 1;
      llvm::Value* dest_dim_size_address =
          output_dim_arrays[dim_index].GetBasePointer();
      // output[i] stores dynamic_dim_(i-1)
      CreateStore(dynamic_dims[dim_index], dest_dim_size_address, alignment);
    }
  });

  //     int dyn_element_total = 1;
  //     dyn_element_total *= *dyn_dim0_size;
  //     dyn_element_total *= *dyn_dim1_size;
  llvm::Value* dyn_element_total = llvm::ConstantInt::get(index_ty, 1);
  for (llvm::Value* dynamic_dim : dynamic_dims) {
    dyn_element_total =
        b_.CreateMul(dyn_element_total,
                     b_.CreateIntCast(dynamic_dim, dyn_element_total->getType(),
                                      /*isSigned=*/true),
                     /*Name=*/"dyn_element_total_pad");
  }

  //   linear_index = block_id * threads_per_block + thread_id;
  //   if (linear_index < max_num_element) {
  //     Index static_index =
  //         delinerized(linerized_index, static_dim0_size, static_dim1_size);
  //     if (linerized_index < dyn_element_total) {
  //       Index dyn_index =
  //           delinerized(linerized_index, *dyn_dim0_size, *dyn_dim1_size);
  //       dest_array[dyn_index.dim0][dyn_index.dim1] =
  //           source_array[static_index.dim0][static_index.dim1];
  //     }
  //   }
  llvm_ir::BodyEmitter body_generator =
      [&](const llvm_ir::IrArray::Index& array_index) -> Status {
    llvm::Value* linearIndex =
        array_index.Linearize(input_shape.dimensions(), &b_);
    auto if_in_dyn_bounds = llvm_ir::EmitIfThenElse(
        b_.CreateICmpULT(linearIndex, dyn_element_total),
        llvm_ir::IrName(ir_name, "in_dyn_bounds"), &b_, false);
    // Set IR builder insertion point to the body of the if structure.
    llvm_ir::SetToFirstInsertPoint(if_in_dyn_bounds.true_block, &b_);
    llvm_ir::IrArray::Index dyn_index(linearIndex, input_shape,
                                      absl::MakeSpan(dynamic_dims), &b_);
    output_array.EmitWriteArrayElement(
        dyn_index,
        source_array.EmitReadArrayElement(array_index, &b_, /*name=*/""), &b_,
        /*use_linear_index=*/false);
    return OkStatus();
  };

  const Shape& data_shape = GetShape(pad_to_static.getOutput().front());
  TF_RETURN_IF_ERROR(ParallelLoopEmitter(body_generator, data_shape,
                                         launch_dimensions, &b_,
                                         {unroll_factor})
                         .EmitLoop(ir_name, index_ty));
  return OkStatus();
}

// Input = {dynamic array(with dynamic dimension meta data at the end)}
// Output = {static array, dynamic_dim0, dynamic_dim1}
Status IrEmitterUnnested::EmitSliceToDynamic(mlir::Operation* op) {
  // TODO(jurahul): Create an op to represent SliceToDynamic.
  auto slice_to_dynamic = mlir::cast<mlir::lmhlo::CustomCallOp>(op);
  int unroll_factor = 1;
  std::string ir_name = GetIrNameFromLoc(slice_to_dynamic.getLoc());

  const Shape& input_shape = GetShape(slice_to_dynamic.getArgs().front());
  TF_ASSIGN_OR_RETURN(LaunchDimensions launch_dimensions,
                      CalculateLaunchDimensions(
                          input_shape, ir_emitter_context_->gpu_device_info(),
                          {unroll_factor}));
  llvm::Type* index_ty = GetIndexTypeForKernel(
      slice_to_dynamic, launch_dimensions.launch_bound(), &b_);
  TF_ASSIGN_OR_RETURN(std::vector<llvm_ir::IrArray> ir_arrays,
                      BuildKernelThunk(slice_to_dynamic, launch_dimensions));

  TF_RET_CHECK(slice_to_dynamic.getOutput().size() == 1);
  const Shape& data_shape = GetShape(slice_to_dynamic.getOutput().front());

  // TODO(jurahul): data_shape here is the static shape of the output (which has
  // a dynamic shape in XLA). Currently, we are mapping that to a static shaped
  // memref. When we change that to a more appropriate representation in MLIR,
  // fix this code to correctly deduce the static shape backing the dynamically
  // shaped memref.

  // calculate the location where metadata needs to be inserted
  //   int* dyn_dim0_size = dest_array + meta_data_offset;
  //   int* dyn_dim1_size = dest_array + meta_data_offset + sizeof(int);
  int32_t raw_data_size = ShapeUtil::ByteSizeOf(data_shape);

  // pseudo code for sliceToDynamic on a 2d array
  //   int* source_array = input[0];
  //   int* dest_array = output[0];
  const llvm_ir::IrArray data_array = ir_arrays.back();
  llvm::Value* dest_buffer = data_array.GetBasePointer();
  llvm::Value* raw_buffer =
      b_.CreateBitCast(dest_buffer, b_.getInt8Ty()->getPointerTo());

  // Load dynamic dimensions from memory.
  std::vector<llvm::Value*> dynamic_dims;
  int alignment = raw_data_size % sizeof(int32_t);
  for (int64_t i = 1; i < slice_to_dynamic.getArgs().size(); ++i) {
    // const int64_t dim_index = i - 1;
    llvm::Value* source_buffer = ir_arrays[i].GetBasePointer();
    llvm::Type* source_buffer_pointee_type = ir_arrays[i].GetBasePointeeType();
    llvm::LoadInst* dyn_dim_size =
        Load(source_buffer_pointee_type, source_buffer, "dyn_dim_size");
    dynamic_dims.push_back(dyn_dim_size);
  }

  // only one thread need to store the dynamic index
  //   int thread_id = GetThreadId();
  //   int block_id = GetBlockId();
  //   if (thread_id == 0 && block_id == 0) {
  //     *dyn_dim0_size = *output[1];
  //     *dyn_dim1_size = *output[2];
  //   }
  KernelSupportLibrary{&b_}.If("is_thread_0", IsBlock0Thread0(&b_), [&] {
    for (int64_t i = 1; i < slice_to_dynamic.getArgs().size(); ++i) {
      const int64_t dim_index = i - 1;
      llvm::Value* metadata = b_.CreateConstInBoundsGEP1_32(
          b_.getInt8Ty(), raw_buffer,
          raw_data_size + dim_index * sizeof(int32_t));
      // output[i] stores dynamic_dim_(i-1)
      CreateStore(dynamic_dims[dim_index], metadata, alignment);
    }
  });

  //     int dyn_element_total = 1;
  //     dyn_element_total *= dyn_dim0_size;
  //     dyn_element_total *= dyn_dim1_size;
  llvm::Value* dyn_element_total = llvm::ConstantInt::get(index_ty, 1);
  for (llvm::Value* dynamic_dim : dynamic_dims) {
    dyn_element_total =
        b_.CreateMul(dyn_element_total,
                     b_.CreateIntCast(dynamic_dim, dyn_element_total->getType(),
                                      /*isSigned=*/true),
                     /*Name=*/"dyn_element_total_slice");
  }

  //   linear_index = block_id * threads_per_block + thread_id;
  //   if (linear_index < max_num_element) {
  //     Index static_index =
  //         delinerized(linerized_index, static_dim0_size, static_dim1_size);
  //     if (linerized_index < dyn_element_total) {
  //       Index dyn_index =
  //           delinerized(linerized_index, *dyn_dim0_size, *dyn_dim1_size);
  //       dest_array[static_index.dim0][static_index.di] =
  //           source_array[dyn_index.dim0][dyn_index.dim1];
  //     }
  //   }
  llvm_ir::BodyEmitter body_generator =
      [&](const llvm_ir::IrArray::Index& array_index) -> Status {
    llvm::Value* linearIndex =
        array_index.Linearize(input_shape.dimensions(), &b_);
    auto if_in_dyn_bounds = llvm_ir::EmitIfThenElse(
        b_.CreateICmpULT(linearIndex, dyn_element_total),
        llvm_ir::IrName(ir_name, "in_dyn_bounds"), &b_, false);
    // Set IR builder insertion point to the body of the if structure.
    llvm_ir::SetToFirstInsertPoint(if_in_dyn_bounds.true_block, &b_);
    llvm_ir::IrArray::Index dyn_index(linearIndex, input_shape,
                                      absl::MakeSpan(dynamic_dims), &b_);

    data_array.EmitWriteArrayElement(
        array_index,
        ir_arrays[0].EmitReadArrayElement(dyn_index, &b_, /*name=*/"",
                                          /*use_linear_index=*/false),
        &b_);
    return OkStatus();
  };

  TF_RETURN_IF_ERROR(ParallelLoopEmitter(body_generator, data_shape,
                                         launch_dimensions, &b_,
                                         {unroll_factor})
                         .EmitLoop(ir_name, index_ty));
  return OkStatus();
}

Status IrEmitterUnnested::EmitConvolutionThunk(mlir::Operation* op) {
  using mlir::dyn_cast;
  using mlir::lmhlo_gpu::Activation;
  using mlir::lmhlo_gpu::ConvBackwardFilterOp;
  using mlir::lmhlo_gpu::ConvBackwardInputOp;
  using mlir::lmhlo_gpu::ConvForwardFusedOp;
  using mlir::lmhlo_gpu::ConvForwardFusedSideInputOp;
  using mlir::lmhlo_gpu::ConvForwardOp;

  // Last 2 operands of the convolution operation are the result and scratch.
  std::vector<BufferAllocation::Slice> operand_slices;
  int64_t num_operands = op->getNumOperands();
  operand_slices.reserve(num_operands - 2);
  for (mlir::Value operand : op->getOperands().drop_back(2)) {
    TF_ASSIGN_OR_RETURN(auto slice, GetAllocationSlice(operand));
    operand_slices.push_back(slice);
  }

  mlir::Value conv_result = op->getOperand(num_operands - 2);
  mlir::Value scratch_result = op->getOperand(num_operands - 1);
  TF_ASSIGN_OR_RETURN(auto conv_result_slice, GetAllocationSlice(conv_result));
  TF_ASSIGN_OR_RETURN(auto scratch_slice, GetAllocationSlice(scratch_result));

  auto apply_layout = [](const Shape& shape,
                         mlir::ArrayRef<int64_t> minor_to_major) {
    return ShapeUtil::MakeShapeWithDenseLayout(
        shape.element_type(), shape.dimensions(), minor_to_major);
  };

  GpuConvDescriptor descriptor;

  auto fill_conv_descriptor = [&](auto op) {
    descriptor.operand0_shape =
        apply_layout(GetShape(op->getOperand(0)),
                     op.getBackendConfig().getOperand_0Layout());
    descriptor.operand1_shape =
        apply_layout(GetShape(op->getOperand(1)),
                     op.getBackendConfig().getOperand_1Layout());
    descriptor.result_shape = apply_layout(
        GetShape(conv_result), op.getBackendConfig().getResultLayout());
    descriptor.dnums = ConvertConvDimensionNumbers(op.getDimensionNumbers());
    descriptor.scratch_size = scratch_slice.size();
    mlir::DenseIntElementsAttr window_strides = op.getWindowStrides().value();
    mlir::DenseIntElementsAttr padding = op.getPadding().value();
    mlir::DenseIntElementsAttr lhs_dilation = op.getLhsDilation().value();
    mlir::DenseIntElementsAttr rhs_dilation = op.getRhsDilation().value();
    mlir::DenseElementsAttr window_reversal = op.getWindowReversal().value();
    for (auto index : llvm::seq<int>(0, window_strides.getNumElements())) {
      WindowDimension* dim = descriptor.window.add_dimensions();
      // Window size for a convolution is the same as the kernel size.
      // Kernel size of the convolution is operand1_shape. We need to look at
      // the convolution dimension numbers kernel spatial dimensions to get
      // the window size.
      int kernel_dim = descriptor.dnums.kernel_spatial_dimensions(index);
      dim->set_size(descriptor.operand0_shape.dimensions(kernel_dim));
      dim->set_stride(window_strides.getValues<int64_t>()[index]);
      dim->set_padding_low(padding.getValues<int64_t>()[index]);
      dim->set_padding_high(padding.getValues<int64_t>()[index]);
      dim->set_base_dilation(lhs_dilation.getValues<int64_t>()[index]);
      dim->set_window_dilation(rhs_dilation.getValues<int64_t>()[index]);
      dim->set_window_reversal(window_reversal.getValues<bool>()[index]);
    }
    descriptor.feature_group_count = op.getFeatureGroupCount();
    {
      auto* algorithm = descriptor.backend_config.mutable_algorithm();
      algorithm->set_algo_id(op.getBackendConfig().getAlgorithm());
      algorithm->set_math_type(op.getBackendConfig().getTensorOpsEnabled()
                                   ? se::dnn::AlgorithmProto::TENSOR_OP_MATH
                                   : se::dnn::AlgorithmProto::DEFAULT_MATH);
      for (int i = 0; i < op.getBackendConfig().getKnobIds().size(); ++i) {
        // N.B. tuning_knobs is a map rather than a repeated field, so this
        // doesn't require reserving space up front.
        (*algorithm
              ->mutable_tuning_knobs())[op.getBackendConfig().getKnobIds()[i]] =
            op.getBackendConfig().getKnobValues()[i];
      }
      algorithm->set_is_cudnn_frontend(
          op.getBackendConfig().getIsCudnnFrontend());
      auto workspace_size = op.getBackendConfig().getWorkspaceSize();
      if (workspace_size >= 0) {
        algorithm->mutable_workspace_size()->set_value(workspace_size);
      }
    }
    descriptor.backend_config.set_conv_result_scale(
        op.getResultScale().convertToDouble());
    descriptor.backend_config.set_reordered_int8_nchw_vect(
        op.getBackendConfig().getIsCudnnReorderedInt8());
  };

  auto set_activation_mode = [&](auto op) -> Status {
    TF_ASSIGN_OR_RETURN(stream_executor::dnn::ActivationMode activation_mode,
                        ConvertConvActivationMode(op.getActivationMode()));
    descriptor.backend_config.set_activation_mode(
        static_cast<int64_t>(activation_mode));
    return OkStatus();
  };

  if (auto conv = dyn_cast<ConvForwardOp>(op)) {
    descriptor.kind = CudnnConvKind::kForward;
    fill_conv_descriptor(conv);
  } else if (auto conv = dyn_cast<ConvBackwardInputOp>(op)) {
    descriptor.kind = CudnnConvKind::kBackwardInput;
    fill_conv_descriptor(conv);
  } else if (auto conv = dyn_cast<ConvBackwardFilterOp>(op)) {
    descriptor.kind = CudnnConvKind::kBackwardFilter;
    fill_conv_descriptor(conv);
  } else if (auto conv = dyn_cast<ConvForwardFusedOp>(op)) {
    descriptor.kind = CudnnConvKind::kForwardActivation;
    fill_conv_descriptor(conv);
    TF_RETURN_IF_ERROR(set_activation_mode(conv));
  } else if (auto conv = dyn_cast<ConvForwardFusedSideInputOp>(op)) {
    descriptor.kind = CudnnConvKind::kForwardActivation;
    fill_conv_descriptor(conv);
    TF_RETURN_IF_ERROR(set_activation_mode(conv));
    descriptor.backend_config.set_side_input_scale(
        conv.getSideInputScale().convertToDouble());
  } else {
    return InternalError("Unexpected operation");
  }
  TF_ASSIGN_OR_RETURN(GpuConvConfig config, GetGpuConvConfig(descriptor, ""));
  AddThunkToThunkSequence(std::make_unique<ConvolutionThunk>(
      GetThunkInfo(op), std::move(config), std::move(operand_slices),
      conv_result_slice, scratch_slice));
  return OkStatus();
}

Status IrEmitterUnnested::EmitGemmThunk(mlir::Operation* op) {
  auto gemm = mlir::dyn_cast<mlir::lmhlo_gpu::GEMMOp>(op);
  TF_RET_CHECK(gemm != nullptr);

  TF_ASSIGN_OR_RETURN(auto a, GetAllocationSlice(gemm.getA()));
  TF_ASSIGN_OR_RETURN(auto b, GetAllocationSlice(gemm.getB()));
  TF_ASSIGN_OR_RETURN(auto c, GetAllocationSlice(gemm.getC()));

  TF_ASSIGN_OR_RETURN(GemmConfig config, GemmConfig::For(gemm));
  auto thunk =
      std::make_unique<GemmThunk>(GetThunkInfo(op), std::move(config), a, b, c);

  AddThunkToThunkSequence(std::move(thunk));
  return OkStatus();
}

#if GOOGLE_CUDA

Status IrEmitterUnnested::EmitCublasLtMatmulThunk(mlir::Operation* op) {
  auto matmul = mlir::dyn_cast<mlir::lmhlo_gpu::CublasLtMatmulOp>(op);
  TF_RET_CHECK(matmul != nullptr);

  TF_ASSIGN_OR_RETURN(auto a, GetAllocationSlice(matmul.getA()));
  TF_ASSIGN_OR_RETURN(auto b, GetAllocationSlice(matmul.getB()));
  TF_ASSIGN_OR_RETURN(auto c, GetAllocationSlice(matmul.getC()));
  TF_ASSIGN_OR_RETURN(auto d, GetAllocationSlice(matmul.getD()));

  BufferAllocation::Slice bias, a_scale, b_scale, c_scale, d_scale, d_amax;
  if (matmul.getBias() != nullptr) {
    TF_ASSIGN_OR_RETURN(bias, GetAllocationSlice(matmul.getBias()));
  }

  BufferAllocation::Slice aux;
  if (matmul.getAux() != nullptr) {
    TF_ASSIGN_OR_RETURN(aux, GetAllocationSlice(matmul.getAux()));
  }

  TF_ASSIGN_OR_RETURN(cublas_lt::MatmulPlan plan,
                      cublas_lt::MatmulPlan::For(matmul));
  auto thunk = std::make_unique<CublasLtMatmulThunk>(
      GetThunkInfo(op), std::move(plan), matmul.getAlgorithm(), a, b, c, d,
      bias, aux, a_scale, b_scale, c_scale, d_scale, d_amax);

  AddThunkToThunkSequence(std::move(thunk));
  return OkStatus();
}

Status IrEmitterUnnested::EmitCublasLtMatmulThunkF8(mlir::Operation* op) {
  auto matmul = mlir::dyn_cast<mlir::lmhlo_gpu::CublasLtMatmulF8Op>(op);
  TF_RET_CHECK(matmul != nullptr);

  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice a,
                      GetAllocationSlice(matmul.getA()));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice b,
                      GetAllocationSlice(matmul.getB()));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice c,
                      GetAllocationSlice(matmul.getC()));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice d,
                      GetAllocationSlice(matmul.getD()));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice a_scale,
                      GetAllocationSlice(matmul.getAScale()));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice b_scale,
                      GetAllocationSlice(matmul.getBScale()));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice c_scale,
                      GetAllocationSlice(matmul.getCScale()));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice d_scale,
                      GetAllocationSlice(matmul.getDScale()));
  BufferAllocation::Slice d_amax;
  if (matmul.getDAmax() != nullptr) {
    TF_ASSIGN_OR_RETURN(d_amax, GetAllocationSlice(matmul.getDAmax()));
  }

  BufferAllocation::Slice bias, aux;  // Not used.

  TF_ASSIGN_OR_RETURN(cublas_lt::MatmulPlan plan,
                      cublas_lt::MatmulPlan::For(matmul));
  auto thunk = std::make_unique<CublasLtMatmulThunk>(
      GetThunkInfo(op), std::move(plan), matmul.getAlgorithm(), a, b, c, d,
      bias, aux, a_scale, b_scale, c_scale, d_scale, d_amax);

  AddThunkToThunkSequence(std::move(thunk));
  return OkStatus();
}

Status IrEmitterUnnested::EmitConvolutionReorderThunk(mlir::Operation* op) {
  using mlir::dyn_cast;
  using mlir::lmhlo_gpu::CudnnConvReorderFilterAndBiasOp;
  using mlir::lmhlo_gpu::CudnnConvReorderFilterOp;

  std::vector<BufferAllocation::Slice> operand_slices;
  std::vector<BufferAllocation::Slice> result_slices;
  std::vector<int64_t> filter_dims;

  auto set_filter_data = [&](auto op) -> Status {
    TF_ASSIGN_OR_RETURN(BufferAllocation::Slice filter_input,
                        GetAllocationSlice(op.getFilterInput()));
    operand_slices.push_back(filter_input);

    TF_ASSIGN_OR_RETURN(BufferAllocation::Slice filter_output,
                        GetAllocationSlice(op.getFilterOutput()));
    result_slices.push_back(filter_output);

    auto filter_dims_values = op.getFilterDims().template getValues<int64_t>();
    filter_dims.assign(filter_dims_values.begin(), filter_dims_values.end());
    return OkStatus();
  };

  if (auto reorder = dyn_cast<CudnnConvReorderFilterAndBiasOp>(op)) {
    TF_RETURN_IF_ERROR(set_filter_data(reorder));

    TF_ASSIGN_OR_RETURN(BufferAllocation::Slice bias_input,
                        GetAllocationSlice(reorder.getBiasInput()));
    operand_slices.push_back(bias_input);

    TF_ASSIGN_OR_RETURN(BufferAllocation::Slice bias_output,
                        GetAllocationSlice(reorder.getBiasOutput()));
    result_slices.push_back(bias_output);
  } else if (auto reorder = dyn_cast<CudnnConvReorderFilterOp>(op)) {
    TF_RETURN_IF_ERROR(set_filter_data(reorder));
  } else {
    return InternalError("Unexpected operation");
  }

  auto thunk = std::make_unique<ConvolutionReorderThunk>(
      GetThunkInfo(op), absl::MakeSpan(filter_dims), std::move(operand_slices),
      std::move(result_slices));

  AddThunkToThunkSequence(std::move(thunk));
  return OkStatus();
}

#endif  // GOOGLE_CUDA

namespace {
// An MLIR value and its name as defined in the ODS spec.
struct NamedValue {
  mlir::Value value;
  absl::string_view name;
};

// Determine if we enable the row optimized codegen.  When we have a
// fusion with only point-wise operations, scalar broadcasting and row
// broadcasting, we can trigger a kernel that vectorize the row loads.
// This speed up the kernel, in particular on A100.
// Returns a pair<bool, int>. The bool mean should we try to enable
// row vectorization.  The int is the number of inputs with the higher
// rank.
std::pair<bool, int> RowVectorizationEnabled(mlir::lmhlo::FusionOp fusion) {
  const auto is_row_major = [](mlir::Value value) {
    // Only tested when the inputs are row-major. So only
    // enable that case. Maybe it would works if only the
    // inner dimensions is contiguous.
    return LayoutUtil::IsMonotonicWithDim0Major(GetShape(value).layout());
  };
  bool row_vectorized =
      fusion.getFusionResults().size() == 1 &&  // Not tested with MOF.
      absl::c_all_of(GetHloOperands(fusion), is_row_major) &&
      absl::c_all_of(GetHloOutputs(fusion), is_row_major);

  // Check that the operations in the fusion are supported.  Each
  // supported operation (or category) must be manually vetted as XLA
  // only unrolls and relies on LLVM to vectorize. But this is brittle.
  // Currently tested and supported operations:
  // Elementwise, scalar and row broadcasting.
  //
  // We also detect at the same time if there is a row broadcasting
  // operation.
  bool some_row_broadcasting = false;
  auto out_rank =
      fusion.getFusionResults()[0].getType().cast<mlir::ShapedType>().getRank();
  int num_big_inputs = 0;
  for (mlir::Operation& op : fusion.getRegion().front()) {
    if (auto load = mlir::dyn_cast<mlir::bufferization::ToTensorOp>(op)) {
      auto rank = load.getResult().getType().cast<mlir::ShapedType>().getRank();
      num_big_inputs += static_cast<int>(rank == out_rank);
      continue;
    } else if (mlir::isa<mlir::memref::TensorStoreOp, mlir::lmhlo::TerminatorOp,
                         mlir::mhlo::ReturnOp, mlir::mhlo::ConstantOp,
                         mlir::lmhlo::ConstantOp>(op)) {
      continue;
    }
    HloOpcode opcode = *MhloToHloOpcode(&op);
    if (HloInstruction::IsOpElementwise(opcode)) {
      continue;
    }

    if (auto broadcast = mlir::dyn_cast<mlir::mhlo::BroadcastInDimOp>(op)) {
      const auto& broadcast_dimensions_size =
          broadcast.getBroadcastDimensions().size();
      if (broadcast_dimensions_size == 0) {
        continue;
      }
      llvm::SmallVector<int64_t> broadcast_dimensions;
      broadcast_dimensions.reserve(broadcast_dimensions_size);
      for (const llvm::APInt& int_value : broadcast.getBroadcastDimensions()) {
        broadcast_dimensions.push_back(int_value.getSExtValue());
      }

      auto rank = GetShape(broadcast.getResult()).rank();
      if (broadcast_dimensions.size() == 1 &&
          broadcast_dimensions.back() == (rank - 1)) {
        some_row_broadcasting = true;
        continue;
      }
    }
    VLOG(2) << "Row vectorization not enabled due to this op: "
            << llvm_ir::DumpToString(&op);
    return std::make_pair(false, 0);
  }
  // Trigger only when there is a row broadcasting.
  return std::make_pair(row_vectorized && some_row_broadcasting,
                        num_big_inputs);
}
}  // namespace

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
Status IrEmitterUnnested::EmitCholeskyThunk(mlir::Operation* op) {
  auto cholesky_op = mlir::cast<mlir::lmhlo_gpu::CholeskyOp>(op);

  const Shape shape = GetShape(cholesky_op.getInput());
  int ndim = shape.dimensions_size();
  CHECK_GE(ndim, 2);
  int64_t n = shape.dimensions(ndim - 1);

  const auto& dims = shape.dimensions();
  int64_t batch_size =
      std::accumulate(dims.begin(), dims.end() - 2, int64_t{1},
                      [](int64_t a, int64_t b) { return a * b; });

  TF_ASSIGN_OR_RETURN(auto operand_buffer,
                      GetAllocationSlice(cholesky_op.getInput()));
  TF_ASSIGN_OR_RETURN(auto a_buffer,
                      GetAllocationSlice(cholesky_op.getOutput()));
  TF_ASSIGN_OR_RETURN(auto workspace_buffer,
                      GetAllocationSlice(cholesky_op.getScratch()));
  TF_ASSIGN_OR_RETURN(auto info_buffer,
                      GetAllocationSlice(cholesky_op.getInfo()));

  ThunkSequence thunks;

  if (operand_buffer != a_buffer) {
    thunks.push_back(std::make_unique<DeviceToDeviceCopyThunk>(
        GetThunkInfo(op),
        /*source_buffer=*/operand_buffer,
        /*destination_buffer=*/a_buffer,
        /*mem_size=*/ShapeUtil::ByteSizeOf(shape),
        /*source_value=*/cholesky_op.getInput(),
        /*destination_value=*/cholesky_op.getOutput()));
  }

  CholeskyOptions options;
  options.set_lower(cholesky_op.getIsLower());
  thunks.push_back(std::make_unique<CholeskyThunk>(
      GetThunkInfo(op), options,
      PtxOptsFromDebugOptions(hlo_module_config_.debug_options()), a_buffer,
      workspace_buffer, info_buffer, shape.element_type(), batch_size, n));

  // Elide the sequential thunk if there's no copy.
  if (thunks.size() == 1) {
    AddThunkToThunkSequence(std::move(thunks[0]));
  } else {
    AddThunkToThunkSequence(
        std::make_unique<SequentialThunk>(GetThunkInfo(op), std::move(thunks)));
  }

  return OkStatus();
}
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

Status IrEmitterUnnested::EmitCustomCallThunk(mlir::Operation* op) {
  auto custom_call = mlir::cast<mlir::lmhlo::CustomCallOp>(op);
  const std::string call_target_name = custom_call.getCallTargetName().str();

  void* call_target = CustomCallTargetRegistry::Global()->Lookup(
      call_target_name, std::string(platform_name()));

  // Typed custom calls only are supported by XLA runtime. It's ok to emit a
  // thunk with an unresolved custom call target, as we'll never execute it.
  bool is_typed_custom_call =
      custom_call.getApiVersion() ==
      mlir::mhlo::CustomCallApiVersion::API_VERSION_TYPED_FFI;

  if (!call_target && !is_typed_custom_call) {
    return Unimplemented(
        "No registered implementation for custom call to \"%s\"",
        call_target_name);
  }

  std::vector<CustomCallThunk::OptionalSlice> operands;
  std::vector<CustomCallThunk::OptionalSlice> results;
  if (custom_call.getTargetArgMapping()) {
    auto values_to_slices_with_token_holes =
        [&](mlir::ValueRange operands,
            mlir::ArrayRef<int64_t> op_to_target_mapping, int64_t num_target)
        -> StatusOr<std::vector<CustomCallThunk::OptionalSlice>> {
      std::vector<CustomCallThunk::OptionalSlice> slices(num_target);
      for (auto index_and_value_it :
           llvm::zip(op_to_target_mapping, operands)) {
        int64_t index = std::get<0>(index_and_value_it);
        mlir::Value value = std::get<1>(index_and_value_it);
        TF_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                            GetAllocationSlice(value));
        slices[index] = slice;
      }
      return slices;
    };

    mlir::lmhlo::CustomCallTargetArgMappingAttr target_mapping =
        *custom_call.getTargetArgMapping();
    TF_ASSIGN_OR_RETURN(operands, values_to_slices_with_token_holes(
                                      custom_call.getArgs(),
                                      target_mapping.getArgsToTargetArgs(),
                                      target_mapping.getNumArgs()));
    TF_ASSIGN_OR_RETURN(results, values_to_slices_with_token_holes(
                                     custom_call.getOutput(),
                                     target_mapping.getResultsToTargetResults(),
                                     target_mapping.getNumResults()));
  } else {
    auto values_to_slices = [&](mlir::ValueRange values)
        -> StatusOr<std::vector<CustomCallThunk::OptionalSlice>> {
      std::vector<CustomCallThunk::OptionalSlice> slices;
      for (mlir::Value value : values) {
        TF_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                            GetAllocationSlice(value));
        slices.push_back(slice);
      }
      return slices;
    };

    TF_ASSIGN_OR_RETURN(operands, values_to_slices(custom_call.getArgs()));
    TF_ASSIGN_OR_RETURN(results, values_to_slices(custom_call.getOutput()));
  }

  CustomCallThunk::CustomCallTarget custom_call_target;

  // For information about this calling convention, see
  // xla/g3doc/custom_call.md.
  switch (custom_call.getApiVersion()) {
    case mlir::mhlo::CustomCallApiVersion::API_VERSION_ORIGINAL:
      using original_call_type =
          void (*)(CustomCallThunk::Stream /*stream*/, void** /*buffers*/,
                   const char* /*opaque*/, size_t /*opaque_len*/);
      custom_call_target = [call_target](CustomCallThunk::Stream stream,
                                         void** buffers, const char* opaque,
                                         size_t opaque_len,
                                         XlaCustomCallStatus*) {
        auto typed_call_target =
            reinterpret_cast<original_call_type>(call_target);
        typed_call_target(stream, buffers, opaque, opaque_len);
      };
      break;
    case mlir::mhlo::CustomCallApiVersion::API_VERSION_STATUS_RETURNING:
    case mlir::mhlo::CustomCallApiVersion::API_VERSION_STATUS_RETURNING_UNIFIED:
      using status_returning_call_type =
          void (*)(CustomCallThunk::Stream /*stream*/, void** /*buffers*/,
                   const char* /*opaque*/, size_t /*opaque_len*/,
                   XlaCustomCallStatus* /*status*/);
      custom_call_target =
          reinterpret_cast<status_returning_call_type>(call_target);
      break;
    case mlir::mhlo::CustomCallApiVersion::API_VERSION_TYPED_FFI:
      custom_call_target = [](CustomCallThunk::Stream, void**, const char*,
                              size_t, XlaCustomCallStatus*) {
        LOG(FATAL) << "Typed FFI custom call must be called by XLA runtime";
      };
      break;
    default:
      return InternalError("Unknown custom-call API version enum value: %d",
                           custom_call.getApiVersion());
  }

  // Thunks support only user-encoded string backend config.
  std::string backend_config;
  if (auto str = custom_call.getBackendConfig()
                     .value_or(mlir::Attribute())
                     .dyn_cast_or_null<mlir::StringAttr>()) {
    backend_config = str.str();
  }

  auto thunk = std::make_unique<CustomCallThunk>(
      GetThunkInfo(op), std::move(custom_call_target), std::move(operands),
      std::move(results), backend_config);
  AddThunkToThunkSequence(std::move(thunk));
  return OkStatus();
}

Status IrEmitterUnnested::EmitFftThunk(mlir::Operation* op) {
  auto fft_op = mlir::cast<mlir::lmhlo::FftOp>(op);
  const Shape operand_shape = GetShape(fft_op.getOperand());
  const Shape output_shape = GetShape(fft_op.getOutput());
  TF_RET_CHECK(LayoutUtil::IsMonotonicWithDim0Major(operand_shape.layout()));
  TF_RET_CHECK(LayoutUtil::IsMonotonicWithDim0Major(output_shape.layout()));

  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice arg_slice,
                      GetAllocationSlice(fft_op.getOperand()));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice dest_slice,
                      GetAllocationSlice(fft_op.getOutput()));
  TF_ASSIGN_OR_RETURN(
      xla::FftType fft_type,
      ConvertFftType(mlir::mhlo::stringifyFftType(fft_op.getFftType())));
  auto fft_length_values = fft_op.getFftLength().getValues<int64_t>();
  std::vector<int64_t> fft_length(fft_length_values.begin(),
                                  fft_length_values.end());

  AddThunkToThunkSequence(
      std::make_unique<FftThunk>(GetThunkInfo(op), fft_type, fft_length,
                                 /*input_buffer=*/arg_slice,
                                 /*output_buffer=*/dest_slice,
                                 /*input_shape=*/operand_shape,
                                 /*output_shape=*/output_shape));
  return OkStatus();
}

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
Status IrEmitterUnnested::EmitTriangularSolveCustomCall(mlir::Operation* op) {
  auto custom_call = mlir::cast<mlir::lmhlo::CustomCallOp>(op);

  auto operands = op->getOperands();
  TF_RET_CHECK(operands.size() == 4);

  // We expect Fortran layout for everything other than the temp buffer (the
  // last operand).  Fortran layout is not XLA default layout with elements 0
  // and 1 swapped.  For example instead of default layout {3,2,1,0} we'd have
  // Fortran layout {2,3,1,0}.
  TF_RET_CHECK(absl::c_all_of(operands.drop_back(1), [&](mlir::Value v) {
    const Shape& shape = GetShape(v);
    const Layout& layout = shape.layout();
    int n = layout.minor_to_major_size();
    if (n < 2) {
      return false;
    }
    // Unfortunately the HLO -> LMHLO -> HLO conversion loses layout information
    // if the shape has any dimensions of size 1: In that case, the new HLO
    // (which we see here) will have an arbitrary value for the location of the
    // size-1 dimension.  Just skip this assertion if the shape has any
    // degenerate dimensions.
    if (absl::c_any_of(shape.dimensions(),
                       [](int64_t dim) { return dim == 1; })) {
      return true;
    }
    return layout.minor_to_major(0) == n - 2 &&
           layout.minor_to_major(1) == n - 1 &&
           std::is_sorted(layout.minor_to_major().begin() + 2,
                          layout.minor_to_major().end(),
                          std::greater<int64_t>());
  }));

  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice a_slice,
                      GetAllocationSlice(operands[0]));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice b_slice,
                      GetAllocationSlice(operands[1]));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice result_slice,
                      GetAllocationSlice(operands[2]));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice temp_slice,
                      GetAllocationSlice(operands[3]));

  const Shape b_shape = GetShape(operands[1]);
  const PrimitiveType elem_ty = b_shape.element_type();

  TriangularSolveOptions backend_config;
  if (auto str = custom_call.getBackendConfig()
                     .value_or(mlir::Attribute())
                     .dyn_cast_or_null<mlir::StringAttr>())
    TF_RETURN_IF_ERROR(
        tsl::HumanReadableJsonToProto(str.str(), &backend_config));

  ThunkSequence thunks;

  // Triangular solve is in-place on 'b', so copy 'b' to the output if they
  // aren't the same buffer.
  if (b_slice != result_slice) {
    thunks.push_back(std::make_unique<DeviceToDeviceCopyThunk>(
        Thunk::ThunkInfo(op),
        /*source_buffer=*/b_slice,
        /*destination_buffer=*/result_slice,
        /*mem_size=*/ShapeUtil::ByteSizeOf(b_shape),
        /*source_value=*/operands[1],
        /*destination_value=*/operands[2]));
  }

  int64_t m = b_shape.dimensions(b_shape.rank() - 2);
  int64_t n = b_shape.dimensions(b_shape.rank() - 1);
  int64_t batch_size = std::accumulate(
      b_shape.dimensions().begin(), b_shape.dimensions().end() - 2, int64_t{1},
      [](int64_t a, int64_t b) { return a * b; });
  int64_t elem_size = ShapeUtil::ByteSizeOfPrimitiveType(elem_ty);
  int64_t a_batch_stride =
      backend_config.left_side() ? m * m * elem_size : n * n * elem_size;
  int64_t b_batch_stride = m * n * elem_size;
  thunks.push_back(std::make_unique<TriangularSolveThunk>(
      GetThunkInfo(op), backend_config,
      PtxOptsFromDebugOptions(hlo_module_config_.debug_options()),
      /*a_buffer=*/a_slice, /*b_buffer=*/result_slice, temp_slice, elem_ty,
      batch_size, m, n, a_batch_stride, b_batch_stride));

  // Elide the sequential thunk if there's no copy.
  if (thunks.size() == 1) {
    AddThunkToThunkSequence(std::move(thunks[0]));
  } else {
    AddThunkToThunkSequence(
        std::make_unique<SequentialThunk>(GetThunkInfo(op), std::move(thunks)));
  }
  return OkStatus();
}
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// Convert the following form of fusion region:
//   fusion() {
//     %0 = tensor_load %external_memref0
//     %1 = tensor_load %external_memref1
//     ...
//     tensor_store %ret, %external_memref2
//   }
// to
//   fusion(%external_memref0, %external_memref1) (^bb(%0, %1) {
//     ...
//     mhlo.return %ret
//   })
//
// So that it's suitable for MHLO -> XLA HLO conversion.
// This function won't be needed once ElementalIrEmitter migrates to take MHLO
// instead.
static Status ProcessFusionForConversion(mlir::Region* region,
                                         std::vector<Shape>* operand_shapes,
                                         std::vector<Shape>* output_shapes) {
  std::vector<mlir::bufferization::ToTensorOp> loads;
  std::vector<mlir::memref::TensorStoreOp> stores;

  region->walk([&](mlir::bufferization::ToTensorOp load) {
    if (load.getMemref().getParentRegion() != region) {
      loads.push_back(load);
    }
  });

  region->walk([&](mlir::memref::TensorStoreOp store) {
    if (store.getMemref().getParentRegion() != region) {
      stores.push_back(store);
    }
  });

  for (auto& load : loads) {
    auto arg = region->addArgument(load.getType(), region->getLoc());
    load.replaceAllUsesWith(arg);
    Shape shape = GetShape(load.getResult());
    operand_shapes->push_back(std::move(shape));
    load.erase();
  }

  std::vector<mlir::Value> returned_values;
  for (auto store : stores) {
    Shape shape = GetShape(store.getMemref());
    output_shapes->push_back(shape);

    returned_values.push_back(store.getTensor());
    store.erase();
  }

  region->back().back().erase();
  auto b = mlir::OpBuilder::atBlockEnd(&region->back());
  auto loc = returned_values[0].getLoc();
  b.create<mlir::mhlo::ReturnOp>(loc, returned_values);
  return OkStatus();
}

Status IrEmitterUnnested::EmitLaunchFunc(mlir::Operation* op) {
  auto launch_func = mlir::cast<mlir::gpu::LaunchFuncOp>(op);
  auto kernel_func =
      mlir::SymbolTable::lookupNearestSymbolFrom<mlir::LLVM::LLVMFuncOp>(
          launch_func, launch_func.getKernel());
  if (!kernel_func) {
    return InternalError("kernel '%s' not found",
                         launch_func.getKernelName().str());
  }

  // Lower kernel module to NVVM.
  auto gpu_module = kernel_func->getParentOfType<mlir::gpu::GPUModuleOp>();
  std::unique_ptr<llvm::Module> llvm_module = mlir::translateModuleToLLVMIR(
      gpu_module, module_->getContext(), gpu_module.getName());
  if (!llvm_module)
    return InternalError("Failed to translate GPU module to LLVM");

  // Add kernel to LLVM module.
  llvm_module->setDataLayout(module_->getDataLayout());
  llvm::Linker::linkModules(*module_, std::move(llvm_module));

  // Retrieve launch dimensions from arith.constant ops.
  auto get_dim3d = [](mlir::gpu::KernelDim3 dim3) {
    auto get_const = [](mlir::Value value) -> int64_t {
      auto const_op = value.getDefiningOp<mlir::arith::ConstantOp>();
      if (!const_op) return -1;
      auto attr = const_op.getValue().cast<mlir::IntegerAttr>();
      if (!attr) return -1;
      return attr.getValue().getSExtValue();
    };
    return LaunchDimensions::Dim3D{get_const(dim3.x), get_const(dim3.y),
                                   get_const(dim3.z)};
  };
  LaunchDimensions launch_dimensions(
      get_dim3d(launch_func.getGridSizeOperandValues()),
      get_dim3d(launch_func.getBlockSizeOperandValues()));

  // Create BufferSlice array from launch_func arguments, using the
  // attribute depicting which arguments are written by the kernel.
  std::vector<KernelArgument> kernel_arguments;
  unsigned num_kernel_operands = launch_func.getNumKernelOperands();
  kernel_arguments.reserve(num_kernel_operands);
  int i = 0;
  mlir::ArrayRef<mlir::Attribute> written_operands =
      mlir::getWrittenOperandsAttribute(launch_func).getValue();
  for (const auto& [operand, written] :
       llvm::zip_first(launch_func.getKernelOperands(),
                       written_operands.take_back(num_kernel_operands))) {
    auto& kernel_argument = kernel_arguments.emplace_back();
    kernel_argument.order = i++;
    kernel_argument.value = operand;
    auto& slice = kernel_argument.slice;
    TF_ASSIGN_OR_RETURN(slice.buffer_slice,
                        GetAllocationSlice(operand, &slice.constant_name));
    slice.shape = GetShape(operand);
    slice.written = written.cast<mlir::BoolAttr>().getValue();
  }

  // Add kernel prototype to module_, kernel thunk to thunk_sequence_.
  std::string kernel_name = GetIrNameFromLoc(launch_func.getLoc());
  TF_ASSIGN_OR_RETURN(
      std::vector<llvm_ir::IrArray> ir_arrays,
      BuildKernelThunkImpl(kernel_name, GetThunkInfo(op),
                           std::move(kernel_arguments), launch_dimensions));

  // Move function body into kernel prototype.
  llvm::Function* prototype_func = b_.GetInsertBlock()->getParent();
  llvm::Function* implementation_func =
      module_->getFunction(kernel_func.getName());
  prototype_func->splice(prototype_func->end(), implementation_func);
  for (const auto& [arg, ir_array] :
       llvm::zip_first(implementation_func->args(), ir_arrays)) {
    arg.replaceAllUsesWith(ir_array.GetBasePointer());
  }
  implementation_func->eraseFromParent();

  // Replace pre-existing return with unconditional branch to next block.
  llvm::Instruction* terminator =
      prototype_func->getEntryBlock().getTerminator();
  llvm::BranchInst::Create(&*std::next(prototype_func->begin()), terminator);
  terminator->eraseFromParent();

  return OkStatus();
}

#if GOOGLE_CUDA
Status IrEmitterUnnested::EmitTritonFusion(
    mlir::Operation* op, tensorflow::AutotuneResult::TritonGemmKey& config) {
  VLOG(3) << llvm_ir::DumpToString(op);
  auto fusion_op = mlir::cast<mlir::lmhlo::FusionOp>(op);

  TF_ASSIGN_OR_RETURN(
      const HloComputation* hlo_computation,
      GetOrCreateSubComputationFromRegion(&fusion_op->getRegion(0),
                                          /*is_fusion=*/false));

  const std::string fingerprint =
      hlo_computation->ToString(HloPrintOptions::Fingerprint()
                                    .set_print_only_essential_constants(false)
                                    .set_print_operand_shape(false));

  // TODO(tdanyluk): Consider removing this level of caching, because we already
  // cache the wrapper_fn now.
  auto cache_it = triton_cache_.find(fingerprint);
  llvm::Function* impl_fn;
  if (cache_it == triton_cache_.end()) {
    const std::string fn_name =
        ir_emitter_context_->name_uniquer()->GetUniqueName(
            llvm_ir::SanitizeFunctionName(
                fusion_op->getName().getStringRef().str()));
    const std::optional<LaunchDimensions> launch_dimensions = TritonWrapper(
        fn_name, hlo_computation,
        ir_emitter_context_->cuda_compute_capability(),
        ir_emitter_context_->gpu_device_info(), config, module_, &MatMul);
    TF_RET_CHECK(launch_dimensions.has_value());
    impl_fn = module_->getFunction(fn_name);
    TF_RET_CHECK(impl_fn);
    triton_cache_[fingerprint] =
        std::make_pair(impl_fn, launch_dimensions.value());
  } else {
    VLOG(10) << "Duplicate computation reused.";
    impl_fn = cache_it->second.first;
  }

  // Call the (cached) impl_fn from the wrapper_fn corresponding to this thunk.
  // Using BuildReusableKernelThunk actually speeds up the compilation
  // considerably, despite the caching of the impl_fn.
  TF_ASSIGN_OR_RETURN(
      std::optional<std::vector<llvm_ir::IrArray>> opt_ir_arrays,
      BuildReusableKernelThunk(
          fusion_op, /*launch_dimensions=*/triton_cache_[fingerprint].second));
  if (!opt_ir_arrays.has_value()) {
    // The kernel was reused, no need to emit code.
    return OkStatus();
  }
  std::vector<llvm_ir::IrArray>& ir_arrays = opt_ir_arrays.value();

  std::vector<llvm::Value*> args;
  args.reserve(ir_arrays.size());
  for (const llvm_ir::IrArray& a : ir_arrays) {
    args.push_back(a.GetBasePointer());
  }
  llvm::Function* wrapper_fn = b_.GetInsertBlock()->getParent();
  llvm::CallInst::Create(
      impl_fn, args, /*NameStr=*/"",
      /*InsertBefore=*/wrapper_fn->getEntryBlock().getTerminator());

  LogAndVerify(module_);

  return OkStatus();
}
#endif  // GOOGLE_CUDA

// TODO(timshen): update the comment once the HandleFusion code path deleted.
//
// This is migrated from IrEmitter::HandleFusion() with IrEmitterUnnested as the
// subclass. The logic is de-virtualized and less scattered.
Status IrEmitterUnnested::EmitLoopFusion(mlir::Operation* op) {
  auto fusion = mlir::cast<mlir::lmhlo::FusionOp>(op);

  TF_ASSIGN_OR_RETURN(const HloComputation* fused_computation,
                      GetOrCreateSubComputationFromRegion(&fusion.getRegion(),
                                                          /*is_fusion=*/true));

  const GpuDeviceInfo gpu_device_info = ir_emitter_context_->gpu_device_info();

  Shape element_shape = GetShape(fusion.getOutputBuffers()[0]);
  int unroll_factor = 1;
  {
    // Unrolling is good to read large inputs with small elements
    // due to vector loads, but increases the register pressure when one
    // thread has to produce multiple output elements.
    // Therefore for fusions with small outputs prefer to use one thread
    // per output element = no unroll.
    // Call 'small' fusions that use less threads than the GPU has.
    int64_t num_elements = ShapeUtil::ElementsIn(element_shape);
    int64_t n_threads_max =
        gpu_device_info.threads_per_core_limit * gpu_device_info.core_count;
    if (num_elements >= n_threads_max && !MayPreventVectorization(fusion)) {
      unroll_factor = ComputeMaxUnrollFactor(fusion, hlo_module_config_);
    }
  }
  VLOG(2) << "Unroll factor: " << unroll_factor;

  bool row_vectorized;
  int num_big_inputs;
  std::tie(row_vectorized, num_big_inputs) = RowVectorizationEnabled(fusion);
  bool few_waves = [fusion, row_vectorized, num_big_inputs]() mutable {
    for (mlir::Operation& op : fusion.getRegion().front()) {
      if (mlir::isa<mlir::bufferization::ToTensorOp,
                    mlir::memref::TensorStoreOp, mlir::lmhlo::TerminatorOp,
                    mlir::mhlo::ReturnOp, mlir::mhlo::ConstantOp>(op)) {
        continue;
      }
      HloOpcode opcode = *MhloToHloOpcode(&op);
      if (HloInstruction::IsOpElementwise(opcode)) {
        continue;
      }
      if (auto broadcast = mlir::dyn_cast<mlir::mhlo::BroadcastInDimOp>(op)) {
        if (broadcast.getBroadcastDimensions().empty() ||
            // More than 3 big inputs cause a speed regression.
            (row_vectorized && num_big_inputs <= 3)) {
          continue;
        }
      }
      VLOG(2) << "few_waves not enabled due to: " << llvm_ir::DumpToString(&op);
      return false;
    }
    return true;
  }();

  LaunchDimensionsConfig launch_config{unroll_factor, few_waves,
                                       row_vectorized};
  // Check that the shapes is supported.
  if (launch_config.row_vectorized &&
      ThreadsPerBlockRowVectorized(element_shape, gpu_device_info,
                                   launch_config) <= 0) {
    VLOG(2) << "Cancelling row_vectorization as the shape isn't supported.";
    launch_config.row_vectorized = false;
    launch_config.few_waves = false;
  }

  TF_ASSIGN_OR_RETURN(
      LaunchDimensions launch_dimensions,
      CalculateLaunchDimensions(element_shape, gpu_device_info, launch_config));

  TF_ASSIGN_OR_RETURN(
      std::optional<std::vector<llvm_ir::IrArray>> opt_ir_arrays,
      BuildReusableKernelThunk(fusion, launch_dimensions));
  if (!opt_ir_arrays.has_value()) {
    // The kernel was reused, no need to emit code.
    return OkStatus();
  }
  std::vector<llvm_ir::IrArray>& ir_arrays = opt_ir_arrays.value();

  absl::Span<llvm_ir::IrArray> operand_arrays =
      absl::MakeSpan(ir_arrays).subspan(0, fusion.getInputBuffers().size());
  absl::Span<llvm_ir::IrArray> output_element_arrays =
      absl::MakeSpan(ir_arrays).subspan(fusion.getInputBuffers().size(),
                                        fusion.getOutputBuffers().size());

  FusedIrEmitter fused_emitter(elemental_emitter_);

  for (int i = 0; i < fusion.getInputBuffers().size(); i++) {
    auto* builder = &b_;
    auto ir_array = operand_arrays[i];
    fused_emitter.BindGenerator(
        *fused_computation->parameter_instruction(i),
        [builder, ir_array](llvm_ir::IrArray::Index index) {
          return ir_array.EmitReadArrayElement(index, builder);
        });
  }
  TF_ASSIGN_OR_RETURN(
      auto element_generator,
      fused_emitter.GetGenerator(*fused_computation->root_instruction()));

  llvm::Type* index_type =
      GetIndexTypeForKernel(fusion, launch_dimensions.launch_bound(), &b_);

  TF_RETURN_IF_ERROR(
      ParallelLoopEmitter(element_generator, output_element_arrays,
                          launch_dimensions, &b_, launch_config)
          .EmitLoop(GetIrNameFromLoc(fusion->getLoc()), index_type));

  b_.SetInsertPoint(b_.GetInsertBlock()->getTerminator());
  return OkStatus();
}

Status IrEmitterUnnested::EmitUnnestedTranspose(
    mlir::lmhlo::FusionOp fusion, HloComputation* fused_computation) {
  std::vector<HloInstruction*> hlo_roots = GetFusionRoots(fused_computation);

  // TODO(cheshire): avoid duplication of FindTiledTranspose function, is it
  // possible?
  auto dims_and_order = FindAnyTiledTranspose(**absl::c_find_if(
      hlo_roots,
      [](HloInstruction* instr) { return FindAnyTiledTranspose(*instr); }));

  // TODO(cheshire): have a more robust way of checking this.
  CHECK(dims_and_order.has_value());

  constexpr int kNumRows = 4;
  CHECK_EQ(WarpSize() % kNumRows, 0);

  // 3D view over the input shape.
  Vector3 dims = dims_and_order->first;
  Vector3 order = dims_and_order->second;
  // We expect that the last dimension is swapped with a different dimension.
  CHECK_NE(order[2], 2);
  Vector3 permuted_dims = {dims[order[0]], dims[order[1]], dims[order[2]]};
  Vector3 tile_sizes{1, 1, 1};
  tile_sizes[order[2]] = WarpSize() / kNumRows;
  Vector3 num_threads{1, 1, WarpSize()};
  num_threads[order[2]] = kNumRows;

  TilingScheme tiling_scheme(
      /*permuted_dims*/ permuted_dims,
      /*tile_sizes=*/tile_sizes,
      /*num_threads=*/num_threads,
      /*indexing_order=*/kLinearIndexingX,
      /*vector_size=*/1,
      /*scaling_factor=*/1,
      /*tiling_dimensions=*/{order[2], 2});
  LaunchDimensions launch_dimensions(
      tiling_scheme.GetNumberOfBlocksPhysical(),
      tiling_scheme.GetNumThreadsPerBlockPhysical());

  TF_ASSIGN_OR_RETURN(
      std::optional<std::vector<llvm_ir::IrArray>> opt_ir_arrays,
      BuildReusableKernelThunk(fusion, launch_dimensions));
  if (!opt_ir_arrays.has_value()) {
    // The kernel was reused, no need to emit code.
    return OkStatus();
  }
  std::vector<llvm_ir::IrArray>& ir_arrays = opt_ir_arrays.value();

  TF_RETURN_IF_ERROR(EmitTransposeTile(
      fusion, fused_computation,
      absl::MakeSpan(ir_arrays).subspan(0, fusion.getInputBuffers().size()),
      absl::MakeSpan(ir_arrays).subspan(fusion.getInputBuffers().size()),
      tiling_scheme, launch_dimensions));
  return OkStatus();
}

Status IrEmitterUnnested::EmitFusion(mlir::Operation* op) {
  auto fusion_op = mlir::cast<mlir::lmhlo::FusionOp>(op);
  TF_ASSIGN_OR_RETURN(
      HloComputation * fused_computation,
      GetOrCreateSubComputationFromRegion(&fusion_op.getRegion(),
                                          /*is_fusion=*/true));

  if (HasAnyUnnestedReductionRoot(fused_computation)) {
    return EmitUnnestedReduction(fusion_op, fused_computation);
  }

  if (HasAnyTiledTransposeRoot(fused_computation)) {
    return EmitUnnestedTranspose(fusion_op, fused_computation);
  }

#if GOOGLE_CUDA
  if (auto backend_config = fusion_op.getBackendConfig()
                                .value_or(mlir::Attribute())
                                .dyn_cast_or_null<mlir::StringAttr>()) {
    tensorflow::AutotuneResult::TritonGemmKey triton_config;
    if (backend_config == kTritonGemmBackendConfig) {
      LOG(WARNING) << "Using fallback triton GEMM config";
      triton_config.set_block_m(64);
      triton_config.set_block_k(64);
      triton_config.set_block_n(64);
      triton_config.set_split_k(1);
      triton_config.set_num_stages(1);
      triton_config.set_num_warps(2);
      return EmitTritonFusion(fusion_op, triton_config);
    } else if (tsl::HumanReadableJsonToProto(backend_config.str(),
                                             &triton_config)
                   .ok()) {
      return EmitTritonFusion(fusion_op, triton_config);
    }
  }
#endif  // GOOGLE_CUDA

  auto fusion_results = fusion_op.getFusionResults();
  TF_RET_CHECK(!fusion_results.empty());
  if (fusion_results.size() > 1) {
    // In the case of root tuple, it can be either reduce or slice input
    // fusion.
    if (IsInputFusibleSlices(op, /*verify_no_strides=*/true)) {
      // The emitter doesn't support all cases. If it's not supported, fallback
      // to ElementalIrEmitter.
      auto status = EmitInputFusibleNonStridedSlices(op);
      if (status.code() == tsl::error::FAILED_PRECONDITION) {
        return EmitLoopFusion(op);
      }
      return status;
    }
  }

  mlir::Operation* fusion_root = fusion_results[0].getDefiningOp();
  if (mlir::isa<mlir::mhlo::ScatterOp>(fusion_root)) {
    return EmitScatter(fusion_op, fused_computation);
  }

  if (!IsSingleInstructionFusion(fusion_op) &&
      CanEmitFusedDynamicUpdateSliceInPlaceForGpu(
          fusion_op, ir_emitter_context_->allocations())) {
    return EmitDynamicUpdateSlice(fusion_op, fused_computation);
  }

  if (auto copy = mlir::dyn_cast<mlir::mhlo::CopyOp>(fusion_root);
      copy && IsSingleInstructionFusion(fusion_op)) {
    auto operands = GetHloOperands(fusion_op);
    auto outputs = GetHloOutputs(fusion_op);
    TF_RET_CHECK(operands.size() == 1);
    TF_RET_CHECK(outputs.size() == 1);
    auto operand_shape = GetShape(operands[0]);
    auto output_shape = GetShape(outputs[0]);

    CHECK(ShapeUtil::Compatible(operand_shape, output_shape));
    auto maybe_slice = GetAllocationSlice(operands[0]);
    if (LayoutUtil::Equal(operand_shape.layout(), output_shape.layout()) &&
        maybe_slice.ok()) {
      // Copy the operand into the output if it's not the same buffer already.
      auto operand_buffer = *maybe_slice;
      auto destination_buffer = *GetAllocationSlice(outputs[0]);
      if (operand_buffer != destination_buffer) {
        AddThunkToThunkSequence(std::make_unique<DeviceToDeviceCopyThunk>(
            GetThunkInfo(op),
            /*source_buffer=*/operand_buffer,
            /*destination_buffer=*/destination_buffer,
            /*mem_size=*/ByteSizeOf(operand_shape),
            /*source_value=*/operands[0],
            /*destination_value=*/outputs[0]));
      }
      return OkStatus();
    }
  }

  return EmitLoopFusion(op);
}

Status IrEmitterUnnested::EmitExtraOutputsForReduce(
    const Shape& reduction_operand_shape,
    const ReductionOutputMap& result_ir_arrays, const IrArray::Index& index,
    const ReductionCodegenInfo& reduction_info,
    const ExtraOutputGensMap& extra_output_gens) {
  if (extra_output_gens.empty()) {
    return OkStatus();
  }

  // Compute all extra output values before writing them. This avoids
  // overwriting aliased input/output buffers before all reads occurred.
  absl::flat_hash_map<const HloInstruction*, llvm::Value*>
      extra_output_ir_values;

  auto get_index = [&](const HloInstruction* instr) {
    const Shape& s = instr->shape();
    return ShapeUtil::EqualIgnoringElementType(reduction_operand_shape, s)
               ? index
               : index.SourceIndexOfBitcast(reduction_operand_shape, s, &b_);
  };

  for (const auto& [instr, generator] : extra_output_gens) {
    TF_ASSIGN_OR_RETURN(llvm::Value* const extra_output_ir_value,
                        generator(get_index(instr)));
    extra_output_ir_values[instr] = extra_output_ir_value;
  }

  for (const auto& [instr, generator] : extra_output_ir_values) {
    absl::Span<llvm_ir::IrArray const> result_ir = result_ir_arrays.at(instr);
    CHECK_EQ(result_ir.size(), 1);
    result_ir[0].EmitWriteArrayElement(
        get_index(instr), generator, &b_, /*use_linear_index=*/
        reduction_info.GetNumPartialResults() == 1);
  }
  return OkStatus();
}

Status IrEmitterUnnested::AssertNonDeterminismIsOkay(
    const std::string& op_name) {
  if (hlo_module_config_.debug_options().xla_gpu_deterministic_ops()) {
    return Unimplemented(
        "HLO instruction %s does not have a deterministic implementation, "
        "but run-to-run determinism is required by "
        "--xla_gpu_deterministic_ops.",
        op_name);
  }
  return OkStatus();
}

Status IrEmitterUnnested::EmitSelectAndScatter(mlir::Operation* op) {
  auto select_and_scatter_op = mlir::cast<mlir::lmhlo::SelectAndScatterOp>(op);

  const Shape source_shape = GetShape(select_and_scatter_op.getSource());
  const Shape operand_shape = GetShape(select_and_scatter_op.getOperand());
  const int64_t rank = operand_shape.rank();

  CHECK_EQ(rank, source_shape.rank());
  if (select_and_scatter_op.getWindowDimensions()) {
    CHECK_EQ(rank, select_and_scatter_op.getWindowDimensions()->size());
  }

  TF_RETURN_IF_ERROR(AssertNonDeterminismIsOkay(
      mlir::mhlo::GetDebugNameFromLocation(select_and_scatter_op.getLoc())));

  std::string name = GetIrNameFromLoc(select_and_scatter_op.getLoc());

  // IrEmitterUnnested implements kSelectAndScatter as a SequentialThunk
  // consisting of two thunks, an initializer KernelThunk that initializes
  // the output and another KernelThunk that accumulates the scattered
  // elements.
  TF_RETURN_IF_ERROR(BuildInitializerThunk(op,
                                           select_and_scatter_op.getInitValue(),
                                           select_and_scatter_op.getOut()));

  TF_ASSIGN_OR_RETURN(
      LaunchDimensions launch_dimensions,
      CalculateLaunchDimensions(source_shape,
                                ir_emitter_context_->gpu_device_info()));

  // Init value is not needed in IR emission.
  TF_ASSIGN_OR_RETURN(std::vector<llvm_ir::IrArray> ir_arrays,
                      BuildKernelThunk(select_and_scatter_op,
                                       {select_and_scatter_op.getOperand(),
                                        select_and_scatter_op.getSource(),
                                        select_and_scatter_op.getOut()},
                                       launch_dimensions));

  CHECK_EQ(ir_arrays.size(), 3);
  const IrArray& operand_array = ir_arrays[0];
  const IrArray& source_array = ir_arrays[1];
  const IrArray& out_array = ir_arrays[2];

  llvm::Type* index_type = GetIndexTypeForKernel(
      select_and_scatter_op, launch_dimensions.launch_bound(), &b_);
  auto index_typed_constant = [&](uint64_t c) -> llvm::Constant* {
    return llvm::ConstantInt::get(index_type, c);
  };

  // kSelectAndScatter is implemented as two kernel launches: the first launch
  // initializes the output array to the given initial value,
  // and the second accumulates the "source" matrix to the
  // selected elements in the output array. The first launch is already
  // implemented by the initializer thunk generated earlier, so this function
  // only needs to take care of the select-and-scatter part.
  //
  // Pseudo code for select-and-scatter:
  //
  // for (coordinates S in the source):  # This loop is parallel.
  //   initialized_flag = false
  //   for (coordinates W in the window):
  //     I = S * stride + W - pad_low
  //     if I within bounds of operand:
  //       if !(initialized_flag and select(selected_value, operand(I))):
  //         selected_value = operand(I)
  //         selected_index = I
  //         initialized_flag = true
  //   if initialized_flag:
  //     output(selected_index) = scatter(output(selected_index), source(S))
  auto loop_body_emitter = [&](const IrArray::Index& source_index) -> Status {
    // Allocate space to keep the currently selected value, its index, and a
    // boolean flag if the value is initialized. The initialized_flag is set
    // false.
    llvm::Value* selected_value_address = llvm_ir::EmitAllocaAtFunctionEntry(
        llvm_ir::PrimitiveTypeToIrType(operand_shape.element_type(), module_),
        "selected_value_address", &b_);

    llvm::AllocaInst* selected_index_address =
        llvm_ir::EmitAllocaAtFunctionEntryWithCount(
            index_type, index_typed_constant(rank), "selected_index_address",
            &b_);

    llvm::AllocaInst* initialized_flag_address =
        llvm_ir::EmitAllocaAtFunctionEntry(b_.getInt1Ty(),
                                           "initialized_flag_address", &b_);
    Store(b_.getInt1(false), initialized_flag_address);

    // Create the inner loop to iterate over the window.
    llvm_ir::ForLoopNest window_loops(absl::StrCat(name, "inner"), &b_,
                                      index_type);

    DimensionVector window_size;
    mlir::DenseIntElementsAttr window_dimensions =
        select_and_scatter_op.getWindowDimensions().value();
    for (const auto& dim : window_dimensions) {
      window_size.push_back(dim.getSExtValue());
      CHECK_GT(dim.getSExtValue(), 0);
    }

    const IrArray::Index window_index = window_loops.AddLoopsForShape(
        ShapeUtil::MakeShape(operand_shape.element_type(), window_size),
        "window");
    llvm_ir::SetToFirstInsertPoint(window_loops.GetInnerLoopBodyBasicBlock(),
                                   &b_);

    // Compute the operand index to visit and evaluate the condition whether the
    // operand index is within the bounds. The unsigned comparison includes
    // checking whether the operand index >= 0.
    std::vector<llvm::Value*> operand_multi_index(source_index.size());
    llvm::Value* in_bounds_condition = b_.getInt1(true);

    auto strides = *select_and_scatter_op.getWindowStrides();
    auto paddings = *select_and_scatter_op.getPadding();

    for (const auto& stride_and_padding :
         llvm::enumerate(llvm::zip(strides, paddings))) {
      const int i = stride_and_padding.index();
      int64_t stride = std::get<0>(stride_and_padding.value()).getSExtValue();
      int64_t padding = std::get<1>(stride_and_padding.value()).getSExtValue();

      llvm::Value* strided_index =
          NSWMul(source_index[i], index_typed_constant(stride));
      operand_multi_index[i] = NSWSub(NSWAdd(strided_index, window_index[i]),
                                      index_typed_constant(padding));
      llvm::Value* index_condition = ICmpULT(
          operand_multi_index[i],
          index_typed_constant(ShapeUtil::GetDimension(operand_shape, i)));
      in_bounds_condition = And(in_bounds_condition, index_condition);
    }

    // Only need to do something if the operand index is within the bounds.
    // First check if the initialized_flag is set.
    llvm_ir::LlvmIfData if_in_bounds =
        llvm_ir::EmitIfThenElse(in_bounds_condition, "in-bounds", &b_);
    llvm_ir::SetToFirstInsertPoint(if_in_bounds.true_block, &b_);
    llvm_ir::LlvmIfData if_initialized = llvm_ir::EmitIfThenElse(
        Load(initialized_flag_address->getAllocatedType(),
             initialized_flag_address),
        "initialized", &b_);

    // If the initialized_flag is false, initialize the selected value and index
    // with the currently visiting operand.
    llvm_ir::SetToFirstInsertPoint(if_initialized.false_block, &b_);
    const auto save_operand_index = [&](const IrArray::Index& operand_index) {
      for (int64_t i = 0; i < rank; ++i) {
        llvm::Value* selected_index_address_slot =
            InBoundsGEP(selected_index_address->getAllocatedType(),
                        selected_index_address, {b_.getInt32(i)});
        Store(operand_index[i], selected_index_address_slot);
      }
    };
    IrArray::Index operand_index(operand_multi_index, operand_shape,
                                 index_type);
    llvm::Value* operand_data =
        operand_array.EmitReadArrayElement(operand_index, &b_);
    Store(operand_data, selected_value_address);
    save_operand_index(operand_index);
    Store(b_.getInt1(true), initialized_flag_address);

    // If the initialized_flag is true, call the `select` function to
    // potentially update the selected value and index with the currently
    // visiting operand.
    llvm_ir::SetToFirstInsertPoint(if_initialized.true_block, &b_);
    llvm::Value* operand_address =
        operand_array.EmitArrayElementAddress(operand_index, &b_);
    llvm::AllocaInst* select_return_buffer = llvm_ir::EmitAllocaAtFunctionEntry(
        llvm_ir::PrimitiveTypeToIrType(PRED, module_), "select_return_buffer",
        &b_);

    TF_ASSIGN_OR_RETURN(
        const HloComputation* select_computation,
        GetOrCreateSubComputationFromRegion(&select_and_scatter_op.getSelect(),
                                            /*is_fusion=*/false));

    TF_RETURN_IF_ERROR(EmitCallToNestedComputation(
        *select_computation, {selected_value_address, operand_address},
        select_return_buffer));
    llvm::Value* result =
        Load(select_return_buffer->getAllocatedType(), select_return_buffer);

    // If the 'select' function returns false, update the selected value and the
    // index to the currently visiting operand.
    llvm::Value* cond =
        ICmpNE(result,
               llvm::ConstantInt::get(
                   llvm_ir::PrimitiveTypeToIrType(PRED, module_), 0),
               "boolean_predicate");
    llvm_ir::LlvmIfData if_select_lhs =
        llvm_ir::EmitIfThenElse(cond, "if-select-lhs", &b_);
    llvm_ir::SetToFirstInsertPoint(if_select_lhs.false_block, &b_);
    Store(Load(operand_array.GetElementLlvmType(), operand_address),
          selected_value_address);
    save_operand_index(operand_index);

    // If the initialized_flag is true, write to the selected index of the
    // output; otherwise the window is outside the source (in the padding) and
    // should be ignored.
    llvm_ir::SetToFirstInsertPoint(window_loops.GetOuterLoopExitBasicBlock(),
                                   &b_);
    llvm_ir::LlvmIfData if_should_store = llvm_ir::EmitIfThenElse(
        Load(initialized_flag_address->getAllocatedType(),
             initialized_flag_address),
        "should-store", &b_, /*emit_else=*/false);
    llvm_ir::SetToFirstInsertPoint(if_should_store.true_block, &b_);

    // After iterating over the window elements, scatter the source element to
    // the selected index of the output. The value we store at the output
    // location is computed by calling the `scatter` function with the source
    // value and the current output value.
    std::vector<llvm::Value*> selected_multi_index;
    for (int64_t i = 0; i < rank; ++i) {
      llvm::Value* selected_index_address_slot =
          InBoundsGEP(selected_index_address->getAllocatedType(),
                      selected_index_address, {b_.getInt32(i)});
      selected_multi_index.push_back(
          Load(selected_index_address->getAllocatedType(),
               selected_index_address_slot));
    }
    const Shape output_shape = GetShape(select_and_scatter_op.getOut());
    llvm::Value* source_value_address =
        source_array.EmitArrayElementAddress(source_index, &b_);
    IrArray::Index selected_index(selected_multi_index, output_shape,
                                  operand_index.GetType());
    llvm::Value* output_value_address =
        out_array.EmitArrayElementAddress(selected_index, &b_);

    TF_ASSIGN_OR_RETURN(
        const HloComputation* scatter_computation,
        GetOrCreateSubComputationFromRegion(&select_and_scatter_op.getScatter(),
                                            /*is_fusion=*/false));

    return EmitAtomicOperationForNestedComputation(
        *scatter_computation, output_value_address, source_value_address,
        source_array.GetElementLlvmType());
  };

  return ParallelLoopEmitter(loop_body_emitter, source_shape, launch_dimensions,
                             &b_)
      .EmitLoop(name, index_type);
}

Status IrEmitterUnnested::EmitWhile(mlir::Operation* op) {
  auto while_op = mlir::cast<mlir::lmhlo::WhileOp>(op);

  auto cond_result = GetHloOutputs(while_op);
  TF_RET_CHECK(cond_result.size() == 1);
  TF_RET_CHECK(cond_result[0]
                   .getType()
                   .cast<mlir::ShapedType>()
                   .getElementType()
                   .isInteger(/*width=*/1))
      << "While condition computation must return bool";

  // Build ForThunk for conformant while loops, otherwise build WhileThunk.
  //
  // If Xla runtime is enabled we always lower to `lmhlo.while` operation and
  // rely on `lmhlo-to-gpu-runtime` to lower while loops with known trip counts
  // to `scf.for` loops.
  if (while_op.getTripCount() &&
      !IsXlaRuntimeExecutableEnabled(hlo_module_config_)) {
    TF_ASSIGN_OR_RETURN(auto thunk, BuildForThunk(while_op, GetThunkInfo(op),
                                                  *while_op.getTripCount()));
    AddThunkToThunkSequence(std::move(thunk));
  } else {
    TF_ASSIGN_OR_RETURN(auto thunk,
                        BuildWhileThunk(while_op, GetThunkInfo(op)));
    AddThunkToThunkSequence(std::move(thunk));
  }
  return OkStatus();
}

Status IrEmitterUnnested::EmitRngGetAndUpdateState(mlir::Operation* op) {
  auto rng_op = mlir::dyn_cast<mlir::lmhlo::RngGetAndUpdateStateOp>(op);

  // Emit a kernel to increment the global state for Philox RNG algorithm.
  TF_ASSIGN_OR_RETURN(
      std::vector<llvm_ir::IrArray> ir_arrays,
      BuildKernelThunk(rng_op, rng_op.getState(), LaunchDimensions()));

  llvm::Value* old_state =
      llvm_ir::RngGetAndUpdateState(rng_op.getDelta(), module_, &b_);

  const Shape shape = GetShape(rng_op.getState());

  llvm::Value* output_address = ir_arrays[0].EmitArrayElementAddress(
      llvm_ir::IrArray::Index(
          /*linear=*/b_.getInt64(0), shape, &b_),
      &b_, "rng_state_address");
  output_address = BitCast(
      output_address, llvm::PointerType::get(
                          old_state->getType(),
                          output_address->getType()->getPointerAddressSpace()));
  Store(old_state, output_address);

  return OkStatus();
}

Status IrEmitterUnnested::EmitScatter(mlir::Operation* op) {
  auto scatter_op = mlir::cast<mlir::lmhlo::ScatterOp>(op);

  TF_ASSIGN_OR_RETURN(auto operand_buffer,
                      GetAllocationSlice(scatter_op.getOperand()));
  TF_ASSIGN_OR_RETURN(auto output_buffer,
                      GetAllocationSlice(scatter_op.getOutput()));

  // Copy the operand into the output if it's not the same buffer already.
  if (operand_buffer != output_buffer) {
    AddThunkToThunkSequence(std::make_unique<DeviceToDeviceCopyThunk>(
        Thunk::ThunkInfo(op),
        /*source_buffer=*/operand_buffer,
        /*destination_buffer=*/output_buffer,
        /*mem_size=*/ShapeUtil::ByteSizeOf(GetShape(scatter_op.getOutput())),
        /*source_value=*/scatter_op.getOperand(),
        /*destination_value=*/scatter_op.getOutput()));
  }

  const Shape& data_shape = GetShape(scatter_op.getUpdates());
  TF_ASSIGN_OR_RETURN(LaunchDimensions launch_dimensions,
                      CalculateLaunchDimensions(
                          data_shape, ir_emitter_context_->gpu_device_info()));

  // Create kernel thunk for all operands except the first one (`operand`). The
  // code generated for scatter below assumes that the input operand is already
  // copied into the output, so does not use it in codegen.
  TF_ASSIGN_OR_RETURN(
      std::vector<llvm_ir::IrArray> ir_arrays,
      BuildKernelThunk(scatter_op, scatter_op.getOperands().drop_front(),
                       launch_dimensions));

  CHECK_EQ(ir_arrays.size(), 3);
  const IrArray& scatter_indices = ir_arrays[0];
  const IrArray& updates = ir_arrays[1];
  const IrArray& output = ir_arrays[2];

  auto get_index_type = [&](int64_t launch_size) {
    return GetIndexTypeForKernel(scatter_op, launch_size, &b_);
  };

  TF_RETURN_IF_ERROR(EmitScatter(
      scatter_op, launch_dimensions, output,
      /*scatter_indices_gen=*/
      [&](const IrArray::Index& index) {
        return scatter_indices.EmitReadArrayElement(index, &b_,
                                                    "scatter_index");
      },
      /*updates_gen=*/
      [&](const IrArray::Index& index) {
        return updates.EmitReadArrayElement(index, &b_, "update");
      },
      /* get_index_type=*/
      get_index_type));

  return OkStatus();
}

Status IrEmitterUnnested::EmitScatter(
    mlir::lmhlo::ScatterOp scatter, const LaunchDimensions& launch_dimensions,
    const llvm_ir::IrArray& output,
    const llvm_ir::ElementGenerator& scatter_indices_gen,
    const llvm_ir::ElementGenerator& updates_gen,
    std::function<llvm::Type*(int64_t)> get_index_type) {
  const Shape operand_shape = GetShape(scatter.getOperand());
  CHECK(ShapeUtil::Equal(GetShape(scatter.getOutput()), operand_shape));

  TF_ASSIGN_OR_RETURN(
      const HloComputation* update_computation,
      GetOrCreateSubComputationFromRegion(&scatter.getUpdateComputation(),
                                          /*is_fusion=*/false));

  ScatterDescriptor desc;
  desc.name = GetIrNameFromLoc(scatter.getLoc());
  desc.operand_shape = operand_shape;
  desc.scatter_indices_shape = GetShape(scatter.getScatterIndices());
  desc.updates_shape = GetShape(scatter.getUpdates());
  desc.dim_numbers = scatter.getScatterDimensionNumbers();
  desc.unique_indices = scatter.getUniqueIndices();
  desc.update_computation = update_computation;
  desc.output = output;
  desc.scatter_indices_gen = scatter_indices_gen;
  desc.updates_gen = updates_gen;
  desc.get_index_type = get_index_type;
  return EmitScatter(desc, launch_dimensions);
}

Status IrEmitterUnnested::EmitScatter(
    const ScatterDescriptor& desc, const LaunchDimensions& launch_dimensions) {
  auto loop_body_emitter = [&](const IrArray::Index& index) -> Status {
    std::vector<llvm::Value*> raw_window_multidim;
    std::vector<llvm::Value*> input_scatter_multidim;
    std::vector<int64_t> raw_window_bounds;

    // Partition the index into window indices and scatter indices.
    for (int64_t i = 0, e = index.size(); i != e; ++i) {
      // For window indices also remember the window size, this comes in handy
      // later.
      if (llvm::is_contained(desc.dim_numbers.getUpdateWindowDims(), i)) {
        raw_window_multidim.push_back(index[i]);
        raw_window_bounds.push_back(desc.updates_shape.dimensions(i));
      } else {
        input_scatter_multidim.push_back(index[i]);
      }
    }
    DCHECK_EQ(raw_window_multidim.size(),
              desc.dim_numbers.getUpdateWindowDims().size());

    // Apply inserted_window_dims to the window dimensions.
    int64_t raw_window_multidim_idx = 0;
    llvm::SmallVector<llvm::Value*> input_window_multidim;
    llvm::SmallVector<int64_t> input_window_bounds;
    const int64_t rank = desc.operand_shape.rank();
    input_window_bounds.reserve(rank);
    input_window_multidim.reserve(rank);

    for (int64_t i = 0; i != rank; ++i) {
      if (llvm::is_contained(desc.dim_numbers.getInsertedWindowDims(), i)) {
        input_window_bounds.push_back(1);  // Trivial dimension.
        input_window_multidim.push_back(index.GetConstantWithIndexType(0));
      } else {
        input_window_bounds.push_back(
            raw_window_bounds[raw_window_multidim_idx]);
        input_window_multidim.push_back(
            raw_window_multidim[raw_window_multidim_idx]);
        ++raw_window_multidim_idx;
      }
    }
    DCHECK_EQ(input_window_multidim.size(), desc.operand_shape.rank());

    // Insert a 1 dimension at the end if index_vector_dim requests one.
    Shape scatter_indices_shape_fixed = desc.scatter_indices_shape;
    if (desc.dim_numbers.getIndexVectorDim() ==
        desc.scatter_indices_shape.rank()) {
      scatter_indices_shape_fixed.add_dimensions(1);
      scatter_indices_shape_fixed.mutable_layout()->add_minor_to_major(
          desc.dim_numbers.getIndexVectorDim());
    }

    // Now load the indices corresponding to the current window from
    // scatter_indices.
    std::vector<llvm::Value*> raw_scatter_index_multidim =
        input_scatter_multidim;
    raw_scatter_index_multidim.insert(raw_scatter_index_multidim.begin() +
                                          desc.dim_numbers.getIndexVectorDim(),
                                      nullptr);
    llvm::Value* is_in_bounds = b_.getTrue();
    for (int64_t i = 0,
                 e = desc.dim_numbers.getScatterDimsToOperandDims().size();
         i != e; ++i) {
      // Our index is stored along index_vector_dim, insert that into the lookup
      // index into scatter_indices.
      raw_scatter_index_multidim[desc.dim_numbers.getIndexVectorDim()] =
          index.GetConstantWithIndexType(i);
      llvm_ir::IrArray::Index raw_scatter_index_index(
          raw_scatter_index_multidim, scatter_indices_shape_fixed,
          index.GetType());

      int64_t operand_dim = desc.dim_numbers.getScatterDimsToOperandDims()[i];
      TF_ASSIGN_OR_RETURN(
          llvm::Value* const loaded_scatter_index,
          desc.scatter_indices_gen(raw_scatter_index_index.SourceIndexOfReshape(
              scatter_indices_shape_fixed, desc.scatter_indices_shape, &b_)));
      // And add the index to our window index. This yields the output index.
      llvm::Value* casted_scatter_index =
          IntCast(loaded_scatter_index, index.GetType(),
                  /*isSigned=*/true);
      llvm::Value* dim_offset =
          Add(input_window_multidim[operand_dim], casted_scatter_index);
      input_window_multidim[operand_dim] = dim_offset;

      // Also do the bounds check now.
      int64_t max_index = desc.operand_shape.dimensions(operand_dim) -
                          input_window_bounds[operand_dim] + 1;
      // is_in_bounds = index >= 0 && index < dim_size-window_size+1
      //   --> index u< dim_size-window_size+1
      is_in_bounds =
          And(is_in_bounds, ICmpULT(casted_scatter_index,
                                    index.GetConstantWithIndexType(max_index)));
    }

    llvm_ir::LlvmIfData if_window_in_bounds_data = llvm_ir::EmitIfThenElse(
        is_in_bounds, "scatter.in_bounds", &b_, /*emit_else=*/false);
    llvm_ir::SetToFirstInsertPoint(if_window_in_bounds_data.true_block, &b_);
    // All done, now just read from the calculated input from the window, and do
    // an atomic store to the calculated location in the output.
    llvm_ir::IrArray::Index input_window_index(
        input_window_multidim, desc.output.GetShape(), index.GetType());
    llvm::Value* output_address =
        desc.output.EmitArrayElementAddress(input_window_index, &b_);
    llvm::Value* input_address = llvm_ir::EmitAllocaAtFunctionEntry(
        llvm_ir::PrimitiveTypeToIrType(desc.updates_shape.element_type(),
                                       module_),
        "input_address", &b_);
    TF_ASSIGN_OR_RETURN(llvm::Value* const input_ir_value,
                        desc.updates_gen(index));
    Store(input_ir_value, input_address);

    if (!desc.unique_indices) {
      return EmitAtomicOperationForNestedComputation(
          *desc.update_computation, output_address, input_address,
          desc.output.GetElementLlvmType());
    } else {
      return EmitCallToNestedComputation(*desc.update_computation,
                                         {output_address, input_address},
                                         output_address);
    }
  };

  // Launch a kernel that reads every element in the updates tensor. We could
  // also do one kernel per window instead if bounds checks turn out to be a
  // bottleneck.
  return ParallelLoopEmitter(loop_body_emitter, desc.updates_shape,
                             launch_dimensions, &b_)
      .EmitLoop(desc.name,
                desc.get_index_type(launch_dimensions.launch_bound()));
}

// This transformation should be migrated off. See b/171334474.
StatusOr<HloComputation*>
IrEmitterUnnested::GetOrCreateSubComputationFromRegion(mlir::Region* region,
                                                       bool is_fusion) {
  std::unique_ptr<HloModule>& module = scratch_nested_computations_[region];
  if (module == nullptr) {
    std::vector<Shape> operand_shapes, output_shapes;
    if (is_fusion) {
      mlir::Operation* clone = region->getParentOp()->clone();
      region = &mlir::cast<mlir::lmhlo::FusionOp>(clone).getRegion();
      TF_RETURN_IF_ERROR(
          ProcessFusionForConversion(region, &operand_shapes, &output_shapes));
    }

    xla::XlaComputation xla_computation;
    mlir::MlirToHloConversionOptions options;
    options.propagate_layouts = true;
    options.propagate_bitcast_layouts_to_backend_config = true;
    TF_RETURN_IF_ERROR(
        ConvertRegionToComputation(region, &xla_computation, options));

    if (is_fusion) {
      region->getParentOp()->erase();
    }

    TF_ASSIGN_OR_RETURN(auto program_shape, xla_computation.GetProgramShape());
    TF_ASSIGN_OR_RETURN(
        module, HloModule::CreateFromProto(xla_computation.proto(),
                                           HloModuleConfig(program_shape)));

    if (is_fusion) {
      HloComputation* fused_computation = module->entry_computation();

      CHECK_EQ(operand_shapes.size(), fused_computation->num_parameters());
      for (int i = 0; i < fused_computation->num_parameters(); i++) {
        *fused_computation->parameter_instruction(i)
             ->mutable_shape()
             ->mutable_layout() = operand_shapes[i].layout();
      }
      HloInstruction* root = fused_computation->root_instruction();
      // Manually fold Tuple(GTE(a, 0), GTE(a, 1), GTE(a, 2), ...) to a.
      // FusedIrEmitter doesn't take GTE ops because we aim to elimiate tuples
      // as much as possible.
      if (root->opcode() == HloOpcode::kTuple) {
        [&] {
          HloInstruction* real_root = nullptr;
          int expected_tuple_index = 0;
          for (HloInstruction* operand : root->operands()) {
            if (operand->opcode() != HloOpcode::kGetTupleElement) {
              return;
            }
            if (real_root == nullptr) {
              real_root = operand->mutable_operand(0);
            } else if (real_root != operand->operand(0)) {
              return;
            }
            if (expected_tuple_index != operand->tuple_index()) {
              return;
            }
            expected_tuple_index++;
          }
          fused_computation->set_root_instruction(real_root);
          std::vector<HloInstruction*> to_be_removed;
          to_be_removed.push_back(root);
          for (HloInstruction* operand : root->operands()) {
            to_be_removed.push_back(operand);
          }
          for (auto instr : to_be_removed) {
            TF_CHECK_OK(fused_computation->RemoveInstruction(instr));
          }

          root = real_root;
        }();
      }

      if (output_shapes.size() > 1) {
        CHECK(root->shape().IsTuple());
        CHECK_EQ(root->shape().tuple_shapes_size(), output_shapes.size());

        for (int i = 0; i < output_shapes.size(); i++) {
          *root->mutable_shape()->mutable_tuple_shapes(i) = output_shapes.at(i);
        }
      } else {
        CHECK_EQ(1, output_shapes.size());
        *root->mutable_shape() = output_shapes[0];
      }
    }
    // Post-process the generated computation:
    // * Sanitize constant names, so that they can be used as LLVM global
    // symbols.
    // * Propagate layouts for tuple types.
    for (HloComputation* computation : module->computations()) {
      for (HloInstruction* instr : computation->MakeInstructionPostOrder()) {
        if (instr->opcode() == HloOpcode::kConstant) {
          // Notice that IR emitters use the name of constants as LLVM symbol
          // names, therefore it's important to not let these constants in the
          // new module collide with constants in the original module by names.
          // Unique them by prepending the module name.
          //
          // TODO(timshen): A better solution would be to plumb the exact
          // constant names through original HLO -> LHLO -> MHLO -> HLO. This is
          // hard because XLA builder doesn't support setting names. Revisit
          // this once we get rid of this function, or don't rely on the op name
          // (which shouldn't be the identity) to generate LLVM symbols.
          instr->SetAndSanitizeName(llvm_ir::SanitizeConstantName(
              module->name() + "_" + instr->name()));
        }
      }
    }
  }
  return module->entry_computation();
}

Status IrEmitterUnnested::EmitSort(mlir::Operation* op) {
  auto sort_op = mlir::cast<mlir::lmhlo::SortOp>(op);

  std::string op_name = GetIrNameFromLoc(sort_op.getLoc());
  llvm::SmallVector<mlir::Value> operands = GetHloOperands(sort_op);
  const Shape& keys_shape = GetShape(operands[0]);
  int64_t dimension_to_sort = sort_op.getDimension();
  for (int64_t i = 0; i < operands.size(); ++i) {
    // We assume that the layout of all involved operands and outputs is the
    // same.
    TF_RET_CHECK(
        LayoutUtil::LayoutsInShapesEqual(keys_shape, GetShape(operands[i])));
    TF_RET_CHECK(LayoutUtil::LayoutsInShapesEqual(
        keys_shape, GetShape(GetHloOutputs(sort_op)[i])));

    // If possible, we share buffers. If that is not possible, we need to copy
    // the values, because the emitter does the sorting in-place.
    TF_ASSIGN_OR_RETURN(auto destination_buffer,
                        GetAllocationSlice(sort_op.getOutput()[i]));
    TF_ASSIGN_OR_RETURN(auto source_address,
                        GetAllocationSlice(sort_op.getOperands()[i]));
    if (destination_buffer != source_address) {
      // TODO(b/26783907): Figure out why we never seem to share buffers for
      // key/value sort.
      VLOG(2) << op_name << " requires initial D2D copy for operand " << i;
      AddThunkToThunkSequence(std::make_unique<DeviceToDeviceCopyThunk>(
          Thunk::ThunkInfo(op),
          /*source_buffer=*/source_address,
          /*destination_buffer=*/destination_buffer,
          /*mem_size=*/ShapeUtil::ByteSizeOf(GetShape(operands[i])),
          /*source_value=*/sort_op.getOperands()[i],
          /*destination_value=*/sort_op.getOutput()[i]));
    }
  }

  uint64_t dimension_to_sort_bound = keys_shape.dimensions(dimension_to_sort);
  int64_t num_stages = Log2Ceiling(dimension_to_sort_bound);
  VLOG(2) << op_name << " requires " << num_stages << " stages.";
  CHECK_GE(1ULL << num_stages, dimension_to_sort_bound);
  CHECK_LT(1ULL << (num_stages - 1), dimension_to_sort_bound);

  // Naive C++ code for the outer loops:
  //
  // for (int64_t stage = 0; stage < Log2Ceiling(dimension_to_sort_bound);
  //     ++stage) {
  //   int64_t first_xor_mask = (1LL << (stage + 1)) - 1;
  //   SortInPlace(first_xor_mask);
  //   for (int64_t mask = stage - 1; mask >= 0; --mask) {
  //     int64_t later_xor_mask = 1LL << mask;
  //     SortInPlace(later_xor_mask);
  //   }
  // }
  //
  // This follows the alternative representation of the algorithm described on
  // Wikipedia: https://en.wikipedia.org/wiki/Bitonic_sorter
  //
  // Each mask specifies how to derive from one position in the array the
  // position with which it should be compared (we calculate the xor of the
  // position with the mask).
  // As an optimization, we can move the 'mask' loop to inside the
  // sorting/comparison loop if the comparisons happen within a small block of
  // the array. To make this work, we collect all consecutive masks that are
  // smaller than our chosen power of 2 tile size, and pass them to SortInPlace.
  // Each thread then processes one tile of data.

  const uint64_t kTileSize = std::min(2048ULL, 1ULL << num_stages);

  // If we cannot combine several xor masks together, we don't use tiling, so we
  // calculate the standard launch dimensions for the shape. However we only
  // need to iterate through ~half of the dimension to sort (rounded up to the
  // next highest power of 2), because each iteration compares one pair of
  // elements.
  Shape standard_iteration_shape = keys_shape;
  uint64_t standard_num_iterations_in_sort_dim = 1ULL << (num_stages - 1);
  standard_iteration_shape.set_dimensions(dimension_to_sort,
                                          standard_num_iterations_in_sort_dim);
  TF_ASSIGN_OR_RETURN(
      LaunchDimensions standard_launch_dimensions,
      CalculateLaunchDimensions(standard_iteration_shape,
                                ir_emitter_context_->gpu_device_info()));

  // Calculate the launch dimensions for the case where we use tiling. We split
  // the dimension that should be sorted into tiles of size 'kTileSize'. This
  // means we first need to round 'dimension_to_sort_bound' up to be a multiple
  // of the tile size.
  int64_t rounded_bound = RoundUpTo(dimension_to_sort_bound, kTileSize);
  Shape iteration_shape = keys_shape;

  // We iterate through the element pairs that should be compared.
  uint64_t num_iterations_in_sort_dim = rounded_bound / 2;
  iteration_shape.set_dimensions(dimension_to_sort, num_iterations_in_sort_dim);
  uint64_t num_iterations = ShapeUtil::ElementsIn(iteration_shape);

  // For correctness reasons we need exactly 'kTileSize' / 2 many threads per
  // block. Each thread is responsible for copying exactly two adjacent elements
  // into shared memory, and then does a comparison of two possibly different
  // elements taken from shared memory.
  const uint64_t kThreadsPerBlock = kTileSize / 2;

  // Check whether we should use any tiling. We might not be able to use it if
  // we have not enough threads, or not enough shared memory.
  int64_t total_shared_memory_needed = 0;
  for (int64_t i = 0; i < operands.size(); ++i) {
    total_shared_memory_needed +=
        kTileSize * ShapeUtil::ByteSizeOfPrimitiveType(
                        GetShape(operands[i]).element_type());
  }
  bool no_tiling =
      kThreadsPerBlock >
          ir_emitter_context_->gpu_device_info().threads_per_block_limit ||
      total_shared_memory_needed >
          ir_emitter_context_->gpu_device_info().shared_memory_per_block;
  VLOG(2) << absl::StreamFormat(
      "%s %s use tiling. No tiling if any of the following is true: "
      "kThreadsPerBlock=%d > threads_per_block_limit=%d, "
      "total_shared_memory_needed=%d > shared_memory_per_block=%d",
      op_name, (no_tiling ? "won't" : "will"), kThreadsPerBlock,
      ir_emitter_context_->gpu_device_info().threads_per_block_limit,
      total_shared_memory_needed,
      ir_emitter_context_->gpu_device_info().shared_memory_per_block);

  uint64_t num_blocks = CeilOfRatio(num_iterations, kThreadsPerBlock);
  LaunchDimensions tiled_launch_dimensions(num_blocks, kThreadsPerBlock);
  VLOG(2) << absl::StreamFormat("%s launch dims: %d blocks, %d threads/block",
                                op_name, num_blocks, kThreadsPerBlock);
  auto emit_kernel = [&](absl::Span<const int64_t> xor_masks) {
    VLOG(2) << absl::StreamFormat(
        "%s uses kernel for xor masks [%s]", op_name,
        absl::StrJoin(xor_masks, ", ", [](std::string* out, int64_t xor_mask) {
          absl::StrAppendFormat(out, "0x%x", xor_mask);
        }));
    LaunchDimensions launch_dimensions = xor_masks.size() > 1
                                             ? tiled_launch_dimensions
                                             : standard_launch_dimensions;
    TF_ASSIGN_OR_RETURN(
        std::vector<llvm_ir::IrArray> ir_arrays,
        BuildKernelThunk(sort_op, sort_op.getOutput(), launch_dimensions));
    std::vector<IrArray> values_arrays;
    values_arrays.reserve(operands.size());
    for (int64_t i = 0; i < operands.size(); ++i) {
      values_arrays.push_back(ir_arrays[i]);
    }
    TF_ASSIGN_OR_RETURN(const HloComputation* comparator,
                        GetOrCreateSubComputationFromRegion(
                            &sort_op.getComparator(), /*is_fusion=*/false));
    return llvm_ir::EmitSortInPlace(
        dimension_to_sort, values_arrays, IrName(op_name), xor_masks, &b_,
        launch_dimensions,
        xor_masks.size() > 1 ? num_iterations_in_sort_dim
                             : standard_num_iterations_in_sort_dim,
        kTileSize,
        [&](absl::Span<llvm::Value* const> operands, llvm::Value* output) {
          return EmitCallToNestedComputation(*comparator, operands, output);
        });
  };
  std::vector<int64_t> xor_masks;
  for (int64_t stage = 0; stage < num_stages; ++stage) {
    for (int64_t mask = stage; mask >= 0; --mask) {
      int64_t xor_mask;
      if (mask == stage) {
        xor_mask = (1LL << (stage + 1)) - 1;
      } else {
        xor_mask = 1LL << mask;
      }
      if (xor_mask >= kTileSize || no_tiling) {
        if (!xor_masks.empty()) {
          TF_RETURN_IF_ERROR(emit_kernel(xor_masks));
          xor_masks.clear();
        }
        TF_RETURN_IF_ERROR(emit_kernel({xor_mask}));
      } else {
        xor_masks.push_back(xor_mask);
      }
    }
  }
  if (!xor_masks.empty()) {
    TF_RETURN_IF_ERROR(emit_kernel(xor_masks));
  }
  return OkStatus();
}

template <typename ThunkType, typename OpT>
Status IrEmitterUnnested::EmitReplicaOrPartitionId(mlir::Operation* op) {
  auto casted = mlir::cast<OpT>(op);
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice result_slice,
                      GetAllocationSlice(casted.getOperand()));
  auto thunk = std::make_unique<ThunkType>(GetThunkInfo(op), result_slice);
  AddThunkToThunkSequence(std::move(thunk));
  return OkStatus();
}

template <typename NcclThunkType, typename OpT>
Status IrEmitterUnnested::EmitCollectivePermute(mlir::Operation* op) {
  auto collective_permute_op = mlir::cast<OpT>(op);

  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice source_slice,
                      GetAllocationSlice(collective_permute_op.getOperand()));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice result_slice,
                      GetAllocationSlice(collective_permute_op.getOutput()));

  const Shape shape = GetShape(collective_permute_op.getOperand());
  const int64_t replica_count = hlo_module_config_.replica_count();
  const int64_t partition_count = hlo_module_config_.num_partitions();

  NcclCollectiveThunk::AsyncExecutor* async_executor = nullptr;
  if (NcclThunkType::IsDegenerate(collective_permute_op, replica_count,
                                  partition_count)) {
    // For a degenerate collective permute, just generate a copy thunk.
    AddThunkToThunkSequence(std::make_unique<DeviceToDeviceCopyThunk>(
        GetThunkInfo(op),
        /*source_buffer=*/source_slice,
        /*destination_buffer=*/result_slice,
        /*mem_size=*/ShapeUtil::ByteSizeOf(shape),
        /*source_value=*/collective_permute_op.getOperand(),
        /*destination_value=*/collective_permute_op.getOutput()));
  } else {
    const NcclCollectiveThunk::Buffer buffer = {
        /*element_count=*/ShapeUtil::ElementsIn(shape),
        /*source_buffer=*/source_slice,
        /*destination_buffer=*/result_slice};
    auto thunk =
        std::make_unique<NcclThunkType>(GetThunkInfo(op), collective_permute_op,
                                        replica_count, partition_count, buffer);
    if constexpr (NcclThunkType::IsAsync()) {
      async_executor = &thunk->async_executor();
    }
    AddThunkToThunkSequence(std::move(thunk));
  }

  // Signal that start thunk not created with nullptr.
  if constexpr (NcclThunkType::IsAsync()) {
    async_executors_.insert({op, async_executor});
  }
  return OkStatus();
}

template <typename NcclThunkType, typename OpT>
Status IrEmitterUnnested::EmitNcclThunk(mlir::Operation* untyped_op) {
  OpT op = mlir::cast<OpT>(untyped_op);
  int64_t replica_count = hlo_module_config_.replica_count();
  int64_t partition_count = hlo_module_config_.num_partitions();
  VLOG(2) << NcclThunkType::GetName() << "; replica count: " << replica_count
          << "; partition count: " << partition_count
          << "; operand count: " << op.getOperands().size()
          << "; NCCL is enabled: " << NcclThunkType::NcclIsEnabled();

  // A given collective op can be degenerate if across all groups formed
  // by it are singleton. In such a case, we don't need to do any communication
  // and we can just copy the input to the output.
  bool is_degenerate =
      NcclThunkType::IsDegenerate(op, replica_count, partition_count);
  bool should_use_nccl_thunk =
      !is_degenerate && NcclThunkType::CanImplement(op);

  // Stash relevant information in NcclCollectiveThunk::Buffer even if we may
  // not generate an NcclCollectiveThunk.
  std::vector<NcclCollectiveThunk::Buffer> buffers;
  buffers.reserve(op.getInputs().size());
  for (auto it : llvm::zip(op.getInputs(), op.getOutputs())) {
    mlir::Value operand = std::get<0>(it);
    mlir::Value result = std::get<1>(it);
    const Shape shape = GetShape(operand);
    TF_ASSIGN_OR_RETURN(auto source_slice, GetAllocationSlice(operand));
    TF_ASSIGN_OR_RETURN(auto dest_slice, GetAllocationSlice(result));
    buffers.push_back(NcclCollectiveThunk::Buffer{
        /*element_count=*/ShapeUtil::ElementsIn(shape),
        /*source_buffer=*/source_slice,
        /*destination_buffer=*/dest_slice,
        /*source_value=*/operand,
        /*destination_value=*/result});
  }

  if (should_use_nccl_thunk) {
    auto thunk =
        std::make_unique<NcclThunkType>(GetThunkInfo(op), op,
                                        /*buffers=*/std::move(buffers));

    if constexpr (NcclThunkType::IsAsync()) {
      async_executors_.insert({untyped_op, &thunk->async_executor()});
    }

    AddThunkToThunkSequence(std::move(thunk));
    return OkStatus();
  }

  // Signal that start thunk not created with nullptr.
  if constexpr (NcclThunkType::IsAsync()) {
    async_executors_.insert({untyped_op, nullptr});
  }

  if (!is_degenerate) {
    CollectiveOpGroupMode group_mode = NcclThunkType::GetGroupMode(op);

    std::string message = absl::StrFormat(
        "Requested %s not implemented on GPU; replica_count: %d; "
        "partition_count: %d, group_mode: %s, operand_count: %d; NCCL support: "
        "%d",
        NcclThunkType::GetName(), replica_count, partition_count,
        CollectiveOpGroupModeToString(group_mode), op.getOperands().size(),
        NcclThunkType::NcclIsEnabled());
    if (!op.getOperands().empty()) {
      const Shape shape = GetShape(op.getOperands().front());
      absl::StrAppendFormat(&message, "; first operand array element-type: %s",
                            PrimitiveType_Name(shape.element_type()));
    }
    return Unimplemented("%s", message);
  }

  VLOG(1) << "Collective call is degenerate, not doing NCCL call";

  // All-gather with one replica is simply the identity function. Buffer
  // assignment expects a copy, so that's what we do.
  ThunkSequence thunks;
  for (int64_t i = 0; i < buffers.size(); i++) {
    const Shape shape = GetShape(op.getOperands()[i]);
    thunks.push_back(std::make_unique<DeviceToDeviceCopyThunk>(
        buffers.size() == 1 ? GetThunkInfo(op) : Thunk::ThunkInfo(op),
        /*source_buffer=*/buffers[i].source_buffer,
        /*destination_buffer=*/buffers[i].destination_buffer,
        /*mem_size=*/ShapeUtil::ByteSizeOf(shape),
        /*source_value=*/buffers[i].source_value,
        /*destination_value=*/buffers[i].destination_value));
  }
  if (thunks.size() == 1) {
    AddThunkToThunkSequence(std::move(thunks[0]));
  } else {
    AddThunkToThunkSequence(
        std::make_unique<SequentialThunk>(GetThunkInfo(op), std::move(thunks)));
  }
  return OkStatus();
}

template <typename NcclThunkType, typename OpT>
Status IrEmitterUnnested::EmitNcclAsyncDone(mlir::Operation* op) {
  auto start_op = mlir::cast<OpT>(op).getToken().getDefiningOp();
  auto async_executor = async_executors_.extract(start_op);
  TF_RET_CHECK(async_executor) << "couldn't find async executor for start op";

  // Can be null if no start thunk was created (e.g. if the start op is
  // degenerate), in which case there's nothing to do here.
  if (async_executor.mapped() != nullptr) {
    AddThunkToThunkSequence(std::make_unique<NcclThunkType>(
        GetThunkInfo(op), *async_executor.mapped()));
  }
  return OkStatus();
}

StatusOr<std::vector<ShapedSlice>> IrEmitterUnnested::GetShapedSlices(
    mlir::Operation::operand_range operands) {
  std::vector<ShapedSlice> shaped_slices;
  shaped_slices.reserve(operands.size());
  for (mlir::Value opnd : operands) {
    TF_ASSIGN_OR_RETURN(auto slice, GetAllocationSlice(opnd));
    shaped_slices.push_back(ShapedSlice{slice, GetShape(opnd)});
  }
  return shaped_slices;
}

StatusOr<std::vector<BufferAllocation::Slice>> IrEmitterUnnested::GetSlices(
    mlir::Operation::operand_range operands) {
  std::vector<BufferAllocation::Slice> slices;
  slices.reserve(operands.size());
  for (mlir::Value opnd : operands) {
    TF_ASSIGN_OR_RETURN(auto slice, GetAllocationSlice(opnd));
    slices.push_back(slice);
  }
  return slices;
}

Status IrEmitterUnnested::EmitInfeed(mlir::Operation* op) {
  mlir::Operation::operand_range operands =
      mlir::cast<mlir::lmhlo::InfeedOp>(op).getOutputs();
  TF_ASSIGN_OR_RETURN(auto shaped_slices, GetShapedSlices(operands));
  auto thunk =
      std::make_unique<InfeedThunk>(GetThunkInfo(op), std::move(shaped_slices));
  AddThunkToThunkSequence(std::move(thunk));

  return OkStatus();
}

Status IrEmitterUnnested::EmitOutfeed(mlir::Operation* op) {
  mlir::Operation::operand_range operands =
      mlir::cast<mlir::lmhlo::OutfeedOp>(op).getInputs();
  TF_ASSIGN_OR_RETURN(auto shaped_slices, GetShapedSlices(operands));
  auto thunk = std::make_unique<OutfeedThunk>(GetThunkInfo(op),
                                              std::move(shaped_slices));
  AddThunkToThunkSequence(std::move(thunk));

  return OkStatus();
}

StatusOr<std::vector<llvm_ir::IrArray>> IrEmitterUnnested::BuildKernelThunkImpl(
    absl::string_view name, Thunk::ThunkInfo thunk_info,
    std::vector<KernelArgument> kernel_arguments,
    const LaunchDimensions& launch_dimensions) {
  std::vector<llvm_ir::IrArray> ir_arrays;

  // Temporarily reorder the values/slices to match the way we supply buffer
  // allocation arguments to the GPU kernels (see below).
  absl::c_sort(kernel_arguments,
               [](const KernelArgument& a, const KernelArgument& b) {
                 return a.slice.buffer_slice.allocation()->index() <
                        b.slice.buffer_slice.allocation()->index();
               });

  // Figure out which buffer allocations need to be passed as arguments to our
  // kernel.  This is simply all of the allocations referenced in slices,
  // plus the XLA temp buffer (if we have it).  We always include the temp
  // buffer because even if the kernel itself doesn't use it, a nested
  // subcomputation within the kernel (e.g. a kMap's computation) might.
  // For XLA runtime, do the same for mlir::Value(s).
  absl::flat_hash_set<const BufferAllocation*> buffers_needed;
  std::vector<mlir::Value> values_needed;
  for (const auto& kernel_argument : kernel_arguments) {
    const BufferSlice& slice = kernel_argument.slice;

    if (slice.buffer_slice.allocation()->is_constant()) continue;

    auto result = buffers_needed.insert(slice.buffer_slice.allocation());
    if (!result.second) continue;

    auto add_if_not_exists = [&](mlir::Value buffer_alloc_arg) {
      if (!absl::c_linear_search(values_needed, buffer_alloc_arg)) {
        values_needed.push_back(buffer_alloc_arg);
      }
    };

    mlir::Value argument = kernel_argument.value;
    auto defining_op = argument.getDefiningOp();
    if (defining_op == nullptr) {
      add_if_not_exists(argument);
      continue;
    }

    if (auto view_op = llvm::dyn_cast<mlir::memref::ViewOp>(defining_op)) {
      argument = view_op.getOperand(0);
      add_if_not_exists(argument);
      continue;
    }

    if (auto cast_op =
            llvm::dyn_cast<mlir::memref::ReinterpretCastOp>(defining_op)) {
      argument = cast_op.getOperand(0);
      if (auto view_op =
              llvm::dyn_cast<mlir::memref::ViewOp>(argument.getDefiningOp())) {
        argument = view_op.getOperand(0);
      }
      add_if_not_exists(argument);
      continue;
    }

    if (auto collapse_shape_op =
            llvm::dyn_cast<mlir::memref::CollapseShapeOp>(defining_op)) {
      argument = collapse_shape_op.getSrc();
      if (auto view_op =
              llvm::dyn_cast<mlir::memref::ViewOp>(argument.getDefiningOp())) {
        argument = view_op.getOperand(0);
      }
      add_if_not_exists(argument);
      continue;
    }

    return Unimplemented(
        "Defining op for argument to GPU kernel not handled: %s",
        defining_op->getName().getStringRef().str());
  }
  std::optional<const BufferAllocation*> temp_buffer;
  for (const BufferAllocation& alloc : ir_emitter_context_->allocations()) {
    if (alloc.IsPreallocatedTempBuffer()) {
      if (!temp_buffer.has_value()) {
        // Retrieve the first seen temp buffer.
        temp_buffer = &alloc;
      }
    }
  }
  if (temp_buffer.has_value()) {
    buffers_needed.insert(*temp_buffer);
  }

  // We'll pass a pointer to each of the elements of `buffers` to our kernel, in
  // this order.
  std::vector<const BufferAllocation*> buffers_ordered;
  absl::c_copy(buffers_needed, std::back_inserter(buffers_ordered));
  absl::c_sort(buffers_ordered,
               [](const BufferAllocation* a, const BufferAllocation* b) {
                 return a->index() < b->index();
               });

  llvm::Function* kernel = BuildKernelPrototype(name, buffers_ordered);

  // Build a map from a BufferAllocation to the corresponding argument in our
  // kernel.
  absl::flat_hash_map<const BufferAllocation*, llvm::Value*> kernel_args;
  {
    auto arg_it = kernel->arg_begin();
    auto buffers_it = buffers_ordered.begin();
    for (; arg_it != kernel->arg_end(); ++arg_it, ++buffers_it) {
      kernel_args[*buffers_it] = arg_it;

      // Annotate all allocations with LLVM's `noalias`.
      // There are three kinds of allocations:
      // * Read-only allocations, aka input parameters that are not aliased with
      // outputs.
      // * Read-write allocations, including all output buffers, some of which
      // may alias with input HLO parameters, but aliased HLO buffers are always
      // assigned with the same allocation.
      // * The temp buffer.
      //
      // Read-only allocations may overlap with each other, but since they are
      // not mutated, they can always be annotated with `noalias` per LLVM
      // semantics.
      //
      // Read-write allocations and the temp buffer don't overlap with any
      // allocations, therefore they can also be annotated with `noalias`.
      kernel->addParamAttr(
          arg_it->getArgNo(),
          llvm::Attribute::get(arg_it->getContext(), llvm::Attribute::NoAlias));
    }
  }

  // Recover the original ordering of values/slices.
  absl::c_sort(kernel_arguments,
               [](const KernelArgument& a, const KernelArgument& b) {
                 return a.order < b.order;
               });

  absl::flat_hash_set<BufferAllocation::Slice> buffers_written;
  for (const auto& kernel_argument : kernel_arguments) {
    const auto& slice = kernel_argument.slice;
    if (slice.written) {
      buffers_written.insert(slice.buffer_slice);
    }
  }

  // For each buffer our kernel might want to touch, bind it to a value derived
  // from our kernel args.
  for (const auto& kernel_argument : kernel_arguments) {
    const auto& slice = kernel_argument.slice;
    const BufferAllocation::Slice& buffer_slice = slice.buffer_slice;

    llvm::Value* loc;
    if (!slice.constant_name.empty()) {
      loc = module_->getGlobalVariable(slice.constant_name);
      CHECK_NE(loc, nullptr)
          << "Could not find variable '" << slice.constant_name << "'";
    } else {
      CHECK(!buffer_slice.allocation()->is_constant());
      loc =
          InBoundsGEP(b_.getInt8Ty(), kernel_args.at(buffer_slice.allocation()),
                      {b_.getInt64(buffer_slice.offset())});
    }

    llvm::Type* ir_type = llvm_ir::ShapeToIrType(slice.shape, module_);
    llvm_ir::IrArray ir_array(CastToTypedValue(slice.shape, loc, &b_), ir_type,
                              slice.shape);
    if (!buffers_written.contains(slice.buffer_slice)) {
      ir_array.MarkInvariantOverWholeProgram(&loc->getContext());
    }

    ir_arrays.push_back(ir_array);
  }

  AnnotateKernelLaunchDimensions(launch_dimensions,
                                 std::string(kernel->getName()), module_);

  AddThunkToThunkSequence(std::make_unique<KernelThunk>(
      thunk_info, buffers_ordered, std::string(kernel->getName()),
      launch_dimensions, std::move(values_needed)));
  return ir_arrays;
}

StatusOr<IrEmitterUnnested::KernelArgument>
IrEmitterUnnested::ValueToKernelArgument(mlir::Value operand, int order,
                                         bool is_written) {
  KernelArgument kernel_argument;
  kernel_argument.order = order;
  kernel_argument.value = operand;
  BufferSlice& slice = kernel_argument.slice;
  TF_ASSIGN_OR_RETURN(slice.buffer_slice,
                      GetAllocationSlice(operand, &slice.constant_name));
  slice.written = is_written;
  slice.shape = GetShape(operand);
  return kernel_argument;
}

StatusOr<std::vector<llvm_ir::IrArray>> IrEmitterUnnested::BuildKernelThunk(
    mlir::Operation* op, mlir::ValueRange operands,
    const LaunchDimensions& launch_dimensions) {
  TF_RET_CHECK(!mlir::isa<mlir::lmhlo::FusionOp>(op));

  std::vector<KernelArgument> kernel_arguments(operands.size());
  for (auto& [i, operand] : llvm::enumerate(operands)) {
    TF_ASSIGN_OR_RETURN(
        kernel_arguments[i],
        ValueToKernelArgument(operand, i, WritesMlirBuffer(op, operand)));
  }
  std::string name = GetIrNameFromLoc(op->getLoc());
  auto thunk_info = GetThunkInfo(op);
  return BuildKernelThunkImpl(name, thunk_info, std::move(kernel_arguments),
                              launch_dimensions);
}

StatusOr<std::vector<llvm_ir::IrArray>> IrEmitterUnnested::BuildKernelThunk(
    mlir::Operation* op, const LaunchDimensions& launch_dimensions) {
  // TODO(tdanyluk): Consider disallowing fusions here, if no fusions use this
  // anymore.
  if (auto fusion = mlir::dyn_cast<mlir::lmhlo::FusionOp>(op)) {
    llvm::SmallVector<mlir::Value> operands = GetHloOperands(op);
    llvm::SmallVector<mlir::Value> outputs = GetHloOutputs(op);

    std::vector<KernelArgument> kernel_arguments(operands.size() +
                                                 outputs.size());
    int i = 0;
    for (mlir::Value op : llvm::concat<mlir::Value>(operands, outputs)) {
      TF_ASSIGN_OR_RETURN(
          kernel_arguments[i],
          ValueToKernelArgument(op, i, /*is_written=*/i >= operands.size()));
      i++;
    }

    std::string name = GetIrNameFromLoc(op->getLoc());
    return BuildKernelThunkImpl(name, GetThunkInfo(op), kernel_arguments,
                                launch_dimensions);
  }
  return BuildKernelThunk(op, op->getOperands(), launch_dimensions);
}

StatusOr<std::vector<IrEmitterUnnested::ReusableKernelArgument>>
IrEmitterUnnested::GetReusableKernelArguments(mlir::lmhlo::FusionOp fusion_op) {
  std::vector<ReusableKernelArgument> kernel_arguments;
  {
    llvm::SmallVector<mlir::Value> operands = GetHloOperands(fusion_op);
    llvm::SmallVector<mlir::Value> outputs = GetHloOutputs(fusion_op);
    kernel_arguments.reserve(operands.size() + outputs.size());

    int i = 0;
    for (mlir::Value value : llvm::concat<mlir::Value>(operands, outputs)) {
      ReusableKernelArgument kernel_argument;
      kernel_argument.value = value;
      kernel_argument.shape = GetShape(value);
      TF_ASSIGN_OR_RETURN(kernel_argument.slice,
                          GetAllocationSlice(value, nullptr));
      // Note: We change this on a later line.
      kernel_argument.written = i >= operands.size();
      kernel_arguments.push_back(std::move(kernel_argument));
      i += 1;
    }
  }

  absl::flat_hash_set<BufferAllocation::Slice> buffers_written;
  for (const ReusableKernelArgument& kernel_argument : kernel_arguments) {
    if (kernel_argument.written) {
      buffers_written.insert(kernel_argument.slice);
    }
  }

  for (int i = 0; i < static_cast<int>(kernel_arguments.size()); ++i) {
    ReusableKernelArgument& kernel_argument = kernel_arguments[i];

    kernel_argument.first_with_same_slice = [&]() -> std::optional<int> {
      for (int j = 0; j < i; ++j) {
        const ReusableKernelArgument& other_kernel_argument =
            kernel_arguments[j];
        if (kernel_argument.slice == other_kernel_argument.slice) {
          return j;
        }
      }
      return std::nullopt;
    }();

    if (kernel_argument.first_with_same_slice.has_value()) {
      const ReusableKernelArgument& same =
          kernel_arguments[kernel_argument.first_with_same_slice.value()];
      kernel_argument.alignment = same.alignment;
      kernel_argument.aliased = same.aliased;
      kernel_argument.written = same.written;
      continue;
    }

    kernel_argument.alignment = [&] {
      const BufferAllocation* alloc = kernel_argument.slice.allocation();
      if (alloc->is_entry_computation_parameter()) {
        return kEntryParameterAlignBytes;
      } else if (alloc->is_constant()) {
        return kConstantBufferAlignBytes;
      } else {
        return kXlaAllocatedBufferAlignBytes;
      }
    }();

    kernel_argument.written = buffers_written.contains(kernel_argument.slice);

    kernel_argument.aliased = kernel_argument.written && [&] {
      for (size_t j = 0; j < kernel_arguments.size(); ++j) {
        const ReusableKernelArgument& other_kernel_argument =
            kernel_arguments[j];
        if (i != j && kernel_argument.slice != other_kernel_argument.slice &&
            kernel_argument.slice.OverlapsWith(other_kernel_argument.slice)) {
          return true;
        }
      }
      return false;
    }();
  }

  return kernel_arguments;
}

std::string IrEmitterUnnested::GetArgumentFingerprint(
    absl::Span<const ReusableKernelArgument> kernel_arguments) {
  return absl::StrJoin(kernel_arguments, ",",
                       [](std::string* s, const ReusableKernelArgument& arg) {
                         if (arg.first_with_same_slice.has_value()) {
                           absl::StrAppend(s, "=",
                                           arg.first_with_same_slice.value());
                           return;
                         }
                         absl::StrAppend(s, arg.alignment);
                         if (arg.aliased) {
                           absl::StrAppend(s, "a");
                         }
                         if (arg.written) {
                           absl::StrAppend(s, "w");
                         }
                       });
}

std::string IrEmitterUnnested::GetFingerprint(
    const HloComputation* fused_computation,
    absl::Span<const ReusableKernelArgument> kernel_arguments,
    absl::string_view discriminator) {
  // We have to print constants, because otherwise we would accidentally reuse
  // kernels which have different builtin constants.
  //
  // It is not a problem to recursively print subcomputations, because we don't
  // have them at this point.
  auto print_options = HloPrintOptions::Fingerprint()
                           .set_print_only_essential_constants(false)
                           .set_print_operand_shape(false);

  return absl::StrCat(discriminator, "(",
                      GetArgumentFingerprint(kernel_arguments), ")",
                      fused_computation->ToString(print_options));
}

StatusOr<mlir::Value> IrEmitterUnnested::RemoveTransformingOperations(
    mlir::Value value) {
  // This is based on BuildKernelThunkImpl, but ViewOp's are not removed,
  // because we want to refer to the exact location of the arguments, not the
  // whole allocation.

  mlir::Operation* defining_op = value.getDefiningOp();
  // Don't remove GetGlobalOp's and ViewOp's:
  // Allow passing constants and views to kernels.
  if (defining_op == nullptr ||
      llvm::dyn_cast<mlir::memref::GetGlobalOp>(defining_op) ||
      llvm::dyn_cast<mlir::memref::ViewOp>(defining_op)) {
    return value;
  }
  if (auto cast_op =
          llvm::dyn_cast<mlir::memref::ReinterpretCastOp>(defining_op)) {
    return cast_op.getOperand(0);
  }
  if (auto collapse_shape_op =
          llvm::dyn_cast<mlir::memref::CollapseShapeOp>(defining_op)) {
    return collapse_shape_op.getSrc();
  }
  return Unimplemented("Defining op for argument to GPU kernel not handled: %s",
                       defining_op->getName().getStringRef().str());
}

StatusOr<ReusableKernelThunk*> IrEmitterUnnested::BuildReusableKernelThunkImpl(
    absl::string_view kernel_name,
    absl::Span<const ReusableKernelArgument> kernel_arguments,
    Thunk::ThunkInfo thunk_info, const LaunchDimensions& launch_dimensions) {
  std::vector<BufferAllocation::Slice> arg_slices;
  arg_slices.reserve(kernel_arguments.size());
  for (const auto& kernel_argument : kernel_arguments) {
    if (!kernel_argument.first_with_same_slice.has_value()) {
      arg_slices.push_back(kernel_argument.slice);
    }
  }

  std::vector<mlir::Value> values;
  values.reserve(kernel_arguments.size());
  for (const auto& kernel_argument : kernel_arguments) {
    if (!kernel_argument.first_with_same_slice.has_value()) {
      TF_ASSIGN_OR_RETURN(mlir::Value value,
                          RemoveTransformingOperations(kernel_argument.value));
      values.push_back(value);
    }
  }

  auto thunk_ptr = std::make_unique<ReusableKernelThunk>(
      std::move(thunk_info), std::move(arg_slices), std::string(kernel_name),
      launch_dimensions, std::move(values));
  ReusableKernelThunk* raw_thunk_ptr = thunk_ptr.get();
  AddThunkToThunkSequence(std::move(thunk_ptr));

  return raw_thunk_ptr;
}

StatusOr<std::optional<std::vector<llvm_ir::IrArray>>>
IrEmitterUnnested::BuildReusableKernelThunk(
    mlir::lmhlo::FusionOp fusion_op, const LaunchDimensions& launch_dimensions,
    absl::string_view discriminator) {
  std::string suggested_kernel_name = GetIrNameFromLoc(fusion_op->getLoc());

  TF_ASSIGN_OR_RETURN(std::vector<ReusableKernelArgument> kernel_arguments,
                      GetReusableKernelArguments(fusion_op));

  TF_ASSIGN_OR_RETURN(
      const HloComputation* fused_computation,
      GetOrCreateSubComputationFromRegion(&fusion_op.getRegion(),
                                          /*is_fusion=*/true));
  std::string fingerprint =
      GetFingerprint(fused_computation, kernel_arguments, discriminator);
  VLOG(4) << "Fingerprint: ";
  XLA_VLOG_LINES(4, fingerprint);

  auto cache_it = kernel_reuse_cache_.find(fingerprint);
  if (cache_it != kernel_reuse_cache_.end()) {
    ReusableKernelThunk* old_thunk = cache_it->second;

    VLOG(3) << "Reuse: " << suggested_kernel_name << " -> "
            << old_thunk->kernel_name();

    // The calculated launch dimensions must be the same for kernels which are
    // deduplicated.
    // TODO(tdanyluk): Consider avoiding the recalculation of launch dimensions
    // when reusing kernels.
    TF_RET_CHECK(old_thunk->launch_dimensions() == launch_dimensions);

    // We are not reusing the ThunkInfo of the old thunk, because the current
    // thunk info must reference the current HLO operation.
    TF_RETURN_IF_ERROR(
        BuildReusableKernelThunkImpl(old_thunk->kernel_name(), kernel_arguments,
                                     GetThunkInfo(fusion_op), launch_dimensions)
            .status());

    return {std::nullopt};
  }

  VLOG(3) << "Generating: " << suggested_kernel_name;

  auto [kernel, ir_arrays] = BuildReusableKernelPrototype(
      suggested_kernel_name, kernel_arguments, launch_dimensions);

  TF_ASSIGN_OR_RETURN(
      ReusableKernelThunk * thunk,
      BuildReusableKernelThunkImpl(kernel->getName().str(), kernel_arguments,
                                   GetThunkInfo(fusion_op), launch_dimensions));
  kernel_reuse_cache_[fingerprint] = thunk;

  return {ir_arrays};
}

std::unique_ptr<Thunk> IrEmitterUnnested::BuildConstantInitializerThunk(
    mlir::Operation* op, absl::Span<const uint8_t> init_value, mlir::Value dest,
    const BufferAllocation::Slice& dest_slice, const Shape& output_shape) {
  int64_t num_bytes = init_value.size();
  if (absl::c_all_of(init_value, [](uint8_t byte) { return byte == 0; })) {
    return std::make_unique<MemzeroThunk>(Thunk::ThunkInfo(op), dest_slice,
                                          dest);
  }

  // If the literal is 8 or 16 bits wide, we can emit a 32-bit memset by
  // repeating the literal 4 or 2 times, so long as the destination buffer is
  // an even multiple of 32 bits long.
  if ((num_bytes == 1 || num_bytes == 2) &&
      ShapeUtil::ByteSizeOf(output_shape) % 4 == 0) {
    uint16_t pattern16;
    if (num_bytes == 1) {
      uint8_t b = init_value.front();
      pattern16 = uint16_t{b} | (uint16_t{b} << 8);
    } else {
      memcpy(&pattern16, init_value.data(), sizeof(pattern16));
    }
    uint32_t pattern32 = uint32_t{pattern16} | (uint32_t{pattern16} << 16);
    return std::make_unique<Memset32BitValueThunk>(Thunk::ThunkInfo(op),
                                                   pattern32, dest_slice, dest);
  }

  // If the literal is an even multiple of 32 bits wide, we can emit a 32-bit
  // memset so long as all 32-bit words of the scalar are equal to each other.
  if (num_bytes >= 4 && num_bytes % 4 == 0 &&
      memcmp(init_value.data(), init_value.data() + 4, init_value.size() - 4) ==
          0) {
    uint32_t word;
    memcpy(&word, init_value.data(), sizeof(word));
    return std::make_unique<Memset32BitValueThunk>(Thunk::ThunkInfo(op), word,
                                                   dest_slice, dest);
  }

  return nullptr;
}

StatusOr<std::unique_ptr<Thunk>>
IrEmitterUnnested::TryBuildConstantInitializerThunk(mlir::Operation* op,
                                                    mlir::Value init_value,
                                                    mlir::Value dest) {
  mlir::DenseElementsAttr const_init;
  if (auto get_global_memref =
          mlir::dyn_cast_or_null<mlir::memref::GetGlobalOp>(
              init_value.getDefiningOp())) {
    auto global_memref =
        mlir::SymbolTable::lookupNearestSymbolFrom<mlir::memref::GlobalOp>(
            get_global_memref, get_global_memref.getNameAttr());
    if (global_memref.getConstant() && global_memref.getInitialValue()) {
      // If the initial value happens to be a constant, generate a specialized
      // thunk.
      const_init = global_memref.getInitialValue()
                       .value()
                       .cast<mlir::DenseElementsAttr>();
    }
  } else if (auto constant = mlir::dyn_cast_or_null<mlir::mhlo::ConstantOp>(
                 init_value.getDefiningOp())) {
    const_init = constant.getValue().dyn_cast<mlir::DenseElementsAttr>();
  }

  if (const_init) {
    std::vector<uint8_t> literal_bytes;
    TF_RETURN_IF_ERROR(
        CopyDenseElementsDataToXlaFormat(const_init, &literal_bytes));

    TF_ASSIGN_OR_RETURN(auto dest_slice, GetAllocationSlice(dest));

    const Shape dest_shape = GetShape(dest);
    auto thunk = BuildConstantInitializerThunk(op, literal_bytes, dest,
                                               dest_slice, dest_shape);
    if (thunk) {
      return {std::move(thunk)};
    }
  }
  return std::unique_ptr<Thunk>();
}

Status IrEmitterUnnested::BuildInitializerThunk(mlir::Operation* op,
                                                mlir::Value init_value,
                                                mlir::Value dest) {
  // initial value must be a scalar memref.
  auto init_type = init_value.getType().dyn_cast<mlir::MemRefType>();
  TF_RET_CHECK(init_type.getRank() == 0);

  TF_ASSIGN_OR_RETURN(std::unique_ptr<Thunk> constant_init_thunk,
                      TryBuildConstantInitializerThunk(op, init_value, dest));
  if (constant_init_thunk) {
    AddThunkToThunkSequence(std::move(constant_init_thunk));
    return OkStatus();
  }

  // Otherwise fall back to our slow initializer code. The thunk in this case
  // will just need the IR arrays for the initial value and the destination.
  const Shape dest_shape = GetShape(dest);
  TF_ASSIGN_OR_RETURN(LaunchDimensions launch_dimensions,
                      CalculateLaunchDimensions(
                          dest_shape, ir_emitter_context_->gpu_device_info()));
  TF_ASSIGN_OR_RETURN(
      std::vector<llvm_ir::IrArray> ir_arrays,
      BuildKernelThunk(op, {init_value, dest}, launch_dimensions));

  const llvm_ir::IrArray init_array = ir_arrays[0];
  const llvm_ir::IrArray dest_array = ir_arrays[1];

  std::string name = GetIrNameFromLoc(op->getLoc());
  TF_RETURN_IF_ERROR(ParallelLoopEmitter(
                         [=](const IrArray::Index& index) {
                           return init_array.EmitReadArrayElement(index, &b_);
                         },
                         {dest_array}, launch_dimensions, &b_)
                         .EmitLoop(GetIrNameFromLoc(op->getLoc())));
  return OkStatus();
}

Status IrEmitterUnnested::BuildFusedInitializerThunk(
    mlir::lmhlo::FusionOp fusion, int output_index) {
  auto reduce = mlir::dyn_cast_or_null<mlir::mhlo::ReduceOp>(
      fusion.getFusionRoots()[output_index]);

  TF_RET_CHECK(reduce);
  TF_RET_CHECK(reduce.getNumResults() == 1);

  mlir::Value init_value = reduce.getInitValues()[0];
  mlir::Value dest = fusion.getOutputBuffers()[output_index];
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<Thunk> constant_init_thunk,
      TryBuildConstantInitializerThunk(fusion, init_value, dest));
  if (constant_init_thunk) {
    AddThunkToThunkSequence(std::move(constant_init_thunk));
    return OkStatus();
  }

  auto input_buffers = fusion.getInputBuffers();

  const Shape dest_shape = GetShape(dest);
  TF_ASSIGN_OR_RETURN(LaunchDimensions launch_dimensions,
                      CalculateLaunchDimensions(
                          dest_shape, ir_emitter_context_->gpu_device_info()));

  TF_ASSIGN_OR_RETURN(
      std::optional<std::vector<llvm_ir::IrArray>> opt_ir_arrays,
      BuildReusableKernelThunk(
          fusion, launch_dimensions,
          /*discriminator=*/absl::StrCat("init_", output_index)));
  if (!opt_ir_arrays.has_value()) {
    // The kernel was reused, no need to emit code.
    return OkStatus();
  }
  std::vector<llvm_ir::IrArray>& ir_arrays = opt_ir_arrays.value();

  const llvm_ir::IrArray dest_array =
      ir_arrays[input_buffers.size() + output_index];

  const HloComputation* fused_computation =
      *GetOrCreateSubComputationFromRegion(&fusion.getRegion(),
                                           /*is_fusion=*/true);

  FusedIrEmitter fused_emitter(elemental_emitter_);
  for (int i = 0; i < fused_computation->num_parameters(); i++) {
    fused_emitter.BindGenerator(
        *fused_computation->parameter_instruction(i),
        [this, &ir_arrays, i](llvm_ir::IrArray::Index index) {
          return ir_arrays[i].EmitReadArrayElement(index, &b_);
        });
  }
  HloInstruction* instr = fused_computation->root_instruction();
  if (instr->opcode() != HloOpcode::kTuple) {
    CHECK_EQ(0, output_index);
  } else {
    instr = instr->mutable_operand(output_index);
  }
  TF_RET_CHECK(instr->shape().IsArray());
  TF_ASSIGN_OR_RETURN(auto generator,
                      fused_emitter.GetGenerator(*instr->operand(1)));
  TF_RETURN_IF_ERROR(
      ParallelLoopEmitter(generator, {dest_array}, launch_dimensions, &b_)
          .EmitLoop(GetIrNameFromLoc(fusion.getLoc())));
  return OkStatus();
}

StatusOr<std::unique_ptr<Thunk>> IrEmitterUnnested::BuildWhileThunk(
    mlir::lmhlo::WhileOp while_op, const Thunk::ThunkInfo& thunk_info) {
  // Generate thunk sequence for while 'condition'.
  mlir::Region* condition = &while_op.getCond();
  TF_ASSIGN_OR_RETURN(
      auto ir_emitter_condition,
      IrEmitterUnnested::Create(hlo_module_config_, ir_emitter_context_));

  TF_RETURN_IF_ERROR(ir_emitter_condition->EmitLmhloRegion(condition));

  // Generate thunk sequence for while 'body'.
  mlir::Region* body = &while_op.getBody();
  TF_ASSIGN_OR_RETURN(
      auto ir_emitter_body,
      IrEmitterUnnested::Create(hlo_module_config_, ir_emitter_context_));

  TF_RETURN_IF_ERROR(ir_emitter_body->EmitLmhloRegion(body));

  // Extract the condition value from the last op (exlucidng the terminator op)
  // in the condition region.
  auto cond_result = GetHloOutputs(while_op);
  TF_RET_CHECK(cond_result.size() == 1);
  TF_ASSIGN_OR_RETURN(auto cond_result_slice,
                      GetAllocationSlice(cond_result[0]));

  return std::unique_ptr<Thunk>(
      new WhileThunk(thunk_info, cond_result_slice,
                     ir_emitter_condition->ConsumeThunkSequence(),
                     ir_emitter_body->ConsumeThunkSequence()));
}

StatusOr<std::unique_ptr<Thunk>> IrEmitterUnnested::BuildForThunk(
    mlir::lmhlo::WhileOp while_op, const Thunk::ThunkInfo& thunk_info,
    const int64_t loop_limit) {
  // Generate thunk sequence for while 'body' (will be used a For loop body).
  TF_ASSIGN_OR_RETURN(
      auto ir_emitter_body,
      IrEmitterUnnested::Create(hlo_module_config_, ir_emitter_context_));
  TF_RETURN_IF_ERROR(ir_emitter_body->EmitLmhloRegion(&while_op.getBody()));

  return std::unique_ptr<Thunk>(new ForThunk(
      thunk_info, loop_limit, ir_emitter_body->ConsumeThunkSequence()));
}

Status IrEmitterUnnested::EmitTargetElementLoop(
    const HloInstruction& hlo, const llvm_ir::ElementGenerator& body_emitter) {
  return InternalError("This should be unreachable");
}

// Gets the output offset as calculated from thread_id.x (to be applied to the
// offset calculated from block_id and thread_id.y).
static llvm::Value* GetStartOffsetX(const TilingScheme& tiling_scheme,
                                    llvm::Value* thread_id_x,
                                    llvm::Type* index_ty,
                                    llvm::IRBuilder<>* b) {
  int64_t multiplier = tiling_scheme.GetIndexingOrder() == kStridedIndexingX
                           ? tiling_scheme.GetVectorSize()
                           : tiling_scheme.GetTileSizeFor(kDimX);
  return b->CreateMul(thread_id_x,
                      llvm::ConstantInt::get(index_ty, multiplier));
}

// Emits loop through the minor (X) dimension of a tile, starting at a given
// offset.
//
// Rough pseudocode:
//
// Given: offset, callback
//
// for (int x = 0; x < x_tile_size / vector_size; ++x) {
//   for (int i = 0; i < vector_size; ++i) {
//      callback(offset + x * stride * vector_size + i);
//   }
// }
static void EmitXTileLoop(
    const IrEmitterUnnested::ThreadIdInfo& thread_id_info,
    const IrArray::Index& tile_origin_index, const TilingScheme& tiling_scheme,
    bool check_x_tile_bounds, llvm::Value* y_loc,
    IrEmitterUnnested::ValueVector2 tile_dimensions, llvm::IRBuilder<>* b,
    const IrEmitterUnnested::EmitElementFunction* emit_elem_function) {
  llvm::Type* index_ty = tile_dimensions[1]->getType();
  KernelSupportLibrary ksl(b, llvm_ir::UnrollMode::kDefaultUnroll);
  auto constant = [&](int64_t val) {
    return llvm::ConstantInt::get(index_ty, val);
  };
  llvm::Value* start_offset_x =
      GetStartOffsetX(tiling_scheme, thread_id_info.thread_id_x, index_ty, b);

  int64_t vector_size = tiling_scheme.GetVectorSize();
  int64_t stride_x = tiling_scheme.GetIndexingOrder() == kLinearIndexingX
                         ? 1
                         : tiling_scheme.GetNumThreadsFor(kDimX);
  KernelSupportLibrary unrolled_ksl(b, llvm_ir::UnrollMode::kFullyUnroll);
  unrolled_ksl.For(
      "tile_loop",
      /*start=*/constant(0),
      /*end=*/constant(tiling_scheme.GetTileSizeFor(kDimX) / vector_size),
      /*step=*/1, [&](llvm::Value* x) {
        for (int64_t i = 0; i < vector_size; i++) {
          llvm::Value* x_offset = b->CreateAdd(
              b->CreateMul(x, constant(stride_x * vector_size)), constant(i));
          llvm::Value* x_loc = b->CreateAdd(x_offset, start_offset_x, "x_loc");
          IrArray::Index source_idx_x =
              tile_origin_index
                  .AddOffsetToDim(y_loc, tiling_scheme.GetTilingDimension(0), b)
                  .AddOffsetToDim(x_loc, tiling_scheme.GetTilingDimension(1),
                                  b);
          auto emit_element = [&] {
            return (*emit_elem_function)(thread_id_info, source_idx_x, y_loc,
                                         x_loc);
          };
          if (check_x_tile_bounds) {
            ksl.If("x_in_tile", b->CreateICmpULT(x_loc, tile_dimensions[1]),
                   emit_element);
          } else {
            emit_element();
          }
        }
      });
}

void IrEmitterUnnested::EmitTile(
    const TilingScheme& tiling_scheme, const IrArray::Index& tile_origin_index,
    const ThreadIdInfo& thread_id_info, ValueVector2 tile_dimensions,
    const IrEmitterUnnested::EmitElementFunction& emit_elem_function) {
  llvm::Type* index_ty = tile_dimensions[0]->getType();
  auto constant = [&](int64_t val) {
    return llvm::ConstantInt::get(index_ty, val);
  };
  llvm::Value* num_threads_y = constant(
      tiling_scheme.GetNumThreadsFor(tiling_scheme.GetTilingDimension(0)));

  KernelSupportLibrary ksl(&b_, llvm_ir::UnrollMode::kDefaultUnroll);

  ksl.For(
      "y_in_tile",
      /*start=*/thread_id_info.thread_id_y,
      /*end=*/
      tile_dimensions[0],
      /*step=*/num_threads_y, [&](llvm::Value* y_loc) {
        auto unroll_inner_tile_loop = [&](bool check_x_tile_bounds) {
          return EmitXTileLoop(thread_id_info, tile_origin_index, tiling_scheme,
                               check_x_tile_bounds, y_loc, tile_dimensions, &b_,
                               &emit_elem_function);
        };

        // Only take this path when we unroll in a way vectorizable by
        // LLVM. Special case when the tile doesn't fit completely for even
        // row size. For odd row size every other row isn't aligned to the
        // vectorized size, so it can't be vectorized by LLVM.
        if (tiling_scheme.GetIndexingOrder() == kStridedIndexingX) {
          ksl.If(
              "is_full_tile",
              b_.CreateICmpEQ(
                  constant(tiling_scheme.GetBlockTileSizeFor(kDimX)),
                  tile_dimensions[1]),
              [&] { unroll_inner_tile_loop(/*check_x_tile_bounds=*/false); },
              [&] { unroll_inner_tile_loop(/*check_x_tile_bounds=*/true); });
        } else {
          unroll_inner_tile_loop(/*check_x_tile_bounds=*/true);
        }
      });
}

static IrArray::Index GetUnnormalizedIndex(
    const IrArray::Index& normalized_shape_index,
    const Shape& unnormalized_shape, llvm::IRBuilder<>* b_,
    absl::Span<const int64_t> dims_in_elems) {
  CHECK_EQ(normalized_shape_index.size(), 3);
  // If the normalization only add a new dimensions of size 1,
  // generate simpler indexing. LLVM doesn't always simplify the more
  // complicated indexing and this prevents it from vectorizing some
  // cases. We do this only for major_to_minor memory layout.
  if (unnormalized_shape.rank() == 2 && unnormalized_shape.has_layout() &&
      unnormalized_shape.dimensions()[0] == normalized_shape_index.dims()[1] &&
      unnormalized_shape.dimensions()[1] == normalized_shape_index.dims()[2] &&
      unnormalized_shape.layout().minor_to_major(1) == 0) {
    CHECK_EQ(normalized_shape_index.dims()[0], 1);
    auto multidim = normalized_shape_index.multidim();
    return IrArray::Index({multidim[1], multidim[2]}, unnormalized_shape,
                          normalized_shape_index.GetType());
  }
  if (unnormalized_shape.rank() == 2 && unnormalized_shape.has_layout() &&
      unnormalized_shape.dimensions()[0] == normalized_shape_index.dims()[2] &&
      unnormalized_shape.dimensions()[1] == normalized_shape_index.dims()[1] &&
      unnormalized_shape.layout().minor_to_major(1) == 1) {
    CHECK_EQ(normalized_shape_index.dims()[0], 1);
    auto multidim = normalized_shape_index.multidim();
    return IrArray::Index({multidim[2], multidim[1]}, unnormalized_shape,
                          normalized_shape_index.GetType());
  }
  llvm::Value* linear = normalized_shape_index.Linearize(dims_in_elems, b_);
  return IrArray::Index(linear, unnormalized_shape, b_);
}

static int GetNumOutputs(const Shape& shape) {
  if (shape.IsTuple()) {
    return shape.tuple_shapes_size();
  }
  return 1;
}

ReductionCodegenState IrEmitterUnnested::GenerateReductionCodegenState(
    mlir::lmhlo::FusionOp fusion, const ReductionCodegenInfo& reduction_info,
    absl::Span<const HloReduceInstruction* const> reduce_instr_index_group,
    FusedIrEmitter& fused_emitter) {
  ReductionCodegenState reduction_codegen_state(reduction_info);
  VLOG(10) << "Emit prologue for reduction: " << llvm_ir::DumpToString(fusion);

  for (const HloReduceInstruction* reduce_hlo : reduce_instr_index_group) {
    int num_partial_results = reduction_codegen_state.GetNumPartialResults();
    for (int op_result_idx = 0;
         op_result_idx < GetNumOutputs(reduce_hlo->shape()); op_result_idx++) {
      Shape result_shape = reduce_hlo->shape().IsTuple()
                               ? reduce_hlo->shape().tuple_shapes(op_result_idx)
                               : reduce_hlo->shape();

      llvm::Type* element_type =
          llvm_ir::PrimitiveTypeToIrType(result_shape.element_type(), module_);
      llvm::AllocaInst* reduction_input_address =
          llvm_ir::EmitAllocaAtFunctionEntry(element_type,
                                             "reduction_input_address", &b_);

      llvm::AllocaInst* partial_result_address =
          llvm_ir::EmitAllocaAtFunctionEntryWithCount(
              element_type, /*element_count=*/b_.getInt32(num_partial_results),
              "partial_reduction_result", &b_);

      const HloInstruction* init_value =
          reduce_hlo->init_values()[op_result_idx];

      // Initialize the partial result with the initial value of the reduction.
      llvm::Value* init_ir_value = (*fused_emitter.GetGenerator(*init_value))(
                                       IrArray::Index(b_.getInt32Ty()))
                                       .value();

      for (int i = 0; i < num_partial_results; ++i) {
        b_.CreateStore(init_ir_value,
                       InBoundsGEP(partial_result_address->getAllocatedType(),
                                   partial_result_address, {b_.getInt32(i)}));
      }

      const TilingScheme& tiling_scheme =
          reduction_codegen_state.GetTilingScheme();
      int64_t num_threads_x = tiling_scheme.GetNumThreadsFor(kDimX);
      llvm::GlobalVariable* shared_cache = [&]() -> llvm::GlobalVariable* {
        if (reduction_codegen_state.IsRowReduction()) {
          // Multi-row reductions do not use shared memory.
          if (RowReductionGetRowsPerWarp(tiling_scheme.GetDimsInElems()[2]) >
              1) {
            return nullptr;
          }
          // Allocate __shared__
          // cache[num_partial_results][num_warps][scaling_factor].
          CHECK_EQ(tiling_scheme.GetNumThreadsPerBlock() % WarpSize(), 0);
          int num_warps = tiling_scheme.GetNumThreadsPerBlock() / WarpSize();
          return AllocateShared(tiling_scheme, element_type,
                                {num_partial_results, num_warps},
                                "shared_cache");
        } else {
          // Allocate __shared__
          // cache[num_partial_results][num_threads][num_threads + 1], where
          // num_threads == num_threads_x == num_threads_y.  The "+1" is used to
          // avoid bank conflicts.
          CHECK_EQ(num_threads_x, tiling_scheme.GetNumThreadsFor(kDimY));
          return AllocateShared(
              tiling_scheme, element_type,
              {num_partial_results, num_threads_x, num_threads_x + 1},
              "shared_cache");
        }
      }();

      llvm_ir::ElementGenerator input_gen =
          *fused_emitter.GetGenerator(*reduce_hlo->inputs()[op_result_idx]);
      reduction_codegen_state.SetCalculationStateFor(
          {shared_cache, init_ir_value, partial_result_address,
           reduction_input_address, input_gen},
          reduce_hlo, op_result_idx);
    }
  }

  return reduction_codegen_state;
}

void IrEmitterUnnested::EmitFullWarpShuffleDownLoopForReduce(
    const HloComputation* reducer,
    absl::Span<TypedPointer const> partial_result_addresses,
    int threads_per_block, int num_results_per_warp) {
  // This only works when the block size is a multiple of 32 threads.

  // We check this here as a mistake in the number of threads per
  // block is very hard to detect.
  CHECK_EQ(threads_per_block % 32, 0);
  CHECK_EQ(WarpSize() % num_results_per_warp, 0);

  for (int distance = 16 / num_results_per_warp; distance >= 1; distance /= 2) {
    absl::InlinedVector<llvm::Value*, 2> reduction_params;

    for (auto acc : partial_result_addresses) {
      reduction_params.push_back(acc.first);
    }

    for (auto [partial_result_address, element_type] :
         partial_result_addresses) {
      int bit_width = llvm_ir::GetSizeInBits(element_type);
      llvm::Value* result_from_other_lane = llvm_ir::EmitAllocaAtFunctionEntry(
          element_type, "result_from_other_lane", &b_);

      reduction_params.push_back(result_from_other_lane);

      // Bitcast cannot be applied to aggregate types (even packed ones), so
      // we bitcast addresses of load/store to intN* of the same bit-width.
      llvm::Type* shuffled_value_type =
          element_type->isStructTy() ? b_.getIntNTy(bit_width) : element_type;
      auto convert_pointer_for_shuffle = [&](llvm::Value* ptr) {
        return b_.CreatePointerBitCastOrAddrSpaceCast(
            ptr, shuffled_value_type->getPointerTo());
      };

      llvm::Value* partial_result =
          b_.CreateLoad(shuffled_value_type,
                        convert_pointer_for_shuffle(partial_result_address),
                        "partial_reduction_result");
      b_.CreateStore(
          EmitFullWarpShuffleDown(partial_result, b_.getInt32(distance), &b_),
          convert_pointer_for_shuffle(result_from_other_lane));
    }

    StatusOr<std::vector<llvm::Value*>> returned_scalars =
        ComputeNestedElementFromAddrs(*reducer, reduction_params);
    TF_CHECK_OK(returned_scalars.status());

    for (int i = 0; i < returned_scalars->size(); i++) {
      b_.CreateStore(/*Val=*/returned_scalars->at(i),
                     /*Ptr=*/partial_result_addresses[i].first);
    }
  }
}

llvm::Value* IrEmitterUnnested::GetOutputAddressForReduction(
    int partial_result_idx, llvm::Type* index_ty,
    const ReductionCodegenState& reduction_codegen_state,
    const TilingKernelInfo& tiling_kernel_info,
    const IrEmitterUnnested::ReductionOutputMap& output_arrays,
    const HloReduceInstruction* reduction, int output_idx) {
  auto constant = [&](uint64_t c) -> llvm::Constant* {
    return llvm::ConstantInt::get(index_ty, c);
  };

  const TilingScheme& tiling_scheme = reduction_codegen_state.GetTilingScheme();
  const ThreadIdInfo& thread_id_info = tiling_kernel_info.thread_id_info;

  IrArray::Index start_offset = [&] {
    llvm::Value* x_loc = thread_id_info.thread_id_x;
    llvm::Value* y_loc = thread_id_info.thread_id_y;
    if (!reduction_codegen_state.IsRowReduction()) {
      std::swap(x_loc, y_loc);
    }
    llvm::Value* start_offset_x =
        GetStartOffsetX(tiling_scheme, x_loc, index_ty, &b_);
    return tiling_kernel_info.tile_origin.AddOffsetToDim(y_loc, kDimY, &b_)
        .AddOffsetToDim(start_offset_x, kDimX, &b_);
  }();

  const IrArray& output_array = output_arrays.at(reduction)[output_idx];
  const Shape& operand_shape = reduction->inputs()[output_idx]->shape();
  Shape reduction_kept_element_shape =
      ShapeUtil::DeleteDimensions(reduction->dimensions(), operand_shape);

  // Given the IrArray index of a reduction input, returns the linear address of
  // the reduction output as if the reduction were going to keep the input shape
  // with the dimensions being reduced moved.
  llvm::Value* untransposed_output_linear_address = [&] {
    const llvm_ir::IrArray::Index index =
        start_offset.AddOffsetToDim(constant(partial_result_idx), kDimX, &b_);
    if (reduction_codegen_state.IsRowReduction()) {
      // For row-reduction, y-coordinate determines which row we write into.
      return index[kDimY];
    }
    // For column reduction, we get the transposed address.
    absl::Span<const int64_t> dims_in_elem = tiling_scheme.GetDimsInElems();
    llvm::Value* x_dim_size =
        index.GetConstantWithIndexType(dims_in_elem[kDimX]);
    llvm::Value* x_block_offset = b_.CreateMul(index[kDimZ], x_dim_size);
    return b_.CreateAdd(x_block_offset, index[kDimX]);
  }();

  // A reduction is allowed to transpose its output.  For example, suppose
  // we are reducing the second dimension of f32[10,20,30]{3,2,1}.  We are
  // allowed to produce as output either f32[10,30]{1,0} (no transpose) or
  // f32[10,30]{0,1} (transposing the two output dims).
  //
  // At this point in the function we have a "partial sum" of input elements
  // (stored in partial_result_addresses), and we need to accumulate it into
  // the correct output element.
  IrArray::Index element_index(
      /*linear=*/untransposed_output_linear_address,
      reduction_kept_element_shape, &b_);
  IrArray::Index output_index(element_index.multidim(), output_array.GetShape(),
                              element_index.GetType());

  return output_array.EmitArrayElementAddress(output_index, &b_,
                                              "output_element_address");
}

llvm::Value* IrEmitterUnnested::EmitBlockId(int32_t num_blocks,
                                            llvm::Type* index_ty) {
  llvm::Value* block_id = gpu::EmitCallToTargetIntrinsic(
      gpu::TargetIntrinsicID::kBlockIdx, {}, {}, &b_);
  if (num_blocks != 0) {
    llvm_ir::AddRangeMetadata(0, num_blocks,
                              llvm::cast<llvm::Instruction>(block_id));
  }
  llvm::Value* linear_block_id =
      b_.CreateIntCast(block_id, index_ty, /*isSigned=*/true, "block.id.x");
  return linear_block_id;
}

void IrEmitterUnnested::EmitPrintfWithThreadId(
    absl::string_view fmt, absl::Span<llvm::Value* const> arguments,
    std::optional<int64_t> thread_id_filter,
    std::optional<int64_t> block_id_filter) {
  llvm::Value* thread_id = EmitThreadId(
      /*threads_per_block=*/1024, b_.getInt32Ty());
  llvm::Value* block_id = EmitBlockId(0, b_.getInt32Ty());
  std::vector<llvm::Value*> updated_arguments = {thread_id, block_id};
  updated_arguments.insert(updated_arguments.end(), arguments.begin(),
                           arguments.end());
  llvm::Value* constraint = b_.getTrue();
  if (thread_id_filter) {
    constraint = b_.CreateAnd(
        constraint, b_.CreateICmpEQ(thread_id, b_.getInt32(*thread_id_filter)));
  }
  if (block_id_filter) {
    constraint = b_.CreateAnd(
        constraint, b_.CreateICmpEQ(block_id, b_.getInt32(*block_id_filter)));
  }
  KernelSupportLibrary ksl(&b_, llvm_ir::UnrollMode::kDefaultUnroll);
  ksl.If(constraint, [&] {
    xla::gpu::EmitPrintf(absl::StrCat("[TID=%d,BID=%d] ", fmt, "\n"),
                         updated_arguments, &b_);
  });
}

void IrEmitterUnnested::EmitPrintfForIndex(
    absl::string_view fmt, const llvm_ir::IrArray::Index& index,
    std::optional<int64_t> thread_id_filter,
    std::optional<int64_t> block_id_filter) {
  EmitPrintfWithThreadId(
      absl::StrCat(fmt, " : ",
                   absl::StrJoin(index.multidim(), ", ",
                                 [&](std::string* out, llvm::Value* v) {
                                   absl::StrAppend(out, "%d");
                                 })),
      index.multidim(), thread_id_filter, block_id_filter);
}

llvm::Value* IrEmitterUnnested::CastSharedToGlobal(llvm::Value* input,
                                                   llvm::Type* element_type,
                                                   llvm::Twine name) {
  return b_.CreateAddrSpaceCast(input,
                                llvm::PointerType::get(element_type,
                                                       /*AddressSpace=*/0),
                                name);
}

void IrEmitterUnnested::WriteReductionOutput(
    llvm::Type* index_ty, const ReductionCodegenState& reduction_codegen_state,
    const TilingKernelInfo& tiling_kernel_info,
    const ReductionOutputMap& output_arrays,
    const HloReduceInstruction* reduction, int partial_result_idx,
    const absl::Span<TypedPointer const> values) {
  const HloComputation* reducer = reduction->to_apply();
  for (auto [oidx, typed_ptr] : llvm::enumerate(values)) {
    auto [output_ptr, type] = typed_ptr;
    llvm::Value* output_address = GetOutputAddressForReduction(
        partial_result_idx, index_ty, reduction_codegen_state,
        tiling_kernel_info, output_arrays, reduction, oidx);
    if (reduction_codegen_state.IsRaceFree()) {
      b_.CreateStore(b_.CreateLoad(type, output_ptr, "output"), output_address);
    } else {
      CHECK_EQ(values.size(), 1);
      TF_CHECK_OK(EmitAtomicOperationForNestedComputation(
          *reducer, output_address, output_ptr, type));
    }
  }
}

void IrEmitterUnnested::EmitReductionOutputForRowReduction(
    const TilingKernelInfo& tiling_kernel_info,
    const ReductionCodegenState& reduction_codegen_state, llvm::Type* index_ty,
    const ReductionOutputMap& output_arrays,
    const HloReduceInstruction* reduction, int partial_result_idx) {
  const HloComputation* reducer = reduction->to_apply();
  const auto& thread_id_info = tiling_kernel_info.thread_id_info;
  auto constant = [&](uint64_t c) -> llvm::Constant* {
    return llvm::ConstantInt::get(index_ty, c);
  };
  auto is_zero = [&](llvm::Value* value) {
    return b_.CreateICmpEQ(value, constant(0));
  };

  int num_outputs = reducer->num_parameters() / 2;
  const TilingScheme& tiling_scheme = reduction_codegen_state.GetTilingScheme();
  absl::InlinedVector<TypedPointer, 2> current_outputs;
  for (int output_idx = 0; output_idx < num_outputs; output_idx++) {
    const ReductionCodegenState::ReductionCalculationState& state =
        reduction_codegen_state.GetCalculationStateFor(reduction, output_idx);
    current_outputs.push_back(
        {InBoundsGEP(state.partial_result_address->getAllocatedType(),
                     state.partial_result_address,
                     {constant(partial_result_idx)}, "current_output"),
         state.partial_result_address->getAllocatedType()});
  }

  int reduced_dimension_size = tiling_scheme.GetDimsInElems()[2];
  int num_rows_per_warp = RowReductionGetRowsPerWarp(reduced_dimension_size);
  EmitFullWarpShuffleDownLoopForReduce(
      reducer, absl::MakeSpan(current_outputs),
      tiling_scheme.GetNumThreadsPerBlockPhysical(), num_rows_per_warp);

  KernelSupportLibrary ksl(&b_);
  llvm::Value* warp_id =
      b_.CreateUDiv(thread_id_info.thread_id_x, constant(WarpSize()));

  auto emit_write_output = [&](llvm::Value* write_condition,
                               const absl::Span<TypedPointer const> values) {
    ksl.If("reduction_write_output", write_condition, [&] {
      WriteReductionOutput(index_ty, reduction_codegen_state,
                           tiling_kernel_info, output_arrays, reduction,
                           partial_result_idx, values);
    });
  };

  if (num_rows_per_warp > 1) {
    llvm::Value* is_writing_thread = is_zero(b_.CreateAnd(
        thread_id_info.thread_id_x, constant(reduced_dimension_size - 1)));
    emit_write_output(is_writing_thread, current_outputs);
    return;
  }

  ksl.If("intra_warp_reduce_write", is_zero(thread_id_info.lane_id), [&] {
    for (int oidx = 0; oidx < num_outputs; oidx++) {
      const ReductionCodegenState::ReductionCalculationState& state =
          reduction_codegen_state.GetCalculationStateFor(reduction, oidx);
      llvm::Value* shmem_output_addr = thread_id_info.GEPIntoSharedMemory(
          &b_, state.shared_cache, {constant(partial_result_idx), warp_id});
      Store(Load(current_outputs[oidx].second, current_outputs[oidx].first),
            shmem_output_addr);
    }
  });

  // TODO(cheshire): Don't we want to sync it once for everything in the
  // output? Not once per each?
  EmitSyncThreads();
  ksl.If("inter_warp_reduce", is_zero(warp_id), [&] {
    absl::InlinedVector<TypedPointer, 2> selected_values;
    for (int oidx = 0; oidx < num_outputs; oidx++) {
      const ReductionCodegenState::ReductionCalculationState& state =
          reduction_codegen_state.GetCalculationStateFor(reduction, oidx);
      llvm::Value* block_accum_addr = thread_id_info.GEPIntoSharedMemory(
          &b_, state.shared_cache,
          {constant(partial_result_idx), thread_id_info.lane_id});

      llvm::Type* element_type =
          state.partial_result_address->getAllocatedType();

      /* Insure initial value address is in generic, not scratch. */
      llvm::Value* initial_value_addr =
          CastSharedToGlobal(llvm_ir::EmitAllocaAtFunctionEntry(
                                 element_type, "initial_value_addr", &b_),
                             element_type);
      b_.CreateStore(state.initial_value, initial_value_addr);

      llvm::Value* warp_exists = b_.CreateICmpULT(
          thread_id_info.thread_id_x,
          constant(tiling_scheme.GetNumThreadsFor(kDimX) / WarpSize()));

      llvm::Value* selected_value =
          b_.CreateSelect(warp_exists, block_accum_addr, initial_value_addr);

      selected_values.push_back({selected_value, element_type});
    }

    // If only one warp is present in the block, then we don't need inter-warp
    // reduction.
    // TODO(b/241414088) If only warp is present, then inter-warp communication
    // using shared memory and synchronization using barrier is also unnecessary
    // and should be removed.
    if (tiling_scheme.GetNumThreadsPerBlock() > WarpSize()) {
      EmitFullWarpShuffleDownLoopForReduce(
          reducer, absl::MakeSpan(selected_values),
          tiling_scheme.GetNumThreadsPerBlock());
    }

    emit_write_output(is_zero(thread_id_info.thread_id_x), selected_values);
  });
}

void IrEmitterUnnested::EmitReductionOutputForColumnReduction(
    const TilingKernelInfo& tiling_kernel_info,
    const ReductionCodegenState& reduction_codegen_state, llvm::Type* index_ty,
    const ReductionOutputMap& output_arrays,
    const HloReduceInstruction* reduction, int partial_result_idx) {
  KernelSupportLibrary ksl(&b_);
  const HloComputation* reducer = reduction->to_apply();
  const auto& thread_id_info = tiling_kernel_info.thread_id_info;

  auto constant = [&](uint64_t c) -> llvm::Constant* {
    return llvm::ConstantInt::get(index_ty, c);
  };
  auto is_zero = [&](llvm::Value* value) {
    return b_.CreateICmpEQ(value, constant(0));
  };
  const TilingScheme& tiling_scheme = reduction_codegen_state.GetTilingScheme();
  int num_outputs = reducer->num_parameters() / 2;

  // Store the transpose in shared memory.
  for (int output_idx = 0; output_idx < num_outputs; output_idx++) {
    const ReductionCodegenState::ReductionCalculationState& state =
        reduction_codegen_state.GetCalculationStateFor(reduction, output_idx);
    llvm::GlobalVariable* shared_cache = state.shared_cache;
    llvm::AddrSpaceCastInst* shmem_output_addr =
        llvm::cast<llvm::AddrSpaceCastInst>(thread_id_info.GEPIntoSharedMemory(
            &b_, shared_cache,
            {constant(partial_result_idx), thread_id_info.thread_id_x,
             thread_id_info.thread_id_y},
            "shmem_output_address"));
    llvm::Value* current_output =
        InBoundsGEP(state.partial_result_address->getAllocatedType(),
                    state.partial_result_address,
                    {constant(partial_result_idx)}, "current_output");

    llvm::Value* current_output_value =
        Load(state.partial_result_address->getAllocatedType(), current_output);
    b_.CreateStore(current_output_value, shmem_output_addr);
  }

  EmitSyncThreads();

  // Get transposed element from shared memory.
  absl::InlinedVector<TypedPointer, 2> shmem_transposed_addrs;
  for (int output_idx = 0; output_idx < num_outputs; output_idx++) {
    const ReductionCodegenState::ReductionCalculationState& state =
        reduction_codegen_state.GetCalculationStateFor(reduction, output_idx);
    llvm::AddrSpaceCastInst* shmem_transposed_addr =
        llvm::cast<llvm::AddrSpaceCastInst>(thread_id_info.GEPIntoSharedMemory(
            &b_, state.shared_cache,
            {constant(partial_result_idx), thread_id_info.thread_id_y,
             thread_id_info.thread_id_x},
            "shmem_transposed_addr"));
    shmem_transposed_addrs.push_back(
        {shmem_transposed_addr, llvm::cast<llvm::GetElementPtrInst>(
                                    shmem_transposed_addr->getPointerOperand())
                                    ->getResultElementType()});
  }

  EmitFullWarpShuffleDownLoopForReduce(reducer,
                                       absl::MakeSpan(shmem_transposed_addrs),
                                       tiling_scheme.GetNumThreadsPerBlock());

  // Some warps in the block are completely outside of the bound of the
  // tensor, so they should not write any output at all.
  llvm::Value* has_output =
      b_.CreateAnd(b_.CreateICmpULT(GetStartOffsetX(tiling_scheme,
                                                    thread_id_info.thread_id_y,
                                                    index_ty, &b_),
                                    tiling_kernel_info.output_tile_bounds[1]),
                   b_.CreateICmpULT(thread_id_info.thread_id_x,
                                    tiling_kernel_info.output_tile_bounds[0]));

  ksl.If("reduction_write_output",
         b_.CreateAnd(has_output, is_zero(thread_id_info.lane_id)), [&] {
           WriteReductionOutput(index_ty, reduction_codegen_state,
                                tiling_kernel_info, output_arrays, reduction,
                                partial_result_idx, shmem_transposed_addrs);
         });
}

llvm::Value* IrEmitterUnnested::EmitThreadId(int64_t threads_per_block,
                                             llvm::Type* index_ty) {
  // Calculate (y, x) coordinates respectively in the 2D view of thread block,
  // defined by (num_thread_y, num_thread_x) from thread_id.
  llvm::CallInst* thread_id_raw = gpu::EmitCallToTargetIntrinsic(
      gpu::TargetIntrinsicID::kThreadIdx, {}, {}, &b_);
  llvm_ir::AddRangeMetadata(0, threads_per_block, thread_id_raw);
  return b_.CreateIntCast(thread_id_raw, index_ty,
                          /*isSigned=*/true, "thread.id.x");
}

StatusOr<IrEmitterUnnested::ThreadIdInfo> IrEmitterUnnested::EmitThreadIdInfo(
    const TilingScheme& tiling_scheme, llvm::Type* index_ty) {
  auto constant = [&](uint64_t c) -> llvm::Constant* {
    return llvm::ConstantInt::get(index_ty, c);
  };
  llvm::Value* thread_id_physical =
      EmitThreadId(tiling_scheme.GetNumThreadsPerBlockPhysical(), index_ty);
  int64_t num_blocks = tiling_scheme.GetNumberOfBlocksPhysical();
  if (num_blocks > (int64_t)std::numeric_limits<uint32_t>::max()) {
    return FailedPrecondition(
        "Number of physical blocks (%d) does not fit in an i32 in tiling "
        "scheme: %s",
        num_blocks, tiling_scheme.ToString());
  }
  llvm::Value* block_id_physical = EmitBlockId(num_blocks, index_ty);

  // Wait this will break coalescing.
  llvm::Value* thread_id_logical = b_.CreateURem(
      thread_id_physical, constant(tiling_scheme.GetNumThreadsPerBlock()));
  llvm::Value* scaling = b_.CreateUDiv(
      thread_id_physical, constant(tiling_scheme.GetNumThreadsPerBlock()));
  llvm::Value* block_id_logical = b_.CreateAdd(
      b_.CreateMul(block_id_physical,
                   constant(tiling_scheme.GetThreadIdScalingFactor())),
      scaling);

  llvm::Value* num_threads_x_v =
      constant(tiling_scheme.GetNumThreadsFor(kDimX));

  llvm::Value* block_exists = b_.CreateICmpULT(
      block_id_logical, constant(tiling_scheme.GetNumberOfBlocks()));
  llvm_ir::EmitEarlyReturn(block_exists, &b_);
  return {{thread_id_logical,
           /*thread_id_x=*/
           b_.CreateURem(thread_id_logical, num_threads_x_v, "thread_id.x"),
           /*thread_id_y=*/
           b_.CreateUDiv(thread_id_logical, num_threads_x_v, "thread_id.y"),
           /*lane_id=*/
           b_.CreateURem(thread_id_logical, constant(WarpSize()), "lane_id"),
           /*block_id=*/block_id_logical,
           /*scaling=*/scaling}};
}

StatusOr<IrEmitterUnnested::TilingKernelInfo>
IrEmitterUnnested::EmitTilingKernel(
    const TilingScheme& tiling_scheme, llvm::Type* index_ty,
    const TileElementGenerator& tile_element_generator) {
  absl::Span<const int64_t> dims_in_elems = tiling_scheme.GetDimsInElems();
  Vector3 dims_in_blocks = tiling_scheme.GetDimsInBlocks();
  auto constant = [&](uint64_t c) -> llvm::Constant* {
    return llvm::ConstantInt::get(index_ty, c);
  };

  TF_ASSIGN_OR_RETURN(ThreadIdInfo thread_id_info,
                      EmitThreadIdInfo(tiling_scheme, index_ty));

  KernelSupportLibrary ksl(&b_, llvm_ir::UnrollMode::kDefaultUnroll);

  int64_t non_tiling_dimension =
      tiling_scheme.GetTilingDimension(0) == 1 ? kDimZ : kDimY;
  const IrArray::Index block_coords(
      thread_id_info.block_id,
      ShapeUtil::MakeShapeWithDenseLayout(
          PRED /*arbitrary*/, dims_in_blocks,
          // This layout determines the iteration order. We want the
          // non-tiling dimension to be the slowest varying dimension.
          {2, 1 - non_tiling_dimension, non_tiling_dimension}),
      &b_);

  ValueVector2 tile_dimensions;
  // Coordinate access is shifted: 0 corresponds to the first non-tiling
  // dimension and 1 corresponds to DimX.
  std::array<int64_t, 2> tiling_coords{1 - non_tiling_dimension, kDimX};
  for (int i = 0; i < 2; ++i) {
    int64_t tile_size_for_dim =
        tiling_scheme.GetBlockTileSizeFor(tiling_coords[i]);
    // Only last row or column may not have full size.
    llvm::Value* is_last =
        b_.CreateICmpEQ(block_coords[tiling_coords[i]],
                        constant(dims_in_blocks[tiling_coords[i]] - 1));
    int64_t partial_row =
        dims_in_elems[tiling_coords[i]] -
        (dims_in_blocks[tiling_coords[i]] - 1) * tile_size_for_dim;
    tile_dimensions[i] =
        b_.CreateSelect(is_last, constant(partial_row),
                        constant(tile_size_for_dim), "tile_bound");
  }

  IrArray::Index tile_origin = [&] {
    std::vector<llvm::Value*> elem_multi_index = block_coords.multidim();
    llvm::Type* index_ty = block_coords.GetType();
    for (int i = 0; i < kDimTot; ++i) {
      elem_multi_index[i] =
          b_.CreateMul(block_coords[i],
                       llvm::ConstantInt::get(
                           index_ty, tiling_scheme.GetBlockTileSizeFor(i)),
                       "tile_origin." + std::to_string(i));
    }
    return IrArray::Index(elem_multi_index, tiling_scheme.GetDimsInElems(),
                          index_ty);
  }();

  auto emit_tile = [&](const IrArray::Index& tile) {
    tile_element_generator(thread_id_info, tile, tile_dimensions);
  };

  if (tiling_scheme.GetBlockTileSizeFor(non_tiling_dimension) == 1) {
    emit_tile(tile_origin);
  } else {
    llvm::Value* starting_tile_index_for_dim =
        tile_origin[non_tiling_dimension];
    llvm::Value* block_size_for_dim =
        constant(tiling_scheme.GetBlockTileSizeFor(non_tiling_dimension));
    llvm::Value* block_id_for_dim =
        b_.CreateUDiv(starting_tile_index_for_dim, block_size_for_dim);
    llvm::Value* last_block_for_dim =
        constant(dims_in_blocks[non_tiling_dimension] - 1);
    llvm::Value* last_block_size_for_dim =
        constant(dims_in_elems[non_tiling_dimension] -
                 (dims_in_blocks[non_tiling_dimension] - 1) *
                     tiling_scheme.GetBlockTileSizeFor(non_tiling_dimension));

    llvm::Value* num_tiles_in_block =
        b_.CreateSelect(b_.CreateICmpEQ(last_block_for_dim, block_id_for_dim),
                        last_block_size_for_dim, block_size_for_dim);
    ksl.For("loop_z",
            /*start=*/constant(0),
            /*end=*/num_tiles_in_block,
            /*step=*/1, [&](llvm::Value* block_dim_induction_var) {
              IrArray::Index tile_index = tile_origin.AddOffsetToDim(
                  block_dim_induction_var, non_tiling_dimension, &b_);
              emit_tile(tile_index);
            });
  }

  return {{tile_dimensions, tile_origin, thread_id_info}};
}

llvm::CallInst* IrEmitterUnnested::EmitSyncThreads() {
  MaybeEmitFenceForAMDGPU(llvm::AtomicOrdering::SequentiallyConsistent,
                          "workgroup");
  return EmitCallToTargetIntrinsic(TargetIntrinsicID::kBarrierId, {}, {}, &b_);
}

static IrArray::Index PermuteIndex(const IrArray::Index& index,
                                   absl::Span<const int64_t> permutation) {
  return IrArray::Index{Permute(index.multidim(), permutation),
                        Permute(index.dims(), permutation), index.GetType()};
}

Status IrEmitterUnnested::EmitTransposeTile(
    mlir::lmhlo::FusionOp fusion, HloComputation* fusion_hlo,
    absl::Span<const llvm_ir::IrArray> operand_arrays,
    absl::Span<const llvm_ir::IrArray> output_arrays,
    const TilingScheme& tiling_scheme,
    const LaunchDimensions& launch_dimensions) {
  std::vector<HloInstruction*> hlo_roots = GetFusionRoots(fusion_hlo);
  const HloInstruction* first_transpose = &FindNonTrivialHero(
      **absl::c_find_if(hlo_roots, [](HloInstruction* instr) {
        return FindAnyTiledTranspose(FindNonTrivialHero(*instr));
      }));

  const Shape& transpose_in_shape = first_transpose->operand(0)->shape();
  auto first_tiled_transpose = FindAnyTiledTranspose(*first_transpose);

  // We need the following invariant:
  // For every tuple element:
  //  -> EITHER it's a kCopy: S{L} -> S{L'}
  //  -> OR it's an elementwise op of shape S{L}
  for (HloInstruction* root : hlo_roots) {
    auto tiled_transpose = FindAnyTiledTranspose(*root);
    if (tiled_transpose) {
      CHECK(*tiled_transpose == *first_tiled_transpose);
    } else {
      CHECK(ShapeUtil::IsReshapeOrTransposeBitcast(
          root->shape(), transpose_in_shape,
          /*ignore_element_type=*/true));
    }
  }

  FusedIrEmitter fused_emitter(elemental_emitter_);
  for (int i = 0; i < fusion_hlo->num_parameters(); i++) {
    llvm_ir::IrArray ir_array = operand_arrays[i];
    HloInstruction* fused_operand = fusion_hlo->parameter_instruction(i);
    fused_emitter.BindGenerator(
        *fused_operand,
        [this, ir_array, fused_operand](const llvm_ir::IrArray::Index& index) {
          return ir_array.EmitReadArrayElement(index, &b_,
                                               fused_operand->name());
        });
  }

  absl::flat_hash_map<const HloInstruction*, llvm::GlobalVariable*> tiles;
  Vector3 permutation;
  for (const auto& [tile_idx, root] : llvm::enumerate(hlo_roots)) {
    if (auto tr = FindAnyTiledTranspose(*root)) {
      permutation = tr->second;
      const HloInstruction& hero = FindNonTrivialHero(*root);
      tiles[&hero] =
          AllocateShared(tiling_scheme,
                         llvm_ir::PrimitiveTypeToIrType(
                             hero.operand(0)->shape().element_type(), module_),
                         {tiling_scheme.GetBlockTileSizeFor(permutation[kDimX]),
                          tiling_scheme.GetBlockTileSizeFor(kDimX) + 1},
                         absl::StrCat("tr_tile_", tile_idx));
    }
  }

  TileElementGenerator tile_generator = [&](const ThreadIdInfo& thread_id_info,
                                            const IrArray::Index& index,
                                            ValueVector2 tile_dimensions) {
    // Copy input parameter values to shared memory buffers:
    // tile[thread_id_y, thread_id_x] = input[index]
    // Note that tile_width and tile_height are flipped here because we
    // are reading a transposed tile.
    EmitTile(
        tiling_scheme, index, thread_id_info, tile_dimensions,
        [&](const ThreadIdInfo& thread_id_info, const IrArray::Index& index,
            llvm::Value* y_loc, llvm::Value* x_loc) {
          // Compute all extra output values before writing them. This avoids
          // overwriting aliased input/output values before all reads occurred.
          std::vector<std::tuple<IrArray, IrArray::Index, llvm::Value*>>
              scheduled_writes;

          for (const auto& [output_idx, root] : llvm::enumerate(hlo_roots)) {
            IrArray::Index input_index = GetUnnormalizedIndex(
                index, transpose_in_shape, &b_, tiling_scheme.GetDimsInElems());
            if (FindAnyTiledTranspose(*root)) {
              const HloInstruction& hero = FindNonTrivialHero(*root);
              llvm_ir::ElementGenerator input_gen =
                  *fused_emitter.GetGenerator(*hero.operand(0));
              IrArray::Index used_index = input_index;
              if (!ShapeUtil::EqualIgnoringElementType(hero.operand(0)->shape(),
                                                       transpose_in_shape)) {
                used_index = used_index.SourceIndexOfBitcast(
                    transpose_in_shape, hero.operand(0)->shape(), &b_);
              }
              llvm::Value* value = *input_gen(used_index);
              llvm::Value* addr = thread_id_info.GEPIntoSharedMemory(
                  &b_, tiles[&hero], {y_loc, x_loc});

              b_.CreateStore(value, addr);
            } else {
              IrArray::Index used_index = input_index;
              if (!ShapeUtil::EqualIgnoringElementType(root->shape(),
                                                       transpose_in_shape)) {
                used_index = used_index.SourceIndexOfBitcast(
                    transpose_in_shape, root->shape(), &b_);
              }
              llvm_ir::ElementGenerator output_gen =
                  *fused_emitter.GetGenerator(*root);
              llvm::Value* output_value = *output_gen(used_index);
              scheduled_writes.emplace_back(output_arrays[output_idx],
                                            used_index, output_value);
            }
          }

          for (const auto& [output, idx, value] : scheduled_writes) {
            output.EmitWriteArrayElement(idx, value, &b_);
          }
        });

    EmitSyncThreads();

    IrArray::Index output_tile_index = PermuteIndex(index, permutation);
    ValueVector2 transposed_tile_dimensions = {tile_dimensions[1],
                                               tile_dimensions[0]};

    EmitTile(
        tiling_scheme, output_tile_index, thread_id_info,
        transposed_tile_dimensions,
        /*emit_elem_function=*/
        [&](const ThreadIdInfo& thread_id_info,
            const llvm_ir::IrArray::Index& index, llvm::Value* y_loc,
            llvm::Value* x_loc) {
          for (const auto& [output_idx, root] : llvm::enumerate(hlo_roots)) {
            if (FindAnyTiledTranspose(*root)) {
              const HloInstruction& hero = FindNonTrivialHero(*root);

              IrArray::Index untiled_index = GetUnnormalizedIndex(
                  index, hero.shape(), &b_,
                  Permute(tiling_scheme.GetDimsInElems(), permutation));
              std::vector<llvm::Value*> idx = {x_loc, y_loc};
              llvm::Value* gep =
                  thread_id_info.GEPIntoSharedMemory(&b_, tiles[&hero], idx);
              llvm::Type* type =
                  thread_id_info.GEPIntoSharedMemoryType(tiles[&hero], idx);
              llvm::Value* loaded = b_.CreateLoad(type, gep, "tiled_buffer");

              FusedIrEmitter fused_emitter(elemental_emitter_);
              fused_emitter.BindGenerator(
                  hero, [&](const IrArray::Index& index) { return loaded; });

              // Apply codegeneration for the code after the real hero.
              TF_ASSIGN_OR_RETURN(llvm_ir::ElementGenerator gen,
                                  fused_emitter.GetGenerator(*root));

              // Both for emission and writing it should be index-as-transformed
              // by the computation.
              IrArray::Index output_index = untiled_index;
              if (root->shape() != hero.shape()) {
                output_index = output_index.SourceIndexOfBitcast(
                    hero.shape(), root->shape(), &b_);
              }
              TF_ASSIGN_OR_RETURN(llvm::Value * generated, gen(output_index));
              output_arrays[output_idx].EmitWriteArrayElement(output_index,
                                                              generated, &b_);
            }
          }
          return OkStatus();
        });
  };

  llvm::Type* index_type = GetIndexTypeForKernel(
      fusion.getOperation(), launch_dimensions.launch_bound(), &b_);
  return EmitTilingKernel(tiling_scheme, index_type, tile_generator).status();
}

namespace {

// Returns true if all the transitive users of hlo before hitting users in
// use_chain_endings are elementwise operations.
bool AreUsersElementwise(
    mlir::Value value,
    const absl::flat_hash_set<mlir::Operation*>& use_chain_endings) {
  return absl::c_all_of(value.getUsers(), [&](mlir::OpOperand use) {
    mlir::Operation* user = use.getOwner();
    return use_chain_endings.count(user) ||
           (HloInstruction::IsOpElementwise(*MhloToHloOpcode(user)) &&
            absl::c_all_of(user->getResults(),
                           [&](const mlir::OpResult result) {
                             return AreUsersElementwise(result,
                                                        use_chain_endings);
                           })

           );
  });
}

// Returns the number of fusion inputs that have the same dimension as the
// given shape, and involve in only elementwise operations.
int64_t NumInputsInvolveInOnlyElementwiseOps(
    mlir::lmhlo::FusionOp fusion, const Shape& op_shape,
    const absl::flat_hash_set<mlir::Operation*>& use_chain_endings) {
  return absl::c_count_if(
      fusion.getFusionParameters(), [&](mlir::Value parameter) {
        Shape parameter_shape = GetShape(parameter);
        return ShapeUtil::SameDimensions(op_shape, parameter_shape) &&
               AreUsersElementwise(parameter, use_chain_endings);
      });
}

// Returns the number of fusion inputs that have more elements than the given
// shape.
int64_t NumInputsWithMoreElementsThan(mlir::lmhlo::FusionOp fusion,
                                      const Shape& shape) {
  int64_t num_elements = ShapeUtil::ElementsIn(shape);
  return absl::c_count_if(
      fusion.getFusionParameters(), [&](mlir::Value parameter) {
        Shape parameter_shape = GetShape(parameter);
        return ShapeUtil::ElementsIn(parameter_shape) > num_elements;
      });
}

// The benefit of unrolling a kInput fusion that is a column reduction comes
// from the vectorization of non-reduction fusion outputs and fusion inputs.
// On the other hand, unrolling can also introduce factors that can cause
// the kernel to run slower. This routine uses a simple heuristic to estimate
// the benefit as well as the overhead of unrolling in order to decide whether
// unrolling is beneficial for the given kInput fusion.
bool IsUnrollingColumnReductionBeneficial(mlir::lmhlo::FusionOp fusion,
                                          HloComputation* fused_computation,
                                          const Shape& input_shape,
                                          int64_t num_kept_minor,
                                          bool reduction_is_race_free) {
  if (num_kept_minor % (WarpSize() * 2) != 0) {
    return false;
  }

  if (input_shape.dimensions()[input_shape.rank() - 1] < 64) {
    return false;
  }

  int64_t can_be_vectorized = 0;
  int64_t cannot_be_vectorized = 0;
  llvm::SmallVector<mlir::Operation*> fusion_roots = fusion.getFusionRoots();
  absl::flat_hash_set<mlir::Operation*> use_chain_endings;
  std::vector<HloInstruction*> hlo_roots = GetFusionRoots(fused_computation);

  for (int i = 0; i < fusion_roots.size(); i++) {
    if (!reduction_is_race_free &&
        IsReductionFromOrToContiguousDimensions(*hlo_roots[i])) {
      // Atomics cannot be vectorized.
      cannot_be_vectorized++;
    } else {
      can_be_vectorized++;
    }
    use_chain_endings.insert(fusion_roots[i]);
  }
  // Fusion inputs that have the same dimension as the reduce input and
  // only involve in elementwise operations can be vectorized.
  can_be_vectorized += NumInputsInvolveInOnlyElementwiseOps(fusion, input_shape,
                                                            use_chain_endings);
  // Fusion inputs with more elements than the reduce op input must participate
  // in non-elementwise operations and we assume that they are not vectorizable
  // for the purpose of estimating the benefit of unrolling. If the kernel is
  // unrolled even with such an assumption,  and the accesses to those inputs
  // turn out to be vectorizable, the compiler will still vectorize them.
  cannot_be_vectorized += NumInputsWithMoreElementsThan(fusion, input_shape);
  return can_be_vectorized >= cannot_be_vectorized;
}

int64_t NearestPowerOfTwo(int64_t v) {
  if (v < 0) {
    return 0;
  }
  int64_t upper = absl::bit_ceil<uint64_t>(v);
  int64_t lower = upper >> 1;
  return upper - v < v - lower ? upper : lower;
}

}  // namespace

// Returns primitive bitwidth for shape of the value.
static int GetPrimitiveBitwidth(mlir::Value i) {
  // TODO(timshen): may not be efficient.
  return primitive_util::BitWidth(GetShape(i).element_type());
}

// Experimentally determined values to achieve optimal number of
// bytes-in-flight. With a bound of #warps/SM which can be concurrently
// scheduled, for small reduced values it can be hard to achieve optimal
// number of bytes-in-flight. In order to address it, we increase the # of
// threads/block (physically, while keeping logical mapping the same), which
// allows larger # of bytes-in-flight.
static int CalculateVirtualThreadScalingFactorForReduction(
    const ReductionDimensions& reduction_dimensions,
    const se::CudaComputeCapability& cc) {
  int64_t dimx = reduction_dimensions.dimensions[kDimX];
  if (reduction_dimensions.is_row_reduction && dimx <= 128) {
    int rows_per_warp = RowReductionGetRowsPerWarp(dimx);
    if (cc.IsAtLeast(se::CudaComputeCapability::AMPERE)) {
      return rows_per_warp * 3;
    }
    return rows_per_warp * 5;
  }
  return 1;
}

llvm::Type* IrEmitterUnnested::ThreadIdInfo::GEPIntoSharedMemoryType(
    llvm::GlobalVariable* shared,
    absl::Span<llvm::Value* const> idx_major_to_minor) const {
  std::vector<llvm::Value*> idxs_scaled;
  idxs_scaled.push_back(llvm::ConstantInt::get(scaling->getType(), 0));
  idxs_scaled.push_back(scaling);
  idxs_scaled.insert(idxs_scaled.end(), idx_major_to_minor.begin(),
                     idx_major_to_minor.end());
  return llvm::GetElementPtrInst::getIndexedType(shared->getValueType(),
                                                 idxs_scaled);
}

llvm::Value* IrEmitterUnnested::ThreadIdInfo::GEPIntoSharedMemory(
    llvm::IRBuilder<>* b, llvm::GlobalVariable* shared,
    absl::Span<llvm::Value* const> idx_major_to_minor,
    const llvm::Twine& name) const {
  std::vector<llvm::Value*> idxs_scaled;
  idxs_scaled.push_back(llvm::ConstantInt::get(scaling->getType(), 0));
  idxs_scaled.push_back(scaling);
  idxs_scaled.insert(idxs_scaled.end(), idx_major_to_minor.begin(),
                     idx_major_to_minor.end());
  llvm::Value* gep =
      b->CreateInBoundsGEP(shared->getValueType(), shared, idxs_scaled, name);

  llvm::PointerType* pointer_in_addressspace =
      llvm::PointerType::getWithSamePointeeType(
          llvm::cast<llvm::PointerType>(gep->getType()), /*AddressSpace=*/0);

  // __shared__ memory uses a different address space, so we cast it to
  // global address space before writing or reading.
  return b->CreateAddrSpaceCast(gep, pointer_in_addressspace);
}

llvm::GlobalVariable* IrEmitterUnnested::AllocateShared(
    const TilingScheme& tiling_scheme, llvm::Type* element_type,
    absl::Span<int64_t const> dimensions_major_to_minor,
    absl::string_view buffer_name) {
  CHECK(!dimensions_major_to_minor.empty());
  llvm::Type* array_type = nullptr;
  for (int i = dimensions_major_to_minor.size() - 1; i >= 0; i--) {
    // Iterate in minor-to-major order.
    int64_t dim = dimensions_major_to_minor[i];
    if (!array_type) {
      array_type = llvm::ArrayType::get(element_type, dim);
    } else {
      array_type = llvm::ArrayType::get(array_type, dim);
    }
  }
  array_type = llvm::ArrayType::get(array_type,
                                    tiling_scheme.GetThreadIdScalingFactor());
  return llvm_ir::AllocateSharedMemoryTile(b_.GetInsertBlock()->getModule(),
                                           array_type, buffer_name);
}

// Whether the reduction can be vectorized.
static bool CanVectorizeReduction(
    se::CudaComputeCapability cc, mlir::lmhlo::FusionOp fusion,
    HloComputation* fused_computation,
    const ReductionDimensions& reduction_dimensions, int num_threads_x,
    Vector3 reduction_tiling, const Shape& input_shape,
    bool reduction_is_race_free) {
  if (!reduction_dimensions.is_row_reduction) {
    return IsUnrollingColumnReductionBeneficial(
        fusion, fused_computation, input_shape,
        reduction_dimensions.dimensions[kDimX], reduction_is_race_free);
  }

  if (reduction_dimensions.dimensions[kDimX] % 2 != 0 ||
      MayPreventVectorization(fusion)) {
    return false;
  }

  // Enabling vectorization if number of threads is <= warpsize leads to half or
  // more of the threads not doing any work.
  if (reduction_dimensions.is_row_reduction && num_threads_x <= WarpSize()) {
    return false;
  }

  if (cc.IsAtLeast(se::CudaComputeCapability::VOLTA)) {
    return true;
  }

  int smallest_input_dtype_bits = std::numeric_limits<int>::max();
  for (mlir::Value operand : fusion.getInputBuffers()) {
    smallest_input_dtype_bits =
        std::min(GetPrimitiveBitwidth(operand), smallest_input_dtype_bits);
  }
  if (cc.IsAtLeast(se::CudaComputeCapability::PASCAL_)) {
    return smallest_input_dtype_bits <= 32 &&
           reduction_dimensions.dimensions[kDimX] %
                   (reduction_tiling[2] * num_threads_x) ==
               0;
  }
  return false;
}

// Projected shmem usage of reduction fusion.
static int64_t ProjectedShmemUsageBytes(
    const ReductionDimensions& reduction_dimensions,
    const std::vector<std::vector<HloInstruction*>>& instr_index_groups) {
  int64_t out = 0;
  // Different groups are computed in parallel on different blocks, so they are
  // not sharing the shmem budget. The overall usage is given by the largest
  // one.
  for (const std::vector<HloInstruction*>& group : instr_index_groups) {
    int64_t sum = 0;
    for (HloInstruction* root : group) {
      if (IsReductionFromOrToContiguousDimensions(*root)) {
        sum += SharedMemoryUsage(*root);
      }
    }
    out = std::max(out, sum);
  }
  return out;
}

StatusOr<ReductionCodegenInfo> IrEmitterUnnested::ComputeReductionCodegenInfo(
    mlir::lmhlo::FusionOp fusion, HloComputation* fused_computation,
    HloInstruction* first_reduce,
    const std::vector<std::vector<HloInstruction*>>& instr_index_groups) {
  Shape input_shape = first_reduce->operand(0)->shape();
  ReductionDimensions reduction_dimensions =
      GetReductionKindAndContiguousComponents(*first_reduce);
  VLOG(10) << "is_row_reduction " << reduction_dimensions.is_row_reduction
           << " " << reduction_dimensions.dimensions[0] << " "
           << reduction_dimensions.dimensions[1] << " "
           << reduction_dimensions.dimensions[2];
  Vector3 reduction_tiling = GetReductionTiling(reduction_dimensions);

  int64_t num_threads_y =
      reduction_dimensions.is_row_reduction ? 1 : WarpSize();
  int64_t num_threads_x = [&] {
    if (reduction_dimensions.is_row_reduction) {
      if (RowReductionGetRowsPerWarp(reduction_dimensions.dimensions[2]) > 1) {
        return reduction_dimensions.dimensions[2];
      }
      // Use 512 as default block size (threads per block) for row reductions.
      // For multi-output fusions, reduce the block size further to decrease
      // register pressure when multiple outputs are computed by each thread.
      int64_t fan_out = fusion.getFusionRoots().size();
      int64_t max_block_size =
          std::max(MinThreadsXRowReduction(),
                   static_cast<int64_t>(512LL / NearestPowerOfTwo(fan_out)));
      return std::min(max_block_size,
                      RoundUpTo(CeilOfRatio(reduction_dimensions.dimensions[2],
                                            reduction_tiling[2]),
                                WarpSize()));
    }
    return WarpSize();
  }();

  se::CudaComputeCapability cc = ir_emitter_context_->cuda_compute_capability();

  int smallest_input_dtype_bits = std::numeric_limits<int>::max();
  for (mlir::Value operand : fusion.getInputBuffers()) {
    smallest_input_dtype_bits =
        std::min(GetPrimitiveBitwidth(operand), smallest_input_dtype_bits);
  }

  TilingScheme::IndexingOrder indexing_order =
      reduction_dimensions.is_row_reduction ? kStridedIndexingX
                                            : kLinearIndexingX;
  int64_t shmem_usage =
      ProjectedShmemUsageBytes(reduction_dimensions, instr_index_groups);
  const int64_t shmem_budget =
      ir_emitter_context_->gpu_device_info().shared_memory_per_block;
  bool reduction_is_race_free = ReductionIsRaceFree(reduction_dimensions);
  bool vectorize =
      // Vectorization might cause us to run out of budget.
      (shmem_usage * 2 <= shmem_budget) &&
      CanVectorizeReduction(cc, fusion, fused_computation, reduction_dimensions,
                            num_threads_x, reduction_tiling, input_shape,
                            reduction_is_race_free);
  int vector_size = vectorize ? 2 : 1;

  int num_partial_results = 1;
  if (!reduction_dimensions.is_row_reduction && vectorize) {
    if (smallest_input_dtype_bits <= 32) {
      // Make sure to use all the data read at once.
      // Instead of hardcoding the granularity, we can query the granularity we
      // need like this:
      //   size_t granularity = 0;
      //   CUresult res = cuCtxGetLimit(&granularity,
      //   CU_LIMIT_MAX_L2_FETCH_GRANULARITY); // 0x05
      // But we need a context to be active. Which isn't the case here.
      num_partial_results = std::min(64 / smallest_input_dtype_bits, 8);

      // Limit register presure, PRED dtype is only one bit.
      num_partial_results = std::min(num_partial_results, 8);
      // Limit register presure for MOF, but still use a minimum of 2.
      int64_t fan_out = fusion.getFusionRoots().size();
      num_partial_results /= fan_out;
      num_partial_results = std::max(num_partial_results, 2);
    } else {
      num_partial_results = 2;
    }
  }

  while (shmem_usage * num_partial_results > shmem_budget) {
    num_partial_results /= 2;
    if (num_partial_results == 1) {
      break;
    }
  }

  VLOG(3) << "Each thread will produce " << num_partial_results << " output(s)";
  reduction_tiling[kDimX] *= num_partial_results;

  Vector3 num_threads = {1, num_threads_y, num_threads_x};
  int virtual_thread_scaling_factor =
      CalculateVirtualThreadScalingFactorForReduction(reduction_dimensions, cc);
  VLOG(2) << "Using virtual thread scaling: " << virtual_thread_scaling_factor;

  TilingScheme tiling_scheme(reduction_dimensions.dimensions, reduction_tiling,
                             num_threads, indexing_order, vector_size,
                             virtual_thread_scaling_factor);
  return ReductionCodegenInfo(tiling_scheme, num_partial_results,
                              reduction_dimensions.is_row_reduction,
                              reduction_is_race_free);
}

// Generate a single element of the tile (update the accumulator state) for a
// given reducer of index `i`.
void IrEmitterUnnested::GenerateElementForReducer(
    const HloReduceInstruction* reduction, llvm::Value* partial_result_index,
    const ReductionCodegenState& codegen_state,
    const llvm_ir::IrArray::Index& index_without_linear,
    const IrArray::Index& input_index, int num_partial_results,
    const ReductionOutputMap& result_ir_arrays) {
  HloComputation* reducer = reduction->to_apply();
  CHECK_EQ(reducer->num_parameters() % 2, 0);

  absl::InlinedVector<llvm::Value*, 2> reduction_accumulators;
  absl::InlinedVector<llvm::Value*, 2> reduction_input_value;
  for (int red_idx = 0; red_idx < reducer->num_parameters() / 2; red_idx++) {
    const ReductionCodegenState::ReductionCalculationState& state =
        codegen_state.GetCalculationStateFor(reduction, red_idx);

    llvm::AllocaInst* input_address = state.input_address;
    llvm::AllocaInst* partial_reduction_result_address =
        state.partial_result_address;
    llvm::Value* const input_ir_value = *state.input_gen(
        num_partial_results > 1 ? index_without_linear : input_index);
    b_.CreateStore(input_ir_value, input_address);
    llvm::Value* partial_result_address =
        InBoundsGEP(partial_reduction_result_address->getAllocatedType(),
                    partial_reduction_result_address, {partial_result_index});
    reduction_accumulators.push_back(partial_result_address);
    reduction_input_value.push_back(input_address);
  }

  absl::InlinedVector<llvm::Value*, 4> reduction_params;
  for (llvm::Value* acc : reduction_accumulators) {
    reduction_params.push_back(acc);
  }
  for (llvm::Value* value : reduction_input_value) {
    reduction_params.push_back(value);
  }

  // Emit a call to the variadic reducer. Since it may be returning a
  // tuple, we can't return it directly as a value. Instead, before
  // the call, we create N (N = # arguments in the tuple) allocas, one
  // for each returned argument, then when we make the call we pass N
  // pointers as last parameters, the called computation writes into
  // those pointers, and we have returned values on the stack (as well
  // as pointers to them).
  StatusOr<std::vector<llvm::Value*>> returned_scalars =
      ComputeNestedElementFromAddrs(*reducer, reduction_params);
  TF_CHECK_OK(returned_scalars.status());

  for (int i = 0; i < returned_scalars->size(); i++) {
    b_.CreateStore(returned_scalars->at(i), reduction_accumulators[i]);
  }
}

Status IrEmitterUnnested::EmitIRForReduction(
    mlir::lmhlo::FusionOp fusion,
    absl::Span<HloInstruction* const> instr_index_group,
    FusedIrEmitter& fused_emitter, const ReductionOutputMap& result_ir_arrays,
    const ReductionCodegenInfo& reduction_info, const Shape& input_shape) {
  std::vector<const HloReduceInstruction*> reductions;
  ExtraOutputGensMap extra_output_gens;

  for (const HloInstruction* hlo : instr_index_group) {
    if (IsReductionFromOrToContiguousDimensions(*hlo)) {
      reductions.push_back(Cast<HloReduceInstruction>(hlo));
    } else {
      extra_output_gens[hlo] = *fused_emitter.GetGenerator(*hlo);
    }
  }

  CHECK(!reductions.empty()) << " expect at least one reduce instructions.";
  const TilingScheme& tiling_scheme = reduction_info.GetTilingScheme();
  CHECK_EQ(tiling_scheme.GetNumThreadsPerBlockPhysical() % WarpSize(), 0);
  llvm::Type* index_ty =
      GetIndexTypeForKernel(fusion,
                            tiling_scheme.GetNumThreadsPerBlockPhysical() *
                                tiling_scheme.GetNumberOfBlocksPhysical(),
                            &b_);
  ReductionCodegenState codegen_state = GenerateReductionCodegenState(
      fusion, reduction_info, reductions, fused_emitter);

  EmitElementFunction emit_reduction_element =
      [&](const ThreadIdInfo& thread_id_info, const IrArray::Index& index,
          llvm::Value* y_loc, llvm::Value* x_loc) {
        IrArray::Index input_index = GetUnnormalizedIndex(
            index, input_shape, &b_,
            codegen_state.GetTilingScheme().GetDimsInElems());
        llvm::Value* partial_result_index =
            codegen_state.IsRowReduction()
                ? b_.getInt32(0)
                : b_.CreateSub(
                      x_loc,
                      GetStartOffsetX(tiling_scheme, thread_id_info.thread_id_x,
                                      index_ty, &b_));

        // Clear the linear index field of the IrArray::Index to enable the use
        // of GetElementPointer with array types. This enables the vectorization
        // of the computation for different partial results. Use this index if
        // 'num_partial_results > 1'.
        int num_partial_results = codegen_state.GetNumPartialResults();
        IrArray::Index index_without_linear{input_index.multidim(), input_shape,
                                            input_index.GetType()};

        // Emit code to generate the input and perform the reduction computation
        // for each reduction instruction.
        for (const HloReduceInstruction* reduce : reductions) {
          GenerateElementForReducer(reduce, partial_result_index, codegen_state,
                                    index_without_linear, input_index,
                                    num_partial_results, result_ir_arrays);
        }

        // Emit code to generate the output for the non-reduction instructions
        // in the fusion, if any.
        TF_CHECK_OK(EmitExtraOutputsForReduce(input_shape, result_ir_arrays,
                                              input_index, reduction_info,
                                              extra_output_gens));
      };

  TF_ASSIGN_OR_RETURN(
      TilingKernelInfo tiling_kernel_info,
      EmitTilingKernel(
          tiling_scheme, index_ty,
          [&](const ThreadIdInfo& thread_id_info, const IrArray::Index& index,
              ValueVector2 tile_dimensions) {
            EmitTile(codegen_state.GetTilingScheme(), index, thread_id_info,
                     tile_dimensions, emit_reduction_element);
          }));

  KernelSupportLibrary ksl(&b_);
  for (const HloReduceInstruction* reduce : reductions) {
    for (int partial_result_idx = 0;
         partial_result_idx < reduction_info.GetNumPartialResults();
         ++partial_result_idx) {
      if (codegen_state.IsRowReduction()) {
        EmitReductionOutputForRowReduction(tiling_kernel_info, codegen_state,
                                           index_ty, result_ir_arrays, reduce,
                                           partial_result_idx);
      } else {
        EmitReductionOutputForColumnReduction(tiling_kernel_info, codegen_state,
                                              index_ty, result_ir_arrays,
                                              reduce, partial_result_idx);
      }
    }
  }

  return OkStatus();
}

namespace {

// Returns whether the `instr` is either a constant, a scalar, or a
// broadcasted constant/scalar.
bool IsBroadcastedConstantOrScalar(const HloInstruction& instr) {
  return instr.IsConstant() || ShapeUtil::IsScalar(instr.shape()) ||
         (HloOpcode::kBroadcast == instr.opcode() &&
          (instr.operand(0)->IsConstant() ||
           ShapeUtil::IsScalar(instr.operand(0)->shape())));
}

// Divides `num_reduces` reduces into groups. Different groups will be executed
// in parallel. Generally speaking, we'd like to run the reduce instructions
// in parallel without incurring too much recomputation overhead. The current
// heuristic is to place reduce instructions who share nothing or only
// (broadcasted) scalars/constants into different groups; otherwise, they are
// placed in the same group. Non-reduce instructions always go with the reduce
// instructions into the same group so long as they share any predecessors.
std::vector<std::vector<HloInstruction*>> GroupDisjointReductions(
    HloComputation* fused_computation) {
  const Shape& root_shape = fused_computation->root_instruction()->shape();
  int num_fusion_outputs =
      fused_computation->root_instruction()->opcode() == HloOpcode::kTuple
          ? root_shape.tuple_shapes_size()
          : 1;
  CHECK_NE(0, num_fusion_outputs);
  if (num_fusion_outputs == 1) {
    return {{fused_computation->root_instruction()}};
  }

  std::vector<HloInstruction*> roots = GetFusionRoots(fused_computation);
  HloInstructionMap<tensorflow::UnionFind<HloInstruction*>> disjoint_sets;

  // TODO(b/249976438): we currently do not treat properly
  // aliasing between inputs and outputs of the fusion, so for now put all
  // non-reduction roots into one group to avoid read-after-write conflicts.
  HloInstruction* first_non_reduction_root = nullptr;

  for (HloInstruction* root : roots) {
    disjoint_sets[root].Get() = root;
    if (!IsReductionFromOrToContiguousDimensions(*root)) {
      if (!first_non_reduction_root) {
        first_non_reduction_root = root;
      } else {
        disjoint_sets[first_non_reduction_root].Merge(&disjoint_sets[root]);
      }
    }
  }

  std::unique_ptr<HloReachabilityMap> reachability_map =
      HloReachabilityMap::Build(fused_computation);
  for (HloInstruction* instr : fused_computation->instructions()) {
    std::vector<HloInstruction*> reached_output_ids;
    bool added_to_reduce = false;
    for (HloInstruction* output : roots) {
      if (IsReductionFromOrToContiguousDimensions(*output) &&
          (IsBroadcastedConstantOrScalar(*instr))) {
        if (added_to_reduce) {
          // Do not group more than one output reduce instructions through
          // broadcasted constants or scalars, as the recomputation should be
          // acceptable.
          VLOG(3) << "Skip broadcasted constant or scalar "
                  << instr->ToString();
          continue;
        }
      }
      // Now group output instructions if they have common predecessors.
      if (reachability_map->IsReachable(instr, output)) {
        VLOG(3) << "Reaching " << output->ToString() << " from "
                << instr->ToString();
        reached_output_ids.push_back(output);
        if (IsReductionFromOrToContiguousDimensions(*output)) {
          added_to_reduce = true;
        }
      }
    }
    for (size_t j = 1; j < reached_output_ids.size(); ++j) {
      disjoint_sets[reached_output_ids[0]].Merge(
          &disjoint_sets[reached_output_ids[j]]);
    }
  }

  // Place output instructions in the same set into the same group.
  HloInstructionMap<std::vector<HloInstruction*>> groups;
  for (HloInstruction* root : roots) {
    groups[disjoint_sets[root].Get()].push_back(root);
  }

  std::vector<std::vector<HloInstruction*>> ret;
  absl::c_for_each(
      groups, [&](auto& iter) { ret.emplace_back(std::move(iter.second)); });
  return ret;
}

}  // namespace

Status IrEmitterUnnested::EmitUnnestedReduction(
    mlir::lmhlo::FusionOp fusion, HloComputation* fused_computation) {
  llvm::SmallVector<mlir::Operation*> fusion_roots = fusion.getFusionRoots();

  // Group disjoint reductions in groups, to be executed in parallel.
  std::vector<std::vector<HloInstruction*>> instr_index_groups =
      GroupDisjointReductions(fused_computation);

  VLOG(2) << StrCat("Generate in ", instr_index_groups.size(), " groups for ",
                    llvm_ir::DumpToString(fusion));

  // hlo_roots has same ordering as fusion_roots.
  auto hlo_roots = GetFusionRoots(fused_computation);
  HloInstruction* first_reduce =
      *absl::c_find_if(hlo_roots, [](HloInstruction* instr) {
        return IsReductionFromOrToContiguousDimensions(*instr);
      });

  // We always use the first reduce as representative to construct
  // ReductionCodegenInfo, since all the reductions are required to have the
  // same shape and layout as verified by `IsFusedReductionOutputConsistent()`.
  TF_ASSIGN_OR_RETURN(
      ReductionCodegenInfo reduction_codegen_info,
      ComputeReductionCodegenInfo(fusion, fused_computation, first_reduce,
                                  instr_index_groups));
  const TilingScheme& tiling_scheme = reduction_codegen_info.GetTilingScheme();

  // block_y_count is set to instr_index_groups.size(), so that each reduction
  // group can be run in parallel by a different BlockIdy.
  LaunchDimensions launch_dimensions(
      {/*x=*/tiling_scheme.GetNumberOfBlocksPhysical(),
       /*y=*/static_cast<int64_t>(instr_index_groups.size()),
       /*z=*/1},
      {/*x=*/tiling_scheme.GetNumThreadsPerBlockPhysical(), /*y=*/1, /*z=*/1});
  VLOG(3) << "Launch dimensions of "
          << mlir::mhlo::GetDebugNameFromLocation(fusion.getLoc()) << ": "
          << launch_dimensions.ToString();
  if (!reduction_codegen_info.IsRaceFree()) {
    for (int i = 0; i < fusion_roots.size(); ++i) {
      if (IsReductionFromOrToContiguousDimensions(*hlo_roots[i])) {
        TF_RETURN_IF_ERROR(BuildFusedInitializerThunk(fusion, i));
      }
    }
  }

  TF_ASSIGN_OR_RETURN(
      std::optional<std::vector<llvm_ir::IrArray>> opt_ir_arrays,
      BuildReusableKernelThunk(fusion, launch_dimensions));
  if (!opt_ir_arrays.has_value()) {
    // The kernel was reused, no need to emit code.
    return OkStatus();
  }
  std::vector<llvm_ir::IrArray>& ir_arrays = opt_ir_arrays.value();

  FusedIrEmitter fused_emitter(elemental_emitter_);
  CHECK_LT(fused_computation->num_parameters(), ir_arrays.size());
  for (int i = 0; i < fused_computation->num_parameters(); i++) {
    llvm_ir::IrArray ir_array = ir_arrays[i];
    HloInstruction* fused_operand = fused_computation->parameter_instruction(i);
    fused_emitter.BindGenerator(
        *fused_operand,
        [this, ir_array, fused_operand](const llvm_ir::IrArray::Index& index) {
          return ir_array.EmitReadArrayElement(index, &b_,
                                               fused_operand->name());
        });
  }

  // Get outputs.
  ReductionOutputMap result_ir_arrays;

  // Skip all parameter buffers first.
  int ir_arrays_idx = fused_computation->num_parameters();
  for (HloInstruction* root : hlo_roots) {
    int get_num_results = GetNumOutputs(root->shape());
    result_ir_arrays[root] =
        absl::MakeSpan(ir_arrays).subspan(ir_arrays_idx, get_num_results);
    ir_arrays_idx += get_num_results;
  }

  KernelSupportLibrary ksl(&b_, llvm_ir::UnrollMode::kDefaultUnroll);

  // Use raw block_id_y to select the i-th parallel reduction to run. Using
  // block_id_y instead of block_id_x simplifies the index calculation
  // for reduction code generation as the block_id_y is orthogonal to
  // the indices used within the reductions.
  llvm::CallInst* raw_block_id_y = gpu::EmitCallToTargetIntrinsic(
      gpu::TargetIntrinsicID::kBlockIdy, {}, {}, &b_);
  llvm_ir::AddRangeMetadata(0, instr_index_groups.size(),
                            llvm::cast<llvm::Instruction>(raw_block_id_y));
  for (int i = 0; i < instr_index_groups.size(); ++i) {
    TF_RETURN_IF_ERROR(ksl.IfWithStatus(
        StrCat("reduce-group-", i),
        b_.CreateICmpEQ(raw_block_id_y, b_.getInt32(i)), [&] {
          return EmitIRForReduction(
              fusion, instr_index_groups[i], fused_emitter, result_ir_arrays,
              reduction_codegen_info, first_reduce->operand(0)->shape());
        }));
  }

  return OkStatus();
}

// Emits code for slices based on the below structure. An if statement with
// a guarding condition is generated for each ROOT slice.
//
// Pseudo code:
//
// Compute values of slice input operands
//
// Compute guarding_cond0
// if (guarding_cond0) {
//   Write to output of slice0
// }
//
// Compute guarding_cond1
// if (guarding_cond1) {
//   Write to output of slice1
// }
//
Status IrEmitterUnnested::EmitElementForInputFusibleSlices(
    const HloComputation* fused_computation,
    absl::Span<const llvm_ir::IrArray> ir_arrays,
    const llvm_ir::IrArray::Index& index) {
  VLOG(10) << "Emitting slice input fusion for "
           << fused_computation->ToString();

  HloInstruction* slice_or_tuple = fused_computation->root_instruction();
  auto slice_instructions = [&]() -> absl::Span<HloInstruction* const> {
    if (slice_or_tuple->opcode() == HloOpcode::kSlice) {
      return absl::Span<HloInstruction* const>(&slice_or_tuple, 1);
    }
    CHECK_EQ(slice_or_tuple->opcode(), HloOpcode::kTuple);
    return slice_or_tuple->operands();
  }();

  // Emit input operand values of slices.
  std::vector<llvm::Value*> input_ir_values;
  FusedIrEmitter fused_emitter(elemental_emitter_);
  for (int i = 0; i < fused_computation->num_parameters(); i++) {
    fused_emitter.BindGenerator(
        *fused_computation->parameter_instruction(i),
        [this, &ir_arrays, i](llvm_ir::IrArray::Index index) {
          return ir_arrays[i].EmitReadArrayElement(index, &b_);
        });
  }
  for (const HloInstruction* slice : slice_instructions) {
    auto input_generator = *fused_emitter.GetGenerator(*slice->operand(0));
    input_ir_values.push_back(input_generator(index).value());
  }

  // Emit for slice_instructions.
  KernelSupportLibrary ksl(&b_, llvm_ir::UnrollMode::kDefaultUnroll);
  for (int64_t i = 0; i < slice_instructions.size(); ++i) {
    HloInstruction* slice = slice_instructions[i];

    // guarding_cond := index >= start && index < limit, for each dim.
    std::vector<llvm::Value*> index_within_ranges;
    for (size_t dim = 0; dim < slice->slice_starts().size(); ++dim) {
      CHECK_EQ(slice->slice_strides(dim), 1);
      auto larger_or_equal_than_start = b_.CreateICmpSGE(
          index.multidim()[dim],
          index.GetConstantWithIndexType(slice->slice_starts(dim)));
      llvm::Value* smaller_than_limit = b_.CreateICmpSLT(
          index.multidim()[dim],
          index.GetConstantWithIndexType(slice->slice_limits(dim)));
      llvm::Value* within_range =
          b_.CreateAnd(larger_or_equal_than_start, smaller_than_limit);
      index_within_ranges.push_back(within_range);
    }
    llvm::Value* guarding_cond = b_.CreateAnd(index_within_ranges);

    auto emit_slice_elem_func = [&] {
      const std::vector<llvm::Value*>& src_multidim = index.multidim();
      std::vector<llvm::Value*> dst_multidim(src_multidim.size());
      for (size_t dim = 0; dim < src_multidim.size(); ++dim) {
        dst_multidim[dim] =
            Sub(src_multidim[dim],
                index.GetConstantWithIndexType(slice->slice_starts(dim)));
      }
      llvm_ir::IrArray src_ir_array =
          ir_arrays[fused_computation->num_parameters() + i];
      IrArray::Index slice_dst_index(dst_multidim, slice->shape(),
                                     index.GetType());
      src_ir_array.EmitWriteArrayElement(slice_dst_index, input_ir_values[i],
                                         &b_);
    };

    ksl.If(StrCat("slice", i), guarding_cond, emit_slice_elem_func);
  }
  return OkStatus();
}

Status IrEmitterUnnested::EmitInputFusibleNonStridedSlices(
    mlir::Operation* op) {
  auto fusion = mlir::cast<mlir::lmhlo::FusionOp>(op);

  constexpr int unroll_factor = 1;

  TF_ASSIGN_OR_RETURN(const HloComputation* fused_computation,
                      GetOrCreateSubComputationFromRegion(&fusion.getRegion(),
                                                          /*is_fusion=*/true));

  TF_ASSIGN_OR_RETURN(Shape element_shape,
                      GetConsistentInputShapeForRootSlices(fused_computation));
  TF_ASSIGN_OR_RETURN(LaunchDimensions launch_dimensions,
                      CalculateLaunchDimensions(
                          element_shape, ir_emitter_context_->gpu_device_info(),
                          {unroll_factor}));

  TF_ASSIGN_OR_RETURN(
      std::optional<std::vector<llvm_ir::IrArray>> opt_ir_arrays,
      BuildReusableKernelThunk(fusion, launch_dimensions));
  if (!opt_ir_arrays.has_value()) {
    // The kernel was reused, no need to emit code.
    return OkStatus();
  }
  std::vector<llvm_ir::IrArray>& ir_arrays = opt_ir_arrays.value();

  return ParallelLoopEmitter(
             [&](const llvm_ir::IrArray::Index index) -> Status {
               return EmitElementForInputFusibleSlices(fused_computation,
                                                       ir_arrays, index);
             },
             element_shape, launch_dimensions, &b_)
      .EmitLoop(
          IrName(GetIrNameFromLoc(fusion.getLoc())),
          GetIndexTypeForKernel(fusion, launch_dimensions.launch_bound(), &b_));
}

Status IrEmitterUnnested::EmitDynamicUpdateSlice(
    mlir::lmhlo::FusionOp fusion_op, const HloComputation* fused_computation) {
  // Fusion node with dynamic-update-slice as the root where the op's input
  // (i.e. array to update) shares the same slice as its output.  In this case
  // we have a special algorithm that modifies the output in place without
  // touching the un-updated elements.
  CHECK_EQ(1, GetHloOutputs(fusion_op).size());

  // Shape of the dynamic-update-slice's "update" operand.
  Shape update_shape =
      fused_computation->root_instruction()->operand(1)->shape();

  TF_ASSIGN_OR_RETURN(
      LaunchDimensions launch_dimensions,
      CalculateLaunchDimensions(update_shape,
                                ir_emitter_context_->gpu_device_info()));

  // Set up kernel thunk and fused ir emitter.
  TF_ASSIGN_OR_RETURN(
      std::optional<std::vector<llvm_ir::IrArray>> opt_ir_arrays,
      BuildReusableKernelThunk(fusion_op, launch_dimensions));
  if (!opt_ir_arrays.has_value()) {
    // The kernel was reused, no need to emit code.
    return OkStatus();
  }
  std::vector<llvm_ir::IrArray>& ir_arrays = opt_ir_arrays.value();

  FusedIrEmitter fused_emitter(elemental_emitter_);
  for (int i = 0; i < fused_computation->num_parameters(); i++) {
    auto fused_operand = fused_computation->parameter_instruction(i);
    fused_emitter.BindGenerator(
        *fused_operand, [this, &ir_arrays, i,
                         fused_operand](const llvm_ir::IrArray::Index& index) {
          return ir_arrays[i].EmitReadArrayElement(index, &b_,
                                                   fused_operand->name());
        });
  }

  // Array to write into.  Because this is an in-place operation, this is the
  // same as operand 0's array.
  const IrArray& output_array = ir_arrays.back();

  return llvm_ir::EmitParallelFusedDynamicUpdateSliceInPlace(
      fused_computation, output_array, &fused_emitter, launch_dimensions, &b_);
}

Status IrEmitterUnnested::EmitScatter(mlir::lmhlo::FusionOp fusion_op,
                                      const HloComputation* fused_computation) {
  auto* root = fused_computation->root_instruction();

  // The initialization from 'operand' is using different loop bounds, so
  // emit it in a separate kernel. Treat it like a loop fusion, writing to
  // the output buffer.
  TF_RETURN_IF_ERROR([&] {
    auto unroll_factor = ComputeMaxUnrollFactor(fusion_op, hlo_module_config_);
    const Shape& element_shape = root->shape();
    TF_ASSIGN_OR_RETURN(
        LaunchDimensions launch_dimensions,
        CalculateLaunchDimensions(element_shape,
                                  ir_emitter_context_->gpu_device_info(),
                                  {unroll_factor, /*few_waves=*/false}));

    TF_ASSIGN_OR_RETURN(
        std::optional<std::vector<llvm_ir::IrArray>> opt_ir_arrays,
        BuildReusableKernelThunk(fusion_op, launch_dimensions,
                                 /*discriminator=*/"init"));
    if (!opt_ir_arrays.has_value()) {
      // The kernel was reused, no need to emit code.
      return OkStatus();
    }
    std::vector<llvm_ir::IrArray>& ir_arrays = opt_ir_arrays.value();

    FusedIrEmitter operand_fused_emitter(elemental_emitter_);
    for (int i = 0; i < fused_computation->num_parameters(); i++) {
      auto fused_operand = fused_computation->parameter_instruction(i);
      operand_fused_emitter.BindGenerator(
          *fused_operand,
          [this, &ir_arrays, i, fused_operand](llvm_ir::IrArray::Index index) {
            return ir_arrays[i].EmitReadArrayElement(index, &b_,
                                                     fused_operand->name());
          });
    }
    TF_ASSIGN_OR_RETURN(auto generator,
                        operand_fused_emitter.GetGenerator(*root->operand(0)));

    TF_RETURN_IF_ERROR(
        ParallelLoopEmitter(generator, {ir_arrays.back()}, launch_dimensions,
                            &b_, {unroll_factor})
            .EmitLoop(IrName(GetIrNameFromLoc(fusion_op.getLoc())),
                      GetIndexTypeForKernel(
                          fusion_op, launch_dimensions.launch_bound(), &b_)));

    return OkStatus();
  }());

  // Now build the actual scatter, reading and writing to the freshly
  // filled output buffer.
  {
    const Shape& updates_shape = root->operand(2)->shape();
    TF_ASSIGN_OR_RETURN(
        LaunchDimensions launch_dimensions,
        CalculateLaunchDimensions(updates_shape,
                                  ir_emitter_context_->gpu_device_info()));

    TF_ASSIGN_OR_RETURN(
        std::optional<std::vector<llvm_ir::IrArray>> opt_ir_arrays,
        BuildReusableKernelThunk(fusion_op, launch_dimensions,
                                 /*discriminator=*/"scatter"));
    if (!opt_ir_arrays.has_value()) {
      // The kernel was reused, no need to emit code.
      return OkStatus();
    }
    std::vector<llvm_ir::IrArray>& ir_arrays = opt_ir_arrays.value();

    // Spin up a new fused emitter for the scatter kernel and emit it.
    FusedIrEmitter scatter_fused_emitter = FusedIrEmitter(elemental_emitter_);
    for (int i = 0; i < fused_computation->num_parameters(); i++) {
      auto fused_operand = fused_computation->parameter_instruction(i);
      scatter_fused_emitter.BindGenerator(
          *fused_operand,
          [this, &ir_arrays, i, fused_operand](llvm_ir::IrArray::Index index) {
            return ir_arrays[i].EmitReadArrayElement(index, &b_,
                                                     fused_operand->name());
          });
    }

    TF_ASSIGN_OR_RETURN(const auto dim_numbers,
                        mlir::LhloDialectEmitter::GetScatterDimensionNumbers(
                            root, fusion_op.getContext()));

    ScatterDescriptor desc;
    desc.name = IrName(root);
    desc.operand_shape = root->operand(0)->shape();
    desc.scatter_indices_shape = root->operand(1)->shape();
    desc.updates_shape = updates_shape;
    desc.dim_numbers = dim_numbers;
    desc.unique_indices = root->unique_indices();
    desc.update_computation = root->called_computations()[0];
    desc.output = ir_arrays.back();
    TF_ASSIGN_OR_RETURN(desc.scatter_indices_gen,
                        scatter_fused_emitter.GetGenerator(*root->operand(1)));
    TF_ASSIGN_OR_RETURN(desc.updates_gen,
                        scatter_fused_emitter.GetGenerator(*root->operand(2)));
    desc.get_index_type = [&](int64_t launch_size) {
      return GetIndexTypeForKernel(root, launch_size, &b_);
    };
    TF_RETURN_IF_ERROR(EmitScatter(desc, launch_dimensions));
  }

  return OkStatus();
}

Status IrEmitterUnnested::EmitOp(mlir::Operation* op) {
  if (mlir::isa<mlir::memref::CollapseShapeOp, mlir::func::ConstantOp,
                mlir::arith::ConstantOp, mlir::memref::ReinterpretCastOp,
                mlir::func::ReturnOp, mlir::lmhlo::TerminatorOp,
                mlir::memref::ViewOp>(op)) {
    return OkStatus();
  }

  if (mlir::isa<mlir::memref::GetGlobalOp>(op)) {
    return EmitConstant(op);
  }

  if (auto call = mlir::dyn_cast<mlir::lmhlo::CustomCallOp>(op)) {
    if (call.getCallTargetName() == "PadToStatic") {
      return EmitPadToStatic(op);
    }
    if (call.getCallTargetName() == "SliceToDynamic") {
      return EmitSliceToDynamic(op);
    }
    const llvm::StringRef call_target = call.getCallTargetName();
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    if (absl::string_view(call_target.data(), call_target.size()) ==
        kTriangularSolveCallTarget) {
      return EmitTriangularSolveCustomCall(op);
    }
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

    return EmitCustomCallThunk(op);
  }

  if (mlir::isa<mlir::lmhlo_gpu::GEMMOp>(op)) {
    return EmitGemmThunk(op);
  }

#if GOOGLE_CUDA
  if (mlir::isa<mlir::lmhlo_gpu::CublasLtMatmulOp>(op)) {
    return EmitCublasLtMatmulThunk(op);
  }
  if (mlir::isa<mlir::lmhlo_gpu::CublasLtMatmulF8Op>(op)) {
    return EmitCublasLtMatmulThunkF8(op);
  }
  if (mlir::isa<mlir::lmhlo_gpu::CudnnConvReorderFilterOp,
                mlir::lmhlo_gpu::CudnnConvReorderFilterAndBiasOp>(op)) {
    return EmitConvolutionReorderThunk(op);
  }
#endif  // GOOGLE_CUDA

  if (mlir::isa<mlir::lmhlo_gpu::ConvForwardOp,
                mlir::lmhlo_gpu::ConvForwardFusedOp,
                mlir::lmhlo_gpu::ConvForwardFusedSideInputOp,
                mlir::lmhlo_gpu::ConvBackwardFilterOp,
                mlir::lmhlo_gpu::ConvBackwardInputOp>(op)) {
    return EmitConvolutionThunk(op);
  }

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  if (mlir::isa<mlir::lmhlo_gpu::CholeskyOp>(op)) {
    return EmitCholeskyThunk(op);
  }
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

  if (mlir::isa<mlir::lmhlo::FftOp>(op)) {
    return EmitFftThunk(op);
  }

  if (mlir::isa<mlir::lmhlo::TriangularSolveOp>(op)) {
    return InternalError(
        "TriangularSolve is implemented as a custom-call; we do not expect to "
        "lower a true HLO TriangularSolve op.");
  }

  if (mlir::isa<mlir::lmhlo::FusionOp>(op)) {
    return EmitFusion(op);
  }

  if (mlir::isa<mlir::lmhlo::SelectAndScatterOp>(op)) {
    return EmitSelectAndScatter(op);
  }

  if (mlir::isa<mlir::lmhlo::RngGetAndUpdateStateOp>(op)) {
    return EmitRngGetAndUpdateState(op);
  }

  if (mlir::isa<mlir::lmhlo::ScatterOp>(op)) {
    return EmitScatter(op);
  }

  if (mlir::isa<mlir::lmhlo::SortOp>(op)) {
    return EmitSort(op);
  }

  if (mlir::isa<mlir::lmhlo::ReplicaIdOp>(op)) {
    return EmitReplicaOrPartitionId<ReplicaIdThunk, mlir::lmhlo::ReplicaIdOp>(
        op);
  }

  if (mlir::isa<mlir::lmhlo::PartitionIdOp>(op)) {
    return EmitReplicaOrPartitionId<PartitionIdThunk,
                                    mlir::lmhlo::PartitionIdOp>(op);
  }

  if (mlir::isa<mlir::lmhlo::CollectivePermuteOp>(op)) {
    return EmitCollectivePermute<NcclCollectivePermuteThunk,
                                 mlir::lmhlo::CollectivePermuteOp>(op);
  }

  if (mlir::isa<mlir::lmhlo_gpu::CollectivePermuteStartOp>(op)) {
    return EmitCollectivePermute<NcclCollectivePermuteStartThunk,
                                 mlir::lmhlo_gpu::CollectivePermuteStartOp>(op);
  }

  if (mlir::isa<mlir::lmhlo_gpu::CollectivePermuteDoneOp>(op)) {
    return EmitNcclAsyncDone<NcclCollectivePermuteDoneThunk,
                             mlir::lmhlo_gpu::CollectivePermuteDoneOp>(op);
  }

  if (mlir::isa<mlir::lmhlo::AllGatherOp>(op)) {
    return EmitNcclThunk<NcclAllGatherThunk, mlir::lmhlo::AllGatherOp>(op);
  }

  if (mlir::isa<mlir::lmhlo_gpu::AllGatherStartOp>(op)) {
    return EmitNcclThunk<NcclAllGatherStartThunk,
                         mlir::lmhlo_gpu::AllGatherStartOp>(op);
  }

  if (mlir::isa<mlir::lmhlo_gpu::AllGatherDoneOp>(op)) {
    return EmitNcclAsyncDone<NcclAllGatherDoneThunk,
                             mlir::lmhlo_gpu::AllGatherDoneOp>(op);
  }

  if (mlir::isa<mlir::lmhlo::AllReduceOp>(op)) {
    return EmitNcclThunk<NcclAllReduceThunk, mlir::lmhlo::AllReduceOp>(op);
  }

  if (mlir::isa<mlir::lmhlo_gpu::AllReduceStartOp>(op)) {
    return EmitNcclThunk<NcclAllReduceStartThunk,
                         mlir::lmhlo_gpu::AllReduceStartOp>(op);
  }

  if (mlir::isa<mlir::lmhlo_gpu::AllReduceDoneOp>(op)) {
    return EmitNcclAsyncDone<NcclAllReduceDoneThunk,
                             mlir::lmhlo_gpu::AllReduceDoneOp>(op);
  }

  if (mlir::isa<mlir::lmhlo::ReduceScatterOp>(op)) {
    return EmitNcclThunk<NcclReduceScatterThunk, mlir::lmhlo::ReduceScatterOp>(
        op);
  }

  if (mlir::isa<mlir::lmhlo_gpu::ReduceScatterStartOp>(op)) {
    return EmitNcclThunk<NcclReduceScatterStartThunk,
                         mlir::lmhlo_gpu::ReduceScatterStartOp>(op);
  }

  if (mlir::isa<mlir::lmhlo_gpu::ReduceScatterDoneOp>(op)) {
    return EmitNcclAsyncDone<NcclReduceScatterDoneThunk,
                             mlir::lmhlo_gpu::ReduceScatterDoneOp>(op);
  }

  if (mlir::isa<mlir::lmhlo::AllToAllOp>(op)) {
    return EmitNcclThunk<NcclAllToAllThunk, mlir::lmhlo::AllToAllOp>(op);
  }

  if (mlir::isa<mlir::lmhlo_gpu::AllToAllStartOp>(op)) {
    return EmitNcclThunk<NcclAllToAllStartThunk,
                         mlir::lmhlo_gpu::AllToAllStartOp>(op);
  }

  if (mlir::isa<mlir::lmhlo_gpu::AllToAllDoneOp>(op)) {
    return EmitNcclAsyncDone<NcclAllToAllDoneThunk,
                             mlir::lmhlo_gpu::AllToAllDoneOp>(op);
  }

  if (mlir::isa<mlir::lmhlo::InfeedOp>(op)) {
    return EmitInfeed(op);
  }

  if (mlir::isa<mlir::lmhlo::OutfeedOp>(op)) {
    return EmitOutfeed(op);
  }

  if (mlir::isa<mlir::lmhlo::CaseOp>(op)) {
    return EmitConditional(op);
  }

  if (mlir::isa<mlir::lmhlo::WhileOp>(op)) {
    return EmitWhile(op);
  }

  if (mlir::isa<mlir::gpu::LaunchFuncOp>(op)) {
    return EmitLaunchFunc(op);
  }

  // Remaining arith.constant ops are the gpu.launch_func dimensions as a result
  // of inlining the fusion region after lowering. They can safely be skipped
  // because constants have no side effects.
  if (mlir::isa<mlir::arith::ConstantOp>(op)) {
    return OkStatus();
  }

  // Point to point communication operations are only implemented as XLA
  // GPU runtime custom calls.
  bool is_gpu_runtime = hlo_module_config_.debug_options()
                            .xla_gpu_enable_xla_runtime_executable();
  if (is_gpu_runtime &&
      mlir::isa<mlir::lmhlo::SendOp, mlir::lmhlo::RecvOp,
                mlir::lmhlo::SendDoneOp, mlir::lmhlo::RecvDoneOp>(op)) {
    return EmitUnreachable(op,
                           "Point-to-point communication operations are not "
                           "implemented as thunks");
  }

  return InternalError("Unrecognized op: %s", llvm_ir::DumpToString(op));
}

Status IrEmitterUnnested::EmitLmhloRegion(mlir::Region* region) {
  for (mlir::Operation& op : llvm::make_early_inc_range(region->front())) {
    TF_RETURN_IF_ERROR(EmitOp(&op));
  }
  return OkStatus();
}

void IrEmitterUnnested::GetDependentDialects(mlir::DialectRegistry& registry) {
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::gpu::GPUDialect, mlir::lmhlo::LmhloDialect,
                  mlir::lmhlo_gpu::LmhloGpuDialect, mlir::mhlo::MhloDialect,
                  mlir::memref::MemRefDialect>();
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerNVVMDialectTranslation(registry);
  mlir::registerROCDLDialectTranslation(registry);
}

Thunk::ThunkInfo IrEmitterUnnested::GetThunkInfo(mlir::Operation* op) {
  Thunk::ThunkInfo thunk_info(op);
  thunk_info.profile_annotation = absl::StrFormat(
      "Thunk:#hlo_op=%s#", mlir::mhlo::GetDebugNameFromLocation(op->getLoc()));
  return thunk_info;
}

}  // namespace gpu
}  // namespace xla
