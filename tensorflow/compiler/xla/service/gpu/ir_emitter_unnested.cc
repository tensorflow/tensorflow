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

#include "tensorflow/compiler/xla/service/gpu/ir_emitter_unnested.h"

#include <algorithm>
#include <cstring>
#include <iterator>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BlockAndValueMapping.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Verifier.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/lhlo_gpu_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/utils/hlo_utils.h"
#include "tensorflow/compiler/mlir/utils/name_utils.h"
#include "tensorflow/compiler/mlir/xla/attribute_exporter.h"
#include "tensorflow/compiler/mlir/xla/hlo_function_importer.h"
#include "tensorflow/compiler/mlir/xla/hlo_utils.h"
#include "tensorflow/compiler/mlir/xla/mlir_hlo_to_hlo.h"
#include "tensorflow/compiler/mlir/xla/type_to_shape.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/permutation_util.h"
#include "tensorflow/compiler/xla/service/collective_ops_utils.h"
#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/bef_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/compiler/xla/service/gpu/collective_permute_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/conditional_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/convolution_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/copy_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/cudnn_batchnorm_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/custom_call_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/fft_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/for_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/gemm_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_constants.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_conv_runner.h"
#include "tensorflow/compiler/xla/service/gpu/hlo_to_ir_bindings.h"
#include "tensorflow/compiler/xla/service/gpu/infeed_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emitter_context.h"
#include "tensorflow/compiler/xla/service/gpu/kernel_mapping_scheme.h"
#include "tensorflow/compiler/xla/service/gpu/kernel_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/launch_dimensions.h"
#include "tensorflow/compiler/xla/service/gpu/memset_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_all_gather_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_all_reduce_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_all_to_all_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_collective_permute_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/outfeed_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/parallel_loop_emitter.h"
#include "tensorflow/compiler/xla/service/gpu/replica_id_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/sequential_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/target_util.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/compiler/xla/service/gpu/triangular_solve_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/while_thunk.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/llvm_ir/buffer_assignment_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/dynamic_update_slice_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/fused_ir_emitter.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_array.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/sort_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/tuple_ops.h"
#include "tensorflow/compiler/xla/service/name_uniquer.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/service/while_loop_analysis.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/union_find.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"

#if GOOGLE_CUDA
#include "tensorflow/compiler/xla/service/gpu/cholesky_thunk.h"
#endif  // GOOGLE_CUDA

namespace xla {
namespace gpu {

namespace {

using absl::InlinedVector;
using absl::nullopt;
using absl::optional;
using absl::StrCat;
using llvm_ir::IrArray;
using llvm_ir::IrName;

const auto kDimX = KernelMappingScheme::DimX;
const auto kDimY = KernelMappingScheme::DimY;
const auto kDimZ = KernelMappingScheme::DimZ;
const auto kDimTot = KernelMappingScheme::DimTot;

const auto kLinearIndexingX = KernelMappingScheme::LinearIndexingX;
const auto kStridedIndexingX = KernelMappingScheme::StridedIndexingX;
const auto kStridedLinearIndexingX =
    KernelMappingScheme::StridedLinearIndexingX;

// If a dimensions is smaller than this, untiled transposition may be more
// efficient.
const int64 kMinDimensionToTransposeTiled = 16;

// Updates the launch dimensions in "thunk" and annotate the launch dimensions
// of the corresponding IR kernel in "llvm_module".
// Precondition: "thunk" must be a KernelThunk.
void UpdateLaunchDimensions(const LaunchDimensions& launch_dims, Thunk* thunk,
                            llvm::Module* llvm_module) {
  CHECK(Thunk::Kind::kKernel == thunk->kind());
  KernelThunk* kernel_thunk = static_cast<KernelThunk*>(thunk);
  kernel_thunk->SetLaunchDimensions(launch_dims);

  // Add __launch_bounds__ to metadata. This limits registers per thread to
  // avoid out-of-resources launching errors.
  llvm::NamedMDNode* nvvm_annotations_node =
      llvm_module->getOrInsertNamedMetadata("nvvm.annotations");
  llvm::Function* ir_kernel =
      llvm_module->getFunction(kernel_thunk->kernel_name().c_str());
  llvm::LLVMContext& llvm_context = llvm_module->getContext();
  llvm::ConstantInt* threads_per_block_ir_value = llvm::ConstantInt::get(
      llvm::IntegerType::get(llvm_context, /*NumBits=*/32),
      launch_dims.thread_counts_per_block().x);
  // Our launch bounds are exact, so we can specify them as reqntidx rather than
  // maxntidx.
  nvvm_annotations_node->addOperand(llvm::MDNode::get(
      llvm_context,
      {llvm::ConstantAsMetadata::get(ir_kernel),
       llvm::MDString::get(llvm_context, "reqntidx"),
       llvm::ConstantAsMetadata::get(threads_per_block_ir_value)}));
}

bool BinarySearchDenseElementsAttr(mlir::DenseIntElementsAttr elements,
                                   int64 v) {
  mlir::APInt value(sizeof(int64) * 8, v, /*isSigned=*/true);
  return std::binary_search(
      elements.begin(), elements.end(), value,
      [](const mlir::APInt& x, const mlir::APInt& y) { return x.slt(y); });
}

// Returns true if the fusion contains any instruction that is likely
// translated to complex LLVM IR, such as loops, and prevent vectorization.
bool MayPreventVectorization(const HloInstruction& hlo) {
  if (hlo.opcode() == HloOpcode::kFusion) {
    return absl::c_any_of(hlo.fused_instructions_computation()->instructions(),
                          [](const HloInstruction* instr) {
                            switch (instr->opcode()) {
                              case HloOpcode::kReduceWindow:
                              case HloOpcode::kSort:
                              case HloOpcode::kDot:
                              case HloOpcode::kSin:
                              case HloOpcode::kCos:
                              case HloOpcode::kPower:
                              case HloOpcode::kAtan2:
                                return true;
                              case HloOpcode::kReduce:
                                return !instr->shape().IsArray();
                              default:
                                return false;
                            }
                          });
  } else if (hlo.IsElementwise()) {
    // Unfused elementwise operations are usually memory bound, unroll them.
    switch (hlo.opcode()) {
        // The following elementwise operation implementations contain branches.
        // LLVM vectorizer doesn't work in that case.
        // The unrolled code is faster when it isn't vectorized.
      case HloOpcode::kSin:
      case HloOpcode::kCos:
      case HloOpcode::kPower:
      case HloOpcode::kAtan2:
        return true;
      default:
        return false;
    }
  } else if (hlo.opcode() == HloOpcode::kReduce && hlo.shape().IsArray()) {
    // TODO(timshen): check if the to_apply() attribute contains instructions
    // that break LLVM vectorization.
    return false;
  }
  return true;
}

bool LmhloOpIsElementwise(mlir::Operation* op) {
  CHECK(op->getDialect() ==
        op->getContext()->getLoadedDialect<mlir::lmhlo::LmhloDialect>());
  auto opcode = *MhloToHloOpcode(op);
  if (HloInstruction::IsOpElementwise(opcode)) {
    return true;
  }
  if (opcode == HloOpcode::kMap) {
    int iota = 0;
    for (const llvm::APInt& i :
         mlir::cast<mlir::lmhlo::MapOp>(op).dimensions()) {
      if (i.getZExtValue() != iota) {
        return false;
      }
      iota++;
    }
    return true;
  }
  // TODO(timshen): not sure about whether porting
  // HloFusionInstruction::IsElementwiseImpl() is necessary. HandleFusion()
  // doesn't use such information.
  return false;
}

bool MayPreventVectorization(mlir::Operation* op) {
  CHECK(op->getDialect() ==
        op->getContext()->getLoadedDialect<mlir::lmhlo::LmhloDialect>());
  auto opcode = *MhloToHloOpcode(op);

  if (auto fusion = mlir::dyn_cast<mlir::lmhlo::FusionOp>(op)) {
    for (mlir::Operation& instr : fusion.region().front()) {
      if (mlir::isa<mlir::lmhlo::TerminatorOp, mlir::mhlo::ReturnOp,
                    mlir::memref::TensorLoadOp, mlir::memref::TensorStoreOp>(
              &instr)) {
        continue;
      }
      CHECK(instr.getDialect() ==
            instr.getContext()->getLoadedDialect<mlir::mhlo::MhloDialect>())
          << MlirToString(op);
      switch (*MhloToHloOpcode(&instr)) {
        case HloOpcode::kReduceWindow:
        case HloOpcode::kSort:
        case HloOpcode::kDot:
        case HloOpcode::kSin:
        case HloOpcode::kCos:
        case HloOpcode::kPower:
        case HloOpcode::kAtan2:
          return true;
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
  } else if (LmhloOpIsElementwise(op)) {
    // Unfused elementwise operations are usually memory bound, unroll them.
    switch (opcode) {
        // The following elementwise operation implementations contain branches.
        // LLVM vectorizer doesn't work in that case.
        // The unrolled code is faster when it isn't vectorized.
      case HloOpcode::kSin:
      case HloOpcode::kCos:
      case HloOpcode::kPower:
      case HloOpcode::kAtan2:
        return true;
      default:
        return false;
    }
  } else if (opcode == HloOpcode::kReduce && GetHloOutputs(op).size() == 1) {
    // TODO(timshen): check if the to_apply() attribute contains instructions
    // that break LLVM vectorization.
    return false;
  }
  return true;
}

std::vector<mlir::Operation*> GetOutputOps(mlir::lmhlo::FusionOp fusion) {
  llvm::SetVector<mlir::Operation*> ops;
  for (mlir::Value output_value : fusion.getFusionResults()) {
    ops.insert(output_value.getDefiningOp());
  }
  return std::vector<mlir::Operation*>(ops.begin(), ops.end());
}

// Computes the maximum valid unroll factor for a given instruction.
int ComputeMaxUnrollFactor(const Shape& shape,
                           const HloModuleConfig& hlo_module_config) {
  int max_unroll_factor =
      hlo_module_config.debug_options().xla_gpu_max_kernel_unroll_factor();

  // Find the largest possible power of two to unroll by.
  // TODO(kramerb): Make this smarter.
  int64 num_elements = ShapeUtil::ElementsIn(shape);
  for (int i = max_unroll_factor; i > 1; i /= 2) {
    if (num_elements % i == 0) {
      return i;
    }
  }

  // Cannot unroll.
  return 1;
}

// Computes the maximum valid unroll factor for a given instruction.
int ComputeMaxUnrollFactor(const HloInstruction* hlo) {
  const Shape& element_shape = hlo->IsMultiOutputFusion()
                                   ? ShapeUtil::GetSubshape(hlo->shape(), {0})
                                   : hlo->shape();
  return ComputeMaxUnrollFactor(element_shape, hlo->GetModule()->config());
}

// Computes the maximum valid unroll factor for a given instruction.
int ComputeMaxUnrollFactor(mlir::Operation* op,
                           const HloModuleConfig& hlo_module_config) {
  Shape element_shape = [&] {
    std::vector<Shape> shapes;
    // Detect multi-output fusion. Notice that for a reduce in the fusion that
    // returns a tuple, we don't want to treat it as multi-output fusion. We
    // want to pass that tuple into ComputeMaxUnrollFactor below. For an actual
    // MOF, just pass the first element of the root tuple.
    if (auto fusion = mlir::dyn_cast<mlir::lmhlo::FusionOp>(op)) {
      std::vector<mlir::Operation*> fusion_outputs = GetOutputOps(fusion);
      for (mlir::Value result : fusion_outputs[0]->getResults()) {
        shapes.push_back(TypeToShape(result.getType()));
      }
    } else {
      for (mlir::Value result : GetHloOutputs(op)) {
        shapes.push_back(TypeToShape(result.getType()));
      }
    }
    if (shapes.size() > 1) {
      return ShapeUtil::MakeTupleShape(shapes);
    }
    return shapes[0];
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
llvm::Type* GetIndexTypeForKernel(const HloInstruction* hlo, int64 launch_size,
                                  llvm::IRBuilder<>* b) {
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
llvm::Type* GetIndexTypeForKernelFromMlir(mlir::Operation* op,
                                          int64 launch_size,
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
    if (!shape_in_range(TypeToShape(result.getType()))) {
      return i64_ty;
    }
  }

  auto hlo_shape_in_range = [&](mlir::Value operand) -> bool {
    return shape_in_range(TypeToShape(operand.getType()));
  };

  // Check the size of input tensors
  if (!absl::c_all_of(op->getOperands(), hlo_shape_in_range)) {
    return i64_ty;
  }

  // Check the size of the internal result tensors
  if (auto fusion = mlir::dyn_cast<mlir::lmhlo::FusionOp>(op)) {
    auto result = fusion.region().walk([&](mlir::Operation* op) {
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

}  // namespace

IrEmitterUnnested::IrEmitterUnnested(const HloModuleConfig& hlo_module_config,
                                     IrEmitterContext* ir_emitter_context)
    : IrEmitter(hlo_module_config, ir_emitter_context, /*is_nested=*/false) {}

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
  string kernel_name = ir_emitter_context_->name_uniquer()->GetUniqueName(
      llvm_ir::SanitizeFunctionName(std::string(name)));

  // Create the kernel and add it to the module.
  llvm::Module* module = ir_emitter_context_->llvm_module();
  llvm::LLVMContext& context = module->getContext();
  llvm::FunctionType* kernel_type = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getVoidTy(context),
      std::vector<llvm::Type*>(args.size(), b_.getInt8PtrTy()),
      /*isVarArg=*/false);
  llvm::Function* kernel =
      llvm::Function::Create(kernel_type, llvm::GlobalValue::ExternalLinkage,
                             kernel_name.c_str(), module);

  // Add dereferenceable and alignment information to each of the kernel's
  // parameters.
  auto arg_it = kernel->arg_begin();
  for (size_t arg_no = 0; arg_no < args.size(); ++arg_no) {
    const BufferAllocation* alloc = args[arg_no];
    llvm::Argument* fn_arg = &*arg_it;
    ++arg_it;

    kernel->addDereferenceableAttr(arg_no + 1, alloc->size());

    const int64 alignment = [&] {
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
      fn_arg->setName("temp_buf");
    } else {
      fn_arg->setName(StrCat("alloc", alloc->index()));
    }
  }

  AnnotateFunctionAsGpuKernel(module, kernel, &b_);

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

StatusOr<BufferAllocation::Slice> IrEmitterUnnested::GetAllocationSliceForMlir(
    mlir::Value v, std::string* constant_name) {
  return xla::gpu::GetAllocationSliceForMlir(
      v, ir_emitter_context_->allocations(), constant_name);
}

Status IrEmitterUnnested::EmitUsingElementalIrEmitter(MlirEmitterInput input) {
  // Replace unnested op with a fused nested op.
  //
  // TODO(timshen): Ultimately this should be a pass. It's currently not a pass,
  // because we don't have a fully functioning LMHLO graph yet.

  mlir::Location loc = input.op->getLoc();
  mlir::lmhlo::FusionOp fusion =
      mlir::OpBuilder(input.op).create<mlir::lmhlo::FusionOp>(loc);
  Shape output_shape;
  mlir::OpBuilder b(&fusion.region());

  const auto load_memrefs = [loc, &b](mlir::ValueRange range) {
    std::vector<mlir::Value> operands;
    for (mlir::Value memref : range) {
      auto load = b.create<mlir::memref::TensorLoadOp>(loc, memref);
      HloFunctionImporter::SetLayoutForMlir(load,
                                            TypeToShape(memref.getType()));
      operands.push_back(load);
    }
    return operands;
  };

  if (auto copy = mlir::dyn_cast<mlir::lmhlo::CopyOp>(input.op)) {
    auto operand = b.create<mlir::memref::TensorLoadOp>(loc, copy.operand());
    HloFunctionImporter::SetLayoutForMlir(
        operand, TypeToShape(copy.operand().getType()));
    auto fused_copy = b.create<mlir::mhlo::CopyOp>(loc, operand);
    output_shape = TypeToShape(copy.output().getType());
    HloFunctionImporter::SetLayoutForMlir(fused_copy, output_shape);
    b.create<mlir::memref::TensorStoreOp>(loc, fused_copy, copy.output());
  } else if (auto reduce = mlir::dyn_cast<mlir::lmhlo::ReduceOp>(input.op)) {
    std::vector<mlir::Value> inputs = load_memrefs(reduce.inputs());
    std::vector<mlir::Value> init_values = load_memrefs(reduce.init_values());
    auto fused_reduce = b.create<mlir::mhlo::ReduceOp>(loc, inputs, init_values,
                                                       reduce.dimensions());
    fused_reduce.body().takeBody(reduce.body());
    CHECK_EQ(fused_reduce.getNumResults(), reduce.out().size());
    std::vector<Shape> output_shapes;
    for (int i = 0; i < reduce.out().size(); i++) {
      b.create<mlir::memref::TensorStoreOp>(loc, fused_reduce.getResult(i),
                                            reduce.out()[i]);
      auto shape = TypeToShape(reduce.out()[i].getType());
      if (i == 0) {
        HloFunctionImporter::SetLayoutForMlir(fused_reduce, shape);
      }
      output_shapes.push_back(shape);
    }
    if (output_shapes.size() == 1) {
      output_shape = output_shapes[0];
    } else {
      output_shape = ShapeUtil::MakeTupleShape(output_shapes);
    }
  } else {
    // Try to generically convert any LMHLO ops to LMHLO fusion + the
    // corresponding MHLO op. Currently we've only looked at elementwise ops and
    // they seem to be well covered.
    //
    // TODO(timshen): Moving forward, we should make it cover all ops if
    // possible, and only special-case the ones it can't.
    std::vector<mlir::Value> outputs;
    mlir::Operation* new_op;
    {
      auto operands = GetHloOperands(input.op);
      outputs = GetHloOutputs(input.op);
      TF_RET_CHECK(outputs.size() == 1) << MlirToString(input.op);

      std::vector<mlir::Value> loads = load_memrefs(operands);
      std::string mhlo_op_name = mlir::hlo::LmhloToMhloOpName(
          input.op->getName().getStringRef(), input.op->getContext());
      TF_RET_CHECK(!mhlo_op_name.empty())
          << "No corresponding MHLO op for given LMHLO op: "
          << MlirToString(input.op);
      mlir::OperationState op_state(loc, mhlo_op_name);

      mlir::BlockAndValueMapping mapper;
      for (mlir::Region& region : input.op->getRegions()) {
        mlir::Region* new_region = op_state.addRegion();
        region.cloneInto(new_region, mapper);
      }

      op_state.addOperands(loads);
      op_state.addAttributes(input.op->getAttrs());
      op_state.addTypes({mlir::RankedTensorType::get(
          outputs[0].getType().cast<mlir::MemRefType>().getShape(),
          outputs[0].getType().cast<mlir::MemRefType>().getElementType())});
      new_op = b.createOperation(op_state);
    }
    TF_RET_CHECK(mlir::succeeded(mlir::verify(new_op)));
    output_shape = TypeToShape(outputs[0].getType());
    HloFunctionImporter::SetLayoutForMlir(new_op, output_shape);
    b.create<mlir::memref::TensorStoreOp>(loc, new_op->getResult(0),
                                          outputs[0]);
  }
  int unroll_factor = 1;
  if (!MayPreventVectorization(input.op)) {
    unroll_factor = ComputeMaxUnrollFactor(input.op, hlo_module_config_);
  }
  input.op->erase();
  input.op = fusion;
  return EmitLoopFusionFromMlir(input, unroll_factor);
}

Status IrEmitterUnnested::EmitConstant(MlirEmitterInput mlir_input) {
  auto get_global = mlir::cast<mlir::memref::GetGlobalOp>(mlir_input.op);
  auto module = get_global->getParentOfType<mlir::ModuleOp>();
  auto global = mlir::cast<mlir::memref::GlobalOp>(
      module.lookupSymbol(get_global.name()));

  auto literal = global.initial_value()->dyn_cast<mlir::DenseElementsAttr>();
  TF_RET_CHECK(literal);

  const bool should_emit_initializer = literal.getType().getNumElements() <= 1;

  TF_ASSIGN_OR_RETURN(int element_bytes,
                      GetElementTypeBytes(literal.getType().getElementType()));
  llvm::ArrayType* global_type = llvm::ArrayType::get(
      b_.getInt8Ty(), literal.getType().getNumElements() * element_bytes);

  GpuExecutable::ConstantInfo info;
  llvm::Constant* initializer;
  if (should_emit_initializer) {
    std::vector<uint8> content;
    TF_RETURN_IF_ERROR(CopyDenseElementsDataToXlaFormat(literal, &content));
    initializer = llvm::ConstantDataArray::get<uint8>(
        ir_emitter_context_->llvm_module()->getContext(), content);
  } else {
    TF_RETURN_IF_ERROR(
        CopyDenseElementsDataToXlaFormat(literal, &info.content));
    initializer = llvm::ConstantAggregateZero::get(global_type);
  }

  // These globals will be looked up by name by GpuExecutable so we need to
  // give them an external linkage.  Not all of their uses are visible in
  // the LLVM IR so we can't give then a linkage that merely preserves their
  // names (like available_externally), we also need to ensure that they stick
  // around even if they're "unused".
  //
  // We may have to be more clever here in the future if we notice that we're
  // keeping around too many globals because of their linkage.
  llvm::GlobalVariable* global_for_const = new llvm::GlobalVariable(
      global_type, /*isConstant=*/should_emit_initializer,
      llvm::GlobalValue::ExternalLinkage,
      /*Initializer=*/initializer, global.sym_name(),
      /*TLMode=*/llvm::GlobalValue::NotThreadLocal,
      /*AddressSpace=*/0,
      /*isExternallyInitialized=*/false);
  global_for_const->setAlignment(llvm::Align(kConstantBufferAlignBytes));
  ir_emitter_context_->llvm_module()->getGlobalList().push_back(
      global_for_const);

  info.symbol_name.assign(global.sym_name().begin(), global.sym_name().end());

  info.allocation_index =
      global->getAttrOfType<mlir::IntegerAttr>("lmhlo.alloc").getInt();
  ir_emitter_context_->constants().push_back(std::move(info));
  return Status::OK();
}

static ConditionalThunkConfig GetConditionalThunkConfig(
    mlir::lmhlo::CaseOp op, std::vector<ThunkSequence> branch_thunk_sequences) {
  ConditionalThunkConfig config;
  config.branch_index_is_bool =
      op.index().getType().cast<mlir::ShapedType>().getElementType().isInteger(
          /*width=*/1);
  config.branch_count = op.branches().size();
  // Pass nullptr as the HloInstruction* to the branch_thunks
  // constructors because these SequentialThunks are logically "part of"
  // this ConditionalThunk, and shouldn't be profiled separately from it.
  config.branch_thunks.reserve(branch_thunk_sequences.size());
  for (auto& branch_thunk_sequence : branch_thunk_sequences) {
    config.branch_thunks.emplace_back(new SequentialThunk(
        Thunk::ThunkInfo(), std::move(branch_thunk_sequence)));
  }
  return config;
}

Status IrEmitterUnnested::EmitConditionalFromMlir(MlirEmitterInput mlir_input) {
  auto conditional = mlir::cast<mlir::lmhlo::CaseOp>(mlir_input.op);

  std::vector<ThunkSequence> branch_thunks;

  int branch_count = conditional.branches().size();
  branch_thunks.reserve(branch_count);

  for (int j = 0; j < branch_count; ++j) {
    mlir::Region* branch_computation = &conditional.branches()[j];
    TF_ASSIGN_OR_RETURN(
        auto ir_emitter,
        IrEmitterUnnested::Create(hlo_module_config_, ir_emitter_context_));
    TF_RETURN_IF_ERROR(ir_emitter->EmitLmhloRegion(branch_computation));
    branch_thunks.push_back(std::move(*ir_emitter->ConsumeThunkSequence()));
  }

  ConditionalThunkConfig config =
      GetConditionalThunkConfig(conditional, std::move(branch_thunks));

  TF_ASSIGN_OR_RETURN(auto slice,
                      GetAllocationSliceForMlir(conditional.index()));
  AddThunkToThunkSequence(std::unique_ptr<Thunk>(new ConditionalThunk(
      mlir_input.thunk_info, std::move(config), slice, {})));
  return Status::OK();
}

// Input = {dynamic array(with dynamic dimension meta data at the end)}
// Output = {static array, dynamic_dim0, dynamic_dim1}
Status IrEmitterUnnested::EmitPadToStaticFromMlir(MlirEmitterInput mlir_input) {
  // TODO(jurahul): Create an op to represent PadToStatic.
  auto pad_to_static = mlir::cast<mlir::lmhlo::CustomCallOp>(mlir_input.op);
  int unroll_factor = 1;
  std::string ir_name = mlir::GetNameFromLoc(pad_to_static.getLoc());

  std::vector<llvm_ir::IrArray> ir_arrays;
  TF_ASSIGN_OR_RETURN(auto kernel_thunk,
                      BuildKernelThunkForMlir(
                          pad_to_static, mlir_input.thunk_info, &ir_arrays));

  const llvm_ir::IrArray source_array = ir_arrays[0];
  const llvm_ir::IrArray output_array = ir_arrays[1];
  auto output_dim_arrays =
      absl::Span<const llvm_ir::IrArray>(ir_arrays).subspan(2);

  // pseudo code for PadToStatic on a 2d array
  //   int* source_array = input[0];
  //   int* dest_array = output[0];
  const Shape& data_shape =
      TypeToShape(pad_to_static.output().front().getType());
  const Shape& input_shape =
      TypeToShape(pad_to_static.args().front().getType());
  llvm::Value* source_buffer = source_array.GetBasePointer();
  llvm::Value* raw_buffer =
      b_.CreateBitCast(source_buffer, b_.getInt8Ty()->getPointerTo());

  // TODO(jurahul): input_shape here is the static shape of the input (which has
  // a dynamic shape in XLA). Currently, we are mapping that to a static shaped
  // memref. When we change that to a more appropriate representation in MLIR,
  // fix this code to correctly deduce the static shape backing the dynamically
  // shaped memref.
  int64 raw_data_size = ShapeUtil::ByteSizeOf(input_shape);

  //   int* dyn_dim0_size = source_array + meta_data_offset;
  //   int* dyn_dim1_size = source_array + meta_data_offset + sizeof(int);
  std::vector<llvm::Value*> dynamic_dims;
  for (int64 i = 1; i < pad_to_static.output().size(); ++i) {
    // Dynamic size of each dimension is attached at the end of the source
    // array(operand(0)). We need to extract these value.
    const Shape& dim_shape = TypeToShape(pad_to_static.output()[i].getType());
    TF_RET_CHECK(Shape::Equal()(dim_shape, ShapeUtil::MakeScalarShape(S32)));

    const int64 dim_index = i - 1;
    llvm::Value* metadata = b_.CreateConstInBoundsGEP1_32(
        b_.getInt8Ty(), raw_buffer, raw_data_size + dim_index * sizeof(int32));
    llvm::Value* dyn_dim_size = b_.CreateLoad(
        b_.CreateBitCast(metadata, b_.getInt32Ty()->getPointerTo()),
        "dyn_dim_size");
    dynamic_dims.push_back(dyn_dim_size);
  }

  // only one thread need to store the dynamic index
  //   int thread_id = GetThreadId();
  //   int block_id = GetBlockId();
  //   if (thread_id == 0 && block_id == 0) {
  //     *output[1] = *dyn_dim0_size;
  //     *output[2] = *dyn_dim1_size;
  //   }
  KernelSupportLibrary{&b_}.If("is_thred_0", IsBlock0Thread0(&b_), [&] {
    for (int64 i = 1; i < pad_to_static.output().size(); ++i) {
      const int64 dim_index = i - 1;
      llvm::Value* dest_dim_size_address =
          output_dim_arrays[dim_index].GetBasePointer();
      // output[i] stores dynamic_dim_(i-1)
      b_.CreateStore(dynamic_dims[i - 1],
                     b_.CreateBitCast(dest_dim_size_address,
                                      b_.getInt32Ty()->getPointerTo()));
    }
  });

  //     int dyn_element_total = 1;
  //     dyn_element_total *= *dyn_dim0_size;
  //     dyn_element_total *= *dyn_dim1_size;
  llvm::Value* dyn_element_total = llvm::ConstantInt::get(b_.getInt32Ty(), 1);
  for (llvm::Value* dynamic_dim : dynamic_dims) {
    dyn_element_total = b_.CreateMul(dyn_element_total, dynamic_dim,
                                     /*Name=*/"dyn_element_total");
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
  llvm_ir::LoopEmitter::BodyEmitter body_generator =
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
    return Status::OK();
  };

  TF_ASSIGN_OR_RETURN(LaunchDimensions launch_dimensions,
                      CalculateLaunchDimensions(
                          input_shape, ir_emitter_context_->gpu_device_info(),
                          {unroll_factor}));
  UpdateLaunchDimensions(launch_dimensions, kernel_thunk.get(),
                         ir_emitter_context_->llvm_module());
  TF_RETURN_IF_ERROR(
      ParallelLoopEmitter(body_generator, data_shape, launch_dimensions, &b_,
                          {unroll_factor})
          .EmitLoop(ir_name,
                    GetIndexTypeForKernelFromMlir(
                        pad_to_static, launch_dimensions.launch_bound(), &b_)));
  thunk_sequence_.emplace_back(std::move(kernel_thunk));
  return Status::OK();
}

// Input = {dynamic array(with dynamic dimension meta data at the end)}
// Output = {static array, dynamic_dim0, dynamic_dim1}
Status IrEmitterUnnested::EmitSliceToDynamicFromMlir(
    MlirEmitterInput mlir_input) {
  // TODO(jurahul): Create an op to represent SliceToDynamic.
  auto slice_to_dynamic = mlir::cast<mlir::lmhlo::CustomCallOp>(mlir_input.op);
  int unroll_factor = 1;
  std::string ir_name = mlir::GetNameFromLoc(slice_to_dynamic.getLoc());

  std::vector<llvm_ir::IrArray> ir_arrays;
  TF_ASSIGN_OR_RETURN(auto kernel_thunk,
                      BuildKernelThunkForMlir(
                          slice_to_dynamic, mlir_input.thunk_info, &ir_arrays));

  const Shape& input_shape =
      TypeToShape(slice_to_dynamic.args().front().getType());
  TF_RET_CHECK(slice_to_dynamic.output().size() == 1);
  const Shape& data_shape =
      TypeToShape(slice_to_dynamic.output().front().getType());

  // TODO(jurahul): data_shape here is the static shape of the output (which has
  // a dynamic shape in XLA). Currently, we are mapping that to a static shaped
  // memref. When we change that to a more appropriate representation in MLIR,
  // fix this code to correctly deduce the static shape backing the dynamically
  // shaped memref.

  // calculate the location where metadata needs to be inserted
  //   int* dyn_dim0_size = dest_array + meta_data_offset;
  //   int* dyn_dim1_size = dest_array + meta_data_offset + sizeof(int);
  int32 raw_data_size = ShapeUtil::ByteSizeOf(data_shape);

  // pseudo code for sliceToDynamic on a 2d array
  //   int* source_array = input[0];
  //   int* dest_array = output[0];
  const llvm_ir::IrArray data_array = ir_arrays.back();
  llvm::Value* dest_buffer = data_array.GetBasePointer();
  llvm::Value* raw_buffer =
      b_.CreateBitCast(dest_buffer, b_.getInt8Ty()->getPointerTo());

  // Load dynamic dimensions from memory.
  std::vector<llvm::Value*> dynamic_dims;
  for (int64 i = 1; i < slice_to_dynamic.args().size(); ++i) {
    // const int64 dim_index = i - 1;
    llvm::Value* source_buffer = ir_arrays[i].GetBasePointer();
    llvm::LoadInst* dyn_dim_size = b_.CreateLoad(source_buffer, "dyn_dim_size");
    dynamic_dims.push_back(dyn_dim_size);
  }

  // only one thread need to store the dynamic index
  //   int thread_id = GetThreadId();
  //   int block_id = GetBlockId();
  //   if (thread_id == 0 && block_id == 0) {
  //     *dyn_dim0_size = *output[1];
  //     *dyn_dim1_size = *output[2];
  //   }
  KernelSupportLibrary{&b_}.If("is_thred_0", IsBlock0Thread0(&b_), [&] {
    for (int64 i = 1; i < slice_to_dynamic.args().size(); ++i) {
      const int64 dim_index = i - 1;
      llvm::Value* metadata = b_.CreateConstInBoundsGEP1_32(
          b_.getInt8Ty(), raw_buffer,
          raw_data_size + dim_index * sizeof(int32));
      // output[i] stores dynamic_dim_(i-1)
      b_.CreateStore(
          dynamic_dims[dim_index],
          b_.CreateBitCast(metadata, b_.getInt32Ty()->getPointerTo()));
    }
  });

  //     int dyn_element_total = 1;
  //     dyn_element_total *= dyn_dim0_size;
  //     dyn_element_total *= dyn_dim1_size;
  llvm::Value* dyn_element_total = llvm::ConstantInt::get(b_.getInt32Ty(), 1);
  for (llvm::Value* dynamic_dim : dynamic_dims) {
    dyn_element_total = b_.CreateMul(dyn_element_total, dynamic_dim,
                                     /*Name=*/"dyn_element_total");
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
  llvm_ir::LoopEmitter::BodyEmitter body_generator =
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
    return Status::OK();
  };

  TF_ASSIGN_OR_RETURN(LaunchDimensions launch_dimensions,
                      CalculateLaunchDimensions(
                          input_shape, ir_emitter_context_->gpu_device_info(),
                          {unroll_factor}));
  UpdateLaunchDimensions(launch_dimensions, kernel_thunk.get(),
                         ir_emitter_context_->llvm_module());

  TF_RETURN_IF_ERROR(
      ParallelLoopEmitter(body_generator, data_shape, launch_dimensions, &b_,
                          {unroll_factor})
          .EmitLoop(ir_name, GetIndexTypeForKernelFromMlir(
                                 slice_to_dynamic,
                                 launch_dimensions.launch_bound(), &b_)));
  thunk_sequence_.emplace_back(std::move(kernel_thunk));
  return Status::OK();
}

Status IrEmitterUnnested::EmitCustomCallFromMlir(MlirEmitterInput input) {
  using mlir::dyn_cast;
  using mlir::isa;

  if (auto call = dyn_cast<mlir::lmhlo::CustomCallOp>(input.op)) {
    if (call.call_target_name() == "PadToStatic") {
      return EmitPadToStaticFromMlir(input);
    }
    if (call.call_target_name() == "SliceToDynamic") {
      return EmitSliceToDynamicFromMlir(input);
    }
    return EmitCustomCallThunkFromMlir(input);
  }

  if (isa<mlir::lmhlo_gpu::GEMMOp, mlir::lmhlo_gpu::GEMM_BiasOp>(input.op)) {
    return EmitGemmThunkFromMlir(input);
  }

  if (mlir::isa<mlir::lmhlo_gpu::ConvForwardOp,
                mlir::lmhlo_gpu::ConvForwardFusedOp,
                mlir::lmhlo_gpu::ConvForwardFusedSideInputOp,
                mlir::lmhlo_gpu::ConvBackwardFilterOp,
                mlir::lmhlo_gpu::ConvBackwardInputOp>(input.op)) {
    return EmitConvolutionThunkFromMlir(input);
  }

  if (isa<mlir::lmhlo_gpu::BatchNormTrainingOp,
          mlir::lmhlo_gpu::BatchNormInferenceOp,
          mlir::lmhlo_gpu::BatchNormGradOp>(input.op)) {
    return EmitBatchNormThunkFromMlir(input);
  }

#if GOOGLE_CUDA
  if (mlir::isa<mlir::lmhlo_gpu::CholeskyOp>(input.op)) {
    return EmitCholeskyThunkFromMlir(input);
  }
#endif  // GOOGLE_CUDA

  return Unimplemented("No registered implementation for custom call to \"%s\"",
                       MlirToString(input.op));
}

Status IrEmitterUnnested::EmitConvolutionThunkFromMlir(MlirEmitterInput input) {
  using mlir::dyn_cast;
  using mlir::lmhlo_gpu::Activation;
  using mlir::lmhlo_gpu::ConvBackwardFilterOp;
  using mlir::lmhlo_gpu::ConvBackwardInputOp;
  using mlir::lmhlo_gpu::ConvForwardFusedOp;
  using mlir::lmhlo_gpu::ConvForwardFusedSideInputOp;
  using mlir::lmhlo_gpu::ConvForwardOp;

  // Last 2 operands of the convolution operation are the result and scratch.
  std::vector<BufferAllocation::Slice> operand_slices;
  int64 num_operands = input.op->getNumOperands();
  operand_slices.reserve(num_operands - 2);
  for (mlir::Value operand : input.op->getOperands().drop_back(2)) {
    TF_ASSIGN_OR_RETURN(auto slice, GetAllocationSliceForMlir(operand));
    operand_slices.push_back(slice);
  }

  mlir::Value conv_result = input.op->getOperand(num_operands - 2);
  mlir::Value scratch_result = input.op->getOperand(num_operands - 1);
  TF_ASSIGN_OR_RETURN(auto conv_result_slice,
                      GetAllocationSliceForMlir(conv_result));
  TF_ASSIGN_OR_RETURN(auto scratch_slice,
                      GetAllocationSliceForMlir(scratch_result));

  auto apply_layout = [](const Shape& shape, mlir::ArrayAttr layout_attrib) {
    mlir::SmallVector<int64, 4> minor_to_major = llvm::to_vector<4>(
        llvm::map_range(layout_attrib, [](mlir::Attribute a) -> int64 {
          return static_cast<int64>(a.cast<mlir::IntegerAttr>().getInt());
        }));
    return ShapeUtil::MakeShapeWithLayout(shape.element_type(),
                                          shape.dimensions(), minor_to_major);
  };

  GpuConvDescriptor descriptor;

  auto fill_conv_descriptor = [&](auto op) {
    descriptor.operand0_shape =
        apply_layout(TypeToShape(input.op->getOperand(0).getType()),
                     op.backend_config().operand_0_layout());
    descriptor.operand1_shape =
        apply_layout(TypeToShape(input.op->getOperand(1).getType()),
                     op.backend_config().operand_1_layout());
    descriptor.result_shape = apply_layout(TypeToShape(conv_result.getType()),
                                           op.backend_config().result_layout());
    descriptor.dnums = ConvertConvDimensionNumbers(op.dimension_numbers());
    descriptor.scratch_size = scratch_slice.size();
    mlir::DenseIntElementsAttr window_strides = op.window_strides().getValue();
    mlir::DenseIntElementsAttr padding = op.padding().getValue();
    mlir::DenseIntElementsAttr lhs_dilation = op.lhs_dilation().getValue();
    mlir::DenseIntElementsAttr rhs_dilation = op.rhs_dilation().getValue();
    mlir::DenseElementsAttr window_reversal = op.window_reversal().getValue();
    for (auto index : llvm::seq<int>(0, window_strides.getNumElements())) {
      WindowDimension* dim = descriptor.window.add_dimensions();
      // Window size for a convolution is the same as the kernel size.
      // Kernel size of the convolution is operand1_shape. We need to look at
      // the convolution dimension numbers kernel spatial dimensions to get
      // the window size.
      int kernel_dim = descriptor.dnums.kernel_spatial_dimensions(index);
      dim->set_size(descriptor.operand0_shape.dimensions(kernel_dim));
      dim->set_stride(window_strides.getValue<int64>(index));
      dim->set_padding_low(padding.getValue<int64>(index));
      dim->set_padding_high(padding.getValue<int64>(index));
      dim->set_base_dilation(lhs_dilation.getValue<int64>(index));
      dim->set_window_dilation(rhs_dilation.getValue<int64>(index));
      dim->set_window_reversal(window_reversal.getValue<bool>(index));
    }
    descriptor.feature_group_count = op.feature_group_count();
    descriptor.backend_config.set_algorithm(
        op.backend_config().algorithm().getInt());
    descriptor.backend_config.set_tensor_ops_enabled(
        op.backend_config().tensor_ops_enabled().getValue());
    descriptor.backend_config.set_conv_result_scale(
        op.result_scale().convertToDouble());
  };

  auto set_activation_mode = [&](auto op) -> Status {
    TF_ASSIGN_OR_RETURN(stream_executor::dnn::ActivationMode activation_mode,
                        ConvertConvActivationMode(op.activation_mode()));
    descriptor.backend_config.set_activation_mode(
        static_cast<int64>(activation_mode));
    return Status::OK();
  };

  if (auto op = dyn_cast<ConvForwardOp>(input.op)) {
    descriptor.kind = CudnnConvKind::kForward;
    fill_conv_descriptor(op);
  } else if (auto op = dyn_cast<ConvBackwardInputOp>(input.op)) {
    descriptor.kind = CudnnConvKind::kBackwardInput;
    fill_conv_descriptor(op);
  } else if (auto op = dyn_cast<ConvBackwardFilterOp>(input.op)) {
    descriptor.kind = CudnnConvKind::kBackwardFilter;
    fill_conv_descriptor(op);
  } else if (auto op = dyn_cast<ConvForwardFusedOp>(input.op)) {
    descriptor.kind = CudnnConvKind::kForwardActivation;
    fill_conv_descriptor(op);
    TF_RETURN_IF_ERROR(set_activation_mode(op));
  } else if (auto op = dyn_cast<ConvForwardFusedSideInputOp>(input.op)) {
    descriptor.kind = CudnnConvKind::kForwardActivation;
    fill_conv_descriptor(op);
    TF_RETURN_IF_ERROR(set_activation_mode(op));
    descriptor.backend_config.set_side_input_scale(
        op.side_input_scale().convertToDouble());
  } else {
    return InternalError("Unexpected operation");
  }
  TF_ASSIGN_OR_RETURN(GpuConvConfig config, GetGpuConvConfig(descriptor, ""));
  AddThunkToThunkSequence(absl::make_unique<ConvolutionThunk>(
      input.thunk_info, std::move(config), std::move(operand_slices),
      conv_result_slice, scratch_slice));
  return Status::OK();
}

Status IrEmitterUnnested::EmitGemmThunkFromMlir(MlirEmitterInput input) {
  auto build_gemm_config = [](auto op, absl::optional<double> gemm_bias_beta =
                                           absl::nullopt) {
    GpuGemmConfig config;
    GemmBackendConfig& backend = config.backend_config;
    config.output_shape = TypeToShape(op.output().getType());
    config.lhs_shape = TypeToShape(op.lhs().getType());
    config.rhs_shape = TypeToShape(op.rhs().getType());
    backend.Clear();
    if (op.algorithm()) {
      backend.set_selected_algorithm(*op.algorithm());
    }
    backend.set_alpha_real(op.alpha_real().convertToDouble());
    backend.set_alpha_imag(op.alpha_imag().convertToDouble());
    backend.set_batch_size(op.batch_size());
    if (gemm_bias_beta.has_value()) {
      backend.set_beta(gemm_bias_beta.value());
    }

    auto& dims = *backend.mutable_dot_dimension_numbers();
    auto mlir_dims = op.dot_dimension_numbers();

    auto fill_dims = [](mlir::DenseElementsAttr mlir_dim, auto* config_attrs) {
      for (llvm::APInt e : mlir_dim.getIntValues())
        config_attrs->Add(e.getSExtValue());
    };
    fill_dims(mlir_dims.lhs_batching_dimensions(),
              dims.mutable_lhs_batch_dimensions());
    fill_dims(mlir_dims.rhs_batching_dimensions(),
              dims.mutable_rhs_batch_dimensions());
    fill_dims(mlir_dims.lhs_contracting_dimensions(),
              dims.mutable_lhs_contracting_dimensions());
    fill_dims(mlir_dims.rhs_contracting_dimensions(),
              dims.mutable_rhs_contracting_dimensions());
    return config;
  };

  GpuGemmConfig config;
  BufferAllocation::Slice lhs, rhs, bias, output;

  const bool use_bef_thunk = BefThunk::SupportsOp(input.op);
  auto make_thunk_for_gemm = [&](bool implements_whole_instruction)
      -> StatusOr<std::unique_ptr<Thunk>> {
    std::unique_ptr<Thunk> thunk;
    if (use_bef_thunk) {
      TF_ASSIGN_OR_RETURN(
          thunk,
          BefThunk::Create(input.thunk_info, input.op,
                           std::vector<BufferAllocation::Slice>{lhs, rhs},
                           std::vector<BufferAllocation::Slice>{output}));
    } else {
      thunk = absl::make_unique<GemmThunk>(input.thunk_info, std::move(config),
                                           lhs, rhs, output,
                                           implements_whole_instruction);
    }
    return thunk;
  };

  if (auto gemm = mlir::dyn_cast<mlir::lmhlo_gpu::GEMMOp>(input.op)) {
    if (!use_bef_thunk) config = build_gemm_config(gemm);
    TF_ASSIGN_OR_RETURN(lhs, GetAllocationSliceForMlir(gemm.lhs()));
    TF_ASSIGN_OR_RETURN(rhs, GetAllocationSliceForMlir(gemm.rhs()));
    TF_ASSIGN_OR_RETURN(output, GetAllocationSliceForMlir(gemm.output()));
  } else if (auto gemm_bias =
                 mlir::dyn_cast<mlir::lmhlo_gpu::GEMM_BiasOp>(input.op)) {
    if (!use_bef_thunk)
      config = build_gemm_config(gemm_bias, gemm_bias.beta().convertToDouble());
    TF_ASSIGN_OR_RETURN(lhs, GetAllocationSliceForMlir(gemm_bias.lhs()));
    TF_ASSIGN_OR_RETURN(rhs, GetAllocationSliceForMlir(gemm_bias.rhs()));
    TF_ASSIGN_OR_RETURN(bias, GetAllocationSliceForMlir(gemm_bias.bias()));
    TF_ASSIGN_OR_RETURN(output, GetAllocationSliceForMlir(gemm_bias.output()));

    // The bias is passed inside the output buffer. If those buffers are shared
    // we can just use it, otherwise copy the bias values into the output buffer
    // first.
    if (bias != output) {
      ThunkSequence thunks;

      thunks.push_back(absl::make_unique<DeviceToDeviceCopyThunk>(
          Thunk::ThunkInfo(),
          /*source_buffer=*/bias,
          /*destination_buffer=*/output,
          /*mem_size=*/
          ShapeUtil::ByteSizeOf(TypeToShape(gemm_bias.output().getType()))));
      TF_ASSIGN_OR_RETURN(
          auto thunk,
          make_thunk_for_gemm(/*implements_whole_instruction=*/false));
      thunks.push_back(std::move(thunk));
      AddThunkToThunkSequence(absl::make_unique<SequentialThunk>(
          input.thunk_info, std::move(thunks)));
      return Status::OK();
    }
  }

  TF_ASSIGN_OR_RETURN(
      auto thunk, make_thunk_for_gemm(/*implements_whole_instruction=*/true));
  AddThunkToThunkSequence(std::move(thunk));
  return Status::OK();
}

namespace {
// An MLIR value and its name as defined in the ODS spec.
struct NamedValue {
  mlir::Value value;
  absl::string_view name;
};

// Verifies that the given batch norm is well formed for thunk emission. This
// requires that all statistics operands (mean, stddev etc) are F32 types and
// all the non-statistics operands need to match in shape, element type, and
// layout (which maps to them having the same memref type).
Status VerifyBatchNormForThunkEmission(
    mlir::ArrayRef<NamedValue> statistics_operands,
    mlir::ArrayRef<NamedValue> other_operands) {
  for (const NamedValue& v : statistics_operands) {
    // Note: MLIR verification will ensure that the operands of the batchnorm
    // LHLO are valid memref types.
    if (!v.value.getType().cast<mlir::MemRefType>().getElementType().isF32()) {
      return Unimplemented("Operand %s of batch norm should have F32 type",
                           v.name);
    }
  }
  if (other_operands.empty()) {
    return Status::OK();
  }

  mlir::Type first_type = other_operands.front().value.getType();
  absl::string_view first_name = other_operands.front().name;

  for (const NamedValue& v : other_operands.drop_front(1)) {
    if (v.value.getType() != first_type) {
      return Unimplemented("%s and %s for batch norm should have same types",
                           v.name, first_name);
    }
  }

  return Status::OK();
}

// Determine if we enable the row optimized codegen.  When we have a
// fusion with only point-wise operations, scalar broadcasting and row
// broadcasting, we can trigger a kernel that vectorize the row loads.
// This speed up the kernel, in particular on A100.
bool RowVectorizationEnabled(mlir::lmhlo::FusionOp fusion) {
  const auto is_row_major = [](mlir::Value value) {
    // Only tested when the inputs are row-major. So only
    // enable that case. Maybe it would works if only the
    // inner dimensions is contiguous.
    return LayoutUtil::IsMonotonicWithDim0Major(
        TypeToShape(value.getType()).layout());
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
  for (mlir::Operation& op : fusion.region().front()) {
    if (mlir::isa<mlir::memref::TensorLoadOp, mlir::memref::TensorStoreOp,
                  mlir::lmhlo::TerminatorOp, mlir::mhlo::ReturnOp,
                  mlir::mhlo::ConstOp, mlir::lmhlo::ConstOp>(op)) {
      continue;
    }
    HloOpcode opcode = *MhloToHloOpcode(&op);
    if (HloInstruction::IsOpElementwise(opcode)) {
      continue;
    }

    if (auto broadcast = mlir::dyn_cast<mlir::mhlo::BroadcastInDimOp>(op)) {
      if (broadcast.broadcast_dimensions().size() == 0) {
        continue;
      }
      std::vector<int64> broadcast_dimensions;
      for (const llvm::APInt& int_value : broadcast.broadcast_dimensions()) {
        broadcast_dimensions.push_back(int_value.getSExtValue());
      }

      auto rank = TypeToShape(broadcast.getResult().getType()).rank();
      if (broadcast_dimensions.size() == 1 &&
          broadcast_dimensions.back() == (rank - 1)) {
        some_row_broadcasting = true;
        continue;
      }
    }
    VLOG(2) << "Row vectorization not enabled due to this op: "
            << MlirToString(&op);
    return false;
  }
  // Trigger only when there is a row broadcasting.
  return row_vectorized && some_row_broadcasting;
}
}  // namespace

Status IrEmitterUnnested::EmitBatchNormThunkFromMlir(MlirEmitterInput input) {
  auto get_batch_norm_config = [](auto op, mlir::Value output) {
    CudnnBatchNormConfig config;
    config.output_shape = TypeToShape(output.getType());
    config.output_type = config.output_shape.element_type();
    config.epsilon = op.epsilon().convertToFloat();
    config.feature_index = op.feature_index();
    return config;
  };

  // The statistics operands for batch norm operations need to be FP32 type.
  // And the rest of the operands need match in shape, layout, and element type
  // to match.
  if (auto bn_train =
          mlir::dyn_cast<mlir::lmhlo_gpu::BatchNormTrainingOp>(input.op)) {
    TF_RETURN_IF_ERROR(VerifyBatchNormForThunkEmission(
        /*statistics_operands=*/
        {{bn_train.scale(), "scale"},
         {bn_train.offset(), "offset"},
         {bn_train.batch_mean(), "batch_mean"},
         {bn_train.batch_stddev(), "batch_stddev"}},
        /*other_operands=*/
        {{bn_train.operand(), "operand"}, {bn_train.output(), "output"}}));
    TF_ASSIGN_OR_RETURN(auto operand,
                        GetAllocationSliceForMlir(bn_train.operand()));
    TF_ASSIGN_OR_RETURN(auto scale,
                        GetAllocationSliceForMlir(bn_train.scale()));
    TF_ASSIGN_OR_RETURN(auto offset,
                        GetAllocationSliceForMlir(bn_train.offset()));

    // BatchNormTraining returns a tuple of three elements: data, calculated
    // mean, and calculated 1/sqrt(variance + epsilon).
    TF_ASSIGN_OR_RETURN(auto output_data,
                        GetAllocationSliceForMlir(bn_train.output()));
    TF_ASSIGN_OR_RETURN(auto output_mean,
                        GetAllocationSliceForMlir(bn_train.batch_mean()));
    TF_ASSIGN_OR_RETURN(auto output_inv_stddev,
                        GetAllocationSliceForMlir(bn_train.batch_stddev()));

    AddThunkToThunkSequence(
        absl::make_unique<CudnnBatchNormForwardTrainingThunk>(
            input.thunk_info,
            /*config=*/get_batch_norm_config(bn_train, bn_train.output()),
            /*operand=*/operand,
            /*scale=*/scale,
            /*offset=*/offset,
            /*output_data=*/output_data,
            /*output_mean=*/output_mean,
            /*output_inv_stddev=*/output_inv_stddev));
    return Status::OK();
  }

  if (auto bn_grad =
          mlir::dyn_cast<mlir::lmhlo_gpu::BatchNormGradOp>(input.op)) {
    TF_RETURN_IF_ERROR(VerifyBatchNormForThunkEmission(
        /*statistics_operands=*/
        {{bn_grad.scale(), "scale"},
         {bn_grad.mean(), "mean"},
         {bn_grad.stddev(), "stddev"},
         {bn_grad.grad_scale(), "grad_scale"},
         {bn_grad.grad_offset(), "grad_offset"}},
        /*other_operands=*/
        {{bn_grad.operand(), "operand"},
         {bn_grad.grad_output(), "grad_output"},
         {bn_grad.grad_operand(), "grad_operand"}}));

    TF_ASSIGN_OR_RETURN(auto operand,
                        GetAllocationSliceForMlir(bn_grad.operand()));
    TF_ASSIGN_OR_RETURN(auto scale, GetAllocationSliceForMlir(bn_grad.scale()));
    TF_ASSIGN_OR_RETURN(auto mean, GetAllocationSliceForMlir(bn_grad.mean()));
    TF_ASSIGN_OR_RETURN(auto inv_stddev,
                        GetAllocationSliceForMlir(bn_grad.stddev()));
    TF_ASSIGN_OR_RETURN(auto grad_output,
                        GetAllocationSliceForMlir(bn_grad.grad_output()));

    // BatchNormGrad returns a tuple of three elements: grad_data, grad_scale,
    // grad_offset.
    TF_ASSIGN_OR_RETURN(auto output_grad_data,
                        GetAllocationSliceForMlir(bn_grad.grad_operand()));
    TF_ASSIGN_OR_RETURN(auto output_grad_scale,
                        GetAllocationSliceForMlir(bn_grad.grad_scale()));
    TF_ASSIGN_OR_RETURN(auto output_grad_offset,
                        GetAllocationSliceForMlir(bn_grad.grad_offset()));

    CudnnBatchNormConfig config;
    config.output_shape = TypeToShape(bn_grad.grad_output().getType());
    config.output_type = config.output_shape.element_type();
    config.epsilon = bn_grad.epsilon().convertToFloat();
    config.feature_index = bn_grad.feature_index();

    AddThunkToThunkSequence(absl::make_unique<CudnnBatchNormBackwardThunk>(
        input.thunk_info,
        /*config=*/get_batch_norm_config(bn_grad, bn_grad.grad_output()),
        /*operand=*/operand,
        /*scale=*/scale,
        /*mean=*/mean,
        /*inv_stddev=*/inv_stddev,
        /*grad_output=*/grad_output,
        /*output_grad_data=*/output_grad_data,
        /*output_grad_scale=*/output_grad_scale,
        /*output_grad_offset=*/output_grad_offset));
    return Status::OK();
  }

  if (auto bn_inference =
          mlir::dyn_cast<mlir::lmhlo_gpu::BatchNormInferenceOp>(input.op)) {
    TF_RETURN_IF_ERROR(
        VerifyBatchNormForThunkEmission(/*statistics_operands=*/
                                        {{bn_inference.scale(), "scale"},
                                         {bn_inference.offset(), "offset"},
                                         {bn_inference.mean(), "mean"},
                                         {bn_inference.stddev(), "stddev"}},
                                        /*other_operands=*/
                                        {{bn_inference.operand(), "operand"},
                                         {bn_inference.output(), "output"}}));

    TF_ASSIGN_OR_RETURN(auto operand,
                        GetAllocationSliceForMlir(bn_inference.operand()));
    TF_ASSIGN_OR_RETURN(auto scale,
                        GetAllocationSliceForMlir(bn_inference.scale()));
    TF_ASSIGN_OR_RETURN(auto offset,
                        GetAllocationSliceForMlir(bn_inference.offset()));
    TF_ASSIGN_OR_RETURN(auto mean,
                        GetAllocationSliceForMlir(bn_inference.mean()));
    TF_ASSIGN_OR_RETURN(auto variance,
                        GetAllocationSliceForMlir(bn_inference.stddev()));
    TF_ASSIGN_OR_RETURN(auto output,
                        GetAllocationSliceForMlir(bn_inference.output()));

    AddThunkToThunkSequence(absl::make_unique<
                            CudnnBatchNormForwardInferenceThunk>(
        input.thunk_info,
        /*config=*/get_batch_norm_config(bn_inference, bn_inference.output()),
        /*operand=*/operand,
        /*scale=*/scale,
        /*offset=*/offset,
        /*mean=*/mean,
        /*variance=*/variance,
        /*output=*/output));
    return Status::OK();
  }

  return Unimplemented("Unsupported batch norm operation");
}

#if GOOGLE_CUDA
Status IrEmitterUnnested::EmitCholeskyThunkFromMlir(MlirEmitterInput input) {
  auto cholesky_op = mlir::cast<mlir::lmhlo_gpu::CholeskyOp>(input.op);

  const Shape shape = TypeToShape(cholesky_op.input().getType());
  int ndim = shape.dimensions_size();
  CHECK_GE(ndim, 2);
  int64 n = shape.dimensions(ndim - 1);

  const auto& dims = shape.dimensions();
  int64 batch_size = std::accumulate(dims.begin(), dims.end() - 2, int64{1},
                                     [](int64 a, int64 b) { return a * b; });

  TF_ASSIGN_OR_RETURN(auto operand_buffer,
                      GetAllocationSliceForMlir(cholesky_op.input()));
  TF_ASSIGN_OR_RETURN(auto a_buffer,
                      GetAllocationSliceForMlir(cholesky_op.output()));
  TF_ASSIGN_OR_RETURN(auto workspace_buffer,
                      GetAllocationSliceForMlir(cholesky_op.scratch()));
  TF_ASSIGN_OR_RETURN(auto info_buffer,
                      GetAllocationSliceForMlir(cholesky_op.info()));

  ThunkSequence thunks;

  if (operand_buffer != a_buffer) {
    thunks.push_back(absl::make_unique<DeviceToDeviceCopyThunk>(
        input.thunk_info,
        /*source_address=*/operand_buffer,
        /*destination_buffer=*/a_buffer,
        /*mem_size=*/ShapeUtil::ByteSizeOf(shape)));
  }

  CholeskyOptions options;
  options.set_lower(cholesky_op.is_lower());
  thunks.push_back(absl::make_unique<CholeskyThunk>(
      input.thunk_info, options, a_buffer, workspace_buffer, info_buffer,
      shape.element_type(), batch_size, n));

  // Elide the sequential thunk if there's no copy.
  if (thunks.size() == 1) {
    AddThunkToThunkSequence(std::move(thunks[0]));
  } else {
    AddThunkToThunkSequence(absl::make_unique<SequentialThunk>(
        input.thunk_info, std::move(thunks)));
  }

  return Status::OK();
}
#endif  // GOOGLE_CUDA

Status IrEmitterUnnested::EmitCustomCallThunkFromMlir(MlirEmitterInput input) {
  auto custom_call = mlir::cast<mlir::lmhlo::CustomCallOp>(input.op);
  const std::string call_target_name = custom_call.call_target_name().str();

  void* call_target = CustomCallTargetRegistry::Global()->Lookup(
      call_target_name, std::string(platform_name()));
  if (!call_target) {
    return Unimplemented(
        "No registered implementation for custom call to \"%s\"",
        call_target_name);
  }

  std::vector<CustomCallThunk::OptionalSlice> operands;
  std::vector<CustomCallThunk::OptionalSlice> results;

  if (custom_call.target_arg_mapping()) {
    auto values_to_slices_with_token_holes =
        [&](mlir::ValueRange operands, mlir::ArrayAttr op_to_target_mapping,
            mlir::IntegerAttr num_target)
        -> StatusOr<std::vector<CustomCallThunk::OptionalSlice>> {
      std::vector<CustomCallThunk::OptionalSlice> slices(num_target.getInt());
      for (auto index_and_value_it :
           llvm::zip(op_to_target_mapping, operands)) {
        mlir::Attribute index_attr = std::get<0>(index_and_value_it);
        mlir::Value value = std::get<1>(index_and_value_it);
        int64 index = index_attr.cast<mlir::IntegerAttr>().getInt();
        TF_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                            GetAllocationSliceForMlir(value));
        slices[index] = slice;
      }
      return slices;
    };

    mlir::lmhlo::CustomCallTargetArgMapping target_mapping =
        *custom_call.target_arg_mapping();
    TF_ASSIGN_OR_RETURN(
        operands, values_to_slices_with_token_holes(
                      custom_call.args(), target_mapping.args_to_target_args(),
                      target_mapping.num_args()));
    TF_ASSIGN_OR_RETURN(results, values_to_slices_with_token_holes(
                                     custom_call.output(),
                                     target_mapping.results_to_target_results(),
                                     target_mapping.num_results()));
  } else {
    auto values_to_slices = [&](mlir::ValueRange values)
        -> StatusOr<std::vector<CustomCallThunk::OptionalSlice>> {
      std::vector<CustomCallThunk::OptionalSlice> slices;
      for (mlir::Value value : values) {
        TF_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                            GetAllocationSliceForMlir(value));
        slices.push_back(slice);
      }
      return slices;
    };

    TF_ASSIGN_OR_RETURN(operands, values_to_slices(custom_call.args()));
    TF_ASSIGN_OR_RETURN(results, values_to_slices(custom_call.output()));
  }

  AddThunkToThunkSequence(absl::make_unique<CustomCallThunk>(
      input.thunk_info, call_target, std::move(operands), std::move(results),
      custom_call.backend_config().str()));
  return Status::OK();
}

Status IrEmitterUnnested::EmitFftThunkFromMlir(MlirEmitterInput input) {
  auto fft_op = mlir::cast<mlir::lmhlo::FftOp>(input.op);
  const Shape operand_shape = TypeToShape(fft_op.operand().getType());
  const Shape output_shape = TypeToShape(fft_op.output().getType());
  TF_RET_CHECK(LayoutUtil::IsMonotonicWithDim0Major(operand_shape.layout()));
  TF_RET_CHECK(LayoutUtil::IsMonotonicWithDim0Major(output_shape.layout()));

  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice arg_slice,
                      GetAllocationSliceForMlir(fft_op.operand()));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice dest_slice,
                      GetAllocationSliceForMlir(fft_op.output()));
  TF_ASSIGN_OR_RETURN(xla::FftType fft_type, ConvertFftType(fft_op.fft_type()));
  auto fft_length_values = fft_op.fft_length().getValues<int64>();
  std::vector<int64> fft_length(fft_length_values.begin(),
                                fft_length_values.end());
  AddThunkToThunkSequence(
      absl::make_unique<FftThunk>(input.thunk_info, fft_type, fft_length,
                                  /*input_buffer=*/arg_slice,
                                  /*output_buffer=*/dest_slice,
                                  /*input_shape=*/operand_shape,
                                  /*output_shape=*/output_shape));
  return Status::OK();
}

Status IrEmitterUnnested::EmitTriangularSolveFromMlir(MlirEmitterInput input) {
  auto triangular_solve_op =
      mlir::cast<mlir::lmhlo::TriangularSolveOp>(input.op);
  auto has_fortran_layout = [](mlir::DenseIntElementsAttr layout_attr) {
    int64_t n = layout_attr.getNumElements();
    return layout_attr.getValue<int64_t>({0}) == n - 2 &&
           layout_attr.getValue<int64_t>({1}) == n - 1;
  };
  TF_RET_CHECK(has_fortran_layout(triangular_solve_op.layout_a()));
  TF_RET_CHECK(has_fortran_layout(triangular_solve_op.layout_b()));
  TF_RET_CHECK(has_fortran_layout(triangular_solve_op.layout_output()));

  const Shape b_shape = TypeToShape(triangular_solve_op.b().getType());

  const Shape output_shape =
      TypeToShape(triangular_solve_op.output().getType());

  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice a_slice,
                      GetAllocationSliceForMlir(triangular_solve_op.a()));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice b_slice,
                      GetAllocationSliceForMlir(triangular_solve_op.b()));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice output_slice,
                      GetAllocationSliceForMlir(triangular_solve_op.output()));
  TF_ASSIGN_OR_RETURN(TriangularSolveOptions_Transpose transpose_a,
                      ConvertTranspose(triangular_solve_op.transpose_a()));

  ThunkSequence thunks;

  // Triangular solve is in-place on 'b', so copy 'b' to the output if they
  // aren't the same buffer.
  if (b_slice != output_slice) {
    thunks.push_back(absl::make_unique<DeviceToDeviceCopyThunk>(
        Thunk::ThunkInfo(),
        /*source_address=*/b_slice,
        /*destination_buffer=*/output_slice,
        /*mem_size=*/ShapeUtil::ByteSizeOf(b_shape)));
  }

  int64 m = b_shape.dimensions(b_shape.rank() - 2);
  int64 n = b_shape.dimensions(b_shape.rank() - 1);
  int64 batch_size = std::accumulate(b_shape.dimensions().begin(),
                                     b_shape.dimensions().end() - 2, int64{1},
                                     [](int64 a, int64 b) { return a * b; });
  int64 elem_size =
      ShapeUtil::ByteSizeOfPrimitiveType(output_shape.element_type());
  int64 a_batch_stride =
      triangular_solve_op.left_side() ? m * m * elem_size : n * n * elem_size;
  int64 b_batch_stride = m * n * elem_size;
  TriangularSolveOptions options;
  options.set_left_side(triangular_solve_op.left_side());
  options.set_lower(triangular_solve_op.lower());
  options.set_unit_diagonal(triangular_solve_op.unit_diagonal());
  options.set_transpose_a(transpose_a);
  thunks.push_back(absl::make_unique<TriangularSolveThunk>(
      input.thunk_info, options,
      /*a_input_buffer=*/a_slice,
      /*b_input_buffer=*/output_slice, output_shape.element_type(), batch_size,
      m, n, a_batch_stride, b_batch_stride));

  // Elide the sequential thunk if there's no copy.
  if (thunks.size() == 1) {
    AddThunkToThunkSequence(std::move(thunks[0]));
  } else {
    AddThunkToThunkSequence(absl::make_unique<SequentialThunk>(
        input.thunk_info, std::move(thunks)));
  }
  return Status::OK();
}

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
  std::vector<mlir::memref::TensorLoadOp> loads;
  std::vector<mlir::memref::TensorStoreOp> stores;

  region->walk([&](mlir::memref::TensorLoadOp load) {
    if (load.memref().getParentRegion() != region) {
      loads.push_back(load);
    }
  });

  region->walk([&](mlir::memref::TensorStoreOp store) {
    if (store.memref().getParentRegion() != region) {
      stores.push_back(store);
    }
  });

  for (auto load : loads) {
    auto arg = region->addArgument(load.getType());
    load.replaceAllUsesWith(arg);
    Shape shape = TypeToShape(load.getType());
    if (auto attr = mlir::GetLayoutFromMlirHlo(load)) {
      std::vector<int64> minor_to_major;
      absl::c_transform(
          attr, std::back_inserter(minor_to_major),
          std::function<int64(const llvm::APInt&)>(&llvm::APInt::getZExtValue));
      *shape.mutable_layout() = LayoutUtil::MakeLayout(minor_to_major);
    } else {
      *shape.mutable_layout() =
          LayoutUtil::MakeDescendingLayout(load.getType().getShape().size());
    }
    operand_shapes->push_back(std::move(shape));
    load.erase();
  }

  std::vector<mlir::Value> returned_values;
  for (auto store : stores) {
    Shape shape = TypeToShape(store.memref().getType());
    if (auto attr = mlir::GetLayoutFromMlirHlo(store)) {
      std::vector<int64> minor_to_major;
      absl::c_transform(
          attr, std::back_inserter(minor_to_major),
          std::function<int64(const llvm::APInt&)>(&llvm::APInt::getZExtValue));
      *shape.mutable_layout() = LayoutUtil::MakeLayout(minor_to_major);
    }
    output_shapes->push_back(shape);

    returned_values.push_back(store.tensor());
    store.erase();
  }

  region->back().back().erase();
  auto b = mlir::OpBuilder::atBlockEnd(&region->back());
  auto loc = returned_values[0].getLoc();
  b.create<mlir::mhlo::ReturnOp>(loc, returned_values);
  return Status::OK();
}

// TODO(timshen): update the comment once the HandleFusion code path deleted.
//
// This is migrated from IrEmitter::HandleFusion() with IrEmitterUnnested as the
// subclass. The logic is de-virtualized and less scattered.
Status IrEmitterUnnested::EmitLoopFusionFromMlir(
    MlirEmitterInput input, absl::optional<int> unroll_factor_override) {
  auto fusion = mlir::cast<mlir::lmhlo::FusionOp>(input.op);
  MlirEmitterContext context;
  context.SetOperation(fusion);

  std::vector<llvm_ir::IrArray> ir_arrays;
  Thunk* kernel_thunk;
  {
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<KernelThunk> kernel_thunk_ptr,
        BuildKernelThunkForMlir(fusion, input.thunk_info, &ir_arrays));
    kernel_thunk = kernel_thunk_ptr.get();
    thunk_sequence_.emplace_back(std::move(kernel_thunk_ptr));
  }

  auto operand_arrays =
      absl::MakeSpan(ir_arrays).subspan(0, context.operand_shapes.size());
  auto output_element_arrays = absl::MakeSpan(ir_arrays).subspan(
      context.operand_shapes.size(), context.output_shapes.size());

  TF_ASSIGN_OR_RETURN(const HloComputation* fused_computation,
                      GetOrCreateSubComputationFromRegion(&fusion.region(),
                                                          /*is_fusion=*/true));

  GpuElementalIrEmitter elemental_emitter(hlo_module_config_, module_, &b_,
                                          GetNestedComputer());
  FusedIrEmitter fused_emitter(&elemental_emitter);

  for (int i = 0; i < context.operand_shapes.size(); i++) {
    auto* builder = &b_;
    auto ir_array = operand_arrays[i];
    fused_emitter.BindGenerator(
        fused_computation->parameter_instruction(i),
        [builder, ir_array](llvm_ir::IrArray::Index index) {
          return ir_array.EmitReadArrayElement(index, builder);
        });
  }
  TF_ASSIGN_OR_RETURN(
      auto element_generator,
      fused_emitter.GetGenerator(fused_computation->root_instruction()));

  int unroll_factor;
  if (unroll_factor_override.has_value()) {
    unroll_factor = *unroll_factor_override;
  } else if (!MayPreventVectorization(fusion)) {
    unroll_factor = ComputeMaxUnrollFactor(fusion, hlo_module_config_);
  } else {
    unroll_factor = 1;
  }

  bool few_waves = [fusion]() mutable {
    for (mlir::Operation& op : fusion.region().front()) {
      if (mlir::isa<mlir::memref::TensorLoadOp, mlir::memref::TensorStoreOp,
                    mlir::lmhlo::TerminatorOp, mlir::mhlo::ReturnOp>(op)) {
        continue;
      }
      HloOpcode opcode = *MhloToHloOpcode(&op);
      if (HloInstruction::IsOpElementwise(opcode)) {
        continue;
      }
      if (auto broadcast = mlir::dyn_cast<mlir::mhlo::BroadcastOp>(op)) {
        if (broadcast.broadcast_sizes().size() == 0) {
          continue;
        }
      }
      return false;
    }
    return true;
  }();

  bool row_vectorized = RowVectorizationEnabled(fusion);
  Shape element_shape = context.output_shapes[0];
  LaunchDimensionsConfig launch_config{unroll_factor, few_waves,
                                       row_vectorized};
  // Check that the shapes is supported.
  launch_config.row_vectorized &=
      ThreadsPerBlockRowVectorized(element_shape,
                                   ir_emitter_context_->gpu_device_info(),
                                   launch_config) > 0;

  TF_ASSIGN_OR_RETURN(LaunchDimensions launch_dimensions,
                      CalculateLaunchDimensions(
                          element_shape, ir_emitter_context_->gpu_device_info(),
                          launch_config));
  UpdateLaunchDimensions(launch_dimensions, kernel_thunk,
                         ir_emitter_context_->llvm_module());
  llvm::Type* index_type = GetIndexTypeForKernelFromMlir(
      fusion, launch_dimensions.launch_bound(), &b_);

  if (context.output_shapes.size() > 1) {
    // For multioutput fusion, we need to emit each operand and the root.
    TF_RETURN_IF_ERROR(
        ParallelLoopEmitter(element_generator, output_element_arrays,
                            launch_dimensions, &b_, launch_config)
            .EmitLoop(context.name, index_type));
  } else {
    TF_RETURN_IF_ERROR(
        ParallelLoopEmitter(element_generator, output_element_arrays[0],
                            launch_dimensions, &b_, launch_config)
            .EmitLoop(context.name, index_type));
  }

  b_.SetInsertPoint(b_.GetInsertBlock()->getTerminator());
  return Status::OK();
}

Status IrEmitterUnnested::EmitFusionFromMlir(MlirEmitterInput mlir_input) {
  auto fusion_op = mlir::cast<mlir::lmhlo::FusionOp>(mlir_input.op);

  // Infer the layout of fusion internal nodes.
  const FusionLayoutAnalysis layout_analysis(fusion_op);

  auto fusion_results = fusion_op.getFusionResults();
  TF_RET_CHECK(!fusion_results.empty());
  if (fusion_results.size() > 1) {
    // In the case of root tuple, it can be either reduce or slice input
    // fusion.
    if (IsInputFusibleSlices(mlir_input.op, /*verify_no_strides=*/true)) {
      // The emitter doesn't support all cases. If it's not supported, fallback
      // to ElementalIrEmitter.
      auto status = EmitInputFusibleNonStridedSlices(mlir_input);
      if (status.code() == tensorflow::error::FAILED_PRECONDITION) {
        return EmitLoopFusionFromMlir(mlir_input);
      }
      return status;
    }

    const bool is_parallel_reduce =
        absl::c_any_of(fusion_results, [&layout_analysis](mlir::Value result) {
          return IsReductionFromOrToContiguousDimensions(result.getDefiningOp(),
                                                         layout_analysis);
        });

    if (is_parallel_reduce) {
      return EmitReductionFromOrToContiguousDimensions(mlir_input,
                                                       layout_analysis);
    }
  }

  mlir::Operation* fusion_root = fusion_results[0].getDefiningOp();
  if (mlir::isa<mlir::mhlo::ScatterOp>(fusion_root)) {
    TF_ASSIGN_OR_RETURN(
        const HloComputation* fused_computation,
        GetOrCreateSubComputationFromRegion(&fusion_op.region(),
                                            /*is_fusion=*/true));
    auto* root = fused_computation->root_instruction();

    ThunkSequence thunks;
    // The initialization from 'operand' is using different loop bounds, so
    // emit it in a separate kernel. Treat it like a loop fusion, writing to
    // the output buffer.
    {
      std::vector<llvm_ir::IrArray> ir_arrays;
      TF_ASSIGN_OR_RETURN(auto operand_thunk,
                          BuildKernelThunkForMlir(
                              mlir_input.op, Thunk::ThunkInfo(), &ir_arrays));
      thunks.push_back(std::move(operand_thunk));

      GpuElementalIrEmitter operand_elemental_emitter(
          hlo_module_config_, ir_emitter_context_->llvm_module(), &b_,
          GetNestedComputer());
      FusedIrEmitter operand_fused_emitter(&operand_elemental_emitter);
      for (int i = 0; i < fused_computation->num_parameters(); i++) {
        auto fused_operand = fused_computation->parameter_instruction(i);
        operand_fused_emitter.BindGenerator(
            fused_operand, [this, &ir_arrays, i,
                            fused_operand](llvm_ir::IrArray::Index index) {
              return ir_arrays[i].EmitReadArrayElement(index, &b_,
                                                       fused_operand->name());
            });
      }
      TF_ASSIGN_OR_RETURN(auto generator,
                          operand_fused_emitter.GetGenerator(root->operand(0)));

      auto unroll_factor =
          ComputeMaxUnrollFactor(fusion_op, hlo_module_config_);
      const Shape& element_shape = root->shape();
      TF_ASSIGN_OR_RETURN(
          LaunchDimensions launch_dimensions,
          CalculateLaunchDimensions(element_shape,
                                    ir_emitter_context_->gpu_device_info(),
                                    {unroll_factor, /*few_waves=*/false}));
      UpdateLaunchDimensions(launch_dimensions, thunks.back().get(),
                             ir_emitter_context_->llvm_module());
      TF_RETURN_IF_ERROR(
          ParallelLoopEmitter(generator, ir_arrays.back(), launch_dimensions,
                              &b_, {unroll_factor})
              .EmitLoop(IrName(mlir::GetNameFromLoc(fusion_op.getLoc())),
                        GetIndexTypeForKernelFromMlir(
                            fusion_op, launch_dimensions.launch_bound(), &b_)));
    }

    // Now build the actual scatter, reading and writing to the freshly
    // filled output buffer.
    {
      std::vector<llvm_ir::IrArray> ir_arrays;
      TF_ASSIGN_OR_RETURN(auto scatter_thunk,
                          BuildKernelThunkForMlir(
                              mlir_input.op, Thunk::ThunkInfo(), &ir_arrays));
      thunks.push_back(std::move(scatter_thunk));
      // Spin up a new fused emitter for the scatter kernel and emit it.
      GpuElementalIrEmitter scatter_elemental_emitter(
          hlo_module_config_, ir_emitter_context_->llvm_module(), &b_,
          GetNestedComputer());
      FusedIrEmitter scatter_fused_emitter(&scatter_elemental_emitter);
      for (int i = 0; i < fused_computation->num_parameters(); i++) {
        auto fused_operand = fused_computation->parameter_instruction(i);
        scatter_fused_emitter.BindGenerator(
            fused_operand, [this, &ir_arrays, i,
                            fused_operand](llvm_ir::IrArray::Index index) {
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
      desc.updates_shape = root->operand(2)->shape();
      desc.dim_numbers = dim_numbers;
      desc.unique_indices = root->unique_indices();
      desc.update_computation = root->called_computations()[0];
      desc.output = ir_arrays.back();
      TF_ASSIGN_OR_RETURN(desc.scatter_indices_gen,
                          scatter_fused_emitter.GetGenerator(root->operand(1)));
      TF_ASSIGN_OR_RETURN(desc.updates_gen,
                          scatter_fused_emitter.GetGenerator(root->operand(2)));
      desc.get_index_type = [&](int64 launch_size) {
        return GetIndexTypeForKernel(root, launch_size, &b_);
      };

      TF_RETURN_IF_ERROR(EmitScatter(desc, thunks.back().get()));
    }
    AddThunkToThunkSequence(absl::make_unique<SequentialThunk>(
        mlir_input.thunk_info, std::move(thunks)));
    return Status::OK();
  }

  if (mlir::isa<mlir::mhlo::ReduceOp>(fusion_root) &&
      IsReductionFromOrToContiguousDimensions(fusion_root, layout_analysis)) {
    // HandleFusion specializes reduction from a multi-dimensional array to
    // a 1D array. The specialized version requires a initializer thunk that
    // initializes the output array to the initial value of the reduce.
    if (mlir_input.op->getNumResults() > 1) {
      // TODO(b/129089333): Support tiled vectorized variadic reduce.
      return Unimplemented(
          "Vectorized variadic reduce is not supported on GPU");
    }
    return EmitReductionFromOrToContiguousDimensions(mlir_input,
                                                     layout_analysis);
  }

  if (CanEmitFusedDynamicUpdateSliceInPlaceForGpu(
          fusion_op, ir_emitter_context_->allocations())) {
    // Fusion node with dynamic-update-slice as the root where the op's input
    // (i.e. array to update) shares the same slice as its output.  In this case
    // we have a special algorithm that modifies the output in place without
    // touching the un-updated elements.
    CHECK_EQ(1, GetHloOutputs(mlir_input.op).size());

    // Set up kernel thunk and fused ir emitter.
    std::vector<llvm_ir::IrArray> ir_arrays;
    TF_ASSIGN_OR_RETURN(
        auto fusion_thunk,
        BuildKernelThunkForMlir(fusion_op, mlir_input.thunk_info, &ir_arrays));

    TF_ASSIGN_OR_RETURN(
        const HloComputation* fused_computation,
        GetOrCreateSubComputationFromRegion(&fusion_op.region(),
                                            /*is_fusion=*/true));

    GpuElementalIrEmitter elemental_emitter(hlo_module_config_,
                                            ir_emitter_context_->llvm_module(),
                                            &b_, GetNestedComputer());

    // Shape of the dynamic-update-slice's "update" operand.
    Shape update_shape =
        fused_computation->root_instruction()->operand(1)->shape();

    // Array to write into.  Because this is an in-place operation, this is the
    // same as operand 0's array.
    const IrArray& output_array = ir_arrays.back();

    TF_ASSIGN_OR_RETURN(
        LaunchDimensions launch_dimensions,
        CalculateLaunchDimensions(update_shape,
                                  ir_emitter_context_->gpu_device_info()));
    UpdateLaunchDimensions(launch_dimensions, fusion_thunk.get(),
                           ir_emitter_context_->llvm_module());
    AddThunkToThunkSequence(std::move(fusion_thunk));

    FusedIrEmitter fused_emitter(&elemental_emitter);

    for (int i = 0; i < fused_computation->num_parameters(); i++) {
      auto fused_operand = fused_computation->parameter_instruction(i);
      fused_emitter.BindGenerator(
          fused_operand, [this, &ir_arrays, i,
                          fused_operand](const llvm_ir::IrArray::Index& index) {
            return ir_arrays[i].EmitReadArrayElement(index, &b_,
                                                     fused_operand->name());
          });
    }

    return llvm_ir::EmitParallelFusedDynamicUpdateSliceInPlace(
        fused_computation, output_array, &fused_emitter, launch_dimensions,
        &b_);
  }

  TF_ASSIGN_OR_RETURN(const bool matched_021,
                      CheckAndEmitHloWithTile021(mlir_input));
  if (matched_021) {
    return Status::OK();
  }

  return EmitLoopFusionFromMlir(mlir_input);
}

Status IrEmitterUnnested::EmitCopyFromMlir(MlirEmitterInput input) {
  auto copy = mlir::cast<mlir::lmhlo::CopyOp>(input.op);
  auto operand_shape = TypeToShape(copy.operand().getType());
  auto output_shape = TypeToShape(copy.output().getType());

  CHECK(ShapeUtil::Compatible(operand_shape, output_shape));
  auto maybe_slice = GetAllocationSliceForMlir(copy.operand());
  if (LayoutUtil::Equal(operand_shape.layout(), output_shape.layout()) &&
      maybe_slice.ok()) {
    // Copy the operand into the output if it's not the same buffer already.
    auto operand_buffer = *maybe_slice;
    auto destination_buffer = *GetAllocationSliceForMlir(copy.output());
    if (operand_buffer != destination_buffer) {
      AddThunkToThunkSequence(absl::make_unique<DeviceToDeviceCopyThunk>(
          input.thunk_info,
          /*source_address=*/operand_buffer,
          /*destination_buffer=*/destination_buffer,
          /*mem_size=*/
          ByteSizeOf(operand_shape)));
    }
    return Status::OK();
  }
  TF_ASSIGN_OR_RETURN(bool matched_021, CheckAndEmitHloWithTile021(input));
  if (matched_021) {
    return Status::OK();
  }

  return EmitUsingElementalIrEmitter(input);
}

Status IrEmitterUnnested::EmitExtraOutputsForReduce(
    absl::Span<const llvm_ir::IrArray> result_ir_arrays,
    const IrArray::Index& index, bool use_linear_index,
    absl::Span<const std::pair<llvm_ir::ElementGenerator, int>>
        extra_output_gens) {
  // Compute all extra output values before writing them. This avoids
  // overwriting aliased input/output buffers before all reads occured.
  absl::InlinedVector<llvm::Value*, 8> extra_output_ir_values;
  for (int i = 0; i < extra_output_gens.size(); ++i) {
    TF_ASSIGN_OR_RETURN(llvm::Value* const extra_output_ir_value,
                        extra_output_gens[i].first(index));
    extra_output_ir_values.push_back(extra_output_ir_value);
  }
  for (int i = 0; i < extra_output_gens.size(); ++i) {
    result_ir_arrays[extra_output_gens[i].second].EmitWriteArrayElement(
        index, extra_output_ir_values[i], &b_, use_linear_index);
  }
  return Status::OK();
}

Status IrEmitterUnnested::EmitReduceFromMlir(MlirEmitterInput mlir_input) {
  const FusionLayoutAnalysis dummy_analysis;
  if (GetHloOutputs(mlir_input.op).size() == 1 &&
      IsReductionFromOrToContiguousDimensions(mlir_input.op, dummy_analysis)) {
    return EmitReductionFromOrToContiguousDimensions(mlir_input,
                                                     dummy_analysis);
  }

  return EmitUsingElementalIrEmitter(mlir_input);
}

Status IrEmitterUnnested::AssertNonDeterminismIsOkay(const string& op_name) {
  if (hlo_module_config_.debug_options().xla_gpu_deterministic_ops()) {
    return Unimplemented(
        "HLO instruction %s does not have a deterministic implementation, "
        "but run-to-run determinism is required by "
        "--xla_gpu_deterministic_ops.",
        op_name);
  }
  return Status::OK();
}

Status IrEmitterUnnested::EmitSelectAndScatterFromMlir(
    MlirEmitterInput mlir_input) {
  auto select_and_scatter_op =
      mlir::cast<mlir::lmhlo::SelectAndScatterOp>(mlir_input.op);

  const Shape source_shape =
      TypeToShape(select_and_scatter_op.source().getType());
  const Shape operand_shape =
      TypeToShape(select_and_scatter_op.operand().getType());
  const int64 rank = operand_shape.rank();

  CHECK_EQ(rank, source_shape.rank());
  if (select_and_scatter_op.window_dimensions()) {
    CHECK_EQ(rank, select_and_scatter_op.window_dimensions()->size());
  }

  TF_RETURN_IF_ERROR(AssertNonDeterminismIsOkay(
      mlir::GetNameFromLoc(select_and_scatter_op.getLoc())));

  std::string name = mlir::GetNameFromLoc(select_and_scatter_op.getLoc());

  ThunkSequence thunks;
  thunks.emplace_back();
  TF_ASSIGN_OR_RETURN(thunks.back(),
                      BuildInitializerThunkForMlir(
                          mlir_input.op, select_and_scatter_op.init_value(),
                          select_and_scatter_op.out()));

  std::vector<llvm_ir::IrArray> ir_arrays;
  thunks.emplace_back();
  // Init value is not needed in IR emission.
  TF_ASSIGN_OR_RETURN(thunks.back(),
                      BuildKernelThunkForMlir(select_and_scatter_op,
                                              {select_and_scatter_op.operand(),
                                               select_and_scatter_op.source(),
                                               select_and_scatter_op.out()},
                                              Thunk::ThunkInfo(), &ir_arrays));

  CHECK_EQ(ir_arrays.size(), 3);
  const IrArray& operand_array = ir_arrays[0];
  const IrArray& source_array = ir_arrays[1];
  const IrArray& out_array = ir_arrays[2];

  auto select_and_scatter_thunk = absl::make_unique<SequentialThunk>(
      mlir_input.thunk_info, std::move(thunks));

  TF_ASSIGN_OR_RETURN(
      LaunchDimensions launch_dimensions,
      CalculateLaunchDimensions(source_shape,
                                ir_emitter_context_->gpu_device_info()));

  llvm::Type* index_type = GetIndexTypeForKernelFromMlir(
      select_and_scatter_op, launch_dimensions.launch_bound(), &b_);
  auto index_typed_constant = [&](uint64 c) -> llvm::Constant* {
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
  //   output(selected_index) = scatter(output(selected_index), source(S))
  auto loop_body_emitter = [&](const IrArray::Index& source_index) -> Status {
    // Allocate space to keep the currently selected value, its index, and a
    // boolean flag if the value is initialized. The initialized_flag is set
    // false.
    llvm::Value* selected_value_address = llvm_ir::EmitAllocaAtFunctionEntry(
        llvm_ir::PrimitiveTypeToIrType(operand_shape.element_type(),
                                       ir_emitter_context_->llvm_module()),
        "selected_value_address", &b_);

    llvm::Value* selected_index_address =
        llvm_ir::EmitAllocaAtFunctionEntryWithCount(
            index_type, index_typed_constant(rank), "selected_index_address",
            &b_);

    llvm::Value* initialized_flag_address = llvm_ir::EmitAllocaAtFunctionEntry(
        b_.getInt1Ty(), "initialized_flag_address", &b_);
    Store(b_.getInt1(false), initialized_flag_address);

    // Create the inner loop to iterate over the window.
    llvm_ir::ForLoopNest window_loops(absl::StrCat(name, "inner"), &b_,
                                      index_type);

    DimensionVector window_size;
    mlir::DenseIntElementsAttr window_dimensions =
        select_and_scatter_op.window_dimensions().getValue();
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

    auto strides = *select_and_scatter_op.window_strides();
    auto paddings = *select_and_scatter_op.padding();

    for (auto stride_and_padding :
         llvm::enumerate(llvm::zip(strides, paddings))) {
      const int i = stride_and_padding.index();
      int64 stride = std::get<0>(stride_and_padding.value()).getSExtValue();
      int64 padding = std::get<1>(stride_and_padding.value()).getSExtValue();

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
        Load(initialized_flag_address), "initialized", &b_);

    // If the initialized_flag is false, initialize the selected value and index
    // with the currently visiting operand.
    llvm_ir::SetToFirstInsertPoint(if_initialized.false_block, &b_);
    const auto save_operand_index = [&](const IrArray::Index& operand_index) {
      for (int64 i = 0; i < rank; ++i) {
        llvm::Value* selected_index_address_slot =
            InBoundsGEP(selected_index_address, {b_.getInt32(i)});
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
    llvm::Value* select_return_buffer = llvm_ir::EmitAllocaAtFunctionEntry(
        llvm_ir::PrimitiveTypeToIrType(PRED,
                                       ir_emitter_context_->llvm_module()),
        "select_return_buffer", &b_);

    TF_ASSIGN_OR_RETURN(
        const HloComputation* select_computation,
        GetOrCreateSubComputationFromRegion(&select_and_scatter_op.select(),
                                            /*is_fusion=*/false));

    TF_RETURN_IF_ERROR(EmitCallToNestedComputation(
        *select_computation, {selected_value_address, operand_address},
        select_return_buffer));
    llvm::Value* result = Load(select_return_buffer);

    // If the 'select' function returns false, update the selected value and the
    // index to the currently visiting operand.
    llvm::Value* cond = ICmpNE(
        result,
        llvm::ConstantInt::get(llvm_ir::PrimitiveTypeToIrType(
                                   PRED, ir_emitter_context_->llvm_module()),
                               0),
        "boolean_predicate");
    llvm_ir::LlvmIfData if_select_lhs =
        llvm_ir::EmitIfThenElse(cond, "if-select-lhs", &b_);
    llvm_ir::SetToFirstInsertPoint(if_select_lhs.false_block, &b_);
    Store(Load(operand_address), selected_value_address);
    save_operand_index(operand_index);

    // After iterating over the window elements, scatter the source element to
    // the selected index of the output. The value we store at the output
    // location is computed by calling the `scatter` function with the source
    // value and the current output value.
    llvm_ir::SetToFirstInsertPoint(window_loops.GetOuterLoopExitBasicBlock(),
                                   &b_);
    std::vector<llvm::Value*> selected_multi_index;
    for (int64 i = 0; i < rank; ++i) {
      llvm::Value* selected_index_address_slot =
          InBoundsGEP(selected_index_address, {b_.getInt32(i)});
      selected_multi_index.push_back(Load(selected_index_address_slot));
    }
    const Shape output_shape =
        TypeToShape(select_and_scatter_op.out().getType());
    llvm::Value* source_value_address =
        source_array.EmitArrayElementAddress(source_index, &b_);
    IrArray::Index selected_index(selected_multi_index, output_shape,
                                  operand_index.GetType());
    llvm::Value* output_value_address =
        out_array.EmitArrayElementAddress(selected_index, &b_);

    TF_ASSIGN_OR_RETURN(
        const HloComputation* scatter_computation,
        GetOrCreateSubComputationFromRegion(&select_and_scatter_op.scatter(),
                                            /*is_fusion=*/false));

    return EmitAtomicOperationForNestedComputation(
        *scatter_computation, output_value_address, source_value_address);
  };

  UpdateLaunchDimensions(
      launch_dimensions,
      // IrEmitterUnnested implements kSelectAndScatter as a SequentialThunk
      // consisting of two thunks, an initializer KernelThunk that initializes
      // the output and another KernelThunk that accumulates the scattered
      // elements.
      select_and_scatter_thunk->thunks().back().get(),
      ir_emitter_context_->llvm_module());
  AddThunkToThunkSequence(std::move(select_and_scatter_thunk));
  return ParallelLoopEmitter(loop_body_emitter, source_shape, launch_dimensions,
                             &b_)
      .EmitLoop(name, index_type);
}

Status IrEmitterUnnested::EmitWhileFromMlir(MlirEmitterInput mlir_input) {
  auto while_op = mlir::cast<mlir::lmhlo::WhileOp>(mlir_input.op);

  auto cond_result = GetHloOutputs(while_op);
  TF_RET_CHECK(cond_result.size() == 1);
  TF_RET_CHECK(cond_result[0]
                   .getType()
                   .cast<mlir::ShapedType>()
                   .getElementType()
                   .isInteger(/*width=*/1))
      << "While condition computation must return bool";

  //  Build ForThunk for conformant while loops, otherwise build WhileThunk.
  if (while_op.trip_count()) {
    TF_ASSIGN_OR_RETURN(
        auto thunk,
        BuildForThunk(while_op, mlir_input.thunk_info, *while_op.trip_count()));
    AddThunkToThunkSequence(std::move(thunk));
  } else {
    TF_ASSIGN_OR_RETURN(auto thunk,
                        BuildWhileThunk(while_op, mlir_input.thunk_info));
    AddThunkToThunkSequence(std::move(thunk));
  }
  return Status::OK();
}

Status IrEmitterUnnested::EmitRngGetAndUpdateState(
    MlirEmitterInput mlir_input) {
  auto rng_op =
      mlir::dyn_cast<mlir::lmhlo::RngGetAndUpdateStateOp>(mlir_input.op);

  // Emit a kernel to increment the global state for Philox RNG algorithm.
  std::vector<llvm_ir::IrArray> ir_arrays;
  TF_ASSIGN_OR_RETURN(
      auto kernel_thunk,
      BuildKernelThunkForMlir(rng_op, rng_op.state(), mlir_input.thunk_info,
                              &ir_arrays));
  AddThunkToThunkSequence(std::move(kernel_thunk));

  llvm::Value* old_state =
      llvm_ir::RngGetAndUpdateState(rng_op.delta(), module_, &b_);

  const Shape shape = TypeToShape(rng_op.state().getType());

  llvm::Value* output_address = ir_arrays[0].EmitArrayElementAddress(
      llvm_ir::IrArray::Index(
          /*linear=*/b_.getInt64(0), shape, &b_),
      &b_, "rng_state_address");
  output_address = BitCast(
      output_address, llvm::PointerType::get(
                          old_state->getType(),
                          output_address->getType()->getPointerAddressSpace()));
  Store(old_state, output_address);

  return Status::OK();
}

Status IrEmitterUnnested::EmitScatterFromMlir(MlirEmitterInput mlir_input) {
  ThunkSequence thunks;

  auto scatter_op = mlir::cast<mlir::lmhlo::ScatterOp>(mlir_input.op);

  if (!scatter_op.unique_indices()) {
    TF_RETURN_IF_ERROR(
        AssertNonDeterminismIsOkay(mlir::GetNameFromLoc(scatter_op.getLoc())));
  }

  TF_ASSIGN_OR_RETURN(auto operand_buffer,
                      GetAllocationSliceForMlir(scatter_op.operand()));
  TF_ASSIGN_OR_RETURN(auto output_buffer,
                      GetAllocationSliceForMlir(scatter_op.output()));

  // Copy the operand into the output if it's not the same buffer already.
  if (operand_buffer != output_buffer) {
    thunks.push_back(absl::make_unique<DeviceToDeviceCopyThunk>(
        Thunk::ThunkInfo(),
        /*source_address=*/operand_buffer,
        /*destination_buffer=*/output_buffer,
        /*mem_size=*/
        ShapeUtil::ByteSizeOf(TypeToShape(scatter_op.output().getType()))));
  }

  // Create kernel thunk for all operands except the first one (`operand`). The
  // code generated for scatter below assumes that the input operand is already
  // copied into the output, so does not use it in codegen.
  std::vector<llvm_ir::IrArray> ir_arrays;
  thunks.emplace_back();
  TF_ASSIGN_OR_RETURN(
      thunks.back(),
      BuildKernelThunkForMlir(scatter_op, scatter_op.getOperands().drop_front(),
                              mlir_input.thunk_info, &ir_arrays));

  CHECK_EQ(ir_arrays.size(), 3);
  const IrArray& scatter_indices = ir_arrays[0];
  const IrArray& updates = ir_arrays[1];
  const IrArray& output = ir_arrays[2];

  auto get_index_type = [&](int64 launch_size) {
    return GetIndexTypeForKernelFromMlir(scatter_op, launch_size, &b_);
  };

  TF_RETURN_IF_ERROR(EmitScatter(
      thunks.back().get(), scatter_op, output,
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

  // Elide the sequential thunk if there's no copy.
  if (thunks.size() == 1) {
    AddThunkToThunkSequence(std::move(thunks[0]));
  } else {
    AddThunkToThunkSequence(absl::make_unique<SequentialThunk>(
        mlir_input.thunk_info, std::move(thunks)));
  }

  return Status::OK();
}

Status IrEmitterUnnested::EmitScatter(
    Thunk* thunk, mlir::lmhlo::ScatterOp scatter,
    const llvm_ir::IrArray& output,
    const llvm_ir::ElementGenerator& scatter_indices_gen,
    const llvm_ir::ElementGenerator& updates_gen,
    std::function<llvm::Type*(int64)> get_index_type) {
  const Shape operand_shape = TypeToShape(scatter.operand().getType());
  CHECK(
      ShapeUtil::Equal(TypeToShape(scatter.output().getType()), operand_shape));

  TF_ASSIGN_OR_RETURN(
      const HloComputation* update_computation,
      GetOrCreateSubComputationFromRegion(&scatter.update_computation(),
                                          /*is_fusion=*/false));

  ScatterDescriptor desc;
  desc.name = mlir::GetNameFromLoc(scatter.getLoc());
  desc.operand_shape = operand_shape;
  desc.scatter_indices_shape = TypeToShape(scatter.scatter_indices().getType());
  desc.updates_shape = TypeToShape(scatter.updates().getType());
  desc.dim_numbers = scatter.scatter_dimension_numbers();
  desc.unique_indices = scatter.unique_indices();
  desc.update_computation = update_computation;
  desc.output = output;
  desc.scatter_indices_gen = scatter_indices_gen;
  desc.updates_gen = updates_gen;
  desc.get_index_type = get_index_type;
  return EmitScatter(desc, thunk);
}

Status IrEmitterUnnested::EmitScatter(const ScatterDescriptor& desc,
                                      Thunk* thunk) {
  if (!desc.unique_indices) {
    TF_RETURN_IF_ERROR(AssertNonDeterminismIsOkay(desc.name));
  }
  auto loop_body_emitter = [&](const IrArray::Index& index) -> Status {
    std::vector<llvm::Value*> raw_window_multidim;
    std::vector<llvm::Value*> input_scatter_multidim;
    std::vector<int64> raw_window_bounds;

    // Partition the index into window indices and scatter indices.
    for (int64 i = 0, e = index.size(); i != e; ++i) {
      // For window indices also remember the window size, this comes in handy
      // later.
      if (BinarySearchDenseElementsAttr(desc.dim_numbers.update_window_dims(),
                                        i)) {
        raw_window_multidim.push_back(index[i]);
        raw_window_bounds.push_back(desc.updates_shape.dimensions(i));
      } else {
        input_scatter_multidim.push_back(index[i]);
      }
    }
    DCHECK_EQ(raw_window_multidim.size(),
              desc.dim_numbers.update_window_dims().size());

    // Apply inserted_window_dims to the window dimensions.
    int64 raw_window_multidim_idx = 0;
    std::vector<llvm::Value*> input_window_multidim;
    std::vector<int64> input_window_bounds;

    for (int64 i = 0, e = desc.operand_shape.rank(); i != e; ++i) {
      if (BinarySearchDenseElementsAttr(desc.dim_numbers.inserted_window_dims(),
                                        i)) {
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
    if (desc.dim_numbers.index_vector_dim().getInt() ==
        desc.scatter_indices_shape.rank()) {
      scatter_indices_shape_fixed.add_dimensions(1);
      scatter_indices_shape_fixed.mutable_layout()->add_minor_to_major(
          desc.dim_numbers.index_vector_dim().getInt());
    }

    // Now load the indices corresponding to the current window from
    // scatter_indices.
    std::vector<llvm::Value*> raw_scatter_index_multidim =
        input_scatter_multidim;
    raw_scatter_index_multidim.insert(
        raw_scatter_index_multidim.begin() +
            desc.dim_numbers.index_vector_dim().getInt(),
        nullptr);
    llvm::Value* is_in_bounds = b_.getTrue();
    for (int64 i = 0,
               e = desc.dim_numbers.scatter_dims_to_operand_dims().size();
         i != e; ++i) {
      // Our index is stored along index_vector_dim, insert that into the lookup
      // index into scatter_indices.
      raw_scatter_index_multidim[desc.dim_numbers.index_vector_dim().getInt()] =
          index.GetConstantWithIndexType(i);
      llvm_ir::IrArray::Index raw_scatter_index_index(
          raw_scatter_index_multidim, scatter_indices_shape_fixed,
          index.GetType());

      int64 operand_dim =
          desc.dim_numbers.scatter_dims_to_operand_dims().getValue<int64>(i);
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
      int64 max_index = desc.operand_shape.dimensions(operand_dim) -
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
          *desc.update_computation, output_address, input_address);
    } else {
      return EmitCallToNestedComputation(*desc.update_computation,
                                         {output_address, input_address},
                                         output_address);
    }
  };

  // Launch a kernel that reads every element in the updates tensor. We could
  // also do one kernel per window instead if bounds checks turn out to be a
  // bottleneck.
  TF_ASSIGN_OR_RETURN(
      LaunchDimensions launch_dimensions,
      CalculateLaunchDimensions(desc.updates_shape,
                                ir_emitter_context_->gpu_device_info()));
  UpdateLaunchDimensions(launch_dimensions, thunk,
                         ir_emitter_context_->llvm_module());

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
      region = &mlir::cast<mlir::lmhlo::FusionOp>(clone).region();
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
        if (instr->shape().IsTuple() &&
            computation == module->entry_computation() &&
            instr != computation->root_instruction()) {
          return InternalError("Non-root tuple types are not handled.");
        }
      }
    }
  }
  return module->entry_computation();
}

Status IrEmitterUnnested::EmitSortFromMlir(MlirEmitterInput mlir_input) {
  auto sort_op = mlir::cast<mlir::lmhlo::SortOp>(mlir_input.op);
  MlirEmitterContext context;
  context.SetOperation(sort_op);

  ThunkSequence thunks;

  const Shape& keys_shape = context.operand_shapes[0];
  int64 dimension_to_sort = sort_op.dimension();
  for (int64 i = 0; i < context.operand_shapes.size(); ++i) {
    // We assume that the layout of all involved operands and outputs is the
    // same.
    TF_RET_CHECK(LayoutUtil::LayoutsInShapesEqual(keys_shape,
                                                  context.operand_shapes[i]));
    TF_RET_CHECK(
        LayoutUtil::LayoutsInShapesEqual(keys_shape, context.output_shapes[i]));

    // If possible, we share buffers. If that is not possible, we need to copy
    // the values, because the emitter does the sorting in-place.
    TF_ASSIGN_OR_RETURN(auto destination_buffer,
                        GetAllocationSliceForMlir(sort_op.output()[i]));
    TF_ASSIGN_OR_RETURN(auto source_address,
                        GetAllocationSliceForMlir(sort_op.operands()[i]));
    if (destination_buffer != source_address) {
      // TODO(b/26783907): Figure out why we never seem to share buffers for
      // key/value sort.
      VLOG(2) << context.name << " requires initial D2D copy for operand " << i;
      thunks.push_back(absl::make_unique<DeviceToDeviceCopyThunk>(
          Thunk::ThunkInfo(),
          /*source_address=*/source_address,
          /*destination_buffer=*/destination_buffer,
          /*mem_size=*/ShapeUtil::ByteSizeOf(context.operand_shapes[i])));
    }
  }

  uint64 dimension_to_sort_bound = keys_shape.dimensions(dimension_to_sort);
  int64 num_stages = tensorflow::Log2Ceiling(dimension_to_sort_bound);
  VLOG(2) << context.name << " requires " << num_stages << " stages.";
  CHECK_GE(1ULL << num_stages, dimension_to_sort_bound);
  CHECK_LT(1ULL << (num_stages - 1), dimension_to_sort_bound);

  // Naive C++ code for the outer loops:
  //
  // for (int64 stage = 0; stage < Log2Ceiling(dimension_to_sort_bound);
  //     ++stage) {
  //   int64 first_xor_mask = (1LL << (stage + 1)) - 1;
  //   SortInPlace(first_xor_mask);
  //   for (int64 mask = stage - 1; mask >= 0; --mask) {
  //     int64 later_xor_mask = 1LL << mask;
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

  const uint64 kTileSize = std::min(2048ULL, 1ULL << num_stages);

  // If we cannot combine several xor masks together, we don't use tiling, so we
  // calculate the standard launch dimensions for the shape. However we only
  // need to iterate through ~half of the dimension to sort (rounded up to the
  // next highest power of 2), because each iteration compares one pair of
  // elements.
  Shape standard_iteration_shape = keys_shape;
  uint64 standard_num_iterations_in_sort_dim = 1ULL << (num_stages - 1);
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
  int64 rounded_bound = RoundUpToNearest(dimension_to_sort_bound, kTileSize);
  Shape iteration_shape = keys_shape;

  // We iterate through the element pairs that should be compared.
  uint64 num_iterations_in_sort_dim = rounded_bound / 2;
  iteration_shape.set_dimensions(dimension_to_sort, num_iterations_in_sort_dim);
  uint64 num_iterations = ShapeUtil::ElementsIn(iteration_shape);

  // For correctness reasons we need exactly 'kTileSize' / 2 many threads per
  // block. Each thread is responsible for copying exactly two adjacent elements
  // into shared memory, and then does a comparison of two possibly different
  // elements taken from shared memory.
  const uint64 kThreadsPerBlock = kTileSize / 2;

  // Check whether we should use any tiling. We might not be able to use it if
  // we have not enough threads, or not enough shared memory. Also it does not
  // give a speedup if the tile size is < 128.
  int64 total_shared_memory_needed = 0;
  for (int64 i = 0; i < context.operand_shapes.size(); ++i) {
    total_shared_memory_needed +=
        kTileSize * ShapeUtil::ByteSizeOfPrimitiveType(
                        context.operand_shapes[i].element_type());
  }
  bool no_tiling =
      kTileSize < 128 ||
      kThreadsPerBlock >
          ir_emitter_context_->gpu_device_info().threads_per_block_limit ||
      total_shared_memory_needed >
          ir_emitter_context_->gpu_device_info().shared_memory_per_block;
  VLOG(2) << absl::StreamFormat(
      "%s %s use tiling. No tiling if any of the following is true: "
      "kTileSize=%d < 128, "
      "kThreadsPerBlock=%d > threads_per_block_limit=%d, "
      "total_shared_memory_needed=%d > shared_memory_per_block=%d",
      context.name, (no_tiling ? "won't" : "will"), kTileSize, kThreadsPerBlock,
      ir_emitter_context_->gpu_device_info().threads_per_block_limit,
      total_shared_memory_needed,
      ir_emitter_context_->gpu_device_info().shared_memory_per_block);

  uint64 num_blocks = CeilOfRatio(num_iterations, kThreadsPerBlock);
  LaunchDimensions tiled_launch_dimensions(num_blocks, kThreadsPerBlock);
  VLOG(2) << absl::StreamFormat("%s launch dims: %d blocks, %d threads/block",
                                context.name, num_blocks, kThreadsPerBlock);

  std::vector<llvm_ir::IrArray> ir_arrays;
  auto emit_kernel = [&](absl::Span<const int64> xor_masks) {
    VLOG(2) << absl::StreamFormat(
        "%s uses kernel for xor masks [%s]", context.name,
        absl::StrJoin(xor_masks, ", ", [](std::string* out, int64 xor_mask) {
          absl::StrAppendFormat(out, "0x%x", xor_mask);
        }));
    thunks.emplace_back();
    TF_ASSIGN_OR_RETURN(
        thunks.back(), BuildKernelThunkForMlir(sort_op, sort_op.output(),
                                               Thunk::ThunkInfo(), &ir_arrays));
    LaunchDimensions launch_dimensions = xor_masks.size() > 1
                                             ? tiled_launch_dimensions
                                             : standard_launch_dimensions;
    UpdateLaunchDimensions(launch_dimensions, thunks.back().get(),
                           ir_emitter_context_->llvm_module());
    std::vector<IrArray> values_arrays;
    values_arrays.reserve(context.operand_shapes.size());
    for (int64 i = 0; i < context.operand_shapes.size(); ++i) {
      values_arrays.push_back(ir_arrays[i]);
    }
    TF_ASSIGN_OR_RETURN(const HloComputation* comparator,
                        GetOrCreateSubComputationFromRegion(
                            &sort_op.comparator(), /*is_fusion=*/false));
    return llvm_ir::EmitSortInPlace(
        dimension_to_sort, values_arrays, IrName(context.name), xor_masks, &b_,
        launch_dimensions,
        xor_masks.size() > 1 ? num_iterations_in_sort_dim
                             : standard_num_iterations_in_sort_dim,
        kTileSize,
        [&](absl::Span<llvm::Value* const> operands, llvm::Value* output) {
          return EmitCallToNestedComputation(*comparator, operands, output);
        });
  };
  std::vector<int64> xor_masks;
  for (int64 stage = 0; stage < num_stages; ++stage) {
    for (int64 mask = stage; mask >= 0; --mask) {
      int64 xor_mask;
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
  VLOG(2) << absl::StreamFormat(
      "%s requires %d thunks (including any D2D copies)", context.name,
      thunks.size());

  AddThunkToThunkSequence(absl::make_unique<SequentialThunk>(
      mlir_input.thunk_info, std::move(thunks)));
  return Status::OK();
}

template <typename ThunkType, typename OpT>
Status IrEmitterUnnested::EmitReplicaOrPartitionIdFromMlir(
    MlirEmitterInput input) {
  auto op = mlir::cast<OpT>(input.op);
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice result_slice,
                      GetAllocationSliceForMlir(op.getOperand()));
  AddThunkToThunkSequence(
      absl::make_unique<ThunkType>(input.thunk_info, result_slice));
  return Status::OK();
}

Status IrEmitterUnnested::EmitCollectivePermuteFromMlir(
    MlirEmitterInput input) {
  auto collective_permute_op =
      mlir::cast<mlir::lmhlo::CollectivePermuteOp>(input.op);

  TF_ASSIGN_OR_RETURN(
      BufferAllocation::Slice source_slice,
      GetAllocationSliceForMlir(collective_permute_op.operand()));
  TF_ASSIGN_OR_RETURN(
      BufferAllocation::Slice result_slice,
      GetAllocationSliceForMlir(collective_permute_op.output()));

  const Shape shape = TypeToShape(collective_permute_op.operand().getType());
  const int64 replica_count = hlo_module_config_.replica_count();
  const int64 partition_count = hlo_module_config_.num_partitions();

  if (NcclCollectivePermuteThunk::IsDegenerate(
          collective_permute_op, replica_count, partition_count)) {
    // For a degenerate collective permute, just generate a copy thunk.
    AddThunkToThunkSequence(absl::make_unique<DeviceToDeviceCopyThunk>(
        input.thunk_info,
        /*source_address=*/source_slice,
        /*destination_buffer=*/result_slice,
        /*mem_size=*/ShapeUtil::ByteSizeOf(shape)));
  } else if (!collective_permute_op.channel_id() && partition_count == 1) {
    // The non-NCCL based collective permute only works for a single partition
    // cross replica case.
    using source_dest_pairs_t = std::vector<std::pair<int64, int64>>;
    TF_ASSIGN_OR_RETURN(
        source_dest_pairs_t source_dest_pairs,
        ConvertNx2Attribute(collective_permute_op.source_target_pairs()));

    AddThunkToThunkSequence(absl::make_unique<CollectivePermuteThunk>(
        input.thunk_info, std::move(source_dest_pairs), source_slice,
        result_slice));
  } else {
    const NcclCollectivePermuteThunk::Buffer buffer = {
        /*element_count=*/ShapeUtil::ElementsIn(shape),
        /*source_buffer=*/source_slice,
        /*destination_buffer=*/result_slice};
    AddThunkToThunkSequence(absl::make_unique<NcclCollectivePermuteThunk>(
        input.thunk_info, collective_permute_op, replica_count, partition_count,
        buffer));
  }
  return Status::OK();
}

template <typename NcclThunkType, typename OpTy>
Status IrEmitterUnnested::EmitNcclThunkFromMlir(MlirEmitterInput input) {
  OpTy op = mlir::cast<OpTy>(input.op);
  int64 replica_count = hlo_module_config_.replica_count();
  int64 partition_count = hlo_module_config_.num_partitions();
  VLOG(2) << NcclThunkType::GetName() << "; replica count: " << replica_count
          << "; partition count: " << partition_count
          << "; operand count: " << op.operands().size()
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
  buffers.reserve(op.operands().size());
  for (auto it : llvm::zip(op.operands(), op.results())) {
    mlir::Value operand = std::get<0>(it);
    mlir::Value result = std::get<1>(it);
    const Shape shape = TypeToShape(operand.getType());
    TF_ASSIGN_OR_RETURN(auto source_slice, GetAllocationSliceForMlir(operand));
    TF_ASSIGN_OR_RETURN(auto dest_slice, GetAllocationSliceForMlir(result));
    buffers.push_back(NcclCollectiveThunk::Buffer{
        /*element_count=*/ShapeUtil::ElementsIn(shape),
        /*source_buffer=*/source_slice,
        /*destination_buffer=*/dest_slice});
  }

  if (should_use_nccl_thunk) {
    auto nccl_thunk =
        absl::make_unique<NcclThunkType>(input.thunk_info, op,
                                         /*buffers=*/std::move(buffers));
    AddThunkToThunkSequence(std::move(nccl_thunk));
    return Status::OK();
  }

  if (!is_degenerate) {
    CollectiveOpGroupMode group_mode = NcclThunkType::GetGroupMode(op);

    string message = absl::StrFormat(
        "Requested %s not implemented on GPU; replica_count: %d; "
        "partition_count: %d, group_mode: %s, operand_count: %d; NCCL support: "
        "%d",
        NcclThunkType::GetName(), replica_count, partition_count,
        CollectiveOpGroupModeToString(group_mode), op.operands().size(),
        NcclThunkType::NcclIsEnabled());
    if (!op.operands().empty()) {
      const Shape shape = TypeToShape(op.operands().front().getType());
      absl::StrAppendFormat(&message, "; first operand array element-type: %s",
                            PrimitiveType_Name(shape.element_type()));
    }
    return Unimplemented("%s", message);
  }

  // All-gather with one replica is simply the identity function. Buffer
  // assignment expects a copy, so that's what we do.
  ThunkSequence thunks;
  for (int64 i = 0; i < buffers.size(); i++) {
    const Shape shape = TypeToShape(op.operands()[i].getType());
    thunks.push_back(absl::make_unique<DeviceToDeviceCopyThunk>(
        buffers.size() == 1 ? input.thunk_info : Thunk::ThunkInfo(),
        /*source_address=*/buffers[i].source_buffer,
        /*destination_buffer=*/buffers[i].destination_buffer,
        /*mem_size=*/ShapeUtil::ByteSizeOf(shape)));
  }
  if (thunks.size() == 1) {
    AddThunkToThunkSequence(std::move(thunks[0]));
  } else {
    AddThunkToThunkSequence(absl::make_unique<SequentialThunk>(
        input.thunk_info, std::move(thunks)));
  }
  return Status::OK();
}

Status IrEmitterUnnested::EmitInfeedFromMlir(MlirEmitterInput input) {
  auto infeed_op = mlir::cast<mlir::lmhlo::InfeedOp>(input.op);

  std::vector<ShapedSlice> dest_slices;
  dest_slices.reserve(infeed_op.outputs().size());

  for (mlir::Value output : infeed_op.outputs()) {
    TF_ASSIGN_OR_RETURN(auto slice, GetAllocationSliceForMlir(output));
    const Shape& shape = TypeToShape(output.getType());
    dest_slices.push_back(ShapedSlice{slice, shape});
  }

  AddThunkToThunkSequence(
      absl::make_unique<InfeedThunk>(input.thunk_info, std::move(dest_slices)));
  return Status::OK();
}

Status IrEmitterUnnested::EmitOutfeedFromMlir(MlirEmitterInput input) {
  auto outfeed_op = mlir::cast<mlir::lmhlo::OutfeedOp>(input.op);

  std::vector<ShapedSlice> source_slices;
  source_slices.reserve(outfeed_op.operands().size());

  for (mlir::Value operand : outfeed_op.operands()) {
    TF_ASSIGN_OR_RETURN(auto slice, GetAllocationSliceForMlir(operand));
    const Shape& shape = TypeToShape(operand.getType());
    source_slices.push_back(ShapedSlice{slice, shape});
  }

  AddThunkToThunkSequence(absl::make_unique<OutfeedThunk>(
      input.thunk_info, std::move(source_slices)));
  return Status::OK();
}

std::unique_ptr<KernelThunk> IrEmitterUnnested::BuildKernelThunkForMlirImpl(
    absl::string_view name, Thunk::ThunkInfo thunk_info,
    absl::Span<const BufferSlice> slices,
    std::vector<llvm_ir::IrArray>* ir_arrays) {
  // Figure out which buffer allocations need to be passed as arguments to our
  // kernel.  This is simply all of the allocations referenced in slices,
  // plus the XLA temp buffer (if we have it).  We always include the temp
  // buffer because even if the kernel itself doesn't use it, a nested
  // subcomputation within the kernel (e.g. a kMap's computation) might.
  std::unordered_set<const BufferAllocation*> buffers_needed;
  for (const auto& slice : slices) {
    buffers_needed.insert(slice.buffer_slice.allocation());
  }
  absl::optional<const BufferAllocation*> temp_buffer;
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
  std::vector<const BufferAllocation*> non_constant_buffers;
  absl::c_copy_if(buffers_needed, std::back_inserter(non_constant_buffers),
                  [](const BufferAllocation* allocation) {
                    return !allocation->is_constant();
                  });

  absl::c_sort(non_constant_buffers,
               [](const BufferAllocation* a, const BufferAllocation* b) {
                 return a->index() < b->index();
               });

  llvm::Function* kernel = BuildKernelPrototype(name, non_constant_buffers);

  // Build a map from a BufferAllocation to the corresponding argument in our
  // kernel.
  std::unordered_map<const BufferAllocation*, llvm::Value*> kernel_args;
  {
    auto arg_it = kernel->arg_begin();
    auto buffers_it = non_constant_buffers.begin();
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

  absl::flat_hash_set<BufferAllocation::Slice> buffers_written;
  for (const auto& slice : slices) {
    if (slice.written) {
      buffers_written.insert(slice.buffer_slice);
    }
  }

  ir_arrays->clear();

  // For each buffer our kernel might want to touch, bind it to a value derived
  // from our kernel args.
  for (const auto& slice : slices) {
    const BufferAllocation::Slice& buffer_slice = slice.buffer_slice;

    llvm::Value* loc;
    if (!slice.constant_name.empty()) {
      loc = ir_emitter_context_->llvm_module()->getGlobalVariable(
          slice.constant_name);
      CHECK_NE(loc, nullptr);
    } else {
      CHECK(!buffer_slice.allocation()->is_constant());
      loc = InBoundsGEP(kernel_args.at(buffer_slice.allocation()),
                        {b_.getInt64(buffer_slice.offset())});
    }

    llvm_ir::IrArray ir_array(CastToTypedValue(slice.shape, loc, &b_),
                              slice.shape);
    if (!buffers_written.contains(slice.buffer_slice)) {
      ir_array.MarkInvariantOverWholeProgram(&loc->getContext());
    }

    ir_arrays->push_back(ir_array);
  }

  return absl::make_unique<KernelThunk>(thunk_info, non_constant_buffers,
                                        std::string(kernel->getName()));
}

StatusOr<std::unique_ptr<KernelThunk>>
IrEmitterUnnested::BuildKernelThunkForMlir(
    mlir::Operation* op, mlir::ValueRange operands, Thunk::ThunkInfo thunk_info,
    std::vector<llvm_ir::IrArray>* ir_arrays) {
  TF_RET_CHECK(!mlir::isa<mlir::lmhlo::FusionOp>(op));

  std::vector<BufferSlice> slices;
  for (mlir::Value operand : operands) {
    slices.emplace_back();
    auto& slice = slices.back();
    TF_ASSIGN_OR_RETURN(slice.buffer_slice, GetAllocationSliceForMlir(
                                                operand, &slice.constant_name));
    slice.written = WritesMlirBuffer(op, operand);
    slice.shape = TypeToShape(operand.getType());
  }
  std::string name = mlir::GetNameFromLoc(op->getLoc());
  return BuildKernelThunkForMlirImpl(name, thunk_info, slices, ir_arrays);
}

StatusOr<std::unique_ptr<KernelThunk>>
IrEmitterUnnested::BuildKernelThunkForMlir(
    mlir::Operation* op, Thunk::ThunkInfo thunk_info,
    std::vector<llvm_ir::IrArray>* ir_arrays) {
  if (auto fusion = mlir::dyn_cast<mlir::lmhlo::FusionOp>(op)) {
    auto operands = GetHloOperands(op);
    auto outputs = GetHloOutputs(op);

    std::vector<BufferSlice> slices;
    for (auto operand : operands) {
      slices.emplace_back();
      auto& slice = slices.back();
      TF_ASSIGN_OR_RETURN(
          slice.buffer_slice,
          GetAllocationSliceForMlir(operand, &slice.constant_name));
      slice.written = false;
      slice.shape = TypeToShape(operand.getType());
    }
    for (auto output : outputs) {
      slices.emplace_back();
      auto& slice = slices.back();
      TF_ASSIGN_OR_RETURN(
          slice.buffer_slice,
          GetAllocationSliceForMlir(output, &slice.constant_name));
      slice.written = true;
      slice.shape = TypeToShape(output.getType());
    }
    std::string name = mlir::GetNameFromLoc(op->getLoc());
    return BuildKernelThunkForMlirImpl(name, thunk_info, slices, ir_arrays);
  }
  return BuildKernelThunkForMlir(op, op->getOperands(), thunk_info, ir_arrays);
}

std::unique_ptr<Thunk> IrEmitterUnnested::BuildConstantInitializerThunk(
    absl::Span<const uint8> init_value, const BufferAllocation::Slice& dest,
    const Shape& output_shape) {
  int64 num_bytes = init_value.size();
  if (absl::c_all_of(init_value, [](uint8 byte) { return byte == 0; })) {
    return absl::make_unique<MemzeroThunk>(Thunk::ThunkInfo(), dest);
  }

  // If the literal is 8 or 16 bits wide, we can emit a 32-bit memset by
  // repeating the literal 4 or 2 times, so long as the destination buffer is
  // an even multiple of 32 bits long.
  if ((num_bytes == 1 || num_bytes == 2) &&
      ShapeUtil::ByteSizeOf(output_shape) % 4 == 0) {
    uint16 pattern16;
    if (num_bytes == 1) {
      uint8 b = init_value.front();
      pattern16 = uint16{b} | (uint16{b} << 8);
    } else {
      memcpy(&pattern16, init_value.data(), sizeof(pattern16));
    }
    uint32 pattern32 = uint32{pattern16} | (uint32{pattern16} << 16);
    return absl::make_unique<Memset32BitValueThunk>(Thunk::ThunkInfo(),
                                                    pattern32, dest);
  }

  // If the literal is an even multiple of 32 bits wide, we can emit a 32-bit
  // memset so long as all 32-bit words of the scalar are equal to each other.
  if (num_bytes >= 4 && num_bytes % 4 == 0 &&
      memcmp(init_value.data(), init_value.data() + 4, init_value.size() - 4) ==
          0) {
    uint32 word;
    memcpy(&word, init_value.data(), sizeof(word));
    return absl::make_unique<Memset32BitValueThunk>(Thunk::ThunkInfo(), word,
                                                    dest);
  }

  return nullptr;
}

StatusOr<std::unique_ptr<Thunk>>
IrEmitterUnnested::TryBuildConstantInitializerThunk(mlir::Value init_value,
                                                    mlir::Value dest) {
  mlir::DenseElementsAttr const_init;
  if (auto get_global_memref =
          mlir::dyn_cast_or_null<mlir::memref::GetGlobalOp>(
              init_value.getDefiningOp())) {
    auto global_memref =
        mlir::SymbolTable::lookupNearestSymbolFrom<mlir::memref::GlobalOp>(
            get_global_memref, get_global_memref.name());
    if (global_memref.constant() && global_memref.initial_value()) {
      // If the initial value happens to be a constant, generate a specialized
      // thunk.
      const_init = global_memref.initial_value()
                       .getValue()
                       .cast<mlir::DenseElementsAttr>();
    }
  } else if (auto constant = mlir::dyn_cast_or_null<mlir::mhlo::ConstOp>(
                 init_value.getDefiningOp())) {
    const_init = constant.value().dyn_cast<mlir::DenseElementsAttr>();
  }

  if (const_init) {
    std::vector<uint8> literal_bytes;
    TF_RETURN_IF_ERROR(
        CopyDenseElementsDataToXlaFormat(const_init, &literal_bytes));

    TF_ASSIGN_OR_RETURN(auto dest_slice, GetAllocationSliceForMlir(dest));

    const Shape dest_shape = TypeToShape(dest.getType());
    auto thunk =
        BuildConstantInitializerThunk(literal_bytes, dest_slice, dest_shape);
    if (thunk) {
      return {std::move(thunk)};
    }
  }
  return std::unique_ptr<Thunk>();
}

StatusOr<std::unique_ptr<Thunk>>
IrEmitterUnnested::BuildInitializerThunkForMlir(mlir::Operation* op,
                                                mlir::Value init_value,
                                                mlir::Value dest) {
  // initial value must be a scalar memref.
  auto init_type = init_value.getType().dyn_cast<mlir::MemRefType>();
  TF_RET_CHECK(init_type.getRank() == 0);

  TF_ASSIGN_OR_RETURN(std::unique_ptr<Thunk> constant_init_thunk,
                      TryBuildConstantInitializerThunk(init_value, dest));
  if (constant_init_thunk) {
    return {std::move(constant_init_thunk)};
  }

  // Otherwise fall back to our slow initializer code. The thunk in this case
  // will just need the IR arrays for the initial value and the destination.
  std::vector<llvm_ir::IrArray> ir_arrays;
  TF_ASSIGN_OR_RETURN(std::unique_ptr<KernelThunk> kernel_thunk,
                      BuildKernelThunkForMlir(op, {init_value, dest},
                                              Thunk::ThunkInfo(), &ir_arrays));
  const llvm_ir::IrArray init_array = ir_arrays[0];
  const llvm_ir::IrArray dest_array = ir_arrays[1];

  const Shape dest_shape = TypeToShape(dest.getType());
  TF_ASSIGN_OR_RETURN(LaunchDimensions launch_dimensions,
                      CalculateLaunchDimensions(
                          dest_shape, ir_emitter_context_->gpu_device_info()));
  UpdateLaunchDimensions(launch_dimensions, kernel_thunk.get(),
                         ir_emitter_context_->llvm_module());

  std::string name = mlir::GetNameFromLoc(op->getLoc());
  TF_RETURN_IF_ERROR(ParallelLoopEmitter(
                         [=](const IrArray::Index& index) {
                           return init_array.EmitReadArrayElement(index, &b_);
                         },
                         dest_array, launch_dimensions, &b_)
                         .EmitLoop(mlir::GetNameFromLoc(op->getLoc())));

  // Convert unique_ptr<KernelThunk> to StatusOr<unique_ptr<Thunk>>.
  return {std::move(kernel_thunk)};
}

StatusOr<std::unique_ptr<Thunk>>
IrEmitterUnnested::BuildFusedInitializerThunkForMlir(
    mlir::lmhlo::FusionOp fusion, int output_index) {
  auto reduce = mlir::dyn_cast_or_null<mlir::mhlo::ReduceOp>(
      fusion.getFusionResults()[output_index].getDefiningOp());

  TF_RET_CHECK(reduce);
  TF_RET_CHECK(reduce.getNumResults() == 1);

  mlir::Value init_value = reduce.init_values()[0];
  mlir::Value dest = fusion.getOutputBuffers()[output_index];
  TF_ASSIGN_OR_RETURN(std::unique_ptr<Thunk> constant_init_thunk,
                      TryBuildConstantInitializerThunk(init_value, dest));
  if (constant_init_thunk) {
    return {std::move(constant_init_thunk)};
  }

  auto input_buffers = fusion.getInputBuffers();

  std::vector<llvm_ir::IrArray> ir_arrays;
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<KernelThunk> kernel_thunk,
      BuildKernelThunkForMlir(fusion, Thunk::ThunkInfo(), &ir_arrays));
  const llvm_ir::IrArray dest_array =
      ir_arrays[input_buffers.size() + output_index];

  const Shape dest_shape = TypeToShape(dest.getType());
  TF_ASSIGN_OR_RETURN(LaunchDimensions launch_dimensions,
                      CalculateLaunchDimensions(
                          dest_shape, ir_emitter_context_->gpu_device_info()));
  UpdateLaunchDimensions(launch_dimensions, kernel_thunk.get(),
                         ir_emitter_context_->llvm_module());

  const HloComputation* fused_computation =
      *GetOrCreateSubComputationFromRegion(&fusion.region(),
                                           /*is_fusion=*/true);

  // If init_value was fused into this reduce we have to generate it first.
  GpuElementalIrEmitter elemental_emitter(hlo_module_config_,
                                          ir_emitter_context_->llvm_module(),
                                          &b_, GetNestedComputer());

  FusedIrEmitter fused_emitter(&elemental_emitter);
  for (int i = 0; i < fused_computation->num_parameters(); i++) {
    fused_emitter.BindGenerator(
        fused_computation->parameter_instruction(i),
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
                      fused_emitter.GetGenerator(instr->operand(1)));
  TF_RETURN_IF_ERROR(
      ParallelLoopEmitter(generator, dest_array, launch_dimensions, &b_)
          .EmitLoop(mlir::GetNameFromLoc(fusion.getLoc())));
  return {std::move(kernel_thunk)};
}

StatusOr<std::unique_ptr<Thunk>> IrEmitterUnnested::BuildWhileThunk(
    mlir::lmhlo::WhileOp while_op, const Thunk::ThunkInfo& thunk_info) {
  // Generate thunk sequence for while 'condition'.
  mlir::Region* condition = &while_op.cond();
  TF_ASSIGN_OR_RETURN(
      auto ir_emitter_condition,
      IrEmitterUnnested::Create(hlo_module_config_, ir_emitter_context_));

  TF_RETURN_IF_ERROR(ir_emitter_condition->EmitLmhloRegion(condition));

  // Generate thunk sequence for while 'body'.
  mlir::Region* body = &while_op.body();
  TF_ASSIGN_OR_RETURN(
      auto ir_emitter_body,
      IrEmitterUnnested::Create(hlo_module_config_, ir_emitter_context_));

  TF_RETURN_IF_ERROR(ir_emitter_body->EmitLmhloRegion(body));

  // Extract the condition value from the last op (exlucidng the terminator op)
  // in the condition region.
  auto cond_result = GetHloOutputs(while_op);
  TF_RET_CHECK(cond_result.size() == 1);
  TF_ASSIGN_OR_RETURN(auto cond_result_slice,
                      GetAllocationSliceForMlir(cond_result[0]));

  return std::unique_ptr<Thunk>(
      new WhileThunk(thunk_info, cond_result_slice,
                     ir_emitter_condition->ConsumeThunkSequence(),
                     ir_emitter_body->ConsumeThunkSequence()));
}

StatusOr<std::unique_ptr<Thunk>> IrEmitterUnnested::BuildForThunk(
    mlir::lmhlo::WhileOp while_op, const Thunk::ThunkInfo& thunk_info,
    const int64 loop_limit) {
  // Generate thunk sequence for while 'body' (will be used a For loop body).
  mlir::Region* body = &while_op.body();
  TF_ASSIGN_OR_RETURN(
      auto ir_emitter_body,
      IrEmitterUnnested::Create(hlo_module_config_, ir_emitter_context_));
  for (mlir::Operation& op : llvm::make_early_inc_range(body->front())) {
    TF_RETURN_IF_ERROR(ir_emitter_body->EmitOp(MlirEmitterInput{&op}));
  }

  return std::unique_ptr<Thunk>(new ForThunk(
      thunk_info, loop_limit, ir_emitter_body->ConsumeThunkSequence()));
}

Status IrEmitterUnnested::EmitTargetElementLoop(
    const HloInstruction& hlo, const llvm_ir::ElementGenerator& body_emitter) {
  return InternalError("This should be unreachable");
}

// Gets the output offset as calculated from thread_id.x (to be applied to the
// offset calculated from block_id and thread_id.y).
static llvm::Value* GetStartOffsetX(const KernelMappingScheme& mapping_scheme,
                                    llvm::Value* thread_id_x,
                                    llvm::Type* index_ty,
                                    llvm::IRBuilder<>* b) {
  auto constant = [&](int64 val) {
    return llvm::ConstantInt::get(index_ty, val);
  };
  if (mapping_scheme.GetIndexingOrder() == kStridedIndexingX) {
    return thread_id_x;
  } else if (mapping_scheme.GetIndexingOrder() == kStridedLinearIndexingX) {
    return b->CreateMul(thread_id_x, constant(mapping_scheme.GetVectorSize()));
  }
  CHECK_EQ(mapping_scheme.GetIndexingOrder(), kLinearIndexingX);
  int64 x_num_steps =
      mapping_scheme.GetTileSizeX() / mapping_scheme.GetNumThreadsX();
  return b->CreateMul(thread_id_x, constant(x_num_steps));
}

// Calls `emit_elem_function()` `x_num_steps` times.  If
// `vector_size`==1, then each element index passed to
// `emit_elem_function()` will be separated by `step_x`. If `vector_size`>1,
// then it must be a multiple of `x_num_steps`.  In that case, it
// triggers a different indexing order that is vectorizable by
// LLVM. It generates many groups of calls to `emit_elem_function`. Each
// group is separated by `step_x` elements.  Inside a group, elements
// are consecutive. If `check_x_tile_bounds` is true, then it will check
// if the element index is in bound compared to `tile_width` before
// calling `emit_elem_function`.
static void UnrollInnerTileLoop(
    bool check_x_tile_bounds, int64 x_num_steps, int64 step_x,
    int64 vector_size, const string& loop_name, KernelSupportLibrary* ksl,
    llvm::Value* start_offset_x, llvm::Value* y_loc, llvm::Value* tile_width,
    const IrArray::Index& source_idx, llvm::IRBuilder<>* b,
    const IrEmitterUnnested::EmitElementFunction* emit_elem_function) {
  llvm::Type* index_ty = tile_width->getType();
  auto constant = [&](int64 val) {
    return llvm::ConstantInt::get(index_ty, val);
  };
  IrArray::Index source_idx_x_base = source_idx.AddOffsetToDim(y_loc, kDimY, b);
  for (int64 j = 0; j < x_num_steps / vector_size; j++) {
    for (int64 i = 0; i < vector_size; i++) {
      int64 linear_index = j * vector_size + i;
      llvm::Value* x_loc = b->CreateAdd(constant(j * step_x * vector_size + i),
                                        start_offset_x, "x_loc");
      IrArray::Index source_idx_x = source_idx_x_base.AddOffsetToDim(
          constant(j * step_x * vector_size + i), kDimX, b);
      auto emit_element = [&] {
        return (*emit_elem_function)(source_idx_x, y_loc, x_loc, linear_index);
      };
      if (check_x_tile_bounds) {
        ksl->If(loop_name + "_x_in_tile", b->CreateICmpULT(x_loc, tile_width),
                emit_element);
      } else {
        emit_element();
      }
    }
  }
}

void IrEmitterUnnested::EmitTile(
    const KernelMappingScheme& mapping_scheme,
    const IrArray::Index& tile_origin_index, const string& loop_name,
    KernelSupportLibrary* ksl, const ThreadIdInfo& thread_id_info,
    llvm::Value* tile_height, llvm::Value* tile_width,
    const IrEmitterUnnested::EmitElementFunction& emit_elem_function) {
  llvm::Type* index_ty = tile_width->getType();
  auto constant = [&](int64 val) {
    return llvm::ConstantInt::get(index_ty, val);
  };
  int64 num_threads_x = mapping_scheme.GetNumThreadsX();
  llvm::Value* num_threads_y = constant(mapping_scheme.GetNumThreadsY());
  int64 tile_size_x = mapping_scheme.GetTileSizeX();

  int64 x_num_steps = tile_size_x / num_threads_x;
  llvm::Value* start_offset_x = GetStartOffsetX(
      mapping_scheme, thread_id_info.thread_id_x, index_ty, &b_);

  // Using dilated mapping scheme, each thread steps with a stride of number
  // of threads.
  // Otherwise, the stride is one, but we multiply each offset by the limit of
  // number of steps which can be made.
  int64 step_x =
      mapping_scheme.GetIndexingOrder() == kLinearIndexingX ? 1 : num_threads_x;
  int64 vector_size = mapping_scheme.GetVectorSize();

  IrArray::Index source_idx =
      tile_origin_index.AddOffsetToDim(start_offset_x, kDimX, &b_);

  auto ceil_of_ratio = [&](llvm::Value* a, llvm::Value* b) {
    return b_.CreateUDiv(b_.CreateAdd(b_.CreateAdd(a, b), constant(-1)), b);
  };

  // True iff all threads always execute all instructions in the tiling
  // dimension X.
  bool x_tile_fits =
      mapping_scheme.GetDimsInElems()[kDimX] % tile_size_x == 0 &&
      mapping_scheme.GetRowContiguous();

  // The outer loop below is simply doing:
  //
  // for (int y_loc=thread_id_y; y_loc<tile_height; y_loc+=num_threads_y)
  //
  //
  // However, in order to avoid an LLVM optimization triggering the ptxas bug,
  // we write this loop in a convoluted way:
  //
  // y_bound = ceil_of_ratio(tile_height - thread_id_y, num_threads_y)
  // for (int y_indvar=0; y_indvar<y_bound; y_indvar+=1)
  //    y_loc = thread_id_y + y_indvar * num_threads_y
  //
  // TODO(cheshire): Once ptxas is fixed and TF switches to it, remove the
  // workaround.
  ksl->For(
      loop_name + "_y_in_tile",
      /*start=*/constant(0),
      /*end=*/
      ceil_of_ratio(b_.CreateSub(tile_height, thread_id_info.thread_id_y),
                    num_threads_y),
      /*step=*/constant(1), [&](llvm::Value* y_indvar) {
        llvm::Value* y_loc = b_.CreateAdd(
            thread_id_info.thread_id_y, b_.CreateMul(y_indvar, num_threads_y));
        auto unroll_inner_tile_loop = [&](bool check_x_tile_bounds) {
          return UnrollInnerTileLoop(check_x_tile_bounds, x_num_steps, step_x,
                                     vector_size, loop_name, ksl,
                                     start_offset_x, y_loc, tile_width,
                                     source_idx, &b_, &emit_elem_function);
        };

        // Only take this path when we unroll in a way vectorizable by
        // LLVM. Special case when the tile doesn't fit completely for even
        // row size. For odd row size every other row isn't aligned to the
        // vectorized size, so it can't be vectorized by LLVM.
        if (!x_tile_fits &&
            mapping_scheme.GetIndexingOrder() == kStridedLinearIndexingX) {
          ksl->If(
              loop_name + "_is_full_tile",
              // For the last block, tile_width will be the number of
              // elements left.
              b_.CreateICmpEQ(constant(mapping_scheme.GetTileSizeX()),
                              tile_width),
              [&] { unroll_inner_tile_loop(/*check_x_tile_bounds=*/false); },
              [&] { unroll_inner_tile_loop(/*check_x_tile_bounds=*/true); });
        } else {
          unroll_inner_tile_loop(/*check_x_tile_bounds=*/!x_tile_fits);
        }
      });
}

// Emits code to process a tensor element in a tile for the given kCopy HLO that
// performs a 0-2-1 transpose.
//
// index: The index for the first output element in the normalized tensor. The
//   normalized tensor is the resulting tensor after collapsing contiguous
//   dimensions that play the same role in the transpose.
// mapping_scheme: Kernel mapping scheme specifying the tiling
void IrEmitterUnnested::EmitTileElementForCopy(
    const Shape& output_shape, const llvm_ir::IrArray& output_array,
    const llvm_ir::IrArray::Index& index,
    const KernelMappingScheme& mapping_scheme, llvm::Value* y_loc,
    llvm::Value* x_loc, absl::Span<llvm::Value* const> param_shmem_buffers) {
  // TODO(jlebar): Add AA metadata to this load.
  llvm::Instruction* load_from_shmem_buffer =
      Load(GEP(param_shmem_buffers[0], {b_.getInt64(0), x_loc, y_loc}),
           "output_element");
  Shape output_reduced_shape = ShapeUtil::MakeShapeWithDescendingLayout(
      output_shape.element_type(), mapping_scheme.GetDimsInElems());
  // When the output_reduced_shape is a 0-2-1 transpose of the input shape,
  // the 0-2-1 transpose is achieved through EmitWriteArrayElement.
  output_array.CastToShape(output_reduced_shape, &b_)
      .EmitWriteArrayElement(index, load_from_shmem_buffer, &b_);
}

static IrArray::Index GetUnnormalizedIndex(
    const IrArray::Index& normalized_shape_index,
    const Shape& unnormalized_shape, llvm::IRBuilder<>* b_,
    const KernelMappingScheme& kernel_mapping_scheme) {
  DCHECK_EQ(normalized_shape_index.size(), 3);
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
  llvm::Value* linear = normalized_shape_index.Linearize(
      kernel_mapping_scheme.GetDimsInElems(), b_);
  return IrArray::Index(linear, unnormalized_shape, b_);
}

// Emits code to process a tensor element in a tile for the given kLoop fusion
// HLO containing parameters that are 0-2-1 transpose of its outputs.
//
// index: The index for the first output element in the normalized tensor, that
//   is the resulting tensor after collapsing contiguous dimensions that play
//   the same role in the transpose.
// kernel_info: Other information to support the kernel code generation.
void IrEmitterUnnested::EmitTileElementForFusion(
    mlir::lmhlo::FusionOp fusion,
    absl::Span<const llvm_ir::IrArray> operand_arrays,
    absl::Span<const llvm_ir::IrArray> output_arrays,
    const llvm_ir::IrArray::Index& index,
    const KernelMappingScheme& mapping_scheme, llvm::Value* y_loc,
    llvm::Value* x_loc, absl::Span<llvm::Value* const> param_shmem_buffers) {
  const HloComputation* fused_computation =
      *GetOrCreateSubComputationFromRegion(&fusion.region(),
                                           /*is_fusion=*/true);
  GpuElementalIrEmitter elem_emitter(hlo_module_config_, module_, &b_,
                                     GetNestedComputer());
  FusedIrEmitter fused_emitter(&elem_emitter);
  for (int i = 0; i < operand_arrays.size(); i++) {
    llvm_ir::ElementGenerator gen;
    if (llvm::Value* param_tile_buffer = param_shmem_buffers[i]) {
      gen = [this, param_tile_buffer, x_loc,
             y_loc](llvm_ir::IrArray::Index index) {
        // TODO(jlebar): Add AA metadata to this load.  Tile buffers are
        // global variables, so LLVM's points-to analysis doesn't help us
        // much.  And we want the AA info to be present before address
        // spaces are inferred (which is pretty late in the pipeline), so
        // even if we had address-space-based AA in LLVM, it wouldn't help
        // us much here.
        return b_.CreateLoad(
            b_.CreateGEP(param_tile_buffer,
                         {index.GetConstantWithIndexType(0), x_loc, y_loc}),
            "tiled_buffer");
      };
    } else {
      auto array = operand_arrays[i];
      auto name = fused_computation->parameter_instruction(i)->name();
      gen = [this, array, name](const llvm_ir::IrArray::Index& index) {
        return array.EmitReadArrayElement(index, &b_, name);
      };
    }
    fused_emitter.BindGenerator(fused_computation->parameter_instruction(i),
                                std::move(gen));
  }
  IrArray::Index untiled_index = GetUnnormalizedIndex(
      index, output_arrays[0].GetShape(), &b_, mapping_scheme);
  llvm_ir::ElementGenerator output_generator =
      *fused_emitter.GetGenerator(fused_computation->root_instruction());
  llvm::Value* output_value = output_generator(untiled_index).ValueOrDie();
  if (output_arrays.size() > 1) {
    DCHECK(output_value->getType()->isStructTy());
    DCHECK_EQ(output_value->getType()->getStructNumElements(),
              output_arrays.size());
    for (int64 i = 0; i < output_arrays.size(); ++i) {
      output_arrays[i].EmitWriteArrayElement(
          untiled_index, ExtractValue(output_value, i), &b_);
    }
  } else {
    output_arrays[0].EmitWriteArrayElement(untiled_index, output_value, &b_);
  }
}

static mlir::Operation* GetReduceFromUnnestedMlir(mlir::Operation* unnested_hlo,
                                                  int index) {
  if (mlir::isa<mlir::lmhlo::ReduceOp>(unnested_hlo)) {
    CHECK_EQ(0, index);
    return unnested_hlo;
  }
  if (auto fusion = mlir::dyn_cast<mlir::lmhlo::FusionOp>(unnested_hlo)) {
    auto results = fusion.getFusionResults();
    CHECK(index < results.size())
        << MlirToString(unnested_hlo) << " vs " << index;
    return results[index].getDefiningOp();
  }
  return nullptr;
}

void IrEmitterUnnested::EmitPrologueForReduction(
    mlir::Operation* unnested_hlo, absl::Span<const int> instr_index_group,
    HloComputation* fused_computation, FusedIrEmitter* fused_emitter,
    absl::Span<const llvm_ir::IrArray> operand_ir_arrays,
    absl::Span<const llvm_ir::IrArray> result_ir_arrays,
    ReductionCodegenInfo* reduction_info,
    const FusionLayoutAnalysis& layout_analysis) {
  VLOG(10) << "Emit prologue for reduction: " << MlirToString(unnested_hlo);
  mlir::Operation* first_reduce = nullptr;
  for (int index : instr_index_group) {
    mlir::Operation* reduce_inst =
        GetReduceFromUnnestedMlir(unnested_hlo, index);

    if (!IsReductionFromOrToContiguousDimensions(reduce_inst,
                                                 layout_analysis)) {
      continue;
    }

    auto results = GetHloOutputs(reduce_inst);
    CHECK_EQ(1, results.size());
    Shape reduce_inst_shape = layout_analysis.GetShape(results[0]);

    VLOG(10) << "Emit prologue for reduction: " << MlirToString(reduce_inst);
    if (first_reduce == nullptr) {
      first_reduce = reduce_inst;
    } else {
      CHECK(absl::c_equal(
          first_reduce->getAttrOfType<mlir::DenseIntElementsAttr>("dimensions"),
          reduce_inst->getAttrOfType<mlir::DenseIntElementsAttr>(
              "dimensions")));
    }

    AddressVector* reduction_input_addresses =
        reduction_info->GetMutableReductionInputAddresses();
    llvm::Type* element_type = llvm_ir::PrimitiveTypeToIrType(
        reduce_inst_shape.element_type(), ir_emitter_context_->llvm_module());
    llvm::AllocaInst* reduction_input_address =
        llvm_ir::EmitAllocaAtFunctionEntry(element_type,
                                           "reduction_input_address", &b_);
    reduction_input_addresses->push_back(reduction_input_address);

    int num_partial_results = reduction_info->GetNumPartialResults();
    AddressVector* partial_result_addresses =
        reduction_info->GetMutablePartialResultAddresses();
    llvm::AllocaInst* partial_result_address =
        llvm_ir::EmitAllocaAtFunctionEntryWithCount(
            element_type, /*ArraySize=*/b_.getInt32(num_partial_results),
            ("partial_reduction_result." + llvm::Twine(index)).str(), &b_);
    partial_result_addresses->push_back(partial_result_address);

    // Initialize the partial result with the initial value of the reduction.
    llvm::Value* init_ir_value;
    if (auto fusion = mlir::dyn_cast<mlir::lmhlo::FusionOp>(unnested_hlo)) {
      const HloInstruction* reduce_hlo = fused_computation->root_instruction();
      if (reduce_hlo->opcode() == HloOpcode::kTuple) {
        reduce_hlo = reduce_hlo->operand(index);
      }
      const HloInstruction* init_value = reduce_hlo->operand(1);

      init_ir_value = (*fused_emitter->GetGenerator(init_value))(
                          IrArray::Index(b_.getInt32Ty()))
                          .ValueOrDie();
    } else {
      init_ir_value = operand_ir_arrays[1].EmitReadArrayElement(
          IrArray::Index(b_.getInt32Ty()), &b_);
    }

    for (int i = 0; i < num_partial_results; ++i) {
      Store(init_ir_value,
            InBoundsGEP(partial_result_address, {b_.getInt32(i)}));
    }
    reduction_info->GetMutableInitialValues()->push_back(init_ir_value);

    auto& mapping_scheme = reduction_info->GetKernelMappingScheme();
    int64 num_threads_x = mapping_scheme.GetNumThreadsX();
    llvm::Type* primitive_type = llvm_ir::PrimitiveTypeToIrType(
        reduce_inst_shape.element_type(), module_);
    llvm::Type* buffer_type = [&] {
      if (reduction_info->IsRowReduction()) {
        // Allocate __shared__ cache[num_partial_results][kWarpSize].
        return llvm::ArrayType::get(
            llvm::ArrayType::get(primitive_type, kWarpSize),
            num_partial_results);
      } else {
        // Allocate __shared__
        // cache[num_partial_results][num_threads][num_threads + 1], where
        // num_threads == num_threads_x == num_threads_y.  The "+1" is used to
        // avoid bank conflicts.
        CHECK_EQ(num_threads_x, mapping_scheme.GetNumThreadsY());
        return llvm::ArrayType::get(
            llvm::ArrayType::get(
                llvm::ArrayType::get(primitive_type, num_threads_x + 1),
                num_threads_x),
            num_partial_results);
      }
    }();
    llvm::GlobalVariable* shared_cache_per_reduce =
        llvm_ir::AllocateSharedMemoryTile(b_.GetInsertBlock()->getModule(),
                                          buffer_type,
                                          absl::StrCat("shared_cache_", index));
    reduction_info->GetMutableSharedCache()->push_back(shared_cache_per_reduce);
  }
  CHECK(first_reduce);
}

void IrEmitterUnnested::EmitFullWarpShuffleDownLoopForAllReduces(
    absl::Span<HloComputation* const> reducers,
    absl::Span<llvm::AllocaInst* const> partial_result_addresses,
    int threads_per_block) {
  CHECK_EQ(reducers.size(), partial_result_addresses.size());
  for (int i = 0; i != reducers.size(); i++) {
    EmitFullWarpShuffleDownLoopForReduce(
        reducers[i], partial_result_addresses[i]->getType()->getElementType(),
        partial_result_addresses[i], threads_per_block);
  }
}

void IrEmitterUnnested::EmitFullWarpShuffleDownLoopForReduce(
    HloComputation* reducer, llvm::Type* element_type,
    llvm::Value* partial_result_address, int threads_per_block) {
  // This only works when the block size is a multiple of 32 threads.
  CHECK_EQ(threads_per_block % 32, 0);
  for (int distance = 16; distance >= 1; distance /= 2) {
    int bit_width = llvm_ir::GetSizeInBits(element_type);
    llvm::Value* result_from_other_lane = llvm_ir::EmitAllocaAtFunctionEntry(
        element_type, "result_from_other_lane", &b_);
    // Bitcast cannot be applied to aggregate types (even packed ones), so
    // we bitcast addresses of load/store to intN* of the same bit-width.
    llvm::Type* shuffled_value_type =
        element_type->isStructTy() ? b_.getIntNTy(bit_width) : element_type;
    auto convert_pointer_for_shuffle = [&](llvm::Value* ptr) {
      return b_.CreatePointerBitCastOrAddrSpaceCast(
          ptr, shuffled_value_type->getPointerTo());
    };
    llvm::Value* partial_result =
        Load(convert_pointer_for_shuffle(partial_result_address),
             "partial_reduction_result");
    Store(EmitFullWarpShuffleDown(partial_result, b_.getInt32(distance), &b_),
          convert_pointer_for_shuffle(result_from_other_lane));
    TF_CHECK_OK(EmitCallToNestedComputation(
        *reducer, {partial_result_address, result_from_other_lane},
        partial_result_address));
  }
}

// Given the IrArray index of a reduction input, returns the linear address of
// the reduction output as if the reduction were going to keep the input shape
// with the dimensions being reduced moved.
static llvm::Value* GetUntransposedOutputLinearAddress(
    llvm::IRBuilder<>* b, const llvm_ir::IrArray::Index& index,
    const ReductionCodegenInfo& reduction_info) {
  const KernelMappingScheme& kernel_mapping_scheme =
      reduction_info.GetKernelMappingScheme();
  if (reduction_info.IsRowReduction()) {
    // For row-reduction, y-coordinate determines which row we write into.
    return index[kDimY];
  }
  // For column reduction, we get the transposed address.
  absl::Span<const int64> dims_in_elem = kernel_mapping_scheme.GetDimsInElems();
  llvm::Value* x_dim_size = index.GetConstantWithIndexType(dims_in_elem[kDimX]);
  llvm::Value* x_block_offset = b->CreateMul(index[kDimZ], x_dim_size);
  return b->CreateAdd(x_block_offset, index[kDimX]);
}

void IrEmitterUnnested::EmitEpilogueForReduction(
    llvm::Type* index_ty, mlir::Operation* unnested_hlo,
    absl::Span<const int> instr_index_group,
    absl::Span<const llvm_ir::IrArray> result_ir_arrays,
    absl::Span<HloComputation* const> reducers,
    const ReductionCodegenInfo& reduction_info,
    const TilingKernelInfo& tiling_kernel_info,
    const FusionLayoutAnalysis& layout_analysis) {
  const KernelMappingScheme& mapping_scheme =
      reduction_info.GetKernelMappingScheme();
  auto constant = [&](uint64 c) -> llvm::Constant* {
    return llvm::ConstantInt::get(index_ty, c);
  };

  IrEmitterUnnested::ThreadIdInfo thread_id_info =
      EmitThreadIdInfo(mapping_scheme.GetThreadsPerBlock(), index_ty,
                       mapping_scheme.GetNumThreadsX());

  IrArray::Index start_offset = [&] {
    llvm::Value* x_loc = thread_id_info.thread_id_x;
    llvm::Value* y_loc = thread_id_info.thread_id_y;
    if (!reduction_info.IsRowReduction()) {
      std::swap(x_loc, y_loc);
    }
    llvm::Value* start_offset_x =
        GetStartOffsetX(mapping_scheme, x_loc, index_ty, &b_);
    return tiling_kernel_info.tile_origin.AddOffsetToDim(y_loc, kDimY, &b_)
        .AddOffsetToDim(start_offset_x, kDimX, &b_);
  }();

  absl::Span<llvm::AllocaInst* const> partial_result_addresses =
      reduction_info.GetPartialResultAddresses();

  int num_partial_results = reduction_info.GetNumPartialResults();

  // Emit an atomic operation that accumulates the partial reduction to the
  // output element. For row reduction, this is only for lane 0 due to the
  // if-statement emitted above.
  //
  // `i` is the compacted index for contiguous-dimension reductions. It's used
  // for accessing `reduction_info` and `reducers`, which are also compacted.
  int i = -1;
  for (int index : instr_index_group) {
    mlir::Operation* reduce_hlo =
        GetReduceFromUnnestedMlir(unnested_hlo, index);
    if (!IsReductionFromOrToContiguousDimensions(reduce_hlo, layout_analysis)) {
      continue;
    }
    i++;
    Shape operand_shape = layout_analysis.GetShape(reduce_hlo->getOperand(0));
    Shape reduction_kept_element_shape = ShapeUtil::FilterDimensions(
        [&](int64 dim) {
          return !absl::c_linear_search(
              reduce_hlo->getAttrOfType<mlir::DenseIntElementsAttr>(
                  "dimensions"),
              dim);
        },
        operand_shape);
    for (int j = 0; j < num_partial_results; ++j) {
      llvm::Value* untransposed_output_linear_address =
          GetUntransposedOutputLinearAddress(
              &b_, start_offset.AddOffsetToDim(constant(j), kDimX, &b_),
              reduction_info);

      // A reduction is allowed to transpose its output.  For example, suppose
      // we are reducing the second dimension of f32[10,20,30]{3,2,1}.  We are
      // allowed to produce as output either f32[10,30]{1,0} (no transpose) or
      // f32[10,30]{0,1} (transposing the two output dims).
      //
      // At this point in the function we have a "partial sum" of input elements
      // (stored in partial_result_addresses), and we need to accumulate it into
      // the correct output element.
      auto output_array = result_ir_arrays[index];
      IrArray::Index element_index(
          /*linear=*/untransposed_output_linear_address,
          reduction_kept_element_shape, &b_);
      IrArray::Index output_index(element_index.multidim(),
                                  output_array.GetShape(),
                                  element_index.GetType());
      llvm::Value* output_address = output_array.EmitArrayElementAddress(
          output_index, &b_, "output_element_address");
      llvm::Value* current_output = b_.CreateInBoundsGEP(
          partial_result_addresses[i], {constant(j)}, "current_output");

      llvm::GlobalVariable* shared_cache = reduction_info.GetSharedCache()[i];

      // __shared__ memory uses a different address space, so we cast it to
      // global address space before writing or reading.
      auto shared_to_global = [&](llvm::Value* input, llvm::Twine name = "") {
        return b_.CreateAddrSpaceCast(
            input,
            llvm::PointerType::get(input->getType()->getPointerElementType(),
                                   /*AddressSpace=*/0),
            name);
      };

      auto is_zero = [&](llvm::Value* value) {
        return b_.CreateICmpEQ(value, constant(0));
      };

      KernelSupportLibrary ksl(&b_);
      llvm::Type* element_type =
          partial_result_addresses[i]->getType()->getElementType();
      if (reduction_info.IsRowReduction()) {
        EmitFullWarpShuffleDownLoopForReduce(
            reducers[i], element_type, current_output,
            mapping_scheme.GetThreadsPerBlock());
        llvm::Value* warp_id =
            b_.CreateUDiv(thread_id_info.thread_id_x, constant(kWarpSize));
        ksl.If("intra_warp_reduce_write", is_zero(thread_id_info.lane_id), [&] {
          llvm::Value* shmem_output_addr =
              shared_to_global(b_.CreateInBoundsGEP(
                  shared_cache, {b_.getInt32(0), constant(j), warp_id}));
          b_.CreateStore(b_.CreateLoad(current_output), shmem_output_addr);
        });

        EmitSyncThreads();
        ksl.If("inter_warp_reduce", is_zero(warp_id), [&] {
          llvm::Value* block_accum_addr = shared_to_global(b_.CreateInBoundsGEP(
              shared_cache,
              {b_.getInt32(0), constant(j), thread_id_info.lane_id}));
          llvm::Value* initial_value = reduction_info.GetInitialValues()[i];
          llvm::Value* initial_value_addr =
              shared_to_global(llvm_ir::EmitAllocaAtFunctionEntry(
                  element_type, "initial_value_addr", &b_));
          b_.CreateStore(initial_value, initial_value_addr);

          llvm::Value* warp_exists = b_.CreateICmpULT(
              thread_id_info.thread_id_x,
              constant(mapping_scheme.GetNumThreadsX() / kWarpSize));

          llvm::Value* selected_value = b_.CreateSelect(
              warp_exists, block_accum_addr, initial_value_addr);

          EmitFullWarpShuffleDownLoopForReduce(
              reducers[i], element_type,
              /*block_accum_addr*/ selected_value,
              mapping_scheme.GetThreadsPerBlock());
          ksl.If("reduction_atomic_update", is_zero(thread_id_info.thread_id_x),
                 [&] {
                   TF_CHECK_OK(EmitAtomicOperationForNestedComputation(
                       *reducers[i], output_address, block_accum_addr));
                 });
        });

      } else {
        llvm::Value* shmem_output_addr = shared_to_global(
            b_.CreateInBoundsGEP(shared_cache, {b_.getInt32(0), constant(j),
                                                thread_id_info.thread_id_x,
                                                thread_id_info.thread_id_y}),
            "shmem_output_address");
        llvm::Value* current_output_value = b_.CreateLoad(current_output);
        b_.CreateStore(current_output_value, shmem_output_addr);

        EmitSyncThreads();

        // Get transposed element from shared memory.
        llvm::Value* shmem_transposed_addr =
            shared_to_global(b_.CreateInBoundsGEP(
                shared_cache,
                {b_.getInt32(0), constant(j), thread_id_info.thread_id_y,
                 thread_id_info.thread_id_x},
                "shmem_transposed_addr"));

        EmitFullWarpShuffleDownLoopForReduce(
            reducers[i], element_type, shmem_transposed_addr,
            mapping_scheme.GetThreadsPerBlock());

        // Some threads in the block are completely outside of the bound of the
        // tensor, so they should not write any output at all.
        llvm::Value* has_output = b_.CreateAnd(
            b_.CreateICmpULT(
                GetStartOffsetX(mapping_scheme, thread_id_info.thread_id_y,
                                index_ty, &b_),
                tiling_kernel_info.output_tile_bounds[kDimX]),
            b_.CreateICmpULT(thread_id_info.thread_id_x,
                             tiling_kernel_info.output_tile_bounds[kDimY]));

        ksl.If("reduction_atomic_update",
               b_.CreateAnd(has_output, is_zero(thread_id_info.lane_id)), [&] {
                 TF_CHECK_OK(EmitAtomicOperationForNestedComputation(
                     *reducers[i], output_address, shmem_transposed_addr));
               });
      }
    }
  }
}

llvm::Value* IrEmitterUnnested::EmitBlockId() {
  return gpu::EmitCallToTargetIntrinsic(gpu::TargetIntrinsicID::kBlockIdx, {},
                                        {}, &b_);
}

void IrEmitterUnnested::EmitPrintfWithThreadId(
    absl::string_view fmt, absl::Span<llvm::Value* const> arguments,
    absl::optional<int64> thread_id_filter,
    absl::optional<int64> block_id_filter) {
  llvm::Value* thread_id = EmitThreadId(1024, b_.getInt32Ty());
  llvm::Value* block_id = EmitBlockId();
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

void IrEmitterUnnested::EmitTileElementForReduction(
    mlir::Operation* unnested_hlo, const Shape& reduction_operand_shape,
    absl::Span<const int> instr_index_group, HloComputation* fused_computation,
    FusedIrEmitter* fused_emitter,
    absl::Span<const llvm_ir::IrArray> operand_ir_arrays,
    absl::Span<const llvm_ir::IrArray> result_ir_arrays,
    absl::Span<HloComputation* const> reducers,
    const llvm_ir::IrArray::Index& index,
    const ReductionCodegenInfo& reduction_info, int64 x_iter_num,
    const FusionLayoutAnalysis& layout_analysis) {
  VLOG(10) << "Emit tile element for reduce " << MlirToString(unnested_hlo);
  int partial_result_index = reduction_info.IsRowReduction() ? 0 : x_iter_num;

  InlinedVector<llvm_ir::ElementGenerator, 1> input_gens;
  std::vector<std::pair<llvm_ir::ElementGenerator, int>> extra_output_gens;

  // Construct the ElementGenerator for each reduction and extra output in the
  // the group of output instructions.
  if (auto fusion = mlir::dyn_cast<mlir::lmhlo::FusionOp>(unnested_hlo)) {
    for (int index : instr_index_group) {
      mlir::Operation* inst = GetReduceFromUnnestedMlir(unnested_hlo, index);

      const HloInstruction* hlo = fused_computation->root_instruction();
      if (hlo->opcode() == HloOpcode::kTuple) {
        hlo = hlo->operand(index);
      }
      if (IsReductionFromOrToContiguousDimensions(inst, layout_analysis)) {
        input_gens.push_back(*fused_emitter->GetGenerator(hlo->operand(0)));
      } else {
        extra_output_gens.emplace_back(*fused_emitter->GetGenerator(hlo),
                                       index);
      }
    }
  } else {
    input_gens.push_back([&](const IrArray::Index& index) {
      return operand_ir_arrays[0].EmitReadArrayElement(index, &b_);
    });
  }

  IrArray::Index input_index =
      GetUnnormalizedIndex(index, reduction_operand_shape, &b_,
                           reduction_info.GetKernelMappingScheme());
  // Clear the linear index field of the IrArray::Index to enable the use of
  // GetElementPointer with array types. This enables the vectorization of
  // the computation for different partial results. Use this index if
  // 'num_partial_results > 1'.
  int num_partial_results = reduction_info.GetNumPartialResults();
  auto index_without_linear = IrArray::Index(
      input_index.multidim(), reduction_operand_shape, input_index.GetType());

  // Emit code to generate the input and perform the reduction computation for
  // each reduction instruction.
  for (int i = 0; i < reducers.size(); i++) {
    llvm::AllocaInst* input_address =
        reduction_info.GetReductionInputAddresses()[i];
    llvm::AllocaInst* partial_reduction_result_address =
        reduction_info.GetPartialResultAddresses()[i];
    llvm::Value* const input_ir_value =
        input_gens[i](num_partial_results > 1 ? index_without_linear
                                              : input_index)
            .ValueOrDie();
    Store(input_ir_value, input_address);
    llvm::Value* partial_result_address = InBoundsGEP(
        partial_reduction_result_address, {b_.getInt32(partial_result_index)});
    TF_CHECK_OK(EmitCallToNestedComputation(
        *reducers[i], {partial_result_address, input_address},
        partial_result_address));
  }

  // Emit code to generate the output for the non-reduction instructions in the
  // fusion, if any.
  TF_CHECK_OK(EmitExtraOutputsForReduce(
      result_ir_arrays, input_index,
      /*use_linear_index=*/num_partial_results == 1, extra_output_gens));
}

llvm::Value* IrEmitterUnnested::EmitThreadId(int64 threads_per_block,
                                             llvm::Type* index_ty) {
  // Calculate (y, x) coordinates respectively in the 2D view of thread block,
  // defined by (num_thread_y, num_thread_x) from thread_id.
  llvm::CallInst* thread_id_raw = gpu::EmitCallToTargetIntrinsic(
      gpu::TargetIntrinsicID::kThreadIdx, {}, {}, &b_);
  llvm_ir::AddRangeMetadata(0, threads_per_block, thread_id_raw);
  return b_.CreateIntCast(thread_id_raw, index_ty,
                          /*isSigned=*/true, "thread.id.x");
}

IrEmitterUnnested::ThreadIdInfo IrEmitterUnnested::EmitThreadIdInfo(
    int64 threads_per_block, llvm::Type* index_ty, int64 num_threads_x) {
  auto constant = [&](uint64 c) -> llvm::Constant* {
    return llvm::ConstantInt::get(index_ty, c);
  };
  llvm::Value* thread_id = EmitThreadId(threads_per_block, index_ty);
  llvm::Value* num_threads_x_v = constant(num_threads_x);
  return {
      /*thread_id=*/thread_id,
      /*thread_id_x=*/b_.CreateURem(thread_id, num_threads_x_v, "thread_id.x"),
      /*thread_id_y=*/b_.CreateUDiv(thread_id, num_threads_x_v, "thread_id.y"),
      /*lane_id=*/b_.CreateURem(thread_id, constant(kWarpSize), "lane_id")};
}

IrEmitterUnnested::TilingKernelInfo IrEmitterUnnested::EmitTilingKernel(
    const KernelMappingScheme& mapping_scheme, llvm::Type* index_ty,
    const TileElementGenerator& tile_element_generator) {
  absl::Span<const int64> dims_in_elems = mapping_scheme.GetDimsInElems();
  std::vector<int64> dims_in_blocks = {
      CeilOfRatio(dims_in_elems[0], mapping_scheme.GetTileSizeZ()),
      CeilOfRatio(dims_in_elems[1], mapping_scheme.GetTileSizeY()),
      CeilOfRatio(dims_in_elems[2], mapping_scheme.GetTileSizeX())};
  auto constant = [&](uint64 c) -> llvm::Constant* {
    return llvm::ConstantInt::get(index_ty, c);
  };

  IrEmitterUnnested::ThreadIdInfo thread_id_info =
      EmitThreadIdInfo(mapping_scheme.GetThreadsPerBlock(), index_ty,
                       mapping_scheme.GetNumThreadsX());

  KernelSupportLibrary ksl(&b_, llvm_ir::UnrollMode::kDefaultUnroll);

  const IrArray::Index block_coords = [&] {
    llvm::Value* block_id = EmitBlockId();
    llvm_ir::AddRangeMetadata(0, mapping_scheme.GetNumberOfBlocks(),
                              llvm::cast<llvm::Instruction>(block_id));
    llvm::Value* linear_block_id =
        b_.CreateIntCast(block_id, index_ty, /*isSigned=*/true, "block.id.x");
    IrArray::Index starting_block(linear_block_id,
                                  ShapeUtil::MakeShapeWithDescendingLayout(
                                      PRED /*arbitrary*/, dims_in_blocks),
                                  &b_);

    std::vector<llvm::Value*> multidim = {
        b_.CreateMul(starting_block[0], constant(mapping_scheme.GetTileSizeZ()),
                     "block_origin.z"),
        starting_block[1], starting_block[2]};
    return IrArray::Index(multidim, dims_in_blocks, index_ty);
  }();

  std::array<llvm::Value*, 3> output_tile_bounds;
  for (int i = kDimY; i < kDimTot; ++i) {
    int64 tile_size_for_dim = mapping_scheme.GetTileSizeFor(i);
    // Only last row or column may not have full size.
    llvm::Value* is_last =
        b_.CreateICmpEQ(block_coords[i], constant(dims_in_blocks[i] - 1));
    int64 partial_row =
        dims_in_elems[i] - (dims_in_blocks[i] - 1) * tile_size_for_dim;
    output_tile_bounds[i] =
        b_.CreateSelect(is_last, constant(partial_row),
                        constant(tile_size_for_dim), "tile_bound");
  }

  IrArray::Index tile_origin = [&] {
    std::vector<llvm::Value*> elem_multi_index = block_coords.multidim();
    llvm::Type* index_ty = block_coords.GetType();
    for (int i = kDimY; i < kDimTot; ++i) {
      elem_multi_index[i] = b_.CreateMul(
          block_coords[i],
          llvm::ConstantInt::get(index_ty, mapping_scheme.GetTileSizeFor(i)),
          "tile_origin." + std::to_string(i));
    }
    return IrArray::Index(elem_multi_index, mapping_scheme.GetDimsInElems(),
                          index_ty);
  }();

  auto emit_tile = [&](const IrArray::Index& tile) {
    tile_element_generator(thread_id_info, tile, "output",
                           output_tile_bounds[1], output_tile_bounds[2], &ksl);
  };

  if (mapping_scheme.GetTileSizeZ() == 1) {
    emit_tile(tile_origin);
  } else {
    llvm::Value* starting_tile_index_for_dim = tile_origin[kDimZ];
    llvm::Value* block_size_for_dim = constant(mapping_scheme.GetTileSizeZ());
    llvm::Value* block_id_for_dim =
        b_.CreateUDiv(starting_tile_index_for_dim, block_size_for_dim);
    llvm::Value* last_block_for_dim = constant(dims_in_blocks[kDimZ] - 1);
    llvm::Value* last_block_size_for_dim =
        constant(dims_in_elems[kDimZ] -
                 (dims_in_blocks[kDimZ] - 1) * mapping_scheme.GetTileSizeZ());

    llvm::Value* num_tiles_in_block =
        b_.CreateSelect(b_.CreateICmpEQ(last_block_for_dim, block_id_for_dim),
                        last_block_size_for_dim, block_size_for_dim);
    ksl.For("loop_z",
            /*start=*/constant(0),
            /*end=*/num_tiles_in_block,
            /*step=*/1, [&](llvm::Value* block_dim_induction_var) {
              IrArray::Index tile_index = tile_origin.AddOffsetToDim(
                  block_dim_induction_var, kDimZ, &b_);
              emit_tile(tile_index);
            });
  }
  return {output_tile_bounds, tile_origin};
}

llvm::CallInst* IrEmitterUnnested::EmitSyncThreads() {
  return EmitCallToTargetIntrinsic(TargetIntrinsicID::kBarrierId, {}, {}, &b_);
}

// Emits a kernel for the given hlo instruction using a tiled 0-2-1 transpose
// algorithm to improve the memory access patterns for the input parameters
// with a shape that is a 0-2-1 transpose of the output tensor shape. The caller
// is responsible for making sure that it is safe to apply the shared memory
// transpose on the input parameters.
//
//
// For the purpose of tiling, the output tensors have a logical shape of three
// components 0-2-1 while the relevant input parameters have a logical shape
// of three components 0-1-2 in the order major to minor. The x- and y-
// dimensions of the tensors are tiled in square tiles with an edge length
// `kTileSize`. Each thread block of `kTileSize` x `kNumRows` threads
// transposes one tile: each thread copies kTileSize/kNumRows elements from
// the input to a shared memory tile, then the otherwise "regular HLO kernel"
// reads from the shared memory instead of the original input.
//
// This is similar to the following CUDA algorithm in TensorFlow:
// https://goo.gl/MStRV6.
//
// `kTileSize` should usually be same as warp size. We currently choose 32 for
// `kTileSize` and 4 for `kNumRows`. The CUDA algorithm uses 8 for `kNumRows`.
//
// TODO(b/33320379): Here each block transposes 1 tile. It may be more
// efficient to launch fewer blocks so each transposes many tiles.
void IrEmitterUnnested::EmitHlo021Tile(
    mlir::Operation* op, Thunk* kernel_thunk, const MlirEmitterContext& context,
    absl::Span<const llvm_ir::IrArray> operand_arrays,
    absl::Span<const llvm_ir::IrArray> output_arrays,
    absl::Span<const int64> reduced_output_dims,
    absl::Span<const int64> tiled_param_ids) {
  constexpr int kNumRows = 4;

  std::string name = mlir::GetNameFromLoc(op->getLoc());

  KernelMappingScheme mapping_scheme(reduced_output_dims,
                                     /*tile_sizes=*/{1, kWarpSize, kWarpSize},
                                     /*num_threads_y=*/kNumRows,
                                     /*num_threads_x=*/kWarpSize,
                                     /*indexing_order=*/kLinearIndexingX,
                                     /*vector_size=*/1,
                                     /*is_row_contiguous=*/false);
  LaunchDimensions launch_dimensions(mapping_scheme.GetNumberOfBlocks(),
                                     mapping_scheme.GetThreadsPerBlock());

  llvm::Type* index_type =
      GetIndexTypeForKernelFromMlir(op, launch_dimensions.launch_bound(), &b_);
  std::vector<IrArray> param_arrays;

  // For each tiled parameter, cast its input IrArray to the corresponding
  // reduced shape and keep the reduced shape live during IR emission.
  std::vector<IrArray> param_in_reduced_shape_arrays;
  std::vector<llvm::Value*> param_shmem_buffers(context.operand_shapes.size(),
                                                nullptr);

  auto get_shared_memory_buffer = [&](llvm::Type* elem_ty,
                                      absl::string_view buffer_name) {
    // For Nvidia GPUs, the warp size is 32 threads and the shared memory bank
    // is organized into 32-way. We usually use the warp size or a multiplier or
    // a the warp size as the size for tiling. This may cause all elements in
    // the same column of a tile use the same memory bank and therefore shared
    // memory bank conflicts. Adding 1 to the minor dimension of the shared
    // memory buffer can reduce such shared memory bank conflicts.
    llvm::Type* buffer_type = llvm::ArrayType::get(
        llvm::ArrayType::get(elem_ty, mapping_scheme.GetTileSizeX() + 1),
        mapping_scheme.GetTileSizeY());
    return llvm_ir::AllocateSharedMemoryTile(b_.GetInsertBlock()->getModule(),
                                             buffer_type, buffer_name);
  };

  for (int64 id = 0; id < context.operand_shapes.size(); id++) {
    const Shape& param_shape = context.operand_shapes[id];
    param_arrays.push_back(operand_arrays[id]);

    if (absl::c_linear_search(tiled_param_ids, id)) {
      param_shmem_buffers[id] = get_shared_memory_buffer(
          llvm_ir::PrimitiveTypeToIrType(param_shape.element_type(), module_),
          IrName(name, StrCat("tile", id)));
      VLOG(3) << "Added shmem buffer for parameter " << id << ": "
              << llvm_ir::DumpToString(*param_shmem_buffers[id]);
      Shape reduced_shape = ShapeUtil::MakeShapeWithDescendingLayout(
          param_shape.element_type(), Permute(reduced_output_dims, {0, 2, 1}));
      param_in_reduced_shape_arrays.push_back(
          param_arrays[id].CastToShape(reduced_shape, &b_));
    } else {
      param_in_reduced_shape_arrays.push_back(IrArray());
    }
  }

  EmitElementFunction element_generator =
      [&](const llvm_ir::IrArray::Index& index, llvm::Value* y_loc,
          llvm::Value* x_loc, int64 x_iter_num) {
        if (auto copy = mlir::dyn_cast<mlir::lmhlo::CopyOp>(op)) {
          CHECK_EQ(1, context.output_shapes.size());
          EmitTileElementForCopy(context.output_shapes[0], output_arrays[0],
                                 index, mapping_scheme, y_loc, x_loc,
                                 param_shmem_buffers);
        } else if (auto fusion = mlir::dyn_cast<mlir::lmhlo::FusionOp>(op)) {
          EmitTileElementForFusion(fusion, operand_arrays, output_arrays, index,
                                   mapping_scheme, y_loc, x_loc,
                                   param_shmem_buffers);
        } else {
          LOG(FATAL) << "Unexpected op: " << MlirToString(op);
        }
      };

  TileElementGenerator tile_generator =
      [&](const ThreadIdInfo& thread_id_info, const IrArray::Index& index,
          const string& loop_name, llvm::Value* tile_height,
          llvm::Value* tile_width, KernelSupportLibrary* ksl) {
        // If shared memory transpose is needed, wait for all threads to reach
        // this point, lest we copy a value from tile to output before the other
        // thread copies it from input to tile. This is `__syncthreads` in CUDA.
        if (!tiled_param_ids.empty()) {
          // Calculate the input tile origin from the output tile origin.
          const IrArray::Index input_tile_origin(
              Permute(index.multidim(), {0, 2, 1}),
              Permute(index.dims(), {0, 2, 1}), index.GetType());

          // Copy input parameter values to shared memory buffers:
          // tile[thread_id_y, thread_id_x] = input[index]
          // Note that tile_width and tile_height are flipped here because we
          // are reading a transposed tile.
          EmitTile(mapping_scheme, input_tile_origin, "input", ksl,
                   thread_id_info, tile_width, tile_height,
                   [&](const IrArray::Index& index, llvm::Value* y_loc,
                       llvm::Value* x_loc, int64 /*x_iter_num*/) {
                     for (int64 id : tiled_param_ids) {
                       IrArray& input_in_logical_shape =
                           param_in_reduced_shape_arrays.at(id);

                       llvm::Value* shmem_buffer = param_shmem_buffers.at(id);
                       llvm::Value* zero =
                           llvm::ConstantInt::get(index_type, 0);
                       // TODO(jlebar): Add AA metadata to this store.  Tile
                       // buffers are global variables, so LLVM can't infer much
                       // about it.
                       auto value = input_in_logical_shape.EmitReadArrayElement(
                           index, &b_, "input_element");
                       auto addr = GEP(shmem_buffer, {zero, y_loc, x_loc});
                       Store(value, addr);
                     }
                   });

          // Wait for all threads to reach this point using `__syncthreads` in
          // CUDA.
          EmitSyncThreads();
        }

        EmitTile(mapping_scheme, index, loop_name, ksl, thread_id_info,
                 tile_height, tile_width, element_generator);
        bool block_contains_multi_tiles = mapping_scheme.GetTileSizeZ() > 1;

        // If a tile block contains multiple tiles and shared memory buffers are
        // used, we need to wait for all threads to finish using the shared
        // memory buffer for the current tile before we move on to process the
        // next tile and overwrite the shared memory buffers.
        if (block_contains_multi_tiles && !tiled_param_ids.empty()) {
          EmitSyncThreads();
        }
      };

  EmitTilingKernel(mapping_scheme, index_type, tile_generator);
  UpdateLaunchDimensions(launch_dimensions, kernel_thunk,
                         ir_emitter_context_->llvm_module());
}

namespace {

// A recursive function to inspect the users of a parameter to determine
// whether it's safe for a parameter to participate in a shared-memory
// transpose.
//
// Consider a fusion parameter P for which we might want to use a shmem
// transpose.  If we do, we use a GPU thread block to preload a tile of P with
// indices [z, y..y+31, x..x+31] to compute an output tile with the same indices
// cooperatively, where z, y, x are the indices for the normalized input/output
// tensor (see the document for FindTranspose021 for the definition of
// normalized tensor for 0-2-1 transpose). This shmem transpose implementation
// requires that the computation of the output tile only read elements within
// the preload tile. If this is not true, we can't use a shmem transpose for P.
//
// If the computation of output element [z, y, x] only requires the element of
// P with the same indices, the shmem transpose implementation can be applied
// to P safely. This is a sufficient but not necessary condition. We check all
// the transitive users of P to see if we can find a user that may cause an
// exception to the situation. If such a user is not found, we conclude that P
// is safe for shmem transpose.
//
// This is trivially true for elementwise operations and some "data-movement"
// ops like kTuple. However, it's not true for operations that can change the
// dimensions of the inputs (e.g. pad, slice) and bitcast operation.
// For example:
//
// fused_computation {
//   param_0 = f32[64,64]{1,0} parameter(0)
//   ROOT bitcast = f32[64,64]{0,1} bitcast(param_0)
// }
// The output element at logical address [0, 63] depends on the input element
// at logical address [63, 0], which would not be within the shared-memory
// block.
//
// TODO(bixia): In order to extend this for kInput fusion, that is reduction
// with transpose, we only need to end the use-chain checking with the input of
// a reduce operations. In this case, the above description on "output" apply
// to the result of such a use-chain, which provides the input to the reduce
// operation.
bool IsInstructionSafeForShmemTranspose(mlir::Operation* op) {
  if (mlir::isa<mlir::memref::TensorStoreOp>(op)) {
    return true;
  }

  HloOpcode opcode;
  if (mlir::isa<mlir::memref::TensorLoadOp>(op)) {
    opcode = HloOpcode::kParameter;
  } else {
    opcode = *MhloToHloOpcode(op);
  }
  if (HloInstruction::IsOpElementwise(opcode)) {
    for (mlir::Value v : op->getResults()) {
      for (mlir::OpOperand use : v.getUsers()) {
        if (!IsInstructionSafeForShmemTranspose(use.getOwner())) {
          return false;
        }
      }
    }
    return true;
  }

  switch (opcode) {
    // Non-elementwise instructions that don't cause the shmem transpose
    // to be unsafe, including the instructions that don't currently fuse.
    case HloOpcode::kGetDimensionSize:
      // The result of the operation doesn't rely on the content of the
      // tensor. As such, there is no need to further inspect its users.
      return true;
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kMap:
    case HloOpcode::kParameter:
    case HloOpcode::kTuple:
    case HloOpcode::kTupleSelect:
      for (mlir::Value v : op->getResults()) {
        for (mlir::OpOperand use : v.getUsers()) {
          if (!IsInstructionSafeForShmemTranspose(use.getOwner())) {
            return false;
          }
        }
      }
      return true;

    default:
      return false;
  }
}

// Given a group of input parameters that are 0-2-1 transpose of the outputs of
// a fusion kernel, returns the input parameters that are safe for the shared
// memory transpose implementation.
//
// When a tile based shared memory transpose is used to implement an input with
// 0-2-1 transpose, we preload a tile of the input elements
// [z, y..y+31, x..x+31] to compute the output tile elements of the same
// indices. Preloading the input tile this way is only safe when the computation
// of the output tile elements do not need any input element outside the
// preloaded tile. We inspect all the transitive users of the input parameter
// up to the fusion root instruction to see if we can find any instruction
// that can make preloading the input tile unsafe.
std::vector<int64> FilterInputsForShmemTranspose(mlir::lmhlo::FusionOp fusion,
                                                 std::vector<int64> input_ids) {
  std::vector<mlir::Value> params = ToStdVector(fusion.getFusionParameters());

  std::vector<int64> filtered_input_ids;
  for (int64 input_id : input_ids) {
    mlir::Value input = params.at(input_id);
    if (IsInstructionSafeForShmemTranspose(input.getDefiningOp())) {
      filtered_input_ids.push_back(input_id);
    }
  }
  return filtered_input_ids;
}

}  // namespace

StatusOr<bool> IrEmitterUnnested::CheckAndEmitHloWithTile021(
    MlirEmitterInput input) {
  CHECK((mlir::isa<mlir::lmhlo::FusionOp, mlir::lmhlo::CopyOp>(input.op)));

  MlirEmitterContext context;
  context.SetOperation(input.op);

  // If the output_shape is reduced to 021 shape, find all the parameters of
  // the HLO that are in the corresponding 012 shape.
  std::vector<int64> params_012;
  optional<std::vector<int64>> reduced_dims_021;
  for (int64 operand_idx = 0; operand_idx < context.operand_shapes.size();
       ++operand_idx) {
    const Shape& operand_shape = context.operand_shapes[operand_idx];
    auto find_transpose_result =
        ShapeUtil::FindTranspose021(operand_shape, context.output_shapes[0]);
    if (!find_transpose_result.has_value()) {
      continue;
    }
    const std::vector<int64>& curr_reduced_dims_021 = *find_transpose_result;
    if (!reduced_dims_021.has_value()) {
      reduced_dims_021 = curr_reduced_dims_021;
    }
    if (!absl::c_equal(*reduced_dims_021, curr_reduced_dims_021)) {
      // There is more than one possible transpose. Instead of picking one
      // transpose, we simply give up here.
      return false;
    }
    params_012.push_back(operand_idx);
  }

  if (!reduced_dims_021.has_value()) {
    return false;
  }

  if ((*reduced_dims_021)[1] < kMinDimensionToTransposeTiled ||
      (*reduced_dims_021)[2] < kMinDimensionToTransposeTiled) {
    return false;
  }

  if (auto fusion_op = mlir::dyn_cast<mlir::lmhlo::FusionOp>(input.op)) {
    params_012 = FilterInputsForShmemTranspose(fusion_op, params_012);
    if (params_012.empty()) {
      return false;
    }
  }

  // Each of our shared memory tiles has 32*33 elements (so ~4kb, if the
  // elements are of size 4 bytes), and CUDA has an architectural limit of
  // 48kb shared memory per SM.  (This is increased to 96kb in Volta, but we
  // don't use this, in part because it eats into our L1 cache space.)
  //
  // For correctness we need to ensure that we don't make more than 48kb worth
  // of shmem tiles per block.  And for performance, we'd probably like to use
  // significantly less, so that we can fit more than one block at a time on a
  // gpu core.
  //
  // We say without benchmarks that we want at least 3 threads/block,
  // corresponding to 3 shmem tiles if the elements are 32 bits wide.  We
  // choose which params get the shmem transpose treatment arbitrarily; it's
  // not clear if there's a Right Choice.
  //
  // This is only sound if tiled transposes are the only place where we use
  // shared memory in fusions.  If in the future other fusible ops use shared
  // memory, we'll have to adjust this heuristic.
  constexpr int kMinBlocksPerCore = 3;
  constexpr int64 kShmemPerCore = 48 * 1024;
  int64 shmem_used = 0;
  for (int64 i = 0; i < params_012.size(); ++i) {
    const Shape& operand_shape = context.operand_shapes[params_012[i]];
    shmem_used +=
        32 * 33 *
        ShapeUtil::ByteSizeOfPrimitiveType(operand_shape.element_type());

    if (kMinBlocksPerCore * shmem_used > kShmemPerCore) {
      // Erase this element and everything after it from params_012.
      params_012.resize(i);
      break;
    }
  }

  if (params_012.empty()) {
    return false;
  }

  std::vector<llvm_ir::IrArray> ir_arrays;
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<KernelThunk> kernel_thunk,
      BuildKernelThunkForMlir(input.op, input.thunk_info, &ir_arrays));
  EmitHlo021Tile(
      input.op, kernel_thunk.get(), context,
      absl::MakeSpan(ir_arrays).subspan(0, context.operand_shapes.size()),
      absl::MakeSpan(ir_arrays).subspan(context.operand_shapes.size()),
      *reduced_dims_021, params_012);
  AddThunkToThunkSequence(std::move(kernel_thunk));
  return true;
}

namespace {

// Returns true if all the transitive users of hlo before hitting users in
// use_chain_endings are elementwise operations.
bool AreUsersElementwise(
    mlir::Value value,
    const absl::flat_hash_set<mlir::Operation*>& use_chain_endings) {
  return absl::c_all_of(value.getUsers(), [&](mlir::OpOperand use) {
    mlir::Operation* user = use.getOwner();
    CHECK_EQ(1, user->getNumResults());
    return use_chain_endings.count(user) ||
           (HloInstruction::IsOpElementwise(*MhloToHloOpcode(user)) &&
            AreUsersElementwise(user->getResult(0), use_chain_endings));
  });
}

// Returns the number of fusion inputs that have the same dimension as the
// given shape, and involve in only elementwise operations.
int64 NumInputsInvolveInOnlyElementwiseOps(
    mlir::lmhlo::FusionOp fusion, const Shape& op_shape,
    const absl::flat_hash_set<mlir::Operation*>& use_chain_endings) {
  return absl::c_count_if(
      fusion.getFusionParameters(), [&](mlir::Value parameter) {
        Shape parameter_shape = TypeToShape(parameter.getType());
        return ShapeUtil::SameDimensions(op_shape, parameter_shape) &&
               AreUsersElementwise(parameter, use_chain_endings);
      });
}

// Returns the number of fusion inputs that have more elements than the given
// shape.
int64 NumInputsWithMoreElementsThan(mlir::lmhlo::FusionOp fusion,
                                    const Shape& shape) {
  int64 num_elements = ShapeUtil::ElementsIn(shape);
  return absl::c_count_if(
      fusion.getFusionParameters(), [&](mlir::Value parameter) {
        Shape parameter_shape = TypeToShape(parameter.getType());
        return ShapeUtil::ElementsIn(parameter_shape) > num_elements;
      });
}

// The benefit of unrolling a kInput fusion that is a column reduction comes
// from the vectorization of non-reduction fusion outputs and fusion inputs.
// On the other hand, unrolling can also introduce factors that can cause
// the kernel to run slower. This routine uses a simple heuristic to estimate
// the benefit as well as the overhead of unrolling in order to decide whether
// unrolling is beneficial for the given kInput fusion.
bool IsUnrollingColumnReductionBeneficial(
    mlir::Operation* unnested_hlo, const Shape& input_shape,
    int64 num_kept_minor, const FusionLayoutAnalysis& layout_analysis) {
  // TODO(b/122468062): Need further investigate to see whether we can
  // remove the constraint on IsPowerOfTwo.
  if (!IsPowerOfTwo(static_cast<uint64>(num_kept_minor))) {
    return false;
  }

  if (IsReductionFromOrToContiguousDimensions(unnested_hlo, layout_analysis)) {
    return true;
  }

  auto fusion = mlir::cast<mlir::lmhlo::FusionOp>(unnested_hlo);
  int64 can_be_vectorized = 0;
  int64 cannot_be_vectorized = 0;
  auto fusion_results = ToStdVector(fusion.getFusionResults());
  absl::flat_hash_set<mlir::Operation*> use_chain_endings;
  if (fusion_results.size() == 1) {
    if (IsReductionFromOrToContiguousDimensions(
            fusion_results[0].getDefiningOp(), layout_analysis)) {
      use_chain_endings.insert(fusion_results[0].getDefiningOp());
      // Atomic.add of the reduction result can't be vectorized.
      cannot_be_vectorized++;
    }
  } else {
    for (mlir::Value result : fusion_results) {
      if (IsReductionFromOrToContiguousDimensions(result.getDefiningOp(),
                                                  layout_analysis)) {
        // Atomic.add of the reduction result can't be vectorized.
        cannot_be_vectorized++;
      } else {
        // Write of the non-reduction result can be vectorized.
        can_be_vectorized++;
      }
      use_chain_endings.insert(result.getDefiningOp());
    }
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

int64 NearestPowerOfTwo(int64 v) {
  if (v < 0) {
    return 0;
  }
  int64 upper = tensorflow::NextPowerOfTwo64(v);
  int64 lower = upper >> 1;
  return upper - v < v - lower ? upper : lower;
}

}  // namespace

ReductionCodegenInfo IrEmitterUnnested::ComputeReductionCodegenInfo(
    mlir::Operation* unnested_hlo, mlir::Operation* first_reduce,
    const FusionLayoutAnalysis& layout_analysis) {
  Shape input_shape = layout_analysis.GetShape(first_reduce->getOperand(0));
  ReductionDimensions reduction_dimensions =
      GetReductionKindAndContiguousComponents(first_reduce);
  VLOG(10) << "is_row_reduction " << reduction_dimensions.is_row_reduction
           << " " << reduction_dimensions.dimensions[0] << " "
           << reduction_dimensions.dimensions[1] << " "
           << reduction_dimensions.dimensions[2];
  auto get_dtype_bits = [](mlir::Value i) {
    // TODO(timshen): may not be efficient.
    return primitive_util::BitWidth(TypeToShape(i.getType()).element_type());
  };

  // For fusion with multiple inputs, use the smallest input dtype to
  // select the reduction_tiling.
  int smallest_input_dtype_bits = get_dtype_bits(first_reduce->getOperand(0));

  for (mlir::Value operand : GetHloOperands(unnested_hlo)) {
    smallest_input_dtype_bits =
        std::min(get_dtype_bits(operand), smallest_input_dtype_bits);
  }
  std::array<int64, 3> reduction_tiling =
      GetReductionTiling(reduction_dimensions, smallest_input_dtype_bits,
                         ir_emitter_context_->cuda_compute_capability());

  int64 num_threads_y = reduction_dimensions.is_row_reduction ? 1 : kWarpSize;
  int64 num_threads_x = [&] {
    if (reduction_dimensions.is_row_reduction) {
      // Use 512 as default block size (threads per block) for row reductions.
      // For multi-output fusions, reduce the block size further to decrease
      // register pressure when multiple outputs are computed by each thread.
      int64 fan_out = 1;
      if (auto fusion = mlir::dyn_cast<mlir::lmhlo::FusionOp>(unnested_hlo)) {
        fan_out = fusion.getFusionResults().size();
      }

      // 64 is the general advice as the smallest block sizes.
      // Moreover, XLA:GPU emitters need at least 32 threads at some places.
      int64 max_block_size = std::max(64LL, 512LL / NearestPowerOfTwo(fan_out));
      return std::min(
          max_block_size,
          RoundUpToNearest(CeilOfRatio(reduction_dimensions.dimensions[2],
                                       reduction_tiling[2]),
                           kWarpSize));
    }
    return kWarpSize;
  }();

  bool tile_fit = reduction_dimensions.dimensions[kDimX] %
                      (reduction_tiling[2] * num_threads_x) ==
                  0;

  int cc_major = 0;
  if (ir_emitter_context_->cuda_compute_capability()) {
    cc_major = ir_emitter_context_->cuda_compute_capability()->cc_major;
  }

  int num_partial_results = 1;
  KernelMappingScheme::IndexingOrder indexing_order = [&]() {
    if (reduction_dimensions.is_row_reduction &&
        // P100, only try to vectorize+coales memory access when the
        // tile size fits exactly and dtypes <= 32 bits
        ((cc_major == 6 && smallest_input_dtype_bits <= 32 && tile_fit) ||
         // On V100, only try to vectorize+coales memory access for
         // rows of even size.  For odd row sizes, every other row
         // isn't aligned, so it can't be vectorized.
         (cc_major >= 7 && reduction_dimensions.dimensions[2] % 2 == 0))) {
      return kStridedLinearIndexingX;
    } else if (!reduction_dimensions.is_row_reduction &&
               IsUnrollingColumnReductionBeneficial(
                   unnested_hlo, input_shape,
                   reduction_dimensions.dimensions[2], layout_analysis)) {
      num_partial_results = 2;
      reduction_tiling[2] *= num_partial_results;
      return kLinearIndexingX;
    } else {
      return kStridedIndexingX;
    }
  }();

  int vector_size = 1;
  if (indexing_order == kStridedLinearIndexingX) {
    // Assuming XLA will perform the unrolling and LLVM will vectorize,
    // disable the unroll for the cases that LLVM doesn't vectorize.
    if (reduction_dimensions.dimensions[2] % 2 == 0 &&
        !MayPreventVectorization(unnested_hlo)) {
      vector_size = 2;
    } else {
      indexing_order = kStridedIndexingX;
    }
  }
  KernelMappingScheme mapping_scheme(
      reduction_dimensions.dimensions,
      {reduction_tiling[0], reduction_tiling[1] * num_threads_y,
       reduction_tiling[2] * num_threads_x},
      num_threads_y, num_threads_x, indexing_order, vector_size);
  return ReductionCodegenInfo(mapping_scheme, num_partial_results,
                              reduction_dimensions.is_row_reduction);
}

void IrEmitterUnnested::EmitIRForReduction(
    mlir::Operation* unnested_hlo, absl::Span<const int> instr_index_group,
    HloComputation* fused_computation, FusedIrEmitter* fused_emitter,
    absl::Span<const llvm_ir::IrArray> operand_ir_arrays,
    absl::Span<const llvm_ir::IrArray> result_ir_arrays,
    ReductionCodegenInfo* reduction_info, const Shape& input_shape,
    const FusionLayoutAnalysis& layout_analysis) {
  std::vector<HloComputation*> reducers;
  for (auto index : instr_index_group) {
    auto reduce = GetReduceFromUnnestedMlir(unnested_hlo, index);
    if (!IsReductionFromOrToContiguousDimensions(reduce, layout_analysis)) {
      continue;
    }
    if (auto unnested_reduce = mlir::dyn_cast<mlir::lmhlo::ReduceOp>(reduce)) {
      reducers.push_back(
          *GetOrCreateSubComputationFromRegion(&unnested_reduce.body(),
                                               /*is_fusion=*/false));
    } else if (auto nested_reduce =
                   mlir::dyn_cast<mlir::mhlo::ReduceOp>(reduce)) {
      HloInstruction* root = fused_computation->root_instruction();
      if (root->opcode() == HloOpcode::kTuple) {
        root = root->mutable_operand(index);
      } else {
        CHECK_EQ(0, index);
      }
      reducers.push_back(root->to_apply());
    } else {
      LOG(FATAL) << "Unexpected reduce op: " << MlirToString(reduce);
    }
  }
  CHECK(!reducers.empty()) << " expect at least one reduce instructions.";

  const KernelMappingScheme& mapping_scheme =
      reduction_info->GetKernelMappingScheme();
  LaunchDimensions launch_dimensions(mapping_scheme.GetNumberOfBlocks(),
                                     mapping_scheme.GetThreadsPerBlock());
  llvm::Type* index_ty = GetIndexTypeForKernelFromMlir(
      unnested_hlo, launch_dimensions.launch_bound(), &b_);
  EmitPrologueForReduction(unnested_hlo, instr_index_group, fused_computation,
                           fused_emitter, operand_ir_arrays, result_ir_arrays,
                           reduction_info, layout_analysis);

  EmitElementFunction emit_reduction_tile =
      [&](const llvm_ir::IrArray::Index& index, llvm::Value* y_loc,
          llvm::Value* x_loc, int64 x_iter_num) {
        EmitTileElementForReduction(
            unnested_hlo, input_shape, instr_index_group, fused_computation,
            fused_emitter, operand_ir_arrays, result_ir_arrays, reducers, index,
            *reduction_info, x_iter_num, layout_analysis);
      };

  TilingKernelInfo tiling_kernel_info = EmitTilingKernel(
      mapping_scheme, index_ty,
      [&](const ThreadIdInfo& thread_id_info, const IrArray::Index& index,
          const string& loop_name, llvm::Value* tile_height,
          llvm::Value* tile_width, KernelSupportLibrary* ksl) {
        EmitTile(reduction_info->GetKernelMappingScheme(), index, loop_name,
                 ksl, thread_id_info, tile_height, tile_width,
                 emit_reduction_tile);
      });
  EmitEpilogueForReduction(index_ty, unnested_hlo, instr_index_group,
                           result_ir_arrays, reducers, *reduction_info,
                           tiling_kernel_info, layout_analysis);
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
std::vector<std::vector<int>> DivideOutputInstructionsIntoGroups(
    HloComputation* fused_computation, int num_reduces) {
  CHECK_NE(0, num_reduces);
  if (num_reduces == 1) {
    return {{0}};
  }

  std::vector<tensorflow::UnionFind<HloInstruction*>> disjoint_sets(
      num_reduces);
  for (size_t i = 0; i < num_reduces; ++i) {
    disjoint_sets[i].Get() =
        fused_computation->root_instruction()->mutable_operand(i);
  }

  std::unique_ptr<HloReachabilityMap> reachability_map =
      HloReachabilityMap::Build(fused_computation);
  for (auto* instr : fused_computation->instructions()) {
    std::vector<int64> reached_output_ids;
    for (size_t oid = 0; oid < num_reduces; ++oid) {
      auto reduce = fused_computation->root_instruction()->mutable_operand(oid);
      if (HloOpcode::kReduce == reduce->opcode() &&
          (IsBroadcastedConstantOrScalar(*instr))) {
        // Do not group output reduce instructions through broadcasted
        // constants or scalars, as the recomputation should be acceptable.
        VLOG(3) << "Skip broadcasted constant or scalar " << instr->ToString();
        continue;
      }
      // Now group output instructions if they have common predecessors.
      if (reachability_map->IsReachable(instr, reduce)) {
        VLOG(3) << "Reaching " << reduce->ToString() << " from "
                << instr->ToString();
        reached_output_ids.push_back(oid);
      }
    }
    for (size_t j = 1; j < reached_output_ids.size(); ++j) {
      disjoint_sets[reached_output_ids[0]].Merge(
          &disjoint_sets[reached_output_ids[j]]);
    }
  }
  // Place output instructions in the same set into the same group.
  HloInstructionMap<std::vector<int>> groups;
  for (size_t oid = 0; oid < num_reduces; ++oid) {
    groups[disjoint_sets[oid].Get()].push_back(oid);
  }

  std::vector<std::vector<int>> ret;
  absl::c_for_each(
      groups, [&](auto& iter) { ret.emplace_back(std::move(iter.second)); });
  return ret;
}

}  // namespace

Status IrEmitterUnnested::EmitReductionFromOrToContiguousDimensions(
    MlirEmitterInput mlir_input, const FusionLayoutAnalysis& layout_analysis) {
  mlir::Operation* unnested_hlo = mlir_input.op;
  auto fusion = mlir::dyn_cast<mlir::lmhlo::FusionOp>(unnested_hlo);

  int num_reduces = 1;
  if (fusion) {
    num_reduces = fusion.getFusionResults().size();
  }

  bool returns_tuple = num_reduces > 1;
  VLOG(10) << "Emitting reduction to vector " << MlirToString(unnested_hlo);

  // Build an initializer thunk to initialize each reduction output.
  ThunkSequence thunks;
  for (int i = 0; i < num_reduces; ++i) {
    mlir::Operation* output_instruction =
        GetReduceFromUnnestedMlir(unnested_hlo, i);
    if (!IsReductionFromOrToContiguousDimensions(output_instruction,
                                                 layout_analysis)) {
      continue;
    }

    if (fusion) {
      TF_ASSIGN_OR_RETURN(std::unique_ptr<Thunk> initializer_thunk,
                          BuildFusedInitializerThunkForMlir(fusion, i));
      thunks.push_back(std::move(initializer_thunk));
    } else {
      auto reduce = mlir::cast<mlir::lmhlo::ReduceOp>(output_instruction);

      TF_RET_CHECK(!returns_tuple);
      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<Thunk> initializer_thunk,
          BuildInitializerThunkForMlir(reduce, reduce.init_values()[0],
                                       reduce.out()[0]));
      thunks.push_back(std::move(initializer_thunk));
    }
  }

  // Build a kernel thunk to compute all the outputs.
  mlir::Operation* first_reduce = nullptr;
  for (int i = 0; i < num_reduces; ++i) {
    mlir::Operation* output_instruction =
        GetReduceFromUnnestedMlir(unnested_hlo, i);
    if (IsReductionFromOrToContiguousDimensions(output_instruction,
                                                layout_analysis)) {
      first_reduce = GetReduceFromUnnestedMlir(unnested_hlo, i);
      break;
    }
  }
  CHECK(first_reduce) << MlirToString(unnested_hlo);
  if (num_reduces > 1) {
    for (int i = 0; i < num_reduces; i++) {
      auto candidate = mlir::dyn_cast<mlir::mhlo::ReduceOp>(
          GetReduceFromUnnestedMlir(unnested_hlo, i));
      if (candidate &&
          !IsFusedReductionOutputConsistent(
              candidate, mlir::cast<mlir::mhlo::ReduceOp>(first_reduce),
              layout_analysis)) {
        return InternalError("Inconsistent reduction fusion outputs");
      }
    }
  }
  Shape input_shape = TypeToShape(first_reduce->getOperand(0).getType());
  // The layout of a reduction input is either set by LayoutAssignment for
  // unnested kReduce or by InstructionFusion for fused kReduce.
  CHECK(input_shape.has_layout()) << "LayoutAssignment or InstructionFusion "
                                     "doesn't set the input layout of "
                                  << MlirToString(first_reduce);

  std::vector<llvm_ir::IrArray> ir_arrays;
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<KernelThunk> kernel_thunk,
      BuildKernelThunkForMlir(unnested_hlo, Thunk::ThunkInfo(), &ir_arrays));

  HloComputation* fused_computation = nullptr;
  if (fusion) {
    TF_ASSIGN_OR_RETURN(fused_computation, GetOrCreateSubComputationFromRegion(
                                               &fusion.region(),
                                               /*is_fusion=*/true));
  }

  // Group output instructions. Each group will be executed in parallel.
  std::vector<std::vector<int>> instr_index_groups =
      DivideOutputInstructionsIntoGroups(fused_computation, num_reduces);

  VLOG(2) << StrCat("Generate in ", instr_index_groups.size(), " groups for ",
                    MlirToString(unnested_hlo));

  absl::optional<GpuElementalIrEmitter> elemental_emitter;
  absl::optional<FusedIrEmitter> optional_fused_emitter;
  FusedIrEmitter* fused_emitter = nullptr;

  absl::Span<const llvm_ir::IrArray> operand_ir_arrays;
  absl::Span<const llvm_ir::IrArray> result_ir_arrays;
  if (fusion) {
    elemental_emitter.emplace(hlo_module_config_,
                              ir_emitter_context_->llvm_module(), &b_,
                              GetNestedComputer());
    optional_fused_emitter.emplace(&*elemental_emitter);
    fused_emitter = &*optional_fused_emitter;

    CHECK_LT(fused_computation->num_parameters(), ir_arrays.size());
    for (int i = 0; i < fused_computation->num_parameters(); i++) {
      auto ir_array = ir_arrays[i];
      auto fused_operand = fused_computation->parameter_instruction(i);
      fused_emitter->BindGenerator(
          fused_operand, [this, ir_array,
                          fused_operand](const llvm_ir::IrArray::Index& index) {
            return ir_array.EmitReadArrayElement(index, &b_,
                                                 fused_operand->name());
          });
    }
    result_ir_arrays = absl::MakeSpan(ir_arrays).subspan(
        fused_computation->num_parameters(), num_reduces);
  } else {
    CHECK_EQ(3, ir_arrays.size());
    operand_ir_arrays = absl::MakeSpan(ir_arrays).subspan(0, 2);
    result_ir_arrays = absl::MakeSpan(ir_arrays).subspan(2);
  }

  KernelSupportLibrary ksl(&b_, llvm_ir::UnrollMode::kDefaultUnroll);
  for (size_t i = 0; i < instr_index_groups.size(); ++i) {
    // Create a new ReductionCodegenInfo instance as it contains states for
    // code generation per reduction group. For now, let's always use the very
    // first reduce as representative to construct ReductionCodegenInfo, since
    // all the reductions are required to have the same shape and layout as
    // verified by `IsFusedReductionOutputConsistent()`. We can loosen the
    // constraint later when the needs arise.
    ReductionCodegenInfo reduction_info = ComputeReductionCodegenInfo(
        unnested_hlo, first_reduce, layout_analysis);
    auto emit_reduction_func = [&] {
      EmitIRForReduction(unnested_hlo, instr_index_groups[i], fused_computation,
                         fused_emitter, operand_ir_arrays, result_ir_arrays,
                         &reduction_info, input_shape, layout_analysis);
    };
    // Use raw block_id_y to select the i-th parallel reduction to run. Using
    // block_id_y instead of block_id_x simplifies the index calculation
    // for reduction code generation as the block_id_y is orthogonal to
    // the indices used within the reductions.
    llvm::CallInst* raw_block_id_y = gpu::EmitCallToTargetIntrinsic(
        gpu::TargetIntrinsicID::kBlockIdy, {}, {}, &b_);
    llvm_ir::AddRangeMetadata(0, instr_index_groups.size(),
                              llvm::cast<llvm::Instruction>(raw_block_id_y));
    llvm::Value* guarding_cond =
        b_.CreateICmpEQ(raw_block_id_y, b_.getInt32(i));
    ksl.If(StrCat("reduce-group-", i), guarding_cond, emit_reduction_func);
  }
  ReductionCodegenInfo reduction_info =
      ComputeReductionCodegenInfo(unnested_hlo, first_reduce, layout_analysis);
  const KernelMappingScheme& mapping_scheme =
      reduction_info.GetKernelMappingScheme();
  // block_y_count is set to instr_index_groups.size(), so that each reduction
  // group can be run in parallel by a different BlockIdy.
  LaunchDimensions launch_dimensions(
      {/*x=*/mapping_scheme.GetNumberOfBlocks(),
       /*y=*/static_cast<int64>(instr_index_groups.size()),
       /*z=*/1},
      {/*x=*/mapping_scheme.GetThreadsPerBlock(), /*y=*/1, /*z=*/1});
  VLOG(3) << "Launch dimensions of "
          << mlir::GetNameFromLoc(unnested_hlo->getLoc())
          << ": number of blocks: " << mapping_scheme.GetNumberOfBlocks()
          << " - threads per block: " << mapping_scheme.GetThreadsPerBlock();
  UpdateLaunchDimensions(launch_dimensions, kernel_thunk.get(),
                         ir_emitter_context_->llvm_module());

  thunks.push_back(std::move(kernel_thunk));
  std::unique_ptr<SequentialThunk> sequential_thunk =
      absl::make_unique<SequentialThunk>(mlir_input.thunk_info,
                                         std::move(thunks));
  AddThunkToThunkSequence(std::move(sequential_thunk));

  return Status::OK();
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
  GpuElementalIrEmitter elem_emitter(hlo_module_config_, module_, &b_,
                                     GetNestedComputer());
  FusedIrEmitter fused_emitter(&elem_emitter);
  for (int i = 0; i < fused_computation->num_parameters(); i++) {
    fused_emitter.BindGenerator(
        fused_computation->parameter_instruction(i),
        [this, &ir_arrays, i](llvm_ir::IrArray::Index index) {
          return ir_arrays[i].EmitReadArrayElement(index, &b_);
        });
  }
  for (const HloInstruction* slice : slice_instructions) {
    auto input_generator = *fused_emitter.GetGenerator(slice->operand(0));
    input_ir_values.push_back(input_generator(index).ValueOrDie());
  }

  // Emit for slice_instructions.
  KernelSupportLibrary ksl(&b_, llvm_ir::UnrollMode::kDefaultUnroll);
  for (int64 i = 0; i < slice_instructions.size(); ++i) {
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
  return Status::OK();
}

Status IrEmitterUnnested::EmitInputFusibleNonStridedSlices(
    MlirEmitterInput mlir_input) {
  auto fusion = mlir::cast<mlir::lmhlo::FusionOp>(mlir_input.op);

  constexpr int unroll_factor = 1;

  std::vector<llvm_ir::IrArray> ir_arrays;
  TF_ASSIGN_OR_RETURN(
      auto kernel_thunk,
      BuildKernelThunkForMlir(fusion, mlir_input.thunk_info, &ir_arrays));

  TF_ASSIGN_OR_RETURN(const HloComputation* fused_computation,
                      GetOrCreateSubComputationFromRegion(&fusion.region(),
                                                          /*is_fusion=*/true));

  TF_ASSIGN_OR_RETURN(Shape element_shape,
                      GetConsistentInputShapeForRootSlices(fused_computation));
  TF_ASSIGN_OR_RETURN(LaunchDimensions launch_dimensions,
                      CalculateLaunchDimensions(
                          element_shape, ir_emitter_context_->gpu_device_info(),
                          {unroll_factor}));
  UpdateLaunchDimensions(launch_dimensions, kernel_thunk.get(),
                         ir_emitter_context_->llvm_module());

  Status emit_status =
      ParallelLoopEmitter(
          [&](const llvm_ir::IrArray::Index index) -> Status {
            return EmitElementForInputFusibleSlices(fused_computation,
                                                    ir_arrays, index);
          },
          element_shape, launch_dimensions, &b_)
          .EmitLoop(IrName(mlir::GetNameFromLoc(fusion.getLoc())),
                    GetIndexTypeForKernelFromMlir(
                        fusion, launch_dimensions.launch_bound(), &b_));

  thunk_sequence_.emplace_back(std::move(kernel_thunk));

  return emit_status;
}

Thunk::ThunkInfo IrEmitterUnnested::GetThunkInfo(
    const HloInstruction* hlo) const {
  CHECK(hlo);
  Thunk::ThunkInfo info;
  info.profile_annotation = absl::StrFormat(
      "Thunk:#hlo_op=%s,hlo_module=%s#", hlo->name(), hlo->GetModule()->name());
  if (const auto* index_map = ir_emitter_context_->profile_index_map()) {
    info.profile_index.emplace(
        static_cast<int64>(index_map->GetProfileIndexFor(*hlo)));
  }
  return info;
}

Status IrEmitterUnnested::EmitOp(MlirEmitterInput mlir_input) {
  if (mlir::isa<mlir::ConstantOp, mlir::memref::ViewOp,
                mlir::memref::ReinterpretCastOp, mlir::ReturnOp,
                mlir::lmhlo::TerminatorOp>(mlir_input.op)) {
    return Status::OK();
  }

  if (mlir::isa<mlir::memref::GetGlobalOp>(mlir_input.op)) {
    return EmitConstant(mlir_input);
  }

  if (auto call = mlir::dyn_cast<mlir::lmhlo::CustomCallOp>(mlir_input.op)) {
    if (call.call_target_name() == "PadToStatic") {
      return EmitPadToStaticFromMlir(mlir_input);
    }
    if (call.call_target_name() == "SliceToDynamic") {
      return EmitSliceToDynamicFromMlir(mlir_input);
    }
    return EmitCustomCallThunkFromMlir(mlir_input);
  }

  if (mlir::isa<mlir::lmhlo_gpu::GEMMOp, mlir::lmhlo_gpu::GEMM_BiasOp>(
          mlir_input.op)) {
    return EmitGemmThunkFromMlir(mlir_input);
  }

  if (mlir::isa<mlir::lmhlo_gpu::ConvForwardOp,
                mlir::lmhlo_gpu::ConvForwardFusedOp,
                mlir::lmhlo_gpu::ConvForwardFusedSideInputOp,
                mlir::lmhlo_gpu::ConvBackwardFilterOp,
                mlir::lmhlo_gpu::ConvBackwardInputOp>(mlir_input.op)) {
    return EmitConvolutionThunkFromMlir(mlir_input);
  }

  if (mlir::isa<mlir::lmhlo_gpu::BatchNormTrainingOp,
                mlir::lmhlo_gpu::BatchNormInferenceOp,
                mlir::lmhlo_gpu::BatchNormGradOp>(mlir_input.op)) {
    return EmitBatchNormThunkFromMlir(mlir_input);
  }

#if GOOGLE_CUDA
  if (mlir::isa<mlir::lmhlo_gpu::CholeskyOp>(mlir_input.op)) {
    return EmitCholeskyThunkFromMlir(mlir_input);
  }
#endif  // GOOGLE_CUDA

  if (mlir::isa<mlir::lmhlo::FftOp>(mlir_input.op)) {
    return EmitFftThunkFromMlir(mlir_input);
  }

  if (mlir::isa<mlir::lmhlo::TriangularSolveOp>(mlir_input.op)) {
    return EmitTriangularSolveFromMlir(mlir_input);
  }

  if (mlir::isa<mlir::lmhlo::FusionOp>(mlir_input.op)) {
    return EmitFusionFromMlir(mlir_input);
  }

  if (mlir::isa<mlir::lmhlo::CopyOp>(mlir_input.op)) {
    return EmitCopyFromMlir(mlir_input);
  }

  if (mlir::isa<mlir::lmhlo::ReduceOp>(mlir_input.op)) {
    return EmitReduceFromMlir(mlir_input);
  }

  if (mlir::isa<mlir::lmhlo::SelectAndScatterOp>(mlir_input.op)) {
    return EmitSelectAndScatterFromMlir(mlir_input);
  }

  if (mlir::isa<mlir::lmhlo::RngGetAndUpdateStateOp>(mlir_input.op)) {
    return EmitRngGetAndUpdateState(mlir_input);
  }

  if (mlir::isa<mlir::lmhlo::ScatterOp>(mlir_input.op)) {
    return EmitScatterFromMlir(mlir_input);
  }

  if (mlir::isa<mlir::lmhlo::SortOp>(mlir_input.op)) {
    return EmitSortFromMlir(mlir_input);
  }

  if (mlir::isa<mlir::lmhlo::ReplicaIdOp>(mlir_input.op)) {
    return EmitReplicaOrPartitionIdFromMlir<ReplicaIdThunk,
                                            mlir::lmhlo::ReplicaIdOp>(
        mlir_input);
  }

  if (mlir::isa<mlir::lmhlo::PartitionIdOp>(mlir_input.op)) {
    return EmitReplicaOrPartitionIdFromMlir<PartitionIdThunk,
                                            mlir::lmhlo::PartitionIdOp>(
        mlir_input);
  }

  if (mlir::isa<mlir::lmhlo::CollectivePermuteOp>(mlir_input.op)) {
    return EmitCollectivePermuteFromMlir(mlir_input);
  }

  if (mlir::isa<mlir::lmhlo::AllGatherOp>(mlir_input.op)) {
    return EmitNcclThunkFromMlir<NcclAllGatherThunk, mlir::lmhlo::AllGatherOp>(
        mlir_input);
  }

  if (mlir::isa<mlir::lmhlo::AllReduceOp>(mlir_input.op)) {
    return EmitNcclThunkFromMlir<NcclAllReduceThunk, mlir::lmhlo::AllReduceOp>(
        mlir_input);
  }

  if (mlir::isa<mlir::lmhlo::AllToAllOp>(mlir_input.op)) {
    return EmitNcclThunkFromMlir<NcclAllToAllThunk, mlir::lmhlo::AllToAllOp>(
        mlir_input);
  }

  if (mlir::isa<mlir::lmhlo::InfeedOp>(mlir_input.op)) {
    return EmitInfeedFromMlir(mlir_input);
  }

  if (mlir::isa<mlir::lmhlo::OutfeedOp>(mlir_input.op)) {
    return EmitOutfeedFromMlir(mlir_input);
  }

  if (mlir::isa<mlir::lmhlo::CaseOp>(mlir_input.op)) {
    return EmitConditionalFromMlir(mlir_input);
  }

  if (mlir::isa<mlir::lmhlo::WhileOp>(mlir_input.op)) {
    return EmitWhileFromMlir(mlir_input);
  }

  return EmitUsingElementalIrEmitter(mlir_input);
}

Status IrEmitterUnnested::EmitLmhloRegion(mlir::Region* region) {
  Thunk::ThunkInfo thunk_info;
  auto module = region->getParentOfType<mlir::ModuleOp>();
  std::string module_name = mlir::GetNameFromLoc(module->getLoc());
  for (mlir::Operation& op : llvm::make_early_inc_range(region->front())) {
    thunk_info.profile_annotation =
        absl::StrFormat("Thunk:#hlo_op=%s,hlo_module=%s#",
                        mlir::GetNameFromLoc(op.getLoc()), module_name);
    TF_RETURN_IF_ERROR(EmitOp(MlirEmitterInput{&op, thunk_info}));
  }
  return Status::OK();
}

void MlirEmitterContext::SetOperation(mlir::Operation* op) {
  this->name = mlir::GetNameFromLoc(op->getLoc());

  auto operands = GetHloOperands(op);
  auto outputs = GetHloOutputs(op);
  for (auto operand : operands) {
    operand_shapes.push_back(TypeToShape(operand.getType()));
  }
  for (auto output : outputs) {
    output_shapes.push_back(TypeToShape(output.getType()));
  }
}

}  // namespace gpu
}  // namespace xla
