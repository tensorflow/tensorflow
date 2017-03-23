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

#include <memory>
#include <string>
#include <vector>

#include "external/llvm/include/llvm/ADT/StringRef.h"
#include "external/llvm/include/llvm/IR/BasicBlock.h"
#include "external/llvm/include/llvm/IR/Function.h"
#include "external/llvm/include/llvm/IR/IRBuilder.h"
#include "external/llvm/include/llvm/IR/Instructions.h"
#include "external/llvm/include/llvm/IR/LLVMContext.h"
#include "external/llvm/include/llvm/IR/Module.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor.h"
#include "tensorflow/compiler/xla/service/gpu/convolution_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/copy_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/for_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/gemm_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/hlo_to_ir_bindings.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emitter.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emitter_context.h"
#include "tensorflow/compiler/xla/service/gpu/kernel_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/parallel_loop_emitter.h"
#include "tensorflow/compiler/xla/service/gpu/partition_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/sequential_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/compiler/xla/service/gpu/tuple_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/while_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/while_transformer.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/llvm_ir/fused_ir_emitter.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ops.h"
#include "tensorflow/compiler/xla/service/name_uniquer.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace gpu {

namespace {

// If a dimensions is smaller than this, untiled transposition may be more
// efficient.
const int64 kMinDimensionToTransposeTiled = 16;

// Returns true if all paths from `hlo` to `root` contain only tuples. The
// result of such an HloInstruction does not need to be materialized, when the
// computation can have a hybrid result.
bool ReachRootViaOnlyTuples(const HloInstruction& hlo,
                            const HloInstruction& root) {
  if (hlo.opcode() != HloOpcode::kTuple) {
    return false;
  }

  if (&hlo == &root) {
    return true;
  }

  for (HloInstruction* user : hlo.users()) {
    if (!ReachRootViaOnlyTuples(*user, root)) {
      return false;
    }
  }

  return true;
}

// If `hlo` is a Transpose, returns its operand; otherwise returns `hlo` itself.
const HloInstruction* StripTranspose(const HloInstruction& hlo) {
  if (hlo.IsRank2Transpose()) {
    return hlo.operand(0);
  }
  return &hlo;
}

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
      launch_dims.threads_per_block());
  nvvm_annotations_node->addOperand(llvm::MDNode::get(
      llvm_context,
      {llvm::ConstantAsMetadata::get(ir_kernel),
       llvm::MDString::get(llvm_context, "maxntidx"),
       llvm::ConstantAsMetadata::get(threads_per_block_ir_value)}));
}
}  // namespace

IrEmitterUnnested::IrEmitterUnnested(const HloModuleConfig& hlo_module_config,
                                     const HloComputation* hlo_computation,
                                     bool has_hybrid_result,
                                     IrEmitterContext* ir_emitter_context)
    : IrEmitter(hlo_module_config, ir_emitter_context, /*is_nested=*/false),
      hlo_computation_(hlo_computation),
      has_hybrid_result_(has_hybrid_result) {
  // Initialize thunk_sequence_ to an empty list of thunks.
  thunk_sequence_.reset(new ThunkSequence());
}

Status IrEmitterUnnested::Postprocess(HloInstruction* hlo) {
  bindings_.UnbindAllLocalIrValues();
  return DfsHloVisitor::Postprocess(hlo);
}

namespace {
bool ImplementedAsMemcpy(const HloInstruction& hlo) {
  // `hlo` needs to satisfy three conditions to be implemented as a
  // host-to-device cuMemcpy.
  //
  // 1. `hlo` is a kCopy instruction.
  // 2. `hlo`'s only operand is a kConstant instruction.
  // 3. `hlo` and its operand have the same shape (thus the same layout too).
  return hlo.opcode() == HloOpcode::kCopy &&
         hlo.operand(0)->opcode() == HloOpcode::kConstant &&
         ShapeUtil::Equal(hlo.operand(0)->shape(), hlo.shape());
}
}  // namespace

llvm::Function* IrEmitterUnnested::BuildKernelPrototype(
    const HloInstruction& inst,
    tensorflow::gtl::ArraySlice<const HloInstruction*> escaped_hlos) {
  // Compute the kernel name. The opcode string may contain "-" which cannot be
  // in a PTX function name, so sanitize the name before uniquifying it.
  string kernel_name = ir_emitter_context_->name_uniquer()->GetUniqueName(
      llvm_ir::SanitizeIrName(inst.name()));

  // Create the kernel and adds it to the module.
  llvm::Module* module = ir_emitter_context_->llvm_module();
  llvm::LLVMContext& context = module->getContext();
  int num_escaped_hlos = escaped_hlos.size();
  llvm::FunctionType* kernel_type = llvm::FunctionType::get(
      llvm::Type::getVoidTy(context),  // The type of function result.
      std::vector<llvm::Type*>(num_escaped_hlos + 1,
                               ir_builder_.getInt8PtrTy()),
      false);  // Not a variadic argument function.
  llvm::Function* kernel =
      llvm::Function::Create(kernel_type, llvm::GlobalValue::ExternalLinkage,
                             kernel_name.c_str(), module);

  // Add dereferenceable information to each of the escaped HLO parameters.
  for (size_t arg_no = 0; arg_no < escaped_hlos.size(); ++arg_no) {
    const HloInstruction* escaped_hlo = escaped_hlos[arg_no];
    const Shape& escaped_hlo_shape = escaped_hlo->shape();
    int64 escaped_hlo_size = llvm_ir::ByteSizeOf(
        escaped_hlo_shape, ir_emitter_context_->llvm_module()->getDataLayout());
    kernel->addDereferenceableAttr(arg_no + 1, escaped_hlo_size);
  }

  // The last argument is a pointer to the temporary buffer memory block.
  // We know that it doesn't alias any of the escaped arguments (the inputs +
  // the result).  We also know how many bytes can be dereferenced in it.
  const llvm::Argument& temp_buffer = *std::prev(kernel->arg_end());
  int64 temp_buffer_arg_no = temp_buffer.getArgNo();
  if (const BufferAllocation* allocation =
          ir_emitter_context_->buffer_assignment().GetTempAllocation()) {
    kernel->addDereferenceableAttr(temp_buffer_arg_no + 1, allocation->size());
  }
  kernel->setDoesNotAlias(temp_buffer_arg_no + 1);

  // Add the declaration of this kernel to llvm.nvvm.annotations so that NVPTX
  // treats it as a CUDA kernel.
  llvm::NamedMDNode* nvvm_annotations_node =
      module->getOrInsertNamedMetadata("nvvm.annotations");
  nvvm_annotations_node->addOperand(llvm::MDNode::get(
      context, {llvm::ConstantAsMetadata::get(kernel),
                llvm::MDString::get(context, "kernel"),
                llvm::ConstantAsMetadata::get(ir_builder_.getInt32(1))}));

  // Update the insert point to the entry basic block.
  llvm::BasicBlock* entry_bb =
      llvm::BasicBlock::Create(context,
                               "entry",  // The name of the basic block.
                               kernel);  // The parent/owner of "entry_bb".
  // Emit a "return void" at entry_bb's end, and sets the insert point before
  // that return instruction.
  ir_builder_.SetInsertPoint(llvm::ReturnInst::Create(context, entry_bb));

  return kernel;
}

Status IrEmitterUnnested::DefaultAction(HloInstruction* hlo) {
  thunk_sequence_->emplace_back(BuildKernelThunk(hlo));
  return IrEmitter::DefaultAction(hlo);
}

Status IrEmitterUnnested::HandleDot(HloInstruction* dot,
                                    HloInstruction* lhs_instruction,
                                    HloInstruction* rhs_instruction) {
  if (ImplementedAsGemm(*dot)) {
    thunk_sequence_->emplace_back(BuildGemmThunk(dot));
    return Status::OK();
  }
  thunk_sequence_->emplace_back(BuildKernelThunk(dot));
  return IrEmitter::HandleDot(dot, lhs_instruction, rhs_instruction);
}

Status IrEmitterUnnested::HandleConvolution(HloInstruction* convolution,
                                            HloInstruction* lhs_instruction,
                                            HloInstruction* rhs_instruction,
                                            const Window& window) {
  if (ImplementedAsDnnConvolution(*convolution)) {
    thunk_sequence_->emplace_back(BuildConvolutionThunk(convolution));
    return Status::OK();
  }
  thunk_sequence_->emplace_back(BuildKernelThunk(convolution));
  return IrEmitter::HandleConvolution(convolution, lhs_instruction,
                                      rhs_instruction, window);
}

namespace {

// Returns the first non-GetTupleElement ancestor instruction of 'hlo'.
// If the first non-GTE ancestor is tuple-shaped, populates 'index' with the
// (possibly nested) tuple indices used on the path from ancestor to 'hlo'.
const HloInstruction* LatestNonGteAncestorAndIndex(const HloInstruction* hlo,
                                                   ShapeIndex* index) {
  if (hlo->opcode() == HloOpcode::kGetTupleElement) {
    const auto* operand = LatestNonGteAncestorAndIndex(hlo->operand(0), index);
    index->push_back(hlo->tuple_index());
    return operand;
  }
  return hlo;
}

// Checks if we can emit code for DynamicUpdateSlice to update data in-place.
// Returns true if operand 0 of DynamicUpdateSlice and its output buffer
// share the same buffer allocation.
// Returns false otherwise.
bool CanUpdateDynamicSliceInPlace(const BufferAssignment& assignment,
                                  HloInstruction* fusion) {
  CHECK_EQ(HloOpcode::kFusion, fusion->opcode());
  HloInstruction* fused_root = fusion->fused_expression_root();
  if (fused_root->opcode() != HloOpcode::kDynamicUpdateSlice) {
    return false;
  }
  // Walk DynamicUpdateSlice operand(0) to fused parameter and get its
  // associated operand. See if it shares an allocation with this operand.
  ShapeIndex index;
  auto* fusion_operand =
      LatestNonGteAncestorAndIndex(fused_root->operand(0), &index);
  if (fusion_operand->opcode() != HloOpcode::kParameter) {
    return false;
  }
  auto* operand = fusion->operand(fusion_operand->parameter_number());

  BufferAllocation::Slice operand_slice =
      assignment.GetUniqueSlice(operand, index).ConsumeValueOrDie();

  BufferAllocation::Slice fusion_slice =
      assignment.GetUniqueTopLevelSlice(fusion).ConsumeValueOrDie();

  return operand_slice == fusion_slice;
}

}  // namespace

Status IrEmitterUnnested::HandleFusion(HloInstruction* fusion) {
  HloInstruction* root = fusion->fused_expression_root();
  // HandleFusion specializes reduction from a multi-dimensional array to a 1D
  // array. The specialized version requires a initializer thunk that
  // initializes the output array to the initial value of the reduce.
  if (HloInstruction::FusionKind::kInput == fusion->fusion_kind()) {
    switch (root->opcode()) {
      case HloOpcode::kReduce: {
        VLOG(3) << "Emitting fused reduction to vector: " << fusion->ToString();
        std::vector<std::unique_ptr<Thunk>> thunks;
        thunks.emplace_back(BuildKernelThunk(fusion));
        TF_RETURN_IF_ERROR(EmitInitializer(
            fusion, static_cast<KernelThunk*>(thunks.back().get())));
        bindings_.UnbindAllLocalIrValues();
        thunks.emplace_back(BuildKernelThunk(fusion));
        thunk_sequence_->emplace_back(
            MakeUnique<SequentialThunk>(std::move(thunks), fusion));
        std::vector<llvm_ir::IrArray> parameter_arrays;
        for (HloInstruction* operand : fusion->operands()) {
          parameter_arrays.push_back(GetIrArray(*operand));
        }
        GpuElementalIrEmitter elemental_emitter(
            hlo_module_config_, ir_emitter_context_->llvm_module(),
            &ir_builder_, GetNestedComputer());
        FusedIrEmitter fused_emitter(parameter_arrays, &elemental_emitter);
        TF_RETURN_IF_ERROR(root->Accept(&fused_emitter));

        Shape input_shape = root->operand(0)->shape();
        // EmitReductionToVector requires the input shape to have a layout, but
        // fused instructions don't have one. So we determine its layout from
        // the fusion's operands. The choice of the layout only affects
        // performance but not correctness.
        auto choose_input_layout = [](
            tensorflow::gtl::ArraySlice<const HloInstruction*> operands,
            Shape* input_shape) -> Status {
          // Prefer the layout of an operand whose shape is compatible with
          // input_shape.
          for (const HloInstruction* operand : operands) {
            if (ShapeUtil::Compatible(*input_shape, operand->shape())) {
              return LayoutUtil::CopyLayoutBetweenShapes(operand->shape(),
                                                         input_shape);
            }
          }
          // If no operand has a compatible shape, prefer an operand that has
          // the same rank at least.
          for (const HloInstruction* operand : operands) {
            if (ShapeUtil::Rank(*input_shape) ==
                ShapeUtil::Rank(operand->shape())) {
              // Do not use CopyLayoutBetweenShapes because input_shape and
              // operand->shape() may be incompatible.
              *input_shape->mutable_layout() = operand->shape().layout();
              return Status::OK();
            }
          }
          // When all the above fails, which is rare, set the default layout.
          LayoutUtil::SetToDefaultLayout(input_shape);
          return Status::OK();
        };
        TF_RETURN_IF_ERROR(
            choose_input_layout(fusion->operands(), &input_shape));

        return EmitReductionToVector(
            root, input_shape, fused_emitter.GetGenerator(root->operand(0)),
            fused_emitter.GetGenerator(root->operand(1)), root->dimensions(),
            root->to_apply());
      }
      default:
        LOG(FATAL) << "Bad opcode for input fusion: "
                   << fusion->fused_expression_root()->opcode();
    }
  } else if (HloInstruction::FusionKind::kLoop == fusion->fusion_kind() &&
             root->opcode() == HloOpcode::kDynamicUpdateSlice &&
             CanUpdateDynamicSliceInPlace(
                 ir_emitter_context_->buffer_assignment(), fusion)) {
    // Loop fusion instruction with DynamicUpdateSlice as fused root.
    // DynamicUpdateSlice's operand(0) and 'fusion' output share the same
    // BufferAllocation::Slice, so it is safe to emit code to update the slice
    // 'in-place'. This avoids copying data outside of the slice update region.

    // Set up kernel thunk and fused ir emitter.
    thunk_sequence_->emplace_back(BuildKernelThunk(fusion));
    std::vector<llvm_ir::IrArray> parameter_arrays;
    for (HloInstruction* operand : fusion->operands()) {
      parameter_arrays.push_back(GetIrArray(*operand));
    }
    GpuElementalIrEmitter elemental_emitter(hlo_module_config_,
                                            ir_emitter_context_->llvm_module(),
                                            &ir_builder_, GetNestedComputer());
    FusedIrEmitter fused_emitter(parameter_arrays, &elemental_emitter);
    TF_RETURN_IF_ERROR(root->Accept(&fused_emitter));

    // Recursively lookup 'fusion_operand' for DynamicUpdateSlice operand 0.
    ShapeIndex index_unused;
    auto* fusion_operand =
        LatestNonGteAncestorAndIndex(root->operand(0), &index_unused);
    CHECK_EQ(HloOpcode::kParameter, fusion_operand->opcode());

    // Operand(0) the input array which shares an allocation with the output.
    const auto* input = root->operand(0);
    llvm::Value* input_base_ptr = fused_emitter.GetIrValueForGTE(input);
    // Operand(1) 'update' is slice with which to update input at operand(0).
    const auto* update = root->operand(1);
    Shape update_shape = update->shape();
    TF_RETURN_IF_ERROR(
        LayoutUtil::CopyLayoutBetweenShapes(fusion->shape(), &update_shape));
    // Operand(2) the dynamic slice indices at which to write 'update'.
    const auto* start_indices = root->operand(2);

    // Create element generators for 'update' and 'start_indices'.
    llvm_ir::ElementGenerator element_generator =
        fused_emitter.GetGenerator(update);
    llvm_ir::ElementGenerator start_generator =
        fused_emitter.GetGenerator(start_indices);

    // Create loop body emitter which emits code to do the following:
    // *) Read dynamic slice start indices into 'start_index'.
    // *) Map requested 'index' and slice 'start_index' to input/output shape
    //    as 'output_index'.
    // *) Reads value from 'update' element generator.
    // *) Writes value to input/output array at 'output_index'.
    auto loop_body_emitter =
        [=](const llvm_ir::IrArray::Index& index) -> Status {
      // Emit IR to read dynamic start indices from hlo->operand(2).
      const int64 rank = ShapeUtil::Rank(input->shape());
      llvm_ir::IrArray::Index start_index(rank);
      for (int64 i = 0; i < rank; ++i) {
        llvm_ir::IrArray::Index dim_index({ir_builder_.getInt64(i)});
        TF_ASSIGN_OR_RETURN(start_index[i], start_generator(dim_index));
      }

      // Calculate 'output_index' at which to write value from update.
      llvm_ir::IrArray::Index output_index(rank);
      for (int64 i = 0; i < rank; ++i) {
        // Emit IR which computes:
        //   output_index = (start_index + index) % dim_size
        llvm::Value* dim_size = llvm::ConstantInt::get(
            index[i]->getType(), input->shape().dimensions(i));
        llvm::Value* start_index0 = ir_builder_.CreateZExtOrBitCast(
            start_index[i], index[i]->getType());
        output_index[i] = ir_builder_.CreateURem(
            ir_builder_.CreateAdd(start_index0, index[i]), dim_size);
      }

      // Read value from 'update'.
      TF_ASSIGN_OR_RETURN(llvm::Value * input_value, element_generator(index));
      // Write value to output array.
      llvm_ir::IrArray(input_base_ptr, input->shape())
          .EmitWriteArrayElement(output_index, input_value, &ir_builder_);
      return Status::OK();
    };

    // Create loop which iterates over 'update' shape.
    LaunchDimensions launch_dimensions = CalculateLaunchDimensions(
        update_shape, ir_emitter_context_->device_description());
    CHECK(Thunk::Kind::kKernel == LastThunk()->kind());
    UpdateLaunchDimensions(launch_dimensions,
                           static_cast<KernelThunk*>(LastThunk()),
                           ir_emitter_context_->llvm_module());
    return ParallelLoopEmitter(loop_body_emitter, update_shape,
                               launch_dimensions, &ir_builder_)
        .EmitLoop();
  }
  if (ImplementedAsGemm(*fusion)) {
    thunk_sequence_->emplace_back(BuildGemmThunk(fusion));
    return Status::OK();
  }
  if (ImplementedAsDnnConvolution(*fusion)) {
    thunk_sequence_->emplace_back(BuildConvolutionThunk(fusion));
    return Status::OK();
  }
  thunk_sequence_->emplace_back(BuildKernelThunk(fusion));
  return IrEmitter::HandleFusion(fusion);
}

namespace {

// Returns the indices of the first elements of all consecutive subarrays of the
// given array. For example:
// ConsecutiveSegments({m, m+1, m+2, n, k, k+1}) = {0, 3, 4}
std::vector<size_t> ConsecutiveSegments(tensorflow::gtl::ArraySlice<int64> xs) {
  std::vector<size_t> is = {0};
  for (size_t i = 1; i < xs.size(); ++i) {
    if (1 != xs[i] - xs[i - 1]) {
      is.push_back(i);
    }
  }
  return is;
}

// Merges the sequences of dimensions of the given shape which start at the
// given indices `segs`.
Shape MergeDimensions(tensorflow::gtl::ArraySlice<size_t> segs,
                      const Shape& shape) {
  std::vector<int64> dimensions;
  for (size_t i = 1; i <= segs.size(); ++i) {
    dimensions.push_back(std::accumulate(
        shape.dimensions().begin() + segs[i - 1],
        shape.dimensions().begin() +
            (segs.size() == i ? shape.dimensions().size() : segs[i]),
        1, std::multiplies<int64>()));
  }
  return ShapeUtil::MakeShapeWithMonotonicDim0MajorLayout(shape.element_type(),
                                                          dimensions);
}

// Returns whether the given shapes and permutation are a 0-2-1 transpose, and
// if so, the normalized and rank-reduced shapes. The shapes must have the same
// dimensions, so this considers layout only.
//
// This function recognizes higher-rank transposes which are elementwise
// equivalent to a 0-2-1 transpose.
std::tuple<bool, Shape, Shape> IsTranspose021(const Shape& a, const Shape& b) {
  CHECK(ShapeUtil::Compatible(a, b));
  std::vector<int64> perm(a.dimensions().size());
  {
    std::vector<int64> layout_a(a.layout().minor_to_major().rbegin(),
                                a.layout().minor_to_major().rend());
    std::vector<int64> layout_b(b.layout().minor_to_major().rbegin(),
                                b.layout().minor_to_major().rend());
    for (size_t i = 0; i < perm.size(); ++i) {
      perm[i] = PositionInContainer(layout_b, layout_a[i]);
    }
  }
  auto segs = ConsecutiveSegments(perm);
  Shape norm_a = ShapeUtil::NormalizeShapeToMonotonicDim0MajorLayout(a);
  Shape norm_b = ShapeUtil::NormalizeShapeToMonotonicDim0MajorLayout(b);
  if (3 == segs.size() && 0 == perm[0]) {
    Shape reduced_a = MergeDimensions(segs, norm_a);
    Shape reduced_b = ShapeUtil::MakeShapeWithMonotonicDim0MajorLayout(
        b.element_type(),
        Permute({0, 2, 1}, AsInt64Slice(reduced_a.dimensions())));
    return std::make_tuple(true, reduced_a, reduced_b);
  }
  return std::make_tuple(false, ShapeUtil::MakeNil(), ShapeUtil::MakeNil());
}

// Returns whether the given shapes are potentially of a 0-2-1 transpose.
// As 0-2-1 is a self-inverse permutation, which shape is input or output is
// arbitrary.
bool AreShapesForTranspose021(const Shape& a, const Shape& b) {
  return 3 == b.dimensions().size() &&
         ShapeUtil::Compatible(
             ShapeUtil::NormalizeShapeToMonotonicDim0MajorLayout(a),
             ShapeUtil::PermuteDimensions(
                 {0, 2, 1},
                 ShapeUtil::NormalizeShapeToMonotonicDim0MajorLayout(b)));
}

// Emits a tiled 0-2-1 transpose, assuming both input and output lain out from
// major to minor. The x- and y- dimensions are tiled in square tiles of edge
// length `tile_size`. Each thread block of `tile_size` threads transposes one
// tile: each thread copies a row from the input to a shared memory tile, then
// copies a column from the shared memory tile to the output.
//
// `tile_size` should usually be same as warp size.
//
// Returns (number of tiles = number of thread blocks needed).
//
// TODO(b/33320379): Here each block transposes 1 tile. It may be more efficient
//                   to launch fewer blocks so each transposes many tiles, and
//                   in any case, the number of blocks we can launch is limited.
//
// This is the same algorithm in CUDA:
// https://github.com/tensorflow/tensorflow/blob/6172351b81af76d0b819fea6bb478cbd4016d6c2/tensorflow/core/kernels/conv_ops_gpu_3.cu.cc#L183
int64 EmitTranspose021Tiled(llvm_ir::IrArray input, llvm_ir::IrArray output,
                            const int64 tile_size, llvm::IRBuilder<>* builder) {
  // Adds `addend` to the given `dim` of `index`.
  auto offset_dim = [builder](llvm_ir::IrArray::Index index,
                              llvm::Value* addend, int64 dim) {
    index[dim] = builder->CreateAdd(index[dim], addend);
    return index;
  };

  CHECK(AreShapesForTranspose021(input.GetShape(), output.GetShape()));

  Shape input_shape =
      ShapeUtil::NormalizeShapeToMonotonicDim0MajorLayout(input.GetShape());
  Shape output_shape =
      ShapeUtil::NormalizeShapeToMonotonicDim0MajorLayout(output.GetShape());
  input = input.CastToShape(input_shape, builder);
  output = output.CastToShape(output_shape, builder);

  llvm::Type* tile_type = llvm::ArrayType::get(
      llvm::ArrayType::get(input.GetElementLlvmType(), tile_size),
      // One extra here to avoid share memory bank conflict
      tile_size + 1);
  auto* tile = new llvm::GlobalVariable(
      *builder->GetInsertBlock()->getParent()->getParent(), tile_type,
      /*isConstant=*/false, llvm::GlobalValue::PrivateLinkage,
      llvm::UndefValue::get(tile_type), "tile", nullptr,
      llvm::GlobalValue::NotThreadLocal,
      /*AddressSpace=*/3 /* GPU shared memory */);

  // let x = threadIdx.x
  llvm::Value* x = llvm_ir::EmitCallToIntrinsic(
      llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x, {}, {}, builder);
  llvm_ir::AddRangeMetadata(0, tile_size, static_cast<llvm::Instruction*>(x));
  x = builder->CreateIntCast(x, builder->getInt64Ty(), /*isSigned=*/true,
                             "thread.id.x");

  // `emit_cp` emits equivalent to following pseudocode:
  // if (tile_size == tile_width && tile_size == tile_height) {
  //   unroll for (y in 0..tile_size) {
  //     emit_cp_element(index + {0, y, 0}, y);
  //   }
  // } else if (x < tile_width) {
  //   for (y in 0..tile_height) {
  //     emit_cp_element(index + {0, y, 0}, y);
  //   }
  // }
  //
  // We use this to emit both the copy from input to tile and the copy from tile
  // to output.
  //
  // `index` is the origin of the row or column in the input or output array.
  //
  // `emit_cp_element(index, y)` emits code to copy a single element between the
  // tile and the input or output array, where `y` is the `y`-position in the
  // tile, whether which is row or column is a function of whether we're copying
  // from input or to output, and `index` is the index into the input or output
  // array.
  auto emit_cp_tile = [builder, tile_size, x, &offset_dim](
      std::function<void(const llvm_ir::IrArray::Index&, llvm::Value*)>
          emit_cp_element,
      llvm::Value* tile_width, llvm::Value* tile_height,
      const llvm_ir::IrArray::Index& index, const string& loop_name) {
    llvm_ir::LlvmIfData if_not_last_row = llvm_ir::EmitIfThenElse(
        builder->CreateAnd(
            builder->CreateICmpEQ(builder->getInt64(tile_size), tile_width),
            builder->CreateICmpEQ(builder->getInt64(tile_size), tile_height)),
        "not_last_row", builder);
    builder->SetInsertPoint(if_not_last_row.true_block->getTerminator());
    for (int64 i = 0; i < tile_size; ++i) {
      emit_cp_element(offset_dim(index, builder->getInt64(i), /*dim=*/1),
                      builder->getInt64(i));
    }
    builder->SetInsertPoint(if_not_last_row.false_block->getTerminator());
    llvm_ir::LlvmIfData if_in_tile = llvm_ir::EmitIfThenElse(
        builder->CreateICmpULT(x, tile_width), "in_tile", builder);
    builder->SetInsertPoint(if_in_tile.true_block->getTerminator());
    auto loop = llvm_ir::ForLoop::EmitForLoop(loop_name, builder->getInt64(0),
                                              tile_height, builder->getInt64(1),
                                              builder);
    llvm_ir::SetToFirstInsertPoint(loop->GetHeaderBasicBlock(), builder);
    builder->SetInsertPoint(loop->GetBodyBasicBlock()->getTerminator());
    emit_cp_element(offset_dim(index, loop->GetIndVarValue(), /*dim=*/1),
                    loop->GetIndVarValue());
    builder->SetInsertPoint(if_not_last_row.after_block->getTerminator());
  };

  auto input_dims_in_tiles = input_shape.dimensions();
  // Unpermuted dimensions are untiled.
  for (int i = 1; i < 3; ++i) {
    input_dims_in_tiles[i] =
        CeilOfRatio<int64>(input_dims_in_tiles[i], tile_size);
  }
  int64 num_tiles =
      std::accumulate(input_dims_in_tiles.begin(), input_dims_in_tiles.end(), 1,
                      std::multiplies<int64>());
  const llvm_ir::IrArray::Index input_tile_index(
      /*linear=*/builder->CreateIntCast(
          llvm_ir::AddRangeMetadata(
              0, num_tiles,
              static_cast<llvm::Instruction*>(llvm_ir::EmitCallToIntrinsic(
                  llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_x, {}, {},
                  builder))),
          builder->getInt64Ty(), /*isSigned=*/true, "block.id.x"),
      ShapeUtil::MakeShapeWithMonotonicDim0MajorLayout(
          PRED /*arbitrary*/, AsInt64Slice(input_dims_in_tiles)),
      builder);
  const llvm_ir::IrArray::Index input_tile_origin = ({
    llvm_ir::IrArray::Index index = input_tile_index;
    for (int i = 1; i < 3; ++i) {
      index[i] = builder->CreateMul(index[i], builder->getInt64(tile_size),
                                    "tile_origin." + std::to_string(i));
    }
    index;
  });
  const llvm_ir::IrArray::Index input_index =
      offset_dim(input_tile_origin, x, /*dim=*/2);
  std::vector<llvm::Value*> tile_dims(input_shape.dimensions().size());
  // Only last row or column may not have full size.
  for (int i = 1; i < 3; ++i) {
    tile_dims[i] = builder->CreateSelect(
        builder->CreateICmpEQ(input_tile_index[i],
                              builder->getInt64(input_dims_in_tiles[i] - 1)),
        builder->getInt64(input_shape.dimensions(i) -
                          (input_dims_in_tiles[i] - 1) * tile_size),
        builder->getInt64(tile_size), "tile_size");
  }

  // Load data from input memory to shared memory tile.
  emit_cp_tile(
      // tile[y, x] = input_array[index]
      [builder, tile, x, &input](const llvm_ir::IrArray::Index& index,
                                 llvm::Value* y) {
        builder->CreateStore(
            input.EmitReadArrayElement(index, builder, "input_element"),
            builder->CreateGEP(tile, {builder->getInt64(0), y, x}));
      },
      tile_dims[2], tile_dims[1], input_index, "input");

  // Wait for all threads to reach this point, lest we copy a value from tile to
  // output before the other thread copies it from input to tile.
  // This is `__syncthreads` in CUDA.
  llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::nvvm_barrier0, {}, {}, builder);

  const llvm_ir::IrArray::Index output_tile_index(
      Permute({0, 2, 1}, input_tile_index.multidim()));
  const llvm_ir::IrArray::Index output_tile_origin(
      Permute({0, 2, 1}, input_tile_origin.multidim()));
  const llvm_ir::IrArray::Index output_index =
      offset_dim(output_tile_origin, x, /*dim=*/2);

  // Store data from shared memory tile to output memory.
  emit_cp_tile(
      // output_array[index] = tile[x, y]
      [builder, tile, x, &output](const llvm_ir::IrArray::Index& index,
                                  llvm::Value* y) {
        output.EmitWriteArrayElement(
            index,
            builder->CreateLoad(
                builder->CreateGEP(tile, {builder->getInt64(0), x, y}),
                "output_element"),
            builder);
      },
      tile_dims[1], tile_dims[2], output_index, "output");

  return num_tiles;
}

}  // namespace

Status IrEmitterUnnested::HandleCopy(HloInstruction* copy,
                                     HloInstruction* operand) {
  if (ImplementedAsMemcpy(*copy)) {
    thunk_sequence_->emplace_back(BuildCopyThunk(copy));
    return Status::OK();
  }
  bool is_transpose_021;
  Shape reduced_input_shape, reduced_output_shape;
  std::tie(is_transpose_021, reduced_input_shape, reduced_output_shape) =
      IsTranspose021(operand->shape(), copy->shape());
  if (is_transpose_021 &&
      reduced_input_shape.dimensions(1) >= kMinDimensionToTransposeTiled &&
      reduced_input_shape.dimensions(2) >= kMinDimensionToTransposeTiled) {
    thunk_sequence_->emplace_back(BuildKernelThunk(copy));
    VLOG(3) << "Emitting tiled 0-2-1 transposition";
    constexpr int64 tile_size = 32;
    int64 num_tiles = EmitTranspose021Tiled(
        GetIrArray(*operand).CastToShape(reduced_input_shape, &ir_builder_),
        GetIrArray(*copy).CastToShape(reduced_output_shape, &ir_builder_),
        tile_size, &ir_builder_);
    UpdateLaunchDimensions(LaunchDimensions(num_tiles, tile_size), LastThunk(),
                           ir_emitter_context_->llvm_module());
    return Status::OK();
  }

  return IrEmitter::HandleCopy(copy, operand);
}

Status IrEmitterUnnested::EmitColumnReduction(
    int64 height, int64 width, HloInstruction* reduce, const Shape& input_shape,
    const llvm_ir::ElementGenerator& input_gen,
    const llvm_ir::ElementGenerator& init_value_gen, HloComputation* reducer) {
  // Divide the input matrix into tiles of size Kx1. For example, when the
  // input matrix is 4x4 and K=2, the tiled matrix looks like
  //
  //   0123
  //   0123
  //   4567
  //   4567  // Numbers indicate tile IDs.
  //
  // Each tile is first partially reduced to a scalar by a thread, and then the
  // scalar is accumulated to the output vector using atomic operations. We
  // choose 16 as the tile size, which matches Eigen's ColumnReduceKernel.
  constexpr int64 kTileSize = 16;
  // If the height is not a multiple of the tile size, we pad the bottom of the
  // input matrix.
  const int64 height_in_tiles = CeilOfRatio(height, kTileSize);

  // for (linear_index = threadIdx.x + blockIdx.x * blockDim.x;
  //      linear_index < height_in_tiles * width;
  //      linear_index += blockDim.x * gridDim.x) {
  //   y_in_tiles = linear_index / width;
  //   x = linear_index % width;
  //
  //   partial_result = init_value;
  //   if (height % kTileSize == 0 ||
  //       y_in_tiles * kTileSize + kTileSize <= height) {
  //     for (element_id_in_tile : range(kTileSize)) {
  //       y = y_in_tiles * kTileSize + element_id_in_tile;
  //       partial_result = Reducer(partial_result, input[y][x]);
  //     }
  //   } else {
  //     for (element_id_in_tile : range(kTileSize)) {
  //       y = y_in_tiles * kTileSize + element_id_in_tile;
  //       if (y < height) {
  //         partial_result = Reducer(partial_result, input[y][x]);
  //       }
  //     }
  //   }
  //   AtomicReducer(&output[x], partial_result);
  // }
  auto loop_body_emitter =
      [=](const llvm_ir::IrArray::Index& tile_index) -> Status {
    // Emit the loop body that reduces one tile.
    llvm::Type* element_ir_type = llvm_ir::PrimitiveTypeToIrType(
        input_shape.element_type(), &ir_builder_);
    llvm::Value* partial_reduction_result_address = ir_builder_.CreateAlloca(
        element_ir_type, /*ArraySize=*/nullptr, "partial_reduction_result");
    {
      TF_ASSIGN_OR_RETURN(llvm::Value * init_ir_value,
                          init_value_gen(llvm_ir::IrArray::Index({})));
      ir_builder_.CreateStore(init_ir_value, partial_reduction_result_address);
    }

    // Emit an inner for-loop that partially reduces the elements in the given
    // tile.
    llvm::Value* y_in_tiles = tile_index[0];
    llvm::Value* x = tile_index[1];

    auto emit_tile_element_loop = [=](bool tile_in_bounds) -> Status {
      std::unique_ptr<llvm_ir::ForLoop> tile_element_loop =
          llvm_ir::ForLoop::EmitForLoop("element_id_in_tile",
                                        ir_builder_.getInt64(0),
                                        ir_builder_.getInt64(kTileSize),
                                        ir_builder_.getInt64(1), &ir_builder_);

      // Emit the body of the partial reduction loop.
      llvm_ir::SetToFirstInsertPoint(tile_element_loop->GetBodyBasicBlock(),
                                     &ir_builder_);
      llvm::Value* y = ir_builder_.CreateNSWAdd(
          ir_builder_.CreateNSWMul(y_in_tiles, ir_builder_.getInt64(kTileSize)),
          tile_element_loop->GetIndVarValue());
      // Unless we know the tile is entirely in bounds, we have to emit a
      // y-in-bounds check before reading from the input.
      if (!tile_in_bounds) {
        llvm_ir::LlvmIfData if_data = llvm_ir::EmitIfThenElse(
            ir_builder_.CreateICmpULT(y, ir_builder_.getInt64(height)),
            "y_in_bounds", &ir_builder_);

        // Emit code that reads the input element and accumulates it to
        // the partial reduction result.
        llvm_ir::SetToFirstInsertPoint(if_data.true_block, &ir_builder_);
      }
      llvm::Value* input_address = ir_builder_.CreateAlloca(element_ir_type);
      {
        // {y,x} is an index to input_matrix_shape [height,width]. We need to
        // convert that to an index to input_shape (the shape of the operand of
        // "reduce"). This conversion is composed of a transposition from
        // input_shape to normalized_input_shape and a reshape from
        // normalized_input_shape to input_matrix_shape.
        const Shape normalized_input_shape =
            ShapeUtil::NormalizeShapeToMonotonicDim0MajorLayout(input_shape);
        const std::vector<int64> transpose_dimension_mapping(
            input_shape.layout().minor_to_major().rbegin(),
            input_shape.layout().minor_to_major().rend());

        const Shape input_matrix_shape =
            ShapeUtil::MakeShapeWithMonotonicDim0MajorLayout(
                input_shape.element_type(), {height, width});
        const llvm_ir::IrArray::Index input_matrix_index(
            {y, x}, input_matrix_shape, &ir_builder_);
        const llvm_ir::IrArray::Index input_index =
            input_matrix_index
                .SourceIndexOfReshape(input_matrix_shape,
                                      normalized_input_shape, &ir_builder_)
                .SourceIndexOfTranspose(normalized_input_shape, input_shape,
                                        transpose_dimension_mapping,
                                        &ir_builder_);
        TF_ASSIGN_OR_RETURN(llvm::Value * input_ir_value,
                            input_gen(input_index));
        ir_builder_.CreateStore(input_ir_value, input_address);
      }
      return (EmitCallToNestedComputation(
          *reducer, {partial_reduction_result_address, input_address},
          partial_reduction_result_address));
    };

    // y_end = kTileSize + y_in_tiles * kTileSize, i.e., the y location that's
    // immediately beyond the tile.
    llvm::Value* y_end = ir_builder_.CreateNSWAdd(
        ir_builder_.getInt64(kTileSize),
        ir_builder_.CreateNSWMul(y_in_tiles, ir_builder_.getInt64(kTileSize)));
    llvm::Value* tile_in_bounds = ir_builder_.CreateOr(
        ir_builder_.CreateICmpULE(y_end, ir_builder_.getInt64(height)),
        ir_builder_.getInt1(height % kTileSize == 0));
    // The tile is entirely in bound if "height" is a multiple of kTileSize or
    // y_end <= height.
    llvm_ir::LlvmIfData if_tile_in_bounds_data =
        llvm_ir::EmitIfThenElse(tile_in_bounds, "tile_in_bounds", &ir_builder_);
    llvm_ir::SetToFirstInsertPoint(if_tile_in_bounds_data.true_block,
                                   &ir_builder_);
    TF_RETURN_IF_ERROR(emit_tile_element_loop(/*tile_in_bounds=*/true));
    llvm_ir::SetToFirstInsertPoint(if_tile_in_bounds_data.false_block,
                                   &ir_builder_);
    TF_RETURN_IF_ERROR(emit_tile_element_loop(/*tile_in_bounds=*/false));

    // After the if-then-else statement on tile_in_bounds, emit atomic
    // operations to accumulate the partial reduction result to the output
    // element.
    llvm_ir::SetToFirstInsertPoint(if_tile_in_bounds_data.after_block,
                                   &ir_builder_);
    const HloInstruction* output =
        reduce->IsFused() ? reduce->fusion_instruction() : reduce;
    llvm::Value* output_address = GetIrArray(*output).EmitArrayElementAddress(
        llvm_ir::IrArray::Index(x, output->shape(), &ir_builder_), &ir_builder_,
        "output_element_address");
    return EmitAtomicOperationForNestedComputation(
        *reducer, output_address, partial_reduction_result_address);
  };

  // Emit a parallel loop that iterate through all input tiles.
  Shape tiled_input_shape = ShapeUtil::MakeShapeWithLayout(
      reduce->shape().element_type(), {height_in_tiles, width}, {1, 0});
  LaunchDimensions launch_dimensions = CalculateLaunchDimensions(
      tiled_input_shape, ir_emitter_context_->device_description());
  CHECK(LastThunk()->kind() == Thunk::Kind::kSequential);
  UpdateLaunchDimensions(
      launch_dimensions,
      static_cast<SequentialThunk*>(LastThunk())->thunks().back().get(),
      ir_emitter_context_->llvm_module());
  return ParallelLoopEmitter(loop_body_emitter, tiled_input_shape,
                             launch_dimensions, &ir_builder_)
      .EmitLoop();
}

Status IrEmitterUnnested::EmitRowReduction(
    int64 depth, int64 height, int64 width, HloInstruction* reduce,
    const Shape& input_shape, const llvm_ir::ElementGenerator& input_gen,
    const llvm_ir::ElementGenerator& init_value_gen, HloComputation* reducer) {
  // A naive algorithm is:
  // 1. Divide the input tensor into tiles of size 1x1xK.
  // 2. Partially reduces each tile to a scalar using one thread.
  // 3. Accumulates that scalar to the output vector using atomic operations.
  //
  // for (linear_index = threadIdx.x + blockIdx.x * blockDim.x;
  //      linear_index < depth * height * width_in_tiles;
  //      linear_index += blockDim.x * gridDim.x) {
  //   int x_in_tiles = linear_index % width_in_tiles;
  //   int y = linear_index / width_in_tiles % height;
  //   int z = linear_index / (height * width_in_tiles);
  //   float partial_result = 0;
  //   for (element_id_in_tile : range(kTileSize)) {
  //     int x = x_in_tiles * kTileSize + element_id_in_tile;
  //     if (x < width)
  //       partial_result = reducer(partial_result, input[z][y][z]);
  //   }
  //   AtomicReducer(&output[y], partial_result);
  // }
  //
  // Three optimizations are performed.
  //
  // 1. To coalesc global memory accesses, dilate the tile with a factor of 32
  // (i.e. the warp size). For example, suppose the width is 8x32=256. Instead
  // of making each tile consecutive, we let make tile 0 column
  // [0,32,64,...,224], tile 1 column [1,33,65,...,225], and so on. This ensures
  // that threads in a warp access consecutive memory in one iteration (i.e.
  // coalesced). In the above example, the warp that contains thread 0-31
  // accesses column 0-31 in the first iteration, and 32-63 in the second
  // iteration, and so on.
  //
  // 2. Partially accumulate partial reduced results computed by threads in the
  // same warp using shfl_down. Using shfl_down is faster than directly using
  // atomic operations because shfl_down transfers the data between threads
  // using shared memory and threads in the same warp run in lock step (thus no
  // extra synchronization needed). See
  // https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
  // for details. The downside is, to produce correct results when using
  // shfl_down, we need to guarantee threads in the same warp work on input
  // elements with the same y, so the number of tiles in each row must be a
  // multiple of 32.
  //
  // 3. Specialize the case that the entire tile is in bounds. When that is
  // true, we don't need to emit "if(x<width)" inside the loop on
  // element_id_in_tile, which makes the code more friendly to optimizations
  // such as LICM.
  //
  // for (linear_index = threadIdx.x + blockIdx.x * blockDim.x;
  //      linear_index < depth * height * width_in_tiles;
  //      linear_index += blockDim.x * gridDim.x) {
  //   int x_in_tiles = linear_index % width_in_tiles;
  //   int y = linear_index / width_in_tiles % height;
  //   int z = linear_index / (height * width_in_tiles);
  //   int warp_id = x_in_tiles / warpSize;
  //   int lane_id = x_in_tiles % warpSize;
  //   float partial_result = 0;
  //   int x = warp_id * kTileSize * warpSize + lane_id;
  //   if (width % (kTileSize * warpSize) == 0 ||
  //       x + (kTileSize - 1) * warpSize < width) {
  //     // The entire tile is in bounds.
  //     for (int element_id_in_tile = 0; element_id_in_tile < kTileSize;
  //        ++element_id_in_tile, x += warpSize) {
  //       partial_result = Reducer(partial_result, input[z][y][x]);
  //     }
  //   } else {
  //     // The tile is partially in bounds.
  //     for (int element_id_in_tile = 0; element_id_in_tile < kTileSize;
  //          ++element_id_in_tile, x += warpSize) {
  //       if (x < width)
  //         partial_result = Reducer(partial_result, input[z][y][x]);
  //     }
  //   }
  //   for (shuffle_distance = 16; shuffle_distance > 0; shuffle_distance /= 2)
  //     partial_result = Reducer(
  //         partial_result,
  //         __shfl_down(partial_result, shuffle_distance));
  //   if (lane_id == 0)
  //     AtomicReducer(&output[y], partial_result);
  // }
  //
  // Choose 8 as the tile size, which matches Eigen's RowReduceKernel.
  constexpr int64 kTileSize = 8;
  // Round the width in tiles up to the nearest multiple of kWarpSize, so that
  // the use of shfl_down is valid.
  const int64 width_in_tiles =
      RoundUpToNearest(CeilOfRatio(width, kTileSize), kWarpSize);

  auto loop_body_emitter =
      [=](const llvm_ir::IrArray::Index& tile_index) -> Status {
    // Emit the loop body that reduces one tile.
    llvm::Type* element_ir_type = llvm_ir::PrimitiveTypeToIrType(
        input_shape.element_type(), &ir_builder_);
    llvm::Value* partial_reduction_result_address = ir_builder_.CreateAlloca(
        element_ir_type, /*ArraySize=*/nullptr, "partial_reduction_result");
    {
      TF_ASSIGN_OR_RETURN(llvm::Value * init_ir_value,
                          init_value_gen(llvm_ir::IrArray::Index({})));
      ir_builder_.CreateStore(init_ir_value, partial_reduction_result_address);
    }

    // Emit an inner for-loop that partially reduces the elements in the given
    // tile.
    llvm::Value* z = tile_index[0];
    llvm::Value* y = tile_index[1];
    llvm::Value* x_tile = tile_index[2];
    llvm::Value* warp_id = ir_builder_.CreateUDiv(
        x_tile, ir_builder_.getInt64(kWarpSize), "warp_id");
    llvm::Value* lane_id = ir_builder_.CreateURem(
        x_tile, ir_builder_.getInt64(kWarpSize), "lane_id");

    // The x-location of the last element in this tile.
    //   last_x = lane_id + warpSize * (kTileSize - 1 + warp_id * kTileSize);
    llvm::Value* last_x = ir_builder_.CreateNSWAdd(
        lane_id,
        ir_builder_.CreateNSWMul(
            ir_builder_.getInt64(kWarpSize),
            ir_builder_.CreateNSWAdd(
                ir_builder_.getInt64(kTileSize - 1),
                ir_builder_.CreateNSWMul(warp_id,
                                         ir_builder_.getInt64(kTileSize)))));

    auto emit_tile_element_loop = [=](bool tile_in_bounds) -> Status {
      std::unique_ptr<llvm_ir::ForLoop> tile_element_loop =
          llvm_ir::ForLoop::EmitForLoop("element_id_in_tile",
                                        ir_builder_.getInt64(0),
                                        ir_builder_.getInt64(kTileSize),
                                        ir_builder_.getInt64(1), &ir_builder_);

      // Emit the body of the partial reduction loop.
      llvm_ir::SetToFirstInsertPoint(tile_element_loop->GetBodyBasicBlock(),
                                     &ir_builder_);
      // x = lane_id + warpSize * (element_id_in_tile + warp_id * kTileSize);
      llvm::Value* x = ir_builder_.CreateNSWAdd(
          lane_id,
          ir_builder_.CreateNSWMul(
              ir_builder_.getInt64(kWarpSize),
              ir_builder_.CreateNSWAdd(
                  tile_element_loop->GetIndVarValue(),
                  ir_builder_.CreateNSWMul(warp_id,
                                           ir_builder_.getInt64(kTileSize)))));

      // Unless we know the tile is entirely in bounds, we have to emit a
      // x-in-bounds check before reading from the input.
      if (!tile_in_bounds) {
        llvm_ir::LlvmIfData if_x_in_bounds_data = llvm_ir::EmitIfThenElse(
            ir_builder_.CreateICmpULT(x, ir_builder_.getInt64(width)),
            "x_in_bounds", &ir_builder_);

        // Points ir_builder_ to the then-block.
        llvm_ir::SetToFirstInsertPoint(if_x_in_bounds_data.true_block,
                                       &ir_builder_);
      }

      // Emit code that reads the input element and accumulates it to the
      // partial reduction result.
      llvm::Value* input_address = ir_builder_.CreateAlloca(element_ir_type);
      {
        // {z,y,x} is an index to input_3d_tensor_shape [depth,height,width]. We
        // need to convert that to an index to input_shape (the shape of the
        // operand of "reduce"). This conversion is composed of a transposition
        // from input_shape to normalized_input_shape and a reshape from
        // normalized_input_shape to input_3d_tensor_shape.
        const Shape normalized_input_shape =
            ShapeUtil::NormalizeShapeToMonotonicDim0MajorLayout(input_shape);
        const std::vector<int64> transpose_dimension_mapping(
            input_shape.layout().minor_to_major().rbegin(),
            input_shape.layout().minor_to_major().rend());
        const Shape input_3d_tensor_shape =
            ShapeUtil::MakeShapeWithMonotonicDim0MajorLayout(
                input_shape.element_type(), {depth, height, width});
        const llvm_ir::IrArray::Index input_3d_tensor_index(
            {z, y, x}, input_3d_tensor_shape, &ir_builder_);
        const llvm_ir::IrArray::Index input_index =
            input_3d_tensor_index
                .SourceIndexOfReshape(input_3d_tensor_shape,
                                      normalized_input_shape, &ir_builder_)
                .SourceIndexOfTranspose(normalized_input_shape, input_shape,
                                        transpose_dimension_mapping,
                                        &ir_builder_);
        TF_ASSIGN_OR_RETURN(llvm::Value * input_ir_value,
                            input_gen(input_index));
        ir_builder_.CreateStore(input_ir_value, input_address);
      }
      return EmitCallToNestedComputation(
          *reducer, {partial_reduction_result_address, input_address},
          partial_reduction_result_address);
    };

    llvm::Value* tile_in_bounds = ir_builder_.CreateOr(
        ir_builder_.getInt1(width % (kTileSize * kWarpSize) == 0),
        ir_builder_.CreateICmpULT(last_x, ir_builder_.getInt64(width)));
    llvm_ir::LlvmIfData if_tile_in_bounds_data =
        llvm_ir::EmitIfThenElse(tile_in_bounds, "tile_in_bounds", &ir_builder_);
    llvm_ir::SetToFirstInsertPoint(if_tile_in_bounds_data.true_block,
                                   &ir_builder_);
    TF_RETURN_IF_ERROR(emit_tile_element_loop(/*tile_in_bounds=*/true));
    llvm_ir::SetToFirstInsertPoint(if_tile_in_bounds_data.false_block,
                                   &ir_builder_);
    TF_RETURN_IF_ERROR(emit_tile_element_loop(/*tile_in_bounds=*/false));

    // After the if-then-else statement on tile_in_bounds, emit calls to
    // shfl_down that accumulate the partial reduction results of all threads
    // from the warp.
    llvm_ir::SetToFirstInsertPoint(if_tile_in_bounds_data.after_block,
                                   &ir_builder_);
    for (int shuffle_distance = 16; shuffle_distance >= 1;
         shuffle_distance /= 2) {
      llvm::Value* partial_reduction_result = ir_builder_.CreateLoad(
          partial_reduction_result_address, "partial_reduction_result");
      llvm::Value* result_from_other_lane = ir_builder_.CreateAlloca(
          element_ir_type, nullptr, "result_from_other_lane");
      ir_builder_.CreateStore(
          EmitShuffleDown(partial_reduction_result,
                          ir_builder_.getInt32(shuffle_distance), &ir_builder_),
          result_from_other_lane);
      TF_RETURN_IF_ERROR(EmitCallToNestedComputation(
          *reducer, {partial_reduction_result_address, result_from_other_lane},
          partial_reduction_result_address));
    }

    const HloInstruction* output =
        reduce->IsFused() ? reduce->fusion_instruction() : reduce;

    // Emit an atomic operation that accumulates the partial reduction result of
    // lane 0 (which holds the partially accumulated result for its warp) to the
    // output element.
    llvm_ir::LlvmIfData if_lane_id_is_zero_data = llvm_ir::EmitIfThenElse(
        ir_builder_.CreateICmpEQ(lane_id, ir_builder_.getInt64(0)),
        "lane_id_is_zero", &ir_builder_);
    llvm_ir::SetToFirstInsertPoint(if_lane_id_is_zero_data.true_block,
                                   &ir_builder_);
    llvm::Value* output_address = GetIrArray(*output).EmitArrayElementAddress(
        llvm_ir::IrArray::Index(y, output->shape(), &ir_builder_), &ir_builder_,
        "output_element_address");
    return EmitAtomicOperationForNestedComputation(
        *reducer, output_address, partial_reduction_result_address);
  };

  // Emit a parallel loop that iterates through every input tiles.
  Shape tiled_input_shape = ShapeUtil::MakeShapeWithLayout(
      reduce->shape().element_type(), {depth, height, width_in_tiles},
      {2, 1, 0});
  LaunchDimensions launch_dimensions = CalculateLaunchDimensions(
      tiled_input_shape, ir_emitter_context_->device_description());
  CHECK(LastThunk()->kind() == Thunk::Kind::kSequential);
  UpdateLaunchDimensions(
      launch_dimensions,
      static_cast<SequentialThunk*>(LastThunk())->thunks().back().get(),
      ir_emitter_context_->llvm_module());
  return ParallelLoopEmitter(loop_body_emitter, tiled_input_shape,
                             launch_dimensions, &ir_builder_)
      .EmitLoop();
}

// Figures out whether `reduce` is a row or column reduction, and which
// dimensions to reduce, and calls either `EmitRowReduction` or
// `EmitColumnReduction` as appropriate.
// Prerequisite: all the dimensions to keep are contiguous in the input layout
//               and, if `reduce` is fused, the fused subgraph is pure
//               elementwise.
Status IrEmitterUnnested::EmitReductionToVector(
    HloInstruction* reduce, const Shape& input_shape,
    const llvm_ir::ElementGenerator& input_gen,
    const llvm_ir::ElementGenerator& init_value_gen,
    tensorflow::gtl::ArraySlice<int64> dimensions_to_reduce,
    HloComputation* reducer) {
  // This emission requires "reduce" to have an input layout. It is either set
  // by LayoutAssignment (for a top-level kReduce) or by InstructionFusion (for
  // a fused kReduce).
  CHECK(input_shape.has_layout()) << "LayoutAssignment or InstructionFusion "
                                     "doesn't set the input layout of "
                                  << reduce->ToString();

  // Specialize multi-dimensional-array-to-vector reduction.
  std::vector<int64> input_dims_to_keep;
  for (int64 input_dim = 0; input_dim < ShapeUtil::Rank(input_shape);
       ++input_dim) {
    if (std::find(dimensions_to_reduce.begin(), dimensions_to_reduce.end(),
                  input_dim) == dimensions_to_reduce.end()) {
      input_dims_to_keep.push_back(input_dim);
    }
  }

  // Sort the dimensions to keep from minor to major, to facilitate checking
  // whether another dimension is major or minor of them.
  std::sort(input_dims_to_keep.begin(), input_dims_to_keep.end(),
            [&input_shape](int64 dim_a, int64 dim_b) {
              return PositionInContainer(input_shape.layout().minor_to_major(),
                                         dim_a) <
                     PositionInContainer(input_shape.layout().minor_to_major(),
                                         dim_b);
            });
  // Now, if output rank is at least 1, `input_dims_to_keep.front()` is
  // minormost and `input_dims_to_keep.back()` is majormost.

  // If the dimensions to keep are minormost, emit a column reduction. As all
  // the dimensions to keep are contiguous, by prerequisite of
  // `EmitReductionToVector`, we only need to check whether the minormost
  // dimension of the input is to keep.
  //
  // If the output is scalar, we could emit either a row or a column reduction.
  // Some tests have shown scalar reduction is no more efficient as row
  // reduction, and is simpler to emit as column reduction, so we emit a column
  // reduction in this case.
  if (input_dims_to_keep.empty() ||
      input_dims_to_keep.front() ==
          LayoutUtil::Minor(input_shape.layout(), 0)) {
    // Column reduction. Treat the result of "input" as a matrix whose width
    // is the most minor dimension and height the product of other dimensions,
    // and treat "reduce" as a column reduction of the input matrix.
    const int64 width = ShapeUtil::ElementsIn(reduce->shape());
    // "width" can be zero, so don't do
    //   height = ShapeUtil::ElementsIn(input_shape) / width;
    int64 height = 1;
    for (int64 input_dim = 0; input_dim < ShapeUtil::Rank(input_shape);
         ++input_dim) {
      if (!std::count(input_dims_to_keep.begin(), input_dims_to_keep.end(),
                      input_dim)) {
        height *= input_shape.dimensions(input_dim);
      }
    }
    return EmitColumnReduction(height, width, reduce, input_shape, input_gen,
                               init_value_gen, reducer);
  } else {
    // Reduce the row dimension of a matrix or reduce dimension 0 and 2 in a
    // 3D tensor. The size of dimension 1 (the height) is the size of the
    // dimension to keep, the size of dimension 0 (the depth) is the product
    // of dimensions that are more major than the dimension to keep, and the
    // size of dimension 2 (the width) is the product of more minor
    // dimensions.
    int64 depth = 1;
    int64 width = 1;
    for (int64 input_dim = 0; input_dim < ShapeUtil::Rank(input_shape);
         ++input_dim) {
      if (PositionInContainer(input_shape.layout().minor_to_major(),
                              input_dim) >
          PositionInContainer(input_shape.layout().minor_to_major(),
                              input_dims_to_keep.back())) {
        depth *= input_shape.dimensions(input_dim);
      } else if (PositionInContainer(input_shape.layout().minor_to_major(),
                                     input_dim) <
                 PositionInContainer(input_shape.layout().minor_to_major(),
                                     input_dims_to_keep.front())) {
        width *= input_shape.dimensions(input_dim);
      }
    }
    const int64 height = ShapeUtil::ElementsIn(reduce->shape());
    return EmitRowReduction(depth, height, width, reduce, input_shape,
                            input_gen, init_value_gen, reducer);
  }
}

Status IrEmitterUnnested::HandleReduce(
    HloInstruction* reduce, HloInstruction* input, HloInstruction* init_value,
    tensorflow::gtl::ArraySlice<int64> dimensions_to_reduce,
    HloComputation* reducer) {
  // HandleReduce specializes reduction from a multi-dimensional array to a 1D
  // array. The specialized version requires an initializer thunk that
  // initializes the output array to the initial value of the reduce.
  if (IsReductionToVector(*reduce) &&
      // NVPTX backend can't do atomic cmpxchg any narrower than 32 bits
      32 <= primitive_util::BitWidth(reduce->shape().element_type())) {
    std::vector<std::unique_ptr<Thunk>> thunks;
    thunks.emplace_back(BuildKernelThunk(reduce));
    TF_RETURN_IF_ERROR(EmitInitializer(
        reduce, static_cast<KernelThunk*>(thunks.back().get())));
    bindings_.UnbindAllLocalIrValues();
    thunks.emplace_back(BuildKernelThunk(reduce));
    thunk_sequence_->emplace_back(
        MakeUnique<SequentialThunk>(std::move(thunks), reduce));
    return EmitReductionToVector(
        reduce, input->shape(),
        [this, input](const llvm_ir::IrArray::Index& index) {
          return GetIrArray(*input).EmitReadArrayElement(index, &ir_builder_);
        },
        [this, init_value](const llvm_ir::IrArray::Index& index) {
          return GetIrArray(*init_value)
              .EmitReadArrayElement(index, &ir_builder_);
        },
        dimensions_to_reduce, reducer);
  }

  thunk_sequence_->emplace_back(BuildKernelThunk(reduce));
  return IrEmitter::HandleReduce(reduce, input, init_value,
                                 dimensions_to_reduce, reducer);
}

Status IrEmitterUnnested::HandleTuple(
    HloInstruction* tuple,
    tensorflow::gtl::ArraySlice<HloInstruction*> operands) {
  bool all_tuple_elements_have_buffer = std::all_of(
      operands.begin(), operands.end(), [this](HloInstruction* tuple_element) {
        return ir_emitter_context_->buffer_assignment().HasTopLevelAllocation(
            tuple_element);
      });
  // Tuples (especially output tuples) can take too many tuple elements,
  // causing the kernel emitted exceeds the parameter space limit
  // (b/31336476). As an optimization, if all tuple elements have a buffer, we
  // collect their buffer addresses in a host array, and then copy that array
  // to the tuple's buffer.
  //
  // Some tuple elements (e.g. const or bitcast of const) might not have a
  // buffer -- their contents are stored in code. In that case, we fall back
  // to emitting kernels which have access to their buffer addresses in code.
  if (all_tuple_elements_have_buffer) {
    std::vector<BufferAllocation::Slice> tuple_element_buffers;
    for (const HloInstruction* tuple_element : operands) {
      tuple_element_buffers.push_back(GetAllocationSlice(*tuple_element));
    }
    thunk_sequence_->emplace_back(MakeUnique<TupleThunk>(
        tuple_element_buffers, GetAllocationSlice(*tuple), tuple));
    return Status::OK();
  }
  // If `inst` is a nested thunk that can be disassembled from the result tuple,
  // GpuExecutable will disassemble it and return it as part of the resultant
  // ShapedBuffer.
  if (has_hybrid_result_ &&
      ReachRootViaOnlyTuples(*tuple, *hlo_computation_->root_instruction())) {
    return Status::OK();
  }
  thunk_sequence_->emplace_back(BuildKernelThunk(tuple));
  return IrEmitter::HandleTuple(tuple, operands);
}

Status IrEmitterUnnested::HandleGetTupleElement(
    HloInstruction* get_tuple_element, HloInstruction* operand) {
  // GetTupleElement IR is emitted in the IR context of the user instruction,
  // and so we do not build a kernel for GetTupleElement instructions.
  return Status::OK();
}

Status IrEmitterUnnested::HandleSelectAndScatter(
    HloInstruction* select_and_scatter) {
  CHECK_EQ(select_and_scatter->operand_count(), 3);
  const auto* operand = select_and_scatter->operand(0);
  const auto* source = select_and_scatter->operand(1);
  const Window& window = select_and_scatter->window();
  PrimitiveType operand_element_type = operand->shape().element_type();
  const int64 rank = ShapeUtil::Rank(operand->shape());
  CHECK_EQ(rank, ShapeUtil::Rank(source->shape()));
  CHECK_EQ(rank, window.dimensions_size());

  {
    std::vector<std::unique_ptr<Thunk>> thunks;
    thunks.emplace_back(BuildKernelThunk(select_and_scatter));
    TF_RETURN_IF_ERROR(EmitInitializer(
        select_and_scatter, static_cast<KernelThunk*>(thunks.back().get())));
    bindings_.UnbindAllLocalIrValues();
    thunks.emplace_back(BuildKernelThunk(select_and_scatter));
    thunk_sequence_->emplace_back(
        MakeUnique<SequentialThunk>(std::move(thunks), select_and_scatter));
  }

  // TODO(b/31410564): Implement dilation rate for select-and-scatter.
  if (window_util::HasDilation(window)) {
    return Unimplemented(
        "Dilation for select-and-scatter not implemented on GPU. "
        "See b/31410564.");
  }

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
  auto loop_body_emitter =
      [=](const llvm_ir::IrArray::Index& source_index) -> Status {
    // Allocate space to keep the currently selected value, its index, and a
    // boolean flag if the value is initialized. The initialized_flag is set
    // false.
    llvm::Value* selected_value_address = llvm_ir::EmitAllocaAtFunctionEntry(
        llvm_ir::PrimitiveTypeToIrType(operand_element_type, &ir_builder_),
        "selected_value_address", &ir_builder_);
    llvm::Value* selected_index_address =
        llvm_ir::EmitAllocaAtFunctionEntryWithCount(
            ir_builder_.getInt64Ty(), ir_builder_.getInt32(rank),
            "selected_index_address", &ir_builder_);
    llvm::Value* initialized_flag_address = llvm_ir::EmitAllocaAtFunctionEntry(
        ir_builder_.getInt1Ty(), "initialized_flag_address", &ir_builder_);
    ir_builder_.CreateStore(ir_builder_.getInt1(false),
                            initialized_flag_address);

    // Create the inner loop to iterate over the window.
    llvm_ir::ForLoopNest window_loops(&ir_builder_);
    std::vector<int64> window_size;
    for (const auto& dim : window.dimensions()) {
      window_size.push_back(dim.size());
      CHECK_GT(dim.size(), 0);
    }
    const llvm_ir::IrArray::Index window_index = window_loops.AddLoopsForShape(
        ShapeUtil::MakeShape(operand_element_type, window_size), "window");
    llvm_ir::SetToFirstInsertPoint(window_loops.GetInnerLoopBodyBasicBlock(),
                                   &ir_builder_);

    // Compute the operand index to visit and evaluate the condition whether the
    // operand index is within the bounds. The unsigned comparison includes
    // checking whether the operand index >= 0.
    llvm_ir::IrArray::Index operand_index(source_index.size());
    llvm::Value* in_bounds_condition = ir_builder_.getInt1(true);
    for (int64 i = 0; i < rank; ++i) {
      llvm::Value* strided_index = ir_builder_.CreateNSWMul(
          source_index[i], ir_builder_.getInt64(window.dimensions(i).stride()));
      operand_index[i] = ir_builder_.CreateNSWSub(
          ir_builder_.CreateNSWAdd(strided_index, window_index[i]),
          ir_builder_.getInt64(window.dimensions(i).padding_low()));
      llvm::Value* index_condition = ir_builder_.CreateICmpULT(
          operand_index[i],
          ir_builder_.getInt64(ShapeUtil::GetDimension(operand->shape(), i)));
      in_bounds_condition =
          ir_builder_.CreateAnd(in_bounds_condition, index_condition);
    }
    CHECK(in_bounds_condition != nullptr);

    // Only need to do something if the operand index is within the bounds.
    // First check if the initialized_flag is set.
    llvm_ir::LlvmIfData if_in_bounds =
        llvm_ir::EmitIfThenElse(in_bounds_condition, "in-bounds", &ir_builder_);
    llvm_ir::SetToFirstInsertPoint(if_in_bounds.true_block, &ir_builder_);
    llvm_ir::LlvmIfData if_initialized = llvm_ir::EmitIfThenElse(
        ir_builder_.CreateLoad(initialized_flag_address), "initialized",
        &ir_builder_);

    // If the initialized_flag is false, initialize the selected value and index
    // with the currently visiting operand.
    llvm_ir::SetToFirstInsertPoint(if_initialized.false_block, &ir_builder_);
    const auto save_operand_index = [&](
        const llvm_ir::IrArray::Index& operand_index) {
      for (int64 i = 0; i < rank; ++i) {
        llvm::Value* selected_index_address_slot =
            ir_builder_.CreateInBoundsGEP(selected_index_address,
                                          {ir_builder_.getInt32(i)});
        ir_builder_.CreateStore(operand_index[i], selected_index_address_slot);
      }
    };
    llvm_ir::IrArray operand_array(GetIrArray(*operand));
    llvm::Value* operand_data =
        operand_array.EmitReadArrayElement(operand_index, &ir_builder_);
    ir_builder_.CreateStore(operand_data, selected_value_address);
    save_operand_index(operand_index);
    ir_builder_.CreateStore(ir_builder_.getInt1(true),
                            initialized_flag_address);

    // If the initialized_flag is true, call the `select` function to
    // potentially update the selected value and index with the currently
    // visiting operand.
    llvm_ir::SetToFirstInsertPoint(if_initialized.true_block, &ir_builder_);
    const Shape output_shape = ShapeUtil::MakeShape(PRED, {});
    llvm::Value* operand_address =
        operand_array.EmitArrayElementAddress(operand_index, &ir_builder_);
    llvm::Value* select_return_buffer = llvm_ir::EmitAllocaAtFunctionEntry(
        llvm_ir::PrimitiveTypeToIrType(PRED, &ir_builder_),
        "select_return_buffer", &ir_builder_);
    TF_RETURN_IF_ERROR(EmitCallToNestedComputation(
        *select_and_scatter->select(),
        {selected_value_address, operand_address}, select_return_buffer));
    llvm::Value* result = ir_builder_.CreateLoad(select_return_buffer);

    // If the 'select' function returns false, update the selected value and the
    // index to the currently visiting operand.
    llvm::Value* cond = ir_builder_.CreateICmpNE(
        result, llvm::ConstantInt::get(
                    llvm_ir::PrimitiveTypeToIrType(PRED, &ir_builder_), 0),
        "boolean_predicate");
    llvm_ir::LlvmIfData if_select_lhs =
        llvm_ir::EmitIfThenElse(cond, "if-select-lhs", &ir_builder_);
    llvm_ir::SetToFirstInsertPoint(if_select_lhs.false_block, &ir_builder_);
    ir_builder_.CreateStore(ir_builder_.CreateLoad(operand_address),
                            selected_value_address);
    save_operand_index(operand_index);

    // After iterating over the window elements, scatter the source element to
    // the selected index of the output. The value we store at the output
    // location is computed by calling the `scatter` function with the source
    // value and the current output value.
    llvm_ir::SetToFirstInsertPoint(window_loops.GetOuterLoopExitBasicBlock(),
                                   &ir_builder_);
    llvm_ir::IrArray::Index selected_index;
    for (int64 i = 0; i < rank; ++i) {
      llvm::Value* selected_index_address_slot = ir_builder_.CreateInBoundsGEP(
          selected_index_address, {ir_builder_.getInt32(i)});
      selected_index.push_back(
          ir_builder_.CreateLoad(selected_index_address_slot));
    }
    llvm::Value* source_value_address =
        GetIrArray(*source).EmitArrayElementAddress(source_index, &ir_builder_);
    llvm::Value* output_value_address =
        GetIrArray(*select_and_scatter)
            .EmitArrayElementAddress(selected_index, &ir_builder_);
    return EmitAtomicOperationForNestedComputation(
        *select_and_scatter->scatter(), output_value_address,
        source_value_address);
  };

  LaunchDimensions launch_dimensions = CalculateLaunchDimensions(
      source->shape(), ir_emitter_context_->device_description());
  UpdateLaunchDimensions(
      launch_dimensions,
      // IrEmitterUnnested implements kSelectAndScatter as a SequentialThunk
      // consisting of two thunks, an initializer KernelThunk that initializes
      // the output and another KernelThunk that accumulates the scattered
      // elements.
      static_cast<SequentialThunk*>(LastThunk())->thunks().back().get(),
      ir_emitter_context_->llvm_module());
  return ParallelLoopEmitter(loop_body_emitter, source->shape(),
                             launch_dimensions, &ir_builder_)
      .EmitLoop();
}

Status IrEmitterUnnested::HandleWhile(HloInstruction* xla_while,
                                      HloInstruction* init,
                                      HloComputation* condition,
                                      HloComputation* body) {
  TF_RET_CHECK(ShapeUtil::IsScalar(condition->root_instruction()->shape()) &&
               condition->root_instruction()->shape().element_type() == PRED)
      << "While condition computation must return bool";
  // Build ForThunk for conformant while loops, otherwise build WhileThunk.
  auto result = CanTransformWhileToFor(xla_while);
  if (result.ok()) {
    auto tuple = result.ConsumeValueOrDie();
    // loop_trip_count = (limit - start + increment - 1) / increment
    const int64 loop_trip_count =
        (std::get<1>(tuple) - std::get<0>(tuple) + std::get<2>(tuple) - 1) /
        std::get<2>(tuple);
    thunk_sequence_->emplace_back(BuildForThunk(xla_while, loop_trip_count));
    VLOG(3) << "Built ForThunk for while: " << xla_while->name();
  } else {
    thunk_sequence_->emplace_back(BuildWhileThunk(xla_while));
    VLOG(3) << "Built WhileThunk for while: " << xla_while->name()
            << " while-to-for transform status: " << result.status();
  }
  return Status::OK();
}

Status IrEmitterUnnested::HandleRng(HloInstruction* random,
                                    RandomDistribution distribution) {
  thunk_sequence_->push_back(BuildKernelThunk(random));
  return IrEmitter::HandleRng(random, distribution);
}

Status IrEmitterUnnested::HandleSelect(HloInstruction* select,
                                       HloInstruction* pred,
                                       HloInstruction* on_true,
                                       HloInstruction* on_false) {
  thunk_sequence_->push_back(BuildKernelThunk(select));
  return IrEmitter::HandleSelect(select, pred, on_true, on_false);
}

llvm::Function* IrEmitterUnnested::EmitBasePointersForHloAndItsOperands(
    const HloInstruction& hlo, std::vector<const HloInstruction*>* io_hlos) {
  const BufferAssignment& buffer_assignment =
      ir_emitter_context_->buffer_assignment();
  // GetTupleElement instructions are implemented by emitting IR that indexes
  // and loads the target tuple element pointer from its operand (possibly
  // recursively). For this reason, GetTupleElement instructions are associated
  // with their operand buffer in 'io_hlos' and 'non_io_hlos' below.
  std::vector<const HloInstruction*> non_io_hlos;
  for (const HloInstruction* operand : hlo.operands()) {
    const HloInstruction* to_lookup = LatestNonGteAncestor(operand);
    if (buffer_assignment.HasTopLevelAllocation(to_lookup) &&
        buffer_assignment.GetUniqueTopLevelSlice(to_lookup)
            .ConsumeValueOrDie()
            .allocation()
            ->IsInputOrOutput()) {
      io_hlos->push_back(operand);
    } else {
      non_io_hlos.push_back(operand);
    }
  }

  CHECK_NE(HloOpcode::kGetTupleElement, hlo.opcode());
  if (buffer_assignment.HasTopLevelAllocation(&hlo) &&
      buffer_assignment.GetUniqueTopLevelSlice(&hlo)
          .ConsumeValueOrDie()
          .allocation()
          ->IsInputOrOutput()) {
    io_hlos->push_back(&hlo);
  } else {
    non_io_hlos.push_back(&hlo);
  }

  llvm::Function* kernel = BuildKernelPrototype(hlo, *io_hlos);
  // bindings_ is reused because the bindings of kConstant to their underlying
  // llvm::Constant can be shared for all HLOs in this computation.
  bindings_.EmitBasePointersForHlos(*io_hlos, non_io_hlos);
  return kernel;
}

std::unique_ptr<Thunk> IrEmitterUnnested::BuildKernelThunk(
    const HloInstruction* inst) {
  std::vector<const HloInstruction*> io_hlos;
  llvm::Function* kernel =
      EmitBasePointersForHloAndItsOperands(*inst, &io_hlos);

  // Compute the input buffer indices.
  std::vector<BufferAllocation::Slice> io_buffers;
  for (const HloInstruction* io_hlo : io_hlos) {
    io_buffers.push_back(GetAllocationSlice(*LatestNonGteAncestor(io_hlo)));
  }

  // Create a KernelThunk that launches the kernel that implements "inst".
  return MakeUnique<KernelThunk>(io_buffers,
                                 llvm_ir::AsString(kernel->getName()), inst);
}

std::unique_ptr<Thunk> IrEmitterUnnested::BuildCopyThunk(
    const HloInstruction* inst) {
  const HloInstruction* operand = inst->operand(0);
  CHECK_EQ(HloOpcode::kConstant, operand->opcode());
  return MakeUnique<CopyThunk>(
      /*source_address=*/LiteralUtil::InternalData(operand->literal()),
      /*destination_buffer=*/GetAllocationSlice(*inst),
      /*mem_size=*/
      llvm_ir::ByteSizeOf(operand->shape(),
                          ir_emitter_context_->llvm_module()->getDataLayout()),
      inst);
}

std::unique_ptr<Thunk> IrEmitterUnnested::BuildGemmThunk(
    const HloInstruction* inst) {
  if (inst->opcode() == HloOpcode::kDot) {
    const HloInstruction* lhs = inst->operand(0);
    const HloInstruction* rhs = inst->operand(1);
    return MakeUnique<GemmThunk>(
        GetAllocationSlice(*lhs),   // The buffer assigned to LHS.
        GetAllocationSlice(*rhs),   // The buffer assigned to RHS.
        GetAllocationSlice(*inst),  // The output buffer.
        lhs->shape(),               // The shape of LHS.
        rhs->shape(),               // The shape of RHS.
        inst->shape(),              // The shape of the output.
        false,                      // Do not transpose LHS.
        false,                      // Do not transpose RHS.
        inst);
  }

  if (inst->opcode() == HloOpcode::kFusion) {
    const HloInstruction* dot = inst->fused_expression_root();
    DCHECK(dot->opcode() == HloOpcode::kDot);
    const HloInstruction* lhs_parameter = StripTranspose(*dot->operand(0));
    const HloInstruction* rhs_parameter = StripTranspose(*dot->operand(1));
    DCHECK(lhs_parameter->opcode() == HloOpcode::kParameter &&
           rhs_parameter->opcode() == HloOpcode::kParameter);
    const HloInstruction* lhs =
        inst->operand(lhs_parameter->parameter_number());
    const HloInstruction* rhs =
        inst->operand(rhs_parameter->parameter_number());

    return MakeUnique<GemmThunk>(
        GetAllocationSlice(*lhs),             // The buffer assigned to LHS.
        GetAllocationSlice(*rhs),             // The buffer assigned to RHS.
        GetAllocationSlice(*inst),            // The output buffer.
        lhs->shape(),                         // The shape of LHS.
        rhs->shape(),                         // The shape of RHS.
        inst->shape(),                        // The shape of the output.
        dot->operand(0)->IsRank2Transpose(),  // Transpose LHS.
        dot->operand(1)->IsRank2Transpose(),  // Trasnpose RHS.
        inst);
  }

  LOG(FATAL) << "Cannot build a GemmThunk for " << inst->ToString();
}

std::unique_ptr<Thunk> IrEmitterUnnested::BuildConvolutionThunk(
    const HloInstruction* inst) {
  const HloInstruction* lhs = inst->operand(0);
  const HloInstruction* rhs = inst->operand(1);
  if (inst->opcode() == HloOpcode::kConvolution) {
    // Forward covolution.
    return MakeUnique<ConvolutionThunk>(
        ConvolutionThunk::ConvolutionKind::kForward,
        /*input_buffer=*/GetAllocationSlice(*lhs),
        /*filter_buffer=*/GetAllocationSlice(*rhs),
        /*output_buffer=*/GetAllocationSlice(*inst),
        /*input_shape=*/lhs->shape(),
        /*filter_shape=*/rhs->shape(),
        /*output_shape=*/inst->shape(), inst->window(),
        inst->convolution_dimension_numbers(), inst);
  }

  // Backward filter convolution, which takes the input (activations) and the
  // gradients, and computes the filter.
  CHECK_EQ(HloOpcode::kFusion, inst->opcode());
  switch (inst->fusion_kind()) {
    case HloInstruction::FusionKind::kConvBackwardFilter:
      return MakeUnique<ConvolutionThunk>(
          ConvolutionThunk::ConvolutionKind::kBackwardFilter,
          /*input_buffer=*/GetAllocationSlice(*lhs),
          /*filter_buffer=*/GetAllocationSlice(*inst),
          /*output_buffer=*/GetAllocationSlice(*rhs),
          /*input_shape=*/lhs->shape(),
          /*filter_shape=*/inst->shape(),
          /*output_shape=*/rhs->shape(), inst->window(),
          inst->convolution_dimension_numbers(), inst);
    case HloInstruction::FusionKind::kConvBackwardInput:
      return MakeUnique<ConvolutionThunk>(
          ConvolutionThunk::ConvolutionKind::kBackwardInput,
          /*input_buffer=*/GetAllocationSlice(*inst),
          /*filter_buffer=*/GetAllocationSlice(*rhs),
          /*output_buffer=*/GetAllocationSlice(*lhs),
          /*input_shape=*/inst->shape(),
          /*filter_shape=*/rhs->shape(),
          /*output_shape=*/lhs->shape(), inst->window(),
          inst->convolution_dimension_numbers(), inst);
    default:
      LOG(FATAL) << "Not a convolution-fusion";
  }
}

Status IrEmitterUnnested::EmitInitializer(const HloInstruction* hlo,
                                          KernelThunk* thunk) {
  bool fused = HloOpcode::kFusion == hlo->opcode();

  const HloInstruction* inst = fused ? hlo->fused_expression_root() : hlo;
  CHECK(inst->opcode() == HloOpcode::kSelectAndScatter ||
        inst->opcode() == HloOpcode::kReduce);
  const HloInstruction* init_value = nullptr;
  switch (inst->opcode()) {
    case HloOpcode::kSelectAndScatter:
      init_value = inst->operand(2);
      break;
    case HloOpcode::kReduce:
      init_value = inst->operand(1);
      break;
    default:
      LOG(FATAL) << "Opcode " << inst->opcode()
                 << " should not need an initializer.";
  }

  if (fused && init_value->opcode() == HloOpcode::kParameter) {
    init_value = hlo->operand(init_value->parameter_number());
  }

  return EmitTargetElementLoopInThunk(
      *hlo,
      [=](const llvm_ir::IrArray::Index& index) {
        return GetIrArray(*init_value)
            .EmitReadArrayElement(index, &ir_builder_);
      },
      thunk);
}

namespace {

// Checks that all buffers used during while loop iteration share the same
// buffer allocation. This includes buffers for while result, while init
// operand, condition parameter, body parameter and body result.
// Returns OK on success, error status otherwise.
Status CheckWhileBuffersShareAllocation(
    const HloInstruction* xla_while,
    const BufferAssignment& buffer_assignment) {
  return ShapeUtil::ForEachSubshape(
      xla_while->shape(),
      [&buffer_assignment, &xla_while](const Shape& /*subshape*/,
                                       const ShapeIndex& index) -> Status {
        auto check = [&buffer_assignment](const HloInstruction* a,
                                          const HloInstruction* b,
                                          const ShapeIndex& index) -> Status {
          const BufferAllocation::Slice slice_a =
              buffer_assignment.GetUniqueSlice(a, index).ConsumeValueOrDie();
          const BufferAllocation::Slice slice_b =
              buffer_assignment.GetUniqueSlice(b, index).ConsumeValueOrDie();
          if (slice_a != slice_b) {
            return InternalError(
                "instruction %s %s does not share allocation with "
                "instruction %s %s",
                a->ToString().c_str(), slice_a.ToString().c_str(),
                b->ToString().c_str(), slice_b.ToString().c_str());
          }
          return Status::OK();
        };
        const HloInstruction* condition_parameter =
            xla_while->while_condition()->parameter_instruction(0);
        const HloComputation* body = xla_while->while_body();
        const HloInstruction* body_parameter = body->parameter_instruction(0);
        const HloInstruction* body_result = body->root_instruction();
        TF_RETURN_IF_ERROR(check(xla_while, xla_while->operand(0), index));
        TF_RETURN_IF_ERROR(check(xla_while, condition_parameter, index));
        TF_RETURN_IF_ERROR(check(xla_while, body_parameter, index));
        TF_RETURN_IF_ERROR(check(xla_while, body_result, index));
        return Status::OK();
      });
}

}  // namespace

std::unique_ptr<Thunk> IrEmitterUnnested::BuildWhileThunk(
    const HloInstruction* hlo) {
  // Check that all while-related buffers share an allocation.
  TF_CHECK_OK(CheckWhileBuffersShareAllocation(
      hlo, ir_emitter_context_->buffer_assignment()));

  // Generate thunk sequence for while 'condition'.
  HloComputation* condition = hlo->while_condition();
  IrEmitterUnnested ir_emitter_condition(hlo_module_config_, condition,
                                         /*has_hybrid_result=*/false,
                                         ir_emitter_context_);
  TF_CHECK_OK(condition->root_instruction()->Accept(&ir_emitter_condition));

  // Generate thunk sequence for while 'body'.
  HloComputation* body = hlo->while_body();
  IrEmitterUnnested ir_emitter_body(hlo_module_config_, body,
                                    false /* has_hybrid_result */,
                                    ir_emitter_context_);
  TF_CHECK_OK(body->root_instruction()->Accept(&ir_emitter_body));

  return MakeUnique<WhileThunk>(
      GetAllocationSlice(*condition->root_instruction()),  // cond result
      ir_emitter_condition.ConsumeThunkSequence(),
      ir_emitter_body.ConsumeThunkSequence(), hlo);
}

std::unique_ptr<Thunk> IrEmitterUnnested::BuildForThunk(
    const HloInstruction* hlo, const int64 loop_limit) {
  // Check that all while-related buffers share an allocation.
  TF_CHECK_OK(CheckWhileBuffersShareAllocation(
      hlo, ir_emitter_context_->buffer_assignment()));

  // Generate thunk sequence for while 'body' (will be used a For loop body).
  HloComputation* body = hlo->while_body();
  IrEmitterUnnested ir_emitter_body(hlo_module_config_, body,
                                    false /* has_hybrid_result */,
                                    ir_emitter_context_);
  TF_CHECK_OK(body->root_instruction()->Accept(&ir_emitter_body));

  return MakeUnique<ForThunk>(loop_limit,
                              ir_emitter_body.ConsumeThunkSequence(), hlo);
}

Status IrEmitterUnnested::EmitTargetElementLoopInThunk(
    const HloInstruction& hlo,
    const llvm_ir::ElementGenerator& element_generator, KernelThunk* thunk) {
  LaunchDimensions launch_dimensions = CalculateLaunchDimensions(
      hlo.shape(), ir_emitter_context_->device_description());
  UpdateLaunchDimensions(launch_dimensions, thunk,
                         ir_emitter_context_->llvm_module());
  // Otherwise, emit a parallel loop that computes the partition that each
  // thread is in charge of.
  return ParallelLoopEmitter(element_generator, GetIrArray(hlo),
                             launch_dimensions, &ir_builder_)
      .EmitLoop();
}

Status IrEmitterUnnested::EmitTargetElementLoop(
    const HloInstruction& hlo,
    const llvm_ir::ElementGenerator& element_generator) {
  CHECK(Thunk::Kind::kKernel == LastThunk()->kind());
  return EmitTargetElementLoopInThunk(hlo, element_generator,
                                      static_cast<KernelThunk*>(LastThunk()));
}

}  // namespace gpu
}  // namespace xla
