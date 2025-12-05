/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/backends/gpu/codegen/llvm/ir_emitter.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

// IWYU pragma: no_include "llvm/IR/Intrinsics.gen.inc"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "xla/backends/gpu/codegen/fusion_emitter.h"
#include "xla/backends/gpu/codegen/llvm/ir_emitter_nested.h"
#include "xla/backends/gpu/codegen/llvm/parallel_loop_emitter.h"
#include "xla/backends/gpu/codegen/llvm/sort_util.h"
#include "xla/backends/gpu/runtime/kernel_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_id.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/primitive_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/elemental_ir_emitter.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/llvm_ir/fused_ir_emitter.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/kernel_support_library.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/llvm_ir/loop_emitter.h"
#include "xla/service/llvm_ir/tuple_ops.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/tma_metadata.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla {

namespace gpu {

IrEmitter::IrEmitter(IrEmitterContext* ir_emitter_context, bool is_nested)
    : ir_emitter_context_(ir_emitter_context),
      module_(ir_emitter_context->llvm_module()),
      b_(module_->getContext()),
      bindings_(&b_, module_, is_nested) {}

absl::Status IrEmitter::DefaultAction(HloInstruction* hlo) {
  ElementalIrEmitter::HloToElementGeneratorMap operand_to_generator;
  for (const HloInstruction* operand : hlo->operands()) {
    operand_to_generator[operand] = [=](const llvm_ir::IrArray::Index& index) {
      return GetIrArray(*operand, *hlo)
          .EmitReadArrayElement(index, &b_, operand->name());
    };
  }
  return EmitTargetElementLoop(
      *hlo, ElementalIrEmitter(module_, &b_)
                .MakeElementGenerator(hlo, operand_to_generator));
}

absl::Status IrEmitter::HandleConstant(HloInstruction* constant) {
  return absl::OkStatus();
}

absl::Status IrEmitter::HandleAddDependency(HloInstruction* add_dependency) {
  VLOG(2) << "HandleAddDependency: " << add_dependency->ToString();
  const HloInstruction* operand = add_dependency->operand(0);
  // Add_Dependency is a no-op, but we still want to bind it to an llvm::Value
  // sometimes, e.g., when it's operand is a constant or a bitcast of a
  // constant.
  if (bindings_.BoundToIrValue(*operand)) {
    bindings_.BindHloToIrValue(*add_dependency, GetBasePointer(*operand));
  }
  return absl::OkStatus();
}

absl::Status IrEmitter::HandleGetTupleElement(
    HloInstruction* get_tuple_element) {
  auto operand = get_tuple_element->operand(0);
  CHECK(bindings_.BoundToIrValue(*operand));
  bindings_.BindHloToIrValue(
      *get_tuple_element,
      llvm_ir::EmitGetTupleElement(
          get_tuple_element->shape(), get_tuple_element->tuple_index(),
          // TODO(b/26344050): tighten the alignment here
          // based on the real element type.
          /*alignment=*/1, GetBasePointer(*operand),
          llvm_ir::ShapeToIrType(operand->shape(), module_->getContext()),
          &b_));
  return absl::OkStatus();
}

absl::Status IrEmitter::HandleSend(HloInstruction*) {
  return Unimplemented("Send is not implemented on GPU");
}

absl::Status IrEmitter::HandleSendDone(HloInstruction*) {
  return Unimplemented("Send-Done is not implemented on GPU");
}

absl::Status IrEmitter::HandleRecv(HloInstruction*) {
  return Unimplemented("Recv is not implemented on GPU");
}

absl::Status IrEmitter::HandleRecvDone(HloInstruction*) {
  return Unimplemented("Recv-done is not implemented on GPU");
}

absl::Status IrEmitter::HandleScatter(HloInstruction*) {
  return Unimplemented("Scatter is not implemented on GPUs.");
}

absl::Status IrEmitter::HandleTuple(HloInstruction* tuple) {
  std::vector<llvm::Value*> base_ptrs;
  for (const HloInstruction* operand : tuple->operands()) {
    base_ptrs.push_back(GetBasePointer(*operand));
  }
  llvm_ir::EmitTuple(GetIrArray(*tuple, *tuple), base_ptrs, &b_);
  return absl::OkStatus();
}

absl::Status IrEmitter::HandleConvolution(HloInstruction* convolution) {
  if (ShapeUtil::IsZeroElementArray(convolution->shape())) {
    // Emit no code for an empty output.
    return absl::OkStatus();
  }
  // TODO(b/31409998): Support convolution with dilation.
  return Unimplemented(
      "Hit a case for convolution that is not implemented on GPU.");
}

absl::Status IrEmitter::HandleFft(HloInstruction* fft) {
  if (ShapeUtil::IsZeroElementArray(fft->shape())) {
    // Emit no code for an empty output.
    return absl::OkStatus();
  }
  return Unimplemented("Hit a case for fft that is not implemented on GPU.");
}

absl::Status IrEmitter::HandleAllReduce(HloInstruction* crs) {
  return Unimplemented(
      "AllReduce cannot be nested inside of fusion, map, etc.");
}

absl::Status IrEmitter::HandleParameter(HloInstruction* parameter) {
  return absl::OkStatus();
}

absl::Status IrEmitter::HandleCall(HloInstruction* call) {
  std::vector<llvm::Value*> operand_addresses;
  for (HloInstruction* operand : call->operands()) {
    operand_addresses.push_back(GetBasePointer(*operand));
  }
  return CallNestedComputation(&b_, *ir_emitter_context_, module_,
                               *call->to_apply(), operand_addresses,
                               GetBasePointer(*call));
}

absl::Status IrEmitter::HandleCustomCall(HloInstruction*) {
  return Unimplemented("custom-call");
}

absl::Status IrEmitter::HandleInfeed(HloInstruction*) {
  // TODO(b/30467474): Implement infeed on GPU.
  return Unimplemented("Infeed is not supported on GPU.");
}

absl::Status IrEmitter::HandleOutfeed(HloInstruction*) {
  // TODO(b/34359662): Implement outfeed on GPU.
  return Unimplemented("Outfeed is not supported on GPU.");
}

absl::Status IrEmitter::HandleBatchNormInference(HloInstruction*) {
  return Unimplemented(
      "The GPU backend does not implement BatchNormInference directly.  It "
      "should be lowered before IR emission to HLO-soup using "
      "BatchNormRewriter.");
}

absl::Status IrEmitter::HandleBatchNormTraining(HloInstruction*) {
  return Unimplemented(
      "The GPU backend does not implement BatchNormTraining directly.  It "
      "should be lowered before IR emission to HLO-soup using "
      "BatchNormRewriter.");
}

absl::Status IrEmitter::HandleBatchNormGrad(HloInstruction*) {
  return Unimplemented(
      "The GPU backend does not implement BatchNormGrad directly.  It should "
      "be lowered before IR emission to HLO-soup using BatchNormRewriter.");
}

std::vector<llvm_ir::IrArray> IrEmitter::ConstructIrArrayForOutputs(
    const HloInstruction& hlo) {
  std::vector<llvm_ir::IrArray> output_arrays;
  if (hlo.shape().IsTuple()) {
    int64_t num_outputs = ShapeUtil::TupleElementCount(hlo.shape());
    output_arrays.reserve(num_outputs);
    for (int64_t i = 0; i < num_outputs; ++i) {
      output_arrays.push_back(GetIrArray(hlo, hlo, {i}));
    }
  } else {
    output_arrays.push_back(GetIrArray(hlo, hlo));
  }
  return output_arrays;
}

absl::Status IrEmitter::EmitTargetElementLoop(
    const HloInstruction& hlo, const llvm_ir::ElementGenerator& body_emitter) {
  return Internal("This should be unreachable");
}

void IrEmitter::BindFusionArguments(const HloInstruction* fusion,
                                    FusedIrEmitter* fused_emitter) {
  for (int i = 0; i < fusion->operand_count(); i++) {
    const HloInstruction* operand = fusion->operand(i);
    fused_emitter->BindGenerator(
        *fusion->fused_parameter(i),
        [this, operand, fusion](llvm_ir::IrArray::Index index) {
          return GetIrArray(*operand, *fusion)
              .EmitReadArrayElement(index, &b_, operand->name());
        });
  }
}

namespace {

struct KernelThunkInfo {
  std::vector<llvm_ir::IrArray> ir_arrays;
  std::unique_ptr<Thunk> thunk;
};

absl::StatusOr<KernelThunkInfo> BuildKernelThunkForNonFusionOp(
    llvm::Module* llvm_module, const HloInstruction* hlo,
    const BufferAssignment& buffer_assignment, ThunkId thunk_id,
    const se::DeviceDescription& gpu_device_info,
    const std::string& sanitized_kernel_name,

    IrEmitter& ir_emitter, const LaunchDimensions& launch_dimensions) {
  std::string suggested_kernel_name(hlo->name());

  TF_ASSIGN_OR_RETURN(auto kernel_arguments,
                      emitters::KernelArguments::Create(
                          buffer_assignment, GetDefaultBufferAlignment(), hlo));

  VLOG(3) << "Generating (without reuse check): " << suggested_kernel_name;

  TF_ASSIGN_OR_RETURN(
      llvm::Function * kernel,
      BuildKernelPrototype(llvm_module, gpu_device_info, suggested_kernel_name,
                           sanitized_kernel_name, kernel_arguments,
                           launch_dimensions, ir_emitter.builder()));

  auto thunk = std::make_unique<KernelThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(hlo, thunk_id),
      kernel->getName().str(), kernel_arguments, launch_dimensions,
      /*cluster_dim=*/std::nullopt,
      /*shmem_bytes=*/0,
      /*tma_metadata=*/se::gpu::TmaMetadata());

  std::vector<llvm_ir::IrArray> ir_arrays;
  ir_arrays.reserve(kernel_arguments.args().size());
  for (const auto& [kernel_argument, llvm_arg] :
       llvm::zip(kernel_arguments.args(), kernel->args())) {
    llvm::Type* ir_type =
        llvm_ir::ShapeToIrType(kernel_argument.shape(), llvm_arg.getContext());
    llvm_ir::IrArray ir_array(&llvm_arg, ir_type, kernel_argument.shape());

    if (!kernel_argument.written()) {
      ir_array.MarkInvariantOverWholeProgram(&llvm_arg.getContext());
    }
    ir_arrays.push_back(ir_array);
  }
  return {KernelThunkInfo{ir_arrays, std::move(thunk)}};
}

llvm::Value* CreateLoad(llvm::Value* address, llvm::Type* data_type,
                        int alignment_bytes, llvm::IRBuilderBase* b) {
  int data_bytes = data_type->getPrimitiveSizeInBits() /
                   primitive_util::BitWidth(PrimitiveType::U8);
  if (alignment_bytes == 0) {
    return b->CreateLoad(data_type, address);
  }

  int alignment_bitwidth =
      alignment_bytes * primitive_util::BitWidth(PrimitiveType::U8);

  llvm::Value* output = llvm::ConstantInt::get(data_type, 0);
  for (int offset_bytes = 0; offset_bytes < data_bytes;
       offset_bytes += alignment_bytes) {
    llvm::Value* offset_address = b->CreateConstInBoundsGEP1_32(
        b->getInt8Ty(), address, offset_bytes, "offset_address");
    llvm::Value* partial_value = b->CreateLoad(b->getIntNTy(alignment_bitwidth),
                                               offset_address, "partial_value");
    llvm::Value* zextd =
        b->CreateZExt(partial_value, output->getType(), "partial_value_zextd");
    llvm::Value* shifted = b->CreateShl(
        zextd, llvm::ConstantInt::get(b->getInt32Ty(), offset_bytes),
        "partial_input_shifted");
    output = b->CreateAdd(output, shifted, "output_updated");
  }
  return output;
}

void CreateStore(llvm::Value* data, llvm::Value* address, int alignment_bytes,
                 llvm::IRBuilderBase* b) {
  int data_bytes = data->getType()->getPrimitiveSizeInBits() /
                   primitive_util::BitWidth(PrimitiveType::U8);
  CHECK_GE(data_bytes, alignment_bytes);
  if (alignment_bytes == 0) {
    b->CreateStore(data, address);
    return;
  }

  int alignment_bitwidth =
      alignment_bytes * primitive_util::BitWidth(PrimitiveType::U8);

  for (int offset_bytes = 0; offset_bytes < data_bytes;
       offset_bytes += alignment_bytes) {
    llvm::Value* offset_address = b->CreateConstInBoundsGEP1_32(
        b->getInt8Ty(), address, offset_bytes, "offset_address");
    llvm::Value* shifted_partial = b->CreateTrunc(
        b->CreateLShr(data,
                      llvm::ConstantInt::get(b->getInt32Ty(), offset_bytes)),
        b->getIntNTy(alignment_bitwidth), "truncated_value");
    b->CreateStore(shifted_partial, offset_address);
  }
}

}  // namespace

absl::StatusOr<ThunkSequence> EmitBitonicSortLLVMIR(
    const HloSortInstruction* sort, llvm::Module* llvm_module,
    IrEmitterContext* ir_emitter_context) {
  std::string op_name(sort->name());

  // Copy of the main context with the local module.
  IrEmitterContext local_ir_emitter_context(
      &ir_emitter_context->hlo_module(),
      &ir_emitter_context->buffer_assignment(),
      &ir_emitter_context->execution_stream_assignment(),
      std::string(ir_emitter_context->platform_name()),
      ir_emitter_context->gpu_device_info(), ir_emitter_context->mlir_context(),
      llvm_module, ir_emitter_context->llvm_module_constants(),
      ir_emitter_context->emit_kernels());

  IrEmitter ir_emitter(&local_ir_emitter_context, /*nested=*/false);

  int64_t dimension_to_sort = sort->sort_dimension();
  const Shape& keys_shape = sort->operand(0)->shape();
  uint64_t dimension_to_sort_bound = keys_shape.dimensions(dimension_to_sort);
  int64_t num_stages = Log2Ceiling(dimension_to_sort_bound);
  VLOG(2) << op_name << " requires " << num_stages << " stages.";
  CHECK_GE(1ULL << num_stages, dimension_to_sort_bound);
  CHECK_LT(1ULL << (num_stages - 1), dimension_to_sort_bound);

  // Naive C++ code for the outer loops:
  //
  // for (int64_t stage = 0; stage <
  // Log2Ceiling(dimension_to_sort_bound);
  //     ++stage) {
  //   int64_t first_xor_mask = (1LL << (stage + 1)) - 1;
  //   SortInPlace(first_xor_mask);
  //   for (int64_t mask = stage - 1; mask >= 0; --mask) {
  //     int64_t later_xor_mask = 1LL << mask;
  //     SortInPlace(later_xor_mask);
  //   }
  // }
  //
  // This follows the alternative representation of the algorithm
  // described on Wikipedia:
  // https://en.wikipedia.org/wiki/Bitonic_sorter
  //
  // Each mask specifies how to derive from one position in the
  // array the position with which it should be compared (we
  // calculate the xor of the position with the mask). As an
  // optimization, we can move the 'mask' loop to inside the
  // sorting/comparison loop if the comparisons happen within a
  // small block of the array. To make this work, we collect all
  // consecutive masks that are smaller than our chosen power of 2
  // tile size, and pass them to SortInPlace. Each block then
  // processes one tile of data.

  const uint64_t kUnrollFactor = 4;
  // Determine the total element size of all sort operands. We need to choose a
  // tile size such that we have enough shared memory to store a tile of
  // elements from each operand.
  uint64_t total_element_size = 0;
  for (int64_t i = 0; i < sort->operand_count(); ++i) {
    total_element_size += ShapeUtil::ByteSizeOfPrimitiveType(
        sort->operand(i)->shape().element_type());
  }
  const uint64_t kMaxSharedMemoryPerBlock =
      ir_emitter_context->gpu_device_info().shared_memory_per_block();
  uint64_t max_tile_size_fitting_into_shared_memory =
      kMaxSharedMemoryPerBlock / total_element_size;
  const uint64_t kMaxThreadsPerBlock =
      ir_emitter_context->gpu_device_info().threads_per_block_limit();
  // Choose the tile size based on actual amount of elements to sort, the amount
  // of shared memory available, and the maximum number of threads per block.
  uint64_t tile_size =
      std::min(std::min(kMaxThreadsPerBlock * kUnrollFactor,
                        max_tile_size_fitting_into_shared_memory),
               uint64_t{1} << num_stages);
  // The tile size needs to be a power of 2.
  tile_size = uint64_t{1} << Log2Floor(tile_size);

  // If we cannot combine several xor masks together, we don't use
  // tiling, so we calculate the standard launch dimensions for the
  // shape. However we only need to iterate through ~half of the
  // dimension to sort (rounded up to the next highest power of 2),
  // because each iteration compares one pair of elements.
  Shape standard_iteration_shape = keys_shape;
  uint64_t standard_num_iterations_in_sort_dim = 1ULL << (num_stages - 1);
  standard_iteration_shape.set_dimensions(
      dimension_to_sort,
      CeilOfRatio(standard_num_iterations_in_sort_dim, kUnrollFactor));

  LaunchDimensions standard_launch_dimensions = CalculateLaunchDimensions(
      standard_iteration_shape, ir_emitter_context->gpu_device_info());

  // Calculate the launch dimensions for the case where we use
  // tiling. We split the dimension that should be sorted into tiles
  // of size 'tile_size'. This means we first need to round
  // 'dimension_to_sort_bound' up to be a multiple of the tile size.
  uint64_t rounded_bound = RoundUpTo(dimension_to_sort_bound, tile_size);
  Shape iteration_shape = keys_shape;

  // We iterate through the element pairs that should be compared.
  uint64_t num_iterations_in_sort_dim =
      CeilOfRatio(rounded_bound, kUnrollFactor);
  iteration_shape.set_dimensions(dimension_to_sort, num_iterations_in_sort_dim);
  uint64_t num_iterations = ShapeUtil::ElementsIn(iteration_shape);

  // For correctness reasons we need exactly `tile_size` / `kUnrollFactor` many
  // threads per block. Each thread is responsible for copying
  // exactly `kUnrollFactor` many adjacent elements into shared memory, and then
  // does `kUnrollFactor` / 2 many comparisons of two elements taken from shared
  // memory.
  const uint64_t kThreadsPerBlock =
      std::max(uint64_t{1}, tile_size / kUnrollFactor);

  uint64_t num_blocks = CeilOfRatio(num_iterations, kThreadsPerBlock);
  LaunchDimensions tiled_launch_dimensions(num_blocks, kThreadsPerBlock);
  VLOG(2) << absl::StreamFormat("%s launch dims: %d blocks, %d threads/block",
                                op_name, num_blocks, kThreadsPerBlock);
  ThunkSequence thunks;
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
        KernelThunkInfo kernel_thunk_info,
        BuildKernelThunkForNonFusionOp(
            llvm_module, sort, ir_emitter_context->buffer_assignment(),
            ir_emitter_context->GetNextThunkId(),
            ir_emitter_context->gpu_device_info(),
            ir_emitter_context->GetSanitizedUniqueName(op_name), ir_emitter,
            launch_dimensions));
    thunks.push_back(std::move(kernel_thunk_info.thunk));

    // The first `operand_count()` elements of `ir_arrays` are the input
    // operands and the rest are the output arrays. Inputs are aliases with
    // outputs, so we need to pass only the outputs to the in-place sort kernel.
    auto output_arrays_span =
        absl::Span<const llvm_ir::IrArray>(kernel_thunk_info.ir_arrays)
            .subspan(sort->operand_count());

    auto* comparator = sort->called_computations().front();
    auto* builder = ir_emitter.builder();
    return llvm_ir::EmitSortInPlace(
        dimension_to_sort, output_arrays_span, llvm_ir::IrName(op_name),
        xor_masks, ir_emitter.builder(), launch_dimensions,
        xor_masks.size() > 1 ? num_iterations_in_sort_dim
                             : standard_num_iterations_in_sort_dim,
        tile_size, kUnrollFactor,
        [&](absl::Span<llvm::Value* const> operands, llvm::Value* output) {
          return CallNestedComputation(builder, local_ir_emitter_context,
                                       llvm_module, *comparator, operands,
                                       output);
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
      if (xor_mask >= tile_size) {
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
  return thunks;
}

// Input = {dynamic array(with dynamic dimension meta data at the
// end)} Output = {static array, dynamic_dim0, dynamic_dim1}
absl::StatusOr<ThunkSequence> EmitPadToStaticLLVMIR(
    const HloCustomCallInstruction* hlo, llvm::Module* llvm_module,
    IrEmitterContext* ir_emitter_context) {
  std::string ir_name = std::string(hlo->name());

  // Copy of the main context with the local module.
  IrEmitterContext local_ir_emitter_context(
      &ir_emitter_context->hlo_module(),
      &ir_emitter_context->buffer_assignment(),
      &ir_emitter_context->execution_stream_assignment(),
      std::string(ir_emitter_context->platform_name()),
      ir_emitter_context->gpu_device_info(), ir_emitter_context->mlir_context(),
      llvm_module, ir_emitter_context->llvm_module_constants(),
      ir_emitter_context->emit_kernels());

  IrEmitter ir_emitter(&local_ir_emitter_context, /*nested=*/false);

  constexpr int kUnrollFactor = 1;
  const Shape& input_shape = hlo->operand(0)->shape();

  LaunchDimensions launch_dimensions = CalculateLaunchDimensions(
      input_shape, ir_emitter_context->gpu_device_info(), {kUnrollFactor});

  TF_ASSIGN_OR_RETURN(
      KernelThunkInfo kernel_thunk_info,
      BuildKernelThunkForNonFusionOp(
          llvm_module, hlo, ir_emitter_context->buffer_assignment(),
          ir_emitter_context->GetNextThunkId(),
          ir_emitter_context->gpu_device_info(),
          ir_emitter_context->GetSanitizedUniqueName(ir_name), ir_emitter,
          launch_dimensions));
  ThunkSequence thunk_sequence;
  thunk_sequence.push_back(std::move(kernel_thunk_info.thunk));

  const llvm_ir::IrArray& source_array = kernel_thunk_info.ir_arrays[0];
  const llvm_ir::IrArray& output_array = kernel_thunk_info.ir_arrays[1];
  auto output_dim_arrays =
      absl::Span<const llvm_ir::IrArray>(kernel_thunk_info.ir_arrays)
          .subspan(2);

  llvm::Type* index_ty = GetIndexTypeForKernel(
      hlo, launch_dimensions.launch_bound(), ir_emitter.builder());

  // pseudo code for PadToStatic on a 2d array
  //   int* source_array = args[0];
  //   int* dest_array = args[1];
  llvm::Value* source_buffer = source_array.GetBasePointer();

  // TODO(jurahul): input_shape here is the static shape of the
  // input (which has a dynamic shape in XLA). Currently, we are
  // mapping that to a static shaped memref. When we change that to
  // a more appropriate representation in MLIR, fix this code to
  // correctly deduce the static shape backing the dynamically
  // shaped memref.
  int64_t raw_data_size = ShapeUtil::ByteSizeOf(input_shape);

  //   int* dyn_dim0_size = source_array + meta_data_offset;
  //   int* dyn_dim1_size = source_array + meta_data_offset +
  //   sizeof(int);
  std::vector<llvm::Value*> dynamic_dims;
  int alignment = raw_data_size % sizeof(int32_t);
  std::vector<ShapeUtil::IndexedShape> output_shapes =
      ShapeUtil::GetLeafShapes(hlo->shape());

  for (int64_t i = 1; i < output_shapes.size(); ++i) {
    // Dynamic size of each dimension is attached at the end of the
    // source array(operand(0)). We need to extract these value.
    const Shape& dim_shape = output_shapes[i].shape;
    TF_RET_CHECK(Shape::Equal()(dim_shape, ShapeUtil::MakeScalarShape(S32)));

    const int64_t dim_index = i - 1;
    llvm::Value* metadata = ir_emitter.builder()->CreateConstInBoundsGEP1_32(
        ir_emitter.builder()->getInt8Ty(), source_buffer,
        raw_data_size + dim_index * sizeof(int32_t));
    llvm::Value* dyn_dim_size =
        CreateLoad(metadata, ir_emitter.builder()->getInt32Ty(), alignment,
                   ir_emitter.builder());
    dynamic_dims.push_back(dyn_dim_size);
  }

  // only one thread need to store the dynamic index
  //   int thread_id = GetThreadId();
  //   int block_id = GetBlockId();
  //   if (thread_id == 0 && block_id == 0) {
  //     *output[1] = *dyn_dim0_size;
  //     *output[2] = *dyn_dim1_size;
  //   }
  KernelSupportLibrary{ir_emitter.builder()}.If(
      "is_thread_0", IsBlock0Thread0(ir_emitter.builder()), [&] {
        for (int64_t i = 1; i < output_shapes.size(); ++i) {
          const int64_t dim_index = i - 1;
          llvm::Value* dest_dim_size_address =
              output_dim_arrays[dim_index].GetBasePointer();
          // output[i] stores dynamic_dim_(i-1)
          CreateStore(dynamic_dims[dim_index], dest_dim_size_address, alignment,
                      ir_emitter.builder());
        }
      });

  //     int dyn_element_total = 1;
  //     dyn_element_total *= *dyn_dim0_size;
  //     dyn_element_total *= *dyn_dim1_size;
  llvm::Value* dyn_element_total = llvm::ConstantInt::get(index_ty, 1);
  for (llvm::Value* dynamic_dim : dynamic_dims) {
    dyn_element_total = ir_emitter.builder()->CreateMul(
        dyn_element_total,
        ir_emitter.builder()->CreateIntCast(dynamic_dim,
                                            dyn_element_total->getType(),
                                            /*isSigned=*/true),
        /*Name=*/"dyn_element_total_pad");
  }

  //   linear_index = block_id * threads_per_block + thread_id;
  //   if (linear_index < max_num_element) {
  //     Index static_index =
  //         delinerized(linerized_index, static_dim0_size,
  //         static_dim1_size);
  //     if (linerized_index < dyn_element_total) {
  //       Index dyn_index =
  //           delinerized(linerized_index, *dyn_dim0_size,
  //           *dyn_dim1_size);
  //       dest_array[dyn_index.dim0][dyn_index.dim1] =
  //           source_array[static_index.dim0][static_index.dim1];
  //     }
  //   }
  llvm_ir::BodyEmitter body_generator =
      [&](const llvm_ir::IrArray::Index& array_index) -> absl::Status {
    llvm::Value* linearIndex =
        array_index.Linearize(input_shape.dimensions(), ir_emitter.builder());
    auto if_in_dyn_bounds = llvm_ir::EmitIfThenElse(
        ir_emitter.builder()->CreateICmpULT(linearIndex, dyn_element_total),
        llvm_ir::IrName(ir_name, "in_dyn_bounds"), ir_emitter.builder(), false);
    // Set IR builder insertion point to the body of the if
    // structure.
    llvm_ir::SetToFirstInsertPoint(if_in_dyn_bounds.true_block,
                                   ir_emitter.builder());
    llvm_ir::IrArray::Index dyn_index(linearIndex, input_shape,
                                      absl::MakeSpan(dynamic_dims),
                                      ir_emitter.builder());
    output_array.EmitWriteArrayElement(
        dyn_index,
        source_array.EmitReadArrayElement(array_index, ir_emitter.builder(),
                                          /*name=*/""),
        ir_emitter.builder(),
        /*use_linear_index=*/false);
    return absl::OkStatus();
  };

  const Shape& data_shape = hlo->shape().tuple_shapes(0);
  TF_RETURN_IF_ERROR(ParallelLoopEmitter(body_generator, data_shape,
                                         launch_dimensions,
                                         ir_emitter.builder(), {kUnrollFactor})
                         .EmitLoop(ir_name, index_ty));
  return thunk_sequence;
}

// Input = {dynamic array(with dynamic dimension meta data at the
// end)} Output = {static array, dynamic_dim0, dynamic_dim1}
absl::StatusOr<ThunkSequence> EmitSliceToDynamicLLVMIR(
    const HloCustomCallInstruction* hlo, llvm::Module* llvm_module,
    IrEmitterContext* ir_emitter_context) {
  std::string ir_name = std::string(hlo->name());

  // Copy of the main context with the local module.
  IrEmitterContext local_ir_emitter_context(
      &ir_emitter_context->hlo_module(),
      &ir_emitter_context->buffer_assignment(),
      &ir_emitter_context->execution_stream_assignment(),
      std::string(ir_emitter_context->platform_name()),
      ir_emitter_context->gpu_device_info(), ir_emitter_context->mlir_context(),
      llvm_module, ir_emitter_context->llvm_module_constants(),
      ir_emitter_context->emit_kernels());

  IrEmitter ir_emitter(&local_ir_emitter_context, /*nested=*/false);
  // TODO(jurahul): Create an op to represent SliceToDynamic.
  constexpr int kUnrollFactor = 1;
  const Shape& input_shape = hlo->operand(0)->shape();

  LaunchDimensions launch_dimensions = CalculateLaunchDimensions(
      input_shape, ir_emitter_context->gpu_device_info(), {kUnrollFactor});
  llvm::Type* index_ty = GetIndexTypeForKernel(
      hlo, launch_dimensions.launch_bound(), ir_emitter.builder());
  TF_ASSIGN_OR_RETURN(
      KernelThunkInfo kernel_thunk_info,
      BuildKernelThunkForNonFusionOp(
          llvm_module, hlo, ir_emitter_context->buffer_assignment(),
          ir_emitter_context->GetNextThunkId(),
          ir_emitter_context->gpu_device_info(),
          ir_emitter_context->GetSanitizedUniqueName(ir_name), ir_emitter,
          launch_dimensions));
  ThunkSequence thunk_sequence;
  thunk_sequence.push_back(std::move(kernel_thunk_info.thunk));

  const Shape& data_shape = ShapeUtil::MakeStaticShape(hlo->shape());
  TF_RET_CHECK(data_shape.IsArray());

  // TODO(jurahul): data_shape here is the static shape of the
  // output (which has a dynamic shape in XLA). Currently, we are
  // mapping that to a static shaped memref. When we change that to
  // a more appropriate representation in MLIR, fix this code to
  // correctly deduce the static shape backing the dynamically
  // shaped memref.

  // calculate the location where metadata needs to be inserted
  //   int* dyn_dim0_size = dest_array + meta_data_offset;
  //   int* dyn_dim1_size = dest_array + meta_data_offset +
  //   sizeof(int);
  int32_t raw_data_size = ShapeUtil::ByteSizeOf(data_shape);

  // pseudo code for sliceToDynamic on a 2d array
  //   int* source_array = args[0];
  //   int* dest_array = args.back();
  const auto& ir_arrays = kernel_thunk_info.ir_arrays;
  const llvm_ir::IrArray& data_array = ir_arrays.back();
  llvm::Value* dest_buffer = data_array.GetBasePointer();

  // Load dynamic dimensions from memory.
  std::vector<llvm::Value*> dynamic_dims;
  int alignment = raw_data_size % sizeof(int32_t);
  for (int64_t i = 1; i < hlo->operand_count(); ++i) {
    llvm::Value* source_buffer = ir_arrays[i].GetBasePointer();
    llvm::Type* source_buffer_pointee_type = ir_arrays[i].GetBasePointeeType();
    llvm::LoadInst* dyn_dim_size = ir_emitter.builder()->CreateLoad(
        source_buffer_pointee_type, source_buffer, "dyn_dim_size");
    dynamic_dims.push_back(dyn_dim_size);
  }

  // only one thread need to store the dynamic index
  //   int thread_id = GetThreadId();
  //   int block_id = GetBlockId();
  //   if (thread_id == 0 && block_id == 0) {
  //     *dyn_dim0_size = *output[1];
  //     *dyn_dim1_size = *output[2];
  //   }
  KernelSupportLibrary{ir_emitter.builder()}.If(
      "is_thread_0", IsBlock0Thread0(ir_emitter.builder()), [&] {
        for (int64_t i = 1; i < hlo->operand_count(); ++i) {
          const int64_t dim_index = i - 1;
          llvm::Value* metadata =
              ir_emitter.builder()->CreateConstInBoundsGEP1_32(
                  ir_emitter.builder()->getInt8Ty(), dest_buffer,
                  raw_data_size + dim_index * sizeof(int32_t));
          // output[i] stores dynamic_dim_(i-1)
          CreateStore(dynamic_dims[dim_index], metadata, alignment,
                      ir_emitter.builder());
        }
      });

  //     int dyn_element_total = 1;
  //     dyn_element_total *= dyn_dim0_size;
  //     dyn_element_total *= dyn_dim1_size;
  llvm::Value* dyn_element_total = llvm::ConstantInt::get(index_ty, 1);
  for (llvm::Value* dynamic_dim : dynamic_dims) {
    dyn_element_total = ir_emitter.builder()->CreateMul(
        dyn_element_total,
        ir_emitter.builder()->CreateIntCast(dynamic_dim,
                                            dyn_element_total->getType(),
                                            /*isSigned=*/true),
        /*Name=*/"dyn_element_total_slice");
  }

  //   linear_index = block_id * threads_per_block + thread_id;
  //   if (linear_index < max_num_element) {
  //     Index static_index =
  //         delinerized(linerized_index, static_dim0_size,
  //         static_dim1_size);
  //     if (linerized_index < dyn_element_total) {
  //       Index dyn_index =
  //           delinerized(linerized_index, *dyn_dim0_size,
  //           *dyn_dim1_size);
  //       dest_array[static_index.dim0][static_index.di] =
  //           source_array[dyn_index.dim0][dyn_index.dim1];
  //     }
  //   }
  llvm_ir::BodyEmitter body_generator =
      [&](const llvm_ir::IrArray::Index& array_index) -> absl::Status {
    llvm::Value* linearIndex =
        array_index.Linearize(input_shape.dimensions(), ir_emitter.builder());
    auto if_in_dyn_bounds = llvm_ir::EmitIfThenElse(
        ir_emitter.builder()->CreateICmpULT(linearIndex, dyn_element_total),
        llvm_ir::IrName(ir_name, "in_dyn_bounds"), ir_emitter.builder(), false);
    // Set IR builder insertion point to the body of the if
    // structure.
    llvm_ir::SetToFirstInsertPoint(if_in_dyn_bounds.true_block,
                                   ir_emitter.builder());
    llvm_ir::IrArray::Index dyn_index(linearIndex, input_shape,
                                      absl::MakeSpan(dynamic_dims),
                                      ir_emitter.builder());

    data_array.EmitWriteArrayElement(
        array_index,
        ir_arrays[0].EmitReadArrayElement(dyn_index, ir_emitter.builder(),
                                          /*name=*/"",
                                          /*use_linear_index=*/false),
        ir_emitter.builder());
    return absl::OkStatus();
  };

  TF_RETURN_IF_ERROR(ParallelLoopEmitter(body_generator, data_shape,
                                         launch_dimensions,
                                         ir_emitter.builder(), {kUnrollFactor})
                         .EmitLoop(ir_name, index_ty));
  return thunk_sequence;
}

absl::StatusOr<ThunkSequence> EmitRngGetAndUpdateStateLLVMIR(
    const HloRngGetAndUpdateStateInstruction* hlo, llvm::Module* llvm_module,
    IrEmitterContext* ir_emitter_context) {
  std::string ir_name = std::string(hlo->name());

  // Copy of the main context with the local module.
  IrEmitterContext local_ir_emitter_context(
      &ir_emitter_context->hlo_module(),
      &ir_emitter_context->buffer_assignment(),
      &ir_emitter_context->execution_stream_assignment(),
      std::string(ir_emitter_context->platform_name()),
      ir_emitter_context->gpu_device_info(), ir_emitter_context->mlir_context(),
      llvm_module, ir_emitter_context->llvm_module_constants(),
      ir_emitter_context->emit_kernels());

  IrEmitter ir_emitter(&local_ir_emitter_context, /*nested=*/false);

  auto& b = *ir_emitter.builder();
  // Emit a kernel to increment the global state for Philox RNG
  // algorithm.
  TF_ASSIGN_OR_RETURN(
      KernelThunkInfo kernel_thunk_info,
      BuildKernelThunkForNonFusionOp(
          llvm_module, hlo, ir_emitter_context->buffer_assignment(),
          ir_emitter_context->GetNextThunkId(),
          ir_emitter_context->gpu_device_info(),
          ir_emitter_context->GetSanitizedUniqueName(ir_name), ir_emitter,
          LaunchDimensions()));
  ThunkSequence thunk_sequence;
  thunk_sequence.push_back(std::move(kernel_thunk_info.thunk));

  auto& ir_arrays = kernel_thunk_info.ir_arrays;
  llvm::Value* old_state =
      llvm_ir::RngGetAndUpdateState(hlo->delta(), llvm_module, &b);
  llvm::Value* output_address = ir_arrays[0].EmitArrayElementAddress(
      llvm_ir::IrArray::Index(
          /*linear=*/b.getInt64(0), hlo->shape(), &b),
      &b, "rng_state_address");
  b.CreateStore(old_state, output_address);
  return thunk_sequence;
}

}  // namespace gpu
}  // namespace xla
