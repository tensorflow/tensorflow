/* Copyright 2025 The OpenXLA Authors.

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
#include "xla/backends/gpu/codegen/sort.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/functional/function_ref.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/IR/ArithAttributes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "xla/backends/gpu/codegen/emitters/ir/xla_gpu_ops.h"
#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/backends/gpu/codegen/kernel_compiler.h"
#include "xla/backends/gpu/codegen/kernels/custom_kernel.h"
#include "xla/backends/gpu/codegen/kernels/ptx_custom_kernel.h"
#include "xla/backends/gpu/runtime/custom_kernel_thunk.h"
#include "xla/backends/gpu/runtime/device_to_device_copy_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/codegen/emitters/computation_partitioner.h"
#include "xla/codegen/emitters/elemental_hlo_to_mlir.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/emitters/kernel_api_builder.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/codegen/emitters/type_util.h"
#include "xla/codegen/llvm_kernel_source.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/future.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

namespace arith = ::mlir::arith;
namespace scf = ::mlir::scf;
namespace tensor = ::mlir::tensor;

using ::llvm::SmallVector;
using ::mlir::ImplicitLocOpBuilder;
using ::mlir::MLIRContext;
using ::mlir::OpBuilder;
using ::mlir::RankedTensorType;
using ::mlir::Type;
using ::mlir::TypeRange;
using ::mlir::Value;
using ::mlir::ValueRange;
using ::mlir::func::FuncOp;

constexpr uint64_t kBitonicSortUnrollFactor = 4;

SmallVector<Value> EmitIfElse(
    ImplicitLocOpBuilder& b, Value condition, TypeRange result_types,
    absl::FunctionRef<SmallVector<Value>(ImplicitLocOpBuilder&)> then_fn,
    absl::FunctionRef<SmallVector<Value>(ImplicitLocOpBuilder&)> else_fn) {
  auto if_op =
      scf::IfOp::create(b, result_types, condition, /*withElseRegion=*/true);

  OpBuilder then_b = if_op.getThenBodyBuilder();
  ImplicitLocOpBuilder implicit_then_b(b.getLoc(), then_b);
  scf::YieldOp::create(implicit_then_b, then_fn(implicit_then_b));

  OpBuilder else_b = if_op.getElseBodyBuilder();
  ImplicitLocOpBuilder implicit_else_b(b.getLoc(), else_b);
  scf::YieldOp::create(implicit_else_b, else_fn(implicit_else_b));

  return SmallVector<Value>(if_op.getResults().begin(),
                            if_op.getResults().end());
}

absl::StatusOr<SmallVector<Value>> EmitIfElseWithStatus(
    ImplicitLocOpBuilder& b, Value condition, TypeRange result_types,
    absl::FunctionRef<absl::StatusOr<SmallVector<Value>>(ImplicitLocOpBuilder&)>
        then_fn,
    absl::FunctionRef<absl::StatusOr<SmallVector<Value>>(ImplicitLocOpBuilder&)>
        else_fn) {
  auto if_op =
      scf::IfOp::create(b, result_types, condition, /*withElseRegion=*/true);

  OpBuilder then_b = if_op.getThenBodyBuilder();
  ImplicitLocOpBuilder implicit_then_b(b.getLoc(), then_b);
  ABSL_ASSIGN_OR_RETURN(SmallVector<Value> then_results, then_fn(implicit_then_b));
  scf::YieldOp::create(implicit_then_b, then_results);

  OpBuilder else_b = if_op.getElseBodyBuilder();
  ImplicitLocOpBuilder implicit_else_b(b.getLoc(), else_b);
  ABSL_ASSIGN_OR_RETURN(SmallVector<Value> else_results, else_fn(implicit_else_b));
  scf::YieldOp::create(implicit_else_b, else_results);

  return SmallVector<Value>(if_op.getResults().begin(),
                            if_op.getResults().end());
}

SmallVector<Value> EmitUpdateIf(
    ImplicitLocOpBuilder& b, Value condition, ValueRange values,
    absl::FunctionRef<SmallVector<Value>(ImplicitLocOpBuilder&)>
        updated_values_fn) {
  return EmitIfElse(b, condition, TypeRange(values), updated_values_fn,
                    [&](ImplicitLocOpBuilder&) {
                      return SmallVector<Value>(values.begin(), values.end());
                    });
}

absl::StatusOr<SmallVector<Value>> EmitUpdateIfWithStatus(
    ImplicitLocOpBuilder& b, Value condition, ValueRange values,
    absl::FunctionRef<absl::StatusOr<SmallVector<Value>>(ImplicitLocOpBuilder&)>
        updated_values_fn) {
  return EmitIfElseWithStatus(
      b, condition, TypeRange(values), updated_values_fn,
      [&](ImplicitLocOpBuilder&) -> absl::StatusOr<SmallVector<Value>> {
        return SmallVector<Value>(values.begin(), values.end());
      });
}

using ReadElementFn = absl::FunctionRef<absl::StatusOr<Value>(
    ImplicitLocOpBuilder&, ValueRange /*tensors*/, int64_t /*operand*/,
    Value /*index*/)>;
using WriteElementFn = absl::FunctionRef<Value(
    ImplicitLocOpBuilder&, Value /*tensor*/, int64_t /*operand*/,
    Value /*index*/, Value /*value*/)>;

absl::StatusOr<SmallVector<Value>> EmitCompareLoopBody(
    ImplicitLocOpBuilder& b, int64_t iteration_bound, int64_t num_threads,
    int64_t unroll_factor, int64_t num_values, Value element_pair_index,
    int64_t xor_mask, ValueRange tensors, ReadElementFn read_element,
    WriteElementFn write_element, FuncOp comparator, bool force_write,
    bool needs_bounds_checks = true) {
  auto index_const = [&](int64_t value) -> Value {
    return arith::ConstantIndexOp::create(b, value);
  };

  int64_t block_size = xor_mask;
  if (xor_mask > 1 && (xor_mask & (xor_mask + 1)) == 0) {
    block_size = (xor_mask + 1) / 2;
  }
  if (block_size >= iteration_bound) {
    return SmallVector<Value>(tensors.begin(), tensors.end());
  }

  if (num_threads % kNumShmemBanks == 0 && unroll_factor > 1) {
    Value bank_row_id = arith::DivUIOp::create(b, element_pair_index,
                                               index_const(kNumShmemBanks));
    Value offset = arith::RemUIOp::create(b, element_pair_index,
                                          index_const(kNumShmemBanks));
    Value first_in_bank_row = arith::MulIOp::create(
        b, bank_row_id, index_const(unroll_factor * kNumShmemBanks));
    element_pair_index = arith::AddIOp::create(b, first_in_bank_row, offset);
  } else {
    element_pair_index = arith::MulIOp::create(b, element_pair_index,
                                               index_const(unroll_factor));
  }

  SmallVector<Value> current_tensors(tensors.begin(), tensors.end());
  for (int64_t i = 0; i < unroll_factor; ++i) {
    Value current_keys_index;
    if (num_threads % kNumShmemBanks == 0) {
      current_keys_index = arith::AddIOp::create(
          b, element_pair_index, index_const(i * kNumShmemBanks));
    } else {
      current_keys_index =
          arith::AddIOp::create(b, element_pair_index, index_const(i));
    }
    if (block_size == 1) {
      current_keys_index =
          arith::MulIOp::create(b, current_keys_index, index_const(2));
    } else if (block_size * 2 < iteration_bound) {
      Value block_id = arith::DivUIOp::create(b, current_keys_index,
                                              index_const(block_size));
      Value index_within_block = arith::RemUIOp::create(
          b, current_keys_index, index_const(block_size));
      Value first_element_in_block =
          arith::MulIOp::create(b, block_id, index_const(2 * block_size));
      current_keys_index =
          arith::AddIOp::create(b, first_element_in_block, index_within_block);
    } else {
      Value is_within_left_block =
          arith::CmpIOp::create(b, arith::CmpIPredicate::slt,
                                current_keys_index, index_const(block_size));
      current_keys_index =
          arith::SelectOp::create(b, is_within_left_block, current_keys_index,
                                  index_const(iteration_bound ^ xor_mask));
    }
    Value compare_keys_index =
        arith::XOrIOp::create(b, current_keys_index, index_const(xor_mask));
    Value index_is_inbounds =
        needs_bounds_checks
            ? arith::CmpIOp::create(b, arith::CmpIPredicate::slt,
                                    compare_keys_index,
                                    index_const(iteration_bound))
                  .getResult()
            : arith::ConstantIntOp::create(b, b.getI1Type(), 1).getResult();

    ABSL_ASSIGN_OR_RETURN(
        current_tensors,
        EmitUpdateIfWithStatus(
            b, index_is_inbounds, current_tensors,
            [&](ImplicitLocOpBuilder& if_b)
                -> absl::StatusOr<SmallVector<Value>> {
              SmallVector<Value> values_to_compare;
              values_to_compare.reserve(num_values * 2);
              for (int64_t param = 0; param < num_values; ++param) {
                ABSL_ASSIGN_OR_RETURN(Value val_compare,
                                 read_element(if_b, current_tensors, param,
                                              compare_keys_index));
                values_to_compare.push_back(val_compare);
                ABSL_ASSIGN_OR_RETURN(Value val_current,
                                 read_element(if_b, current_tensors, param,
                                              current_keys_index));
                values_to_compare.push_back(val_current);
              }
              Value pred =
                  PureCallOp::create(if_b, comparator, values_to_compare)
                      .getResult(0);
              Value zero =
                  arith::ConstantIntOp::create(if_b, if_b.getI8Type(), 0);
              Value is_smaller_than = arith::CmpIOp::create(
                  if_b, arith::CmpIPredicate::ne, pred, zero);

              auto write_swapped_order =
                  [&](ImplicitLocOpBuilder& swap_b) -> SmallVector<Value> {
                SmallVector<Value> updated(current_tensors.begin(),
                                           current_tensors.end());
                for (int64_t param = 0; param < num_values; ++param) {
                  Value value1 = values_to_compare[param * 2];
                  Value value2 = values_to_compare[param * 2 + 1];
                  updated[param] = write_element(swap_b, updated[param], param,
                                                 current_keys_index, value1);
                  updated[param] = write_element(swap_b, updated[param], param,
                                                 compare_keys_index, value2);
                }
                return updated;
              };

              if (force_write) {
                auto write_original_order =
                    [&](ImplicitLocOpBuilder& orig_b) -> SmallVector<Value> {
                  SmallVector<Value> updated(current_tensors.begin(),
                                             current_tensors.end());
                  for (int64_t param = 0; param < num_values; ++param) {
                    Value value1 = values_to_compare[param * 2];
                    Value value2 = values_to_compare[param * 2 + 1];
                    updated[param] =
                        write_element(orig_b, updated[param], param,
                                      current_keys_index, value2);
                    updated[param] =
                        write_element(orig_b, updated[param], param,
                                      compare_keys_index, value1);
                  }
                  return updated;
                };
                return EmitIfElse(if_b, is_smaller_than,
                                  TypeRange(current_tensors),
                                  write_swapped_order, write_original_order);
              }
              return EmitUpdateIf(if_b, is_smaller_than, current_tensors,
                                  write_swapped_order);
            }));
  }
  return current_tensors;
}

absl::StatusOr<SmallVector<Value>> EmitTiledCompareLoop(
    ImplicitLocOpBuilder& b, ValueRange tiled_keys_index, Value thread_id_x,
    const HloSortInstruction* sort, int64_t dimension_to_sort_bound,
    int64_t num_threads, absl::Span<const int64_t> xor_masks,
    ValueRange output_tensors, bool emit_iota_operands, int64_t tile_size,
    int64_t unroll_factor, FuncOp comparator) {
  int64_t dimension_to_sort = sort->sort_dimension();
  int64_t num_values = sort->operand_count();
  auto index_const = [&](int64_t value) -> Value {
    return arith::ConstantIndexOp::create(b, value);
  };

  SmallVector<Value> shmem_tensors;
  shmem_tensors.reserve(num_values);
  int64_t shmem_tile_size = std::max(tile_size, int64_t{64});
  for (int64_t param = 0; param < num_values; ++param) {
    Type elem_type = emitters::PrimitiveTypeToMlirType(
        sort->operand(param)->shape().element_type(), b);
    shmem_tensors.push_back(AllocateSharedOp::create(
        b, RankedTensorType::get({shmem_tile_size}, elem_type)));
  }

  Value unroll = index_const(unroll_factor);
  Value base_keys_index =
      arith::MulIOp::create(b, tiled_keys_index[dimension_to_sort], unroll);
  Value base_cache_index = arith::MulIOp::create(b, thread_id_x, unroll);

  // Copy `unroll_factor` adjacent elements from global memory (or kIota) to
  // shared memory.
  for (int64_t i = 0; i < unroll_factor; ++i) {
    Value offset = index_const(i);
    Value current_keys_index =
        arith::AddIOp::create(b, base_keys_index, offset);
    Value in_bounds =
        arith::CmpIOp::create(b, arith::CmpIPredicate::slt, current_keys_index,
                              index_const(dimension_to_sort_bound));
    ABSL_ASSIGN_OR_RETURN(
        shmem_tensors,
        EmitUpdateIfWithStatus(
            b, in_bounds, shmem_tensors,
            [&](ImplicitLocOpBuilder& if_b)
                -> absl::StatusOr<SmallVector<Value>> {
              Value cache_index =
                  arith::AddIOp::create(if_b, base_cache_index, offset);
              SmallVector<Value> elem_index(tiled_keys_index.begin(),
                                            tiled_keys_index.end());
              elem_index[dimension_to_sort] = current_keys_index;
              SmallVector<Value> updated(shmem_tensors.begin(),
                                         shmem_tensors.end());
              for (int64_t param = 0; param < num_values; ++param) {
                Value value;
                if (emit_iota_operands &&
                    HloPredicateIsOp<HloOpcode::kIota>(sort->operand(param))) {
                  ABSL_ASSIGN_OR_RETURN(auto iota_vals,
                                   emitters::EmitIota(sort->operand(param),
                                                      elem_index, if_b));
                  value = iota_vals.front();
                } else {
                  value = tensor::ExtractOp::create(if_b, output_tensors[param],
                                                    elem_index);
                }
                updated[param] = tensor::InsertOp::create(
                    if_b, value, updated[param], ValueRange{cache_index});
              }
              return updated;
            }));
  }

  auto read_shmem = [](ImplicitLocOpBuilder& body_b, ValueRange tensors,
                       int64_t operand, Value index) -> absl::StatusOr<Value> {
    return tensor::ExtractOp::create(body_b, tensors[operand],
                                     ValueRange{index})
        .getResult();
  };
  auto write_shmem = [](ImplicitLocOpBuilder& body_b, Value tensor,
                        int64_t operand, Value index, Value value) -> Value {
    return tensor::InsertOp::create(body_b, value, tensor, ValueRange{index});
  };

  for (int64_t xor_mask : xor_masks) {
    auto synced =
        SyncThreadsOp::create(b, TypeRange(shmem_tensors), shmem_tensors)
            .getResults();
    shmem_tensors.assign(synced.begin(), synced.end());

    auto emit_compare_loop =
        [&](ImplicitLocOpBuilder& body_b, int64_t index_bound,
            bool needs_bounds_checks,
            ValueRange tensors) -> absl::StatusOr<SmallVector<Value>> {
      return EmitCompareLoopBody(
          body_b, index_bound, num_threads, unroll_factor / 2, num_values,
          thread_id_x, xor_mask, tensors, read_shmem, write_shmem, comparator,
          /*force_write=*/false, needs_bounds_checks);
    };

    if (dimension_to_sort_bound % tile_size == 0) {
      ABSL_ASSIGN_OR_RETURN(
          shmem_tensors,
          emit_compare_loop(b, tile_size, /*needs_bounds_checks=*/false,
                            shmem_tensors));
    } else {
      // Only the last tile is partial; it starts at
      // RoundDownTo(dimension_to_sort_bound, tile_size).
      Value is_last_tile = arith::CmpIOp::create(
          b, arith::CmpIPredicate::sge, base_keys_index,
          index_const(RoundDownTo(dimension_to_sort_bound, tile_size)));
      ABSL_ASSIGN_OR_RETURN(shmem_tensors,
                       EmitIfElseWithStatus(
                           b, is_last_tile, TypeRange(shmem_tensors),
                           [&](ImplicitLocOpBuilder& then_b) {
                             return emit_compare_loop(
                                 then_b, dimension_to_sort_bound % tile_size,
                                 /*needs_bounds_checks=*/true, shmem_tensors);
                           },
                           [&](ImplicitLocOpBuilder& else_b) {
                             return emit_compare_loop(
                                 else_b, tile_size,
                                 /*needs_bounds_checks=*/false, shmem_tensors);
                           }));
    }
  }
  auto synced =
      SyncThreadsOp::create(b, TypeRange(shmem_tensors), shmem_tensors)
          .getResults();
  shmem_tensors.assign(synced.begin(), synced.end());

  // Copy sorted tile from shared memory back to global memory.
  SmallVector<Value> updated_outputs(output_tensors.begin(),
                                     output_tensors.end());
  for (int64_t i = 0; i < unroll_factor; ++i) {
    Value offset = index_const(i);
    Value current_keys_index =
        arith::AddIOp::create(b, base_keys_index, offset);
    Value in_bounds =
        arith::CmpIOp::create(b, arith::CmpIPredicate::slt, current_keys_index,
                              index_const(dimension_to_sort_bound));
    updated_outputs = EmitUpdateIf(
        b, in_bounds, updated_outputs,
        [&](ImplicitLocOpBuilder& if_b) -> SmallVector<Value> {
          Value cache_index =
              arith::AddIOp::create(if_b, base_cache_index, offset);
          SmallVector<Value> elem_index(tiled_keys_index.begin(),
                                        tiled_keys_index.end());
          elem_index[dimension_to_sort] = current_keys_index;
          SmallVector<Value> updated(updated_outputs.begin(),
                                     updated_outputs.end());
          for (int64_t param = 0; param < num_values; ++param) {
            Value value = tensor::ExtractOp::create(if_b, shmem_tensors[param],
                                                    ValueRange{cache_index});
            updated[param] = tensor::InsertOp::create(
                if_b, value, updated[param], elem_index);
          }
          return updated;
        });
  }
  return updated_outputs;
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> CreateBitonicSortMlirModule(
    MLIRContext& mlir_context, const HloSortInstruction* sort,
    const HloInstruction* hlo_with_buffers,
    const BufferAssignment* buffer_assignment,
    absl::string_view entry_function_name, absl::Span<const int64_t> xor_masks,
    const LaunchDimensions& launch_dimensions,
    int64_t num_iterations_in_sort_dim, int64_t tile_size,
    int64_t unroll_factor, bool emit_iota_operands) {
  mlir::OpBuilder builder(&mlir_context);
  auto loc =
      mlir::NameLoc::get(builder.getStringAttr(hlo_with_buffers->name()));
  mlir::OwningOpRef<mlir::ModuleOp> module = llvm_ir::CreateMlirModuleOp(loc);

  bool force_64_bit = launch_dimensions.launch_bound() * unroll_factor >
                      std::numeric_limits<int32_t>::max();
  emitters::SetIndexDataLayout(*module, *hlo_with_buffers, force_64_bit);

  ABSL_ASSIGN_OR_RETURN(FuncOp entry_func,
                   emitters::EmitKernelApi(
                       *module, *hlo_with_buffers, buffer_assignment,
                       GetDefaultBufferAlignment(), entry_function_name));
  SetBackendKind(&mlir_context, entry_func, BackendKind::kGpu);

  emitters::PartitionedComputations computations(sort->to_apply(),
                                                 &mlir_context);
  ABSL_ASSIGN_OR_RETURN(auto call_targets, emitters::EmitPartitionedComputations(
                                          *module, computations));
  FuncOp comparator = call_targets(sort->to_apply()->root_instruction());

  ImplicitLocOpBuilder b(loc, entry_func);
  b.setInsertionPointToStart(entry_func.addEntryBlock());

  ValueRange output_args =
      entry_func.getArguments().drop_front(hlo_with_buffers->operand_count());
  SmallVector<Value> output_tensors(output_args.begin(), output_args.end());

  const Shape& keys_shape = sort->operand(0)->shape();
  int64_t rank = keys_shape.dimensions().size();
  int64_t dimension_to_sort = sort->sort_dimension();
  int64_t dimension_to_sort_bound = keys_shape.dimensions(dimension_to_sort);

  Shape iteration_shape = keys_shape;
  iteration_shape.set_dimensions(dimension_to_sort, num_iterations_in_sort_dim);
  std::vector<int64_t> minor_to_major;
  minor_to_major.reserve(rank);
  minor_to_major.push_back(dimension_to_sort);
  for (int64_t dim : LayoutUtil::MinorToMajor(keys_shape)) {
    if (dim != dimension_to_sort) {
      minor_to_major.push_back(dim);
    }
  }
  *iteration_shape.mutable_layout() = LayoutUtil::MakeLayout(minor_to_major);

  if (ShapeUtil::ElementsIn(iteration_shape) == 0) {
    mlir::func::ReturnOp::create(b, output_tensors);
    return module;
  }

  SmallVector<Value> thread_and_block_ids =
      MlirKernelEmitter::EmitThreadAndBlockIds(b, launch_dimensions);
  Value thread_id_x = thread_and_block_ids[0];

  IndexingMap indexing_map = emitters::GetDefaultWorkItemIndexingMap(
      launch_dimensions.AsWorkDimensions(), iteration_shape, &mlir_context);

  int64_t num_threads = launch_dimensions.thread_counts_per_block().x;
  int64_t num_values = sort->operand_count();

  if (xor_masks.empty()) {
    // A sort dimension of size 1 needs no comparisons; only iota operands have
    // to be written to their output buffers.
    Value in_bounds =
        emitters::CheckConstraints(indexing_map, thread_and_block_ids, {}, b);
    ABSL_ASSIGN_OR_RETURN(
        output_tensors,
        EmitUpdateIfWithStatus(
            b, in_bounds, output_tensors,
            [&](ImplicitLocOpBuilder& if_b)
                -> absl::StatusOr<SmallVector<Value>> {
              SmallVector<Value, 3> keys_index = emitters::ApplyIndexing(
                  indexing_map, thread_and_block_ids, {}, if_b);
              SmallVector<Value> updated(output_tensors.begin(),
                                         output_tensors.end());
              for (int64_t operand = 0; operand < num_values; ++operand) {
                if (!HloPredicateIsOp<HloOpcode::kIota>(
                        sort->operand(operand))) {
                  continue;
                }
                ABSL_ASSIGN_OR_RETURN(auto iota_vals,
                                 emitters::EmitIota(sort->operand(operand),
                                                    keys_index, if_b));
                updated[operand] = tensor::InsertOp::create(
                    if_b, iota_vals.front(), updated[operand], keys_index);
              }
              return updated;
            }));
  } else if (xor_masks.size() > 1) {
    SmallVector<Value, 3> keys_index =
        emitters::ApplyIndexing(indexing_map, thread_and_block_ids, {}, b);
    ABSL_ASSIGN_OR_RETURN(
        output_tensors,
        EmitTiledCompareLoop(b, keys_index, thread_id_x, sort,
                             dimension_to_sort_bound, num_threads, xor_masks,
                             output_tensors, emit_iota_operands, tile_size,
                             unroll_factor, comparator));
  } else {
    Value in_bounds =
        emitters::CheckConstraints(indexing_map, thread_and_block_ids, {}, b);
    ABSL_ASSIGN_OR_RETURN(
        output_tensors,
        EmitUpdateIfWithStatus(
            b, in_bounds, output_tensors,
            [&](ImplicitLocOpBuilder& if_b)
                -> absl::StatusOr<SmallVector<Value>> {
              SmallVector<Value, 3> keys_index = emitters::ApplyIndexing(
                  indexing_map, thread_and_block_ids, {}, if_b);
              auto read_global = [&](ImplicitLocOpBuilder& body_b,
                                     ValueRange tensors, int64_t operand,
                                     Value index) -> absl::StatusOr<Value> {
                SmallVector<Value> elem_index(keys_index.begin(),
                                              keys_index.end());
                elem_index[dimension_to_sort] = index;
                if (emit_iota_operands && HloPredicateIsOp<HloOpcode::kIota>(
                                              sort->operand(operand))) {
                  ABSL_ASSIGN_OR_RETURN(auto iota_vals,
                                   emitters::EmitIota(sort->operand(operand),
                                                      elem_index, body_b));
                  return iota_vals.front();
                }
                return tensor::ExtractOp::create(body_b, tensors[operand],
                                                 elem_index)
                    .getResult();
              };
              auto write_global = [&](ImplicitLocOpBuilder& body_b,
                                      Value tensor, int64_t operand,
                                      Value index, Value value) -> Value {
                SmallVector<Value> elem_index(keys_index.begin(),
                                              keys_index.end());
                elem_index[dimension_to_sort] = index;
                return tensor::InsertOp::create(body_b, value, tensor,
                                                elem_index);
              };
              return EmitCompareLoopBody(
                  if_b, dimension_to_sort_bound, /*num_threads=*/1,
                  unroll_factor / 2, num_values, keys_index[dimension_to_sort],
                  xor_masks[0], output_tensors, read_global, write_global,
                  comparator, /*force_write=*/emit_iota_operands);
            }));
  }

  mlir::func::ReturnOp::create(b, output_tensors);
  return module;
}

}  // namespace

AsyncThunkSequence EmitBitonicSort(const HloSortInstruction* sort,
                                   IrEmitterContext* ir_emitter_context) {
  std::string op_name(sort->name());

  int64_t dimension_to_sort = sort->sort_dimension();
  const Shape& keys_shape = sort->operand(0)->shape();
  uint64_t dimension_to_sort_bound = keys_shape.dimensions(dimension_to_sort);
  if (ShapeUtil::IsZeroElementArray(keys_shape)) {
    return ThunkSequence{};
  }
  int64_t num_stages = Log2Ceiling(dimension_to_sort_bound);
  VLOG(2) << op_name << " requires " << num_stages << " stages.";
  TF_RET_CHECK((uint64_t{1} << num_stages) >= dimension_to_sort_bound);
  TF_RET_CHECK(num_stages == 0 ||
               (uint64_t{1} << (num_stages - 1)) < dimension_to_sort_bound);

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
  uint64_t tile_size =
      std::min(std::min(kMaxThreadsPerBlock * kBitonicSortUnrollFactor,
                        max_tile_size_fitting_into_shared_memory),
               uint64_t{1} << num_stages);
  tile_size = uint64_t{1} << Log2Floor(tile_size);

  Shape standard_iteration_shape = keys_shape;
  uint64_t standard_num_iterations_in_sort_dim =
      num_stages == 0 ? 1
                      : CeilOfRatio(uint64_t{1} << (num_stages - 1),
                                    kBitonicSortUnrollFactor / 2);
  standard_iteration_shape.set_dimensions(dimension_to_sort,
                                          standard_num_iterations_in_sort_dim);

  LaunchDimensions standard_launch_dimensions = CalculateLaunchDimensions(
      standard_iteration_shape, ir_emitter_context->gpu_device_info());

  uint64_t rounded_bound = RoundUpTo(dimension_to_sort_bound, tile_size);
  Shape iteration_shape = keys_shape;
  uint64_t num_iterations_in_sort_dim =
      CeilOfRatio(rounded_bound, kBitonicSortUnrollFactor);
  iteration_shape.set_dimensions(dimension_to_sort, num_iterations_in_sort_dim);
  uint64_t num_iterations = ShapeUtil::ElementsIn(iteration_shape);

  const uint64_t kThreadsPerBlock =
      std::max(uint64_t{1}, tile_size / kBitonicSortUnrollFactor);

  uint64_t num_blocks = CeilOfRatio(num_iterations, kThreadsPerBlock);
  LaunchDimensions tiled_launch_dimensions(num_blocks, kThreadsPerBlock);
  VLOG(2) << absl::StreamFormat("%s launch dims: %d blocks, %d threads/block",
                                op_name, num_blocks, kThreadsPerBlock);

  bool is_fusion = sort->parent()->IsFusionComputation();
  const HloInstruction* hlo_with_buffers =
      is_fusion ? sort->parent()->FusionInstruction() : sort;
  ABSL_ASSIGN_OR_RETURN(auto kernel_arguments,
                   emitters::KernelArguments::Create(
                       ir_emitter_context->buffer_assignment(),
                       GetDefaultBufferAlignment(), hlo_with_buffers));

  std::vector<tsl::Future<std::unique_ptr<Thunk>>> thunks;
  bool emit_iota_operands = true;
  auto emit_thunk = [&](absl::Span<const int64_t> xor_masks) -> absl::Status {
    VLOG(2) << absl::StreamFormat(
        "%s uses kernel for xor masks [%s]", op_name,
        absl::StrJoin(xor_masks, ", ", [](std::string* out, int64_t xor_mask) {
          absl::StrAppendFormat(out, "0x%x", xor_mask);
        }));

    LaunchDimensions launch_dimensions = xor_masks.size() > 1
                                             ? tiled_launch_dimensions
                                             : standard_launch_dimensions;
    std::string sanitized_kernel_name =
        ir_emitter_context->GetSanitizedUniqueName(op_name);
    Thunk::ThunkInfo thunk_info = Thunk::ThunkInfo::WithProfileAnnotation(
        hlo_with_buffers, ir_emitter_context->GetNextThunkId());

    BorrowedMlirContext borrowed_context =
        ir_emitter_context->BorrowMlirContext();
    MLIRContext* mlir_context = borrowed_context->get();
    mlir_context->appendDialectRegistry(
        MlirKernelEmitter::GetDialectRegistry());
    mlir_context->loadAllAvailableDialects();

    ABSL_ASSIGN_OR_RETURN(
        mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
        CreateBitonicSortMlirModule(
            *mlir_context, sort, hlo_with_buffers,
            &ir_emitter_context->buffer_assignment(), sanitized_kernel_name,
            xor_masks, launch_dimensions,
            xor_masks.size() > 1 ? num_iterations_in_sort_dim
                                 : standard_num_iterations_in_sort_dim,
            tile_size, kBitonicSortUnrollFactor, emit_iota_operands));
    emit_iota_operands = false;

    MlirKernelSource mlir_source(nullptr, std::move(mlir_module));

    // CompileMlirToLlvm inserts a PDL wait at the kernel entry when PDL is
    // enabled for this module and device, so launch with PDL in that case.
    bool use_pdl = IsPdlEnabled(
        sort->GetModule()->config().debug_options(),
        ir_emitter_context->gpu_device_info().gpu_compute_capability());
    Future<LlvmKernelSource> llvm_source_future =
        ir_emitter_context->kernel_compiler()->CompileMlirToLlvm(
            ir_emitter_context->gpu_device_info(), *sort->GetModule(),
            sanitized_kernel_name, /*unroll_factor=*/0, std::move(mlir_source),
            std::move(borrowed_context));

    thunks.push_back(
        std::move(llvm_source_future)
            .Map([ir_emitter_context, sanitized_kernel_name, kernel_arguments,
                  launch_dimensions, thunk_info = std::move(thunk_info),
                  use_pdl](LlvmKernelSource source) mutable
                     -> Future<std::unique_ptr<Thunk>> {
              llvm::orc::ThreadSafeModule safe_module =
                  std::move(source).thread_safe_module();
              ABSL_RETURN_IF_ERROR(AnnotateMlirKernelLlvmModule(
                  ir_emitter_context->gpu_device_info(), launch_dimensions,
                  sanitized_kernel_name, ir_emitter_context->data_layout(),
                  ir_emitter_context->target_triple(),
                  safe_module.getModuleUnlocked()));

              int devices_in_process =
                  ir_emitter_context->gpu_topology().num_devices_per_process();
              return ir_emitter_context->kernel_compiler()
                  ->CompileToTargetBinary(
                      LlvmKernelSource(std::move(safe_module)))
                  .Map([sanitized_kernel_name, kernel_arguments,
                        launch_dimensions, thunk_info = std::move(thunk_info),
                        use_pdl,
                        devices_in_process](std::vector<uint8_t> cubin) mutable
                           -> absl::StatusOr<std::unique_ptr<Thunk>> {
                    ABSL_ASSIGN_OR_RETURN(
                        CustomKernel custom_kernel,
                        kernel::CreateOwnedCubinCustomKernel(
                            sanitized_kernel_name, std::move(cubin),
                            kernel_arguments.args().size(),
                            launch_dimensions.block_counts(),
                            launch_dimensions.thread_counts_per_block(),
                            /*shared_memory_bytes=*/0));
                    return std::make_unique<CustomKernelThunk>(
                        std::move(thunk_info), std::move(custom_kernel),
                        kernel_arguments, devices_in_process, use_pdl);
                  });
            }));

    return absl::OkStatus();
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
          ABSL_RETURN_IF_ERROR(emit_thunk(xor_masks));
          xor_masks.clear();
        }
        ABSL_RETURN_IF_ERROR(emit_thunk({xor_mask}));
      } else {
        xor_masks.push_back(xor_mask);
      }
    }
  }
  if (!xor_masks.empty()) {
    ABSL_RETURN_IF_ERROR(emit_thunk(xor_masks));
  }
  if (num_stages == 0 &&
      absl::c_any_of(sort->operands(), HloPredicateIsOp<HloOpcode::kIota>)) {
    ABSL_RETURN_IF_ERROR(emit_thunk({}));
  }
  return tsl::JoinFutures(absl::MakeSpan(thunks))
      .Map([](std::vector<std::unique_ptr<Thunk>> thunks) {
        return ThunkSequence(std::move(thunks));
      });
}

AsyncThunkSequence SortFusion::Emit(IrEmitterContext& ir_emitter_context,
                                    const HloFusionInstruction& fusion) const {
  std::vector<BufferAllocation::Slice> src_buffers;
  std::vector<BufferAllocation::Slice> dst_buffers;
  std::vector<Shape> src_shapes;
  src_buffers.reserve(fusion.operand_count());
  dst_buffers.reserve(fusion.operand_count());
  src_shapes.reserve(fusion.operand_count());
  const HloSortInstruction* sort =
      Cast<HloSortInstruction>(fusion.fused_expression_root());
  Shape keys_shape = sort->operand(0)->shape();
  for (int64_t i = 0; i < sort->operand_count(); ++i) {
    // We assume that the layout of all involved operands and
    // outputs is the same.
    TF_RET_CHECK(LayoutUtil::LayoutsInShapesEqual(
        keys_shape, sort->operand(i)->shape(),
        Layout::Equal().IgnoreMemorySpace().IgnoreElementSize()));
    ShapeIndex shape_index =
        sort->operand_count() > 1 ? ShapeIndex({i}) : ShapeIndex({});
    TF_RET_CHECK(LayoutUtil::LayoutsInShapesEqual(
        keys_shape, ShapeUtil::GetSubshape(sort->shape(), shape_index),
        Layout::Equal().IgnoreMemorySpace().IgnoreElementSize()));
    // We expect only parameters or iotas as operand of sort.
    if (HloPredicateIsOp<HloOpcode::kParameter>(sort->operand(i))) {
      const HloInstruction* src_instr =
          fusion.operand(sort->operand(i)->parameter_number());
      ABSL_ASSIGN_OR_RETURN(
          BufferAllocation::Slice slice,
          ir_emitter_context.buffer_assignment().GetUniqueSlice(src_instr, {}));
      src_buffers.push_back(slice);
      src_shapes.push_back(sort->operand(i)->shape());
      ABSL_ASSIGN_OR_RETURN(slice,
                       ir_emitter_context.buffer_assignment().GetUniqueSlice(
                           &fusion, shape_index));
      dst_buffers.push_back(slice);
    } else {
      TF_RET_CHECK(HloPredicateIsOp<HloOpcode::kIota>(sort->operand(i)));
    }
  }

  ThunkSequence thunks;
  for (int i = 0; i < src_buffers.size(); ++i) {
    if (src_buffers[i] != dst_buffers[i]) {
      thunks.emplace_back(std::make_unique<DeviceToDeviceCopyThunk>(
          Thunk::ThunkInfo::WithProfileAnnotation(
              &fusion, ir_emitter_context.GetNextThunkId()),
          /*source_buffer=*/ShapedSlice{src_buffers[i], src_shapes[i]},
          /*destination_buffer=*/ShapedSlice{dst_buffers[i], src_shapes[i]},
          /*mem_size=*/src_buffers[i].size()));
    }
  }
  return EmitBitonicSort(sort, &ir_emitter_context)
      .Map(
          [thunks = std::move(thunks)](ThunkSequence sort_thunks) mutable {
            thunks.insert(thunks.end(),
                          std::make_move_iterator(sort_thunks.begin()),
                          std::make_move_iterator(sort_thunks.end()));
            return std::move(thunks);
          });
}

}  // namespace gpu
}  // namespace xla
