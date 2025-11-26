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

#include "xla/backends/gpu/codegen/emitters/dynamic_padder.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "xla/codegen/emitters/computation_partitioner.h"
#include "xla/codegen/emitters/elemental_hlo_to_mlir.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/emitters/kernel_api_builder.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

namespace {

using llvm::SmallVector;
using mlir::ImplicitLocOpBuilder;
using mlir::Value;
using mlir::ValueRange;
using mlir::func::ReturnOp;

const Shape& GetIndexShape(const Shape& shape) {
  return shape.IsTuple() ? shape.tuple_shapes(0) : shape;
}

Value EmitIsZeroThreadCheck(ImplicitLocOpBuilder& b,
                            ValueRange thread_and_block_ids) {
  Value is_block_0_thread_0 =
      b.create<mlir::arith::ConstantOp>(b.getI1Type(), b.getBoolAttr(true));
  for (auto id : thread_and_block_ids) {
    is_block_0_thread_0 = b.create<mlir::arith::AndIOp>(
        is_block_0_thread_0, b.create<mlir::arith::CmpIOp>(
                                 mlir::arith::CmpIPredicate::eq, id,
                                 b.create<mlir::arith::ConstantIndexOp>(0)));
  }
  return is_block_0_thread_0;
}

Value EmitBoundsCheck(ImplicitLocOpBuilder& b, ValueRange multi_index,
                      ValueRange shape) {
  Value result =
      b.create<mlir::arith::ConstantOp>(b.getI1Type(), b.getBoolAttr(true));
  for (auto [index, dim] : llvm::zip_equal(multi_index, shape)) {
    result = b.create<mlir::arith::AndIOp>(
        result, b.create<mlir::arith::CmpIOp>(
                    mlir::arith::CmpIPredicate::ult, index,
                    b.create<mlir::arith::IndexCastOp>(b.getIndexType(), dim)));
  }
  return result;
}

Value LinearizeMultiIndex(mlir::ImplicitLocOpBuilder& b, ValueRange multi_index,
                          ValueRange dynamic_dim_sizes) {
  Value linear_index = b.create<mlir::arith::ConstantIndexOp>(0);
  Value stride = b.create<mlir::arith::ConstantIndexOp>(1);
  CHECK_EQ(multi_index.size(), dynamic_dim_sizes.size());

  int64_t rank = multi_index.size();

  for (int64_t i = rank - 1; i >= 0; --i) {
    linear_index = b.create<mlir::arith::AddIOp>(
        linear_index, b.create<mlir::arith::MulIOp>(stride, multi_index[i]));

    stride = b.create<mlir::arith::MulIOp>(
        stride, b.create<mlir::arith::IndexCastOp>(b.getIndexType(),
                                                   dynamic_dim_sizes[i]));
  }

  return linear_index;
}

llvm::SmallVector<mlir::Value> DelinearizeIndex(mlir::ImplicitLocOpBuilder& b,
                                                Value linear_index,
                                                llvm::ArrayRef<int64_t> shape) {
  llvm::SmallVector<mlir::Value> multi_index(shape.size());

  int64_t stride = 1;
  for (int64_t i = shape.size() - 1; i >= 0; --i) {
    Value cur_index = b.create<mlir::arith::DivSIOp>(
        linear_index, b.create<mlir::arith::ConstantIndexOp>(stride));

    cur_index = b.create<mlir::arith::RemSIOp>(
        cur_index, b.create<mlir::arith::ConstantIndexOp>(shape[i]));

    stride *= shape[i];
    multi_index[i] = cur_index;
  }
  return multi_index;
}

// Input = {dynamic array(with dynamic dimension meta data at the end)}
// Output = {static array, dynamic_dim0, dynamic_dim1}
// For a tensor with static dimension [2][<=5] and dynamic dimension [2][3]
// (`_` stands for padding)
// Input = {{1,2,3,4,5,6,_,_,_,_,2,3}}
// Output = {{1,2,3,_,_,4,5,6_,_}, 2, 3}

// pseudo code for padToStatic on a 2d array
//   ```
// void padToStatic(int** input, int** output, int threads_per_block,
//                  int meta_data_offset, int max_num_element,
//                  int static_dim0_size, int static_dim1_size) {
//   int* source_array = input[0];
//   int* dest_array = output[0];

//   // extract the dynamic dimension from the source array's metadata
//   int* dyn_dim0_size = source_array + meta_data_offset;
//   int* dyn_dim1_size = source_array + meta_data_offset + sizeof(int);

//   // only one thread need to store the dynamic index
//   int thread_id = GetThreadId();
//   int block_id = GetBlockId();
//   if (thread_id == 0 && block_id == 0) {
//     *output[1] = *dyn_dim0_size;
//     *output[2] = *dyn_dim1_size;
//   }

//   int dyn_element_total = 1;
//   dyn_element_total *= *dyn_dim0_size;
//   dyn_element_total *= *dyn_dim1_size;
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
//   return;
// }
//   ```
void EmitPadToStatic(ImplicitLocOpBuilder& b,
                     const HloFusionInstruction& fusion,
                     mlir::func::FuncOp entry_func,
                     llvm::ArrayRef<Value> thread_and_block_ids,
                     const IndexingMap& indexing) {
  int64_t rank = fusion.operand(0)->shape().dimensions().size();
  auto static_dim_sizes = fusion.operand(0)->shape().dimensions();

  // Dynamic size of each dimension is attached at the end of the source
  // array(operand(0)). We need to extract these value.
  SmallVector<Value> dynamic_dim_sizes;
  dynamic_dim_sizes.reserve(rank);
  for (int64_t i = 0; i < rank; ++i) {
    auto get_dynamic_dim_size_op = b.create<GetDynamicDimSizeOp>(
        entry_func.getArgument(0), b.getI64IntegerAttr(i));
    get_dynamic_dim_size_op->setAttr(
        "xla.range", b.getIndexArrayAttr({0, static_dim_sizes[i] - 1}));

    dynamic_dim_sizes.push_back(get_dynamic_dim_size_op.getResult());
  }

  SmallVector<Value> dynamic_dim_operands =
      llvm::to_vector_of<Value>(entry_func.getArguments().drop_front(2));

  // Write dynamic dimension sizes to the operands. Only do this on the first
  // thread of the first block.
  dynamic_dim_operands = llvm::to_vector_of<Value>(
      b.create<mlir::scf::IfOp>(
           EmitIsZeroThreadCheck(b, thread_and_block_ids),
           [&](mlir::OpBuilder& then_builder, mlir::Location then_loc) {
             ImplicitLocOpBuilder then_b(then_loc, then_builder);

             SmallVector<Value> stored_dynamic_dim_operands(rank);
             for (int64_t i = 0; i < rank; ++i) {
               stored_dynamic_dim_operands[i] =
                   then_b.create<mlir::tensor::InsertOp>(
                       dynamic_dim_sizes[i], dynamic_dim_operands[i],
                       ValueRange{});
             }
             then_b.create<mlir::scf::YieldOp>(stored_dynamic_dim_operands);
           },
           [&](mlir::OpBuilder& else_builder, mlir::Location else_loc) {
             ImplicitLocOpBuilder else_b(else_loc, else_builder);

             else_b.create<mlir::scf::YieldOp>(dynamic_dim_operands);
           })
          .getResults());

  auto element_type = mlir::dyn_cast<mlir::RankedTensorType>(
                          entry_func.getArgument(0).getType())
                          .getElementType();

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
  auto body_builder = [&](ImplicitLocOpBuilder& nested_b,
                          ValueRange symbol_values, ValueRange map_results,
                          ValueRange output_tensors) -> SmallVector<Value> {
    Value input_value =
        b.create<mlir::scf::IfOp>(
             EmitBoundsCheck(nested_b, map_results, dynamic_dim_sizes),
             [&](mlir::OpBuilder& then_builder, mlir::Location then_loc) {
               ImplicitLocOpBuilder then_b(then_loc, then_builder);

               SmallVector<Value> input_index =
                   DelinearizeIndex(then_b,
                                    LinearizeMultiIndex(nested_b, map_results,
                                                        dynamic_dim_sizes),
                                    static_dim_sizes);

               Value input_value = then_b.create<mlir::tensor::ExtractOp>(
                   entry_func.getArgument(0), input_index);
               then_b.create<mlir::scf::YieldOp>(input_value);
             },

             [&](mlir::OpBuilder& else_builder, mlir::Location else_loc) {
               ImplicitLocOpBuilder else_b(else_loc, else_builder);

               Value input_value = else_b.create<mlir::arith::ConstantOp>(
                   element_type, else_builder.getZeroAttr(element_type));
               else_b.create<mlir::scf::YieldOp>(input_value);
             })
            .getResult(0);

    return {nested_b.create<mlir::tensor::InsertOp>(
        input_value, output_tensors.front(), map_results)};
  };

  Value updated_output =
      emitters::EmitXlaLoopOp(b, thread_and_block_ids,
                              entry_func.getArgument(1), indexing, body_builder)
          .front();

  SmallVector<Value> results;
  results.push_back(updated_output);
  results.append(dynamic_dim_operands.begin(), dynamic_dim_operands.end());
  b.create<ReturnOp>(results);
}

}  // namespace

LaunchDimensions DynamicPadder::launch_dimensions() const {
  const Shape& indexing_shape =
      GetIndexShape(analysis_.fusion_spec().fusion_root(0).shape());
  return CalculateLaunchDimensions(indexing_shape, analysis_.device_info(),
                                   config_);
}

std::optional<IndexingMap> DynamicPadder::ComputeThreadIdToOutputIndexing(
    int64_t root_index, mlir::MLIRContext* mlir_context) const {
  auto launch_dims = launch_dimensions();
  return GetDefaultThreadIdIndexingMap(
      launch_dims, config_.unroll_factor,
      GetIndexShape(analysis_.fusion_root(root_index).shape()), mlir_context);
}

std::optional<std::vector<IndexingMap>>
DynamicPadder::ComputeThreadIdToInputIndexing(
    int64_t root_index, mlir::MLIRContext* mlir_context) const {
  return std::nullopt;
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
DynamicPadder::CreateMLIRModule(
    mlir::MLIRContext& mlir_context, const HloFusionInstruction& fusion,
    const std::string& entry_function_name,
    const BufferAssignment* buffer_assignment) const {
  mlir::OpBuilder builder(&mlir_context);
  auto loc = mlir::NameLoc::get(builder.getStringAttr(fusion.name()));
  mlir::OwningOpRef<mlir::ModuleOp> module = llvm_ir::CreateMlirModuleOp(loc);

  TF_ASSIGN_OR_RETURN(mlir::func::FuncOp entry_func,
                      emitters::EmitKernelApi(
                          *module, fusion, buffer_assignment,
                          GetDefaultBufferAlignment(), entry_function_name));
  SetBackendKind(&mlir_context, entry_func, BackendKind::kGpu);
  emitters::SetIndexDataLayout(module.get(), fusion);

  mlir::ImplicitLocOpBuilder b(loc, builder);
  b.setInsertionPointToStart(entry_func.addEntryBlock());

  SmallVector<Value> thread_and_block_ids = EmitThreadAndBlockIds(b);
  std::optional<IndexingMap> indexing =
      ComputeThreadIdToOutputIndexing(0, &mlir_context);

  EmitPadToStatic(b, fusion, entry_func, thread_and_block_ids, *indexing);

  return module;
}

absl::Status DynamicPadder::EmitEntryFunction(
    const emitters::PartitionedComputations& computations,
    const emitters::CallTargetProvider& call_targets,
    mlir::func::FuncOp entry_function,
    const HloFusionInstruction& fusion) const {
  return absl::UnimplementedError("Not implemented yet.");
}

}  // namespace gpu
}  // namespace xla
