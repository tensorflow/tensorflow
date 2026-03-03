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

#include "xla/backends/gpu/codegen/triton/lowering_util.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"
#include "xla/backends/gpu/codegen/triton/tma_utils.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/stream_executor/gpu/tma_metadata.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace xla::gpu::triton {
namespace {
namespace ttir = ::mlir::triton;
namespace arith = ::mlir::arith;
}  // namespace

absl::StatusOr<stream_executor::ThreadDim> ExtractThreadDims(
    mlir::ModuleOp triton_module, mlir::LLVM::LLVMFuncOp func_op) {
  // Extract the launch information from the Triton module.
  auto threads_per_warp_attr =
      triton_module->getAttrOfType<mlir::IntegerAttr>("ttg.threads-per-warp");
  if (!threads_per_warp_attr) {
    return absl::InternalError("ttg.threads-per-warp attribute not found.");
  }
  auto num_warps_attr =
      triton_module->getAttrOfType<mlir::IntegerAttr>("ttg.num-warps");
  if (!num_warps_attr) {
    return absl::InternalError("ttg.num-warps attribute not found.");
  }
  // AMD/ROCm Triton backend does not support warp specialization.
  // Consequently, `ttg.total-num-warps` and  `nvvm.reqntid` are not added
  // to triton module/function.
  // ThreadDim is therefore calculated from the Module attributes and not
  // retrieved from `nvvm.reqntid`.
  auto target = triton_module->getAttrOfType<mlir::StringAttr>("ttg.target");
  if (!target) {
    return absl::InternalError("ttg.target attribute not found.");
  }
  if (target.getValue().find("gfx") != std::string::npos) {
    stream_executor::ThreadDim thread_dims(
        num_warps_attr.getInt() * threads_per_warp_attr.getInt(), 1, 1);
    return thread_dims;
  }
  auto total_num_warps_attr =
      triton_module->getAttrOfType<mlir::IntegerAttr>("ttg.total-num-warps");
  if (!total_num_warps_attr) {
    return absl::InternalError("ttg.total-num-warps attribute not found.");
  }
  auto reqntid_attr =
      func_op->getAttrOfType<mlir::DenseI32ArrayAttr>("nvvm.reqntid");
  if (!reqntid_attr) {
    return absl::InternalError("nvvm.reqntid attribute not found.");
  }
  auto reqntids = reqntid_attr.asArrayRef();
  if (reqntids.empty()) {
    return absl::InternalError("nvvm.reqntid attribute is empty.");
  }
  if (reqntids.size() > 3) {
    return absl::InternalError(
        "nvvm.reqntid attribute has more than 3 dimensions.");
  }

  // Validate the launch information.
  if (num_warps_attr.getInt() != total_num_warps_attr.getInt()) {
    VLOG(6)
        << "num_warps and total_num_warps are different! This can happen if "
           "Triton compilation decides to use a different number of warps than "
           "configured. e.g. auto warp specialization can do that.";
  }
  int64_t expected_total_threads = xla::Product<int32_t>(reqntids);
  int64_t actual_total_threads =
      total_num_warps_attr.getInt() * threads_per_warp_attr.getInt();
  if (actual_total_threads != expected_total_threads) {
    return absl::InternalError(absl::StrCat(
        "Expected total threads as per reqntid attribute to be ",
        expected_total_threads, " but got ", actual_total_threads,
        " as per ttg.total-num-warps and tt.threads-per-warp attributes."));
  }

  stream_executor::ThreadDim thread_dims(reqntids[0],
                                         reqntids.size() > 1 ? reqntids[1] : 1,
                                         reqntids.size() > 2 ? reqntids[2] : 1);
  return thread_dims;
}

absl::StatusOr<stream_executor::gpu::TmaMetadata> ExtractTmaMetadata(
    mlir::LLVM::LLVMFuncOp func_op) {
  stream_executor::gpu::TmaMetadata tma_metadata;
  for (auto [idx, arg] : llvm::enumerate(func_op.getArguments())) {
    if (auto attr =
            func_op.getArgAttrOfType<mlir::triton::xla::TmaDescriptorAttr>(
                idx, "tt.tma_descriptor")) {
      TF_ASSIGN_OR_RETURN(
          auto tma_desc,
          CreateTmaDescriptor(attr.getGlobalShape(), attr.getTileShape(),
                              attr.getTileStrides(), attr.getLayout(),
                              attr.getElementByteSize(),
                              attr.getSwizzleMode().getValue()));
      tma_metadata.arg_index_to_tma_info.insert({idx, tma_desc});
    }
  }
  return tma_metadata;
}

std::vector<llvm::Metadata*> ExtractNvvmAnnotations(
    llvm::Module* ll_triton_module) {
  std::vector<llvm::Metadata*> captured_nvvm_annotations;
  llvm::NamedMDNode* nvvm_annotations =
      ll_triton_module->getNamedMetadata("nvvm.annotations");
  if (nvvm_annotations) {
    for (llvm::MDNode* operand : nvvm_annotations->operands()) {
      captured_nvvm_annotations.push_back(operand);
    }
    ll_triton_module->eraseNamedMetadata(nvvm_annotations);
  }
  return captured_nvvm_annotations;
}

llvm::SmallVector<int64_t> ComputeStrides(llvm::ArrayRef<int64_t> shape,
                                          llvm::ArrayRef<int64_t> layout) {
  CHECK_EQ(shape.size(), layout.size());
  llvm::SmallVector<int64_t> result(shape.size());
  int64_t stride = 1;
  for (int64_t dim : layout) {
    result[dim] = stride;
    stride *= shape[dim];
  }
  return result;
}

llvm::SmallVector<unsigned> GetRetainedDims(
    llvm::ArrayRef<unsigned> reduced_dims, size_t rank) {
  llvm::SmallVector<unsigned> result;
  result.reserve(rank);
  for (auto [i, dim] : llvm::enumerate(reduced_dims)) {
    for (unsigned j = result.size() + i; j < dim; ++j) {
      result.push_back(j);
    }
  }
  while (result.size() < rank) {
    result.push_back(result.size() + reduced_dims.size());
  }
  return result;
}

llvm::SmallVector<mlir::Value> IndexCast(mlir::ImplicitLocOpBuilder& builder,
                                         mlir::Type type,
                                         mlir::ValueRange values) {
  llvm::SmallVector<mlir::Value> result;
  result.reserve(values.size());
  for (auto value : values) {
    result.push_back(arith::IndexCastOp::create(builder, type, value));
  }
  return result;
}

mlir::Value ExpandAndBroadcastValue(mlir::ImplicitLocOpBuilder& builder,
                                    mlir::Value value, int dim,
                                    mlir::RankedTensorType tile_type) {
  for (int i = 0; i < tile_type.getRank(); ++i) {
    if (i != dim) {
      value = ttir::ExpandDimsOp::create(builder, value, i);
    }
  }
  return ttir::BroadcastOp::create(builder, tile_type, value);
}

bool IsGuaranteedDivisible(mlir::Value value, int64_t divisor) {
  if (auto const_op = value.getDefiningOp<arith::ConstantIndexOp>()) {
    return const_op.value() % divisor == 0;
  }
  if (auto apply_indexing = value.getDefiningOp<::xla::ApplyIndexingOp>()) {
    mlir::AffineMap affine_map = apply_indexing.getIndexingMap().GetAffineMap();
    if (affine_map.getNumResults() != 1) {
      return false;
    }
    return affine_map.getResult(0).isMultipleOf(divisor);
  }
  return false;
}

std::pair<mlir::Value, mlir::Value> CreateTensorOfPointersAndMask(
    mlir::ImplicitLocOpBuilder& builder, mlir::Value base_ptr,
    llvm::ArrayRef<int64_t> original_shape, llvm::ArrayRef<int64_t> layout,
    mlir::ValueRange offsets, llvm::ArrayRef<int64_t> sizes,
    llvm::ArrayRef<int64_t> strides, llvm::ArrayRef<unsigned> reduced_dims,
    llvm::ArrayRef<int64_t> tile_shape) {
  CHECK_EQ(original_shape.size(), layout.size());
  CHECK_EQ(original_shape.size(), offsets.size());
  CHECK_EQ(original_shape.size(), sizes.size());
  CHECK_EQ(original_shape.size(), strides.size());
  CHECK_EQ(original_shape.size(), reduced_dims.size() + tile_shape.size());

  llvm::SmallVector<int64_t> shape_strides =
      ComputeStrides(original_shape, layout);
  llvm::SmallVector<unsigned> retained_dims =
      GetRetainedDims(reduced_dims, tile_shape.size());

  mlir::Type i64_type = builder.getI64Type();
  auto i64_tile_type = mlir::RankedTensorType::get(tile_shape, i64_type);

  // Combines the values using op, if rhs is present. Otherwise returns lhs.
  auto add_if = [&](auto op, mlir::Value lhs, mlir::Value rhs) -> mlir::Value {
    if (rhs) {
      return decltype(op)::create(builder, lhs.getType(), lhs, rhs);
    }
    return lhs;
  };

  llvm::SmallVector<mlir::Value> cast_offsets =
      IndexCast(builder, i64_type, offsets);

  mlir::Value range_tile, mask_tile;
  for (auto [i, dim] : llvm::enumerate(retained_dims)) {
    auto i64_row_type = mlir::RankedTensorType::get({sizes[dim]}, i64_type);

    // Create iota range row tensor.
    mlir::Value range = ttir::MakeRangeOp::create(
        builder, i64_row_type.clone(builder.getI32Type()), 0, sizes[dim]);
    range = arith::ExtSIOp::create(builder, i64_row_type, range);

    // Multiply range by tile stride.
    mlir::Value stride = arith::ConstantOp::create(
        builder, mlir::DenseIntElementsAttr::get(i64_row_type, strides[dim]));
    range = arith::MulIOp::create(builder, range, stride);

    // Expand and broadcast range to tile shape.
    range = ExpandAndBroadcastValue(builder, range, i, i64_tile_type);

    // Create a mask for values that are inside bounds.
    // We need to check the offset alignment with the tile size as well,
    // otherwise we might load/store outside the valid range of the tile, even
    // if the original shape is divisible by the tile size.
    if (original_shape[dim] % sizes[dim] != 0 ||
        !IsGuaranteedDivisible(offsets[dim], sizes[dim])) {
      mlir::Value upper_bound =
          arith::ConstantIntOp::create(builder, i64_type, original_shape[dim]);
      upper_bound =
          arith::SubIOp::create(builder, upper_bound, cast_offsets[dim]);
      upper_bound = ttir::SplatOp::create(builder, i64_tile_type, upper_bound);
      mlir::Value mask = arith::CmpIOp::create(
          builder, arith::CmpIPredicate::slt, range, upper_bound);

      // Combine mask with previous iteration.
      mask_tile = add_if(arith::AndIOp(), mask, mask_tile);
    }

    // Multiply range by shape strides.
    mlir::Value shape_stride = arith::ConstantOp::create(
        builder,
        mlir::DenseIntElementsAttr::get(i64_tile_type, shape_strides[dim]));
    range = arith::MulIOp::create(builder, range, shape_stride);

    // Combine range with previous iteration.
    range_tile = add_if(arith::AddIOp(), range, range_tile);
  }

  // Sum up block-uniform offsets multiplied by strides.
  mlir::Value block_offset;
  for (auto [cast_offset, shape_stride] :
       llvm::zip_equal(cast_offsets, shape_strides)) {
    mlir::Value offset = arith::MulIOp::create(
        builder, cast_offset,
        arith::ConstantIntOp::create(builder, i64_type, shape_stride));
    // Combine offset with previous iteration.
    block_offset = add_if(arith::AddIOp(), offset, block_offset);
  }
  if (block_offset && mlir::isa<mlir::RankedTensorType>(base_ptr.getType())) {
    // Splat block_offset if base_ptr is a tensor otherwise AddPtrOp will fail.
    block_offset = ttir::SplatOp::create(builder, i64_tile_type, block_offset);
  }
  // Add the accumulated offsets to the base pointer.
  mlir::Value block_ptr = add_if(ttir::AddPtrOp(), base_ptr, block_offset);

  // Splat block-uniform pointer and add range offsets.
  // In case the base_ptr is already a tensor, we can skip the splat.
  mlir::Value ptr_tile = block_ptr;
  if (!mlir::isa<mlir::RankedTensorType>(ptr_tile.getType())) {
    auto ptr_tile_type =
        mlir::RankedTensorType::get(tile_shape, base_ptr.getType());
    ptr_tile = ttir::SplatOp::create(builder, ptr_tile_type, block_ptr);
  }
  ptr_tile = add_if(ttir::AddPtrOp(), ptr_tile, range_tile);

  return std::make_pair(ptr_tile, mask_tile);
}

}  // namespace xla::gpu::triton
