/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/hlo_matchers.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <utility>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo

namespace mlir {
namespace odml {

namespace {

// Facilitate access to 1-d backing data for a tensor so that values in a 1-d
// slice of the tensor can be accessed as if part of an ArrayView.
class StridedArrayViewBase {
 protected:
  StridedArrayViewBase(ArrayRef<int64_t> shape, ArrayRef<int64_t> index,
                       int64_t axis) {
    assert(shape.size() == index.size());
    assert(axis < shape.size());
    assert(axis >= 0);
    assert(index[axis] == 0);
    offset_ = IndexToOffset(shape, index);
    stride_ = StrideForAxis(shape, axis);
    size_ = shape[axis];
  }

  // Returns the size of the 1-d slice across the tensor.
  int64_t size() const { return size_; }

  // Calculates the next index in a tensor excluding a specified axis.
  //
  // Returns the next index where one exists.
  // If there is no valid next index, returns `std::nullopt`.
  //
  // `index` should have the same size as `shape`.
  // Each value `dim` in `index` should be in [0, shape[dim]).
  static std::optional<SmallVector<int64_t>> NextTensorIndex(
      SmallVector<int64_t> index, ArrayRef<int64_t> shape, int64_t fixed_axis) {
#ifndef NDEBUG
    assert(shape.size() == index.size());
    assert(fixed_axis < shape.size());
    assert(fixed_axis >= 0);
    assert(index[fixed_axis] == 0);
    for (size_t i = 0; i < shape.size(); ++i) {
      assert(index[i] < shape[i]);
      assert(index[i] >= 0);
    }
#endif  // NDEBUG
    for (int64_t dim = shape.size() - 1; dim >= 0; --dim) {
      if (dim == fixed_axis) continue;
      ++index[dim];
      if (index[dim] < shape[dim]) return std::move(index);
      index[dim] = 0;
    }
    return std::nullopt;
  }

 protected:
  // Calculates how many values to skip across a 1-D contiguous array that holds
  // backing data for a given shape to access the value at a given index along a
  // StridedArrayView across a higher dimensional shape.
  //
  // The index `i` must be in [0, shape[axis])` where `shape` is the shape
  // of the tensor and `axis` is the axis along the tensor that the
  // StridedArrayView indexes along.
  int64_t OffsetForIndex(int64_t i) const { return offset_ + i * stride_; }

 private:
  // Calculates how many values to skip across a 1-D contiguous array that holds
  // backing data for a given shape to access the next value along a given axis.
  //
  // `axis` should be a valid dimension in `shape`.
  static int64_t StrideForAxis(ArrayRef<int64_t> shape, int64_t axis) {
    int64_t stride = 1;  // Start with the trailing dimension.
    for (int64_t dim = shape.size() - 1; dim > axis; --dim) {
      stride *= shape[dim];
    }
    return stride;
  }

  // Calculates how many values to skip across a 1-D contiguous array that holds
  // backing data for a given shape to access data at a specified index.
  //
  // `index` should have the same size as `shape`.
  // Each value `dim` in `index` should be in [0, shape[dim]).
  static int64_t IndexToOffset(ArrayRef<int64_t> shape,
                               ArrayRef<int64_t> index) {
#ifndef NDEBUG
    assert(shape.size() == index.size());
    for (size_t i = 0; i < shape.size(); ++i) {
      assert(index[i] < shape[i]);
      assert(index[i] >= 0);
    }
#endif  // NDEBUG
    int64_t offset = 0;
    int64_t stride = 1;
    for (int64_t dim = shape.size() - 1; dim >= 0; --dim) {
      offset += index[dim] * stride;
      stride *= shape[dim];
    }
    return offset;
  }

  int64_t offset_;
  int64_t stride_;
  int64_t size_;
};

template <typename T>
class StridedArrayView;  // Class requires specialization.

// Wraps a DenseIntElementsAttr that holds backing data for a tensor so that
// int64_t values in a 1-d slice of the tensor can be accessed as if part of an
// ArrayView.
template <>
class StridedArrayView<DenseIntElementsAttr> : StridedArrayViewBase {
 public:
  StridedArrayView(const DenseIntElementsAttr& data, ArrayRef<int64_t> shape,
                   ArrayRef<int64_t> index, int64_t axis)
      : StridedArrayViewBase(shape, index, axis), data_(data) {
    int64_t element_count = 1;
    for (int64_t i = 0, e = shape.size(); i < e; ++i) {
      element_count *= shape[i];
    }
    assert(data.getNumElements() == element_count);
  }

  using StridedArrayViewBase::NextTensorIndex;
  using StridedArrayViewBase::size;

  int64_t operator[](int64_t i) const {
    return data_.getValues<APInt>()[OffsetForIndex(i)].getSExtValue();
  }

 private:
  const DenseIntElementsAttr& data_;
};

// It matches %iota generated from the following mlir codes:
//
// %iota_r1 = "stablehlo.iota"(){iota_dimension = 0} :() -> tensor<Lxi32>
// %iota = "stablehlo.broadcast_in_dim(%iota_r1){
//    broadcast_dimensions = array<i64: $dimensions[0]>},
//
// where %dimensions is of size 1. It usually comes from an IotaOp that is
// folded to IotaOp (rank1) + BroadCastInDimOp.
bool MatchIotaBroadCastInDim(ArrayRef<int64_t> dimensions, Value iota) {
  if (auto iota_broadcast =
          dyn_cast_or_null<stablehlo::BroadcastInDimOp>(iota.getDefiningOp())) {
    if (iota_broadcast.getBroadcastDimensions() != dimensions) return false;
    return isa_and_nonnull<stablehlo::IotaOp>(
        iota_broadcast.getOperand().getDefiningOp());
  }
  return false;
}

// Matches %iota generated from the following mlir codes (rank 2 example):
//
// %iota = stablehlo.constant dense<[[0, 1, 2, ..., L],
//                                  [0, 1, 2, ..., L]
//                                  ...
//                                  [0, 1, 2, ..., L]]>,
// where $dimensions is of size 1.
//
// StridedArrayViews are used to check the iota property across the constant
// data so that the iota dimension does not need to be the (inner) z-dimension.
bool MatchIotaConst(ArrayRef<int64_t> dimensions, Value iota) {
  if (dimensions.size() != 1) return false;
  DenseIntElementsAttr iota_const_attr;
  if (!matchPattern(iota, m_Constant(&iota_const_attr))) return false;

  auto iota_type = iota_const_attr.getType();
  auto iota_shape = iota_type.getShape();
  auto reduce_dim = dimensions[0];
  if (reduce_dim < 0) reduce_dim += iota_type.getRank();

  auto index =
      std::optional<SmallVector<int64_t>>(std::in_place, iota_type.getRank());
  while (index.has_value()) {
    StridedArrayView<DenseIntElementsAttr> array_view(
        iota_const_attr, iota_shape, *index, reduce_dim);
    for (int64_t i = 0; i < array_view.size(); ++i) {
      if (array_view[i] != i) return false;
    }
    index = StridedArrayView<DenseIntElementsAttr>::NextTensorIndex(
        std::move(*index), iota_shape, reduce_dim);
  }

  return true;
}

// Matches %iota generated from the following code (rank 3 example):
//
// %iota_r1 = "stablehlo.iota"(){iota_dimension = 0 : i32} : () ->
// tensor<44xi32> %iota = "stablehlo.reshape"(%iota_r1): (tensor<44xi32>) ->
// tensor<1x1x44xi32>
//
// Where $dimensions is of size 1 and $dimensions[0] = 2.
//
// In general matches a 1-D Iota with multiple dimensions of size 1 added
// through a reshape.
bool MatchReshapedIota(ArrayRef<int64_t> dimensions, Value iota) {
  if (dimensions.size() != 1) return false;
  auto reshape_op =
      dyn_cast_or_null<stablehlo::ReshapeOp>(iota.getDefiningOp());
  if (!reshape_op) return false;
  Value reshape_operand = reshape_op.getOperand();
  auto operand_type =
      mlir::dyn_cast<RankedTensorType>(reshape_operand.getType());
  if (!operand_type || !operand_type.hasStaticShape()) return false;
  auto reshape_type = mlir::cast<RankedTensorType>(reshape_op.getType());

  // Reshape can take a 1-D iota input and add extra dims of size one.
  if (operand_type.getRank() != 1) return false;
  if (!isa_and_nonnull<stablehlo::IotaOp>(reshape_operand.getDefiningOp()))
    return false;

  int64_t iota_dim = dimensions[0];
  for (int64_t i = 0, e = reshape_type.getRank(); i < e; ++i) {
    if (i == iota_dim) {
      if (reshape_type.getDimSize(i) != operand_type.getDimSize(0))
        return false;
    } else if (reshape_type.getDimSize(i) != 1) {
      return false;
    }
  }
  return true;
}

// It returns "true" when Value $iota is obtained from the following mlir code:
//
// $iota = "stablehlo.iota"(){iota_dimension = $dimensions[0]},
//
// where $dimensions must have size 1 and iota can have rank>=1.
// It usually used for matching rank 1 iota since the iotaOp will be folded to
// IotaOp + BroadCastInDimOp except for the case when result shape is rank 1.
bool MatchSingleIota(ArrayRef<int64_t> dimensions, Value iota) {
  if (dimensions.size() != 1) return false;
  if (auto iota_op =
          dyn_cast_or_null<stablehlo::IotaOp>(iota.getDefiningOp())) {
    return dimensions[0] == iota_op.getIotaDimension();
  }
  return false;
}

// It matches %iota generated from the following mlir codes:
//
// %iota_r1 = stablehlo.constant dense<[0, 1, 2, ..., L]>
// %iota = "stablehlo.broadcast_in_dim(%iota_r1){
//    broadcast_dimensions = array<i64: $dimensions[0]>},
//
// where $dimensions is of size 1. It usually comes from an IotaOp that is
// folded to ConstOp (folded rank1 iota) + BroadCastInDimOp.
bool MatchConstIotaBroadCastInDim(ArrayRef<int64_t> dimensions, Value iota) {
  if (dimensions.size() != 1) return false;
  Value broadcast_operand;
  if (auto iota_broadcast =
          dyn_cast_or_null<stablehlo::BroadcastInDimOp>(iota.getDefiningOp())) {
    if (iota_broadcast.getBroadcastDimensions() != dimensions) return false;
    broadcast_operand = iota_broadcast.getOperand();
  } else {
    return false;
  }
  DenseElementsAttr range_const;
  if (!matchPattern(broadcast_operand, m_Constant(&range_const))) return false;
  int index = 0;
  for (auto value : range_const.getValues<APInt>()) {
    if (value != index++) return false;
  }
  return true;
}
}  // namespace

// The following 5 different forms of stablehlo::iota will be matched:
// 1. IotaOp.
// 2. IotaOp + BroadCastInDim.
// 3. IotaOp + Reshape.
// 4. Constant (folded Iota) + BroadCastInDim.
// 5. Constant (folded result).
// Moreover, the dimensions has to match the iota_dimension.
bool MatchIota(ArrayRef<int64_t> dimensions, Value iota) {
  return MatchSingleIota(dimensions, iota) ||
         MatchIotaBroadCastInDim(dimensions, iota) ||
         MatchReshapedIota(dimensions, iota) ||
         MatchConstIotaBroadCastInDim(dimensions, iota) ||
         MatchIotaConst(dimensions, iota);
}

bool MatchIota(DenseIntElementsAttr dimensions, Value iota) {
  llvm::SmallVector<int64_t> dims;
  for (auto val : dimensions.getValues<APInt>()) {
    dims.push_back(val.getSExtValue());
  }
  return MatchIota(llvm::ArrayRef<int64_t>(dims), iota);
}

}  // namespace odml
}  // namespace mlir
