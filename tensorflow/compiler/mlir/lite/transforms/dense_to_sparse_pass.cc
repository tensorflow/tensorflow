/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// This transformation pass convert dense tensor to sparse format.
#include "tensorflow/compiler/mlir/lite/transforms/dense_to_sparse_pass.h"

#include "absl/memory/memory.h"
#include "Eigen/Core"  // from @eigen_archive
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/kernels/internal/utils/sparsity_format_converter.h"

//===----------------------------------------------------------------------===//
// The DenseToSparse Pass.
//
namespace mlir {
namespace TFL {

namespace {

// If sparsity level is below this threshold, keep the tensor in dense format.
constexpr float kMinSparsityLevel = 0.3;
// Heuristic to check if a block configuration is correct for float constants.
constexpr float kBlockOverRandomSparsityRatio = 0.9;
// After quantization, some non-zero values are set to 0.
// Lower the ratio for identifying block configuration for quantized constants.
constexpr float kBlockOverRandomSparsityRatioQuant = 0.8;

Eigen::half APFloatToEigenHalf(const APFloat& val) {
  uint16_t raw_data = val.bitcastToAPInt().getZExtValue();
  return Eigen::numext::bit_cast<Eigen::half>(raw_data);
}

APFloat EigenHalfToAPFloat(const Eigen::half& val) {
  uint16_t raw_data = Eigen::numext::bit_cast<uint16_t>(val);
  return APFloat(APFloat::IEEEhalf(), APInt(16, raw_data));
}

void PopulateEncodingParams(const std::vector<int>& block_size,
                            std::vector<int>* traversal_order,
                            std::vector<TfLiteDimensionType>* format,
                            std::vector<int>* b_map, std::vector<int>* b_size) {
  const int dims_count = block_size.size();
  traversal_order->resize(dims_count);
  format->resize(dims_count);
  for (int i = 0; i < dims_count; i++) {
    (*traversal_order)[i] = i;
  }
  for (int i = 0; i < dims_count - 1; i++) {
    (*format)[i] = kTfLiteDimDense;
  }
  (*format)[dims_count - 1] = kTfLiteDimSparseCSR;
  *b_map = {};
  *b_size = {};
  int block_rank = 0;
  for (int i = 0; i < dims_count; i++) {
    if (block_size[i] != 1) {
      traversal_order->push_back(block_rank + dims_count);
      format->push_back(kTfLiteDimDense);
      block_rank++;
      b_map->push_back(i);
      b_size->push_back(block_size[i]);
    }
  }
}

inline float GetSparsity(const int num_zeros, const int num_elements) {
  return (1.0 * num_zeros / num_elements);
}

float CalculateRandomSparsity(const ElementsAttr& attr,
                              const ShapedType& type) {
  int num_elements = type.getNumElements();
  int num_zeros = 0;

  if (mlir::isa<FloatType>(type.getElementType())) {
    for (const auto val : attr.getValues<APFloat>()) {
      if (val.isZero()) {
        num_zeros++;
      }
    }
  } else if (mlir::isa<quant::QuantizedType>(type.getElementType())) {
    for (const auto val : attr.getValues<int8_t>()) {
      if (val == 0) {
        num_zeros++;
      }
    }
  }

  return GetSparsity(num_zeros, num_elements);
}

float CalculateBlockSparsity(const ElementsAttr& attr, const ShapedType& type,
                             const std::vector<int>& block_size) {
  float sparsity = 0;
  std::vector<int> shape(2);
  shape[0] = type.getDimSize(0);
  shape[1] = type.getDimSize(1);

  std::vector<int> traversal_order = {};
  std::vector<TfLiteDimensionType> format = {};
  std::vector<int> b_size = {};
  std::vector<int> b_map = {};
  PopulateEncodingParams(block_size, &traversal_order, &format, &b_map,
                         &b_size);

  if (type.getElementType().isF32()) {
    tflite_migration::internal::sparsity::FormatConverter<float>
        format_converter(shape, traversal_order, format, b_size, b_map);
    std::vector<float> data;
    data.reserve(type.getNumElements());
    for (const auto val : attr.getValues<float>()) data.push_back(val);
    format_converter.DenseToSparse(data.data());
    sparsity =
        GetSparsity(type.getNumElements() - format_converter.GetData().size(),
                    type.getNumElements());
  } else if (type.getElementType().isF16()) {
    tflite_migration::internal::sparsity::FormatConverter<Eigen::half>
        format_converter(shape, traversal_order, format, b_size, b_map);
    std::vector<Eigen::half> data;
    data.reserve(type.getNumElements());
    for (const auto& val : attr.getValues<APFloat>())
      data.push_back(APFloatToEigenHalf(val));
    format_converter.DenseToSparse(data.data());
    sparsity =
        GetSparsity(type.getNumElements() - format_converter.GetData().size(),
                    type.getNumElements());
  } else if (mlir::isa<quant::QuantizedType>(type.getElementType())) {
    tflite_migration::internal::sparsity::FormatConverter<int8_t>
        format_converter(shape, traversal_order, format, b_size, b_map);
    std::vector<int8_t> data;
    data.reserve(type.getNumElements());
    for (const auto val : attr.getValues<int8_t>()) data.push_back(val);
    format_converter.DenseToSparse(data.data());
    sparsity =
        GetSparsity(type.getNumElements() - format_converter.GetData().size(),
                    type.getNumElements());
  }

  return sparsity;
}

typedef struct InspectResult {
  // Whether the weight tensor is sparse enough to be compressed.
  bool can_compress;
  // If the weight tensor cannot be encoded in a block configuration that the op
  // supports, a Densify() op will be inserted afterwards to fall back to dense
  // execution.
  bool needs_densify;
  // Among the supported block configs of an op, which got selected to encode
  // the sparse weight.
  std::vector<int> selected_block_size;
} InspectResult;

InspectResult InspectWeight(
    Operation* inst, const std::vector<std::vector<int>>& supported_block_size,
    const float ratio_threshold) {
  ElementsAttr attr;
  ShapedType type;
  InspectResult result = {};
  if (auto cst = dyn_cast<ConstOp>(inst)) {
    attr = cst.getValue();
    type = mlir::cast<ShapedType>(cst.getType());
  } else if (auto cst = dyn_cast<QConstOp>(inst)) {
    attr = cst.getValue();
    type = mlir::cast<ShapedType>(cst.getType());
  } else {
    result.can_compress = false;
    return result;
  }

  // Currently we only support compressing weights of ops:
  //   Conv, DepthwiseConv, TransposeConv, whose filter has rank 4, and
  //   FullyConnected, whose filter has rank 2.
  if (type.getRank() != 2 && type.getRank() != 4) {
    result.can_compress = false;
    return result;
  }

  float random_sparsity = CalculateRandomSparsity(attr, type);
  if (random_sparsity < kMinSparsityLevel) {
    result.can_compress = false;
    return result;
  }

  result.can_compress = true;

  float curr_sparsity = 0;
  std::vector<int> selected_block_size;
  result.needs_densify = true;
  for (const auto& block_size : supported_block_size) {
    curr_sparsity = CalculateBlockSparsity(attr, type, block_size);
    if (curr_sparsity / random_sparsity > ratio_threshold) {
      selected_block_size = block_size;
      result.can_compress = true;
      result.needs_densify = false;
      result.selected_block_size = selected_block_size;
      break;
    }
  }

  return result;
}

template <typename T>
std::vector<T> BuildSparsityParameterAttribute(
    const std::vector<int>& block_size, const T* dense_buffer, Operation* inst,
    OpBuilder* builder, SparsityParameterAttr* s_param) {
  ElementsAttr attr;
  ShapedType type;
  if (auto cst = dyn_cast<ConstOp>(inst)) {
    attr = cst.getValue();
    type = mlir::cast<ShapedType>(cst.getType());
  } else if (auto cst = dyn_cast<QConstOp>(inst)) {
    attr = cst.getValue();
    type = mlir::cast<ShapedType>(cst.getType());
  } else {
    assert(false && "Expected a constant-like op");
  }
  const int dims_count = type.getRank();
  std::vector<int> shape(dims_count);
  for (int i = 0; i < dims_count; i++) {
    shape[i] = type.getDimSize(i);
  }

  std::vector<int> traversal_order = {};
  std::vector<TfLiteDimensionType> format = {};
  std::vector<int> b_size = {};
  std::vector<int> b_map = {};
  PopulateEncodingParams(block_size, &traversal_order, &format, &b_map,
                         &b_size);

  tflite_migration::internal::sparsity::FormatConverter<T> format_converter(
      shape, traversal_order, format, b_size, b_map);
  format_converter.DenseToSparse(dense_buffer);
  const auto& metadata = format_converter.GetDimMetadata();
  const auto& compressed_data = format_converter.GetData();
  const int dim_size = metadata.size() / 2;
  std::vector<DimensionMetadataAttr> dim_metadata(traversal_order.size());
  for (int i = 0; i < dim_size; i++) {
    if (format[i] == kTfLiteDimDense) {
      dim_metadata[i] = DimensionMetadataAttr::get(
          builder->getContext(),
          ::mlir::TFL::DimensionTypeAttr::get(
              builder->getContext(), ::mlir::TFL::DimensionType::DENSE),
          metadata[2 * i][0], {}, {});
    } else {
      dim_metadata[i] = DimensionMetadataAttr::get(
          builder->getContext(),
          ::mlir::TFL::DimensionTypeAttr::get(
              builder->getContext(), ::mlir::TFL::DimensionType::SPARSE_CSR),
          0, metadata[2 * i], metadata[2 * i + 1]);
    }
  }
  *s_param = SparsityParameterAttr::get(builder->getContext(), traversal_order,
                                        b_map, dim_metadata);

  return compressed_data;
}
}  // namespace

void DenseToSparsePass::runOnOperation() {
  func::FuncOp func = getOperation();
  OpBuilder builder(func);

  func.walk([&](SparseOpInterface sparse_op) {
    const auto& sparse_operands = sparse_op.GetSparseOperands();
    std::vector<std::vector<int>> supported_block_size;
    for (int operand : sparse_operands) {
      auto* op = sparse_op.getOperation();
      auto value = op->getOperand(operand);

      auto* inst = value.getDefiningOp();
      if (!inst) {
        continue;
      }

      // There could be a Dequantize op after the weight tensor in cases like
      // fp16 post-training quantization. We need to get the weight from the
      // input of the Dequantize op.
      if (isa<DequantizeOp>(inst)) {
        op = inst;
        value = inst->getOperand(0);
        inst = value.getDefiningOp();
        if (!inst) {
          continue;
        }
        operand = 0;
      }

      ShapedType type;
      float ratio_threshold = kBlockOverRandomSparsityRatio;
      if (isa<ConstOp>(inst)) {
        supported_block_size = sparse_op.GetFloatBlockSize();
        type = mlir::cast<ShapedType>(dyn_cast<ConstOp>(inst).getType());
      } else if (isa<QConstOp>(inst)) {
        supported_block_size = sparse_op.GetQuantizedBlockSize();
        type = mlir::cast<ShapedType>(dyn_cast<QConstOp>(inst).getType());
        ratio_threshold = kBlockOverRandomSparsityRatioQuant;
      } else {
        continue;
      }

      InspectResult result =
          InspectWeight(inst, supported_block_size, ratio_threshold);
      if (!result.can_compress) {
        continue;
      }

      // The weight is not block sparse. Encode with random sparsity.
      if (result.selected_block_size.empty()) {
        result.selected_block_size = std::vector<int>(type.getRank(), 1);
      }

      builder.setInsertionPoint(op);
      SparsityParameterAttr s_param;
      if (auto cst = dyn_cast<ConstOp>(inst)) {
        auto attr = cst.getValue();
        auto type = mlir::cast<ShapedType>(cst.getType());
        if (type.getElementType().isF32()) {
          std::vector<float> dense_data;
          dense_data.reserve(type.getNumElements());
          for (const auto val : attr.getValues<float>())
            dense_data.push_back(val);
          std::vector<float> compressed_data =
              BuildSparsityParameterAttribute<float>(result.selected_block_size,
                                                     dense_data.data(), inst,
                                                     &builder, &s_param);
          auto compressed_data_type = RankedTensorType::get(
              {static_cast<int64_t>(compressed_data.size())},
              builder.getF32Type());
          auto new_value = DenseElementsAttr::get<float>(compressed_data_type,
                                                         compressed_data);
          auto s_const = builder.create<SparseConstOp>(
              op->getLoc(), cst.getValue(), s_param, new_value);
          value.replaceAllUsesWith(s_const.getResult());
          cst.erase();
        } else if (type.getElementType().isF16()) {
          std::vector<Eigen::half> dense_data;
          dense_data.reserve(type.getNumElements());
          for (const auto& val : attr.getValues<APFloat>())
            dense_data.push_back(APFloatToEigenHalf(val));
          std::vector<Eigen::half> compressed_data =
              BuildSparsityParameterAttribute<Eigen::half>(
                  result.selected_block_size, dense_data.data(), inst, &builder,
                  &s_param);
          std::vector<APFloat> apfloat_data;
          apfloat_data.reserve(type.getNumElements());
          for (const auto& val : compressed_data)
            apfloat_data.push_back(EigenHalfToAPFloat(val));
          auto compressed_data_type = RankedTensorType::get(
              {static_cast<int64_t>(compressed_data.size())},
              type.getElementType());
          auto new_value =
              DenseElementsAttr::get(compressed_data_type, apfloat_data);
          auto s_const = builder.create<SparseConstOp>(
              op->getLoc(), cst.getValue(), s_param, new_value);
          value.replaceAllUsesWith(s_const.getResult());
          cst.erase();
        }
      } else if (auto cst = dyn_cast<QConstOp>(inst)) {
        auto attr = cst.getValue();
        auto type = mlir::cast<ShapedType>(cst.getType());
        std::vector<int8_t> dense_data;
        dense_data.reserve(type.getNumElements());
        for (const auto& val : attr.getValues<int8_t>())
          dense_data.push_back(val);
        std::vector<int8_t> compressed_data =
            BuildSparsityParameterAttribute<int8_t>(result.selected_block_size,
                                                    dense_data.data(), inst,
                                                    &builder, &s_param);
        auto compressed_data_type = RankedTensorType::get(
            {static_cast<int64_t>(compressed_data.size())},
            builder.getIntegerType(8, true));
        auto new_value = DenseElementsAttr::get<int8_t>(compressed_data_type,
                                                        compressed_data);
        auto s_qconst =
            builder.create<SparseQConstOp>(op->getLoc(), cst.getQtypeAttr(),
                                           cst.getValue(), s_param, new_value);
        value.replaceAllUsesWith(s_qconst.getResult());
        cst.erase();
      }

      if (result.needs_densify) {
        auto value = op->getOperand(operand);
        auto densify =
            builder.create<DensifyOp>(op->getLoc(), value.getType(), value);
        value.replaceAllUsesWith(densify);
        densify.setOperand(value);
      }
    }
  });
}


// Creates an instance of the TensorFlow Lite dialect DenseToSparse pass.
std::unique_ptr<OperationPass<func::FuncOp>> CreateDenseToSparsePass() {
  return std::make_unique<DenseToSparsePass>();
}

}  // namespace TFL
}  // namespace mlir
