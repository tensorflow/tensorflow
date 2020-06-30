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

#include "absl/memory/memory.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/lite/tools/optimize/sparsity/format_converter.h"

//===----------------------------------------------------------------------===//
// The DenseToSparse Pass.
//
namespace mlir {
namespace TFL {

namespace {
// If sparsity level is below this threadshold, keep the tensor in dense format.
const float kMinSparsityLevel = 0.3;
// Heuristic to check if a block configuration is correct.
const float kBlockOverRandomSparsityRatio = 0.9;

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

float CalculateRandomSparsity(const ElementsAttr& attr,
                              const ShapedType& type) {
  int num_elements = type.getNumElements();
  int num_zeros = 0;

  if (type.getElementType().isF32()) {
    for (const auto val : attr.getValues<float>()) {
      if (val == 0.f) {
        num_zeros++;
      }
    }
  } else if (type.getElementType().isa<quant::QuantizedType>()) {
    for (const auto val : attr.getValues<int8_t>()) {
      if (val == 0) {
        num_zeros++;
      }
    }
  }

  return 1.0 * num_zeros / num_elements;
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
    tflite::optimize::sparsity::FormatConverter<float> format_converter(
        shape, traversal_order, format, b_size, b_map);
    std::vector<float> data;
    data.reserve(type.getNumElements());
    for (const auto val : attr.getValues<float>()) data.push_back(val);
    format_converter.DenseToSparse(data.data());
    sparsity =
        1 - 1.0 * format_converter.GetData().size() / type.getNumElements();
  } else if (type.getElementType().isa<quant::QuantizedType>()) {
    tflite::optimize::sparsity::FormatConverter<int8_t> format_converter(
        shape, traversal_order, format, b_size, b_map);
    std::vector<int8_t> data;
    data.reserve(type.getNumElements());
    for (const auto val : attr.getValues<int8_t>()) data.push_back(val);
    format_converter.DenseToSparse(data.data());
    sparsity =
        1 - 1.0 * format_converter.GetData().size() / type.getNumElements();
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
    Operation* inst,
    const std::vector<std::vector<int>>& supported_block_size) {
  ElementsAttr attr;
  ShapedType type;
  InspectResult result = {};
  if (auto cst = dyn_cast<ConstOp>(inst)) {
    attr = cst.value();
    type = cst.getType().cast<ShapedType>();
  } else if (auto cst = dyn_cast<QConstOp>(inst)) {
    attr = cst.value();
    type = cst.getType().cast<ShapedType>();
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
    if (curr_sparsity / random_sparsity > kBlockOverRandomSparsityRatio) {
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
    const std::vector<int>& block_size, Operation* inst, OpBuilder* builder,
    SparsityParameterAttr* s_param) {
  ElementsAttr attr;
  ShapedType type;
  if (auto cst = dyn_cast<ConstOp>(inst)) {
    attr = cst.value();
    type = cst.getType().cast<ShapedType>();
  } else if (auto cst = dyn_cast<QConstOp>(inst)) {
    attr = cst.value();
    type = cst.getType().cast<ShapedType>();
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

  tflite::optimize::sparsity::FormatConverter<T> format_converter(
      shape, traversal_order, format, b_size, b_map);
  std::vector<T> data;
  data.reserve(type.getNumElements());
  for (const auto val : attr.getValues<T>()) data.push_back(val);
  format_converter.DenseToSparse(data.data());
  auto metadata = format_converter.GetDimMetadata();
  auto compressed_data = format_converter.GetData();
  const int dim_size = metadata.size() / 2;
  std::vector<Attribute> dim_metadata(traversal_order.size());
  for (int i = 0; i < dim_size; i++) {
    if (format[i] == kTfLiteDimDense) {
      dim_metadata[i] = DimensionMetadataAttr::get(
          builder->getStringAttr("DENSE"),
          builder->getI32IntegerAttr(metadata[2 * i][0]),
          builder->getArrayAttr({}), builder->getArrayAttr({}),
          builder->getContext());
    } else {
      dim_metadata[i] = DimensionMetadataAttr::get(
          builder->getStringAttr("SPARSE_CSR"), builder->getI32IntegerAttr(0),
          builder->getI32ArrayAttr(metadata[2 * i]),
          builder->getI32ArrayAttr(metadata[2 * i + 1]), builder->getContext());
    }
  }
  *s_param = SparsityParameterAttr::get(
      builder->getI32ArrayAttr(traversal_order),
      builder->getI32ArrayAttr(b_map), builder->getArrayAttr(dim_metadata),
      builder->getContext());

  return compressed_data;
}

// This pass encodes sparse weights in the model in the proper format, and adds
// Densify() op if necessary. The general algorithm is:
//   1. Get list of operands (weights) of an op that can be sparse.
//   2. Get list of supported block configurations of the op.
//   3. Calculate random sparsity of the weight.
//     3.1. If sparsity level is below the encoding threshold, keep in dense.
//     3.2. If sparsity level is above the encoding threshold, go to 4.
//   4. Try to encode the weight with supported block configurations. If the
//      weight was pruned with the same block config, the blocked sparsity level
//      should match the random sparsity.
//     4.1. Return the matching block config if found.
//     4.2. If no matching block config is found, encode the weight with random
//          sparsity, and add Densify() op to fall back to dense execution.
struct DenseToSparse : public PassWrapper<DenseToSparse, FunctionPass> {
  void runOnFunction() override;
};

void DenseToSparse::runOnFunction() {
  FuncOp func = getFunction();
  OpBuilder builder(func);

  func.walk([&](SparseOpInterface sparse_op) {
    const auto& sparse_operands = sparse_op.GetSparseOperands();
    std::vector<std::vector<int>> supported_block_size;
    for (const int operand : sparse_operands) {
      auto* op = sparse_op.getOperation();
      const auto& value = op->getOperand(operand);

      auto* inst = value.getDefiningOp();
      if (!inst) {
        continue;
      }

      ShapedType type;
      if (isa<ConstOp>(inst)) {
        supported_block_size = sparse_op.GetFloatBlockSize();
        type = dyn_cast<ConstOp>(inst).getType().cast<ShapedType>();
      } else if (isa<QConstOp>(inst)) {
        supported_block_size = sparse_op.GetQuantizedBlockSize();
        type = dyn_cast<QConstOp>(inst).getType().cast<ShapedType>();
      } else {
        continue;
      }

      InspectResult result = InspectWeight(inst, supported_block_size);
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
        std::vector<float> compressed_data =
            BuildSparsityParameterAttribute<float>(result.selected_block_size,
                                                   inst, &builder, &s_param);
        auto compressed_data_type = RankedTensorType::get(
            {static_cast<int64_t>(compressed_data.size())},
            builder.getF32Type());
        auto new_value = DenseElementsAttr::get<float>(compressed_data_type,
                                                       compressed_data);
        auto s_const = builder.create<SparseConstOp>(op->getLoc(), cst.value(),
                                                     s_param, new_value);
        value.replaceAllUsesWith(s_const.getResult());
        cst.erase();
      } else if (auto cst = dyn_cast<QConstOp>(inst)) {
        std::vector<int8_t> compressed_data =
            BuildSparsityParameterAttribute<int8_t>(result.selected_block_size,
                                                    inst, &builder, &s_param);
        auto compressed_data_type = RankedTensorType::get(
            {static_cast<int64_t>(compressed_data.size())},
            builder.getIntegerType(8, true));
        auto new_value = DenseElementsAttr::get<int8_t>(compressed_data_type,
                                                        compressed_data);
        auto s_qconst = builder.create<SparseQConstOp>(
            op->getLoc(), cst.qtypeAttr(), cst.value(), s_param, new_value);
        value.replaceAllUsesWith(s_qconst.getResult());
        cst.erase();
      }

      if (result.needs_densify) {
        const auto value = op->getOperand(operand);
        auto densify =
            builder.create<DensifyOp>(op->getLoc(), value.getType(), value);
        value.replaceAllUsesWith(densify);
        densify.setOperand(value);
      }
    }
  });
}

}  // namespace

// Creates an instance of the TensorFlow Lite dialect DenseToSparse pass.
std::unique_ptr<OperationPass<FuncOp>> CreateDenseToSparsePass() {
  return absl::make_unique<DenseToSparse>();
}

static PassRegistration<DenseToSparse> pass(
    "tfl-dense-to-sparse", "Convert dense tensor to sparse format.");

}  // namespace TFL
}  // namespace mlir
