/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/quantization/tensorflow/utils/tf_to_uniform_attribute_utils.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/common/quantization_lib/quantization_utils.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/ops/uniform_op_quant_spec.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/core/util/quantization/uniform_quant_ops_attr.pb.h"

namespace mlir::quant {

using QuantMethod = tensorflow::quantization::QuantizationMethod::PresetMethod;

enum class OpType {
  kDynamicRangeOp,  // Dynamic Range kernels only have rhs attr.
  kUnaryOp,         // Unary ops have one min/max attr.
  kBinaryOp,        // Binary ops have lhs/rhs attr.
  kQuantizationOp,  // Quantization ops have input/output attr.
};

// For each op type, the following axis carries axis information:
// kDynamicRangeOp: rhs_quantization_axis will carry axis information.
// kUnaryOp: quantization_axis will carry axis information.
// kBinaryOp: Among {lhs, rhs, output}_quantization_axis, only check rhs.
// kQuantizationOp: Among {input, output}_quantization_axis, only check input.
// We therefore check exemplary 3 axes {rhs_, input_, }quantization_axis from
// previous accumulations.
constexpr std::array<absl::string_view, 3> kQuantizationAxisAttrs = {
    "input_quantization_axis", "quantization_axis", "rhs_quantization_axis"};

// Common suffixes for attributes used in FillQuantizationAttributes.
constexpr std::array<absl::string_view, 2> kSuffixes = {"_min_val", "_max_val"};

Attribute GetWindowStridesValue(
    PatternRewriter& rewriter, llvm::StringMap<Attribute>& identifier_to_attr) {
  ArrayAttr stride = identifier_to_attr["strides"].dyn_cast<ArrayAttr>();
  const int stride_h = stride[1].cast<IntegerAttr>().getInt();
  const int stride_w = stride[2].cast<IntegerAttr>().getInt();
  return rewriter.getI64ArrayAttr({stride_h, stride_w});
}

Attribute GetLhsDilationValue(PatternRewriter& rewriter,
                              llvm::StringMap<Attribute>& identifier_to_attr) {
  return rewriter.getI64ArrayAttr({1, 1});
}

Attribute GetRhsDilationValue(PatternRewriter& rewriter,
                              llvm::StringMap<Attribute>& identifier_to_attr) {
  ArrayAttr dilations = identifier_to_attr["dilations"].dyn_cast<ArrayAttr>();
  const int dilation_h = dilations[1].cast<IntegerAttr>().getInt();
  const int dilation_w = dilations[2].cast<IntegerAttr>().getInt();
  return rewriter.getI64ArrayAttr({dilation_h, dilation_w});
}

Attribute GetPaddingValue(PatternRewriter& rewriter,
                          llvm::StringMap<Attribute>& identifier_to_attr) {
  llvm::StringRef padding =
      identifier_to_attr["padding"].dyn_cast<StringAttr>().getValue();
  return rewriter.getStringAttr(padding);
}

Attribute GetExplicitPaddingValue(
    PatternRewriter& rewriter, llvm::StringMap<Attribute>& identifier_to_attr) {
  ArrayAttr explicit_padding =
      identifier_to_attr["explicit_paddings"].dyn_cast<ArrayAttr>();
  return explicit_padding;
}

Attribute GetDimensionNumbersValue(
    PatternRewriter& rewriter, llvm::StringMap<Attribute>& identifier_to_attr) {
  // Only NHWC is lifted in TF-quant and the corresponding dimension number is
  // [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f].

  tensorflow::UniformQuantizedConvolutionDimensionNumbersAttr dimension_numbers;
  if (!tensorflow::protobuf::TextFormat::ParseFromString(
          R"pb(
            input_batch_dimension: 0
            input_feature_dimension: 3
            input_spatial_dimensions: 1
            input_spatial_dimensions: 2
            kernel_output_feature_dimension: 3
            kernel_input_feature_dimension: 2
            kernel_spatial_dimensions: 0
            kernel_spatial_dimensions: 1
            output_batch_dimension: 0
            output_feature_dimension: 3
            output_spatial_dimensions: 1
            output_spatial_dimensions: 2
          )pb",
          &dimension_numbers)) {
    return rewriter.getStringAttr("");
  }
  return rewriter.getStringAttr(dimension_numbers.SerializeAsString());
}

Attribute GetBatchGroupCountValue(
    PatternRewriter& rewriter, llvm::StringMap<Attribute>& identifier_to_attr) {
  // Only 1 case is supported.
  return rewriter.getI64IntegerAttr(1);
}

Attribute GetQuantizationAxis(PatternRewriter& rewriter, Operation* op,
                              const int operand_index) {
  auto* defining_op = op->getOperand(operand_index).getDefiningOp();
  for (auto attr : kQuantizationAxisAttrs) {
    if (defining_op->hasAttr(attr)) {
      return defining_op->getAttr(attr);
    }
  }
  // Not found.
  return rewriter.getI64IntegerAttr(-1);
}

LogicalResult CheckIfAttrIs8Bit(const std::string& attr, Operation* op,
                                bool& is_8_bit) {
  Type element_type;
  if (attr == "lhs_quantization" || attr == "input_quantization" ||
      attr == "quantization") {
    if (op->getNumOperands() < 1) {
      return failure();
    }
    element_type = getElementTypeOrSelf(op->getOperand(0).getType());
  }
  if (attr == "rhs_quantization") {
    if (op->getNumOperands() < 2) {
      return failure();
    }
    element_type = getElementTypeOrSelf(op->getOperand(1).getType());
  }
  if (attr == "output_quantization") {
    if (op->getNumResults() < 1) {
      return failure();
    }
    element_type = getElementTypeOrSelf(op->getOpResult(0).getType());
  }
  if (element_type) {
    is_8_bit = element_type.isa<TF::Qint8Type>();
    return success();
  }
  return failure();
}

LogicalResult FillQuantizationAttributes(
    PatternRewriter& rewriter, Operation* op, NamedAttrList& attrs,
    llvm::StringMap<Attribute>& identifier_to_attr, OpType op_type) {
  absl::flat_hash_map<std::string, int> min_max_scheme_for_8bit = {
      {"min", -128}, {"max", 127}};
  absl::flat_hash_map<std::string, int> min_max_schema_for_32bit = {
      {"min", -2147483648}, {"max", 2147483647}};

  std::vector<std::string> quantization_attributes;
  switch (op_type) {
    case OpType::kDynamicRangeOp:
      quantization_attributes = {"rhs_quantization"};
      break;
    case OpType::kUnaryOp:
      quantization_attributes = {"quantization"};
      break;
    case OpType::kBinaryOp:
      quantization_attributes = {"lhs_quantization", "rhs_quantization",
                                 "output_quantization"};
      break;
    case OpType::kQuantizationOp:
      quantization_attributes = {"input_quantization", "output_quantization"};
      break;
    default:
      quantization_attributes = {};
      break;
  }

  for (const auto& attr : quantization_attributes) {
    bool attr_is_8_bit;
    if (failed(CheckIfAttrIs8Bit(attr, op, attr_is_8_bit))) {
      return failure();
    }
    for (int i = 0; i < kSuffixes.size(); i++) {
      int64_t quant_val;
      if (attr_is_8_bit) {
        quant_val = i == 0 ? min_max_scheme_for_8bit["min"]
                           : min_max_scheme_for_8bit["max"];
      } else {
        quant_val = i == 0 ? min_max_schema_for_32bit["min"]
                           : min_max_schema_for_32bit["max"];
      }
      std::string attr_minmax = absl::StrCat(attr, kSuffixes[i]);
      attrs.push_back(rewriter.getNamedAttr(
          attr_minmax, rewriter.getI64IntegerAttr(quant_val)));
    }
  }
  return success();
}

// This LogicalResult covers both the hybrid and fully quantized op cases.
LogicalResult FillAttributesForUniformQuantizedDotOp(
    PatternRewriter& rewriter, Operation* op,
    llvm::StringMap<Attribute>& identifier_to_attr,
    QuantMethod quantization_method, bool enable_per_channel_quantization) {
  NamedAttrList attrs;

  if (quantization_method ==
      tensorflow::quantization::QuantizationMethod::METHOD_DYNAMIC_RANGE_INT8) {
    // Fill quantization related attributes for Hybrid op.
    if (failed(FillQuantizationAttributes(rewriter, op, attrs,
                                          identifier_to_attr,
                                          OpType::kDynamicRangeOp))) {
      return failure();
    }
  } else {
    // Fill quantization related attributes for fully quantized op.
    if (failed(FillQuantizationAttributes(
            rewriter, op, attrs, identifier_to_attr, OpType::kBinaryOp))) {
      return failure();
    }
    // Per-channel activation is not supported
    attrs.push_back(rewriter.getNamedAttr("lhs_quantization_axis",
                                          rewriter.getI64IntegerAttr(-1)));
  }

  std::unique_ptr<OpQuantSpec> spec = GetUniformOpQuantSpec(op);
  absl::flat_hash_set<int> operands = spec->quantizable_operands;
  int quant_dim = -1;
  if (enable_per_channel_quantization && operands.size() == 1) {
    quant_dim = spec->coeff_op_quant_dim[*(operands.begin())];
  }
  attrs.push_back(rewriter.getNamedAttr("rhs_quantization_axis",
                                        rewriter.getI64IntegerAttr(quant_dim)));
  attrs.push_back(rewriter.getNamedAttr("output_quantization_axis",
                                        rewriter.getI64IntegerAttr(quant_dim)));

  op->setAttrs(rewriter.getDictionaryAttr(attrs));

  return success();
}

// This LogicalResult covers both the hybrid and fully quantized op cases.
LogicalResult FillAttributesForUniformQuantizedConvolutionOp(
    PatternRewriter& rewriter, Operation* op,
    llvm::StringMap<Attribute>& identifier_to_attr,
    QuantMethod quantization_method, bool enable_per_channel_quantization) {
  NamedAttrList attrs;
  absl::flat_hash_map<std::string, Attribute (*)(PatternRewriter&,
                                                 llvm::StringMap<Attribute>&)>
      attribute_getter_map;

  attribute_getter_map = {{"window_strides", GetWindowStridesValue},
                          {"lhs_dilation", GetLhsDilationValue},
                          {"rhs_dilation", GetRhsDilationValue},
                          {"padding", GetPaddingValue},
                          {"explicit_padding", GetExplicitPaddingValue},
                          {"dimension_numbers", GetDimensionNumbersValue},
                          {"batch_group_count", GetBatchGroupCountValue}};

  for (auto& attr : op->getAttrs()) {
    llvm::StringRef attr_name = attr.getName().getValue();
    if (attribute_getter_map.find(attr_name.str()) !=
        attribute_getter_map.end()) {
      auto attr_val =
          (attribute_getter_map[attr_name.str()])(rewriter, identifier_to_attr);
      attrs.push_back(rewriter.getNamedAttr(attr_name, attr_val));
    }
  }

  auto feature_group_cnt_attr = llvm::StringRef("feature_group_count");
  int feature_group_cnt = 1;
  ShapedType input_shape = op->getOperand(0).getType().dyn_cast<ShapedType>();
  if (!input_shape) {
    return op->emitError(
        "Only input with known shape is supported for Uniform Quantized "
        "opset.");
  }

  if (op->getParentOfType<func::FuncOp>().getName().contains("depthwise_")) {
    feature_group_cnt = input_shape.getDimSize(3);
  }

  attrs.push_back(rewriter.getNamedAttr(
      feature_group_cnt_attr, rewriter.getI64IntegerAttr(feature_group_cnt)));

  if (quantization_method ==
      tensorflow::quantization::QuantizationMethod::METHOD_DYNAMIC_RANGE_INT8) {
    // Fill quantization related attributes for Hybrid op.
    if (failed(FillQuantizationAttributes(rewriter, op, attrs,
                                          identifier_to_attr,
                                          OpType::kDynamicRangeOp))) {
      return failure();
    }
  } else {
    // Fill quantization related attributes for fully quantized op.
    if (failed(FillQuantizationAttributes(
            rewriter, op, attrs, identifier_to_attr, OpType::kBinaryOp))) {
      return failure();
    }
  }

  if (quantization_method !=
      tensorflow::quantization::QuantizationMethod::METHOD_DYNAMIC_RANGE_INT8) {
    // Per-channel activation is not supported
    attrs.push_back(rewriter.getNamedAttr("lhs_quantization_axis",
                                          rewriter.getI64IntegerAttr(-1)));
  }

  std::unique_ptr<OpQuantSpec> spec = GetUniformOpQuantSpec(op);
  absl::flat_hash_set<int> operands = spec->quantizable_operands;
  int quant_dim = -1;
  if (enable_per_channel_quantization && operands.size() == 1) {
    quant_dim = spec->coeff_op_quant_dim[*(operands.begin())];
  }
  attrs.push_back(rewriter.getNamedAttr("rhs_quantization_axis",
                                        rewriter.getI64IntegerAttr(quant_dim)));
  attrs.push_back(rewriter.getNamedAttr("output_quantization_axis",
                                        rewriter.getI64IntegerAttr(quant_dim)));

  op->setAttrs(rewriter.getDictionaryAttr(attrs));

  return success();
}

LogicalResult FillAttributesForUniformQuantizedAddOp(
    PatternRewriter& rewriter, Operation* op,
    llvm::StringMap<Attribute>& identifier_to_attr,
    const QuantMethod quantization_method,
    const bool enable_per_channel_quantization) {
  NamedAttrList attrs;

  // Fill quantization related attributes.
  if (failed(FillQuantizationAttributes(rewriter, op, attrs, identifier_to_attr,
                                        OpType::kBinaryOp))) {
    return failure();
  }
  Attribute activation_quantization_axis = rewriter.getI64IntegerAttr(-1);
  if (enable_per_channel_quantization) {
    // If either of lhs or rhs is per-channel quantized, the quantization axis
    // must match for lhs, rhs, and output.
    activation_quantization_axis =
        GetQuantizationAxis(rewriter, op, /*operand_index=*/0);
    if (activation_quantization_axis == rewriter.getI64IntegerAttr(-1)) {
      activation_quantization_axis =
          GetQuantizationAxis(rewriter, op, /*operand_index=*/1);
    }
  }
  attrs.push_back(rewriter.getNamedAttr("lhs_quantization_axis",
                                        activation_quantization_axis));
  attrs.push_back(rewriter.getNamedAttr("rhs_quantization_axis",
                                        activation_quantization_axis));
  attrs.push_back(rewriter.getNamedAttr("output_quantization_axis",
                                        activation_quantization_axis));
  op->setAttrs(rewriter.getDictionaryAttr(attrs));

  return success();
}

LogicalResult FillAttributesForUniformQuantizedClipByValueOp(
    PatternRewriter& rewriter, Operation* op,
    llvm::StringMap<Attribute>& identifier_to_attr,
    QuantMethod quantization_method, bool enable_per_channel_quantization) {
  NamedAttrList attrs;

  // Fill quantization related attributes.
  if (failed(FillQuantizationAttributes(rewriter, op, attrs, identifier_to_attr,
                                        OpType::kUnaryOp))) {
    return failure();
  }

  Attribute activation_quantization_axis = rewriter.getI64IntegerAttr(-1);
  if (enable_per_channel_quantization) {
    activation_quantization_axis =
        GetQuantizationAxis(rewriter, op, /*operand_index=*/0);
  }
  attrs.push_back(
      rewriter.getNamedAttr("quantization_axis", activation_quantization_axis));
  op->setAttrs(rewriter.getDictionaryAttr(attrs));

  return success();
}

LogicalResult FillAttributesForUniformRequantizeOp(
    PatternRewriter& rewriter, Operation* op,
    llvm::StringMap<Attribute>& identifier_to_attr,
    QuantMethod quantization_method, bool enable_per_channel_quantization) {
  NamedAttrList attrs;

  // Fill quantization related attributes.
  if (failed(FillQuantizationAttributes(rewriter, op, attrs, identifier_to_attr,
                                        OpType::kQuantizationOp))) {
    return failure();
  }

  Attribute activation_quantization_axis = rewriter.getI64IntegerAttr(-1);
  Attribute output_quantization_axis = rewriter.getI64IntegerAttr(-1);
  // TODO(b/296916785): Revisit axis assignment logic.
  if (enable_per_channel_quantization) {
    activation_quantization_axis =
        GetQuantizationAxis(rewriter, op, /*operand_index=*/0);

    auto output_scale_type = op->getOperand(3).getType().dyn_cast<ShapedType>();
    if (!output_scale_type) {
      return failure();
    }
    if (output_scale_type.hasRank() && 0 < output_scale_type.getRank()) {
      output_quantization_axis = activation_quantization_axis;
    }
  }
  // For per-axis -> per-axis requantization, input and output quantization
  // axis must be equal.
  attrs.push_back(rewriter.getNamedAttr("input_quantization_axis",
                                        activation_quantization_axis));
  attrs.push_back(rewriter.getNamedAttr("output_quantization_axis",
                                        output_quantization_axis));
  op->setAttrs(rewriter.getDictionaryAttr(attrs));

  return success();
}

LogicalResult FillAttributesForUniformQuantizeOp(
    PatternRewriter& rewriter, Operation* op,
    llvm::StringMap<Attribute>& identifier_to_attr,
    QuantMethod quantization_method, bool enable_per_channel_quantization) {
  NamedAttrList attrs;

  // Fill quantization related attributes.
  if (failed(FillQuantizationAttributes(rewriter, op, attrs, identifier_to_attr,
                                        OpType::kUnaryOp))) {
    return failure();
  }
  Attribute quantization_axis = rewriter.getI64IntegerAttr(-1);
  // TODO(b/296916785): Revisit axis assignment logic.
  if (enable_per_channel_quantization) {
    quantization_axis = rewriter.getI64IntegerAttr(3);
  }

  attrs.push_back(
      rewriter.getNamedAttr("quantization_axis", quantization_axis));
  op->setAttrs(rewriter.getDictionaryAttr(attrs));
  return success();
}
}  // namespace mlir::quant
