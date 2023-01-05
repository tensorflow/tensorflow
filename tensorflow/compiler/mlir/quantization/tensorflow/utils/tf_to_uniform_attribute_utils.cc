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

#include <functional>
#include <memory>
#include <set>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/ops/uniform_op_quant_spec.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/utils.h"
#include "tensorflow/core/util/quantization/uniform_quant_ops_attr.pb.h"

namespace mlir::quant {

using QuantMethod =
    tensorflow::quantization::QuantizationMethod::ExperimentalMethod;

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

void FillQuantizationAttributes(PatternRewriter& rewriter, Operation* op,
                                NamedAttrList& attrs,
                                llvm::StringMap<Attribute>& identifier_to_attr,
                                QuantMethod quantization_method) {
  // TODO(b/259374419): Support broader quantization schemes
  absl::flat_hash_map<std::string, int> min_max_scheme_for_8bit_narrow;
  min_max_scheme_for_8bit_narrow = {{"min", -127}, {"max", 127}};

  std::set<std::string> quantization_attributes;
  if (quantization_method ==
      tensorflow::quantization::QuantizationMethod::DYNAMIC_RANGE) {
    quantization_attributes = {
        "rhs_quantization_min_val",
        "rhs_quantization_max_val",
    };
  } else {
    quantization_attributes = {
        "lhs_quantization_min_val",    "lhs_quantization_max_val",
        "rhs_quantization_min_val",    "rhs_quantization_max_val",
        "output_quantization_min_val", "output_quantization_max_val",
    };
  }

  for (const auto& attr : quantization_attributes) {
    auto quant_val = absl::StrContains(attr, "min")
                         ? min_max_scheme_for_8bit_narrow["min"]
                         : min_max_scheme_for_8bit_narrow["max"];
    auto quant_val_attr = rewriter.getI64IntegerAttr(quant_val);
    attrs.push_back(rewriter.getNamedAttr(attr, quant_val_attr));
  }
}

LogicalResult FillAttributesForUniformQuantizedDotOp(
    PatternRewriter& rewriter, Operation* op,
    llvm::StringMap<Attribute>& identifier_to_attr,
    QuantMethod quantization_method, bool enable_per_channel_quantization) {
  NamedAttrList attrs;

  // Fill quantization related attributes.
  FillQuantizationAttributes(rewriter, op, attrs, identifier_to_attr,
                             quantization_method);

  if (!(quantization_method ==
        tensorflow::quantization::QuantizationMethod::DYNAMIC_RANGE)) {
    // Per-channel activation is not supported
    attrs.push_back(rewriter.getNamedAttr("lhs_quantization_axis",
                                          rewriter.getI64IntegerAttr(-1)));
    attrs.push_back(rewriter.getNamedAttr("output_quantization_axis",
                                          rewriter.getI64IntegerAttr(-1)));
  }

  std::unique_ptr<OpQuantSpec> spec = GetUniformOpQuantSpec(op);
  absl::flat_hash_set<int> operands = spec->quantizable_operands;
  int quant_dim = -1;
  if (enable_per_channel_quantization && operands.size() == 1) {
    quant_dim = spec->coeff_op_quant_dim[*(spec->quantizable_operands.begin())];
  }
  attrs.push_back(rewriter.getNamedAttr("rhs_quantization_axis",
                                        rewriter.getI64IntegerAttr(quant_dim)));

  op->setAttrs(rewriter.getDictionaryAttr(attrs));

  return success();
}

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

  // Fill quantization related attributes.
  FillQuantizationAttributes(rewriter, op, attrs, identifier_to_attr,
                             quantization_method);

  if (quantization_method !=
      tensorflow::quantization::QuantizationMethod::DYNAMIC_RANGE) {
    // Per-channel activation is not supported
    attrs.push_back(rewriter.getNamedAttr("lhs_quantization_axis",
                                          rewriter.getI64IntegerAttr(-1)));
    attrs.push_back(rewriter.getNamedAttr("output_quantization_axis",
                                          rewriter.getI64IntegerAttr(-1)));
  }

  std::unique_ptr<OpQuantSpec> spec = GetUniformOpQuantSpec(op);
  absl::flat_hash_set<int> operands = spec->quantizable_operands;
  int quant_dim = -1;
  if (enable_per_channel_quantization && operands.size() == 1) {
    quant_dim = spec->coeff_op_quant_dim[*(spec->quantizable_operands.begin())];
  }
  attrs.push_back(rewriter.getNamedAttr("rhs_quantization_axis",
                                        rewriter.getI64IntegerAttr(quant_dim)));

  op->setAttrs(rewriter.getDictionaryAttr(attrs));

  return success();
}

}  // namespace mlir::quant
