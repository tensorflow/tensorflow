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

#include "tensorflow/compiler/mlir/lite/utils/constant_utils.h"

#include <string>

#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_attributes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/mangling_util.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/platform/status.h"

namespace mlir {
namespace TFL {

tsl::StatusOr<arith::ConstantOp> CreateConstOpWithSingleValue(
    PatternRewriter* rewriter, Location loc, ShapedType shaped_type,
    int value) {
  Type element_type = shaped_type.getElementType();
  ShapedType scalar_type = RankedTensorType::get({}, element_type);
  Attribute attr;
  if (element_type.isF16()) {
    auto floatType = mlir::FloatType::getF16(element_type.getContext());
    auto floatAttr = mlir::FloatAttr::get(floatType, static_cast<float>(value));
    std::vector<Attribute> floatValues({floatAttr});
    attr = DenseElementsAttr::get(scalar_type, floatValues);
  } else if (element_type.isBF16()) {
    auto floatType = mlir::FloatType::getBF16(element_type.getContext());
    auto floatAttr = mlir::FloatAttr::get(floatType, static_cast<float>(value));
    std::vector<Attribute> floatValues({floatAttr});
    attr = DenseElementsAttr::get(scalar_type, floatValues);
  } else if (element_type.isF32()) {
    attr =
        DenseElementsAttr::get<float>(scalar_type, static_cast<float>(value));
  } else if (auto complex_type = element_type.dyn_cast<mlir::ComplexType>()) {
    auto etype = complex_type.getElementType();
    if (etype.isF32()) {
      tensorflow::TensorProto repr;
      repr.set_dtype(tensorflow::DT_COMPLEX64);

      tensorflow::TensorShapeProto* shape = repr.mutable_tensor_shape();
      shape->set_unknown_rank(false);
      shape->add_dim()->set_size(int64_t{1});
      std::string content;
      auto complex_value = std::complex<float>(static_cast<float>(value), 0.0f);
      content.assign(reinterpret_cast<const char*>(&complex_value),
                     sizeof(complex_value));
      repr.set_tensor_content(content);
      std::string mangled = tensorflow::mangling_util::MangleTensor(repr);

      attr = mlir::TF::TensorProtoAttr::get(scalar_type, mangled);
    } else {
      return tensorflow::Status(absl::StatusCode::kInvalidArgument,
                                "Unsupported type");
    }
  } else if (auto itype = element_type.dyn_cast<mlir::IntegerType>()) {
    if (element_type.isSignedInteger()) {
      switch (itype.getWidth()) {
        case 8:
          attr = DenseElementsAttr::get<int8_t>(scalar_type,
                                                static_cast<int8_t>(value));
          break;
        case 16:
          attr = DenseElementsAttr::get<int16_t>(scalar_type,
                                                 static_cast<int16_t>(value));
          break;
        case 32:
          attr = DenseElementsAttr::get<int32_t>(scalar_type,
                                                 static_cast<int32_t>(value));
          break;
        case 64:
          attr = DenseElementsAttr::get<int64_t>(scalar_type,
                                                 static_cast<int64_t>(value));
          break;
        default:
          return tensorflow::Status(absl::StatusCode::kInvalidArgument,
                                    "Unsupported type");
      }
    } else {
      switch (itype.getWidth()) {
        case 8:
          attr = DenseElementsAttr::get<uint8_t>(scalar_type,
                                                 static_cast<uint8_t>(value));
          break;
        case 16:
          attr = DenseElementsAttr::get<uint16_t>(scalar_type,
                                                  static_cast<uint16_t>(value));
          break;
        case 32:
          attr = DenseElementsAttr::get<uint32_t>(scalar_type,
                                                  static_cast<uint32_t>(value));
          break;
        case 64:
          attr = DenseElementsAttr::get<uint64_t>(scalar_type,
                                                  static_cast<uint64_t>(value));
          break;
        default:
          return tensorflow::Status(absl::StatusCode::kInvalidArgument,
                                    "Unsupported type");
      }
    }
  } else {
    return tensorflow::Status(absl::StatusCode::kInvalidArgument,
                              "Unsupported type");
  }
  return rewriter->create<arith::ConstantOp>(loc, scalar_type, attr);
}

}  // namespace TFL
}  // namespace mlir
