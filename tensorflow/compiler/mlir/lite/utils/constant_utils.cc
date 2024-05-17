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

#include <complex>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_attributes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/mangling_util.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/platform/status.h"
#include "tsl/platform/statusor.h"

namespace mlir {
namespace TFL {

absl::StatusOr<TypedAttr> CreateTypedAttr(ShapedType shaped_type, int value) {
  Type element_type = shaped_type.getElementType();
  if (element_type.isF16()) {
    auto floatType = mlir::FloatType::getF16(element_type.getContext());
    auto floatAttr = mlir::FloatAttr::get(floatType, static_cast<float>(value));
    std::vector<Attribute> floatValues({floatAttr});
    return DenseElementsAttr::get(shaped_type, floatValues);
  } else if (element_type.isBF16()) {
    auto floatType = mlir::FloatType::getBF16(element_type.getContext());
    auto floatAttr = mlir::FloatAttr::get(floatType, static_cast<float>(value));
    std::vector<Attribute> floatValues({floatAttr});
    return DenseElementsAttr::get(shaped_type, floatValues);
  } else if (element_type.isF32()) {
    return DenseElementsAttr::get<float>(shaped_type,
                                         static_cast<float>(value));
  } else if (auto complex_type =
                 mlir::dyn_cast<mlir::ComplexType>(element_type)) {
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

      return mlir::TF::TensorProtoAttr::get(shaped_type, mangled);
    } else {
      return tensorflow::Status(absl::StatusCode::kInvalidArgument,
                                "Unsupported type");
    }
  } else if (auto itype = mlir::dyn_cast<mlir::IntegerType>(element_type)) {
    if (element_type.isSignedInteger()) {
      switch (itype.getWidth()) {
        case 8:
          return DenseElementsAttr::get<int8_t>(shaped_type,
                                                static_cast<int8_t>(value));
          break;
        case 16:
          return DenseElementsAttr::get<int16_t>(shaped_type,
                                                 static_cast<int16_t>(value));
          break;
        case 32:
          return DenseElementsAttr::get<int32_t>(shaped_type,
                                                 static_cast<int32_t>(value));
          break;
        case 64:
          return DenseElementsAttr::get<int64_t>(shaped_type,
                                                 static_cast<int64_t>(value));
          break;
        default:
          return tensorflow::Status(absl::StatusCode::kInvalidArgument,
                                    "Unsupported type");
      }
    } else {
      switch (itype.getWidth()) {
        case 8:
          return DenseElementsAttr::get<uint8_t>(shaped_type,
                                                 static_cast<uint8_t>(value));
          break;
        case 16:
          return DenseElementsAttr::get<uint16_t>(shaped_type,
                                                  static_cast<uint16_t>(value));
          break;
        case 32:
          return DenseElementsAttr::get<uint32_t>(shaped_type,
                                                  static_cast<uint32_t>(value));
          break;
        case 64:
          return DenseElementsAttr::get<uint64_t>(shaped_type,
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
}

// Returns a Constant op with a splat vector value.
absl::StatusOr<arith::ConstantOp> CreateConstOpWithVectorValue(
    PatternRewriter* rewriter, Location loc, ShapedType shaped_type,
    int value) {
  ShapedType dense_type = RankedTensorType::get(shaped_type.getShape(),
                                                shaped_type.getElementType());
  auto attr = CreateTypedAttr(dense_type, value);

  return rewriter->create<arith::ConstantOp>(loc, dense_type,
                                             cast<TypedAttr>(*attr));
}

absl::StatusOr<arith::ConstantOp> CreateConstOpWithSingleValue(
    PatternRewriter* rewriter, Location loc, ShapedType shaped_type,
    int value) {
  ShapedType scalar_type =
      RankedTensorType::get({}, shaped_type.getElementType());
  auto attr = CreateTypedAttr(scalar_type, value);

  return rewriter->create<arith::ConstantOp>(loc, scalar_type,
                                             cast<TypedAttr>(*attr));
}

}  // namespace TFL
}  // namespace mlir
