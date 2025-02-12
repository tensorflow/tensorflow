/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/utils/convert_type.h"

#include "absl/status/statusor.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_generated.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/errors.h"

namespace tflite {

using absl::StatusOr;

namespace errors = tensorflow::errors;

tflite::TensorType ConvertTypeToTensorType(mlir::Type type) {
  if (type.isF16()) {
    return tflite::TensorType_FLOAT16;
  } else if (type.isBF16()) {
    return tflite::TensorType_BFLOAT16;
  } else if (type.isF32()) {
    return tflite::TensorType_FLOAT32;
  } else if (type.isF64()) {
    return tflite::TensorType_FLOAT64;
  } else if (mlir::isa<mlir::TF::StringType>(type)) {
    return tflite::TensorType_STRING;
  } else if (auto complex_type = mlir::dyn_cast<mlir::ComplexType>(type)) {
    if (complex_type.getElementType().isF32()) {
      return tflite::TensorType_COMPLEX64;
    } else if (complex_type.getElementType().isF64()) {
      return tflite::TensorType_COMPLEX128;
    }
    llvm_unreachable("invalid complex Type in conversion");
  } else if (auto itype = mlir::dyn_cast<mlir::IntegerType>(type)) {
    switch (itype.getWidth()) {
      case 1:
        return tflite::TensorType_BOOL;
      case 4:
        if (itype.isUnsigned())
          llvm_unreachable("invalid unsigned 4bit integer Type in conversion");
        else
          return tflite::TensorType_INT4;
      case 8:
        if (itype.isUnsigned())
          return tflite::TensorType_UINT8;
        else
          return tflite::TensorType_INT8;
      case 16:
        return tflite::TensorType_INT16;
      case 32:
        return tflite::TensorType_INT32;
      case 64:
        if (itype.isUnsigned())
          return tflite::TensorType_UINT64;
        else
          return tflite::TensorType_INT64;
      default:
        llvm_unreachable("invalid integer Type in conversion");
    }
  }
  llvm_unreachable("invalid Type in conversion");
}

mlir::Type ConvertElementType(tflite::TensorType type, mlir::Builder builder) {
  switch (type) {
    case tflite::TensorType_FLOAT16:
      return builder.getF16Type();
    case tflite::TensorType_BFLOAT16:
      return builder.getBF16Type();
    case tflite::TensorType_FLOAT32:
      return builder.getF32Type();
    case tflite::TensorType_FLOAT64:
      return builder.getF64Type();
    case tflite::TensorType_INT32:
      return builder.getIntegerType(32);
    case tflite::TensorType_UINT16:
      return builder.getIntegerType(16, /*isSigned=*/false);
    case tflite::TensorType_UINT32:
      return builder.getIntegerType(32, /*isSigned=*/false);
    case tflite::TensorType_UINT8:
      return builder.getIntegerType(8, /*isSigned=*/false);
    case tflite::TensorType_INT64:
      return builder.getIntegerType(64);
    case tflite::TensorType_STRING:
      return mlir::TF::StringType::get(builder.getContext());
    case tflite::TensorType_BOOL:
      return builder.getI1Type();
    case tflite::TensorType_INT16:
      return builder.getIntegerType(16);
    case tflite::TensorType_COMPLEX64:
      return mlir::ComplexType::get(builder.getF32Type());
    case tflite::TensorType_COMPLEX128:
      return mlir::ComplexType::get(builder.getF64Type());
    case tflite::TensorType_INT4:
      return builder.getIntegerType(4);
    case tflite::TensorType_INT8:
      return builder.getIntegerType(8);
    case tflite::TensorType_UINT64:
      return builder.getIntegerType(64, /*isSigned=*/false);
    case tflite::TensorType_RESOURCE:
      return mlir::TF::ResourceType::get(builder.getContext());
    case tflite::TensorType_VARIANT:
      return mlir::TF::VariantType::get(builder.getContext());
  }
}

tensorflow::DataType TflTypeToTfType(tflite::TensorType type) {
  switch (type) {
    case tflite::TensorType_BOOL:
      return tensorflow::DT_BOOL;
    case tflite::TensorType_COMPLEX64:
      return tensorflow::DT_COMPLEX64;
    case tflite::TensorType_COMPLEX128:
      return tensorflow::DT_COMPLEX128;
    case tflite::TensorType_FLOAT16:
      return tensorflow::DT_HALF;
    case tflite::TensorType_BFLOAT16:
      return tensorflow::DT_BFLOAT16;
    case tflite::TensorType_FLOAT32:
      return tensorflow::DT_FLOAT;
    case tflite::TensorType_FLOAT64:
      return tensorflow::DT_DOUBLE;
    // TODO(b/246806634): Tensorflow DT_INT4 type doesn't exist yet
    case tflite::TensorType_INT4:
      return tensorflow::DT_INT8;
    case tflite::TensorType_INT8:
      return tensorflow::DT_INT8;
    case tflite::TensorType_INT16:
      return tensorflow::DT_INT16;
    case tflite::TensorType_INT32:
      return tensorflow::DT_INT32;
    case tflite::TensorType_UINT32:
      return tensorflow::DT_UINT32;
    case tflite::TensorType_INT64:
      return tensorflow::DT_INT64;
    case tflite::TensorType_STRING:
      return tensorflow::DT_STRING;
    case tflite::TensorType_UINT8:
      return tensorflow::DT_UINT8;
    case tflite::TensorType_UINT16:
      return tensorflow::DT_UINT16;
    case tflite::TensorType_UINT64:
      return tensorflow::DT_UINT64;
    case tflite::TensorType_RESOURCE:
      return tensorflow::DT_RESOURCE;
    case tflite::TensorType_VARIANT:
      return tensorflow::DT_VARIANT;
  }
}

absl::StatusOr<tflite::TensorType> TfTypeToTflType(tensorflow::DataType type) {
  switch (type) {
    case tensorflow::DT_BOOL:
      return tflite::TensorType_BOOL;
    case tensorflow::DT_COMPLEX64:
      return tflite::TensorType_COMPLEX64;
    case tensorflow::DT_COMPLEX128:
      return tflite::TensorType_COMPLEX128;
    case tensorflow::DT_HALF:
      return tflite::TensorType_FLOAT16;
    case tensorflow::DT_BFLOAT16:
      return tflite::TensorType_BFLOAT16;
    case tensorflow::DT_FLOAT:
      return tflite::TensorType_FLOAT32;
    case tensorflow::DT_DOUBLE:
      return tflite::TensorType_FLOAT64;
    case tensorflow::DT_INT8:
      return tflite::TensorType_INT8;
    case tensorflow::DT_INT16:
      return tflite::TensorType_INT16;
    case tensorflow::DT_INT32:
      return tflite::TensorType_INT32;
    case tensorflow::DT_UINT32:
      return tflite::TensorType_UINT32;
    case tensorflow::DT_INT64:
      return tflite::TensorType_INT64;
    case tensorflow::DT_UINT64:
      return tflite::TensorType_UINT64;
    case tensorflow::DT_STRING:
      return tflite::TensorType_STRING;
    case tensorflow::DT_UINT8:
      return tflite::TensorType_UINT8;
    case tensorflow::DT_RESOURCE:
      return tflite::TensorType_RESOURCE;
    case tensorflow::DT_VARIANT:
      return tflite::TensorType_VARIANT;
    default:
      return errors::InvalidArgument("unsupported tensor data type", type);
  }
}

mlir::Type GetShapeStrippedType(mlir::TypeAttr type_attr) {
  auto type = type_attr.getValue();
  auto shaped_type = mlir::dyn_cast<mlir::ShapedType>(type);
  if (shaped_type) {
    return shaped_type.getElementType();
  } else {
    return type;
  }
}

bool NotFromQuantOpOrSameQuantType(mlir::Value val, mlir::TypeAttr qtype_attr) {
  auto val_defn_op = val.getDefiningOp();
  mlir::TFL::QuantizeOp q_op =
      llvm::dyn_cast_or_null<mlir::TFL::QuantizeOp>(val_defn_op);
  if (!q_op) return true;

  // Ignore shape details - we're really only trying to
  // check if quantization is the same.
  auto stripped_src_qtype = GetShapeStrippedType(q_op.getQtypeAttr());
  auto stripped_qtype = GetShapeStrippedType(qtype_attr);
  return stripped_src_qtype == stripped_qtype;
}

}  // namespace tflite
