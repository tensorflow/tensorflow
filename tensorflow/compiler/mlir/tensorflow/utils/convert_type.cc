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

#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"

#include <limits>

#include "absl/strings/str_cat.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Support/DebugStringHelper.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dynamic_shape_utils.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

using mlir::Builder;
using mlir::ShapedType;
using mlir::Type;

Status ConvertDataType(DataType dtype, Builder builder, Type* type) {
  switch (dtype) {
    case DT_HALF:
      *type = builder.getF16Type();
      return OkStatus();
    case DT_FLOAT:
      *type = builder.getF32Type();
      return OkStatus();
    case DT_DOUBLE:
      *type = builder.getF64Type();
      return OkStatus();
    case DT_BOOL:
      *type = builder.getIntegerType(1);
      return OkStatus();
    case DT_INT8:
      *type = builder.getIntegerType(8);
      return OkStatus();
    case DT_INT16:
      *type = builder.getIntegerType(16);
      return OkStatus();
    case DT_INT32:
      *type = builder.getIntegerType(32);
      return OkStatus();
    case DT_INT64:
      *type = builder.getIntegerType(64);
      return OkStatus();
    case DT_UINT8:
      *type = builder.getIntegerType(8, /*isSigned=*/false);
      return OkStatus();
    case DT_UINT16:
      *type = builder.getIntegerType(16, /*isSigned=*/false);
      return OkStatus();
    case DT_UINT32:
      *type = builder.getIntegerType(32, /*isSigned=*/false);
      return OkStatus();
    case DT_UINT64:
      *type = builder.getIntegerType(64, /*isSigned=*/false);
      return OkStatus();
    case DT_BFLOAT16:
      *type = builder.getBF16Type();
      return OkStatus();
    case DT_COMPLEX64:
      *type = mlir::ComplexType::get(builder.getF32Type());
      return OkStatus();
    case DT_COMPLEX128:
      *type = mlir::ComplexType::get(builder.getF64Type());
      return OkStatus();
    case tensorflow::DT_FLOAT8_E4M3FN:
      *type = builder.getFloat8E4M3FNType();
      return ::tensorflow::OkStatus();
    case tensorflow::DT_FLOAT8_E5M2:
      *type = builder.getFloat8E5M2Type();
      return ::tensorflow::OkStatus();
    case DT_INT4:
      *type = builder.getIntegerType(4, /*isSigned=*/true);
      return ::tensorflow::OkStatus();
    case DT_UINT4:
      *type = builder.getIntegerType(4, /*isSigned=*/false);
      return ::tensorflow::OkStatus();
#define HANDLE_TF_TYPE(tftype, enumerant, name)             \
  case DT_##enumerant:                                      \
    *type = builder.getType<mlir::tf_type::tftype##Type>(); \
    return OkStatus();
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.def"

    default:
      return errors::Unimplemented(absl::StrCat(
          "Converting DataType '", DataTypeString(dtype), "' to MLIR Type"));
  }
}

Status ConvertScalarTypeToDataType(Type type, DataType* dtype) {
  if (type.isF16()) {
    *dtype = DT_HALF;
    return OkStatus();
  } else if (type.isF32()) {
    *dtype = DT_FLOAT;
    return OkStatus();
  } else if (type.isF64()) {
    *dtype = DT_DOUBLE;
    return OkStatus();
  } else if (type.isBF16()) {
    *dtype = DT_BFLOAT16;
    return OkStatus();
  } else if (type.isFloat8E4M3FN()) {
    *dtype = DT_FLOAT8_E4M3FN;
    return OkStatus();
  } else if (type.isFloat8E5M2()) {
    *dtype = DT_FLOAT8_E5M2;
    return OkStatus();
  } else if (auto itype = type.dyn_cast<mlir::IntegerType>()) {
    switch (itype.getWidth()) {
      case 1:
        *dtype = DT_BOOL;
        return OkStatus();
      case 4:
        *dtype = itype.isUnsigned() ? DT_UINT4 : DT_INT4;
        return OkStatus();
      case 8:
        *dtype = itype.isUnsigned() ? DT_UINT8 : DT_INT8;
        return OkStatus();
      case 16:
        *dtype = itype.isUnsigned() ? DT_UINT16 : DT_INT16;
        return OkStatus();
      case 32:
        *dtype = itype.isUnsigned() ? DT_UINT32 : DT_INT32;
        return OkStatus();
      case 64:
        *dtype = itype.isUnsigned() ? DT_UINT64 : DT_INT64;
        return OkStatus();
      default:
        return errors::Unimplemented(
            absl::StrCat("Converting ", debugString(type), " to DataType"));
    }
  } else if (auto complex_type = type.dyn_cast<mlir::ComplexType>()) {
    auto etype = complex_type.getElementType();
    if (etype.isF32()) {
      *dtype = DT_COMPLEX64;
      return OkStatus();
    } else if (etype.isF64()) {
      *dtype = DT_COMPLEX128;
      return OkStatus();
    }
    return errors::Unimplemented(
        absl::StrCat("Converting ", debugString(type), " to DataType"));
  }

#define HANDLE_TF_TYPE(tftype, enumerant, name)  \
  if (type.isa<mlir::tf_type::tftype##Type>()) { \
    *dtype = DT_##enumerant;                     \
    return OkStatus();                           \
  }
// NOLINTNEXTLINE
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.def"

  return errors::Unimplemented(
      absl::StrCat("Converting ", debugString(type), " to DataType"));
}

Status ConvertToDataType(Type type, DataType* dtype) {
  if (auto stype = type.dyn_cast<ShapedType>()) {
    TF_RETURN_IF_ERROR(
        ConvertScalarTypeToDataType(stype.getElementType(), dtype));
  } else {
    TF_RETURN_IF_ERROR(ConvertScalarTypeToDataType(type, dtype));
  }
  return OkStatus();
}

void ConvertToMlirShape(const TensorShape& input_shape,
                        llvm::SmallVectorImpl<int64_t>* shape) {
  shape->reserve(input_shape.dims());
  for (const auto& d : input_shape) {
    shape->push_back(d.size == kTFDynamicSize ? ShapedType::kDynamic : d.size);
  }
}

Status ConvertToMlirShape(const TensorShapeProto& input_shape,
                          llvm::SmallVectorImpl<int64_t>* shape) {
  shape->reserve(input_shape.dim_size());
  auto& dims = input_shape.dim();
  for (auto& d : dims) {
    if (d.size() > std::numeric_limits<int64_t>::max()) {
      return errors::InvalidArgument("Shape element overflows");
    }
    shape->push_back(d.size() == kTFDynamicSize ? ShapedType::kDynamic
                                                : d.size());
  }
  return OkStatus();
}

absl::StatusOr<mlir::Type> ConvertToMlirTensorType(
    const TensorShapeProto& shape, DataType dtype, mlir::Builder* builder) {
  mlir::Type element_type;
  TF_RETURN_IF_ERROR(ConvertDataType(dtype, *builder, &element_type));
  if (shape.unknown_rank()) {
    return mlir::UnrankedTensorType::get(element_type);
  }
  llvm::SmallVector<int64_t, 4> shape_dims;
  TF_RETURN_IF_ERROR(ConvertToMlirShape(shape, &shape_dims));
  return GetTypeFromTFTensorShape(shape_dims, element_type);
}

}  // namespace tensorflow
