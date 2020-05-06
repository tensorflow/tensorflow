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

#include "absl/strings/str_cat.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Support/DebugStringHelper.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
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
      return Status::OK();
    case DT_FLOAT:
      *type = builder.getF32Type();
      return Status::OK();
    case DT_DOUBLE:
      *type = builder.getF64Type();
      return Status::OK();
    case DT_BOOL:
      *type = builder.getIntegerType(1);
      return Status::OK();
    case DT_INT8:
      *type = builder.getIntegerType(8);
      return Status::OK();
    case DT_INT16:
      *type = builder.getIntegerType(16);
      return Status::OK();
    case DT_INT32:
      *type = builder.getIntegerType(32);
      return Status::OK();
    case DT_INT64:
      *type = builder.getIntegerType(64);
      return Status::OK();
    case DT_UINT8:
      *type = builder.getIntegerType(8, /*isSigned=*/false);
      return Status::OK();
    case DT_UINT16:
      *type = builder.getIntegerType(16, /*isSigned=*/false);
      return Status::OK();
    case DT_UINT32:
      *type = builder.getIntegerType(32, /*isSigned=*/false);
      return Status::OK();
    case DT_UINT64:
      *type = builder.getIntegerType(64, /*isSigned=*/false);
      return Status::OK();
    case DT_BFLOAT16:
      *type = builder.getBF16Type();
      return Status::OK();
    case DT_COMPLEX64:
      *type = mlir::ComplexType::get(builder.getF32Type());
      return Status::OK();
    case DT_COMPLEX128:
      *type = mlir::ComplexType::get(builder.getF64Type());
      return Status::OK();
#define HANDLE_TF_TYPE(tftype, enumerant, name)        \
  case DT_##enumerant:                                 \
    *type = builder.getType<mlir::TF::tftype##Type>(); \
    return Status::OK();
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.def"

    default:
      return errors::Unimplemented(absl::StrCat(
          "Converting DataType '", DataTypeString(dtype), "' to MLIR Type"));
  }
}

Status ConvertScalarTypeToDataType(Type type, DataType* dtype) {
  switch (type.getKind()) {
    case mlir::StandardTypes::F16:
      *dtype = DT_HALF;
      return Status::OK();
    case mlir::StandardTypes::F32:
      *dtype = DT_FLOAT;
      return Status::OK();
    case mlir::StandardTypes::F64:
      *dtype = DT_DOUBLE;
      return Status::OK();
    case mlir::StandardTypes::BF16:
      *dtype = DT_BFLOAT16;
      return Status::OK();
    case mlir::StandardTypes::Integer: {
      const auto& itype = type.cast<mlir::IntegerType>();
      switch (itype.getWidth()) {
        case 1:
          *dtype = DT_BOOL;
          return Status::OK();
        case 8:
          *dtype = itype.isUnsigned() ? DT_UINT8 : DT_INT8;
          return Status::OK();
        case 16:
          *dtype = itype.isUnsigned() ? DT_UINT16 : DT_INT16;
          return Status::OK();
        case 32:
          *dtype = itype.isUnsigned() ? DT_UINT32 : DT_INT32;
          return Status::OK();
        case 64:
          *dtype = itype.isUnsigned() ? DT_UINT64 : DT_INT64;
          return Status::OK();
        default:
          return errors::Unimplemented(
              absl::StrCat("Converting ", debugString(type), " to DataType"));
      }
    }
    case mlir::StandardTypes::Complex: {
      auto etype = type.cast<mlir::ComplexType>().getElementType();
      if (etype.isF32()) {
        *dtype = DT_COMPLEX64;
        return Status::OK();
      } else if (etype.isF64()) {
        *dtype = DT_COMPLEX128;
        return Status::OK();
      }
      return errors::Unimplemented(
          absl::StrCat("Converting ", debugString(type), " to DataType"));
    }
#define HANDLE_TF_TYPE(tftype, enumerant, name) \
  case mlir::TF::TensorFlowTypes::enumerant:    \
    *dtype = DT_##enumerant;                    \
    return Status::OK();
// NOLINTNEXTLINE
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.def"
    default:
      return errors::Unimplemented(
          absl::StrCat("Converting ", debugString(type), " to DataType"));
  }
}

Status ConvertToDataType(Type type, DataType* dtype) {
  if (auto stype = type.dyn_cast<ShapedType>()) {
    TF_RETURN_IF_ERROR(
        ConvertScalarTypeToDataType(stype.getElementType(), dtype));
  } else {
    TF_RETURN_IF_ERROR(ConvertScalarTypeToDataType(type, dtype));
  }
  return Status::OK();
}

void ConvertToMlirShape(const TensorShape& input_shape,
                        llvm::SmallVectorImpl<int64_t>* shape) {
  shape->reserve(input_shape.dims());
  for (const auto& d : input_shape) {
    shape->push_back(d.size);
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
    shape->push_back(d.size());
  }
  return Status::OK();
}

StatusOr<mlir::Type> ConvertToMlirTensorType(const TensorShapeProto& shape,
                                             DataType dtype,
                                             mlir::Builder* builder) {
  mlir::Type element_type;
  TF_RETURN_IF_ERROR(ConvertDataType(dtype, *builder, &element_type));
  if (shape.unknown_rank()) {
    return mlir::UnrankedTensorType::get(element_type);
  }
  llvm::SmallVector<int64_t, 4> shape_dims;
  TF_RETURN_IF_ERROR(ConvertToMlirShape(shape, &shape_dims));
  return mlir::RankedTensorType::get(shape_dims, element_type);
}

}  // namespace tensorflow
