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
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "mlir/IR/Types.h"  // TF:local_config_mlir
#include "mlir/Support/DebugStringHelper.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"

using mlir::Builder;
using mlir::ShapedType;
using mlir::Type;

namespace tensorflow {

Status ConvertDataType(const DataType& dtype, Builder builder, Type* type) {
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
    case DT_UINT8:
      *type = builder.getIntegerType(8);
      return Status::OK();
    case DT_INT16:
    case DT_UINT16:
      *type = builder.getIntegerType(16);
      return Status::OK();
    case DT_INT32:
    case DT_UINT32:
      *type = builder.getIntegerType(32);
      return Status::OK();
    case DT_INT64:
    case DT_UINT64:
      *type = builder.getIntegerType(64);
      return Status::OK();
    case DT_BFLOAT16:
      *type = builder.getBF16Type();
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
          *dtype = DT_INT8;
          return Status::OK();
        case 16:
          *dtype = DT_INT16;
          return Status::OK();
        case 32:
          *dtype = DT_INT32;
          return Status::OK();
        case 64:
          *dtype = DT_INT64;
          return Status::OK();
        default:
          return errors::Unimplemented(
              absl::StrCat("Converting ", debugString(type), " to DataType"));
      }
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

}  // namespace tensorflow
