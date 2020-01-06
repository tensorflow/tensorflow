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

#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"

#include <limits>

#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"  // TF:llvm-project
#include "mlir/IR/Builders.h"  // TF:llvm-project
#include "mlir/IR/StandardTypes.h"  // TF:llvm-project
#include "mlir/IR/Types.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/mangling_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace tensorflow {

using llvm::ArrayRef;
using llvm::SmallVector;
using mlir::Builder;
using mlir::DenseFPElementsAttr;
using mlir::DenseIntElementsAttr;
using mlir::ElementsAttr;
using mlir::OpaqueElementsAttr;
using mlir::RankedTensorType;
using mlir::ShapedType;
using mlir::Type;
using tensorflow::errors::InvalidArgument;

static TensorProto ConvertToProto(const Tensor& input_tensor,
                                  bool use_tensor_content = true) {
  TensorProto tensor_proto;
  // Using tensor content (mostly*) reduces serialization overhead during RPC
  // calls, but is less human reader friendly. People reading protobufs are less
  // frequent than serialization, so default to using tensor content
  // representation.
  // * For scalars and short strings it may be marginally worse and a more
  //   intelligent decision could be made by caller.
  if (use_tensor_content)
    input_tensor.AsProtoTensorContent(&tensor_proto);
  else
    input_tensor.AsProtoField(&tensor_proto);
  return tensor_proto;
}

static std::string MangleTensor(const Tensor& tensor) {
  return mangling_util::MangleTensor(ConvertToProto(tensor));
}

// Converts a TensorFlow tensor into an MLIR elements attribute.
template <typename T>
StatusOr<ElementsAttr> ConvertFlatTensor(const Tensor& input_tensor,
                                         ShapedType type, Builder* builder) {
  auto arr = input_tensor.flat<T>();
  return mlir::DenseElementsAttr::get(
      type, llvm::makeArrayRef(arr.data(), arr.size()));
}

StatusOr<ElementsAttr> ConvertTensor(const Tensor& input_tensor,
                                     Builder* builder) {
  const auto& input_dtype = input_tensor.dtype();
  const auto& input_shape = input_tensor.shape();
  Type elt_type;
  TF_RETURN_IF_ERROR(ConvertDataType(input_dtype, *builder, &elt_type));
  SmallVector<int64_t, 4> shape;
  ConvertToMlirShape(input_shape, &shape);
  auto type = RankedTensorType::get(shape, elt_type);

#define CONVERT_FLAT(DTYPE, CTYPE) \
  case DTYPE:                      \
    return ConvertFlatTensor<CTYPE>(input_tensor, type, builder);

  // TODO(fengliuai): customize the conversions for more types.
  switch (input_dtype) {
    CONVERT_FLAT(DT_BOOL, bool)
    CONVERT_FLAT(DT_FLOAT, float)
    CONVERT_FLAT(DT_DOUBLE, double)
    CONVERT_FLAT(DT_INT32, int32)
    CONVERT_FLAT(DT_INT64, int64)
    default:
      // TODO(shpeisman): restructure code to reuse dialect pointer across
      // calls.
      auto* dialect = builder->getContext()->getRegisteredDialect("tf");
      return OpaqueElementsAttr::get(dialect, type, MangleTensor(input_tensor));
  }

#undef CONVERT_FLAT
}

StatusOr<ElementsAttr> ConvertTensorProto(const TensorProto& input_tensor,
                                          Builder* builder) {
  Tensor t;
  if (!t.FromProto(input_tensor))
    return InvalidArgument("Failed to parse input_tensor.");
  return ConvertTensor(t, builder);
}

void ConvertToTensorShapeProto(ArrayRef<int64_t> shape,
                               TensorShapeProto* output_shape) {
  for (auto d : shape) {
    output_shape->add_dim()->set_size(d);
  }
}

PartialTensorShape ConvertTypeToTensorShape(const mlir::Type& type) {
  if (type.isa<mlir::UnrankedTensorType>()) {
    // An empty PartialTensorShape indicates an unranked tensor.
    return PartialTensorShape();
  }

  if (auto tensor_type = type.dyn_cast<mlir::RankedTensorType>()) {
    TensorShapeProto tensor_shape_proto;
    ConvertToTensorShapeProto(tensor_type.getShape(), &tensor_shape_proto);
    return PartialTensorShape(tensor_shape_proto);
  }

  // If type is not a RankedTensor or UnrankedTensor, it must be a scalar.
  // Empty TensorShape indicates a scalar.
  return TensorShape();
}

// Converts an MLIR opaque elements attribute to a TensorFlow tensor proto.
Status ConvertOpaqueElementsAttr(const ElementsAttr attr,
                                 TensorProto* output_tensor) {
  if (attr.isa<OpaqueElementsAttr>()) {
    auto mangled_tensor = attr.cast<OpaqueElementsAttr>().getValue();
    absl::string_view tensor_view(mangled_tensor.data(), mangled_tensor.size());
    return mangling_util::DemangleTensor(tensor_view, output_tensor);
  }
  return InvalidArgument("Unexpected elements attribute type from MLIR.");
}

// Converts an MLIR elements attribute to a TensorFlow tensor proto
// with the double_val field updated.
Status ConvertDoubleElementsAttr(const ElementsAttr attr,
                                 TensorProto* output_tensor) {
  if (auto elts = attr.dyn_cast<DenseFPElementsAttr>()) {
    if (elts.isSplat()) {
      output_tensor->add_double_val(elts.getSplatValue<double>());
    } else {
      for (auto value : elts.getValues<double>())
        output_tensor->add_double_val(value);
    }
    return Status::OK();
  }
  return ConvertOpaqueElementsAttr(attr, output_tensor);
}

// Converts an MLIR elements attribute to a TensorFlow tensor proto
// with the float_val field updated.
Status ConvertFloatElementsAttr(const ElementsAttr attr,
                                TensorProto* output_tensor) {
  if (auto elts = attr.dyn_cast<DenseFPElementsAttr>()) {
    if (elts.isSplat()) {
      output_tensor->add_float_val(elts.getSplatValue<float>());
    } else {
      for (auto value : elts.getValues<float>())
        output_tensor->add_float_val(value);
    }
    return Status::OK();
  }
  return ConvertOpaqueElementsAttr(attr, output_tensor);
}

// Converts an MLIR elements attribute to a TensorFlow tensor proto
// with the half_val field updated.
Status ConvertHalfElementsAttr(const ElementsAttr attr,
                               TensorProto* output_tensor) {
  if (auto elts = attr.dyn_cast<DenseFPElementsAttr>()) {
    if (elts.isSplat()) {
      output_tensor->add_half_val(
          (*elts.begin()).bitcastToAPInt().getSExtValue());
    } else {
      for (auto value : elts.getFloatValues())
        output_tensor->add_half_val(value.bitcastToAPInt().getSExtValue());
    }
    return Status::OK();
  }
  return ConvertOpaqueElementsAttr(attr, output_tensor);
}

// Converts an MLIR elements attribute to a TensorFlow tensor proto
// with the int_val field updated.
Status ConvertIntElementsAttr(const mlir::ElementsAttr attr,
                              TensorProto* output_tensor) {
  if (auto elts = attr.dyn_cast<DenseIntElementsAttr>()) {
    if (elts.isSplat()) {
      output_tensor->add_int_val((*elts.begin()).getSExtValue());
    } else {
      for (auto val : elts) output_tensor->add_int_val(val.getSExtValue());
    }
    return Status::OK();
  }
  return ConvertOpaqueElementsAttr(attr, output_tensor);
}

// Converts an MLIR elements attribute to a TensorFlow tensor proto
// with the int64_val field updated.
Status ConvertInt64ElementsAttr(const mlir::ElementsAttr attr,
                                TensorProto* output_tensor) {
  if (auto elts = attr.dyn_cast<DenseIntElementsAttr>()) {
    if (elts.isSplat()) {
      output_tensor->add_int64_val((*elts.begin()).getSExtValue());
    } else {
      for (auto val : elts) output_tensor->add_int64_val(val.getSExtValue());
    }
    return Status::OK();
  }
  return ConvertOpaqueElementsAttr(attr, output_tensor);
}

// Converts an MLIR elements attribute to a TensorFlow tensor proto
// with bool_val field updated.
Status ConvertBoolElementsAttr(const mlir::ElementsAttr attr,
                               TensorProto* output_tensor) {
  if (auto elts = attr.dyn_cast<DenseIntElementsAttr>()) {
    for (auto val : elts) {
      output_tensor->add_bool_val(val.getBoolValue());
    }
    return Status::OK();
  }
  return ConvertOpaqueElementsAttr(attr, output_tensor);
}

Status ConvertToTensorProto(const ElementsAttr attr,
                            TensorProto* output_tensor) {
  auto type = attr.getType();
  auto shape = type.getShape();
  DataType output_dtype;
  TF_RETURN_IF_ERROR(ConvertToDataType(type, &output_dtype));
  output_tensor->set_dtype(output_dtype);
  ConvertToTensorShapeProto(shape, output_tensor->mutable_tensor_shape());

  switch (output_dtype) {
    case DT_FLOAT:
      return ConvertFloatElementsAttr(attr, output_tensor);
    case DT_HALF:
      // Handles both DenseFPElementsAttr and OpaqueElementsAttr.
      return ConvertHalfElementsAttr(attr, output_tensor);
    case DT_DOUBLE:
      return ConvertDoubleElementsAttr(attr, output_tensor);
    case DT_QUINT8:
    case DT_UINT8:
    case DT_INT8:
    case DT_QUINT16:
    case DT_UINT16:
    case DT_INT16:
    case DT_INT32:
      return ConvertIntElementsAttr(attr, output_tensor);
    case DT_INT64:
      return ConvertInt64ElementsAttr(attr, output_tensor);
    case DT_BOOL:
      return ConvertBoolElementsAttr(attr, output_tensor);
    default:
      return ConvertOpaqueElementsAttr(attr.cast<OpaqueElementsAttr>(),
                                       output_tensor);
  }
}

Status ConvertToTensor(const mlir::ElementsAttr attr, Tensor* output_tensor) {
  TensorProto tensor_proto;
  TF_RETURN_IF_ERROR(ConvertToTensorProto(attr, &tensor_proto));
  if (!output_tensor->FromProto(tensor_proto)) {
    return InvalidArgument("Couldn't convert tensor proto to tensor.");
  }
  return Status::OK();
}

StatusOr<mlir::ElementsAttr> DecodeOpaqueTensor(
    const mlir::OpaqueElementsAttr input_attr, mlir::Builder builder) {
  // TODO(antiagainst): The following logic, albeit simple, involves copying the
  // tensor content multiple times, which is bad. Figure out a better way to
  // achieve the purpose.
  Tensor tensor;
  TF_RETURN_IF_ERROR(ConvertToTensor(input_attr, &tensor));
  return ConvertTensor(tensor, &builder);
}

}  // namespace tensorflow
