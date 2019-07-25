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
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "mlir/IR/Types.h"  // TF:local_config_mlir
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
using mlir::Attribute;
using mlir::BoolAttr;
using mlir::Builder;
using mlir::DenseFPElementsAttr;
using mlir::DenseIntElementsAttr;
using mlir::ElementsAttr;
using mlir::FloatAttr;
using mlir::IntegerAttr;
using mlir::OpaqueElementsAttr;
using mlir::ShapedType;
using mlir::SplatElementsAttr;
using mlir::Type;
using tensorflow::errors::InvalidArgument;

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
      return InvalidArgument("Shape element overflows");
    }
    shape->push_back(d.size());
  }
  return Status::OK();
}

// Converts a TensorFlow tensor to an MLIR opaque elements attribute.
StatusOr<ElementsAttr> ConvertToOpaqueElementsAttr(const Tensor& input_tensor,
                                                   ShapedType type,
                                                   Builder* builder) {
  TensorProto tensor_proto;
  input_tensor.AsProtoTensorContent(&tensor_proto);
  // TODO(shpeisman): restructure code to reuse dialect pointer across calls.
  auto* dialect = builder->getContext()->getRegisteredDialect("tf");
  return builder->getOpaqueElementsAttr(
      dialect, type, mangling_util::MangleTensor(tensor_proto));
}

// Template predicate that provides a constant member `value` equal to true if
// a sequence of `From` values can be copied wholesale to locations for `To`
// values.

// Primary template declaration
template <typename From, typename To, typename Enable = void>
struct IsBatchCopyable;

// Partial template specialization: allow wholesale copy for the same type
template <typename Self>
struct IsBatchCopyable<Self, Self> : std::true_type {};

// SFINAE: integral types depend on the bitwidth
template <typename From, typename To>
struct IsBatchCopyable<
    From, To,
    typename std::enable_if<std::is_integral<From>::value &&
                            std::is_integral<To>::value>::type> {
  static constexpr bool value =
      std::numeric_limits<From>::digits == std::numeric_limits<To>::digits;
};

// Converts a TensorFlow tensor into an MLIR elements attribute.
template <typename T>
StatusOr<ElementsAttr> ConvertFlatTensor(const Tensor& input_tensor,
                                         ShapedType type, Builder* builder) {
  auto arr = input_tensor.flat<T>();
  return mlir::DenseElementsAttr::get(
      type, llvm::makeArrayRef(arr.data(), arr.size()));
}

// Converts a TensorFlow tensor proto with DT_BOOL data type into an MLIR
// elements attribute.
StatusOr<ElementsAttr> ConvertBoolTensor(const Tensor& input_tensor,
                                         ShapedType type, Builder* builder) {
  // When the repeated "bool_val" field only has one element, it is converted to
  // a splat elements attribute; When it has more than one element, it is
  // converted to a dense elements attribute; otherwise, convert the whole
  // tensor to an opaque elements attribute if the "tensor_content" field is
  // set.
  return ConvertToOpaqueElementsAttr(input_tensor, type, builder);
}

StatusOr<ElementsAttr> ConvertTensor(const Tensor& input_tensor,
                                     Builder* builder) {
  const auto& input_dtype = input_tensor.dtype();
  const auto& input_shape = input_tensor.shape();
  Type elt_type;
  TF_RETURN_IF_ERROR(ConvertDataType(input_dtype, *builder, &elt_type));
  SmallVector<int64_t, 4> shape;
  ConvertToMlirShape(input_shape, &shape);
  auto type = builder->getTensorType(shape, elt_type);

  // TODO(fengliuai): customize the conversions for more types.
  switch (input_dtype) {
    case DT_FLOAT:
      return ConvertFlatTensor<float>(input_tensor, type, builder);
    case DT_INT32:
      return ConvertFlatTensor<int32>(input_tensor, type, builder);
    case DT_INT64:
      return ConvertFlatTensor<int64>(input_tensor, type, builder);
    case DT_BOOL:
      return ConvertBoolTensor(input_tensor, type, builder);
    default:
      // The value of the opaque elements attribute contains the whole tensor
      // proto, not just the tensor content.

      // TODO(shpeisman): restructure code to reuse dialect pointer across
      // calls.
      auto* dialect = builder->getContext()->getRegisteredDialect("tf");

      TensorProto tensor_proto;
      input_tensor.AsProtoTensorContent(&tensor_proto);
      return builder->getOpaqueElementsAttr(
          dialect, type, mangling_util::MangleTensor(tensor_proto));
  }
}

StatusOr<ElementsAttr> ConvertTensorProto(const TensorProto& input_tensor,
                                          Builder* builder) {
  Tensor t;
  if (!t.FromProto(input_tensor))
    return InvalidArgument("Failed to parse input_tensor.");
  return ConvertTensor(t, builder);
}

Status ConvertToTensorShapeProto(ArrayRef<int64_t> shape,
                                 TensorShapeProto* output_shape) {
  for (auto d : shape) {
    output_shape->add_dim()->set_size(d);
  }
  return Status::OK();
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
// with the float_val field updated.
Status ConvertFloatElementsAttr(const ElementsAttr attr,
                                TensorProto* output_tensor) {
  if (auto elts = attr.dyn_cast<DenseFPElementsAttr>()) {
    for (auto value : elts.getValues<float>()) {
      output_tensor->add_float_val(value);
    }
  } else {
    return ConvertOpaqueElementsAttr(attr, output_tensor);
  }
  return Status::OK();
}

// Converts an MLIR elements attribute to a TensorFlow tensor proto
// with the int_val field updated.
Status ConvertIntElementsAttr(const mlir::ElementsAttr attr,
                              TensorProto* output_tensor) {
  if (auto elts = attr.dyn_cast<DenseIntElementsAttr>()) {
    for (auto val : elts) {
      output_tensor->add_int_val(val.getSExtValue());
    }
  } else {
    return ConvertOpaqueElementsAttr(attr, output_tensor);
  }
  return Status::OK();
}

// Converts an MLIR elements attribute to a TensorFlow tensor proto
// with the int64_val field updated.
Status ConvertInt64ElementsAttr(const mlir::ElementsAttr attr,
                                TensorProto* output_tensor) {
  if (auto elts = attr.dyn_cast<DenseIntElementsAttr>()) {
    for (auto val : elts) {
      output_tensor->add_int64_val(val.getSExtValue());
    }
  } else {
    return ConvertOpaqueElementsAttr(attr, output_tensor);
  }
  return Status::OK();
}

// Converts an MLIR elements attribute to a TensorFlow tensor proto
// with bool_val field updated.
Status ConvertBoolElementsAttr(const mlir::ElementsAttr attr,
                               TensorProto* output_tensor) {
  if (auto elts = attr.dyn_cast<DenseIntElementsAttr>()) {
    for (auto val : elts) {
      output_tensor->add_bool_val(val.getBoolValue());
    }
  } else {
    return ConvertOpaqueElementsAttr(attr, output_tensor);
  }
  return Status::OK();
}

Status ConvertToTensorProto(const ElementsAttr attr,
                            TensorProto* output_tensor) {
  auto type = attr.getType();
  auto shape = type.getShape();
  DataType output_dtype;
  TF_RETURN_IF_ERROR(ConvertToDataType(type, &output_dtype));
  output_tensor->set_dtype(output_dtype);
  TF_RETURN_IF_ERROR(
      ConvertToTensorShapeProto(shape, output_tensor->mutable_tensor_shape()));

  switch (output_dtype) {
    case DT_FLOAT:
      return ConvertFloatElementsAttr(attr, output_tensor);
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
