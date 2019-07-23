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

// Converts an TensorFlow tensor proto to an MLIR opaque elements attribute.
StatusOr<ElementsAttr> ConvertToOpaqueElementsAttr(
    const TensorProto& input_tensor, ShapedType type, Builder* builder) {
  // TODO(shpeisman): restructure code to reuse dialect pointer across calls.
  auto* dialect = builder->getContext()->getRegisteredDialect("tf");
  return builder->getOpaqueElementsAttr(
      dialect, type, mangling_util::MangleTensor(input_tensor));
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

// Converts an TensorFlow tensor proto to an MLIR dense elements attribute.
// To save the memory held by the attribute, the value is casted to the
// specified type.
template <typename ProtoT, typename MlirT>
typename std::enable_if<IsBatchCopyable<ProtoT, MlirT>::value,
                        StatusOr<ElementsAttr>>::type
ConvertToDenseElementsAttr(
    const tensorflow::protobuf::RepeatedField<ProtoT>& values, ShapedType type,
    Builder* builder) {
  return mlir::DenseElementsAttr::get(
      type, llvm::makeArrayRef(values.data(), values.size()));
}

template <typename ProtoT, typename MlirT>
typename std::enable_if<!IsBatchCopyable<ProtoT, MlirT>::value,
                        StatusOr<ElementsAttr>>::type
ConvertToDenseElementsAttr(
    const tensorflow::protobuf::RepeatedField<ProtoT>& values, ShapedType type,
    Builder* builder) {
  std::vector<MlirT> buff;
  buff.reserve(values.size());
  for (auto value : values) {
    buff.push_back(value);
  }
  return mlir::DenseElementsAttr::get(type, llvm::makeArrayRef(buff));
}

// Convert a TensorFlow tensor from its raw serialization into a
// DenseElementAttr. This is a wrapper around mlir::DenseElementsAttr that
// creates a temporary copy of the data for satisfying strict aliasing
// defensively. TODO(aminim): this extra copy should not be needed,
// DenseElementAttr will perform a similar copy internally.
// Template parameter `T` must match the element type of the `type` argument
// (this is checked in DenseElementsAttr::get()).
template <typename T>
mlir::DenseElementsAttr ConvertToDenseElementsAttr(const absl::Cord& values,
                                                   ShapedType type,
                                                   Builder* builder) {
  DCHECK_EQ((values.size() % sizeof(T)), 0)
      << "unexpected size vs elt type mismatch";
  int n_elements = values.size() / sizeof(T);
  auto data = absl::make_unique<T[]>(n_elements);
  // This assumes that the endianess conversion was handled when loading the
  // tensor in memory.
  values.CopyToArray(reinterpret_cast<char*>(data.get()));
  return mlir::DenseElementsAttr::get(
      type, llvm::makeArrayRef(data.get(), n_elements));
}

// Converts an TensorFlow tensor proto with DT_FLOAT data type into an MLIR
// elements attribute.
StatusOr<ElementsAttr> ConvertFloatTensor(const TensorProto& input_tensor,
                                          ShapedType type, Builder* builder) {
  // When the repeated "float_val" field only has one element, it is converted
  // to a splat elements attribute; When it has more than one element, it is
  // converted to a dense elements attribute; otherwise, convert the whole
  // tensor to an opaque elements attribute if the "tensor_content" field is
  // set.
  auto repeated_val_size = input_tensor.float_val_size();
  if (repeated_val_size == 1 || repeated_val_size == type.getNumElements()) {
    return ConvertToDenseElementsAttr<float, float>(input_tensor.float_val(),
                                                    type, builder);
  }
  auto raw_data = input_tensor.tensor_content();
  if (raw_data.size() == type.getSizeInBits() / 8)
    return ConvertToDenseElementsAttr<float>(raw_data, type, builder);
  return ConvertToOpaqueElementsAttr(input_tensor, type, builder);
}

// Converts an TensorFlow tensor proto with DT_INT32, DT_INT16, DT_INT8,
// DT_UINT8, DT_QUINT8 data type into an MLIR elements attribute.
template <typename T>
StatusOr<ElementsAttr> ConvertIntTensor(const TensorProto& input_tensor,
                                        ShapedType type, Builder* builder) {
  // When the repeated "int_val" field only has one element, it is converted to
  // a splat elements attribute; When it has more than one element, it is
  // converted to a dense elements attribute; otherwise, convert the whole
  // tensor to an opaque elements attribute if the "tensor_content" field is
  // set.
  auto repeated_val_size = input_tensor.int_val_size();
  if (repeated_val_size == 1 || repeated_val_size == type.getNumElements()) {
    return ConvertToDenseElementsAttr<int32_t, T>(input_tensor.int_val(), type,
                                                  builder);
  }
  auto raw_data = input_tensor.tensor_content();
  if (raw_data.size() == type.getSizeInBits() / 8)
    return ConvertToDenseElementsAttr<int32_t>(raw_data, type, builder);

  return ConvertToOpaqueElementsAttr(input_tensor, type, builder);
}

// Converts an TensorFlow tensor proto with DT_INT64 data type into an MLIR
// elements attribute.
StatusOr<ElementsAttr> ConvertInt64Tensor(const TensorProto& input_tensor,
                                          ShapedType type, Builder* builder) {
  // When the repeated "int64_val" field only has one element, it is converted
  // to a splat elements attribute; When it has more than one element, it is
  // converted to a dense elements attribute; otherwise, convert the whole
  // tensor to an opaque elements attribute if the "tensor_content" field is
  // set.
  auto repeated_val_size = input_tensor.int64_val_size();
  if (repeated_val_size == 1 || repeated_val_size == type.getNumElements()) {
    return ConvertToDenseElementsAttr<decltype(input_tensor.int64_val(0)),
                                      uint64_t>(input_tensor.int64_val(), type,
                                                builder);
  }
  auto raw_data = input_tensor.tensor_content();
  if (raw_data.size() == type.getSizeInBits() / 8)
    return ConvertToDenseElementsAttr<int64_t>(raw_data, type, builder);
  return ConvertToOpaqueElementsAttr(input_tensor, type, builder);
}

// Converts an TensorFlow tensor proto with DT_BOOL data type into an MLIR
// elements attribute.
StatusOr<ElementsAttr> ConvertBoolTensor(const TensorProto& input_tensor,
                                         ShapedType type, Builder* builder) {
  // When the repeated "bool_val" field only has one element, it is converted to
  // a splat elements attribute; When it has more than one element, it is
  // converted to a dense elements attribute; otherwise, convert the whole
  // tensor to an opaque elements attribute if the "tensor_content" field is
  // set.
  auto repeated_val_size = input_tensor.bool_val_size();
  if (repeated_val_size == 1 || repeated_val_size == type.getNumElements()) {
    const auto& proto = input_tensor.bool_val();
    return mlir::DenseElementsAttr::get(
        type, llvm::makeArrayRef(proto.data(), proto.size()));
  }
  return ConvertToOpaqueElementsAttr(input_tensor, type, builder);
}

StatusOr<ElementsAttr> ConvertTensorProto(const TensorProto& input_tensor,
                                          Builder* builder) {
  const auto& input_dtype = input_tensor.dtype();
  const auto& input_shape = input_tensor.tensor_shape();
  Type elt_type;
  TF_RETURN_IF_ERROR(ConvertDataType(input_dtype, *builder, &elt_type));
  SmallVector<int64_t, 4> shape;
  TF_RETURN_IF_ERROR(ConvertToMlirShape(input_shape, &shape));
  auto type = builder->getTensorType(shape, elt_type);

  // TODO(fengliuai): customize the conversions for more types.
  switch (input_dtype) {
    case DT_FLOAT:
      return ConvertFloatTensor(input_tensor, type, builder);
    case DT_INT32:
      return ConvertIntTensor<uint32_t>(input_tensor, type, builder);
    case DT_INT64:
      return ConvertInt64Tensor(input_tensor, type, builder);
    case DT_BOOL:
      return ConvertBoolTensor(input_tensor, type, builder);
    default:
      // The value of the opaque elements attribute contains the whole tensor
      // proto, not just the tensor content.

      // TODO(shpeisman): restructure code to reuse dialect pointer across
      // calls.
      auto* dialect = builder->getContext()->getRegisteredDialect("tf");

      return builder->getOpaqueElementsAttr(
          dialect, type, mangling_util::MangleTensor(input_tensor));
  }
}

StatusOr<mlir::ElementsAttr> ConvertTensor(const Tensor& input_tensor,
                                           mlir::Builder* builder) {
  TensorProto input_proto;
  // This decodes the tensor content into a proper proto field.
  input_tensor.AsProtoField(&input_proto);
  return ConvertTensorProto(input_proto, builder);
}

Status ConvertToTensorShapeProto(ArrayRef<int64_t> shape,
                                 TensorShapeProto* output_shape) {
  for (auto d : shape) {
    output_shape->add_dim()->set_size(d);
  }
  return Status::OK();
}

// Converts an MLIR opaque elements attribute to an TensorFlow tensor proto.
Status ConvertOpaqueElementsAttr(const ElementsAttr attr,
                                 TensorProto* output_tensor) {
  if (attr.isa<OpaqueElementsAttr>()) {
    auto mangled_tensor = attr.cast<OpaqueElementsAttr>().getValue();
    absl::string_view tensor_view(mangled_tensor.data(), mangled_tensor.size());
    return mangling_util::DemangleTensor(tensor_view, output_tensor);
  }
  return InvalidArgument("Unexpected elements attribute type from MLIR.");
}

// Converts an MLIR elements attribute to an TensorFlow tensor proto
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

// Converts an MLIR elements attribute to an TensorFlow tensor proto
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

// Converts an MLIR elements attribute to an TensorFlow tensor proto
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

// Converts an MLIR elements attribute to an TensorFlow tensor proto
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
