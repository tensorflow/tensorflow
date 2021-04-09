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

#include "absl/base/casts.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/mangling_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/bfloat16.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace tensorflow {

using llvm::ArrayRef;
using llvm::SmallVector;
using mlir::Builder;
using mlir::DenseFPElementsAttr;
using mlir::DenseIntElementsAttr;
using mlir::DenseStringElementsAttr;
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
                                         ShapedType type) {
  auto arr = input_tensor.flat<T>();
  return mlir::DenseElementsAttr::get(
      type, llvm::makeArrayRef(arr.data(), arr.size()));
}

ElementsAttr ConvertBf16Tensor(const Tensor& input_tensor,
                               RankedTensorType type) {
  auto buffer = llvm::makeArrayRef(static_cast<char*>(input_tensor.data()),
                                   input_tensor.TotalBytes());
  return mlir::DenseElementsAttr::getFromRawBuffer(
      type, buffer,
      /*isSplatBuffer=*/type.getNumElements() == 1);
}

ElementsAttr ConvertHalfTensor(const Tensor& tensor, RankedTensorType type) {
  auto buffer = llvm::makeArrayRef(static_cast<char*>(tensor.data()),
                                   tensor.TotalBytes());
  return mlir::DenseElementsAttr::getFromRawBuffer(
      type, buffer,
      /*isSplatBuffer=*/type.getNumElements() == 1);
}

StatusOr<ElementsAttr> ConvertStringTensor(const Tensor& input_tensor,
                                           ShapedType type) {
  // Extract to a vector of StringRefs for converting.
  auto arr = input_tensor.flat<tstring>();
  std::vector<mlir::StringRef> string_refs;
  string_refs.reserve(arr.size());
  for (int i = 0; i < arr.size(); i++) {
    const auto& val = arr(i);
    string_refs.push_back({val.data(), val.size()});
  }

  return DenseStringElementsAttr::get(type, string_refs);
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
    return ConvertFlatTensor<CTYPE>(input_tensor, type);

  // TODO(fengliuai): customize the conversions for quantized and string types.
  switch (input_dtype) {
    CONVERT_FLAT(DT_BOOL, bool)
    CONVERT_FLAT(DT_FLOAT, float)
    CONVERT_FLAT(DT_DOUBLE, double)
    CONVERT_FLAT(DT_INT8, int8)
    CONVERT_FLAT(DT_INT16, int16)
    CONVERT_FLAT(DT_INT32, int32)
    CONVERT_FLAT(DT_INT64, int64)
    CONVERT_FLAT(DT_UINT8, uint8)
    CONVERT_FLAT(DT_UINT16, uint16)
    CONVERT_FLAT(DT_UINT32, uint32)
    CONVERT_FLAT(DT_UINT64, uint64)
    CONVERT_FLAT(DT_COMPLEX64, std::complex<float>)
    CONVERT_FLAT(DT_COMPLEX128, std::complex<double>)

    // BFLOAT16 is a special case that it needs to be cast to double type to
    // match its storage type.
    case DT_BFLOAT16:
      return ConvertBf16Tensor(input_tensor, type);
    case DT_HALF:
      return ConvertHalfTensor(input_tensor, type);

    case DT_STRING:
      return ConvertStringTensor(input_tensor, type);

    default:
      // TODO(shpeisman): restructure code to reuse dialect pointer across
      // calls.
      auto* dialect = builder->getContext()->getLoadedDialect("tf");
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

mlir::TF::ShapeAttr ConvertTypeToTensorShapeAttr(const mlir::Type& type) {
  if (type.isa<mlir::UnrankedTensorType>()) {
    return mlir::TF::ShapeAttr::get(type.getContext(), llvm::None);
  }

  if (auto tensor_type = type.dyn_cast<mlir::RankedTensorType>()) {
    return mlir::TF::ShapeAttr::get(type.getContext(), tensor_type.getShape());
  }

  // If type is not a RankedTensor or UnrankedTensor, it must be a scalar.
  // Empty TensorShape indicates a scalar.
  return mlir::TF::ShapeAttr::get(type.getContext(), ArrayRef<int64_t>());
}

// Converts the tensor shape proto into an MLIR shape attribute.
StatusOr<mlir::Attribute> ConvertTensorShapeProto(const TensorShapeProto& shape,
                                                  mlir::MLIRContext* context) {
  if (shape.unknown_rank())
    return mlir::TF::ShapeAttr::get(context, llvm::None);

  llvm::SmallVector<int64_t, 4> dims;
  dims.reserve(shape.dim().size());
  for (const auto& dim : shape.dim()) {
    dims.push_back(dim.size());
  }
  return mlir::TF::ShapeAttr::get(context, llvm::makeArrayRef(dims));
}

// Converts an MLIR dense string elements attribute to a TensorFlow tensor
// proto.
void ConvertStringElementsAttr(
    const DenseStringElementsAttr attr,
    protobuf::RepeatedPtrField<std::string>* output) {
  for (const auto& val : attr.getRawStringData())
    output->Add({val.data(), val.size()});
}

template <typename T>
void ConvertComplexElementsAttr(const mlir::DenseElementsAttr attr,
                                protobuf::RepeatedField<T>* output) {
  for (const auto& val : attr.getValues<std::complex<T>>()) {
    output->Add(val.real());
    output->Add(val.imag());
  }
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

// Converts an MLIR elements attribute and adds it to specified repeated field.
template <typename T>
void ConvertElementsAttr(const mlir::DenseElementsAttr attr,
                         protobuf::RepeatedField<T>* output) {
  if (attr.isSplat()) {
    output->Add(attr.getSplatValue<T>());
  } else {
    output->Reserve(attr.getNumElements());
    for (auto value : attr.getValues<T>()) output->AddAlreadyReserved(value);
  }
}

// Converts an MLIR elements attribute containing half values and adds it to
// specified repeated field.
void ConvertHalfElementsAttr(const mlir::DenseElementsAttr attr,
                             protobuf::RepeatedField<int>* output) {
  if (attr.isSplat()) {
    output->Add(attr.getSplatValue<Eigen::half>().x);
  } else {
    output->Reserve(attr.getNumElements());
    for (const Eigen::half value : attr.getValues<Eigen::half>())
      output->AddAlreadyReserved(value.x);
  }
}

// Converts an MLIR elements attribute containing int values and adds it to
// specified repeated field.
void ConvertIntElementsAttr(const mlir::DenseIntElementsAttr attr,
                            protobuf::RepeatedField<int>* output) {
  if (attr.isSplat()) {
    output->Add((*attr.begin()).getSExtValue());
  } else {
    output->Reserve(attr.getNumElements());
    for (const llvm::APInt val : attr)
      output->AddAlreadyReserved(val.getSExtValue());
  }
}

void ConvertBfloat16ElementsAttr(const mlir::DenseElementsAttr attr,
                                 protobuf::RepeatedField<int>* output) {
  if (attr.isSplat()) {
    output->Add(attr.getSplatValue<bfloat16>().value);
  } else {
    output->Reserve(attr.getNumElements());
    for (const bfloat16 value : attr.getValues<bfloat16>())
      output->AddAlreadyReserved(value.value);
  }
}

Status ConvertToTensorProto(const ElementsAttr attr, TensorProto* output) {
  auto type = attr.getType();
  auto shape = type.getShape();
  DataType output_dtype;
  TF_RETURN_IF_ERROR(ConvertToDataType(type, &output_dtype));
  output->set_dtype(output_dtype);
  ConvertToTensorShapeProto(shape, output->mutable_tensor_shape());

  if (attr.isa<OpaqueElementsAttr>())
    return ConvertOpaqueElementsAttr(attr.cast<OpaqueElementsAttr>(), output);

  auto dense_attr = attr.dyn_cast<mlir::DenseElementsAttr>();
  if (!dense_attr) return errors::InvalidArgument("Unsupported elements attr");

  switch (output_dtype) {
    case DT_FLOAT:
      ConvertElementsAttr<float>(dense_attr, output->mutable_float_val());
      break;
    case DT_HALF:
      ConvertHalfElementsAttr(dense_attr, output->mutable_half_val());
      break;
    case DT_DOUBLE:
      ConvertElementsAttr(dense_attr, output->mutable_double_val());
      break;
    case DT_QUINT8:
    case DT_UINT8:
    case DT_INT8:
    case DT_QUINT16:
    case DT_UINT16:
    case DT_INT16:
    case DT_INT32:
      ConvertIntElementsAttr(dense_attr.cast<DenseIntElementsAttr>(),
                             output->mutable_int_val());
      break;
    case DT_UINT32:
      ConvertElementsAttr(dense_attr, output->mutable_uint32_val());
      break;
    case DT_UINT64:
      ConvertElementsAttr(dense_attr, output->mutable_uint64_val());
      break;
    case DT_INT64:
      ConvertElementsAttr(dense_attr, output->mutable_int64_val());
      break;
    case DT_BOOL:
      ConvertElementsAttr(dense_attr, output->mutable_bool_val());
      break;
    case DT_BFLOAT16:
      ConvertBfloat16ElementsAttr(dense_attr, output->mutable_half_val());
      break;
    case DT_STRING:
      ConvertStringElementsAttr(dense_attr.cast<DenseStringElementsAttr>(),
                                output->mutable_string_val());
      break;
    case DT_COMPLEX64:
      ConvertComplexElementsAttr(dense_attr, output->mutable_scomplex_val());
      break;
    case DT_COMPLEX128:
      ConvertComplexElementsAttr(dense_attr, output->mutable_dcomplex_val());
      break;
    default:
      return errors::Unimplemented(absl::StrCat("Unimplemented data type ",
                                                DataTypeString(output_dtype)));
  }
  return Status::OK();
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
