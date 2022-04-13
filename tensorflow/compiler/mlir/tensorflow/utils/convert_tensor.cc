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

#include <cstdint>
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
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/tensor_util.h"
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
  return ElementsAttr(mlir::DenseElementsAttr::get(
      type, llvm::makeArrayRef(arr.data(), arr.size())));
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

  return ElementsAttr(DenseStringElementsAttr::get(type, string_refs));
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

  // TODO(fengliuai): customize the conversions for quantized types.
  switch (input_dtype) {
    CONVERT_FLAT(DT_BOOL, bool)
    CONVERT_FLAT(DT_FLOAT, float)
    CONVERT_FLAT(DT_DOUBLE, double)
    CONVERT_FLAT(DT_INT8, int8)
    CONVERT_FLAT(DT_INT16, int16)
    CONVERT_FLAT(DT_INT32, int32)
    CONVERT_FLAT(DT_INT64, int64_t)
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
      return ElementsAttr(
          OpaqueElementsAttr::get(dialect, type, MangleTensor(input_tensor)));
  }

#undef CONVERT_FLAT
}

// Returns the number of elements present in this TensorProto, or -1 if that
// could not be determined. This might be less than the shape of the proto might
// indicate, if we're storing a splat tensor.
int NumberOfMaterializedElements(const TensorProto& tensor) {
  if (!tensor.tensor_content().empty()) return -1;
    // We don't know which element type this protocol buffer is storing, and the
    // metaprogramming facilities for TensorProto are too limited to check their
    // number without knowing this, so we need to manually dispatch to each
    // possible member of TensorProto, depening on its dtype.
#define MATCH(DTYPE, FIELD) \
  case DTYPE:               \
    return tensor.FIELD##_val().size()

  switch (tensor.dtype()) {
    MATCH(DT_FLOAT, float);
    MATCH(DT_DOUBLE, double);
    MATCH(DT_INT8, int);
    MATCH(DT_UINT8, int);
    MATCH(DT_INT16, int);
    MATCH(DT_UINT16, int);
    MATCH(DT_INT32, int);
    MATCH(DT_UINT32, uint32);
    MATCH(DT_INT64, int64);
    MATCH(DT_UINT64, uint64);
    MATCH(DT_BOOL, bool);
    MATCH(DT_HALF, half);
    MATCH(DT_BFLOAT16, half);
    MATCH(DT_STRING, string);

    // TODO(b/188995810): DenseElementsAttr::get doesn't support complex
    // Attributes being passed, so we bail out for now. This should just be
    //   MATCH(DT_COMPLEX64, scomplex) / 2;
    //   MATCH(DT_COMPLEX128, dcomplex) / 2;
    // when DenseElementsAttr is updated.
    case DT_COMPLEX64:
    case DT_COMPLEX128:
    default:
      return -1;
  }
}

StatusOr<ElementsAttr> ConvertTensorProto(const TensorProto& input_tensor,
                                          Builder* builder) {
  // If there is only one actual element in the proto, but its shape would
  // indicate there are more values, then this is representing a splat tensor.
  // We can create an MLIR Attribute more efficiently in this case.
  TensorShape input_tensor_shape(input_tensor.tensor_shape());
  if (NumberOfMaterializedElements(input_tensor) == 1 &&
      input_tensor_shape.num_elements() > 1) {
    // We first convert this TensorProto to one of shape [1]. We then create an
    // Attribute for that proto, and finally splat the Attribute.

    TensorProto tensor_copy = input_tensor;
    auto* shape = tensor_copy.mutable_tensor_shape();
    shape->clear_dim();
    shape->add_dim()->set_size(1);

    TF_ASSIGN_OR_RETURN(ElementsAttr single_attr,
                        ConvertTensorProto(tensor_copy, builder));

    llvm::SmallVector<int64_t> original_dimensions;
    for (auto dim : input_tensor_shape) original_dimensions.push_back(dim.size);
    return ElementsAttr(mlir::SplatElementsAttr::get(
        single_attr.getType().clone(original_dimensions),
        single_attr.getValues<mlir::Attribute>()[0]));
  }

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

template <typename T>
void ConvertElementsAttr(const mlir::DenseElementsAttr attr,
                         protobuf::RepeatedField<T>* output) {
  if (attr.isSplat()) {
    if (attr.getSplatValue<T>() != T(0)) output->Add(attr.getSplatValue<T>());
  } else {
    output->Reserve(attr.getNumElements());
    for (auto value : attr.getValues<T>()) output->AddAlreadyReserved(value);
  }
}

// Converts an MLIR elements attribute and adds it to specified repeated field.
template <typename T, typename Cord>
void ConvertFloatElementsAttr(const mlir::DenseElementsAttr attr,
                              protobuf::RepeatedField<T>* output,
                              Cord* tensor_content) {
  if (attr.isSplat()) {
    if (attr.getSplatValue<T>() != T(0)) output->Add(attr.getSplatValue<T>());
  } else {
    port::CopyFromArray(tensor_content, attr.getRawData().data(),
                        attr.getRawData().size());
  }
}

// Converts an MLIR elements attribute containing half values and adds it to
// specified repeated field.
void ConvertHalfElementsAttr(const mlir::DenseElementsAttr attr,
                             protobuf::RepeatedField<int>* output) {
  if (attr.isSplat()) {
    if (attr.getSplatValue<Eigen::half>() != Eigen::half(0))
      output->Add(
          Eigen::numext::bit_cast<uint16_t>(attr.getSplatValue<Eigen::half>()));
  } else {
    output->Reserve(attr.getNumElements());
    for (const Eigen::half value : attr.getValues<Eigen::half>())
      output->AddAlreadyReserved(Eigen::numext::bit_cast<uint16_t>(value));
  }
}

// Converts an MLIR elements attribute containing signed int values and adds it
// to specified repeated field.
template <typename T, typename U = T, typename Cord>
void ConvertIntElementsAttr(const mlir::DenseElementsAttr attr,
                            protobuf::RepeatedField<T>* output,
                            Cord* tensor_content) {
  if (attr.isSplat()) {
    if (attr.getSplatValue<U>() != U(0)) output->Add(attr.getSplatValue<U>());
  } else {
    port::CopyFromArray(tensor_content, attr.getRawData().data(),
                        attr.getRawData().size());
  }
}

// Converts an MLIR elements attribute containing unsigned int values and adds
// it to specified repeated field.
template <typename T, typename U = T, typename Cord>
void ConvertUIntElementsAttr(const mlir::DenseElementsAttr attr,
                             protobuf::RepeatedField<T>* output,
                             Cord* tensor_content) {
  if (attr.isSplat()) {
    if (attr.getSplatValue<U>() != U(0)) output->Add(attr.getSplatValue<U>());
  } else {
    port::CopyFromArray(tensor_content, attr.getRawData().data(),
                        attr.getRawData().size());
  }
}

void ConvertBfloat16ElementsAttr(const mlir::DenseElementsAttr attr,
                                 protobuf::RepeatedField<int>* output) {
  if (attr.isSplat()) {
    if (attr.getSplatValue<bfloat16>() != bfloat16(0))
      output->Add(
          Eigen::numext::bit_cast<uint16_t>(attr.getSplatValue<bfloat16>()));
  } else {
    output->Reserve(attr.getNumElements());
    for (const bfloat16 value : attr.getValues<bfloat16>())
      output->AddAlreadyReserved(Eigen::numext::bit_cast<uint16_t>(value));
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
    case DT_BOOL:
      ConvertElementsAttr(dense_attr, output->mutable_bool_val());
      break;
    case DT_BFLOAT16:
      ConvertBfloat16ElementsAttr(dense_attr, output->mutable_half_val());
      break;
    case DT_COMPLEX64:
      ConvertComplexElementsAttr(dense_attr, output->mutable_scomplex_val());
      break;
    case DT_COMPLEX128:
      ConvertComplexElementsAttr(dense_attr, output->mutable_dcomplex_val());
      break;
    case DT_DOUBLE:
      ConvertFloatElementsAttr(dense_attr, output->mutable_double_val(),
                               output->mutable_tensor_content());
      break;
    case DT_HALF:
      ConvertHalfElementsAttr(dense_attr, output->mutable_half_val());
      break;
    case DT_FLOAT:
      ConvertFloatElementsAttr(dense_attr, output->mutable_float_val(),
                               output->mutable_tensor_content());
      break;
    case DT_QUINT8:
    case DT_INT8:
      ConvertUIntElementsAttr<int, int8_t>(dense_attr,
                                           output->mutable_int_val(),
                                           output->mutable_tensor_content());
      break;
    case DT_QUINT16:
    case DT_INT16:
      ConvertIntElementsAttr<int, int16_t>(dense_attr,
                                           output->mutable_int_val(),
                                           output->mutable_tensor_content());
      break;
    case DT_INT32:
      ConvertIntElementsAttr(dense_attr, output->mutable_int_val(),
                             output->mutable_tensor_content());
      break;
    case DT_INT64:
      ConvertIntElementsAttr(dense_attr, output->mutable_int64_val(),
                             output->mutable_tensor_content());
      break;
    case DT_STRING:
      ConvertStringElementsAttr(dense_attr.cast<DenseStringElementsAttr>(),
                                output->mutable_string_val());
      break;
    case DT_UINT8:
      ConvertUIntElementsAttr<int, uint8_t>(dense_attr,
                                            output->mutable_int_val(),
                                            output->mutable_tensor_content());
      break;
    case DT_UINT16:
      ConvertUIntElementsAttr<int, uint16_t>(dense_attr,
                                             output->mutable_int_val(),
                                             output->mutable_tensor_content());
      break;
    case DT_UINT32:
      ConvertUIntElementsAttr(dense_attr, output->mutable_uint32_val(),
                              output->mutable_tensor_content());
      break;
    case DT_UINT64:
      ConvertUIntElementsAttr(dense_attr, output->mutable_uint64_val(),
                              output->mutable_tensor_content());
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
