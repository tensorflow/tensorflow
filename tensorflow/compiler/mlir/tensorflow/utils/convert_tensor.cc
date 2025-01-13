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

#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/AsmState.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/DialectResourceBlobManager.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_attributes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dynamic_shape_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/mangling_util.h"
#include "xla/tsl/platform/errors.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/bfloat16.h"
#include "tensorflow/core/platform/tstring.h"
#include "tsl/platform/ml_dtypes.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {

using llvm::ArrayRef;
using llvm::SmallVector;
using mlir::Builder;
using mlir::DenseStringElementsAttr;
using mlir::ElementsAttr;
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

template <typename ElementType>
static absl::Status CopyDataIntoBlob(mlir::AsmResourceBlob& blob,
                                     absl::string_view raw_src_data) {
  ArrayRef<ElementType> data = blob.getDataAs<ElementType>();
  llvm::MutableArrayRef<ElementType> raw_dest_data =
      mlir::MutableArrayRef<ElementType>(const_cast<ElementType*>(data.data()),
                                         data.size());
  if (raw_src_data.size() != blob.getData().size()) {
    return absl::InvalidArgumentError(
        "Size mismatch between raw_src_data and blob data");
  }
  // Memcpy.
  std::memcpy(raw_dest_data.data(), raw_src_data.data(), raw_src_data.size());

  return absl::OkStatus();
}

// Converts a TensorFlow tensor into an MLIR elements attribute.
template <typename ElementType>
absl::StatusOr<ElementsAttr> ConvertFlatTensor(const Tensor& input_tensor,
                                               ShapedType shaped_type,
                                               bool convert_to_dense_resource) {
  // Only convert to dense resource if the data type is integer or floating.
  if (convert_to_dense_resource && DataTypeCanUseMemcpy(input_tensor.dtype()) &&
      (DataTypeIsInteger(input_tensor.dtype()) ||
       DataTypeIsFloating(input_tensor.dtype()))) {
    auto element_type = shaped_type.getElementType();
    auto num_elements = shaped_type.getNumElements();
    auto bit_width = element_type.getIntOrFloatBitWidth();
    auto tensor_data = input_tensor.tensor_data();
    mlir::AsmResourceBlob blob;

    if (llvm::isa<mlir::IntegerType>(element_type)) {
      switch (bit_width) {
        case 1:
          blob = mlir::HeapAsmResourceBlob::allocate(num_elements,
                                                     /*align=*/64,
                                                     /*dataIsMutable=*/true);
          TF_RETURN_IF_ERROR(CopyDataIntoBlob<uint8_t>(blob, tensor_data));
          return mlir::DenseResourceElementsAttr::get(
              shaped_type, "dense_elements_i1", std::move(blob));
        case 8:
          blob = mlir::HeapAsmResourceBlob::allocate(num_elements,
                                                     /*align=*/64,
                                                     /*dataIsMutable=*/true);
          TF_RETURN_IF_ERROR(CopyDataIntoBlob<ElementType>(blob, tensor_data));
          return mlir::DenseResourceElementsAttr::get(
              shaped_type, "dense_elements_i8", std::move(blob));
        case 16:
          blob = mlir::HeapAsmResourceBlob::allocate(2 * num_elements,
                                                     /*align=*/64,
                                                     /*dataIsMutable=*/true);
          TF_RETURN_IF_ERROR(CopyDataIntoBlob<ElementType>(blob, tensor_data));
          return mlir::DenseResourceElementsAttr::get(
              shaped_type, "dense_elements_i16", std::move(blob));
        case 32:
          blob = mlir::HeapAsmResourceBlob::allocate(4 * num_elements,
                                                     /*align=*/64,
                                                     /*dataIsMutable=*/true);
          TF_RETURN_IF_ERROR(CopyDataIntoBlob<ElementType>(blob, tensor_data));
          return mlir::DenseResourceElementsAttr::get(
              shaped_type, "dense_elements_i32", std::move(blob));
        case 64:
          blob = mlir::HeapAsmResourceBlob::allocate(8 * num_elements,
                                                     /*align=*/64,
                                                     /*dataIsMutable=*/true);
          TF_RETURN_IF_ERROR(CopyDataIntoBlob<ElementType>(blob, tensor_data));
          return mlir::DenseResourceElementsAttr::get(
              shaped_type, "dense_elements_i64", std::move(blob));
        default:
          return absl::InvalidArgumentError("Unsupported bit width");
      }
    } else if (llvm::isa<mlir::FloatType>(element_type)) {
      mlir::AsmResourceBlob blob;
      switch (bit_width) {
        case 8:
          blob = mlir::HeapAsmResourceBlob::allocate(num_elements, /*align=*/64,
                                                     /*dataIsMutable=*/true);
          TF_RETURN_IF_ERROR(CopyDataIntoBlob<uint8_t>(blob, tensor_data));
          return mlir::DenseResourceElementsAttr::get(
              shaped_type, "dense_elements_f8", std::move(blob));
        case 16:
          blob = mlir::HeapAsmResourceBlob::allocate(2 * num_elements,
                                                     /*align=*/64,
                                                     /*dataIsMutable=*/true);
          TF_RETURN_IF_ERROR(CopyDataIntoBlob<uint16_t>(blob, tensor_data));
          return mlir::DenseResourceElementsAttr::get(
              shaped_type, "dense_elements_f16", std::move(blob));
        case 32: {
          blob = mlir::HeapAsmResourceBlob::allocate(4 * num_elements,
                                                     /*align=*/64,
                                                     /*dataIsMutable=*/true);
          TF_RETURN_IF_ERROR(CopyDataIntoBlob<float>(blob, tensor_data));
          return mlir::DenseResourceElementsAttr::get(
              shaped_type, "dense_elements_f32", std::move(blob));
        }
        case 64:
          blob = mlir::HeapAsmResourceBlob::allocate(8 * num_elements,
                                                     /*align=*/64,
                                                     /*dataIsMutable=*/true);
          TF_RETURN_IF_ERROR(CopyDataIntoBlob<uint64_t>(blob, tensor_data));
          return mlir::DenseResourceElementsAttr::get(
              shaped_type, "dense_elements_f64", std::move(blob));
        default:
          return absl::InvalidArgumentError("Unsupported bit width");
      }
    } else {
      return absl::InvalidArgumentError("Unsupported element type");
    }
  } else {
    auto tensor_data = llvm::ArrayRef(input_tensor.flat<ElementType>().data(),
                                      input_tensor.flat<ElementType>().size());
    return ElementsAttr(mlir::DenseElementsAttr::get(shaped_type, tensor_data));
  }
}

ElementsAttr ConvertTensorOfCustomFloatType(const Tensor& tensor,
                                            RankedTensorType type) {
  auto buffer =
      llvm::ArrayRef(static_cast<char*>(tensor.data()), tensor.TotalBytes());
  return mlir::DenseElementsAttr::getFromRawBuffer(type, buffer);
}

absl::StatusOr<ElementsAttr> ConvertStringTensor(const Tensor& input_tensor,
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

absl::StatusOr<ElementsAttr> ConvertTensor(const Tensor& input_tensor,
                                           Builder* builder,
                                           bool convert_to_dense_resource) {
  const auto& input_dtype = input_tensor.dtype();
  const auto& input_shape = input_tensor.shape();
  Type elt_type;
  TF_RETURN_IF_ERROR(ConvertDataType(input_dtype, *builder, &elt_type));
  SmallVector<int64_t, 4> shape;
  ConvertToMlirShape(input_shape, &shape);
  auto type = RankedTensorType::get(shape, elt_type);

#define CONVERT_FLAT(DTYPE, CTYPE)                      \
  case DTYPE:                                           \
    return ConvertFlatTensor<CTYPE>(input_tensor, type, \
                                    convert_to_dense_resource);

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

    case DT_BFLOAT16:
    case DT_HALF:
    case DT_FLOAT8_E5M2:
    case DT_FLOAT8_E4M3FN:
    case DT_FLOAT8_E4M3FNUZ:
    case DT_FLOAT8_E4M3B11FNUZ:
    case DT_FLOAT8_E5M2FNUZ:
      return ConvertTensorOfCustomFloatType(input_tensor, type);
    case DT_STRING:
      return ConvertStringTensor(input_tensor, type);
    default:
      // TODO(hinsu): Remove mangling now that there is a special attribute.
      return ElementsAttr(
          mlir::TF::TensorProtoAttr::get(type, MangleTensor(input_tensor)));
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

absl::StatusOr<ElementsAttr> ConvertTensorProto(
    const TensorProto& input_tensor, Builder* builder,
    bool convert_to_dense_resource) {
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
        single_attr.getShapedType().clone(original_dimensions),
        single_attr.getValues<mlir::Attribute>()[0]));
  }

  Tensor t;
  if (!t.FromProto(input_tensor))
    return InvalidArgument("Failed to parse input_tensor.");
  return ConvertTensor(t, builder, convert_to_dense_resource);
}

void ConvertToTensorShapeProto(ArrayRef<int64_t> shape,
                               TensorShapeProto* output_shape) {
  for (auto d : shape) {
    output_shape->add_dim()->set_size(ShapedType::isDynamic(d) ? kTFDynamicSize
                                                               : d);
  }
}

PartialTensorShape ConvertTypeToTensorShape(const mlir::Type& type) {
  if (mlir::isa<mlir::UnrankedTensorType>(type)) {
    // An empty PartialTensorShape indicates an unranked tensor.
    return PartialTensorShape();
  }

  if (auto tensor_type = mlir::dyn_cast<mlir::RankedTensorType>(type)) {
    TensorShapeProto tensor_shape_proto;
    ConvertToTensorShapeProto(tensor_type.getShape(), &tensor_shape_proto);
    return PartialTensorShape(tensor_shape_proto);
  }

  // If type is not a RankedTensor or UnrankedTensor, it must be a scalar.
  // Empty TensorShape indicates a scalar.
  return TensorShape();
}

mlir::TF::ShapeAttr ConvertTypeToTensorShapeAttr(const mlir::Type& type) {
  if (mlir::isa<mlir::UnrankedTensorType>(type)) {
    return mlir::TF::ShapeAttr::get(type.getContext(), std::nullopt);
  }

  if (auto tensor_type = mlir::dyn_cast<mlir::RankedTensorType>(type)) {
    return mlir::TF::ShapeAttr::get(type.getContext(), tensor_type.getShape());
  }

  // If type is not a RankedTensor or UnrankedTensor, it must be a scalar.
  // Empty TensorShape indicates a scalar.
  return mlir::TF::ShapeAttr::get(type.getContext(), ArrayRef<int64_t>());
}

absl::StatusOr<TensorSpecProto> ConvertTypeToTensorSpecProto(
    const mlir::Type& type) {
  DataType dtype;
  TF_RETURN_IF_ERROR(ConvertToDataType(type, &dtype));
  TensorSpecProto tensor_spec;
  tensor_spec.set_dtype(dtype);
  *tensor_spec.mutable_shape() = ConvertTypeToTensorShape(type).AsProto();
  return tensor_spec;
}

// Converts the tensor shape proto into an MLIR shape attribute.
absl::StatusOr<mlir::Attribute> ConvertTensorShapeProto(
    const TensorShapeProto& shape, mlir::MLIRContext* context) {
  if (shape.unknown_rank())
    return mlir::TF::ShapeAttr::get(context, std::nullopt);

  llvm::SmallVector<int64_t, 4> dims;
  dims.reserve(shape.dim().size());
  for (const auto& dim : shape.dim()) {
    dims.push_back(dim.size() == kTFDynamicSize ? ShapedType::kDynamic
                                                : dim.size());
  }
  return mlir::TF::ShapeAttr::get(context, llvm::ArrayRef(dims));
}

// Converts an MLIR dense string elements attribute to a TensorFlow tensor
// proto.
absl::Status ConvertStringElementsAttr(
    const DenseStringElementsAttr attr,
    protobuf::RepeatedPtrField<std::string>* output) {
  for (const auto& val : attr.getRawStringData())
    output->Add({val.data(), val.size()});
  return absl::OkStatus();
}

template <typename T>
absl::Status ConvertComplexElementsAttr(const mlir::ElementsAttr elem_attr,
                                        protobuf::RepeatedField<T>* output) {
  auto attr = llvm::dyn_cast<mlir::DenseElementsAttr>(elem_attr);
  if (!attr)
    return absl::InvalidArgumentError("Unsupported elements attr found");

  auto elementType = attr.getType().getElementType();
  if (!llvm::isa<mlir::ComplexType>(elementType)) {
    return absl::InvalidArgumentError("Complex elements attr not found");
  }

  auto complex_elem_ty =
      llvm::cast<mlir::ComplexType>(elementType).getElementType();
  if (complex_elem_ty.isF32()) {
    for (const auto& val : attr.getValues<std::complex<mlir::APFloat>>()) {
      output->Add(val.real().convertToFloat());
      output->Add(val.imag().convertToFloat());
    }
  } else if (complex_elem_ty.isF64()) {
    for (const auto& val : attr.getValues<std::complex<mlir::APFloat>>()) {
      output->Add(val.real().convertToDouble());
      output->Add(val.imag().convertToDouble());
    }
  } else {
    return absl::InvalidArgumentError("Unsupported complex element type");
  }
  return absl::OkStatus();
}

// Converts an Tensor proto attribute to a TensorFlow tensor proto.
absl::Status ConvertTensorProtoAttr(const mlir::TF::TensorProtoAttr attr,
                                    TensorProto* output_tensor) {
  auto mangled_tensor = attr.getValue();
  absl::string_view tensor_view(mangled_tensor.data(), mangled_tensor.size());
  return mangling_util::DemangleTensor(tensor_view, output_tensor);
}

template <typename T>
absl::Status ConvertElementsAttr(const mlir::ElementsAttr elem_attr,
                                 protobuf::RepeatedField<T>* output) {
  auto attr = llvm::dyn_cast<mlir::DenseElementsAttr>(elem_attr);
  if (!attr)
    return absl::InvalidArgumentError("Unsupported elements attr found");
  if (attr.isSplat()) {
    if (attr.getSplatValue<T>() != T(0)) output->Add(attr.getSplatValue<T>());
  } else {
    output->Reserve(attr.getNumElements());
    for (auto value : attr.getValues<T>()) output->AddAlreadyReserved(value);
  }
  return absl::OkStatus();
}

// Converts an MLIR elements attribute and adds it to specified repeated field.
template <typename T, typename Cord>
absl::Status ConvertFloatElementsAttr(const mlir::ElementsAttr elem_attr,
                                      protobuf::RepeatedField<T>* output,
                                      Cord* tensor_content) {
  if (auto attr = llvm::dyn_cast<mlir::DenseElementsAttr>(elem_attr)) {
    if (attr.isSplat()) {
      if (attr.getSplatValue<T>() != T(0)) output->Add(attr.getSplatValue<T>());
    } else {
      port::CopyFromArray(tensor_content, attr.getRawData().data(),
                          attr.getRawData().size());
    }
  } else if (auto dense_resource_ttr =
                 llvm::dyn_cast<mlir::DenseResourceElementsAttr>(elem_attr)) {
    mlir::AsmResourceBlob* blob = dense_resource_ttr.getRawHandle().getBlob();
    if (blob) {
      size_t dst_block_length = blob->getData().size();
      const char* raw_dst_block = blob->getData().data();
      if constexpr (std::is_same_v<Cord, std::string>) {
        *tensor_content = absl::string_view(raw_dst_block, dst_block_length);
      } else {
        *tensor_content = absl::MakeCordFromExternal(
            absl::string_view(raw_dst_block, dst_block_length),
            [](absl::string_view data) {});
      }
    } else {
      return absl::InvalidArgumentError("No blob found in dense resource");
    }
  } else {
    return absl::InvalidArgumentError("Unsupported elements attr found");
  }
  return absl::OkStatus();
}

// Converts an MLIR elements attribute containing half values and adds it to
// specified repeated field.
absl::Status ConvertHalfElementsAttr(const mlir::ElementsAttr elem_attr,
                                     protobuf::RepeatedField<int>* output) {
  auto attr = llvm::dyn_cast<mlir::DenseElementsAttr>(elem_attr);
  if (!attr)
    return absl::InvalidArgumentError(
        "DenseResourceElementsAttr of type half found");
  if (attr.isSplat()) {
    if (attr.getSplatValue<Eigen::half>() != Eigen::half(0))
      output->Add(
          Eigen::numext::bit_cast<uint16_t>(attr.getSplatValue<Eigen::half>()));
  } else {
    output->Reserve(attr.getNumElements());
    for (const Eigen::half value : attr.getValues<Eigen::half>())
      output->AddAlreadyReserved(Eigen::numext::bit_cast<uint16_t>(value));
  }
  return absl::OkStatus();
}

// Converts an MLIR elements attribute containing signed int values and adds it
// to specified repeated field.
template <typename T, typename U = T, typename Cord>
absl::Status ConvertIntElementsAttr(const mlir::ElementsAttr elem_attr,
                                    protobuf::RepeatedField<T>* output,
                                    Cord* tensor_content) {
  if (auto attr = llvm::dyn_cast<mlir::DenseElementsAttr>(elem_attr)) {
    if (attr.isSplat()) {
      if (attr.getSplatValue<U>() != U(0))
        output->Add(static_cast<T>(attr.getSplatValue<U>()));
    } else {
      port::CopyFromArray(tensor_content, attr.getRawData().data(),
                          attr.getRawData().size());
    }
  } else if (auto dense_resource_ttr =
                 llvm::dyn_cast<mlir::DenseResourceElementsAttr>(elem_attr)) {
    mlir::AsmResourceBlob* blob = dense_resource_ttr.getRawHandle().getBlob();
    if (blob) {
      size_t dst_block_length = blob->getData().size();
      const char* raw_dst_block = blob->getData().data();
      if constexpr (std::is_same_v<Cord, std::string>) {
        *tensor_content = absl::string_view(raw_dst_block, dst_block_length);
      } else {
        *tensor_content = absl::MakeCordFromExternal(
            absl::string_view(raw_dst_block, dst_block_length),
            [](absl::string_view data) {});
      }
    } else {
      return absl::InvalidArgumentError("No blob found in dense resource");
    }
  } else {
    return absl::InvalidArgumentError("Unsupported elements attr found");
  }
  return absl::OkStatus();
}

// Converts an MLIR elements attribute containing unsigned int values and adds
// it to specified repeated field.
template <typename T, typename U = T, typename Cord>
absl::Status ConvertUIntElementsAttr(const mlir::ElementsAttr elem_attr,
                                     protobuf::RepeatedField<T>* output,
                                     Cord* tensor_content) {
  if (auto attr = llvm::dyn_cast<mlir::DenseElementsAttr>(elem_attr)) {
    if (attr.isSplat()) {
      if (attr.getSplatValue<U>() != U(0))
        output->Add(static_cast<T>(attr.getSplatValue<U>()));
    } else {
      port::CopyFromArray(tensor_content, attr.getRawData().data(),
                          attr.getRawData().size());
    }
  } else if (auto dense_resource_ttr =
                 llvm::dyn_cast<mlir::DenseResourceElementsAttr>(elem_attr)) {
    mlir::AsmResourceBlob* blob = dense_resource_ttr.getRawHandle().getBlob();
    if (blob) {
      size_t dst_block_length = blob->getData().size();
      const char* raw_dst_block = blob->getData().data();
      if constexpr (std::is_same_v<Cord, std::string>) {
        *tensor_content = absl::string_view(raw_dst_block, dst_block_length);
      } else {
        *tensor_content = absl::MakeCordFromExternal(
            absl::string_view(raw_dst_block, dst_block_length),
            [](absl::string_view data) {});
      }
    } else {
      return absl::InvalidArgumentError("No blob found in dense resource");
    }
  } else {
    return absl::InvalidArgumentError("Unsupported elements attr found");
  }
  return absl::OkStatus();
}

absl::Status ConvertBfloat16ElementsAttr(const mlir::ElementsAttr elem_attr,
                                         protobuf::RepeatedField<int>* output) {
  auto attr = llvm::dyn_cast<mlir::DenseElementsAttr>(elem_attr);
  if (!attr)
    return absl::InvalidArgumentError("Unsupported elements attr found");
  if (attr.isSplat()) {
    if (attr.getSplatValue<bfloat16>() != bfloat16(0))
      output->Add(
          Eigen::numext::bit_cast<uint16_t>(attr.getSplatValue<bfloat16>()));
  } else {
    output->Reserve(attr.getNumElements());
    for (const bfloat16 value : attr.getValues<bfloat16>())
      output->AddAlreadyReserved(Eigen::numext::bit_cast<uint16_t>(value));
  }
  return absl::OkStatus();
}

template <typename T>
absl::Status ConvertFloat8ElementsAttr(const mlir::ElementsAttr elem_attr,
                                       std::string* output) {
  auto attr = llvm::dyn_cast<mlir::DenseElementsAttr>(elem_attr);
  if (!attr)
    return absl::InvalidArgumentError("Unsupported elements attr found");
  if (attr.isSplat()) {
    if (attr.getSplatValue<T>() != T(0))
      output->push_back(
          Eigen::numext::bit_cast<uint8_t>(attr.getSplatValue<T>()));
  } else {
    output->reserve(attr.getNumElements());
    for (const T value : attr.getValues<T>())
      output->push_back(Eigen::numext::bit_cast<uint8_t>(value));
  }
  return absl::OkStatus();
}

absl::Status ConvertToTensorProto(const ElementsAttr attr,
                                  TensorProto* output) {
  auto type = attr.getShapedType();
  auto shape = type.getShape();
  DataType output_dtype;
  TF_RETURN_IF_ERROR(ConvertToDataType(type, &output_dtype));
  output->set_dtype(output_dtype);
  ConvertToTensorShapeProto(shape, output->mutable_tensor_shape());

  if (auto tensor_attr = mlir::dyn_cast<mlir::TF::TensorProtoAttr>(attr))
    return ConvertTensorProtoAttr(tensor_attr, output);

  switch (output_dtype) {
    case DT_BOOL:
      TF_RETURN_IF_ERROR(ConvertElementsAttr(attr, output->mutable_bool_val()));
      break;
    case DT_BFLOAT16:
      TF_RETURN_IF_ERROR(
          ConvertBfloat16ElementsAttr(attr, output->mutable_half_val()));
      break;
    case DT_COMPLEX64:
      TF_RETURN_IF_ERROR(
          ConvertComplexElementsAttr(attr, output->mutable_scomplex_val()));
      break;
    case DT_COMPLEX128:
      TF_RETURN_IF_ERROR(
          ConvertComplexElementsAttr(attr, output->mutable_dcomplex_val()));
      break;
    case DT_DOUBLE:
      TF_RETURN_IF_ERROR(
          ConvertFloatElementsAttr(attr, output->mutable_double_val(),
                                   output->mutable_tensor_content()));
      break;
    case DT_HALF:
      TF_RETURN_IF_ERROR(
          ConvertHalfElementsAttr(attr, output->mutable_half_val()));
      break;
    case DT_FLOAT:
      TF_RETURN_IF_ERROR(ConvertFloatElementsAttr(
          attr, output->mutable_float_val(), output->mutable_tensor_content()));
      break;
    case DT_FLOAT8_E5M2:
      TF_RETURN_IF_ERROR(ConvertFloat8ElementsAttr<tsl::float8_e5m2>(
          attr, output->mutable_float8_val()));
      break;
    case DT_FLOAT8_E4M3FN:
      TF_RETURN_IF_ERROR(ConvertFloat8ElementsAttr<tsl::float8_e4m3fn>(
          attr, output->mutable_float8_val()));
      break;
    case DT_FLOAT8_E4M3FNUZ:
      TF_RETURN_IF_ERROR(ConvertFloat8ElementsAttr<tsl::float8_e4m3fnuz>(
          attr, output->mutable_float8_val()));
      break;
    case DT_FLOAT8_E4M3B11FNUZ:
      TF_RETURN_IF_ERROR(ConvertFloat8ElementsAttr<tsl::float8_e4m3b11fnuz>(
          attr, output->mutable_float8_val()));
      break;
    case DT_FLOAT8_E5M2FNUZ:
      TF_RETURN_IF_ERROR(ConvertFloat8ElementsAttr<tsl::float8_e5m2fnuz>(
          attr, output->mutable_float8_val()));
      break;
    case tensorflow::DT_INT4:
      TF_RETURN_IF_ERROR(ConvertIntElementsAttr<int, tsl::int4>(
          attr, output->mutable_int_val(), output->mutable_tensor_content()));
      break;
    case tensorflow::DT_UINT4:
      TF_RETURN_IF_ERROR(ConvertUIntElementsAttr<int, tsl::uint4>(
          attr, output->mutable_int_val(), output->mutable_tensor_content()));
      break;
    case DT_QUINT8:
    case DT_INT8:
      TF_RETURN_IF_ERROR(ConvertIntElementsAttr<int, int8_t>(
          attr, output->mutable_int_val(), output->mutable_tensor_content()));
      break;
    case DT_QUINT16:
    case DT_INT16:
      TF_RETURN_IF_ERROR(ConvertIntElementsAttr<int, int16_t>(
          attr, output->mutable_int_val(), output->mutable_tensor_content()));
      break;
    case DT_INT32:
      TF_RETURN_IF_ERROR(ConvertIntElementsAttr(
          attr, output->mutable_int_val(), output->mutable_tensor_content()));
      break;
    case DT_INT64:
      TF_RETURN_IF_ERROR(ConvertIntElementsAttr(
          attr, output->mutable_int64_val(), output->mutable_tensor_content()));
      break;
    case DT_STRING:
      TF_RETURN_IF_ERROR(
          ConvertStringElementsAttr(mlir::cast<DenseStringElementsAttr>(attr),
                                    output->mutable_string_val()));
      break;
    case DT_UINT8:
      TF_RETURN_IF_ERROR(ConvertUIntElementsAttr<int, uint8_t>(
          attr, output->mutable_int_val(), output->mutable_tensor_content()));
      break;
    case DT_UINT16:
      TF_RETURN_IF_ERROR(ConvertUIntElementsAttr<int, uint16_t>(
          attr, output->mutable_int_val(), output->mutable_tensor_content()));
      break;
    case DT_UINT32:
      TF_RETURN_IF_ERROR(
          ConvertUIntElementsAttr(attr, output->mutable_uint32_val(),
                                  output->mutable_tensor_content()));
      break;
    case DT_UINT64:
      TF_RETURN_IF_ERROR(
          ConvertUIntElementsAttr(attr, output->mutable_uint64_val(),
                                  output->mutable_tensor_content()));
      break;
    default:
      return absl::UnimplementedError(absl::StrCat(
          "Unimplemented data type ", DataTypeString(output_dtype)));
  }
  return absl::OkStatus();
}

absl::Status ConvertToTensor(const mlir::ElementsAttr attr,
                             Tensor* output_tensor) {
  TensorProto tensor_proto;
  TF_RETURN_IF_ERROR(ConvertToTensorProto(attr, &tensor_proto));
  if (!output_tensor->FromProto(tensor_proto)) {
    return InvalidArgument("Couldn't convert tensor proto to tensor.");
  }
  return absl::OkStatus();
}

}  // namespace tensorflow
