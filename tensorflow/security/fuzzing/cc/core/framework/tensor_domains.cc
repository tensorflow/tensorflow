/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/security/fuzzing/cc/core/framework/tensor_domains.h"

#include "fuzztest/fuzztest.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/security/fuzzing/cc/core/framework/datatype_domains.h"
#include "tensorflow/security/fuzzing/cc/core/framework/tensor_shape_domains.h"

namespace tensorflow::fuzzing {
namespace {

using ::fuzztest::Arbitrary;
using ::fuzztest::Domain;
using ::fuzztest::Filter;
using ::fuzztest::FlatMap;
using ::fuzztest::Map;
using ::fuzztest::VectorOf;

template <class T>
auto AnyStatusOrTensor(const TensorShape& shape) {
  return Map(
      [shape](const std::vector<T>& contents) -> StatusOr<Tensor> {
        Tensor tensor;
        TF_RETURN_IF_ERROR(
            Tensor::BuildTensor(DataTypeToEnum<T>::v(), shape, &tensor));
        auto flat_tensor = tensor.flat<T>();
        for (int i = 0; i < contents.size(); ++i) {
          flat_tensor(i) = contents[i];
        }
        return tensor;
      },
      VectorOf(Arbitrary<T>()).WithSize(shape.num_elements()));
}

Domain<StatusOr<Tensor>> AnyStatusOrTensorOfShapeAndType(
    const TensorShape& shape, DataType data_type) {
  switch (data_type) {
    case DT_FLOAT:
      return AnyStatusOrTensor<EnumToDataType<DT_FLOAT>::Type>(shape);
    case DT_DOUBLE:
      return AnyStatusOrTensor<EnumToDataType<DT_DOUBLE>::Type>(shape);
    case DT_INT32:
      return AnyStatusOrTensor<EnumToDataType<DT_INT32>::Type>(shape);
    case DT_UINT8:
      return AnyStatusOrTensor<EnumToDataType<DT_UINT8>::Type>(shape);
    case DT_INT16:
      return AnyStatusOrTensor<EnumToDataType<DT_INT16>::Type>(shape);
    case DT_INT8:
      return AnyStatusOrTensor<EnumToDataType<DT_INT8>::Type>(shape);
    case DT_INT64:
      return AnyStatusOrTensor<EnumToDataType<DT_INT64>::Type>(shape);
    case DT_BOOL:
      return AnyStatusOrTensor<EnumToDataType<DT_BOOL>::Type>(shape);
    case DT_UINT16:
      return AnyStatusOrTensor<EnumToDataType<DT_UINT16>::Type>(shape);
    case DT_UINT32:
      return AnyStatusOrTensor<EnumToDataType<DT_UINT32>::Type>(shape);
    case DT_UINT64:
      return AnyStatusOrTensor<EnumToDataType<DT_UINT64>::Type>(shape);
    // TODO(b/268338352): Add unsupported types
    // DT_STRING, DT_COMPLEX64, DT_QINT8, DT_QUINT8, DT_QINT32, DT_BFLOAT16,
    // DT_QINT16, DT_COMPLEX128, DT_HALF, DT_RESOURCE, DT_VARIANT,
    // DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN
    default:
      LOG(FATAL) << "Unsupported data type: " << data_type;  // Crash OK
  }
}

}  // namespace

Domain<Tensor> AnyValidTensorOfShapeAndType(const TensorShape& shape,
                                            DataType datatype) {
  return Map(
      [](const StatusOr<Tensor>& t) { return *t; },
      Filter(
          [](const StatusOr<Tensor>& inner_t) { return inner_t.status().ok(); },
          AnyStatusOrTensorOfShapeAndType(shape, datatype)));
}

Domain<Tensor> AnyValidTensor(Domain<TensorShape> tensor_shape_domain,
                              Domain<DataType> datatype_domain) {
  return FlatMap(AnyValidTensorOfShapeAndType, tensor_shape_domain,
                 datatype_domain);
}

}  // namespace tensorflow::fuzzing
