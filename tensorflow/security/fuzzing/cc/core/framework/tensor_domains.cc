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

#include <limits>
#include <string>

#include "fuzztest/fuzztest.h"
#include "absl/log/log.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/security/fuzzing/cc/core/framework/tensor_shape_domains.h"
#include "tsl/platform/errors.h"

namespace tensorflow::fuzzing {
namespace {

using ::fuzztest::Arbitrary;
using ::fuzztest::Domain;
using ::fuzztest::Filter;
using ::fuzztest::FlatMap;
using ::fuzztest::InRange;
using ::fuzztest::Map;
using ::fuzztest::VectorOf;

template <class T>
Domain<T> DomainRange(double min, double max) {
  // Need to convert limits to T, and check that we are within bounds.
  T min_t = std::numeric_limits<T>::lowest() / 2;
  if (min > static_cast<double>(min_t)) min_t = static_cast<T>(min);
  T max_t = std::numeric_limits<T>::max() / 2;
  if (max < static_cast<double>(max_t)) max_t = static_cast<T>(max);
  return InRange(min_t, max_t);
}

template <>
Domain<bool> DomainRange(double min, double max) {
  return Arbitrary<bool>();
}

template <typename T>
auto StatusOrAnyTensor(const TensorShape& shape, Domain<T> content_domain) {
  return Map(
      [shape](const std::vector<T>& contents) -> absl::StatusOr<Tensor> {
        Tensor tensor;
        TF_RETURN_IF_ERROR(
            Tensor::BuildTensor(DataTypeToEnum<T>::v(), shape, &tensor));
        auto flat_tensor = tensor.flat<T>();
        for (int i = 0; i < contents.size(); ++i) {
          flat_tensor(i) = contents[i];
        }
        return tensor;
      },
      VectorOf(content_domain).WithSize(shape.num_elements()));
}

#define NUMERIC_TENSOR_HELPER(data_type) \
  case data_type:                        \
    return StatusOrAnyTensor(            \
        shape, DomainRange<EnumToDataType<data_type>::Type>(min, max));

Domain<absl::StatusOr<Tensor>> StatusOrAnyNumericTensor(
    const TensorShape& shape, DataType data_type, double min, double max) {
  switch (data_type) {
    NUMERIC_TENSOR_HELPER(DT_FLOAT);
    NUMERIC_TENSOR_HELPER(DT_DOUBLE);
    NUMERIC_TENSOR_HELPER(DT_INT32);
    NUMERIC_TENSOR_HELPER(DT_UINT8);
    NUMERIC_TENSOR_HELPER(DT_INT16);
    NUMERIC_TENSOR_HELPER(DT_INT8);
    NUMERIC_TENSOR_HELPER(DT_INT64);
    NUMERIC_TENSOR_HELPER(DT_UINT16);
    NUMERIC_TENSOR_HELPER(DT_UINT32);
    NUMERIC_TENSOR_HELPER(DT_UINT64);
    NUMERIC_TENSOR_HELPER(DT_BOOL);
    // TODO(b/268338352): Add unsupported types
    // DT_BOOL, DT_STRING, DT_COMPLEX64, DT_QINT8, DT_QUINT8, DT_QINT32,
    // DT_BFLOAT16, DT_QINT16, DT_COMPLEX128, DT_HALF, DT_RESOURCE, DT_VARIANT,
    // DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN
    default:
      LOG(FATAL) << "Unsupported data type: " << data_type;  // Crash OK
  }
}

Domain<Tensor> FilterInvalid(Domain<absl::StatusOr<Tensor>> domain) {
  return Map([](const absl::StatusOr<Tensor>& t) { return *t; },
             Filter(
                 [](const absl::StatusOr<Tensor>& inner_t) {
                   return inner_t.status().ok();
                 },
                 domain));
}

}  // namespace

Domain<Tensor> AnyValidNumericTensor(const TensorShape& shape,
                                     DataType datatype, double min,
                                     double max) {
  return FilterInvalid(StatusOrAnyNumericTensor(shape, datatype, min, max));
}

Domain<Tensor> AnyValidNumericTensor(Domain<TensorShape> tensor_shape_domain,
                                     Domain<DataType> datatype_domain,
                                     double min, double max) {
  return FlatMap(
      [min, max](const TensorShape& shape, DataType datatype) {
        return AnyValidNumericTensor(shape, datatype, min, max);
      },
      tensor_shape_domain, datatype_domain);
}

Domain<Tensor> AnySmallValidNumericTensor(DataType datatype) {
  return fuzzing::AnyValidNumericTensor(fuzzing::AnyValidTensorShape(
                                            /*max_rank=*/5,
                                            /*dim_lower_bound=*/0,
                                            /*dim_upper_bound=*/10),
                                        fuzztest::Just(datatype),
                                        /*min=*/-10,
                                        /*max=*/10);
}

Domain<Tensor> AnyValidStringTensor(const TensorShape& tensor_shape,
                                    Domain<std::string> string_domain) {
  return FilterInvalid(StatusOrAnyTensor<tstring>(
      tensor_shape,
      Map([](const std::string& s) { return static_cast<tstring>(s); },
          string_domain)));
}

}  // namespace tensorflow::fuzzing
