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

#include "tensorflow/compiler/tf2xla/type_util.h"

#include <array>

#include "absl/status/statusor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

// Conversion utilities should support any primitive type,
//  excluding string, resource, variant, invalid.
bool DataTypeSupportsXlaConversion(DataType dt) {
  switch (dt) {
    case DataType::DT_STRING:
    case DataType::DT_RESOURCE:
    case DataType::DT_VARIANT:
    case DataType::DT_INVALID:
      return false;
    default:
      // All other types should be supported.
      break;
  }
  return !IsRefType(dt);
}

TEST(DataTypeToPrimitiveTypeTest, AllDataTypesSupported) {
  for (int i = tensorflow::DataType_MIN; i < tensorflow::DataType_MAX; ++i) {
    if (tensorflow::DataType_IsValid(i)) {
      DataType dt = static_cast<DataType>(i);
      if (DataTypeSupportsXlaConversion(dt)) {
        xla::PrimitiveType out_type;
        EXPECT_TRUE(DataTypeToPrimitiveType(dt, &out_type).ok());
      }
    }
  }
}

TEST(EncodePrimitiveTypeAsDataType, AllPrimitiveTypesSupported) {
  for (int i = tensorflow::DataType_MIN; i < tensorflow::DataType_MAX; ++i) {
    DataType dt = static_cast<DataType>(i);
    xla::PrimitiveType xla_type;
    // If conversion to primitive type works, then the reverse mapping should
    // also work.
    if (DataTypeToPrimitiveType(dt, &xla_type).ok()) {
      absl::StatusOr<DataType> data_type_or =
          EncodePrimitiveTypeAsDataType(xla_type);
      EXPECT_TRUE(data_type_or.ok());
      // Non-quantized inputs should map directly back to the original type.
      if (!DataTypeIsQuantized(dt)) {
        EXPECT_EQ(*data_type_or, dt);
      }
    }
  }
}

TEST(EncodePrimitiveTypeAsDataType, QuantizedTypesMapToUnquantized) {
  static std::array<DataType, 5> quantized_inputs = {
      DT_QINT8, DT_QINT16, DT_QINT32, DT_QUINT8, DT_QUINT16};
  static std::array<DataType, 5> expected_outputs = {
      DT_INT8, DT_INT16, DT_INT32, DT_UINT8, DT_UINT16};

  for (int i = 0; i < quantized_inputs.size(); ++i) {
    xla::PrimitiveType xla_type;
    EXPECT_TRUE(DataTypeToPrimitiveType(quantized_inputs[i], &xla_type).ok());
    absl::StatusOr<DataType> data_type_or =
        EncodePrimitiveTypeAsDataType(xla_type);
    EXPECT_TRUE(data_type_or.ok());
    EXPECT_EQ(*data_type_or, expected_outputs[i]);
  }
}

}  // namespace
}  // namespace tensorflow
