// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/tensor_pool.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "third_party/qairt/latest/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

namespace {

// TODO(Alen): The current test coverage is not exhaustive.
// Some corner cases may not be tested. Narrowed types may lead to unexpected
// behavior.

TEST(TensorPoolConvertStaticTensorTest, ConvertNonStaticTensor) {
  TensorPool tensor_pool;

  auto& tensor_wrapper = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_FLOAT_32, QuantizeParamsWrapperVariant{}, {1, 2, 3});

  auto* res = tensor_pool.ConvertStaticTensorFrom<float>(tensor_wrapper);
  ASSERT_EQ(res, nullptr);
}

TEST(TensorPoolConvertStaticTensorTest, ExceedRangeAndFailToConvert) {
  TensorPool tensor_pool;

  std::vector<std::int32_t> tensor_data{
      std::numeric_limits<std::int32_t>::min(),
      std::numeric_limits<std::int32_t>::max()};
  auto& tensor_wrapper = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, QuantizeParamsWrapperVariant{}, {2},
      sizeof(decltype(tensor_data)::value_type) * tensor_data.size(),
      tensor_data.data());

  auto* res = tensor_pool.ConvertStaticTensorFrom<std::int16_t>(tensor_wrapper);
  ASSERT_EQ(res, nullptr);
}

TEST(TensorPoolConvertStaticTensorTest, SameTypeConversionFloat32) {
  TensorPool tensor_pool;

  std::vector<float> tensor_data{0, 1, 2, 3, 4, 5};
  auto& tensor_wrapper = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_FLOAT_32, QuantizeParamsWrapperVariant{}, {1, 2, 3},
      sizeof(decltype(tensor_data)::value_type) * tensor_data.size(),
      tensor_data.data());

  auto* res = tensor_pool.ConvertStaticTensorFrom<float>(tensor_wrapper);
  ASSERT_NE(res, nullptr);

  auto converted_data = res->GetStaticTensorData<float>();
  ASSERT_TRUE(converted_data.has_value());

  ASSERT_EQ(tensor_data.size(), converted_data->size());
  for (size_t i = 0; i < tensor_data.size(); ++i) {
    ASSERT_FLOAT_EQ(tensor_data[i], (*converted_data)[i]);
  }
}

TEST(TensorPoolConvertStaticTensorTest, SameTypeConversionInt32) {
  TensorPool tensor_pool;

  std::vector<std::int32_t> tensor_data{0, 1, 2, 3, 4, 5};
  auto& tensor_wrapper = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, QuantizeParamsWrapperVariant{}, {1, 2, 3},
      sizeof(decltype(tensor_data)::value_type) * tensor_data.size(),
      tensor_data.data());

  auto* res = tensor_pool.ConvertStaticTensorFrom<std::int32_t>(tensor_wrapper);
  ASSERT_NE(res, nullptr);

  auto converted_data = res->GetStaticTensorData<std::int32_t>();
  ASSERT_TRUE(converted_data.has_value());

  ASSERT_EQ(tensor_data.size(), converted_data->size());
  for (size_t i = 0; i < tensor_data.size(); ++i) {
    ASSERT_EQ(tensor_data[i], (*converted_data)[i]);
  }
}

TEST(TensorPoolConvertStaticTensorTest, ExpandTypeConversionFloat32) {
  TensorPool tensor_pool;

  std::vector<float> tensor_data{0, 1, 2, 3, 4, 5};
  auto& tensor_wrapper = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_FLOAT_32, QuantizeParamsWrapperVariant{}, {1, 2, 3},
      sizeof(decltype(tensor_data)::value_type) * tensor_data.size(),
      tensor_data.data());

  auto* res = tensor_pool.ConvertStaticTensorFrom<double>(tensor_wrapper);
  ASSERT_NE(res, nullptr);

  auto converted_data = res->GetStaticTensorData<double>();
  ASSERT_TRUE(converted_data.has_value());

  ASSERT_EQ(tensor_data.size(), converted_data->size());
  for (size_t i = 0; i < tensor_data.size(); ++i) {
    ASSERT_DOUBLE_EQ(tensor_data[i], (*converted_data)[i]);
  }
}

TEST(TensorPoolConvertStaticTensorTest, ExpandTypeConversionInt32) {
  TensorPool tensor_pool;

  std::vector<std::int32_t> tensor_data{0, 1, 2, 3, 4, 5};
  auto& tensor_wrapper = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, QuantizeParamsWrapperVariant{}, {1, 2, 3},
      sizeof(decltype(tensor_data)::value_type) * tensor_data.size(),
      tensor_data.data());

  auto* res = tensor_pool.ConvertStaticTensorFrom<std::int64_t>(tensor_wrapper);
  ASSERT_NE(res, nullptr);

  auto converted_data = res->GetStaticTensorData<std::int64_t>();
  ASSERT_TRUE(converted_data.has_value());

  ASSERT_EQ(tensor_data.size(), converted_data->size());
  for (size_t i = 0; i < tensor_data.size(); ++i) {
    ASSERT_EQ(tensor_data[i], (*converted_data)[i]);
  }
}

TEST(TensorPoolConvertStaticTensorTest, NarrowTypeConversionFloat32) {
  TensorPool tensor_pool;

  std::vector<double> tensor_data{0, 1, 2, 3, 4, 5};
  auto& tensor_wrapper = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_FLOAT_64, QuantizeParamsWrapperVariant{}, {1, 2, 3},
      sizeof(decltype(tensor_data)::value_type) * tensor_data.size(),
      tensor_data.data());

  auto* res = tensor_pool.ConvertStaticTensorFrom<float>(tensor_wrapper);
  ASSERT_NE(res, nullptr);

  auto converted_data = res->GetStaticTensorData<float>();
  ASSERT_TRUE(converted_data.has_value());

  ASSERT_EQ(tensor_data.size(), converted_data->size());
  for (size_t i = 0; i < tensor_data.size(); ++i) {
    ASSERT_DOUBLE_EQ(tensor_data[i], (*converted_data)[i]);
  }
}

TEST(TensorPoolConvertStaticTensorTest, NarrowTypeConversionInt32) {
  TensorPool tensor_pool;

  std::vector<std::int64_t> tensor_data{0, 1, 2, 3, 4, 5};
  auto& tensor_wrapper = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_64, QuantizeParamsWrapperVariant{}, {1, 2, 3},
      sizeof(decltype(tensor_data)::value_type) * tensor_data.size(),
      tensor_data.data());

  auto* res = tensor_pool.ConvertStaticTensorFrom<std::int32_t>(tensor_wrapper);
  ASSERT_NE(res, nullptr);

  auto converted_data = res->GetStaticTensorData<std::int32_t>();
  ASSERT_TRUE(converted_data.has_value());

  ASSERT_EQ(tensor_data.size(), converted_data->size());
  for (size_t i = 0; i < tensor_data.size(); ++i) {
    ASSERT_EQ(tensor_data[i], (*converted_data)[i]);
  }
}

}  // namespace

}  // namespace qnn
