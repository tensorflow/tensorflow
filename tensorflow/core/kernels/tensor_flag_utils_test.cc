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

#include "tensorflow/core/kernels/tensor_flag_utils.h"

#include <vector>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/test.h"

namespace {

using ::int64_t;
using tensorflow::DataType;
using tensorflow::int32;
using tensorflow::Tensor;
using tensorflow::TTypes;
using tensorflow::error::INVALID_ARGUMENT;
using tensorflow::tensor_flag_utils::FindConfigValueForKey;
using tensorflow::tensor_flag_utils::GetLinearBucket;
using tensorflow::tensor_flag_utils::GetPowerBucket;
using tensorflow::tensor_flag_utils::ValidateScalarQuantityShardingConfig;
using tensorflow::tensor_flag_utils::ValidateSparseMatrixShardingConfig;

TEST(SparseUtilsTest, ValidateSparseMatrixShardingConfig) {
  // Only a default is specified.
  {
    Tensor t(DataType::DT_FLOAT, {});
    t.scalar<float>()() = 0.7;
    EXPECT_TRUE(ValidateSparseMatrixShardingConfig(t).ok());
  }
  {
    Tensor t(DataType::DT_FLOAT, {});
    t.scalar<float>()() = 1.0;
    EXPECT_TRUE(ValidateSparseMatrixShardingConfig(t).ok());
  }

  // Misshapen.
  {
    Tensor t(DataType::DT_FLOAT, {1, 1});
    int indx = 0;
    for (const float v : {60.0}) {
      t.flat<float>()(indx++) = v;
    }
    EXPECT_EQ(INVALID_ARGUMENT, ValidateSparseMatrixShardingConfig(t).code());
  }
  {
    Tensor t(DataType::DT_FLOAT, {1, 2});
    int indx = 0;
    for (const float v : {
             60.0,
             50.0,
         }) {
      t.flat<float>()(indx++) = v;
    }
    EXPECT_EQ(INVALID_ARGUMENT, ValidateSparseMatrixShardingConfig(t).code());
  }

  // Only one key is specified.
  {
    Tensor t(DataType::DT_FLOAT, {1, 3});
    int indx = 0;
    for (const float v : {30.0, 20.0, 1.0}) {
      t.flat<float>()(indx++) = v;
    }
    EXPECT_TRUE(ValidateSparseMatrixShardingConfig(t).ok());
  }

  // Two keys are specified.
  {
    Tensor t(DataType::DT_FLOAT, {2, 3});
    int indx = 0;
    for (const float v : {60.0, 50.0, 0.41, 30.0, 20.0, 0.7}) {
      t.flat<float>()(indx++) = v;
    }
    EXPECT_TRUE(ValidateSparseMatrixShardingConfig(t).ok());
  }

  // Out of range.
  {
    Tensor t(DataType::DT_FLOAT, {2, 3});
    int indx = 0;
    for (const float v : {60.0, 40.0, 0.41, 30.0, 20.0, 10.7}) {
      t.flat<float>()(indx++) = v;
    }
    EXPECT_EQ(INVALID_ARGUMENT, ValidateSparseMatrixShardingConfig(t).code());
  }
  {
    Tensor t(DataType::DT_FLOAT, {2, 3});
    int indx = 0;
    for (const float v : {60.0, 40.0, 0.41, 30.0, 20.0, -0.7}) {
      t.flat<float>()(indx++) = v;
    }
    EXPECT_EQ(INVALID_ARGUMENT, ValidateSparseMatrixShardingConfig(t).code());
  }
  {
    Tensor t(DataType::DT_FLOAT, {2, 3});
    int indx = 0;
    for (const float v : {60.0, -40.0, 0.41, 30.0, 20.0, 0.7}) {
      t.flat<float>()(indx++) = v;
    }
    EXPECT_EQ(INVALID_ARGUMENT, ValidateSparseMatrixShardingConfig(t).code());
  }
  {
    Tensor t(DataType::DT_FLOAT, {});
    t.scalar<float>()() = -0.5;
    EXPECT_EQ(INVALID_ARGUMENT, ValidateSparseMatrixShardingConfig(t).code());
  }
  {
    Tensor t(DataType::DT_FLOAT, {});
    t.scalar<float>()() = 0;
    EXPECT_EQ(INVALID_ARGUMENT, ValidateSparseMatrixShardingConfig(t).code());
  }
  {
    Tensor t(DataType::DT_FLOAT, {});
    t.scalar<float>()() = 1.2;
    EXPECT_EQ(INVALID_ARGUMENT, ValidateSparseMatrixShardingConfig(t).code());
  }
}

TEST(SparseUtilsTest, ValidateScalarQuantityShardingConfig) {
  // Only a default is specified.
  {
    Tensor t(DataType::DT_FLOAT, {});
    t.scalar<float>()() = 0.7;
    EXPECT_TRUE(ValidateScalarQuantityShardingConfig(t).ok());
  }
  {
    Tensor t(DataType::DT_FLOAT, {});
    t.scalar<float>()() = 1.0;
    EXPECT_TRUE(ValidateScalarQuantityShardingConfig(t).ok());
  }
  {
    Tensor t(DataType::DT_FLOAT, {});
    t.scalar<float>()() = 1.2;
    EXPECT_EQ(INVALID_ARGUMENT, ValidateScalarQuantityShardingConfig(t).code());
  }

  // Misshapen.
  {
    Tensor t(DataType::DT_FLOAT, {1, 1});
    int indx = 0;
    for (const float v : {60.0}) {
      t.flat<float>()(indx++) = v;
    }
    EXPECT_EQ(INVALID_ARGUMENT, ValidateScalarQuantityShardingConfig(t).code());
  }
  {
    Tensor t(DataType::DT_FLOAT, {1, 2});
    int indx = 0;
    for (const float v : {
             60.0,
             50.0,
         }) {
      t.flat<float>()(indx++) = v;
    }
    EXPECT_EQ(INVALID_ARGUMENT, ValidateScalarQuantityShardingConfig(t).code());
  }

  // Two keys are specified.
  {
    Tensor t(DataType::DT_FLOAT, {1, 3});
    int indx = 0;
    for (const float v : {30.0, 20.0, 1.0}) {
      t.flat<float>()(indx++) = v;
    }
    EXPECT_EQ(INVALID_ARGUMENT, ValidateScalarQuantityShardingConfig(t).code());
  }

  // Only one key is specified.
  {
    Tensor t(DataType::DT_FLOAT, {2, 2});
    int indx = 0;
    for (const float v : {60.0, 0.41, 30.0, 0.7}) {
      t.flat<float>()(indx++) = v;
    }
    EXPECT_TRUE(ValidateScalarQuantityShardingConfig(t).ok());
  }

  // Out of range.
  {
    Tensor t(DataType::DT_FLOAT, {2, 2});
    int indx = 0;
    for (const float v : {60.0, 0.41, 30.0, 10.7}) {
      t.flat<float>()(indx++) = v;
    }
    EXPECT_EQ(INVALID_ARGUMENT, ValidateScalarQuantityShardingConfig(t).code());
  }
  {
    Tensor t(DataType::DT_FLOAT, {2, 2});
    int indx = 0;
    for (const float v : {60.0, 0.41, 30.0, -0.7}) {
      t.flat<float>()(indx++) = v;
    }
    EXPECT_EQ(INVALID_ARGUMENT, ValidateScalarQuantityShardingConfig(t).code());
  }
  {
    Tensor t(DataType::DT_FLOAT, {2, 2});
    int indx = 0;
    for (const float v : {-40.0, 0.41, 20.0, 0.7}) {
      t.flat<float>()(indx++) = v;
    }
    EXPECT_EQ(INVALID_ARGUMENT, ValidateScalarQuantityShardingConfig(t).code());
  }
  {
    Tensor t(DataType::DT_FLOAT, {});
    t.scalar<float>()() = -0.5;
    EXPECT_EQ(INVALID_ARGUMENT, ValidateScalarQuantityShardingConfig(t).code());
  }
  {
    Tensor t(DataType::DT_FLOAT, {});
    t.scalar<float>()() = 0;
    EXPECT_EQ(INVALID_ARGUMENT, ValidateScalarQuantityShardingConfig(t).code());
  }
  {
    Tensor t(DataType::DT_FLOAT, {});
    t.scalar<float>()() = 1.2;
    EXPECT_EQ(INVALID_ARGUMENT, ValidateScalarQuantityShardingConfig(t).code());
  }
}

TEST(SparseUtils, FindConfigValueForKey) {
  {
    float data[] = {60.0, 50.0, 0.41, 30.0, 20.0, 0.1, 0, 0, 0.7};
    TTypes<float>::ConstMatrix config_mat(data, 3, 3);
    auto val = FindConfigValueForKey<float, int32>(config_mat, {70, 40});
    EXPECT_FLOAT_EQ(0.1, val);
    val = FindConfigValueForKey<float, int32>(config_mat, {60, 50});
    EXPECT_FLOAT_EQ(0.41, val);
    val = FindConfigValueForKey<float, int32>(config_mat, {60, 60});
    EXPECT_FLOAT_EQ(0.41, val);
    val = FindConfigValueForKey<float, int32>(config_mat, {60, 40});
    EXPECT_FLOAT_EQ(0.1, val);
    val = FindConfigValueForKey<float, int32>(config_mat, {50, 60});
    EXPECT_FLOAT_EQ(0.1, val);
    val = FindConfigValueForKey<float, int32>(config_mat, {20, 30});
    EXPECT_FLOAT_EQ(0.7, val);
    val = FindConfigValueForKey<float, int32>(config_mat, {30, 10});
    EXPECT_FLOAT_EQ(0.7, val);
  }
  {
    float data[] = {0, 0, 0.7};
    TTypes<float>::ConstMatrix config_mat(data, 1, 3);
    auto val = FindConfigValueForKey<float, int64_t>(config_mat, {70, 40});
    EXPECT_FLOAT_EQ(0.7, val);
    val = FindConfigValueForKey<float, int64_t>(config_mat, {60, 50});
    EXPECT_FLOAT_EQ(0.7, val);
    val = FindConfigValueForKey<float, int64_t>(config_mat, {60, 60});
    EXPECT_FLOAT_EQ(0.7, val);
    val = FindConfigValueForKey<float, int64_t>(config_mat, {60, 40});
    EXPECT_FLOAT_EQ(0.7, val);
    val = FindConfigValueForKey<float, int64_t>(config_mat, {50, 60});
    EXPECT_FLOAT_EQ(0.7, val);
    val = FindConfigValueForKey<float, int64_t>(config_mat, {20, 30});
    EXPECT_FLOAT_EQ(0.7, val);
    val = FindConfigValueForKey<float, int64_t>(config_mat, {30, 10});
    EXPECT_FLOAT_EQ(0.7, val);
  }
  {
    float data[] = {60.0, 50.0, 0.41, 0, 0, 0.7};
    TTypes<float>::ConstMatrix config_mat(data, 2, 3);
    auto val = FindConfigValueForKey<float, int32>(config_mat, {70, 40});
    EXPECT_FLOAT_EQ(0.7, val);
    val = FindConfigValueForKey<float, int32>(config_mat, {60, 50});
    EXPECT_FLOAT_EQ(0.41, val);
    val = FindConfigValueForKey<float, int32>(config_mat, {60, 60});
    EXPECT_FLOAT_EQ(0.41, val);
    val = FindConfigValueForKey<float, int32>(config_mat, {60, 40});
    EXPECT_FLOAT_EQ(0.7, val);
    val = FindConfigValueForKey<float, int32>(config_mat, {50, 60});
    EXPECT_FLOAT_EQ(0.7, val);
    val = FindConfigValueForKey<float, int32>(config_mat, {20, 30});
    EXPECT_FLOAT_EQ(0.7, val);
    val = FindConfigValueForKey<float, int32>(config_mat, {30, 10});
    EXPECT_FLOAT_EQ(0.7, val);
  }
  {
    float data[] = {60.0, 0.41, 50.0, 0.14, 0, 0.7};
    TTypes<float>::ConstMatrix config_mat(data, 3, 2);
    auto val = FindConfigValueForKey<float, int32>(config_mat, 70);
    EXPECT_FLOAT_EQ(0.41, val);
    val = FindConfigValueForKey<float, int32>(config_mat, 60);
    EXPECT_FLOAT_EQ(0.41, val);
    val = FindConfigValueForKey<float, int32>(config_mat, 55);
    EXPECT_FLOAT_EQ(0.14, val);
    val = FindConfigValueForKey<float, int32>(config_mat, 50);
    EXPECT_FLOAT_EQ(0.14, val);
    val = FindConfigValueForKey<float, int32>(config_mat, 20);
    EXPECT_FLOAT_EQ(0.7, val);
    val = FindConfigValueForKey<float, int32>(config_mat, 30);
    EXPECT_FLOAT_EQ(0.7, val);
  }
}

TEST(SparseUtils, GetLinearBucket) {
  EXPECT_EQ(11, GetLinearBucket(11, 5));
  EXPECT_EQ(11, GetLinearBucket(12, 5));
  EXPECT_EQ(1, GetLinearBucket(int64_t{4}, int64_t{5}));
}

TEST(SparseUtils, GetPowerBucket) {
  EXPECT_EQ(6, GetPowerBucket(11, 5));
  EXPECT_EQ(6, GetPowerBucket(12, 5));
  EXPECT_EQ(1332, GetPowerBucket(1335, 11));
  EXPECT_EQ(5, GetPowerBucket(int64_t{5}, int64_t{4}));
  EXPECT_EQ(1, GetPowerBucket(int64_t{4}, int64_t{1}));
}

}  // namespace
