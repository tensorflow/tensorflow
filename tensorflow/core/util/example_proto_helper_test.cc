/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/util/example_proto_helper.h"

#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

constexpr char kDenseInt64Key[] = "dense_int64";
constexpr char kDenseFloatKey[] = "dense_float";
constexpr char kDenseStringKey[] = "dense_string";

constexpr char kSparseInt64Key[] = "sparse_int64";
constexpr char kSparseFloatKey[] = "sparse_float";
constexpr char kSparseStringKey[] = "sparse_string";

// Note that this method is also extensively tested by the python unit test:
// tensorflow/python/kernel_tests/parsing_ops_test.py
class SingleExampleProtoToTensorsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Setup dense feature configuration.
    FixedLenFeature int64_dense_config;
    int64_dense_config.key = kDenseInt64Key;
    int64_dense_config.dtype = DT_INT64;
    int64_dense_config.shape = TensorShape({1});
    int64_dense_config.default_value = Tensor(DT_INT64, TensorShape({1}));
    int64_dense_config.default_value.scalar<int64>()() = 0;
    dense_vec_.push_back(int64_dense_config);

    FixedLenFeature float_dense_config;
    float_dense_config.key = kDenseFloatKey;
    float_dense_config.dtype = DT_FLOAT;
    float_dense_config.shape = TensorShape({1});
    float_dense_config.default_value = Tensor(DT_FLOAT, TensorShape({1}));
    float_dense_config.default_value.scalar<float>()() = 0.0;
    dense_vec_.push_back(float_dense_config);

    FixedLenFeature string_dense_config;
    string_dense_config.key = kDenseStringKey;
    string_dense_config.dtype = DT_STRING;
    string_dense_config.shape = TensorShape({1});
    string_dense_config.default_value = Tensor(DT_STRING, TensorShape({1}));
    string_dense_config.default_value.scalar<tstring>()() = "default";
    dense_vec_.push_back(string_dense_config);

    // Setup sparse feature configuration.
    VarLenFeature int64_sparse_config;
    int64_sparse_config.key = kSparseInt64Key;
    int64_sparse_config.dtype = DT_INT64;
    sparse_vec_.push_back(int64_sparse_config);

    VarLenFeature float_sparse_config;
    float_sparse_config.key = kSparseFloatKey;
    float_sparse_config.dtype = DT_FLOAT;
    sparse_vec_.push_back(float_sparse_config);

    VarLenFeature string_sparse_config;
    string_sparse_config.key = kSparseStringKey;
    string_sparse_config.dtype = DT_STRING;
    sparse_vec_.push_back(string_sparse_config);
  }

  std::vector<FixedLenFeature> dense_vec_;
  std::vector<VarLenFeature> sparse_vec_;
};

TEST_F(SingleExampleProtoToTensorsTest, SparseOnlyTrivial) {
  Example ex;
  // Set up a feature for each of our supported types.
  (*ex.mutable_features()->mutable_feature())[kSparseInt64Key]
      .mutable_int64_list()
      ->add_value(42);
  (*ex.mutable_features()->mutable_feature())[kSparseFloatKey]
      .mutable_float_list()
      ->add_value(4.2);
  (*ex.mutable_features()->mutable_feature())[kSparseStringKey]
      .mutable_bytes_list()
      ->add_value("forty-two");

  std::vector<Tensor*> output_dense_values(0);
  std::vector<std::vector<Tensor>> output_sparse_values_tmp(3);
  for (int i = 0; i < 3; ++i) {
    output_sparse_values_tmp[i] = std::vector<Tensor>(1);
  }

  std::vector<FixedLenFeature> empty_dense_vec;
  TF_EXPECT_OK(SingleExampleProtoToTensors(ex, "", 0, empty_dense_vec,
                                           sparse_vec_, &output_dense_values,
                                           &output_sparse_values_tmp));

  const std::vector<Tensor>& int64_tensor_vec = output_sparse_values_tmp[0];
  EXPECT_EQ(1, int64_tensor_vec.size());
  EXPECT_EQ(42, int64_tensor_vec[0].vec<int64>()(0));

  const std::vector<Tensor>& float_tensor_vec = output_sparse_values_tmp[1];
  EXPECT_EQ(1, float_tensor_vec.size());
  EXPECT_NEAR(4.2, float_tensor_vec[0].vec<float>()(0), 0.001);

  const std::vector<Tensor>& string_tensor_vec = output_sparse_values_tmp[2];
  EXPECT_EQ(1, string_tensor_vec.size());
  EXPECT_EQ("forty-two", string_tensor_vec[0].vec<tstring>()(0));
}

TEST_F(SingleExampleProtoToTensorsTest, SparseOnlyEmpty) {
  Example empty;
  std::vector<Tensor*> output_dense_values(0);
  std::vector<std::vector<Tensor>> output_sparse_values_tmp(3);
  for (int i = 0; i < 3; ++i) {
    output_sparse_values_tmp[i] = std::vector<Tensor>(1);
  }

  std::vector<FixedLenFeature> empty_dense_vec;
  TF_EXPECT_OK(SingleExampleProtoToTensors(empty, "", 0, empty_dense_vec,
                                           sparse_vec_, &output_dense_values,
                                           &output_sparse_values_tmp));

  // Each feature will still have a tensor vector, however the tensor
  // in the vector will be empty.
  const std::vector<Tensor>& int64_tensor_vec = output_sparse_values_tmp[0];
  EXPECT_EQ(1, int64_tensor_vec.size());
  EXPECT_EQ(0, int64_tensor_vec[0].vec<int64>().size());

  const std::vector<Tensor>& float_tensor_vec = output_sparse_values_tmp[1];
  EXPECT_EQ(1, float_tensor_vec.size());
  EXPECT_EQ(0, float_tensor_vec[0].vec<float>().size());

  const std::vector<Tensor>& string_tensor_vec = output_sparse_values_tmp[2];
  EXPECT_EQ(1, string_tensor_vec.size());
  EXPECT_EQ(0, string_tensor_vec[0].vec<tstring>().size());
}

TEST_F(SingleExampleProtoToTensorsTest, DenseOnlyTrivial) {
  Example ex;
  // Set up a feature for each of our supported types.
  (*ex.mutable_features()->mutable_feature())[kDenseInt64Key]
      .mutable_int64_list()
      ->add_value(42);
  (*ex.mutable_features()->mutable_feature())[kDenseFloatKey]
      .mutable_float_list()
      ->add_value(4.2);
  (*ex.mutable_features()->mutable_feature())[kDenseStringKey]
      .mutable_bytes_list()
      ->add_value("forty-two");

  std::vector<Tensor*> output_dense_values(3);
  Tensor int64_dense_output(DT_INT64, TensorShape({1, 1}));
  output_dense_values[0] = &int64_dense_output;

  Tensor float_dense_output(DT_FLOAT, TensorShape({1, 1}));
  output_dense_values[1] = &float_dense_output;

  Tensor str_dense_output(DT_STRING, TensorShape({1, 1}));
  output_dense_values[2] = &str_dense_output;

  std::vector<VarLenFeature> empty_sparse_vec;
  std::vector<std::vector<Tensor>> output_sparse_values_tmp;
  TF_EXPECT_OK(SingleExampleProtoToTensors(
      ex, "", 0, dense_vec_, empty_sparse_vec, &output_dense_values,
      &output_sparse_values_tmp));
  EXPECT_TRUE(output_sparse_values_tmp.empty());

  EXPECT_EQ(1, int64_dense_output.matrix<int64>().size());
  EXPECT_EQ(42, int64_dense_output.matrix<int64>()(0, 0));

  EXPECT_EQ(1, float_dense_output.matrix<float>().size());
  EXPECT_NEAR(4.2, float_dense_output.matrix<float>()(0, 0), 0.001);

  EXPECT_EQ(1, str_dense_output.matrix<tstring>().size());
  EXPECT_EQ("forty-two", str_dense_output.matrix<tstring>()(0, 0));
}

TEST_F(SingleExampleProtoToTensorsTest, DenseOnlyDefaults) {
  std::vector<Tensor*> output_dense_values(3);
  Tensor int64_dense_output(DT_INT64, TensorShape({1, 1}));
  output_dense_values[0] = &int64_dense_output;

  Tensor float_dense_output(DT_FLOAT, TensorShape({1, 1}));
  output_dense_values[1] = &float_dense_output;

  Tensor str_dense_output(DT_STRING, TensorShape({1, 1}));
  output_dense_values[2] = &str_dense_output;

  Example empty;

  std::vector<VarLenFeature> empty_sparse_vec;
  std::vector<std::vector<Tensor>> output_sparse_values_tmp;
  TF_EXPECT_OK(SingleExampleProtoToTensors(
      empty, "", 0, dense_vec_, empty_sparse_vec, &output_dense_values,
      &output_sparse_values_tmp));

  EXPECT_EQ(1, int64_dense_output.matrix<int64>().size());
  EXPECT_EQ(0, int64_dense_output.matrix<int64>()(0, 0));

  EXPECT_EQ(1, float_dense_output.matrix<float>().size());
  EXPECT_NEAR(0.0, float_dense_output.matrix<float>()(0, 0), 0.001);

  EXPECT_EQ(1, str_dense_output.matrix<tstring>().size());
  EXPECT_EQ("default", str_dense_output.matrix<tstring>()(0, 0));
}

}  // namespace
}  // namespace tensorflow
