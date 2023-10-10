/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/shim/tf_tensor_view.h"

#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/tstring.h"

namespace tflite {
namespace shim {
namespace {

using ::tensorflow::protobuf::TextFormat;

TEST(TfTensorView, Bool) {
  ::tensorflow::TensorProto tf_tensor_pb;
  ASSERT_TRUE(TextFormat::ParseFromString(
      R"pb(
        dtype: DT_BOOL
        tensor_shape {
          dim:
          [ { size: 3 }
            , { size: 2 }]
        }
        bool_val: [ false, false, false, false, false, false ]
      )pb",
      &tf_tensor_pb));
  ::tensorflow::Tensor tf_tensor;
  ASSERT_TRUE(tf_tensor.FromProto(tf_tensor_pb));

  // Test move assignment
  auto t_premove_or = TensorView::New(&tf_tensor);
  ASSERT_TRUE(t_premove_or.ok()) << t_premove_or.status();
  auto t = std::move(t_premove_or.value());

  auto tensor_data_as_vector = t.Data<bool>();
  for (int i = 0; i < 3 * 2; ++i) tensor_data_as_vector[i] = i % 5 == 0;

  ASSERT_THAT(tf_tensor.SummarizeValue(10, true),
              ::testing::Eq("[[1 0]\n [0 0]\n [0 1]]"));
}

TEST(TfTensorView, Int32) {
  ::tensorflow::TensorProto tf_tensor_pb;
  ASSERT_TRUE(TextFormat::ParseFromString(
      R"pb(
        dtype: DT_INT32
        tensor_shape {
          dim:
          [ { size: 3 }
            , { size: 2 }]
        }
        int_val: [ 0, 0, 0, 0, 0, 0 ]
      )pb",
      &tf_tensor_pb));
  ::tensorflow::Tensor tf_tensor;
  ASSERT_TRUE(tf_tensor.FromProto(tf_tensor_pb));

  // Test move assignment
  auto t_premove_or = TensorView::New(&tf_tensor);
  ASSERT_TRUE(t_premove_or.ok()) << t_premove_or.status();
  auto t = std::move(t_premove_or.value());

  auto tensor_data_as_vector = t.Data<int32_t>();
  for (int i = 0; i < 3 * 2; ++i) tensor_data_as_vector[i] = i;

  ASSERT_THAT(tf_tensor.SummarizeValue(10, true),
              ::testing::Eq("[[0 1]\n [2 3]\n [4 5]]"));
}

TEST(TfTensorView, Int64) {
  ::tensorflow::TensorProto tf_tensor_pb;
  ASSERT_TRUE(TextFormat::ParseFromString(
      R"pb(
        dtype: DT_INT64
        tensor_shape {
          dim:
          [ { size: 3 }
            , { size: 2 }]
        }
        int_val: [ 0, 0, 0, 0, 0, 0 ]
      )pb",
      &tf_tensor_pb));
  ::tensorflow::Tensor tf_tensor;
  ASSERT_TRUE(tf_tensor.FromProto(tf_tensor_pb));
  auto t_or = TensorView::New(&tf_tensor);
  ASSERT_TRUE(t_or.ok()) << t_or.status();
  auto& t = t_or.value();

  auto tensor_data_as_vector = t.Data<int64_t>();
  for (int i = 0; i < 3 * 2; ++i) tensor_data_as_vector[i] = i;

  ASSERT_THAT(tf_tensor.SummarizeValue(10, true),
              ::testing::Eq("[[0 1]\n [2 3]\n [4 5]]"));
}

TEST(TfTensorView, Float) {
  ::tensorflow::TensorProto tf_tensor_pb;
  ASSERT_TRUE(TextFormat::ParseFromString(
      R"pb(
        dtype: DT_FLOAT
        tensor_shape {
          dim:
          [ { size: 3 }
            , { size: 2 }]
        }
        float_val: [ 0, 0, 0, 0, 0, 0 ]
      )pb",
      &tf_tensor_pb));
  ::tensorflow::Tensor tf_tensor;
  ASSERT_TRUE(tf_tensor.FromProto(tf_tensor_pb));
  auto t_or = TensorView::New(&tf_tensor);
  ASSERT_TRUE(t_or.ok()) << t_or.status();
  auto& t = t_or.value();

  auto tensor_data_as_vector = t.Data<float>();
  for (int i = 0; i < 3 * 2; ++i)
    tensor_data_as_vector[i] = static_cast<float>(i) / 2.0;

  ASSERT_THAT(tf_tensor.SummarizeValue(10, true),
              ::testing::Eq("[[0 0.5]\n [1 1.5]\n [2 2.5]]"));
}

TEST(TfTensorView, Double) {
  ::tensorflow::TensorProto tf_tensor_pb;
  ASSERT_TRUE(TextFormat::ParseFromString(
      R"pb(
        dtype: DT_DOUBLE
        tensor_shape {
          dim:
          [ { size: 3 }
            , { size: 2 }]
        }
        double_val: [ 0, 0, 0, 0, 0, 0 ]
      )pb",
      &tf_tensor_pb));
  ::tensorflow::Tensor tf_tensor;
  ASSERT_TRUE(tf_tensor.FromProto(tf_tensor_pb));
  auto t_or = TensorView::New(&tf_tensor);
  ASSERT_TRUE(t_or.ok()) << t_or.status();
  auto& t = t_or.value();

  auto tensor_data_as_vector = t.Data<double>();
  for (int i = 0; i < 3 * 2; ++i)
    tensor_data_as_vector[i] = static_cast<double>(i) / 2.0;

  ASSERT_THAT(tf_tensor.SummarizeValue(10, true),
              ::testing::Eq("[[0 0.5]\n [1 1.5]\n [2 2.5]]"));
}

TEST(TfTensorView, Str) {
  ::tensorflow::TensorProto tf_tensor_pb;
  ASSERT_TRUE(TextFormat::ParseFromString(
      R"pb(
        dtype: DT_STRING
        tensor_shape {
          dim:
          [ { size: 3 }
            , { size: 2 }]
        }
        string_val: [ "", "", "", "", "", "" ]
      )pb",
      &tf_tensor_pb));
  ::tensorflow::Tensor tf_tensor;
  ASSERT_TRUE(tf_tensor.FromProto(tf_tensor_pb));
  auto t_or = TensorView::New(&tf_tensor);
  ASSERT_TRUE(t_or.ok()) << t_or.status();
  auto& t = t_or.value();

  auto tensor_data_as_vector = t.Data<::tensorflow::tstring>();
  tensor_data_as_vector[0] = "a";
  tensor_data_as_vector[1] = "bc";
  tensor_data_as_vector[2] = "def";
  tensor_data_as_vector[3] = "g";
  tensor_data_as_vector[4] = "hi";
  tensor_data_as_vector[5] = "";

  EXPECT_THAT(t.Data<::tensorflow::tstring>(),
              ::testing::ElementsAre("a", "bc", "def", "g", "hi", ""));

  const auto& const_tf_tensor = tf_tensor;
  const auto const_t_or = TensorView::New(&const_tf_tensor);
  ASSERT_TRUE(const_t_or.ok()) << const_t_or.status();
  const auto& const_t = const_t_or.value();

  EXPECT_THAT(const_t.Data<::tensorflow::tstring>(),
              ::testing::ElementsAre("a", "bc", "def", "g", "hi", ""));

  const char expectation[] = R"(
[["a" "bc"]
 ["def" "g"]
 ["hi" ""]])";

  EXPECT_THAT(tf_tensor.SummarizeValue(10, true),
              ::testing::Eq(absl::string_view(expectation).substr(1)));
}

}  // namespace
}  // namespace shim
}  // namespace tflite
