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

#include "tensorflow/core/framework/full_type.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

namespace full_type {

namespace {

TEST(ReplicateInput, Default) {
  FullTypeDef t;
  t.set_type_id(TFT_ARRAY);

  const auto ret = ReplicateInput()({t});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.ValueOrDie();
  EXPECT_EQ(rt.type_id(), TFT_PRODUCT);
  ASSERT_EQ(rt.args_size(), 1);
  EXPECT_EQ(rt.args(0).type_id(), TFT_ARRAY);
}

TEST(ReplicateInput, Duplicate) {
  FullTypeDef t;
  t.set_type_id(TFT_ARRAY);

  const auto ret = ReplicateInput(0, 2)({t});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.ValueOrDie();
  EXPECT_EQ(rt.type_id(), TFT_PRODUCT);
  ASSERT_EQ(rt.args_size(), 2);
  EXPECT_EQ(rt.args(0).type_id(), TFT_ARRAY);
  EXPECT_EQ(rt.args(1).type_id(), TFT_ARRAY);
}

TEST(ReplicateInput, FirstOfMultipleArgs) {
  FullTypeDef t1;
  t1.set_type_id(TFT_ARRAY);
  FullTypeDef t2;
  t2.set_type_id(TFT_TENSOR);

  const auto ret = ReplicateInput(0, 2)({t1, t2});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.ValueOrDie();
  EXPECT_EQ(rt.type_id(), TFT_PRODUCT);
  ASSERT_EQ(rt.args_size(), 2);
  EXPECT_EQ(rt.args(0).type_id(), TFT_ARRAY);
  EXPECT_EQ(rt.args(1).type_id(), TFT_ARRAY);
}

TEST(ReplicateInput, SecondOfMultipleArgs) {
  FullTypeDef t1;
  t1.set_type_id(TFT_ARRAY);
  FullTypeDef t2;
  t2.set_type_id(TFT_TENSOR);

  const auto ret = ReplicateInput(1, 2)({t1, t2});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.ValueOrDie();
  EXPECT_EQ(rt.type_id(), TFT_PRODUCT);
  ASSERT_EQ(rt.args_size(), 2);
  EXPECT_EQ(rt.args(0).type_id(), TFT_TENSOR);
  EXPECT_EQ(rt.args(1).type_id(), TFT_TENSOR);
}

TEST(ReplicateInput, Unset) {
  FullTypeDef t;

  const auto ret = ReplicateInput()({t});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.ValueOrDie();
  EXPECT_EQ(rt.type_id(), TFT_UNSET);
}

TEST(ReplicateIdenticalInputs, Single) {
  FullTypeDef t;
  t.set_type_id(TFT_ARRAY);

  const auto ret = ReplicateIdenticalInputs()({t});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.ValueOrDie();
  EXPECT_EQ(rt.type_id(), TFT_PRODUCT);
  ASSERT_EQ(rt.args_size(), 1);
  EXPECT_EQ(rt.args(0).type_id(), TFT_ARRAY);
}

TEST(ReplicateIdenticalInputs, Double) {
  FullTypeDef t;
  t.set_type_id(TFT_ARRAY);

  const auto ret = ReplicateIdenticalInputs()({t, t});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.ValueOrDie();
  EXPECT_EQ(rt.type_id(), TFT_PRODUCT);
  ASSERT_EQ(rt.args_size(), 1);
  EXPECT_EQ(rt.args(0).type_id(), TFT_ARRAY);
}

TEST(ReplicateIdenticalInputs, Unset) {
  FullTypeDef t;
  t.set_type_id(TFT_UNSET);

  const auto ret = ReplicateIdenticalInputs()({t});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.ValueOrDie();
  EXPECT_EQ(rt.type_id(), TFT_UNSET);
}

TEST(ReplicateIdenticalInputs, UnsetComponents) {
  FullTypeDef t1;
  FullTypeDef t2;

  const auto ret = ReplicateIdenticalInputs()({t1, t2});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.ValueOrDie();
  EXPECT_EQ(rt.type_id(), TFT_UNSET);
}

TEST(ReplicateIdenticalInputs, UsesPartialInfo_FirstUnknown) {
  FullTypeDef t1;
  FullTypeDef t2;
  t2.set_type_id(TFT_ARRAY);

  const auto ret = ReplicateIdenticalInputs()({t1, t2});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.ValueOrDie();
  EXPECT_EQ(rt.type_id(), TFT_PRODUCT);
  ASSERT_EQ(rt.args_size(), 1);
  EXPECT_EQ(rt.args(0).type_id(), TFT_ARRAY);
}

TEST(ReplicateIdenticalInputs, UsesPartialInfo_SecondUnknown) {
  FullTypeDef t1;
  t1.set_type_id(TFT_ARRAY);
  FullTypeDef t2;

  const auto ret = ReplicateIdenticalInputs()({t1, t2});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.ValueOrDie();
  EXPECT_EQ(rt.type_id(), TFT_PRODUCT);
  ASSERT_EQ(rt.args_size(), 1);
  EXPECT_EQ(rt.args(0).type_id(), TFT_ARRAY);
}

TEST(ReplicateIdenticalInputs, RejectsMismatched) {
  FullTypeDef t1;
  t1.set_type_id(TFT_ARRAY);
  FullTypeDef t2;
  t2.set_type_id(TFT_TENSOR);

  const auto ret = ReplicateIdenticalInputs()({t1, t2});
  EXPECT_FALSE(ret.status().ok());
}

}  // namespace

}  // namespace full_type

}  // namespace tensorflow
