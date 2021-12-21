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
  EXPECT_THAT(ret.status().error_message(),
              ::testing::HasSubstr("expected identical input types"));
}

TEST(UnaryContainerCreate, Basic) {
  FullTypeDef t1;
  t1.set_type_id(TFT_ANY);
  FullTypeDef t2;
  t2.set_type_id(TFT_TENSOR);
  FullTypeDef t3;
  t3.set_type_id(TFT_ANY);

  const auto ret = UnaryContainerCreate(TFT_ARRAY, 1)({t1, t2, t3});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.ValueOrDie();
  EXPECT_EQ(rt.type_id(), TFT_PRODUCT);
  ASSERT_EQ(rt.args_size(), 1);
  EXPECT_EQ(rt.args(0).type_id(), TFT_ARRAY);
  ASSERT_EQ(rt.args(0).args_size(), 1);
  EXPECT_EQ(rt.args(0).args(0).type_id(), TFT_TENSOR);
}

TEST(UnaryContainerAdd, Basic) {
  FullTypeDef t1;
  t1.set_type_id(TFT_ANY);
  FullTypeDef t2;
  t2.set_type_id(TFT_TENSOR);
  FullTypeDef t3;
  t3.set_type_id(TFT_ARRAY);

  const auto ret =
      UnaryContainerAdd(TFT_ARRAY, /*container_idx=*/2, /*element_idx=*/1,
                        /*homogeneous=*/false)({t1, t2, t3});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.ValueOrDie();
  EXPECT_EQ(rt.type_id(), TFT_PRODUCT);
  ASSERT_EQ(rt.args_size(), 1);
  EXPECT_EQ(rt.args(0).type_id(), TFT_ARRAY);
  ASSERT_EQ(rt.args(0).args_size(), 1);
  EXPECT_EQ(rt.args(0).args(0).type_id(), TFT_TENSOR);
}

TEST(UnaryContainerAdd, RejectsMismatchedContainerType) {
  FullTypeDef t1;
  t1.set_type_id(TFT_TENSOR);
  FullTypeDef t2;
  t2.set_type_id(TFT_DATASET);

  const auto ret =
      UnaryContainerAdd(TFT_ARRAY, /*container_idx=*/1, /*element_idx=*/0,
                        /*homogeneous=*/false)({t1, t2});
  EXPECT_THAT(ret.status().error_message(),
              ::testing::HasSubstr("expected container type"));
}

TEST(UnaryContainerAdd, IgnoresUnsetContainerType) {
  FullTypeDef t1;
  t1.set_type_id(TFT_TENSOR);
  FullTypeDef t2;

  const auto ret =
      UnaryContainerAdd(TFT_ARRAY, /*container_idx=*/1, /*element_idx=*/0,
                        /*homogeneous=*/false)({t1, t2});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.ValueOrDie();
  EXPECT_EQ(rt.type_id(), TFT_PRODUCT);
  ASSERT_EQ(rt.args_size(), 1);
  EXPECT_EQ(rt.args(0).type_id(), TFT_ARRAY);
  ASSERT_EQ(rt.args(0).args_size(), 1);
  EXPECT_EQ(rt.args(0).args(0).type_id(), TFT_TENSOR);
}

TEST(UnaryContainerAdd, UnsetElementTypeRemainsUnset) {
  FullTypeDef t1;
  t1.set_type_id(TFT_ANY);
  FullTypeDef t2;
  FullTypeDef t3;
  t3.set_type_id(TFT_ARRAY);

  const auto ret =
      UnaryContainerAdd(TFT_ARRAY, /*container_idx=*/2, /*element_idx=*/1,
                        /*homogeneous=*/false)({t1, t2, t3});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.ValueOrDie();
  EXPECT_EQ(rt.type_id(), TFT_PRODUCT);
  ASSERT_EQ(rt.args_size(), 1);
  EXPECT_EQ(rt.args(0).type_id(), TFT_ARRAY);
  ASSERT_EQ(rt.args(0).args_size(), 0);
}

TEST(UnaryContainerAdd, UnsetElementTypeKeepsOriginalElementType) {
  FullTypeDef t1;
  t1.set_type_id(TFT_ARRAY);
  t1.add_args()->set_type_id(TFT_TENSOR);
  FullTypeDef t2;

  const auto ret =
      UnaryContainerAdd(TFT_ARRAY, /*container_idx=*/0, /*element_idx=*/1,
                        /*homogeneous=*/false)({t1, t2});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.ValueOrDie();
  EXPECT_EQ(rt.type_id(), TFT_PRODUCT);
  ASSERT_EQ(rt.args_size(), 1);
  EXPECT_EQ(rt.args(0).type_id(), TFT_ARRAY);
  ASSERT_EQ(rt.args(0).args_size(), 1);
  EXPECT_EQ(rt.args(0).args(0).type_id(), TFT_TENSOR);
}

TEST(UnaryContainerAdd, KeepsContainerTypeIfElementIsSubtype) {
  // TODO(mdan): We may want to refine the type if homogeneous.
  FullTypeDef t1;
  t1.set_type_id(TFT_ARRAY);
  t1.add_args()->set_type_id(TFT_ANY);
  FullTypeDef t2;
  t2.set_type_id(TFT_TENSOR);

  const auto ret =
      UnaryContainerAdd(TFT_ARRAY, /*container_idx=*/0, /*element_idx=*/1,
                        /*homogeneous=*/true)({t1, t2});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.ValueOrDie();
  EXPECT_EQ(rt.type_id(), TFT_PRODUCT);
  ASSERT_EQ(rt.args_size(), 1);
  EXPECT_EQ(rt.args(0).type_id(), TFT_ARRAY);
  ASSERT_EQ(rt.args(0).args_size(), 1);
  EXPECT_EQ(rt.args(0).args(0).type_id(), TFT_ANY);
}

TEST(UnaryContainerAdd, RejectsMismatchedElementTypesHeterogenous) {
  // TODO(mdan): Implement if needed (see full_type_inference_util.cc).
  FullTypeDef t1;
  t1.set_type_id(TFT_ARRAY);
  t1.add_args()->set_type_id(TFT_TENSOR);
  FullTypeDef t2;
  t2.set_type_id(TFT_DATASET);

  const auto ret =
      UnaryContainerAdd(TFT_ARRAY, /*container_idx=*/0, /*element_idx=*/1,
                        /*homogeneous=*/false)({t1, t2});
  EXPECT_THAT(ret.status().error_message(),
              ::testing::HasSubstr("need union types"));
}

TEST(UnaryContainerAdd, RejectsMismatchedElementTypesHomogeneous) {
  FullTypeDef t1;
  t1.set_type_id(TFT_ARRAY);
  t1.add_args()->set_type_id(TFT_TENSOR);
  FullTypeDef t2;
  t2.set_type_id(TFT_DATASET);

  const auto ret =
      UnaryContainerAdd(TFT_ARRAY, /*container_idx=*/0, /*element_idx=*/1,
                        /*homogeneous=*/true)({t1, t2});
  EXPECT_THAT(ret.status().error_message(),
              ::testing::HasSubstr("expected a subtype"));
}

TEST(UnaryContainerAdd, RejectsSupertypeElementTypeHeterogeneous) {
  // TODO(mdan): Implement if needed (see full_type_inference_util.cc).
  FullTypeDef t1;
  t1.set_type_id(TFT_ARRAY);
  t1.add_args()->set_type_id(TFT_TENSOR);
  FullTypeDef t2;
  t2.set_type_id(TFT_ANY);

  const auto ret =
      UnaryContainerAdd(TFT_ARRAY, /*container_idx=*/0, /*element_idx=*/1,
                        /*homogeneous=*/false)({t1, t2});
  EXPECT_THAT(ret.status().error_message(),
              ::testing::HasSubstr("need union types"));
}

TEST(UnaryContainerAdd, RejectsSupertypeElementTypeHomogeneous) {
  // TODO(mdan): This might be acceptable.
  FullTypeDef t1;
  t1.set_type_id(TFT_ARRAY);
  t1.add_args()->set_type_id(TFT_TENSOR);
  FullTypeDef t2;
  t2.set_type_id(TFT_ANY);

  const auto ret =
      UnaryContainerAdd(TFT_ARRAY, /*container_idx=*/0, /*element_idx=*/1,
                        /*homogeneous=*/true)({t1, t2});
  EXPECT_THAT(ret.status().error_message(),
              ::testing::HasSubstr("expected a subtype"));
}

}  // namespace

}  // namespace full_type

}  // namespace tensorflow
