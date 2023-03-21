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

#include <functional>
#include <string>
#include <vector>

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

  const auto ret = ReplicateInput()({t}, {});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.value();
  EXPECT_EQ(rt.type_id(), TFT_PRODUCT);
  ASSERT_EQ(rt.args_size(), 1);
  EXPECT_EQ(rt.args(0).type_id(), TFT_ARRAY);
}

TEST(ReplicateInput, Duplicate) {
  FullTypeDef t;
  t.set_type_id(TFT_ARRAY);

  const auto ret = ReplicateInput(0, 2)({t}, {});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.value();
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

  const auto ret = ReplicateInput(0, 2)({t1, t2}, {});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.value();
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

  const auto ret = ReplicateInput(1, 2)({t1, t2}, {});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.value();
  EXPECT_EQ(rt.type_id(), TFT_PRODUCT);
  ASSERT_EQ(rt.args_size(), 2);
  EXPECT_EQ(rt.args(0).type_id(), TFT_TENSOR);
  EXPECT_EQ(rt.args(1).type_id(), TFT_TENSOR);
}

TEST(ReplicateInput, Unset) {
  FullTypeDef t;

  const auto ret = ReplicateInput()({t}, {});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.value();
  EXPECT_EQ(rt.type_id(), TFT_UNSET);
}

TEST(Merge, Single) {
  FullTypeDef t;
  t.set_type_id(TFT_ARRAY);

  const auto ret = Merge()({t}, {});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.value();
  EXPECT_EQ(rt.type_id(), TFT_PRODUCT);
  ASSERT_EQ(rt.args_size(), 1);
  EXPECT_EQ(rt.args(0).type_id(), TFT_ARRAY);
}

TEST(Merge, Double) {
  FullTypeDef t;
  t.set_type_id(TFT_ARRAY);

  const auto ret = Merge()({t, t}, {});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.value();
  EXPECT_EQ(rt.type_id(), TFT_PRODUCT);
  ASSERT_EQ(rt.args_size(), 1);
  EXPECT_EQ(rt.args(0).type_id(), TFT_ARRAY);
}

TEST(Merge, Unset) {
  FullTypeDef t;
  t.set_type_id(TFT_UNSET);

  const auto ret = Merge()({t}, {});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.value();
  EXPECT_EQ(rt.type_id(), TFT_UNSET);
}

TEST(Merge, UnsetComponents) {
  FullTypeDef t1;
  FullTypeDef t2;

  const auto ret = Merge()({t1, t2}, {});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.value();
  EXPECT_EQ(rt.type_id(), TFT_UNSET);
}

void ExpectInferredArrayOfTensor(StatusOr<FullTypeDef> ret) {
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.value();
  EXPECT_EQ(rt.type_id(), TFT_PRODUCT);
  ASSERT_EQ(rt.args_size(), 1);
  EXPECT_EQ(rt.args(0).type_id(), TFT_ARRAY);
  ASSERT_EQ(rt.args(0).args_size(), 1);
  EXPECT_EQ(rt.args(0).args(0).type_id(), TFT_TENSOR);
}

TEST(Merge, RejectsMismatched) {
  FullTypeDef t1;
  t1.set_type_id(TFT_ARRAY);
  FullTypeDef t2;
  t2.set_type_id(TFT_TENSOR);

  const auto ret = Merge()({t1, t2}, {});
  EXPECT_THAT(ret.status().error_message(),
              ::testing::HasSubstr("expected compatible input types"));
}

TEST(Merge, UsesPartialInfo) {
  FullTypeDef t1;
  FullTypeDef t2;
  t2.set_type_id(TFT_ARRAY);
  t2.add_args()->set_type_id(TFT_TENSOR);

  ExpectInferredArrayOfTensor(Merge()({t1, t2}, {}));
  ExpectInferredArrayOfTensor(Merge()({t2, t1}, {}));
}

TEST(Merge, SelectsMostSpecificOfSubtypes) {
  FullTypeDef t1;
  t1.set_type_id(TFT_ARRAY);
  t1.add_args()->set_type_id(TFT_ANY);
  FullTypeDef t2;
  t2.set_type_id(TFT_ARRAY);
  t2.add_args()->set_type_id(TFT_TENSOR);

  ExpectInferredArrayOfTensor(Merge()({t1, t2}, {}));
  ExpectInferredArrayOfTensor(Merge()({t2, t1}, {}));
}

TEST(UnaryContainerCreate, Basic) {
  FullTypeDef t1;
  t1.set_type_id(TFT_ANY);
  FullTypeDef t2;
  t2.set_type_id(TFT_TENSOR);
  FullTypeDef t3;
  t3.set_type_id(TFT_ANY);

  const auto ret = UnaryContainerCreate(TFT_ARRAY, 1)({t1, t2, t3}, {});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.value();
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
                        /*homogeneous=*/false)({t1, t2, t3}, {});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.value();
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
                        /*homogeneous=*/false)({t1, t2}, {});
  EXPECT_THAT(ret.status().error_message(),
              ::testing::HasSubstr("expected container type"));
}

TEST(UnaryContainerAdd, IgnoresUnsetContainerType) {
  FullTypeDef t1;
  t1.set_type_id(TFT_TENSOR);
  FullTypeDef t2;

  const auto ret =
      UnaryContainerAdd(TFT_ARRAY, /*container_idx=*/1, /*element_idx=*/0,
                        /*homogeneous=*/false)({t1, t2}, {});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.value();
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
                        /*homogeneous=*/false)({t1, t2, t3}, {});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.value();
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
                        /*homogeneous=*/false)({t1, t2}, {});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.value();
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
                        /*homogeneous=*/true)({t1, t2}, {});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.value();
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
                        /*homogeneous=*/false)({t1, t2}, {});
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
                        /*homogeneous=*/true)({t1, t2}, {});
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
                        /*homogeneous=*/false)({t1, t2}, {});
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
                        /*homogeneous=*/true)({t1, t2}, {});
  EXPECT_THAT(ret.status().error_message(),
              ::testing::HasSubstr("expected a subtype"));
}

TEST(MultiaryUnstack, Basic) {
  FullTypeDef t1;
  t1.set_type_id(TFT_TENSOR);

  const auto ret = MultiaryUnstack(TFT_DATASET, UnstackTensor)({t1}, {});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.value();
  EXPECT_EQ(rt.type_id(), TFT_PRODUCT);
  ASSERT_EQ(rt.args_size(), 1);
  EXPECT_EQ(rt.args(0).type_id(), TFT_DATASET);
  ASSERT_EQ(rt.args(0).args_size(), 1);
  ASSERT_EQ(rt.args(0).args(0).type_id(), TFT_PRODUCT);
  ASSERT_EQ(rt.args(0).args(0).args_size(), 1);
  ASSERT_EQ(rt.args(0).args(0).args(0).type_id(), TFT_TENSOR);
}

TEST(MultiaryUnstack, Ternary) {
  FullTypeDef t1;
  t1.set_type_id(TFT_RAGGED);
  t1.add_args()->set_type_id(TFT_STRING);
  FullTypeDef t2;
  t2.set_type_id(TFT_TENSOR);
  FullTypeDef t3;
  t3.set_type_id(TFT_RAGGED);
  t3.add_args()->set_type_id(TFT_INT64);

  const auto ret =
      MultiaryUnstack(TFT_DATASET, UnstackTensor)({t1, t2, t3}, {});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.value();
  EXPECT_EQ(rt.type_id(), TFT_PRODUCT);
  ASSERT_EQ(rt.args_size(), 1);
  EXPECT_EQ(rt.args(0).type_id(), TFT_DATASET);
  ASSERT_EQ(rt.args(0).args_size(), 1);
  ASSERT_EQ(rt.args(0).args(0).type_id(), TFT_PRODUCT);
  ASSERT_EQ(rt.args(0).args(0).args_size(), 3);
  ASSERT_EQ(rt.args(0).args(0).args(0).type_id(), TFT_RAGGED);
  ASSERT_EQ(rt.args(0).args(0).args(0).args_size(), 1);
  ASSERT_EQ(rt.args(0).args(0).args(0).args(0).type_id(), TFT_STRING);
  ASSERT_EQ(rt.args(0).args(0).args(1).type_id(), TFT_TENSOR);
  ASSERT_EQ(rt.args(0).args(0).args(2).type_id(), TFT_RAGGED);
  ASSERT_EQ(rt.args(0).args(0).args(2).args_size(), 1);
  ASSERT_EQ(rt.args(0).args(0).args(2).args(0).type_id(), TFT_INT64);
}

TEST(MapContainer, Basic) {
  FullTypeDef cont_t;
  cont_t.set_type_id(TFT_DATASET);
  FullTypeDef* el_t = cont_t.add_args();
  el_t->set_type_id(TFT_PRODUCT);
  (el_t->add_args())->set_type_id(TFT_TENSOR);

  const auto ret = ContainerMap(TFT_DATASET, 0, BatchTensor)({cont_t}, {});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.value();
  EXPECT_EQ(rt.type_id(), TFT_PRODUCT);
  ASSERT_EQ(rt.args_size(), 1);
  EXPECT_EQ(rt.args(0).type_id(), TFT_DATASET);
  ASSERT_EQ(rt.args(0).args_size(), 1);
  ASSERT_EQ(rt.args(0).args(0).type_id(), TFT_PRODUCT);
  ASSERT_EQ(rt.args(0).args(0).args_size(), 1);
  ASSERT_EQ(rt.args(0).args(0).args(0).type_id(), TFT_TENSOR);
}

TEST(MapContainer, Ternary) {
  FullTypeDef t1;
  t1.set_type_id(TFT_ANY);
  FullTypeDef cont_t;
  cont_t.set_type_id(TFT_DATASET);
  FullTypeDef* el_t = cont_t.add_args();
  el_t->set_type_id(TFT_PRODUCT);
  FullTypeDef* e1 = el_t->add_args();
  e1->set_type_id(TFT_RAGGED);
  e1->add_args()->set_type_id(TFT_STRING);
  FullTypeDef* e2 = el_t->add_args();
  e2->set_type_id(TFT_TENSOR);
  FullTypeDef* e3 = el_t->add_args();
  e3->set_type_id(TFT_RAGGED);
  e3->add_args()->set_type_id(TFT_INT64);
  FullTypeDef t3;
  t3.set_type_id(TFT_ANY);

  const auto ret =
      ContainerMap(TFT_DATASET, 1, BatchTensor)({t1, cont_t, t3}, {});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.value();
  EXPECT_EQ(rt.type_id(), TFT_PRODUCT);
  ASSERT_EQ(rt.args_size(), 1);
  EXPECT_EQ(rt.args(0).type_id(), TFT_DATASET);
  ASSERT_EQ(rt.args(0).args_size(), 1);
  ASSERT_EQ(rt.args(0).args(0).type_id(), TFT_PRODUCT);
  ASSERT_EQ(rt.args(0).args(0).args_size(), 3);
  ASSERT_EQ(rt.args(0).args(0).args(0).type_id(), TFT_RAGGED);
  ASSERT_EQ(rt.args(0).args(0).args(0).args_size(), 1);
  ASSERT_EQ(rt.args(0).args(0).args(0).args(0).type_id(), TFT_STRING);
  ASSERT_EQ(rt.args(0).args(0).args(1).type_id(), TFT_TENSOR);
  ASSERT_EQ(rt.args(0).args(0).args(2).type_id(), TFT_RAGGED);
  ASSERT_EQ(rt.args(0).args(0).args(2).args_size(), 1);
  ASSERT_EQ(rt.args(0).args(0).args(2).args(0).type_id(), TFT_INT64);
}

TEST(MapCovariant, Basic) {
  FullTypeDef t;
  t.set_type_id(TFT_TENSOR);
  t.add_args()->set_type_id(TFT_INT32);

  const auto ret = MapCovariant(TFT_TENSOR, TFT_DATASET, 0)({t}, {});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.value();
  ASSERT_EQ(rt.type_id(), TFT_PRODUCT);
  ASSERT_EQ(rt.args_size(), 1);
  EXPECT_EQ(rt.args(0).type_id(), TFT_DATASET);
  ASSERT_EQ(rt.args(0).args_size(), 1);
  EXPECT_EQ(rt.args(0).args(0).type_id(), TFT_INT32);
  ASSERT_EQ(rt.args(0).args(0).args_size(), 0);
}

TEST(MapCovariant, IgnoresUnset) {
  FullTypeDef t;
  t.set_type_id(TFT_UNSET);

  const auto ret = MapCovariant(TFT_TENSOR, TFT_DATASET, 0)({t}, {});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.value();
  EXPECT_EQ(rt.type_id(), TFT_UNSET);
  ASSERT_EQ(rt.args_size(), 0);
}

TEST(MapCovariant, RejectsMismatchedType) {
  FullTypeDef t;
  t.set_type_id(TFT_TENSOR);
  t.add_args()->set_type_id(TFT_INT32);

  const auto ret = MapCovariant(TFT_ARRAY, TFT_DATASET, 0)({t}, {});
  EXPECT_THAT(ret.status().error_message(),
              ::testing::HasSubstr("expected type"));
}

// Create a type inference function for the Tuple.Basic test (in a function so
// that when the function is used, the local variables used to create it are
// out-of-scope.) Return "Tuple([ReplicateInput(), Tensor(TFT_INT32)])", a case
// simimlar to the `Merge` op which has two outputs where the second output is
// always an int32 index.
static TypeInferenceFn tuple_func() {
  std::vector<TypeInferenceFn> func_list{ReplicateInput(), Tensor(TFT_INT32)};
  return Tuple(func_list);
}

TEST(Tuple, Basic) {
  const TypeInferenceFn ret_func = tuple_func();
  FullTypeDef t_in;
  t_in.set_type_id(TFT_TENSOR);
  t_in.add_args()->set_type_id(TFT_FLOAT);
  const auto ret = ret_func({t_in}, {});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.value();
  EXPECT_EQ(rt.type_id(), TFT_PRODUCT);
  ASSERT_EQ(rt.args_size(), 2);
  EXPECT_EQ(rt.args(0).type_id(), TFT_TENSOR);
  ASSERT_EQ(rt.args(0).args_size(), 1);
  EXPECT_EQ(rt.args(0).args(0).type_id(), TFT_FLOAT);
  EXPECT_EQ(rt.args(1).type_id(), TFT_TENSOR);
  ASSERT_EQ(rt.args(1).args_size(), 1);
  EXPECT_EQ(rt.args(1).args(0).type_id(), TFT_INT32);
}

TEST(Tuple, Unset) {
  const TypeInferenceFn ret_func = tuple_func();
  FullTypeDef t_in;  // input is TFT_UNSET
  const auto ret = ret_func({t_in}, {});
  TF_EXPECT_OK(ret.status());

  const FullTypeDef& rt = ret.value();
  EXPECT_EQ(rt.type_id(), TFT_UNSET);
  ASSERT_EQ(rt.args_size(), 0);
}

}  // namespace

}  // namespace full_type

}  // namespace tensorflow
