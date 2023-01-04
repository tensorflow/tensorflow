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

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/full_type.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

namespace full_type {

namespace {

// TODO(mdan): Use ParseTextProto, ProtoEquals when available in a clean lib.

TEST(Nullary, Basic) {
  OpTypeConstructor ctor = Nullary(TFT_TENSOR);

  OpDef op;
  op.add_output_arg();

  TF_ASSERT_OK(ctor(&op));

  const FullTypeDef& t = op.output_arg(0).experimental_full_type();
  EXPECT_EQ(t.type_id(), TFT_TENSOR);
  EXPECT_EQ(t.args_size(), 0);
}

TEST(Unary, Basic) {
  OpTypeConstructor ctor = Unary(TFT_TENSOR, "T");

  OpDef op;
  op.add_output_arg();

  TF_ASSERT_OK(ctor(&op));

  const FullTypeDef& t = op.output_arg(0).experimental_full_type();
  EXPECT_EQ(t.type_id(), TFT_TENSOR);
  EXPECT_EQ(t.args_size(), 1);
  EXPECT_EQ(t.args(0).type_id(), TFT_VAR);
  EXPECT_EQ(t.args(0).args_size(), 0);
  EXPECT_EQ(t.args(0).s(), "T");
}

TEST(UnaryGeneric, Basic) {
  OpTypeConstructor ctor = UnaryGeneric(TFT_TENSOR);

  OpDef op;
  op.add_output_arg();

  TF_ASSERT_OK(ctor(&op));

  const FullTypeDef& t = op.output_arg(0).experimental_full_type();
  EXPECT_EQ(t.type_id(), TFT_TENSOR);
  EXPECT_EQ(t.args_size(), 1);
  EXPECT_EQ(t.args(0).type_id(), TFT_ANY);
  EXPECT_EQ(t.args(0).args_size(), 0);
}

TEST(UnaryTensorContainer, Fixed) {
  OpTypeConstructor ctor = UnaryTensorContainer(TFT_ARRAY, TFT_INT32);

  OpDef op;
  op.add_output_arg();

  TF_ASSERT_OK(ctor(&op));

  const FullTypeDef& t = op.output_arg(0).experimental_full_type();
  EXPECT_EQ(t.type_id(), TFT_ARRAY);
  EXPECT_EQ(t.args_size(), 1);
  EXPECT_EQ(t.args(0).type_id(), TFT_TENSOR);
  EXPECT_EQ(t.args(0).args_size(), 1);
  EXPECT_EQ(t.args(0).args(0).type_id(), TFT_INT32);
  EXPECT_EQ(t.args(0).args(0).args_size(), 0);
}

TEST(UnaryTensorContainer, Dependent) {
  OpTypeConstructor ctor = UnaryTensorContainer(TFT_ARRAY, "T");

  OpDef op;
  op.add_output_arg();

  TF_ASSERT_OK(ctor(&op));

  const FullTypeDef& t = op.output_arg(0).experimental_full_type();
  EXPECT_EQ(t.type_id(), TFT_ARRAY);
  EXPECT_EQ(t.args_size(), 1);
  EXPECT_EQ(t.args(0).type_id(), TFT_TENSOR);
  EXPECT_EQ(t.args(0).args_size(), 1);
  EXPECT_EQ(t.args(0).args(0).type_id(), TFT_VAR);
  EXPECT_EQ(t.args(0).args(0).args_size(), 0);
  EXPECT_EQ(t.args(0).args(0).s(), "T");
}

TEST(VariadicTensorContainer, Basic) {
  OpTypeConstructor ctor = VariadicTensorContainer(TFT_ARRAY, "T");

  OpDef op;
  op.add_output_arg();

  TF_ASSERT_OK(ctor(&op));

  const FullTypeDef& t = op.output_arg(0).experimental_full_type();
  EXPECT_EQ(t.type_id(), TFT_ARRAY);
  EXPECT_EQ(t.args_size(), 1);
  EXPECT_EQ(t.args(0).type_id(), TFT_FOR_EACH);
  EXPECT_EQ(t.args(0).args_size(), 3);
  EXPECT_EQ(t.args(0).args(0).type_id(), TFT_PRODUCT);
  EXPECT_EQ(t.args(0).args(0).args_size(), 0);
  EXPECT_EQ(t.args(0).args(1).type_id(), TFT_TENSOR);
  EXPECT_EQ(t.args(0).args(1).args_size(), 1);
  EXPECT_EQ(t.args(0).args(1).args(0).type_id(), TFT_VAR);
  EXPECT_EQ(t.args(0).args(1).args(0).args_size(), 0);
  EXPECT_EQ(t.args(0).args(1).args(0).s(), "T");
  EXPECT_EQ(t.args(0).args(2).type_id(), TFT_VAR);
  EXPECT_EQ(t.args(0).args(2).args_size(), 0);
  EXPECT_EQ(t.args(0).args(2).s(), "T");
}

TEST(SpecializeType, Fixed) {
  OpDef op;
  FullTypeDef* t = op.add_output_arg()->mutable_experimental_full_type();
  t->set_type_id(TFT_ARRAY);
  t->add_args()->set_type_id(TFT_TENSOR);
  t->mutable_args(0)->add_args()->set_type_id(TFT_INT32);
  t->add_args()->set_type_id(TFT_DATASET);
  t->mutable_args(1)->add_args()->set_type_id(TFT_FLOAT);

  AttrSlice empty;

  FullTypeDef ft;
  TF_ASSERT_OK(SpecializeType(empty, op, ft));

  EXPECT_EQ(ft.type_id(), TFT_PRODUCT);
  EXPECT_EQ(ft.args_size(), 1);

  const FullTypeDef& t_actual = ft.args(0);
  EXPECT_EQ(t_actual.type_id(), TFT_ARRAY);
  EXPECT_EQ(t_actual.args_size(), 2);
  EXPECT_EQ(t_actual.args(0).type_id(), TFT_TENSOR);
  EXPECT_EQ(t_actual.args(0).args_size(), 1);
  EXPECT_EQ(t_actual.args(0).args(0).type_id(), TFT_INT32);
  EXPECT_EQ(t_actual.args(0).args(0).args_size(), 0);
  EXPECT_EQ(t_actual.args(1).type_id(), TFT_DATASET);
  EXPECT_EQ(t_actual.args(1).args_size(), 1);
  EXPECT_EQ(t_actual.args(1).args(0).type_id(), TFT_FLOAT);
  EXPECT_EQ(t_actual.args(1).args(0).args_size(), 0);
}

TEST(SpecializeType, Idempotence) {
  OpDef op;
  FullTypeDef* t = op.add_output_arg()->mutable_experimental_full_type();
  t->set_type_id(TFT_ARRAY);
  t->add_args()->set_type_id(TFT_TENSOR);
  t->mutable_args(0)->add_args()->set_type_id(TFT_INT32);
  t->add_args()->set_type_id(TFT_DATASET);
  t->mutable_args(1)->add_args()->set_type_id(TFT_FLOAT);

  AttrSlice empty;

  FullTypeDef ft;
  TF_ASSERT_OK(SpecializeType(empty, op, ft));
  TF_ASSERT_OK(SpecializeType(empty, op, ft));

  EXPECT_EQ(ft.type_id(), TFT_PRODUCT);
  EXPECT_EQ(ft.args_size(), 1);
}

TEST(SpecializeType, VarExpandsFromSingleAttribute) {
  OpDef op;
  FullTypeDef* t = op.add_output_arg()->mutable_experimental_full_type();
  t->set_type_id(TFT_ARRAY);
  t->add_args()->set_type_id(TFT_TENSOR);
  t->mutable_args(0)->add_args()->set_type_id(TFT_VAR);
  t->mutable_args(0)->mutable_args(0)->set_s("T");

  AttrValue attr;
  attr.set_type(DT_INT32);
  NodeDef ndef;
  (*ndef.mutable_attr())["T"] = attr;

  AttrSlice attrs(ndef);

  FullTypeDef ft;
  TF_ASSERT_OK(SpecializeType(attrs, op, ft));

  EXPECT_EQ(ft.type_id(), TFT_PRODUCT);
  EXPECT_EQ(ft.args_size(), 1);

  const FullTypeDef& t_actual = ft.args(0);
  EXPECT_EQ(t_actual.type_id(), TFT_ARRAY);
  EXPECT_EQ(t_actual.args_size(), 1);
  EXPECT_EQ(t_actual.args(0).type_id(), TFT_TENSOR);
  EXPECT_EQ(t_actual.args(0).args_size(), 1);
  EXPECT_EQ(t_actual.args(0).args(0).type_id(), TFT_INT32);
  EXPECT_EQ(t_actual.args(0).args(0).args_size(), 0);
}

TEST(SpecializeType, VarExpandsFromDefaultForSingleAttribute) {
  OpDef op;
  FullTypeDef* t = op.add_output_arg()->mutable_experimental_full_type();
  t->set_type_id(TFT_ARRAY);
  t->add_args()->set_type_id(TFT_TENSOR);
  t->mutable_args(0)->add_args()->set_type_id(TFT_VAR);
  t->mutable_args(0)->mutable_args(0)->set_s("T");

  AttrValue attr;
  attr.set_type(DT_INT32);

  // Create a default for attribute "T"
  OpDef::AttrDef* attr_with_default = op.add_attr();
  attr_with_default->set_name("T");
  (*attr_with_default->mutable_default_value()) = attr;

  NodeDef ndef;
  // ndef does not specify the "T" attribute, so default value is used

  AttrSlice attrs(ndef);

  FullTypeDef ft;
  TF_ASSERT_OK(SpecializeType(attrs, op, ft));

  EXPECT_EQ(ft.type_id(), TFT_PRODUCT);
  EXPECT_EQ(ft.args_size(), 1);

  const FullTypeDef& t_actual = ft.args(0);
  EXPECT_EQ(t_actual.type_id(), TFT_ARRAY);
  EXPECT_EQ(t_actual.args_size(), 1);
  EXPECT_EQ(t_actual.args(0).type_id(), TFT_TENSOR);
  EXPECT_EQ(t_actual.args(0).args_size(), 1);
  EXPECT_EQ(t_actual.args(0).args(0).type_id(), TFT_INT32);
  EXPECT_EQ(t_actual.args(0).args(0).args_size(), 0);
}

TEST(SpecializeType, VarExpandsFromSingleElementTypeListAttribute) {
  OpDef op;
  FullTypeDef* t = op.add_output_arg()->mutable_experimental_full_type();
  t->set_type_id(TFT_ARRAY);
  t->add_args()->set_type_id(TFT_TENSOR);
  t->mutable_args(0)->add_args()->set_type_id(TFT_VAR);
  t->mutable_args(0)->mutable_args(0)->set_s("T");

  AttrValue attr;
  attr.mutable_list()->add_type(DT_INT32);
  NodeDef ndef;
  (*ndef.mutable_attr())["T"] = attr;

  AttrSlice attrs(ndef);

  FullTypeDef ft;
  TF_ASSERT_OK(SpecializeType(attrs, op, ft));

  EXPECT_EQ(ft.type_id(), TFT_PRODUCT);
  EXPECT_EQ(ft.args_size(), 1);

  const FullTypeDef& t_actual = ft.args(0);
  EXPECT_EQ(t_actual.type_id(), TFT_ARRAY);
  EXPECT_EQ(t_actual.args_size(), 1);
  EXPECT_EQ(t_actual.args(0).type_id(), TFT_TENSOR);
  EXPECT_EQ(t_actual.args(0).args_size(), 1);
  EXPECT_EQ(t_actual.args(0).args(0).type_id(), TFT_INT32);
  EXPECT_EQ(t_actual.args(0).args(0).args_size(), 0);
}

TEST(SpecializeType, VarRejectsMultipleElementTypeListAttribute) {
  OpDef op;
  FullTypeDef* t = op.add_output_arg()->mutable_experimental_full_type();
  t->set_type_id(TFT_ARRAY);
  t->add_args()->set_type_id(TFT_TENSOR);
  t->mutable_args(0)->add_args()->set_type_id(TFT_VAR);
  t->mutable_args(0)->mutable_args(0)->set_s("T");

  AttrValue attr;
  attr.mutable_list()->add_type(DT_INT32);
  attr.mutable_list()->add_type(DT_FLOAT);
  NodeDef ndef;
  (*ndef.mutable_attr())["T"] = attr;

  AttrSlice attrs(ndef);

  FullTypeDef ft;
  EXPECT_FALSE(SpecializeType(attrs, op, ft).ok());
}

TEST(SpecializeType, VarRejectsEmptyTypeListAttribute) {
  OpDef op;
  FullTypeDef* t = op.add_output_arg()->mutable_experimental_full_type();
  t->set_type_id(TFT_ARRAY);
  t->add_args()->set_type_id(TFT_TENSOR);
  t->mutable_args(0)->add_args()->set_type_id(TFT_VAR);
  t->mutable_args(0)->mutable_args(0)->set_s("T");

  AttrValue attr;
  attr.mutable_list();
  NodeDef ndef;
  (*ndef.mutable_attr())["T"] = attr;

  AttrSlice attrs(ndef);

  FullTypeDef ft;
  EXPECT_FALSE(SpecializeType(attrs, op, ft).ok());
}

TEST(SpecializeType, ForEachExpandsFromSingleAttribute) {
  OpDef op;
  FullTypeDef* t = op.add_output_arg()->mutable_experimental_full_type();
  t->set_type_id(TFT_FOR_EACH);
  t->add_args()->set_type_id(TFT_PRODUCT);
  t->add_args()->set_type_id(TFT_TENSOR);
  t->mutable_args(1)->add_args()->set_type_id(TFT_VAR);
  t->mutable_args(1)->mutable_args(0)->set_s("T");
  t->add_args()->set_type_id(TFT_VAR);
  t->mutable_args(2)->set_s("T");

  AttrValue attr;
  attr.set_type(DT_INT32);
  NodeDef ndef;
  (*ndef.mutable_attr())["T"] = attr;

  AttrSlice attrs(ndef);

  FullTypeDef ft;
  TF_ASSERT_OK(SpecializeType(attrs, op, ft));

  EXPECT_EQ(ft.type_id(), TFT_PRODUCT);
  EXPECT_EQ(ft.args_size(), 1);

  const FullTypeDef& t_actual = ft.args(0);
  EXPECT_EQ(t_actual.type_id(), TFT_PRODUCT);
  EXPECT_EQ(t_actual.args_size(), 1);
  EXPECT_EQ(t_actual.args(0).type_id(), TFT_TENSOR);
  EXPECT_EQ(t_actual.args(0).args_size(), 1);
  EXPECT_EQ(t_actual.args(0).args(0).type_id(), TFT_INT32);
  EXPECT_EQ(t_actual.args(0).args(0).args_size(), 0);
}

TEST(SpecializeType, ForEachExpandsFromListAttribute) {
  OpDef op;
  FullTypeDef* t = op.add_output_arg()->mutable_experimental_full_type();
  t->set_type_id(TFT_FOR_EACH);
  t->add_args()->set_type_id(TFT_PRODUCT);
  t->add_args()->set_type_id(TFT_TENSOR);
  t->mutable_args(1)->add_args()->set_type_id(TFT_VAR);
  t->mutable_args(1)->mutable_args(0)->set_s("T");
  t->add_args()->set_type_id(TFT_VAR);
  t->mutable_args(2)->set_s("T");

  AttrValue attr;
  attr.mutable_list()->add_type(DT_INT32);
  attr.mutable_list()->add_type(DT_FLOAT);
  NodeDef ndef;
  (*ndef.mutable_attr())["T"] = attr;

  AttrSlice attrs(ndef);

  FullTypeDef ft;
  TF_ASSERT_OK(SpecializeType(attrs, op, ft));

  EXPECT_EQ(ft.type_id(), TFT_PRODUCT);
  EXPECT_EQ(ft.args_size(), 1);

  const FullTypeDef& t_actual = ft.args(0);
  EXPECT_EQ(t_actual.type_id(), TFT_PRODUCT);
  EXPECT_EQ(t_actual.args_size(), 2);
  EXPECT_EQ(t_actual.args(0).type_id(), TFT_TENSOR);
  EXPECT_EQ(t_actual.args(0).args_size(), 1);
  EXPECT_EQ(t_actual.args(0).args(0).type_id(), TFT_INT32);
  EXPECT_EQ(t_actual.args(0).args(0).args_size(), 0);
  EXPECT_EQ(t_actual.args(1).type_id(), TFT_TENSOR);
  EXPECT_EQ(t_actual.args(1).args_size(), 1);
  EXPECT_EQ(t_actual.args(1).args(0).type_id(), TFT_FLOAT);
  EXPECT_EQ(t_actual.args(1).args(0).args_size(), 0);
}

TEST(SpecializeType, ForEachDistributesNestedVar) {
  OpDef op;
  FullTypeDef* t = op.add_output_arg()->mutable_experimental_full_type();
  t->set_type_id(TFT_FOR_EACH);
  t->add_args()->set_type_id(TFT_PRODUCT);
  t->add_args()->set_type_id(TFT_TENSOR);
  t->mutable_args(1)->add_args()->set_type_id(TFT_VAR);
  t->mutable_args(1)->mutable_args(0)->set_s("ForEachTarget");
  t->mutable_args(1)->add_args()->set_type_id(TFT_VAR);
  t->mutable_args(1)->mutable_args(1)->set_s("GlobalVar");
  t->add_args()->set_type_id(TFT_VAR);
  t->mutable_args(2)->set_s("ForEachTarget");

  NodeDef ndef;
  AttrValue attr;

  attr.mutable_list()->add_type(DT_INT32);
  attr.mutable_list()->add_type(DT_INT64);
  (*ndef.mutable_attr())["ForEachTarget"] = attr;

  attr.set_type(DT_FLOAT);
  (*ndef.mutable_attr())["GlobalVar"] = attr;

  AttrSlice attrs(ndef);

  FullTypeDef ft;
  TF_ASSERT_OK(SpecializeType(attrs, op, ft));

  EXPECT_EQ(ft.type_id(), TFT_PRODUCT);
  EXPECT_EQ(ft.args_size(), 1);

  const FullTypeDef& t_actual = ft.args(0);
  EXPECT_EQ(t_actual.type_id(), TFT_PRODUCT);
  EXPECT_EQ(t_actual.args_size(), 2);
  EXPECT_EQ(t_actual.args(0).type_id(), TFT_TENSOR);
  EXPECT_EQ(t_actual.args(0).args_size(), 2);
  EXPECT_EQ(t_actual.args(0).args(0).type_id(), TFT_INT32);
  EXPECT_EQ(t_actual.args(0).args(0).args_size(), 0);
  EXPECT_EQ(t_actual.args(0).args(1).type_id(), TFT_FLOAT);
  EXPECT_EQ(t_actual.args(0).args(1).args_size(), 0);
  EXPECT_EQ(t_actual.args(1).type_id(), TFT_TENSOR);
  EXPECT_EQ(t_actual.args(1).args_size(), 2);
  EXPECT_EQ(t_actual.args(1).args(0).type_id(), TFT_INT64);
  EXPECT_EQ(t_actual.args(1).args(0).args_size(), 0);
  EXPECT_EQ(t_actual.args(1).args(1).type_id(), TFT_FLOAT);
  EXPECT_EQ(t_actual.args(1).args(1).args_size(), 0);
}

TEST(SpecializeType, ForEachDistributesNestedForEach) {
  OpDef op;
  FullTypeDef* t = op.add_output_arg()->mutable_experimental_full_type();
  t->set_type_id(TFT_FOR_EACH);
  t->add_args()->set_type_id(TFT_PRODUCT);

  FullTypeDef* inner = t->add_args();
  inner->set_type_id(TFT_FOR_EACH);
  inner->add_args()->set_type_id(TFT_PRODUCT);
  inner->add_args()->set_type_id(TFT_ARRAY);
  inner->mutable_args(1)->add_args()->set_type_id(TFT_VAR);
  inner->mutable_args(1)->mutable_args(0)->set_s("InnerForEach");
  inner->mutable_args(1)->add_args()->set_type_id(TFT_VAR);
  inner->mutable_args(1)->mutable_args(1)->set_s("OuterForEach");
  inner->add_args()->set_type_id(TFT_VAR);
  inner->mutable_args(2)->set_s("InnerForEach");

  t->add_args()->set_type_id(TFT_VAR);
  t->mutable_args(2)->set_s("OuterForEach");

  NodeDef ndef;
  AttrValue attr;

  attr.mutable_list()->add_type(DT_INT32);
  attr.mutable_list()->add_type(DT_INT64);
  (*ndef.mutable_attr())["OuterForEach"] = attr;

  attr.set_type(DT_FLOAT);
  (*ndef.mutable_attr())["InnerForEach"] = attr;

  AttrSlice attrs(ndef);

  FullTypeDef ft;
  TF_ASSERT_OK(SpecializeType(attrs, op, ft));

  EXPECT_EQ(ft.type_id(), TFT_PRODUCT);
  EXPECT_EQ(ft.args_size(), 1);

  const FullTypeDef& t_actual = ft.args(0);
  EXPECT_EQ(t_actual.type_id(), TFT_PRODUCT);
  EXPECT_EQ(t_actual.args_size(), 2);
  EXPECT_EQ(t_actual.args(0).type_id(), TFT_PRODUCT);
  EXPECT_EQ(t_actual.args(0).args_size(), 1);
  EXPECT_EQ(t_actual.args(0).args(0).type_id(), TFT_ARRAY);
  EXPECT_EQ(t_actual.args(0).args(0).args_size(), 2);
  EXPECT_EQ(t_actual.args(0).args(0).args(0).type_id(), TFT_FLOAT);
  EXPECT_EQ(t_actual.args(0).args(0).args(0).args_size(), 0);
  EXPECT_EQ(t_actual.args(0).args(0).args(1).type_id(), TFT_INT32);
  EXPECT_EQ(t_actual.args(0).args(0).args(1).args_size(), 0);
  EXPECT_EQ(t_actual.args(1).type_id(), TFT_PRODUCT);
  EXPECT_EQ(t_actual.args(1).args_size(), 1);
  EXPECT_EQ(t_actual.args(1).args(0).type_id(), TFT_ARRAY);
  EXPECT_EQ(t_actual.args(1).args(0).args_size(), 2);
  EXPECT_EQ(t_actual.args(1).args(0).args(0).type_id(), TFT_FLOAT);
  EXPECT_EQ(t_actual.args(1).args(0).args(0).args_size(), 0);
  EXPECT_EQ(t_actual.args(1).args(0).args(1).type_id(), TFT_INT64);
  EXPECT_EQ(t_actual.args(1).args(0).args(1).args_size(), 0);
}

TEST(SpecializeType, ForEachOverridesTargetOfNestedForEach) {
  OpDef op;
  FullTypeDef* t = op.add_output_arg()->mutable_experimental_full_type();
  t->set_type_id(TFT_FOR_EACH);
  t->add_args()->set_type_id(TFT_PRODUCT);

  FullTypeDef* inner = t->add_args();
  inner->set_type_id(TFT_FOR_EACH);
  inner->add_args()->set_type_id(TFT_PRODUCT);
  inner->add_args()->set_type_id(TFT_ARRAY);
  inner->mutable_args(1)->add_args()->set_type_id(TFT_VAR);
  inner->mutable_args(1)->mutable_args(0)->set_s("T");
  inner->add_args()->set_type_id(TFT_VAR);
  inner->mutable_args(2)->set_s("T");

  t->add_args()->set_type_id(TFT_VAR);
  t->mutable_args(2)->set_s("T");

  NodeDef ndef;
  AttrValue attr;

  attr.mutable_list()->add_type(DT_FLOAT);
  attr.mutable_list()->add_type(DT_DOUBLE);
  (*ndef.mutable_attr())["T"] = attr;

  AttrSlice attrs(ndef);

  FullTypeDef ft;
  TF_ASSERT_OK(SpecializeType(attrs, op, ft));

  EXPECT_EQ(ft.type_id(), TFT_PRODUCT);
  EXPECT_EQ(ft.args_size(), 1);

  const FullTypeDef& t_actual = ft.args(0);
  EXPECT_EQ(t_actual.type_id(), TFT_PRODUCT);
  EXPECT_EQ(t_actual.args_size(), 2);
  EXPECT_EQ(t_actual.args(0).type_id(), TFT_PRODUCT);
  EXPECT_EQ(t_actual.args(0).args_size(), 1);
  EXPECT_EQ(t_actual.args(0).args(0).type_id(), TFT_ARRAY);
  EXPECT_EQ(t_actual.args(0).args(0).args_size(), 1);
  EXPECT_EQ(t_actual.args(0).args(0).args(0).type_id(), TFT_FLOAT);
  EXPECT_EQ(t_actual.args(0).args(0).args(0).args_size(), 0);
  EXPECT_EQ(t_actual.args(1).type_id(), TFT_PRODUCT);
  EXPECT_EQ(t_actual.args(1).args_size(), 1);
  EXPECT_EQ(t_actual.args(1).args(0).type_id(), TFT_ARRAY);
  EXPECT_EQ(t_actual.args(1).args(0).args_size(), 1);
  EXPECT_EQ(t_actual.args(1).args(0).args(0).type_id(), TFT_DOUBLE);
  EXPECT_EQ(t_actual.args(1).args(0).args(0).args_size(), 0);
}

TEST(SpecializeType, ForEachRejectsMalformedInput) {
  OpDef op;
  FullTypeDef* t = op.add_output_arg()->mutable_experimental_full_type();
  t->set_type_id(TFT_FOR_EACH);
  t->add_args()->set_type_id(TFT_PRODUCT);

  NodeDef ndef;
  AttrSlice attrs(ndef);

  FullTypeDef ft;
  EXPECT_FALSE(SpecializeType(attrs, op, ft).ok());
}

TEST(SpecializeType, RemovesLegacyVariant) {
  OpDef op;
  FullTypeDef* t = op.add_output_arg()->mutable_experimental_full_type();
  t->set_type_id(TFT_ARRAY);
  t->add_args()->set_type_id(TFT_TENSOR);
  t->mutable_args(0)->add_args()->set_type_id(TFT_LEGACY_VARIANT);
  t->add_args()->set_type_id(TFT_TENSOR);
  t->mutable_args(1)->add_args()->set_type_id(TFT_FLOAT);

  AttrSlice empty;

  FullTypeDef ft;
  TF_ASSERT_OK(SpecializeType(empty, op, ft));

  EXPECT_EQ(ft.type_id(), TFT_PRODUCT);
  EXPECT_EQ(ft.args_size(), 1);

  const FullTypeDef& t_actual = ft.args(0);
  EXPECT_EQ(t_actual.type_id(), TFT_ARRAY);
  EXPECT_EQ(t_actual.args_size(), 0);
}

TEST(SpecializeType, RemovesLegacyVariantAfterExpansion) {
  OpDef op;
  FullTypeDef* t = op.add_output_arg()->mutable_experimental_full_type();
  t->set_type_id(TFT_ARRAY);
  t->add_args()->set_type_id(TFT_TENSOR);
  t->mutable_args(0)->add_args()->set_type_id(TFT_VAR);
  t->mutable_args(0)->mutable_args(0)->set_s("T");

  AttrValue attr;
  attr.set_type(DT_VARIANT);
  NodeDef ndef;
  (*ndef.mutable_attr())["T"] = attr;

  AttrSlice attrs(ndef);

  FullTypeDef ft;
  TF_ASSERT_OK(SpecializeType(attrs, op, ft));

  EXPECT_EQ(ft.type_id(), TFT_PRODUCT);
  EXPECT_EQ(ft.args_size(), 1);

  const FullTypeDef& t_actual = ft.args(0);
  EXPECT_EQ(t_actual.type_id(), TFT_ARRAY);
  EXPECT_EQ(t_actual.args_size(), 0);
}

TEST(GetArgDefaults, DefaultUnsetFromNoArgs) {
  FullTypeDef t;

  const auto& d = GetArgDefaultUnset(t, 0);

  EXPECT_EQ(d.type_id(), TFT_UNSET);
}

TEST(GetArgDefaults, DefaultUnsetFromOutOfBounds) {
  FullTypeDef t;
  t.add_args()->set_type_id(TFT_TENSOR);

  const auto& d = GetArgDefaultUnset(t, 1);

  EXPECT_EQ(d.type_id(), TFT_UNSET);
}

TEST(GetArgDefaults, NoDefaultUnsetFromArg) {
  FullTypeDef t;
  t.add_args()->set_type_id(TFT_TENSOR);
  t.mutable_args(0)->add_args();

  const auto& d = GetArgDefaultUnset(t, 0);

  EXPECT_EQ(d.type_id(), TFT_TENSOR);
  EXPECT_EQ(d.args_size(), 1);
}

TEST(GetArgDefaults, DefaultAnyFromNoArgs) {
  FullTypeDef t;

  const auto& d = GetArgDefaultAny(t, 0);

  EXPECT_EQ(d.type_id(), TFT_ANY);
}

TEST(GetArgDefaults, DefaultAnyFromOutOfBounds) {
  FullTypeDef t;
  t.add_args()->set_type_id(TFT_TENSOR);

  const auto& d = GetArgDefaultAny(t, 1);

  EXPECT_EQ(d.type_id(), TFT_ANY);
}

TEST(GetArgDefaults, DefaultAnyFromUnset) {
  FullTypeDef t;
  t.add_args();

  const auto& d = GetArgDefaultAny(t, 0);

  EXPECT_EQ(d.type_id(), TFT_ANY);
}

TEST(GetArgDefaults, NoDefaultAnyFromArg) {
  FullTypeDef t;
  t.add_args()->set_type_id(TFT_TENSOR);
  t.mutable_args(0)->add_args();

  const auto& d = GetArgDefaultAny(t, 0);

  EXPECT_EQ(d.type_id(), TFT_TENSOR);
  EXPECT_EQ(d.args_size(), 1);
}

TEST(IsEqual, Reflexivity) {
  FullTypeDef t;
  t.set_type_id(TFT_TENSOR);
  t.add_args()->set_type_id(TFT_INT32);
  t.add_args()->set_type_id(TFT_INT64);

  EXPECT_TRUE(IsEqual(t, t));
}

TEST(IsEqual, Copy) {
  FullTypeDef t;
  t.set_type_id(TFT_TENSOR);
  t.add_args()->set_type_id(TFT_INT32);
  t.add_args()->set_type_id(TFT_INT64);

  FullTypeDef u;
  u = t;
  EXPECT_TRUE(IsEqual(t, u));
  EXPECT_TRUE(IsEqual(u, t));
}

TEST(IsEqual, DifferentTypesNotEqual) {
  FullTypeDef t;
  t.set_type_id(TFT_TENSOR);
  t.add_args()->set_type_id(TFT_INT32);
  t.add_args()->set_type_id(TFT_INT64);

  FullTypeDef u;
  u = t;
  u.set_type_id(TFT_ARRAY);

  EXPECT_FALSE(IsEqual(t, u));
  EXPECT_FALSE(IsEqual(u, t));
}

TEST(IsEqual, DifferentAritiesNotEqual) {
  FullTypeDef t;
  t.set_type_id(TFT_TENSOR);
  t.add_args()->set_type_id(TFT_INT32);
  t.add_args()->set_type_id(TFT_INT64);

  FullTypeDef u;
  u = t;
  u.add_args()->set_type_id(TFT_FLOAT);

  EXPECT_FALSE(IsEqual(t, u));
  EXPECT_FALSE(IsEqual(u, t));
}

TEST(IsEqual, MissingArgsEquivalentToAny) {
  FullTypeDef t;
  t.set_type_id(TFT_TENSOR);
  t.add_args()->set_type_id(TFT_INT32);

  FullTypeDef u;
  u = t;
  u.add_args()->set_type_id(TFT_ANY);

  EXPECT_TRUE(IsEqual(t, u));
  EXPECT_TRUE(IsEqual(u, t));
}

TEST(IsEqual, DifferentArgsNotEqual) {
  FullTypeDef t;
  t.set_type_id(TFT_TENSOR);
  t.add_args()->set_type_id(TFT_INT32);
  t.add_args()->set_type_id(TFT_INT64);

  FullTypeDef u;
  u = t;
  u.mutable_args(1)->set_type_id(TFT_FLOAT);

  EXPECT_FALSE(IsEqual(t, u));
  EXPECT_FALSE(IsEqual(u, t));
}

TEST(IsEqual, DifferentStringValuesNotEqual) {
  FullTypeDef t;
  t.set_type_id(TFT_VAR);
  t.set_s("T");

  FullTypeDef u;
  u = t;
  u.set_type_id(TFT_VAR);
  u.set_s("U");

  EXPECT_FALSE(IsEqual(t, u));
  EXPECT_FALSE(IsEqual(u, t));
}

TEST(IsSubtype, Reflexivity) {
  FullTypeDef t;
  t.set_type_id(TFT_TENSOR);
  t.add_args()->set_type_id(TFT_INT32);
  t.add_args()->set_type_id(TFT_INT64);

  EXPECT_TRUE(IsSubtype(t, t));
}

TEST(IsSubtype, Copy) {
  FullTypeDef t;
  t.set_type_id(TFT_TENSOR);
  t.add_args()->set_type_id(TFT_INT32);
  t.add_args()->set_type_id(TFT_INT64);

  FullTypeDef u;
  u = t;
  EXPECT_TRUE(IsSubtype(t, u));
}

TEST(IsSubtype, Any) {
  FullTypeDef t;
  t.set_type_id(TFT_TENSOR);
  t.add_args()->set_type_id(TFT_INT32);
  t.add_args()->set_type_id(TFT_INT64);

  FullTypeDef u;
  u.set_type_id(TFT_ANY);

  EXPECT_TRUE(IsSubtype(t, u));
  EXPECT_FALSE(IsSubtype(u, t));
}

TEST(IsSubtype, Unset) {
  FullTypeDef t;
  t.set_type_id(TFT_TENSOR);
  t.add_args()->set_type_id(TFT_INT32);
  t.add_args()->set_type_id(TFT_INT64);

  FullTypeDef u;
  u.set_type_id(TFT_UNSET);

  EXPECT_TRUE(IsSubtype(t, u));
  EXPECT_FALSE(IsSubtype(u, t));
}

TEST(IsSubtype, Covariance) {
  FullTypeDef t;
  t.set_type_id(TFT_TENSOR);
  t.add_args()->set_type_id(TFT_ARRAY);
  t.mutable_args(0)->add_args()->set_type_id(TFT_INT32);

  FullTypeDef u;
  u.set_type_id(TFT_TENSOR);
  u.add_args()->set_type_id(TFT_ANY);

  EXPECT_TRUE(IsSubtype(t, u, /*covariant=*/true));
  EXPECT_FALSE(IsSubtype(u, t, /*covariant=*/true));

  EXPECT_FALSE(IsSubtype(t, u, /*covariant=*/false));
  EXPECT_TRUE(IsSubtype(u, t, /*covariant=*/false));
}

TEST(IsSubtype, DifferentTypesNotSubtype) {
  FullTypeDef t;
  t.set_type_id(TFT_TENSOR);
  t.add_args()->set_type_id(TFT_INT32);
  t.add_args()->set_type_id(TFT_INT64);

  FullTypeDef u;
  u = t;
  u.set_type_id(TFT_ARRAY);

  EXPECT_FALSE(IsSubtype(t, u));
  EXPECT_FALSE(IsSubtype(u, t));
}

TEST(IsSubtype, DifferentAritiesDefaultToAny) {
  FullTypeDef t;
  t.set_type_id(TFT_TENSOR);
  t.add_args()->set_type_id(TFT_INT32);
  t.add_args()->set_type_id(TFT_INT64);

  FullTypeDef u;
  u = t;
  u.add_args()->set_type_id(TFT_FLOAT);

  EXPECT_FALSE(IsSubtype(t, u));
  EXPECT_TRUE(IsSubtype(u, t));
}

TEST(IsSubtype, DifferentArgsNotSubtype) {
  FullTypeDef t;
  t.set_type_id(TFT_TENSOR);
  t.add_args()->set_type_id(TFT_INT32);
  t.add_args()->set_type_id(TFT_INT64);

  FullTypeDef u;
  u = t;
  u.mutable_args(1)->set_type_id(TFT_FLOAT);

  EXPECT_FALSE(IsSubtype(t, u));
  EXPECT_FALSE(IsSubtype(u, t));
}

}  // namespace

}  // namespace full_type

}  // namespace tensorflow
