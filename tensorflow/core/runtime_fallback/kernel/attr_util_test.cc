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
#include "tensorflow/core/runtime_fallback/kernel/attr_util.h"

#include <vector>

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"
#include "tfrt/core_runtime/op_attr_type.h"  // from @tf_runtime
#include "tfrt/core_runtime/op_attrs.h"  // from @tf_runtime
#include "tfrt/host_context/kernel_utils.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime

using llvm::ArrayRef;
using tfrt::OpAttrs;
using tfrt::OpAttrType;

namespace tensorflow {
namespace {

TEST(AttrUtilTest, TestGetBoolAttr) {
  OpAttrs opattrs;
  TF_ASSERT_OK(AddOpAttr("foo", "bool$true", &opattrs));
  TF_ASSERT_OK(AddOpAttr("bar", "bool$false", &opattrs));

  ASSERT_TRUE(opattrs.GetAsserting<bool>("foo"));
  ASSERT_FALSE(opattrs.GetAsserting<bool>("bar"));
}

TEST(AttrUtilTest, TestGetIntAttr) {
  OpAttrs opattrs;
  TF_ASSERT_OK(AddOpAttr("foo", "i32$-2", &opattrs));
  TF_ASSERT_OK(AddOpAttr("bar", "i32$0", &opattrs));
  TF_ASSERT_OK(AddOpAttr("baz", "i32$123", &opattrs));

  ASSERT_EQ(opattrs.GetAsserting<int32>("foo"), -2);
  ASSERT_EQ(opattrs.GetAsserting<int32>("bar"), 0);
  ASSERT_EQ(opattrs.GetAsserting<int32>("baz"), 123);

  Status s = AddOpAttr("invalid", "i32$4.5", &opattrs);
  ASSERT_FALSE(s.ok());
}

TEST(AttrUtilTest, TestGetDTypeAttr) {
  OpAttrs opattrs;
  TF_ASSERT_OK(AddOpAttr("foo", "tfdtype$DT_INT32", &opattrs));
  TF_ASSERT_OK(AddOpAttr("bar", "tfdtype$DT_FLOAT", &opattrs));

  ASSERT_EQ(opattrs.GetAsserting<OpAttrType>("foo"), OpAttrType::I32);
  ASSERT_EQ(opattrs.GetAsserting<OpAttrType>("bar"), OpAttrType::F32);
}

TEST(AttrUtilTest, TestGetIntListAttr) {
  OpAttrs opattrs;
  TF_ASSERT_OK(AddOpAttr("foo", "list(i32)$", &opattrs));
  TF_ASSERT_OK(AddOpAttr("bar", "list(i32)$1", &opattrs));
  TF_ASSERT_OK(AddOpAttr("baz", "list(i32)$1,2,3", &opattrs));

  // std::vector<int32> v1, v2, v3;
  ArrayRef<int32> v1, v2, v3;
  std::vector<int32> expected_v1;
  std::vector<int32> expected_v2 = {1};
  std::vector<int32> expected_v3 = {1, 2, 3};
  ArrayRef<int32> expected_v1_ref(expected_v1);
  ArrayRef<int32> expected_v2_ref(expected_v2);
  ArrayRef<int32> expected_v3_ref(expected_v3);

  ASSERT_TRUE(opattrs.GetArray<int32>("foo", &v1));
  ASSERT_TRUE(opattrs.GetArray<int32>("bar", &v2));
  ASSERT_TRUE(opattrs.GetArray<int32>("baz", &v3));
  ASSERT_EQ(v1, expected_v1_ref);
  ASSERT_EQ(v2, expected_v2_ref);
  ASSERT_EQ(v3, expected_v3_ref);
}

TEST(AttrUtilTest, TestGetStrAttr) {
  OpAttrs opattrs;
  TF_ASSERT_OK(AddOpAttr("foo", "string$", &opattrs));
  TF_ASSERT_OK(AddOpAttr("bar", "string$test", &opattrs));

  ASSERT_EQ(opattrs.GetStringAsserting("foo"), "");
  ASSERT_EQ(opattrs.GetStringAsserting("bar"), "test");
}

TEST(AttrUtilTest, TestGetPaddingAttr) {
  OpAttrs opattrs;
  TF_ASSERT_OK(AddOpAttr("foo", "padding$VALID", &opattrs));
  TF_ASSERT_OK(AddOpAttr("bar", "padding$SAME", &opattrs));

  ASSERT_EQ(opattrs.GetStringAsserting("foo"), "VALID");
  ASSERT_EQ(opattrs.GetStringAsserting("bar"), "SAME");
}
}  // namespace
}  // namespace tensorflow
