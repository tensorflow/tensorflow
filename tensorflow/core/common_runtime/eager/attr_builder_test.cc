/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/eager/attr_builder.h"

#include <memory>
#include <vector>

#include "tensorflow/c/c_api.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace {

TEST(AttrTypeMap, Lookup) {
  const AttrTypeMap* m = nullptr;
  // Unknown ops are assumed to be functions.
  // Their maps are filled with default attributes.
  bool is_function = false;
  Status s = AttrTypeMapForOp("SomeFunctionName", &m, &is_function);
  EXPECT_TRUE(s.ok());
  EXPECT_TRUE(is_function);
  ASSERT_NE(m->end(), m->find("executor_type"));
  EXPECT_EQ(TF_ATTR_STRING, m->find("executor_type")->second);
  ASSERT_NE(m->end(), m->find("config_proto"));
  EXPECT_EQ(TF_ATTR_STRING, m->find("config_proto")->second);

  is_function = true;
  s = AttrTypeMapForOp("MatMul", &m, &is_function);
  EXPECT_FALSE(is_function);
  ASSERT_TRUE(s.ok()) << s;

  TF_AttrType t;
  unsigned char is_list = 1;
  s = AttrTypeByName(*m, "ThisAttribyteCannotPossiblyExist", &t, &is_list);
  EXPECT_FALSE(s.ok());
  EXPECT_NE(is_list, 0);
  s = AttrTypeByName(*m, "transpose_a", &t, &is_list);
  ASSERT_TRUE(s.ok()) << s;
  EXPECT_EQ(TF_ATTR_BOOL, t);
  EXPECT_EQ(is_list, 0);

  s = AttrTypeMapForOp("Squeeze", &m, &is_function);
  ASSERT_TRUE(s.ok()) << s;
  s = AttrTypeByName(*m, "squeeze_dims", &t, &is_list);
  ASSERT_TRUE(s.ok()) << s;
  EXPECT_EQ(TF_ATTR_INT, t);
  EXPECT_NE(is_list, 0);
}

TEST(AttrTypeMap, CacheKey) {
  AttrBuilder a("op_name");
  a.NumInputs(2);
  a.Set("T", TF_FLOAT);
  tensorflow::Fprint128 cache_key = a.CacheKey("cpu:0");

  ASSERT_FALSE(cache_key == a.CacheKey("cpu:1"));
  ASSERT_TRUE(cache_key == a.CacheKey("cpu:0"));

  a.Set("x", 1.0);
  ASSERT_FALSE(cache_key == a.CacheKey("cpu:0"));
}

string ToString(const AttrValueMap& m) {
  std::vector<string> strs;
  for (const auto& e : m) {
    strs.push_back(absl::StrCat(e.first, " -> ", e.second.DebugString()));
  }
  return absl::StrJoin(strs, "\n");
}

TEST(AttrBuilder, FillAttrValueMapWithoutDefaults_MatMul) {
  AttrBuilder a("MatMul");
  a.Set("transpose_a", true);
  a.Set("transpose_b", false);

  AttrValueMap m;
  a.FillAttrValueMapWithoutDefaults(&m);
  // Only non-default value must end up in the map
  ASSERT_EQ(1, m.size()) << ToString(m);
  ASSERT_EQ(true, m["transpose_a"].b()) << ToString(m);
}

TEST(AttrBuilder, FillAttrValueMapWithoutDefaults_UnknownOp) {
  AttrBuilder a("SomeUnknownOp");
  a.Set("transpose_a", true);
  a.Set("transpose_b", false);

  AttrValueMap m;
  a.FillAttrValueMapWithoutDefaults(&m);
  // Only non-default value must end up in the map
  ASSERT_EQ(2, m.size()) << ToString(m);
  ASSERT_EQ(true, m["transpose_a"].b()) << ToString(m);
  ASSERT_EQ(false, m["transpose_b"].b()) << ToString(m);
}

TEST(AttrBuilder, GetTypeAndNumber) {
  AttrBuilder a("Concat");
  a.Set("T", DT_FLOAT);
  a.Set("N", 2);
  DataType type;
  ASSERT_TRUE(a.GetType("T", &type));
  ASSERT_EQ(DT_FLOAT, type);
  int64_t num;
  ASSERT_TRUE(a.GetInt("N", &num));
  ASSERT_EQ(2, num);
}

TEST(AttrBuilder, GetTypeList) {
  AttrBuilder a("IdentityN");
  a.Set("T", gtl::ArraySlice<DataType>({DT_FLOAT, DT_INT64}));
  absl::InlinedVector<DataType, 4> type_list;
  Status s = a.GetTypeList("T", &type_list);
  ASSERT_TRUE(s.ok()) << s;
  ASSERT_EQ(2, type_list.size()) << type_list.size();
  ASSERT_EQ(DT_FLOAT, type_list[0]) << type_list[0];
  ASSERT_EQ(DT_INT64, type_list[1]) << type_list[1];
}

}  // namespace
}  // namespace tensorflow
