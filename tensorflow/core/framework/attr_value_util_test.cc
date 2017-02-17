/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/attr_value_util.h"

#include <vector>
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

// A few helpers to construct AttrValue protos.
template <typename T>
AttrValue V(T value) {
  AttrValue ret;
  SetAttrValue(value, &ret);
  return ret;
}

AttrValue P(const string& p) {
  AttrValue ret;
  ret.set_placeholder(p);
  return ret;
}

AttrValue F(const string& name,
            std::vector<std::pair<string, AttrValue>> pairs) {
  AttrValue ret;
  ret.mutable_func()->set_name(name);
  ret.mutable_func()->mutable_attr()->insert(pairs.begin(), pairs.end());
  return ret;
}

AttrValue Fs(
    std::vector<std::pair<string, std::vector<std::pair<string, AttrValue>>>>
        funcs) {
  AttrValue ret;
  for (const auto& func : funcs) {
    NameAttrList* entry = ret.mutable_list()->add_func();
    entry->set_name(func.first);
    entry->mutable_attr()->insert(func.second.begin(), func.second.end());
  }
  return ret;
}

TEST(AttrValueUtil, HasType) {
  // OK
  EXPECT_TRUE(AttrValueHasType(V(123), "int").ok());
  EXPECT_TRUE(AttrValueHasType(V(1.2), "float").ok());
  EXPECT_TRUE(AttrValueHasType(V(DT_FLOAT), "type").ok());
  EXPECT_TRUE(AttrValueHasType(F("f", {}), "func").ok());
  EXPECT_TRUE(AttrValueHasType(Fs({{"f", {}}, {"g", {}}}), "list(func)").ok());

  // not OK.
  EXPECT_FALSE(AttrValueHasType(V(123), "func").ok());
  EXPECT_FALSE(AttrValueHasType(V(1.2), "int").ok());
  EXPECT_FALSE(AttrValueHasType(V(DT_FLOAT), "shape").ok());
  EXPECT_FALSE(AttrValueHasType(F("f", {}), "string").ok());
  EXPECT_FALSE(AttrValueHasType(P("T"), "float").ok());
  EXPECT_FALSE(AttrValueHasType(V(static_cast<DataType>(1000)), "type").ok());
  std::vector<DataType> list_type({static_cast<DataType>(1000)});
  EXPECT_FALSE(AttrValueHasType(V(list_type), "list(type)").ok());
}

SubstituteFunc ReplaceTWith(const AttrValue& val) {
  return [val](const string& placeholder, AttrValue* target) {
    if (placeholder == "T") {
      *target = val;
      return true;
    } else {
      return false;
    }
  };
}

TEST(AttrValueUtil, Basic) {
  auto v = F("MatMul", {{"dtype", P("T")},
                        {"transpose_a", V(false)},
                        {"transpose_b", V(true)},
                        {"use_cublas", V(true)}});
  TF_EXPECT_OK(AttrValueHasType(v, "func"));
  EXPECT_TRUE(HasPlaceHolder(v));

  EXPECT_EQ(
      SummarizeAttrValue(v),
      "MatMul[dtype=$T, transpose_a=false, transpose_b=true, use_cublas=true]");

  SubstitutePlaceholders(ReplaceTWith(V(DT_FLOAT)), &v);
  EXPECT_TRUE(!HasPlaceHolder(v));
  EXPECT_EQ(SummarizeAttrValue(v),
            "MatMul[dtype=DT_FLOAT, transpose_a=false, transpose_b=true, "
            "use_cublas=true]");
}

TEST(AttrValueUtil, Shaped) {
  auto v =
      F("OpRequiresShape", {{"shape_full", V(TensorShape({1, 0}))},
                            {"shape_part", V(PartialTensorShape({-1, 1, 0}))}});
  TF_EXPECT_OK(AttrValueHasType(v, "func"));
  EXPECT_FALSE(HasPlaceHolder(v));

  EXPECT_EQ(SummarizeAttrValue(v),
            "OpRequiresShape[shape_full=[1,0], shape_part=[?,1,0]]");
}

TEST(AttrValueUtil, DeepAttr) {
  auto v = Fs({{"f", {{"T", P("T")}}}, {"g", {{"T", P("T")}}}});
  TF_EXPECT_OK(AttrValueHasType(v, "list(func)"));
  EXPECT_TRUE(HasPlaceHolder(v));

  for (int i = 0; i < 3; ++i) {
    v = F("f", {{"T", P("T")}, {"F", v}});
    EXPECT_TRUE(HasPlaceHolder(v));
  }
  EXPECT_EQ(SummarizeAttrValue(v),
            "f[F=f[F=f[F=[f[T=$T], g[T=$T]], T=$T], T=$T], T=$T]");

  SubstitutePlaceholders(ReplaceTWith(F("x", {})), &v);
  EXPECT_TRUE(!HasPlaceHolder(v));
  EXPECT_EQ(SummarizeAttrValue(v),
            "f[F=f[F=f[F=[f[T=x[]], g[T=x[]]], T=x[]], T=x[]], T=x[]]");
}

}  // namespace tensorflow
