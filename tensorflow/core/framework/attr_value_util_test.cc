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
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/protobuf.h"
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

TEST(AttrValueUtil, SummarizeAttrValueDoesNotElideShortStrings) {
  AttrValue attr_value;
  SetAttrValue(string(40, '-'), &attr_value);
  EXPECT_EQ(strings::StrCat("\"", string(40, '-'), "\""),
            SummarizeAttrValue(attr_value));
}

TEST(AttrValueUtil, SummarizeAttrValueElidesLongStrings) {
  AttrValue attr_value;
  SetAttrValue(string(80, '-'), &attr_value);
  EXPECT_EQ("\"----------...----------\"", SummarizeAttrValue(attr_value));
}

TEST(AttrValueUtil, SummarizeAttrValueDoesNotElideShortLists) {
  std::vector<int> alist(10);
  std::iota(alist.begin(), alist.end(), 0);

  AttrValue attr_value;
  SetAttrValue(alist, &attr_value);
  EXPECT_EQ("[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]", SummarizeAttrValue(attr_value));
}

TEST(AttrValueUtil, SummarizeAttrValueElidesLongLists) {
  std::vector<int> alist(30);
  std::iota(alist.begin(), alist.end(), 0);

  AttrValue attr_value;
  SetAttrValue(alist, &attr_value);
  EXPECT_EQ("[0, 1, 2, 3, 4, ..., 25, 26, 27, 28, 29]",
            SummarizeAttrValue(attr_value));
}

AttrValue FromText(const string& text) {
  AttrValue attr;
  EXPECT_TRUE(protobuf::TextFormat::MergeFromString(text, &attr));
  return attr;
}

void ExpectDifferent(const AttrValue& a1, const AttrValue& a2) {
  EXPECT_FALSE(AreAttrValuesEqual(a1, a2));
  EXPECT_FALSE(AreAttrValuesEqual(a2, a1));
  EXPECT_NE(AttrValueHash(a1), AttrValueHash(a2));
}

TEST(AttrValueEquality, StringAndFuncTensors) {
  AttrValue a = FromText(R"(
      tensor {
        dtype: DT_STRING
        tensor_shape {
          dim {
            size: 2
          }
        }
        string_val: 'reader_dataset_ops_test/tmphtXHks/text_line.0.txt'
        string_val: 'reader_dataset_ops_test/tmphtXHks/text_line.1.txt'
      })");
  EXPECT_TRUE(AreAttrValuesEqual(a, a));
  EXPECT_EQ(AttrValueHash(a), AttrValueHash(a));

  AttrValue b = a;
  (*b.mutable_tensor()->mutable_string_val(0))[3] = '1';
  ExpectDifferent(a, b);

  AttrValue c1;
  c1.mutable_func()->set_name("func_name");
  (*c1.mutable_func()->mutable_attr())["attr1"] = a;
  (*c1.mutable_func()->mutable_attr())["attr2"] = b;
  EXPECT_TRUE(AreAttrValuesEqual(c1, c1));
  EXPECT_EQ(AttrValueHash(c1), AttrValueHash(c1));

  ExpectDifferent(c1, a);

  AttrValue c2 = c1;
  c2.mutable_func()->set_name("func_name2");
  ExpectDifferent(c1, c2);

  c2 = c1;
  (*c2.mutable_func()->mutable_attr())["attr3"] = b;
  ExpectDifferent(c1, c2);

  c2 = c1;
  (*c2.mutable_func()->mutable_attr())["attr2"] = a;
  ExpectDifferent(c1, c2);

  c2 = c1;
  c2.mutable_func()->mutable_attr()->erase("attr2");
  ExpectDifferent(c1, c2);
}

}  // namespace tensorflow
