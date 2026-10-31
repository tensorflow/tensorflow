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

#include <numeric>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include <cstdint>
#include <string>
#include "absl/base/casts.h"
#include "tensorflow/core/framework/types.pb.h"
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

AttrValue P(const std::string& p) {
  AttrValue ret;
  ret.set_placeholder(p);
  return ret;
}

AttrValue F(const std::string& name,
            std::vector<std::pair<std::string, AttrValue>> pairs) {
  AttrValue ret;
  ret.mutable_func()->set_name(name);
  ret.mutable_func()->mutable_attr()->insert(pairs.begin(), pairs.end());
  return ret;
}

AttrValue Fs(
    std::vector<
        std::pair<std::string, std::vector<std::pair<std::string, AttrValue>>>>
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
  return [val](const std::string& placeholder, AttrValue* target) {
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
  SetAttrValue(std::string(40, '-'), &attr_value);
  EXPECT_EQ(absl::StrCat("\"", std::string(40, '-'), "\""),
            SummarizeAttrValue(attr_value));
}

TEST(AttrValueUtil, SummarizeAttrValueElidesLongStrings) {
  AttrValue attr_value;
  SetAttrValue(std::string(80, '-'), &attr_value);
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
  std::vector<int> alist(110);
  std::iota(alist.begin(), alist.end(), 0);

  AttrValue attr_value;
  SetAttrValue(alist, &attr_value);
  EXPECT_EQ(
      "[0, 1, 2, 3, 4, ..., 105, 106, 107, 108, "
      "109]{attr_hash=14506120815048308275}",
      SummarizeAttrValue(attr_value));
}

TEST(AttrValueUtil, TensorByteSizeNumElementsOverflows) {
  TensorProto proto;
  proto.mutable_tensor_shape()->add_dim()->set_size(9223372036854775807L);
  proto.mutable_tensor_shape()->add_dim()->set_size(2092026309338556617L);
  proto.set_dtype(DT_INT32);
  EXPECT_EQ(attr_value_util_internal::TensorByteSize(proto), -1);
}

TEST(AttrValueUtil, TensorByteSizeShouldNotOverflow) {
  {
    TensorProto proto;
    proto.mutable_tensor_shape()->add_dim()->set_size(4611686018427387904L);
    proto.set_dtype(DT_INT32);
    EXPECT_EQ(attr_value_util_internal::TensorByteSize(proto), -1);
  }
  {
    TensorProto proto;
    proto.mutable_tensor_shape()->add_dim()->set_size(46123445412334L);
    proto.set_dtype(DT_INT32);
    EXPECT_NE(attr_value_util_internal::TensorByteSize(proto), -1);
  }
}

AttrValue FromText(const std::string& text) {
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

namespace {

void ExpectMixedListEquality(const AttrValue& a, const AttrValue& b,
                             bool equal) {
  ASSERT_GT(a.list().func_size(), 0);
  ASSERT_GT(b.list().func_size(), 0);
  for (bool allow_false_negatives : {false, true}) {
    EXPECT_EQ(AreAttrValuesEqual(a, b, allow_false_negatives), equal);
    EXPECT_EQ(AreAttrValuesEqual(b, a, allow_false_negatives), equal);
  }
  EXPECT_EQ(AttrValueHash(a) == AttrValueHash(b), equal);
}

AttrValue MixedFuncList() {
  AttrValue a;
  auto* list = a.mutable_list();
  list->add_func()->set_name("f");
  list->add_s(std::string("a\0b", 3));
  list->add_i(7);
  list->add_f(1.5f);
  list->add_b(true);
  list->add_type(DT_INT32);
  list->add_shape()->add_dim()->set_size(2);
  auto* tensor = list->add_tensor();
  tensor->set_dtype(DT_INT32);
  tensor->mutable_tensor_shape()->add_dim()->set_size(1);
  tensor->add_int_val(1);
  return a;
}

// Unknown field 100, wire type varint. Parsing works with full and lite protos.
template <typename Proto>
void AddUnknownField(Proto* proto, char value) {
  std::string wire = proto->SerializeAsString();
  wire.append("\xa0\x06", 2);
  wire.push_back(value);
  ASSERT_TRUE(proto->ParseFromString(wire));
}

}  // namespace

TEST(AttrValueEquality, MixedFuncListFloatBits) {
  AttrValue a;
  a.mutable_list()->add_func()->set_name("f");
  a.mutable_list()->add_f(0.0f);
  AttrValue b = a;
  ExpectMixedListEquality(a, b, true);
  b.mutable_list()->set_f(0, -0.0f);
  ExpectMixedListEquality(a, b, false);

  a.mutable_list()->set_f(0, absl::bit_cast<float>(uint32_t{0x7fc00001}));
  b = a;
  ExpectMixedListEquality(a, b, true);
  b.mutable_list()->set_f(0, absl::bit_cast<float>(uint32_t{0x7fc00002}));
  ExpectMixedListEquality(a, b, false);
  b.mutable_list()->set_f(0, absl::bit_cast<float>(uint32_t{0xffc00001}));
  ExpectMixedListEquality(a, b, false);
}

TEST(AttrValueEquality, MixedFuncListKnownFields) {
  const AttrValue a = MixedFuncList();
  ExpectMixedListEquality(a, AttrValue(a), true);
  for (int field = 0; field < 7; ++field) {
    SCOPED_TRACE(field);
    AttrValue b = a;
    auto* list = b.mutable_list();
    switch (field) {
      case 0:
        list->set_s(0, "different");
        break;
      case 1:
        list->set_i(0, 8);
        break;
      case 2:
        list->set_f(0, 2.5f);
        break;
      case 3:
        list->set_b(0, false);
        break;
      case 4:
        list->set_type(0, DT_FLOAT);
        break;
      case 5:
        list->mutable_shape(0)->mutable_dim(0)->set_size(3);
        break;
      case 6:
        list->mutable_tensor(0)->set_int_val(0, 2);
        break;
    }
    ExpectMixedListEquality(a, b, false);
    b = a;
    list = b.mutable_list();
    switch (field) {
      case 0:
        list->clear_s();
        break;
      case 1:
        list->clear_i();
        break;
      case 2:
        list->clear_f();
        break;
      case 3:
        list->clear_b();
        break;
      case 4:
        list->clear_type();
        break;
      case 5:
        list->clear_shape();
        break;
      case 6:
        list->clear_tensor();
        break;
    }
    ExpectMixedListEquality(a, b, false);
  }
  AttrValue b = a;
  b.mutable_list()->add_i(9);
  AttrValue c = b;
  c.mutable_list()->set_i(0, 9);
  c.mutable_list()->set_i(1, 7);
  ExpectMixedListEquality(b, c, false);
}

TEST(AttrValueEquality, MixedFuncListUnknownFields) {
  for (bool mixed : {false, true}) {
    AttrValue a;
    if (mixed)
      a = MixedFuncList();
    else
      a.mutable_list()->add_func()->set_name("f");
    AttrValue b = a;
    AddUnknownField(a.mutable_list(), 1);
    ExpectMixedListEquality(a, b, true);
    AddUnknownField(b.mutable_list(), 2);
    ExpectMixedListEquality(a, b, true);
  }
  for (bool tensor : {false, true}) {
    AttrValue a = MixedFuncList();
    AttrValue b = a;
    if (tensor)
      AddUnknownField(a.mutable_list()->mutable_tensor(0), 1);
    else
      AddUnknownField(a.mutable_list()->mutable_shape(0), 1);
    ExpectMixedListEquality(a, b, false);
    b = a;
    ExpectMixedListEquality(a, b, true);
    if (tensor)
      AddUnknownField(b.mutable_list()->mutable_tensor(0), 2);
    else
      AddUnknownField(b.mutable_list()->mutable_shape(0), 2);
    ExpectMixedListEquality(a, b, false);
  }
}

TEST(AttrValueEquality, MixedFuncListTensorEncoding) {
  const AttrValue a = MixedFuncList();
  AttrValue b = a;
  auto* tensor = b.mutable_list()->mutable_tensor(0);
  tensor->clear_int_val();
  tensor->set_tensor_content(std::string("\x01\x00\x00\x00", 4));
  // Direct tensor fields retain serialized comparison under both policies.
  ExpectMixedListEquality(a, b, false);

  AttrValue c = a;
  AttrValue tensor_attr;
  auto* nested = tensor_attr.mutable_tensor();
  nested->set_dtype(DT_INT32);
  nested->mutable_tensor_shape()->add_dim()->set_size(1);
  nested->add_int_val(1);
  (*c.mutable_list()->mutable_func(0)->mutable_attr())["t"] = tensor_attr;
  AttrValue d = c;
  auto* d_nested = (*d.mutable_list()->mutable_func(0)->mutable_attr())["t"]
                       .mutable_tensor();
  d_nested->clear_int_val();
  d_nested->set_tensor_content(std::string("\x01\x00\x00\x00", 4));
  ExpectMixedListEquality(c, d, true);
  EXPECT_EQ(AttrValueHash(c), AttrValueHash(d));
}

TEST(AttrValueEquality, GiantTensors) {
  AttrValue tensor = FromText(R"(
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1024
          }
          dim {
            size: 1024
          }
          dim {
            size: 1024
          }
          dim {
            size: 1024
          }
        }
        int_val: 0
      })");
  EXPECT_TRUE(AreAttrValuesEqual(tensor, tensor));
}

namespace {

AttrValue BuildDeepFuncAttr(int depth) {
  AttrValue current;
  current.set_placeholder("T");
  for (int i = 0; i < depth; ++i) {
    AttrValue parent;
    parent.mutable_func()->set_name("f");
    (*parent.mutable_func()->mutable_attr())["x"].Swap(&current);
    current.Swap(&parent);
  }
  return current;
}

AttrValue BuildDeepListFuncAttr(int depth) {
  AttrValue current = V(1);
  for (int i = 0; i < depth; ++i) {
    AttrValue parent;
    NameAttrList* func = parent.mutable_list()->add_func();
    func->set_name("f");
    (*func->mutable_attr())["x"].Swap(&current);
    current.Swap(&parent);
  }
  return current;
}

}  // namespace

TEST(AttrValueUtil, ParseAttrValueAcceptsStringBracesAndComments) {
  AttrValue tensor_out;
  std::string braces_in_string =
      "{ dtype: DT_STRING tensor_shape {} string_val: \"" +
      std::string(150, '{') + "\" }";
  EXPECT_TRUE(ParseAttrValue("tensor", braces_in_string, &tensor_out));
  ASSERT_EQ(tensor_out.tensor().string_val_size(), 1);
  EXPECT_EQ(tensor_out.tensor().string_val(0).size(), 150);

  AttrValue func_out;
  EXPECT_TRUE(
      ParseAttrValue("func",
                     "# comment > }\n"
                     "{ name: \"fn\" attr { key: \"k\" value { i: 3 } } }",
                     &func_out));
  EXPECT_EQ(func_out.func().name(), "fn");
}

TEST(AttrValueUtil, ParseAttrValueScannerQuotesAndDelimiters) {
  AttrValue out;
  std::string escaped =
      "{ dtype: DT_STRING tensor_shape {} string_val: \"\\\"\\\\" +
      std::string(150, '{') + "\" }";
  EXPECT_TRUE(ParseAttrValue("tensor", escaped, &out));
  ASSERT_EQ(out.tensor().string_val_size(), 1);
  EXPECT_EQ(out.tensor().string_val(0).size(), 152);

  std::string single_quoted =
      "{ dtype: DT_STRING tensor_shape {} string_val: '" +
      std::string(150, '<') + "' }";
  EXPECT_TRUE(ParseAttrValue("tensor", single_quoted, &out));

  std::string deep_after_escaped_backslash =
      "{ name: \"f\\\\\" attr { key: \"x\" value { func ";
  for (int i = 0; i < 150; ++i) {
    deep_after_escaped_backslash +=
        "{ name: \"f\" attr { key: \"x\" value { func ";
  }
  deep_after_escaped_backslash += "{ name: \"leaf\" }";
  for (int i = 0; i < 150; ++i) {
    deep_after_escaped_backslash += " } } }";
  }
  deep_after_escaped_backslash += " } } }";
  EXPECT_FALSE(ParseAttrValue("func", deep_after_escaped_backslash, &out));

  EXPECT_FALSE(ParseAttrValue("func", "} { name: \"f\" }", &out));
  EXPECT_FALSE(ParseAttrValue("tensor", "{ string_val: \"abc }", &out));
  EXPECT_FALSE(ParseAttrValue("tensor", "{ string_val: \"abc\\", &out));
}

TEST(AttrValueUtil, ParseAttrValueRejectsDeepNestingAndCommentBypass) {
  constexpr int kDepth = 15000;
  std::string deep_tensor = "# " + std::string(kDepth * 3, '>') + "\n";
  for (int i = 0; i < kDepth; ++i) {
    deep_tensor += "{ variant_val { tensors ";
  }
  deep_tensor += "{ dtype: DT_INT32 int_val: 1 }";
  for (int i = 0; i < kDepth; ++i) {
    deep_tensor += " } }";
  }
  AttrValue out;
  EXPECT_FALSE(ParseAttrValue("tensor", deep_tensor, &out));

  std::string deep_func;
  for (int i = 0; i < kDepth; ++i) {
    deep_func += "{ name: \"f\" attr { key: \"x\" value { func ";
  }
  deep_func += "{ name: \"leaf\" }";
  for (int i = 0; i < kDepth; ++i) {
    deep_func += " } } }";
  }
  EXPECT_FALSE(ParseAttrValue("func", deep_func, &out));
  EXPECT_FALSE(
      ParseAttrValue("list(int)", "[1] func: [" + deep_func + "]", &out));
}

TEST(AttrValueUtil, ListFuncNestingLimitBoundary) {
  AttrValue ok = BuildDeepListFuncAttr(100);
  AttrValue deep = BuildDeepListFuncAttr(101);
  const auto subst = ReplaceTWith(V(DT_FLOAT));
  EXPECT_FALSE(HasPlaceHolder(ok));
  EXPECT_TRUE(HasPlaceHolder(deep));
  EXPECT_TRUE(SubstitutePlaceholders(subst, &ok));
  EXPECT_FALSE(SubstitutePlaceholders(subst, &deep));
  EXPECT_TRUE(AreAttrValuesEqual(ok, AttrValue(ok)));
  EXPECT_FALSE(AreAttrValuesEqual(deep, AttrValue(deep)));
  EXPECT_EQ(AttrValueHash(ok), AttrValueHash(AttrValue(ok)));
  EXPECT_EQ(FastAttrValueHash(ok), FastAttrValueHash(AttrValue(ok)));
  EXPECT_EQ(AttrValueHash(deep), AttrValueHash(AttrValue(deep)));
  EXPECT_EQ(FastAttrValueHash(deep), FastAttrValueHash(AttrValue(deep)));
  const std::string kLimit = "<AttrValue nesting limit exceeded>";
  EXPECT_EQ(SummarizeAttrValue(ok).find(kLimit), std::string::npos);
  EXPECT_NE(SummarizeAttrValue(deep).find(kLimit), std::string::npos);
}

TEST(AttrValueUtil, ExcessiveFuncNestingIsBounded) {
  constexpr int kDepth = 150;
  AttrValue a = BuildDeepFuncAttr(kDepth);
  AttrValue b = BuildDeepFuncAttr(kDepth);

  EXPECT_NE(SummarizeAttrValue(a).find("<AttrValue nesting limit exceeded>"),
            std::string::npos);
  EXPECT_FALSE(AreAttrValuesEqual(a, b));
  EXPECT_EQ(AttrValueHash(a), AttrValueHash(b));
  EXPECT_EQ(FastAttrValueHash(a), FastAttrValueHash(b));
  EXPECT_TRUE(HasPlaceHolder(a));
  EXPECT_FALSE(SubstitutePlaceholders(ReplaceTWith(V(DT_FLOAT)), &a));

  AttrValue list_a;
  list_a.mutable_list()->add_func()->Swap(a.mutable_func());
  EXPECT_FALSE(AreAttrValuesEqual(list_a, V(1)));
  EXPECT_FALSE(AreAttrValuesEqual(V(1), list_a));
  a.mutable_func()->Swap(list_a.mutable_list()->mutable_func(0));
}

}  // namespace tensorflow
