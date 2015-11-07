#include "tensorflow/core/framework/attr_value_util.h"

#include <gtest/gtest.h>

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
            std::vector<std::pair<string, AttrValue> > pairs) {
  AttrValue ret;
  ret.mutable_func()->set_name(name);
  ret.mutable_func()->mutable_attr()->insert(pairs.begin(), pairs.end());
  return ret;
}

TEST(AttrValueUtil, HasType) {
  // OK
  EXPECT_TRUE(AttrValueHasType(V(123), "int").ok());
  EXPECT_TRUE(AttrValueHasType(V(1.2), "float").ok());
  EXPECT_TRUE(AttrValueHasType(V(DT_FLOAT), "type").ok());
  EXPECT_TRUE(AttrValueHasType(F("f", {}), "func").ok());

  // not OK.
  EXPECT_FALSE(AttrValueHasType(V(123), "func").ok());
  EXPECT_FALSE(AttrValueHasType(V(1.2), "int").ok());
  EXPECT_FALSE(AttrValueHasType(V(DT_FLOAT), "shape").ok());
  EXPECT_FALSE(AttrValueHasType(F("f", {}), "string").ok());
  EXPECT_FALSE(AttrValueHasType(P("T"), "float").ok());
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
  TF_CHECK_OK(AttrValueHasType(v, "func"));
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

TEST(AttrValueUtil, DeepAttr) {
  auto v = F("f", {{"T", P("T")}});
  TF_CHECK_OK(AttrValueHasType(v, "func"));
  EXPECT_TRUE(HasPlaceHolder(v));

  for (int i = 0; i < 3; ++i) {
    v = F("f", {{"T", P("T")}, {"F", v}});
    EXPECT_TRUE(HasPlaceHolder(v));
  }
  EXPECT_EQ(SummarizeAttrValue(v), "f[F=f[F=f[F=f[T=$T], T=$T], T=$T], T=$T]");

  SubstitutePlaceholders(ReplaceTWith(F("x", {})), &v);
  EXPECT_TRUE(!HasPlaceHolder(v));
  EXPECT_EQ(SummarizeAttrValue(v),
            "f[F=f[F=f[F=f[T=x[]], T=x[]], T=x[]], T=x[]]");
}

}  // namespace tensorflow
