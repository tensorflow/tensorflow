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

#include "tensorflow/core/framework/op_gen_lib.h"

#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(OpGenLibTest, MultilinePBTxt) {
  // Non-multiline pbtxt
  const string pbtxt = R"(foo: "abc"
foo: ""
foo: "\n\n"
foo: "abc\nEND"
  foo: "ghi\njkl\n"
bar: "quotes:\""
)";

  // Field "foo" converted to multiline but not "bar".
  const string ml_foo = R"(foo: <<END
abc
END
foo: <<END

END
foo: <<END



END
foo: <<END0
abc
END
END0
  foo: <<END
ghi
jkl

END
bar: "quotes:\""
)";

  // Both fields "foo" and "bar" converted to multiline.
  const string ml_foo_bar = R"(foo: <<END
abc
END
foo: <<END

END
foo: <<END



END
foo: <<END0
abc
END
END0
  foo: <<END
ghi
jkl

END
bar: <<END
quotes:"
END
)";

  // ToMultiline
  EXPECT_EQ(ml_foo, PBTxtToMultiline(pbtxt, {"foo"}));
  EXPECT_EQ(pbtxt, PBTxtToMultiline(pbtxt, {"baz"}));
  EXPECT_EQ(ml_foo_bar, PBTxtToMultiline(pbtxt, {"foo", "bar"}));

  // FromMultiline
  EXPECT_EQ(pbtxt, PBTxtFromMultiline(pbtxt));
  EXPECT_EQ(pbtxt, PBTxtFromMultiline(ml_foo));
  EXPECT_EQ(pbtxt, PBTxtFromMultiline(ml_foo_bar));
}

TEST(OpGenLibTest, PBTxtToMultilineErrorCases) {
  // Everything correct.
  EXPECT_EQ("f: <<END\n7\nEND\n", PBTxtToMultiline("f: \"7\"\n", {"f"}));

  // In general, if there is a problem parsing in PBTxtToMultiline, it leaves
  // the line alone.

  // No colon
  EXPECT_EQ("f \"7\"\n", PBTxtToMultiline("f \"7\"\n", {"f"}));
  // Only converts strings.
  EXPECT_EQ("f: 7\n", PBTxtToMultiline("f: 7\n", {"f"}));
  // No quote after colon.
  EXPECT_EQ("f: 7\"\n", PBTxtToMultiline("f: 7\"\n", {"f"}));
  // Only one quote
  EXPECT_EQ("f: \"7\n", PBTxtToMultiline("f: \"7\n", {"f"}));
  // Illegal escaping
  EXPECT_EQ("f: \"7\\\"\n", PBTxtToMultiline("f: \"7\\\"\n", {"f"}));
}

TEST(OpGenLibTest, PBTxtToMultilineComments) {
  const string pbtxt = R"(f: "bar"  # Comment 1
    f: "\n"  # Comment 2
)";
  const string ml = R"(f: <<END
bar
END  # Comment 1
    f: <<END


END  # Comment 2
)";

  EXPECT_EQ(ml, PBTxtToMultiline(pbtxt, {"f"}));
  EXPECT_EQ(pbtxt, PBTxtFromMultiline(ml));
}

}  // namespace
}  // namespace tensorflow
