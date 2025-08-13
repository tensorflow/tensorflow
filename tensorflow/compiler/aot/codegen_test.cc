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

#include "tensorflow/compiler/aot/codegen.h"

#include <algorithm>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/TargetSelect.h"
#include "tensorflow/compiler/aot/compile.h"
#include "xla/cpu_function_runtime.h"
#include "xla/service/cpu/cpu_aot_compilation_result.h"
#include "xla/shape_util.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace tfcompile {
namespace {

using ::xla::cpu_function_runtime::BufferInfo;

void ExpectErrorContains(const absl::Status& status, absl::string_view str) {
  EXPECT_NE(absl::OkStatus(), status);
  EXPECT_TRUE(absl::StrContains(status.message(), str))
      << "expected error: " << status.message() << " to contain: " << str;
}

TEST(ValidateCppIdent, Simple) {
  TF_EXPECT_OK(ValidateCppIdent("a", ""));
  TF_EXPECT_OK(ValidateCppIdent("abc", ""));
  TF_EXPECT_OK(ValidateCppIdent("_abc", ""));
  TF_EXPECT_OK(ValidateCppIdent("_abc123", ""));
  // Make sure we didn't skip a valid letter or digit
  string ident;
  for (char c = 'a'; c <= 'z'; c++) {
    ident.append(1, c);
  }
  for (char c = 'A'; c <= 'Z'; c++) {
    ident.append(1, c);
  }
  for (char c = '0'; c <= '9'; c++) {
    ident.append(1, c);
  }
  ident += "_";
  TF_EXPECT_OK(ValidateCppIdent(ident, ""));

  ExpectErrorContains(ValidateCppIdent("", ""), "empty identifier");
  ExpectErrorContains(ValidateCppIdent(" ", ""), "illegal leading char");
  ExpectErrorContains(ValidateCppIdent("0", ""), "illegal leading char");
  ExpectErrorContains(ValidateCppIdent(".", ""), "illegal leading char");
  ExpectErrorContains(ValidateCppIdent(":", ""), "illegal leading char");
  ExpectErrorContains(ValidateCppIdent("a.", ""), "illegal char");
  ExpectErrorContains(ValidateCppIdent("a:", ""), "illegal char");
  ExpectErrorContains(ValidateCppIdent("a:", ""), "illegal char");
}

class ParseCppClassTest : public ::testing::Test {
 protected:
  void ExpectOK(const string& cpp_class, const string& want_class_name,
                const std::vector<string>& want_namespaces) {
    string class_name;
    std::vector<string> namespaces;
    TF_EXPECT_OK(ParseCppClass(cpp_class, &class_name, &namespaces));
    EXPECT_EQ(class_name, want_class_name);
    EXPECT_EQ(namespaces, want_namespaces);
  }

  void ExpectFail(const string& cpp_class) {
    string class_name;
    std::vector<string> namespaces;
    EXPECT_NE(ParseCppClass(cpp_class, &class_name, &namespaces),
              absl::OkStatus())
        << cpp_class;
  }
};

TEST_F(ParseCppClassTest, ParseOK) {
  ExpectOK("MyClass", "MyClass", {});
  ExpectOK("_MyClass", "_MyClass", {});
  ExpectOK("a::MyClass", "MyClass", {"a"});
  ExpectOK("a::foo::MyClass", "MyClass", {"a", "foo"});
  ExpectOK("a::foo::b::MyClass", "MyClass", {"a", "foo", "b"});
  ExpectOK("a::foo::b::bar::MyClass", "MyClass", {"a", "foo", "b", "bar"});
  ExpectOK("foo::MyClass", "MyClass", {"foo"});
  ExpectOK("_foo::MyClass", "MyClass", {"_foo"});
  ExpectOK("_foo::_MyClass", "_MyClass", {"_foo"});
  ExpectOK("::foo::bar::MyClass", "MyClass", {"foo", "bar"});
  ExpectOK("::_foo::MyClass", "MyClass", {"_foo"});
  ExpectOK("::_foo::_MyClass", "_MyClass", {"_foo"});
  // Make sure we didn't skip a valid letter or digit
  string ident;
  for (char c = 'a'; c <= 'z'; c++) {
    ident.append(1, c);
  }
  for (char c = 'A'; c <= 'Z'; c++) {
    ident.append(1, c);
  }
  for (char c = '0'; c <= '9'; c++) {
    ident.append(1, c);
  }
  ident += "_";
  ExpectOK(ident, ident, {});
  ExpectOK(ident + "::" + ident, ident, {ident});
  ExpectOK(ident + "::" + ident + "::" + ident, ident, {ident, ident});
}

TEST_F(ParseCppClassTest, ParseFail) {
  ExpectFail("");
  ExpectFail("::");
  ExpectFail("0");
  ExpectFail("a.b");
  ExpectFail("a:b");
  ExpectFail(":foo::bar");
  ExpectFail("good::.bad");
  ExpectFail("good:::bad");
  ExpectFail("good::bad::");
  ExpectFail("good::::bad");
  ExpectFail("::::bad");
  ExpectFail("good:: bad");
  ExpectFail("good::0bad");
}

static void CompareWithGoldenFile(
    const string& tensorflow_relative_golden_file_name,
    const string& expected_contents, bool ignore_cr) {
  // Get rid of all CR characters, we may be running under windows.
  string sanitized_expected_contents(expected_contents);
  if (ignore_cr) {
    sanitized_expected_contents.erase(
        std::remove(sanitized_expected_contents.begin(),
                    sanitized_expected_contents.end(), '\r'),
        sanitized_expected_contents.end());
  }

  // To update the golden file, flip update_golden to true and run the
  // following:
  // blaz test --test_strategy=local \
  //   "third_party/tensorflow/compiler/aot:codegen_test"
  const bool update_golden = false;
  string golden_file_name =
      GetDataDependencyFilepath(tensorflow_relative_golden_file_name);

  if (update_golden) {
    TF_EXPECT_OK(
        WriteStringToFile(Env::Default(), golden_file_name, expected_contents));
  }

  string golden_file_contents;
  TF_ASSERT_OK(ReadFileToString(Env::Default(), golden_file_name,
                                &golden_file_contents));
  if (ignore_cr) {
    golden_file_contents.erase(std::remove(golden_file_contents.begin(),
                                           golden_file_contents.end(), '\r'),
                               golden_file_contents.end());
  }
  EXPECT_EQ(golden_file_contents, expected_contents);
}

}  // namespace
}  // namespace tfcompile
}  // namespace tensorflow
