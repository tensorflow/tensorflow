/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {
// The returned array is only valid for the lifetime of the input vector.
// We're using const casting because we need to pass in an argv-style array of
// char* pointers for the API, even though we know they won't be altered.
std::vector<char*> CharPointerVectorFromStrings(
    const std::vector<tensorflow::string>& strings) {
  std::vector<char*> result;
  for (const tensorflow::string& string : strings) {
    result.push_back(const_cast<char*>(string.c_str()));
  }
  return result;
}
}

TEST(CommandLineFlagsTest, BasicUsage) {
  int some_int = 10;
  bool some_switch = false;
  tensorflow::string some_name = "something";
  int argc = 4;
  std::vector<tensorflow::string> argv_strings = {
      "program_name", "--some_int=20", "--some_switch",
      "--some_name=somethingelse"};
  std::vector<char*> argv_array = CharPointerVectorFromStrings(argv_strings);
  bool parsed_ok =
      ParseFlags(&argc, argv_array.data(), {Flag("some_int", &some_int),
                                            Flag("some_switch", &some_switch),
                                            Flag("some_name", &some_name)});
  EXPECT_EQ(true, parsed_ok);
  EXPECT_EQ(20, some_int);
  EXPECT_EQ(true, some_switch);
  EXPECT_EQ("somethingelse", some_name);
  EXPECT_EQ(argc, 1);
}

TEST(CommandLineFlagsTest, BadIntValue) {
  int some_int = 10;
  int argc = 2;
  std::vector<tensorflow::string> argv_strings = {"program_name",
                                                  "--some_int=notanumber"};
  std::vector<char*> argv_array = CharPointerVectorFromStrings(argv_strings);
  bool parsed_ok =
      ParseFlags(&argc, argv_array.data(), {Flag("some_int", &some_int)});

  EXPECT_EQ(false, parsed_ok);
  EXPECT_EQ(10, some_int);
  EXPECT_EQ(argc, 1);
}

TEST(CommandLineFlagsTest, BadBoolValue) {
  bool some_switch = false;
  int argc = 2;
  std::vector<tensorflow::string> argv_strings = {"program_name",
                                                  "--some_switch=notabool"};
  std::vector<char*> argv_array = CharPointerVectorFromStrings(argv_strings);
  bool parsed_ok =
      ParseFlags(&argc, argv_array.data(), {Flag("some_switch", &some_switch)});

  EXPECT_EQ(false, parsed_ok);
  EXPECT_EQ(false, some_switch);
  EXPECT_EQ(argc, 1);
}

}  // namespace tensorflow
