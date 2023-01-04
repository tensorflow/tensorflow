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
#include <cstdint>
#include <cstdlib>
#include <string>
#include <string_view>

#include "fuzztest/fuzztest.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"

// This is a fuzzer for tensorflow::ParseAttrValue.

namespace {
using tensorflow::StringPiece;

void FuzzTest(std::string_view type, std::string_view text_string) {
  // ParseAttrValue converts text protos into the types of attr_value.proto,
  // which are string, int, float, bool, DataType, TensorShapeProto,
  // TensorProto, NameAttrList, and list of any previously mentioned data type.

  // This fuzzer tests the ParseAttrValue's ability to not crash.
  tensorflow::AttrValue out;

  tensorflow::ParseAttrValue(type, text_string, &out);
}
FUZZ_TEST(CC_FUZZING, FuzzTest)
    .WithDomains(
        fuzztest::ElementOf<std::string>(
            {"string", "int", "float", "bool", "type", "shape", "tensor",
             "list(string)", "list(int)", "list(float)", "list(bool)",
             "list(type)", "list(shape)", "list(tensor)", "list(list(string))",
             "list(list(int))", "list(list(float))", "list(list(bool))",
             "list(list(type))", "list(list(shape))", "list(list(tensor))",
             // Invalid values
             "invalid", "123"}),
        fuzztest::Arbitrary<std::string>());

}  // namespace
