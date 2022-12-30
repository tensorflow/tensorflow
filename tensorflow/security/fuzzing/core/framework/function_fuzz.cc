// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <vector>

#include <gtest/gtest.h>
#include "fuzztest/fuzztest.h"
#include "tensorflow/security/fuzzing/domains/function_def.h"

namespace tensorflow {
namespace fuzzing {

using AttrValueVariantFuzzType = std::variant<std::string, int, float, std::vector<std::string>>;
using AttrsFuzzType = std::vector<std::pair<std::string, AttrValueVariantFuzzType>>;
using AttrsVectorType = std::vector<std::pair<std::string, FunctionDefHelper::AttrValueWrapper>>;

namespace domain {

AttrsVectorType AttrsVectorGenerator(AttrsFuzzType attrsFuzz) {
  AttrsVectorType attrs;
  attrs.reserve(attrsFuzz.size());
  std::transform(attrsFuzz.cbegin(), attrsFuzz.cend(), std::back_inserter(attrs),
                 [](std::pair<std::string, AttrValueVariantFuzzType> attr) {
                   return std::visit([&](auto&& value) {
                     return std::make_pair(attr.first, FunctionDefHelper::AttrValueWrapper{value});
                   }, attr.second);
                 });
  return attrs;
}

auto Attrs() {
  return fuzztest::Map(AttrsVectorGenerator, fuzztest::Arbitrary<AttrsFuzzType>());
}

auto RandomNode() {
  return fuzztest::Map([](std::vector<string> ret, std::string op, std::vector<string> arg,
                          AttrsVectorType attr, std::vector<string> dep,
                          std::string device, std::string name) {
      return FunctionDefHelper::Node {ret, op, arg, attr, dep, device, name};
    }, fuzztest::VectorOf(fuzztest::PrintableAsciiString()).WithMinSize(1), fuzztest::PrintableAsciiString(),
       fuzztest::VectorOf(fuzztest::PrintableAsciiString()), Attrs(), fuzztest::VectorOf(fuzztest::PrintableAsciiString()),
       fuzztest::PrintableAsciiString(), fuzztest::PrintableAsciiString());
}

}  // end namespace domain

void AttrValueWrapperFuzz(AttrValueVariantFuzzType in) {
  auto value = std::visit([](auto&& value) {
      return FunctionDefHelper::AttrValueWrapper(value);
    }, in);
}
FUZZ_TEST(FunctionFuzz, AttrValueWrapperFuzz);

void FunctionRefWithAttrsFuzz(std::string name, std::vector<std::pair<std::string, FunctionDefHelper::AttrValueWrapper>> attrs) {
  FunctionDefHelper::FunctionRef(name, attrs);
}
FUZZ_TEST(FunctionFuzz, FunctionRefWithAttrsFuzz).WithDomains(fuzztest::PrintableAsciiString(), domain::Attrs());

void FunctionRefFuzz(std::string name) {
  FunctionDefHelper::FunctionRef(name);
}
FUZZ_TEST(FunctionFuzz, FunctionRefFuzz).WithDomains(fuzztest::PrintableAsciiString());

void NodeGetNameFuzz(FunctionDefHelper::Node node) {
  node.GetName();
}
FUZZ_TEST(FunctionFuzz, NodeGetNameFuzz).WithDomains(domain::RandomNode());

void NodeToNodeDefFuzz(FunctionDefHelper::Node node) {
  node.ToNodeDef();
}
FUZZ_TEST(FunctionFuzz, NodeToNodeDefFuzz).WithDomains(domain::RandomNode());

void DebugStringFuzz(FunctionDef def) {
  auto str = DebugString(def);
  EXPECT_FALSE(str.empty());
}
FUZZ_TEST(FunctionFuzz, DebugStringFuzz).WithDomains(domain::FunctionDef());

void FunctionDefsEqualFuzz(FunctionDef def1, FunctionDef def2) {
  FunctionDefsEqual(def1, def2);
  EXPECT_TRUE(FunctionDefsEqual(def1, def1));
}
FUZZ_TEST(FunctionFuzz, FunctionDefsEqualFuzz).WithDomains(domain::FunctionDef(), domain::FunctionDef());

void FunctionDefHashFuzz(FunctionDef def) {
  FunctionDefHash(def);
}
FUZZ_TEST(FunctionFuzz, FunctionDefHashFuzz).WithDomains(domain::FunctionDef());

}  // end namespace fuzzing
}  // end namespace tensorflow
