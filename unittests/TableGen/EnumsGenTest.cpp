//===- EnumsGenTest.cpp - TableGen EnumsGen Tests -------------------------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringSwitch.h"
#include "gmock/gmock.h"
#include <type_traits>

// Pull in generated enum utility declarations
#include "EnumsGenTest.h.inc"
// And definitions
#include "EnumsGenTest.cpp.inc"

using ::testing::StrEq;

// Test namespaces and enum class/utility names
using Outer::Inner::ConvertToEnum;
using Outer::Inner::ConvertToString;
using Outer::Inner::MyEnum;

TEST(EnumsGenTest, GeneratedEnumDefinition) {
  EXPECT_EQ(0u, static_cast<uint64_t>(MyEnum::CaseA));
  EXPECT_EQ(10u, static_cast<uint64_t>(MyEnum::CaseB));
}

TEST(EnumsGenTest, GeneratedDenseMapInfo) {
  llvm::DenseMap<MyEnum, std::string> myMap;

  myMap[MyEnum::CaseA] = "zero";
  myMap[MyEnum::CaseB] = "ten";

  EXPECT_THAT(myMap[MyEnum::CaseA], StrEq("zero"));
  EXPECT_THAT(myMap[MyEnum::CaseB], StrEq("ten"));
}

TEST(EnumsGenTest, GeneratedSymbolToStringFn) {
  EXPECT_THAT(ConvertToString(MyEnum::CaseA), StrEq("CaseA"));
  EXPECT_THAT(ConvertToString(MyEnum::CaseB), StrEq("CaseB"));
}

TEST(EnumsGenTest, GeneratedStringToSymbolFn) {
  EXPECT_EQ(llvm::Optional<MyEnum>(MyEnum::CaseA), ConvertToEnum("CaseA"));
  EXPECT_EQ(llvm::Optional<MyEnum>(MyEnum::CaseB), ConvertToEnum("CaseB"));
  EXPECT_EQ(llvm::None, ConvertToEnum("X"));
}

TEST(EnumsGenTest, GeneratedUnderlyingType) {
  bool v =
      std::is_same<uint64_t, std::underlying_type<Uint64Enum>::type>::value;
  EXPECT_TRUE(v);
}
