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
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
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
using Outer::Inner::StrEnum;

TEST(EnumsGenTest, GeneratedStrEnumDefinition) {
  EXPECT_EQ(0u, static_cast<uint64_t>(StrEnum::CaseA));
  EXPECT_EQ(10u, static_cast<uint64_t>(StrEnum::CaseB));
}

TEST(EnumsGenTest, GeneratedI32EnumDefinition) {
  EXPECT_EQ(5u, static_cast<uint64_t>(I32Enum::Case5));
  EXPECT_EQ(10u, static_cast<uint64_t>(I32Enum::Case10));
}

TEST(EnumsGenTest, GeneratedDenseMapInfo) {
  llvm::DenseMap<StrEnum, std::string> myMap;

  myMap[StrEnum::CaseA] = "zero";
  myMap[StrEnum::CaseB] = "one";

  EXPECT_THAT(myMap[StrEnum::CaseA], StrEq("zero"));
  EXPECT_THAT(myMap[StrEnum::CaseB], StrEq("one"));
}

TEST(EnumsGenTest, GeneratedSymbolToStringFn) {
  EXPECT_THAT(ConvertToString(StrEnum::CaseA), StrEq("CaseA"));
  EXPECT_THAT(ConvertToString(StrEnum::CaseB), StrEq("CaseB"));
}

TEST(EnumsGenTest, GeneratedStringToSymbolFn) {
  EXPECT_EQ(llvm::Optional<StrEnum>(StrEnum::CaseA), ConvertToEnum("CaseA"));
  EXPECT_EQ(llvm::Optional<StrEnum>(StrEnum::CaseB), ConvertToEnum("CaseB"));
  EXPECT_EQ(llvm::None, ConvertToEnum("X"));
}

TEST(EnumsGenTest, GeneratedUnderlyingType) {
  bool v = std::is_same<uint32_t, std::underlying_type<I32Enum>::type>::value;
  EXPECT_TRUE(v);
}

TEST(EnumsGenTest, GeneratedBitEnumDefinition) {
  EXPECT_EQ(0u, static_cast<uint32_t>(BitEnumWithNone::None));
  EXPECT_EQ(1u, static_cast<uint32_t>(BitEnumWithNone::Bit1));
  EXPECT_EQ(4u, static_cast<uint32_t>(BitEnumWithNone::Bit3));
}

TEST(EnumsGenTest, GeneratedSymbolToStringFnForBitEnum) {
  EXPECT_THAT(stringifyBitEnumWithNone(BitEnumWithNone::None), StrEq("None"));
  EXPECT_THAT(stringifyBitEnumWithNone(BitEnumWithNone::Bit1), StrEq("Bit1"));
  EXPECT_THAT(stringifyBitEnumWithNone(BitEnumWithNone::Bit3), StrEq("Bit3"));
  EXPECT_THAT(
      stringifyBitEnumWithNone(BitEnumWithNone::Bit1 | BitEnumWithNone::Bit3),
      StrEq("Bit1|Bit3"));
}

TEST(EnumsGenTest, GeneratedStringToSymbolForBitEnum) {
  EXPECT_EQ(symbolizeBitEnumWithNone("None"), BitEnumWithNone::None);
  EXPECT_EQ(symbolizeBitEnumWithNone("Bit1"), BitEnumWithNone::Bit1);
  EXPECT_EQ(symbolizeBitEnumWithNone("Bit3"), BitEnumWithNone::Bit3);
  EXPECT_EQ(symbolizeBitEnumWithNone("Bit3|Bit1"),
            BitEnumWithNone::Bit3 | BitEnumWithNone::Bit1);

  EXPECT_EQ(symbolizeBitEnumWithNone("Bit2"), llvm::None);
  EXPECT_EQ(symbolizeBitEnumWithNone("Bit3|Bit4"), llvm::None);

  EXPECT_EQ(symbolizeBitEnumWithoutNone("None"), llvm::None);
}

TEST(EnumsGenTest, GeneratedOperator) {
  EXPECT_TRUE(bitEnumContains(BitEnumWithNone::Bit1 | BitEnumWithNone::Bit3,
                              BitEnumWithNone::Bit1));
  EXPECT_FALSE(bitEnumContains(BitEnumWithNone::Bit1 & BitEnumWithNone::Bit3,
                               BitEnumWithNone::Bit1));
}
