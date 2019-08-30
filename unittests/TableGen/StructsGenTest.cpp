//===- StructsGenTest.cpp - TableGen StructsGen Tests ---------------------===//
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

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Identifier.h"
#include "mlir/IR/StandardTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringSwitch.h"
#include "gmock/gmock.h"
#include <type_traits>

namespace mlir {

// Pull in generated enum utility declarations
#include "StructAttrGenTest.h.inc"
// And definitions
#include "StructAttrGenTest.cpp.inc"
// Helper that returns an example test::TestStruct for testing its
// implementation.
static test::TestStruct getTestStruct(mlir::MLIRContext *context) {
  auto integerType = mlir::IntegerType::get(32, context);
  auto integerAttr = mlir::IntegerAttr::get(integerType, 127);

  auto floatType = mlir::FloatType::getF16(context);
  auto floatAttr = mlir::FloatAttr::get(floatType, 0.25);

  auto elementsType = mlir::RankedTensorType::get({2, 3}, integerType);
  auto elementsAttr =
      mlir::DenseElementsAttr::get(elementsType, {1, 2, 3, 4, 5, 6});

  return test::TestStruct::get(integerAttr, floatAttr, elementsAttr, context);
}

// Validates that test::TestStruct::classof correctly identifies a valid
// test::TestStruct.
TEST(StructsGenTest, ClassofTrue) {
  mlir::MLIRContext context;
  auto structAttr = getTestStruct(&context);
  ASSERT_TRUE(test::TestStruct::classof(structAttr));
}

// Validates that test::TestStruct::classof fails when an extra attribute is in
// the class.
TEST(StructsGenTest, ClassofExtraFalse) {
  mlir::MLIRContext context;
  mlir::DictionaryAttr structAttr = getTestStruct(&context);
  auto expectedValues = structAttr.getValue();
  ASSERT_EQ(expectedValues.size(), 3);

  // Copy the set of named attributes.
  llvm::SmallVector<mlir::NamedAttribute, 5> newValues(expectedValues.begin(),
                                                       expectedValues.end());

  // Add an extra NamedAttribute.
  auto wrongId = mlir::Identifier::get("wrong", &context);
  auto wrongAttr = mlir::NamedAttribute(wrongId, expectedValues[0].second);
  newValues.push_back(wrongAttr);

  // Make a new DictionaryAttr and validate.
  auto badDictionary = mlir::DictionaryAttr::get(newValues, &context);
  ASSERT_FALSE(test::TestStruct::classof(badDictionary));
}

// Validates that test::TestStruct::classof fails when a NamedAttribute has an
// incorrect name.
TEST(StructsGenTest, ClassofBadNameFalse) {
  mlir::MLIRContext context;
  mlir::DictionaryAttr structAttr = getTestStruct(&context);
  auto expectedValues = structAttr.getValue();
  ASSERT_EQ(expectedValues.size(), 3);

  // Create a copy of all but the first NamedAttributes.
  llvm::SmallVector<mlir::NamedAttribute, 4> newValues(
      expectedValues.begin() + 1, expectedValues.end());

  // Add a copy of the first attribute with the wrong Identifier.
  auto wrongId = mlir::Identifier::get("wrong", &context);
  auto wrongAttr = mlir::NamedAttribute(wrongId, expectedValues[0].second);
  newValues.push_back(wrongAttr);

  auto badDictionary = mlir::DictionaryAttr::get(newValues, &context);
  ASSERT_FALSE(test::TestStruct::classof(badDictionary));
}

// Validates that test::TestStruct::classof fails when a NamedAttribute is
// missing.
TEST(StructsGenTest, ClassofMissingFalse) {
  mlir::MLIRContext context;
  mlir::DictionaryAttr structAttr = getTestStruct(&context);
  auto expectedValues = structAttr.getValue();
  ASSERT_EQ(expectedValues.size(), 3);

  // Copy a subset of the structures Named Attributes.
  llvm::SmallVector<mlir::NamedAttribute, 3> newValues(
      expectedValues.begin() + 1, expectedValues.end());

  // Make a new DictionaryAttr and validate it is not a validte TestStruct.
  auto badDictionary = mlir::DictionaryAttr::get(newValues, &context);
  ASSERT_FALSE(test::TestStruct::classof(badDictionary));
}

// Validate the accessor for the FloatAttr value.
TEST(StructsGenTest, GetFloat) {
  mlir::MLIRContext context;
  auto structAttr = getTestStruct(&context);
  auto returnedAttr = structAttr.sample_float();
  EXPECT_EQ(returnedAttr.getValueAsDouble(), 0.25);
}

// Validate the accessor for the IntegerAttr value.
TEST(StructsGenTest, GetInteger) {
  mlir::MLIRContext context;
  auto structAttr = getTestStruct(&context);
  auto returnedAttr = structAttr.sample_integer();
  EXPECT_EQ(returnedAttr.getInt(), 127);
}

// Validate the accessor for the ElementsAttr value.
TEST(StructsGenTest, GetElements) {
  mlir::MLIRContext context;
  auto structAttr = getTestStruct(&context);
  auto returnedAttr = structAttr.sample_elements();
  auto denseAttr = returnedAttr.dyn_cast<mlir::DenseElementsAttr>();
  ASSERT_TRUE(denseAttr);

  for (const auto &valIndexIt : llvm::enumerate(denseAttr.getIntValues())) {
    EXPECT_EQ(valIndexIt.value(), valIndexIt.index() + 1);
  }
}

} // namespace mlir
