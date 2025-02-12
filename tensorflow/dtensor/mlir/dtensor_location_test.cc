/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/dtensor/mlir/dtensor_location.h"

#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/utils/name_utils.h"
#include "tensorflow/core/platform/test.h"

namespace {

void CheckFileLineColLocation(mlir::Location loc, unsigned line,
                              unsigned column) {
  ASSERT_TRUE(mlir::isa<mlir::FileLineColLoc>(loc));
  auto file_line_col_loc = mlir::cast<mlir::FileLineColLoc>(loc);
  EXPECT_EQ(file_line_col_loc.getFilename(), "test.cc");
  EXPECT_EQ(file_line_col_loc.getLine(), line);
  EXPECT_EQ(file_line_col_loc.getColumn(), column);
}

TEST(DTensorLocationTest, HandlesEmptyLocation) {
  mlir::MLIRContext ctx;
  mlir::Location loc = mlir::FileLineColLoc::get(&ctx, "test.cc", 10, 20);
  loc = tensorflow::dtensor::DTensorLocation(loc, "test.cc", 21);

  ASSERT_TRUE(mlir::isa<mlir::CallSiteLoc>(loc));
  auto callsite_loc = mlir::cast<mlir::CallSiteLoc>(loc);
  CheckFileLineColLocation(callsite_loc.getCallee(), 21, 0);
  CheckFileLineColLocation(callsite_loc.getCaller(), 10, 20);

  constexpr char stack[] = R"stack(>> test.cc:10:20
>> test.cc:21:0)stack";
  EXPECT_EQ(tensorflow::dtensor::DTensorLocationToString(loc), stack);
}

TEST(DTensorLocationTest, HandlesMultipleCalls) {
  mlir::MLIRContext ctx;
  mlir::Location test_loc = mlir::FileLineColLoc::get(&ctx, "test.cc", 10, 20);
  test_loc = tensorflow::dtensor::DTensorLocation(test_loc, "test.cc", 21);
  test_loc = tensorflow::dtensor::DTensorLocation(test_loc, "test.cc", 22);
  test_loc = tensorflow::dtensor::DTensorLocation(test_loc, "test.cc", 23);
  test_loc = tensorflow::dtensor::DTensorLocation(test_loc, "test.cc", 24);

  auto verify_loc = test_loc;
  for (int i = 0; i < 4; ++i) {
    ASSERT_TRUE(mlir::isa<mlir::CallSiteLoc>(verify_loc));
    auto callsite_loc = mlir::cast<mlir::CallSiteLoc>(verify_loc);
    auto callee_loc = callsite_loc.getCallee();
    CheckFileLineColLocation(callee_loc, 24 - i, 0);
    verify_loc = callsite_loc.getCaller();
  }
  CheckFileLineColLocation(verify_loc, 10, 20);

  constexpr char stack[] = R"stack(>> test.cc:10:20
>> test.cc:21:0
>> test.cc:22:0
>> test.cc:23:0
>> test.cc:24:0)stack";
  EXPECT_EQ(tensorflow::dtensor::DTensorLocationToString(test_loc), stack);
}

TEST(DTensorLocationTest, HandlesNameLoc) {
  mlir::MLIRContext ctx;
  mlir::Location test_loc =
      mlir::NameLoc::get(mlir::StringAttr::get(&ctx, "op@"),
                         mlir::FileLineColLoc::get(&ctx, "test.cc", 10, 20));
  test_loc = tensorflow::dtensor::DTensorLocation(test_loc, "test.cc", 21);
  ASSERT_EQ(mlir::GetNameFromLoc(test_loc), "op");
  ASSERT_TRUE(mlir::isa<mlir::CallSiteLoc>(test_loc));
  auto callsite_loc = mlir::cast<mlir::CallSiteLoc>(test_loc);
  mlir::Location caller_loc =
      mlir::cast<mlir::CallSiteLoc>(test_loc).getCaller();
  ASSERT_TRUE(mlir::isa<mlir::NameLoc>(caller_loc));
  CheckFileLineColLocation(mlir::cast<mlir::NameLoc>(caller_loc).getChildLoc(),
                           10, 20);

  mlir::Location callee_loc = callsite_loc.getCallee();
  ASSERT_TRUE(mlir::isa<mlir::NameLoc>(callee_loc));
  CheckFileLineColLocation(mlir::cast<mlir::NameLoc>(callee_loc).getChildLoc(),
                           21, 0);

  constexpr char stack[] = R"stack(>> test.cc:10:20
>> test.cc:21:0)stack";
  EXPECT_EQ(tensorflow::dtensor::DTensorLocationToString(test_loc), stack);
}

TEST(DTensorLocationTest, HandlesNameLocWithName) {
  mlir::MLIRContext ctx;
  mlir::Location test_loc =
      mlir::NameLoc::get(mlir::StringAttr::get(&ctx, "op@"),
                         mlir::FileLineColLoc::get(&ctx, "test.cc", 10, 20));
  test_loc =
      tensorflow::dtensor::DTensorLocation(test_loc, "test.cc", 21, "nested");
  EXPECT_EQ(mlir::GetNameFromLoc(test_loc), "op/nested");
  constexpr char stack[] = R"stack(>> test.cc:10:20
>> test.cc:21:0)stack";
  EXPECT_EQ(tensorflow::dtensor::DTensorLocationToString(test_loc), stack);
}

}  // namespace
