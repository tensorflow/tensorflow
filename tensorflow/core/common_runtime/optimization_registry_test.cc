/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/optimization_registry.h"

#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

class TestOptimization : public GraphOptimizationPass {
 public:
  static int count_;
  Status Run(const GraphOptimizationPassOptions& options) override {
    ++count_;
    return Status::OK();
  }
};

int TestOptimization::count_ = 0;

REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 1,
                      TestOptimization);

TEST(OptimizationRegistry, OptimizationPass) {
  EXPECT_EQ(0, TestOptimization::count_);
  GraphOptimizationPassOptions options;
  EXPECT_EQ(Status::OK(),
            OptimizationPassRegistry::Global()->RunGrouping(
                OptimizationPassRegistry::PRE_PLACEMENT, options));
  EXPECT_EQ(1, TestOptimization::count_);
}

class UpdateFuncLibPass : public GraphOptimizationPass {
 public:
  Status Run(const GraphOptimizationPassOptions& options) override {
    return options.flib_def->AddFunctionDef(test::function::WXPlusB());
  }
};

REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, 1,
                      UpdateFuncLibPass);

class OptimizationPassTest : public ::testing::Test {
 public:
  OptimizationPassTest() {
    FunctionDefLibrary func_def_lib;
    *func_def_lib.add_function() = test::function::XTimesTwo();
    flib_def_.reset(
        new FunctionLibraryDefinition(OpRegistry::Global(), func_def_lib));
  }

  void RunPass() {
    GraphOptimizationPassOptions options;
    options.flib_def = flib_def_.get();
    EXPECT_EQ(Status::OK(),
              OptimizationPassRegistry::Global()->RunGrouping(
                  OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, options));
  }

  const FunctionDef* GetFunctionDef(const string& func) const {
    return flib_def_->Find(func);
  }

 private:
  std::unique_ptr<FunctionLibraryDefinition> flib_def_;
};

TEST_F(OptimizationPassTest, UpdateFuncLibPass) {
  RunPass();
  auto f1 = GetFunctionDef("XTimesTwo");
  ASSERT_NE(f1, nullptr);
  EXPECT_EQ(test::function::XTimesTwo().DebugString(), f1->DebugString());

  auto f2 = GetFunctionDef("WXPlusB");
  ASSERT_NE(f2, nullptr);
  EXPECT_EQ(test::function::WXPlusB().DebugString(), f2->DebugString());
}

}  // namespace tensorflow
