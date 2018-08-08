/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/contrib/lite/delegates/eager/delegate.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/contrib/lite/delegates/eager/test_util.h"

namespace tflite {
namespace eager {
namespace {

using ::testing::ContainsRegex;
using ::testing::ElementsAre;

// TODO(nupurgarg): Add a test with multiple interpreters for one delegate.

class DelegateTest : public testing::EagerModelTest {
 public:
  DelegateTest() {
    // The delegate needs to be constructed before the interpreter because the
    // interpreter references data contained in the delegate.
    delegate_.reset(new EagerDelegate());
    interpreter_.reset(new Interpreter(&error_reporter_));
  }

  ~DelegateTest() override {
    // The delegate needs to be destructed after the interpreter because the
    // interpreter references data contained in the delegate.
    delete interpreter_.release();
    delete delegate_.release();
  }

  void ConfigureDelegate() {
    CHECK(delegate_->Apply(interpreter_.get()) == kTfLiteOk);
  }

 private:
  std::unique_ptr<EagerDelegate> delegate_;
};

TEST_F(DelegateTest, FullGraph) {
  // Define the graph.
  AddTensors(9, {0, 3}, {8}, kTfLiteFloat32, {3});

  AddTfOp(testing::kUnpack, {0}, {1, 2});
  AddTfOp(testing::kUnpack, {3}, {4, 5});
  AddTfOp(testing::kAdd, {1, 4}, {6});
  AddTfOp(testing::kAdd, {2, 5}, {7});
  AddTfOp(testing::kMul, {6, 7}, {8});

  // Apply the delegate.
  ConfigureDelegate();

  // Define inputs.
  SetShape(0, {2, 2, 1});
  SetValues(0, {1.1f, 2.2f, 3.3f, 4.4f});
  SetShape(3, {2, 2, 1});
  SetValues(3, {1.1f, 2.2f, 3.3f, 4.4f});

  ASSERT_TRUE(Invoke());

  ASSERT_THAT(GetShape(8), ElementsAre(2, 1));
  ASSERT_THAT(GetValues(8), ElementsAre(14.52f, 38.72f));
}

TEST_F(DelegateTest, MixedGraph) {
  AddTensors(9, {0, 3}, {8}, kTfLiteFloat32, {3});

  AddTfOp(testing::kUnpack, {0}, {1, 2});
  AddTfOp(testing::kUnpack, {3}, {4, 5});
  AddTfOp(testing::kAdd, {1, 4}, {6});
  AddTfOp(testing::kAdd, {2, 5}, {7});
  AddTfLiteMulOp({6, 7}, {8});

  ConfigureDelegate();

  SetShape(0, {2, 2, 1});
  SetValues(0, {1.1f, 2.2f, 3.3f, 4.4f});
  SetShape(3, {2, 2, 1});
  SetValues(3, {1.1f, 2.2f, 3.3f, 4.4f});

  ASSERT_TRUE(Invoke());

  ASSERT_THAT(GetShape(8), ElementsAre(2, 1));
  ASSERT_THAT(GetValues(8), ElementsAre(14.52f, 38.72f));
}

TEST_F(DelegateTest, SplitGraph) {
  AddTensors(10, {0}, {9}, kTfLiteFloat32, {3});

  AddTfOp(testing::kUnpack, {0}, {1, 2});
  AddTfOp(testing::kAdd, {1, 2}, {3});
  AddTfOp(testing::kUnpack, {3}, {4, 5});

  AddTfLiteMulOp({4, 5}, {6});

  AddTfOp(testing::kUnpack, {6}, {7, 8});
  AddTfOp(testing::kAdd, {7, 8}, {9});

  ConfigureDelegate();

  SetShape(0, {2, 2, 2, 1});
  SetValues(0, {3.0f, 1.0f, 0.5f, -1.0f, 0.0f, 1.0f, 1.5f, 3.0f});

  ASSERT_TRUE(Invoke());

  ASSERT_THAT(GetShape(9), ElementsAre(1));
  ASSERT_THAT(GetValues(9), ElementsAre(10.0f));
}

TEST_F(DelegateTest, OnlyTFLite) {
  // Only TFLite single op model.
  AddTensors(10, {0, 1}, {2}, kTfLiteFloat32, {3});
  AddTfLiteMulOp({0, 1}, {2});

  ConfigureDelegate();

  SetShape(0, {2, 2, 1});
  SetValues(0, {1.1f, 2.2f, 3.3f, 4.4f});
  SetShape(1, {2, 2, 1});
  SetValues(1, {1.0f, 2.0f, 3.0f, 4.0f});

  ASSERT_TRUE(Invoke());

  ASSERT_THAT(GetShape(2), ElementsAre(2, 2, 1));
  ASSERT_THAT(GetValues(2), ElementsAre(1.1f, 4.4f, 9.9f, 17.6f));
}

}  // namespace
}  // namespace eager
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
