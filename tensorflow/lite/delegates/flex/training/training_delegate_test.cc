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

#include "tensorflow/lite/delegates/flex/training/training_delegate.h"

#include <cstdint>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/flex/test_util.h"
#include "tensorflow/lite/shared_library.h"

namespace tflite {
namespace flex {
namespace testing {

using ::testing::ElementsAre;

class TrainingDelegateTest : public testing::FlexModelTest {
 protected:
  void SetUp() override {
    delegate_ = absl::make_unique<TrainingFlexDelegate>();
    interpreter_ = absl::make_unique<Interpreter>(&error_reporter_);
    interpreter_->SetCancellationFunction(delegate_.get(),
                                          TrainingFlexDelegate::ShouldCancel);
  }

  void TearDown() override {
    // The delegate needs to be destructed after the interpreter because the
    // interpreter references data contained in the delegate.
    interpreter_.reset();
    delegate_.reset();
  }

 public:
  TrainingDelegateTest() : delegate_(nullptr) {}

  void ConfigureDelegate() {
    ASSERT_EQ(
        interpreter_->ModifyGraphWithDelegate(delegate_->GetTfLiteDelegate()),
        kTfLiteOk);
  }

  void Cancel() { delegate_->Cancel(); }

 private:
  std::unique_ptr<TrainingFlexDelegate> delegate_;
};

TEST_F(TrainingDelegateTest, TestFullGraph) {
  AddTensors(3, {0, 1}, {2}, kTfLiteInt32, {2});

  AddTfOp(testing::kAdd, {0, 1}, {2});

  ConfigureDelegate();

  SetShape(0, {2, 2});
  SetTypedValues<int>(0, {1, 2, 3, 4});
  SetShape(1, {2, 2});
  SetTypedValues<int>(1, {4, 3, 2, 1});

  ASSERT_TRUE(Invoke());

  ASSERT_THAT(GetShape(2), ElementsAre(2, 2));
  ASSERT_THAT(GetTypedValues<int>(2), ElementsAre(5, 5, 5, 5));
  ASSERT_EQ(GetType(2), kTfLiteInt32);
}

TEST_F(TrainingDelegateTest, TestCancellation1) {
  AddTensors(3, {0, 1}, {2}, kTfLiteInt32, {2});

  AddTfOp(testing::kAdd, {0, 1}, {2});

  ConfigureDelegate();

  SetShape(0, {2, 2});
  SetTypedValues<int>(0, {1, 2, 3, 4});
  SetShape(1, {2, 2});
  SetTypedValues<int>(1, {4, 3, 2, 1});

  ASSERT_TRUE(Invoke());

  ASSERT_THAT(GetShape(2), ElementsAre(2, 2));
  ASSERT_THAT(GetTypedValues<int>(2), ElementsAre(5, 5, 5, 5));
  ASSERT_EQ(GetType(2), kTfLiteInt32);

  Cancel();
  // Op should be cancelled.
  ASSERT_FALSE(Invoke());
}

TEST_F(TrainingDelegateTest, TestCancellation2) {
  // Define the graph.
  AddTensors(2, {0}, {1}, kTfLiteBool, {1});

  // We need an op that checks the CancellationManager status.
  AddTfOp(testing::kLoopCond, {0}, {1});

  // Apply the delegate.
  ConfigureDelegate();

  // Define inputs.
  SetShape(0, {1});

  ASSERT_TRUE(Invoke());

  Cancel();
  // Op should be cancelled.
  ASSERT_FALSE(Invoke());
}

TEST_F(TrainingDelegateTest, TestCancellationTwoThreads) {
  AddTensors(3, {0, 1}, {2}, kTfLiteInt32, {2});

  AddTfOp(testing::kAdd, {0, 1}, {2});

  ConfigureDelegate();

  SetShape(0, {2, 2});
  SetTypedValues<int>(0, {1, 2, 3, 4});
  SetShape(1, {2, 2});
  SetTypedValues<int>(1, {4, 3, 2, 1});

  std::thread invoke_thread([this]() {
    bool result = true;
    result = this->Invoke();
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    result = this->Invoke();
    ASSERT_FALSE(result);
  });

  std::thread cancel_thread([this]() { this->Cancel(); });

  invoke_thread.join();
  cancel_thread.join();
}

// TODO(b/179048124): Add test case with ReduceDataset op.
// TODO(b/179048124): Add integration test with real models.
// TODO(b/179048124): Add proper test with multiple threads.

}  // namespace testing
}  // namespace flex
}  // namespace tflite
