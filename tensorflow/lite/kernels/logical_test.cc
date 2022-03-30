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
#include <initializer_list>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAre;

class LogicalOpModel : public SingleOpModel {
 public:
  LogicalOpModel(std::initializer_list<int> input1_shape,
                 std::initializer_list<int> input2_shape, BuiltinOperator op) {
    input1_ = AddInput(TensorType_BOOL);
    input2_ = AddInput(TensorType_BOOL);
    output_ = AddOutput(TensorType_BOOL);
    ConfigureBuiltinOp(op);
    BuildInterpreter({input1_shape, input2_shape});
  }

  int input1() { return input1_; }
  int input2() { return input2_; }

  std::vector<bool> GetOutput() { return ExtractVector<bool>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 private:
  int input1_;
  int input2_;
  int output_;

  void ConfigureBuiltinOp(BuiltinOperator op) {
    switch (op) {
      case BuiltinOperator_LOGICAL_OR: {
        SetBuiltinOp(op, BuiltinOptions_LogicalOrOptions,
                     CreateLogicalOrOptions(builder_).Union());
        break;
      }
      case BuiltinOperator_LOGICAL_AND: {
        SetBuiltinOp(op, BuiltinOptions_LogicalAndOptions,
                     CreateLogicalAndOptions(builder_).Union());
        break;
      }
      default: { FAIL() << "We shouldn't get here."; }
    }
  }
};

TEST(LogicalTest, LogicalOr) {
  LogicalOpModel model({1, 1, 1, 4}, {1, 1, 1, 4}, BuiltinOperator_LOGICAL_OR);
  model.PopulateTensor<bool>(model.input1(), {true, false, false, true});
  model.PopulateTensor<bool>(model.input2(), {true, false, true, false});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput(), ElementsAre(true, false, true, true));
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 1, 1, 4));
}

TEST(LogicalTest, BroadcastLogicalOr) {
  LogicalOpModel model({1, 1, 1, 4}, {1, 1, 1, 1}, BuiltinOperator_LOGICAL_OR);
  model.PopulateTensor<bool>(model.input1(), {true, false, false, true});
  model.PopulateTensor<bool>(model.input2(), {false});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput(), ElementsAre(true, false, false, true));
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 1, 1, 4));
}

TEST(LogicalTest, LogicalAnd) {
  LogicalOpModel model({1, 1, 1, 4}, {1, 1, 1, 4}, BuiltinOperator_LOGICAL_AND);
  model.PopulateTensor<bool>(model.input1(), {true, false, false, true});
  model.PopulateTensor<bool>(model.input2(), {true, false, true, false});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput(), ElementsAre(true, false, false, false));
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 1, 1, 4));
}

TEST(LogicalTest, BroadcastLogicalAnd) {
  LogicalOpModel model({1, 1, 1, 4}, {1, 1, 1, 1}, BuiltinOperator_LOGICAL_AND);
  model.PopulateTensor<bool>(model.input1(), {true, false, false, true});
  model.PopulateTensor<bool>(model.input2(), {true});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutput(), ElementsAre(true, false, false, true));
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1, 1, 1, 4));
}

}  // namespace
}  // namespace tflite
