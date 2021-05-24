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

#include "tensorflow/lite/kernels/gradient/bcast_grad_args.h"

#include <cstdint>
#include <vector>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/testing/util.h"

namespace tflite {
namespace ops {
namespace custom {

namespace {

using testing::ElementsAreArray;

class BcastGradArgsInt32OpModel : public SingleOpModel {
 public:
  BcastGradArgsInt32OpModel(const TensorData& input1, const TensorData& input2,
                            const TensorData& output1,
                            const TensorData& output2) {
    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output1_ = AddOutput(output1);
    output2_ = AddOutput(output2);

    std::vector<uint8_t> custom_option;
    SetCustomOp("BroadcastGradientArgs", custom_option,
                Register_BROADCAST_GRADIENT_ARGS);
    BuildInterpreter({GetShape(input1_), GetShape(input2_)});
  }

  void SetInput1(const std::vector<int>& data) {
    PopulateTensor(input1_, data);
  }
  void SetInput2(const std::vector<int>& data) {
    PopulateTensor(input2_, data);
  }

  std::vector<int> GetOutput1() { return ExtractVector<int>(output1_); }
  std::vector<int> GetOutput1Shape() { return GetTensorShape(output1_); }
  std::vector<int> GetOutput2() { return ExtractVector<int>(output2_); }
  std::vector<int> GetOutput2Shape() { return GetTensorShape(output2_); }

 protected:
  int input1_;
  int input2_;
  int output1_;
  int output2_;
};

TEST(BcastGradArgsInt32OpModel, AllEqualsInt32DTypes) {
  BcastGradArgsInt32OpModel model(
      /*input1=*/{TensorType_INT32, {4}},
      /*input2=*/{TensorType_INT32, {4}},
      /*output1=*/{TensorType_INT32, {}},
      /*output2=*/{TensorType_INT32, {}});
  model.SetInput1({3, 1, 2, 3});
  model.SetInput2({3, 1, 2, 3});
  model.Invoke();

  EXPECT_THAT(model.GetOutput1().size(), 0);
  EXPECT_THAT(model.GetOutput2().size(), 0);
}

TEST(BcastGradArgsInt32OpModel, BroadcastableDimAtInput1Int32DTypes) {
  BcastGradArgsInt32OpModel model(
      /*input1=*/{TensorType_INT32, {4}},
      /*input2=*/{TensorType_INT32, {4}},
      /*output1=*/{TensorType_INT32, {}},
      /*output2=*/{TensorType_INT32, {}});
  model.SetInput1({3, 4, 1, 3});
  model.SetInput2({3, 4, 2, 3});
  model.Invoke();

  EXPECT_THAT(model.GetOutput1(), ElementsAreArray({2}));
  EXPECT_THAT(model.GetOutput2().size(), 0);
}

TEST(BcastGradArgsInt32OpModel, BroadcastableDimAtInput2Int32DTypes) {
  BcastGradArgsInt32OpModel model(
      /*input1=*/{TensorType_INT32, {4}},
      /*input2=*/{TensorType_INT32, {4}},
      /*output1=*/{TensorType_INT32, {}},
      /*output2=*/{TensorType_INT32, {}});
  model.SetInput1({3, 4, 2, 3});
  model.SetInput2({3, 1, 2, 3});
  model.Invoke();

  EXPECT_THAT(model.GetOutput1().size(), 0);
  EXPECT_THAT(model.GetOutput2(), ElementsAreArray({1}));
}

TEST(BcastGradArgsInt32OpModel, DifferentInputSizesInt32DTypes) {
  BcastGradArgsInt32OpModel model(
      /*input1=*/{TensorType_INT32, {4}},
      /*input2=*/{TensorType_INT32, {3}},
      /*output1=*/{TensorType_INT32, {}},
      /*output2=*/{TensorType_INT32, {}});
  model.SetInput1({3, 4, 2, 3});
  model.SetInput2({4, 2, 3});
  model.Invoke();

  EXPECT_THAT(model.GetOutput1().size(), 0);
  EXPECT_THAT(model.GetOutput2(), ElementsAreArray({0}));
}

TEST(BcastGradArgsInt32OpModel, NonBroadcastableDimsInt32DTypes) {
  BcastGradArgsInt32OpModel model(
      /*input1=*/{TensorType_INT32, {4}},
      /*input2=*/{TensorType_INT32, {4}},
      /*output1=*/{TensorType_INT32, {}},
      /*output2=*/{TensorType_INT32, {}});
  model.SetInput1({3, 4, 2, 3});
  model.SetInput2({9, 9, 9, 9});
  EXPECT_THAT(model.InvokeUnchecked(), kTfLiteError);
}

class BcastGradArgsInt64OpModel : public SingleOpModel {
 public:
  BcastGradArgsInt64OpModel(const TensorData& input1, const TensorData& input2,
                            const TensorData& output1,
                            const TensorData& output2) {
    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output1_ = AddOutput(output1);
    output2_ = AddOutput(output2);

    std::vector<uint8_t> custom_option;
    SetCustomOp("BroadcastGradientArgs", custom_option,
                Register_BROADCAST_GRADIENT_ARGS);
    BuildInterpreter({GetShape(input1_), GetShape(input2_)});
  }

  void SetInput1(const std::vector<int64_t>& data) {
    PopulateTensor(input1_, data);
  }
  void SetInput2(const std::vector<int64_t>& data) {
    PopulateTensor(input2_, data);
  }

  std::vector<int64_t> GetOutput1() { return ExtractVector<int64_t>(output1_); }
  std::vector<int> GetOutput1Shape() { return GetTensorShape(output1_); }
  std::vector<int64_t> GetOutput2() { return ExtractVector<int64_t>(output2_); }
  std::vector<int> GetOutput2Shape() { return GetTensorShape(output2_); }

 protected:
  int input1_;
  int input2_;
  int output1_;
  int output2_;
};

TEST(BcastGradArgsInt32OpModel, AllEqualsInt64DTypes) {
  BcastGradArgsInt64OpModel model(
      /*input1=*/{TensorType_INT64, {4}},
      /*input2=*/{TensorType_INT64, {4}},
      /*output1=*/{TensorType_INT64, {}},
      /*output2=*/{TensorType_INT64, {}});
  model.SetInput1({3, 1, 2, 3});
  model.SetInput2({3, 1, 2, 3});
  model.Invoke();

  EXPECT_THAT(model.GetOutput1().size(), 0);
  EXPECT_THAT(model.GetOutput2().size(), 0);
}

TEST(BcastGradArgsInt32OpModel, BroadcastableDimAtInput1Int64DTypes) {
  BcastGradArgsInt64OpModel model(
      /*input1=*/{TensorType_INT64, {4}},
      /*input2=*/{TensorType_INT64, {4}},
      /*output1=*/{TensorType_INT64, {}},
      /*output2=*/{TensorType_INT64, {}});
  model.SetInput1({3, 4, 1, 3});
  model.SetInput2({3, 4, 2, 3});
  model.Invoke();

  EXPECT_THAT(model.GetOutput1(), ElementsAreArray({2}));
  EXPECT_THAT(model.GetOutput2().size(), 0);
}

TEST(BcastGradArgsInt32OpModel, BroadcastableDimAtInput2Int64DTypes) {
  BcastGradArgsInt64OpModel model(
      /*input1=*/{TensorType_INT64, {4}},
      /*input2=*/{TensorType_INT64, {4}},
      /*output1=*/{TensorType_INT64, {}},
      /*output2=*/{TensorType_INT64, {}});
  model.SetInput1({3, 4, 2, 3});
  model.SetInput2({3, 1, 2, 3});
  model.Invoke();

  EXPECT_THAT(model.GetOutput1().size(), 0);
  EXPECT_THAT(model.GetOutput2(), ElementsAreArray({1}));
}

TEST(BcastGradArgsInt32OpModel, DifferentInputSizesInt64DTypes) {
  BcastGradArgsInt64OpModel model(
      /*input1=*/{TensorType_INT64, {4}},
      /*input2=*/{TensorType_INT64, {3}},
      /*output1=*/{TensorType_INT64, {}},
      /*output2=*/{TensorType_INT64, {}});
  model.SetInput1({3, 4, 2, 3});
  model.SetInput2({4, 2, 3});
  model.Invoke();

  EXPECT_THAT(model.GetOutput1().size(), 0);
  EXPECT_THAT(model.GetOutput2(), ElementsAreArray({0}));
}

TEST(BcastGradArgsInt32OpModel, NonBroadcastableDimsInt64DTypes) {
  BcastGradArgsInt64OpModel model(
      /*input1=*/{TensorType_INT64, {4}},
      /*input2=*/{TensorType_INT64, {4}},
      /*output1=*/{TensorType_INT64, {}},
      /*output2=*/{TensorType_INT64, {}});
  model.SetInput1({3, 4, 2, 3});
  model.SetInput2({9, 9, 9, 9});
  EXPECT_THAT(model.InvokeUnchecked(), kTfLiteError);
}

}  // namespace
}  // namespace custom
}  // namespace ops
}  // namespace tflite
