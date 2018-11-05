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
#include "tensorflow/lite/kernels/kernel_util.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/testing/util.h"

namespace tflite {
namespace {

void ReportError(TfLiteContext* context, const char* format, ...) {}

class KernelUtilTest : public ::testing::Test {
 public:
  KernelUtilTest() {
    context_.ReportError = ReportError;

    tensor1_.dims = nullptr;
    tensor2_.dims = nullptr;
    tensor1_.allocation_type = kTfLiteMmapRo;
    tensor2_.allocation_type = kTfLiteMmapRo;
  }
  ~KernelUtilTest() override {
    TfLiteTensorFree(&tensor1_);
    TfLiteTensorFree(&tensor2_);
  }

  void SetShape(TfLiteTensor* tensor, std::initializer_list<int> dims) {
    TfLiteTensorFree(tensor);
    tensor->dims = TfLiteIntArrayCreate(dims.size());
    int i = 0;
    for (int d : dims) {
      tensor->dims->data[i] = d;
      ++i;
    }
  }

  std::vector<int> GetShape(TfLiteIntArray* dims) {
    std::vector<int> result;
    for (int i = 0; i < dims->size; ++i) {
      result.push_back(dims->data[i]);
    }
    return result;
  }

 protected:
  TfLiteContext context_;
  TfLiteTensor tensor1_;
  TfLiteTensor tensor2_;
};

TEST_F(KernelUtilTest, SameShapeEmpty) {
  EXPECT_TRUE(HaveSameShapes(&tensor1_, &tensor2_));

  SetShape(&tensor1_, {1, 2, 3});
  EXPECT_FALSE(HaveSameShapes(&tensor1_, &tensor2_));

  SetShape(&tensor2_, {1, 2});
  EXPECT_FALSE(HaveSameShapes(&tensor1_, &tensor2_));

  SetShape(&tensor2_, {1, 2, 3, 4});
  EXPECT_FALSE(HaveSameShapes(&tensor1_, &tensor2_));

  SetShape(&tensor2_, {1, 2, 3});
  EXPECT_TRUE(HaveSameShapes(&tensor1_, &tensor2_));

  SetShape(&tensor2_, {});
  EXPECT_FALSE(HaveSameShapes(&tensor1_, &tensor2_));

  SetShape(&tensor1_, {});
  EXPECT_TRUE(HaveSameShapes(&tensor1_, &tensor2_));
}

TEST_F(KernelUtilTest, BroadcastShapeIncompatibleDim) {
  TfLiteIntArray* output = nullptr;
  SetShape(&tensor1_, {1, 2});
  SetShape(&tensor2_, {1, 3});
  EXPECT_NE(kTfLiteOk, CalculateShapeForBroadcast(&context_, &tensor1_,
                                                  &tensor2_, &output));
  EXPECT_EQ(output, nullptr);
}

TEST_F(KernelUtilTest, BroadcastShapeOnes) {
  TfLiteIntArray* output = nullptr;
  SetShape(&tensor1_, {1, 1});
  SetShape(&tensor2_, {1, 3});
  EXPECT_EQ(kTfLiteOk, CalculateShapeForBroadcast(&context_, &tensor1_,
                                                  &tensor2_, &output));
  TfLiteIntArrayFree(output);

  SetShape(&tensor1_, {1, 2});
  SetShape(&tensor2_, {1, 1});
  EXPECT_EQ(kTfLiteOk, CalculateShapeForBroadcast(&context_, &tensor1_,
                                                  &tensor2_, &output));
  TfLiteIntArrayFree(output);
}

TEST_F(KernelUtilTest, BroadcastShapeScalars) {
  TfLiteIntArray* output = nullptr;
  SetShape(&tensor1_, {1, 2});
  SetShape(&tensor2_, {});
  EXPECT_EQ(kTfLiteOk, CalculateShapeForBroadcast(&context_, &tensor1_,
                                                  &tensor2_, &output));
  EXPECT_THAT(GetShape(output), ::testing::ElementsAre(1, 2));
  TfLiteIntArrayFree(output);

  SetShape(&tensor1_, {});
  SetShape(&tensor2_, {2});
  EXPECT_EQ(kTfLiteOk, CalculateShapeForBroadcast(&context_, &tensor1_,
                                                  &tensor2_, &output));
  EXPECT_THAT(GetShape(output), ::testing::ElementsAre(2));
  TfLiteIntArrayFree(output);
}

TEST_F(KernelUtilTest, BroadcastShapeDifferentSizes) {
  TfLiteIntArray* output = nullptr;
  SetShape(&tensor1_, {1, 2});
  SetShape(&tensor2_, {3, 1, 1});
  EXPECT_EQ(kTfLiteOk, CalculateShapeForBroadcast(&context_, &tensor1_,
                                                  &tensor2_, &output));
  EXPECT_THAT(GetShape(output), ::testing::ElementsAre(3, 1, 2));
  TfLiteIntArrayFree(output);

  SetShape(&tensor1_, {1, 2, 3, 4});
  SetShape(&tensor2_, {1, 3, 1});
  EXPECT_EQ(kTfLiteOk, CalculateShapeForBroadcast(&context_, &tensor1_,
                                                  &tensor2_, &output));
  EXPECT_THAT(GetShape(output), ::testing::ElementsAre(1, 2, 3, 4));
  TfLiteIntArrayFree(output);
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
