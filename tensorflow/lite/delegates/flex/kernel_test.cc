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
#include "tensorflow/lite/delegates/flex/kernel.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/flex/delegate_data.h"
#include "tensorflow/lite/delegates/flex/test_util.h"

namespace tflite {
namespace flex {
namespace {

using ::testing::ContainsRegex;
using ::testing::ElementsAre;

TfLiteStatus GenericPrepare(TfLiteContext* context, TfLiteDelegate* delegate,
                            const std::vector<int>& supported_nodes) {
  TfLiteIntArray* size_and_nodes =
      ConvertVectorToTfLiteIntArray(supported_nodes);
  TF_LITE_ENSURE_STATUS(context->ReplaceNodeSubsetsWithDelegateKernels(
      context, flex::GetKernel(), size_and_nodes, delegate));
  TfLiteIntArrayFree(size_and_nodes);
  return kTfLiteOk;
}

class KernelTest : public testing::FlexModelTest {
 public:
  KernelTest() {
    CHECK(delegate_data_.Prepare(tensorflow::SessionOptions{}).ok());
    interpreter_.reset(new Interpreter(&error_reporter_));
  }

  template <typename T>
  void ConfigureDelegate(T prepare_function) {
    delegate_.data_ = &delegate_data_;
    delegate_.flags = kTfLiteDelegateFlagsAllowDynamicTensors;
    delegate_.FreeBufferHandle = nullptr;
    delegate_.Prepare = prepare_function;
    delegate_.CopyFromBufferHandle = [](TfLiteContext* context,
                                        TfLiteDelegate* delegate,
                                        TfLiteBufferHandle buffer_handle,
                                        TfLiteTensor* output) {
      auto* delegate_data = reinterpret_cast<DelegateData*>(delegate->data_);
      tensorflow::StringPiece values = delegate_data->GetBufferMap(context)
                                           ->GetTensor(buffer_handle)
                                           .tensor_data();
      memcpy(output->data.raw, values.data(), values.size());
      return kTfLiteOk;
    };
    CHECK(interpreter_->ModifyGraphWithDelegate(&delegate_) == kTfLiteOk);
  }

 private:
  DelegateData delegate_data_;
  TfLiteDelegate delegate_;
};

TEST_F(KernelTest, FullGraph) {
  // Define the graph.
  AddTensors(9, {0, 3}, {8}, kTfLiteFloat32, {3});

  AddTfOp(testing::kUnpack, {0}, {1, 2});
  AddTfOp(testing::kUnpack, {3}, {4, 5});
  AddTfOp(testing::kAdd, {1, 4}, {6});
  AddTfOp(testing::kAdd, {2, 5}, {7});
  AddTfOp(testing::kMul, {6, 7}, {8});

  // Apply Delegate.
  ConfigureDelegate([](TfLiteContext* context, TfLiteDelegate* delegate) {
    return GenericPrepare(context, delegate, {0, 1, 2, 3, 4});
  });

  // Define inputs.
  SetShape(0, {2, 2, 1});
  SetValues(0, {1.1f, 2.2f, 3.3f, 4.4f});
  SetShape(3, {2, 2, 1});
  SetValues(3, {1.1f, 2.2f, 3.3f, 4.4f});

  ASSERT_TRUE(Invoke());

  ASSERT_THAT(GetShape(8), ElementsAre(2, 1));
  ASSERT_THAT(GetValues(8), ElementsAre(14.52f, 38.72f));

  // Try again with different inputs
  SetShape(0, {2, 3, 1});
  SetValues(0, {2.0f, 2.0f, 3.0f, 3.0f, 4.0f, 4.0f});
  SetShape(3, {2, 3, 1});
  SetValues(3, {2.0f, 2.0f, 3.0f, 3.0f, 4.0f, 4.0f});

  ASSERT_TRUE(Invoke());

  ASSERT_THAT(GetShape(8), ElementsAre(3, 1));
  ASSERT_THAT(GetValues(8), ElementsAre(24.0f, 32.0f, 48.0f));
}

TEST_F(KernelTest, BadTensorFlowOp) {
  AddTensors(2, {0}, {1}, kTfLiteFloat32, {3});
  AddTfOp(testing::kNonExistent, {0}, {1});

  ConfigureDelegate([](TfLiteContext* context, TfLiteDelegate* delegate) {
    return GenericPrepare(context, delegate, {0});
  });

  SetShape(0, {2, 2, 1});
  SetValues(0, {1.1f, 2.2f, 3.3f, 4.4f});

  ASSERT_FALSE(Invoke());
  ASSERT_THAT(error_reporter().error_messages(),
              ContainsRegex("while processing attributes of 'NonExistentOp'"));
}

TEST_F(KernelTest, BadNumberOfOutputs) {
  AddTensors(3, {0}, {1, 2}, kTfLiteFloat32, {3});
  AddTfOp(testing::kIdentity, {0}, {1, 2});

  ConfigureDelegate([](TfLiteContext* context, TfLiteDelegate* delegate) {
    return GenericPrepare(context, delegate, {0});
  });

  SetShape(0, {2, 2, 1});
  SetValues(0, {1.1f, 2.2f, 3.3f, 4.4f});

  ASSERT_FALSE(Invoke());
  ASSERT_THAT(error_reporter().error_messages(),
              ContainsRegex("Unexpected number of outputs"));
}

TEST_F(KernelTest, IncompatibleNodeDef) {
  AddTensors(2, {0}, {1}, kTfLiteFloat32, {3});

  // Cast is a TF op, but we don't add the proper nodedef to it in AddTfOp.
  AddTfOp(testing::kIncompatibleNodeDef, {0}, {1});

  ConfigureDelegate([](TfLiteContext* context, TfLiteDelegate* delegate) {
    return GenericPrepare(context, delegate, {0});
  });

  SetShape(0, {2, 2, 1});
  SetValues(0, {1.1f, 2.2f, 3.3f, 4.4f});

  ASSERT_FALSE(Invoke());
  ASSERT_THAT(error_reporter().error_messages(),
              ContainsRegex("while executing 'Cast' via Eager"));
}

TEST_F(KernelTest, WrongSetOfNodes) {
  AddTensors(4, {0}, {3}, kTfLiteFloat32, {3});
  AddTfOp(testing::kUnpack, {0}, {1, 2});
  AddTfLiteMulOp({1, 2}, {3});

  // Specify that testing::kMul (#1) is supported when it actually isn't.
  ConfigureDelegate([](TfLiteContext* context, TfLiteDelegate* delegate) {
    return GenericPrepare(context, delegate, {0, 1});
  });

  ASSERT_NE(interpreter_->AllocateTensors(), kTfLiteOk);
  ASSERT_THAT(error_reporter().error_messages(),
              ContainsRegex("Invalid NodeDef in Flex op"));
}

TEST_F(KernelTest, MixedGraph) {
  AddTensors(9, {0, 3}, {8}, kTfLiteFloat32, {3});

  AddTfOp(testing::kUnpack, {0}, {1, 2});
  AddTfOp(testing::kUnpack, {3}, {4, 5});
  AddTfOp(testing::kAdd, {1, 4}, {6});
  AddTfOp(testing::kAdd, {2, 5}, {7});
  AddTfLiteMulOp({6, 7}, {8});

  ConfigureDelegate([](TfLiteContext* context, TfLiteDelegate* delegate) {
    return GenericPrepare(context, delegate, {0, 1, 2, 3});
  });

  SetShape(0, {2, 2, 1});
  SetValues(0, {1.1f, 2.2f, 3.3f, 4.4f});
  SetShape(3, {2, 2, 1});
  SetValues(3, {1.1f, 2.2f, 3.3f, 4.4f});

  ASSERT_TRUE(Invoke());

  ASSERT_THAT(GetShape(8), ElementsAre(2, 1));
  ASSERT_THAT(GetValues(8), ElementsAre(14.52f, 38.72f));
}

// We will build a complex graph where most of the ops are TF ops, but one
// of them, right in the middle is handle natively by TF Lite. This results
// in two flex subgraphs to handle the TF ops, and some of the tensors
// connect those two subgraphs directly.
TEST_F(KernelTest, SplitGraph) {
  std::vector<float> a = {3.0f, 1.0f, 0.5f, -1.0f, 4.0f, -1.0f, -2.0f, 5.0f};
  std::vector<float> b = {0.0f, 1.0f, 1.5f, 3.0f};

  AddTensors(18, {0, 1}, {17}, kTfLiteFloat32, {3});

  // Split the first input. Each branch below uses one half of it.
  AddTfOp(testing::kUnpack, {0}, {2, 10});

  // The left branch: l = (a0 + b0) * (a2 + b2) + (a1 + b1) * (a3 + b3) = 10
  AddTfOp(testing::kAdd, {1, 2}, {3});     // => 3, 2, 2, 2
  AddTfOp(testing::kUnpack, {3}, {4, 5});  // => 3, 2 --- 2, 2
  AddTfLiteMulOp({4, 5}, {6});             // => 6, 4
  AddTfOp(testing::kUnpack, {6}, {7, 8});  // => 6 -- 4
  AddTfOp(testing::kAdd, {7, 8}, {9});     // => 10

  // The right branch: r = (a4 + a6) + (a5 + a7) = 6
  AddTfOp(testing::kUnpack, {10}, {11, 12});  // => 4, -1 --- -2, 5
  AddTfOp(testing::kAdd, {11, 12}, {13});     // => 2, 4
  AddTfOp(testing::kUnpack, {13}, {14, 15});  // => 2 --- 4
  AddTfOp(testing::kAdd, {14, 15}, {16});     // => 6

  // The two branches added together:
  AddTfOp(testing::kAdd, {9, 16}, {17});  // => 16

  ConfigureDelegate([](TfLiteContext* context, TfLiteDelegate* delegate) {
    // All ops by #3 are TF ops, handled by the delegate. However, because #4
    // depends on the non-TF op, two subgraphs are necessary:
    //    TF subgraph 1: 0, 1, 2, 6, 7, 8, 9
    //    TF Lite Op: 3
    //    TF subgraph 2: 4, 5, 10
    return GenericPrepare(context, delegate, {0, 1, 2, 4, 5, 6, 7, 8, 9, 10});
  });

  SetShape(0, {2, 2, 2, 1});
  SetValues(0, a);
  SetShape(1, {2, 2, 1});
  SetValues(1, b);

  ASSERT_TRUE(Invoke());

  ASSERT_THAT(GetShape(17), ElementsAre(1));
  ASSERT_THAT(GetValues(17), ElementsAre(16.0f));

  // Same as above but with slightly different output.
  // We still expect the result to be l + r where
  //     l = (a0 + b0) * (a2 + b2) + (a1 + b1) * (a3 + b3)
  //     r = (a4 + a6) + (a5 + a7)
  SetShape(0, {2, 2, 2, 1});
  SetValues(0, {4.0f, 1.0f, 1.5f, -2.0f, 2.0f, 0.0f, -2.0f, 3.0f});
  SetShape(1, {2, 2, 1});
  SetValues(1, {0.0f, 2.0f, 1.5f, 3.0f});
  // So l = (4 + 0) * (1.5 + 1.5) + (1 + 2) * (-2 + 3) =  12 + 3 = 15
  //    r = (2 - 2) + (0 + 3) = 3

  ASSERT_TRUE(Invoke());

  ASSERT_THAT(GetShape(17), ElementsAre(1));
  ASSERT_THAT(GetValues(17), ElementsAre(18.0f));
}

}  // namespace
}  // namespace flex
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
