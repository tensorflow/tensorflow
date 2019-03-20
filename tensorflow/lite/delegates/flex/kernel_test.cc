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
using ::testing::ElementsAreArray;

TfLiteStatus GenericPrepare(TfLiteContext* context, TfLiteDelegate* delegate,
                            const std::vector<int>& supported_nodes) {
  TfLiteIntArray* size_and_nodes =
      ConvertVectorToTfLiteIntArray(supported_nodes);
  TF_LITE_ENSURE_STATUS(context->ReplaceNodeSubsetsWithDelegateKernels(
      context, flex::GetKernel(), size_and_nodes, delegate));
  TfLiteIntArrayFree(size_and_nodes);
  return kTfLiteOk;
}

// There is no easy way to pass a parameter into the TfLiteDelegate's
// 'prepare' function, so we keep a global map for testing purpused.
// To avoid collisions use: GetPrepareFunction<__LINE__>().
std::map<int, std::vector<int>>* GetGlobalOpLists() {
  static auto* op_list = new std::map<int, std::vector<int>>;
  return op_list;
}

class KernelTest : public testing::FlexModelTest {
 public:
  static constexpr int kOnes = 1;  // This is the index of a tensor of 1's.
  static constexpr int kTwos = 2;  // This is the index of a tensor of 2's.
  static constexpr int kMaxTensors = 30;

  static void SetUpTestSuite() { GetGlobalOpLists()->clear(); }

  KernelTest() {
    CHECK(delegate_data_.Prepare(tensorflow::SessionOptions{}).ok());
    interpreter_.reset(new Interpreter(&error_reporter_));
  }

  typedef TfLiteStatus (*PrepareFunction)(TfLiteContext* context,
                                          TfLiteDelegate* delegate);

  template <int KEY>
  PrepareFunction GetPrepareFunction() {
    GetGlobalOpLists()->insert({KEY, tf_ops_});
    return [](TfLiteContext* context, TfLiteDelegate* delegate) {
      return GenericPrepare(context, delegate, GetGlobalOpLists()->at(KEY));
    };
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
      auto* buffer_map = delegate_data->GetBufferMap(context);
      if (!buffer_map->HasTensor(buffer_handle)) {
        context->ReportError(context, "Tensor '%d' not found", buffer_handle);
        return kTfLiteError;
      }
      tensorflow::StringPiece values =
          buffer_map->GetTensor(buffer_handle).tensor_data();
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

  ASSERT_NE(interpreter_->AllocateTensors(), kTfLiteOk);
  ASSERT_THAT(error_reporter().error_messages(),
              ContainsRegex("Op type not registered 'NonExistentOp'"));
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
    // All ops but #3 are TF ops, handled by the delegate. However, because #4
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

class MultipleSubgraphsTest : public KernelTest {
 public:
  static constexpr int kInput = 0;

  void PrepareInterpreter(PrepareFunction prepare,
                          const std::vector<float>& input) {
    ConfigureDelegate(prepare);

    SetShape(kOnes, {3});
    SetValues(kOnes, {1.0f, 1.0f, 1.0f});
    SetShape(kTwos, {3});
    SetValues(kTwos, {2.0f, 2.0f, 2.0f});

    SetValues(kInput, input);
  }

  std::vector<float> Apply(const std::vector<float>& input,
                           std::function<float(float)> function) {
    std::vector<float> result;
    for (float f : input) {
      result.push_back(function(f));
    }
    return result;
  }
};

TEST_F(MultipleSubgraphsTest, ForwardabilityIsLocal) {
  AddTensors(kMaxTensors, {kInput, kOnes, kTwos}, {12}, kTfLiteFloat32, {3});

  // Only TF tensors can be forwarded, so we build a small first graph
  // to produce tensor #10. Here #10 is forwardable, because it is only
  // used once, as an output.
  AddTfOp(testing::kAdd, {0, kOnes}, {3});
  AddTfOp(testing::kAdd, {0, kOnes}, {10});

  // The second TF graph, separated from the former by a TF Lite
  // multiplication, will consume tensor #10, which is not forwardable here
  // since it is used by more than one op. The existing code will forward the
  // tensor anyway, because it was deemed to be forwardable by the previous
  // subgraph.
  AddTfLiteMulOp({3, kTwos}, {4});
  AddTfOp(testing::kAdd, {10, 4}, {11});
  AddTfOp(testing::kAdd, {11, 10}, {7});

  // And a simple TF Lite op trying to access tensor #10, which was removed
  // from the buffer map. It will cause Invoke() to fail.
  AddTfLiteMulOp({10, 7}, {12});

  auto input = {3.0f, 4.0f, 5.0f};
  PrepareInterpreter(GetPrepareFunction<__LINE__>(), input);

  ASSERT_TRUE(Invoke());
  ASSERT_THAT(GetValues(12), ElementsAreArray(Apply(input, [](float in) {
                return (4 * in + 4) * (in + 1);
              })));
}

// Subgraphs should not remove input tensors from the buffer_map, since
// they could be necessary for downstream graphs.
TEST_F(MultipleSubgraphsTest, DoNotRemoveInputTensors) {
  AddTensors(kMaxTensors, {kInput, kOnes, kTwos}, {12}, kTfLiteFloat32, {3});

  // Only TF tensors can be removed, so we build a small first graph
  // to produce tensor #10. We make sure it is used by more than one
  // op, so it is not forwardable here.
  AddTfOp(testing::kAdd, {0, kOnes}, {3});
  AddTfOp(testing::kAdd, {0, kOnes}, {10});
  AddTfOp(testing::kAdd, {10, kOnes}, {15});
  AddTfOp(testing::kAdd, {10, kOnes}, {16});

  // The second TF graph, separated from the former by a TF Lite
  // multiplication, will consume tensor #10. The existing code will remove
  // from the buffer_map all tensors that are not outputs, so #10 will
  // disappear. Note that we are using #10 in two ops, so it is not forwardable
  // either.
  AddTfLiteMulOp({3, kTwos}, {4});
  AddTfOp(testing::kAdd, {10, 4}, {11});
  AddTfOp(testing::kAdd, {10, 11}, {7});

  // And a simple TF Lite op trying to access tensor #10, which was removed
  // from the buffer map. It will cause Invoke() to fail.
  AddTfLiteMulOp({10, 7}, {12});

  auto input = {3.0f, 4.0f, 5.0f};
  PrepareInterpreter(GetPrepareFunction<__LINE__>(), input);

  ASSERT_TRUE(Invoke());
  ASSERT_THAT(GetValues(12), ElementsAreArray(Apply(input, [](float in) {
                return (4 * in + 4) * (in + 1);
              })));
}

// A tensor is deemed forwardable but it happens to be the input to
// more than one subgraph. It should not be forwarded, otherwise its
// contents will be overwritten.
TEST_F(MultipleSubgraphsTest, DoNotForwardInputTensors) {
  AddTensors(kMaxTensors, {kInput, kOnes, kTwos}, {12}, kTfLiteFloat32, {3});

  // Only TF tensors can be forwarded, so we build a small first graph
  // to produce tensor #10.
  AddTfOp(testing::kAdd, {0, kOnes}, {3});
  AddTfOp(testing::kAdd, {0, kOnes}, {10});

  // The second TF graph, separated from the former by a TF Lite
  // multiplication, will consume tensor #10 and will think it is forwardable
  // because it is used by a single op. However, the subgraph doesn't have
  // enough information to make that judgment, as the input tensor could be
  // used by another graph further downstream. The existing code will forward
  // the tensor and remove it from the buffer_map, causing a failure later.
  AddTfLiteMulOp({3, kTwos}, {4});
  AddTfOp(testing::kAdd, {10, 4}, {11});
  AddTfOp(testing::kAdd, {11, 4}, {7});

  // And a simple TF Lite op trying to access tensor #10, which was removed
  // from the buffer map. It will cause Invoke() to fail.
  AddTfLiteMulOp({10, 7}, {12});

  auto input = {3.0f, 4.0f, 5.0f};
  PrepareInterpreter(GetPrepareFunction<__LINE__>(), input);

  ASSERT_TRUE(Invoke());
  ASSERT_THAT(GetValues(12), ElementsAreArray(Apply(input, [](float in) {
                return (5 * in + 5) * (in + 1);
              })));
}

}  // namespace
}  // namespace flex
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
