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

#include <functional>
#include <initializer_list>
#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/delegates/flex/delegate.h"
#include "tensorflow/lite/delegates/flex/delegate_data.h"
#include "tensorflow/lite/delegates/flex/test_util.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace flex {
namespace testing {

using ::testing::ContainsRegex;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

// A testing flex delegate that supports every node regardless whether it's
// actually supported or not. It's only for testing certain scenarios.
class TestFlexDelegate : public FlexDelegate {
 protected:
  bool IsNodeSupportedByDelegate(const TfLiteRegistration* registration,
                                 const TfLiteNode* node,
                                 TfLiteContext* context) const override {
    return true;
  }
};

class KernelTest : public testing::FlexModelTest {
 public:
  static constexpr int kOnes = 1;  // This is the index of a tensor of 1's.
  static constexpr int kTwos = 2;  // This is the index of a tensor of 2's.
  static constexpr int kMaxTensors = 30;

  KernelTest() {
    interpreter_ = std::make_unique<Interpreter>(&error_reporter_);
  }

  TfLiteStatus ApplyFlexDelegate(
      std::unique_ptr<FlexDelegate> delegate = nullptr) {
    auto flex_delegate = FlexDelegate::Create(std::move(delegate));
    delegate_data_ =
        reinterpret_cast<FlexDelegate*>(flex_delegate->data_)->mutable_data();
    CHECK_OK(delegate_data_->Prepare(tensorflow::SessionOptions{}));
    return interpreter_->ModifyGraphWithDelegate(std::move(flex_delegate));
  }

  const std::map<int, int>& GetTensorReleaseMap(DelegateKernel* kernel) {
    return kernel->GetTensorReleaseMap();
  }

 protected:
  tflite::flex::DelegateData* delegate_data_;
};

TEST_F(KernelTest, FullGraph) {
  // Define the graph.
  AddTensors(9, {0, 3}, {8}, kTfLiteFloat32, {3});

  AddTfOp(testing::kUnpack, {0}, {1, 2});
  AddTfOp(testing::kUnpack, {3}, {4, 5});
  AddTfOp(testing::kAdd, {1, 4}, {6});
  AddTfOp(testing::kAdd, {2, 5}, {7});
  AddTfOp(testing::kMul, {6, 7}, {8});

  ApplyFlexDelegate();

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

TEST_F(KernelTest, ValidateTensorReleaseMap) {
  // Define the graph.
  //        0           3
  //        |           |
  //      Unpack_0    Unpack_1
  //       /  \        / \
  //      1    2      4   5
  //      |____|_______|__|
  //         | |__________|
  //         |      |
  //        Add_2  Add_3
  //         |      |
  //         6      7
  //         \______/
  //             |
  //            Mul_4
  //             |
  //             8
  AddTensors(9, {0, 3}, {8}, kTfLiteFloat32, {3});
  AddTfOp(testing::kUnpack, {0}, {1, 2});
  AddTfOp(testing::kUnpack, {3}, {4, 5});
  AddTfOp(testing::kAdd, {1, 4}, {6});
  AddTfOp(testing::kAdd, {2, 5}, {7});
  AddTfOp(testing::kMul, {6, 7}, {8});

  ApplyFlexDelegate();

  const int node_size = interpreter_->primary_subgraph().nodes_size();
  const std::pair<TfLiteNode, TfLiteRegistration>* node_and_reg =
      interpreter_->primary_subgraph().node_and_registration(node_size - 1);

  DelegateKernel* delegate_kernel =
      reinterpret_cast<DelegateKernel*>(node_and_reg->first.user_data);
  const auto& tensor_release_map = GetTensorReleaseMap(delegate_kernel);
  // Validate the tensor release mapping.
  EXPECT_THAT(
      tensor_release_map,
      UnorderedElementsAre(Pair(0, 0), Pair(1, 2), Pair(2, 3), Pair(3, 1),
                           Pair(4, 2), Pair(5, 3), Pair(6, 4), Pair(7, 4)));
}

TEST_F(KernelTest, PersistEagerTensor) {
  // Define the graph.
  //        0           3
  //        |           |
  //      Unpack_0    Unpack_1
  //       /  \        / \
  //      1    2      4   5
  //      |____|_______|__|
  //         | |__________|
  //         |      |
  //        Add_2  Add_3
  //         |      |
  //         6      7
  //         | \   /
  //         | TFL_MUL
  //         |    |
  //         |    8
  //         |____|
  //             AddN
  //              |
  //              9
  AddTensors(10, {0, 3}, {9}, kTfLiteFloat32, {3});

  AddTfOp(testing::kUnpack, {0}, {1, 2});
  AddTfOp(testing::kUnpack, {3}, {4, 5});
  AddTfOp(testing::kAdd, {1, 4}, {6});
  AddTfOp(testing::kAdd, {2, 5}, {7});
  AddTfLiteMulOp({6, 7}, {8});
  AddTfOp(testing::kAdd, {6, 8}, {9});

  ApplyFlexDelegate();

  // Define inputs.
  SetShape(0, {2, 2, 1});
  SetValues(0, {1.1f, 2.2f, 3.3f, 4.4f});
  SetShape(3, {2, 2, 1});
  SetValues(3, {1.1f, 2.2f, 3.3f, 4.4f});

  ASSERT_TRUE(Invoke());
  // Validates that tensor 6 should be preserved in the buffer map.
  auto* buffer_map =
      delegate_data_->GetBufferMap(interpreter_->primary_subgraph().context());
  EXPECT_TRUE(buffer_map->HasTensor(6));
  EXPECT_FALSE(buffer_map->HasTensor(7));
}

TEST_F(KernelTest, BadTensorFlowOp) {
  AddTensors(2, {0}, {1}, kTfLiteFloat32, {3});
  AddTfOp(testing::kNonExistent, {0}, {1});

  ASSERT_NE(
      ApplyFlexDelegate(std::unique_ptr<FlexDelegate>(new TestFlexDelegate())),
      kTfLiteOk);

  ASSERT_THAT(error_reporter().error_messages(),
              ContainsRegex("Op type not registered 'NonExistentOp'"));
}

TEST_F(KernelTest, BadNumberOfOutputs) {
  AddTensors(3, {0}, {1, 2}, kTfLiteFloat32, {3});
  AddTfOp(testing::kIdentity, {0}, {1, 2});

  ApplyFlexDelegate();

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

  ASSERT_NE(ApplyFlexDelegate(), kTfLiteOk);

  ASSERT_THAT(error_reporter().error_messages(),
              ContainsRegex("No attr named 'SrcT' in NodeDef"));
}

TEST_F(KernelTest, WrongSetOfNodes) {
  AddTensors(4, {0}, {3}, kTfLiteFloat32, {3});
  AddTfOp(testing::kUnpack, {0}, {1, 2});
  AddTfLiteMulOp({1, 2}, {3});

  // Specify that testing::kMul (#1) is supported when it actually isn't so that
  // we choose to use the TestFlexDelegate that supports every node regardless
  // whether it's actually supported or not.
  ASSERT_NE(
      ApplyFlexDelegate(std::unique_ptr<FlexDelegate>(new TestFlexDelegate())),
      kTfLiteOk);

  ASSERT_THAT(error_reporter().error_messages(),
              ContainsRegex("Cannot convert empty data into a valid NodeDef"));
}

TEST_F(KernelTest, MixedGraph) {
  AddTensors(9, {0, 3}, {8}, kTfLiteFloat32, {3});

  AddTfOp(testing::kUnpack, {0}, {1, 2});
  AddTfOp(testing::kUnpack, {3}, {4, 5});
  AddTfOp(testing::kAdd, {1, 4}, {6});
  AddTfOp(testing::kAdd, {2, 5}, {7});
  AddTfLiteMulOp({6, 7}, {8});

  ApplyFlexDelegate();

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

  ApplyFlexDelegate();

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

  void PrepareInterpreter(const std::vector<float>& input) {
    ApplyFlexDelegate();

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
  PrepareInterpreter(input);

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
  PrepareInterpreter(input);

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
  PrepareInterpreter(input);

  ASSERT_TRUE(Invoke());
  ASSERT_THAT(GetValues(12), ElementsAreArray(Apply(input, [](float in) {
                return (5 * in + 5) * (in + 1);
              })));
}

tensorflow::OpDef MakeOpDef(int num_inputs, int num_outputs) {
  tensorflow::OpRegistrationData op_reg_data;
  tensorflow::OpDefBuilder b("dummy");
  for (int i = 0; i < num_inputs; ++i) {
    b.Input(tensorflow::strings::StrCat("i", i, ": float"));
  }
  for (int i = 0; i < num_outputs; ++i) {
    b.Output(tensorflow::strings::StrCat("o", i, ": float"));
  }
  CHECK_OK(b.Attr("foo:string").Finalize(&op_reg_data));
  return op_reg_data.op_def;
}

tensorflow::PartialTensorShape S(std::initializer_list<int64_t> dims) {
  return tensorflow::PartialTensorShape(dims);
}

TEST(ValidateOutputTensorShapeConsistencyTest, ShapeHandleDebugString) {
  // Setup test to contain an input tensor list of size 3.
  tensorflow::OpDef op_def = MakeOpDef(4, 1);
  tensorflow::NodeDef def;
  tensorflow::shape_inference::InferenceContext c(
      0, def, op_def, {S({1}), S({2, 3}), S({4, 5, 6}), {}}, {}, {}, {});
  c.SetInput(3, c.UnknownShape());

  std::vector<tensorflow::shape_inference::ShapeHandle> shapes;
  EXPECT_EQ("[1]", c.DebugString(c.input(0)));
  EXPECT_EQ("[2,3]", c.DebugString(c.input(1)));
  EXPECT_EQ("[4,5,6]", c.DebugString(c.input(2)));
  // c.DebugString() returns "?" for the unknown shape which is different with
  // "-1" of TFLite. But this is intended behavior since we should use dynamic
  // tensor for unknown shape so the shape comparison must fail.
  EXPECT_EQ("?", c.DebugString(c.input(3)));
}

TEST(ValidateOutputTensorShapeConsistencyTest, GetShapeDebugString) {
  TfLiteIntArray* dims1 = TfLiteIntArrayCreate(1);
  dims1->data[0] = 1;
  EXPECT_EQ("[1]", GetShapeDebugString(dims1));
  TfLiteIntArrayFree(dims1);

  TfLiteIntArray* dims2 = TfLiteIntArrayCreate(2);
  dims2->data[0] = 2;
  dims2->data[1] = 3;
  EXPECT_EQ("[2,3]", GetShapeDebugString(dims2));
  TfLiteIntArrayFree(dims2);

  TfLiteIntArray* dims3 = TfLiteIntArrayCreate(3);
  dims3->data[0] = 4;
  dims3->data[1] = 5;
  dims3->data[2] = 6;
  EXPECT_EQ("[4,5,6]", GetShapeDebugString(dims3));
  TfLiteIntArrayFree(dims3);
}

}  // namespace testing
}  // namespace flex
}  // namespace tflite
