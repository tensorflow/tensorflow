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
#include "tensorflow/contrib/lite/delegates/eager/kernel.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/contrib/lite/delegates/eager/delegate_data.h"
#include "tensorflow/contrib/lite/delegates/eager/test_util.h"

namespace tflite {
namespace eager {
namespace {

using ::testing::ContainsRegex;
using ::testing::ElementsAre;

TfLiteStatus GenericPrepare(TfLiteContext* context, TfLiteDelegate* delegate,
                            const std::vector<int>& supported_nodes) {
  TfLiteIntArray* size_and_nodes =
      ConvertVectorToTfLiteIntArray(supported_nodes);
  TF_LITE_ENSURE_STATUS(context->ReplaceSubgraphsWithDelegateKernels(
      context, eager::GetKernel(), size_and_nodes, delegate));
  TfLiteIntArrayFree(size_and_nodes);
  return kTfLiteOk;
}

class KernelTest : public testing::EagerModelTest {
 public:
  KernelTest() {
    CHECK(DelegateData::Create(&delegate_data_).ok());
    interpreter_.reset(new Interpreter(&error_reporter_));
  }

  ~KernelTest() override {
    // The data needs to be released before the interpreter because the
    // interpreter references the data.
    delegate_data_.reset();
    interpreter_.reset();
  }

  template <typename T>
  void ConfigureDelegate(T prepare_function) {
    delegate_.data_ = delegate_data_.get();
    delegate_.FreeBufferHandle = nullptr;
    delegate_.Prepare = prepare_function;
    delegate_.CopyFromBufferHandle = [](TfLiteContext* context,
                                        TfLiteDelegate* delegate,
                                        TfLiteBufferHandle buffer_handle,
                                        void* data, size_t size) {
      auto* delegate_data = reinterpret_cast<DelegateData*>(delegate->data_);
      tensorflow::StringPiece values = delegate_data->GetBufferMap(context)
                                           ->GetTensor(buffer_handle)
                                           .tensor_data();
      memcpy(data, values.data(), values.size());
      return kTfLiteOk;
    };
    CHECK(interpreter_->ModifyGraphWithDelegate(
              &delegate_, /*allow_dynamic_tensors=*/true) == kTfLiteOk);
  }

 private:
  std::unique_ptr<DelegateData> delegate_data_;
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

  SetShape(0, {2, 2, 1});
  SetValues(0, {1.1f, 2.2f, 3.3f, 4.4f});

  ASSERT_FALSE(Invoke());
  ASSERT_THAT(error_reporter().error_messages(),
              ContainsRegex("Invalid NodeDef in Eager op"));
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

TEST_F(KernelTest, SplitGraph) {
  AddTensors(10, {0}, {9}, kTfLiteFloat32, {3});

  AddTfOp(testing::kUnpack, {0}, {1, 2});
  AddTfOp(testing::kAdd, {1, 2}, {3});
  AddTfOp(testing::kUnpack, {3}, {4, 5});

  AddTfLiteMulOp({4, 5}, {6});

  AddTfOp(testing::kUnpack, {6}, {7, 8});
  AddTfOp(testing::kAdd, {7, 8}, {9});

  ConfigureDelegate([](TfLiteContext* context, TfLiteDelegate* delegate) {
    return GenericPrepare(context, delegate, {0, 1, 2, 4, 5});
  });

  SetShape(0, {2, 2, 2, 1});
  SetValues(0, {3.0f, 1.0f, 0.5f, -1.0f, 0.0f, 1.0f, 1.5f, 3.0f});

  ASSERT_TRUE(Invoke());

  ASSERT_THAT(GetShape(9), ElementsAre(1));
  ASSERT_THAT(GetValues(9), ElementsAre(10.0f));
}

}  // namespace
}  // namespace eager
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
