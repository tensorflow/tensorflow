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
#include "absl/memory/memory.h"
#include "third_party/flatbuffers/include/flatbuffers/flexbuffers.h"
#include "tensorflow/contrib/lite/delegates/eager/delegate_data.h"
#include "tensorflow/contrib/lite/kernels/test_util.h"
#include "tensorflow/contrib/lite/testing/util.h"

namespace tflite {
namespace eager {
namespace {

using tensorflow::protobuf::TextFormat;
using ::testing::ContainsRegex;
using ::testing::ElementsAre;

// We will use these are custom_names, so they need to be static.
static const char kIdentity[] = "Identity";
static const char kUnpack[] = "Unpack";
static const char kAdd[] = "Add";
static const char kMul[] = "Mul";

TfLiteStatus GenericPrepare(TfLiteContext* context, TfLiteDelegate* delegate,
                            const std::vector<int>& supported_nodes) {
  TfLiteIntArray* size_and_nodes =
      ConvertVectorToTfLiteIntArray(supported_nodes);
  TF_LITE_ENSURE_STATUS(context->ReplaceSubgraphsWithDelegateKernels(
      context, eager::GetKernel(), size_and_nodes, delegate));
  TfLiteIntArrayFree(size_and_nodes);
  return kTfLiteOk;
}

class KernelTest : public ::testing::Test {
 public:
  KernelTest() {
    CHECK(DelegateData::Create(&delegate_data_).ok());
    interpreter_.reset(new Interpreter(&error_reporter_));
  }

  bool Invoke() { return interpreter_->Invoke() == kTfLiteOk; }

  void SetValues(int tensor_index, const std::vector<float>& values) {
    float* v = interpreter_->typed_tensor<float>(tensor_index);
    for (float f : values) {
      *v++ = f;
    }
  }

  std::vector<float> GetValues(int tensor_index) {
    TfLiteTensor* o = interpreter_->tensor(tensor_index);
    return std::vector<float>(o->data.f, o->data.f + o->bytes / sizeof(float));
  }

  void SetShape(int tensor_index, const std::vector<int>& values) {
    ASSERT_EQ(interpreter_->ResizeInputTensor(tensor_index, values), kTfLiteOk);
    ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  }

  std::vector<int> GetShape(int tensor_index) {
    std::vector<int> result;
    auto* dims = interpreter_->tensor(tensor_index)->dims;
    for (int i = 0; i < dims->size; ++i) {
      result.push_back(dims->data[i]);
    }
    return result;
  }

  template <typename T>
  void ConfigureDelegate(T prepare_function) {
    delegate_.data_ = delegate_data_.get();
    delegate_.FreeBufferHandle = nullptr;
    delegate_.Prepare = prepare_function;
    delegate_.CopyFromBufferHandle = [](TfLiteDelegate* delegate,
                                        TfLiteBufferHandle buffer_handle,
                                        void* data, size_t size) {
      auto* delegate_data = reinterpret_cast<DelegateData*>(delegate->data_);
      tensorflow::StringPiece values =
          delegate_data->GetBufferMap()->GetTensor(buffer_handle).tensor_data();
      memcpy(data, values.data(), values.size());
      return kTfLiteOk;
    };
    CHECK(interpreter_->ModifyGraphWithDelegate(
              &delegate_, /*allow_dynamic_tensors=*/true) == kTfLiteOk);
  }

  void AddOp(const char* name, const std::vector<int>& inputs,
             const std::vector<int>& outputs) {
    auto attr = [](const string& key, const string& value) {
      return " attr{ key: '" + key + "' value {" + value + "}}";
    };

    string attributes;
    if (name == string(kUnpack)) {
      attributes = attr("T", "type: DT_FLOAT") + attr("num", "i: 2") +
                   attr("axis", "i: 0");
    } else if (name == string(kIdentity)) {
      attributes = attr("T", "type: DT_FLOAT");
    } else if (name == string(kAdd)) {
      attributes = attr("T", "type: DT_FLOAT");
    } else if (name == string(kMul)) {
      attributes = attr("T", "type: DT_FLOAT");
    }
    AddTfOp(name, attributes, inputs, outputs);
  }

  void AddTensors(int num_tensors, const std::vector<int>& inputs,
                  const std::vector<int>& outputs) {
    interpreter_->AddTensors(num_tensors);
    for (int i = 0; i < num_tensors; ++i) {
      TfLiteQuantizationParams quant;
      CHECK_EQ(interpreter_->SetTensorParametersReadWrite(i, kTfLiteFloat32,
                                                          /*name=*/"",
                                                          /*dims=*/{3}, quant),
               kTfLiteOk);
    }

    CHECK_EQ(interpreter_->SetInputs(inputs), kTfLiteOk);
    CHECK_EQ(interpreter_->SetOutputs(outputs), kTfLiteOk);
  }

  const TestErrorReporter& error_reporter() const { return error_reporter_; }

  void AddTfLiteOp(const char* name, const std::vector<int>& inputs,
                   const std::vector<int>& outputs) {
    CHECK_EQ(string(name), kMul);  // can only add MUL
    static TfLiteRegistration reg = {nullptr, nullptr, nullptr, nullptr};
    reg.builtin_code = BuiltinOperator_MUL;
    reg.prepare = [](TfLiteContext* context, TfLiteNode* node) {
      auto* i0 = &context->tensors[node->inputs->data[0]];
      auto* o = &context->tensors[node->outputs->data[0]];
      return context->ResizeTensor(context, o, TfLiteIntArrayCopy(i0->dims));
    };
    reg.invoke = [](TfLiteContext* context, TfLiteNode* node) {
      auto* i0 = &context->tensors[node->inputs->data[0]];
      auto* i1 = &context->tensors[node->inputs->data[1]];
      auto* o = &context->tensors[node->outputs->data[0]];
      for (int i = 0; i < o->bytes / sizeof(float); ++i) {
        o->data.f[i] = i0->data.f[i] * i1->data.f[i];
      }
      return kTfLiteOk;
    };

    CHECK_EQ(interpreter_->AddNodeWithParameters(inputs, outputs, nullptr, 0,
                                                 nullptr, &reg),
             kTfLiteOk);
  }

 private:
  void AddTfOp(const char* name, const string& nodedef_str,
               const std::vector<int>& inputs,
               const std::vector<int>& outputs) {
    static TfLiteRegistration reg = {nullptr, nullptr, nullptr, nullptr};
    reg.builtin_code = BuiltinOperator_CUSTOM;
    reg.custom_name = name;

    tensorflow::NodeDef nodedef;
    CHECK(TextFormat::ParseFromString(nodedef_str + " op: '" + name + "'",
                                      &nodedef));
    string serialized_nodedef;
    CHECK(nodedef.SerializeToString(&serialized_nodedef));
    flexbuffers::Builder fbb;
    fbb.Vector([&]() {
      fbb.String(nodedef.op());
      fbb.String(serialized_nodedef);
    });
    fbb.Finish();

    flexbuffers_.push_back(fbb.GetBuffer());
    auto& buffer = flexbuffers_.back();
    CHECK_EQ(interpreter_->AddNodeWithParameters(
                 inputs, outputs, reinterpret_cast<const char*>(buffer.data()),
                 buffer.size(), nullptr, &reg),
             kTfLiteOk);
  }

  std::unique_ptr<Interpreter> interpreter_;
  std::unique_ptr<DelegateData> delegate_data_;
  TfLiteDelegate delegate_;
  std::vector<std::vector<uint8_t>> flexbuffers_;
  TestErrorReporter error_reporter_;
};

TEST_F(KernelTest, FullGraph) {
  // Define the graph.
  AddTensors(9, {0, 3}, {8});

  AddOp(kUnpack, {0}, {1, 2});
  AddOp(kUnpack, {3}, {4, 5});
  AddOp(kAdd, {1, 4}, {6});
  AddOp(kAdd, {2, 5}, {7});
  AddOp(kMul, {6, 7}, {8});

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
  AddTensors(2, {0}, {1});
  AddOp("NonExistentOp", {0}, {1});

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
  AddTensors(3, {0}, {1, 2});
  AddOp(kIdentity, {0}, {1, 2});

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
  AddTensors(2, {0}, {1});

  // Cast is a TF op, but we don't add the proper nodedef to it in AddOp.
  AddOp("Cast", {0}, {1});

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
  AddTensors(4, {0}, {3});
  AddOp(kUnpack, {0}, {1, 2});
  AddTfLiteOp(kMul, {1, 2}, {3});

  // Specify that kMul (#1) is supported when it actually isn't.
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
  AddTensors(9, {0, 3}, {8});

  AddOp(kUnpack, {0}, {1, 2});
  AddOp(kUnpack, {3}, {4, 5});
  AddOp(kAdd, {1, 4}, {6});
  AddOp(kAdd, {2, 5}, {7});
  AddTfLiteOp(kMul, {6, 7}, {8});

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
  AddTensors(10, {0}, {9});

  AddOp(kUnpack, {0}, {1, 2});
  AddOp(kAdd, {1, 2}, {3});
  AddOp(kUnpack, {3}, {4, 5});

  AddTfLiteOp(kMul, {4, 5}, {6});

  AddOp(kUnpack, {6}, {7, 8});
  AddOp(kAdd, {7, 8}, {9});

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
