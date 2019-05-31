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

#include "tensorflow/lite/experimental/micro/micro_interpreter.h"

#include "tensorflow/lite/experimental/micro/testing/micro_test.h"

namespace tflite {
namespace {
void* MockInit(TfLiteContext* context, const char* buffer, size_t length) {
  // Do nothing.
  return nullptr;
}

void MockFree(TfLiteContext* context, void* buffer) {
  // Do nothing.
}

TfLiteStatus MockPrepare(TfLiteContext* context, TfLiteNode* node) {
  return kTfLiteOk;
}

TfLiteStatus MockInvoke(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = &context->tensors[node->inputs->data[0]];
  const int32_t* input_data = input->data.i32;
  const TfLiteTensor* weight = &context->tensors[node->inputs->data[1]];
  const uint8_t* weight_data = weight->data.uint8;
  TfLiteTensor* output = &context->tensors[node->outputs->data[0]];
  int32_t* output_data = output->data.i32;
  output_data[0] = input_data[0] + weight_data[0];
  return kTfLiteOk;
}

class MockOpResolver : public OpResolver {
 public:
  const TfLiteRegistration* FindOp(BuiltinOperator op,
                                   int version) const override {
    return nullptr;
  }
  const TfLiteRegistration* FindOp(const char* op, int version) const override {
    if (strcmp(op, "mock_custom") == 0) {
      static TfLiteRegistration r = {MockInit, MockFree, MockPrepare,
                                     MockInvoke};
      return &r;
    } else {
      return nullptr;
    }
  }
};

class StackAllocator : public flatbuffers::Allocator {
 public:
  StackAllocator() : data_(data_backing_), data_size_(0) {}

  uint8_t* allocate(size_t size) override {
    if ((data_size_ + size) > kStackAllocatorSize) {
      // TODO(petewarden): Add error reporting beyond returning null!
      return nullptr;
    }
    uint8_t* result = data_;
    data_ += size;
    data_size_ += size;
    return result;
  }

  void deallocate(uint8_t* p, size_t) override {}

  static StackAllocator& instance() {
    // Avoid using true dynamic memory allocation to be portable to bare metal.
    static char inst_memory[sizeof(StackAllocator)];
    static StackAllocator* inst = new (inst_memory) StackAllocator;
    return *inst;
  }

  static constexpr int kStackAllocatorSize = 4096;

 private:
  uint8_t data_backing_[kStackAllocatorSize];
  uint8_t* data_;
  int data_size_;
};

const Model* BuildMockModel() {
  using flatbuffers::Offset;
  flatbuffers::FlatBufferBuilder builder(StackAllocator::kStackAllocatorSize,
                                         &StackAllocator::instance());
  constexpr size_t buffer_data_size = 1;
  const uint8_t buffer_data[buffer_data_size] = {21};
  constexpr size_t buffers_size = 2;
  const Offset<Buffer> buffers[buffers_size] = {
      CreateBuffer(builder),
      CreateBuffer(builder,
                   builder.CreateVector(buffer_data, buffer_data_size))};
  constexpr size_t tensor_shape_size = 1;
  const int32_t tensor_shape[tensor_shape_size] = {1};
  constexpr size_t tensors_size = 3;
  const Offset<Tensor> tensors[tensors_size] = {
      CreateTensor(builder,
                   builder.CreateVector(tensor_shape, tensor_shape_size),
                   TensorType_INT32, 0,
                   builder.CreateString("test_input_tensor"), 0, false),
      CreateTensor(builder,
                   builder.CreateVector(tensor_shape, tensor_shape_size),
                   TensorType_UINT8, 1,
                   builder.CreateString("test_weight_tensor"), 0, false),
      CreateTensor(builder,
                   builder.CreateVector(tensor_shape, tensor_shape_size),
                   TensorType_INT32, 0,
                   builder.CreateString("test_output_tensor"), 0, false),
  };
  constexpr size_t inputs_size = 1;
  const int32_t inputs[inputs_size] = {0};
  constexpr size_t outputs_size = 1;
  const int32_t outputs[outputs_size] = {2};
  constexpr size_t operator_inputs_size = 2;
  const int32_t operator_inputs[operator_inputs_size] = {0, 1};
  constexpr size_t operator_outputs_size = 1;
  const int32_t operator_outputs[operator_outputs_size] = {2};
  constexpr size_t operators_size = 1;
  const Offset<Operator> operators[operators_size] = {CreateOperator(
      builder, 0, builder.CreateVector(operator_inputs, operator_inputs_size),
      builder.CreateVector(operator_outputs, operator_outputs_size),
      BuiltinOptions_NONE)};
  constexpr size_t subgraphs_size = 1;
  const Offset<SubGraph> subgraphs[subgraphs_size] = {
      CreateSubGraph(builder, builder.CreateVector(tensors, tensors_size),
                     builder.CreateVector(inputs, inputs_size),
                     builder.CreateVector(outputs, outputs_size),
                     builder.CreateVector(operators, operators_size),
                     builder.CreateString("test_subgraph"))};
  constexpr size_t operator_codes_size = 1;
  const Offset<OperatorCode> operator_codes[operator_codes_size] = {
      CreateOperatorCodeDirect(builder, BuiltinOperator_CUSTOM, "mock_custom",
                               0)};
  const Offset<Model> model_offset = CreateModel(
      builder, 0, builder.CreateVector(operator_codes, operator_codes_size),
      builder.CreateVector(subgraphs, subgraphs_size),
      builder.CreateString("test_model"),
      builder.CreateVector(buffers, buffers_size));
  FinishModelBuffer(builder, model_offset);
  void* model_pointer = builder.GetBufferPointer();
  const Model* model = flatbuffers::GetRoot<Model>(model_pointer);
  return model;
}

}  // namespace
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TestInterpreter) {
  const tflite::Model* model = tflite::BuildMockModel();
  TF_LITE_MICRO_EXPECT_NE(nullptr, model);
  tflite::MockOpResolver mock_resolver;
  constexpr size_t allocator_buffer_size = 1024;
  uint8_t allocator_buffer[allocator_buffer_size];
  tflite::SimpleTensorAllocator simple_tensor_allocator(allocator_buffer,
                                                        allocator_buffer_size);
  tflite::MicroInterpreter interpreter(
      model, mock_resolver, &simple_tensor_allocator, micro_test::reporter);
  TF_LITE_MICRO_EXPECT_EQ(1, interpreter.inputs_size());
  TF_LITE_MICRO_EXPECT_EQ(1, interpreter.outputs_size());

  TfLiteTensor* input = interpreter.input(0);
  TF_LITE_MICRO_EXPECT_NE(nullptr, input);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt32, input->type);
  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(4, input->bytes);
  TF_LITE_MICRO_EXPECT_NE(nullptr, input->data.i32);
  input->data.i32[0] = 21;

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, interpreter.Invoke());

  TfLiteTensor* output = interpreter.output(0);
  TF_LITE_MICRO_EXPECT_NE(nullptr, output);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteInt32, output->type);
  TF_LITE_MICRO_EXPECT_EQ(1, output->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(4, output->bytes);
  TF_LITE_MICRO_EXPECT_NE(nullptr, output->data.i32);
  TF_LITE_MICRO_EXPECT_EQ(42, output->data.i32[0]);
}

TF_LITE_MICRO_TESTS_END
