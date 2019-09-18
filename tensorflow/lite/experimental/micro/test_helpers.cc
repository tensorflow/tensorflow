/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/experimental/micro/test_helpers.h"

namespace tflite {
namespace {

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

flatbuffers::FlatBufferBuilder* BuilderInstance() {
  static char inst_memory[sizeof(flatbuffers::FlatBufferBuilder)];
  static flatbuffers::FlatBufferBuilder* inst =
      new (inst_memory) flatbuffers::FlatBufferBuilder(
          StackAllocator::kStackAllocatorSize, &StackAllocator::instance());
  return inst;
}

const Model* BuildMockModel() {
  using flatbuffers::Offset;
  flatbuffers::FlatBufferBuilder* builder = BuilderInstance();

  constexpr size_t buffer_data_size = 1;
  const uint8_t buffer_data[buffer_data_size] = {21};
  constexpr size_t buffers_size = 2;
  const Offset<Buffer> buffers[buffers_size] = {
      CreateBuffer(*builder),
      CreateBuffer(*builder,
                   builder->CreateVector(buffer_data, buffer_data_size))};
  constexpr size_t tensor_shape_size = 1;
  const int32_t tensor_shape[tensor_shape_size] = {1};
  constexpr size_t tensors_size = 3;
  const Offset<Tensor> tensors[tensors_size] = {
      CreateTensor(*builder,
                   builder->CreateVector(tensor_shape, tensor_shape_size),
                   TensorType_INT32, 0,
                   builder->CreateString("test_input_tensor"), 0, false),
      CreateTensor(*builder,
                   builder->CreateVector(tensor_shape, tensor_shape_size),
                   TensorType_UINT8, 1,
                   builder->CreateString("test_weight_tensor"), 0, false),
      CreateTensor(*builder,
                   builder->CreateVector(tensor_shape, tensor_shape_size),
                   TensorType_INT32, 0,
                   builder->CreateString("test_output_tensor"), 0, false),
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
      *builder, 0, builder->CreateVector(operator_inputs, operator_inputs_size),
      builder->CreateVector(operator_outputs, operator_outputs_size),
      BuiltinOptions_NONE)};
  constexpr size_t subgraphs_size = 1;
  const Offset<SubGraph> subgraphs[subgraphs_size] = {
      CreateSubGraph(*builder, builder->CreateVector(tensors, tensors_size),
                     builder->CreateVector(inputs, inputs_size),
                     builder->CreateVector(outputs, outputs_size),
                     builder->CreateVector(operators, operators_size),
                     builder->CreateString("test_subgraph"))};
  constexpr size_t operator_codes_size = 1;
  const Offset<OperatorCode> operator_codes[operator_codes_size] = {
      CreateOperatorCodeDirect(*builder, BuiltinOperator_CUSTOM, "mock_custom",
                               0)};
  const Offset<Model> model_offset = CreateModel(
      *builder, 0, builder->CreateVector(operator_codes, operator_codes_size),
      builder->CreateVector(subgraphs, subgraphs_size),
      builder->CreateString("test_model"),
      builder->CreateVector(buffers, buffers_size));
  FinishModelBuffer(*builder, model_offset);
  void* model_pointer = builder->GetBufferPointer();
  const Model* model = flatbuffers::GetRoot<Model>(model_pointer);
  return model;
}

}  // namespace

const Model* GetMockModel() {
  static Model* model = nullptr;
  if (!model) {
    model = const_cast<Model*>(BuildMockModel());
  }
  return model;
}

const Tensor* Create1dFlatbufferTensor(int size) {
  using flatbuffers::Offset;
  flatbuffers::FlatBufferBuilder* builder = BuilderInstance();
  constexpr size_t tensor_shape_size = 1;
  const int32_t tensor_shape[tensor_shape_size] = {size};
  const Offset<Tensor> tensor_offset = CreateTensor(
      *builder, builder->CreateVector(tensor_shape, tensor_shape_size),
      TensorType_INT32, 0, builder->CreateString("test_tensor"), 0, false);
  builder->Finish(tensor_offset);
  void* tensor_pointer = builder->GetBufferPointer();
  const Tensor* tensor = flatbuffers::GetRoot<Tensor>(tensor_pointer);
  return tensor;
}

const Tensor* CreateMissingQuantizationFlatbufferTensor(int size) {
  using flatbuffers::Offset;
  flatbuffers::FlatBufferBuilder* builder = BuilderInstance();
  const Offset<QuantizationParameters> quant_params =
      CreateQuantizationParameters(*builder, 0, 0, 0, 0,
                                   QuantizationDetails_NONE, 0, 0);
  constexpr size_t tensor_shape_size = 1;
  const int32_t tensor_shape[tensor_shape_size] = {size};
  const Offset<Tensor> tensor_offset = CreateTensor(
      *builder, builder->CreateVector(tensor_shape, tensor_shape_size),
      TensorType_INT32, 0, builder->CreateString("test_tensor"), quant_params,
      false);
  builder->Finish(tensor_offset);
  void* tensor_pointer = builder->GetBufferPointer();
  const Tensor* tensor = flatbuffers::GetRoot<Tensor>(tensor_pointer);
  return tensor;
}

const flatbuffers::Vector<flatbuffers::Offset<Buffer>>*
CreateFlatbufferBuffers() {
  using flatbuffers::Offset;
  flatbuffers::FlatBufferBuilder* builder = BuilderInstance();
  constexpr size_t buffers_size = 1;
  const Offset<Buffer> buffers[buffers_size] = {
      CreateBuffer(*builder),
  };
  const flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<Buffer>>>
      buffers_offset = builder->CreateVector(buffers, buffers_size);
  builder->Finish(buffers_offset);
  void* buffers_pointer = builder->GetBufferPointer();
  const flatbuffers::Vector<flatbuffers::Offset<Buffer>>* result =
      flatbuffers::GetRoot<flatbuffers::Vector<flatbuffers::Offset<Buffer>>>(
          buffers_pointer);
  return result;
}

}  // namespace tflite
