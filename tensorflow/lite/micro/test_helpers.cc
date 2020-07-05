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

#include "tensorflow/lite/micro/test_helpers.h"

#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <new>

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace testing {
namespace {

class StackAllocator : public flatbuffers::Allocator {
 public:
  StackAllocator() : data_(data_backing_), data_size_(0) {}

  uint8_t* allocate(size_t size) override {
    TFLITE_DCHECK((data_size_ + size) <= kStackAllocatorSize);
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

  static constexpr size_t kStackAllocatorSize = 8192;

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

// A wrapper around FlatBuffer API to help build model easily.
class ModelBuilder {
 public:
  typedef int32_t Tensor;
  typedef int Operator;
  typedef int Node;

  // `builder` needs to be available until BuildModel is called.
  explicit ModelBuilder(flatbuffers::FlatBufferBuilder* builder)
      : builder_(builder) {}

  // Registers an operator that will be used in the model.
  Operator RegisterOp(BuiltinOperator op, const char* custom_code,
                      int32_t version);

  // Adds a tensor to the model.
  Tensor AddTensor(TensorType type, std::initializer_list<int32_t> shape) {
    return AddTensorImpl(type, /* is_variable */ false, shape);
  }

  // Adds a variable tensor to the model.
  Tensor AddVariableTensor(TensorType type,
                           std::initializer_list<int32_t> shape) {
    return AddTensorImpl(type, /* is_variable */ true, shape);
  }

  // Adds a node to the model with given input and output Tensors.
  Node AddNode(Operator op, std::initializer_list<Tensor> inputs,
               std::initializer_list<Tensor> outputs);

  void AddMetadata(const char* description_string,
                   const int32_t* metadata_buffer_data, size_t num_elements);

  // Constructs the flatbuffer model using `builder_` and return a pointer to
  // it. The returned model has the same lifetime as `builder_`.
  const Model* BuildModel(std::initializer_list<Tensor> inputs,
                          std::initializer_list<Tensor> outputs);

 private:
  // Adds a tensor to the model.
  Tensor AddTensorImpl(TensorType type, bool is_variable,
                       std::initializer_list<int32_t> shape);

  flatbuffers::FlatBufferBuilder* builder_;

  static constexpr int kMaxOperatorCodes = 10;
  flatbuffers::Offset<tflite::OperatorCode> operator_codes_[kMaxOperatorCodes];
  int next_operator_code_id_ = 0;

  static constexpr int kMaxOperators = 50;
  flatbuffers::Offset<tflite::Operator> operators_[kMaxOperators];
  int next_operator_id_ = 0;

  static constexpr int kMaxTensors = 50;
  flatbuffers::Offset<tflite::Tensor> tensors_[kMaxTensors];

  static constexpr int kMaxMetadataBuffers = 10;

  static constexpr int kMaxMetadatas = 10;
  flatbuffers::Offset<Metadata> metadata_[kMaxMetadatas];

  flatbuffers::Offset<Buffer> metadata_buffers_[kMaxMetadataBuffers];

  int nbr_of_metadata_buffers_ = 0;

  int next_tensor_id_ = 0;
};

ModelBuilder::Operator ModelBuilder::RegisterOp(BuiltinOperator op,
                                                const char* custom_code,
                                                int32_t version) {
  TFLITE_DCHECK(next_operator_code_id_ <= kMaxOperatorCodes);
  operator_codes_[next_operator_code_id_] =
      tflite::CreateOperatorCodeDirect(*builder_, op, custom_code, version);
  next_operator_code_id_++;
  return next_operator_code_id_ - 1;
}

ModelBuilder::Node ModelBuilder::AddNode(
    ModelBuilder::Operator op,
    std::initializer_list<ModelBuilder::Tensor> inputs,
    std::initializer_list<ModelBuilder::Tensor> outputs) {
  TFLITE_DCHECK(next_operator_id_ <= kMaxOperators);
  operators_[next_operator_id_] = tflite::CreateOperator(
      *builder_, op, builder_->CreateVector(inputs.begin(), inputs.size()),
      builder_->CreateVector(outputs.begin(), outputs.size()),
      BuiltinOptions_NONE);
  next_operator_id_++;
  return next_operator_id_ - 1;
}

void ModelBuilder::AddMetadata(const char* description_string,
                               const int32_t* metadata_buffer_data,
                               size_t num_elements) {
  metadata_[ModelBuilder::nbr_of_metadata_buffers_] =
      CreateMetadata(*builder_, builder_->CreateString(description_string),
                     1 + ModelBuilder::nbr_of_metadata_buffers_);

  metadata_buffers_[nbr_of_metadata_buffers_] = tflite::CreateBuffer(
      *builder_, builder_->CreateVector((uint8_t*)metadata_buffer_data,
                                        sizeof(uint32_t) * num_elements));

  ModelBuilder::nbr_of_metadata_buffers_++;
}

const Model* ModelBuilder::BuildModel(
    std::initializer_list<ModelBuilder::Tensor> inputs,
    std::initializer_list<ModelBuilder::Tensor> outputs) {
  // Model schema requires an empty buffer at idx 0.
  size_t buffer_size = 1 + ModelBuilder::nbr_of_metadata_buffers_;
  flatbuffers::Offset<Buffer> buffers[kMaxMetadataBuffers];
  buffers[0] = tflite::CreateBuffer(*builder_);

  // Place the metadata buffers first in the buffer since the indices for them
  // have already been set in AddMetadata()
  for (int i = 1; i < ModelBuilder::nbr_of_metadata_buffers_ + 1; ++i) {
    buffers[i] = metadata_buffers_[i - 1];
  }

  // TFLM only supports single subgraph.
  constexpr size_t subgraphs_size = 1;
  const flatbuffers::Offset<SubGraph> subgraphs[subgraphs_size] = {
      tflite::CreateSubGraph(
          *builder_, builder_->CreateVector(tensors_, next_tensor_id_),
          builder_->CreateVector(inputs.begin(), inputs.size()),
          builder_->CreateVector(outputs.begin(), outputs.size()),
          builder_->CreateVector(operators_, next_operator_id_),
          builder_->CreateString("test_subgraph"))};

  flatbuffers::Offset<Model> model_offset;
  if (ModelBuilder::nbr_of_metadata_buffers_ > 0) {
    model_offset = tflite::CreateModel(
        *builder_, 0,
        builder_->CreateVector(operator_codes_, next_operator_code_id_),
        builder_->CreateVector(subgraphs, subgraphs_size),
        builder_->CreateString("teset_model"),
        builder_->CreateVector(buffers, buffer_size), 0,
        builder_->CreateVector(metadata_,
                               ModelBuilder::nbr_of_metadata_buffers_));
  } else {
    model_offset = tflite::CreateModel(
        *builder_, 0,
        builder_->CreateVector(operator_codes_, next_operator_code_id_),
        builder_->CreateVector(subgraphs, subgraphs_size),
        builder_->CreateString("teset_model"),
        builder_->CreateVector(buffers, buffer_size));
  }

  tflite::FinishModelBuffer(*builder_, model_offset);
  void* model_pointer = builder_->GetBufferPointer();
  const Model* model = flatbuffers::GetRoot<Model>(model_pointer);
  return model;
}

ModelBuilder::Tensor ModelBuilder::AddTensorImpl(
    TensorType type, bool is_variable, std::initializer_list<int32_t> shape) {
  TFLITE_DCHECK(next_tensor_id_ <= kMaxTensors);
  tensors_[next_tensor_id_] = tflite::CreateTensor(
      *builder_, builder_->CreateVector(shape.begin(), shape.size()), type,
      /* buffer */ 0, /* name */ 0, /* quantization */ 0,
      /* is_variable */ is_variable,
      /* sparsity */ 0);
  next_tensor_id_++;
  return next_tensor_id_ - 1;
}

const Model* BuildSimpleStatefulModel() {
  using flatbuffers::Offset;
  flatbuffers::FlatBufferBuilder* fb_builder = BuilderInstance();

  ModelBuilder model_builder(fb_builder);

  const int op_id =
      model_builder.RegisterOp(BuiltinOperator_CUSTOM, "simple_stateful_op", 0);
  const int input_tensor = model_builder.AddTensor(TensorType_UINT8, {3});
  const int median_tensor = model_builder.AddTensor(TensorType_UINT8, {3});
  const int invoke_count_tensor =
      model_builder.AddTensor(TensorType_INT32, {1});

  model_builder.AddNode(op_id, {input_tensor},
                        {median_tensor, invoke_count_tensor});
  return model_builder.BuildModel({input_tensor},
                                  {median_tensor, invoke_count_tensor});
}

const Model* BuildSimpleModelWithBranch() {
  using flatbuffers::Offset;
  flatbuffers::FlatBufferBuilder* fb_builder = BuilderInstance();

  ModelBuilder model_builder(fb_builder);
  /* Model structure
           | t0
    +------|
    |      v
    |   +---------+
    |   |   n0    |
    |   |         |
    |   +---------+
    v           +
                |
  +---------+   | t1
  |   n1    |   |
  |         |   |
  +---------+   |
     |          |
 t2  |          v
     |   +---------+
     +-->|    n2   |
         |         |
         +-------|-+
                 |t3
                 v
  */
  const int op_id =
      model_builder.RegisterOp(BuiltinOperator_CUSTOM, "mock_custom",
                               /* version= */ 0);
  const int t0 = model_builder.AddTensor(TensorType_FLOAT32, {2, 2, 3});
  const int t1 = model_builder.AddTensor(TensorType_FLOAT32, {2, 2, 3});
  const int t2 = model_builder.AddTensor(TensorType_FLOAT32, {2, 2, 3});
  const int t3 = model_builder.AddTensor(TensorType_FLOAT32, {2, 2, 3});
  model_builder.AddNode(op_id, {t0}, {t1});      // n0
  model_builder.AddNode(op_id, {t0}, {t2});      // n1
  model_builder.AddNode(op_id, {t1, t2}, {t3});  // n2
  return model_builder.BuildModel({t0}, {t3});
}

const Model* BuildModelWithOfflinePlanning(int number_of_tensors,
                                           const int32_t* metadata_buffer,
                                           NodeConnection* node_conn,
                                           int num_conns) {
  using flatbuffers::Offset;
  flatbuffers::FlatBufferBuilder* fb_builder = BuilderInstance();

  ModelBuilder model_builder(fb_builder);

  const int op_id =
      model_builder.RegisterOp(BuiltinOperator_CUSTOM, "mock_custom",
                               /* version= */ 0);

  for (int i = 0; i < number_of_tensors; ++i) {
    model_builder.AddTensor(TensorType_FLOAT32, {2, 2, 3});
  }

  for (int i = 0; i < num_conns; ++i) {
    model_builder.AddNode(op_id, node_conn[i].input, node_conn[i].output);
  }

  model_builder.AddMetadata(
      "OfflineMemoryAllocation", metadata_buffer,
      number_of_tensors + tflite::testing::kOfflinePlannerHeaderSize);

  return model_builder.BuildModel(node_conn[0].input,
                                  node_conn[num_conns - 1].output);
}

const Model* BuildSimpleMockModel() {
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
  constexpr size_t tensors_size = 4;
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
      CreateTensor(*builder,
                   builder->CreateVector(tensor_shape, tensor_shape_size),
                   TensorType_INT32, 0,
                   builder->CreateString("test_output2_tensor"), 0, false),
  };
  constexpr size_t inputs_size = 1;
  const int32_t inputs[inputs_size] = {0};
  constexpr size_t outputs_size = 2;
  const int32_t outputs[outputs_size] = {2, 3};
  constexpr size_t operator_inputs_size = 2;
  const int32_t operator_inputs[operator_inputs_size] = {0, 1};
  constexpr size_t operator_outputs_size = 1;
  const int32_t operator_outputs[operator_outputs_size] = {2};
  const int32_t operator2_outputs[operator_outputs_size] = {3};
  constexpr size_t operators_size = 2;
  const Offset<Operator> operators[operators_size] = {
      CreateOperator(
          *builder, 0,
          builder->CreateVector(operator_inputs, operator_inputs_size),
          builder->CreateVector(operator_outputs, operator_outputs_size),
          BuiltinOptions_NONE),
      CreateOperator(
          *builder, 0,
          builder->CreateVector(operator_inputs, operator_inputs_size),
          builder->CreateVector(operator2_outputs, operator_outputs_size),
          BuiltinOptions_NONE),
  };
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

const Model* BuildComplexMockModel() {
  using flatbuffers::Offset;
  flatbuffers::FlatBufferBuilder* builder = BuilderInstance();

  constexpr size_t buffer_data_size = 1;
  const uint8_t buffer_data_1[buffer_data_size] = {21};
  const uint8_t buffer_data_2[buffer_data_size] = {21};
  const uint8_t buffer_data_3[buffer_data_size] = {21};
  constexpr size_t buffers_size = 7;
  const Offset<Buffer> buffers[buffers_size] = {
      // Op 1 buffers:
      CreateBuffer(*builder),
      CreateBuffer(*builder),
      CreateBuffer(*builder,
                   builder->CreateVector(buffer_data_1, buffer_data_size)),
      // Op 2 buffers:
      CreateBuffer(*builder),
      CreateBuffer(*builder,
                   builder->CreateVector(buffer_data_2, buffer_data_size)),
      // Op 3 buffers:
      CreateBuffer(*builder),
      CreateBuffer(*builder,
                   builder->CreateVector(buffer_data_3, buffer_data_size)),
  };
  constexpr size_t tensor_shape_size = 1;
  const int32_t tensor_shape[tensor_shape_size] = {1};

  constexpr size_t tensors_size = 10;
  const Offset<Tensor> tensors[tensors_size] = {
      // Op 1 inputs:
      CreateTensor(
          *builder, builder->CreateVector(tensor_shape, tensor_shape_size),
          TensorType_INT32, 0, builder->CreateString("test_input_tensor_1"), 0,
          false /* is_variable */),
      CreateTensor(
          *builder, builder->CreateVector(tensor_shape, tensor_shape_size),
          TensorType_INT32, 1, builder->CreateString("test_variable_tensor_1"),
          0, true /* is_variable */),
      CreateTensor(
          *builder, builder->CreateVector(tensor_shape, tensor_shape_size),
          TensorType_UINT8, 2, builder->CreateString("test_weight_tensor_1"), 0,
          false /* is_variable */),
      // Op 1 output / Op 2 input:
      CreateTensor(
          *builder, builder->CreateVector(tensor_shape, tensor_shape_size),
          TensorType_INT32, 0, builder->CreateString("test_output_tensor_1"), 0,
          false /* is_variable */),
      // Op 2 inputs:
      CreateTensor(
          *builder, builder->CreateVector(tensor_shape, tensor_shape_size),
          TensorType_INT32, 1, builder->CreateString("test_variable_tensor_2"),
          0, true /* is_variable */),
      CreateTensor(
          *builder, builder->CreateVector(tensor_shape, tensor_shape_size),
          TensorType_UINT8, 2, builder->CreateString("test_weight_tensor_2"), 0,
          false /* is_variable */),
      // Op 2 output / Op 3 input:
      CreateTensor(
          *builder, builder->CreateVector(tensor_shape, tensor_shape_size),
          TensorType_INT32, 0, builder->CreateString("test_output_tensor_2"), 0,
          false /* is_variable */),
      // Op 3 inputs:
      CreateTensor(
          *builder, builder->CreateVector(tensor_shape, tensor_shape_size),
          TensorType_INT32, 1, builder->CreateString("test_variable_tensor_3"),
          0, true /* is_variable */),
      CreateTensor(
          *builder, builder->CreateVector(tensor_shape, tensor_shape_size),
          TensorType_UINT8, 2, builder->CreateString("test_weight_tensor_3"), 0,
          false /* is_variable */),
      // Op 3 output:
      CreateTensor(
          *builder, builder->CreateVector(tensor_shape, tensor_shape_size),
          TensorType_INT32, 0, builder->CreateString("test_output_tensor_3"), 0,
          false /* is_variable */),
  };

  constexpr size_t operators_size = 3;
  Offset<Operator> operators[operators_size];
  {
    // Set Op 1 attributes:
    constexpr size_t operator_inputs_size = 3;
    const int32_t operator_inputs[operator_inputs_size] = {0, 1, 2};
    constexpr size_t operator_outputs_size = 1;
    const int32_t operator_outputs[operator_outputs_size] = {3};

    operators[0] = {CreateOperator(
        *builder, 0,
        builder->CreateVector(operator_inputs, operator_inputs_size),
        builder->CreateVector(operator_outputs, operator_outputs_size),
        BuiltinOptions_NONE)};
  }

  {
    // Set Op 2 attributes
    constexpr size_t operator_inputs_size = 3;
    const int32_t operator_inputs[operator_inputs_size] = {3, 4, 5};
    constexpr size_t operator_outputs_size = 1;
    const int32_t operator_outputs[operator_outputs_size] = {6};

    operators[1] = {CreateOperator(
        *builder, 0,
        builder->CreateVector(operator_inputs, operator_inputs_size),
        builder->CreateVector(operator_outputs, operator_outputs_size),
        BuiltinOptions_NONE)};
  }

  {
    // Set Op 3 attributes
    constexpr size_t operator_inputs_size = 3;
    const int32_t operator_inputs[operator_inputs_size] = {6, 7, 8};
    constexpr size_t operator_outputs_size = 1;
    const int32_t operator_outputs[operator_outputs_size] = {9};

    operators[2] = {CreateOperator(
        *builder, 0,
        builder->CreateVector(operator_inputs, operator_inputs_size),
        builder->CreateVector(operator_outputs, operator_outputs_size),
        BuiltinOptions_NONE)};
  }

  constexpr size_t inputs_size = 1;
  const int32_t inputs[inputs_size] = {0};
  constexpr size_t outputs_size = 1;
  const int32_t outputs[outputs_size] = {9};

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

const TfLiteRegistration* SimpleStatefulOp::getRegistration() {
  return GetMutableRegistration();
}

TfLiteRegistration* SimpleStatefulOp::GetMutableRegistration() {
  static TfLiteRegistration r;
  r.init = Init;
  r.prepare = Prepare;
  r.invoke = Invoke;
  return &r;
}

void* SimpleStatefulOp::Init(TfLiteContext* context, const char* buffer,
                             size_t length) {
  TFLITE_DCHECK(context->AllocateBufferForEval == nullptr);
  TFLITE_DCHECK(context->GetScratchBuffer == nullptr);
  TFLITE_DCHECK(context->RequestScratchBufferInArena == nullptr);

  void* raw;
  TFLITE_DCHECK(context->AllocatePersistentBuffer(context, sizeof(OpData),
                                                  &raw) == kTfLiteOk);
  OpData* data = reinterpret_cast<OpData*>(raw);
  *data = {};
  return raw;
}

TfLiteStatus SimpleStatefulOp::Prepare(TfLiteContext* context,
                                       TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  // Make sure that the input is in uint8 with at least 1 data entry.
  const TfLiteTensor* input = tflite::GetInput(context, node, kInputTensor);
  if (input->type != kTfLiteUInt8) return kTfLiteError;
  if (NumElements(input->dims) == 0) return kTfLiteError;

  // Allocate a temporary buffer with the same size of input for sorting.
  TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
      context, sizeof(uint8_t) * NumElements(input->dims),
      &data->sorting_buffer));
  return kTfLiteOk;
}

TfLiteStatus SimpleStatefulOp::Invoke(TfLiteContext* context,
                                      TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  data->invoke_count += 1;

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const uint8_t* input_data = GetTensorData<uint8_t>(input);
  int size = NumElements(input->dims);

  uint8_t* sorting_buffer = reinterpret_cast<uint8_t*>(
      context->GetScratchBuffer(context, data->sorting_buffer));
  // Copy inputs data to the sorting buffer. We don't want to mutate the input
  // tensor as it might be used by a another node.
  for (int i = 0; i < size; i++) {
    sorting_buffer[i] = input_data[i];
  }

  // In place insertion sort on `sorting_buffer`.
  for (int i = 1; i < size; i++) {
    for (int j = i; j > 0 && sorting_buffer[j] < sorting_buffer[j - 1]; j--) {
      std::swap(sorting_buffer[j], sorting_buffer[j - 1]);
    }
  }

  TfLiteTensor* median = GetOutput(context, node, kMedianTensor);
  uint8_t* median_data = GetTensorData<uint8_t>(median);
  TfLiteTensor* invoke_count = GetOutput(context, node, kInvokeCount);
  int32_t* invoke_count_data = GetTensorData<int32_t>(invoke_count);

  median_data[0] = sorting_buffer[size / 2];
  invoke_count_data[0] = data->invoke_count;
  return kTfLiteOk;
}

const TfLiteRegistration* MockCustom::getRegistration() {
  return GetMutableRegistration();
}

TfLiteRegistration* MockCustom::GetMutableRegistration() {
  static TfLiteRegistration r;
  r.init = Init;
  r.prepare = Prepare;
  r.invoke = Invoke;
  r.free = Free;
  return &r;
}

void* MockCustom::Init(TfLiteContext* context, const char* buffer,
                       size_t length) {
  // We don't support delegate in TFL micro. This is a weak check to test if
  // context struct being zero-initialized.
  TFLITE_DCHECK(context->ReplaceNodeSubsetsWithDelegateKernels == nullptr);
  freed_ = false;
  // Do nothing.
  return nullptr;
}

void MockCustom::Free(TfLiteContext* context, void* buffer) { freed_ = true; }

TfLiteStatus MockCustom::Prepare(TfLiteContext* context, TfLiteNode* node) {
  return kTfLiteOk;
}

TfLiteStatus MockCustom::Invoke(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = tflite::GetInput(context, node, 0);
  const int32_t* input_data = input->data.i32;
  const TfLiteTensor* weight = tflite::GetInput(context, node, 1);
  const uint8_t* weight_data = weight->data.uint8;
  TfLiteTensor* output = GetOutput(context, node, 0);
  int32_t* output_data = output->data.i32;
  output_data[0] =
      0;  // Catch output tensor sharing memory with an input tensor
  output_data[0] = input_data[0] + weight_data[0];
  return kTfLiteOk;
}

bool MockCustom::freed_ = false;

AllOpsResolver GetOpResolver() {
  AllOpsResolver op_resolver;
  op_resolver.AddCustom("mock_custom", MockCustom::GetMutableRegistration());
  op_resolver.AddCustom("simple_stateful_op",
                        SimpleStatefulOp::GetMutableRegistration());

  return op_resolver;
}

const Model* GetSimpleMockModel() {
  static Model* model = nullptr;
  if (!model) {
    model = const_cast<Model*>(BuildSimpleMockModel());
  }
  return model;
}

const Model* GetComplexMockModel() {
  static Model* model = nullptr;
  if (!model) {
    model = const_cast<Model*>(BuildComplexMockModel());
  }
  return model;
}

const Model* GetSimpleModelWithBranch() {
  static Model* model = nullptr;
  if (!model) {
    model = const_cast<Model*>(BuildSimpleModelWithBranch());
  }
  return model;
}

const Model* GetModelWithOfflinePlanning(int num_tensors,
                                         const int32_t* metadata_buffer,
                                         NodeConnection* node_conn,
                                         int num_conns) {
  const Model* model = BuildModelWithOfflinePlanning(
      num_tensors, metadata_buffer, node_conn, num_conns);
  return model;
}

const Model* GetSimpleStatefulModel() {
  static Model* model = nullptr;
  if (!model) {
    model = const_cast<Model*>(BuildSimpleStatefulModel());
  }
  return model;
}

const Tensor* Create1dFlatbufferTensor(int size, bool is_variable) {
  using flatbuffers::Offset;
  flatbuffers::FlatBufferBuilder* builder = BuilderInstance();
  constexpr size_t tensor_shape_size = 1;
  const int32_t tensor_shape[tensor_shape_size] = {size};
  const Offset<Tensor> tensor_offset = CreateTensor(
      *builder, builder->CreateVector(tensor_shape, tensor_shape_size),
      TensorType_INT32, 0, builder->CreateString("test_tensor"), 0,
      is_variable);
  builder->Finish(tensor_offset);
  void* tensor_pointer = builder->GetBufferPointer();
  const Tensor* tensor = flatbuffers::GetRoot<Tensor>(tensor_pointer);
  return tensor;
}

const Tensor* CreateQuantizedFlatbufferTensor(int size) {
  using flatbuffers::Offset;
  flatbuffers::FlatBufferBuilder* builder = BuilderInstance();
  const Offset<QuantizationParameters> quant_params =
      CreateQuantizationParameters(
          *builder,
          /*min=*/builder->CreateVector<float>({0.1f}),
          /*max=*/builder->CreateVector<float>({0.2f}),
          /*scale=*/builder->CreateVector<float>({0.3f}),
          /*zero_point=*/builder->CreateVector<int64_t>({100ll}));

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

int TestStrcmp(const char* a, const char* b) {
  if ((a == nullptr) || (b == nullptr)) {
    return -1;
  }
  while ((*a != 0) && (*a == *b)) {
    a++;
    b++;
  }
  return *reinterpret_cast<const unsigned char*>(a) -
         *reinterpret_cast<const unsigned char*>(b);
}

// Wrapper to forward kernel errors to the interpreter's error reporter.
void ReportOpError(struct TfLiteContext* context, const char* format, ...) {
  ErrorReporter* error_reporter = static_cast<ErrorReporter*>(context->impl_);
  va_list args;
  va_start(args, format);
  TF_LITE_REPORT_ERROR(error_reporter, format, args);
  va_end(args);
}

// Create a TfLiteIntArray from an array of ints.  The first element in the
// supplied array must be the size of the array expressed as an int.
TfLiteIntArray* IntArrayFromInts(const int* int_array) {
  return const_cast<TfLiteIntArray*>(
      reinterpret_cast<const TfLiteIntArray*>(int_array));
}

// Create a TfLiteFloatArray from an array of floats.  The first element in the
// supplied array must be the size of the array expressed as a float.
TfLiteFloatArray* FloatArrayFromFloats(const float* floats) {
  static_assert(sizeof(float) == sizeof(int),
                "assumes sizeof(float) == sizeof(int) to perform casting");
  int size = static_cast<int>(floats[0]);
  *reinterpret_cast<int32_t*>(const_cast<float*>(floats)) = size;
  return reinterpret_cast<TfLiteFloatArray*>(const_cast<float*>(floats));
}

TfLiteTensor CreateTensor(TfLiteIntArray* dims, bool is_variable) {
  TfLiteTensor result;
  result.dims = dims;
  result.params = {};
  result.quantization = {kTfLiteNoQuantization, nullptr};
  result.is_variable = is_variable;
  result.allocation_type = kTfLiteMemNone;
  return result;
}

TfLiteTensor CreateFloatTensor(const float* data, TfLiteIntArray* dims,
                               bool is_variable) {
  TfLiteTensor result = CreateTensor(dims, is_variable);
  result.type = kTfLiteFloat32;
  result.data.f = const_cast<float*>(data);
  result.bytes = ElementCount(*dims) * sizeof(float);
  return result;
}

void PopulateFloatTensor(TfLiteTensor* tensor, float* begin, float* end) {
  float* p = begin;
  float* v = tensor->data.f;
  while (p != end) {
    *v++ = *p++;
  }
}

TfLiteTensor CreateBoolTensor(const bool* data, TfLiteIntArray* dims,
                              bool is_variable) {
  TfLiteTensor result = CreateTensor(dims, is_variable);
  result.type = kTfLiteBool;
  result.data.b = const_cast<bool*>(data);
  result.bytes = ElementCount(*dims) * sizeof(bool);
  return result;
}

TfLiteTensor CreateInt32Tensor(const int32_t* data, TfLiteIntArray* dims,
                               bool is_variable) {
  TfLiteTensor result = CreateTensor(dims, is_variable);
  result.type = kTfLiteInt32;
  result.data.i32 = const_cast<int32_t*>(data);
  result.bytes = ElementCount(*dims) * sizeof(int32_t);
  return result;
}

TfLiteTensor CreateQuantizedTensor(const uint8_t* data, TfLiteIntArray* dims,
                                   float scale, int zero_point,
                                   bool is_variable) {
  TfLiteTensor result = CreateTensor(dims, is_variable);
  result.type = kTfLiteUInt8;
  result.data.uint8 = const_cast<uint8_t*>(data);
  result.params = {scale, zero_point};
  result.quantization = {kTfLiteAffineQuantization, nullptr};
  result.bytes = ElementCount(*dims) * sizeof(uint8_t);
  return result;
}

TfLiteTensor CreateQuantizedTensor(const int8_t* data, TfLiteIntArray* dims,
                                   float scale, int zero_point,
                                   bool is_variable) {
  TfLiteTensor result = CreateTensor(dims, is_variable);
  result.type = kTfLiteInt8;
  result.data.int8 = const_cast<int8_t*>(data);
  result.params = {scale, zero_point};
  result.quantization = {kTfLiteAffineQuantization, nullptr};
  result.bytes = ElementCount(*dims) * sizeof(int8_t);
  return result;
}

TfLiteTensor CreateQuantizedTensor(const int16_t* data, TfLiteIntArray* dims,
                                   float scale, int zero_point,
                                   bool is_variable) {
  TfLiteTensor result = CreateTensor(dims, is_variable);
  result.type = kTfLiteInt16;
  result.data.i16 = const_cast<int16_t*>(data);
  result.params = {scale, zero_point};
  result.quantization = {kTfLiteAffineQuantization, nullptr};
  result.bytes = ElementCount(*dims) * sizeof(int16_t);
  return result;
}

TfLiteTensor CreateQuantizedBiasTensor(const float* data, int32_t* quantized,
                                       TfLiteIntArray* dims, float input_scale,
                                       float weights_scale, bool is_variable) {
  float bias_scale = input_scale * weights_scale;
  tflite::SymmetricQuantize(data, quantized, ElementCount(*dims), bias_scale);
  TfLiteTensor result = CreateTensor(dims, is_variable);
  result.type = kTfLiteInt32;
  result.data.i32 = const_cast<int32_t*>(quantized);
  // Quantized int32 tensors always have a zero point of 0, since the range of
  // int32 values is large, and because zero point costs extra cycles during
  // processing.
  result.params = {bias_scale, 0};
  result.quantization = {kTfLiteAffineQuantization, nullptr};
  result.bytes = ElementCount(*dims) * sizeof(int32_t);
  return result;
}

// Quantizes int32 bias tensor with per-channel weights determined by input
// scale multiplied by weight scale for each channel.
TfLiteTensor CreatePerChannelQuantizedBiasTensor(
    const float* input, int32_t* quantized, TfLiteIntArray* dims,
    float input_scale, float* weight_scales, float* scales, int* zero_points,
    TfLiteAffineQuantization* affine_quant, int quantized_dimension,
    bool is_variable) {
  int input_size = ElementCount(*dims);
  int num_channels = dims->data[quantized_dimension];
  // First element is reserved for array length
  zero_points[0] = num_channels;
  scales[0] = static_cast<float>(num_channels);
  float* scales_array = &scales[1];
  for (int i = 0; i < num_channels; i++) {
    scales_array[i] = input_scale * weight_scales[i];
    zero_points[i + 1] = 0;
  }

  SymmetricPerChannelQuantize(input, quantized, input_size, num_channels,
                              scales_array);

  affine_quant->scale = FloatArrayFromFloats(scales);
  affine_quant->zero_point = IntArrayFromInts(zero_points);
  affine_quant->quantized_dimension = quantized_dimension;

  TfLiteTensor result = CreateTensor(dims, is_variable);
  result.type = kTfLiteInt32;
  result.data.i32 = const_cast<int32_t*>(quantized);
  result.quantization = {kTfLiteAffineQuantization, affine_quant};
  result.bytes = ElementCount(*dims) * sizeof(int32_t);
  return result;
}

TfLiteTensor CreateSymmetricPerChannelQuantizedTensor(
    const float* input, int8_t* quantized, TfLiteIntArray* dims, float* scales,
    int* zero_points, TfLiteAffineQuantization* affine_quant,
    int quantized_dimension, bool is_variable) {
  int channel_count = dims->data[quantized_dimension];
  scales[0] = static_cast<float>(channel_count);
  zero_points[0] = channel_count;

  SignedSymmetricPerChannelQuantize(input, dims, quantized_dimension, quantized,
                                    &scales[1]);

  for (int i = 0; i < channel_count; i++) {
    zero_points[i + 1] = 0;
  }

  affine_quant->scale = FloatArrayFromFloats(scales);
  affine_quant->zero_point = IntArrayFromInts(zero_points);
  affine_quant->quantized_dimension = quantized_dimension;

  TfLiteTensor result = CreateTensor(dims, is_variable);
  result.type = kTfLiteInt8;
  result.data.int8 = const_cast<int8_t*>(quantized);
  result.quantization = {kTfLiteAffineQuantization, affine_quant};
  result.bytes = ElementCount(*dims) * sizeof(int8_t);
  return result;
}

}  // namespace testing
}  // namespace tflite
