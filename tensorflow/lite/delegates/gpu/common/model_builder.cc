/* Copyright 2019-2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/common/model_builder.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/gpu/common/custom_parsers.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/lstm_parser.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_builder_helper.h"
#include "tensorflow/lite/delegates/gpu/common/model_builder_internal.h"
#include "tensorflow/lite/delegates/gpu/common/model_transformer.h"
#include "tensorflow/lite/delegates/gpu/common/object_reader.h"
#include "tensorflow/lite/delegates/gpu/common/operation_parser.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/transformations/model_transformations.h"
#include "tensorflow/lite/delegates/utils.h"
#include "tensorflow/lite/kernels/internal/reference/dequantize.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/tools/versioning/gpu_compatibility.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace gpu {
namespace {

absl::Status GetFullyConnectedAttributes(int weights_tensor_id,
                                         int bias_tensor_id,
                                         ObjectReader* reader,
                                         FullyConnectedAttributes* attr) {
  Tensor<HW, DataType::FLOAT32> weights;
  RETURN_IF_ERROR(reader->ReadTensor(weights_tensor_id, &weights));
  attr->weights.data = std::move(weights.data);
  attr->weights.id = weights.id;
  attr->weights.shape.h = 1;
  attr->weights.shape.w = 1;
  attr->weights.shape.o = weights.shape.h;
  attr->weights.shape.i = weights.shape.w;
  reader->ReadTensor(bias_tensor_id, &attr->bias).IgnoreError();  // optional
  return absl::OkStatus();
}

template <typename ParamsT>
absl::Status RetrieveBuiltinData(const TfLiteNode* tflite_node,
                                 const ParamsT** tf_options) {
  *tf_options = static_cast<const ParamsT*>(tflite_node->builtin_data);
  if (!*tf_options) {
    return absl::InternalError("Unable to retrieve builtin_data.");
  }
  return absl::OkStatus();
}

template <typename ParamsT>
absl::Status RetrieveCustomInitialData(const TfLiteNode* tflite_node,
                                       const ParamsT** tf_options) {
  *tf_options = static_cast<const ParamsT*>(tflite_node->custom_initial_data);
  if (!*tf_options) {
    return absl::InternalError("Unable to retrieve custom_initial_data.");
  }
  return absl::OkStatus();
}

// Creates a simple node that holds tensor value.
absl::Status NewConstNode(TensorFloat32 t, GraphFloat32* graph, Value** value) {
  ConstTensorAttributes attr;
  attr.tensor = std::move(t);
  Node* node = graph->NewNode();
  node->operation.attributes = attr;
  node->operation.type = ToString(OperationType::CONSTANT);
  *value = graph->NewValue();
  RETURN_IF_ERROR(graph->SetProducer(node->id, (*value)->id));
  // Keep data inside this tensor.
  (*value)->tensor.ref = attr.tensor.id;
  (*value)->tensor.type = attr.tensor.kType;
  (*value)->tensor.shape = attr.tensor.shape;
  return absl::OkStatus();
}

absl::Status ParseInputsWithConstTensor(Node* node, ObjectReader* reader,
                                        TensorOrScalar* tensor_or_scalar) {
  const std::string& opname = node->operation.type;

  // Determine runtime/constant tensors.
  const TfLiteTensor* input0 = reader->GetInputTensor(0);
  if (!input0) {
    return absl::InvalidArgumentError("Couldn't get the 1st input tensor for " +
                                      opname);
  }
  const TfLiteTensor* input1 = reader->GetInputTensor(1);
  if (!input1) {
    return absl::InvalidArgumentError("Couldn't get the 2nd input tensor for " +
                                      opname);
  }
  const bool constant_tensor0 = IsConstantTensor(input0);
  const bool constant_tensor1 = IsConstantTensor(input1);
  if (constant_tensor0 && constant_tensor1) {
    return absl::InvalidArgumentError("No runtime input tensors for " + opname);
  }
  const bool runtime_tensor0 = !constant_tensor0;
  const bool runtime_tensor1 = !constant_tensor1;

  if (runtime_tensor0 && runtime_tensor1) {
    RETURN_IF_ERROR(reader->AddInput(node, 0));
    RETURN_IF_ERROR(reader->AddInput(node, 1));
  } else {
    int runtime_tensor = 0;
    int constant_tensor = 1;
    TfLiteIntArray* constant_dims = input1->dims;
    if (constant_tensor0 && runtime_tensor1) {
      runtime_tensor = 1;
      constant_tensor = 0;
      constant_dims = input0->dims;
    }
    RETURN_IF_ERROR(reader->AddInput(node, runtime_tensor));
    if (constant_dims->size <= 0 || NumElements(constant_dims) == 1) {
      Tensor<Scalar, DataType::FLOAT32> tensor;
      RETURN_IF_ERROR(reader->ReadTensor(constant_tensor, &tensor));
      *tensor_or_scalar = tensor.data[0];
    } else {
      if (CheckIfLinearConvertible(constant_dims).ok()) {
        Tensor<Linear, DataType::FLOAT32> tensor;
        RETURN_IF_ERROR(reader->ReadTensor(constant_tensor, &tensor));
        *tensor_or_scalar = std::move(tensor);
      } else if (constant_dims->size == 2) {
        Tensor<HW, DataType::FLOAT32> tensor_hw;
        RETURN_IF_ERROR(reader->ReadTensor(constant_tensor, &tensor_hw));
        Tensor<HWC, DataType::FLOAT32> tensor;
        tensor.id = tensor_hw.id;
        tensor.shape = HWC(1, tensor_hw.shape.h, tensor_hw.shape.w);
        tensor.data = tensor_hw.data;
        *tensor_or_scalar = std::move(tensor);
      } else {
        Tensor<HWC, DataType::FLOAT32> tensor;
        RETURN_IF_ERROR(reader->ReadTensor(constant_tensor, &tensor));
        *tensor_or_scalar = std::move(tensor);
      }
    }
  }
  return absl::OkStatus();
}

absl::Status MaybeFuseActivationForElementwiseNode(
    OperationType operation_type, const TfLiteNode* tflite_node,
    GraphFloat32* graph, Node* node) {
  TfLiteFusedActivation activation = kTfLiteActNone;
  switch (operation_type) {
    case OperationType::MUL: {
      const TfLiteMulParams* tf_options;
      if (RetrieveBuiltinData(tflite_node, &tf_options).ok()) {
        activation = tf_options->activation;
      }
      break;
    }
    case OperationType::ADD: {
      const TfLiteAddParams* tf_options;
      if (RetrieveBuiltinData(tflite_node, &tf_options).ok()) {
        activation = tf_options->activation;
      }
      break;
    }
    case OperationType::SUB: {
      const TfLiteSubParams* tf_options;
      if (RetrieveBuiltinData(tflite_node, &tf_options).ok()) {
        activation = tf_options->activation;
      }
      break;
    }
    case OperationType::DIV: {
      const TfLiteDivParams* tf_options;
      if (RetrieveBuiltinData(tflite_node, &tf_options).ok()) {
        activation = tf_options->activation;
      }
      break;
    }
    default:
      // No activation expected.
      activation = kTfLiteActNone;
  }

  if (activation) {
    return MaybeFuseActivation(activation, graph, node);
  }
  return absl::OkStatus();
}

struct TensorInfo {
  std::vector<std::pair<TfLiteNode*, TfLiteRegistration*>> producers;
  std::vector<std::pair<TfLiteNode*, TfLiteRegistration*>> consumers;
};

absl::Status GetTensorInfo(const TfLiteContext* context, int tensor_id,
                           TensorInfo* result) {
  TfLiteIntArray* execution_plan = nullptr;
  if (context->GetExecutionPlan(const_cast<TfLiteContext*>(context),
                                &execution_plan) != kTfLiteOk) {
    return absl::UnavailableError("Unable to get graph execution plan.");
  }
  for (int i = 0; i < execution_plan->size; ++i) {
    const int node_index = execution_plan->data[i];
    TfLiteNode* node = nullptr;
    TfLiteRegistration* registration = nullptr;
    if (context->GetNodeAndRegistration(const_cast<TfLiteContext*>(context),
                                        node_index, &node,
                                        &registration) != kTfLiteOk) {
      return absl::UnavailableError(
          "Unable to get node and registration for node.");
    }
    for (int j = 0; j < node->inputs->size; ++j) {
      if (tensor_id == node->inputs->data[j]) {
        result->consumers.push_back({node, registration});
      }
    }
    for (int j = 0; j < node->outputs->size; ++j) {
      if (tensor_id == node->outputs->data[j]) {
        result->producers.push_back({node, registration});
      }
    }
  }
  return absl::OkStatus();
}

bool IsLogicalCode(int32_t builtin_code) {
  return builtin_code == kTfLiteBuiltinGreater ||
         builtin_code == kTfLiteBuiltinGreaterEqual ||
         builtin_code == kTfLiteBuiltinLess ||
         builtin_code == kTfLiteBuiltinLessEqual ||
         builtin_code == kTfLiteBuiltinEqual ||
         builtin_code == kTfLiteBuiltinNotEqual;
}

bool IsLogicalOp(tflite::gpu::OperationType op_type) {
  return op_type == tflite::gpu::OperationType::GREATER ||
         op_type == tflite::gpu::OperationType::GREATER_EQUAL ||
         op_type == tflite::gpu::OperationType::LESS ||
         op_type == tflite::gpu::OperationType::LESS_EQUAL ||
         op_type == tflite::gpu::OperationType::EQUAL ||
         op_type == tflite::gpu::OperationType::NOT_EQUAL;
}

class BatchedMatMulOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    return CheckGpuDelegateCompatibility(context, tflite_node, registration);
  }

  absl::Status Parse(const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration,
                     GraphFloat32* graph, ObjectReader* reader) final {
    if (reader->GetNumberOfRuntimeInputs() == 2) {
      Node* node = graph->NewNode();
      node->operation.type = ToString(OperationType::BATCHED_MATMUL);
      RETURN_IF_ERROR(reader->AddInput(node, 0));
      RETURN_IF_ERROR(reader->AddInput(node, 1));
      RETURN_IF_ERROR(reader->AddOutputs(node));
      return absl::OkStatus();
    } else if (reader->GetNumberOfRuntimeInputs() == 1) {
      // Second input is constant, replace with Convolution2D
      const TfLiteTensor* second_input = reader->GetInputTensor(1);
      if (!IsConstantTensor(second_input) || second_input->dims->size != 2) {
        // first input must be runtime and second is 2d constant tensor
        return absl::UnavailableError("Not supported batched mat mul case");
      }
      Node* node = graph->NewNode();
      node->operation.type = ToString(OperationType::CONVOLUTION_2D);
      RETURN_IF_ERROR(reader->AddInput(node, 0));
      RETURN_IF_ERROR(reader->AddOutputs(node));

      Tensor<HW, DataType::FLOAT32> weights;
      RETURN_IF_ERROR(reader->ReadTensor(1, &weights));
      Convolution2DAttributes attr;
      attr.weights.data.resize(weights.shape.w * weights.shape.h);
      for (int i = 0; i < weights.shape.w; ++i) {
        for (int j = 0; j < weights.shape.h; ++j) {
          attr.weights.data[i * weights.shape.h + j] =
              weights.data[j * weights.shape.w + i];
        }
      }
      attr.weights.id = weights.id;
      attr.weights.shape.h = 1;
      attr.weights.shape.w = 1;
      attr.weights.shape.o = weights.shape.w;
      attr.weights.shape.i = weights.shape.h;
      attr.strides = HW(1, 1);
      attr.dilations = HW(1, 1);
      attr.padding.appended = HW(0, 0);
      attr.padding.prepended = HW(0, 0);
      node->operation.attributes = std::move(attr);
      return absl::OkStatus();
    } else {
      return absl::UnavailableError("Not supported batched mat mul case");
    }
    return absl::OkStatus();
  }
};

class CastOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    TfLiteType src_type = context->tensors[tflite_node->inputs->data[0]].type;
    TfLiteType dst_type = context->tensors[tflite_node->outputs->data[0]].type;
    if (src_type == kTfLiteBool &&
        (dst_type == kTfLiteFloat16 || dst_type == kTfLiteFloat32)) {
      // check that we have next sequence:
      //   logical_op->bool_tensor->CAST->float_tensor.
      TensorInfo input_tensor_info;
      RETURN_IF_ERROR(GetTensorInfo(context, tflite_node->inputs->data[0],
                                    &input_tensor_info));
      if (input_tensor_info.producers.size() != 1 ||
          input_tensor_info.consumers.size() != 1 ||
          !IsLogicalCode(input_tensor_info.producers[0].second->builtin_code)) {
        return absl::UnavailableError("Not supported cast case");
      }
    }
    return CheckGpuDelegateCompatibility(context, tflite_node, registration);
  }

  absl::Status Parse(const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration,
                     GraphFloat32* graph, ObjectReader* reader) final {
    Node* node = graph->NewNode();
    node->operation.type = ToString(OperationType::CAST);
    RETURN_IF_ERROR(reader->AddInput(node, 0));
    RETURN_IF_ERROR(reader->AddOutputs(node));
    return absl::OkStatus();
  }
};

class ClampOperationsParser : public TFLiteOperationParser {
 public:
  explicit ClampOperationsParser(float clamp_a, float clamp_b)
      : clamp_a_(clamp_a), clamp_b_(clamp_b) {}
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    return absl::OkStatus();
  }

  absl::Status Parse(const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration,
                     GraphFloat32* graph, ObjectReader* reader) final {
    // clamp(v, a, b) = clamp(v - a, 0.0, b - a) + a;
    // We replace clamp(...) with sequence of elementwise ops:
    // substaction -> usual relu with alpha = 0.0 -> addition.
    // node_sub = v0 = v - a // add op (add -a)
    // node_relu = v1 = clamp(v0, 0.0, clip); // relu op alpha = 0.0,
    // clip = b - a;
    // node_add = v2 = v1 + a // add op (add a)
    Node* node_sub = graph->NewNode();
    Node* node_relu = graph->NewNode();
    Node* node_add = graph->NewNode();

    ElementwiseAttributes sub_attr;
    sub_attr.param = -clamp_a_;
    node_sub->operation.type = ToString(OperationType::ADD);
    node_sub->operation.attributes = std::move(sub_attr);

    ReLUAttributes relu_attr;
    relu_attr.alpha = 0.0f;
    relu_attr.clip = clamp_b_ - clamp_a_;
    node_relu->operation.type = ToString(OperationType::RELU);
    node_relu->operation.attributes = relu_attr;

    ElementwiseAttributes add_attr;
    add_attr.param = clamp_a_;
    node_add->operation.type = ToString(OperationType::ADD);
    node_add->operation.attributes = std::move(add_attr);

    RETURN_IF_ERROR(reader->AddInput(node_sub, 0));
    auto input = graph->FindInputs(node_sub->id)[0];

    Value* v0 = graph->NewValue();
    Value* v1 = graph->NewValue();
    v0->tensor.type = input->tensor.type;
    v0->tensor.shape = input->tensor.shape;
    v1->tensor.type = input->tensor.type;
    v1->tensor.shape = input->tensor.shape;

    RETURN_IF_ERROR(graph->SetProducer(node_sub->id, v0->id));
    RETURN_IF_ERROR(graph->AddConsumer(node_relu->id, v0->id));
    RETURN_IF_ERROR(graph->SetProducer(node_relu->id, v1->id));
    RETURN_IF_ERROR(graph->AddConsumer(node_add->id, v1->id));

    RETURN_IF_ERROR(reader->AddOutputs(node_add));
    return absl::OkStatus();
  }

 private:
  const float clamp_a_, clamp_b_;
};

class ConcatenationOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    RETURN_IF_ERROR(CheckMaxSupportedOpVersion(registration, 2));

    // TODO(eignasheva): add proper tensor availability checking
    // for (uint32_t idx = 0; idx < tflite_node->inputs->size; ++idx) {
    //   RETURN_IF_ERROR(CheckTensorIsAvailable(context, tflite_node, idx));
    // }
    // TODO(eignasheva): add axis checking.
    return CheckGpuDelegateCompatibility(context, tflite_node, registration);
  }

  absl::Status Parse(const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration,
                     GraphFloat32* graph, ObjectReader* reader) final {
    ConcatAttributes attr;
    // Read inputs first to make sure const node is added to a graph before
    // concat node to ensure topological order.
    std::vector<const Value*> inputs;
    for (uint32_t idx = 0; idx < tflite_node->inputs->size; ++idx) {
      Value* value;
      const auto status = reader->ReadValue(idx, &value);
      if (status.ok()) {
        inputs.push_back(value);
      } else {
        TensorFloat32 tensor;
        RETURN_IF_ERROR(reader->ReadTensor(idx, &tensor));
        Value* value;
        RETURN_IF_ERROR(NewConstNode(std::move(tensor), graph, &value));
        inputs.push_back(value);
      }
    }

    for (int i = 0; i < inputs.size(); ++i) {
      for (int j = 0; j < i; ++j) {
        if (inputs[i] == inputs[j]) {
          Node* node_copy = graph->NewNode();
          node_copy->operation.type = ToString(OperationType::COPY);
          RETURN_IF_ERROR(graph->AddConsumer(node_copy->id, inputs[j]->id));
          Value* copy_value = graph->NewValue();
          copy_value->tensor.type = inputs[j]->tensor.type;
          copy_value->tensor.shape = inputs[j]->tensor.shape;
          RETURN_IF_ERROR(graph->SetProducer(node_copy->id, copy_value->id));
          inputs[i] = copy_value;
          break;
        }
      }
    }

    Node* node = graph->NewNode();
    node->operation.type = ToString(OperationType::CONCAT);
    RETURN_IF_ERROR(reader->AddOutputs(node));
    for (int i = 0; i < inputs.size(); ++i) {
      RETURN_IF_ERROR(graph->AddConsumer(node->id, inputs[i]->id));
    }

    std::vector<BHWC> input_shapes;
    for (auto input : graph->FindInputs(node->id)) {
      input_shapes.push_back(input->tensor.shape);
    }
    RETURN_IF_ERROR(SetAxis(input_shapes, &attr.axis));

    // Guess axis.
    BHWC output_shape = graph->FindOutputs(node->id)[0]->tensor.shape;
    for (auto input : graph->FindInputs(node->id)) {
      if (input->tensor.shape.h != output_shape.h) {
        attr.axis = Axis::HEIGHT;
        break;
      }
      if (input->tensor.shape.w != output_shape.w) {
        attr.axis = Axis::WIDTH;
        break;
      }
      if (input->tensor.shape.c != output_shape.c) {
        attr.axis = Axis::CHANNELS;
        break;
      }
    }
    const TfLiteConcatenationParams* tf_options;
    RETURN_IF_ERROR(RetrieveBuiltinData(tflite_node, &tf_options));
    RETURN_IF_ERROR(MaybeFuseActivation(tf_options->activation, graph, node));
    node->operation.attributes = attr;
    return absl::OkStatus();
  }

 private:
  absl::Status SetAxis(const std::vector<BHWC>& input_shapes, Axis* axis) {
    *axis = Axis::BATCH;
    for (int i = 1; i < input_shapes.size(); i++) {
      if (input_shapes[0].h != input_shapes[i].h &&
          input_shapes[0].w != input_shapes[i].w &&
          input_shapes[0].c != input_shapes[i].c) {
        *axis = Axis::HEIGHT;
        break;
      }
    }
    if (*axis == Axis::BATCH) return absl::OkStatus();
    for (int i = 1; i < input_shapes.size(); i++) {
      if (input_shapes[0].b != input_shapes[i].b &&
          input_shapes[0].w != input_shapes[i].w &&
          input_shapes[0].c != input_shapes[i].c) {
        *axis = Axis::WIDTH;
        break;
      }
    }
    if (*axis == Axis::HEIGHT) return absl::OkStatus();
    for (int i = 1; i < input_shapes.size(); i++) {
      if (input_shapes[0].b != input_shapes[i].b &&
          input_shapes[0].h != input_shapes[i].h &&
          input_shapes[0].c != input_shapes[i].c) {
        *axis = Axis::CHANNELS;
        break;
      }
    }
    if (*axis == Axis::WIDTH) return absl::OkStatus();
    for (int i = 1; i < input_shapes.size(); i++) {
      if (input_shapes[0].b != input_shapes[i].b &&
          input_shapes[0].w != input_shapes[i].w &&
          input_shapes[0].h != input_shapes[i].h) {
        return absl::UnimplementedError(
            "Can concatenate tensors only by batch, height, width, or "
            "channels.");
      }
    }
    return absl::OkStatus();
  }
};

class Conv2DOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    RETURN_IF_ERROR(CheckMaxSupportedOpVersion(registration, 6));
    return CheckGpuDelegateCompatibility(context, tflite_node, registration);
  }

  absl::Status Parse(const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration,
                     GraphFloat32* graph, ObjectReader* reader) final {
    const TfLiteConvParams* tf_options;
    RETURN_IF_ERROR(RetrieveBuiltinData(tflite_node, &tf_options));
    Convolution2DAttributes attr;
    RETURN_IF_ERROR(ReadAttributes(tflite_node, tf_options, reader, &attr));

    const int runtime_inputs = reader->GetNumberOfRuntimeInputs();
    if (runtime_inputs == 2) {
      // weights are second runtime input
      const TfLiteTensor* src_tensor = reader->GetInputTensor(0);
      const TfLiteTensor* weights_tensor = reader->GetInputTensor(1);
      BHWC src_shape, weights_shape;
      RETURN_IF_ERROR(ExtractTensorShape(*src_tensor, &src_shape));
      RETURN_IF_ERROR(ExtractTensorShape(*weights_tensor, &weights_shape));
      if (src_shape.c != weights_shape.c) {
        return absl::InternalError(
            "No support of CONVOLUTION_2D with runtime grouped weights.");
      }

      Node* node = graph->NewNode();
      node->operation.type = ToString(OperationType::CONVOLUTION_2D);
      node->operation.attributes = std::move(attr);
      RETURN_IF_ERROR(reader->AddInput(node, 0));
      RETURN_IF_ERROR(reader->AddInput(node, 1));
      RETURN_IF_ERROR(reader->AddOutputs(node));
      RETURN_IF_ERROR(MaybeFuseActivation(tf_options->activation, graph, node));
      return absl::OkStatus();
    } else {
      // weights are constants
      const int src_group_size = attr.weights.shape.i;
      const int dst_group_size = attr.weights.shape.o / attr.groups;
      const bool supported_grouped_conv =
          src_group_size % 4 == 0 && dst_group_size % 4 == 0;
      if (attr.groups != 1 && !supported_grouped_conv) {
        // Not supported case, replace with usual convolutions:
        return ResolveGroupedConvolution(attr, tf_options, reader, graph);
      } else {
        Node* node = graph->NewNode();
        node->operation.type = ToString(OperationType::CONVOLUTION_2D);
        node->operation.attributes = std::move(attr);
        RETURN_IF_ERROR(reader->AddInput(node, 0));
        RETURN_IF_ERROR(reader->AddOutputs(node));
        RETURN_IF_ERROR(
            MaybeFuseActivation(tf_options->activation, graph, node));
        return absl::OkStatus();
      }
    }
  }

 private:
  absl::Status ReadAttributes(const TfLiteNode* tflite_node,
                              const TfLiteConvParams* tf_options,
                              ObjectReader* reader,
                              Convolution2DAttributes* attr) {
    const TfLiteTensor* src_tensor = reader->GetInputTensor(0);
    BHWC src_shape;
    RETURN_IF_ERROR(ExtractTensorShape(*src_tensor, &src_shape));
    const int runtime_inputs = reader->GetNumberOfRuntimeInputs();
    if (runtime_inputs == 1) {
      RETURN_IF_ERROR(reader->ReadTensor(1, &attr->weights));
      attr->groups = src_shape.c / attr->weights.shape.i;
    } else {
      const TfLiteTensor* weights_tensor = reader->GetInputTensor(1);
      if (!weights_tensor) {
        return absl::InternalError("Expected second runtime tensor.");
      }
      BHWC weights_shape;
      RETURN_IF_ERROR(ExtractTensorShape(*weights_tensor, &weights_shape));
      attr->weights.shape = OHWI(weights_shape.b, weights_shape.h,
                                 weights_shape.w, weights_shape.c);
      attr->groups = 1;
    }
    reader->ReadTensor(2, &attr->bias).IgnoreError();  // bias is optional
    attr->strides = ToHW(tf_options->stride_height, tf_options->stride_width);
    attr->dilations = HW(tf_options->dilation_height_factor,
                         tf_options->dilation_width_factor);
    UpdatePadding(tf_options->padding, src_shape, attr);
    return absl::OkStatus();
  }

  // Replace single grouped convolution(N = groups count) with this sequence:
  //  split input to N tensors in channels dim
  //  N usual convs
  //  concat N tensors to 1 output in channels dim
  absl::Status ResolveGroupedConvolution(const Convolution2DAttributes& attr,
                                         const TfLiteConvParams* tf_options,
                                         ObjectReader* reader,
                                         GraphFloat32* graph) {
    const TfLiteTensor* src_tensor = reader->GetInputTensor(0);
    const TfLiteTensor* dst_tensor = reader->GetOutputTensor(0);
    BHWC src_shape, dst_shape;
    RETURN_IF_ERROR(ExtractTensorShape(*src_tensor, &src_shape));
    RETURN_IF_ERROR(ExtractTensorShape(*dst_tensor, &dst_shape));

    DataType src_type = DataType::FLOAT32;
    if (src_tensor->type == kTfLiteFloat16) {
      src_type = DataType::FLOAT16;
    }
    DataType dst_type = DataType::FLOAT32;
    if (dst_tensor->type == kTfLiteFloat16) {
      dst_type = DataType::FLOAT16;
    }

    const int src_group_size = attr.weights.shape.i;
    const int dst_group_size = attr.weights.shape.o / attr.groups;

    Node* split_node = graph->NewNode();
    RETURN_IF_ERROR(reader->AddInput(split_node, 0));
    {
      SplitAttributes split_attr;
      split_attr.axis = Axis::CHANNELS;
      split_node->operation.type = ToString(OperationType::SPLIT);
      split_node->operation.attributes = split_attr;
    }

    std::vector<Node*> conv_nodes(attr.groups);
    std::vector<Value*> conv_src(attr.groups);
    std::vector<Value*> conv_dst(attr.groups);
    for (int i = 0; i < attr.groups; ++i) {
      conv_nodes[i] = graph->NewNode();
      conv_src[i] = graph->NewValue();
      conv_dst[i] = graph->NewValue();
      conv_src[i]->tensor.shape = src_shape;
      conv_src[i]->tensor.type = src_type;
      conv_src[i]->tensor.shape.c = src_group_size;
      conv_dst[i]->tensor.shape = dst_shape;
      conv_dst[i]->tensor.type = dst_type;
      conv_dst[i]->tensor.shape.c = dst_group_size;
      Convolution2DAttributes conv_attr;
      conv_attr = attr;
      conv_attr.groups = 1;
      conv_attr.weights.id = -1;
      conv_attr.weights.shape.o = dst_group_size;
      conv_attr.weights.data.resize(
          conv_attr.weights.shape.DimensionsProduct());
      for (int out_i = 0; out_i < dst_group_size; ++out_i) {
        for (int in_i = 0; in_i < src_group_size; ++in_i) {
          for (int ky = 0; ky < attr.weights.shape.h; ++ky) {
            for (int kx = 0; kx < attr.weights.shape.w; ++kx) {
              const int src_index = attr.weights.shape.LinearIndex(
                  {{i * dst_group_size + out_i, ky, kx, in_i}});
              const int dst_index =
                  conv_attr.weights.shape.LinearIndex({{out_i, ky, kx, in_i}});
              conv_attr.weights.data[dst_index] = attr.weights.data[src_index];
            }
          }
        }
      }
      conv_attr.bias.shape.v = dst_group_size;
      conv_attr.bias.data.resize(conv_attr.bias.shape.DimensionsProduct());
      for (int out_i = 0; out_i < dst_group_size; ++out_i) {
        if (i * dst_group_size + out_i < attr.bias.data.size()) {
          conv_attr.bias.data[out_i] =
              attr.bias.data[i * dst_group_size + out_i];
        } else {
          conv_attr.bias.data[out_i] = 0.0f;
        }
      }
      conv_nodes[i]->operation.type = ToString(OperationType::CONVOLUTION_2D);
      conv_nodes[i]->operation.attributes = conv_attr;

      RETURN_IF_ERROR(graph->SetProducer(split_node->id, conv_src[i]->id));
      RETURN_IF_ERROR(graph->AddConsumer(conv_nodes[i]->id, conv_src[i]->id));
      RETURN_IF_ERROR(graph->SetProducer(conv_nodes[i]->id, conv_dst[i]->id));
    }

    Node* concat_node = graph->NewNode();
    {
      ConcatAttributes concat_attr;
      concat_attr.axis = Axis::CHANNELS;
      concat_node->operation.type = ToString(OperationType::CONCAT);
      concat_node->operation.attributes = concat_attr;
    }
    for (int i = 0; i < attr.groups; ++i) {
      RETURN_IF_ERROR(graph->AddConsumer(concat_node->id, conv_dst[i]->id));
    }
    RETURN_IF_ERROR(reader->AddOutputs(concat_node));
    RETURN_IF_ERROR(
        MaybeFuseActivation(tf_options->activation, graph, concat_node));
    return absl::OkStatus();
  }
};

class CumsumOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    RETURN_IF_ERROR(CheckMaxSupportedOpVersion(registration, 1));
    return CheckGpuDelegateCompatibility(context, tflite_node, registration);
  }

  absl::Status Parse(const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration,
                     GraphFloat32* graph, ObjectReader* reader) final {
    Node* node = graph->NewNode();
    CumsumAttributes attr;
    const TfLiteTensor* input_tensor = reader->GetInputTensor(0);
    const TfLiteTensor* axis_tensor = reader->GetInputTensor(1);
    const TfLiteIntArray* shape = input_tensor->dims;
    const int tflite_axis = GetTensorData<int32_t>(axis_tensor)[0];
    const Axis axes[4] = {Axis::BATCH, Axis::WIDTH, Axis::HEIGHT,
                          Axis::CHANNELS};
    attr.axis = axes[tflite_axis + 4 - shape->size];
    node->operation.type = ToString(OperationType::CUMSUM);
    Tensor<BHWC, DataType::FLOAT32> inputs;
    node->operation.attributes = std::move(attr);
    RETURN_IF_ERROR(reader->AddInput(node, 0));
    RETURN_IF_ERROR(reader->AddOutputs(node));
    return absl::OkStatus();
  }
};

// Doesn't have a kernel implementation.
class DensifyOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    RETURN_IF_ERROR(CheckMaxSupportedOpVersion(registration, 1));
    return CheckGpuDelegateCompatibility(context, tflite_node, registration);
  }

  absl::Status Parse(const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration,
                     GraphFloat32* graph, ObjectReader* reader) final {
    Node* node = graph->NewNode();
    node->operation.type = ToString(OperationType::DENSIFY);
    const TfLiteTensor* const_tensor = reader->GetInputTensor(0);
    if (!const_tensor->sparsity) {
      return absl::InvalidArgumentError("Input tensor must be sparse.");
    }
    TensorFloat32 sparse_tensor;
    RETURN_IF_ERROR(reader->ReadTensor(0, &sparse_tensor));
    DensifyAttributes attributes;
    attributes.tensor = std::move(sparse_tensor);
    node->operation.attributes = attributes;
    return reader->AddOutputs(node);
  }
};

class DepthwiseConvolutionOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    RETURN_IF_ERROR(CheckMaxSupportedOpVersion(registration, 6));
    return CheckGpuDelegateCompatibility(context, tflite_node, registration);
  }

  absl::Status Parse(const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration,
                     GraphFloat32* graph, ObjectReader* reader) final {
    Node* node = graph->NewNode();
    node->operation.type = ToString(OperationType::DEPTHWISE_CONVOLUTION);
    RETURN_IF_ERROR(reader->AddInput(node, 0));
    RETURN_IF_ERROR(reader->AddOutputs(node));

    DepthwiseConvolution2DAttributes attr;
    const int runtime_inputs = reader->GetNumberOfRuntimeInputs();
    if (runtime_inputs == 2) {
      RETURN_IF_ERROR(reader->AddInput(node, 1));
      auto weights_shape = graph->FindInputs(node->id)[1]->tensor.shape;
      attr.weights.shape = OHWI(weights_shape.b, weights_shape.h,
                                weights_shape.w, weights_shape.c);
    } else {  // runtime_inputs == 1;
      RETURN_IF_ERROR(reader->ReadTensor(1, &attr.weights));
    }
    reader->ReadTensor(2, &attr.bias).IgnoreError();  // bias is optional
    const TfLiteDepthwiseConvParams* tf_options;
    RETURN_IF_ERROR(RetrieveBuiltinData(tflite_node, &tf_options));
    attr.strides = ToHW(tf_options->stride_height, tf_options->stride_width);
    attr.dilations = HW(std::max(1, tf_options->dilation_height_factor),
                        std::max(1, tf_options->dilation_width_factor));
    UpdatePadding(tf_options->padding,
                  graph->FindInputs(node->id)[0]->tensor.shape, &attr);
    RETURN_IF_ERROR(MaybeFuseActivation(tf_options->activation, graph, node));
    const int depth_multiplier = tf_options->depth_multiplier;
    if (depth_multiplier != 1) {
      const TfLiteTensor* input = reader->GetInputTensor(0);
      const TfLiteTensor* filter = reader->GetInputTensor(1);
      const TfLiteTensor* output = reader->GetOutputTensor(0);
      TransposeWeights(input, filter, output, depth_multiplier, &attr);
    }
    node->operation.attributes = std::move(attr);
    return absl::OkStatus();
  }

 private:
  // TFLite CPU stores weights as:
  //   [1, kernel_height, kernel_width, input_depth * depth_multiplier]
  // TFLite GPU stores weights as:
  //   [depth_multiplier, kernel_height, kernel_width, input_depth]
  static void TransposeWeights(const TfLiteTensor* input,
                               const TfLiteTensor* filter,
                               const TfLiteTensor* output, int depth_multiplier,
                               DepthwiseConvolution2DAttributes* attr) {
    const int input_depth = input->dims->data[3];
    const int filter_height = filter->dims->data[1];
    const int filter_width = filter->dims->data[2];
    const int output_depth = output->dims->data[3];
    Tensor<OHWI, DataType::FLOAT32> weights;
    weights.id = attr->weights.id;
    weights.shape =
        OHWI(output_depth, filter_height, filter_width, input_depth);
    weights.data.resize(weights.shape.DimensionsProduct());
    float* dst = &weights.data[0];
    for (int j = 0; j < output_depth; ++j) {
      const float* src = attr->weights.data.data() + j;
      for (int i = 0; i < filter_height * filter_width; ++i) {
        *dst = *src;
        dst++;
        src += output_depth;
      }
    }
    attr->weights = std::move(weights);
  }
};

class DepthToSpaceOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    return CheckGpuDelegateCompatibility(context, tflite_node, registration);
  }

  absl::Status Parse(const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration,
                     GraphFloat32* graph, ObjectReader* reader) final {
    Node* node = graph->NewNode();
    node->operation.type = ToString(OperationType::DEPTH_TO_SPACE);
    RETURN_IF_ERROR(reader->AddInput(node, 0));
    RETURN_IF_ERROR(reader->AddOutputs(node));
    const TfLiteDepthToSpaceParams* tf_options;
    RETURN_IF_ERROR(RetrieveBuiltinData(tflite_node, &tf_options));
    SpaceToDepthAttributes attr;
    attr.block_size = tf_options->block_size;
    node->operation.attributes = attr;
    return absl::OkStatus();
  }
};

class DequantizeOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    RETURN_IF_ERROR(CheckMaxSupportedOpVersion(registration, 3));
    return CheckGpuDelegateCompatibility(context, tflite_node, registration);
  }

  absl::Status Parse(const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration,
                     GraphFloat32* graph, ObjectReader* reader) final {
    // 'Dequantize' is rewritten as QuantizeAndDequantize since we are dealing
    // with floating-point versions of the original tensors.
    const int runtime_inputs = reader->GetNumberOfRuntimeInputs();
    if (runtime_inputs == 0) {
      // constant input, can be dequantized here
      ConstTensorAttributes attr;
      RETURN_IF_ERROR(reader->ReadTensor(0, &attr.tensor));
      Node* node = graph->NewNode();
      node->operation.attributes = attr;
      node->operation.type = ToString(OperationType::CONSTANT);
      return reader->AddOutputs(node);
    }
    Node* node = graph->NewNode();
    node->operation.type = ToString(OperationType::QUANTIZE_AND_DEQUANTIZE);
    // Non-constant dequantization.
    RETURN_IF_ERROR(reader->AddInput(node, 0));
    RETURN_IF_ERROR(reader->AddOutputs(node));

    // Quantization attributes should already be present in the input tensor.
    auto input_value = graph->FindInputs(node->id)[0];
    if (!input_value->quant_params) {
      if (runtime_inputs == 1) {
        // DEQUANTIZE op is preceded by DENSIFY op and doesn't have any
        // quantization params. The DEQUANTIZE op latter will be removed from
        // the graph in `MergeDensify` graph transformation.
        return absl::OkStatus();
      }
      return absl::InvalidArgumentError(
          "Encountered Dequantize input with no quant params");
    }
    QuantizeAndDequantizeAttributes attr;
    attr.min = input_value->quant_params.value().min;
    attr.max = input_value->quant_params.value().max;
    attr.scale = input_value->quant_params.value().scale;

    node->operation.attributes = attr;
    return absl::OkStatus();
  }
};

class ElementwiseOperationParser : public TFLiteOperationParser {
 public:
  explicit ElementwiseOperationParser(OperationType operation_type)
      : operation_type_(operation_type) {}

  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    const int kMaxSupportedOpVersion =
        operation_type_ == OperationType::MUL ? 3 : 2;
    RETURN_IF_ERROR(
        CheckMaxSupportedOpVersion(registration, kMaxSupportedOpVersion));
    if (IsLogicalOp(operation_type_)) {
      TensorInfo output_tensor_info;
      RETURN_IF_ERROR(GetTensorInfo(context, tflite_node->outputs->data[0],
                                    &output_tensor_info));
      if (output_tensor_info.producers.size() != 1 ||
          output_tensor_info.consumers.size() != 1) {
        return absl::UnavailableError("Not supported logical op case");
      }
      const auto& next_node = output_tensor_info.consumers[0];
      TfLiteType dst_type =
          context->tensors[next_node.first->outputs->data[0]].type;
      if (next_node.second->builtin_code == kTfLiteBuiltinCast &&
          (dst_type == kTfLiteFloat16 || dst_type == kTfLiteFloat32)) {
        return absl::OkStatus();
      } else {
        return absl::UnimplementedError("Not supported logical op case.");
      }
    }
    return CheckGpuDelegateCompatibility(context, tflite_node, registration);
  }

  absl::Status Parse(const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration,
                     GraphFloat32* graph, ObjectReader* reader) final {
    Node* node = graph->NewNode();
    node->operation.type = ToString(operation_type_);
    if (operation_type_ == OperationType::ADD) {
      ElementwiseAttributes attr;
      node->operation.attributes = std::move(attr);
    }

    if (IsOneArgumentOperation()) {
      RETURN_IF_ERROR(reader->VerifyInputsConstsOutputs(tflite_node,
                                                        /*runtime_inputs=*/1,
                                                        /*const_inputs=*/0,
                                                        /*outputs=*/1));

      RETURN_IF_ERROR(reader->AddInput(node, 0));
    } else if (IsTwoArgumentOperation() &&
               reader
                   ->VerifyInputsConstsOutputs(tflite_node,
                                               /*runtime_inputs=*/2,
                                               /*const_inputs=*/0,
                                               /*outputs=*/1)
                   .ok()) {
      if (tflite_node->inputs->size != 2) {
        return absl::InvalidArgumentError("Applies only two input tensors");
      }
      const TfLiteTensor* input0 = reader->GetInputTensor(0);
      const TfLiteTensor* input1 = reader->GetInputTensor(1);

      // TODO(b/166831113): Support the same inputs for operations.
      if (input0 == input1) {
        if (operation_type_ == OperationType::MUL) {
          // replace MUL(A, A) with SQUARE(A)
          node->operation.type = ToString(OperationType::SQUARE);
          RETURN_IF_ERROR(reader->AddInput(node, 0));
        } else if (operation_type_ == OperationType::ADD) {
          // replace ADD(A, A) with MUL(A, 2.0)
          node->operation.type = ToString(OperationType::MUL);
          ElementwiseAttributes attr;
          attr.param = 2.0f;
          node->operation.attributes = std::move(attr);
          RETURN_IF_ERROR(reader->AddInput(node, 0));
        } else {
          return absl::UnimplementedError(
              "No support of few identical inputs in the same operation.");
        }
      } else {
        int input_tensor0 = 0;
        int input_tensor1 = 1;
        if (operation_type_ == OperationType::MUL ||
            operation_type_ == OperationType::ADD) {
          // The "larger" input tensor must be bound to 1st input and the
          // "smaller" input tensor must be bound to 2nd input.
          BHWC shape0;
          RETURN_IF_ERROR(ExtractTensorShape(*input0, &shape0));
          BHWC shape1;
          RETURN_IF_ERROR(ExtractTensorShape(*input1, &shape1));
          if (shape0.h <= shape1.h && shape0.w <= shape1.w &&
              shape0.c == shape1.c) {
            input_tensor0 = 1;
            input_tensor1 = 0;
          }
        }

        RETURN_IF_ERROR(reader->AddInput(node, input_tensor0));
        RETURN_IF_ERROR(reader->AddInput(node, input_tensor1));
      }
    } else if (IsTwoArgumentOperationWithConst()) {
      RETURN_IF_ERROR(reader->VerifyInputsConstsOutputs(tflite_node,
                                                        /*runtime_inputs=*/1,
                                                        /*const_inputs=*/1,
                                                        /*outputs=*/1));
      ElementwiseAttributes attr;
      RETURN_IF_ERROR(ParseInputsWithConstTensor(node, reader, &attr.param));
      attr.runtime_tensor_is_second =
          IsConstantTensor(reader->GetInputTensor(0));
      node->operation.attributes = std::move(attr);
    } else {
      return absl::InvalidArgumentError("Incorrect operation type passed");
    }

    RETURN_IF_ERROR(reader->AddOutputs(node));
    return MaybeFuseActivationForElementwiseNode(operation_type_, tflite_node,
                                                 graph, node);
  }

 private:
  absl::Status GetActivation(const TfLiteNode* tflite_node,
                             TfLiteFusedActivation* activation) const {
    if (operation_type_ == OperationType::DIV) {
      const TfLiteDivParams* tf_options;
      auto status = RetrieveBuiltinData(tflite_node, &tf_options);
      *activation = status.ok() ? tf_options->activation : kTfLiteActNone;
      return absl::OkStatus();
    }
    if (operation_type_ == OperationType::SUB) {
      const TfLiteSubParams* tf_options;
      auto status = RetrieveBuiltinData(tflite_node, &tf_options);
      *activation = status.ok() ? tf_options->activation : kTfLiteActNone;
      return absl::OkStatus();
    }

    // Return kTfLiteActNone as other ops either do not have TfLiteXxxParams or
    // TfLiteXxxParams.activation.
    *activation = kTfLiteActNone;
    return absl::OkStatus();
  }

  bool IsOneArgumentOperation() const {
    switch (operation_type_) {
      case OperationType::ABS:
      case OperationType::COPY:
      case OperationType::COS:
      case OperationType::ELU:
      case OperationType::EXP:
      case OperationType::FLOOR:
      case OperationType::LOG:
      case OperationType::NEG:
      case OperationType::RSQRT:
      case OperationType::SIGMOID:
      case OperationType::SIN:
      case OperationType::SQRT:
      case OperationType::SQUARE:
      case OperationType::TANH:
        return true;
      default:
        return false;
    }
  }

  bool IsTwoArgumentOperation() const {
    switch (operation_type_) {
      case OperationType::ADD:
      case OperationType::DIV:
      case OperationType::EQUAL:
      case OperationType::FLOOR_DIV:
      case OperationType::FLOOR_MOD:
      case OperationType::GREATER:
      case OperationType::GREATER_EQUAL:
      case OperationType::LESS:
      case OperationType::LESS_EQUAL:
      case OperationType::MAXIMUM:
      case OperationType::MINIMUM:
      case OperationType::MUL:
      case OperationType::NOT_EQUAL:
      case OperationType::POW:
      case OperationType::SQUARED_DIFF:
      case OperationType::SUB:
        return true;
      default:
        return false;
    }
  }

  bool IsTwoArgumentOperationWithConst() const {
    switch (operation_type_) {
      case OperationType::ADD:
      case OperationType::DIV:
      case OperationType::EQUAL:
      case OperationType::FLOOR_DIV:
      case OperationType::FLOOR_MOD:
      case OperationType::GREATER:
      case OperationType::GREATER_EQUAL:
      case OperationType::LESS:
      case OperationType::LESS_EQUAL:
      case OperationType::MAXIMUM:
      case OperationType::MINIMUM:
      case OperationType::MUL:
      case OperationType::NOT_EQUAL:
      case OperationType::POW:
      case OperationType::SQUARED_DIFF:
      case OperationType::SUB:
        return true;
      default:
        return false;
    }
  }

  OperationType operation_type_;
};

class FullyConnectedOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    RETURN_IF_ERROR(CheckMaxSupportedOpVersion(registration, 9));
    // TODO(eignasheva): check input shape
    return CheckGpuDelegateCompatibility(context, tflite_node, registration);
  }

  absl::Status Parse(const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration,
                     GraphFloat32* graph, ObjectReader* reader) final {
    const TfLiteFullyConnectedParams* tf_options;
    RETURN_IF_ERROR(RetrieveBuiltinData(tflite_node, &tf_options));

    if (reader->GetNumberOfRuntimeInputs() == 2) {
      // Create Convolution2D, so as it supports runtime weights.
      Node* node = graph->NewNode();
      node->operation.type = ToString(OperationType::CONVOLUTION_2D);
      RETURN_IF_ERROR(reader->AddInput(node, 0));
      RETURN_IF_ERROR(reader->AddInput(node, 1));
      RETURN_IF_ERROR(reader->AddOutputs(node));

      Convolution2DAttributes attr;
      reader->ReadTensor(2, &attr.bias).IgnoreError();  // bias is optional

      attr.strides = HW(1, 1);
      attr.dilations = HW(1, 1);
      attr.padding.appended = HW(0, 0);
      attr.padding.prepended = HW(0, 0);
      RETURN_IF_ERROR(MaybeFuseActivation(tf_options->activation, graph, node));
      node->operation.attributes = std::move(attr);
      return absl::OkStatus();
    }
    Node* node = graph->NewNode();
    RETURN_IF_ERROR(reader->AddInput(node, 0));

    if (tf_options->weights_format !=
        kTfLiteFullyConnectedWeightsFormatDefault) {
      return absl::UnimplementedError(
          "Unsupported FullyConnected weights format.");
    }

    FullyConnectedAttributes attr;
    RETURN_IF_ERROR(GetFullyConnectedAttributes(1, 2, reader, &attr));
    const int weights_width = attr.weights.shape.i;

    auto input = graph->FindInputs(node->id)[0];
    int batch_size = input->tensor.shape.b;
    if (input->tensor.shape.DimensionsProduct() / batch_size != weights_width) {
      return absl::UnimplementedError(
          "Amount of input data should match weights width");
    }

    Node* conv = node;
    if (input->tensor.shape.h != 1 || input->tensor.shape.w != 1) {
      auto& reshape = node;
      conv = graph->NewNode();  // reset conv pointer!
      Value* reshaped_value = graph->NewValue();
      reshaped_value->tensor.type = DataType::FLOAT32;
      reshaped_value->tensor.shape =
          BHWC(input->tensor.shape.b, 1, 1, weights_width);
      RETURN_IF_ERROR(graph->SetProducer(reshape->id, reshaped_value->id));
      reshape->operation.type = ToString(OperationType::RESHAPE);
      ReshapeAttributes attr;
      attr.new_shape = reshaped_value->tensor.shape;
      reshape->operation.attributes = attr;
      RETURN_IF_ERROR(graph->AddConsumer(conv->id, reshaped_value->id));
    }

    conv->operation.type = ToString(OperationType::FULLY_CONNECTED);
    conv->operation.attributes = std::move(attr);
    absl::Status result = reader->AddOutputs(conv);
    RETURN_IF_ERROR(MaybeFuseActivation(tf_options->activation, graph, conv));

    return result;
  }
};

class HardSwishOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    return CheckGpuDelegateCompatibility(context, tflite_node, registration);
  }

  absl::Status Parse(const TfLiteNode*, const TfLiteRegistration*,
                     GraphFloat32* graph, ObjectReader* reader) final {
    Node* node = graph->NewNode();
    node->operation.type = ToString(OperationType::HARD_SWISH);
    RETURN_IF_ERROR(reader->AddInput(node, 0));
    return reader->AddOutputs(node);
  }
};

// Basic LSTM Cell:
//
//  1name = name is at input  index 1
//  name1 = name is at output index 1
//
//    0input     1prev_activ
//       \        /
//        [[concat]]
//             \
//       concat_temp2  2weights  3biases
//              \      /        /
//             [[fully-connected]]
//               \
//         activ_temp3    4prev_state
//                 \      /
//                 [[LSTM]]
//                 /      \
//           new_state1    activation0
//
// For full LSTM cells, see this blog post:
// https://colah.github.io/posts/2015-08-Understanding-LSTMs/
// In addition to Peephole connections and Combined Input Forget Gates (CIFG)
// described in that post, this code also adds the following optional features:
// - Configurable activations (sigmoid or TANH)
// - L2 Normalization of gates: https://arxiv.org/abs/1607.06450
// - Output projection:
//     https://www.isca-speech.org/archive/interspeech_2014/i14_0338.html
// - Configurable clipping of cell state and output state.
class LSTMOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    RETURN_IF_ERROR(CheckMaxSupportedOpVersion(registration, 4));
    return CheckGpuDelegateCompatibility(context, tflite_node, registration);
  }

  absl::Status Parse(const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration,
                     GraphFloat32* graph, ObjectReader* reader) final {
    const TfLiteLSTMParams* tf_options;
    RETURN_IF_ERROR(RetrieveBuiltinData(tflite_node, &tf_options));
    switch (tf_options->kernel_type) {
      case kTfLiteLSTMFullKernel:
        return ParseFull(tflite_node, registration, graph, reader, tf_options);
      case kTfLiteLSTMBasicKernel:
        return ParseBasic(tflite_node, registration, graph, reader, tf_options);
    }
  }

  absl::flat_hash_map<int, ValueId> GetNewValueIdsForVariableInputNodes()
      final {
    return new_variable_input_value_map_;
  }

 private:
  absl::Status ParseBasic(const TfLiteNode* tflite_node,
                          const TfLiteRegistration* registration,
                          GraphFloat32* graph, ObjectReader* reader,
                          const TfLiteLSTMParams* tf_options) {
    if (tflite_node->inputs->size != 5) {
      return absl::InvalidArgumentError("LSTM should have 5 input tensors");
    }
    if (tflite_node->outputs->size != 4) {
      return absl::InvalidArgumentError("LSTM should have 4 output tensors");
    }
    RETURN_IF_ERROR(CheckBasicParameters(tf_options));

    Node* concat_node = graph->NewNode();
    concat_node->operation.type = ToString(OperationType::CONCAT);
    ConcatAttributes concat_attr;
    concat_attr.axis = Axis::CHANNELS;
    concat_node->operation.attributes = concat_attr;

    Node* fc_node = graph->NewNode();
    fc_node->operation.type = ToString(OperationType::FULLY_CONNECTED);
    FullyConnectedAttributes fc_attr;
    RETURN_IF_ERROR(GetFullyConnectedAttributes(2, 3, reader, &fc_attr));
    fc_node->operation.attributes = std::move(fc_attr);

    Node* lstm_node = graph->NewNode();
    lstm_node->operation.type = ToString(OperationType::LSTM);
    LstmAttributes lstm_attr;
    lstm_attr.kernel_type = LstmKernelType::BASIC;
    lstm_node->operation.attributes = lstm_attr;

    Value* concat_temp;
    int concat_tensor_idx = tflite_node->outputs->data[2];
    RETURN_IF_ERROR(
        reader->ReadValueByTensorIdx(concat_tensor_idx, &concat_temp));
    Value* activ_temp;
    int activ_tensor_idx = tflite_node->outputs->data[3];
    RETURN_IF_ERROR(
        reader->ReadValueByTensorIdx(activ_tensor_idx, &activ_temp));

    RETURN_IF_ERROR(reader->AddInput(concat_node, 0));  // input
    RETURN_IF_ERROR(reader->AddInput(concat_node, 1));  // prev_activ
    RETURN_IF_ERROR(graph->SetProducer(concat_node->id, concat_temp->id));

    RETURN_IF_ERROR(graph->AddConsumer(fc_node->id, concat_temp->id));
    RETURN_IF_ERROR(graph->SetProducer(fc_node->id, activ_temp->id));

    RETURN_IF_ERROR(graph->AddConsumer(lstm_node->id, activ_temp->id));
    RETURN_IF_ERROR(reader->AddInput(lstm_node, 4));   // prev_state
    RETURN_IF_ERROR(reader->AddOutput(lstm_node, 1));  // new_state
    RETURN_IF_ERROR(reader->AddOutput(lstm_node, 0));  // activation

    return absl::OkStatus();
  }

  absl::Status CheckBasicParameters(const TfLiteLSTMParams* tf_options) {
    if (tf_options->activation != kTfLiteActTanh) {
      return absl::UnimplementedError("Only TANH activation is supported.");
    }
    if (tf_options->cell_clip != 0.0f) {
      return absl::UnimplementedError("cell_clip is not supported.");
    }
    if (tf_options->proj_clip != 0.0f) {
      return absl::UnimplementedError("proj_clip is not supported.");
    }
    return absl::OkStatus();
  }

  absl::Status ParseFull(const TfLiteNode* tflite_node,
                         const TfLiteRegistration* registration,
                         GraphFloat32* graph, ObjectReader* reader,
                         const TfLiteLSTMParams* tf_options) {
    // Invoke full LSTM parser
    RETURN_IF_ERROR(ParseLSTMAttributes(tflite_node, registration, graph,
                                        reader, tf_options,
                                        &new_variable_input_value_map_));
    return absl::OkStatus();
  }

  absl::Status CheckFullParameters(const TfLiteLSTMParams* tf_options) {
    if (tf_options->activation != kTfLiteActSigmoid &&
        tf_options->activation != kTfLiteActTanh) {
      return absl::UnimplementedError(
          "Only sigmoid or tanh activation is supported.");
    }

    return absl::OkStatus();
  }

  absl::flat_hash_map<int, ValueId> new_variable_input_value_map_;
};

class OneHotOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    RETURN_IF_ERROR(CheckMaxSupportedOpVersion(registration, 1));
    return CheckGpuDelegateCompatibility(context, tflite_node, registration);
  }

  absl::Status Parse(const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration,
                     GraphFloat32* graph, ObjectReader* reader) final {
    Node* node = graph->NewNode();
    OneHotAttributes attr;
    const TfLiteTensor* on_tensor = reader->GetInputTensor(2);
    const TfLiteTensor* off_tensor = reader->GetInputTensor(3);
    attr.on_value = GetTensorData<float>(on_tensor)[0];
    attr.off_value = GetTensorData<float>(off_tensor)[0];
    node->operation.type = ToString(OperationType::ONE_HOT);
    node->operation.attributes = std::move(attr);
    RETURN_IF_ERROR(reader->AddInput(node, 0));
    RETURN_IF_ERROR(reader->AddOutputs(node));
    return absl::OkStatus();
  }
};

class PackOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    return CheckGpuDelegateCompatibility(context, tflite_node, registration);
  }

  absl::Status Parse(const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration,
                     GraphFloat32* graph, ObjectReader* reader) final {
    if (tflite_node->inputs->size == 1) {
      // Pack with single input can be replaced with Reshape
      Node* node = graph->NewNode();
      node->operation.type = ToString(OperationType::RESHAPE);
      RETURN_IF_ERROR(reader->AddInput(node, 0));
      RETURN_IF_ERROR(reader->AddOutputs(node));
      // New shape comes from output shape.
      ReshapeAttributes attr;
      attr.new_shape = graph->FindOutputs(node->id)[0]->tensor.shape;
      node->operation.attributes = attr;
      return absl::OkStatus();
    } else {
      // Pack with few inputs can be replaced with Concat
      const TfLitePackParams* tf_options;
      RETURN_IF_ERROR(RetrieveBuiltinData(tflite_node, &tf_options));

      // Read inputs first to make sure const node is added to a graph before
      // concat node to ensure topological order.
      std::vector<const Value*> inputs;
      for (uint32_t idx = 0; idx < tflite_node->inputs->size; ++idx) {
        Value* value;
        const auto status = reader->ReadValue(idx, &value);
        if (status.ok()) {
          inputs.push_back(value);
        } else {
          TensorFloat32 tensor;
          RETURN_IF_ERROR(reader->ReadTensor(idx, &tensor));
          Value* value;
          RETURN_IF_ERROR(NewConstNode(std::move(tensor), graph, &value));
          inputs.push_back(value);
        }
      }

      const TfLiteTensor* output = reader->GetOutputTensor(0);
      ConcatAttributes attr;
      RETURN_IF_ERROR(
          ExtractAxisFromIndex(*output, tf_options->axis, &attr.axis));
      BHWC output_shape;
      RETURN_IF_ERROR(ExtractTensorShape(*output, &output_shape));
      BHWC input_required_shape = output_shape;
      input_required_shape.set(attr.axis, 1);
      for (int i = 0; i < inputs.size(); ++i) {
        BHWC input_shape = inputs[i]->tensor.shape;
        if (input_shape != input_required_shape) {
          // GPU delegates does not support implicit shapes transformations
          // adding explicit Reshape
          Node* node_reshape = graph->NewNode();
          node_reshape->operation.type = ToString(OperationType::RESHAPE);
          ReshapeAttributes reshape_attr;
          reshape_attr.new_shape = input_required_shape;
          node_reshape->operation.attributes = reshape_attr;
          RETURN_IF_ERROR(graph->AddConsumer(node_reshape->id, inputs[i]->id));
          Value* copy_value = graph->NewValue();
          copy_value->tensor.type = inputs[i]->tensor.type;
          copy_value->tensor.shape = input_required_shape;
          RETURN_IF_ERROR(graph->SetProducer(node_reshape->id, copy_value->id));
          inputs[i] = copy_value;
        }
      }

      Node* node = graph->NewNode();
      node->operation.type = ToString(OperationType::CONCAT);
      RETURN_IF_ERROR(reader->AddOutputs(node));
      for (const Value* input : inputs) {
        RETURN_IF_ERROR(graph->AddConsumer(node->id, input->id));
      }
      node->operation.attributes = attr;
      return absl::OkStatus();
    }
  }
};

class PReLUOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    RETURN_IF_ERROR(CheckMaxSupportedOpVersion(registration, 1));
    // TODO(eignasheva): add params check
    return absl::OkStatus();
  }
  absl::Status Parse(const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration,
                     GraphFloat32* graph, ObjectReader* reader) final {
    Node* node = graph->NewNode();
    node->operation.type = ToString(OperationType::PRELU);
    RETURN_IF_ERROR(reader->AddInput(node, 0));
    auto input_shape = graph->FindInputs(node->id)[0]->tensor.shape;

    PReLUAttributes attr;
    Tensor<Linear, DataType::FLOAT32> linear_alpha;
    absl::Status status = reader->ReadTensor(1, &linear_alpha);
    if (status.ok()) {
      if (linear_alpha.shape.v != input_shape.c) {
        return absl::InvalidArgumentError(
            "Linear alpha shape does not match the number of input channels.");
      }
      attr.alpha = std::move(linear_alpha);
    } else {
      Tensor<HWC, DataType::FLOAT32> hwc_alpha;
      RETURN_IF_ERROR(reader->ReadTensor(1, &hwc_alpha));
      if (hwc_alpha.shape.h != input_shape.h ||
          hwc_alpha.shape.w != input_shape.w ||
          hwc_alpha.shape.c != input_shape.c) {
        return absl::InvalidArgumentError(
            "Alpha shape does not match input shape.");
      }
      attr.alpha = std::move(hwc_alpha);
    }
    node->operation.attributes = std::move(attr);
    return reader->AddOutputs(node);
  }
};

class PadOperationParser : public TFLiteOperationParser {
 public:
  explicit PadOperationParser(bool mirror_pad) : mirror_pad_(mirror_pad) {}

  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    RETURN_IF_ERROR(CheckMaxSupportedOpVersion(registration, 2));
    return CheckGpuDelegateCompatibility(context, tflite_node, registration);
  }

  absl::Status Parse(const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration,
                     GraphFloat32* graph, ObjectReader* reader) final {
    Node* node = graph->NewNode();
    node->operation.type = ToString(OperationType::PAD);
    RETURN_IF_ERROR(reader->AddInput(node, 0));
    RETURN_IF_ERROR(reader->AddOutputs(node));

    PadAttributes attr;
    if (mirror_pad_) {
      attr.type = PaddingContentType::REFLECT;
    } else /*zero pad*/ {
      attr.type = PaddingContentType::ZEROS;
    }

    Tensor<HW, DataType::INT32> paddings;
    RETURN_IF_ERROR(reader->ReadTensor(1, &paddings));

    if (paddings.shape.h == 4 && paddings.shape.w == 2) {
      // 4x2 tensor with paddings.
      attr.prepended = BHWC(paddings.data[0], paddings.data[2],
                            paddings.data[4], paddings.data[6]);
      attr.appended = BHWC(paddings.data[1], paddings.data[3], paddings.data[5],
                           paddings.data[7]);
    } else if (paddings.shape.h == 3 && paddings.shape.w == 2) {
      // 3x2 tensor with paddings.
      attr.prepended =
          BHWC(1, paddings.data[0], paddings.data[2], paddings.data[4]);
      attr.appended =
          BHWC(1, paddings.data[1], paddings.data[3], paddings.data[5]);
    } else {
      // It shouldn't fail here since it's checked at IsSupported().
      return absl::InvalidArgumentError(
          "Paddings tensor has unexpected shape.");
    }
    node->operation.attributes = attr;
    return absl::OkStatus();
  }

 private:
  bool mirror_pad_ = false;
};

class Pooling2DOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    RETURN_IF_ERROR(CheckMaxSupportedOpVersion(registration, 2));
    return CheckGpuDelegateCompatibility(context, tflite_node, registration);
  }

 public:
  explicit Pooling2DOperationParser(PoolingType type) : type_(type) {}

  absl::Status Parse(const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration,
                     GraphFloat32* graph, ObjectReader* reader) final {
    Node* node = graph->NewNode();
    node->operation.type = ToString(OperationType::POOLING_2D);
    RETURN_IF_ERROR(reader->AddInput(node, 0));
    RETURN_IF_ERROR(reader->AddOutput(node, 0));

    Pooling2DAttributes attr;
    attr.type = type_;

    auto input_shape = graph->FindInputs(node->id)[0]->tensor.shape;

    // Check whether there are custom options encoded. It happens if operation
    // is MaxPoolingWithArgmax2D. There is no way to read
    // tflite_node->builtin_code, so, simply check whether custom data is
    // available.
    const TfLitePoolParams* tf_options;
    if (!RetrieveCustomInitialData(tflite_node, &tf_options).ok()) {
      RETURN_IF_ERROR(RetrieveBuiltinData(tflite_node, &tf_options));
    }

    RETURN_IF_ERROR(MaybeFuseActivation(tf_options->activation, graph, node));
    // Second output is optional. It is not required, it but must be added after
    // MaybeAddFusedActivation function is called
    reader->AddOutput(node, 1).IgnoreError();

    // First output is the result of pooling operation, while second output is
    // indices used for pooling.
    auto outputs = graph->FindOutputs(node->id);
    attr.output_indices = outputs.size() == 2;
    if (attr.output_indices) {
      // Fix data type for output indices. In the model it is set as float32.
      outputs[1]->tensor.type = DataType::INT32;
    }
    RETURN_IF_ERROR(ParsePoolingAttributes(tf_options, input_shape, &attr));
    node->operation.attributes = attr;
    return absl::OkStatus();
  }

 private:
  const PoolingType type_;
};

class ReduceOperationParser : public TFLiteOperationParser {
 public:
  explicit ReduceOperationParser(OperationType operation_type)
      : operation_type_(operation_type) {}

  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    return CheckGpuDelegateCompatibility(context, tflite_node, registration);
  }

  absl::Status Parse(const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration,
                     GraphFloat32* graph, ObjectReader* reader) final {
    Node* node = graph->NewNode();
    node->operation.type = ToString(operation_type_);
    RETURN_IF_ERROR(reader->AddInput(node, 0));

    const TfLiteReducerParams* tf_options;
    RETURN_IF_ERROR(RetrieveBuiltinData(tflite_node, &tf_options));

    ReduceAttributes attr;
    const TfLiteTensor* input = reader->GetInputTensor(0);
    const TfLiteTensor* axes = reader->GetInputTensor(1);
    for (int i = 0; i < NumElements(axes->dims); i++) {
      Axis axis;
      RETURN_IF_ERROR(ExtractAxisFromIndex(*input, axes->data.i32[i], &axis));
      attr.dims.insert(axis);
    }
    node->operation.attributes = attr;

    if (!tf_options->keep_dims) {
      // GPU delegates does not support implicit shapes transformations
      // adding explicit Reshape
      const auto& input_tensor = graph->FindInputs(node->id)[0]->tensor;
      auto reduce_output_shape = input_tensor.shape;
      for (auto axis : attr.dims) {
        reduce_output_shape.set(axis, 1);
      }
      Node* node_reshape = graph->NewNode();
      node_reshape->operation.type = ToString(OperationType::RESHAPE);
      ReshapeAttributes reshape_attr;
      const TfLiteTensor* output = reader->GetOutputTensor(0);
      RETURN_IF_ERROR(ExtractTensorShape(*output, &reshape_attr.new_shape));
      node_reshape->operation.attributes = reshape_attr;
      Value* reduce_result = graph->NewValue();
      reduce_result->tensor.type = input_tensor.type;
      reduce_result->tensor.shape = reduce_output_shape;

      RETURN_IF_ERROR(graph->SetProducer(node->id, reduce_result->id));
      RETURN_IF_ERROR(graph->AddConsumer(node_reshape->id, reduce_result->id));
      RETURN_IF_ERROR(reader->AddOutputs(node_reshape));
    } else {
      RETURN_IF_ERROR(reader->AddOutputs(node));
    }
    return absl::OkStatus();
  }

 private:
  const OperationType operation_type_;
};

class QuantizeOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    RETURN_IF_ERROR(CheckMaxSupportedOpVersion(registration, 2));
    return CheckGpuDelegateCompatibility(context, tflite_node, registration);
  }

  absl::Status Parse(const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration,
                     GraphFloat32* graph, ObjectReader* reader) final {
    // 'Quantize' is rewritten as QuantizeAndDequantize since we are dealing
    // with floating-point versions of the original tensors.
    Node* node = graph->NewNode();
    node->operation.type = ToString(OperationType::QUANTIZE_AND_DEQUANTIZE);
    RETURN_IF_ERROR(reader->AddInput(node, 0));
    RETURN_IF_ERROR(reader->AddOutputs(node));

    // Quantization attributes should already be present in the output tensor.
    auto output_value = graph->FindOutputs(node->id)[0];
    if (!output_value->quant_params) {
      return absl::InvalidArgumentError(
          "Encountered Quantize output with no quant params");
    }
    QuantizeAndDequantizeAttributes attr;
    attr.min = output_value->quant_params.value().min;
    attr.max = output_value->quant_params.value().max;
    attr.scale = output_value->quant_params.value().scale;

    node->operation.attributes = attr;
    return absl::OkStatus();
  }
};

class ReLUOperationParser : public TFLiteOperationParser {
 public:
  explicit ReLUOperationParser(int clip) : clip_(clip) {}

  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    RETURN_IF_ERROR(CheckMaxSupportedOpVersion(registration, 2));
    return absl::OkStatus();
  }

  absl::Status Parse(const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration,
                     GraphFloat32* graph, ObjectReader* reader) final {
    Node* node = graph->NewNode();
    node->operation.type = ToString(OperationType::RELU);
    RETURN_IF_ERROR(reader->AddInput(node, 0));

    ReLUAttributes attr;
    const TfLiteLeakyReluParams* tf_options;
    auto status = RetrieveBuiltinData(tflite_node, &tf_options);
    attr.alpha = status.ok() ? tf_options->alpha : 0;
    attr.clip = clip_;
    node->operation.attributes = attr;
    return reader->AddOutputs(node);
  }

 private:
  const int clip_;
};

class ResamplerOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    return CheckGpuDelegateCompatibility(context, tflite_node, registration);
  }

  absl::Status Parse(const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration,
                     GraphFloat32* graph, ObjectReader* reader) final {
    Node* node = graph->NewNode();
    RETURN_IF_ERROR(reader->AddInput(node, 0));  // src
    RETURN_IF_ERROR(reader->AddInput(node, 1));  // warp
    RETURN_IF_ERROR(reader->AddOutputs(node));

    node->operation.type = ToString(OperationType::RESAMPLER);

    auto src_shape = graph->FindInputs(node->id)[0]->tensor.shape;
    auto warp_shape = graph->FindInputs(node->id)[1]->tensor.shape;

    auto output_value = graph->FindOutputs(node->id)[0];
    output_value->tensor.shape =
        BHWC(src_shape.b, warp_shape.h, warp_shape.w, src_shape.c);
    return absl::OkStatus();
  }
};

class ReshapeOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    RETURN_IF_ERROR(CheckMaxSupportedOpVersion(registration, 1));
    // TODO(eignasheva): add shape checking
    return CheckGpuDelegateCompatibility(context, tflite_node, registration);
  }

  absl::Status Parse(const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration,
                     GraphFloat32* graph, ObjectReader* reader) final {
    Node* node = graph->NewNode();
    node->operation.type = ToString(OperationType::RESHAPE);
    RETURN_IF_ERROR(reader->AddInput(node, 0));
    RETURN_IF_ERROR(reader->AddOutputs(node));
    // Here we may have extra inputs. Other tensors were supposed to
    // define new shape, but in TFLite these are ignored.
    // TODO(akulik): check that shapes match?

    // New shape comes from output shape.
    ReshapeAttributes attr;
    attr.new_shape = graph->FindOutputs(node->id)[0]->tensor.shape;
    node->operation.attributes = attr;
    return absl::OkStatus();
  }
};

class Resize2DOperationParser : public TFLiteOperationParser {
 public:
  explicit Resize2DOperationParser(SamplingType sampling_type)
      : sampling_type_(sampling_type) {}

  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    RETURN_IF_ERROR(CheckMaxSupportedOpVersion(registration, 3));
    return CheckGpuDelegateCompatibility(context, tflite_node, registration);
  }

  absl::Status Parse(const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration,
                     GraphFloat32* graph, ObjectReader* reader) final {
    Node* node = graph->NewNode();
    node->operation.type = ToString(OperationType::RESIZE);
    RETURN_IF_ERROR(reader->AddInput(node, 0));
    RETURN_IF_ERROR(reader->AddOutputs(node));
    // Here we may have extra inputs. Other tensors were supposed to
    // define new shape, but in TFLite these are ignored.

    Resize2DAttributes attr;
    RETURN_IF_ERROR(GetAlignCornersValue(tflite_node, &attr.align_corners));
    RETURN_IF_ERROR(
        GetHalfPixelCentersValue(tflite_node, &attr.half_pixel_centers));
    attr.type = sampling_type_;
    attr.new_shape.CopyAllDefinedAxis(
        graph->FindOutputs(node->id)[0]->tensor.shape);
    node->operation.attributes = attr;
    return absl::OkStatus();
  }

 private:
  absl::Status GetAlignCornersValue(const TfLiteNode* tflite_node,
                                    bool* align_corners) {
    switch (sampling_type_) {
      case SamplingType::BILINEAR:
        return GetAlignCornersValueForType<TfLiteResizeBilinearParams>(
            tflite_node, align_corners);
      case SamplingType::NEAREST:
        return GetAlignCornersValueForType<TfLiteResizeNearestNeighborParams>(
            tflite_node, align_corners);
      case SamplingType::UNKNOWN:
        return absl::InternalError("Sampling type is not specified");
    }
    return absl::OkStatus();
  }

  template <class T>
  absl::Status GetAlignCornersValueForType(const TfLiteNode* tflite_node,
                                           bool* align_corners) {
    const T* tf_options;
    RETURN_IF_ERROR(RetrieveBuiltinData(tflite_node, &tf_options));
    *align_corners = tf_options->align_corners;
    return absl::OkStatus();
  }

  absl::Status GetHalfPixelCentersValue(const TfLiteNode* tflite_node,
                                        bool* half_pixel_centers) {
    if (sampling_type_ == SamplingType::BILINEAR) {
      const TfLiteResizeBilinearParams* tf_options;
      RETURN_IF_ERROR(RetrieveBuiltinData(tflite_node, &tf_options));
      if (tf_options->align_corners && tf_options->half_pixel_centers) {
        return absl::InternalError(
            "If half_pixel_centers is True, align_corners must be False.");
      }
      *half_pixel_centers = tf_options->half_pixel_centers;
    } else {
      const TfLiteResizeNearestNeighborParams* tf_options;
      RETURN_IF_ERROR(RetrieveBuiltinData(tflite_node, &tf_options));
      *half_pixel_centers = tf_options->half_pixel_centers;
    }
    return absl::OkStatus();
  }

  SamplingType sampling_type_ = SamplingType::UNKNOWN;
};

class SelectV2OperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    RETURN_IF_ERROR(CheckMaxSupportedOpVersion(registration, 1));
    return CheckGpuDelegateCompatibility(context, tflite_node, registration);
  }

  absl::Status Parse(const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration,
                     GraphFloat32* graph, ObjectReader* reader) final {
    Node* node = graph->NewNode();
    SelectV2Attributes attr;
    const TfLiteTensor* cond_tensor = reader->GetInputTensor(0);
    const TfLiteTensor* true_tensor = reader->GetInputTensor(1);
    const TfLiteTensor* false_tensor = reader->GetInputTensor(2);
    const bool is_if_constant = true_tensor->allocation_type == kTfLiteMmapRo;
    const bool is_else_constant =
        false_tensor->allocation_type == kTfLiteMmapRo;
    BHWC cond_shape, true_shape, false_shape;
    RETURN_IF_ERROR(ExtractTensorShape(*cond_tensor, &cond_shape));
    if (true_tensor->dims->size == 0) {
      attr.broadcast_true = true;
    } else {
      RETURN_IF_ERROR(ExtractTensorShape(*true_tensor, &true_shape));
      attr.broadcast_true = true_shape.DimensionsProduct() == 1;
    }
    if (false_tensor->dims->size == 0) {
      attr.broadcast_false = true;
    } else {
      RETURN_IF_ERROR(ExtractTensorShape(*false_tensor, &false_shape));
      attr.broadcast_false = false_shape.DimensionsProduct() == 1;
    }
    node->operation.type = ToString(OperationType::SELECT_V2);
    Value* if_value;
    Value* else_value;
    Tensor<BHWC, DataType::FLOAT32> if_tensor;
    Tensor<BHWC, DataType::FLOAT32> else_tensor;
    if (!attr.broadcast_true) {
      if (is_if_constant) {
        RETURN_IF_ERROR(reader->ReadTensor(1, &if_tensor));
      }
    } else {
      Tensor<Scalar, DataType::FLOAT32> if_scalar_tensor;
      RETURN_IF_ERROR(reader->ReadTensor(1, &if_scalar_tensor));
      if_tensor.shape = BHWC(1, 1, 1, 1);
      if_tensor.data.push_back(if_scalar_tensor.data[0]);
    }
    if (!attr.broadcast_false) {
      if (is_else_constant) {
        RETURN_IF_ERROR(reader->ReadTensor(2, &else_tensor));
      }
    } else {
      Tensor<Scalar, DataType::FLOAT32> else_scalar_tensor;
      RETURN_IF_ERROR(reader->ReadTensor(2, &else_scalar_tensor));
      else_tensor.shape = BHWC(1, 1, 1, 1);
      else_tensor.data.push_back(else_scalar_tensor.data[0]);
    }
    node->operation.attributes = std::move(attr);
    RETURN_IF_ERROR(reader->AddInput(node, 0));
    if (is_if_constant) {
      RETURN_IF_ERROR(NewConstNode(if_tensor, graph, &if_value));
      RETURN_IF_ERROR(graph->AddConsumer(node->id, if_value->id));
    } else {
      RETURN_IF_ERROR(reader->AddInput(node, 1));
    }
    if (is_else_constant) {
      RETURN_IF_ERROR(NewConstNode(else_tensor, graph, &else_value));
      RETURN_IF_ERROR(graph->AddConsumer(node->id, else_value->id));
    } else {
      RETURN_IF_ERROR(reader->AddInput(node, 2));
    }
    RETURN_IF_ERROR(reader->AddOutputs(node));
    return absl::OkStatus();
  }
};

class SliceOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    RETURN_IF_ERROR(CheckMaxSupportedOpVersion(registration, 2));
    return CheckGpuDelegateCompatibility(context, tflite_node, registration);
  }

  absl::Status Parse(const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration,
                     GraphFloat32* graph, ObjectReader* reader) final {
    Node* node = graph->NewNode();
    node->operation.type = ToString(OperationType::SLICE);
    RETURN_IF_ERROR(reader->AddOutputs(node));
    Value* input;
    RETURN_IF_ERROR(reader->ReadValue(0, &input));
    RETURN_IF_ERROR(graph->AddConsumer(node->id, input->id));

    const TfLiteTensor* tfl_input = reader->GetInputTensor(0);
    const int input_dims = tfl_input->dims->size;

    SliceAttributes attr;
    attr.strides = BHWC(1, 1, 1, 1);
    Tensor<Linear, DataType::INT32> starts, sizes;
    RETURN_IF_ERROR(reader->ReadTensor(1, &starts));
    RETURN_IF_ERROR(reader->ReadTensor(2, &sizes));
    if (starts.data.size() != sizes.data.size()) {
      return absl::InvalidArgumentError("Starts amount != sizes amount.");
    }
    BHWC bhwc_starts(0, 0, 0, 0);
    BHWC bhwc_sizes = input->tensor.shape;
    if (input_dims == 4) {
      // input in BHWC layout
      if (starts.data.size() == 4) {
        bhwc_starts.b = starts.data[0];
        bhwc_starts.h = starts.data[1];
        bhwc_starts.w = starts.data[2];
        bhwc_starts.c = starts.data[3];
        bhwc_sizes.b = sizes.data[0];
        bhwc_sizes.h = sizes.data[1];
        bhwc_sizes.w = sizes.data[2];
        bhwc_sizes.c = sizes.data[3];
      } else if (starts.data.size() == 3) {
        // if input is 4D(BHWC) and args 3D, we assume that args in HWC layout
        bhwc_starts.h = starts.data[0];
        bhwc_starts.w = starts.data[1];
        bhwc_starts.c = starts.data[2];
        bhwc_sizes.h = sizes.data[0];
        bhwc_sizes.w = sizes.data[1];
        bhwc_sizes.c = sizes.data[2];
      } else {
        return absl::UnimplementedError(
            "Slicing is supported for 3 or 4 dimensional tensors only.");
      }
    } else if (input_dims == 3) {
      // input in BWC layout
      if (starts.data.size() == 3) {
        bhwc_starts.b = starts.data[0];
        bhwc_starts.w = starts.data[1];
        bhwc_starts.c = starts.data[2];
        bhwc_sizes.b = sizes.data[0];
        bhwc_sizes.w = sizes.data[1];
        bhwc_sizes.c = sizes.data[2];
      } else {
        return absl::UnimplementedError(
            "Slicing is supported for 3 or 4 dimensional tensors only.");
      }
    } else {
      return absl::UnimplementedError(
          "Slicing is supported for 3 or 4 dimensional tensors only.");
    }
    const auto& in_shape = input->tensor.shape;
    if (bhwc_sizes.b == -1) {
      bhwc_sizes.b = in_shape.b - bhwc_starts.b;
    }
    if (bhwc_sizes.h == -1) {
      bhwc_sizes.h = in_shape.h - bhwc_starts.h;
    }
    if (bhwc_sizes.w == -1) {
      bhwc_sizes.w = in_shape.w - bhwc_starts.w;
    }
    if (bhwc_sizes.c == -1) {
      bhwc_sizes.c = in_shape.c - bhwc_starts.c;
    }
    attr.starts = bhwc_starts;
    attr.ends =
        BHWC(bhwc_starts.b + bhwc_sizes.b, bhwc_starts.h + bhwc_sizes.h,
             bhwc_starts.w + bhwc_sizes.w, bhwc_starts.c + bhwc_sizes.c);
    RETURN_IF_ERROR(UpdateIfNegative(in_shape, &attr));

    auto out_shape = graph->FindOutputs(node->id)[0]->tensor.shape;
    if ((attr.ends.b - attr.starts.b) != out_shape.b) {
      return absl::UnimplementedError("Output batch don't match");
    }
    if ((attr.ends.h - attr.starts.h) != out_shape.h) {
      return absl::UnimplementedError("Output height doesn't match");
    }
    if ((attr.ends.w - attr.starts.w) != out_shape.w) {
      return absl::UnimplementedError("Output width doesn't match");
    }
    if ((attr.ends.c - attr.starts.c) != out_shape.c) {
      return absl::UnimplementedError("Output channels don't match");
    }
    node->operation.attributes = attr;
    return absl::OkStatus();
  }

 private:
  absl::Status UpdateIfNegative(const BHWC& input_shape,
                                SliceAttributes* attr) {
    if (attr->ends.h < 0) {
      attr->ends.h = input_shape.h + attr->ends.h;
    }
    if (attr->ends.w < 0) {
      attr->ends.w = input_shape.w + attr->ends.w;
    }
    if (attr->ends.c < 0) {
      attr->ends.c = input_shape.c + attr->ends.c;
    }
    if (attr->ends.b < 0) {
      attr->ends.b = input_shape.b + attr->ends.b;
    }
    return absl::OkStatus();
  }
};

class SoftmaxOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    RETURN_IF_ERROR(CheckMaxSupportedOpVersion(registration, 2));
    return CheckGpuDelegateCompatibility(context, tflite_node, registration);
  }

  absl::Status Parse(const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration,
                     GraphFloat32* graph, ObjectReader* reader) final {
    Node* node = graph->NewNode();
    node->operation.type = ToString(OperationType::SOFTMAX);
    RETURN_IF_ERROR(reader->AddInput(node, 0));
    RETURN_IF_ERROR(reader->AddOutputs(node));

    const TfLiteSoftmaxParams* tf_options;
    RETURN_IF_ERROR(RetrieveBuiltinData(tflite_node, &tf_options));
    if (tf_options->beta != 1) {
      // there is multiply by scalar operation fused in softmax. Make a layer
      // out of it before softmax.
      return absl::UnimplementedError("Softmax.beta != 1 is not supported.");
      // auto mul_node = reader->NewPassthroughNode(node);
      // mul_node->operation.type = ToString(OperationType::MUL);
    }
    SoftmaxAttributes attr;
    attr.axis = Axis::CHANNELS;  // always by channels
    node->operation.attributes = attr;
    return absl::OkStatus();
  }
};

class SpaceToDepthOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    RETURN_IF_ERROR(CheckMaxSupportedOpVersion(registration, 2));
    // TODO(impjdi): Dims check.
    return CheckGpuDelegateCompatibility(context, tflite_node, registration);
  }

  absl::Status Parse(const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration,
                     GraphFloat32* graph, ObjectReader* reader) final {
    Node* node = graph->NewNode();
    node->operation.type = ToString(OperationType::SPACE_TO_DEPTH);
    RETURN_IF_ERROR(reader->AddInput(node, 0));
    RETURN_IF_ERROR(reader->AddOutputs(node));
    const TfLiteSpaceToDepthParams* tf_options;
    RETURN_IF_ERROR(RetrieveBuiltinData(tflite_node, &tf_options));
    SpaceToDepthAttributes attr;
    attr.block_size = tf_options->block_size;
    node->operation.attributes = attr;
    return absl::OkStatus();
  }
};

class SplitOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    return CheckGpuDelegateCompatibility(context, tflite_node, registration);
  }

  absl::Status Parse(const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration,
                     GraphFloat32* graph, ObjectReader* reader) final {
    const TfLiteSplitParams* split_params;
    RETURN_IF_ERROR(RetrieveBuiltinData(tflite_node, &split_params));
    if (split_params->num_splits == 1) {
      // Adding Identity reshape that will be removed.
      Node* node = graph->NewNode();
      node->operation.type = ToString(OperationType::RESHAPE);
      RETURN_IF_ERROR(reader->AddInput(node, 1));
      RETURN_IF_ERROR(reader->AddOutputs(node));
      // New shape comes from output shape.
      ReshapeAttributes attr;
      attr.new_shape = graph->FindOutputs(node->id)[0]->tensor.shape;
      node->operation.attributes = attr;
      return absl::OkStatus();
    }
    const TfLiteTensor* input = reader->GetInputTensor(1);
    const TfLiteTensor* axis_tensor = reader->GetInputTensor(0);
    SplitAttributes attr;
    RETURN_IF_ERROR(
        ExtractAxisFromIndex(*input, axis_tensor->data.i32[0], &attr.axis));

    Node* node = graph->NewNode();
    node->operation.type = ToString(OperationType::SPLIT);
    node->operation.attributes = attr;
    RETURN_IF_ERROR(reader->AddInput(node, 1));
    for (int i = 0; i < tflite_node->outputs->size; ++i) {
      RETURN_IF_ERROR(reader->AddOutput(node, i));
    }
    return absl::OkStatus();
  }
};

class SplitVOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    return CheckGpuDelegateCompatibility(context, tflite_node, registration);
  }

  absl::Status Parse(const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration,
                     GraphFloat32* graph, ObjectReader* reader) final {
    const TfLiteSplitVParams* split_params;
    RETURN_IF_ERROR(RetrieveBuiltinData(tflite_node, &split_params));
    if (split_params->num_splits == 1) {
      // Adding Identity reshape that will be removed.
      Node* node = graph->NewNode();
      node->operation.type = ToString(OperationType::RESHAPE);
      RETURN_IF_ERROR(reader->AddInput(node, 0));
      RETURN_IF_ERROR(reader->AddOutputs(node));
      // New shape comes from output shape.
      ReshapeAttributes attr;
      attr.new_shape = graph->FindOutputs(node->id)[0]->tensor.shape;
      node->operation.attributes = attr;
      return absl::OkStatus();
    }
    const TfLiteTensor* input = reader->GetInputTensor(0);
    const TfLiteTensor* axis_tensor = reader->GetInputTensor(2);
    SplitAttributes attr;
    RETURN_IF_ERROR(
        ExtractAxisFromIndex(*input, axis_tensor->data.i32[0], &attr.axis));

    Node* node = graph->NewNode();
    node->operation.type = ToString(OperationType::SPLIT);
    node->operation.attributes = attr;
    RETURN_IF_ERROR(reader->AddInput(node, 0));
    for (int i = 0; i < tflite_node->outputs->size; ++i) {
      RETURN_IF_ERROR(reader->AddOutput(node, i));
    }
    return absl::OkStatus();
  }
};

class StridedSliceOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    RETURN_IF_ERROR(CheckMaxSupportedOpVersion(registration, 2));
    return CheckGpuDelegateCompatibility(context, tflite_node, registration);
  }

  absl::Status Parse(const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration,
                     GraphFloat32* graph, ObjectReader* reader) final {
    Node* node = graph->NewNode();
    node->operation.type = ToString(OperationType::SLICE);
    RETURN_IF_ERROR(reader->AddOutputs(node));
    Value* input;
    RETURN_IF_ERROR(reader->ReadValue(0, &input));
    RETURN_IF_ERROR(graph->AddConsumer(node->id, input->id));

    Tensor<Linear, DataType::INT32> tmp;
    RETURN_IF_ERROR(reader->ReadTensor(1, &tmp));

    bool read_without_batch = tmp.data.size() == 3;
    bool read_with_batch = tmp.data.size() == 4;
    if (!read_without_batch && !read_with_batch) {
      // Error: Must be catched in IsSupported()
      return absl::UnimplementedError(
          "Slicing is supported for 3 or 4 dimensional tensors only.");
    }

    const TfLiteStridedSliceParams* tf_options;
    RETURN_IF_ERROR(RetrieveBuiltinData(tflite_node, &tf_options));
    RETURN_IF_ERROR(CheckOptionsSupport(tf_options));

    auto out_shape = graph->FindOutputs(node->id)[0]->tensor.shape;

    SliceAttributes attr;
    if (read_without_batch) {
      RETURN_IF_ERROR(ReadAttribsWithoutBatch(reader, tf_options,
                                              input->tensor.shape, &attr));
    }
    if (read_with_batch) {
      RETURN_IF_ERROR(
          ReadAttribsWithBatch(reader, tf_options, input->tensor.shape, &attr));
    }
    if (attr.strides.b == 0 || attr.strides.h == 0 || attr.strides.w == 0 ||
        attr.strides.c == 0) {
      return absl::InvalidArgumentError("stride values must be non-zero");
    }
    if (attr.strides.b < 0 || attr.strides.h < 0 || attr.strides.w < 0 ||
        attr.strides.c < 0) {
      return absl::UnimplementedError("Reverse slices are not supported.");
    }
    if ((attr.ends.b - attr.starts.b + attr.strides.b - 1) / attr.strides.b !=
        out_shape.b) {
      return absl::UnimplementedError("Output batch don't match");
    }
    if ((attr.ends.h - attr.starts.h + attr.strides.h - 1) / attr.strides.h !=
        out_shape.h) {
      return absl::UnimplementedError("Output height doesn't match");
    }
    if ((attr.ends.w - attr.starts.w + attr.strides.w - 1) / attr.strides.w !=
        out_shape.w) {
      return absl::UnimplementedError("Output width doesn't match");
    }
    if ((attr.ends.c - attr.starts.c + attr.strides.c - 1) / attr.strides.c !=
        out_shape.c) {
      return absl::UnimplementedError("Output channels don't match");
    }
    node->operation.attributes = attr;
    return absl::OkStatus();
  }

 private:
  absl::Status UpdateWithMask(const TfLiteStridedSliceParams* tf_options,
                              const BHWC& input_shape, int ignore_b,
                              int ignore_h, int ignore_w, int ignore_c,
                              SliceAttributes* attr) {
    if (tf_options->begin_mask & ignore_h) {
      attr->starts.h = 0;
    }
    if (tf_options->begin_mask & ignore_w) {
      attr->starts.w = 0;
    }
    if (tf_options->begin_mask & ignore_c) {
      attr->starts.c = 0;
    }
    if (tf_options->begin_mask & ignore_b) {
      attr->starts.b = 0;
    }

    if (tf_options->end_mask & ignore_h) {
      attr->ends.h = input_shape.h;
    }
    if (tf_options->end_mask & ignore_w) {
      attr->ends.w = input_shape.w;
    }
    if (tf_options->end_mask & ignore_c) {
      attr->ends.c = input_shape.c;
    }
    if (tf_options->end_mask & ignore_b) {
      attr->ends.b = input_shape.b;
    }
    return absl::OkStatus();
  }

  absl::Status UpdateIfNegative(const BHWC& input_shape,
                                SliceAttributes* attr) {
    if (attr->ends.h < 0) {
      attr->ends.h = input_shape.h + attr->ends.h;
    }
    if (attr->ends.w < 0) {
      attr->ends.w = input_shape.w + attr->ends.w;
    }
    if (attr->ends.c < 0) {
      attr->ends.c = input_shape.c + attr->ends.c;
    }
    if (attr->ends.b < 0) {
      attr->ends.b = input_shape.b + attr->ends.b;
    }

    if (attr->starts.h < 0) {
      attr->starts.h = input_shape.h + attr->starts.h;
    }
    if (attr->starts.w < 0) {
      attr->starts.w = input_shape.w + attr->starts.w;
    }
    if (attr->starts.c < 0) {
      attr->starts.c = input_shape.c + attr->starts.c;
    }
    if (attr->starts.b < 0) {
      attr->starts.b = input_shape.b + attr->starts.b;
    }

    return absl::OkStatus();
  }

  absl::Status ReadAttribsWithBatch(const ObjectReader* reader,
                                    const TfLiteStridedSliceParams* tf_options,
                                    const BHWC& input_shape,
                                    SliceAttributes* attr) {
    auto read_bhwc = [&](int tensor_index, BHWC* bhwc) -> absl::Status {
      Tensor<Linear, DataType::INT32> t;
      RETURN_IF_ERROR(reader->ReadTensor(tensor_index, &t));
      *bhwc = BHWC(t.data[0], t.data[1], t.data[2], t.data[3]);
      return absl::OkStatus();
    };

    RETURN_IF_ERROR(read_bhwc(1, &attr->starts));
    RETURN_IF_ERROR(read_bhwc(2, &attr->ends));
    RETURN_IF_ERROR(read_bhwc(3, &attr->strides));
    RETURN_IF_ERROR(UpdateIfNegative(input_shape, attr));
    RETURN_IF_ERROR(UpdateWithMask(tf_options, input_shape, 1, 2, 4, 8, attr));
    return absl::OkStatus();
  }

  absl::Status ReadAttribsWithoutBatch(
      const ObjectReader* reader, const TfLiteStridedSliceParams* tf_options,
      const BHWC& input_shape, SliceAttributes* attr) {
    auto read_hwc = [&](int tensor_index, BHWC* bhwc) -> absl::Status {
      Tensor<Linear, DataType::INT32> t;
      RETURN_IF_ERROR(reader->ReadTensor(tensor_index, &t));
      *bhwc = BHWC(0, t.data[0], t.data[1], t.data[2]);
      return absl::OkStatus();
    };

    RETURN_IF_ERROR(read_hwc(1, &attr->starts));
    RETURN_IF_ERROR(read_hwc(2, &attr->ends));
    RETURN_IF_ERROR(read_hwc(3, &attr->strides));
    RETURN_IF_ERROR(UpdateIfNegative(input_shape, attr));
    RETURN_IF_ERROR(UpdateWithMask(tf_options, input_shape, 0, 1, 2, 4, attr));
    attr->starts.b = 0;
    attr->ends.b = input_shape.b;
    attr->strides.b = 1;
    return absl::OkStatus();
  }
  absl::Status CheckOptionsSupport(const TfLiteStridedSliceParams* tf_options) {
    if (tf_options->ellipsis_mask) {
      return absl::UnimplementedError("Slice does not support ellipsis_mask.");
    }
    if (tf_options->new_axis_mask) {
      return absl::UnimplementedError("Slice does not support new_axis_mask.");
    }
    if (tf_options->shrink_axis_mask) {
      return absl::UnimplementedError(
          "Slice does not support shrink_axis_mask parameter. ");
    }
    return absl::OkStatus();
  }
};

class TileOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    return CheckGpuDelegateCompatibility(context, tflite_node, registration);
  }

  absl::Status Parse(const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration,
                     GraphFloat32* graph, ObjectReader* reader) final {
    Node* node = graph->NewNode();
    node->operation.type = ToString(OperationType::TILE);
    RETURN_IF_ERROR(reader->AddInput(node, 0));
    RETURN_IF_ERROR(reader->AddOutputs(node));
    return absl::OkStatus();
  }
};

// Builtin op version of TRANSPOSE_CONV.
class TransposeConvBuiltinOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    RETURN_IF_ERROR(CheckMaxSupportedOpVersion(registration, 3));
    return CheckGpuDelegateCompatibility(context, tflite_node, registration);
  }

  // TFLite's TRANSPOSE_CONV expects 3-4 input tensors (output shape, weights,
  // input, and an optional bias) and allows configurable padding & stride.
  // TODO(impjdi): Translate output_shape to attr.adjacent.
  absl::Status Parse(const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration,
                     GraphFloat32* graph, ObjectReader* reader) final {
    auto* node = graph->NewNode();
    node->operation.type = ToString(OperationType::CONVOLUTION_TRANSPOSED);
    Value* input;
    RETURN_IF_ERROR(reader->ReadValue(2, &input));
    RETURN_IF_ERROR(graph->AddConsumer(node->id, input->id));
    RETURN_IF_ERROR(reader->AddOutputs(node));

    const TfLiteTransposeConvParams* tf_options;
    RETURN_IF_ERROR(RetrieveBuiltinData(tflite_node, &tf_options));

    ConvolutionTransposedAttributes attr;
    attr.stride = tf_options
                      ? HW(tf_options->stride_height, tf_options->stride_width)
                      : HW(1, 1);
    const int runtime_inputs = reader->GetNumberOfRuntimeInputs();
    if (runtime_inputs == 2) {
      RETURN_IF_ERROR(reader->AddInput(node, 1));
      auto weights_shape = graph->FindInputs(node->id)[1]->tensor.shape;
      attr.weights.shape = OHWI(weights_shape.b, weights_shape.h,
                                weights_shape.w, weights_shape.c);
    } else {  // runtime_inputs == 1;
      RETURN_IF_ERROR(reader->ReadTensor(1, &attr.weights));
    }
    reader->ReadTensor(3, &attr.bias).IgnoreError();  // bias is optional

    UpdatePadding(tf_options->padding,
                  graph->FindInputs(node->id)[0]->tensor.shape, &attr);
    node->operation.attributes = std::move(attr);
    return absl::OkStatus();
  }
};

// Custom op version of TRANSPOSE_CONV.
class TransposeConvCustomOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    return CheckGpuDelegateCompatibility(context, tflite_node, registration);
  }

  absl::Status Parse(const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration,
                     GraphFloat32* graph, ObjectReader* reader) final {
    auto* node = graph->NewNode();
    node->operation.type = ToString(OperationType::CONVOLUTION_TRANSPOSED);
    RETURN_IF_ERROR(reader->AddInput(node, 0));
    RETURN_IF_ERROR(reader->AddOutputs(node));

    const TfLiteTransposeConvParams* tf_options;
    auto status = RetrieveCustomInitialData(tflite_node, &tf_options);

    ConvolutionTransposedAttributes attr;
    attr.stride = status.ok()
                      ? HW(tf_options->stride_height, tf_options->stride_width)
                      : HW(1, 1);
    RETURN_IF_ERROR(reader->ReadTensor(1, &attr.weights));
    reader->ReadTensor(2, &attr.bias).IgnoreError();  // bias is optional

    UpdatePadding(status.ok() ? tf_options->padding : kTfLitePaddingUnknown,
                  graph->FindInputs(node->id)[0]->tensor.shape, &attr);
    node->operation.attributes = std::move(attr);
    return absl::OkStatus();
  }
};

class TransposeOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    RETURN_IF_ERROR(CheckMaxSupportedOpVersion(registration, 2));
    return CheckGpuDelegateCompatibility(context, tflite_node, registration);
  }

  absl::Status Parse(const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration,
                     GraphFloat32* graph, ObjectReader* reader) final {
    Node* node = graph->NewNode();
    node->operation.type = ToString(OperationType::TRANSPOSE);
    RETURN_IF_ERROR(reader->AddInput(node, 0));
    RETURN_IF_ERROR(reader->AddOutputs(node));

    TransposeAttributes attr;
    Tensor<Linear, DataType::INT32> perm;
    RETURN_IF_ERROR(reader->ReadTensor(1, &perm));
    std::map<Axis, int> axis_to_index = {{Axis::BATCH, 0},
                                         {Axis::HEIGHT, 1},
                                         {Axis::WIDTH, 2},
                                         {Axis::CHANNELS, 3}};
    if (perm.data.size() == 4) {
      attr.perm = BHWC(perm.data[0], perm.data[1], perm.data[2], perm.data[3]);
    } else if (perm.data.size() == 3) {
      std::vector<Axis> index_to_axis = {Axis::BATCH, Axis::WIDTH,
                                         Axis::CHANNELS};
      attr.perm.b = axis_to_index[index_to_axis[perm.data[0]]];
      attr.perm.h = 1;
      attr.perm.w = axis_to_index[index_to_axis[perm.data[1]]];
      attr.perm.c = axis_to_index[index_to_axis[perm.data[2]]];
    } else if (perm.data.size() == 2) {
      std::vector<Axis> index_to_axis = {Axis::BATCH, Axis::CHANNELS};
      attr.perm.b = axis_to_index[index_to_axis[perm.data[0]]];
      attr.perm.h = 1;
      attr.perm.w = 2;
      attr.perm.c = axis_to_index[index_to_axis[perm.data[1]]];
    } else {
      return absl::InvalidArgumentError(
          "Permutation for transpose is invalid.");
    }

    node->operation.attributes = attr;
    return absl::OkStatus();
  }
};

class Unpooling2DOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    return CheckGpuDelegateCompatibility(context, tflite_node, registration);
  }

  absl::Status Parse(const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration,
                     GraphFloat32* graph, ObjectReader* reader) final {
    Node* node = graph->NewNode();
    node->operation.type = ToString(OperationType::MAX_UNPOOLING_2D);
    RETURN_IF_ERROR(reader->AddInput(node, 0));
    RETURN_IF_ERROR(reader->AddInput(node, 1));
    RETURN_IF_ERROR(reader->AddOutputs(node));
    auto input_shape = graph->FindInputs(node->id)[0]->tensor.shape;
    MaxUnpooling2DAttributes attr;

    const TfLitePoolParams* tf_options;
    RETURN_IF_ERROR(RetrieveCustomInitialData(tflite_node, &tf_options));

    attr.kernel = ToHW(tf_options->filter_height, tf_options->filter_width);
    attr.strides = ToHW(tf_options->stride_height, tf_options->stride_width);
    UpdatePadding(tf_options->padding, input_shape, &attr);

    node->operation.attributes = attr;

    auto output_value = graph->FindOutputs(node->id)[0];
    output_value->tensor.shape = CalculateOutputShape(input_shape, attr);
    return absl::OkStatus();
  }
};

// TODO(impjdi): BATCH_TO_SPACE/SPACE_TO_BATCH shouldn't be supported.
class BatchToSpaceOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    return absl::OkStatus();
  }

  absl::Status Parse(const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration,
                     GraphFloat32* graph, ObjectReader* reader) final {
    auto* node = graph->NewNode();
    node->operation.type = ToString(OperationType::BATCH_TO_SPACE);
    RETURN_IF_ERROR(reader->AddInput(node, 0));
    RETURN_IF_ERROR(reader->AddOutputs(node));

    BatchToSpaceAttributes bs_attr;
    Tensor<Linear, DataType::INT32> block;
    RETURN_IF_ERROR(reader->ReadTensor(1, &block));
    if (block.shape.v != 2) {
      return absl::InternalError("Space has to be HxW.");
    }
    bs_attr.block.h = block.data[0];
    bs_attr.block.w = block.data[1];

    Tensor<HW, DataType::INT32> crop;
    RETURN_IF_ERROR(reader->ReadTensor(2, &crop));
    auto crop_shape = crop.shape;
    if (crop_shape.h != 2 && crop_shape.w != 2) {
      return absl::InternalError("Space has to be HxW.");
    }

    bs_attr.crop.prepended.h = crop.data[0];
    bs_attr.crop.prepended.w = crop.data[2];

    bs_attr.crop.appended.h = crop.data[1];
    bs_attr.crop.appended.w = crop.data[3];

    node->operation.attributes = std::move(bs_attr);
    return absl::OkStatus();
  }
};

class SpaceToBatchOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    return CheckGpuDelegateCompatibility(context, tflite_node, registration);
  }

  absl::Status Parse(const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration,
                     GraphFloat32* graph, ObjectReader* reader) final {
    auto* node = graph->NewNode();
    node->operation.type = ToString(OperationType::SPACE_TO_BATCH);
    RETURN_IF_ERROR(reader->AddInput(node, 0));
    RETURN_IF_ERROR(reader->AddOutputs(node));
    SpaceToBatchAttributes sb_attr;
    Tensor<Linear, DataType::INT32> block;
    RETURN_IF_ERROR(reader->ReadTensor(1, &block));
    if (block.shape.v != 2) {
      return absl::InternalError("Space has to be HxW.");
    }
    sb_attr.block.h = block.data[0];
    sb_attr.block.w = block.data[1];

    Tensor<HW, DataType::INT32> padding;
    RETURN_IF_ERROR(reader->ReadTensor(2, &padding));
    auto padding_shape = padding.shape;

    if (padding_shape.h != 2 && padding_shape.w != 2) {
      return absl::InternalError("Space has to be HxW.");
    }

    sb_attr.padding.prepended.h = padding.data[0];
    sb_attr.padding.prepended.w = padding.data[2];

    sb_attr.padding.appended.h = padding.data[1];
    sb_attr.padding.appended.w = padding.data[3];

    node->operation.attributes = std::move(sb_attr);
    return absl::OkStatus();
  }
};

class MeanOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    return CheckGpuDelegateCompatibility(context, tflite_node, registration);
  }

  absl::Status Parse(const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration,
                     GraphFloat32* graph, ObjectReader* reader) final {
    auto* node = graph->NewNode();
    node->operation.type = ToString(OperationType::MEAN);
    RETURN_IF_ERROR(reader->AddInput(node, 0));
    RETURN_IF_ERROR(reader->AddOutputs(node));

    MeanAttributes attr;
    const TfLiteTensor* input = reader->GetInputTensor(0);
    const TfLiteTensor* axes = reader->GetInputTensor(1);
    for (int i = 0; i < NumElements(axes->dims); i++) {
      Axis axis;
      RETURN_IF_ERROR(ExtractAxisFromIndex(*input, axes->data.i32[i], &axis));
      attr.dims.insert(axis);
    }
    node->operation.attributes = attr;
    return absl::OkStatus();
  }
};

class UnsupportedOperationParser : public TFLiteOperationParser {
 public:
  absl::Status IsSupported(const TfLiteContext* context,
                           const TfLiteNode* tflite_node,
                           const TfLiteRegistration* registration) final {
    return absl::UnimplementedError("Operation is not supported.");
  }

  absl::Status Parse(const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration,
                     GraphFloat32* graph, ObjectReader* reader) final {
    return absl::UnimplementedError("Operation is not supported.");
  }
};

absl::Status IsSupported(
    const TfLiteContext* context, TfLiteNode* node,
    const TfLiteRegistration* registration, bool allow_quant_ops = false,
    const absl::flat_hash_set<TfLiteBuiltinOperator>* excluded_ops = nullptr) {
  return NewOperationParser(registration, allow_quant_ops, excluded_ops)
      ->IsSupported(context, node, registration);
}

bool IsAllAllowedTensors(TfLiteContext* context,
                         const TfLiteIntArray* tensor_indices,
                         const std::vector<TfLiteType>& allowed_types) {
  for (int i = 0; i < tensor_indices->size; ++i) {
    int tensor_idx = tensor_indices->data[i];
    if (tensor_idx == kTfLiteOptionalTensor) continue;
    const TfLiteTensor* t = &context->tensors[tensor_idx];
    if (t->dims && t->dims->size >= 5) {
      return false;
    }
    bool type_supported = false;
    for (auto allowed_type : allowed_types) {
      if (t->type == allowed_type) {
        type_supported = true;
        break;
      }
    }
    if (t->allocation_type == kTfLiteArenaRw && !type_supported) {
      return false;
    }
  }
  return true;
}
}  // namespace

std::unique_ptr<TFLiteOperationParser> NewOperationParser(
    const TfLiteRegistration* registration, bool allow_quant_ops,
    const absl::flat_hash_set<TfLiteBuiltinOperator>* excluded_ops) {
  const auto builtin_code = registration->builtin_code;
  if (excluded_ops != nullptr &&
      excluded_ops->contains(
          static_cast<TfLiteBuiltinOperator>(builtin_code))) {
    return std::make_unique<UnsupportedOperationParser>();
  }
  switch (builtin_code) {
    case kTfLiteBuiltinAbs:
      return std::make_unique<ElementwiseOperationParser>(OperationType::ABS);
    case kTfLiteBuiltinAdd:
      return std::make_unique<ElementwiseOperationParser>(OperationType::ADD);
    case kTfLiteBuiltinAveragePool2d:
      return std::make_unique<Pooling2DOperationParser>(PoolingType::AVERAGE);
    case kTfLiteBuiltinBatchMatmul:
      return std::make_unique<BatchedMatMulOperationParser>();
    case kTfLiteBuiltinCast:
      return std::make_unique<CastOperationParser>();
    case kTfLiteBuiltinConcatenation:
      return std::make_unique<ConcatenationOperationParser>();
    case kTfLiteBuiltinConv2d:
      return std::make_unique<Conv2DOperationParser>();
    case kTfLiteBuiltinCos:
      return std::make_unique<ElementwiseOperationParser>(OperationType::COS);
    case kTfLiteBuiltinCumsum:
      return std::make_unique<CumsumOperationParser>();
    case kTfLiteBuiltinDensify:
      return std::make_unique<DensifyOperationParser>();
    case kTfLiteBuiltinDepthwiseConv2d:
      return std::make_unique<DepthwiseConvolutionOperationParser>();
    case kTfLiteBuiltinDepthToSpace:
      return std::make_unique<DepthToSpaceOperationParser>();
    case kTfLiteBuiltinDequantize:
      if (allow_quant_ops) {
        return std::make_unique<DequantizeOperationParser>();
      }
      break;
    case kTfLiteBuiltinDiv:
      return std::make_unique<ElementwiseOperationParser>(OperationType::DIV);
    case kTfLiteBuiltinEqual:
      return std::make_unique<ElementwiseOperationParser>(OperationType::EQUAL);
    case kTfLiteBuiltinElu:
      return std::make_unique<ElementwiseOperationParser>(OperationType::ELU);
    case kTfLiteBuiltinExp:
      return std::make_unique<ElementwiseOperationParser>(OperationType::EXP);
    case kTfLiteBuiltinFloor:
      return std::make_unique<ElementwiseOperationParser>(OperationType::FLOOR);
    case kTfLiteBuiltinFloorDiv:
      return std::make_unique<ElementwiseOperationParser>(
          OperationType::FLOOR_DIV);
    case kTfLiteBuiltinFloorMod:
      return std::make_unique<ElementwiseOperationParser>(
          OperationType::FLOOR_MOD);
    case kTfLiteBuiltinFullyConnected:
      return std::make_unique<FullyConnectedOperationParser>();
    case kTfLiteBuiltinGreater:
      return std::make_unique<ElementwiseOperationParser>(
          OperationType::GREATER);
    case kTfLiteBuiltinGreaterEqual:
      return std::make_unique<ElementwiseOperationParser>(
          OperationType::GREATER_EQUAL);
    case kTfLiteBuiltinHardSwish:
      return std::make_unique<HardSwishOperationParser>();
    case kTfLiteBuiltinLess:
      return std::make_unique<ElementwiseOperationParser>(OperationType::LESS);
    case kTfLiteBuiltinLessEqual:
      return std::make_unique<ElementwiseOperationParser>(
          OperationType::LESS_EQUAL);
    case kTfLiteBuiltinLogistic:
      return std::make_unique<ElementwiseOperationParser>(
          OperationType::SIGMOID);
    case kTfLiteBuiltinLog:
      return std::make_unique<ElementwiseOperationParser>(OperationType::LOG);
    case kTfLiteBuiltinLstm:
      return std::make_unique<LSTMOperationParser>();
    case kTfLiteBuiltinMaximum:
      return std::make_unique<ElementwiseOperationParser>(
          OperationType::MAXIMUM);
    case kTfLiteBuiltinMaxPool2d:
      return std::make_unique<Pooling2DOperationParser>(PoolingType::MAX);
    case kTfLiteBuiltinMean:
      return std::make_unique<MeanOperationParser>();
    case kTfLiteBuiltinMinimum:
      return std::make_unique<ElementwiseOperationParser>(
          OperationType::MINIMUM);
    case kTfLiteBuiltinMirrorPad:
      return std::make_unique<PadOperationParser>(/*mirror_pad=*/true);
    case kTfLiteBuiltinMul:
      return std::make_unique<ElementwiseOperationParser>(OperationType::MUL);
    case kTfLiteBuiltinNeg:
      return std::make_unique<ElementwiseOperationParser>(OperationType::NEG);
    case kTfLiteBuiltinNotEqual:
      return std::make_unique<ElementwiseOperationParser>(
          OperationType::NOT_EQUAL);
    case kTfLiteBuiltinOneHot:
      return std::make_unique<OneHotOperationParser>();
    case kTfLiteBuiltinPack:
      return std::make_unique<PackOperationParser>();
    case kTfLiteBuiltinPad:
      return std::make_unique<PadOperationParser>(/*mirror_pad=*/false);
    case kTfLiteBuiltinPow:
      return std::make_unique<ElementwiseOperationParser>(OperationType::POW);
    case kTfLiteBuiltinReduceMax:
      return std::make_unique<ReduceOperationParser>(
          OperationType::REDUCE_MAXIMUM);
    case kTfLiteBuiltinReduceMin:
      return std::make_unique<ReduceOperationParser>(
          OperationType::REDUCE_MINIMUM);
    case kTfLiteBuiltinReduceProd:
      return std::make_unique<ReduceOperationParser>(
          OperationType::REDUCE_PRODUCT);
    case kTfLiteBuiltinQuantize:
      if (allow_quant_ops) {
        return std::make_unique<QuantizeOperationParser>();
      }
      break;
    case kTfLiteBuiltinRelu:
      return std::make_unique<ReLUOperationParser>(0);
    case kTfLiteBuiltinRelu6:
      return std::make_unique<ReLUOperationParser>(6);
    case kTfLiteBuiltinReluN1To1:
      return std::make_unique<ClampOperationsParser>(-1.0, 1.0);
    case kTfLiteBuiltinLeakyRelu:
      return std::make_unique<ReLUOperationParser>(0);
    case kTfLiteBuiltinPrelu:
      return std::make_unique<PReLUOperationParser>();
    case kTfLiteBuiltinReshape:
      return std::make_unique<ReshapeOperationParser>();
    case kTfLiteBuiltinResizeBilinear:
      return std::make_unique<Resize2DOperationParser>(SamplingType::BILINEAR);
    case kTfLiteBuiltinResizeNearestNeighbor:
      return std::make_unique<Resize2DOperationParser>(SamplingType::NEAREST);
    case kTfLiteBuiltinRsqrt:
      return std::make_unique<ElementwiseOperationParser>(OperationType::RSQRT);
    case kTfLiteBuiltinSelectV2:
      return std::make_unique<SelectV2OperationParser>();
    case kTfLiteBuiltinSin:
      return std::make_unique<ElementwiseOperationParser>(OperationType::SIN);
    case kTfLiteBuiltinSlice:
      return std::make_unique<SliceOperationParser>();
    case kTfLiteBuiltinSoftmax:
      return std::make_unique<SoftmaxOperationParser>();
    case kTfLiteBuiltinSpaceToDepth:
      return std::make_unique<SpaceToDepthOperationParser>();
    case kTfLiteBuiltinSplit:
      return std::make_unique<SplitOperationParser>();
    case kTfLiteBuiltinSplitV:
      return std::make_unique<SplitVOperationParser>();
    case kTfLiteBuiltinSqrt:
      return std::make_unique<ElementwiseOperationParser>(OperationType::SQRT);
    case kTfLiteBuiltinSquare:
      return std::make_unique<ElementwiseOperationParser>(
          OperationType::SQUARE);
    case kTfLiteBuiltinSquaredDifference:
      return std::make_unique<ElementwiseOperationParser>(
          OperationType::SQUARED_DIFF);
    case kTfLiteBuiltinStridedSlice:
      return std::make_unique<StridedSliceOperationParser>();
    case kTfLiteBuiltinSub:
      return std::make_unique<ElementwiseOperationParser>(OperationType::SUB);
    case kTfLiteBuiltinSum:
      return std::make_unique<ReduceOperationParser>(OperationType::REDUCE_SUM);
    case kTfLiteBuiltinTanh:
      return std::make_unique<ElementwiseOperationParser>(OperationType::TANH);
    case kTfLiteBuiltinTile:
      return std::make_unique<TileOperationParser>();
    case kTfLiteBuiltinTranspose:
      return std::make_unique<TransposeOperationParser>();
    case kTfLiteBuiltinTransposeConv:
      return std::make_unique<TransposeConvBuiltinOperationParser>();
    case kTfLiteBuiltinCustom: {
      const absl::string_view custom_name = registration->custom_name;
      if (custom_name == "Convolution2DTransposeBias") {
        return std::make_unique<TransposeConvCustomOperationParser>();
      }
      if (custom_name == "MaxPoolingWithArgmax2D") {
        return std::make_unique<Pooling2DOperationParser>(PoolingType::MAX);
      }
      if (custom_name == "MaxUnpooling2D") {
        return std::make_unique<Unpooling2DOperationParser>();
      }
      if (custom_name == "Resampler") {
        return std::make_unique<ResamplerOperationParser>();
      }
      return NewCustomOperationParser(registration->custom_name);
    }
  }
  return std::make_unique<UnsupportedOperationParser>();
}

// TODO(impjdi): Check number of input/output tensors and their dimensions.
// TODO(impjdi): Check ops' parameters.
TfLiteIntArray* GetOpsToReplace(
    TfLiteContext* context, bool allow_quant_ops, int max_delegated_partitions,
    const absl::flat_hash_set<TfLiteBuiltinOperator>* excluded_ops) {
  delegates::IsNodeSupportedFn node_supported_fn =
      [=](TfLiteContext* context, TfLiteNode* node,
          TfLiteRegistration* registration,
          std::string* unsupported_details) -> bool {
    const auto status =
        IsSupported(context, node, registration, allow_quant_ops, excluded_ops);
    if (!status.ok()) {
      if (unsupported_details) {
        *unsupported_details = std::string(status.message());
      }
      return false;
    }

    std::vector<TfLiteType> allowed_in_types = {kTfLiteFloat32, kTfLiteFloat16};
    std::vector<TfLiteType> allowed_out_types = {kTfLiteFloat32,
                                                 kTfLiteFloat16};
    if (allow_quant_ops) {
      // Since we only check non-constant tensors, type cannot be Int32.
      allowed_in_types.push_back(kTfLiteInt8);
      allowed_in_types.push_back(kTfLiteUInt8);
      allowed_out_types.push_back(kTfLiteInt8);
      allowed_out_types.push_back(kTfLiteUInt8);
    }
    if (IsLogicalCode(registration->builtin_code)) {
      allowed_out_types.push_back(kTfLiteBool);
    }
    if (registration->builtin_code == kTfLiteBuiltinCast) {
      allowed_in_types.push_back(kTfLiteBool);
      allowed_in_types.push_back(kTfLiteFloat32);
      allowed_in_types.push_back(kTfLiteInt32);
      allowed_out_types.push_back(kTfLiteFloat32);
      allowed_out_types.push_back(kTfLiteInt32);
    }
    if (registration->builtin_code == kTfLiteBuiltinOneHot) {
      allowed_in_types.push_back(kTfLiteInt32);
    }
    if (!IsAllAllowedTensors(context, node->inputs, allowed_in_types) ||
        !IsAllAllowedTensors(context, node->outputs, allowed_out_types)) {
      if (unsupported_details) {
        *unsupported_details =
            "OP is supported, but tensor type/shape isn't compatible.";
      }
      return false;
    }
    return true;
  };

  delegates::FP16GraphPartitionHelper partition_helper(context,
                                                       node_supported_fn);
  std::set<std::string> unsupported_nodes_info;
  if (partition_helper.Partition(&unsupported_nodes_info) != kTfLiteOk) {
    return TfLiteIntArrayCreate(0);
  }

  // By default, we simply get 1st largest partition as 'max_delegate_partions'
  // is set to 1 by default.
  std::vector<int> ops_to_replace =
      partition_helper.GetNodesOfFirstNLargestPartitions(
          max_delegated_partitions);

  if (!unsupported_nodes_info.empty() &&
      partition_helper.num_total_nodes() > ops_to_replace.size()) {
    std::string unsupported = absl::StrJoin(unsupported_nodes_info, "\n");
    std::string error_message = absl::StrCat(
        "Following operations are not supported by GPU delegate:\n",
        unsupported, "\n");
    if (!ops_to_replace.empty()) {
      absl::StrAppend(
          &error_message, ops_to_replace.size(),
          " operations will run on the GPU, and the remaining ",
          partition_helper.num_total_nodes() - ops_to_replace.size());
    } else {
      absl::StrAppend(&error_message,
                      "No operations will run on the GPU, and all ",
                      partition_helper.num_total_nodes());
    }
    absl::StrAppend(&error_message, " operations will run on the CPU.");
    TF_LITE_KERNEL_LOG(context, error_message.c_str());
  }
  return ConvertVectorToTfLiteIntArray(ops_to_replace);
}

// Creates inputs and outputs passed by io_tensors parameters in the resulting
// graph. We force it to make sure that delegated subgraph has same order of
// inputs and outputs with the original one. When delegated model is built from
// the tflite model representation tensors are created lazily, so there is no
// guarantee that the order will match the source model tensors order.
absl::Status PrecreateIOTensors(
    TfLiteContext* context, GraphFloat32* graph, const std::vector<int>& io_ids,
    absl::flat_hash_map<int, int>* quant_conversion_map,
    absl::flat_hash_map<int, Value*>* tensor_to_value) {
  for (const auto& id : io_ids) {
    const TfLiteTensor& tflite_tensor = context->tensors[id];
    if (tflite::IsConstantTensor(&tflite_tensor)) continue;
    RETURN_IF_ERROR(ObjectReader::ReadNonConstantTensor(
        context, tensor_to_value, quant_conversion_map, graph, id));
  }
  return absl::OkStatus();
}

absl::Status CopyVariableTensorOutputs(
    TfLiteNode* tflite_node, TfLiteRegistration* registration,
    GraphFloat32* graph, ObjectReader& reader,
    const absl::flat_hash_map<int, ValueId>& new_variable_tensor_values) {
  absl::flat_hash_map<int, ValueId> new_variable_tensor_values_copy(
      new_variable_tensor_values);
  // Retrieve the final value id for the variable input tensors.
  for (int i = 0; i < tflite_node->inputs->size; i++) {
    int tensor_idx = tflite_node->inputs->data[i];
    Value* value;
    if (!reader.ReadValueByTensorIdx(tensor_idx, &value).ok()) continue;
    if (value->tensor.is_variable_input) {
      if (new_variable_tensor_values_copy.find(i) ==
          new_variable_tensor_values_copy.end()) {
        return absl::InvalidArgumentError(
            absl::StrCat(GetOpNameByRegistration(*registration),
                         " did not provide a new value for the variable input "
                         "tensor with index ",
                         tensor_idx));
      } else {
        Node* node = graph->NewNode();
        node->operation.type = ToString(OperationType::COPY);
        RETURN_IF_ERROR(graph->AddConsumer(
            node->id, new_variable_tensor_values_copy.at(i)));
        RETURN_IF_ERROR(reader.AddUpdate(node, i));
        new_variable_tensor_values_copy.erase(
            new_variable_tensor_values_copy.find(i));
      }
    }
  }
  if (!new_variable_tensor_values_copy.empty()) {
    return absl::InvalidArgumentError(
        "More input variable tensors asked to be copied than present on the "
        "node");
  }
  return absl::OkStatus();
}

absl::Status BuildModel(TfLiteContext* context,
                        const TfLiteDelegateParams* delegate_params,
                        GraphFloat32* graph,
                        absl::flat_hash_map<int, int>* quant_conversion_map) {
  std::vector<int> inputs(delegate_params->input_tensors->size);
  std::vector<int> outputs(delegate_params->output_tensors->size);
  for (int i = 0; i < delegate_params->input_tensors->size; i++) {
    inputs[i] = delegate_params->input_tensors->data[i];
  }
  for (int i = 0; i < delegate_params->output_tensors->size; i++) {
    outputs[i] = delegate_params->output_tensors->data[i];
  }
  return BuildModelEnforceIO(context, delegate_params, inputs, outputs, graph,
                             quant_conversion_map);
}

absl::Status BuildModelEnforceIO(
    TfLiteContext* context, const TfLiteDelegateParams* delegate_params,
    const std::vector<int>& input_ids, const std::vector<int>& output_ids,
    GraphFloat32* graph, absl::flat_hash_map<int, int>* quant_conversion_map) {
  std::vector<std::unique_ptr<TFLiteOperationParser>> operations;
  std::vector<int> tflite_nodes;
  for (int i = 0; i < delegate_params->nodes_to_replace->size; ++i) {
    TfLiteNode* tflite_node = nullptr;
    TfLiteRegistration* registration = nullptr;
    RETURN_IF_ERROR(GetNodeAndRegistration(
        context, delegate_params->nodes_to_replace->data[i], &tflite_node,
        &registration));
    if (registration->builtin_code == kTfLiteBuiltinDequantize &&
        context->tensors[tflite_node->inputs->data[0]].type ==
            TfLiteType::kTfLiteFloat16 &&
        context->tensors[tflite_node->inputs->data[0]].allocation_type ==
            TfLiteAllocationType::kTfLiteMmapRo) {
      // Ignore Fp16 Dequantize nodes only if they are the final nodes before
      // weights, i.e. no other nodes preceded them (e.g. DENSIFY).
      continue;
    }
    auto op_parser = NewOperationParser(
        registration, /*allow_quant_ops=*/quant_conversion_map != nullptr);
    if (!op_parser) {
      return absl::UnimplementedError(
          absl::StrCat("Operation ", registration->builtin_code, "(",
                       registration->custom_name,
                       ") is not supported by TFLite GPU Delegate."));
    }
    operations.push_back(std::move(op_parser));
    tflite_nodes.push_back(i);
  }
  absl::flat_hash_map<int, Value*> tensor_to_value;
  std::vector<ValueId> variable_inputs_to_value_id;

  RETURN_IF_ERROR(PrecreateIOTensors(context, graph, input_ids,
                                     quant_conversion_map, &tensor_to_value));
  RETURN_IF_ERROR(PrecreateIOTensors(context, graph, output_ids,
                                     quant_conversion_map, &tensor_to_value));
  for (int i = 0; i < operations.size(); ++i) {
    TfLiteNode* tflite_node;
    TfLiteRegistration* registration;
    RETURN_IF_ERROR(GetNodeAndRegistration(
        context, delegate_params->nodes_to_replace->data[tflite_nodes[i]],
        &tflite_node, &registration));
    ObjectReader reader(graph, context, tflite_node, &tensor_to_value,
                        quant_conversion_map);
    const auto status =
        operations[i]->Parse(tflite_node, registration, graph, &reader);
    if (!status.ok()) {
      return absl::InternalError(absl::StrCat(
          GetOpNameByRegistration(*registration), ": ", status.message()));
    }

    absl::flat_hash_map<int, ValueId> new_value_for_variable_input_tensors =
        operations[i]->GetNewValueIdsForVariableInputNodes();

    RETURN_IF_ERROR(
        CopyVariableTensorOutputs(tflite_node, registration, graph, reader,
                                  new_value_for_variable_input_tensors));
  }

  // Variable input tensors expect to be unchanged throughout model execution.
  // They need to be an output of the graph in order to have them unchanged.
  for (auto value_id : variable_inputs_to_value_id) {
    if (!graph->IsGraphOutput(value_id)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Variable input tensors must be a graph output. Value ",
                       value_id, " is not a graph output"));
    }
  }
  return absl::OkStatus();
}

absl::Status BuildFinalModel(
    TfLiteContext* context, const TfLiteDelegateParams* delegate_params,
    GraphFloat32* graph, absl::flat_hash_map<int, int>* quant_conversion_map) {
  RETURN_IF_ERROR(
      BuildModel(context, delegate_params, graph, quant_conversion_map));

  // Apply general transformations on the graph.
  ModelTransformer transformer(graph);
  if (!ApplyModelTransformations(&transformer)) {
    return absl::InternalError("Graph transformations failed");
  }
  return absl::OkStatus();
}

namespace {

class DelegateContext {
 public:
  struct DelegateData {
    std::vector<int> input_ids;
    std::vector<int> output_ids;
    GraphFloat32* graph;
    std::unique_ptr<absl::flat_hash_map<int, int>> quant_conversion_map;
  };
  bool Init(TfLiteContext* context,
            const TfLiteDelegateParams* delegate_params) {
    const auto* delegate_data =
        reinterpret_cast<DelegateData*>(delegate_params->delegate->data_);
    return delegate_data->graph &&
           BuildModelEnforceIO(context, delegate_params,
                               delegate_data->input_ids,
                               delegate_data->output_ids, delegate_data->graph,
                               delegate_data->quant_conversion_map.get())
               .ok();
  }
};

TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate) {
  TfLiteRegistration registration{};
  registration.init = [](TfLiteContext* context, const char* buffer,
                         size_t) -> void* {
    auto* delegate_context = new DelegateContext();
    if (!delegate_context->Init(
            context, reinterpret_cast<const TfLiteDelegateParams*>(buffer))) {
      delete delegate_context;
      return nullptr;
    }
    return delegate_context;
  };
  registration.free = [](TfLiteContext* context, void* buffer) -> void {
    delete reinterpret_cast<DelegateContext*>(buffer);
  };
  registration.prepare = [](TfLiteContext* context,
                            TfLiteNode* node) -> TfLiteStatus {
    return node->user_data ? kTfLiteOk : kTfLiteError;
  };

  const auto* delegate_data =
      reinterpret_cast<const DelegateContext::DelegateData*>(delegate->data_);
  TfLiteIntArray* ops_to_replace = GetOpsToReplace(
      context, static_cast<bool>(delegate_data->quant_conversion_map));
  const auto status = context->ReplaceNodeSubsetsWithDelegateKernels(
      context, registration, ops_to_replace, delegate);
  TfLiteIntArrayFree(ops_to_replace);
  return status;
}

}  // namespace

absl::Status BuildFromFlatBuffer(const tflite::FlatBufferModel& flatbuffer,
                                 const tflite::OpResolver& op_resolver,
                                 GraphFloat32* graph, bool allow_quant_ops) {
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder interpreter_builder(flatbuffer, op_resolver);
  if (interpreter_builder(&interpreter) != kTfLiteOk || !interpreter) {
    return absl::InternalError("Unable to prepare TfLite interpreter.");
  }
  TfLiteDelegate delegate;

  DelegateContext::DelegateData delegate_data{interpreter->inputs(),
                                              interpreter->outputs(), graph};
  if (allow_quant_ops) {
    delegate_data.quant_conversion_map =
        std::make_unique<absl::flat_hash_map<int, int>>();
  }

  delegate.data_ = &delegate_data;
  delegate.flags = kTfLiteDelegateFlagsNone;
  delegate.Prepare = DelegatePrepare;
  delegate.CopyFromBufferHandle = nullptr;
  delegate.CopyToBufferHandle = nullptr;
  delegate.FreeBufferHandle = nullptr;

  if (interpreter->ModifyGraphWithDelegate(&delegate) != kTfLiteOk) {
    return absl::InternalError("Conversion from TfLite model failed.");
  }

  ModelTransformer transformer(graph);
  if (!ApplyModelTransformations(&transformer)) {
    return absl::InternalError("Graph transformations failed");
  }

  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
