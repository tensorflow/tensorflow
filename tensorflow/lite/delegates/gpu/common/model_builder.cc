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

#include "tensorflow/lite/delegates/gpu/common/model_builder.h"

#include <stddef.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/context.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace gpu {
namespace {

using ::absl::make_unique;
using ::absl::StrCat;

// Creates a node that consumes output from the given node. Because output need
// to stay the same, newly created node will inherit the output from the given
// node, which will in turn get newly created copy of output. This is necessary
// to preserve reference consistency if another node was pointing at that
// output:
//   node(output)
// will turn into:
//   node(copy(output)) <- passthrough_node(output)
Status NewPassthroughNode(GraphFloat32* graph, Node* node,
                          const Value<TensorRefFloat32>* output,
                          Node** passthru_node) {
  *passthru_node = graph->NewNode();
  // Make copies for every output in the original node.
  RETURN_IF_ERROR(graph->SetProducer((*passthru_node)->id, output->id));
  Value<TensorRefFloat32>* copy_output = graph->NewValue();
  RETURN_IF_ERROR(graph->SetProducer(node->id, copy_output->id));
  RETURN_IF_ERROR(graph->AddConsumer((*passthru_node)->id, copy_output->id));
  copy_output->tensor = output->tensor;
  copy_output->tensor.ref = -1;
  return OkStatus();
}

template <typename T>
Status CreateVectorCopyData(const TfLiteTensor& tensor,
                            std::vector<T>* tensor_data) {
  if (tensor.bytes % sizeof(T) != 0) {
    return InvalidArgumentError(
        StrCat("Input data size ", tensor.bytes,
               " is not aligned to expected type: ", sizeof(T)));
  }
  tensor_data->resize(tensor.bytes / sizeof(T));
  std::memcpy(&(*tensor_data)[0], tensor.data.uint8, tensor.bytes);
  return OkStatus();
}

template <typename ShapeT>
Status SetAllDimensions(const TfLiteIntArray* dimensions, ShapeT* shape);

template <>
Status SetAllDimensions<Scalar>(const TfLiteIntArray* dimensions,
                                Scalar* shape) {
  if (dimensions->size < 0) {
    return InvalidArgumentError("Invalid Scalar dimensions");
  }
  for (int i = 0; i < dimensions->size; ++i) {
    if (dimensions->data[i] != 1) {
      return InvalidArgumentError("Dimension can not be reduced to scalar.");
    }
  }
  shape->v = 1;
  return OkStatus();
}

template <>
Status SetAllDimensions<Linear>(const TfLiteIntArray* dimensions,
                                Linear* shape) {
  if (dimensions->size <= 0) {
    return InvalidArgumentError("Dimension is empty.");
  }
  for (int i = 0; i < dimensions->size - 1; ++i) {
    if (dimensions->data[i] != 1) {
      return InvalidArgumentError("Dimension can not be reduced to linear.");
    }
  }
  shape->v = dimensions->data[dimensions->size - 1];
  return OkStatus();
}

template <>
Status SetAllDimensions<HWC>(const TfLiteIntArray* dimensions, HWC* shape) {
  if (dimensions->size != 4) {
    return InvalidArgumentError("Dimensions are not HWC");
  }
  if (dimensions->data[0] != 1) {
    return UnimplementedError("Batch size is not equal to 1.");
  }
  shape->h = dimensions->data[1];
  shape->w = dimensions->data[2];
  shape->c = dimensions->data[3];
  return OkStatus();
}

template <>
Status SetAllDimensions<HW>(const TfLiteIntArray* dimensions, HW* shape) {
  if (dimensions->size != 2) {
    return InvalidArgumentError("Dimensions are not HW");
  }
  shape->h = dimensions->data[0];
  shape->w = dimensions->data[1];
  return OkStatus();
}

template <>
Status SetAllDimensions<OHWI>(const TfLiteIntArray* dimensions, OHWI* shape) {
  if (dimensions->size != 4) {
    return InvalidArgumentError(
        StrCat("Dimensions are not OHWI: ", dimensions->size));
  }
  shape->o = dimensions->data[0];
  shape->h = dimensions->data[1];
  shape->w = dimensions->data[2];
  shape->i = dimensions->data[3];
  return OkStatus();
}

template <>
Status SetAllDimensions<IHWO>(const TfLiteIntArray* dimensions, IHWO* shape) {
  if (dimensions->size != 4) {
    return InvalidArgumentError(
        StrCat("Dimensions are not IHWO: ", dimensions->size));
  }
  shape->i = dimensions->data[0];
  shape->h = dimensions->data[1];
  shape->w = dimensions->data[2];
  shape->o = dimensions->data[3];
  return OkStatus();
}

template <>
Status SetAllDimensions<BHWC>(const TfLiteIntArray* dimensions, BHWC* shape) {
  if (dimensions->size != 4) {
    return InvalidArgumentError("Dimensions are not BHWC");
  }
  shape->b = dimensions->data[0];
  shape->h = dimensions->data[1];
  shape->w = dimensions->data[2];
  shape->c = dimensions->data[3];
  return OkStatus();
}

DataType ToDataType(TfLiteType type) {
  switch (type) {
    case kTfLiteFloat32:
      return DataType::FLOAT32;
    case kTfLiteInt32:
      return DataType::INT32;
    case kTfLiteInt64:
      return DataType::INT64;
    case kTfLiteUInt8:
      return DataType::UINT8;
    default:
      return DataType::UNKNOWN;
  }
}

int GetNumberOfRuntimeInputsForNode(const TfLiteContext* context,
                                    const TfLiteNode* tflite_node) {
  int number_of_runtime_inputs = 0;
  for (int i = 0; i < tflite_node->inputs->size; i++) {
    if (!IsConstantTensor(&context->tensors[tflite_node->inputs->data[i]])) {
      number_of_runtime_inputs++;
    }
  }
  return number_of_runtime_inputs;
}

int GetNumberOfRuntimeOutputsForNode(const TfLiteContext* context,
                                     const TfLiteNode* tflite_node) {
  int number_of_runtime_outputs = 0;
  for (int i = 0; i < tflite_node->outputs->size; i++) {
    if (!IsConstantTensor(&context->tensors[tflite_node->outputs->data[i]])) {
      number_of_runtime_outputs++;
    }
  }
  return number_of_runtime_outputs;
}

Status CheckTensorIsAvailable(const TfLiteContext* context,
                              const TfLiteNode* tflite_node, int idx) {
  // If tensor id is in range, it's guaranteed that it'll be available.
  if (idx >= tflite_node->inputs->size) {
    return OutOfRangeError(
        absl::StrFormat("Requested index goes beyond array size (%d vs %d).",
                        idx, tflite_node->inputs->data[idx]));
  }
  return OkStatus();
}

class ObjectReader {
 public:
  ObjectReader(GraphFloat32* graph, TfLiteContext* context,
               const TfLiteNode* tflite_node,
               std::vector<Value<TensorRefFloat32>*>* tensor_to_value)
      : graph_(graph),
        context_(context),
        tflite_node_(tflite_node),
        tensor_to_value_(tensor_to_value) {}

  Status ReadValue(uint32_t idx, Value<TensorRefFloat32>** value) {
    if (idx >= tflite_node_->inputs->size) {
      return OutOfRangeError(StrCat("ReadValue: input tensor index: ", idx));
    }
    RETURN_IF_ERROR(
        ReadValueByTensorIdx(tflite_node_->inputs->data[idx], value));
    return OkStatus();
  }

  int GetNumberOfRuntimeInputs() {
    return GetNumberOfRuntimeInputsForNode(context_, tflite_node_);
  }

  Status GetTensorDims(uint32_t idx, TfLiteIntArray* dimensions) {
    if (idx >= tflite_node_->inputs->size) {
      return OutOfRangeError(StrCat("Input tensor index: ", idx));
    }
    int32_t tensor_idx = tflite_node_->inputs->data[idx];
    if (tensor_idx < 0 || tensor_idx > context_->tensors_size) {
      return OutOfRangeError(StrCat("Tensor index: ", tensor_idx));
    }
    const TfLiteTensor& tflite_tensor = context_->tensors[tensor_idx];
    *dimensions = *tflite_tensor.dims;
    return OkStatus();
  }

  template <typename TensorT>
  Status ReadTensor(uint32_t idx, TensorT* t) const {
    RETURN_IF_ERROR(CheckTensorIsAvailable(context_, tflite_node_, idx));
    int32_t tensor_idx = tflite_node_->inputs->data[idx];
    const TfLiteTensor& tflite_tensor = context_->tensors[tensor_idx];
    RETURN_IF_ERROR(CreateVectorCopyData(tflite_tensor, &t->data));

    // Axis and data layout depend on operation this tensor is used in. So,
    // postpone resolutions until operations are parsed.
    t->id = tensor_idx;
    return SetAllDimensions(tflite_tensor.dims, &t->shape);
  }

  Status AddOutput(const Node* node, int id) {
    if (tflite_node_->outputs->size <= id) {
      return InvalidArgumentError(
          StrCat("Data id ", id, " must be less than tflite node outputs size ",
                 tflite_node_->outputs->size));
    }
    int output_tensor_idx = tflite_node_->outputs->data[id];
    Value<TensorRefFloat32>* value;
    RETURN_IF_ERROR(ReadValueByTensorIdx(output_tensor_idx, &value));
    RETURN_IF_ERROR(graph_->SetProducer(node->id, value->id));
    return OkStatus();
  }

  Status AddOutputs(const Node* node) {
    for (int i = 0; i < tflite_node_->outputs->size; ++i) {
      RETURN_IF_ERROR(AddOutput(node, i));
    }
    return OkStatus();
  }

  Status AddInput(const Node* node, uint32_t idx) {
    Value<TensorRefFloat32>* input;
    RETURN_IF_ERROR(ReadValue(idx, &input));
    return graph_->AddConsumer(node->id, input->id);
  }

  Status ReadValueByTensorIdx(uint32_t tensor_idx,
                              Value<TensorRefFloat32>** value) {
    if (tensor_idx >= tensor_to_value_->size()) {
      return OutOfRangeError(
          StrCat("ReadValue: input tensor index: ", tensor_idx));
    }
    if ((*tensor_to_value_)[tensor_idx] == nullptr) {
      const TfLiteTensor& tflite_tensor = context_->tensors[tensor_idx];
      if (tflite::IsConstantTensor(&tflite_tensor)) {
        return NotFoundError(
            StrCat("ReadValue: value is a constant tensor: ", tensor_idx));
      }
      Value<TensorRefFloat32>* value = graph_->NewValue();
      RETURN_IF_ERROR(
          ConvertTfLiteTensorToTensorRef(tflite_tensor, &value->tensor));
      value->tensor.ref = tensor_idx;
      (*tensor_to_value_)[tensor_idx] = value;
    }
    *value = (*tensor_to_value_)[tensor_idx];
    return OkStatus();
  }

 private:
  GraphFloat32* graph_ = nullptr;
  const TfLiteContext* context_ = nullptr;
  const TfLiteNode* tflite_node_ = nullptr;
  std::vector<Value<TensorRefFloat32>*>* tensor_to_value_;
};

Status CheckInputsOutputs(const TfLiteContext* context,
                          const TfLiteNode* tflite_node, int inputs,
                          int outputs) {
  int runtime_inputs = GetNumberOfRuntimeInputsForNode(context, tflite_node);
  if (runtime_inputs != inputs) {
    return InternalError(
        absl::StrFormat("Expected %d input tensor(s), but node has %d runtime "
                        "input(s).",
                        inputs, runtime_inputs));
  }
  int runtime_outputs = GetNumberOfRuntimeOutputsForNode(context, tflite_node);
  if (runtime_outputs != outputs) {
    return InternalError(
        absl::StrFormat("Expected %d output tensor(s), but node has %d runtime "
                        "output(s).",
                        outputs, runtime_outputs));
  }
  return OkStatus();
}

// A parser responsible for parsing TFLite operation and adding it to a graph.
class TFLiteOperationParser {
 public:
  virtual ~TFLiteOperationParser() {}

  // Parses TFLite operation. This method allows expanding fused operations
  // into more than one node.
  virtual Status Parse(const TfLiteNode* tflite_node,
                       const TfLiteRegistration* registration,
                       GraphFloat32* graph, ObjectReader* reader) = 0;

  // Verifies whether passed tflite node may be built by GPU delegate or not.
  virtual Status IsSupported(const TfLiteContext* context,
                             const TfLiteNode* tflite_node,
                             const TfLiteRegistration* registration) = 0;
};

Status CheckActivationSupported(TfLiteFusedActivation fused_activation) {
  if (fused_activation == kTfLiteActNone) {
    return OkStatus();
  }
  switch (fused_activation) {
    case kTfLiteActRelu:
    case kTfLiteActRelu1:
    case kTfLiteActRelu6:
    case kTfLiteActTanh:
      return OkStatus();
    default:
      return NotFoundError(absl::StrFormat("Unsupported fused activation: %d.",
                                           fused_activation));
  }
}

// If there is fused activation present, then there will be another node created
// that will have identical output as the given node. New operation node will
// depend on the given node output.
Status MaybeFuseActivation(TfLiteFusedActivation fused_activation,
                           const std::vector<uint32_t>& output_indices,
                           GraphFloat32* graph, Node* node) {
  if (fused_activation == kTfLiteActNone) {
    return OkStatus();
  }
  const auto& outputs = graph->FindOutputs(node->id);
  if (outputs.empty()) {
    return InternalError("Empty outputs in fused node");
  }
  switch (fused_activation) {
    case kTfLiteActRelu:
    case kTfLiteActRelu1:
    case kTfLiteActRelu6: {
      ReLUAttributes attr;
      attr.clip = fused_activation == kTfLiteActRelu
                      ? 0.0f
                      : (fused_activation == kTfLiteActRelu1 ? 1.0f : 6.0f);
      for (auto index : output_indices) {
        Node* activation_node;
        RETURN_IF_ERROR(
            NewPassthroughNode(graph, node, outputs[index], &activation_node));
        activation_node->operation.type = ToString(OperationType::RELU);
        activation_node->operation.attributes = attr;
      }
      break;
    }
    case kTfLiteActTanh:
      for (auto index : output_indices) {
        Node* activation_node;
        RETURN_IF_ERROR(
            NewPassthroughNode(graph, node, outputs[index], &activation_node));
        activation_node->operation.type = ToString(OperationType::TANH);
      }
      break;
    default:
      return NotFoundError(
          StrCat("Unsupported fused activation: ", fused_activation));
  }
  return OkStatus();
}

Status MaybeFuseActivationToTheSingleOutput(
    TfLiteFusedActivation fused_activation, GraphFloat32* graph, Node* node) {
  if (graph->FindOutputs(node->id).size() != 1) {
    return InternalError("Number of outputs exceeds 1");
  }
  return MaybeFuseActivation(fused_activation, {0}, graph, node);
}

HW ToHW(int32_t h, int32_t w) { return HW(h > 0 ? h : 1, w > 0 ? w : 1); }

template <typename AttrT>
void UpdatePadding(const TfLitePadding& padding, const BHWC& input_shape,
                   AttrT* attr) {
  if (padding == kTfLitePaddingSame) {
    attr->padding = CalculateSamePadding(input_shape, *attr);
  } else {
    attr->padding.prepended = HW(0, 0);
    attr->padding.appended = HW(0, 0);
  }
}

Status GetFullyConnectedAttributes(int weights_tensor_id, int bias_tensor_id,
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

  return OkStatus();
}

template <typename ParamsType>
Status RetrieveBuiltinData(const TfLiteNode* tflite_node,
                           ParamsType** tf_options) {
  const auto* params =
      reinterpret_cast<const ParamsType*>(tflite_node->builtin_data);
  if (!params) {
    return InternalError("Unable to retrieve builtin_data.");
  }
  *tf_options = const_cast<ParamsType*>(params);
  return OkStatus();
}

template <typename ParamsType>
Status RetrieveCustomInitialData(const TfLiteNode* tflite_node,
                                 ParamsType** tf_options) {
  const auto* params =
      reinterpret_cast<const ParamsType*>(tflite_node->custom_initial_data);
  if (!params) {
    return InternalError("Unable to retrieve custom_initial_data.");
  }
  *tf_options = const_cast<ParamsType*>(params);
  return OkStatus();
}

Status CheckMaxSupportedOpVersion(const TfLiteRegistration* registration,
                                  int max_version) {
  const int op_version = registration->version;
  if (op_version > max_version) {
    return UnimplementedError(
        absl::StrFormat("Max version supported: %d. Requested version %d.",
                        max_version, op_version));
  }
  return OkStatus();
}

Status CheckExactSupportedOpVersion(const TfLiteRegistration* registration,
                                    int expected_version) {
  int op_version = registration->version;
  if (op_version != expected_version) {
    return UnimplementedError(
        absl::StrFormat("Only version %d is supported. Requested version %d.",
                        expected_version, op_version));
  }
  return OkStatus();
}

Status CheckKernels(int kernel_h, int kernel_w) {
  if (kernel_h <= 0 || kernel_w <= 0) {
    return InvalidArgumentError(absl::StrFormat(
        "Incorrect kernel values: kernel_height = %d, kernel_width = %d.",
        kernel_h, kernel_w));
  }
  return OkStatus();
}

Status CheckStrides(int strides_h, int strides_w) {
  if (strides_h <= 0 || strides_w <= 0) {
    return InvalidArgumentError(absl::StrFormat(
        "Incorrect stride values: stride_height = %d, stride_width = %d.",
        strides_h, strides_w));
  }
  return OkStatus();
}

Status CheckDilation(int dilation_h, int dilation_w) {
  if (dilation_h <= 0 || dilation_w <= 0) {
    return InvalidArgumentError(
        absl::StrFormat("Incorrect dilation values: dilation_factor = %d, "
                        "dilation_factor = %d.",
                        dilation_h, dilation_w));
  }
  return OkStatus();
}

Status CheckStridesAndDilation(int strides_h, int strides_w, int dilation_h,
                               int dilation_w) {
  RETURN_IF_ERROR(CheckStrides(strides_h, strides_w));
  RETURN_IF_ERROR(CheckDilation(dilation_h, dilation_w));
  return OkStatus();
}

Status CheckKernelsAndStrides(int kernel_h, int kernel_w, int strides_h,
                              int strides_w) {
  RETURN_IF_ERROR(CheckKernels(kernel_h, kernel_w));
  RETURN_IF_ERROR(CheckStrides(strides_h, strides_w));
  return OkStatus();
}

class Conv2DOperationParser : public TFLiteOperationParser {
 public:
  Status IsSupported(const TfLiteContext* context,
                     const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration) final {
    RETURN_IF_ERROR(CheckMaxSupportedOpVersion(registration, 1));
    RETURN_IF_ERROR(
        CheckInputsOutputs(context, tflite_node, /*inputs=*/1, /*outputs=*/1));
    RETURN_IF_ERROR(CheckTensorIsAvailable(context, tflite_node, 1));
    TfLiteConvParams* tf_options = nullptr;
    RETURN_IF_ERROR(RetrieveBuiltinData(tflite_node, &tf_options));
    RETURN_IF_ERROR(CheckStridesAndDilation(
        tf_options->stride_height, tf_options->stride_width,
        tf_options->dilation_height_factor, tf_options->dilation_width_factor));
    RETURN_IF_ERROR(CheckActivationSupported(tf_options->activation));
    return OkStatus();
  }

  Status Parse(const TfLiteNode* tflite_node,
               const TfLiteRegistration* registration, GraphFloat32* graph,
               ObjectReader* reader) final {
    Node* node = graph->NewNode();
    node->operation.type = ToString(OperationType::CONVOLUTION_2D);
    RETURN_IF_ERROR(reader->AddInput(node, 0));
    RETURN_IF_ERROR(reader->AddOutputs(node));

    Convolution2DAttributes attr;
    RETURN_IF_ERROR(reader->ReadTensor(1, &attr.weights));
    reader->ReadTensor(2, &attr.bias).IgnoreError();  // bias is optional

    const auto* tf_options =
        reinterpret_cast<const TfLiteConvParams*>(tflite_node->builtin_data);
    if (!tf_options) {
      return InternalError("Missing tflite params");
    }
    attr.strides = ToHW(tf_options->stride_height, tf_options->stride_width);
    attr.dilations = HW(tf_options->dilation_height_factor,
                        tf_options->dilation_width_factor);
    UpdatePadding(tf_options->padding,
                  graph->FindInputs(node->id)[0]->tensor.shape, &attr);
    RETURN_IF_ERROR(MaybeFuseActivationToTheSingleOutput(tf_options->activation,
                                                         graph, node));
    node->operation.attributes = std::move(attr);
    return OkStatus();
  }
};

// Creates a simple node that holds tensor value.
Status NewConstNode(TensorFloat32 t, GraphFloat32* graph,
                    Value<TensorRefFloat32>** value) {
  ConstTensorAttributes attr;
  attr.tensor = std::move(t);
  Node* node = graph->NewNode();
  node->operation.attributes = attr;
  node->operation.type = ToString(OperationType::CONST);
  *value = graph->NewValue();
  RETURN_IF_ERROR(graph->SetProducer(node->id, (*value)->id));
  // Keep data inside this tensor.
  (*value)->tensor.ref = attr.tensor.id;
  (*value)->tensor.type = attr.tensor.kType;
  (*value)->tensor.shape = attr.tensor.shape;
  return OkStatus();
}

class ConcatenationOperationParser : public TFLiteOperationParser {
 public:
  Status IsSupported(const TfLiteContext* context,
                     const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration) final {
    RETURN_IF_ERROR(CheckMaxSupportedOpVersion(registration, 1));

    // TODO(eignasheva): add proper tensor availability checking
    // for (uint32_t idx = 0; idx < tflite_node->inputs->size; ++idx) {
    //   RETURN_IF_ERROR(CheckTensorIsAvailable(context, tflite_node, idx));
    // }
    // TODO(eignasheva): add axis checking.
    TfLiteConcatenationParams* tf_options = nullptr;
    RETURN_IF_ERROR(RetrieveBuiltinData(tflite_node, &tf_options));
    return OkStatus();
  }

  Status Parse(const TfLiteNode* tflite_node,
               const TfLiteRegistration* registration, GraphFloat32* graph,
               ObjectReader* reader) final {
    ConcatAttributes attr;
    // Read inputs first to make sure const node is added to a graph before
    // concat node to ensure topological order.
    std::vector<const Value<TensorRefFloat32>*> inputs;
    for (uint32_t idx = 0; idx < tflite_node->inputs->size; ++idx) {
      Value<TensorRefFloat32>* value;
      const auto status = reader->ReadValue(idx, &value);
      if (status.ok()) {
        inputs.push_back(value);
      } else {
        TensorFloat32 tensor;
        RETURN_IF_ERROR(reader->ReadTensor(idx, &tensor));
        Value<TensorRefFloat32>* value;
        RETURN_IF_ERROR(NewConstNode(std::move(tensor), graph, &value));
        inputs.push_back(value);
      }
    }

    Node* node = graph->NewNode();
    node->operation.type = ToString(OperationType::CONCAT);
    RETURN_IF_ERROR(reader->AddOutputs(node));
    for (const Value<TensorRefFloat32>* input : inputs) {
      RETURN_IF_ERROR(graph->AddConsumer(node->id, input->id));
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
    const auto* tf_options = reinterpret_cast<const TfLiteConcatenationParams*>(
        tflite_node->builtin_data);
    if (!tf_options) {
      return InternalError("Missing tflite params");
    }
    RETURN_IF_ERROR(MaybeFuseActivationToTheSingleOutput(tf_options->activation,
                                                         graph, node));
    node->operation.attributes = attr;
    return OkStatus();
  }

 private:
  Status SetAxis(const std::vector<BHWC>& input_shapes, Axis* axis) {
    *axis = Axis::BATCH;
    for (int i = 1; i < input_shapes.size(); i++) {
      if (input_shapes[0].h != input_shapes[i].h &&
          input_shapes[0].w != input_shapes[i].w &&
          input_shapes[0].c != input_shapes[i].c) {
        *axis = Axis::HEIGHT;
        break;
      }
    }
    if (*axis == Axis::BATCH) return OkStatus();
    for (int i = 1; i < input_shapes.size(); i++) {
      if (input_shapes[0].b != input_shapes[i].b &&
          input_shapes[0].w != input_shapes[i].w &&
          input_shapes[0].c != input_shapes[i].c) {
        *axis = Axis::WIDTH;
        break;
      }
    }
    if (*axis == Axis::HEIGHT) return OkStatus();
    for (int i = 1; i < input_shapes.size(); i++) {
      if (input_shapes[0].b != input_shapes[i].b &&
          input_shapes[0].h != input_shapes[i].h &&
          input_shapes[0].c != input_shapes[i].c) {
        *axis = Axis::CHANNELS;
        break;
      }
    }
    if (*axis == Axis::WIDTH) return OkStatus();
    for (int i = 1; i < input_shapes.size(); i++) {
      if (input_shapes[0].b != input_shapes[i].b &&
          input_shapes[0].w != input_shapes[i].w &&
          input_shapes[0].h != input_shapes[i].h) {
        return UnimplementedError(
            "Can concatenate tensors only by batch, height, width, or "
            "channels.");
      }
    }
    return OkStatus();
  }
};

class DepthwiseConvolutionOperationParser : public TFLiteOperationParser {
 public:
  Status IsSupported(const TfLiteContext* context,
                     const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration) final {
    RETURN_IF_ERROR(CheckMaxSupportedOpVersion(registration, 2));
    RETURN_IF_ERROR(
        CheckInputsOutputs(context, tflite_node, /*inputs=*/1, /*outputs=*/1));
    RETURN_IF_ERROR(CheckTensorIsAvailable(context, tflite_node, 1));
    TfLiteDepthwiseConvParams* tf_options = nullptr;
    RETURN_IF_ERROR(RetrieveBuiltinData(tflite_node, &tf_options));
    RETURN_IF_ERROR(CheckStridesAndDilation(
        tf_options->stride_height, tf_options->stride_width,
        tf_options->dilation_height_factor, tf_options->dilation_width_factor));
    RETURN_IF_ERROR(CheckActivationSupported(tf_options->activation));
    return OkStatus();
  }

  Status Parse(const TfLiteNode* tflite_node,
               const TfLiteRegistration* registration, GraphFloat32* graph,
               ObjectReader* reader) final {
    Node* node = graph->NewNode();
    node->operation.type = ToString(OperationType::DEPTHWISE_CONVOLUTION);
    RETURN_IF_ERROR(reader->AddInput(node, 0));
    RETURN_IF_ERROR(reader->AddOutputs(node));

    DepthwiseConvolution2DAttributes attr;
    RETURN_IF_ERROR(reader->ReadTensor(1, &attr.weights));
    reader->ReadTensor(2, &attr.bias).IgnoreError();  // bias is optional
    const auto* tf_options = reinterpret_cast<const TfLiteDepthwiseConvParams*>(
        tflite_node->builtin_data);
    if (!tf_options) {
      return InternalError("Missing tflite params");
    }
    attr.strides = ToHW(tf_options->stride_height, tf_options->stride_width);
    attr.dilations = HW(std::max(1, tf_options->dilation_height_factor),
                        std::max(1, tf_options->dilation_width_factor));
    UpdatePadding(tf_options->padding,
                  graph->FindInputs(node->id)[0]->tensor.shape, &attr);
    RETURN_IF_ERROR(MaybeFuseActivationToTheSingleOutput(tf_options->activation,
                                                         graph, node));
    node->operation.attributes = std::move(attr);
    return OkStatus();
  }
};

class ReshapeOperationParser : public TFLiteOperationParser {
 public:
  Status IsSupported(const TfLiteContext* context,
                     const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration) final {
    RETURN_IF_ERROR(CheckMaxSupportedOpVersion(registration, 1));
    RETURN_IF_ERROR(
        CheckInputsOutputs(context, tflite_node, /*inputs=*/1, /*outputs=*/1));
    // TODO(eignasheva): add shape checking
    return OkStatus();
  }

  Status Parse(const TfLiteNode* tflite_node,
               const TfLiteRegistration* registration, GraphFloat32* graph,
               ObjectReader* reader) final {
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
    return OkStatus();
  }
};

Status ParsePoolingAttributes(const TfLitePoolParams* tf_options,
                              const BHWC& input_shape,
                              Pooling2DAttributes* attr) {
  attr->kernel = ToHW(tf_options->filter_height, tf_options->filter_width);
  attr->strides = ToHW(tf_options->stride_height, tf_options->stride_width);
  UpdatePadding(tf_options->padding, input_shape, attr);
  return OkStatus();
}

class Pooling2DOperationParser : public TFLiteOperationParser {
  Status IsSupported(const TfLiteContext* context,
                     const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration) final {
    RETURN_IF_ERROR(CheckMaxSupportedOpVersion(registration, 1));
    TfLitePoolParams* tf_options = nullptr;
    auto status = RetrieveCustomInitialData(tflite_node, &tf_options);
    if (status.ok()) {  // custom case with indices as a second output
      RETURN_IF_ERROR(CheckInputsOutputs(context, tflite_node, /*inputs=*/1,
                                         /*outputs=*/2));
    } else {  // common pooling with 1 output
      RETURN_IF_ERROR(RetrieveBuiltinData(tflite_node, &tf_options));
      RETURN_IF_ERROR(CheckInputsOutputs(context, tflite_node, /*inputs=*/1,
                                         /*outputs=*/1));
    }
    RETURN_IF_ERROR(CheckKernelsAndStrides(
        tf_options->filter_height, tf_options->filter_width,
        tf_options->stride_height, tf_options->stride_width));
    RETURN_IF_ERROR(CheckActivationSupported(tf_options->activation));
    return OkStatus();
  }

 public:
  explicit Pooling2DOperationParser(PoolingType type) : type_(type) {}

  Status Parse(const TfLiteNode* tflite_node,
               const TfLiteRegistration* registration, GraphFloat32* graph,
               ObjectReader* reader) final {
    Node* node = graph->NewNode();
    node->operation.type = ToString(OperationType::POOLING_2D);
    RETURN_IF_ERROR(reader->AddInput(node, 0));
    RETURN_IF_ERROR(reader->AddOutput(node, 0));

    Pooling2DAttributes attr;
    attr.type = type_;

    auto input_shape = graph->FindInputs(node->id)[0]->tensor.shape;

    // check whether there are custom options encoded. It happens if operation
    // is MaxPoolingWithArgmax2D. There is no way to read
    // tflite_node->builtin_code, so, simply check whether custom data is
    // available.
    auto* tf_options = reinterpret_cast<const TfLitePoolParams*>(
        tflite_node->custom_initial_data);
    if (!tf_options) {
      tf_options =
          reinterpret_cast<const TfLitePoolParams*>(tflite_node->builtin_data);
    }
    if (!tf_options) {
      return InternalError("Missing tflite params");
    }

    std::vector<uint32_t> max_tensor_id{0};
    RETURN_IF_ERROR(MaybeFuseActivation(tf_options->activation, max_tensor_id,
                                        graph, node));
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
    return OkStatus();
  }

 private:
  const PoolingType type_;
};

class Unpooling2DOperationParser : public TFLiteOperationParser {
 public:
  Status IsSupported(const TfLiteContext* context,
                     const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration) final {
    TfLitePoolParams* tf_options = nullptr;
    RETURN_IF_ERROR(
        CheckInputsOutputs(context, tflite_node, /*inputs=*/2, /*outputs=*/1));
    RETURN_IF_ERROR(RetrieveCustomInitialData(tflite_node, &tf_options));
    RETURN_IF_ERROR(CheckKernelsAndStrides(
        tf_options->filter_height, tf_options->filter_width,
        tf_options->stride_height, tf_options->stride_width));
    return OkStatus();
  }

  Status Parse(const TfLiteNode* tflite_node,
               const TfLiteRegistration* registration, GraphFloat32* graph,
               ObjectReader* reader) final {
    Node* node = graph->NewNode();
    node->operation.type = ToString(OperationType::MAX_UNPOOLING_2D);
    RETURN_IF_ERROR(reader->AddInput(node, 0));
    RETURN_IF_ERROR(reader->AddInput(node, 1));
    RETURN_IF_ERROR(reader->AddOutputs(node));
    auto input_shape = graph->FindInputs(node->id)[0]->tensor.shape;
    MaxUnpooling2DAttributes attr;
    const auto* tf_options = reinterpret_cast<const TfLitePoolParams*>(
        tflite_node->custom_initial_data);
    if (!tf_options) {
      return InternalError("Missing tflite params");
    }
    attr.kernel = ToHW(tf_options->filter_height, tf_options->filter_width);
    attr.strides = ToHW(tf_options->stride_height, tf_options->stride_width);
    UpdatePadding(tf_options->padding, input_shape, &attr);

    node->operation.attributes = attr;

    auto output_value = graph->FindOutputs(node->id)[0];
    output_value->tensor.shape = CalculateOutputShape(input_shape, attr);
    return OkStatus();
  }
};

class SoftMaxOperationParser : public TFLiteOperationParser {
 public:
  Status IsSupported(const TfLiteContext* context,
                     const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration) final {
    RETURN_IF_ERROR(CheckMaxSupportedOpVersion(registration, 1));
    RETURN_IF_ERROR(
        CheckInputsOutputs(context, tflite_node, /*inputs=*/1, /*outputs=*/1));
    TfLiteSoftmaxParams* tf_options = nullptr;
    RETURN_IF_ERROR(RetrieveBuiltinData(tflite_node, &tf_options));
    if (tf_options->beta != 1) {
      // TODO(eignasheva): figure out, what's wrong with softmax.
      return UnimplementedError("Softmax.beta != 1 is not supported.");
    }
    return OkStatus();
  }
  Status Parse(const TfLiteNode* tflite_node,
               const TfLiteRegistration* registration, GraphFloat32* graph,
               ObjectReader* reader) final {
    Node* node = graph->NewNode();
    node->operation.type = ToString(OperationType::SOFT_MAX);
    RETURN_IF_ERROR(reader->AddInput(node, 0));
    RETURN_IF_ERROR(reader->AddOutputs(node));

    const auto* tf_options =
        reinterpret_cast<const TfLiteSoftmaxParams*>(tflite_node->builtin_data);
    if (!tf_options) {
      return InternalError("Missing tflite params");
    }
    if (tf_options->beta != 1) {
      // there is multiply by scalar operation fused in SoftMax. Make a layer
      // out of it before SoftMax.
      return UnimplementedError("Softmax.beta != 1 is not supported.");
      // auto mul_node = reader->NewPassthroughNode(node);
      // mul_node->operation.type = ToString(OperationType::MUL);
    }
    SoftMaxAttributes attr;
    attr.axis = Axis::CHANNELS;  // always by channels
    node->operation.attributes = attr;
    return OkStatus();
  }
};

class AddOperationParser : public TFLiteOperationParser {
 public:
  Status IsSupported(const TfLiteContext* context,
                     const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration) final {
    RETURN_IF_ERROR(CheckMaxSupportedOpVersion(registration, 1));
    // TODO(eignasheva): add shapes check.
    TfLiteAddParams* tf_options = nullptr;
    RETURN_IF_ERROR(RetrieveBuiltinData(tflite_node, &tf_options));
    return OkStatus();
  }
  Status Parse(const TfLiteNode* tflite_node,
               const TfLiteRegistration* registration, GraphFloat32* graph,
               ObjectReader* reader) final {
    Node* node = graph->NewNode();
    node->operation.type = ToString(OperationType::ADD);
    RETURN_IF_ERROR(reader->AddOutputs(node));

    AddAttributes attr;
    for (int idx = 0; idx < tflite_node->inputs->size; ++idx) {
      if (!reader->AddInput(node, idx).ok()) {
        if (tflite_node->inputs->size != 2) {
          return InvalidArgumentError(
              "Broadcast Add should accept 2 inputs, one input tensor and "
              "broadcasted tensor");
        }
        TfLiteIntArray dims;
        RETURN_IF_ERROR(reader->GetTensorDims(1, &dims));
        if (dims.size <= 0) {
          Tensor<Scalar, DataType::FLOAT32> tensor;
          RETURN_IF_ERROR(reader->ReadTensor(1, &tensor));
          attr.param = tensor.data[0];
        } else {
          Tensor<Linear, DataType::FLOAT32> tensor;
          RETURN_IF_ERROR(reader->ReadTensor(1, &tensor));
          attr.param = std::move(tensor);
        }
      }
    }
    node->operation.attributes = std::move(attr);

    const auto* tf_options =
        reinterpret_cast<const TfLiteAddParams*>(tflite_node->builtin_data);
    if (!tf_options) {
      return InternalError("Missing tflite params");
    }
    RETURN_IF_ERROR(MaybeFuseActivationToTheSingleOutput(tf_options->activation,
                                                         graph, node));
    return OkStatus();
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
class LstmOperationParser : public TFLiteOperationParser {
 public:
  Status IsSupported(const TfLiteContext* context,
                     const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration) final {
    RETURN_IF_ERROR(CheckExactSupportedOpVersion(registration, 2));
    // TODO(eignasheva): Fix bad check.
    // RETURN_IF_ERROR(CheckInputsOutputs(context, tflite_node, /*inputs=*/5,
    //                                    /*outputs=*/4));
    TfLiteLSTMParams* tf_options = nullptr;
    RETURN_IF_ERROR(RetrieveBuiltinData(tflite_node, &tf_options));
    RETURN_IF_ERROR(CheckParameters(tf_options));
    return OkStatus();
  }

  Status Parse(const TfLiteNode* tflite_node,
               const TfLiteRegistration* registration, GraphFloat32* graph,
               ObjectReader* reader) final {
    if (tflite_node->inputs->size != 5) {
      return InvalidArgumentError("LSTM should have 5 input tensors");
    }
    if (tflite_node->outputs->size != 4) {
      return InvalidArgumentError("LSTM should have 4 output tensors");
    }

    const auto* params =
        reinterpret_cast<const TfLiteLSTMParams*>(tflite_node->builtin_data);
    if (!params) {
      return InternalError("Missing tflite params");
    }
    RETURN_IF_ERROR(CheckParameters(params));

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

    Value<TensorRefFloat32>* concat_temp;
    int concat_tensor_idx = tflite_node->outputs->data[2];
    RETURN_IF_ERROR(
        reader->ReadValueByTensorIdx(concat_tensor_idx, &concat_temp));
    Value<TensorRefFloat32>* activ_temp;
    int activ_tensor_idx = tflite_node->outputs->data[3];
    RETURN_IF_ERROR(
        reader->ReadValueByTensorIdx(activ_tensor_idx, &activ_temp));

    RETURN_IF_ERROR(reader->AddInput(concat_node, 0));  // input
    RETURN_IF_ERROR(reader->AddInput(concat_node, 1));  // prev_activ
    RETURN_IF_ERROR(graph->SetProducer(concat_node->id, concat_temp->id));

    RETURN_IF_ERROR(graph->AddConsumer(fc_node->id, concat_temp->id));
    RETURN_IF_ERROR(graph->SetProducer(fc_node->id, activ_temp->id));

    RETURN_IF_ERROR(graph->AddConsumer(lstm_node->id, activ_temp->id));
    RETURN_IF_ERROR(reader->AddInput(lstm_node, 4));       // prev_state
    RETURN_IF_ERROR(reader->AddOutput(lstm_node, 1));      // new_state
    RETURN_IF_ERROR(reader->AddOutput(lstm_node, 0));      // activation

    return OkStatus();
  }

 private:
  Status CheckParameters(const TfLiteLSTMParams* tf_options) {
    if (tf_options->kernel_type !=
        TfLiteLSTMKernelType::kTfLiteLSTMBasicKernel) {
      return UnimplementedError("Only kTfLiteLSTMBasicKernel is supported.");
    }
    if (tf_options->activation != kTfLiteActTanh) {
      return UnimplementedError("Only TANH activation is supported.");
    }
    if (tf_options->cell_clip != 0.0f) {
      return UnimplementedError("cell_clip is not supported.");
    }
    if (tf_options->proj_clip != 0.0f) {
      return UnimplementedError("proj_clip is not supported.");
    }
    return OkStatus();
  }
};

class ResizeBilinearOperationParser : public TFLiteOperationParser {
 public:
  Status IsSupported(const TfLiteContext* context,
                     const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration) final {
    RETURN_IF_ERROR(CheckMaxSupportedOpVersion(registration, 1));
    RETURN_IF_ERROR(
        CheckInputsOutputs(context, tflite_node, /*inputs=*/1, /*outputs=*/1));

    // TODO(eignasheva): check shapes.
    TfLiteResizeBilinearParams* tf_options = nullptr;
    RETURN_IF_ERROR(RetrieveBuiltinData(tflite_node, &tf_options));
    return OkStatus();
  }
  Status Parse(const TfLiteNode* tflite_node,
               const TfLiteRegistration* registration, GraphFloat32* graph,
               ObjectReader* reader) final {
    Node* node = graph->NewNode();
    node->operation.type = ToString(OperationType::UPSAMPLE_2D);
    RETURN_IF_ERROR(reader->AddInput(node, 0));
    RETURN_IF_ERROR(reader->AddOutputs(node));
    // Here we may have extra inputs. Other tensors were supposed to
    // define new shape, but in TFLite these are ignored.

    const auto* tf_options =
        reinterpret_cast<const TfLiteResizeBilinearParams*>(
            tflite_node->builtin_data);
    if (!tf_options) {
      return InternalError("Missing tflite params");
    }
    Upsample2DAttributes attr;
    attr.align_corners = tf_options->align_corners;
    attr.type = UpsamplingType::BILINEAR;
    attr.new_shape.CopyAllDefinedAxis(
        graph->FindOutputs(node->id)[0]->tensor.shape);
    node->operation.attributes = attr;
    return OkStatus();
  }
};

class PadOperationParser : public TFLiteOperationParser {
 public:
  Status IsSupported(const TfLiteContext* context,
                     const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration) final {
    RETURN_IF_ERROR(CheckMaxSupportedOpVersion(registration, 1));
    RETURN_IF_ERROR(
        CheckInputsOutputs(context, tflite_node, /*inputs=*/1, /*outputs=*/1));
    RETURN_IF_ERROR(CheckTensorIsAvailable(context, tflite_node, 1));
    return OkStatus();
  }
  Status Parse(const TfLiteNode* tflite_node,
               const TfLiteRegistration* registration, GraphFloat32* graph,
               ObjectReader* reader) final {
    Node* node = graph->NewNode();
    node->operation.type = ToString(OperationType::PAD);
    RETURN_IF_ERROR(reader->AddInput(node, 0));
    RETURN_IF_ERROR(reader->AddOutputs(node));

    PadAttributes attr;
    attr.type = PaddingContentType::ZEROS;
    Tensor<HW, DataType::INT32> paddings;
    RETURN_IF_ERROR(reader->ReadTensor(1, &paddings));

    // 4x2 tensor with paddings.
    if (paddings.shape.h != 4 || paddings.shape.w != 2) {
      return InvalidArgumentError("Paddings tensor has unexpected shape.");
    }
    if (paddings.data[0] != 0 || paddings.data[1] != 0) {
      return UnimplementedError("Padding for BATCH channel is not supported.");
    }
    attr.prepended = HWC(paddings.data[2], paddings.data[4], paddings.data[6]);
    attr.appended = HWC(paddings.data[3], paddings.data[5], paddings.data[7]);
    node->operation.attributes = attr;
    return OkStatus();
  }
};

class ElementwiseOperationParser : public TFLiteOperationParser {
 public:
  explicit ElementwiseOperationParser(OperationType operation_type)
      : operation_type_(operation_type) {}
  Status IsSupported(const TfLiteContext* context,
                     const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration) final {
    RETURN_IF_ERROR(CheckMaxSupportedOpVersion(registration, 1));
    if (IsTwoArgumentOperation()) {
      RETURN_IF_ERROR(CheckInputsOutputs(context, tflite_node, /*inputs=*/2,
                                         /*outputs=*/1));
      TfLiteSubParams* tf_options = nullptr;
      RETURN_IF_ERROR(RetrieveBuiltinData(tflite_node, &tf_options));
      RETURN_IF_ERROR(CheckActivationSupported(tf_options->activation));
    } else if (!IsOneArgumentOperation()) {
      return InvalidArgumentError("Incorrect operation type passed");
    }

    return OkStatus();
  }
  Status Parse(const TfLiteNode* tflite_node,
               const TfLiteRegistration* registration, GraphFloat32* graph,
               ObjectReader* reader) final {
    Node* node = graph->NewNode();
    node->operation.type = ToString(operation_type_);

    if (IsOneArgumentOperation()) {
      RETURN_IF_ERROR(reader->AddInput(node, 0));
    } else if (IsTwoArgumentOperation()) {
      if (tflite_node->inputs->size != 2) {
        return InvalidArgumentError("Applies only two input tensors");
      }
      RETURN_IF_ERROR(reader->AddInput(node, 0));
      RETURN_IF_ERROR(reader->AddInput(node, 1));

      TfLiteFusedActivation activation = kTfLiteActNone;
      switch (operation_type_) {
        case OperationType::SUB: {
          const auto* tf_options = reinterpret_cast<const TfLiteSubParams*>(
              tflite_node->builtin_data);
          if (tf_options != nullptr) {
            activation = tf_options->activation;
          }
          break;
        }
        case OperationType::DIV: {
          const auto* tf_options = reinterpret_cast<const TfLiteDivParams*>(
              tflite_node->builtin_data);
          if (tf_options != nullptr) {
            activation = tf_options->activation;
          }
          break;
        }
        default:
          // No activation expected.
          activation = kTfLiteActNone;
      }

      if (activation) {
        RETURN_IF_ERROR(
            MaybeFuseActivationToTheSingleOutput(activation, graph, node));
      }
    } else {
      return InvalidArgumentError("Incorrect operation type passed");
    }

    return reader->AddOutputs(node);
  }

 private:
  bool IsOneArgumentOperation() const {
    switch (operation_type_) {
      case OperationType::ABS:
      case OperationType::SIN:
      case OperationType::COS:
      case OperationType::LOG:
      case OperationType::SQRT:
      case OperationType::RSQRT:
      case OperationType::SQUARE:
      case OperationType::SIGMOID:
      case OperationType::TANH:
        return true;
      default:
        return false;
    }
  }

  bool IsTwoArgumentOperation() const {
    switch (operation_type_) {
      case OperationType::SUB:
      case OperationType::DIV:
      case OperationType::POW:
      case OperationType::SQUARED_DIFF:
        return true;
      default:
        return false;
    }
  }

  OperationType operation_type_;
};

class PReLuOperationParser : public TFLiteOperationParser {
 public:
  Status IsSupported(const TfLiteContext* context,
                     const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration) final {
    RETURN_IF_ERROR(CheckMaxSupportedOpVersion(registration, 1));
    // TODO(eignasheva): add params check
    return OkStatus();
  }
  Status Parse(const TfLiteNode* tflite_node,
               const TfLiteRegistration* registration, GraphFloat32* graph,
               ObjectReader* reader) final {
    Node* node = graph->NewNode();
    node->operation.type = ToString(OperationType::PRELU);
    RETURN_IF_ERROR(reader->AddInput(node, 0));
    auto input_shape = graph->FindInputs(node->id)[0]->tensor.shape;

    PReLUAttributes attr;
    Tensor<Linear, DataType::FLOAT32> linear_alpha;
    Status status = reader->ReadTensor(1, &linear_alpha);
    if (status.ok()) {
      if (linear_alpha.shape.v != input_shape.c) {
        return InvalidArgumentError(
            "Linear alpha shape does not match the number of input channels.");
      }
      attr.alpha = std::move(linear_alpha);
    } else {
      Tensor<HWC, DataType::FLOAT32> hwc_alpha;
      RETURN_IF_ERROR(reader->ReadTensor(1, &hwc_alpha));
      if (hwc_alpha.shape.h != input_shape.h ||
          hwc_alpha.shape.w != input_shape.w ||
          hwc_alpha.shape.c != input_shape.c) {
        return InvalidArgumentError("Alpha shape does not match input shape.");
      }
      attr.alpha = std::move(hwc_alpha);
    }
    node->operation.attributes = std::move(attr);
    return reader->AddOutputs(node);
  }
};

class ReLuOperationParser : public TFLiteOperationParser {
 public:
  Status IsSupported(const TfLiteContext* context,
                     const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration) final {
    RETURN_IF_ERROR(CheckMaxSupportedOpVersion(registration, 1));
    return OkStatus();
  }
  explicit ReLuOperationParser(int clip) : clip_(clip) {}
  Status Parse(const TfLiteNode* tflite_node,
               const TfLiteRegistration* registration, GraphFloat32* graph,
               ObjectReader* reader) final {
    Node* node = graph->NewNode();
    node->operation.type = ToString(OperationType::RELU);
    RETURN_IF_ERROR(reader->AddInput(node, 0));

    ReLUAttributes attr;
    TfLiteLeakyReluParams* tf_options = nullptr;
    RetrieveBuiltinData(tflite_node, &tf_options).IgnoreError();
    attr.alpha = tf_options ? tf_options->alpha : 0;
    attr.clip = clip_;
    node->operation.attributes = attr;
    return reader->AddOutputs(node);
  }

 private:
  int clip_;
};

class MulOperationParser : public TFLiteOperationParser {
 public:
  Status IsSupported(const TfLiteContext* context,
                     const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration) final {
    RETURN_IF_ERROR(CheckMaxSupportedOpVersion(registration, 1));
    // TODO(eignasheva): add params check
    return OkStatus();
  }
  Status Parse(const TfLiteNode* tflite_node,
               const TfLiteRegistration* registration, GraphFloat32* graph,
               ObjectReader* reader) final {
    Node* node = graph->NewNode();
    if (reader->GetNumberOfRuntimeInputs() == 2) {
      // ApplyMask operation
      node->operation.type = ToString(OperationType::APPLY_MASK);
      RETURN_IF_ERROR(reader->AddInput(node, 0));
      RETURN_IF_ERROR(reader->AddInput(node, 1));
    } else {
      node->operation.type = ToString(OperationType::MULTIPLY_SCALAR);
      RETURN_IF_ERROR(reader->AddInput(node, 0));
      MultiplyScalarAttributes attr;
      TfLiteIntArray dims;
      RETURN_IF_ERROR(reader->GetTensorDims(1, &dims));
      if (dims.size <= 0) {
        Tensor<Scalar, DataType::FLOAT32> tensor;
        RETURN_IF_ERROR(reader->ReadTensor(1, &tensor));
        attr.param = tensor.data[0];
      } else {
        Tensor<Linear, DataType::FLOAT32> tensor;
        RETURN_IF_ERROR(reader->ReadTensor(1, &tensor));
        attr.param = std::move(tensor);
      }
      node->operation.attributes = std::move(attr);
    }
    return reader->AddOutputs(node);
  }
};

class FullyConnectedOperationParser : public TFLiteOperationParser {
 public:
  Status IsSupported(const TfLiteContext* context,
                     const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration) final {
    RETURN_IF_ERROR(CheckMaxSupportedOpVersion(registration, 1));
    TfLiteFullyConnectedParams* tf_options = nullptr;
    RETURN_IF_ERROR(RetrieveBuiltinData(tflite_node, &tf_options));
    if (tf_options->weights_format !=
        kTfLiteFullyConnectedWeightsFormatDefault) {
      return UnimplementedError("Unsupported FullyConnected weights format.");
    }
    // TODO(eignasheva): check input shape
    return OkStatus();
  }
  Status Parse(const TfLiteNode* tflite_node,
               const TfLiteRegistration* registration, GraphFloat32* graph,
               ObjectReader* reader) final {
    Node* node = graph->NewNode();
    RETURN_IF_ERROR(reader->AddInput(node, 0));

    const auto* tf_options =
        reinterpret_cast<const TfLiteFullyConnectedParams*>(
            tflite_node->builtin_data);
    if (tf_options->weights_format !=
        kTfLiteFullyConnectedWeightsFormatDefault) {
      return UnimplementedError("Unsupported FullyConnected weights format.");
    }

    FullyConnectedAttributes attr;
    RETURN_IF_ERROR(GetFullyConnectedAttributes(1, 2, reader, &attr));

    Tensor<HW, DataType::FLOAT32> weights;
    RETURN_IF_ERROR(reader->ReadTensor(1, &weights));
    auto input = graph->FindInputs(node->id)[0];
    int batch_size = input->tensor.shape.b;
    if (input->tensor.shape.DimensionsProduct() / batch_size !=
        weights.shape.w) {
      return UnimplementedError(
          "Amount of input data should match weights width");
    }

    Node* conv = node;
    if (input->tensor.shape.h != 1 || input->tensor.shape.w != 1) {
      auto& reshape = node;
      conv = graph->NewNode();  // reset conv pointer!
      Value<TensorRefFloat32>* reshaped_value = graph->NewValue();
      reshaped_value->tensor.shape = BHWC(1, 1, 1, weights.shape.w);
      RETURN_IF_ERROR(graph->SetProducer(reshape->id, reshaped_value->id));
      reshape->operation.type = ToString(OperationType::RESHAPE);
      ReshapeAttributes attr;
      attr.new_shape = reshaped_value->tensor.shape;
      reshape->operation.attributes = attr;
      RETURN_IF_ERROR(graph->AddConsumer(conv->id, reshaped_value->id));
    }

    conv->operation.type = ToString(OperationType::FULLY_CONNECTED);
    conv->operation.attributes = std::move(attr);
    Status result = reader->AddOutputs(conv);
    RETURN_IF_ERROR(MaybeFuseActivationToTheSingleOutput(tf_options->activation,
                                                         graph, conv));

    return result;
  }
};

class StridedSliceOperationParser : public TFLiteOperationParser {
 public:
  Status IsSupported(const TfLiteContext* context,
                     const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration) final {
    RETURN_IF_ERROR(CheckMaxSupportedOpVersion(registration, 1));
    TfLiteStridedSliceParams* tf_options = nullptr;
    RETURN_IF_ERROR(RetrieveBuiltinData(tflite_node, &tf_options));
    RETURN_IF_ERROR(CheckOptionsSupport(tf_options));
    return OkStatus();
  }
  Status Parse(const TfLiteNode* tflite_node,
               const TfLiteRegistration* registration, GraphFloat32* graph,
               ObjectReader* reader) final {
    Node* node = graph->NewNode();
    node->operation.type = ToString(OperationType::SLICE);
    RETURN_IF_ERROR(reader->AddOutputs(node));
    Value<TensorRefFloat32>* input;
    RETURN_IF_ERROR(reader->ReadValue(0, &input));
    RETURN_IF_ERROR(graph->AddConsumer(node->id, input->id));

    Tensor<Linear, DataType::INT32> tmp;
    RETURN_IF_ERROR(reader->ReadTensor(1, &tmp));

    bool read_without_batch = tmp.data.size() == 3;
    bool read_with_batch = tmp.data.size() == 4;
    if (!read_without_batch && !read_with_batch) {
      return UnimplementedError(
          "Slicing is supported for 3 or 4 dimensional tensors only.");
    }

    const auto* tf_options = reinterpret_cast<const TfLiteStridedSliceParams*>(
        tflite_node->builtin_data);
    auto out_shape = graph->FindOutputs(node->id)[0]->tensor.shape;
    if (!tf_options) {
      return InternalError("Missing tflite params");
    }
    RETURN_IF_ERROR(CheckOptionsSupport(tf_options));

    SliceAttributes attr;
    if (read_without_batch) {
      RETURN_IF_ERROR(ReadAttribsWithoutBatch(reader, tf_options,
                                              input->tensor.shape, &attr));
    }
    if (read_with_batch) {
      RETURN_IF_ERROR(
          ReadAttribsWithBatch(reader, tf_options, input->tensor.shape, &attr));
    }
    if (attr.strides.h < 0 || attr.strides.w < 0 || attr.strides.c < 0) {
      return UnimplementedError("Reverse slices are not supported.");
    }
    if (attr.ends.h - attr.starts.h != out_shape.h) {
      return UnimplementedError("Output height doesn't match");
    }
    if (attr.ends.w - attr.starts.w != out_shape.w) {
      return UnimplementedError("Output width doesn't match");
    }
    if (attr.ends.c - attr.starts.c != out_shape.c) {
      return UnimplementedError("Output channels don't match");
    }
    node->operation.attributes = attr;
    return OkStatus();
  }

 private:
  Status UpdateWithMask(const TfLiteStridedSliceParams* tf_options,
                        const BHWC& input_shape, int ignore_h, int ignore_w,
                        int ignore_c, SliceAttributes* attr) {
    if (tf_options->begin_mask & ignore_h) {
      attr->starts.h = 0;
    }
    if (tf_options->begin_mask & ignore_w) {
      attr->starts.w = 0;
    }
    if (tf_options->begin_mask & ignore_c) {
      attr->starts.c = 0;
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
    return OkStatus();
  }

  Status UpdateIfNegative(const BHWC& input_shape, SliceAttributes* attr) {
    if (attr->ends.h < 0) {
      attr->ends.h = input_shape.h + attr->ends.h;
    }
    if (attr->ends.w < 0) {
      attr->ends.w = input_shape.w + attr->ends.w;
    }
    if (attr->ends.c < 0) {
      attr->ends.c = input_shape.c + attr->ends.c;
    }
    return OkStatus();
  }

  Status ReadAttribsWithBatch(const ObjectReader* reader,
                              const TfLiteStridedSliceParams* tf_options,
                              const BHWC& input_shape, SliceAttributes* attr) {
    auto read_hwc = [&](int tensor_index, HWC* hwc) -> Status {
      Tensor<Linear, DataType::INT32> t;
      RETURN_IF_ERROR(reader->ReadTensor(tensor_index, &t));
      if (t.data[0] != 1 && t.data[0] != 0) {
        return UnimplementedError(
            "Slicing for BATCH channel is not supported. If you use batch it "
            "should be 0 or 1");
      }
      *hwc = HWC(t.data[1], t.data[2], t.data[3]);
      return OkStatus();
    };

    RETURN_IF_ERROR(read_hwc(1, &attr->starts));
    RETURN_IF_ERROR(read_hwc(2, &attr->ends));
    RETURN_IF_ERROR(read_hwc(3, &attr->strides));
    RETURN_IF_ERROR(UpdateIfNegative(input_shape, attr));
    RETURN_IF_ERROR(UpdateWithMask(tf_options, input_shape, 2, 4, 8, attr));
    return OkStatus();
  }

  Status ReadAttribsWithoutBatch(const ObjectReader* reader,
                                 const TfLiteStridedSliceParams* tf_options,
                                 const BHWC& input_shape,
                                 SliceAttributes* attr) {
    auto read_hwc = [&](int tensor_index, HWC* hwc) -> Status {
      Tensor<Linear, DataType::INT32> t;
      RETURN_IF_ERROR(reader->ReadTensor(tensor_index, &t));
      *hwc = HWC(t.data[0], t.data[1], t.data[2]);
      return OkStatus();
    };

    RETURN_IF_ERROR(read_hwc(1, &attr->starts));
    RETURN_IF_ERROR(read_hwc(2, &attr->ends));
    RETURN_IF_ERROR(read_hwc(3, &attr->strides));
    RETURN_IF_ERROR(UpdateIfNegative(input_shape, attr));
    RETURN_IF_ERROR(UpdateWithMask(tf_options, input_shape, 1, 2, 4, attr));
    return OkStatus();
  }
  Status CheckOptionsSupport(const TfLiteStridedSliceParams* tf_options) {
    if (tf_options->ellipsis_mask) {
      return UnimplementedError("Slice does not support ellipsis_mask.");
    }
    if (tf_options->new_axis_mask) {
      return UnimplementedError("Slice does not support new_axis_mask.");
    }
    if (tf_options->shrink_axis_mask) {
      return UnimplementedError(
          "Slice does not support shrink_axis_mask parameter. ");
    }
    return OkStatus();
  }
};

class TransposeConvOperationParser : public TFLiteOperationParser {
 public:
  Status IsSupported(const TfLiteContext* context,
                     const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration) final {
    RETURN_IF_ERROR(CheckMaxSupportedOpVersion(registration, 1));
    RETURN_IF_ERROR(CheckTensorIsAvailable(context, tflite_node, 1));
    TfLiteTransposeConvParams* tf_options = nullptr;
    RETURN_IF_ERROR(RetrieveBuiltinData(tflite_node, &tf_options));
    RETURN_IF_ERROR(
        CheckStrides(tf_options->stride_height, tf_options->stride_width));
    return OkStatus();
  }
  // TFLite's TRANSPOSE_CONV expects 3 input (output shape, weights, and input)
  // and allows configurable padding & stride.
  // TODO(impjdi): Translate output_shape to attr.adjacent.
  Status Parse(const TfLiteNode* tflite_node,
               const TfLiteRegistration* registration, GraphFloat32* graph,
               ObjectReader* reader) final {
    auto* node = graph->NewNode();
    node->operation.type = ToString(OperationType::CONVOLUTION_TRANSPOSED);
    Value<TensorRefFloat32>* input;
    RETURN_IF_ERROR(reader->ReadValue(2, &input));
    RETURN_IF_ERROR(graph->AddConsumer(node->id, input->id));
    RETURN_IF_ERROR(reader->AddOutputs(node));

    const auto* tf_options = reinterpret_cast<const TfLiteTransposeConvParams*>(
        tflite_node->builtin_data);
    if (!tf_options) {
      return InternalError("Missing tflite options.");
    }
    ConvolutionTransposedAttributes attr;
    attr.stride = tf_options
                      ? HW(tf_options->stride_height, tf_options->stride_width)
                      : HW(1, 1);
    RETURN_IF_ERROR(reader->ReadTensor(1, &attr.weights));

    // TFLite does not support bias.

    UpdatePadding(tf_options->padding,
                  graph->FindInputs(node->id)[0]->tensor.shape, &attr);
    node->operation.attributes = std::move(attr);
    return OkStatus();
  }
};

class Convolution2DTransposeBiasParser : public TFLiteOperationParser {
 public:
  Status IsSupported(const TfLiteContext* context,
                     const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration) final {
    RETURN_IF_ERROR(CheckTensorIsAvailable(context, tflite_node, 1));
    TfLiteTransposeConvParams* tf_options = nullptr;
    RETURN_IF_ERROR(RetrieveCustomInitialData(tflite_node, &tf_options));
    RETURN_IF_ERROR(
        CheckStrides(tf_options->stride_height, tf_options->stride_width));
    return OkStatus();
  }
  Status Parse(const TfLiteNode* tflite_node,
               const TfLiteRegistration* registration, GraphFloat32* graph,
               ObjectReader* reader) final {
    auto* node = graph->NewNode();
    node->operation.type = ToString(OperationType::CONVOLUTION_TRANSPOSED);
    RETURN_IF_ERROR(reader->AddInput(node, 0));
    RETURN_IF_ERROR(reader->AddOutputs(node));

    const auto* params = reinterpret_cast<const TfLiteTransposeConvParams*>(
        tflite_node->custom_initial_data);
    ConvolutionTransposedAttributes attr;
    attr.stride =
        params ? HW(params->stride_height, params->stride_width) : HW(1, 1);

    RETURN_IF_ERROR(reader->ReadTensor(1, &attr.weights));
    reader->ReadTensor(2, &attr.bias).IgnoreError();  // bias is optional

    UpdatePadding(params->padding, graph->FindInputs(node->id)[0]->tensor.shape,
                  &attr);

    node->operation.attributes = std::move(attr);
    return OkStatus();
  }
};

class SpaceToBatchOperationParser : public TFLiteOperationParser {
 public:
  Status IsSupported(const TfLiteContext* context,
                     const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration) final {
    return OkStatus();
  }
  Status Parse(const TfLiteNode* tflite_node,
               const TfLiteRegistration* registration, GraphFloat32* graph,
               ObjectReader* reader) final {
    auto* node = graph->NewNode();
    node->operation.type = ToString(OperationType::SPACE_TO_BATCH);
    RETURN_IF_ERROR(reader->AddInput(node, 0));
    RETURN_IF_ERROR(reader->AddOutputs(node));
    SpaceToBatchAttributes sb_attr;
    Tensor<Linear, DataType::INT32> block;
    RETURN_IF_ERROR(reader->ReadTensor(1, &block));
    if (block.shape.v != 2) {
      return InternalError("Space has to be HxW.");
    }
    sb_attr.block.h = block.data[0];
    sb_attr.block.w = block.data[1];

    Tensor<HW, DataType::INT32> padding;
    RETURN_IF_ERROR(reader->ReadTensor(2, &padding));
    auto padding_shape = padding.shape;

    if (padding_shape.h != 2 && padding_shape.w != 2) {
      return InternalError("Space has to be HxW.");
    }

    sb_attr.padding.prepended.h = padding.data[0];
    sb_attr.padding.prepended.w = padding.data[2];

    sb_attr.padding.appended.h = padding.data[1];
    sb_attr.padding.appended.w = padding.data[3];

    node->operation.attributes = std::move(sb_attr);
    return OkStatus();
  }
};

class BatchToSpaceOperationParser : public TFLiteOperationParser {
 public:
  Status IsSupported(const TfLiteContext* context,
                     const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration) final {
    return OkStatus();
  }
  Status Parse(const TfLiteNode* tflite_node,
               const TfLiteRegistration* registration, GraphFloat32* graph,
               ObjectReader* reader) final {
    auto* node = graph->NewNode();
    node->operation.type = ToString(OperationType::BATCH_TO_SPACE);
    RETURN_IF_ERROR(reader->AddInput(node, 0));
    RETURN_IF_ERROR(reader->AddOutputs(node));

    BatchToSpaceAttributes bs_attr;
    Tensor<Linear, DataType::INT32> block;
    RETURN_IF_ERROR(reader->ReadTensor(1, &block));
    if (block.shape.v != 2) {
      return InternalError("Space has to be HxW.");
    }
    bs_attr.block.h = block.data[0];
    bs_attr.block.w = block.data[1];

    Tensor<HW, DataType::INT32> crop;
    RETURN_IF_ERROR(reader->ReadTensor(2, &crop));
    auto crop_shape = crop.shape;
    if (crop_shape.h != 2 && crop_shape.w != 2) {
      return InternalError("Space has to be HxW.");
    }

    bs_attr.crop.prepended.h = crop.data[0];
    bs_attr.crop.prepended.w = crop.data[2];

    bs_attr.crop.appended.h = crop.data[1];
    bs_attr.crop.appended.w = crop.data[3];

    node->operation.attributes = std::move(bs_attr);
    return OkStatus();
  }
};

class UnsupportedOperationParser : public TFLiteOperationParser {
  Status IsSupported(const TfLiteContext* context,
                     const TfLiteNode* tflite_node,
                     const TfLiteRegistration* registration) final {
    return UnimplementedError("Operation is not supported.");
  }

  Status Parse(const TfLiteNode* tflite_node,
               const TfLiteRegistration* registration, GraphFloat32* graph,
               ObjectReader* reader) final {
    return UnimplementedError("Operation is not supported.");
  }
};

std::unique_ptr<TFLiteOperationParser> NewOperationParser(
    const TfLiteRegistration* registration) {
  const auto builtin_code = registration->builtin_code;
  const absl::string_view custom_name = registration->custom_name;
  switch (builtin_code) {
    case kTfLiteBuiltinAbs:
      return make_unique<ElementwiseOperationParser>(OperationType::ABS);
    case kTfLiteBuiltinAdd:
      return make_unique<AddOperationParser>();
    case kTfLiteBuiltinAveragePool2d:
      return make_unique<Pooling2DOperationParser>(PoolingType::AVERAGE);
    case kTfLiteBuiltinConcatenation:
      return make_unique<ConcatenationOperationParser>();
    case kTfLiteBuiltinConv2d:
      return make_unique<Conv2DOperationParser>();
    case kTfLiteBuiltinCos:
      return make_unique<ElementwiseOperationParser>(OperationType::COS);
    case kTfLiteBuiltinDepthwiseConv2d:
      return make_unique<DepthwiseConvolutionOperationParser>();
    case kTfLiteBuiltinDiv:
      return make_unique<ElementwiseOperationParser>(OperationType::DIV);
    case kTfLiteBuiltinFullyConnected:
      return make_unique<FullyConnectedOperationParser>();
    case kTfLiteBuiltinLogistic:
      return make_unique<ElementwiseOperationParser>(OperationType::SIGMOID);
    case kTfLiteBuiltinLog:
      return make_unique<ElementwiseOperationParser>(OperationType::LOG);
    case kTfLiteBuiltinLstm:
      return make_unique<LstmOperationParser>();
    case kTfLiteBuiltinMaxPool2d:
      return make_unique<Pooling2DOperationParser>(PoolingType::MAX);
    case kTfLiteBuiltinMul:
      return make_unique<MulOperationParser>();
    case kTfLiteBuiltinPad:
      return make_unique<PadOperationParser>();
    case kTfLiteBuiltinPow:
      return make_unique<ElementwiseOperationParser>(OperationType::POW);
    case kTfLiteBuiltinRelu:
      return make_unique<ReLuOperationParser>(0);
    case kTfLiteBuiltinRelu6:
      return make_unique<ReLuOperationParser>(6);
    case kTfLiteBuiltinLeakyRelu:
      return make_unique<ReLuOperationParser>(0);
    case kTfLiteBuiltinPrelu:
      return make_unique<PReLuOperationParser>();
    case kTfLiteBuiltinReshape:
      return make_unique<ReshapeOperationParser>();
    case kTfLiteBuiltinResizeBilinear:
      return make_unique<ResizeBilinearOperationParser>();
    case kTfLiteBuiltinRsqrt:
      return make_unique<ElementwiseOperationParser>(OperationType::RSQRT);
    case kTfLiteBuiltinSin:
      return make_unique<ElementwiseOperationParser>(OperationType::SIN);
    case kTfLiteBuiltinSoftmax:
      return make_unique<SoftMaxOperationParser>();
    case kTfLiteBuiltinStridedSlice:
      return make_unique<StridedSliceOperationParser>();
    case kTfLiteBuiltinSqrt:
      return make_unique<ElementwiseOperationParser>(OperationType::SQRT);
    case kTfLiteBuiltinSquare:
      return make_unique<ElementwiseOperationParser>(OperationType::SQUARE);
    case kTfLiteBuiltinSquaredDifference:
      return make_unique<ElementwiseOperationParser>(
          OperationType::SQUARED_DIFF);
    case kTfLiteBuiltinSub:
      return make_unique<ElementwiseOperationParser>(OperationType::SUB);
    case kTfLiteBuiltinTanh:
      return make_unique<ElementwiseOperationParser>(OperationType::TANH);
    case kTfLiteBuiltinTransposeConv:
      return make_unique<TransposeConvOperationParser>();

    case kTfLiteBuiltinCustom:
      if (custom_name == "Convolution2DTransposeBias") {
        return make_unique<Convolution2DTransposeBiasParser>();
      }
      if (custom_name == "MaxPoolingWithArgmax2D") {
        return make_unique<Pooling2DOperationParser>(PoolingType::MAX);
      }
      if (custom_name == "MaxUnpooling2D") {
        return make_unique<Unpooling2DOperationParser>();
      }
      break;
  }
  return make_unique<UnsupportedOperationParser>();
}

}  // namespace

Status ConvertTfLiteTensorToTensorRef(const TfLiteTensor& tflite_tensor,
                                      TensorRefFloat32* tensor_ref) {
  tensor_ref->type = ToDataType(tflite_tensor.type);
  const TfLiteIntArray* dims = tflite_tensor.dims;
  switch (dims->size) {
    case 1:
      tensor_ref->shape = BHWC(dims->data[0], 1, 1, 1);
      break;
    case 2:
      tensor_ref->shape = BHWC(dims->data[0], 1, 1, dims->data[1]);
      break;
    case 3:
      tensor_ref->shape = BHWC(dims->data[0], 1, dims->data[1], dims->data[2]);
      break;
    case 4:
      tensor_ref->shape =
          BHWC(dims->data[0], dims->data[1], dims->data[2], dims->data[3]);
      break;
    default:
      return InvalidArgumentError(StrCat(
          "Tensor ref has unsupported number of dimensions: ", dims->size));
  }
  return OkStatus();
}

Status IsSupported(const TfLiteContext* context, TfLiteNode* node,
                   const TfLiteRegistration* registration) {
  return NewOperationParser(registration)
      ->IsSupported(context, node, registration);
}

bool IsAllFloatTensors(const TfLiteContext* context,
                       const TfLiteIntArray* array) {
  for (int i = 0; i < array->size; ++i) {
    const TfLiteTensor* t = context->tensors + array->data[i];
    if (t->allocation_type == kTfLiteArenaRw && t->type != kTfLiteFloat32) {
      return false;
    }
  }
  return true;
}

std::string GetOpNameByRegistration(const TfLiteRegistration* registration) {
  auto op = registration->builtin_code;
  std::string result =
      EnumNameBuiltinOperator(static_cast<BuiltinOperator>(op));
  if (op == kTfLiteBuiltinCustom) {
    result += " " + std::string(registration->custom_name);
  }
  return result;
}

Status GetNodeAndRegistration(TfLiteContext* context, int node_id,
                              TfLiteNode** tflite_node,
                              TfLiteRegistration** registration) {
  if (context->GetNodeAndRegistration(context, node_id, tflite_node,
                                      registration) != kTfLiteOk) {
    return InvalidArgumentError(
        StrCat("Couldn't get node and registration info for op: ", node_id));
  }
  return OkStatus();
}

// TODO(impjdi): Check number of input/output tensors and their dimensions.
// TODO(impjdi): Check ops' parameters.
TfLiteIntArray* GetOpsToReplace(TfLiteContext* context) {
  TfLiteIntArray* execution_plan = nullptr;
  if (context->GetExecutionPlan(context, &execution_plan) != kTfLiteOk) {
    context->ReportError(context, "Unable to get graph execution plan.");
    return nullptr;
  }
  TfLiteIntArray* subgraph = TfLiteIntArrayCreate(execution_plan->size);
  subgraph->size = 0;
  std::set<std::string> errors;
  for (int i = 0; i < execution_plan->size; ++i) {
    TfLiteNode* node = nullptr;
    TfLiteRegistration* registration = nullptr;
    auto status = GetNodeAndRegistration(context, i, &node, &registration);
    if (!status.ok()) {
      context->ReportError(context, status.error_message().c_str());
      return nullptr;
    }
    status = IsSupported(context, node, registration);
    if (status.ok() &&
        // TODO(eignasheva): resolve sub operation support for metal delegate
        // registration->builtin_code != kTfLiteBuiltinSub &&
        IsAllFloatTensors(context, node->inputs) &&
        IsAllFloatTensors(context, node->outputs)) {
      if (errors.empty()) subgraph->data[subgraph->size++] = i;
    } else {
      errors.insert(GetOpNameByRegistration(registration) + ": " +
                    status.error_message());
    }
  }
  if (!errors.empty()) {
    std::string unsupported = absl::StrJoin(errors, "\n");
    std::string error_message =
        "Next operations are not supported by GPU delegate:\n" + unsupported +
        "\nFirst " + std::to_string(subgraph->size) +
        " operations will run on the GPU, and the remaining " +
        std::to_string(execution_plan->size - subgraph->size) + " on the CPU.";
    context->ReportError(context, error_message.c_str());
  }
  return subgraph;
}

Status BuildModel(TfLiteContext* context,
                  const TfLiteDelegateParams* delegate_params,
                  GraphFloat32* graph) {
  std::vector<std::unique_ptr<TFLiteOperationParser>> operations;
  for (int i = 0; i < delegate_params->nodes_to_replace->size; ++i) {
    TfLiteNode* tflite_node = nullptr;
    TfLiteRegistration* registration = nullptr;
    RETURN_IF_ERROR(GetNodeAndRegistration(
        context, delegate_params->nodes_to_replace->data[i], &tflite_node,
        &registration));
    auto op_parser = NewOperationParser(registration);
    if (!op_parser) {
      return UnimplementedError(
          StrCat("Operation ", registration->builtin_code, "(",
                 registration->custom_name,
                 ") is not supported by TFLite GPU Delegate."));
    }
    operations.push_back(std::move(op_parser));
  }
  std::vector<Value<TensorRefFloat32>*> tensor_to_value(context->tensors_size,
                                                        nullptr);
  for (int i = 0; i < delegate_params->nodes_to_replace->size; ++i) {
    TfLiteNode* tflite_node = nullptr;
    TfLiteRegistration* registration = nullptr;
    RETURN_IF_ERROR(GetNodeAndRegistration(
        context, delegate_params->nodes_to_replace->data[i], &tflite_node,
        &registration));
    ObjectReader reader(graph, context, tflite_node, &tensor_to_value);
    RETURN_IF_ERROR(
        operations[i]->Parse(tflite_node, registration, graph, &reader));
  }
  return OkStatus();
}

}  // namespace gpu
}  // namespace tflite
