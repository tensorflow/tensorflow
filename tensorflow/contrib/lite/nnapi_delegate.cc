/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/contrib/lite/nnapi_delegate.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "tensorflow/contrib/lite/builtin_op_data.h"
#include "tensorflow/contrib/lite/error_reporter.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/nnapi/NeuralNetworksShim.h"

namespace tflite {

// TODO(aselle): FATAL leaves resources hanging.
void FATAL(const char* format, ...) {
  va_list args;
  va_start(args, format);
  vfprintf(stderr, format, args);
  va_end(args);
  fflush(stderr);
  exit(1);
}

// TODO(aselle): Change the error model to use status codes.
#define CHECK_TFLITE_SUCCESS(x)                       \
  if (x != kTfLiteOk) {                               \
    FATAL("Aborting since tflite returned failure."); \
  }

#define CHECK_NN(x)                                   \
  if (x != ANEURALNETWORKS_NO_ERROR) {                \
    FATAL("Aborting since tflite returned failure."); \
  }

NNAPIAllocation::NNAPIAllocation(const char* filename,
                                 ErrorReporter* error_reporter)
    : MMAPAllocation(filename, error_reporter) {
  if (mmapped_buffer_ != MAP_FAILED)
    CHECK_NN(ANeuralNetworksMemory_createFromFd(buffer_size_bytes_, PROT_READ,
                                                mmap_fd_, 0, &handle_));
}

NNAPIAllocation::~NNAPIAllocation() {
  if (handle_) {
    ANeuralNetworksMemory_free(handle_);
  }
}

NNAPIDelegate::~NNAPIDelegate() {
  if (nn_model_) {
    ANeuralNetworksModel_free(nn_model_);
    nn_model_ = nullptr;
    // TODO(aselle): Is this thread-safe and callable multiple times?
  }
  // ANeuralNetworksShutdown();
}

// Adds the tensors of the interpreter to the NN API model.
// Returns the number of operands added.
uint32_t addTensorOperands(tflite::Interpreter* interpreter,
                           ANeuralNetworksModel* nn_model) {
  uint32_t next_id = 0;
  for (size_t i = 0; i < interpreter->tensors_size(); i++) {
    int32_t nn_type = 0;
    float scale = 1.0f;
    int32_t zeroPoint = 0;
    TfLiteTensor* tensor = interpreter->tensor(i);
    switch (tensor->type) {
      case kTfLiteNoType:
        // Tensors added during initialization of Ops don't have a type yet and
        // should not be registered with the NNAPI.
        continue;
      case kTfLiteFloat32:
        nn_type = ANEURALNETWORKS_TENSOR_FLOAT32;
        break;
      case kTfLiteUInt8:
        nn_type = ANEURALNETWORKS_TENSOR_QUANT8_ASYMM;
        scale = tensor->params.scale;
        zeroPoint = tensor->params.zero_point;
        break;
      case kTfLiteInt32:
        nn_type = ANEURALNETWORKS_TENSOR_INT32;
        scale = tensor->params.scale;
        zeroPoint = tensor->params.zero_point;
        break;
      default:
        FATAL("Unsupported type.");
    }
    // TODO(aselle): Note, many of these are intermediate results. Do I need
    // to ever specify these sizes. I am currently below doing setValue
    // on all of them, but I shouldn't in the future.
    // Answer(jeanluc): If all the operators can set the dimension correctly,
    // you won't need to.
    ANeuralNetworksOperandType operand_type{
        nn_type, static_cast<uint32_t>(tensor->dims->size),
        reinterpret_cast<uint32_t*>(tensor->dims->data), scale, zeroPoint};
    CHECK_NN(ANeuralNetworksModel_addOperand(nn_model, &operand_type));

    // TODO(aselle): Based on Michael's suggestion, limiting this to read
    // only memory
    if (tensor->allocation_type == kTfLiteMmapRo) {
      if (const NNAPIAllocation* alloc = dynamic_cast<const NNAPIAllocation*>(
              static_cast<const Allocation*>(tensor->allocation))) {
        CHECK_NN(ANeuralNetworksModel_setOperandValueFromMemory(
            nn_model, i, alloc->memory(), alloc->offset(tensor->data.raw),
            tensor->bytes));
      } else {
        CHECK_NN(ANeuralNetworksModel_setOperandValue(
            nn_model, i, tensor->data.raw, tensor->bytes));
      }
    }
    ++next_id;
  }
  return next_id;
}

// Adds the operations and their parameters to the NN API model.
// 'next-id' is the operand ID of the next operand of the model.
void AddOpsAndParams(tflite::Interpreter* interpreter,
                     ANeuralNetworksModel* nn_model, uint32_t next_id) {
  for (size_t i = 0; i < interpreter->nodes_size(); i++) {
    const auto* node_and_registration = interpreter->node_and_registration(i);
    const TfLiteNode& node = node_and_registration->first;
    const TfLiteRegistration& registration = node_and_registration->second;
    tflite::BuiltinOperator builtin =
        static_cast<tflite::BuiltinOperator>(registration.builtin_code);

    // Add the parameters.
    std::vector<uint32_t> augmented_inputs(
        node.inputs->data, node.inputs->data + node.inputs->size);

    auto add_scalar_int32 = [&nn_model, &augmented_inputs,
                             &next_id](int value) {
      ANeuralNetworksOperandType operand_type{.type = ANEURALNETWORKS_INT32};
      CHECK_NN(ANeuralNetworksModel_addOperand(nn_model, &operand_type))
      CHECK_NN(ANeuralNetworksModel_setOperandValue(nn_model, next_id, &value,
                                                    sizeof(int32_t)))
      augmented_inputs.push_back(next_id++);
    };

    auto add_scalar_float32 = [&nn_model, &augmented_inputs,
                               &next_id](float value) {
      ANeuralNetworksOperandType operand_type{.type = ANEURALNETWORKS_FLOAT32};
      CHECK_NN(ANeuralNetworksModel_addOperand(nn_model, &operand_type))
      CHECK_NN(ANeuralNetworksModel_setOperandValue(nn_model, next_id, &value,
                                                    sizeof(float)))
      augmented_inputs.push_back(next_id++);
    };

    auto duplicate_state_tensor_float32 =
        [interpreter, &nn_model, &augmented_inputs, &next_id](int tensor_id) {
          const TfLiteTensor* tensor = interpreter->tensor(tensor_id);
          CHECK_NN(ANeuralNetworksModel_setOperandValue(
              nn_model, tensor_id, tensor->data.raw, tensor->bytes));
          augmented_inputs.push_back(tensor_id);
        };

    auto add_add_params = [&add_scalar_int32]() { add_scalar_int32(0); };

    auto add_pooling_params = [&add_scalar_int32](void* data) {
      auto builtin = reinterpret_cast<TfLitePoolParams*>(data);
      add_scalar_int32(builtin->padding);
      add_scalar_int32(builtin->stride_width);
      add_scalar_int32(builtin->stride_height);
      add_scalar_int32(builtin->filter_width);
      add_scalar_int32(builtin->filter_height);
      add_scalar_int32(builtin->activation);
    };

    auto add_convolution_params = [&add_scalar_int32](void* data) {
      auto builtin = reinterpret_cast<TfLiteConvParams*>(data);
      add_scalar_int32(builtin->padding);
      add_scalar_int32(builtin->stride_width);
      add_scalar_int32(builtin->stride_height);
      add_scalar_int32(builtin->activation);
    };

    auto add_depthwise_conv_params = [&add_scalar_int32](void* data) {
      auto builtin = reinterpret_cast<TfLiteDepthwiseConvParams*>(data);
      add_scalar_int32(builtin->padding);
      add_scalar_int32(builtin->stride_width);
      add_scalar_int32(builtin->stride_height);
      add_scalar_int32(builtin->depth_multiplier);
      add_scalar_int32(builtin->activation);
    };

    auto add_fully_connected_params = [&add_scalar_int32](void* data) {
      auto builtin = reinterpret_cast<TfLiteFullyConnectedParams*>(data);
      add_scalar_int32(builtin->activation);
    };

    auto add_concatenation_params = [&add_scalar_int32](void* data) {
      auto builtin = reinterpret_cast<TfLiteConcatenationParams*>(data);
      add_scalar_int32(builtin->axis);
      if (builtin->activation != kTfLiteActNone) {
        FATAL("Concatenation does not support fused activation in NNAPI");
      }
    };

    auto add_softmax_params = [&add_scalar_float32](void* data) {
      auto builtin = reinterpret_cast<TfLiteSoftmaxParams*>(data);
      add_scalar_float32(builtin->beta);
    };

    auto add_space_to_depth_params = [&add_scalar_int32](void* data) {
      auto builtin = reinterpret_cast<TfLiteSpaceToDepthParams*>(data);
      add_scalar_int32(builtin->block_size);
    };

    auto add_lstm_params = [&add_scalar_int32,
                            &add_scalar_float32](void* data) {
      auto builtin = reinterpret_cast<TfLiteLSTMParams*>(data);
      add_scalar_int32(builtin->activation);
      add_scalar_float32(builtin->cell_clip);
      add_scalar_float32(builtin->proj_clip);
    };

#if 0
    auto add_reshape_params = [&](void* data) {
      auto builtin = reinterpret_cast<TfLiteReshapeParams*>(data);
      uint32_t tensor_size_shape = builtin->num_dimensions;
      ANeuralNetworksOperandType operand_type{
          ANEURALNETWORKS_TENSOR_INT32,
          {static_cast<uint32_t>(1),
           reinterpret_cast<uint32_t*>(&tensor_size_shape)},
          0,
          0};
      CHECK_NN(ANeuralNetworksModel_addOperand(nn_model, &operand_type))
      CHECK_NN(ANeuralNetworksModel_setOperandValue(
          nn_model, next_id, builtin->shape,
          sizeof(int) * builtin->num_dimensions));
      augmented_inputs.push_back(next_id++);
    };
#endif

    ANeuralNetworksOperationType nn_op_type;
    switch (builtin) {
      case tflite::BuiltinOperator_ADD:
        nn_op_type = ANEURALNETWORKS_ADD;
        add_add_params();
        break;
      case tflite::BuiltinOperator_AVERAGE_POOL_2D:
        add_pooling_params(node.builtin_data);
        nn_op_type = ANEURALNETWORKS_AVERAGE_POOL_2D;
        break;
      case tflite::BuiltinOperator_MAX_POOL_2D:
        add_pooling_params(node.builtin_data);
        nn_op_type = ANEURALNETWORKS_MAX_POOL_2D;
        break;
      case tflite::BuiltinOperator_L2_POOL_2D:
        add_pooling_params(node.builtin_data);
        nn_op_type = ANEURALNETWORKS_L2_POOL_2D;
        break;
      case tflite::BuiltinOperator_CONV_2D:
        add_convolution_params(node.builtin_data);
        nn_op_type = ANEURALNETWORKS_CONV_2D;
        break;
      case tflite::BuiltinOperator_RELU:
        nn_op_type = ANEURALNETWORKS_RELU;
        break;
      case tflite::BuiltinOperator_RELU6:
        nn_op_type = ANEURALNETWORKS_RELU6;
        break;
      case tflite::BuiltinOperator_TANH:
        nn_op_type = ANEURALNETWORKS_TANH;
        break;
      case tflite::BuiltinOperator_LOGISTIC:
        nn_op_type = ANEURALNETWORKS_LOGISTIC;
        break;
      case tflite::BuiltinOperator_DEPTHWISE_CONV_2D:
        add_depthwise_conv_params(node.builtin_data);
        nn_op_type = ANEURALNETWORKS_DEPTHWISE_CONV_2D;
        break;
      case tflite::BuiltinOperator_CONCATENATION:
        add_concatenation_params(node.builtin_data);
        nn_op_type = ANEURALNETWORKS_CONCATENATION;
        break;
      case tflite::BuiltinOperator_SOFTMAX:
        add_softmax_params(node.builtin_data);
        nn_op_type = ANEURALNETWORKS_SOFTMAX;
        break;
      case tflite::BuiltinOperator_FULLY_CONNECTED:
        add_fully_connected_params(node.builtin_data);
        nn_op_type = ANEURALNETWORKS_FULLY_CONNECTED;
        break;
      case tflite::BuiltinOperator_RESHAPE:
        nn_op_type = ANEURALNETWORKS_RESHAPE;
        // add_reshape_params(node.builtin_data);
        break;
      case tflite::BuiltinOperator_SPACE_TO_DEPTH:
        add_space_to_depth_params(node.builtin_data);
        nn_op_type = ANEURALNETWORKS_SPACE_TO_DEPTH;
        break;
      case tflite::BuiltinOperator_LSTM: {
        duplicate_state_tensor_float32(
            node.outputs->data[/*kOutputStateTensor*/ 1]);
        duplicate_state_tensor_float32(
            node.outputs->data[/*kCellStateTensor*/ 2]);
        add_lstm_params(node.builtin_data);
        nn_op_type = ANEURALNETWORKS_LSTM;
        break;
      }
      case tflite::BuiltinOperator_CONCAT_EMBEDDINGS:
      case tflite::BuiltinOperator_LSH_PROJECTION:
      case tflite::BuiltinOperator_SVDF:
      case tflite::BuiltinOperator_HASHTABLE_LOOKUP:
      case tflite::BuiltinOperator_RNN:
      case tflite::BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_RNN:
      case tflite::BuiltinOperator_EMBEDDING_LOOKUP:
      case tflite::BuiltinOperator_EMBEDDING_LOOKUP_SPARSE:
      case tflite::BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM:
      case tflite::BuiltinOperator_L2_NORMALIZATION:
      case tflite::BuiltinOperator_LOCAL_RESPONSE_NORMALIZATION:
      case tflite::BuiltinOperator_MUL:
      case tflite::BuiltinOperator_PAD:
      case tflite::BuiltinOperator_RESIZE_BILINEAR:
      case tflite::BuiltinOperator_CALL:
      case tflite::BuiltinOperator_SKIP_GRAM:
      case tflite::BuiltinOperator_RELU_N1_TO_1:
      case tflite::BuiltinOperator_GATHER:
      case tflite::BuiltinOperator_SPACE_TO_BATCH_ND:
      case tflite::BuiltinOperator_BATCH_TO_SPACE_ND:
      case tflite::BuiltinOperator_TRANSPOSE:
      case tflite::BuiltinOperator_MEAN:
      case tflite::BuiltinOperator_DIV:
      case tflite::BuiltinOperator_SUB:
      case tflite::BuiltinOperator_SQUEEZE:
      case tflite::BuiltinOperator_STRIDED_SLICE:
        FATAL("Op code %d is currently not delegated to NNAPI", builtin);
        nn_op_type = -1;  // set to invalid
        break;
      case tflite::BuiltinOperator_CUSTOM:
        FATAL("Custom operations are not supported when using NNAPI.");
        nn_op_type = -1;  // set to invalid
        break;
    }

    // Add the operation.
    CHECK_NN(ANeuralNetworksModel_addOperation(
        nn_model, nn_op_type, static_cast<uint32_t>(augmented_inputs.size()),
        augmented_inputs.data(), static_cast<uint32_t>(node.outputs->size),
        reinterpret_cast<uint32_t*>(node.outputs->data)));
  }
}

TfLiteStatus NNAPIDelegate::BuildGraph(Interpreter* interpreter) {
  // TODO(aselle): This is not correct. need to handle resize invalidation.
  if (nn_model_ && nn_compiled_model_) return kTfLiteOk;

  if (!nn_model_) {
    CHECK_NN(ANeuralNetworksModel_create(&nn_model_));

    uint32_t next_id = addTensorOperands(interpreter, nn_model_);
    AddOpsAndParams(interpreter, nn_model_, next_id);
    CHECK_NN(ANeuralNetworksModel_identifyInputsAndOutputs(
        nn_model_, static_cast<uint32_t>(interpreter->inputs().size()),
        reinterpret_cast<const uint32_t*>(interpreter->inputs().data()),
        static_cast<uint32_t>(interpreter->outputs().size()),
        reinterpret_cast<const uint32_t*>(interpreter->outputs().data())));
    CHECK_NN(ANeuralNetworksModel_finish(nn_model_));
  }
  if (!nn_compiled_model_) {
    CHECK_NN(ANeuralNetworksCompilation_create(nn_model_, &nn_compiled_model_));
    CHECK_NN(ANeuralNetworksCompilation_finish(nn_compiled_model_));
  }
  return kTfLiteOk;
}

TfLiteStatus NNAPIDelegate::Invoke(Interpreter* interpreter) {
  if (!nn_model_) {
    TF_LITE_ENSURE_STATUS(BuildGraph(interpreter));
  }

  ANeuralNetworksExecution* execution = nullptr;
  CHECK_NN(ANeuralNetworksExecution_create(nn_compiled_model_, &execution));

  // Currently perform deep copy of input buffer
  for (size_t i = 0; i < interpreter->inputs().size(); i++) {
    int input = interpreter->inputs()[i];
    // TODO(aselle): Is this what we want or do we want input instead?
    // TODO(aselle): This should be called setInputValue maybe to be cons.
    TfLiteTensor* tensor = interpreter->tensor(input);
    CHECK_NN(ANeuralNetworksExecution_setInput(
        execution, i, nullptr, tensor->data.raw, tensor->bytes));
  }
  // Tell nn api where to place final data.
  for (size_t i = 0; i < interpreter->outputs().size(); i++) {
    int output = interpreter->outputs()[i];
    TfLiteTensor* tensor = interpreter->tensor(output);
    CHECK_NN(ANeuralNetworksExecution_setOutput(
        execution, i, nullptr, tensor->data.raw, tensor->bytes));
  }
  // Currently use blocking compute.
  ANeuralNetworksEvent* event = nullptr;
  CHECK_NN(ANeuralNetworksExecution_startCompute(execution, &event));
  CHECK_NN(ANeuralNetworksEvent_wait(event));
  ANeuralNetworksEvent_free(event);
  ANeuralNetworksExecution_free(execution);

#if 0
  printf("From the NN API:\n");
  TfLiteTensor* tensor = interpreter->tensor(interpreter->outputs()[0]);
  if (float* data =
          interpreter->typed_tensor<float>(interpreter->outputs()[0])) {
    size_t num = tensor->bytes / sizeof(float);
    for (float* p = data; p < data + num; p++) {
      printf(" %f", *p);
    }
    printf("\n");
  }
#endif

  return kTfLiteOk;
}

}  // namespace tflite
