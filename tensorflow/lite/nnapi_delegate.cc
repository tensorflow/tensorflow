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

#include "tensorflow/lite/nnapi_delegate.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/nnapi/nnapi_implementation.h"

#ifdef __ANDROID__
#include <android/log.h>
#include <sys/system_properties.h>
#endif

namespace tflite {

void logError(const char* format, ...) {
  // stderr is convenient for native tests, but is not captured for apps
  va_list args_for_stderr;
  va_start(args_for_stderr, format);
  vfprintf(stderr, format, args_for_stderr);
  va_end(args_for_stderr);
  fprintf(stderr, "\n");
  fflush(stderr);
#ifdef __ANDROID__
  // produce logcat output for general consumption
  va_list args_for_log;
  va_start(args_for_log, format);
  __android_log_vprint(ANDROID_LOG_ERROR, "tflite", format, args_for_log);
  va_end(args_for_log);
#endif
}

#define FATAL(...)       \
  logError(__VA_ARGS__); \
  exit(1);

// TODO(aselle): Change the error model to use status codes.
#define CHECK_TFLITE_SUCCESS(x)                                           \
  if (x != kTfLiteOk) {                                                   \
    FATAL("Aborting since tflite returned failure nnapi_delegate.cc:%d.", \
          __LINE__);                                                      \
  }

#define CHECK_NN(x)                                                     \
  if (x != ANEURALNETWORKS_NO_ERROR) {                                  \
    FATAL("Aborting since NNAPI returned failure nnapi_delegate.cc:%d", \
          __LINE__);                                                    \
  }

#define RETURN_ERROR_IF_TFLITE_FAILED(x)                                       \
  if (x != kTfLiteOk) {                                                        \
    logError(                                                                  \
        "Returning error since TFLite returned failure nnapi_delegate.cc:%d.", \
        __LINE__);                                                             \
    return kTfLiteError;                                                       \
  }

#define RETURN_ERROR_IF_NN_FAILED(x)                                          \
  if (x != ANEURALNETWORKS_NO_ERROR) {                                        \
    logError(                                                                 \
        "Returning error since NNAPI returned failure nnapi_delegate.cc:%d.", \
        __LINE__);                                                            \
    return kTfLiteError;                                                      \
  }

// Tracking of NNAPI operand ids
static const int64_t kOperandIdNotSet = -1;
static const int64_t kOperandNotNeeded = -2;

NNAPIAllocation::NNAPIAllocation(const char* filename,
                                 ErrorReporter* error_reporter)
    : MMAPAllocation(filename, error_reporter) {
  if (mmapped_buffer_ != MAP_FAILED)
    CHECK_NN(NnApiImplementation()->ANeuralNetworksMemory_createFromFd(
        buffer_size_bytes_, PROT_READ, mmap_fd_, 0, &handle_));
}

NNAPIAllocation::~NNAPIAllocation() {
  if (handle_) {
    NnApiImplementation()->ANeuralNetworksMemory_free(handle_);
  }
}

NNAPIDelegate::~NNAPIDelegate() {
  if (nn_compiled_model_) {
    NnApiImplementation()->ANeuralNetworksCompilation_free(nn_compiled_model_);
    nn_compiled_model_ = nullptr;
  }
  if (nn_model_) {
    NnApiImplementation()->ANeuralNetworksModel_free(nn_model_);
    nn_model_ = nullptr;
    // TODO(aselle): Is this thread-safe and callable multiple times?
  }
  // ANeuralNetworksShutdown();
}

// Adds the tensors of the subgraph to the NN API model.
TfLiteStatus addTensorOperands(tflite::Subgraph* subgraph,
                               ANeuralNetworksModel* nn_model,
                               uint32_t* no_of_operands_added,
                               std::vector<int64_t>* nnapi_ids) {
  const NnApi* nnapi = NnApiImplementation();
  uint32_t next_id = 0;
  for (size_t i = 0; i < subgraph->tensors_size(); i++) {
    // Skip temporaries and RNN back-edges.
    if ((*nnapi_ids)[i] == kOperandNotNeeded) continue;

    (*nnapi_ids)[i] = int64_t(next_id);

    int32_t nn_type = 0;
    // NNAPI requires 32-bit float scale to be zero, tflite doesn't care
    float scale = 0.0f;
    int32_t zeroPoint = 0;
    TfLiteTensor* tensor = subgraph->tensor(i);
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
        logError("Unsupported tensor type %d", tensor->type);
        return kTfLiteError;
    }
    if (tensor->dims->size == 0) {
      logError("NNAPI doesn't support tensors with rank 0 (index %d name %s)",
               i, tensor->name);
      return kTfLiteError;
    }
    if (tensor->dims->size > 4) {
      logError("NNAPI doesn't support tensors with rank > 4 (index %d name %s)",
               i, tensor->name);
      return kTfLiteError;
    }
    // TODO(aselle): Note, many of these are intermediate results. Do I need
    // to ever specify these sizes. I am currently below doing setValue
    // on all of them, but I shouldn't in the future.
    // Answer(jeanluc): If all the operators can set the dimension correctly,
    // you won't need to.
    ANeuralNetworksOperandType operand_type{
        nn_type, static_cast<uint32_t>(tensor->dims->size),
        reinterpret_cast<uint32_t*>(tensor->dims->data), scale, zeroPoint};
    RETURN_ERROR_IF_NN_FAILED(
        nnapi->ANeuralNetworksModel_addOperand(nn_model, &operand_type));
    // TODO(aselle): Based on Michael's suggestion, limiting this to read
    // only memory
    if (tensor->allocation_type == kTfLiteMmapRo) {
      if (const NNAPIAllocation* alloc = dynamic_cast<const NNAPIAllocation*>(
              static_cast<const Allocation*>(tensor->allocation))) {
        RETURN_ERROR_IF_NN_FAILED(
            nnapi->ANeuralNetworksModel_setOperandValueFromMemory(
                nn_model, next_id, alloc->memory(),
                alloc->offset(tensor->data.raw), tensor->bytes));
      } else {
        RETURN_ERROR_IF_NN_FAILED(nnapi->ANeuralNetworksModel_setOperandValue(
            nn_model, next_id, tensor->data.raw, tensor->bytes));
      }
    } else if (tensor->bytes == 0) {
      // These size 0 tensors are optional tensors reserved.
      RETURN_ERROR_IF_NN_FAILED(nnapi->ANeuralNetworksModel_setOperandValue(
          nn_model, next_id, nullptr, 0));
    }

    ++next_id;
  }
  *no_of_operands_added = next_id;
  return kTfLiteOk;
}

void MapAndAddTensorIds(const int* from_ids_buf, size_t from_ids_count,
                        std::vector<uint32_t>* into,
                        const std::vector<int64_t>& map) {
  for (size_t i = 0; i < from_ids_count; i++) {
    int from_id = from_ids_buf[i];
    if (from_id == kOptionalTensor) {
      into->push_back(from_id);
    } else {
      into->push_back(map[from_id]);
    }
  }
}

// Adds the operations and their parameters to the NN API model.
// 'next-id' is the operand ID of the next operand of the model.
TfLiteStatus AddOpsAndParams(
    tflite::Subgraph* subgraph, ANeuralNetworksModel* nn_model,
    uint32_t next_id, std::vector<int>* model_state_inputs,
    std::vector<int>* model_state_outputs,
    const std::vector<int64_t>& tensor_id_to_nnapi_id) {
  const NnApi* nnapi = NnApiImplementation();
  for (size_t i = 0; i < subgraph->nodes_size(); i++) {
    const auto* node_and_registration = subgraph->node_and_registration(i);
    const TfLiteNode& node = node_and_registration->first;
    const TfLiteRegistration& registration = node_and_registration->second;
    tflite::BuiltinOperator builtin =
        static_cast<tflite::BuiltinOperator>(registration.builtin_code);

    // Add the parameters.
    std::vector<uint32_t> augmented_inputs, augmented_outputs;
    MapAndAddTensorIds(node.inputs->data, node.inputs->size, &augmented_inputs,
                       tensor_id_to_nnapi_id);
    MapAndAddTensorIds(node.outputs->data, node.outputs->size,
                       &augmented_outputs, tensor_id_to_nnapi_id);

    auto add_scalar_int32 = [nnapi, &nn_model, &augmented_inputs,
                             &next_id](int value) {
      ANeuralNetworksOperandType operand_type{.type = ANEURALNETWORKS_INT32};
      CHECK_NN(nnapi->ANeuralNetworksModel_addOperand(nn_model, &operand_type))
      CHECK_NN(nnapi->ANeuralNetworksModel_setOperandValue(
          nn_model, next_id, &value, sizeof(int32_t)))
      augmented_inputs.push_back(next_id++);
    };

    auto add_scalar_float32 = [nnapi, &nn_model, &augmented_inputs,
                               &next_id](float value) {
      ANeuralNetworksOperandType operand_type{.type = ANEURALNETWORKS_FLOAT32};
      CHECK_NN(nnapi->ANeuralNetworksModel_addOperand(nn_model, &operand_type))
      CHECK_NN(nnapi->ANeuralNetworksModel_setOperandValue(
          nn_model, next_id, &value, sizeof(float)))
      augmented_inputs.push_back(next_id++);
    };

    auto add_vector_int32 = [&](const int* values, uint32_t num_values) {
      ANeuralNetworksOperandType operand_type{
          .type = ANEURALNETWORKS_TENSOR_INT32,
          .dimensionCount = 1,
          .dimensions = &num_values};
      CHECK_NN(nnapi->ANeuralNetworksModel_addOperand(nn_model, &operand_type))
      CHECK_NN(nnapi->ANeuralNetworksModel_setOperandValue(
          nn_model, next_id, values, sizeof(int32_t) * num_values));
      augmented_inputs.push_back(next_id++);
    };

    // Handle state tensors of RNN, LSTM, SVDF.
    // For each state_out tensor, a corresponding state_in operand needs to be
    // created for NNAPI.
    auto duplicate_state_tensor_float32 =
        [nnapi, subgraph, &nn_model, &next_id, &augmented_inputs,
         &model_state_inputs, &model_state_outputs](int tensor_id) {
          const TfLiteTensor* tensor = subgraph->tensor(tensor_id);
          ANeuralNetworksOperandType operand_type{
              ANEURALNETWORKS_TENSOR_FLOAT32,
              static_cast<uint32_t>(tensor->dims->size),
              reinterpret_cast<uint32_t*>(tensor->dims->data),
              tensor->params.scale, tensor->params.zero_point};
          CHECK_NN(
              nnapi->ANeuralNetworksModel_addOperand(nn_model, &operand_type));
          augmented_inputs.push_back(next_id);
          model_state_inputs->push_back(next_id);
          model_state_outputs->push_back(tensor_id);
          next_id++;
        };
    auto check_and_add_activation = [&add_scalar_int32](int activation) {
      if (activation > kTfLiteActRelu6) {
        logError("NNAPI only supports RELU, RELU1 and RELU6 activations");
        return kTfLiteError;
      }
      add_scalar_int32(activation);
      return kTfLiteOk;
    };

    auto add_add_params = [&add_scalar_int32](void* data) {
      auto* builtin = reinterpret_cast<TfLiteAddParams*>(data);
      if (builtin->activation > kTfLiteActRelu6) {
        logError("NNAPI only supports RELU, RELU1 and RELU6 activations");
        return kTfLiteError;
      }
      add_scalar_int32(builtin->activation);
      return kTfLiteOk;
    };

    auto add_pooling_params = [&add_scalar_int32,
                               &check_and_add_activation](void* data) {
      auto builtin = reinterpret_cast<TfLitePoolParams*>(data);
      add_scalar_int32(builtin->padding);
      add_scalar_int32(builtin->stride_width);
      add_scalar_int32(builtin->stride_height);
      add_scalar_int32(builtin->filter_width);
      add_scalar_int32(builtin->filter_height);
      return check_and_add_activation(builtin->activation);
    };

    auto add_convolution_params = [&add_scalar_int32,
                                   &check_and_add_activation](void* data) {
      auto builtin = reinterpret_cast<TfLiteConvParams*>(data);
      add_scalar_int32(builtin->padding);
      add_scalar_int32(builtin->stride_width);
      add_scalar_int32(builtin->stride_height);
      return check_and_add_activation(builtin->activation);
    };

    auto add_depthwise_conv_params = [&add_scalar_int32,
                                      &check_and_add_activation](void* data) {
      auto builtin = reinterpret_cast<TfLiteDepthwiseConvParams*>(data);
      add_scalar_int32(builtin->padding);
      add_scalar_int32(builtin->stride_width);
      add_scalar_int32(builtin->stride_height);
      add_scalar_int32(builtin->depth_multiplier);
      return check_and_add_activation(builtin->activation);
    };

    auto add_fully_connected_params = [&check_and_add_activation](void* data) {
      auto builtin = reinterpret_cast<TfLiteFullyConnectedParams*>(data);
      return check_and_add_activation(builtin->activation);
    };

    auto add_concatenation_params = [&add_scalar_int32](void* data) {
      auto builtin = reinterpret_cast<TfLiteConcatenationParams*>(data);
      add_scalar_int32(builtin->axis);
      if (builtin->activation != kTfLiteActNone) {
        logError("Concatenation does not support fused activation in NNAPI");
        return kTfLiteError;
      }
      return kTfLiteOk;
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

    // LSTM in NNAPI requires scratch tensor as an output operand.
    auto add_lstm_scratch_tensor_float32 = [nnapi, subgraph, &node, &nn_model,
                                            &next_id, &augmented_outputs]() {
      if (node.temporaries->size == 0) return;
      int scratch_buffer_index = node.temporaries->data[0];
      const TfLiteTensor* tensor = subgraph->tensor(scratch_buffer_index);
      ANeuralNetworksOperandType operand_type{
          ANEURALNETWORKS_TENSOR_FLOAT32,
          static_cast<uint32_t>(tensor->dims->size),
          reinterpret_cast<uint32_t*>(tensor->dims->data), tensor->params.scale,
          tensor->params.zero_point};
      CHECK_NN(nnapi->ANeuralNetworksModel_addOperand(nn_model, &operand_type));
      augmented_outputs.insert(augmented_outputs.begin(), next_id++);
    };

    auto add_mean_params = [&add_scalar_int32](void* data) {
      auto builtin = reinterpret_cast<TfLiteReducerParams*>(data);
      add_scalar_int32(builtin->keep_dims);
    };

    auto add_svdf_params = [&add_scalar_int32](void* data) {
      auto builtin = reinterpret_cast<TfLiteSVDFParams*>(data);
      add_scalar_int32(builtin->rank);
      add_scalar_int32(builtin->activation);
    };

    auto add_rnn_params = [&add_scalar_int32](void* data) {
      auto builtin = reinterpret_cast<TfLiteRNNParams*>(data);
      add_scalar_int32(builtin->activation);
    };

    auto add_squeeze_params = [&](void* data) {
      const auto* builtin = reinterpret_cast<TfLiteSqueezeParams*>(data);
      // Note that we add the squeeze dimensions even if the dimensions were
      // unspecified (empty), as NNAPI requires the operand.
      add_vector_int32(builtin->squeeze_dims,
                       static_cast<uint32_t>(builtin->num_squeeze_dims));
    };

    // Handle optional input tensors.
    auto add_optional_tensors = [nnapi, &nn_model, &augmented_inputs,
                                 &next_id](int nn_type) {
      for (size_t idx = 0; idx < augmented_inputs.size(); idx++) {
        if (augmented_inputs[idx] == kOptionalTensor) {
          const std::vector<uint32_t> dim = {0, 0};
          ANeuralNetworksOperandType operand_type{nn_type, 2, dim.data(), 0, 0};
          CHECK_NN(
              nnapi->ANeuralNetworksModel_addOperand(nn_model, &operand_type))
          CHECK_NN(nnapi->ANeuralNetworksModel_setOperandValue(
              nn_model, next_id, nullptr, 0))
          augmented_inputs[idx] = next_id++;
        }
      }
    };

    int nnapi_version = 10;
    ANeuralNetworksOperationType nn_op_type;

    switch (builtin) {
      case tflite::BuiltinOperator_ADD:
        nn_op_type = ANEURALNETWORKS_ADD;
        RETURN_ERROR_IF_TFLITE_FAILED(add_add_params(node.builtin_data));
        break;
      case tflite::BuiltinOperator_MUL:
        nn_op_type = ANEURALNETWORKS_MUL;
        RETURN_ERROR_IF_TFLITE_FAILED(add_add_params(node.builtin_data));
        break;
      case tflite::BuiltinOperator_AVERAGE_POOL_2D:
        RETURN_ERROR_IF_TFLITE_FAILED(add_pooling_params(node.builtin_data));
        nn_op_type = ANEURALNETWORKS_AVERAGE_POOL_2D;
        break;
      case tflite::BuiltinOperator_MAX_POOL_2D:
        RETURN_ERROR_IF_TFLITE_FAILED(add_pooling_params(node.builtin_data));
        nn_op_type = ANEURALNETWORKS_MAX_POOL_2D;
        break;
      case tflite::BuiltinOperator_L2_POOL_2D:
        RETURN_ERROR_IF_TFLITE_FAILED(add_pooling_params(node.builtin_data));
        nn_op_type = ANEURALNETWORKS_L2_POOL_2D;
        break;
      case tflite::BuiltinOperator_CONV_2D: {
        auto builtin = reinterpret_cast<TfLiteConvParams*>(node.builtin_data);
        if (builtin->dilation_width_factor != 1 ||
            builtin->dilation_height_factor != 1 || node.inputs->size != 3) {
          logError("NNAPI does not support dilated Conv2D.");
          return kTfLiteError;
        }
      }
        RETURN_ERROR_IF_TFLITE_FAILED(
            add_convolution_params(node.builtin_data));
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
      case tflite::BuiltinOperator_FLOOR:
        nn_op_type = ANEURALNETWORKS_FLOOR;
        break;
      case tflite::BuiltinOperator_LOGISTIC:
        nn_op_type = ANEURALNETWORKS_LOGISTIC;
        break;
      case tflite::BuiltinOperator_DEPTHWISE_CONV_2D:
        RETURN_ERROR_IF_TFLITE_FAILED(
            add_depthwise_conv_params(node.builtin_data));
        nn_op_type = ANEURALNETWORKS_DEPTHWISE_CONV_2D;
        break;
      case tflite::BuiltinOperator_CONCATENATION:
        RETURN_ERROR_IF_TFLITE_FAILED(
            add_concatenation_params(node.builtin_data));
        nn_op_type = ANEURALNETWORKS_CONCATENATION;
        break;
      case tflite::BuiltinOperator_SOFTMAX:
        add_softmax_params(node.builtin_data);
        nn_op_type = ANEURALNETWORKS_SOFTMAX;
        break;
      case tflite::BuiltinOperator_FULLY_CONNECTED:
        RETURN_ERROR_IF_TFLITE_FAILED(
            add_fully_connected_params(node.builtin_data));
        nn_op_type = ANEURALNETWORKS_FULLY_CONNECTED;
        break;
      case tflite::BuiltinOperator_RESHAPE:
        if (node.inputs->size != 2) {
          logError("NNAPI only supports 2-input RESHAPE");
          return kTfLiteError;
        }
        nn_op_type = ANEURALNETWORKS_RESHAPE;
        // add_reshape_params(node.builtin_data);
        break;
      case tflite::BuiltinOperator_SPACE_TO_DEPTH:
        add_space_to_depth_params(node.builtin_data);
        nn_op_type = ANEURALNETWORKS_SPACE_TO_DEPTH;
        break;
      case tflite::BuiltinOperator_LSTM: {
        if (node.inputs->size + /* no of params */ 3 != 21) {
          logError("NNAPI only supports 21-input LSTMs");
          return kTfLiteError;
        }
        duplicate_state_tensor_float32(
            node.outputs->data[/*kOutputStateTensor*/ 0]);
        duplicate_state_tensor_float32(
            node.outputs->data[/*kCellStateTensor*/ 1]);
        add_lstm_params(node.builtin_data);
        add_lstm_scratch_tensor_float32();
        add_optional_tensors(ANEURALNETWORKS_TENSOR_FLOAT32);
        nn_op_type = ANEURALNETWORKS_LSTM;
        break;
      }
      case tflite::BuiltinOperator_SVDF: {
        duplicate_state_tensor_float32(node.outputs->data[/*kStateTensor*/ 0]);
        add_svdf_params(node.builtin_data);
        nn_op_type = ANEURALNETWORKS_SVDF;
        break;
      }
      case tflite::BuiltinOperator_RNN: {
        duplicate_state_tensor_float32(
            node.outputs->data[/*kHiddenStateTensor*/ 0]);
        add_rnn_params(node.builtin_data);
        nn_op_type = ANEURALNETWORKS_RNN;
        break;
      }
      case tflite::BuiltinOperator_EMBEDDING_LOOKUP:
        nn_op_type = ANEURALNETWORKS_EMBEDDING_LOOKUP;
        break;
      case tflite::BuiltinOperator_PAD:
        nnapi_version = 11;  // require NNAPI 1.1
        nn_op_type = ANEURALNETWORKS_PAD;
        break;
      case tflite::BuiltinOperator_MEAN:
        nnapi_version = 11;  // require NNAPI 1.1
        add_mean_params(node.builtin_data);
        nn_op_type = ANEURALNETWORKS_MEAN;
        break;
      case tflite::BuiltinOperator_DIV:
        nnapi_version = 11;  // require NNAPI 1.1
        nn_op_type = ANEURALNETWORKS_DIV;
        RETURN_ERROR_IF_TFLITE_FAILED(check_and_add_activation(
            reinterpret_cast<TfLiteDivParams*>(node.builtin_data)->activation));
        break;
      case tflite::BuiltinOperator_SUB:
        nnapi_version = 11;  // require NNAPI 1.1
        nn_op_type = ANEURALNETWORKS_SUB;
        RETURN_ERROR_IF_TFLITE_FAILED(check_and_add_activation(
            reinterpret_cast<TfLiteSubParams*>(node.builtin_data)->activation));
        break;
      case tflite::BuiltinOperator_SQUEEZE:
        nnapi_version = 11;  // requires NNAPI 1.1
        add_squeeze_params(node.builtin_data);
        nn_op_type = ANEURALNETWORKS_SQUEEZE;
        break;
      case tflite::BuiltinOperator_TRANSPOSE:
        // The permutation input tensor value dictates the output dimensions.
        // TODO(b/110888333): Support dynamically-sized tensors in delegates.
        if ((node.inputs->size > 1) &&
            (subgraph->tensor(node.inputs->data[1])->allocation_type !=
             kTfLiteMmapRo)) {
          logError("NNAPI does not yet support dynamic tensors.");
          return kTfLiteError;
        }
        nnapi_version = 11;  // require NNAPI 1.1
        nn_op_type = ANEURALNETWORKS_TRANSPOSE;
        break;
      case tflite::BuiltinOperator_L2_NORMALIZATION:
        nn_op_type = ANEURALNETWORKS_L2_NORMALIZATION;
        if (reinterpret_cast<TfLiteL2NormParams*>(node.builtin_data)
                ->activation != kTfLiteActNone) {
          logError(
              "NNAPI does not support L2Normalization with fused activations");
          return kTfLiteError;
        }
        if ((node.inputs->size > 0) &&
            (subgraph->tensor(node.inputs->data[0])->dims->size != 4)) {
          logError("NNAPI only supports input rank 4 for L2Normalization");
          return kTfLiteError;
        }
        break;
      case tflite::BuiltinOperator_HASHTABLE_LOOKUP:
        if (subgraph->tensor(node.outputs->data[0])->type != kTfLiteFloat32) {
          logError("NNAPI only support HASHTABLE_LOOKUP with float32 output",
                   builtin);
          return kTfLiteError;
        }
        nn_op_type = ANEURALNETWORKS_HASHTABLE_LOOKUP;
        break;
      case tflite::BuiltinOperator_CONCAT_EMBEDDINGS:
      case tflite::BuiltinOperator_LSH_PROJECTION:
      case tflite::BuiltinOperator_BIDIRECTIONAL_SEQUENCE_RNN:
      case tflite::BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_RNN:
      case tflite::BuiltinOperator_EMBEDDING_LOOKUP_SPARSE:
      case tflite::BuiltinOperator_BIDIRECTIONAL_SEQUENCE_LSTM:
      case tflite::BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM:
      case tflite::BuiltinOperator_LOCAL_RESPONSE_NORMALIZATION:
      case tflite::BuiltinOperator_PADV2:
      case tflite::BuiltinOperator_RESIZE_BILINEAR:
      case tflite::BuiltinOperator_RESIZE_NEAREST_NEIGHBOR:
      case tflite::BuiltinOperator_CALL:
      case tflite::BuiltinOperator_SKIP_GRAM:
      case tflite::BuiltinOperator_RELU_N1_TO_1:
      case tflite::BuiltinOperator_GATHER:
      case tflite::BuiltinOperator_SPACE_TO_BATCH_ND:
      case tflite::BuiltinOperator_BATCH_TO_SPACE_ND:
      case tflite::BuiltinOperator_TOPK_V2:
      case tflite::BuiltinOperator_SPLIT:
      case tflite::BuiltinOperator_STRIDED_SLICE:
      case tflite::BuiltinOperator_EXP:
      case tflite::BuiltinOperator_COS:
      case tflite::BuiltinOperator_LOG_SOFTMAX:
      case tflite::BuiltinOperator_DEQUANTIZE:
      case tflite::BuiltinOperator_DELEGATE:
      case tflite::BuiltinOperator_CAST:
      case tflite::BuiltinOperator_PRELU:
      case tflite::BuiltinOperator_MAXIMUM:
      case tflite::BuiltinOperator_MINIMUM:
      case tflite::BuiltinOperator_ARG_MAX:
      case tflite::BuiltinOperator_ARG_MIN:
      case tflite::BuiltinOperator_GREATER:
      case tflite::BuiltinOperator_GREATER_EQUAL:
      case tflite::BuiltinOperator_LESS:
      case tflite::BuiltinOperator_LESS_EQUAL:
      case tflite::BuiltinOperator_NEG:
      case tflite::BuiltinOperator_SELECT:
      case tflite::BuiltinOperator_SLICE:
      case tflite::BuiltinOperator_SIN:
      case tflite::BuiltinOperator_LOG:
      case tflite::BuiltinOperator_TRANSPOSE_CONV:
      case tflite::BuiltinOperator_TILE:
      case tflite::BuiltinOperator_EXPAND_DIMS:
      case tflite::BuiltinOperator_SPARSE_TO_DENSE:
      case tflite::BuiltinOperator_EQUAL:
      case tflite::BuiltinOperator_NOT_EQUAL:
      case tflite::BuiltinOperator_SUM:
      case tflite::BuiltinOperator_REDUCE_MAX:
      case tflite::BuiltinOperator_REDUCE_MIN:
      case tflite::BuiltinOperator_REDUCE_PROD:
      case tflite::BuiltinOperator_SQRT:
      case tflite::BuiltinOperator_RSQRT:
      case tflite::BuiltinOperator_SHAPE:
      case tflite::BuiltinOperator_POW:
      case tflite::BuiltinOperator_FAKE_QUANT:
      case tflite::BuiltinOperator_PACK:
      case tflite::BuiltinOperator_LOGICAL_OR:
      case tflite::BuiltinOperator_ONE_HOT:
      case tflite::BuiltinOperator_LOGICAL_AND:
      case tflite::BuiltinOperator_LOGICAL_NOT:
      case tflite::BuiltinOperator_UNPACK:
      case tflite::BuiltinOperator_FLOOR_DIV:
      case tflite::BuiltinOperator_REDUCE_ANY:
      case tflite::BuiltinOperator_SQUARE:
      case tflite::BuiltinOperator_ZEROS_LIKE:
      case tflite::BuiltinOperator_FILL:
      case tflite::BuiltinOperator_FLOOR_MOD:
      case tflite::BuiltinOperator_RANGE:
      case tflite::BuiltinOperator_LEAKY_RELU:
      case tflite::BuiltinOperator_SQUARED_DIFFERENCE:
      case tflite::BuiltinOperator_MIRROR_PAD:
      case tflite::BuiltinOperator_ABS:
      case tflite::BuiltinOperator_SPLIT_V:
      case tflite::BuiltinOperator_UNIQUE:
      case tflite::BuiltinOperator_CEIL:
      case tflite::BuiltinOperator_REVERSE_V2:
      case tflite::BuiltinOperator_ADD_N:
      case tflite::BuiltinOperator_GATHER_ND:
      case tflite::BuiltinOperator_WHERE:
      case tflite::BuiltinOperator_RANK:
      case tflite::BuiltinOperator_ELU:
      case tflite::BuiltinOperator_REVERSE_SEQUENCE:
      case tflite::BuiltinOperator_MATRIX_DIAG:
      case tflite::BuiltinOperator_QUANTIZE:
      case tflite::BuiltinOperator_MATRIX_SET_DIAG:
        logError("Op code %d is currently not delegated to NNAPI", builtin);
        return kTfLiteError;
        break;
      case tflite::BuiltinOperator_CUSTOM:
        logError("Custom operations are not supported when using NNAPI.");
        return kTfLiteError;
        break;
    }

    if (nnapi_version == 11 && nnapi->android_sdk_version < 28) {
      logError("Op %d needs NNAPI1.1", builtin);
      return kTfLiteError;
    }

    // Add the operation.
    RETURN_ERROR_IF_NN_FAILED(nnapi->ANeuralNetworksModel_addOperation(
        nn_model, nn_op_type, static_cast<uint32_t>(augmented_inputs.size()),
        augmented_inputs.data(),
        static_cast<uint32_t>(augmented_outputs.size()),
        reinterpret_cast<uint32_t*>(augmented_outputs.data())));
  }
  return kTfLiteOk;
}

TfLiteStatus NNAPIDelegate::BuildGraph(Subgraph* subgraph) {
  if (nn_model_ && nn_compiled_model_) return model_status_;

  const NnApi* nnapi = NnApiImplementation();
  // TODO(aselle): This is not correct. need to handle resize invalidation.
  if (!nn_model_) {
    CHECK_NN(nnapi->ANeuralNetworksModel_create(&nn_model_));

    // Find which tensors should be added to NNAPI. TFLite has temporaries
    // and RNN back-edges which are are not valid for NNAPI. We look through all
    // inputs and outputs and mark the mapping in tensor_id_to_nnapi_id with
    // kOperandIdNotSet. addTensorOperands will replace those with the
    // corresponding NNAPI operand ids and skip kOperandNotNeeded entries.
    std::vector<int64_t> tensor_id_to_nnapi_id(subgraph->tensors_size(),
                                               kOperandNotNeeded);
    auto set_ids_to_not_set = [&tensor_id_to_nnapi_id](const int* buf,
                                                       size_t count) {
      for (int j = 0; j < count; j++) {
        auto tensor_id = buf[j];
        if (tensor_id != kOptionalTensor) {
          tensor_id_to_nnapi_id[tensor_id] = kOperandIdNotSet;
        }
      }
    };
    for (size_t i = 0; i < subgraph->nodes_size(); i++) {
      const auto* node_and_registration = subgraph->node_and_registration(i);
      const TfLiteNode& node = node_and_registration->first;
      set_ids_to_not_set(node.inputs->data, node.inputs->size);
      set_ids_to_not_set(node.outputs->data, node.outputs->size);
    }
    set_ids_to_not_set(subgraph->inputs().data(), subgraph->inputs().size());
    set_ids_to_not_set(subgraph->outputs().data(), subgraph->outputs().size());

    uint32_t next_id = 0;
    RETURN_ERROR_IF_TFLITE_FAILED(addTensorOperands(
        subgraph, nn_model_, &next_id, &tensor_id_to_nnapi_id));
    RETURN_ERROR_IF_TFLITE_FAILED(
        AddOpsAndParams(subgraph, nn_model_, next_id, &model_states_inputs_,
                        &model_states_outputs_, tensor_id_to_nnapi_id));

    std::vector<uint32_t> augmented_inputs;
    MapAndAddTensorIds(subgraph->inputs().data(), subgraph->inputs().size(),
                       &augmented_inputs, tensor_id_to_nnapi_id);
    augmented_inputs.insert(augmented_inputs.end(),
                            model_states_inputs_.begin(),
                            model_states_inputs_.end());
    std::vector<uint32_t> augmented_outputs;
    MapAndAddTensorIds(subgraph->outputs().data(), subgraph->outputs().size(),
                       &augmented_outputs, tensor_id_to_nnapi_id);
    MapAndAddTensorIds(model_states_outputs_.data(),
                       model_states_outputs_.size(), &augmented_outputs,
                       tensor_id_to_nnapi_id);

    CHECK_NN(nnapi->ANeuralNetworksModel_identifyInputsAndOutputs(
        nn_model_, static_cast<uint32_t>(augmented_inputs.size()),
        reinterpret_cast<const uint32_t*>(augmented_inputs.data()),
        static_cast<uint32_t>(augmented_outputs.size()),
        reinterpret_cast<const uint32_t*>(augmented_outputs.data())));

    if (nnapi->android_sdk_version >= 28) {
      CHECK_NN(nnapi->ANeuralNetworksModel_relaxComputationFloat32toFloat16(
          nn_model_, subgraph->GetAllowFp16PrecisionForFp32()));
    }
    CHECK_NN(nnapi->ANeuralNetworksModel_finish(nn_model_));
  }
  if (!nn_compiled_model_) {
    CHECK_NN(nnapi->ANeuralNetworksCompilation_create(nn_model_,
                                                      &nn_compiled_model_));
    CHECK_NN(nnapi->ANeuralNetworksCompilation_finish(nn_compiled_model_));
  }
  return kTfLiteOk;
}

TfLiteStatus NNAPIDelegate::Invoke(Subgraph* subgraph) {
  if (!nn_model_) {
    model_status_ = BuildGraph(subgraph);
    if (model_status_ != kTfLiteOk) {
      logError("Failed to build graph for NNAPI");
    }
  }
  if (model_status_ != kTfLiteOk) {
    return model_status_;
  }

  const NnApi* nnapi = NnApiImplementation();
  ANeuralNetworksExecution* execution = nullptr;
  CHECK_NN(
      nnapi->ANeuralNetworksExecution_create(nn_compiled_model_, &execution));

  // Currently perform deep copy of input buffer
  for (size_t i = 0; i < subgraph->inputs().size(); i++) {
    int input = subgraph->inputs()[i];
    // TODO(aselle): Is this what we want or do we want input instead?
    // TODO(aselle): This should be called setInputValue maybe to be cons.
    TfLiteTensor* tensor = subgraph->tensor(input);
    CHECK_NN(nnapi->ANeuralNetworksExecution_setInput(
        execution, i, nullptr, tensor->data.raw, tensor->bytes));
  }

  // Tell nn api where to place final data.
  for (size_t i = 0; i < subgraph->outputs().size(); i++) {
    int output = subgraph->outputs()[i];
    TfLiteTensor* tensor = subgraph->tensor(output);
    CHECK_NN(nnapi->ANeuralNetworksExecution_setOutput(
        execution, i, nullptr, tensor->data.raw, tensor->bytes));
  }

  // The state_out of previous invocation need to be mapped to state_in of
  // current invocation.
  for (size_t i = 0; i < model_states_outputs_.size(); i++) {
    int state_tensor_idx = model_states_outputs_[i];
    TfLiteTensor* tensor = subgraph->tensor(state_tensor_idx);
    // Here we are using a deep copy for state_in tensors so that we are not
    // reading and writing into the same buffer during a invocation.
    // TODO(miaowang): using double shared buffer to minimize the copies.
    CHECK_NN(nnapi->ANeuralNetworksExecution_setInput(
        execution, i + subgraph->inputs().size(), nullptr, tensor->data.raw,
        tensor->bytes));
    // Tell NNAPI where to output the state_out.
    CHECK_NN(nnapi->ANeuralNetworksExecution_setOutput(
        execution, i + subgraph->outputs().size(), nullptr, tensor->data.raw,
        tensor->bytes));
  }

  // Currently use blocking compute.
  ANeuralNetworksEvent* event = nullptr;
  CHECK_NN(nnapi->ANeuralNetworksExecution_startCompute(execution, &event));
  CHECK_NN(nnapi->ANeuralNetworksEvent_wait(event));
  nnapi->ANeuralNetworksEvent_free(event);
  nnapi->ANeuralNetworksExecution_free(execution);

#if 0
  printf("From the NN API:\n");
  TfLiteTensor* tensor = subgraph->tensor(subgraph->outputs()[0]);
  if (float* data =
          subgraph->typed_tensor<float>(subgraph->outputs()[0])) {
    size_t num = tensor->bytes / sizeof(float);
    for (float* p = data; p < data + num; p++) {
      printf(" %f", *p);
    }
    printf("\n");
  }
#endif

  return kTfLiteOk;
}

bool NNAPIDelegate::IsSupported() {
  return NnApiImplementation()->nnapi_exists;
}

}  // namespace tflite
