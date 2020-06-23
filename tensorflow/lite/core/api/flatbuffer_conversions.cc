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

#include "tensorflow/lite/core/api/flatbuffer_conversions.h"

#include <cstddef>
#include <cstdint>
#include <memory>

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

namespace {

// Utility class for safely allocating POD data. This is useful for avoiding
// leaks in cases where op params are allocated but fail to propagate to the
// parsed op data (e.g., when model parameters are invalid).
class SafeBuiltinDataAllocator {
 public:
  class BuiltinDataDeleter {
   public:
    explicit BuiltinDataDeleter(BuiltinDataAllocator* allocator)
        : allocator_(allocator) {}

    void operator()(void* data) { allocator_->Deallocate(data); }

   private:
    BuiltinDataAllocator* allocator_;
  };

  template <typename T>
  using BuiltinDataPtr = std::unique_ptr<T, BuiltinDataDeleter>;

  explicit SafeBuiltinDataAllocator(BuiltinDataAllocator* allocator)
      : allocator_(allocator) {}

  template <typename T>
  BuiltinDataPtr<T> Allocate() {
    return BuiltinDataPtr<T>(allocator_->AllocatePOD<T>(),
                             BuiltinDataDeleter(allocator_));
  }

 private:
  BuiltinDataAllocator* allocator_;
};

// All the Parse functions take some pointers as params and this function has
// the common DCHECKs to catch if any of those are nullptr.
void CheckParsePointerParams(const Operator* op, ErrorReporter* error_reporter,
                             BuiltinDataAllocator* allocator,
                             void** builtin_data) {
  TFLITE_DCHECK(op != nullptr);
  TFLITE_DCHECK(error_reporter != nullptr);
  TFLITE_DCHECK(allocator != nullptr);
  TFLITE_DCHECK(builtin_data != nullptr);
}

// Copies the contents from the flatbuffer int vector `flatbuffer` into the
// int array `buffer`. `flat_vector` and `buffer` represent the same
// configuration operation for a given operation.
TfLiteStatus FlatBufferIntVectorToArray(
    int max_size_of_buffer, const flatbuffers::Vector<int32_t>* flat_vector,
    int* buffer, ErrorReporter* error_reporter, const char* op_name) {
  if (!flat_vector) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Input array not provided for operation '%s'.\n",
                         op_name);
    return kTfLiteError;
  } else {
    size_t num_dimensions = flat_vector->size();
    if (num_dimensions > max_size_of_buffer / sizeof(int)) {
      TF_LITE_REPORT_ERROR(
          error_reporter,
          "Found too many dimensions in the input array of operation '%s'.\n",
          op_name);
      return kTfLiteError;
    } else {
      for (size_t i = 0; i < num_dimensions; ++i) {
        buffer[i] = flat_vector->Get(i);
      }
    }
  }
  return kTfLiteOk;
}

// Converts the flatbuffer activation to what is used at runtime.
TfLiteFusedActivation ConvertActivation(ActivationFunctionType activation) {
  switch (activation) {
    case ActivationFunctionType_NONE:
      return kTfLiteActNone;
    case ActivationFunctionType_RELU:
      return kTfLiteActRelu;
    case ActivationFunctionType_RELU_N1_TO_1:
      return kTfLiteActReluN1To1;
    case ActivationFunctionType_RELU6:
      return kTfLiteActRelu6;
    case ActivationFunctionType_TANH:
      return kTfLiteActTanh;
    case ActivationFunctionType_SIGN_BIT:
      return kTfLiteActSignBit;
  }
  return kTfLiteActNone;
}

// Converts the flatbuffer padding enum to what is used at runtime.
TfLitePadding ConvertPadding(Padding padding) {
  switch (padding) {
    case Padding_SAME:
      return kTfLitePaddingSame;
    case Padding_VALID:
      return kTfLitePaddingValid;
  }
  return kTfLitePaddingUnknown;
}

}  // namespace

TfLiteStatus ConvertTensorType(TensorType tensor_type, TfLiteType* type,
                               ErrorReporter* error_reporter) {
  switch (tensor_type) {
    case TensorType_FLOAT16:
      *type = kTfLiteFloat16;
      return kTfLiteOk;
    case TensorType_FLOAT32:
      *type = kTfLiteFloat32;
      return kTfLiteOk;
    case TensorType_FLOAT64:
      *type = kTfLiteFloat64;
      return kTfLiteOk;
    case TensorType_INT16:
      *type = kTfLiteInt16;
      return kTfLiteOk;
    case TensorType_INT32:
      *type = kTfLiteInt32;
      return kTfLiteOk;
    case TensorType_UINT8:
      *type = kTfLiteUInt8;
      return kTfLiteOk;
    case TensorType_INT8:
      *type = kTfLiteInt8;
      return kTfLiteOk;
    case TensorType_INT64:
      *type = kTfLiteInt64;
      return kTfLiteOk;
    case TensorType_STRING:
      *type = kTfLiteString;
      return kTfLiteOk;
    case TensorType_BOOL:
      *type = kTfLiteBool;
      return kTfLiteOk;
    case TensorType_COMPLEX64:
      *type = kTfLiteComplex64;
      return kTfLiteOk;
    default:
      *type = kTfLiteNoType;
      TF_LITE_REPORT_ERROR(error_reporter,
                           "Unsupported data type %d in tensor\n", tensor_type);
      return kTfLiteError;
  }
}

TfLiteStatus ParseConv2D(const Operator* op, BuiltinOperator,
                         ErrorReporter* error_reporter,
                         BuiltinDataAllocator* allocator, void** builtin_data) {
  CheckParsePointerParams(op, error_reporter, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteConvParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteConvParams>();
  TF_LITE_ENSURE(error_reporter, params != nullptr);

  const Conv2DOptions* schema_params = op->builtin_options_as_Conv2DOptions();

  if (schema_params != nullptr) {
    params->padding = ConvertPadding(schema_params->padding());
    params->stride_width = schema_params->stride_w();
    params->stride_height = schema_params->stride_h();
    params->activation =
        ConvertActivation(schema_params->fused_activation_function());

    params->dilation_width_factor = schema_params->dilation_w_factor();
    params->dilation_height_factor = schema_params->dilation_h_factor();
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better undertand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

TfLiteStatus ParseDepthwiseConv2D(const Operator* op, BuiltinOperator,
                                  ErrorReporter* error_reporter,
                                  BuiltinDataAllocator* allocator,
                                  void** builtin_data) {
  CheckParsePointerParams(op, error_reporter, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);

  std::unique_ptr<TfLiteDepthwiseConvParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteDepthwiseConvParams>();
  TF_LITE_ENSURE(error_reporter, params != nullptr);

  const DepthwiseConv2DOptions* schema_params =
      op->builtin_options_as_DepthwiseConv2DOptions();

  if (schema_params != nullptr) {
    params->padding = ConvertPadding(schema_params->padding());
    params->stride_width = schema_params->stride_w();
    params->stride_height = schema_params->stride_h();
    params->depth_multiplier = schema_params->depth_multiplier();
    params->activation =
        ConvertActivation(schema_params->fused_activation_function());

    params->dilation_width_factor = schema_params->dilation_w_factor();
    params->dilation_height_factor = schema_params->dilation_h_factor();
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better undertand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseDequantize(const Operator*, BuiltinOperator, ErrorReporter*,
                             BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

TfLiteStatus ParseFullyConnected(const Operator* op, BuiltinOperator,
                                 ErrorReporter* error_reporter,
                                 BuiltinDataAllocator* allocator,
                                 void** builtin_data) {
  CheckParsePointerParams(op, error_reporter, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);

  std::unique_ptr<TfLiteFullyConnectedParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteFullyConnectedParams>();
  TF_LITE_ENSURE(error_reporter, params != nullptr);

  const FullyConnectedOptions* schema_params =
      op->builtin_options_as_FullyConnectedOptions();

  if (schema_params != nullptr) {
    params->activation =
        ConvertActivation(schema_params->fused_activation_function());
    params->keep_num_dims = schema_params->keep_num_dims();
    params->asymmetric_quantize_inputs =
        schema_params->asymmetric_quantize_inputs();

    switch (schema_params->weights_format()) {
      case FullyConnectedOptionsWeightsFormat_DEFAULT:
        params->weights_format = kTfLiteFullyConnectedWeightsFormatDefault;
        break;
      case FullyConnectedOptionsWeightsFormat_SHUFFLED4x16INT8:
        params->weights_format =
            kTfLiteFullyConnectedWeightsFormatShuffled4x16Int8;
        break;
      default:
        TF_LITE_REPORT_ERROR(error_reporter,
                             "Unhandled fully-connected weights format.");
        return kTfLiteError;
    }
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better undertand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

TfLiteStatus ParseReshape(const Operator* op, BuiltinOperator,
                          ErrorReporter* error_reporter,
                          BuiltinDataAllocator* allocator,
                          void** builtin_data) {
  CheckParsePointerParams(op, error_reporter, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);

  std::unique_ptr<TfLiteReshapeParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteReshapeParams>();
  TF_LITE_ENSURE(error_reporter, params != nullptr);

  const ReshapeOptions* schema_params = op->builtin_options_as_ReshapeOptions();

  if (schema_params != nullptr) {
    const flatbuffers::Vector<int32_t>* new_shape = schema_params->new_shape();
    // TODO(b/147203660): We need to figure out when dynamic reshape
    // (new_shape is a tensor) happens, why the option is not a nullptr.
    // But nonethless, we should only copy when new_shape is not a nullptr.
    if (new_shape != nullptr) {
      TF_LITE_ENSURE_STATUS(
          FlatBufferIntVectorToArray(sizeof(params->shape), new_shape,
                                     params->shape, error_reporter, "reshape"));
      params->num_dimensions = new_shape->size();
    } else {
      // TODO(b/157480169) TODO(b/147203660): We should either return
      // kTfLiteError or fill in some reasonable defaults in the params struct.
      // We are not doing so until we better undertand the ramifications of
      // changing the legacy behavior.
    }
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better undertand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseQuantize(const Operator*, BuiltinOperator, ErrorReporter*,
                           BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

TfLiteStatus ParseSoftmax(const Operator* op, BuiltinOperator,
                          ErrorReporter* error_reporter,
                          BuiltinDataAllocator* allocator,
                          void** builtin_data) {
  CheckParsePointerParams(op, error_reporter, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteSoftmaxParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteSoftmaxParams>();
  TF_LITE_ENSURE(error_reporter, params != nullptr);

  const SoftmaxOptions* schema_params = op->builtin_options_as_SoftmaxOptions();

  if (schema_params != nullptr) {
    params->beta = schema_params->beta();
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better undertand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

TfLiteStatus ParseSvdf(const Operator* op, BuiltinOperator,
                       ErrorReporter* error_reporter,
                       BuiltinDataAllocator* allocator, void** builtin_data) {
  CheckParsePointerParams(op, error_reporter, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteSVDFParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteSVDFParams>();
  TF_LITE_ENSURE(error_reporter, params != nullptr);

  const SVDFOptions* schema_params = op->builtin_options_as_SVDFOptions();
  if (schema_params != nullptr) {
    params->rank = schema_params->rank();
    params->activation =
        ConvertActivation(schema_params->fused_activation_function());
    params->asymmetric_quantize_inputs =
        schema_params->asymmetric_quantize_inputs();
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better undertand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

TfLiteStatus ParseOpData(const Operator* op, BuiltinOperator op_type,
                         ErrorReporter* error_reporter,
                         BuiltinDataAllocator* allocator, void** builtin_data) {
  auto parseLSHProjectionType = [](LSHProjectionType type) {
    switch (type) {
      case LSHProjectionType_SPARSE:
        return kTfLiteLshProjectionSparse;
      case LSHProjectionType_DENSE:
        return kTfLiteLshProjectionDense;
      default:
        return kTfLiteLshProjectionUnknown;
    }
  };
  auto parseCombinerType = [](CombinerType type) {
    switch (type) {
      case CombinerType_MEAN:
        return kTfLiteCombinerTypeMean;
      case CombinerType_SQRTN:
        return kTfLiteCombinerTypeSqrtn;
      case CombinerType_SUM:
      default:
        return kTfLiteCombinerTypeSum;
    }
  };

  SafeBuiltinDataAllocator safe_allocator(allocator);
  *builtin_data = nullptr;
  switch (op_type) {
    case BuiltinOperator_CONV_2D: {
      return ParseConv2D(op, op_type, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_DEPTHWISE_CONV_2D: {
      return ParseDepthwiseConv2D(op, op_type, error_reporter, allocator,
                                  builtin_data);
    }

    case BuiltinOperator_DEQUANTIZE: {
      return ParseDequantize(op, op_type, error_reporter, allocator,
                             builtin_data);
    }

    case BuiltinOperator_FULLY_CONNECTED: {
      return ParseFullyConnected(op, op_type, error_reporter, allocator,
                                 builtin_data);
    }

    case BuiltinOperator_QUANTIZE: {
      return ParseQuantize(op, op_type, error_reporter, allocator,
                           builtin_data);
    }

    case BuiltinOperator_RESHAPE: {
      return ParseReshape(op, op_type, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_SOFTMAX: {
      return ParseSoftmax(op, op_type, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_SVDF: {
      return ParseSvdf(op, op_type, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_CAST: {
      auto params = safe_allocator.Allocate<TfLiteCastParams>();
      TF_LITE_ENSURE(error_reporter, params != nullptr);
      if (const auto* schema_params = op->builtin_options_as_CastOptions()) {
        TF_LITE_ENSURE_STATUS(ConvertTensorType(schema_params->in_data_type(),
                                                &params->in_data_type,
                                                error_reporter));
        TF_LITE_ENSURE_STATUS(ConvertTensorType(schema_params->out_data_type(),
                                                &params->out_data_type,
                                                error_reporter));
      }
      *builtin_data = params.release();
      return kTfLiteOk;
    }
    case BuiltinOperator_LSH_PROJECTION: {
      auto params = safe_allocator.Allocate<TfLiteLSHProjectionParams>();
      TF_LITE_ENSURE(error_reporter, params != nullptr);
      if (const auto* lshParams =
              op->builtin_options_as_LSHProjectionOptions()) {
        params->type = parseLSHProjectionType(lshParams->type());
      }
      *builtin_data = params.release();
      return kTfLiteOk;
    }
    case BuiltinOperator_AVERAGE_POOL_2D:
    case BuiltinOperator_MAX_POOL_2D:
    case BuiltinOperator_L2_POOL_2D: {
      auto params = safe_allocator.Allocate<TfLitePoolParams>();
      TF_LITE_ENSURE(error_reporter, params != nullptr);
      if (const auto* pool_params = op->builtin_options_as_Pool2DOptions()) {
        params->padding = ConvertPadding(pool_params->padding());
        params->stride_width = pool_params->stride_w();
        params->stride_height = pool_params->stride_h();
        params->filter_width = pool_params->filter_width();
        params->filter_height = pool_params->filter_height();
        params->activation =
            ConvertActivation(pool_params->fused_activation_function());
      }
      *builtin_data = params.release();
      return kTfLiteOk;
    }
    case BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_RNN: {
      auto params = safe_allocator.Allocate<TfLiteSequenceRNNParams>();
      TF_LITE_ENSURE(error_reporter, params != nullptr);
      if (const auto* sequence_rnn_params =
              op->builtin_options_as_SequenceRNNOptions()) {
        params->activation =
            ConvertActivation(sequence_rnn_params->fused_activation_function());
        params->time_major = sequence_rnn_params->time_major();
        params->asymmetric_quantize_inputs =
            sequence_rnn_params->asymmetric_quantize_inputs();
      }
      *builtin_data = params.release();
      return kTfLiteOk;
    }
    case BuiltinOperator_BIDIRECTIONAL_SEQUENCE_RNN: {
      auto params =
          safe_allocator.Allocate<TfLiteBidirectionalSequenceRNNParams>();
      TF_LITE_ENSURE(error_reporter, params != nullptr);
      if (const auto* bidi_sequence_rnn_params =
              op->builtin_options_as_BidirectionalSequenceRNNOptions()) {
        params->activation = ConvertActivation(
            bidi_sequence_rnn_params->fused_activation_function());
        params->time_major = bidi_sequence_rnn_params->time_major();
        params->merge_outputs = bidi_sequence_rnn_params->merge_outputs();
        params->asymmetric_quantize_inputs =
            bidi_sequence_rnn_params->asymmetric_quantize_inputs();
      }
      *builtin_data = params.release();
      return kTfLiteOk;
    }
    case BuiltinOperator_RNN: {
      auto params = safe_allocator.Allocate<TfLiteRNNParams>();
      TF_LITE_ENSURE(error_reporter, params != nullptr);
      if (const auto* rnn_params = op->builtin_options_as_RNNOptions()) {
        params->activation =
            ConvertActivation(rnn_params->fused_activation_function());
        params->asymmetric_quantize_inputs =
            rnn_params->asymmetric_quantize_inputs();
      }
      *builtin_data = params.release();
      return kTfLiteOk;
    }
    case BuiltinOperator_EMBEDDING_LOOKUP_SPARSE: {
      auto params =
          safe_allocator.Allocate<TfLiteEmbeddingLookupSparseParams>();
      TF_LITE_ENSURE(error_reporter, params != nullptr);
      if (const auto* embedding_params =
              op->builtin_options_as_EmbeddingLookupSparseOptions()) {
        params->combiner = parseCombinerType(embedding_params->combiner());
      }
      *builtin_data = params.release();
      return kTfLiteOk;
    }

    case BuiltinOperator_HASHTABLE_LOOKUP:
      // no-op.
      return kTfLiteOk;
    case BuiltinOperator_CONCATENATION: {
      auto params = safe_allocator.Allocate<TfLiteConcatenationParams>();
      TF_LITE_ENSURE(error_reporter, params != nullptr);
      if (const auto* concatenation_params =
              op->builtin_options_as_ConcatenationOptions()) {
        params->activation = ConvertActivation(
            concatenation_params->fused_activation_function());
        params->axis = concatenation_params->axis();
      }
      *builtin_data = params.release();
      return kTfLiteOk;
    }
    case BuiltinOperator_MUL: {
      auto params = safe_allocator.Allocate<TfLiteMulParams>();
      TF_LITE_ENSURE(error_reporter, params != nullptr);
      if (const auto* schema_params = op->builtin_options_as_MulOptions()) {
        params->activation =
            ConvertActivation(schema_params->fused_activation_function());
      }
      *builtin_data = params.release();
      return kTfLiteOk;
    }
    case BuiltinOperator_ADD: {
      auto params = safe_allocator.Allocate<TfLiteAddParams>();
      TF_LITE_ENSURE(error_reporter, params != nullptr);
      if (const auto* schema_params = op->builtin_options_as_AddOptions()) {
        params->activation =
            ConvertActivation(schema_params->fused_activation_function());
      }
      *builtin_data = params.release();
      return kTfLiteOk;
    }
    case BuiltinOperator_DIV: {
      auto params = safe_allocator.Allocate<TfLiteDivParams>();
      TF_LITE_ENSURE(error_reporter, params != nullptr);
      if (const auto* schema_params = op->builtin_options_as_DivOptions()) {
        params->activation =
            ConvertActivation(schema_params->fused_activation_function());
      }
      *builtin_data = params.release();
      return kTfLiteOk;
    }
    case BuiltinOperator_SUB: {
      auto params = safe_allocator.Allocate<TfLiteSubParams>();
      TF_LITE_ENSURE(error_reporter, params != nullptr);
      if (const auto* schema_params = op->builtin_options_as_SubOptions()) {
        params->activation =
            ConvertActivation(schema_params->fused_activation_function());
      }
      *builtin_data = params.release();
      return kTfLiteOk;
    }
    case BuiltinOperator_L2_NORMALIZATION: {
      auto params = safe_allocator.Allocate<TfLiteL2NormParams>();
      TF_LITE_ENSURE(error_reporter, params != nullptr);
      if (const auto* schema_params = op->builtin_options_as_L2NormOptions()) {
        params->activation =
            ConvertActivation(schema_params->fused_activation_function());
      }
      *builtin_data = params.release();
      return kTfLiteOk;
    }
    case BuiltinOperator_LOCAL_RESPONSE_NORMALIZATION: {
      auto params = safe_allocator.Allocate<TfLiteLocalResponseNormParams>();
      TF_LITE_ENSURE(error_reporter, params != nullptr);
      if (const auto* schema_params =
              op->builtin_options_as_LocalResponseNormalizationOptions()) {
        params->radius = schema_params->radius();
        params->bias = schema_params->bias();
        params->alpha = schema_params->alpha();
        params->beta = schema_params->beta();
      }
      *builtin_data = params.release();
      return kTfLiteOk;
    }
    case BuiltinOperator_LSTM: {
      auto params = safe_allocator.Allocate<TfLiteLSTMParams>();
      TF_LITE_ENSURE(error_reporter, params != nullptr);
      if (const auto* lstm_params = op->builtin_options_as_LSTMOptions()) {
        params->activation =
            ConvertActivation(lstm_params->fused_activation_function());
        params->cell_clip = lstm_params->cell_clip();
        params->proj_clip = lstm_params->proj_clip();
        switch (lstm_params->kernel_type()) {
          case LSTMKernelType_FULL:
            params->kernel_type = kTfLiteLSTMFullKernel;
            break;
          case LSTMKernelType_BASIC:
            params->kernel_type = kTfLiteLSTMBasicKernel;
            break;
          default:
            TF_LITE_REPORT_ERROR(error_reporter,
                                 "Unhandled LSTM kernel type: %d",
                                 lstm_params->kernel_type());
            return kTfLiteError;
        }
        params->asymmetric_quantize_inputs =
            lstm_params->asymmetric_quantize_inputs();
      } else {
        TF_LITE_REPORT_ERROR(error_reporter,
                             "No valid LSTM builtin options exist");
        return kTfLiteError;
      }
      *builtin_data = params.release();
      return kTfLiteOk;
    }
    case BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM: {
      auto params =
          safe_allocator.Allocate<TfLiteUnidirectionalSequenceLSTMParams>();
      TF_LITE_ENSURE(error_reporter, params != nullptr);
      if (const auto* seq_lstm_params =
              op->builtin_options_as_UnidirectionalSequenceLSTMOptions()) {
        params->activation =
            ConvertActivation(seq_lstm_params->fused_activation_function());
        params->cell_clip = seq_lstm_params->cell_clip();
        params->proj_clip = seq_lstm_params->proj_clip();
        params->time_major = seq_lstm_params->time_major();
        params->asymmetric_quantize_inputs =
            seq_lstm_params->asymmetric_quantize_inputs();
      }
      *builtin_data = params.release();
      return kTfLiteOk;
    }
    case BuiltinOperator_BIDIRECTIONAL_SEQUENCE_LSTM: {
      auto params =
          safe_allocator.Allocate<TfLiteBidirectionalSequenceLSTMParams>();
      TF_LITE_ENSURE(error_reporter, params != nullptr);
      if (const auto* bidi_lstm_params =
              op->builtin_options_as_BidirectionalSequenceLSTMOptions()) {
        params->activation =
            ConvertActivation(bidi_lstm_params->fused_activation_function());
        params->cell_clip = bidi_lstm_params->cell_clip();
        params->proj_clip = bidi_lstm_params->proj_clip();
        params->merge_outputs = bidi_lstm_params->merge_outputs();
        params->time_major = bidi_lstm_params->time_major();
        params->asymmetric_quantize_inputs =
            bidi_lstm_params->asymmetric_quantize_inputs();
      }
      *builtin_data = params.release();
      return kTfLiteOk;
    }
    case BuiltinOperator_RESIZE_BILINEAR: {
      auto params = safe_allocator.Allocate<TfLiteResizeBilinearParams>();
      TF_LITE_ENSURE(error_reporter, params != nullptr);
      if (const auto* schema_params =
              op->builtin_options_as_ResizeBilinearOptions()) {
        params->align_corners = schema_params->align_corners();
        params->half_pixel_centers = schema_params->half_pixel_centers();
      } else {
        // Some older models did not populate the ResizeBilinearOptions field in
        // the flatbuffer, so ensure it's set to a sensible default.
        params->align_corners = false;
        params->half_pixel_centers = false;
      }
      *builtin_data = params.release();
      return kTfLiteOk;
    }
    case BuiltinOperator_RESIZE_NEAREST_NEIGHBOR: {
      auto params =
          safe_allocator.Allocate<TfLiteResizeNearestNeighborParams>();
      TF_LITE_ENSURE(error_reporter, params != nullptr);
      if (const auto* schema_params =
              op->builtin_options_as_ResizeNearestNeighborOptions()) {
        params->align_corners = schema_params->align_corners();
        params->half_pixel_centers = schema_params->half_pixel_centers();
      } else {
        params->align_corners = false;
        params->half_pixel_centers = false;
      }
      *builtin_data = params.release();
      return kTfLiteOk;
    }
    case BuiltinOperator_SKIP_GRAM: {
      auto params = safe_allocator.Allocate<TfLiteSkipGramParams>();
      TF_LITE_ENSURE(error_reporter, params != nullptr);
      if (const auto* skip_gram_params =
              op->builtin_options_as_SkipGramOptions()) {
        params->ngram_size = skip_gram_params->ngram_size();
        params->max_skip_size = skip_gram_params->max_skip_size();
        params->include_all_ngrams = skip_gram_params->include_all_ngrams();
      }
      *builtin_data = params.release();
      return kTfLiteOk;
    }
    case BuiltinOperator_SPACE_TO_DEPTH: {
      auto params = safe_allocator.Allocate<TfLiteSpaceToDepthParams>();
      TF_LITE_ENSURE(error_reporter, params != nullptr);
      if (const auto* schema_params =
              op->builtin_options_as_SpaceToDepthOptions()) {
        params->block_size = schema_params->block_size();
      }
      *builtin_data = params.release();
      return kTfLiteOk;
    }
    case BuiltinOperator_DEPTH_TO_SPACE: {
      auto params = safe_allocator.Allocate<TfLiteDepthToSpaceParams>();
      TF_LITE_ENSURE(error_reporter, params != nullptr);
      if (const auto* schema_params =
              op->builtin_options_as_DepthToSpaceOptions()) {
        params->block_size = schema_params->block_size();
      }
      *builtin_data = params.release();
      return kTfLiteOk;
    }
    case BuiltinOperator_GATHER: {
      auto params = safe_allocator.Allocate<TfLiteGatherParams>();
      TF_LITE_ENSURE(error_reporter, params != nullptr);
      params->axis = 0;
      if (const auto* gather_params = op->builtin_options_as_GatherOptions()) {
        params->axis = gather_params->axis();
      }

      *builtin_data = params.release();
      return kTfLiteOk;
    }
    case BuiltinOperator_MEAN:
    case BuiltinOperator_REDUCE_MAX:
    case BuiltinOperator_REDUCE_MIN:
    case BuiltinOperator_REDUCE_PROD:
    case BuiltinOperator_REDUCE_ANY:
    case BuiltinOperator_SUM: {
      auto params = safe_allocator.Allocate<TfLiteReducerParams>();
      TF_LITE_ENSURE(error_reporter, params != nullptr);
      if (const auto* schema_params = op->builtin_options_as_ReducerOptions()) {
        params->keep_dims = schema_params->keep_dims();
      }
      *builtin_data = params.release();
      return kTfLiteOk;
    }
    case BuiltinOperator_SPLIT: {
      auto params = safe_allocator.Allocate<TfLiteSplitParams>();
      TF_LITE_ENSURE(error_reporter, params != nullptr);
      if (const auto* schema_params = op->builtin_options_as_SplitOptions()) {
        params->num_splits = schema_params->num_splits();
      }
      *builtin_data = params.release();
      return kTfLiteOk;
    }
    case BuiltinOperator_SPLIT_V: {
      auto params = safe_allocator.Allocate<TfLiteSplitParams>();
      TF_LITE_ENSURE(error_reporter, params != nullptr);
      if (const auto* schema_params = op->builtin_options_as_SplitVOptions()) {
        params->num_splits = schema_params->num_splits();
      }
      *builtin_data = params.release();
      return kTfLiteOk;
    }
    case BuiltinOperator_SQUEEZE: {
      auto params = safe_allocator.Allocate<TfLiteSqueezeParams>();
      TF_LITE_ENSURE(error_reporter, params != nullptr);
      if (const auto* schema_params = op->builtin_options_as_SqueezeOptions()) {
        const auto* squeeze_dims = schema_params->squeeze_dims();
        TF_LITE_ENSURE_STATUS(FlatBufferIntVectorToArray(
            sizeof(params->squeeze_dims), squeeze_dims, params->squeeze_dims,
            error_reporter, "squeeze"));
        params->num_squeeze_dims = squeeze_dims->size();
      }
      *builtin_data = params.release();
      return kTfLiteOk;
    }
    case BuiltinOperator_STRIDED_SLICE: {
      auto params = safe_allocator.Allocate<TfLiteStridedSliceParams>();
      TF_LITE_ENSURE(error_reporter, params != nullptr);
      if (const auto* schema_params =
              op->builtin_options_as_StridedSliceOptions()) {
        params->begin_mask = schema_params->begin_mask();
        params->end_mask = schema_params->end_mask();
        params->ellipsis_mask = schema_params->ellipsis_mask();
        params->new_axis_mask = schema_params->new_axis_mask();
        params->shrink_axis_mask = schema_params->shrink_axis_mask();
      }
      *builtin_data = params.release();
      return kTfLiteOk;
    }
    case BuiltinOperator_ARG_MAX: {
      auto params = safe_allocator.Allocate<TfLiteArgMaxParams>();
      TF_LITE_ENSURE(error_reporter, params != nullptr);
      if (const auto* schema_params = op->builtin_options_as_ArgMaxOptions()) {
        TF_LITE_ENSURE_STATUS(ConvertTensorType(schema_params->output_type(),
                                                &params->output_type,
                                                error_reporter));
      }
      *builtin_data = params.release();
      return kTfLiteOk;
    }
    case BuiltinOperator_ARG_MIN: {
      auto params = safe_allocator.Allocate<TfLiteArgMinParams>();
      TF_LITE_ENSURE(error_reporter, params != nullptr);
      if (const auto* schema_params = op->builtin_options_as_ArgMinOptions()) {
        TF_LITE_ENSURE_STATUS(ConvertTensorType(schema_params->output_type(),
                                                &params->output_type,
                                                error_reporter));
      }
      *builtin_data = params.release();
      return kTfLiteOk;
    }
    case BuiltinOperator_TRANSPOSE_CONV: {
      auto params = safe_allocator.Allocate<TfLiteTransposeConvParams>();
      TF_LITE_ENSURE(error_reporter, params != nullptr);
      if (const auto* transpose_conv_params =
              op->builtin_options_as_TransposeConvOptions()) {
        params->padding = ConvertPadding(transpose_conv_params->padding());
        params->stride_width = transpose_conv_params->stride_w();
        params->stride_height = transpose_conv_params->stride_h();
      }
      *builtin_data = params.release();
      return kTfLiteOk;
    }
    case BuiltinOperator_SPARSE_TO_DENSE: {
      auto params = safe_allocator.Allocate<TfLiteSparseToDenseParams>();
      TF_LITE_ENSURE(error_reporter, params != nullptr);
      if (const auto* sparse_to_dense_params =
              op->builtin_options_as_SparseToDenseOptions()) {
        params->validate_indices = sparse_to_dense_params->validate_indices();
      }
      *builtin_data = params.release();
      return kTfLiteOk;
    }
    case BuiltinOperator_SHAPE: {
      auto params = safe_allocator.Allocate<TfLiteShapeParams>();
      TF_LITE_ENSURE(error_reporter, params != nullptr);
      if (const auto* schema_params = op->builtin_options_as_ShapeOptions()) {
        TF_LITE_ENSURE_STATUS(ConvertTensorType(
            schema_params->out_type(), &params->out_type, error_reporter));
      }
      *builtin_data = params.release();
      return kTfLiteOk;
    }
    case BuiltinOperator_PACK: {
      auto params = safe_allocator.Allocate<TfLitePackParams>();
      TF_LITE_ENSURE(error_reporter, params != nullptr);
      if (const auto* pack_params = op->builtin_options_as_PackOptions()) {
        params->values_count = pack_params->values_count();
        params->axis = pack_params->axis();
      }
      *builtin_data = params.release();
      return kTfLiteOk;
    }
    case BuiltinOperator_DELEGATE: {
      // TODO(ycling): Revisit when supporting saving delegated models.
      TF_LITE_REPORT_ERROR(error_reporter,
                           "DELEGATE op shouldn't exist in model.");
      return kTfLiteError;
    }
    case BuiltinOperator_FAKE_QUANT: {
      auto params = safe_allocator.Allocate<TfLiteFakeQuantParams>();
      TF_LITE_ENSURE(error_reporter, params != nullptr);
      if (const auto* schema_params =
              op->builtin_options_as_FakeQuantOptions()) {
        params->min = schema_params->min();
        params->max = schema_params->max();
        params->num_bits = schema_params->num_bits();
        params->narrow_range = schema_params->narrow_range();
      }
      *builtin_data = params.release();
      return kTfLiteOk;
    }
    case BuiltinOperator_ONE_HOT: {
      auto params = safe_allocator.Allocate<TfLiteOneHotParams>();
      TF_LITE_ENSURE(error_reporter, params != nullptr);
      if (const auto* schema_params = op->builtin_options_as_OneHotOptions()) {
        params->axis = schema_params->axis();
      }
      *builtin_data = params.release();
      return kTfLiteOk;
    }
    case BuiltinOperator_UNPACK: {
      auto params = safe_allocator.Allocate<TfLiteUnpackParams>();
      TF_LITE_ENSURE(error_reporter, params != nullptr);
      if (const auto* unpack_params = op->builtin_options_as_UnpackOptions()) {
        params->num = unpack_params->num();
        params->axis = unpack_params->axis();
      }
      *builtin_data = params.release();
      return kTfLiteOk;
    }
    case BuiltinOperator_LEAKY_RELU: {
      auto params = safe_allocator.Allocate<TfLiteLeakyReluParams>();
      TF_LITE_ENSURE(error_reporter, params != nullptr);
      if (const auto* leaky_relu_params =
              op->builtin_options_as_LeakyReluOptions()) {
        params->alpha = leaky_relu_params->alpha();
      }
      *builtin_data = params.release();
      return kTfLiteOk;
    }
    case BuiltinOperator_MIRROR_PAD: {
      auto params = safe_allocator.Allocate<TfLiteMirrorPaddingParams>();
      TF_LITE_ENSURE(error_reporter, params != nullptr);
      const auto* mirror_pad_params = op->builtin_options_as_MirrorPadOptions();
      if (mirror_pad_params != nullptr) {
        params->mode =
            mirror_pad_params->mode() == tflite::MirrorPadMode_REFLECT
                ? TfLiteMirrorPaddingMode::kTfLiteMirrorPaddingReflect
                : TfLiteMirrorPaddingMode::kTfLiteMirrorPaddingSymmetric;
      }
      *builtin_data = params.release();
      return kTfLiteOk;
    }
    case BuiltinOperator_UNIQUE: {
      auto params = safe_allocator.Allocate<TfLiteUniqueParams>();
      TF_LITE_ENSURE(error_reporter, params != nullptr);
      const auto* unique_params = op->builtin_options_as_UniqueOptions();
      if (unique_params != nullptr) {
        params->index_out_type =
            unique_params->idx_out_type() == tflite::TensorType_INT64
                ? TfLiteType::kTfLiteInt64
                : TfLiteType::kTfLiteInt32;
      }
      *builtin_data = params.release();
      return kTfLiteOk;
    }
    case BuiltinOperator_REVERSE_SEQUENCE: {
      auto params = safe_allocator.Allocate<TfLiteReverseSequenceParams>();
      TF_LITE_ENSURE(error_reporter, params != nullptr);
      if (const auto* reverse_seq_params =
              op->builtin_options_as_ReverseSequenceOptions()) {
        params->seq_dim = reverse_seq_params->seq_dim();
        params->batch_dim = reverse_seq_params->batch_dim();
      }
      *builtin_data = params.release();
      return kTfLiteOk;
    }
    case BuiltinOperator_IF: {
      auto params = safe_allocator.Allocate<TfLiteIfParams>();
      TF_LITE_ENSURE(error_reporter, params != nullptr);
      if (const auto* if_params = op->builtin_options_as_IfOptions()) {
        params->then_subgraph_index = if_params->then_subgraph_index();
        params->else_subgraph_index = if_params->else_subgraph_index();
      }
      *builtin_data = params.release();
      return kTfLiteOk;
    }
    case BuiltinOperator_WHILE: {
      auto params = safe_allocator.Allocate<TfLiteWhileParams>();
      TF_LITE_ENSURE(error_reporter, params != nullptr);
      if (const auto* while_params = op->builtin_options_as_WhileOptions()) {
        params->cond_subgraph_index = while_params->cond_subgraph_index();
        params->body_subgraph_index = while_params->body_subgraph_index();
      }
      *builtin_data = params.release();
      return kTfLiteOk;
    }
    case BuiltinOperator_BATCH_MATMUL: {
      auto params = safe_allocator.Allocate<TfLiteBatchMatMulParams>();
      TF_LITE_ENSURE(error_reporter, params != nullptr);
      if (const auto* bmm_params =
              op->builtin_options_as_BatchMatMulOptions()) {
        params->adj_x = bmm_params->adj_x();
        params->adj_y = bmm_params->adj_y();
      }
      *builtin_data = params.release();
      return kTfLiteOk;
    }
    // Below are the ops with no builtin_data structure.
    case BuiltinOperator_ABS:
    case BuiltinOperator_BATCH_TO_SPACE_ND:
    // TODO(aselle): Implement call in BuiltinOptions, but nullptrs are
    // ok for now, since there is no call implementation either.
    case BuiltinOperator_CALL:
    case BuiltinOperator_CONCAT_EMBEDDINGS:
    case BuiltinOperator_COS:
    case BuiltinOperator_CUSTOM:
    case BuiltinOperator_ELU:
    case BuiltinOperator_EMBEDDING_LOOKUP:
    case BuiltinOperator_EQUAL:
    case BuiltinOperator_EXP:
    case BuiltinOperator_EXPAND_DIMS:
    case BuiltinOperator_CEIL:
    case BuiltinOperator_FLOOR:
    case BuiltinOperator_GREATER:
    case BuiltinOperator_GREATER_EQUAL:
    case BuiltinOperator_HARD_SWISH:
    case BuiltinOperator_LESS:
    case BuiltinOperator_LESS_EQUAL:
    case BuiltinOperator_LOG:
    case BuiltinOperator_LOGISTIC:
    case BuiltinOperator_LOG_SOFTMAX:
    case BuiltinOperator_MATRIX_DIAG:
    case BuiltinOperator_MATRIX_SET_DIAG:
    case BuiltinOperator_MAXIMUM:
    case BuiltinOperator_MINIMUM:
    case BuiltinOperator_NEG:
    case BuiltinOperator_NOT_EQUAL:
    case BuiltinOperator_PAD:
    case BuiltinOperator_PADV2:
    case BuiltinOperator_PRELU:
    case BuiltinOperator_RELU:
    case BuiltinOperator_RELU6:
    case BuiltinOperator_RELU_N1_TO_1:
    case BuiltinOperator_ROUND:
    case BuiltinOperator_RSQRT:
    case BuiltinOperator_SELECT:
    case BuiltinOperator_SELECT_V2:
    case BuiltinOperator_SIN:
    case BuiltinOperator_SLICE:
    case BuiltinOperator_SPACE_TO_BATCH_ND:
    case BuiltinOperator_SQRT:
    case BuiltinOperator_TANH:
    case BuiltinOperator_TILE:
    case BuiltinOperator_TOPK_V2:
    case BuiltinOperator_TRANSPOSE:
    case BuiltinOperator_POW:
    case BuiltinOperator_LOGICAL_OR:
    case BuiltinOperator_LOGICAL_AND:
    case BuiltinOperator_LOGICAL_NOT:
    case BuiltinOperator_FLOOR_DIV:
    case BuiltinOperator_SQUARE:
    case BuiltinOperator_ZEROS_LIKE:
    case BuiltinOperator_FILL:
    case BuiltinOperator_FLOOR_MOD:
    case BuiltinOperator_RANGE:
    case BuiltinOperator_SQUARED_DIFFERENCE:
    case BuiltinOperator_REVERSE_V2:
    case BuiltinOperator_ADD_N:
    case BuiltinOperator_GATHER_ND:
    case BuiltinOperator_WHERE:
    case BuiltinOperator_RANK:
    case BuiltinOperator_NON_MAX_SUPPRESSION_V4:
    case BuiltinOperator_NON_MAX_SUPPRESSION_V5:
    case BuiltinOperator_SCATTER_ND:
    case BuiltinOperator_DENSIFY:
    case BuiltinOperator_SEGMENT_SUM:
      return kTfLiteOk;
  }
  return kTfLiteError;
}  // NOLINT[readability/fn_size]

}  // namespace tflite
