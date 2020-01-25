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

#include <cstdlib>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
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

// Copies the contents from the flatbuffer int vector `flatbuffer` into the
// int array `buffer`. `flat_vector` and `buffer` represent the same
// configuration operation for a given operation.
TfLiteStatus FlatBufferIntVectorToArray(
    int max_size_of_buffer, const flatbuffers::Vector<int32_t>* flat_vector,
    int* buffer, ErrorReporter* error_reporter, const char* op_name) {
  if (!flat_vector) {
    error_reporter->Report("Input array not provided for operation '%s'.\n",
                           op_name);
    return kTfLiteError;
  } else {
    size_t num_dimensions = flat_vector->size();
    if (num_dimensions > max_size_of_buffer / sizeof(int)) {
      error_reporter->Report(
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

}  // namespace

TfLiteStatus ConvertTensorType(TensorType tensor_type, TfLiteType* type,
                               ErrorReporter* error_reporter) {
  *type = kTfLiteNoType;
  switch (tensor_type) {
    case TensorType_FLOAT32:
      *type = kTfLiteFloat32;
      break;
    case TensorType_FLOAT16:
      *type = kTfLiteFloat16;
      break;
    case TensorType_INT16:
      *type = kTfLiteInt16;
      break;
    case TensorType_INT32:
      *type = kTfLiteInt32;
      break;
    case TensorType_UINT8:
      *type = kTfLiteUInt8;
      break;
    case TensorType_INT8:
      *type = kTfLiteInt8;
      break;
    case TensorType_INT64:
      *type = kTfLiteInt64;
      break;
    case TensorType_STRING:
      *type = kTfLiteString;
      break;
    case TensorType_BOOL:
      *type = kTfLiteBool;
      break;
    case TensorType_COMPLEX64:
      *type = kTfLiteComplex64;
      break;
  }
  if (*type == kTfLiteNoType) {
    error_reporter->Report("Unsupported data type %d in tensor\n", tensor_type);
    return kTfLiteError;
  }
  return kTfLiteOk;
}

// Parse the appropriate data out of the op.
//
// This handles builtin data explicitly as there are flatbuffer schemas.
// If it returns kTfLiteOk, it passes the data out with `builtin_data`, which
// need to be released by calling `free`.`
// If it returns kTfLiteError, `builtin_data` will be `nullptr`.
TfLiteStatus ParseOpData(const Operator* op, BuiltinOperator op_type,
                         ErrorReporter* error_reporter,
                         BuiltinDataAllocator* allocator, void** builtin_data) {
  auto parse_padding = [](Padding padding) {
    switch (padding) {
      case Padding_SAME:
        return kTfLitePaddingSame;
      case Padding_VALID:
        return kTfLitePaddingValid;
    }
    return kTfLitePaddingUnknown;
  };
  auto parse_activation = [](ActivationFunctionType activation) {
    switch (activation) {
      case ActivationFunctionType_NONE:
        return kTfLiteActNone;
      case ActivationFunctionType_RELU:
        return kTfLiteActRelu;
      case ActivationFunctionType_RELU_N1_TO_1:
        return kTfLiteActRelu1;
      case ActivationFunctionType_RELU6:
        return kTfLiteActRelu6;
      case ActivationFunctionType_TANH:
        return kTfLiteActTanh;
      case ActivationFunctionType_SIGN_BIT:
        return kTfLiteActSignBit;
    }
    return kTfLiteActNone;
  };
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
      auto params = safe_allocator.Allocate<TfLiteConvParams>();
      if (auto* conv_params = op->builtin_options_as_Conv2DOptions()) {
        params->padding = parse_padding(conv_params->padding());
        params->stride_width = conv_params->stride_w();
        params->stride_height = conv_params->stride_h();
        params->activation =
            parse_activation(conv_params->fused_activation_function());

        params->dilation_width_factor = conv_params->dilation_w_factor();
        params->dilation_height_factor = conv_params->dilation_h_factor();
      }
      *builtin_data = reinterpret_cast<void*>(params.release());
      break;
    }
    case BuiltinOperator_CAST: {
      auto params = safe_allocator.Allocate<TfLiteCastParams>();
      if (const auto* schema_params = op->builtin_options_as_CastOptions()) {
        auto in_status =
            ConvertTensorType(schema_params->in_data_type(),
                              &params->in_data_type, error_reporter);
        auto out_status =
            ConvertTensorType(schema_params->out_data_type(),
                              &params->out_data_type, error_reporter);
        if (in_status != kTfLiteOk || out_status != kTfLiteOk) {
          return kTfLiteError;
        }
      }
      *builtin_data = reinterpret_cast<void*>(params.release());
      break;
    }
    case BuiltinOperator_LSH_PROJECTION: {
      auto params = safe_allocator.Allocate<TfLiteLSHProjectionParams>();
      if (const auto* lshParams =
              op->builtin_options_as_LSHProjectionOptions()) {
        params->type = parseLSHProjectionType(lshParams->type());
      }
      *builtin_data = reinterpret_cast<void*>(params.release());
      break;
    }
    case BuiltinOperator_AVERAGE_POOL_2D:
    case BuiltinOperator_MAX_POOL_2D:
    case BuiltinOperator_L2_POOL_2D: {
      auto params = safe_allocator.Allocate<TfLitePoolParams>();
      if (const auto* pool_params = op->builtin_options_as_Pool2DOptions()) {
        params->padding = parse_padding(pool_params->padding());
        params->stride_width = pool_params->stride_w();
        params->stride_height = pool_params->stride_h();
        params->filter_width = pool_params->filter_width();
        params->filter_height = pool_params->filter_height();
        params->activation =
            parse_activation(pool_params->fused_activation_function());
      }
      *builtin_data = reinterpret_cast<void*>(params.release());
      break;
    }
    case BuiltinOperator_DEPTHWISE_CONV_2D: {
      auto params = safe_allocator.Allocate<TfLiteDepthwiseConvParams>();
      if (const auto* conv_params =
              op->builtin_options_as_DepthwiseConv2DOptions()) {
        params->padding = parse_padding(conv_params->padding());
        params->stride_width = conv_params->stride_w();
        params->stride_height = conv_params->stride_h();
        params->depth_multiplier = conv_params->depth_multiplier();
        params->activation =
            parse_activation(conv_params->fused_activation_function());

        params->dilation_width_factor = conv_params->dilation_w_factor();
        params->dilation_height_factor = conv_params->dilation_h_factor();
      }
      *builtin_data = reinterpret_cast<void*>(params.release());
      break;
    }
    case BuiltinOperator_SVDF: {
      auto params = safe_allocator.Allocate<TfLiteSVDFParams>();
      if (const auto* svdf_params = op->builtin_options_as_SVDFOptions()) {
        params->rank = svdf_params->rank();
        params->activation =
            parse_activation(svdf_params->fused_activation_function());
      }
      *builtin_data = reinterpret_cast<void*>(params.release());
      break;
    }
    case BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_RNN: {
      auto params = safe_allocator.Allocate<TfLiteSequenceRNNParams>();
      if (const auto* sequence_rnn_params =
              op->builtin_options_as_SequenceRNNOptions()) {
        params->activation =
            parse_activation(sequence_rnn_params->fused_activation_function());
        params->time_major = sequence_rnn_params->time_major();
      }
      *builtin_data = reinterpret_cast<void*>(params.release());
      break;
    }
    case BuiltinOperator_BIDIRECTIONAL_SEQUENCE_RNN: {
      auto params =
          safe_allocator.Allocate<TfLiteBidirectionalSequenceRNNParams>();
      if (const auto* bidi_sequence_rnn_params =
              op->builtin_options_as_BidirectionalSequenceRNNOptions()) {
        params->activation = parse_activation(
            bidi_sequence_rnn_params->fused_activation_function());
        params->time_major = bidi_sequence_rnn_params->time_major();
        params->merge_outputs = bidi_sequence_rnn_params->merge_outputs();
      }
      *builtin_data = reinterpret_cast<void*>(params.release());
      break;
    }
    case BuiltinOperator_RNN: {
      auto params = safe_allocator.Allocate<TfLiteRNNParams>();
      if (const auto* rnn_params = op->builtin_options_as_RNNOptions()) {
        params->activation =
            parse_activation(rnn_params->fused_activation_function());
      }
      *builtin_data = reinterpret_cast<void*>(params.release());
      break;
    }
    case BuiltinOperator_EMBEDDING_LOOKUP_SPARSE: {
      auto params =
          safe_allocator.Allocate<TfLiteEmbeddingLookupSparseParams>();
      if (const auto* embedding_params =
              op->builtin_options_as_EmbeddingLookupSparseOptions()) {
        params->combiner = parseCombinerType(embedding_params->combiner());
      }
      *builtin_data = reinterpret_cast<void*>(params.release());
      break;
    }
    case BuiltinOperator_FULLY_CONNECTED: {
      auto params = safe_allocator.Allocate<TfLiteFullyConnectedParams>();
      if (const auto* fully_connected_params =
              op->builtin_options_as_FullyConnectedOptions()) {
        params->activation = parse_activation(
            fully_connected_params->fused_activation_function());
        params->keep_num_dims = fully_connected_params->keep_num_dims();
        switch (fully_connected_params->weights_format()) {
          case FullyConnectedOptionsWeightsFormat_DEFAULT:
            params->weights_format = kTfLiteFullyConnectedWeightsFormatDefault;
            break;
          case FullyConnectedOptionsWeightsFormat_SHUFFLED4x16INT8:
            params->weights_format =
                kTfLiteFullyConnectedWeightsFormatShuffled4x16Int8;
            break;
          default:
            error_reporter->Report("Unhandled fully-connected weights format.");
            return kTfLiteError;
        }
      }
      *builtin_data = reinterpret_cast<void*>(params.release());
      break;
    }
    case BuiltinOperator_HASHTABLE_LOOKUP:
      // no-op.
      break;
    case BuiltinOperator_SOFTMAX: {
      auto params = safe_allocator.Allocate<TfLiteSoftmaxParams>();
      if (const auto* softmax_params =
              op->builtin_options_as_SoftmaxOptions()) {
        params->beta = softmax_params->beta();
      }
      *builtin_data = reinterpret_cast<void*>(params.release());
      break;
    }
    case BuiltinOperator_CONCATENATION: {
      auto params = safe_allocator.Allocate<TfLiteConcatenationParams>();
      if (const auto* concatenation_params =
              op->builtin_options_as_ConcatenationOptions()) {
        params->activation =
            parse_activation(concatenation_params->fused_activation_function());
        params->axis = concatenation_params->axis();
      }
      *builtin_data = reinterpret_cast<void*>(params.release());
      break;
    }
    case BuiltinOperator_MUL: {
      auto params = safe_allocator.Allocate<TfLiteMulParams>();
      if (const auto* schema_params = op->builtin_options_as_MulOptions()) {
        params->activation =
            parse_activation(schema_params->fused_activation_function());
      }
      *builtin_data = reinterpret_cast<void*>(params.release());
      break;
    }
    case BuiltinOperator_ADD: {
      auto params = safe_allocator.Allocate<TfLiteAddParams>();
      if (const auto* schema_params = op->builtin_options_as_AddOptions()) {
        params->activation =
            parse_activation(schema_params->fused_activation_function());
      }
      *builtin_data = reinterpret_cast<void*>(params.release());
      break;
    }
    case BuiltinOperator_DIV: {
      auto params = safe_allocator.Allocate<TfLiteDivParams>();
      if (const auto* schema_params = op->builtin_options_as_DivOptions()) {
        params->activation =
            parse_activation(schema_params->fused_activation_function());
      }
      *builtin_data = reinterpret_cast<void*>(params.release());
      break;
    }
    case BuiltinOperator_SUB: {
      auto params = safe_allocator.Allocate<TfLiteSubParams>();
      if (const auto* schema_params = op->builtin_options_as_SubOptions()) {
        params->activation =
            parse_activation(schema_params->fused_activation_function());
      }
      *builtin_data = reinterpret_cast<void*>(params.release());
      break;
    }
    case BuiltinOperator_L2_NORMALIZATION: {
      auto params = safe_allocator.Allocate<TfLiteL2NormParams>();
      if (const auto* schema_params = op->builtin_options_as_L2NormOptions()) {
        params->activation =
            parse_activation(schema_params->fused_activation_function());
      }
      *builtin_data = reinterpret_cast<void*>(params.release());
      break;
    }
    case BuiltinOperator_LOCAL_RESPONSE_NORMALIZATION: {
      auto params = safe_allocator.Allocate<TfLiteLocalResponseNormParams>();
      if (const auto* schema_params =
              op->builtin_options_as_LocalResponseNormalizationOptions()) {
        params->radius = schema_params->radius();
        params->bias = schema_params->bias();
        params->alpha = schema_params->alpha();
        params->beta = schema_params->beta();
      }
      *builtin_data = reinterpret_cast<void*>(params.release());
      break;
    }
    case BuiltinOperator_LSTM: {
      auto params = safe_allocator.Allocate<TfLiteLSTMParams>();
      if (const auto* lstm_params = op->builtin_options_as_LSTMOptions()) {
        params->activation =
            parse_activation(lstm_params->fused_activation_function());
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
            error_reporter->Report("Unhandled LSTM kernel type: %d",
                                   lstm_params->kernel_type());
            return kTfLiteError;
        }
      } else {
        error_reporter->Report("No valid LSTM builtin options exist");
        return kTfLiteError;
      }
      *builtin_data = reinterpret_cast<void*>(params.release());
      break;
    }
    case BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM: {
      auto params =
          safe_allocator.Allocate<TfLiteUnidirectionalSequenceLSTMParams>();
      if (const auto* seq_lstm_params =
              op->builtin_options_as_UnidirectionalSequenceLSTMOptions()) {
        params->activation =
            parse_activation(seq_lstm_params->fused_activation_function());
        params->cell_clip = seq_lstm_params->cell_clip();
        params->proj_clip = seq_lstm_params->proj_clip();
        params->time_major = seq_lstm_params->time_major();
      }
      *builtin_data = reinterpret_cast<void*>(params.release());
      break;
    }
    case BuiltinOperator_BIDIRECTIONAL_SEQUENCE_LSTM: {
      auto params =
          safe_allocator.Allocate<TfLiteBidirectionalSequenceLSTMParams>();
      if (const auto* bidi_lstm_params =
              op->builtin_options_as_BidirectionalSequenceLSTMOptions()) {
        params->activation =
            parse_activation(bidi_lstm_params->fused_activation_function());
        params->cell_clip = bidi_lstm_params->cell_clip();
        params->proj_clip = bidi_lstm_params->proj_clip();
        params->merge_outputs = bidi_lstm_params->merge_outputs();
        params->time_major = bidi_lstm_params->time_major();
      }
      *builtin_data = reinterpret_cast<void*>(params.release());
      break;
    }
    case BuiltinOperator_RESIZE_BILINEAR: {
      auto params = safe_allocator.Allocate<TfLiteResizeBilinearParams>();
      if (const auto* schema_params =
              op->builtin_options_as_ResizeBilinearOptions()) {
        params->align_corners = schema_params->align_corners();
        params->half_pixel_centers = schema_params->half_pixel_centers();
      }
      *builtin_data = reinterpret_cast<void*>(params.release());
      break;
    }
    case BuiltinOperator_RESIZE_NEAREST_NEIGHBOR: {
      // Large functions confuse MacOS builds with XCode 8 so a lambda is
      // required to minimize function size. TODO(b/118447267): Simplify
      // ParseOpData function and reduce its length.
      [&]() {
        auto params =
            safe_allocator.Allocate<TfLiteResizeNearestNeighborParams>();
        if (const auto* schema_params =
                op->builtin_options_as_ResizeNearestNeighborOptions()) {
          params->align_corners = schema_params->align_corners();
        }
        *builtin_data = reinterpret_cast<void*>(params.release());
      }();
      break;
    }
    case BuiltinOperator_RESHAPE: {
      auto params = safe_allocator.Allocate<TfLiteReshapeParams>();
      if (const auto* schema_params = op->builtin_options_as_ReshapeOptions()) {
        auto* new_shape = schema_params->new_shape();
        // TODO(b/147203660): We need to figure out when dynamic reshape
        // (new_shape is a tensor) happens, why the option is not a nullptr.
        // But nonethless, we should only copy when new_shape is not a nullptr.
        if (new_shape) {
          TF_LITE_ENSURE_STATUS(FlatBufferIntVectorToArray(
              sizeof(params->shape), new_shape, params->shape, error_reporter,
              "reshape"));
          params->num_dimensions = new_shape->size();
        }
      }
      *builtin_data = reinterpret_cast<void*>(params.release());
      break;
    }
    case BuiltinOperator_SKIP_GRAM: {
      auto params = safe_allocator.Allocate<TfLiteSkipGramParams>();
      if (const auto* skip_gram_params =
              op->builtin_options_as_SkipGramOptions()) {
        params->ngram_size = skip_gram_params->ngram_size();
        params->max_skip_size = skip_gram_params->max_skip_size();
        params->include_all_ngrams = skip_gram_params->include_all_ngrams();
      }
      *builtin_data = reinterpret_cast<void*>(params.release());
      break;
    }
    case BuiltinOperator_SPACE_TO_DEPTH: {
      auto params = safe_allocator.Allocate<TfLiteSpaceToDepthParams>();
      if (const auto* schema_params =
              op->builtin_options_as_SpaceToDepthOptions()) {
        params->block_size = schema_params->block_size();
      }
      *builtin_data = reinterpret_cast<void*>(params.release());
      break;
    }
    case BuiltinOperator_DEPTH_TO_SPACE: {
      auto params = safe_allocator.Allocate<TfLiteDepthToSpaceParams>();
      if (const auto* schema_params =
              op->builtin_options_as_DepthToSpaceOptions()) {
        params->block_size = schema_params->block_size();
      }
      *builtin_data = reinterpret_cast<void*>(params.release());
      break;
    }
    case BuiltinOperator_GATHER: {
      auto params = safe_allocator.Allocate<TfLiteGatherParams>();
      params->axis = 0;
      if (const auto* gather_params = op->builtin_options_as_GatherOptions()) {
        params->axis = gather_params->axis();
      }

      *builtin_data = reinterpret_cast<void*>(params.release());
      break;
    }
    case BuiltinOperator_MEAN:
    case BuiltinOperator_REDUCE_MAX:
    case BuiltinOperator_REDUCE_MIN:
    case BuiltinOperator_REDUCE_PROD:
    case BuiltinOperator_REDUCE_ANY:
    case BuiltinOperator_SUM: {
      auto params = safe_allocator.Allocate<TfLiteReducerParams>();
      if (const auto* schema_params = op->builtin_options_as_ReducerOptions()) {
        params->keep_dims = schema_params->keep_dims();
      }
      *builtin_data = reinterpret_cast<void*>(params.release());
      break;
    }
    case BuiltinOperator_SPLIT: {
      auto params = safe_allocator.Allocate<TfLiteSplitParams>();
      if (const auto* schema_params = op->builtin_options_as_SplitOptions()) {
        params->num_splits = schema_params->num_splits();
      }
      *builtin_data = reinterpret_cast<void*>(params.release());
      break;
    }
    case BuiltinOperator_SPLIT_V: {
      auto params = safe_allocator.Allocate<TfLiteSplitParams>();
      if (const auto* schema_params = op->builtin_options_as_SplitVOptions()) {
        params->num_splits = schema_params->num_splits();
      }
      *builtin_data = reinterpret_cast<void*>(params.release());
      break;
    }
    case BuiltinOperator_SQUEEZE: {
      auto params = safe_allocator.Allocate<TfLiteSqueezeParams>();
      if (const auto* schema_params = op->builtin_options_as_SqueezeOptions()) {
        const auto& squeeze_dims = schema_params->squeeze_dims();
        TF_LITE_ENSURE_STATUS(FlatBufferIntVectorToArray(
            sizeof(params->squeeze_dims), squeeze_dims, params->squeeze_dims,
            error_reporter, "squeeze"));
        params->num_squeeze_dims = squeeze_dims->size();
      }
      *builtin_data = reinterpret_cast<void*>(params.release());
      break;
    }
    case BuiltinOperator_STRIDED_SLICE: {
      auto params = safe_allocator.Allocate<TfLiteStridedSliceParams>();
      if (const auto* schema_params =
              op->builtin_options_as_StridedSliceOptions()) {
        params->begin_mask = schema_params->begin_mask();
        params->end_mask = schema_params->end_mask();
        params->ellipsis_mask = schema_params->ellipsis_mask();
        params->new_axis_mask = schema_params->new_axis_mask();
        params->shrink_axis_mask = schema_params->shrink_axis_mask();
      }
      *builtin_data = reinterpret_cast<void*>(params.release());
      break;
    }
    case BuiltinOperator_ARG_MAX: {
      auto params = safe_allocator.Allocate<TfLiteArgMaxParams>();
      if (const auto* schema_params = op->builtin_options_as_ArgMaxOptions()) {
        ConvertTensorType(schema_params->output_type(), &params->output_type,
                          error_reporter);
      }
      *builtin_data = reinterpret_cast<void*>(params.release());
      break;
    }
    case BuiltinOperator_ARG_MIN: {
      auto params = safe_allocator.Allocate<TfLiteArgMinParams>();
      if (const auto* schema_params = op->builtin_options_as_ArgMinOptions()) {
        ConvertTensorType(schema_params->output_type(), &params->output_type,
                          error_reporter);
      }
      *builtin_data = reinterpret_cast<void*>(params.release());
      break;
    }
    case BuiltinOperator_TRANSPOSE_CONV: {
      auto params = safe_allocator.Allocate<TfLiteTransposeConvParams>();
      if (const auto* transpose_conv_params =
              op->builtin_options_as_TransposeConvOptions()) {
        params->padding = parse_padding(transpose_conv_params->padding());
        params->stride_width = transpose_conv_params->stride_w();
        params->stride_height = transpose_conv_params->stride_h();
      }
      *builtin_data = reinterpret_cast<void*>(params.release());
      break;
    }
    case BuiltinOperator_SPARSE_TO_DENSE: {
      auto params = safe_allocator.Allocate<TfLiteSparseToDenseParams>();
      if (const auto* sparse_to_dense_params =
              op->builtin_options_as_SparseToDenseOptions()) {
        params->validate_indices = sparse_to_dense_params->validate_indices();
      }
      *builtin_data = reinterpret_cast<void*>(params.release());
      break;
    }
    case BuiltinOperator_SHAPE: {
      auto params = safe_allocator.Allocate<TfLiteShapeParams>();
      if (const auto* schema_params = op->builtin_options_as_ShapeOptions()) {
        ConvertTensorType(schema_params->out_type(), &params->out_type,
                          error_reporter);
      }
      *builtin_data = static_cast<void*>(params.release());
      break;
    }
    case BuiltinOperator_PACK: {
      auto params = safe_allocator.Allocate<TfLitePackParams>();
      if (const auto* pack_params = op->builtin_options_as_PackOptions()) {
        params->values_count = pack_params->values_count();
        params->axis = pack_params->axis();
      }
      *builtin_data = reinterpret_cast<void*>(params.release());
      break;
    }
    case BuiltinOperator_DELEGATE: {
      // TODO(ycling): Revisit when supporting saving delegated models.
      error_reporter->Report("DELEGATE op shouldn't exist in model.");
      return kTfLiteError;
    }
    case BuiltinOperator_FAKE_QUANT: {
      auto params = safe_allocator.Allocate<TfLiteFakeQuantParams>();
      if (const auto* schema_params =
              op->builtin_options_as_FakeQuantOptions()) {
        params->min = schema_params->min();
        params->max = schema_params->max();
        params->num_bits = schema_params->num_bits();
        params->narrow_range = schema_params->narrow_range();
      }
      *builtin_data = static_cast<void*>(params.release());
      break;
    }
    case BuiltinOperator_ONE_HOT: {
      auto params = safe_allocator.Allocate<TfLiteOneHotParams>();
      if (const auto* schema_params = op->builtin_options_as_OneHotOptions()) {
        params->axis = schema_params->axis();
      }
      *builtin_data = static_cast<void*>(params.release());
      break;
    }
    case BuiltinOperator_UNPACK: {
      auto params = safe_allocator.Allocate<TfLiteUnpackParams>();
      if (const auto* unpack_params = op->builtin_options_as_UnpackOptions()) {
        params->num = unpack_params->num();
        params->axis = unpack_params->axis();
      }
      *builtin_data = reinterpret_cast<void*>(params.release());
      break;
    }
    case BuiltinOperator_LEAKY_RELU: {
      auto params = safe_allocator.Allocate<TfLiteLeakyReluParams>();
      if (const auto* leaky_relu_params =
              op->builtin_options_as_LeakyReluOptions()) {
        params->alpha = leaky_relu_params->alpha();
      }
      *builtin_data = reinterpret_cast<void*>(params.release());
      break;
    }
    case BuiltinOperator_MIRROR_PAD: {
      auto params = safe_allocator.Allocate<TfLiteMirrorPaddingParams>();
      const auto* mirror_pad_params = op->builtin_options_as_MirrorPadOptions();
      if (mirror_pad_params != nullptr) {
        params->mode =
            mirror_pad_params->mode() == tflite::MirrorPadMode_REFLECT
                ? TfLiteMirrorPaddingMode::kTfLiteMirrorPaddingReflect
                : TfLiteMirrorPaddingMode::kTfLiteMirrorPaddingSymmetric;
      }
      *builtin_data = reinterpret_cast<void*>(params.release());
      break;
    }
    case BuiltinOperator_UNIQUE: {
      auto params = safe_allocator.Allocate<TfLiteUniqueParams>();
      const auto* unique_params = op->builtin_options_as_UniqueOptions();
      if (unique_params != nullptr) {
        params->index_out_type =
            unique_params->idx_out_type() == tflite::TensorType_INT64
                ? TfLiteType::kTfLiteInt64
                : TfLiteType::kTfLiteInt32;
      }
      *builtin_data = reinterpret_cast<void*>(params.release());
      break;
    }
    case BuiltinOperator_REVERSE_SEQUENCE: {
      auto params = safe_allocator.Allocate<TfLiteReverseSequenceParams>();
      if (const auto* reverse_seq_params =
              op->builtin_options_as_ReverseSequenceOptions()) {
        params->seq_dim = reverse_seq_params->seq_dim();
        params->batch_dim = reverse_seq_params->batch_dim();
      }
      *builtin_data = reinterpret_cast<void*>(params.release());
      break;
    }
    case BuiltinOperator_IF: {
      TfLiteIfParams* params = allocator->AllocatePOD<TfLiteIfParams>();
      if (const auto* if_params = op->builtin_options_as_IfOptions()) {
        params->then_subgraph_index = if_params->then_subgraph_index();
        params->else_subgraph_index = if_params->else_subgraph_index();
      }
      *builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_WHILE: {
      TfLiteWhileParams* params = allocator->AllocatePOD<TfLiteWhileParams>();
      if (const auto* while_params = op->builtin_options_as_WhileOptions()) {
        params->cond_subgraph_index = while_params->cond_subgraph_index();
        params->body_subgraph_index = while_params->body_subgraph_index();
      }
      *builtin_data = reinterpret_cast<void*>(params);
      break;
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
    case BuiltinOperator_DEQUANTIZE:
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
    case BuiltinOperator_QUANTIZE:
    case BuiltinOperator_NON_MAX_SUPPRESSION_V4:
    case BuiltinOperator_NON_MAX_SUPPRESSION_V5:
    case BuiltinOperator_SCATTER_ND:
    case BuiltinOperator_DENSIFY:
    case BuiltinOperator_SEGMENT_SUM:
      break;
  }
  return kTfLiteOk;
}  // NOLINT[readability/fn_size]

}  // namespace tflite
