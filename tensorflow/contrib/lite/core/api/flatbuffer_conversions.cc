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

#include "tensorflow/contrib/lite/core/api/flatbuffer_conversions.h"

#include <cstdlib>

#include "tensorflow/contrib/lite/c/builtin_op_data.h"

namespace tflite {

namespace {

// Copies the contents from the flatbuffer int vector `flatbuffer` into the
// int array `buffer`. `flat_vector` and `buffer` represent the same
// configuration operation for a given operation.
void FlatBufferIntVectorToArray(int max_size_of_buffer,
                                const flatbuffers::Vector<int32_t>* flat_vector,
                                int* buffer, ErrorReporter* error_reporter) {
  if (!flat_vector) {
    error_reporter->Report("Input array not provided for operation.\n");
  } else {
    int num_dimensions = flat_vector->Length();
    if (num_dimensions > max_size_of_buffer / sizeof(int)) {
      error_reporter->Report(
          "Found too many dimensions in the operation's input array.\n");
    } else {
      for (int i = 0; i < num_dimensions; ++i) {
        buffer[i] = flat_vector->Get(i);
      }
    }
  }
}

// Allocate a structure using malloc, but make sure the structure is a POD
// structure that doesn't require constructors to run. The reason we do this,
// is that Interpreter's C extension part will take ownership so destructors
// will not be run during deallocation.
template <class T>
T* MallocPOD() {
  static_assert(std::is_pod<T>::value, "Builtin data structure must be POD.");
  return static_cast<T*>(malloc(sizeof(T)));
}

}  // namespace

TfLiteStatus ConvertTensorType(TensorType tensor_type, TfLiteType* type,
                               ErrorReporter* error_reporter) {
  switch (tensor_type) {
    case TensorType_FLOAT32:
      *type = kTfLiteFloat32;
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
    default:
      error_reporter->Report("Unimplemented data type %s (%d) in tensor\n",
                             EnumNameTensorType(tensor_type), tensor_type);
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
                         ErrorReporter* error_reporter, void** builtin_data) {
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

  *builtin_data = nullptr;
  switch (op_type) {
    case BuiltinOperator_CONV_2D: {
      TfLiteConvParams* params = MallocPOD<TfLiteConvParams>();
      if (auto* conv_params = op->builtin_options_as_Conv2DOptions()) {
        params->padding = parse_padding(conv_params->padding());
        params->stride_width = conv_params->stride_w();
        params->stride_height = conv_params->stride_h();
        params->activation =
            parse_activation(conv_params->fused_activation_function());

        params->dilation_width_factor = conv_params->dilation_w_factor();
        params->dilation_height_factor = conv_params->dilation_h_factor();
      }
      *builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_CAST: {
      TfLiteCastParams* params = MallocPOD<TfLiteCastParams>();
      if (auto* schema_params = op->builtin_options_as_CastOptions()) {
        auto in_status =
            ConvertTensorType(schema_params->in_data_type(),
                              &params->in_data_type, error_reporter);
        auto out_status =
            ConvertTensorType(schema_params->out_data_type(),
                              &params->out_data_type, error_reporter);
        if (in_status != kTfLiteOk || out_status != kTfLiteOk) {
          free(params);
          return kTfLiteError;
        }
      }
      *builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_LSH_PROJECTION: {
      TfLiteLSHProjectionParams* params =
          MallocPOD<TfLiteLSHProjectionParams>();
      if (auto* lshParams = op->builtin_options_as_LSHProjectionOptions()) {
        params->type = parseLSHProjectionType(lshParams->type());
      }
      *builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_AVERAGE_POOL_2D:
    case BuiltinOperator_MAX_POOL_2D:
    case BuiltinOperator_L2_POOL_2D: {
      TfLitePoolParams* params = MallocPOD<TfLitePoolParams>();
      if (auto* pool_params = op->builtin_options_as_Pool2DOptions()) {
        params->padding = parse_padding(pool_params->padding());
        params->stride_width = pool_params->stride_w();
        params->stride_height = pool_params->stride_h();
        params->filter_width = pool_params->filter_width();
        params->filter_height = pool_params->filter_height();
        params->activation =
            parse_activation(pool_params->fused_activation_function());
      }
      *builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_DEPTHWISE_CONV_2D: {
      TfLiteDepthwiseConvParams* params =
          MallocPOD<TfLiteDepthwiseConvParams>();
      if (auto* conv_params = op->builtin_options_as_DepthwiseConv2DOptions()) {
        params->padding = parse_padding(conv_params->padding());
        params->stride_width = conv_params->stride_w();
        params->stride_height = conv_params->stride_h();
        params->depth_multiplier = conv_params->depth_multiplier();
        params->activation =
            parse_activation(conv_params->fused_activation_function());

        params->dilation_width_factor = conv_params->dilation_w_factor();
        params->dilation_height_factor = conv_params->dilation_h_factor();
      }
      *builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_SVDF: {
      TfLiteSVDFParams* params = MallocPOD<TfLiteSVDFParams>();
      if (auto* svdf_params = op->builtin_options_as_SVDFOptions()) {
        params->rank = svdf_params->rank();
        params->activation =
            parse_activation(svdf_params->fused_activation_function());
      }
      *builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_BIDIRECTIONAL_SEQUENCE_RNN:
    case BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_RNN: {
      TfLiteSequenceRNNParams* params = MallocPOD<TfLiteSequenceRNNParams>();
      if (auto* sequence_rnn_params =
              op->builtin_options_as_SequenceRNNOptions()) {
        params->activation =
            parse_activation(sequence_rnn_params->fused_activation_function());
        params->time_major = sequence_rnn_params->time_major();
      }
      *builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_RNN: {
      TfLiteRNNParams* params = MallocPOD<TfLiteRNNParams>();
      if (auto* rnn_params = op->builtin_options_as_RNNOptions()) {
        params->activation =
            parse_activation(rnn_params->fused_activation_function());
      }
      *builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_EMBEDDING_LOOKUP_SPARSE: {
      TfLiteEmbeddingLookupSparseParams* params =
          MallocPOD<TfLiteEmbeddingLookupSparseParams>();
      if (auto* embedding_params =
              op->builtin_options_as_EmbeddingLookupSparseOptions()) {
        params->combiner = parseCombinerType(embedding_params->combiner());
      }
      *builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_FULLY_CONNECTED: {
      TfLiteFullyConnectedParams* params =
          MallocPOD<TfLiteFullyConnectedParams>();
      if (auto* fully_connected_params =
              op->builtin_options_as_FullyConnectedOptions()) {
        params->activation = parse_activation(
            fully_connected_params->fused_activation_function());
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
      *builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_HASHTABLE_LOOKUP:
      // no-op.
      break;
    case BuiltinOperator_SOFTMAX: {
      TfLiteSoftmaxParams* params = MallocPOD<TfLiteSoftmaxParams>();
      if (auto* softmax_params = op->builtin_options_as_SoftmaxOptions()) {
        params->beta = softmax_params->beta();
      }
      *builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_CONCATENATION: {
      TfLiteConcatenationParams* params =
          MallocPOD<TfLiteConcatenationParams>();
      if (auto* concatenation_params =
              op->builtin_options_as_ConcatenationOptions()) {
        params->activation =
            parse_activation(concatenation_params->fused_activation_function());
        params->axis = concatenation_params->axis();
      }
      *builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_MUL: {
      auto* params = MallocPOD<TfLiteMulParams>();
      if (auto* schema_params = op->builtin_options_as_MulOptions()) {
        params->activation =
            parse_activation(schema_params->fused_activation_function());
      }
      *builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_ADD: {
      auto* params = MallocPOD<TfLiteAddParams>();
      if (auto* schema_params = op->builtin_options_as_AddOptions()) {
        params->activation =
            parse_activation(schema_params->fused_activation_function());
      }
      *builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_DIV: {
      auto* params = MallocPOD<TfLiteDivParams>();
      if (auto* schema_params = op->builtin_options_as_DivOptions()) {
        params->activation =
            parse_activation(schema_params->fused_activation_function());
      }
      *builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_SUB: {
      auto* params = MallocPOD<TfLiteSubParams>();
      if (auto* schema_params = op->builtin_options_as_SubOptions()) {
        params->activation =
            parse_activation(schema_params->fused_activation_function());
      }
      *builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_L2_NORMALIZATION: {
      auto* params = MallocPOD<TfLiteL2NormParams>();
      if (auto* schema_params = op->builtin_options_as_L2NormOptions()) {
        params->activation =
            parse_activation(schema_params->fused_activation_function());
      }
      *builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_LOCAL_RESPONSE_NORMALIZATION: {
      auto* params = MallocPOD<TfLiteLocalResponseNormParams>();
      if (auto* schema_params =
              op->builtin_options_as_LocalResponseNormalizationOptions()) {
        params->radius = schema_params->radius();
        params->bias = schema_params->bias();
        params->alpha = schema_params->alpha();
        params->beta = schema_params->beta();
      }
      *builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_BIDIRECTIONAL_SEQUENCE_LSTM:
    case BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM:
    case BuiltinOperator_LSTM: {
      TfLiteLSTMParams* params = MallocPOD<TfLiteLSTMParams>();
      if (auto* lstm_params = op->builtin_options_as_LSTMOptions()) {
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
        }
      }
      *builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_RESIZE_BILINEAR: {
      auto* params = MallocPOD<TfLiteResizeBilinearParams>();
      if (auto* schema_params =
              op->builtin_options_as_ResizeBilinearOptions()) {
        params->align_corners = schema_params->align_corners();
      }
      *builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_RESHAPE: {
      auto* params = MallocPOD<TfLiteReshapeParams>();
      if (auto* schema_params = op->builtin_options_as_ReshapeOptions()) {
        auto* new_shape = schema_params->new_shape();
        FlatBufferIntVectorToArray(sizeof(params->shape), new_shape,
                                   params->shape, error_reporter);
        params->num_dimensions = new_shape->Length();
      }
      *builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_SKIP_GRAM: {
      TfLiteSkipGramParams* params = MallocPOD<TfLiteSkipGramParams>();
      if (auto* skip_gram_params = op->builtin_options_as_SkipGramOptions()) {
        params->ngram_size = skip_gram_params->ngram_size();
        params->max_skip_size = skip_gram_params->max_skip_size();
        params->include_all_ngrams = skip_gram_params->include_all_ngrams();
      }
      *builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_SPACE_TO_DEPTH: {
      auto* params = MallocPOD<TfLiteSpaceToDepthParams>();
      if (auto* schema_params = op->builtin_options_as_SpaceToDepthOptions()) {
        params->block_size = schema_params->block_size();
      }
      *builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_GATHER: {
      TfLiteGatherParams* params = MallocPOD<TfLiteGatherParams>();
      params->axis = 0;
      if (auto* gather_params = op->builtin_options_as_GatherOptions()) {
        params->axis = gather_params->axis();
      }

      *builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_MEAN:
    case BuiltinOperator_REDUCE_MAX:
    case BuiltinOperator_REDUCE_MIN:
    case BuiltinOperator_REDUCE_PROD:
    case BuiltinOperator_REDUCE_ANY:
    case BuiltinOperator_SUM: {
      auto* params = MallocPOD<TfLiteReducerParams>();
      if (auto* schema_params = op->builtin_options_as_ReducerOptions()) {
        params->keep_dims = schema_params->keep_dims();
      }
      *builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_SPLIT: {
      auto* params = MallocPOD<TfLiteSplitParams>();
      if (auto* schema_params = op->builtin_options_as_SplitOptions()) {
        params->num_splits = schema_params->num_splits();
      }
      *builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_SQUEEZE: {
      auto* params = MallocPOD<TfLiteSqueezeParams>();
      if (auto* schema_params = op->builtin_options_as_SqueezeOptions()) {
        const auto& squeeze_dims = schema_params->squeeze_dims();
        FlatBufferIntVectorToArray(sizeof(params->squeeze_dims), squeeze_dims,
                                   params->squeeze_dims, error_reporter);
        params->num_squeeze_dims = squeeze_dims->Length();
      }
      *builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_STRIDED_SLICE: {
      auto* params = MallocPOD<TfLiteStridedSliceParams>();
      if (auto* schema_params = op->builtin_options_as_StridedSliceOptions()) {
        params->begin_mask = schema_params->begin_mask();
        params->end_mask = schema_params->end_mask();
        params->ellipsis_mask = schema_params->ellipsis_mask();
        params->new_axis_mask = schema_params->new_axis_mask();
        params->shrink_axis_mask = schema_params->shrink_axis_mask();
      }
      *builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_ARG_MAX: {
      auto* params = MallocPOD<TfLiteArgMaxParams>();
      if (auto* schema_params = op->builtin_options_as_ArgMaxOptions()) {
        ConvertTensorType(schema_params->output_type(), &params->output_type,
                          error_reporter);
      }
      *builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_ARG_MIN: {
      auto* params = MallocPOD<TfLiteArgMinParams>();
      if (const auto* schema_params = op->builtin_options_as_ArgMinOptions()) {
        ConvertTensorType(schema_params->output_type(), &params->output_type,
                          error_reporter);
      }
      *builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_TRANSPOSE_CONV: {
      TfLiteTransposeConvParams* params =
          MallocPOD<TfLiteTransposeConvParams>();
      if (auto* transpose_conv_params =
              op->builtin_options_as_TransposeConvOptions()) {
        params->padding = parse_padding(transpose_conv_params->padding());
        params->stride_width = transpose_conv_params->stride_w();
        params->stride_height = transpose_conv_params->stride_h();
      }
      *builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_SPARSE_TO_DENSE: {
      TfLiteSparseToDenseParams* params =
          MallocPOD<TfLiteSparseToDenseParams>();
      if (auto* sparse_to_dense_params =
              op->builtin_options_as_SparseToDenseOptions()) {
        params->validate_indices = sparse_to_dense_params->validate_indices();
      }
      *builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_SHAPE: {
      auto* params = MallocPOD<TfLiteShapeParams>();
      if (auto* schema_params = op->builtin_options_as_ShapeOptions()) {
        ConvertTensorType(schema_params->out_type(), &params->out_type,
                          error_reporter);
      }
      *builtin_data = static_cast<void*>(params);
      break;
    }
    case BuiltinOperator_PACK: {
      TfLitePackParams* params = MallocPOD<TfLitePackParams>();
      if (auto* pack_params = op->builtin_options_as_PackOptions()) {
        params->values_count = pack_params->values_count();
        params->axis = pack_params->axis();
      }
      *builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_DELEGATE: {
      // TODO(ycling): Revisit when supporting saving delegated models.
      error_reporter->Report("DELEGATE op shouldn't exist in model.");
      return kTfLiteError;
    }
    case BuiltinOperator_FAKE_QUANT: {
      auto* params = MallocPOD<TfLiteFakeQuantParams>();
      if (auto* schema_params = op->builtin_options_as_FakeQuantOptions()) {
        params->min = schema_params->min();
        params->max = schema_params->max();
        params->num_bits = schema_params->num_bits();
        params->narrow_range = schema_params->narrow_range();
      }
      *builtin_data = static_cast<void*>(params);
      break;
    }
    case BuiltinOperator_ONE_HOT: {
      auto* params = MallocPOD<TfLiteOneHotParams>();
      if (auto* schema_params = op->builtin_options_as_OneHotOptions()) {
        params->axis = schema_params->axis();
      }
      *builtin_data = static_cast<void*>(params);
      break;
    }
    case BuiltinOperator_UNPACK: {
      TfLiteUnpackParams* params = MallocPOD<TfLiteUnpackParams>();
      if (auto* unpack_params = op->builtin_options_as_UnpackOptions()) {
        params->num = unpack_params->num();
        params->axis = unpack_params->axis();
      }
      *builtin_data = reinterpret_cast<void*>(params);
      break;
    }

    // Below are the ops with no builtin_data strcture.
    case BuiltinOperator_BATCH_TO_SPACE_ND:
    // TODO(aselle): Implement call in BuiltinOptions, but nullptrs are
    // ok for now, since there is no call implementation either.
    case BuiltinOperator_CALL:
    case BuiltinOperator_CONCAT_EMBEDDINGS:
    case BuiltinOperator_CUSTOM:
    case BuiltinOperator_DEQUANTIZE:
    case BuiltinOperator_EMBEDDING_LOOKUP:
    case BuiltinOperator_EQUAL:
    case BuiltinOperator_EXP:
    case BuiltinOperator_EXPAND_DIMS:
    case BuiltinOperator_FLOOR:
    case BuiltinOperator_GREATER:
    case BuiltinOperator_GREATER_EQUAL:
    case BuiltinOperator_LESS:
    case BuiltinOperator_LESS_EQUAL:
    case BuiltinOperator_LOG:
    case BuiltinOperator_LOGISTIC:
    case BuiltinOperator_LOG_SOFTMAX:
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
    case BuiltinOperator_RSQRT:
    case BuiltinOperator_SELECT:
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
      break;
  }
  return kTfLiteOk;
}  // NOLINT[readability/fn_size]

}  // namespace tflite
