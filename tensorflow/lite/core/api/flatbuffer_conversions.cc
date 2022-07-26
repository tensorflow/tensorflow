/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

// Converts the flatbuffer mirror padding enum to what is used at runtime.
TfLiteMirrorPaddingMode ConvertMirrorPadding(MirrorPadMode padding) {
  switch (padding) {
    case MirrorPadMode_REFLECT:
      return kTfLiteMirrorPaddingReflect;
    case MirrorPadMode_SYMMETRIC:
      return kTfLiteMirrorPaddingSymmetric;
  }
  return kTfLiteMirrorPaddingUnknown;
}

#ifndef TF_LITE_STATIC_MEMORY
TfLiteStatus ParseOpDataTfLite(const Operator* op, BuiltinOperator op_type,
                               ErrorReporter* error_reporter,
                               BuiltinDataAllocator* allocator,
                               void** builtin_data) {
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
    case BuiltinOperator_ABS: {
      return ParseAbs(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_ADD: {
      return ParseAdd(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_ADD_N: {
      return ParseAddN(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_ARG_MAX: {
      return ParseArgMax(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_ARG_MIN: {
      return ParseArgMin(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_ASSIGN_VARIABLE: {
      return ParseAssignVariable(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_AVERAGE_POOL_2D: {
      return ParsePool(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_BATCH_MATMUL: {
      return ParseBatchMatMul(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_BATCH_TO_SPACE_ND: {
      return ParseBatchToSpaceNd(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_BROADCAST_ARGS: {
      return ParseBroadcastArgs(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_BROADCAST_TO: {
      return ParseBroadcastTo(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_CALL_ONCE: {
      return ParseCallOnce(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_CEIL: {
      return ParseCeil(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_CONCATENATION: {
      return ParseConcatenation(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_CONV_2D: {
      return ParseConv2D(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_CUMSUM: {
      return ParseCumsum(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_DEPTH_TO_SPACE: {
      return ParseDepthToSpace(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_DEPTHWISE_CONV_2D: {
      return ParseDepthwiseConv2D(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_DEQUANTIZE: {
      return ParseDequantize(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_DIV: {
      return ParseDiv(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_ELU: {
      return ParseElu(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_EXP: {
      return ParseExp(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_EXPAND_DIMS: {
      return ParseExpandDims(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_FILL: {
      return ParseFill(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_FLOOR: {
      return ParseFloor(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_FLOOR_DIV: {
      return ParseFloorDiv(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_FLOOR_MOD: {
      return ParseFloorMod(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_FULLY_CONNECTED: {
      return ParseFullyConnected(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_GATHER_ND: {
      return ParseGatherNd(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_GREATER: {
      return ParseGreater(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_GREATER_EQUAL: {
      return ParseGreaterEqual(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_HARD_SWISH: {
      return ParseHardSwish(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_L2_NORMALIZATION: {
      return ParseL2Normalization(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_L2_POOL_2D: {
      return ParsePool(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_LEAKY_RELU: {
      return ParseLeakyRelu(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_LESS: {
      return ParseLess(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_LESS_EQUAL: {
      return ParseLessEqual(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_LOG: {
      return ParseLog(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_LOGICAL_AND: {
      return ParseLogicalAnd(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_LOGICAL_NOT: {
      return ParseLogicalNot(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_LOGICAL_OR: {
      return ParseLogicalOr(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_LOGISTIC: {
      return ParseLogistic(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_LOG_SOFTMAX: {
      return ParseLogSoftmax(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_LSTM: {
      return ParseLSTM(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_MAXIMUM: {
      return ParseMaximum(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_MAX_POOL_2D: {
      return ParsePool(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_MIRROR_PAD: {
      return ParseMirrorPad(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_MEAN: {
      return ParseReducer(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_MINIMUM: {
      return ParseMinimum(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_MUL: {
      return ParseMul(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_NEG: {
      return ParseNeg(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_NOT_EQUAL: {
      return ParseNotEqual(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_PACK: {
      return ParsePack(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_PAD: {
      return ParsePad(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_PADV2: {
      return ParsePadV2(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_POW: {
      return ParsePow(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_PRELU: {
      return ParsePrelu(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_QUANTIZE: {
      return ParseQuantize(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_READ_VARIABLE: {
      return ParseReadVariable(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_REDUCE_ANY: {
      return ParseReducer(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_REDUCE_ALL: {
      return ParseReducer(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_REDUCE_MAX: {
      return ParseReducer(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_REDUCE_MIN: {
      return ParseReducer(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_REDUCE_PROD: {
      return ParseReducer(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_RELU: {
      return ParseRelu(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_RELU6: {
      return ParseRelu6(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_RESHAPE: {
      return ParseReshape(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_RESIZE_BILINEAR: {
      return ParseResizeBilinear(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_RESIZE_NEAREST_NEIGHBOR: {
      return ParseResizeNearestNeighbor(op, error_reporter, allocator,
                                        builtin_data);
    }

    case BuiltinOperator_ROUND: {
      return ParseRound(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_RSQRT: {
      return ParseRsqrt(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_SHAPE: {
      return ParseShape(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_SIN: {
      return ParseSin(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_SOFTMAX: {
      return ParseSoftmax(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_SPACE_TO_BATCH_ND: {
      return ParseSpaceToBatchNd(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_SPACE_TO_DEPTH: {
      return ParseSpaceToDepth(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_SPLIT: {
      return ParseSplit(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_SPLIT_V: {
      return ParseSplitV(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_SQRT: {
      return ParseSqrt(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_SQUARE: {
      return ParseSquare(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_SQUARED_DIFFERENCE: {
      return ParseSquaredDifference(op, error_reporter, allocator,
                                    builtin_data);
    }

    case BuiltinOperator_SQUEEZE: {
      return ParseSqueeze(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_STRIDED_SLICE: {
      return ParseStridedSlice(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_SUB: {
      return ParseSub(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_SUM: {
      return ParseReducer(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_SVDF: {
      return ParseSvdf(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_TANH: {
      return ParseTanh(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_TRANSPOSE_CONV: {
      return ParseTransposeConv(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_UNPACK: {
      return ParseUnpack(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_VAR_HANDLE: {
      return ParseVarHandle(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_ZEROS_LIKE: {
      return ParseZerosLike(op, error_reporter, allocator, builtin_data);
    }

    case BuiltinOperator_CAST: {
      return ParseCast(op, error_reporter, allocator, builtin_data);
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
    case BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM: {
      return ParseUnidirectionalSequenceLSTM(op, error_reporter, allocator,
                                             builtin_data);
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

    case BuiltinOperator_GATHER: {
      return ParseGather(op, error_reporter, allocator, builtin_data);
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
    case BuiltinOperator_DELEGATE: {
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
    case BuiltinOperator_CONV_3D:
    case BuiltinOperator_CONV_3D_TRANSPOSE: {
      auto params = safe_allocator.Allocate<TfLiteConv3DParams>();
      TF_LITE_ENSURE(error_reporter, params != nullptr);
      if (const auto* conv3d_params = op->builtin_options_as_Conv3DOptions()) {
        params->padding = ConvertPadding(conv3d_params->padding());
        params->activation =
            ConvertActivation(conv3d_params->fused_activation_function());
        params->stride_depth = conv3d_params->stride_d();
        params->stride_height = conv3d_params->stride_h();
        params->stride_width = conv3d_params->stride_w();
        params->dilation_depth_factor = conv3d_params->dilation_d_factor();
        params->dilation_height_factor = conv3d_params->dilation_h_factor();
        params->dilation_width_factor = conv3d_params->dilation_w_factor();
      }
      *builtin_data = params.release();
      return kTfLiteOk;
    }
    case BuiltinOperator_HASHTABLE: {
      auto params = safe_allocator.Allocate<TfLiteHashtableParams>();
      TF_LITE_ENSURE(error_reporter, params != nullptr);
      if (const auto* hashtable_params =
              op->builtin_options_as_HashtableOptions()) {
        params->table_id = hashtable_params->table_id();
        TF_LITE_ENSURE_STATUS(ConvertTensorType(
            hashtable_params->key_dtype(), &params->key_dtype, error_reporter));
        TF_LITE_ENSURE_STATUS(ConvertTensorType(hashtable_params->value_dtype(),
                                                &params->value_dtype,
                                                error_reporter));
      }
      *builtin_data = params.release();
      return kTfLiteOk;
    }
    case BuiltinOperator_MULTINOMIAL: {
      auto params = safe_allocator.Allocate<TfLiteRandomParams>();
      TF_LITE_ENSURE(error_reporter, params != nullptr);
      if (const auto* multinomial_params =
              op->builtin_options_as_RandomOptions()) {
        params->seed = multinomial_params->seed();
        params->seed2 = multinomial_params->seed2();
      }
      *builtin_data = params.release();
      return kTfLiteOk;
    }
    case BuiltinOperator_RANDOM_STANDARD_NORMAL: {
      auto params = safe_allocator.Allocate<TfLiteRandomParams>();
      TF_LITE_ENSURE(error_reporter, params != nullptr);
      if (const auto* random_std_normal_params =
              op->builtin_options_as_RandomOptions()) {
        params->seed = random_std_normal_params->seed();
        params->seed2 = random_std_normal_params->seed2();
      }
      *builtin_data = params.release();
      return kTfLiteOk;
    }
    case BuiltinOperator_BUCKETIZE: {
      auto params = safe_allocator.Allocate<TfLiteBucketizeParams>();
      TF_LITE_ENSURE(error_reporter, params != nullptr);
      if (const auto* bucketize_params =
              op->builtin_options_as_BucketizeOptions()) {
        const flatbuffers::Vector<float>* boundaries =
            bucketize_params->boundaries();
        if (boundaries == nullptr) {
          TF_LITE_REPORT_ERROR(
              error_reporter,
              "boundaries array not provided for operation 'bucketize'.\n");
          return kTfLiteError;
        }
        params->num_boundaries = boundaries->size();
        if (boundaries->data() == nullptr) {
          TF_LITE_REPORT_ERROR(error_reporter,
                               "boundaries.data() returned nullptr for "
                               "operation 'bucketize'.\n");
          return kTfLiteError;
        }
        params->boundaries = boundaries->data();
      }
      *builtin_data = params.release();
      return kTfLiteOk;
    }
    case BuiltinOperator_RANDOM_UNIFORM: {
      auto params = safe_allocator.Allocate<TfLiteRandomParams>();
      TF_LITE_ENSURE(error_reporter, params != nullptr);
      if (const auto* random_uniform_params =
              op->builtin_options_as_RandomOptions()) {
        params->seed = random_uniform_params->seed();
        params->seed2 = random_uniform_params->seed2();
      }
      *builtin_data = params.release();
      return kTfLiteOk;
    }
    case BuiltinOperator_GELU: {
      auto params = safe_allocator.Allocate<TfLiteGeluParams>();
      TF_LITE_ENSURE(error_reporter, params != nullptr);
      if (const auto* gelu_params = op->builtin_options_as_GeluOptions()) {
        params->approximate = gelu_params->approximate();
      }
      *builtin_data = params.release();
      return kTfLiteOk;
    }
    // Below are the ops with no builtin_data structure.
    // TODO(aselle): Implement call in BuiltinOptions, but nullptrs are
    // ok for now, since there is no call implementation either.
    case BuiltinOperator_CALL:
    case BuiltinOperator_COMPLEX_ABS:
    case BuiltinOperator_CONCAT_EMBEDDINGS:
    case BuiltinOperator_COS:
    case BuiltinOperator_CUSTOM:
    case BuiltinOperator_DENSIFY:
    case BuiltinOperator_DYNAMIC_UPDATE_SLICE:
    case BuiltinOperator_EMBEDDING_LOOKUP:
    case BuiltinOperator_EQUAL:
    case BuiltinOperator_HASHTABLE_FIND:
    case BuiltinOperator_HASHTABLE_IMPORT:
    case BuiltinOperator_HASHTABLE_SIZE:
    case BuiltinOperator_IMAG:
    case BuiltinOperator_MATRIX_DIAG:
    case BuiltinOperator_MATRIX_SET_DIAG:
    case BuiltinOperator_NON_MAX_SUPPRESSION_V4:
    case BuiltinOperator_NON_MAX_SUPPRESSION_V5:
    case BuiltinOperator_RELU_N1_TO_1:
    case BuiltinOperator_RELU_0_TO_1:
    case BuiltinOperator_SCATTER_ND:
    case BuiltinOperator_SELECT:
    case BuiltinOperator_SELECT_V2:
    case BuiltinOperator_SLICE:
    case BuiltinOperator_TILE:
    case BuiltinOperator_TOPK_V2:
    case BuiltinOperator_TRANSPOSE:
    case BuiltinOperator_RANGE:
    case BuiltinOperator_RANK:
    case BuiltinOperator_REAL:
    case BuiltinOperator_RFFT2D:
    case BuiltinOperator_SEGMENT_SUM:
    case BuiltinOperator_REVERSE_V2:
    case BuiltinOperator_UNSORTED_SEGMENT_MAX:
    case BuiltinOperator_UNSORTED_SEGMENT_PROD:
    case BuiltinOperator_UNSORTED_SEGMENT_SUM:
    case BuiltinOperator_ATAN2:
    case BuiltinOperator_SIGN:
    case BuiltinOperator_WHERE:
      return kTfLiteOk;
    case BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES:
      return kTfLiteError;
  }
  return kTfLiteError;
}  // NOLINT[readability/fn_size]
#endif  // !defined(TF_LITE_STATIC_MEMORY)
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
    case TensorType_UINT16:
      *type = kTfLiteUInt16;
      return kTfLiteOk;
    case TensorType_INT32:
      *type = kTfLiteInt32;
      return kTfLiteOk;
    case TensorType_UINT32:
      *type = kTfLiteUInt32;
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
    case TensorType_UINT64:
      *type = kTfLiteUInt64;
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
    case TensorType_COMPLEX128:
      *type = kTfLiteComplex128;
      return kTfLiteOk;
    case TensorType_RESOURCE:
      *type = kTfLiteResource;
      return kTfLiteOk;
    case TensorType_VARIANT:
      *type = kTfLiteVariant;
      return kTfLiteOk;
    default:
      *type = kTfLiteNoType;
      TF_LITE_REPORT_ERROR(error_reporter,
                           "Unsupported data type %d in tensor\n", tensor_type);
      return kTfLiteError;
  }
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseAbs(const Operator*, ErrorReporter*, BuiltinDataAllocator*,
                      void**) {
  return kTfLiteOk;
}

TfLiteStatus ParseAdd(const Operator* op, ErrorReporter* error_reporter,
                      BuiltinDataAllocator* allocator, void** builtin_data) {
  CheckParsePointerParams(op, error_reporter, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteAddParams, SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteAddParams>();
  TF_LITE_ENSURE(error_reporter, params != nullptr);

  const AddOptions* schema_params = op->builtin_options_as_AddOptions();

  if (schema_params != nullptr) {
    params->activation =
        ConvertActivation(schema_params->fused_activation_function());
    params->pot_scale_int16 = schema_params->pot_scale_int16();
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better undertand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

TfLiteStatus ParseAddN(const Operator* op, ErrorReporter* error_reporter,
                       BuiltinDataAllocator* allocator, void** builtin_data) {
  return kTfLiteOk;
}

TfLiteStatus ParseArgMax(const Operator* op, ErrorReporter* error_reporter,
                         BuiltinDataAllocator* allocator, void** builtin_data) {
  CheckParsePointerParams(op, error_reporter, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteArgMaxParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteArgMaxParams>();
  TF_LITE_ENSURE(error_reporter, params != nullptr);

  const ArgMaxOptions* schema_params = op->builtin_options_as_ArgMaxOptions();

  if (schema_params != nullptr) {
    TF_LITE_ENSURE_STATUS(ConvertTensorType(
        schema_params->output_type(), &params->output_type, error_reporter));
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better undertand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

TfLiteStatus ParseArgMin(const Operator* op, ErrorReporter* error_reporter,
                         BuiltinDataAllocator* allocator, void** builtin_data) {
  CheckParsePointerParams(op, error_reporter, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteArgMinParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteArgMinParams>();
  TF_LITE_ENSURE(error_reporter, params != nullptr);

  const ArgMinOptions* schema_params = op->builtin_options_as_ArgMinOptions();

  if (schema_params != nullptr) {
    TF_LITE_ENSURE_STATUS(ConvertTensorType(
        schema_params->output_type(), &params->output_type, error_reporter));
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
TfLiteStatus ParseAssignVariable(const Operator*, ErrorReporter*,
                                 BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseBatchMatMul(const Operator* op, ErrorReporter* error_reporter,
                              BuiltinDataAllocator* allocator,
                              void** builtin_data) {
  CheckParsePointerParams(op, error_reporter, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  auto params = safe_allocator.Allocate<TfLiteBatchMatMulParams>();
  TF_LITE_ENSURE(error_reporter, params != nullptr);
  if (const auto* bmm_params = op->builtin_options_as_BatchMatMulOptions()) {
    params->adj_x = bmm_params->adj_x();
    params->adj_y = bmm_params->adj_y();
    params->asymmetric_quantize_inputs =
        bmm_params->asymmetric_quantize_inputs();
  }
  *builtin_data = params.release();
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseBatchToSpaceNd(const Operator*, ErrorReporter*,
                                 BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseBroadcastArgs(const Operator*, ErrorReporter*,
                                BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseBroadcastTo(const Operator*, ErrorReporter*,
                              BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

TfLiteStatus ParseCallOnce(const Operator* op, ErrorReporter* error_reporter,
                           BuiltinDataAllocator* allocator,
                           void** builtin_data) {
  CheckParsePointerParams(op, error_reporter, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteCallOnceParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteCallOnceParams>();
  TF_LITE_ENSURE(error_reporter, params != nullptr);

  const CallOnceOptions* schema_params =
      op->builtin_options_as_CallOnceOptions();

  if (schema_params != nullptr) {
    params->init_subgraph_index = schema_params->init_subgraph_index();

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
TfLiteStatus ParseCast(const Operator* op, ErrorReporter* error_reporter,
                       BuiltinDataAllocator* allocator, void** builtin_data) {
  CheckParsePointerParams(op, error_reporter, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  auto params = safe_allocator.Allocate<TfLiteCastParams>();
  TF_LITE_ENSURE(error_reporter, params != nullptr);
  if (const auto* schema_params = op->builtin_options_as_CastOptions()) {
    TF_LITE_ENSURE_STATUS(ConvertTensorType(
        schema_params->in_data_type(), &params->in_data_type, error_reporter));
    TF_LITE_ENSURE_STATUS(ConvertTensorType(schema_params->out_data_type(),
                                            &params->out_data_type,
                                            error_reporter));
  }
  *builtin_data = params.release();
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseCeil(const Operator*, ErrorReporter*, BuiltinDataAllocator*,
                       void**) {
  return kTfLiteOk;
}

TfLiteStatus ParseConcatenation(const Operator* op,
                                ErrorReporter* error_reporter,
                                BuiltinDataAllocator* allocator,
                                void** builtin_data) {
  CheckParsePointerParams(op, error_reporter, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteConcatenationParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteConcatenationParams>();
  TF_LITE_ENSURE(error_reporter, params != nullptr);

  const ConcatenationOptions* schema_params =
      op->builtin_options_as_ConcatenationOptions();

  if (schema_params != nullptr) {
    params->activation =
        ConvertActivation(schema_params->fused_activation_function());
    params->axis = schema_params->axis();
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better undertand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

TfLiteStatus ParseConv2D(const Operator* op, ErrorReporter* error_reporter,
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

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseCumsum(const Operator* op, ErrorReporter* error_reporter,
                         BuiltinDataAllocator* allocator, void** builtin_data) {
  CheckParsePointerParams(op, error_reporter, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  auto params = safe_allocator.Allocate<TfLiteCumsumParams>();
  TF_LITE_ENSURE(error_reporter, params != nullptr);
  if (const auto* cumsum_params = op->builtin_options_as_CumsumOptions()) {
    params->exclusive = cumsum_params->exclusive();
    params->reverse = cumsum_params->reverse();
  }
  *builtin_data = params.release();
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseCos(const Operator*, ErrorReporter*, BuiltinDataAllocator*,
                      void**) {
  return kTfLiteOk;
}

TfLiteStatus ParseDepthToSpace(const Operator* op,
                               ErrorReporter* error_reporter,
                               BuiltinDataAllocator* allocator,
                               void** builtin_data) {
  CheckParsePointerParams(op, error_reporter, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteDepthToSpaceParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteDepthToSpaceParams>();
  TF_LITE_ENSURE(error_reporter, params != nullptr);

  const auto* schema_params = op->builtin_options_as_DepthToSpaceOptions();
  if (schema_params != nullptr) {
    params->block_size = schema_params->block_size();
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better undertand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

TfLiteStatus ParseDepthwiseConv2D(const Operator* op,
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
TfLiteStatus ParseDequantize(const Operator*, ErrorReporter*,
                             BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

TfLiteStatus ParseDiv(const Operator* op, ErrorReporter* error_reporter,
                      BuiltinDataAllocator* allocator, void** builtin_data) {
  CheckParsePointerParams(op, error_reporter, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  auto params = safe_allocator.Allocate<TfLiteDivParams>();
  TF_LITE_ENSURE(error_reporter, params != nullptr);
  if (const auto* schema_params = op->builtin_options_as_DivOptions()) {
    params->activation =
        ConvertActivation(schema_params->fused_activation_function());
  }
  *builtin_data = params.release();
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseElu(const Operator*, ErrorReporter*, BuiltinDataAllocator*,
                      void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseEqual(const Operator*, ErrorReporter*, BuiltinDataAllocator*,
                        void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseExp(const Operator*, ErrorReporter*, BuiltinDataAllocator*,
                      void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseExpandDims(const Operator*, ErrorReporter*,
                             BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseFill(const Operator*, ErrorReporter*, BuiltinDataAllocator*,
                       void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseFloor(const Operator*, ErrorReporter*, BuiltinDataAllocator*,
                        void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseFloorDiv(const Operator*, ErrorReporter*,
                           BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseFloorMod(const Operator*, ErrorReporter*,
                           BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

TfLiteStatus ParseFullyConnected(const Operator* op,
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

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseGather(const Operator* op, ErrorReporter* error_reporter,
                         BuiltinDataAllocator* allocator, void** builtin_data) {
  CheckParsePointerParams(op, error_reporter, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  auto params = safe_allocator.Allocate<TfLiteGatherParams>();
  TF_LITE_ENSURE(error_reporter, params != nullptr);
  params->axis = 0;
  params->batch_dims = 0;
  if (const auto* gather_params = op->builtin_options_as_GatherOptions()) {
    params->axis = gather_params->axis();
    params->batch_dims = gather_params->batch_dims();
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseGatherNd(const Operator*, ErrorReporter*,
                           BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseGreater(const Operator*, ErrorReporter*,
                          BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseGreaterEqual(const Operator*, ErrorReporter*,
                               BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseHardSwish(const Operator*, ErrorReporter*,
                            BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

TfLiteStatus ParseIf(const Operator* op, ErrorReporter* error_reporter,
                     BuiltinDataAllocator* allocator, void** builtin_data) {
  CheckParsePointerParams(op, error_reporter, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteIfParams, SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteIfParams>();
  TF_LITE_ENSURE(error_reporter, params != nullptr);

  const IfOptions* schema_params = op->builtin_options_as_IfOptions();

  if (schema_params != nullptr) {
    params->then_subgraph_index = schema_params->then_subgraph_index();
    params->else_subgraph_index = schema_params->else_subgraph_index();
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better undertand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

TfLiteStatus ParseL2Normalization(const Operator* op,
                                  ErrorReporter* error_reporter,
                                  BuiltinDataAllocator* allocator,
                                  void** builtin_data) {
  CheckParsePointerParams(op, error_reporter, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteL2NormParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteL2NormParams>();
  TF_LITE_ENSURE(error_reporter, params != nullptr);

  const L2NormOptions* schema_params = op->builtin_options_as_L2NormOptions();

  if (schema_params != nullptr) {
    params->activation =
        ConvertActivation(schema_params->fused_activation_function());
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better undertand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

TfLiteStatus ParseLeakyRelu(const Operator* op, ErrorReporter* error_reporter,
                            BuiltinDataAllocator* allocator,
                            void** builtin_data) {
  CheckParsePointerParams(op, error_reporter, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  auto params = safe_allocator.Allocate<TfLiteLeakyReluParams>();
  TF_LITE_ENSURE(error_reporter, params != nullptr);
  if (const auto* leaky_relu_params =
          op->builtin_options_as_LeakyReluOptions()) {
    params->alpha = leaky_relu_params->alpha();
  }
  *builtin_data = params.release();
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseLess(const Operator*, ErrorReporter*, BuiltinDataAllocator*,
                       void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseLessEqual(const Operator*, ErrorReporter*,
                            BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseLog(const Operator*, ErrorReporter*, BuiltinDataAllocator*,
                      void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseLogicalAnd(const Operator*, ErrorReporter*,
                             BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseLogicalNot(const Operator*, ErrorReporter*,
                             BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseLogicalOr(const Operator*, ErrorReporter*,
                            BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseLogistic(const Operator*, ErrorReporter*,
                           BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseLogSoftmax(const Operator*, ErrorReporter*,
                             BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

TfLiteStatus ParseLSTM(const Operator* op, ErrorReporter* error_reporter,
                       BuiltinDataAllocator* allocator, void** builtin_data) {
  CheckParsePointerParams(op, error_reporter, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
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
        TF_LITE_REPORT_ERROR(error_reporter, "Unhandled LSTM kernel type: %d",
                             lstm_params->kernel_type());
        return kTfLiteError;
    }
    params->asymmetric_quantize_inputs =
        lstm_params->asymmetric_quantize_inputs();
  } else {
    TF_LITE_REPORT_ERROR(error_reporter, "No valid LSTM builtin options exist");
    return kTfLiteError;
  }
  *builtin_data = params.release();
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseMaximum(const Operator*, ErrorReporter*,
                          BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseMinimum(const Operator*, ErrorReporter*,
                          BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

TfLiteStatus ParseMirrorPad(const Operator* op, ErrorReporter* error_reporter,
                            BuiltinDataAllocator* allocator,
                            void** builtin_data) {
  CheckParsePointerParams(op, error_reporter, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteMirrorPaddingParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteMirrorPaddingParams>();
  TF_LITE_ENSURE(error_reporter, params != nullptr);

  const MirrorPadOptions* schema_params =
      op->builtin_options_as_MirrorPadOptions();

  if (schema_params != nullptr) {
    params->mode = ConvertMirrorPadding(schema_params->mode());
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better undertand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

TfLiteStatus ParseMul(const Operator* op, ErrorReporter* error_reporter,
                      BuiltinDataAllocator* allocator, void** builtin_data) {
  CheckParsePointerParams(op, error_reporter, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteMulParams, SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteMulParams>();
  TF_LITE_ENSURE(error_reporter, params != nullptr);

  const MulOptions* schema_params = op->builtin_options_as_MulOptions();

  if (schema_params != nullptr) {
    params->activation =
        ConvertActivation(schema_params->fused_activation_function());
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
TfLiteStatus ParseNeg(const Operator*, ErrorReporter*, BuiltinDataAllocator*,
                      void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseNotEqual(const Operator*, ErrorReporter*,
                           BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

TfLiteStatus ParsePack(const Operator* op, ErrorReporter* error_reporter,
                       BuiltinDataAllocator* allocator, void** builtin_data) {
  CheckParsePointerParams(op, error_reporter, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLitePackParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLitePackParams>();
  TF_LITE_ENSURE(error_reporter, params != nullptr);

  const PackOptions* schema_params = op->builtin_options_as_PackOptions();

  if (schema_params != nullptr) {
    params->values_count = schema_params->values_count();
    params->axis = schema_params->axis();
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
TfLiteStatus ParsePad(const Operator*, ErrorReporter*, BuiltinDataAllocator*,
                      void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParsePadV2(const Operator*, ErrorReporter*, BuiltinDataAllocator*,
                        void**) {
  return kTfLiteOk;
}

TfLiteStatus ParsePool(const Operator* op, ErrorReporter* error_reporter,
                       BuiltinDataAllocator* allocator, void** builtin_data) {
  CheckParsePointerParams(op, error_reporter, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLitePoolParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLitePoolParams>();
  TF_LITE_ENSURE(error_reporter, params != nullptr);

  const Pool2DOptions* schema_params = op->builtin_options_as_Pool2DOptions();

  if (schema_params != nullptr) {
    params->padding = ConvertPadding(schema_params->padding());
    params->stride_width = schema_params->stride_w();
    params->stride_height = schema_params->stride_h();
    params->filter_width = schema_params->filter_width();
    params->filter_height = schema_params->filter_height();
    params->activation =
        ConvertActivation(schema_params->fused_activation_function());
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
TfLiteStatus ParsePow(const Operator*, ErrorReporter*, BuiltinDataAllocator*,
                      void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParsePrelu(const Operator*, ErrorReporter*, BuiltinDataAllocator*,
                        void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseQuantize(const Operator*, ErrorReporter*,
                           BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseReadVariable(const Operator*, ErrorReporter*,
                               BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

TfLiteStatus ParseReducer(const Operator* op, ErrorReporter* error_reporter,
                          BuiltinDataAllocator* allocator,
                          void** builtin_data) {
  CheckParsePointerParams(op, error_reporter, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);

  std::unique_ptr<TfLiteReducerParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteReducerParams>();
  TF_LITE_ENSURE(error_reporter, params != nullptr);

  const ReducerOptions* schema_params = op->builtin_options_as_ReducerOptions();

  if (schema_params != nullptr) {
    params->keep_dims = schema_params->keep_dims();
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
TfLiteStatus ParseRelu(const Operator*, ErrorReporter*, BuiltinDataAllocator*,
                       void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseRelu6(const Operator*, ErrorReporter*, BuiltinDataAllocator*,
                        void**) {
  return kTfLiteOk;
}

TfLiteStatus ParseReshape(const Operator* op, ErrorReporter* error_reporter,
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

TfLiteStatus ParseResizeBilinear(const Operator* op,
                                 ErrorReporter* error_reporter,
                                 BuiltinDataAllocator* allocator,
                                 void** builtin_data) {
  CheckParsePointerParams(op, error_reporter, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteResizeBilinearParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteResizeBilinearParams>();
  TF_LITE_ENSURE(error_reporter, params != nullptr);

  const ResizeBilinearOptions* schema_params =
      op->builtin_options_as_ResizeBilinearOptions();

  if (schema_params != nullptr) {
    params->align_corners = schema_params->align_corners();
    params->half_pixel_centers = schema_params->half_pixel_centers();
  } else {
    params->align_corners = false;
    params->half_pixel_centers = false;
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

TfLiteStatus ParseResizeNearestNeighbor(const Operator* op,
                                        ErrorReporter* error_reporter,
                                        BuiltinDataAllocator* allocator,
                                        void** builtin_data) {
  CheckParsePointerParams(op, error_reporter, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteResizeNearestNeighborParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteResizeNearestNeighborParams>();
  TF_LITE_ENSURE(error_reporter, params != nullptr);

  const ResizeNearestNeighborOptions* schema_params =
      op->builtin_options_as_ResizeNearestNeighborOptions();

  if (schema_params != nullptr) {
    params->align_corners = schema_params->align_corners();
    params->half_pixel_centers = schema_params->half_pixel_centers();
  } else {
    params->align_corners = false;
    params->half_pixel_centers = false;
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseRound(const Operator*, ErrorReporter*, BuiltinDataAllocator*,
                        void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseRsqrt(const Operator*, ErrorReporter*, BuiltinDataAllocator*,
                        void**) {
  return kTfLiteOk;
}

TfLiteStatus ParseShape(const Operator* op, ErrorReporter* error_reporter,
                        BuiltinDataAllocator* allocator, void** builtin_data) {
  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteShapeParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteShapeParams>();
  TF_LITE_ENSURE(error_reporter, params != nullptr);

  const ShapeOptions* schema_params = op->builtin_options_as_ShapeOptions();

  if (schema_params != nullptr) {
    TF_LITE_ENSURE_STATUS(ConvertTensorType(schema_params->out_type(),
                                            &params->out_type, error_reporter));
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
TfLiteStatus ParseSin(const Operator*, ErrorReporter*, BuiltinDataAllocator*,
                      void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseSlice(const Operator*, ErrorReporter*, BuiltinDataAllocator*,
                        void**) {
  return kTfLiteOk;
}

TfLiteStatus ParseSoftmax(const Operator* op, ErrorReporter* error_reporter,
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

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseSpaceToBatchNd(const Operator*, ErrorReporter*,
                                 BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

TfLiteStatus ParseSpaceToDepth(const Operator* op,
                               ErrorReporter* error_reporter,
                               BuiltinDataAllocator* allocator,
                               void** builtin_data) {
  CheckParsePointerParams(op, error_reporter, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteSpaceToDepthParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteSpaceToDepthParams>();
  TF_LITE_ENSURE(error_reporter, params != nullptr);

  const auto* schema_params = op->builtin_options_as_SpaceToDepthOptions();
  if (schema_params != nullptr) {
    params->block_size = schema_params->block_size();
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better undertand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

TfLiteStatus ParseSplit(const Operator* op, ErrorReporter* error_reporter,
                        BuiltinDataAllocator* allocator, void** builtin_data) {
  CheckParsePointerParams(op, error_reporter, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteSplitParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteSplitParams>();
  TF_LITE_ENSURE(error_reporter, params != nullptr);

  const SplitOptions* schema_params = op->builtin_options_as_SplitOptions();

  if (schema_params != nullptr) {
    params->num_splits = schema_params->num_splits();
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better undertand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

TfLiteStatus ParseSplitV(const Operator* op, ErrorReporter* error_reporter,
                         BuiltinDataAllocator* allocator, void** builtin_data) {
  CheckParsePointerParams(op, error_reporter, allocator, builtin_data);
  SafeBuiltinDataAllocator safe_allocator(allocator);

  std::unique_ptr<TfLiteSplitVParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteSplitVParams>();
  TF_LITE_ENSURE(error_reporter, params != nullptr);

  const SplitVOptions* schema_params = op->builtin_options_as_SplitVOptions();

  if (schema_params != nullptr) {
    params->num_splits = schema_params->num_splits();
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better undertand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

TfLiteStatus ParseUnidirectionalSequenceLSTM(const Operator* op,
                                             ErrorReporter* error_reporter,
                                             BuiltinDataAllocator* allocator,
                                             void** builtin_data) {
  CheckParsePointerParams(op, error_reporter, allocator, builtin_data);
  SafeBuiltinDataAllocator safe_allocator(allocator);
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

TfLiteStatus ParseSqueeze(const Operator* op, ErrorReporter* error_reporter,
                          BuiltinDataAllocator* allocator,
                          void** builtin_data) {
  CheckParsePointerParams(op, error_reporter, allocator, builtin_data);
  SafeBuiltinDataAllocator safe_allocator(allocator);

  std::unique_ptr<TfLiteSqueezeParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteSqueezeParams>();
  TF_LITE_ENSURE(error_reporter, params != nullptr);

  const SqueezeOptions* schema_params = op->builtin_options_as_SqueezeOptions();

  if (schema_params != nullptr) {
    const auto* squeeze_dims = schema_params->squeeze_dims();
    if (squeeze_dims != nullptr) {
      TF_LITE_ENSURE_STATUS(FlatBufferIntVectorToArray(
          sizeof(params->squeeze_dims), squeeze_dims, params->squeeze_dims,
          error_reporter, "squeeze"));
      params->num_squeeze_dims = squeeze_dims->size();
    } else {
      params->num_squeeze_dims = 0;
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
TfLiteStatus ParseSqrt(const Operator*, ErrorReporter*, BuiltinDataAllocator*,
                       void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseSquare(const Operator*, ErrorReporter*, BuiltinDataAllocator*,
                         void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseSquaredDifference(const Operator*, ErrorReporter*,
                                    BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

TfLiteStatus ParseStridedSlice(const Operator* op,
                               ErrorReporter* error_reporter,
                               BuiltinDataAllocator* allocator,
                               void** builtin_data) {
  CheckParsePointerParams(op, error_reporter, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteStridedSliceParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteStridedSliceParams>();
  TF_LITE_ENSURE(error_reporter, params != nullptr);

  const StridedSliceOptions* schema_params =
      op->builtin_options_as_StridedSliceOptions();

  if (schema_params != nullptr) {
    params->begin_mask = schema_params->begin_mask();
    params->end_mask = schema_params->end_mask();
    params->ellipsis_mask = schema_params->ellipsis_mask();
    params->new_axis_mask = schema_params->new_axis_mask();
    params->shrink_axis_mask = schema_params->shrink_axis_mask();
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better undertand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

TfLiteStatus ParseSub(const Operator* op, ErrorReporter* error_reporter,
                      BuiltinDataAllocator* allocator, void** builtin_data) {
  CheckParsePointerParams(op, error_reporter, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteSubParams, SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteSubParams>();
  TF_LITE_ENSURE(error_reporter, params != nullptr);

  const SubOptions* schema_params = op->builtin_options_as_SubOptions();

  if (schema_params != nullptr) {
    params->activation =
        ConvertActivation(schema_params->fused_activation_function());
    params->pot_scale_int16 = schema_params->pot_scale_int16();
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better undertand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

TfLiteStatus ParseSvdf(const Operator* op, ErrorReporter* error_reporter,
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

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseTanh(const Operator*, ErrorReporter*, BuiltinDataAllocator*,
                       void**) {
  return kTfLiteOk;
}
//
// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseTranspose(const Operator*, ErrorReporter*,
                            BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

TfLiteStatus ParseTransposeConv(const Operator* op,
                                ErrorReporter* error_reporter,
                                BuiltinDataAllocator* allocator,
                                void** builtin_data) {
  CheckParsePointerParams(op, error_reporter, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteTransposeConvParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteTransposeConvParams>();
  TF_LITE_ENSURE(error_reporter, params != nullptr);
  const TransposeConvOptions* transpose_conv_params =
      op->builtin_options_as_TransposeConvOptions();
  if (transpose_conv_params != nullptr) {
    params->padding = ConvertPadding(transpose_conv_params->padding());
    params->stride_width = transpose_conv_params->stride_w();
    params->stride_height = transpose_conv_params->stride_h();
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better undertand the ramifications of changing the legacy behavior.
  }
  *builtin_data = params.release();
  return kTfLiteOk;
}

TfLiteStatus ParseUnpack(const Operator* op, ErrorReporter* error_reporter,
                         BuiltinDataAllocator* allocator, void** builtin_data) {
  CheckParsePointerParams(op, error_reporter, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteUnpackParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteUnpackParams>();
  TF_LITE_ENSURE(error_reporter, params != nullptr);

  const UnpackOptions* schema_params = op->builtin_options_as_UnpackOptions();

  if (schema_params != nullptr) {
    params->num = schema_params->num();
    params->axis = schema_params->axis();
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better undertand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

TfLiteStatus ParseVarHandle(const Operator* op, ErrorReporter* error_reporter,
                            BuiltinDataAllocator* allocator,
                            void** builtin_data) {
  CheckParsePointerParams(op, error_reporter, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteVarHandleParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteVarHandleParams>();
  TF_LITE_ENSURE(error_reporter, params != nullptr);

  const VarHandleOptions* schema_params =
      op->builtin_options_as_VarHandleOptions();

  if (schema_params != nullptr) {
    if (schema_params->container()) {
      params->container = schema_params->container()->c_str();
    }
    if (schema_params->shared_name()) {
      params->shared_name = schema_params->shared_name()->c_str();
    }
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better undertand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

TfLiteStatus ParseWhile(const Operator* op, ErrorReporter* error_reporter,
                        BuiltinDataAllocator* allocator, void** builtin_data) {
  CheckParsePointerParams(op, error_reporter, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteWhileParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteWhileParams>();
  TF_LITE_ENSURE(error_reporter, params != nullptr);

  const WhileOptions* schema_params = op->builtin_options_as_WhileOptions();

  if (schema_params != nullptr) {
    params->cond_subgraph_index = schema_params->cond_subgraph_index();
    params->body_subgraph_index = schema_params->body_subgraph_index();
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
TfLiteStatus ParseZerosLike(const Operator*, ErrorReporter*,
                            BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

TfLiteStatus ParseOpData(const Operator* op, BuiltinOperator op_type,
                         ErrorReporter* error_reporter,
                         BuiltinDataAllocator* allocator, void** builtin_data) {
// TODO(b/145762662): It would be preferable to have the build graph for TF Lite
// Micro not have the ParseOpData function at all. This would require splitting
// the current file into two separate files, one of which defines the
// ParseOpData function and the other that defines the operator specific parse
// functions (e.g. ParseAdd).
//
// Such a split was attempted but was not worth the effort at the time because
// of the following reasons:
//  * We could either duplicate the functions and the SafeBuiltinDataAllocator
//    class in the anonymous namespace of this file, or attempt to make a common
//    library with these helper functions and class.
//  * Making a common library with a separate build target was not feasible as
//    it introduced circular dependencies due to the ErrorReporter and a common
//    .cc and .h within the same api build target the also cause circular
//    dependencies due to the  BuiltinDataAllocator class.
//  * If all the builtin operators were to have their own parse functions, or we
//    were ok with some amount of code duplication, then this split of the .cc
//    files would be a lot more feasible.
#ifdef TF_LITE_STATIC_MEMORY
  TF_LITE_REPORT_ERROR(
      error_reporter,
      "ParseOpData is unsupported on TfLiteMicro, please use the operator "
      "specific parse functions (e.g. ParseAdd etc.).\n");
  return kTfLiteError;
#else
  return ParseOpDataTfLite(op, op_type, error_reporter, allocator,
                           builtin_data);
#endif
}

}  // namespace tflite
