/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/core/api/flatbuffer_conversions.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "flatbuffers/vector.h"  // from @flatbuffers
#include "tensorflow/compiler/mlir/lite/core/c/builtin_op_data.h"
#include "tensorflow/compiler/mlir/lite/core/c/tflite_types.h"
#include "tensorflow/compiler/mlir/lite/kernels/internal/compatibility_macros.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_generated.h"

/// Check whether the value `a` is true, and if not return
/// absl::InvalidArgumentError from the current function, while also
/// reporting the location of the error.
#define TFL_FILE_ENSURE(a)                                                   \
  do {                                                                       \
    if (!(a)) {                                                              \
      auto error_message =                                                   \
          absl::StrFormat("%s:%d %s was not true.", __FILE__, __LINE__, #a); \
      ABSL_LOG(ERROR) << error_message;                                      \
      return absl::InvalidArgumentError(error_message);                      \
    }                                                                        \
  } while (0)

#define TFL_FILE_ENSURE_STATUS(a) \
  do {                            \
    const absl::Status s = (a);   \
    if (!s.ok()) {                \
      return s;                   \
    }                             \
  } while (0)

namespace tflite_file {
namespace flatbuffer_conversions {
using absl::OkStatus;
using tflite::ActivationFunctionType;
using tflite::ActivationFunctionType_NONE;
using tflite::ActivationFunctionType_RELU;
using tflite::ActivationFunctionType_RELU6;
using tflite::ActivationFunctionType_RELU_N1_TO_1;
using tflite::ActivationFunctionType_SIGN_BIT;
using tflite::ActivationFunctionType_TANH;
using tflite::AddOptions;
using tflite::ArgMaxOptions;
using tflite::ArgMinOptions;
using tflite::BuiltinOperator;
using tflite::BuiltinOperator_ABS;
using tflite::BuiltinOperator_ADD;
using tflite::BuiltinOperator_ADD_N;
using tflite::BuiltinOperator_ARG_MAX;
using tflite::BuiltinOperator_ARG_MIN;
using tflite::BuiltinOperator_ASSIGN_VARIABLE;
using tflite::BuiltinOperator_ATAN2;
using tflite::BuiltinOperator_AVERAGE_POOL_2D;
using tflite::BuiltinOperator_BATCH_MATMUL;
using tflite::BuiltinOperator_BATCH_TO_SPACE_ND;
using tflite::BuiltinOperator_BIDIRECTIONAL_SEQUENCE_LSTM;
using tflite::BuiltinOperator_BIDIRECTIONAL_SEQUENCE_RNN;
using tflite::BuiltinOperator_BITCAST;
using tflite::BuiltinOperator_BITWISE_XOR;
using tflite::BuiltinOperator_BROADCAST_ARGS;
using tflite::BuiltinOperator_BROADCAST_TO;
using tflite::BuiltinOperator_BUCKETIZE;
using tflite::BuiltinOperator_CALL;
using tflite::BuiltinOperator_CALL_ONCE;
using tflite::BuiltinOperator_CAST;
using tflite::BuiltinOperator_CEIL;
using tflite::BuiltinOperator_COMPLEX_ABS;
using tflite::BuiltinOperator_CONCAT_EMBEDDINGS;
using tflite::BuiltinOperator_CONCATENATION;
using tflite::BuiltinOperator_CONV_2D;
using tflite::BuiltinOperator_CONV_3D;
using tflite::BuiltinOperator_CONV_3D_TRANSPOSE;
using tflite::BuiltinOperator_COS;
using tflite::BuiltinOperator_CUMSUM;
using tflite::BuiltinOperator_CUSTOM;
using tflite::BuiltinOperator_DELEGATE;
using tflite::BuiltinOperator_DENSIFY;
using tflite::BuiltinOperator_DEPTH_TO_SPACE;
using tflite::BuiltinOperator_DEPTHWISE_CONV_2D;
using tflite::BuiltinOperator_DEQUANTIZE;
using tflite::BuiltinOperator_DILATE;
using tflite::BuiltinOperator_DIV;
using tflite::BuiltinOperator_DYNAMIC_UPDATE_SLICE;
using tflite::BuiltinOperator_ELU;
using tflite::BuiltinOperator_EMBEDDING_LOOKUP;
using tflite::BuiltinOperator_EMBEDDING_LOOKUP_SPARSE;
using tflite::BuiltinOperator_EQUAL;
using tflite::BuiltinOperator_EXP;
using tflite::BuiltinOperator_EXPAND_DIMS;
using tflite::BuiltinOperator_FAKE_QUANT;
using tflite::BuiltinOperator_FILL;
using tflite::BuiltinOperator_FLOOR;
using tflite::BuiltinOperator_FLOOR_DIV;
using tflite::BuiltinOperator_FLOOR_MOD;
using tflite::BuiltinOperator_FULLY_CONNECTED;
using tflite::BuiltinOperator_GATHER;
using tflite::BuiltinOperator_GATHER_ND;
using tflite::BuiltinOperator_GELU;
using tflite::BuiltinOperator_GREATER;
using tflite::BuiltinOperator_GREATER_EQUAL;
using tflite::BuiltinOperator_HARD_SWISH;
using tflite::BuiltinOperator_HASHTABLE;
using tflite::BuiltinOperator_HASHTABLE_FIND;
using tflite::BuiltinOperator_HASHTABLE_IMPORT;
using tflite::BuiltinOperator_HASHTABLE_LOOKUP;
using tflite::BuiltinOperator_HASHTABLE_SIZE;
using tflite::BuiltinOperator_IF;
using tflite::BuiltinOperator_IMAG;
using tflite::BuiltinOperator_L2_NORMALIZATION;
using tflite::BuiltinOperator_L2_POOL_2D;
using tflite::BuiltinOperator_LEAKY_RELU;
using tflite::BuiltinOperator_LESS;
using tflite::BuiltinOperator_LESS_EQUAL;
using tflite::BuiltinOperator_LOCAL_RESPONSE_NORMALIZATION;
using tflite::BuiltinOperator_LOG;
using tflite::BuiltinOperator_LOG_SOFTMAX;
using tflite::BuiltinOperator_LOGICAL_AND;
using tflite::BuiltinOperator_LOGICAL_NOT;
using tflite::BuiltinOperator_LOGICAL_OR;
using tflite::BuiltinOperator_LOGISTIC;
using tflite::BuiltinOperator_LSH_PROJECTION;
using tflite::BuiltinOperator_LSTM;
using tflite::BuiltinOperator_MATRIX_DIAG;
using tflite::BuiltinOperator_MATRIX_SET_DIAG;
using tflite::BuiltinOperator_MAX_POOL_2D;
using tflite::BuiltinOperator_MAXIMUM;
using tflite::BuiltinOperator_MEAN;
using tflite::BuiltinOperator_MINIMUM;
using tflite::BuiltinOperator_MIRROR_PAD;
using tflite::BuiltinOperator_MUL;
using tflite::BuiltinOperator_MULTINOMIAL;
using tflite::BuiltinOperator_NEG;
using tflite::BuiltinOperator_NON_MAX_SUPPRESSION_V4;
using tflite::BuiltinOperator_NON_MAX_SUPPRESSION_V5;
using tflite::BuiltinOperator_NOT_EQUAL;
using tflite::BuiltinOperator_ONE_HOT;
using tflite::BuiltinOperator_PACK;
using tflite::BuiltinOperator_PAD;
using tflite::BuiltinOperator_PADV2;
using tflite::BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES;
using tflite::BuiltinOperator_POW;
using tflite::BuiltinOperator_PRELU;
using tflite::BuiltinOperator_QUANTIZE;
using tflite::BuiltinOperator_RANDOM_STANDARD_NORMAL;
using tflite::BuiltinOperator_RANDOM_UNIFORM;
using tflite::BuiltinOperator_RANGE;
using tflite::BuiltinOperator_RANK;
using tflite::BuiltinOperator_READ_VARIABLE;
using tflite::BuiltinOperator_REAL;
using tflite::BuiltinOperator_REDUCE_ALL;
using tflite::BuiltinOperator_REDUCE_ANY;
using tflite::BuiltinOperator_REDUCE_MAX;
using tflite::BuiltinOperator_REDUCE_MIN;
using tflite::BuiltinOperator_REDUCE_PROD;
using tflite::BuiltinOperator_REDUCE_WINDOW;
using tflite::BuiltinOperator_RELU;
using tflite::BuiltinOperator_RELU6;
using tflite::BuiltinOperator_RELU_0_TO_1;
using tflite::BuiltinOperator_RELU_N1_TO_1;
using tflite::BuiltinOperator_RESHAPE;
using tflite::BuiltinOperator_RESIZE_BILINEAR;
using tflite::BuiltinOperator_RESIZE_NEAREST_NEIGHBOR;
using tflite::BuiltinOperator_REVERSE_SEQUENCE;
using tflite::BuiltinOperator_REVERSE_V2;
using tflite::BuiltinOperator_RFFT2D;
using tflite::BuiltinOperator_RIGHT_SHIFT;
using tflite::BuiltinOperator_RNN;
using tflite::BuiltinOperator_ROUND;
using tflite::BuiltinOperator_RSQRT;
using tflite::BuiltinOperator_SCATTER_ND;
using tflite::BuiltinOperator_SEGMENT_SUM;
using tflite::BuiltinOperator_SELECT;
using tflite::BuiltinOperator_SELECT_V2;
using tflite::BuiltinOperator_SHAPE;
using tflite::BuiltinOperator_SIGN;
using tflite::BuiltinOperator_SIN;
using tflite::BuiltinOperator_SKIP_GRAM;
using tflite::BuiltinOperator_SLICE;
using tflite::BuiltinOperator_SOFTMAX;
using tflite::BuiltinOperator_SPACE_TO_BATCH_ND;
using tflite::BuiltinOperator_SPACE_TO_DEPTH;
using tflite::BuiltinOperator_SPARSE_TO_DENSE;
using tflite::BuiltinOperator_SPLIT;
using tflite::BuiltinOperator_SPLIT_V;
using tflite::BuiltinOperator_SQRT;
using tflite::BuiltinOperator_SQUARE;
using tflite::BuiltinOperator_SQUARED_DIFFERENCE;
using tflite::BuiltinOperator_SQUEEZE;
using tflite::BuiltinOperator_STABLEHLO_ABS;
using tflite::BuiltinOperator_STABLEHLO_ADD;
using tflite::BuiltinOperator_STABLEHLO_AND;
using tflite::BuiltinOperator_STABLEHLO_BROADCAST_IN_DIM;
using tflite::BuiltinOperator_STABLEHLO_CASE;
using tflite::BuiltinOperator_STABLEHLO_CBRT;
using tflite::BuiltinOperator_STABLEHLO_CLAMP;
using tflite::BuiltinOperator_STABLEHLO_COMPARE;
using tflite::BuiltinOperator_STABLEHLO_COMPOSITE;
using tflite::BuiltinOperator_STABLEHLO_CONCATENATE;
using tflite::BuiltinOperator_STABLEHLO_CONVERT;
using tflite::BuiltinOperator_STABLEHLO_CONVOLUTION;
using tflite::BuiltinOperator_STABLEHLO_COSINE;
using tflite::BuiltinOperator_STABLEHLO_CUSTOM_CALL;
using tflite::BuiltinOperator_STABLEHLO_DIVIDE;
using tflite::BuiltinOperator_STABLEHLO_DOT_GENERAL;
using tflite::BuiltinOperator_STABLEHLO_DYNAMIC_SLICE;
using tflite::BuiltinOperator_STABLEHLO_DYNAMIC_UPDATE_SLICE;
using tflite::BuiltinOperator_STABLEHLO_EXPONENTIAL;
using tflite::BuiltinOperator_STABLEHLO_FLOOR;
using tflite::BuiltinOperator_STABLEHLO_GATHER;
using tflite::BuiltinOperator_STABLEHLO_IOTA;
using tflite::BuiltinOperator_STABLEHLO_LOG;
using tflite::BuiltinOperator_STABLEHLO_LOGISTIC;
using tflite::BuiltinOperator_STABLEHLO_MAXIMUM;
using tflite::BuiltinOperator_STABLEHLO_MINIMUM;
using tflite::BuiltinOperator_STABLEHLO_MULTIPLY;
using tflite::BuiltinOperator_STABLEHLO_NEGATE;
using tflite::BuiltinOperator_STABLEHLO_OR;
using tflite::BuiltinOperator_STABLEHLO_PAD;
using tflite::BuiltinOperator_STABLEHLO_POWER;
using tflite::BuiltinOperator_STABLEHLO_REDUCE;
using tflite::BuiltinOperator_STABLEHLO_REDUCE_WINDOW;
using tflite::BuiltinOperator_STABLEHLO_REMAINDER;
using tflite::BuiltinOperator_STABLEHLO_RESHAPE;
using tflite::BuiltinOperator_STABLEHLO_RNG_BIT_GENERATOR;
using tflite::BuiltinOperator_STABLEHLO_RSQRT;
using tflite::BuiltinOperator_STABLEHLO_SCATTER;
using tflite::BuiltinOperator_STABLEHLO_SELECT;
using tflite::BuiltinOperator_STABLEHLO_SHIFT_LEFT;
using tflite::BuiltinOperator_STABLEHLO_SLICE;
using tflite::BuiltinOperator_STABLEHLO_SORT;
using tflite::BuiltinOperator_STABLEHLO_SUBTRACT;
using tflite::BuiltinOperator_STABLEHLO_TANH;
using tflite::BuiltinOperator_STABLEHLO_TRANSPOSE;
using tflite::BuiltinOperator_STABLEHLO_WHILE;
using tflite::BuiltinOperator_STRIDED_SLICE;
using tflite::BuiltinOperator_SUB;
using tflite::BuiltinOperator_SUM;
using tflite::BuiltinOperator_SVDF;
using tflite::BuiltinOperator_TANH;
using tflite::BuiltinOperator_TILE;
using tflite::BuiltinOperator_TOPK_V2;
using tflite::BuiltinOperator_TRANSPOSE;
using tflite::BuiltinOperator_TRANSPOSE_CONV;
using tflite::BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM;
using tflite::BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_RNN;
using tflite::BuiltinOperator_UNIQUE;
using tflite::BuiltinOperator_UNPACK;
using tflite::BuiltinOperator_UNSORTED_SEGMENT_MAX;
using tflite::BuiltinOperator_UNSORTED_SEGMENT_MIN;
using tflite::BuiltinOperator_UNSORTED_SEGMENT_PROD;
using tflite::BuiltinOperator_UNSORTED_SEGMENT_SUM;
using tflite::BuiltinOperator_VAR_HANDLE;
using tflite::BuiltinOperator_WHERE;
using tflite::BuiltinOperator_WHILE;
using tflite::BuiltinOperator_ZEROS_LIKE;
using tflite::CallOnceOptions;
using tflite::CombinerType;
using tflite::CombinerType_MEAN;
using tflite::CombinerType_SQRTN;
using tflite::CombinerType_SUM;
using tflite::ConcatenationOptions;
using tflite::Conv2DOptions;
using tflite::DepthwiseConv2DOptions;
using tflite::FullyConnectedOptions;
using tflite::FullyConnectedOptionsWeightsFormat_DEFAULT;
using tflite::FullyConnectedOptionsWeightsFormat_SHUFFLED4x16INT8;
using tflite::IfOptions;
using tflite::L2NormOptions;
using tflite::LSHProjectionType;
using tflite::LSHProjectionType_DENSE;
using tflite::LSHProjectionType_SPARSE;
using tflite::LSTMKernelType_BASIC;
using tflite::LSTMKernelType_FULL;
using tflite::MirrorPadMode;
using tflite::MirrorPadMode_REFLECT;
using tflite::MirrorPadMode_SYMMETRIC;
using tflite::MirrorPadOptions;
using tflite::MulOptions;
using tflite::Operator;
using tflite::PackOptions;
using tflite::Padding;
using tflite::Padding_SAME;
using tflite::Padding_VALID;
using tflite::Pool2DOptions;
using tflite::ReducerOptions;
using tflite::ReduceWindowFunction_ADD;
using tflite::ReduceWindowFunction_ALL;
using tflite::ReduceWindowFunction_ANY;
using tflite::ReduceWindowFunction_MAXIMUM;
using tflite::ReduceWindowFunction_MINIMUM;
using tflite::ReduceWindowFunction_MUL;
using tflite::ReduceWindowFunction_UNSUPPORTED;
using tflite::ReshapeOptions;
using tflite::ResizeBilinearOptions;
using tflite::ResizeNearestNeighborOptions;
using tflite::RngAlgorithm;
using tflite::RngAlgorithm_DEFAULT;
using tflite::RngAlgorithm_PHILOX;
using tflite::RngAlgorithm_THREEFRY;
using tflite::ShapeOptions;
using tflite::SoftmaxOptions;
using tflite::SplitOptions;
using tflite::SplitVOptions;
using tflite::SqueezeOptions;
using tflite::StableHLOCompositeOptions;
using tflite::StablehloGatherOptions;
using tflite::StablehloPadOptions;
using tflite::StablehloReduceWindowOptions;
using tflite::StablehloRngBitGeneratorOptions;
using tflite::StablehloScatterOptions;
using tflite::StridedSliceOptions;
using tflite::SubOptions;
using tflite::SVDFOptions;
using tflite::TensorType;
using tflite::TensorType_BFLOAT16;
using tflite::TensorType_BOOL;
using tflite::TensorType_COMPLEX128;
using tflite::TensorType_COMPLEX64;
using tflite::TensorType_FLOAT16;
using tflite::TensorType_FLOAT32;
using tflite::TensorType_FLOAT64;
using tflite::TensorType_INT16;
using tflite::TensorType_INT32;
using tflite::TensorType_INT4;
using tflite::TensorType_INT64;
using tflite::TensorType_INT8;
using tflite::TensorType_RESOURCE;
using tflite::TensorType_STRING;
using tflite::TensorType_UINT16;
using tflite::TensorType_UINT32;
using tflite::TensorType_UINT64;
using tflite::TensorType_UINT8;
using tflite::TensorType_VARIANT;
using tflite::TransposeConvOptions;
using tflite::UnpackOptions;
using tflite::VarHandleOptions;
using tflite::WhileOptions;

// LINT.IfChange
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
void CheckParsePointerParams(const Operator* op,
                             BuiltinDataAllocator* allocator,
                             void** builtin_data) {
  TFLITE_DCHECK(op != nullptr);
  TFLITE_DCHECK(allocator != nullptr);
  TFLITE_DCHECK(builtin_data != nullptr);
}

// Copies the contents from the flatbuffer int vector `flatbuffer` into the
// int array `buffer`. `flat_vector` and `buffer` represent the same
// configuration operation for a given operation.
template <typename DataType = int32_t>
static absl::Status FlatBufferIntVectorToArray(
    int max_size_of_buffer, const flatbuffers::Vector<DataType>* flat_vector,
    DataType* buffer, const char* op_name) {
  if (!flat_vector) {
    auto error_message = absl::StrFormat(
        "Input array not provided for operation '%s'.\n", op_name);
    ABSL_LOG(ERROR) << error_message;
    return absl::InvalidArgumentError(error_message);
  } else {
    size_t num_dimensions = flat_vector->size();
    if (num_dimensions > max_size_of_buffer / sizeof(DataType)) {
      auto error_message = absl::StrFormat(
          "Found too many dimensions in the input array of operation '%s'.\n",
          op_name);
      ABSL_LOG(ERROR) << error_message;
      return absl::InvalidArgumentError(error_message);
    } else {
      for (size_t i = 0; i < num_dimensions; ++i) {
        buffer[i] = flat_vector->Get(i);
      }
    }
  }
  return OkStatus();
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

TfLiteRngAlgorithm ConvertRngAlgorithm(RngAlgorithm algorithm) {
  switch (algorithm) {
    case RngAlgorithm_THREEFRY:
      return kTfLiteRngAlgorithmThreefry;
    case RngAlgorithm_PHILOX:
      return kTfLiteRngAlgorithmPhilox;
    case RngAlgorithm_DEFAULT:
      return kTfLiteRngAlgorithmDefault;
  }
  return kTfLiteRngAlgorithmUnknown;
}

#ifndef TF_LITE_STATIC_MEMORY
absl::Status ParseOpDataTfLite(const Operator* op, BuiltinOperator op_type,
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
      return ParseAbs(op, allocator, builtin_data);
    }

    case BuiltinOperator_ADD: {
      return ParseAdd(op, allocator, builtin_data);
    }

    case BuiltinOperator_ADD_N: {
      return ParseAddN(op, allocator, builtin_data);
    }

    case BuiltinOperator_ARG_MAX: {
      return ParseArgMax(op, allocator, builtin_data);
    }

    case BuiltinOperator_ARG_MIN: {
      return ParseArgMin(op, allocator, builtin_data);
    }

    case BuiltinOperator_ASSIGN_VARIABLE: {
      return ParseAssignVariable(op, allocator, builtin_data);
    }

    case BuiltinOperator_AVERAGE_POOL_2D: {
      return ParsePool(op, allocator, builtin_data);
    }

    case BuiltinOperator_BATCH_MATMUL: {
      return ParseBatchMatMul(op, allocator, builtin_data);
    }

    case BuiltinOperator_BATCH_TO_SPACE_ND: {
      return ParseBatchToSpaceNd(op, allocator, builtin_data);
    }

    case BuiltinOperator_BROADCAST_ARGS: {
      return ParseBroadcastArgs(op, allocator, builtin_data);
    }

    case BuiltinOperator_BROADCAST_TO: {
      return ParseBroadcastTo(op, allocator, builtin_data);
    }

    case BuiltinOperator_CALL_ONCE: {
      return ParseCallOnce(op, allocator, builtin_data);
    }

    case BuiltinOperator_CEIL: {
      return ParseCeil(op, allocator, builtin_data);
    }

    case BuiltinOperator_CONCATENATION: {
      return ParseConcatenation(op, allocator, builtin_data);
    }

    case BuiltinOperator_CONV_2D: {
      return ParseConv2D(op, allocator, builtin_data);
    }

    case BuiltinOperator_CUMSUM: {
      return ParseCumsum(op, allocator, builtin_data);
    }

    case BuiltinOperator_DEPTH_TO_SPACE: {
      return ParseDepthToSpace(op, allocator, builtin_data);
    }

    case BuiltinOperator_DEPTHWISE_CONV_2D: {
      return ParseDepthwiseConv2D(op, allocator, builtin_data);
    }

    case BuiltinOperator_DEQUANTIZE: {
      return ParseDequantize(op, allocator, builtin_data);
    }

    case BuiltinOperator_DIV: {
      return ParseDiv(op, allocator, builtin_data);
    }

    case BuiltinOperator_ELU: {
      return ParseElu(op, allocator, builtin_data);
    }

    case BuiltinOperator_EMBEDDING_LOOKUP: {
      return ParseEmbeddingLookup(op, allocator, builtin_data);
    }

    case BuiltinOperator_EXP: {
      return ParseExp(op, allocator, builtin_data);
    }

    case BuiltinOperator_EXPAND_DIMS: {
      return ParseExpandDims(op, allocator, builtin_data);
    }

    case BuiltinOperator_FILL: {
      return ParseFill(op, allocator, builtin_data);
    }

    case BuiltinOperator_FLOOR: {
      return ParseFloor(op, allocator, builtin_data);
    }

    case BuiltinOperator_FLOOR_DIV: {
      return ParseFloorDiv(op, allocator, builtin_data);
    }

    case BuiltinOperator_FLOOR_MOD: {
      return ParseFloorMod(op, allocator, builtin_data);
    }

    case BuiltinOperator_FULLY_CONNECTED: {
      return ParseFullyConnected(op, allocator, builtin_data);
    }

    case BuiltinOperator_GATHER_ND: {
      return ParseGatherNd(op, allocator, builtin_data);
    }

    case BuiltinOperator_GREATER: {
      return ParseGreater(op, allocator, builtin_data);
    }

    case BuiltinOperator_GREATER_EQUAL: {
      return ParseGreaterEqual(op, allocator, builtin_data);
    }

    case BuiltinOperator_HARD_SWISH: {
      return ParseHardSwish(op, allocator, builtin_data);
    }

    case BuiltinOperator_L2_NORMALIZATION: {
      return ParseL2Normalization(op, allocator, builtin_data);
    }

    case BuiltinOperator_L2_POOL_2D: {
      return ParsePool(op, allocator, builtin_data);
    }

    case BuiltinOperator_LEAKY_RELU: {
      return ParseLeakyRelu(op, allocator, builtin_data);
    }

    case BuiltinOperator_LESS: {
      return ParseLess(op, allocator, builtin_data);
    }

    case BuiltinOperator_LESS_EQUAL: {
      return ParseLessEqual(op, allocator, builtin_data);
    }

    case BuiltinOperator_LOG: {
      return ParseLog(op, allocator, builtin_data);
    }

    case BuiltinOperator_LOGICAL_AND: {
      return ParseLogicalAnd(op, allocator, builtin_data);
    }

    case BuiltinOperator_LOGICAL_NOT: {
      return ParseLogicalNot(op, allocator, builtin_data);
    }

    case BuiltinOperator_LOGICAL_OR: {
      return ParseLogicalOr(op, allocator, builtin_data);
    }

    case BuiltinOperator_LOGISTIC: {
      return ParseLogistic(op, allocator, builtin_data);
    }

    case BuiltinOperator_LOG_SOFTMAX: {
      return ParseLogSoftmax(op, allocator, builtin_data);
    }

    case BuiltinOperator_LSTM: {
      return ParseLSTM(op, allocator, builtin_data);
    }

    case BuiltinOperator_MAXIMUM: {
      return ParseMaximum(op, allocator, builtin_data);
    }

    case BuiltinOperator_MAX_POOL_2D: {
      return ParsePool(op, allocator, builtin_data);
    }

    case BuiltinOperator_MIRROR_PAD: {
      return ParseMirrorPad(op, allocator, builtin_data);
    }

    case BuiltinOperator_MEAN: {
      return ParseReducer(op, allocator, builtin_data);
    }

    case BuiltinOperator_MINIMUM: {
      return ParseMinimum(op, allocator, builtin_data);
    }

    case BuiltinOperator_MUL: {
      return ParseMul(op, allocator, builtin_data);
    }

    case BuiltinOperator_NEG: {
      return ParseNeg(op, allocator, builtin_data);
    }

    case BuiltinOperator_NOT_EQUAL: {
      return ParseNotEqual(op, allocator, builtin_data);
    }

    case BuiltinOperator_PACK: {
      return ParsePack(op, allocator, builtin_data);
    }

    case BuiltinOperator_PAD: {
      return ParsePad(op, allocator, builtin_data);
    }

    case BuiltinOperator_PADV2: {
      return ParsePadV2(op, allocator, builtin_data);
    }

    case BuiltinOperator_POW: {
      return ParsePow(op, allocator, builtin_data);
    }

    case BuiltinOperator_PRELU: {
      return ParsePrelu(op, allocator, builtin_data);
    }

    case BuiltinOperator_QUANTIZE: {
      return ParseQuantize(op, allocator, builtin_data);
    }

    case BuiltinOperator_READ_VARIABLE: {
      return ParseReadVariable(op, allocator, builtin_data);
    }

    case BuiltinOperator_REDUCE_ANY: {
      return ParseReducer(op, allocator, builtin_data);
    }

    case BuiltinOperator_REDUCE_ALL: {
      return ParseReducer(op, allocator, builtin_data);
    }

    case BuiltinOperator_REDUCE_MAX: {
      return ParseReducer(op, allocator, builtin_data);
    }

    case BuiltinOperator_REDUCE_MIN: {
      return ParseReducer(op, allocator, builtin_data);
    }

    case BuiltinOperator_REDUCE_PROD: {
      return ParseReducer(op, allocator, builtin_data);
    }

    case BuiltinOperator_RELU: {
      return ParseRelu(op, allocator, builtin_data);
    }

    case BuiltinOperator_RELU6: {
      return ParseRelu6(op, allocator, builtin_data);
    }

    case BuiltinOperator_RESHAPE: {
      return ParseReshape(op, allocator, builtin_data);
    }

    case BuiltinOperator_RESIZE_BILINEAR: {
      return ParseResizeBilinear(op, allocator, builtin_data);
    }

    case BuiltinOperator_RESIZE_NEAREST_NEIGHBOR: {
      return ParseResizeNearestNeighbor(op, allocator, builtin_data);
    }

    case BuiltinOperator_ROUND: {
      return ParseRound(op, allocator, builtin_data);
    }

    case BuiltinOperator_RSQRT: {
      return ParseRsqrt(op, allocator, builtin_data);
    }

    case BuiltinOperator_SELECT_V2: {
      return ParseSelectV2(op, allocator, builtin_data);
    }

    case BuiltinOperator_SHAPE: {
      return ParseShape(op, allocator, builtin_data);
    }

    case BuiltinOperator_SIN: {
      return ParseSin(op, allocator, builtin_data);
    }

    case BuiltinOperator_SOFTMAX: {
      return ParseSoftmax(op, allocator, builtin_data);
    }

    case BuiltinOperator_SPACE_TO_BATCH_ND: {
      return ParseSpaceToBatchNd(op, allocator, builtin_data);
    }

    case BuiltinOperator_SPACE_TO_DEPTH: {
      return ParseSpaceToDepth(op, allocator, builtin_data);
    }

    case BuiltinOperator_SPLIT: {
      return ParseSplit(op, allocator, builtin_data);
    }

    case BuiltinOperator_SPLIT_V: {
      return ParseSplitV(op, allocator, builtin_data);
    }

    case BuiltinOperator_SQRT: {
      return ParseSqrt(op, allocator, builtin_data);
    }

    case BuiltinOperator_SQUARE: {
      return ParseSquare(op, allocator, builtin_data);
    }

    case BuiltinOperator_SQUARED_DIFFERENCE: {
      return ParseSquaredDifference(op, allocator, builtin_data);
    }

    case BuiltinOperator_SQUEEZE: {
      return ParseSqueeze(op, allocator, builtin_data);
    }

    case BuiltinOperator_STRIDED_SLICE: {
      return ParseStridedSlice(op, allocator, builtin_data);
    }

    case BuiltinOperator_SUB: {
      return ParseSub(op, allocator, builtin_data);
    }

    case BuiltinOperator_SUM: {
      return ParseReducer(op, allocator, builtin_data);
    }

    case BuiltinOperator_SVDF: {
      return ParseSvdf(op, allocator, builtin_data);
    }

    case BuiltinOperator_TANH: {
      return ParseTanh(op, allocator, builtin_data);
    }

    case BuiltinOperator_TRANSPOSE_CONV: {
      return ParseTransposeConv(op, allocator, builtin_data);
    }

    case BuiltinOperator_UNPACK: {
      return ParseUnpack(op, allocator, builtin_data);
    }

    case BuiltinOperator_VAR_HANDLE: {
      return ParseVarHandle(op, allocator, builtin_data);
    }

    case BuiltinOperator_ZEROS_LIKE: {
      return ParseZerosLike(op, allocator, builtin_data);
    }

    case BuiltinOperator_BITWISE_XOR: {
      return ParseBitwiseXor(op, allocator, builtin_data);
    }

    case BuiltinOperator_RIGHT_SHIFT: {
      return ParseRightShift(op, allocator, builtin_data);
    }

    case BuiltinOperator_CAST: {
      return ParseCast(op, allocator, builtin_data);
    }
    case BuiltinOperator_LSH_PROJECTION: {
      auto params = safe_allocator.Allocate<TfLiteLSHProjectionParams>();
      TFL_FILE_ENSURE(params != nullptr);
      if (const auto* lshParams =
              op->builtin_options_as_LSHProjectionOptions()) {
        params->type = parseLSHProjectionType(lshParams->type());
      }
      *builtin_data = params.release();
      return OkStatus();
    }
    case BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_RNN: {
      auto params = safe_allocator.Allocate<TfLiteSequenceRNNParams>();
      TFL_FILE_ENSURE(params != nullptr);
      if (const auto* sequence_rnn_params =
              op->builtin_options_as_SequenceRNNOptions()) {
        params->activation =
            ConvertActivation(sequence_rnn_params->fused_activation_function());
        params->time_major = sequence_rnn_params->time_major();
        params->asymmetric_quantize_inputs =
            sequence_rnn_params->asymmetric_quantize_inputs();
      }
      *builtin_data = params.release();
      return OkStatus();
    }
    case BuiltinOperator_BIDIRECTIONAL_SEQUENCE_RNN: {
      auto params =
          safe_allocator.Allocate<TfLiteBidirectionalSequenceRNNParams>();
      TFL_FILE_ENSURE(params != nullptr);
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
      return OkStatus();
    }
    case BuiltinOperator_RNN: {
      auto params = safe_allocator.Allocate<TfLiteRNNParams>();
      TFL_FILE_ENSURE(params != nullptr);
      if (const auto* rnn_params = op->builtin_options_as_RNNOptions()) {
        params->activation =
            ConvertActivation(rnn_params->fused_activation_function());
        params->asymmetric_quantize_inputs =
            rnn_params->asymmetric_quantize_inputs();
      }
      *builtin_data = params.release();
      return OkStatus();
    }
    case BuiltinOperator_EMBEDDING_LOOKUP_SPARSE: {
      auto params =
          safe_allocator.Allocate<TfLiteEmbeddingLookupSparseParams>();
      TFL_FILE_ENSURE(params != nullptr);
      if (const auto* embedding_params =
              op->builtin_options_as_EmbeddingLookupSparseOptions()) {
        params->combiner = parseCombinerType(embedding_params->combiner());
      }
      *builtin_data = params.release();
      return OkStatus();
    }

    case BuiltinOperator_HASHTABLE_LOOKUP:
      // no-op.
      return OkStatus();

    case BuiltinOperator_LOCAL_RESPONSE_NORMALIZATION: {
      auto params = safe_allocator.Allocate<TfLiteLocalResponseNormParams>();
      TFL_FILE_ENSURE(params != nullptr);
      if (const auto* schema_params =
              op->builtin_options_as_LocalResponseNormalizationOptions()) {
        params->radius = schema_params->radius();
        params->bias = schema_params->bias();
        params->alpha = schema_params->alpha();
        params->beta = schema_params->beta();
      }
      *builtin_data = params.release();
      return OkStatus();
    }
    case BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM: {
      return ParseUnidirectionalSequenceLSTM(op, allocator, builtin_data);
    }
    case BuiltinOperator_BIDIRECTIONAL_SEQUENCE_LSTM: {
      auto params =
          safe_allocator.Allocate<TfLiteBidirectionalSequenceLSTMParams>();
      TFL_FILE_ENSURE(params != nullptr);
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
      return OkStatus();
    }
    case BuiltinOperator_SKIP_GRAM: {
      auto params = safe_allocator.Allocate<TfLiteSkipGramParams>();
      TFL_FILE_ENSURE(params != nullptr);
      if (const auto* skip_gram_params =
              op->builtin_options_as_SkipGramOptions()) {
        params->ngram_size = skip_gram_params->ngram_size();
        params->max_skip_size = skip_gram_params->max_skip_size();
        params->include_all_ngrams = skip_gram_params->include_all_ngrams();
      }
      *builtin_data = params.release();
      return OkStatus();
    }

    case BuiltinOperator_GATHER: {
      return ParseGather(op, allocator, builtin_data);
    }
    case BuiltinOperator_SPARSE_TO_DENSE: {
      auto params = safe_allocator.Allocate<TfLiteSparseToDenseParams>();
      TFL_FILE_ENSURE(params != nullptr);
      if (const auto* sparse_to_dense_params =
              op->builtin_options_as_SparseToDenseOptions()) {
        params->validate_indices = sparse_to_dense_params->validate_indices();
      }
      *builtin_data = params.release();
      return OkStatus();
    }
    case BuiltinOperator_DELEGATE: {
      auto error_msg = "DELEGATE op shouldn't exist in model.";
      ABSL_LOG(ERROR) << error_msg;
      return absl::InvalidArgumentError(error_msg);
    }
    case BuiltinOperator_FAKE_QUANT: {
      auto params = safe_allocator.Allocate<TfLiteFakeQuantParams>();
      TFL_FILE_ENSURE(params != nullptr);
      if (const auto* schema_params =
              op->builtin_options_as_FakeQuantOptions()) {
        params->min = schema_params->min();
        params->max = schema_params->max();
        params->num_bits = schema_params->num_bits();
        params->narrow_range = schema_params->narrow_range();
      }
      *builtin_data = params.release();
      return OkStatus();
    }
    case BuiltinOperator_ONE_HOT: {
      auto params = safe_allocator.Allocate<TfLiteOneHotParams>();
      TFL_FILE_ENSURE(params != nullptr);
      if (const auto* schema_params = op->builtin_options_as_OneHotOptions()) {
        params->axis = schema_params->axis();
      }
      *builtin_data = params.release();
      return OkStatus();
    }
    case BuiltinOperator_UNIQUE: {
      auto params = safe_allocator.Allocate<TfLiteUniqueParams>();
      TFL_FILE_ENSURE(params != nullptr);
      const auto* unique_params = op->builtin_options_as_UniqueOptions();
      if (unique_params != nullptr) {
        params->index_out_type =
            unique_params->idx_out_type() == tflite::TensorType_INT64
                ? TfLiteType::kTfLiteInt64
                : TfLiteType::kTfLiteInt32;
      }
      *builtin_data = params.release();
      return OkStatus();
    }
    case BuiltinOperator_REVERSE_SEQUENCE: {
      auto params = safe_allocator.Allocate<TfLiteReverseSequenceParams>();
      TFL_FILE_ENSURE(params != nullptr);
      if (const auto* reverse_seq_params =
              op->builtin_options_as_ReverseSequenceOptions()) {
        params->seq_dim = reverse_seq_params->seq_dim();
        params->batch_dim = reverse_seq_params->batch_dim();
      }
      *builtin_data = params.release();
      return OkStatus();
    }
    case BuiltinOperator_IF: {
      auto params = safe_allocator.Allocate<TfLiteIfParams>();
      TFL_FILE_ENSURE(params != nullptr);
      if (const auto* if_params = op->builtin_options_as_IfOptions()) {
        params->then_subgraph_index = if_params->then_subgraph_index();
        params->else_subgraph_index = if_params->else_subgraph_index();
      }
      *builtin_data = params.release();
      return OkStatus();
    }
    case BuiltinOperator_WHILE: {
      auto params = safe_allocator.Allocate<TfLiteWhileParams>();
      TFL_FILE_ENSURE(params != nullptr);
      if (const auto* while_params = op->builtin_options_as_WhileOptions()) {
        params->cond_subgraph_index = while_params->cond_subgraph_index();
        params->body_subgraph_index = while_params->body_subgraph_index();
      }
      *builtin_data = params.release();
      return OkStatus();
    }
    case BuiltinOperator_CONV_3D:
    case BuiltinOperator_CONV_3D_TRANSPOSE: {
      auto params = safe_allocator.Allocate<TfLiteConv3DParams>();
      TFL_FILE_ENSURE(params != nullptr);
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
      return OkStatus();
    }
    case BuiltinOperator_HASHTABLE: {
      auto params = safe_allocator.Allocate<TfLiteHashtableParams>();
      TFL_FILE_ENSURE(params != nullptr);
      if (const auto* hashtable_params =
              op->builtin_options_as_HashtableOptions()) {
        params->table_id = hashtable_params->table_id();
        TFL_FILE_ENSURE_STATUS(ConvertTensorType(hashtable_params->key_dtype(),
                                                 &params->key_dtype));
        TFL_FILE_ENSURE_STATUS(ConvertTensorType(
            hashtable_params->value_dtype(), &params->value_dtype));
      }
      *builtin_data = params.release();
      return OkStatus();
    }
    case BuiltinOperator_MULTINOMIAL: {
      auto params = safe_allocator.Allocate<TfLiteRandomParams>();
      TFL_FILE_ENSURE(params != nullptr);
      if (const auto* multinomial_params =
              op->builtin_options_as_RandomOptions()) {
        params->seed = multinomial_params->seed();
        params->seed2 = multinomial_params->seed2();
      }
      *builtin_data = params.release();
      return OkStatus();
    }
    case BuiltinOperator_RANDOM_STANDARD_NORMAL: {
      auto params = safe_allocator.Allocate<TfLiteRandomParams>();
      TFL_FILE_ENSURE(params != nullptr);
      if (const auto* random_std_normal_params =
              op->builtin_options_as_RandomOptions()) {
        params->seed = random_std_normal_params->seed();
        params->seed2 = random_std_normal_params->seed2();
      }
      *builtin_data = params.release();
      return OkStatus();
    }
    case BuiltinOperator_BUCKETIZE: {
      auto params = safe_allocator.Allocate<TfLiteBucketizeParams>();
      TFL_FILE_ENSURE(params != nullptr);
      if (const auto* bucketize_params =
              op->builtin_options_as_BucketizeOptions()) {
        const flatbuffers::Vector<float>* boundaries =
            bucketize_params->boundaries();
        if (boundaries == nullptr) {
          auto error_message =
              "boundaries array not provided for operation 'bucketize'.\n";
          ABSL_LOG(ERROR) << error_message;
          return absl::InvalidArgumentError(error_message);
        }
        params->num_boundaries = boundaries->size();
        if (boundaries->data() == nullptr) {
          auto error_message =
              "boundaries.data() returned nullptr for "
              "operation 'bucketize'.\n";
          ABSL_LOG(ERROR) << error_message;
          return absl::InvalidArgumentError(error_message);
        }
        params->boundaries = boundaries->data();
      }
      *builtin_data = params.release();
      return OkStatus();
    }
    case BuiltinOperator_RANDOM_UNIFORM: {
      auto params = safe_allocator.Allocate<TfLiteRandomParams>();
      TFL_FILE_ENSURE(params != nullptr);
      if (const auto* random_uniform_params =
              op->builtin_options_as_RandomOptions()) {
        params->seed = random_uniform_params->seed();
        params->seed2 = random_uniform_params->seed2();
      }
      *builtin_data = params.release();
      return OkStatus();
    }
    case BuiltinOperator_GELU: {
      auto params = safe_allocator.Allocate<TfLiteGeluParams>();
      TFL_FILE_ENSURE(params != nullptr);
      if (const auto* gelu_params = op->builtin_options_as_GeluOptions()) {
        params->approximate = gelu_params->approximate();
      }
      *builtin_data = params.release();
      return OkStatus();
    }
    case BuiltinOperator_STABLEHLO_SCATTER: {
      return ParseStablehloScatter(op, allocator, builtin_data);
    }
    case BuiltinOperator_STABLEHLO_RNG_BIT_GENERATOR: {
      return ParseStablehloRngBitGenerator(op, allocator, builtin_data);
    }
    case BuiltinOperator_STABLEHLO_GATHER: {
      return ParseStablehloGather(op, allocator, builtin_data);
    }
    case BuiltinOperator_STABLEHLO_REDUCE_WINDOW: {
      return ParseStablehloReduceWindow(op, allocator, builtin_data);
    }
    case BuiltinOperator_REDUCE_WINDOW: {
      auto params = safe_allocator.Allocate<TfLiteReduceWindowParams>();
      TFL_FILE_ENSURE(params != nullptr);
      if (const auto* reduce_params =
              op->builtin_options_2_as_ReduceWindowOptions()) {
        switch (reduce_params->reduce_function()) {
          case ReduceWindowFunction_ADD:
            params->reduce_function = TfLiteReduceWindowFunctionAdd;
            break;
          case ReduceWindowFunction_MUL:
            params->reduce_function = TfLiteReduceWindowFunctionMul;
            break;
          case ReduceWindowFunction_MINIMUM:
            params->reduce_function = TfLiteReduceWindowFunctionMin;
            break;
          case ReduceWindowFunction_MAXIMUM:
            params->reduce_function = TfLiteReduceWindowFunctionMax;
            break;
          case ReduceWindowFunction_ALL:
            params->reduce_function = TfLiteReduceWindowFunctionAll;
            break;
          case ReduceWindowFunction_ANY:
            params->reduce_function = TfLiteReduceWindowFunctionAny;
            break;
          case ReduceWindowFunction_UNSUPPORTED:
          default:
            return absl::InvalidArgumentError("Unsupported reduce function");
        }
      }
      *builtin_data = params.release();
      return OkStatus();
    }
    case BuiltinOperator_STABLEHLO_PAD: {
      return ParseStablehloPad(op, allocator, builtin_data);
    }
    case BuiltinOperator_STABLEHLO_COMPOSITE: {
      return ParseStablehloComposite(op, allocator, builtin_data);
    }
    case BuiltinOperator_STABLEHLO_SHIFT_LEFT: {
      return ParseStablehloShiftLeft(op, allocator, builtin_data);
    }
    case BuiltinOperator_STABLEHLO_CASE: {
      return ParseStablehloCase(op, allocator, builtin_data);
    }
    // TODO: skip param parsing for now since ops below don't have kernels
    case BuiltinOperator_STABLEHLO_SLICE:
    case BuiltinOperator_STABLEHLO_BROADCAST_IN_DIM:
    case BuiltinOperator_STABLEHLO_CONVOLUTION:
    case BuiltinOperator_STABLEHLO_LOGISTIC:
    case BuiltinOperator_STABLEHLO_ADD:
    case BuiltinOperator_STABLEHLO_DIVIDE:
    case BuiltinOperator_STABLEHLO_MULTIPLY:
    case BuiltinOperator_STABLEHLO_MAXIMUM:
    case BuiltinOperator_STABLEHLO_RESHAPE:
    case BuiltinOperator_STABLEHLO_CLAMP:
    case BuiltinOperator_STABLEHLO_CONCATENATE:
    case BuiltinOperator_STABLEHLO_CUSTOM_CALL:
    case BuiltinOperator_STABLEHLO_REDUCE:
    case BuiltinOperator_STABLEHLO_ABS:
    case BuiltinOperator_STABLEHLO_AND:
    case BuiltinOperator_STABLEHLO_COSINE:
    case BuiltinOperator_STABLEHLO_EXPONENTIAL:
    case BuiltinOperator_STABLEHLO_FLOOR:
    case BuiltinOperator_STABLEHLO_LOG:
    case BuiltinOperator_STABLEHLO_MINIMUM:
    case BuiltinOperator_STABLEHLO_NEGATE:
    case BuiltinOperator_STABLEHLO_OR:
    case BuiltinOperator_STABLEHLO_POWER:
    case BuiltinOperator_STABLEHLO_REMAINDER:
    case BuiltinOperator_STABLEHLO_RSQRT:
    case BuiltinOperator_STABLEHLO_SELECT:
    case BuiltinOperator_STABLEHLO_SUBTRACT:
    case BuiltinOperator_STABLEHLO_TANH:
    case BuiltinOperator_STABLEHLO_DYNAMIC_SLICE:
    case BuiltinOperator_STABLEHLO_DYNAMIC_UPDATE_SLICE:
    case BuiltinOperator_STABLEHLO_IOTA:
    case BuiltinOperator_STABLEHLO_COMPARE:
    case BuiltinOperator_STABLEHLO_CONVERT:
    case BuiltinOperator_STABLEHLO_DOT_GENERAL:
    case BuiltinOperator_STABLEHLO_SORT:
    case BuiltinOperator_STABLEHLO_WHILE:
    case BuiltinOperator_STABLEHLO_TRANSPOSE:
    case BuiltinOperator_STABLEHLO_CBRT:

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
    case BuiltinOperator_UNSORTED_SEGMENT_MIN:
    case BuiltinOperator_UNSORTED_SEGMENT_PROD:
    case BuiltinOperator_UNSORTED_SEGMENT_SUM:
    case BuiltinOperator_ATAN2:
    case BuiltinOperator_SIGN:
    case BuiltinOperator_BITCAST:
    case BuiltinOperator_WHERE:
    case BuiltinOperator_DILATE:
      return OkStatus();
    case BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES:
      return absl::UnimplementedError("Unsupported op");
  }
  return absl::UnimplementedError("Unsupported op");
}  // NOLINT[readability/fn_size]
#endif  // !defined(TF_LITE_STATIC_MEMORY)
}  // namespace

absl::Status ConvertTensorType(TensorType tensor_type, TfLiteType* type) {
  switch (tensor_type) {
    case TensorType_FLOAT16:
      *type = kTfLiteFloat16;
      return OkStatus();
    case TensorType_BFLOAT16:
      *type = kTfLiteBFloat16;
      return OkStatus();
    case TensorType_FLOAT32:
      *type = kTfLiteFloat32;
      return OkStatus();
    case TensorType_FLOAT64:
      *type = kTfLiteFloat64;
      return OkStatus();
    case TensorType_INT16:
      *type = kTfLiteInt16;
      return OkStatus();
    case TensorType_UINT16:
      *type = kTfLiteUInt16;
      return OkStatus();
    case TensorType_INT32:
      *type = kTfLiteInt32;
      return OkStatus();
    case TensorType_UINT32:
      *type = kTfLiteUInt32;
      return OkStatus();
    case TensorType_UINT8:
      *type = kTfLiteUInt8;
      return OkStatus();
    case TensorType_INT8:
      *type = kTfLiteInt8;
      return OkStatus();
    case TensorType_INT64:
      *type = kTfLiteInt64;
      return OkStatus();
    case TensorType_UINT64:
      *type = kTfLiteUInt64;
      return OkStatus();
    case TensorType_STRING:
      *type = kTfLiteString;
      return OkStatus();
    case TensorType_BOOL:
      *type = kTfLiteBool;
      return OkStatus();
    case TensorType_COMPLEX64:
      *type = kTfLiteComplex64;
      return OkStatus();
    case TensorType_COMPLEX128:
      *type = kTfLiteComplex128;
      return OkStatus();
    case TensorType_RESOURCE:
      *type = kTfLiteResource;
      return OkStatus();
    case TensorType_VARIANT:
      *type = kTfLiteVariant;
      return OkStatus();
    case TensorType_INT4:
      *type = kTfLiteInt4;
      return OkStatus();
    default:
      *type = kTfLiteNoType;
      auto error_message =
          absl::StrFormat("Unsupported data type %d in tensor", tensor_type);
      ABSL_LOG(ERROR) << error_message;
      return absl::InvalidArgumentError(error_message);
  }
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseAbs(const Operator*, BuiltinDataAllocator*, void**) {
  return OkStatus();
}

absl::Status ParseAdd(const Operator* op, BuiltinDataAllocator* allocator,
                      void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteAddParams, SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteAddParams>();
  TFL_FILE_ENSURE(params != nullptr);

  const AddOptions* schema_params = op->builtin_options_as_AddOptions();

  if (schema_params != nullptr) {
    params->activation =
        ConvertActivation(schema_params->fused_activation_function());
    params->pot_scale_int16 = schema_params->pot_scale_int16();
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return OkStatus();
}

absl::Status ParseAddN(const Operator* op, BuiltinDataAllocator* allocator,
                       void** builtin_data) {
  return OkStatus();
}

absl::Status ParseArgMax(const Operator* op, BuiltinDataAllocator* allocator,
                         void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteArgMaxParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteArgMaxParams>();
  TFL_FILE_ENSURE(params != nullptr);

  const ArgMaxOptions* schema_params = op->builtin_options_as_ArgMaxOptions();

  if (schema_params != nullptr) {
    TFL_FILE_ENSURE_STATUS(
        ConvertTensorType(schema_params->output_type(), &params->output_type));
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return OkStatus();
}

absl::Status ParseArgMin(const Operator* op, BuiltinDataAllocator* allocator,
                         void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteArgMinParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteArgMinParams>();
  TFL_FILE_ENSURE(params != nullptr);

  const ArgMinOptions* schema_params = op->builtin_options_as_ArgMinOptions();

  if (schema_params != nullptr) {
    TFL_FILE_ENSURE_STATUS(
        ConvertTensorType(schema_params->output_type(), &params->output_type));
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseAssignVariable(const Operator*, BuiltinDataAllocator*,
                                 void**) {
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseBatchMatMul(const Operator* op,
                              BuiltinDataAllocator* allocator,
                              void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  auto params = safe_allocator.Allocate<TfLiteBatchMatMulParams>();
  TFL_FILE_ENSURE(params != nullptr);
  if (const auto* bmm_params = op->builtin_options_as_BatchMatMulOptions()) {
    params->adj_x = bmm_params->adj_x();
    params->adj_y = bmm_params->adj_y();
    params->asymmetric_quantize_inputs =
        bmm_params->asymmetric_quantize_inputs();
  }
  *builtin_data = params.release();
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseBatchToSpaceNd(const Operator*, BuiltinDataAllocator*,
                                 void**) {
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseBroadcastArgs(const Operator*, BuiltinDataAllocator*,
                                void**) {
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseBroadcastTo(const Operator*, BuiltinDataAllocator*, void**) {
  return OkStatus();
}

absl::Status ParseCallOnce(const Operator* op, BuiltinDataAllocator* allocator,
                           void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteCallOnceParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteCallOnceParams>();
  TFL_FILE_ENSURE(params != nullptr);

  const CallOnceOptions* schema_params =
      op->builtin_options_as_CallOnceOptions();

  if (schema_params != nullptr) {
    params->init_subgraph_index = schema_params->init_subgraph_index();

  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseCast(const Operator* op, BuiltinDataAllocator* allocator,
                       void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  auto params = safe_allocator.Allocate<TfLiteCastParams>();
  TFL_FILE_ENSURE(params != nullptr);
  if (const auto* schema_params = op->builtin_options_as_CastOptions()) {
    TFL_FILE_ENSURE_STATUS(ConvertTensorType(schema_params->in_data_type(),
                                             &params->in_data_type));
    TFL_FILE_ENSURE_STATUS(ConvertTensorType(schema_params->out_data_type(),
                                             &params->out_data_type));
  }
  *builtin_data = params.release();
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseCeil(const Operator*, BuiltinDataAllocator*, void**) {
  return OkStatus();
}

absl::Status ParseConcatenation(const Operator* op,
                                BuiltinDataAllocator* allocator,
                                void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteConcatenationParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteConcatenationParams>();
  TFL_FILE_ENSURE(params != nullptr);

  const ConcatenationOptions* schema_params =
      op->builtin_options_as_ConcatenationOptions();

  if (schema_params != nullptr) {
    params->activation =
        ConvertActivation(schema_params->fused_activation_function());
    params->axis = schema_params->axis();
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return OkStatus();
}

absl::Status ParseConv2D(const Operator* op, BuiltinDataAllocator* allocator,
                         void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteConvParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteConvParams>();
  TFL_FILE_ENSURE(params != nullptr);

  const Conv2DOptions* schema_params = op->builtin_options_as_Conv2DOptions();

  if (schema_params != nullptr) {
    params->padding = ConvertPadding(schema_params->padding());
    params->stride_width = schema_params->stride_w();
    params->stride_height = schema_params->stride_h();
    params->activation =
        ConvertActivation(schema_params->fused_activation_function());

    params->dilation_width_factor = schema_params->dilation_w_factor();
    params->dilation_height_factor = schema_params->dilation_h_factor();
    TFL_FILE_ENSURE_STATUS(ConvertTensorType(
        schema_params->quantized_bias_type(), &params->quantized_bias_type));
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseCumsum(const Operator* op, BuiltinDataAllocator* allocator,
                         void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  auto params = safe_allocator.Allocate<TfLiteCumsumParams>();
  TFL_FILE_ENSURE(params != nullptr);
  if (const auto* cumsum_params = op->builtin_options_as_CumsumOptions()) {
    params->exclusive = cumsum_params->exclusive();
    params->reverse = cumsum_params->reverse();
  }
  *builtin_data = params.release();
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseCos(const Operator*, BuiltinDataAllocator*, void**) {
  return OkStatus();
}

absl::Status ParseDepthToSpace(const Operator* op,
                               BuiltinDataAllocator* allocator,
                               void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteDepthToSpaceParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteDepthToSpaceParams>();
  TFL_FILE_ENSURE(params != nullptr);

  const auto* schema_params = op->builtin_options_as_DepthToSpaceOptions();
  if (schema_params != nullptr) {
    params->block_size = schema_params->block_size();
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return OkStatus();
}

absl::Status ParseDepthwiseConv2D(const Operator* op,
                                  BuiltinDataAllocator* allocator,
                                  void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);

  std::unique_ptr<TfLiteDepthwiseConvParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteDepthwiseConvParams>();
  TFL_FILE_ENSURE(params != nullptr);

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
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseDequantize(const Operator*, BuiltinDataAllocator*, void**) {
  return OkStatus();
}

absl::Status ParseDiv(const Operator* op, BuiltinDataAllocator* allocator,
                      void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  auto params = safe_allocator.Allocate<TfLiteDivParams>();
  TFL_FILE_ENSURE(params != nullptr);
  if (const auto* schema_params = op->builtin_options_as_DivOptions()) {
    params->activation =
        ConvertActivation(schema_params->fused_activation_function());
  }
  *builtin_data = params.release();
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseElu(const Operator*, BuiltinDataAllocator*, void**) {
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseEmbeddingLookup(const Operator*, BuiltinDataAllocator*,
                                  void**) {
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseEqual(const Operator*, BuiltinDataAllocator*, void**) {
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseExp(const Operator*, BuiltinDataAllocator*, void**) {
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseExpandDims(const Operator*, BuiltinDataAllocator*, void**) {
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseFill(const Operator*, BuiltinDataAllocator*, void**) {
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseFloor(const Operator*, BuiltinDataAllocator*, void**) {
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseFloorDiv(const Operator*, BuiltinDataAllocator*, void**) {
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseFloorMod(const Operator*, BuiltinDataAllocator*, void**) {
  return OkStatus();
}

absl::Status ParseFullyConnected(const Operator* op,
                                 BuiltinDataAllocator* allocator,
                                 void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);

  std::unique_ptr<TfLiteFullyConnectedParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteFullyConnectedParams>();
  TFL_FILE_ENSURE(params != nullptr);

  const FullyConnectedOptions* schema_params =
      op->builtin_options_as_FullyConnectedOptions();

  if (schema_params != nullptr) {
    params->activation =
        ConvertActivation(schema_params->fused_activation_function());
    params->keep_num_dims = schema_params->keep_num_dims();
    params->asymmetric_quantize_inputs =
        schema_params->asymmetric_quantize_inputs();
    TFL_FILE_ENSURE_STATUS(ConvertTensorType(
        schema_params->quantized_bias_type(), &params->quantized_bias_type));
    switch (schema_params->weights_format()) {
      case FullyConnectedOptionsWeightsFormat_DEFAULT:
        params->weights_format = kTfLiteFullyConnectedWeightsFormatDefault;
        break;
      case FullyConnectedOptionsWeightsFormat_SHUFFLED4x16INT8:
        params->weights_format =
            kTfLiteFullyConnectedWeightsFormatShuffled4x16Int8;
        break;
      default:
        auto error_message = "Unhandled fully-connected weights format.";
        ABSL_LOG(ERROR) << error_message;
        return absl::InvalidArgumentError(error_message);
    }
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseGather(const Operator* op, BuiltinDataAllocator* allocator,
                         void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  auto params = safe_allocator.Allocate<TfLiteGatherParams>();
  TFL_FILE_ENSURE(params != nullptr);
  params->axis = 0;
  params->batch_dims = 0;
  if (const auto* gather_params = op->builtin_options_as_GatherOptions()) {
    params->axis = gather_params->axis();
    params->batch_dims = gather_params->batch_dims();
  }

  *builtin_data = params.release();
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseGatherNd(const Operator*, BuiltinDataAllocator*, void**) {
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseGreater(const Operator*, BuiltinDataAllocator*, void**) {
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseGreaterEqual(const Operator*, BuiltinDataAllocator*, void**) {
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseHardSwish(const Operator*, BuiltinDataAllocator*, void**) {
  return OkStatus();
}

absl::Status ParseIf(const Operator* op, BuiltinDataAllocator* allocator,
                     void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteIfParams, SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteIfParams>();
  TFL_FILE_ENSURE(params != nullptr);

  const IfOptions* schema_params = op->builtin_options_as_IfOptions();

  if (schema_params != nullptr) {
    params->then_subgraph_index = schema_params->then_subgraph_index();
    params->else_subgraph_index = schema_params->else_subgraph_index();
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return OkStatus();
}

absl::Status ParseL2Normalization(const Operator* op,
                                  BuiltinDataAllocator* allocator,
                                  void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteL2NormParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteL2NormParams>();
  TFL_FILE_ENSURE(params != nullptr);

  const L2NormOptions* schema_params = op->builtin_options_as_L2NormOptions();

  if (schema_params != nullptr) {
    params->activation =
        ConvertActivation(schema_params->fused_activation_function());
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return OkStatus();
}

absl::Status ParseLeakyRelu(const Operator* op, BuiltinDataAllocator* allocator,
                            void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  auto params = safe_allocator.Allocate<TfLiteLeakyReluParams>();
  TFL_FILE_ENSURE(params != nullptr);
  if (const auto* leaky_relu_params =
          op->builtin_options_as_LeakyReluOptions()) {
    params->alpha = leaky_relu_params->alpha();
  }
  *builtin_data = params.release();
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseLess(const Operator*, BuiltinDataAllocator*, void**) {
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseLessEqual(const Operator*, BuiltinDataAllocator*, void**) {
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseLog(const Operator*, BuiltinDataAllocator*, void**) {
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseLogicalAnd(const Operator*, BuiltinDataAllocator*, void**) {
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseLogicalNot(const Operator*, BuiltinDataAllocator*, void**) {
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseLogicalOr(const Operator*, BuiltinDataAllocator*, void**) {
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseLogistic(const Operator*, BuiltinDataAllocator*, void**) {
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseLogSoftmax(const Operator*, BuiltinDataAllocator*, void**) {
  return OkStatus();
}

absl::Status ParseLSTM(const Operator* op, BuiltinDataAllocator* allocator,
                       void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  auto params = safe_allocator.Allocate<TfLiteLSTMParams>();
  TFL_FILE_ENSURE(params != nullptr);
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
        auto error_message = absl::StrFormat("Unhandled LSTM kernel type: %d",
                                             lstm_params->kernel_type());
        ABSL_LOG(ERROR) << error_message;
        return absl::InvalidArgumentError(error_message);
    }
    params->asymmetric_quantize_inputs =
        lstm_params->asymmetric_quantize_inputs();
  } else {
    auto error_message = "No valid LSTM builtin options exist";
    ABSL_LOG(ERROR) << error_message;
    return absl::InvalidArgumentError(error_message);
  }
  *builtin_data = params.release();
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseMaximum(const Operator*, BuiltinDataAllocator*, void**) {
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseMinimum(const Operator*, BuiltinDataAllocator*, void**) {
  return OkStatus();
}

absl::Status ParseMirrorPad(const Operator* op, BuiltinDataAllocator* allocator,
                            void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteMirrorPaddingParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteMirrorPaddingParams>();
  TFL_FILE_ENSURE(params != nullptr);

  const MirrorPadOptions* schema_params =
      op->builtin_options_as_MirrorPadOptions();

  if (schema_params != nullptr) {
    params->mode = ConvertMirrorPadding(schema_params->mode());
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return OkStatus();
}

absl::Status ParseMul(const Operator* op, BuiltinDataAllocator* allocator,
                      void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteMulParams, SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteMulParams>();
  TFL_FILE_ENSURE(params != nullptr);

  const MulOptions* schema_params = op->builtin_options_as_MulOptions();

  if (schema_params != nullptr) {
    params->activation =
        ConvertActivation(schema_params->fused_activation_function());
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseNeg(const Operator*, BuiltinDataAllocator*, void**) {
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseNotEqual(const Operator*, BuiltinDataAllocator*, void**) {
  return OkStatus();
}

absl::Status ParsePack(const Operator* op, BuiltinDataAllocator* allocator,
                       void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLitePackParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLitePackParams>();
  TFL_FILE_ENSURE(params != nullptr);

  const PackOptions* schema_params = op->builtin_options_as_PackOptions();

  if (schema_params != nullptr) {
    params->values_count = schema_params->values_count();
    params->axis = schema_params->axis();
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParsePad(const Operator*, BuiltinDataAllocator*, void**) {
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParsePadV2(const Operator*, BuiltinDataAllocator*, void**) {
  return OkStatus();
}

absl::Status ParsePool(const Operator* op, BuiltinDataAllocator* allocator,
                       void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLitePoolParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLitePoolParams>();
  TFL_FILE_ENSURE(params != nullptr);

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
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParsePow(const Operator*, BuiltinDataAllocator*, void**) {
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParsePrelu(const Operator*, BuiltinDataAllocator*, void**) {
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseQuantize(const Operator*, BuiltinDataAllocator*, void**) {
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseReadVariable(const Operator*, BuiltinDataAllocator*, void**) {
  return OkStatus();
}

absl::Status ParseReducer(const Operator* op, BuiltinDataAllocator* allocator,
                          void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);

  std::unique_ptr<TfLiteReducerParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteReducerParams>();
  TFL_FILE_ENSURE(params != nullptr);

  const ReducerOptions* schema_params = op->builtin_options_as_ReducerOptions();

  if (schema_params != nullptr) {
    params->keep_dims = schema_params->keep_dims();
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseRelu(const Operator*, BuiltinDataAllocator*, void**) {
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseRelu6(const Operator*, BuiltinDataAllocator*, void**) {
  return OkStatus();
}

absl::Status ParseReshape(const Operator* op, BuiltinDataAllocator* allocator,
                          void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);

  std::unique_ptr<TfLiteReshapeParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteReshapeParams>();
  TFL_FILE_ENSURE(params != nullptr);

  const ReshapeOptions* schema_params = op->builtin_options_as_ReshapeOptions();

  if (schema_params != nullptr) {
    const flatbuffers::Vector<int32_t>* new_shape = schema_params->new_shape();
    if (new_shape != nullptr) {
      TFL_FILE_ENSURE_STATUS(FlatBufferIntVectorToArray(
          sizeof(params->shape), new_shape, params->shape, "reshape"));
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
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return OkStatus();
}

absl::Status ParseResizeBilinear(const Operator* op,
                                 BuiltinDataAllocator* allocator,
                                 void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteResizeBilinearParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteResizeBilinearParams>();
  TFL_FILE_ENSURE(params != nullptr);

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
  return OkStatus();
}

absl::Status ParseResizeNearestNeighbor(const Operator* op,
                                        BuiltinDataAllocator* allocator,
                                        void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteResizeNearestNeighborParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteResizeNearestNeighborParams>();
  TFL_FILE_ENSURE(params != nullptr);

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
  return OkStatus();
}

absl::Status ParseStablehloReduceWindow(const Operator* op,
                                        BuiltinDataAllocator* allocator,
                                        void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  auto params = safe_allocator.Allocate<TfLiteStablehloReduceWindowParams>();

  const StablehloReduceWindowOptions* schema_params =
      op->builtin_options_2_as_StablehloReduceWindowOptions();
  if (schema_params) {
    if (!schema_params->window_dimensions() ||
        schema_params->window_dimensions()->empty()) {
      auto error_message =
          "'window_dimensions' attribute is not optional for "
          "'stablehlo.reduce_window' and cannot be empty.";
      ABSL_LOG(ERROR) << error_message;
      return absl::InvalidArgumentError(error_message);
    }

    const size_t rank = schema_params->window_dimensions()->size();

    auto LoadAttr = [](int64_t* params_array, size_t params_array_size_bytes,
                       const flatbuffers::Vector<int64_t>* flatbuffer_vector,
                       const char* attr_name, const size_t expected_size,
                       const int64_t fill_value) -> absl::Status {
      if (flatbuffer_vector && !flatbuffer_vector->empty()) {
        if (expected_size != 0 && flatbuffer_vector->size() != expected_size) {
          auto error_message = absl::StrFormat(
              "'%s' attribute of 'stablehlo.reduce_window' does not have the "
              "expected size (%llu != %llu).",
              attr_name, flatbuffer_vector->size(), expected_size);
          ABSL_LOG(ERROR) << error_message;
          return absl::InvalidArgumentError(error_message);
        }
        absl::Status status = FlatBufferIntVectorToArray(
            params_array_size_bytes, flatbuffer_vector, params_array,
            "stablehlo.reduce_window");
        if (!status.ok()) {
          auto error_message = absl::StrFormat("%s Check the '%s' attribute.",
                                               status.message(), attr_name);
          ABSL_LOG(ERROR) << error_message;
          return absl::InvalidArgumentError(error_message);
        }
      } else {
        std::fill_n(params_array, params_array_size_bytes / sizeof(int64_t),
                    fill_value);
      }
      return OkStatus();
    };

    TFL_FILE_ENSURE_STATUS(
        LoadAttr(params->window_dimensions, sizeof(params->window_dimensions),
                 schema_params->window_dimensions(), "window_dimensions",
                 /*expected_size=*/rank, /*fill_value=*/1));
    TFL_FILE_ENSURE_STATUS(
        LoadAttr(params->window_strides, sizeof(params->window_strides),
                 schema_params->window_strides(), "window_strides",
                 /*expected_size=*/rank, /*fill_value=*/1));
    TFL_FILE_ENSURE_STATUS(
        LoadAttr(params->base_dilations, sizeof(params->base_dilations),
                 schema_params->base_dilations(), "base_dilations",
                 /*expected_size=*/rank, /*fill_value=*/1));
    TFL_FILE_ENSURE_STATUS(
        LoadAttr(params->window_dilations, sizeof(params->window_dilations),
                 schema_params->window_dilations(), "window_dilations",
                 /*expected_size=*/rank, /*fill_value=*/1));
    TFL_FILE_ENSURE_STATUS(LoadAttr(params->padding, sizeof(params->padding),
                                    schema_params->padding(), "padding",
                                    /*expected_size=*/2 * rank,
                                    /*fill_value=*/0));

    params->body_subgraph_index = schema_params->body_subgraph_index();
    *builtin_data = params.release();
    return OkStatus();
  }
  auto error_message =
      "Could not get 'stablehlo.reduce_window' operation parameters.";
  ABSL_LOG(ERROR) << error_message;
  return absl::InvalidArgumentError(error_message);
}

absl::Status ParseStablehloScatter(const Operator* op,
                                   BuiltinDataAllocator* allocator,
                                   void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteStablehloScatterParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteStablehloScatterParams>();
  TFL_FILE_ENSURE(params != nullptr);

  const StablehloScatterOptions* schema_params =
      op->builtin_options_2_as_StablehloScatterOptions();
  if (schema_params) {
    params->indices_are_sorted = schema_params->indices_are_sorted();

    if (schema_params->update_window_dims()) {
      TFL_FILE_ENSURE_STATUS(FlatBufferIntVectorToArray<int64_t>(
          schema_params->update_window_dims()->size() * sizeof(int64_t),
          schema_params->update_window_dims(), params->update_window_dims,
          "stablehlo_scatter"));
      params->num_update_window_dims =
          schema_params->update_window_dims()->size();
    }

    if (schema_params->inserted_window_dims()) {
      TFL_FILE_ENSURE_STATUS(FlatBufferIntVectorToArray<int64_t>(
          schema_params->inserted_window_dims()->size() * sizeof(int64_t),
          schema_params->inserted_window_dims(), params->inserted_window_dims,
          "stablehlo_scatter"));
      params->num_inserted_window_dims =
          schema_params->inserted_window_dims()->size();
    }

    if (schema_params->scatter_dims_to_operand_dims()) {
      TFL_FILE_ENSURE_STATUS(FlatBufferIntVectorToArray<int64_t>(
          schema_params->scatter_dims_to_operand_dims()->size() *
              sizeof(int64_t),
          schema_params->scatter_dims_to_operand_dims(),
          params->scatter_dims_to_operand_dims, "stablehlo_scatter"));
      params->num_scatter_dims_to_operand_dims =
          schema_params->scatter_dims_to_operand_dims()->size();
    }

    params->index_vector_dim = schema_params->index_vector_dim();
    params->unique_indices = schema_params->unique_indices();
    params->update_computation_subgraph_index =
        schema_params->update_computation_subgraph_index();
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }
  *builtin_data = params.release();
  return OkStatus();
}

absl::Status ParseStablehloRngBitGenerator(const Operator* op,
                                           BuiltinDataAllocator* allocator,
                                           void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteStablehloRngBitGeneratorParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteStablehloRngBitGeneratorParams>();
  TFL_FILE_ENSURE(params != nullptr);

  const StablehloRngBitGeneratorOptions* schema_params =
      op->builtin_options_2_as_StablehloRngBitGeneratorOptions();
  if (schema_params != nullptr) {
    params->algorithm = ConvertRngAlgorithm(schema_params->algorithm());
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return OkStatus();
}

absl::Status ParseStablehloGather(const Operator* op,
                                  BuiltinDataAllocator* allocator,
                                  void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteStablehloGatherParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteStablehloGatherParams>();
  TFL_FILE_ENSURE(params != nullptr);

  const StablehloGatherOptions* schema_params =
      op->builtin_options_2_as_StablehloGatherOptions();

  if (schema_params != nullptr) {
    TFL_FILE_ENSURE_STATUS(FlatBufferIntVectorToArray<int64_t>(
        /*max_size_of_buffer=*/schema_params->offset_dims()->size() *
            sizeof(int64_t),
        /*flat_vector=*/schema_params->offset_dims(),
        /*buffer=*/params->offset_dims,
        /*op_name=*/"stablehlo_gather"));
    params->num_offset_dims = schema_params->offset_dims()->size();

    TFL_FILE_ENSURE_STATUS(FlatBufferIntVectorToArray<int64_t>(
        schema_params->collapsed_slice_dims()->size() * sizeof(int64_t),
        schema_params->collapsed_slice_dims(), params->collapsed_slice_dims,
        "stablehlo_gather"));
    params->num_collapsed_slice_dims =
        schema_params->collapsed_slice_dims()->size();

    TFL_FILE_ENSURE_STATUS(FlatBufferIntVectorToArray<int64_t>(
        schema_params->start_index_map()->size() * sizeof(int64_t),
        schema_params->start_index_map(), params->start_index_map,
        "stablehlo_gather"));
    params->num_start_index_map = schema_params->start_index_map()->size();

    params->index_vector_dim = schema_params->index_vector_dim();

    TFL_FILE_ENSURE_STATUS(FlatBufferIntVectorToArray<int64_t>(
        schema_params->slice_sizes()->size() * sizeof(int64_t),
        schema_params->slice_sizes(), params->slice_sizes, "stablehlo_gather"));
    params->num_slice_sizes = schema_params->slice_sizes()->size();

    params->indices_are_sorted = schema_params->indices_are_sorted();
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return OkStatus();
}

absl::Status ParseStablehloPad(const Operator* op,
                               BuiltinDataAllocator* allocator,
                               void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  auto params = safe_allocator.Allocate<TfLiteStablehloPadParams>();
  const StablehloPadOptions* schema_params =
      op->builtin_options_2_as_StablehloPadOptions();

  if (schema_params) {
    auto LoadAttr =
        [](int64_t* params_array, const size_t params_array_size_bytes,
           const flatbuffers::Vector<int64_t>* const flatbuffer_vector,
           const char* const attr_name) -> absl::Status {
      absl::Status status =
          FlatBufferIntVectorToArray(params_array_size_bytes, flatbuffer_vector,
                                     params_array, "stablehlo.pad");
      if (!status.ok()) {
        auto error_message = absl::StrFormat("%s Check the '%s' attribute.",
                                             status.message(), attr_name);
        ABSL_LOG(ERROR) << error_message;
        return absl::InvalidArgumentError(error_message);
      }
      return status;
    };

    TFL_FILE_ENSURE_STATUS(
        LoadAttr(params->edge_padding_low, sizeof(params->edge_padding_low),
                 schema_params->edge_padding_low(), "edge_padding_low"));
    TFL_FILE_ENSURE_STATUS(
        LoadAttr(params->edge_padding_high, sizeof(params->edge_padding_high),
                 schema_params->edge_padding_high(), "edge_padding_high"));
    TFL_FILE_ENSURE_STATUS(
        LoadAttr(params->interior_padding, sizeof(params->interior_padding),
                 schema_params->interior_padding(), "interior_padding"));
    if (schema_params->edge_padding_low()->size() !=
            schema_params->edge_padding_high()->size() ||
        schema_params->edge_padding_low()->size() !=
            schema_params->interior_padding()->size()) {
      auto error_message =
          "'stablehlo.pad' operation parameter array sizes are not consistent.";
      ABSL_LOG(ERROR) << error_message;
      return absl::InvalidArgumentError(error_message);
    }
    *builtin_data = params.release();
    return OkStatus();
  }
  auto error_message = "Could not get 'stablehlo.pad' operation parameters.";
  ABSL_LOG(ERROR) << error_message;
  return absl::InvalidArgumentError(error_message);
}

absl::Status ParseStablehloComposite(const Operator* op,
                                     BuiltinDataAllocator* allocator,
                                     void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  auto params = safe_allocator.Allocate<TfLiteStablehloCompositeParams>();
  const StableHLOCompositeOptions* schema_params =
      op->builtin_options_2_as_StableHLOCompositeOptions();
  if (schema_params) {
    params->name = schema_params->name()->c_str();
    params->version = schema_params->version();
    params->subgraph_index = schema_params->decomposition_subgraph_index();
    params->attributes = schema_params->composite_attributes()->data();
    params->attributes_size = schema_params->composite_attributes()->size();
    *builtin_data = params.release();
    return OkStatus();
  }
  auto error_message =
      "Could not get 'stablehlo.composite' operation parameters.";
  ABSL_LOG(ERROR) << error_message;
  return absl::InvalidArgumentError(error_message);
}

absl::Status ParseStablehloShiftLeft(const Operator* op,
                                     BuiltinDataAllocator* allocator,
                                     void** builtin_data) {
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseRound(const Operator*, BuiltinDataAllocator*, void**) {
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseRsqrt(const Operator*, BuiltinDataAllocator*, void**) {
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseSelectV2(const Operator*, BuiltinDataAllocator*, void**) {
  return OkStatus();
}

absl::Status ParseShape(const Operator* op, BuiltinDataAllocator* allocator,
                        void** builtin_data) {
  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteShapeParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteShapeParams>();
  TFL_FILE_ENSURE(params != nullptr);

  const ShapeOptions* schema_params = op->builtin_options_as_ShapeOptions();

  if (schema_params != nullptr) {
    TFL_FILE_ENSURE_STATUS(
        ConvertTensorType(schema_params->out_type(), &params->out_type));
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseSin(const Operator*, BuiltinDataAllocator*, void**) {
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseSlice(const Operator*, BuiltinDataAllocator*, void**) {
  return OkStatus();
}

absl::Status ParseSoftmax(const Operator* op, BuiltinDataAllocator* allocator,
                          void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteSoftmaxParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteSoftmaxParams>();
  TFL_FILE_ENSURE(params != nullptr);

  const SoftmaxOptions* schema_params = op->builtin_options_as_SoftmaxOptions();

  if (schema_params != nullptr) {
    params->beta = schema_params->beta();
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseSpaceToBatchNd(const Operator*, BuiltinDataAllocator*,
                                 void**) {
  return OkStatus();
}

absl::Status ParseSpaceToDepth(const Operator* op,
                               BuiltinDataAllocator* allocator,
                               void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteSpaceToDepthParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteSpaceToDepthParams>();
  TFL_FILE_ENSURE(params != nullptr);

  const auto* schema_params = op->builtin_options_as_SpaceToDepthOptions();
  if (schema_params != nullptr) {
    params->block_size = schema_params->block_size();
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return OkStatus();
}

absl::Status ParseSplit(const Operator* op, BuiltinDataAllocator* allocator,
                        void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteSplitParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteSplitParams>();
  TFL_FILE_ENSURE(params != nullptr);

  const SplitOptions* schema_params = op->builtin_options_as_SplitOptions();

  if (schema_params != nullptr) {
    params->num_splits = schema_params->num_splits();
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return OkStatus();
}

absl::Status ParseSplitV(const Operator* op, BuiltinDataAllocator* allocator,
                         void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);
  SafeBuiltinDataAllocator safe_allocator(allocator);

  std::unique_ptr<TfLiteSplitVParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteSplitVParams>();
  TFL_FILE_ENSURE(params != nullptr);

  const SplitVOptions* schema_params = op->builtin_options_as_SplitVOptions();

  if (schema_params != nullptr) {
    params->num_splits = schema_params->num_splits();
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return OkStatus();
}

absl::Status ParseUnidirectionalSequenceLSTM(const Operator* op,
                                             BuiltinDataAllocator* allocator,
                                             void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);
  SafeBuiltinDataAllocator safe_allocator(allocator);
  auto params =
      safe_allocator.Allocate<TfLiteUnidirectionalSequenceLSTMParams>();
  TFL_FILE_ENSURE(params != nullptr);
  if (const auto* seq_lstm_params =
          op->builtin_options_as_UnidirectionalSequenceLSTMOptions()) {
    params->activation =
        ConvertActivation(seq_lstm_params->fused_activation_function());
    params->cell_clip = seq_lstm_params->cell_clip();
    params->proj_clip = seq_lstm_params->proj_clip();
    params->time_major = seq_lstm_params->time_major();
    params->asymmetric_quantize_inputs =
        seq_lstm_params->asymmetric_quantize_inputs();
    params->diagonal_recurrent_tensors =
        seq_lstm_params->diagonal_recurrent_tensors();
  }
  *builtin_data = params.release();
  return OkStatus();
}

absl::Status ParseSqueeze(const Operator* op, BuiltinDataAllocator* allocator,
                          void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);
  SafeBuiltinDataAllocator safe_allocator(allocator);

  std::unique_ptr<TfLiteSqueezeParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteSqueezeParams>();
  TFL_FILE_ENSURE(params != nullptr);

  const SqueezeOptions* schema_params = op->builtin_options_as_SqueezeOptions();

  if (schema_params != nullptr) {
    const auto* squeeze_dims = schema_params->squeeze_dims();
    if (squeeze_dims != nullptr) {
      TFL_FILE_ENSURE_STATUS(
          FlatBufferIntVectorToArray(sizeof(params->squeeze_dims), squeeze_dims,
                                     params->squeeze_dims, "squeeze"));
      params->num_squeeze_dims = squeeze_dims->size();
    } else {
      params->num_squeeze_dims = 0;
    }
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseSqrt(const Operator*, BuiltinDataAllocator*, void**) {
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseSquare(const Operator*, BuiltinDataAllocator*, void**) {
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseSquaredDifference(const Operator*, BuiltinDataAllocator*,
                                    void**) {
  return OkStatus();
}

absl::Status ParseStridedSlice(const Operator* op,
                               BuiltinDataAllocator* allocator,
                               void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteStridedSliceParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteStridedSliceParams>();
  TFL_FILE_ENSURE(params != nullptr);

  const StridedSliceOptions* schema_params =
      op->builtin_options_as_StridedSliceOptions();

  if (schema_params != nullptr) {
    params->begin_mask = schema_params->begin_mask();
    params->end_mask = schema_params->end_mask();
    params->ellipsis_mask = schema_params->ellipsis_mask();
    params->new_axis_mask = schema_params->new_axis_mask();
    params->shrink_axis_mask = schema_params->shrink_axis_mask();
    params->offset = schema_params->offset();
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return OkStatus();
}

absl::Status ParseSub(const Operator* op, BuiltinDataAllocator* allocator,
                      void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteSubParams, SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteSubParams>();
  TFL_FILE_ENSURE(params != nullptr);

  const SubOptions* schema_params = op->builtin_options_as_SubOptions();

  if (schema_params != nullptr) {
    params->activation =
        ConvertActivation(schema_params->fused_activation_function());
    params->pot_scale_int16 = schema_params->pot_scale_int16();
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return OkStatus();
}

absl::Status ParseSvdf(const Operator* op, BuiltinDataAllocator* allocator,
                       void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteSVDFParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteSVDFParams>();
  TFL_FILE_ENSURE(params != nullptr);

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
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseTanh(const Operator*, BuiltinDataAllocator*, void**) {
  return OkStatus();
}
//
// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseTranspose(const Operator*, BuiltinDataAllocator*, void**) {
  return OkStatus();
}

absl::Status ParseTransposeConv(const Operator* op,
                                BuiltinDataAllocator* allocator,
                                void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteTransposeConvParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteTransposeConvParams>();
  TFL_FILE_ENSURE(params != nullptr);
  const TransposeConvOptions* transpose_conv_params =
      op->builtin_options_as_TransposeConvOptions();
  if (transpose_conv_params != nullptr) {
    params->padding = ConvertPadding(transpose_conv_params->padding());
    params->stride_width = transpose_conv_params->stride_w();
    params->stride_height = transpose_conv_params->stride_h();

    params->activation =
        ConvertActivation(transpose_conv_params->fused_activation_function());
    TFL_FILE_ENSURE_STATUS(
        ConvertTensorType(transpose_conv_params->quantized_bias_type(),
                          &params->quantized_bias_type));
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }
  *builtin_data = params.release();
  return OkStatus();
}

absl::Status ParseUnpack(const Operator* op, BuiltinDataAllocator* allocator,
                         void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteUnpackParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteUnpackParams>();
  TFL_FILE_ENSURE(params != nullptr);

  const UnpackOptions* schema_params = op->builtin_options_as_UnpackOptions();

  if (schema_params != nullptr) {
    params->num = schema_params->num();
    params->axis = schema_params->axis();
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return OkStatus();
}

absl::Status ParseVarHandle(const Operator* op, BuiltinDataAllocator* allocator,
                            void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteVarHandleParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteVarHandleParams>();
  TFL_FILE_ENSURE(params != nullptr);

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
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return OkStatus();
}

absl::Status ParseWhile(const Operator* op, BuiltinDataAllocator* allocator,
                        void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteWhileParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteWhileParams>();
  TFL_FILE_ENSURE(params != nullptr);

  const WhileOptions* schema_params = op->builtin_options_as_WhileOptions();

  if (schema_params != nullptr) {
    params->cond_subgraph_index = schema_params->cond_subgraph_index();
    params->body_subgraph_index = schema_params->body_subgraph_index();
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return OkStatus();
}

absl::Status ParseStablehloCase(const Operator* op,
                                BuiltinDataAllocator* allocator,
                                void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  auto params = safe_allocator.Allocate<TfLiteStablehloCaseParams>();

  const tflite::StablehloCaseOptions* schema_params =
      op->builtin_options_2_as_StablehloCaseOptions();

  if (!schema_params) {
    return absl::InternalError("Could not get 'stablehlo.case' params");
  }

  TFL_FILE_ENSURE_STATUS(FlatBufferIntVectorToArray(
      sizeof(params->branch_subgraph_indices),
      schema_params->branch_subgraph_indices(), params->branch_subgraph_indices,
      "stablehlo.case"));

  params->num_branches = schema_params->branch_subgraph_indices()->size();
  *builtin_data = params.release();

  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseZerosLike(const Operator*, BuiltinDataAllocator*, void**) {
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseBitwiseXor(const Operator*, BuiltinDataAllocator*, void**) {
  return OkStatus();
}

// We have this parse function instead of directly returning OkStatus() from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
absl::Status ParseRightShift(const Operator*, BuiltinDataAllocator*, void**) {
  return OkStatus();
}

absl::Status ParseOpData(const Operator* op, BuiltinOperator op_type,
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
  auto error_message =
      "ParseOpData is unsupported on TfLiteMicro, please use the operator "
      "specific parse functions (e.g. ParseAdd etc.).\n";
  ABSL_LOG(ERROR) << error_message;
  return absl::UnimplementedError(error_message);
#else
  return ParseOpDataTfLite(op, op_type, allocator, builtin_data);
#endif
}

}  // namespace flatbuffer_conversions
}  // namespace tflite_file
// LINT.ThenChange(//tensorflow/lite/core/api/flatbuffer_conversions.cc)
