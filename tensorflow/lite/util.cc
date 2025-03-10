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
#include "tensorflow/lite/util.h"

#include <cassert>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/array.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/macros.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

TfLiteStatus UnresolvedOpInvoke(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_KERNEL_LOG(context,
                     "Encountered an unresolved custom op. Did you miss "
                     "a custom op or delegate?");
  return kTfLiteError;
}

}  // namespace

bool IsFlexOp(const char* custom_name) {
  return custom_name && strncmp(custom_name, kFlexCustomCodePrefix,
                                strlen(kFlexCustomCodePrefix)) == 0;
}

TfLiteIntArray* ConvertVectorToTfLiteIntArray(const std::vector<int>& input) {
  return BuildTfLiteArray(input).release();
}

TfLiteIntArray* ConvertArrayToTfLiteIntArray(const int ndims, const int* dims) {
  return BuildTfLiteArray(ndims, dims).release();
}

bool EqualArrayAndTfLiteIntArray(const TfLiteIntArray* a, const int b_size,
                                 const int* b) {
  if (!a) return false;
  if (a->size != b_size) return false;
  for (int i = 0; i < a->size; ++i) {
    if (a->data[i] != b[i]) return false;
  }
  return true;
}

size_t CombineHashes(std::initializer_list<size_t> hashes) {
  size_t result = 0;
  // Hash combiner used by TensorFlow core.
  for (size_t hash : hashes) {
    result = result ^
             (hash + 0x9e3779b97f4a7800ULL + (result << 10) + (result >> 4));
  }
  return result;
}

TfLiteStatus GetSizeOfType(TfLiteContext* context, const TfLiteType type,
                           size_t* bytes) {
  // TODO(levp): remove the default case so that new types produce compilation
  // error.
  switch (type) {
    case kTfLiteFloat32:
      *bytes = sizeof(float);
      break;
    case kTfLiteInt32:
      *bytes = sizeof(int32_t);
      break;
    case kTfLiteUInt32:
      *bytes = sizeof(uint32_t);
      break;
    case kTfLiteUInt8:
      *bytes = sizeof(uint8_t);
      break;
    case kTfLiteInt64:
      *bytes = sizeof(int64_t);
      break;
    case kTfLiteUInt64:
      *bytes = sizeof(uint64_t);
      break;
    case kTfLiteBool:
      *bytes = sizeof(bool);
      break;
    case kTfLiteComplex64:
      *bytes = sizeof(std::complex<float>);
      break;
    case kTfLiteComplex128:
      *bytes = sizeof(std::complex<double>);
      break;
    case kTfLiteUInt16:
      *bytes = sizeof(uint16_t);
      break;
    case kTfLiteInt16:
      *bytes = sizeof(int16_t);
      break;
    case kTfLiteInt8:
      *bytes = sizeof(int8_t);
      break;
    case kTfLiteFloat16:
      *bytes = sizeof(TfLiteFloat16);
      break;
    case kTfLiteBFloat16:
      *bytes = sizeof(TfLiteBFloat16);
      break;
    case kTfLiteFloat64:
      *bytes = sizeof(double);
      break;
    case kTfLiteInt4:
      // TODO(b/246647008): Multiplying this value by the number of elements
      // does not yield the size of a tensor when 4-bit values are packed
      // 2 to a byte.
      *bytes = sizeof(int8_t);
      break;
    default:
      if (context) {
        TF_LITE_KERNEL_LOG(
            context,
            "Type %d is unsupported. Only float16, float32, float64, int8, "
            "int16, int32, int64, uint8, uint64, bool, complex64 and "
            "complex128 supported currently.",
            type);
      }
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteRegistration CreateUnresolvedCustomOp(const char* custom_op_name) {
  return TfLiteRegistration{nullptr,
                            nullptr,
                            nullptr,
                            /*invoke*/ &UnresolvedOpInvoke,
                            nullptr,
                            BuiltinOperator_CUSTOM,
                            custom_op_name,
                            1};
}

bool IsUnresolvedCustomOp(const TfLiteRegistration& registration) {
  return registration.builtin_code == tflite::BuiltinOperator_CUSTOM &&
         registration.invoke == &UnresolvedOpInvoke;
}

std::string GetOpNameByRegistration(const TfLiteRegistration& registration) {
  auto op = registration.builtin_code;
  std::string result =
      EnumNameBuiltinOperator(static_cast<BuiltinOperator>(op));
  if ((op == kTfLiteBuiltinCustom || op == kTfLiteBuiltinDelegate) &&
      registration.custom_name) {
    result += " " + std::string(registration.custom_name);
  }
  return result;
}

bool IsValidationSubgraph(const char* name) {
  // NOLINTNEXTLINE: can't use absl::StartsWith as absl is not allowed.
  return name && std::string(name).find(kValidationSubgraphNamePrefix) == 0;
}

TfLiteStatus MultiplyAndCheckOverflow(size_t a, size_t b, size_t* product) {
  // Multiplying a * b where a and b are size_t cannot result in overflow in a
  // size_t accumulator if both numbers have no non-zero bits in their upper
  // half.
  constexpr size_t size_t_bits = 8 * sizeof(size_t);
  constexpr size_t overflow_upper_half_bit_position = size_t_bits / 2;
  *product = a * b;
  // If neither integers have non-zero bits past 32 bits can't overflow.
  // Otherwise check using slow devision.
  if (TFLITE_EXPECT_FALSE((a | b) >> overflow_upper_half_bit_position != 0)) {
    if (a != 0 && *product / a != b) return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus BytesRequired(TfLiteType type, const int* dims, size_t dims_size,
                           size_t* bytes, TfLiteContext* context_) {
  TF_LITE_ENSURE(context_, bytes != nullptr);
  // When 'dims_size' is 0, we simply assume it's a scalar. Therefore, we start
  // 'count' as 1.
  size_t count = 1;
  for (int k = 0; k < dims_size; k++) {
    size_t old_count = count;
    TF_LITE_ENSURE_MSG(
        context_,
        MultiplyAndCheckOverflow(old_count, dims[k], &count) == kTfLiteOk,
        "BytesRequired number of elements overflowed.\n");
  }
  size_t type_size = 0;
  TF_LITE_ENSURE_OK(context_, GetSizeOfType(context_, type, &type_size));
  TF_LITE_ENSURE_MSG(
      context_, MultiplyAndCheckOverflow(type_size, count, bytes) == kTfLiteOk,
      "BytesRequired number of bytes overflowed.\n");

  // GetSizeOfType doesn't work for kTfLiteInt4 due to it having 2 values packed
  // into 1 byte so the output of GetSizeOfType is the same as int8 aka 1 byte.
  // Thus the required bytes must be divided by half after everything for int4.
  if (type == kTfLiteInt4) {
    *bytes = (*bytes + 1) / 2;
  }

  return kTfLiteOk;
}

TensorUniquePtr BuildTfLiteTensor() {
  auto tensor = TensorUniquePtr((TfLiteTensor*)calloc(1, sizeof(TfLiteTensor)));
  tensor->buffer_handle = kTfLiteNullBufferHandle;
  return tensor;
}

TensorUniquePtr BuildTfLiteTensor(TfLiteType type, const std::vector<int>& dims,
                                  TfLiteAllocationType allocation_type) {
  return BuildTfLiteTensor(type, BuildTfLiteArray(dims), allocation_type);
}

// Allocates an appropriate sized buffer underneath returned tensor
// based on the value of `dims`. Since arena allocated tensors should not
// be managed by the user, we do not permit `kTfLiteArena` as a
// valid allocation type.
TensorUniquePtr BuildTfLiteTensor(TfLiteType type, IntArrayUniquePtr dims,
                                  TfLiteAllocationType allocation_type) {
  assert(allocation_type != kTfLiteArenaRw &&
         allocation_type != kTfLiteArenaRwPersistent);
  TfLiteIntArray* dims_data = dims.release();
  if (!dims_data) {
    return nullptr;
  }
  size_t bytes;
  auto compute_bytes_stat =
      BytesRequired(type, dims_data->data, dims_data->size, &bytes, nullptr);
  if (compute_bytes_stat != kTfLiteOk) {
    return nullptr;
  }
  TensorUniquePtr t = BuildTfLiteTensor();
  TfLiteTensorReset(type, /*name=*/nullptr, dims_data, /*quantization=*/{},
                    /*buffer=*/nullptr, bytes, allocation_type,
                    /*allocation=*/nullptr, /*is_variable=*/false,
                    /*tensor=*/t.get());
  TfLiteTensorRealloc(bytes, t.get());
  return t;
}

int GetBuiltinDataSize(BuiltinOperator op) {
  switch (op) {
    case BuiltinOperator_ADD:
      return sizeof(TfLiteAddParams);
    case BuiltinOperator_AVERAGE_POOL_2D:
      return sizeof(TfLitePoolParams);
    case BuiltinOperator_CONCATENATION:
      return sizeof(TfLiteConcatenationParams);
    case BuiltinOperator_CONV_2D:
      return sizeof(TfLiteConvParams);
    case BuiltinOperator_DEPTHWISE_CONV_2D:
      return sizeof(TfLiteDepthwiseConvParams);
    case BuiltinOperator_DEPTH_TO_SPACE:
      return sizeof(TfLiteDepthToSpaceParams);
    case BuiltinOperator_DEQUANTIZE:
      return 0;
    case BuiltinOperator_EMBEDDING_LOOKUP:
      return 0;
    case BuiltinOperator_FLOOR:
      return 0;
    case BuiltinOperator_FULLY_CONNECTED:
      return sizeof(TfLiteFullyConnectedParams);
    case BuiltinOperator_HASHTABLE_LOOKUP:
      return 0;
    case BuiltinOperator_L2_NORMALIZATION:
      return sizeof(TfLiteL2NormParams);
    case BuiltinOperator_L2_POOL_2D:
      return sizeof(TfLitePoolParams);
    case BuiltinOperator_LOCAL_RESPONSE_NORMALIZATION:
      return sizeof(TfLiteLocalResponseNormParams);
    case BuiltinOperator_LOGISTIC:
      return 0;
    case BuiltinOperator_LSH_PROJECTION:
      return sizeof(TfLiteLSHProjectionParams);
    case BuiltinOperator_LSTM:
      return sizeof(TfLiteLSTMParams);
    case BuiltinOperator_MAX_POOL_2D:
      return sizeof(TfLitePoolParams);
    case BuiltinOperator_MUL:
      return sizeof(TfLiteMulParams);
    case BuiltinOperator_RELU:
      return 0;
    case BuiltinOperator_RELU_N1_TO_1:
      return 0;
    case BuiltinOperator_RELU6:
      return 0;
    case BuiltinOperator_RESHAPE:
      return sizeof(TfLiteReshapeParams);
    case BuiltinOperator_RESIZE_BILINEAR:
      return sizeof(TfLiteResizeBilinearParams);
    case BuiltinOperator_RNN:
      return sizeof(TfLiteRNNParams);
    case BuiltinOperator_SOFTMAX:
      return sizeof(TfLiteSoftmaxParams);
    case BuiltinOperator_SPACE_TO_DEPTH:
      return sizeof(TfLiteSpaceToDepthParams);
    case BuiltinOperator_SVDF:
      return sizeof(TfLiteSVDFParams);
    case BuiltinOperator_TANH:
      return 0;
    case BuiltinOperator_CONCAT_EMBEDDINGS:
      return 0;
    case BuiltinOperator_SKIP_GRAM:
      return sizeof(TfLiteSkipGramParams);
    case BuiltinOperator_CALL:
      return sizeof(TfLiteCallOnceParams);
    case BuiltinOperator_CUSTOM:
      return 0;
    case BuiltinOperator_EMBEDDING_LOOKUP_SPARSE:
      return sizeof(TfLiteEmbeddingLookupSparseParams);
    case BuiltinOperator_PAD:
      return sizeof(TfLitePadParams);
    case BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_RNN:
      return sizeof(TfLiteSequenceRNNParams);
    case BuiltinOperator_GATHER:
      return sizeof(TfLiteGatherParams);
    case BuiltinOperator_BATCH_TO_SPACE_ND:
      return sizeof(TfLiteBatchToSpaceNDParams);
    case BuiltinOperator_SPACE_TO_BATCH_ND:
      return sizeof(TfLiteSpaceToBatchNDParams);
    case BuiltinOperator_TRANSPOSE:
      return sizeof(TfLiteTransposeParams);
    case BuiltinOperator_MEAN:
      return sizeof(TfLiteReducerParams);
    case BuiltinOperator_SUB:
      return sizeof(TfLiteSubParams);
    case BuiltinOperator_DIV:
      return sizeof(TfLiteDivParams);
    case BuiltinOperator_SQUEEZE:
      return sizeof(TfLiteSqueezeParams);
    case BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM:
      return sizeof(TfLiteUnidirectionalSequenceLSTMParams);
    case BuiltinOperator_STRIDED_SLICE:
      return sizeof(TfLiteStridedSliceParams);
    case BuiltinOperator_BIDIRECTIONAL_SEQUENCE_RNN:
      return sizeof(TfLiteBidirectionalSequenceRNNParams);
    case BuiltinOperator_EXP:
      return 0;
    case BuiltinOperator_TOPK_V2:
      return 0;
    case BuiltinOperator_SPLIT:
      return sizeof(TfLiteSplitParams);
    case BuiltinOperator_LOG_SOFTMAX:
      return 0;
    case BuiltinOperator_DELEGATE:
      return sizeof(TfLiteDelegateParams);
    case BuiltinOperator_BIDIRECTIONAL_SEQUENCE_LSTM:
      return sizeof(TfLiteBidirectionalSequenceLSTMParams);
    case BuiltinOperator_CAST:
      return sizeof(TfLiteCastParams);
    case BuiltinOperator_PRELU:
      return 0;
    case BuiltinOperator_MAXIMUM:
      return 0;
    case BuiltinOperator_ARG_MAX:
      return sizeof(TfLiteArgMaxParams);
    case BuiltinOperator_MINIMUM:
      return 0;
    case BuiltinOperator_LESS:
      return 0;
    case BuiltinOperator_NEG:
      return 0;
    case BuiltinOperator_PADV2:
      return sizeof(TfLitePadV2Params);
    case BuiltinOperator_GREATER:
      return 0;
    case BuiltinOperator_GREATER_EQUAL:
      return 0;
    case BuiltinOperator_LESS_EQUAL:
      return 0;
    case BuiltinOperator_SELECT:
      return 0;
    case BuiltinOperator_SLICE:
      return 0;
    case BuiltinOperator_SIN:
      return 0;
    case BuiltinOperator_TRANSPOSE_CONV:
      return sizeof(TfLiteTransposeConvParams);
    case BuiltinOperator_SPARSE_TO_DENSE:
      return sizeof(TfLiteSparseToDenseParams);
    case BuiltinOperator_TILE:
      return 0;
    case BuiltinOperator_EXPAND_DIMS:
      return 0;
    case BuiltinOperator_EQUAL:
      return 0;
    case BuiltinOperator_NOT_EQUAL:
      return 0;
    case BuiltinOperator_LOG:
      return 0;
    case BuiltinOperator_SUM:
      return sizeof(TfLiteReducerParams);
    case BuiltinOperator_SQRT:
      return 0;
    case BuiltinOperator_RSQRT:
      return 0;
    case BuiltinOperator_SHAPE:
      return sizeof(TfLiteShapeParams);
    case BuiltinOperator_POW:
      return 0;
    case BuiltinOperator_ARG_MIN:
      return sizeof(TfLiteArgMinParams);
    case BuiltinOperator_FAKE_QUANT:
      return sizeof(TfLiteFakeQuantParams);
    case BuiltinOperator_REDUCE_PROD:
      return sizeof(TfLiteReducerParams);
    case BuiltinOperator_REDUCE_MAX:
      return sizeof(TfLiteReducerParams);
    case BuiltinOperator_PACK:
      return sizeof(TfLitePackParams);
    case BuiltinOperator_LOGICAL_OR:
      return 0;
    case BuiltinOperator_ONE_HOT:
      return sizeof(TfLiteOneHotParams);
    case BuiltinOperator_LOGICAL_AND:
      return 0;
    case BuiltinOperator_LOGICAL_NOT:
      return 0;
    case BuiltinOperator_UNPACK:
      return sizeof(TfLiteUnpackParams);
    case BuiltinOperator_REDUCE_MIN:
      return sizeof(TfLiteReducerParams);
    case BuiltinOperator_FLOOR_DIV:
      return 0;
    case BuiltinOperator_REDUCE_ANY:
      return sizeof(TfLiteReducerParams);
    case BuiltinOperator_SQUARE:
      return 0;
    case BuiltinOperator_ZEROS_LIKE:
      return 0;
    case BuiltinOperator_FILL:
      return 0;
    case BuiltinOperator_FLOOR_MOD:
      return 0;
    case BuiltinOperator_RANGE:
      return 0;
    case BuiltinOperator_RESIZE_NEAREST_NEIGHBOR:
      return sizeof(TfLiteResizeNearestNeighborParams);
    case BuiltinOperator_LEAKY_RELU:
      return sizeof(TfLiteLeakyReluParams);
    case BuiltinOperator_SQUARED_DIFFERENCE:
      return 0;
    case BuiltinOperator_MIRROR_PAD:
      return sizeof(TfLiteMirrorPaddingParams);
    case BuiltinOperator_ABS:
      return 0;
    case BuiltinOperator_SPLIT_V:
      return sizeof(TfLiteSplitVParams);
    case BuiltinOperator_UNIQUE:
      return sizeof(TfLiteUniqueParams);
    case BuiltinOperator_CEIL:
      return 0;
    case BuiltinOperator_REVERSE_V2:
      return 0;
    case BuiltinOperator_ADD_N:
      return 0;
    case BuiltinOperator_GATHER_ND:
      return 0;
    case BuiltinOperator_COS:
      return 0;
    case BuiltinOperator_WHERE:
      return 0;
    case BuiltinOperator_RANK:
      return sizeof(TfLiteRankParams);
    case BuiltinOperator_ELU:
      return 0;
    case BuiltinOperator_REVERSE_SEQUENCE:
      return sizeof(TfLiteReverseSequenceParams);
    case BuiltinOperator_MATRIX_DIAG:
      return sizeof(TfLiteMatrixDiagParams);
    case BuiltinOperator_QUANTIZE:
      return 0;
    case BuiltinOperator_MATRIX_SET_DIAG:
      return sizeof(TfLiteMatrixSetDiagParams);
    case BuiltinOperator_ROUND:
      return 0;
    case BuiltinOperator_HARD_SWISH:
      return 0;
    case BuiltinOperator_IF:
      return sizeof(TfLiteIfParams);
    case BuiltinOperator_WHILE:
      return sizeof(TfLiteWhileParams);
    case BuiltinOperator_NON_MAX_SUPPRESSION_V4:
      return 0;
    case BuiltinOperator_NON_MAX_SUPPRESSION_V5:
      return 0;
    case BuiltinOperator_SCATTER_ND:
      return 0;
    case BuiltinOperator_SELECT_V2:
      return 0;
    case BuiltinOperator_DENSIFY:
      return 0;
    case BuiltinOperator_SEGMENT_SUM:
      return 0;
    case BuiltinOperator_BATCH_MATMUL:
      return sizeof(TfLiteBatchMatMulParams);
    case BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES:
      return 0;
    case BuiltinOperator_CUMSUM:
      return sizeof(TfLiteCumsumParams);
    case BuiltinOperator_CALL_ONCE:
      return sizeof(TfLiteCallOnceParams);
    case BuiltinOperator_BROADCAST_TO:
      return 0;
    case BuiltinOperator_RFFT2D:
      return 0;
    case BuiltinOperator_CONV_3D:
      return sizeof(TfLiteConv3DParams);
    case BuiltinOperator_IMAG:
      return 0;
    case BuiltinOperator_REAL:
      return 0;
    case BuiltinOperator_COMPLEX_ABS:
      return 0;
    case BuiltinOperator_HASHTABLE:
      return sizeof(TfLiteHashtableParams);
    case BuiltinOperator_HASHTABLE_FIND:
      return 0;
    case BuiltinOperator_HASHTABLE_IMPORT:
      return 0;
    case BuiltinOperator_HASHTABLE_SIZE:
      return 0;
    case BuiltinOperator_REDUCE_ALL:
      return sizeof(TfLiteReducerParams);
    case BuiltinOperator_CONV_3D_TRANSPOSE:
      return sizeof(TfLiteConv3DParams);
    case BuiltinOperator_VAR_HANDLE:
      return sizeof(TfLiteVarHandleParams);
    case BuiltinOperator_READ_VARIABLE:
      return 0;
    case BuiltinOperator_ASSIGN_VARIABLE:
      return 0;
    case BuiltinOperator_BROADCAST_ARGS:
      return 0;
    case BuiltinOperator_RANDOM_STANDARD_NORMAL:
      return sizeof(TfLiteRandomParams);
    case BuiltinOperator_BUCKETIZE:
      return sizeof(TfLiteBucketizeParams);
    case BuiltinOperator_RANDOM_UNIFORM:
      return sizeof(TfLiteRandomParams);
    case BuiltinOperator_MULTINOMIAL:
      return sizeof(TfLiteRandomParams);
    case BuiltinOperator_GELU:
      return sizeof(TfLiteGeluParams);
    case BuiltinOperator_DYNAMIC_UPDATE_SLICE:
      return 0;
    case BuiltinOperator_RELU_0_TO_1:
      return 0;
    case BuiltinOperator_UNSORTED_SEGMENT_PROD:
      return 0;
    case BuiltinOperator_UNSORTED_SEGMENT_MAX:
      return 0;
    case BuiltinOperator_UNSORTED_SEGMENT_SUM:
      return 0;
    case BuiltinOperator_ATAN2:
      return 0;
    case BuiltinOperator_UNSORTED_SEGMENT_MIN:
      return 0;
    case BuiltinOperator_SIGN:
      return 0;
    case BuiltinOperator_BITCAST:
      return 0;
    case BuiltinOperator_BITWISE_XOR:
      return 0;
    case BuiltinOperator_RIGHT_SHIFT:
      return 0;
    case BuiltinOperator_STABLEHLO_LOGISTIC:
      return 0;
    case BuiltinOperator_STABLEHLO_ADD:
      return 0;
    case BuiltinOperator_STABLEHLO_DIVIDE:
      return 0;
    case BuiltinOperator_STABLEHLO_MULTIPLY:
      return 0;
    case BuiltinOperator_STABLEHLO_MAXIMUM:
      return 0;
    case BuiltinOperator_STABLEHLO_RESHAPE:
      return 0;
    case BuiltinOperator_STABLEHLO_CLAMP:
      return 0;
    case BuiltinOperator_STABLEHLO_CONCATENATE:
      return 0;
    case BuiltinOperator_STABLEHLO_BROADCAST_IN_DIM:
      return 0;
    case BuiltinOperator_STABLEHLO_CONVOLUTION:
      return 0;
    case BuiltinOperator_STABLEHLO_SLICE:
      return 0;
    case BuiltinOperator_STABLEHLO_CUSTOM_CALL:
      return 0;
    case BuiltinOperator_STABLEHLO_REDUCE:
      return 0;
    case BuiltinOperator_STABLEHLO_ABS:
      return 0;
    case BuiltinOperator_STABLEHLO_AND:
      return 0;
    case BuiltinOperator_STABLEHLO_COSINE:
      return 0;
    case BuiltinOperator_STABLEHLO_EXPONENTIAL:
      return 0;
    case BuiltinOperator_STABLEHLO_FLOOR:
      return 0;
    case BuiltinOperator_STABLEHLO_LOG:
      return 0;
    case BuiltinOperator_STABLEHLO_MINIMUM:
      return 0;
    case BuiltinOperator_STABLEHLO_NEGATE:
      return 0;
    case BuiltinOperator_STABLEHLO_OR:
      return 0;
    case BuiltinOperator_STABLEHLO_POWER:
      return 0;
    case BuiltinOperator_STABLEHLO_REMAINDER:
      return 0;
    case BuiltinOperator_STABLEHLO_RSQRT:
      return 0;
    case BuiltinOperator_STABLEHLO_SELECT:
      return 0;
    case BuiltinOperator_STABLEHLO_SUBTRACT:
      return 0;
    case BuiltinOperator_STABLEHLO_TANH:
      return 0;
    case BuiltinOperator_STABLEHLO_SCATTER:
      return sizeof(TfLiteStablehloScatterParams);
    case BuiltinOperator_STABLEHLO_COMPARE:
      return 0;
    case BuiltinOperator_STABLEHLO_CONVERT:
      return 0;
    case BuiltinOperator_STABLEHLO_DYNAMIC_SLICE:
      return 0;
    case BuiltinOperator_STABLEHLO_DYNAMIC_UPDATE_SLICE:
      return 0;
    case BuiltinOperator_STABLEHLO_PAD:
      return sizeof(TfLiteStablehloPadParams);
    case BuiltinOperator_STABLEHLO_IOTA:
      return 0;
    case BuiltinOperator_STABLEHLO_DOT_GENERAL:
      return 0;
    case BuiltinOperator_STABLEHLO_REDUCE_WINDOW:
      return sizeof(TfLiteStablehloReduceWindowParams);
    case BuiltinOperator_STABLEHLO_SORT:
      return 0;
    case BuiltinOperator_STABLEHLO_WHILE:
      return 0;
    case BuiltinOperator_STABLEHLO_GATHER:
      return sizeof(TfLiteStablehloGatherParams);
    case BuiltinOperator_STABLEHLO_TRANSPOSE:
      return 0;
    case BuiltinOperator_DILATE:
      return 0;
    case BuiltinOperator_STABLEHLO_RNG_BIT_GENERATOR:
      return sizeof(TfLiteStablehloRngBitGeneratorParams);
    case BuiltinOperator_REDUCE_WINDOW:
      return sizeof(TfLiteReduceWindowParams);
    case BuiltinOperator_STABLEHLO_COMPOSITE:
      return sizeof(TfLiteStablehloCompositeParams);
    case BuiltinOperator_STABLEHLO_SHIFT_LEFT:
      return 0;
    case BuiltinOperator_STABLEHLO_CBRT:
      return 0;
    case BuiltinOperator_STABLEHLO_CASE:
      return sizeof(TfLiteStablehloCaseParams);
  }
  return 0;
}

}  // namespace tflite
