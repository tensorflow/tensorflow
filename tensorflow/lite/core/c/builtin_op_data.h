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
/// WARNING: Users of TensorFlow Lite should not include this file directly,
/// but should instead include
/// "third_party/tensorflow/lite/c/builtin_op_data.h".
/// Only the TensorFlow Lite implementation itself should include this
/// file directly.
#ifndef TENSORFLOW_LITE_CORE_C_BUILTIN_OP_DATA_H_
#define TENSORFLOW_LITE_CORE_C_BUILTIN_OP_DATA_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "tensorflow/lite/core/c/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// TfLiteReshapeParams can't have dynamic data so we fix the maximum possible
// number of dimensions.
#define TFLITE_RESHAPE_PARAMS_MAX_DIMENSION_COUNT 8
#define TFLITE_STABLEHLO_SCATTER_PARAMS_MAX_DIMENSION_COUNT 8
#define TFLITE_STABLEHLO_GATHER_PARAMS_MAX_DIMENSION_COUNT 8
#define TFLITE_STABLEHLO_REDUCE_WINDOW_PARAMS_MAX_DIMENSION_COUNT 8
#define TFLITE_STABLEHLO_PAD_PARAMS_MAX_DIMENSION_COUNT 8

// TODO(aselle): Consider using "if this then that" for testing.

// Useful placeholder to put in otherwise empty structs to avoid size warnings.
typedef struct {
  char dummy;
} EmptyStructPlaceholder;

// IMPORTANT: All new members of structs must be added at the end to ensure
// backwards compatibility.

// Possible padding types (for convolutions)
typedef enum {
  kTfLitePaddingUnknown = 0,
  kTfLitePaddingSame,
  kTfLitePaddingValid,
} TfLitePadding;

typedef enum {
  kTfLiteMirrorPaddingUnknown = 0,
  kTfLiteMirrorPaddingReflect,
  kTfLiteMirrorPaddingSymmetric,
} TfLiteMirrorPaddingMode;

// TODO(b/130259536): We should move this out of builtin_op_data.
typedef struct {
  int width;
  int height;
  int width_offset;
  int height_offset;
} TfLitePaddingValues;

typedef struct {
  TfLiteMirrorPaddingMode mode;
} TfLiteMirrorPaddingParams;

// Possible fused activation functions.
typedef enum {
  kTfLiteActNone = 0,
  kTfLiteActRelu,
  kTfLiteActReluN1To1,  // min(max(-1, x), 1)
  kTfLiteActRelu6,      // min(max(0, x), 6)
  kTfLiteActTanh,
  kTfLiteActSignBit,
  kTfLiteActSigmoid,
} TfLiteFusedActivation;

typedef struct {
  // Parameters for CONV_2D version 1.
  TfLitePadding padding;
  int stride_width;
  int stride_height;
  TfLiteFusedActivation activation;

  // Parameters for CONV_2D version 2.
  // Note: Version 2 supports dilation values not equal to 1.
  int dilation_width_factor;
  int dilation_height_factor;

  // Parameters for CONV_2D version 7 or above.
  // Used to determine the default value for the quantized bias.
  TfLiteType quantized_bias_type;
} TfLiteConvParams;

typedef struct {
  TfLitePadding padding;
  int stride_width;
  int stride_height;
  int stride_depth;
  int dilation_width_factor;
  int dilation_height_factor;
  int dilation_depth_factor;
  TfLiteFusedActivation activation;
} TfLiteConv3DParams;

typedef TfLiteConv3DParams TfLiteConv3DTransposeParams;

typedef struct {
  TfLitePadding padding;
  int stride_width;
  int stride_height;
  int filter_width;
  int filter_height;
  TfLiteFusedActivation activation;
  struct {
    TfLitePaddingValues padding;
  } computed;
} TfLitePoolParams;

typedef struct {
  // Parameters for DepthwiseConv version 1 or above.
  TfLitePadding padding;
  int stride_width;
  int stride_height;
  // `depth_multiplier` is redundant. It's used by CPU kernels in
  // TensorFlow 2.0 or below, but ignored in versions above.
  //
  // The information can be deduced from the shape of input and the shape of
  // weights. Since the TFLiteConverter toolchain doesn't support partially
  // specified shapes, relying on `depth_multiplier` stops us from supporting
  // graphs with dynamic shape tensors.
  //
  // Note: Some of the delegates (e.g. NNAPI, GPU) are still relying on this
  // field.
  int depth_multiplier;
  TfLiteFusedActivation activation;
  // Parameters for DepthwiseConv version 2 or above.
  int dilation_width_factor;
  int dilation_height_factor;
} TfLiteDepthwiseConvParams;

typedef struct {
  int rank;
  TfLiteFusedActivation activation;

  // Parameter for SVDF version 4.
  bool asymmetric_quantize_inputs;
} TfLiteSVDFParams;

typedef struct {
  TfLiteFusedActivation activation;

  // Parameter for RNN version 3.
  bool asymmetric_quantize_inputs;
} TfLiteRNNParams;

typedef struct {
  bool time_major;
  TfLiteFusedActivation activation;

  // Parameter for Sequence RNN version 3.
  bool asymmetric_quantize_inputs;
} TfLiteSequenceRNNParams;

typedef struct {
  bool time_major;
  TfLiteFusedActivation activation;
  bool merge_outputs;

  // Parameter for Bidirectional RNN version 3.
  bool asymmetric_quantize_inputs;
} TfLiteBidirectionalSequenceRNNParams;

typedef enum {
  kTfLiteFullyConnectedWeightsFormatDefault = 0,
  kTfLiteFullyConnectedWeightsFormatShuffled4x16Int8 = 1,
} TfLiteFullyConnectedWeightsFormat;

typedef struct {
  // Parameters for FullyConnected version 1 or above.
  TfLiteFusedActivation activation;

  // Parameters for FullyConnected version 2 or above.
  TfLiteFullyConnectedWeightsFormat weights_format;

  // Parameters for FullyConnected version 5 or above.
  // If set to true, then the number of dimensions in the input and the output
  // tensors are the same. Furthermore, all but the last dimension of the input
  // and output shapes will be equal.
  bool keep_num_dims;

  // Parameters for FullyConnected version 7 or above.
  // If set to true and the weights are quantized, then non constant inputs
  // are quantized at evaluation time with asymmetric quantization.
  bool asymmetric_quantize_inputs;

  // Parameters for FullyConnected version 10 or above.
  // Used to determine the default value for the quantized bias.
  TfLiteType quantized_bias_type;
} TfLiteFullyConnectedParams;

typedef enum {
  kTfLiteLshProjectionUnknown = 0,
  kTfLiteLshProjectionSparse = 1,
  kTfLiteLshProjectionDense = 2,
} TfLiteLSHProjectionType;

typedef struct {
  TfLiteLSHProjectionType type;
} TfLiteLSHProjectionParams;

typedef struct {
  float beta;
} TfLiteSoftmaxParams;

typedef struct {
  int axis;
  TfLiteFusedActivation activation;
} TfLiteConcatenationParams;

typedef struct {
  TfLiteFusedActivation activation;
  // Parameter added for the version 4.
  bool pot_scale_int16;
} TfLiteAddParams;

typedef struct {
  EmptyStructPlaceholder placeholder;
} TfLiteSpaceToBatchNDParams;

typedef struct {
  EmptyStructPlaceholder placeholder;
} TfLiteBatchToSpaceNDParams;

typedef struct {
  bool adj_x;
  bool adj_y;
  // Parameters for BatchMatMul version 4 or above.
  // If set to true and the weights are quantized, then non constant inputs
  // are quantized at evaluation time with asymmetric quantization.
  bool asymmetric_quantize_inputs;
} TfLiteBatchMatMulParams;

typedef struct {
  TfLiteFusedActivation activation;
} TfLiteMulParams;

typedef struct {
  TfLiteFusedActivation activation;
  // Parameter added for the version 5.
  bool pot_scale_int16;
} TfLiteSubParams;

typedef struct {
  TfLiteFusedActivation activation;
} TfLiteDivParams;

typedef struct {
  TfLiteFusedActivation activation;
} TfLiteL2NormParams;

typedef struct {
  int radius;
  float bias;
  float alpha;
  float beta;
} TfLiteLocalResponseNormParams;

typedef enum {
  kTfLiteLSTMFullKernel = 0,
  kTfLiteLSTMBasicKernel
} TfLiteLSTMKernelType;

typedef struct {
  // Parameters for LSTM version 1.
  TfLiteFusedActivation activation;
  float cell_clip;
  float proj_clip;

  // Parameters for LSTM version 2.
  // kTfLiteLSTMBasicKernel is only supported in version 2 or above.
  TfLiteLSTMKernelType kernel_type;

  // Parameters for LSTM version 4.
  bool asymmetric_quantize_inputs;
} TfLiteLSTMParams;

typedef struct {
  // Parameters needed for the underlying LSTM.
  TfLiteFusedActivation activation;
  float cell_clip;
  float proj_clip;

  // If set to true then the first dimension is time, otherwise batch.
  bool time_major;

  // Parameter for unidirectional sequence RNN version 3.
  bool asymmetric_quantize_inputs;

  // Parameter for unidirectional sequence RNN version 4.
  bool diagonal_recurrent_tensors;
} TfLiteUnidirectionalSequenceLSTMParams;

typedef struct {
  // Parameters supported by version 1:
  // Parameters inherited for the LSTM kernel.
  TfLiteFusedActivation activation;
  float cell_clip;
  float proj_clip;

  // If true, store the outputs of both directions in the first output.
  bool merge_outputs;

  // Parameters supported by version 2:
  // If set to true then the first dimension is time, otherwise batch.
  bool time_major;

  // Parameters supported by version 3:
  // If set to true, then hybrid ops use asymmetric quantization for inputs.
  bool asymmetric_quantize_inputs;
} TfLiteBidirectionalSequenceLSTMParams;

typedef struct {
  bool align_corners;
  // half_pixel_centers assumes pixels are of half the actual dimensions, and
  // yields more accurate resizes. Corresponds to the same argument for the
  // original TensorFlow op in TF2.0.
  bool half_pixel_centers;
} TfLiteResizeBilinearParams;

typedef struct {
  bool align_corners;
  bool half_pixel_centers;
} TfLiteResizeNearestNeighborParams;

typedef struct {
  EmptyStructPlaceholder placeholder;
} TfLitePadParams;

typedef struct {
  EmptyStructPlaceholder placeholder;
} TfLitePadV2Params;

typedef struct {
  // These fields are only used in old models for backward compatibility.
  // In the current implementation, we use the 2nd input of the op as the shape,
  // and these fields are unused.
  int32_t shape[TFLITE_RESHAPE_PARAMS_MAX_DIMENSION_COUNT];
  int num_dimensions;
} TfLiteReshapeParams;

typedef struct {
  int ngram_size;
  int max_skip_size;
  bool include_all_ngrams;
} TfLiteSkipGramParams;

typedef struct {
  int block_size;
} TfLiteSpaceToDepthParams;

typedef struct {
  int block_size;
} TfLiteDepthToSpaceParams;

typedef struct {
  TfLiteType in_data_type;
  TfLiteType out_data_type;
} TfLiteCastParams;

typedef enum {
  kTfLiteCombinerTypeSum = 0,
  kTfLiteCombinerTypeMean = 1,
  kTfLiteCombinerTypeSqrtn = 2,
} TfLiteCombinerType;

typedef struct {
  TfLiteCombinerType combiner;
} TfLiteEmbeddingLookupSparseParams;

typedef struct {
  int axis;
  int batch_dims;
} TfLiteGatherParams;

typedef struct {
  EmptyStructPlaceholder placeholder;
} TfLiteTransposeParams;

typedef struct {
  bool keep_dims;
} TfLiteReducerParams;

typedef struct {
  int num_splits;
} TfLiteSplitParams;

typedef struct {
  int num_splits;
} TfLiteSplitVParams;

typedef struct {
  // TODO(ahentz): We can't have dynamic data in this struct, at least not yet.
  // For now we will fix the maximum possible number of dimensions.
  int32_t squeeze_dims[8];
  int num_squeeze_dims;
} TfLiteSqueezeParams;

typedef struct {
  int begin_mask;
  int end_mask;
  int ellipsis_mask;
  int new_axis_mask;
  int shrink_axis_mask;

  // Parameters supported by version 8:
  // If true, then the end tensor is an offset of the begin tensor.
  bool offset;
} TfLiteStridedSliceParams;

typedef struct {
  TfLiteType output_type;
} TfLiteArgMaxParams;

typedef struct {
  TfLiteType output_type;
} TfLiteArgMinParams;

typedef struct {
  // Parameters supported by version 1:
  TfLitePadding padding;
  int stride_width;
  int stride_height;

  // Parameters supported by version 4:
  TfLiteFusedActivation activation;

  // Parameters for TransposeConv version 5 or above.
  // Used to determine the default value for the quantized bias.
  TfLiteType quantized_bias_type;
} TfLiteTransposeConvParams;

typedef struct {
  bool validate_indices;
} TfLiteSparseToDenseParams;

typedef struct {
  TfLiteType out_type;
} TfLiteShapeParams;

typedef struct {
  EmptyStructPlaceholder placeholder;
} TfLiteRankParams;

typedef struct {
  // Parameters supported by version 1:
  float min;
  float max;
  int num_bits;

  // Parameters supported by version 2:
  bool narrow_range;
} TfLiteFakeQuantParams;

typedef struct {
  int values_count;
  int axis;
} TfLitePackParams;

typedef struct {
  int axis;
} TfLiteOneHotParams;

typedef struct {
  int num;
  int axis;
} TfLiteUnpackParams;

typedef struct {
  float alpha;
} TfLiteLeakyReluParams;

typedef struct {
  TfLiteType index_out_type;
} TfLiteUniqueParams;

typedef struct {
  int seq_dim;
  int batch_dim;
} TfLiteReverseSequenceParams;

typedef struct {
  EmptyStructPlaceholder placeholder;
} TfLiteMatrixDiagParams;

typedef struct {
  EmptyStructPlaceholder placeholder;
} TfLiteMatrixSetDiagParams;

typedef struct {
  int then_subgraph_index;
  int else_subgraph_index;
} TfLiteIfParams;

typedef struct {
  int cond_subgraph_index;
  int body_subgraph_index;
} TfLiteWhileParams;

typedef struct {
  bool exclusive;
  bool reverse;
} TfLiteCumsumParams;

typedef struct {
  int init_subgraph_index;
} TfLiteCallOnceParams;

typedef struct {
  int table_id;
  TfLiteType key_dtype;
  TfLiteType value_dtype;
} TfLiteHashtableParams;

typedef struct {
  const char* container;
  const char* shared_name;
} TfLiteVarHandleParams;

typedef struct {
  int seed;
  int seed2;
} TfLiteRandomParams;

typedef struct {
  int num_boundaries;
  // This points to the memory stored in the model (flatbuffer),
  // and is not owned.
  const float* boundaries;
} TfLiteBucketizeParams;

typedef struct {
  bool approximate;
} TfLiteGeluParams;

typedef struct {
  int64_t dimension;
} TfLiteStablehloConcatenateParams;

typedef struct {
  // See the stablehlo spec for the explanation of the attributes:
  // https://github.com/openxla/stablehlo/blob/main/docs/spec.md#scatter
  bool indices_are_sorted;
  int64_t
      update_window_dims[TFLITE_STABLEHLO_SCATTER_PARAMS_MAX_DIMENSION_COUNT];
  int num_update_window_dims;
  int64_t
      inserted_window_dims[TFLITE_STABLEHLO_SCATTER_PARAMS_MAX_DIMENSION_COUNT];
  int num_inserted_window_dims;
  int64_t scatter_dims_to_operand_dims
      [TFLITE_STABLEHLO_SCATTER_PARAMS_MAX_DIMENSION_COUNT];
  int num_scatter_dims_to_operand_dims;
  int64_t index_vector_dim;
  bool unique_indices;
  int update_computation_subgraph_index;
} TfLiteStablehloScatterParams;

typedef enum {
  kTfLiteRngAlgorithmUnknown = 0,
  // An algorithm auto-selected by the system according to device type.
  kTfLiteRngAlgorithmDefault,
  // The Philox algorithm, as described in paper
  // ['Parallel Random Numbers: As Easy as 1, 2, 3']
  // (https://www.thesalmons.org/john/random123/papers/random123sc11.pdf)
  kTfLiteRngAlgorithmPhilox,
  // The ThreeFry algorithm, as described in paper
  // ['Parallel Random Numbers: As Easy as 1, 2, 3']
  // (https://www.thesalmons.org/john/random123/papers/random123sc11.pdf)
  kTfLiteRngAlgorithmThreefry,
} TfLiteRngAlgorithm;

typedef struct {
  TfLiteRngAlgorithm algorithm;
} TfLiteStablehloRngBitGeneratorParams;

typedef struct {
  // See the stablehlo spec for the explanation of the attributes:
  // https://github.com/openxla/stablehlo/blob/main/docs/spec.md#gather
  int64_t offset_dims[TFLITE_STABLEHLO_GATHER_PARAMS_MAX_DIMENSION_COUNT];
  int num_offset_dims;
  int64_t
      collapsed_slice_dims[TFLITE_STABLEHLO_GATHER_PARAMS_MAX_DIMENSION_COUNT];
  int num_collapsed_slice_dims;
  int64_t start_index_map[TFLITE_STABLEHLO_GATHER_PARAMS_MAX_DIMENSION_COUNT];
  int num_start_index_map;
  int64_t index_vector_dim;
  int64_t slice_sizes[TFLITE_STABLEHLO_GATHER_PARAMS_MAX_DIMENSION_COUNT];
  int num_slice_sizes;
  bool indices_are_sorted;
} TfLiteStablehloGatherParams;

typedef struct {
  // See the stablehlo spec for the explanation of the attributes:
  // https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reduce_window
  int64_t window_dimensions
      [TFLITE_STABLEHLO_REDUCE_WINDOW_PARAMS_MAX_DIMENSION_COUNT];
  int64_t
      window_strides[TFLITE_STABLEHLO_REDUCE_WINDOW_PARAMS_MAX_DIMENSION_COUNT];
  int64_t
      base_dilations[TFLITE_STABLEHLO_REDUCE_WINDOW_PARAMS_MAX_DIMENSION_COUNT];
  int64_t window_dilations
      [TFLITE_STABLEHLO_REDUCE_WINDOW_PARAMS_MAX_DIMENSION_COUNT];
  int64_t
      padding[2 * TFLITE_STABLEHLO_REDUCE_WINDOW_PARAMS_MAX_DIMENSION_COUNT];
  int body_subgraph_index;
} TfLiteStablehloReduceWindowParams;

enum TfLiteReduceWindowFunction {
  TfLiteReduceWindowFunctionUnsupported,
  TfLiteReduceWindowFunctionAdd,
  TfLiteReduceWindowFunctionMul,
  TfLiteReduceWindowFunctionMin,
  TfLiteReduceWindowFunctionMax,
  TfLiteReduceWindowFunctionAll,
  TfLiteReduceWindowFunctionAny
};

typedef struct {
  enum TfLiteReduceWindowFunction reduce_function;
} TfLiteReduceWindowParams;

typedef struct {
  // See the stablehlo spec for the explanation of the attributes:
  // https://github.com/openxla/stablehlo/blob/main/docs/spec.md#pad
  int64_t edge_padding_low[TFLITE_STABLEHLO_PAD_PARAMS_MAX_DIMENSION_COUNT];
  int64_t edge_padding_high[TFLITE_STABLEHLO_PAD_PARAMS_MAX_DIMENSION_COUNT];
  int64_t interior_padding[TFLITE_STABLEHLO_PAD_PARAMS_MAX_DIMENSION_COUNT];
} TfLiteStablehloPadParams;

typedef struct {
  const char* name;
  int32_t subgraph_index;
  int32_t version;
  const uint8_t* attributes;
  size_t attributes_size;
} TfLiteStablehloCompositeParams;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_CORE_C_BUILTIN_OP_DATA_H_
