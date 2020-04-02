/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_OPERATIONS_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_OPERATIONS_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/types/variant.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {

// Non exhaustive list of operations.
enum class OperationType {
  UNKNOWN = 0,
  ABS,
  ADD,
  BATCH_TO_SPACE,
  BATCH_NORMALIZATION,
  CONCAT,
  CONST,
  CONVOLUTION_2D,
  CONVOLUTION_TRANSPOSED,
  COS,
  DEPTHWISE_CONVOLUTION,
  DIV,
  EXP,
  FULLY_CONNECTED,
  HARD_SWISH,
  LOG,
  LSTM,
  MAXIMUM,
  MAX_UNPOOLING_2D,
  MEAN,
  MINIMUM,
  MUL,
  PAD,
  POOLING_2D,
  POW,
  PRELU,
  // Used to accurately run inference on quantized models.
  QUANTIZE_AND_DEQUANTIZE,
  RELU,
  RESHAPE,
  RESIZE,
  RSQRT,
  SIGMOID,
  SIN,
  SLICE,
  SOFTMAX,
  SPACE_TO_BATCH,
  SPACE_TO_DEPTH,
  SQRT,
  SQUARE,
  SQUARED_DIFF,
  SUB,
  TANH,
  TRANSPOSE,
};

std::string ToString(enum OperationType op);

OperationType OperationTypeFromString(const std::string& name);

typedef absl::variant<absl::monostate, Tensor<Linear, DataType::FLOAT32>, float>
    TensorOrScalar;

struct Padding2D {
  Padding2D() = default;
  Padding2D& operator=(const Padding2D& value);
  bool operator==(const Padding2D& value);
  bool operator!=(const Padding2D& value);
  Padding2D& operator-(const Padding2D& value);

  // Padding values for every axis (if needed), where 'prepended' defines
  // padding for the beginning of each axis and 'appended' represents end part
  // of the corresponding axis.
  HW prepended = HW(-1, -1);
  HW appended = HW(-1, -1);
};

struct Padding3D {
  Padding3D() = default;
  Padding3D& operator=(const Padding3D& value);
  bool operator==(const Padding3D& value);
  bool operator!=(const Padding3D& value);
  Padding3D& operator-(const Padding3D& value);

  // Padding values for every axis (if needed), where 'prepended' defines
  // padding for the beginning of each axis and 'appended' represents end part
  // of the corresponding axis.
  HWD prepended = HWD(0, 0, 0);
  HWD appended = HWD(0, 0, 0);
};

struct Crop2D : public Padding2D {};

struct SpaceToBatchAttributes {
  HW block;
  Padding2D padding;
};

struct BatchToSpaceAttributes {
  HW block;
  Crop2D crop;
};

enum class PoolingType {
  UNDEFINED = 0,

  // average pooling
  AVERAGE = 1,

  // max pooling
  MAX = 2,
};

struct Pooling2DAttributes {
  PoolingType type = PoolingType::UNDEFINED;
  // Strides for every axis.
  HW strides = HW(-1, -1);
  HW kernel = HW(-1, -1);
  Padding2D padding;
  // NOTE(akulik): technically the number of outputs from Pooling node indicates
  // whether indices are needed or not, but I decided to keep it inside
  // attributes to simplify processing.
  bool output_indices = false;
};

struct Pooling3DAttributes {
  PoolingType type = PoolingType::UNDEFINED;
  // Strides for every axis.
  HWD strides = HWD(0, 0, 0);
  HWD kernel = HWD(0, 0, 0);
  Padding3D padding;
  // NOTE(akulik): technically the number of outputs from Pooling node indicates
  // whether indices are needed or not, but I decided to keep it inside
  // attributes to simplify processing.
  bool output_indices = false;
};

struct MaxUnpooling2DAttributes {
  // Strides for every axis.
  HW strides = HW(-1, -1);
  HW kernel = HW(-1, -1);
  Padding2D padding;
};

struct MaxUnpooling3DAttributes {
  // Strides for every axis.
  HWD strides = HWD(0, 0, 0);
  HWD kernel = HWD(0, 0, 0);
  Padding3D padding;
};

struct MeanAttributes {
  // The vector of dimensions to calculate mean along.
  std::set<Axis> dims;
};

struct ConcatAttributes {
  // Defines axis by which to concat on.
  Axis axis = Axis::UNKNOWN;
};

// @return shape of a tensor after MaxUnpooling2D operation is applied to
//         the given input.
BHWC CalculateOutputShape(const BHWC& input,
                          const MaxUnpooling2DAttributes& attr);

// @return shape of a tensor after MaxUnpooling3D operation is applied to
//         the given input.
BHWDC CalculateOutputShape(const BHWDC& input,
                           const MaxUnpooling3DAttributes& attr);

// @return shape of a tensor after Pooling2D operation is applied to the given
//         input.
BHWC CalculateOutputShape(const BHWC& input, const Pooling2DAttributes& attr);

// @return shape of a tensor after Pooling3D operation is applied to the given
//         input.
BHWDC CalculateOutputShape(const BHWDC& input, const Pooling3DAttributes& attr);

// @return shape of a tensor after Concat operation is applied to the given
//         input.
absl::Status CalculateOutputShape(const std::vector<BHWC>& input,
                                  const ConcatAttributes& attr,
                                  BHWC* output_shape);

// @return padding for pooling operation to make sure output keep the same shape
// as the given input.
Padding2D CalculateSamePadding(const BHWC& input,
                               const Pooling2DAttributes& attr);

// @return padding for pooling operation to make sure output keep the same shape
// as the given input.
Padding3D CalculateSamePadding(const BHWDC& input,
                               const Pooling3DAttributes& attr);

// @return padding for max unpooling operation to make sure output keep the same
// shape as the given input.
Padding2D CalculateSamePadding(const BHWC& input,
                               const MaxUnpooling2DAttributes& attr);

// @return padding for max unpooling operation to make sure output keep the same
// shape as the given input.
Padding3D CalculateSamePadding(const BHWDC& input,
                               const MaxUnpooling3DAttributes& attr);

struct Convolution2DAttributes {
  HW strides = HW(1, 1);    // Along each axis.
  HW dilations = HW(1, 1);  // Along each axis.
  Padding2D padding;

  Tensor<OHWI, DataType::FLOAT32> weights;
  Tensor<Linear, DataType::FLOAT32> bias;  // optional
};

struct Convolution3DAttributes {
  HWD strides = HWD(0, 0, 0);    // Along each axis.
  HWD dilations = HWD(0, 0, 0);  // Along each axis.
  Padding3D padding;

  Tensor<OHWDI, DataType::FLOAT32> weights;
  Tensor<Linear, DataType::FLOAT32> bias;  // optional
};

// @return shape of a tensor after Convolution2D operation is applied to
//         the given input.
BHWC CalculateOutputShape(const BHWC& input,
                          const Convolution2DAttributes& attr);

// @return shape of a tensor after Convolution3D operation is applied to
//         the given input.
BHWDC CalculateOutputShape(const BHWDC& input,
                           const Convolution3DAttributes& attr);

// @return padding for convolution operation to make sure output keep the same
// shape as the given input.
Padding2D CalculateSamePadding(const BHWC& input,
                               const Convolution2DAttributes& attr);

// @return padding for convolution operation to make sure output keep the same
// shape as the given input.
Padding3D CalculateSamePadding(const BHWDC& input,
                               const Convolution3DAttributes& attr);

struct ConvolutionTransposedAttributes {
  HW stride = HW(1, 1);  // Along each axis.
  HW adjacent;           // TODO(sorokin): No op on Flow.
  Padding2D padding;

  Tensor<OHWI, DataType::FLOAT32> weights;
  Tensor<Linear, DataType::FLOAT32> bias;  // optional
};

struct ConvolutionTransposed3DAttributes {
  HWD stride = HWD(0, 0, 0);  // Along each axis.
  Padding3D padding;

  Tensor<OHWDI, DataType::FLOAT32> weights;
  Tensor<Linear, DataType::FLOAT32> bias;  // optional
};

Padding2D CalculateSamePadding(const BHWC& input,
                               const ConvolutionTransposedAttributes& attr);

Padding3D CalculateSamePadding(const BHWDC& input,
                               const ConvolutionTransposed3DAttributes& attr);

// @return shape of a tensor after ConvolutionTransposed operation is applied to
//         the given input.
BHWC CalculateOutputShape(const BHWC& input,
                          const ConvolutionTransposedAttributes& attr);

// @return shape of a tensor after ConvolutionTransposed3D operation is applied
// to
//         the given input.
BHWDC CalculateOutputShape(const BHWDC& input,
                           const ConvolutionTransposed3DAttributes& attr);

struct DepthwiseConvolution2DAttributes : public Convolution2DAttributes {};
struct DepthwiseConvolution3DAttributes : public Convolution3DAttributes {};

// @return shape of a tensor after DepthwiseConvolution2D operation is applied
//         to the given input.
BHWC CalculateOutputShape(const BHWC& input,
                          const DepthwiseConvolution2DAttributes& attr);

// @return shape of a tensor after DepthwiseConvolution3D operation is applied
//         to the given input.
BHWDC CalculateOutputShape(const BHWDC& input,
                           const DepthwiseConvolution3DAttributes& attr);

// @return padding for depthwise convolution operation to make sure output keep
// the same shape as the given input.
Padding2D CalculateSamePadding(const BHWC& input,
                               const DepthwiseConvolution2DAttributes& attr);

// @return padding for depthwise convolution operation to make sure output keep
// the same shape as the given input.
Padding3D CalculateSamePadding(const BHWDC& input,
                               const DepthwiseConvolution3DAttributes& attr);

// f(x):= {
//   if x < 0  : x -> alpha * x
//   if x >= 0 : x -> min(clip, x)
// }
//
// Examples:
//   - ReLU: clip = 0, alpha = 0
//   - ReLU6: clip = 6, alpha = 0
//   - Leaky ReLU: clip = 0, alpha = a
struct ReLUAttributes {
  // clip <= 0 mean it is not set.
  float clip = 0;

  float alpha = 0;
};

struct PReLUAttributes {
  // clip <= 0 mean it is not set.
  float clip = 0;

  // If alpha is linear, then it is sharded across CHANNELS axis, otherwise
  // full shape alpha is required.
  absl::variant<Tensor<Linear, DataType::FLOAT32>,
                Tensor<HWC, DataType::FLOAT32>>
      alpha;
};

struct SoftmaxAttributes {
  Axis axis = Axis::UNKNOWN;
};

enum LstmKernelType {
  FULL = 0,
  BASIC = 1,  // Currently, only basic is supported.
};

struct LstmAttributes {
  LstmKernelType kernel_type = LstmKernelType::BASIC;
};

struct MultiplyAttributes {
  TensorOrScalar param;
};

enum class SamplingType {
  UNKNOWN = 0,
  NEAREST = 1,
  BILINEAR = 2,
};

struct Resize2DAttributes {
  HW new_shape;

  SamplingType type = SamplingType::UNKNOWN;

  // If true, the centers of the 4 corner pixels of the input and output tensors
  // are aligned, preserving the values at the corner pixels. Defaults to false.
  bool align_corners = false;
  // half_pixel_centers assumes pixels are of half the actual dimensions, and
  // yields more accurate resizes. Only applicable to BILINEAR sampling.
  bool half_pixel_centers = false;
};

// TODO(b/147771327): rename to Resize3D
struct Resize3DAttributes {
  HWD new_shape;

  SamplingType type = SamplingType::NEAREST;

  // If true, the centers of the 8 corner pixels of the input and output tensors
  // are aligned, preserving the values at the corner pixels. Defaults to false.
  bool align_corners = false;
};

float CalculateResizeScale(int32_t input_size, int32_t output_size,
                           const Resize2DAttributes& attr);

float CalculateResizeScale(int32_t input_size, int32_t output_size,
                           const Resize3DAttributes& attr);

// @return shape of a tensor after scale operation is applied to the given
// input.
BHWC CalculateOutputShape(const BHWC& input, const Resize2DAttributes& attr);

// @return shape of a tensor after scale operation is applied to the given
// input.
BHWDC CalculateOutputShape(const BHWDC& input, const Resize3DAttributes& attr);

enum class PaddingContentType {
  ZEROS = 0,
  REFLECT = 1,
  EDGE = 2,
};

struct PadAttributes {
  PaddingContentType type = PaddingContentType::ZEROS;

  BHWC prepended;
  BHWC appended;
};

// @return shape of a tensor after Pad operation is applied to the given input.
BHWC CalculateOutputShape(const BHWC& input, const PadAttributes& attr);

struct ConstTensorAttributes {
  Tensor<BHWC, DataType::FLOAT32> tensor;
};

// Simple slicing without advanced support for shrinking, reverse slicing etc.
struct SliceAttributes {
  // Specifies start and end dimensions for slicing.
  BHWC starts;
  BHWC ends;

  // Stride should be >= 1.
  BHWC strides;
};

// @return shape of a tensor after Slice2D operation is applied to the given
//         input.
BHWC CalculateOutputShape(const BHWC& input, const SliceAttributes& attr);

struct AddAttributes {
  TensorOrScalar param;
};

struct FullyConnectedAttributes {
  Tensor<OHWI, DataType::FLOAT32> weights;
  Tensor<Linear, DataType::FLOAT32> bias;
};

// @return shape of a tensor after FullyConnected operation is applied to
// the given input.
BHWC CalculateOutputShape(const BHWC& input,
                          const FullyConnectedAttributes& attr);

// @return shape of a tensor after Mean operation is applied to the given input.
BHWC CalculateOutputShape(const BHWC& input, const MeanAttributes& attr);

struct ElementwiseAttributes {
  TensorOrScalar param;
};

struct ReshapeAttributes {
  BHWC new_shape;
};

struct TransposeAttributes {
  // A permutation of the dimensions of input tensor
  BHWC perm;
};

// @return shape of a tensor after Transpose operation is applied to
// the given input.
BHWC CalculateOutputShape(const BHWC& input, const TransposeAttributes& attr);

struct SpaceToDepthAttributes {
  int block_size;
};

// These help perform a combination of Quantize & Dequantize to adjust float
// values like quantized inference would.
struct QuantizeAndDequantizeAttributes {
  float min = 0;
  float max = 0;
  float scale = 0;
};

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_OPERATIONS_H_
