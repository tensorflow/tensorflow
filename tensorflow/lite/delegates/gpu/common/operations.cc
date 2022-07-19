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

#include "tensorflow/lite/delegates/gpu/common/operations.h"

#include <algorithm>
#include <cstdint>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"

namespace tflite {
namespace gpu {

Padding2D& Padding2D::operator=(const Padding2D& value) {
  prepended = value.prepended;
  appended = value.appended;
  return *this;
}

bool Padding2D::operator==(const Padding2D& value) {
  return this->prepended == value.prepended && this->appended == value.appended;
}

bool Padding2D::operator!=(const Padding2D& value) { return !(*this == value); }

Padding2D& Padding2D::operator-(const Padding2D& value) {
  prepended.h -= value.prepended.h;
  prepended.w -= value.prepended.w;
  appended.h -= value.appended.h;
  appended.w -= value.appended.w;
  return *this;
}

Padding3D& Padding3D::operator=(const Padding3D& value) {
  prepended = value.prepended;
  appended = value.appended;
  return *this;
}

bool Padding3D::operator==(const Padding3D& value) {
  return this->prepended == value.prepended && this->appended == value.appended;
}

bool Padding3D::operator!=(const Padding3D& value) { return !(*this == value); }

Padding3D& Padding3D::operator-(const Padding3D& value) {
  prepended.h -= value.prepended.h;
  prepended.w -= value.prepended.w;
  prepended.d -= value.prepended.d;
  appended.h -= value.appended.h;
  appended.w -= value.appended.w;
  appended.d -= value.appended.d;
  return *this;
}

std::string ToString(enum OperationType op) {
  switch (op) {
    case OperationType::ABS:
      return "abs";
    case OperationType::ADD:
      return "add";
    case OperationType::BATCH_NORMALIZATION:
      return "batch_normalization";
    case OperationType::BATCH_TO_SPACE:
      return "batch_to_space";
    case OperationType::BATCHED_MATMUL:
      return "batched_matmul";
    case OperationType::CAST:
      return "cast";
    case OperationType::CONCAT:
      return "concat";
    case OperationType::CONSTANT:
      return "const";
    case OperationType::CONVOLUTION_2D:
      return "convolution_2d";
    case OperationType::CONVOLUTION_TRANSPOSED:
      return "convolution_transposed";
    case OperationType::COPY:
      return "copy";
    case OperationType::COS:
      return "cos";
    case OperationType::CUMSUM:
      return "cumsum";
    case OperationType::DENSIFY:
      return "densify";
    case OperationType::DEPTHWISE_CONVOLUTION:
      return "depthwise_convolution";
    case OperationType::DEPTH_TO_SPACE:
      return "depth_to_space";
    case OperationType::DIV:
      return "div";
    case OperationType::ELU:
      return "elu";
    case OperationType::EQUAL:
      return "equal";
    case OperationType::EXP:
      return "exp";
    case OperationType::FLOOR:
      return "floor";
    case OperationType::FLOOR_DIV:
      return "floor_div";
    case OperationType::FLOOR_MOD:
      return "floor_mod";
    case OperationType::FULLY_CONNECTED:
      return "fully_connected";
    case OperationType::FULLY_CONNECTED_INT8:
      return "fully_connected_int8";
    case OperationType::GATHER:
      return "gather";
    case OperationType::GREATER:
      return "greater";
    case OperationType::GREATER_EQUAL:
      return "greater_equal";
    case OperationType::HARD_SWISH:
      return "hard_swish";
    case OperationType::LESS:
      return "less";
    case OperationType::LESS_EQUAL:
      return "less_equal";
    case OperationType::LOG:
      return "log";
    case OperationType::LSTM:
      return "lstm";
    case OperationType::MAXIMUM:
      return "maximum";
    case OperationType::MAX_UNPOOLING_2D:
      return "max_unpooling";
    case OperationType::MEAN:
      return "mean";
    case OperationType::MEAN_STDDEV_NORMALIZATION:
      return "mean_stddev_normalization";
    case OperationType::MINIMUM:
      return "minimum";
    case OperationType::MUL:
      return "mul";
    case OperationType::NEG:
      return "neg";
    case OperationType::NOT_EQUAL:
      return "not_equal";
    case OperationType::ONE_HOT:
      return "one_hot";
    case OperationType::PAD:
      return "pad";
    case OperationType::POOLING_2D:
      return "pooling_2d";
    case OperationType::POW:
      return "pow";
    case OperationType::PRELU:
      return "prelu";
    case OperationType::QUANTIZE_AND_DEQUANTIZE:
      return "quantize_and_dequantize";
    case OperationType::REDUCE_MAXIMUM:
      return "reduce_maximum";
    case OperationType::REDUCE_MINIMUM:
      return "reduce_minimum";
    case OperationType::REDUCE_PRODUCT:
      return "reduce_product";
    case OperationType::REDUCE_SUM:
      return "reduce_sum";
    case OperationType::RELU:
      return "relu";
    case OperationType::RESAMPLER:
      return "resampler";
    case OperationType::RESHAPE:
      return "reshape";
    case OperationType::RESIZE:
      return "resize";
    case OperationType::RSQRT:
      return "rsqrt";
    case OperationType::SELECT_V2:
      return "select_v2";
    case OperationType::SIGMOID:
      return "sigmoid";
    case OperationType::SIN:
      return "sin";
    case OperationType::SLICE:
      return "slice";
    case OperationType::SOFTMAX:
      return "softmax";
    case OperationType::SPACE_TO_BATCH:
      return "space_to_batch";
    case OperationType::SPACE_TO_DEPTH:
      return "space_to_depth";
    case OperationType::SPLIT:
      return "split";
    case OperationType::SQRT:
      return "sqrt";
    case OperationType::SQUARE:
      return "square";
    case OperationType::SQUARED_DIFF:
      return "squared_diff";
    case OperationType::SUB:
      return "subtract";
    case OperationType::TANH:
      return "tanh";
    case OperationType::TILE:
      return "tile";
    case OperationType::TRANSPOSE:
      return "transpose";
    case OperationType::UNKNOWN:
      return "unknown_operation";
  }
}

OperationType OperationTypeFromString(const std::string& name) {
  static const auto* operations =
      new std::unordered_map<std::string, OperationType>({
          {"abs", OperationType::ABS},
          {"add", OperationType::ADD},
          {"batch_normalization", OperationType::BATCH_NORMALIZATION},
          {"batched_matmul", OperationType::BATCHED_MATMUL},
          {"cast", OperationType::CAST},
          {"concat", OperationType::CONCAT},
          {"const", OperationType::CONSTANT},
          {"convolution_2d", OperationType::CONVOLUTION_2D},
          {"convolution_transposed", OperationType::CONVOLUTION_TRANSPOSED},
          {"copy", OperationType::COPY},
          {"cos", OperationType::COS},
          {"cumsum", OperationType::CUMSUM},
          {"densify", OperationType::DENSIFY},
          {"depthwise_convolution", OperationType::DEPTHWISE_CONVOLUTION},
          {"depth_to_space", OperationType::DEPTH_TO_SPACE},
          {"div", OperationType::DIV},
          {"elu", OperationType::ELU},
          {"equal", OperationType::EQUAL},
          {"exp", OperationType::EXP},
          {"floor", OperationType::FLOOR},
          {"floor_div", OperationType::FLOOR_DIV},
          {"floor_mod", OperationType::FLOOR_MOD},
          {"fully_connected", OperationType::FULLY_CONNECTED},
          {"fully_connected_int8", OperationType::FULLY_CONNECTED_INT8},
          {"gather", OperationType::GATHER},
          {"greater", OperationType::GREATER},
          {"greater_equal", OperationType::GREATER_EQUAL},
          {"hard_swish", OperationType::HARD_SWISH},
          {"less", OperationType::LESS},
          {"less_equal", OperationType::LESS_EQUAL},
          {"log", OperationType::LOG},
          {"lstm", OperationType::LSTM},
          {"maximum", OperationType::MAXIMUM},
          {"max_unpooling", OperationType::MAX_UNPOOLING_2D},
          {"mean", OperationType::MEAN},
          {"mean_stddev_normalization",
           OperationType::MEAN_STDDEV_NORMALIZATION},
          {"minimum", OperationType::MINIMUM},
          {"mul", OperationType::MUL},
          {"neg", OperationType::NEG},
          {"not_equal", OperationType::NOT_EQUAL},
          {"one_hot", OperationType::ONE_HOT},
          {"pad", OperationType::PAD},
          {"pooling_2d", OperationType::POOLING_2D},
          {"pow", OperationType::POW},
          {"prelu", OperationType::PRELU},
          {"quantize_and_dequantize", OperationType::QUANTIZE_AND_DEQUANTIZE},
          {"reduce_maximum", OperationType::REDUCE_MAXIMUM},
          {"reduce_minimum", OperationType::REDUCE_MINIMUM},
          {"reduce_product", OperationType::REDUCE_PRODUCT},
          {"reduce_sum", OperationType::REDUCE_SUM},
          {"relu", OperationType::RELU},
          {"resampler", OperationType::RESAMPLER},
          {"resize", OperationType::RESIZE},
          {"reshape", OperationType::RESHAPE},
          {"rsqrt", OperationType::RSQRT},
          {"select_v2", OperationType::SELECT_V2},
          {"sigmoid", OperationType::SIGMOID},
          {"sin", OperationType::SIN},
          {"slice", OperationType::SLICE},
          {"softmax", OperationType::SOFTMAX},
          {"space_to_depth", OperationType::SPACE_TO_DEPTH},
          {"split", OperationType::SPLIT},
          {"sqrt", OperationType::SQRT},
          {"square", OperationType::SQUARE},
          {"squared_diff", OperationType::SQUARED_DIFF},
          {"subtract", OperationType::SUB},
          {"tanh", OperationType::TANH},
          {"tile", OperationType::TILE},
          {"transpose", OperationType::TRANSPOSE},
      });
  auto op = operations->find(name);
  return op == operations->end() ? OperationType::UNKNOWN : op->second;
}

namespace {

template <typename T>
T DivideRoundUp(T n, T divisor) {
  return (n - 1) / divisor + 1;
}

int32_t CalculateOutputSizeBeforeStrides(int32_t input, int32_t kernel,
                                         int32_t padding, int32_t dilation) {
  const int32_t dilated_kernel = (kernel - 1) * dilation + 1;
  return input + padding - dilated_kernel + 1;
}

template <Axis T>
int32_t CalculateOutputWithoutStrides(const BHWC& input,
                                      const Convolution2DAttributes& attr) {
  return CalculateOutputSizeBeforeStrides(
      input.get<T>(), attr.weights.shape.get<T>(),
      attr.padding.prepended.get<T>() + attr.padding.appended.get<T>(),
      attr.dilations.get<T>());
}

template <Axis T>
int32_t CalculateOutputWithoutStrides(const BHWDC& input,
                                      const Convolution3DAttributes& attr) {
  return CalculateOutputSizeBeforeStrides(
      input.get<T>(), attr.weights.shape.get<T>(),
      attr.padding.prepended.get<T>() + attr.padding.appended.get<T>(),
      attr.dilations.get<T>());
}

template <Axis T>
int32_t CalculateOutputWithoutStrides(const BHWC& input,
                                      const Pooling2DAttributes& attr) {
  return CalculateOutputSizeBeforeStrides(
      input.get<T>(), attr.kernel.get<T>(),
      attr.padding.prepended.get<T>() + attr.padding.appended.get<T>(),
      /*dilation=*/1);
}

template <Axis T>
int32_t CalculateOutputWithoutStrides(const BHWDC& input,
                                      const Pooling3DAttributes& attr) {
  return CalculateOutputSizeBeforeStrides(
      input.get<T>(), attr.kernel.get<T>(),
      attr.padding.prepended.get<T>() + attr.padding.appended.get<T>(),
      /*dilation=*/1);
}

template <Axis T>
int32_t CalculateOutput(const BHWC& input,
                        const ConvolutionTransposedAttributes& attr) {
  return (input.get<T>() - 1) * attr.stride.get<T>() -
         (attr.padding.prepended.get<T>() + attr.padding.appended.get<T>()) +
         attr.weights.shape.get<T>() + attr.adjacent.get<T>();
}

template <Axis T>
int32_t CalculateOutput(const BHWDC& input,
                        const ConvolutionTransposed3DAttributes& attr) {
  return (input.get<T>() - 1) * attr.stride.get<T>() -
         (attr.padding.prepended.get<T>() + attr.padding.appended.get<T>()) +
         attr.weights.shape.get<T>();
}

inline int32_t StridedSize(int32_t size, int32_t stride) {
  return stride == 0 ? -1 : DivideRoundUp(size, stride);
}

template <Axis AxisT, typename AttrT>
int32_t CalculateOutput(const BHWC& input, const AttrT& attr) {
  return StridedSize(CalculateOutputWithoutStrides<AxisT>(input, attr),
                     attr.strides.template get<AxisT>());
}

template <Axis AxisT, typename AttrT>
int32_t CalculateOutput(const BHWDC& input, const AttrT& attr) {
  return StridedSize(CalculateOutputWithoutStrides<AxisT>(input, attr),
                     attr.strides.template get<AxisT>());
}

int32_t CalculateSamePadding(int32_t input, int32_t kernel, int32_t dilation,
                             int32_t stride) {
  const int32_t dilated_kernel = (kernel - 1) * dilation + 1;
  return std::max(0, dilated_kernel - (input - 1) % stride - 1);
}

// Returns a padding that should be present to make sure image size stays
// the same.
template <Axis AxisT>
int32_t CalculateSamePadding(const BHWC& input,
                             const Convolution2DAttributes& attr) {
  return CalculateSamePadding(
      input.get<AxisT>(), attr.weights.shape.get<AxisT>(),
      attr.dilations.get<AxisT>(), attr.strides.get<AxisT>());
}

// Returns a padding that should be present to make sure image size stays
// the same.
template <Axis AxisT>
int32_t CalculateSamePadding(const BHWDC& input,
                             const Convolution3DAttributes& attr) {
  return CalculateSamePadding(
      input.get<AxisT>(), attr.weights.shape.get<AxisT>(),
      attr.dilations.get<AxisT>(), attr.strides.get<AxisT>());
}

template <Axis AxisT>
int32_t CalculateSamePadding(const BHWC& input,
                             const ConvolutionTransposedAttributes& attr) {
  return CalculateSamePadding(input.get<AxisT>(),
                              attr.weights.shape.get<AxisT>(),
                              /*dilation=*/1, attr.stride.get<AxisT>());
}

template <Axis AxisT>
int32_t CalculateSamePadding(const BHWDC& input,
                             const ConvolutionTransposed3DAttributes& attr) {
  return CalculateSamePadding(input.get<AxisT>(),
                              attr.weights.shape.get<AxisT>(),
                              /*dilation=*/1, attr.stride.get<AxisT>());
}

template <Axis AxisT>
int32_t CalculateSamePadding(const BHWC& input,
                             const Pooling2DAttributes& attr) {
  return CalculateSamePadding(input.get<AxisT>(), attr.kernel.get<AxisT>(),
                              /*dilation=*/1, attr.strides.get<AxisT>());
}

template <Axis AxisT>
int32_t CalculateSamePadding(const BHWDC& input,
                             const Pooling3DAttributes& attr) {
  return CalculateSamePadding(input.get<AxisT>(), attr.kernel.get<AxisT>(),
                              /*dilation=*/1, attr.strides.get<AxisT>());
}

template <Axis AxisT>
int32_t CalculateSamePadding(const BHWC& input,
                             const MaxUnpooling2DAttributes& attr) {
  return CalculateSamePadding(input.get<AxisT>(), attr.kernel.get<AxisT>(),
                              /*dilation=*/1, attr.strides.get<AxisT>());
}

template <Axis AxisT>
int32_t CalculateSamePadding(const BHWDC& input,
                             const MaxUnpooling3DAttributes& attr) {
  return CalculateSamePadding(input.get<AxisT>(), attr.kernel.get<AxisT>(),
                              /*dilation=*/1, attr.strides.get<AxisT>());
}

Padding2D MakeSamePadding(const BHWC& input,
                          const ConvolutionTransposedAttributes& attr) {
  int32_t padding_height = CalculateSamePadding<Axis::HEIGHT>(input, attr);
  int32_t padding_width = CalculateSamePadding<Axis::WIDTH>(input, attr);
  Padding2D padding;
  padding.prepended = HW(padding_height / 2, padding_width / 2);
  padding.appended = HW(padding_height - padding_height / 2,
                        padding_width - padding_width / 2);
  return padding;
}

Padding3D MakeSamePadding(const BHWDC& input,
                          const ConvolutionTransposed3DAttributes& attr) {
  int32_t padding_height = CalculateSamePadding<Axis::HEIGHT>(input, attr);
  int32_t padding_width = CalculateSamePadding<Axis::WIDTH>(input, attr);
  int32_t padding_depth = CalculateSamePadding<Axis::DEPTH>(input, attr);
  Padding3D padding;
  padding.prepended =
      HWD(padding_height / 2, padding_width / 2, padding_depth / 2);
  padding.appended =
      HWD(padding_height - padding_height / 2,
          padding_width - padding_width / 2, padding_depth - padding_depth / 2);
  return padding;
}

// If padding depends on input, convert it into fixed padding.
template <class AttrT>
Padding2D MakeSamePadding(const BHWC& input, const AttrT& attr) {
  int32_t padding_height = CalculateSamePadding<Axis::HEIGHT>(input, attr);
  int32_t padding_width = CalculateSamePadding<Axis::WIDTH>(input, attr);
  Padding2D padding;
  padding.prepended = HW(padding_height / 2, padding_width / 2);
  padding.appended = HW(padding_height - padding_height / 2,
                        padding_width - padding_width / 2);
  return padding;
}

// If padding depends on input, convert it into fixed padding.
template <class AttrT>
Padding3D MakeSamePadding(const BHWDC& input, const AttrT& attr) {
  int32_t padding_height = CalculateSamePadding<Axis::HEIGHT>(input, attr);
  int32_t padding_width = CalculateSamePadding<Axis::WIDTH>(input, attr);
  int32_t padding_depth = CalculateSamePadding<Axis::DEPTH>(input, attr);
  Padding3D padding;
  padding.prepended =
      HWD(padding_height / 2, padding_width / 2, padding_depth / 2);
  padding.appended =
      HWD(padding_height - padding_height / 2,
          padding_width - padding_width / 2, padding_depth - padding_depth / 2);
  return padding;
}

}  // namespace

BHWC CalculateOutputShape(const BHWC& input,
                          const MaxUnpooling2DAttributes& attr) {
  return BHWC(input.b,
              input.h * attr.strides.h - attr.padding.prepended.h -
                  attr.padding.appended.h,
              input.w * attr.strides.w - attr.padding.prepended.w -
                  attr.padding.appended.w,
              input.c);
}

BHWDC CalculateOutputShape(const BHWDC& input,
                           const MaxUnpooling3DAttributes& attr) {
  return BHWDC(input.b,
               input.h * attr.strides.h - attr.padding.prepended.h -
                   attr.padding.appended.h,
               input.w * attr.strides.w - attr.padding.prepended.w -
                   attr.padding.appended.w,
               input.d * attr.strides.d - attr.padding.prepended.d -
                   attr.padding.appended.d,
               input.c);
}

BHWC CalculateOutputShape(const BHWC& input, const Pooling2DAttributes& attr) {
  return BHWC(input.b, CalculateOutput<Axis::HEIGHT>(input, attr),
              CalculateOutput<Axis::WIDTH>(input, attr), input.c);
}

BHWDC CalculateOutputShape(const BHWDC& input,
                           const Pooling3DAttributes& attr) {
  return BHWDC(input.b, CalculateOutput<Axis::HEIGHT>(input, attr),
               CalculateOutput<Axis::WIDTH>(input, attr),
               CalculateOutput<Axis::DEPTH>(input, attr), input.c);
}

BHWC CalculateOutputShape(const BHWC& input,
                          const Convolution2DAttributes& attr) {
  return BHWC(input.b, CalculateOutput<Axis::HEIGHT>(input, attr),
              CalculateOutput<Axis::WIDTH>(input, attr),
              attr.weights.shape.get<Axis::OUTPUT_CHANNELS>());
}

BHWDC CalculateOutputShape(const BHWDC& input,
                           const Convolution3DAttributes& attr) {
  return BHWDC(input.b, CalculateOutput<Axis::HEIGHT>(input, attr),
               CalculateOutput<Axis::WIDTH>(input, attr),
               CalculateOutput<Axis::DEPTH>(input, attr),
               attr.weights.shape.get<Axis::OUTPUT_CHANNELS>());
}

BHWC CalculateOutputShape(const BHWC& input,
                          const ConvolutionTransposedAttributes& attr) {
  return BHWC(input.b, CalculateOutput<Axis::HEIGHT>(input, attr),
              CalculateOutput<Axis::WIDTH>(input, attr),
              attr.weights.shape.get<Axis::OUTPUT_CHANNELS>());
}

BHWDC CalculateOutputShape(const BHWDC& input,
                           const ConvolutionTransposed3DAttributes& attr) {
  return BHWDC(input.b, CalculateOutput<Axis::HEIGHT>(input, attr),
               CalculateOutput<Axis::WIDTH>(input, attr),
               CalculateOutput<Axis::DEPTH>(input, attr),
               attr.weights.shape.get<Axis::OUTPUT_CHANNELS>());
}

BHWC CalculateOutputShape(const BHWC& input,
                          const DepthwiseConvolution2DAttributes& attr) {
  return BHWC(input.b, CalculateOutput<Axis::HEIGHT>(input, attr),
              CalculateOutput<Axis::WIDTH>(input, attr),
              attr.weights.shape.get<Axis::OUTPUT_CHANNELS>() *
                  attr.weights.shape.get<Axis::INPUT_CHANNELS>());
}

BHWDC CalculateOutputShape(const BHWDC& input,
                           const DepthwiseConvolution3DAttributes& attr) {
  return BHWDC(input.b, CalculateOutput<Axis::HEIGHT>(input, attr),
               CalculateOutput<Axis::WIDTH>(input, attr),
               CalculateOutput<Axis::DEPTH>(input, attr),
               attr.weights.shape.get<Axis::OUTPUT_CHANNELS>() *
                   attr.weights.shape.get<Axis::INPUT_CHANNELS>());
}

BHWC CalculateOutputShape(const BHWC& input, const SliceAttributes& attr) {
  return BHWC(StridedSize(attr.ends.b - attr.starts.b, attr.strides.b),
              StridedSize(attr.ends.h - attr.starts.h, attr.strides.h),
              StridedSize(attr.ends.w - attr.starts.w, attr.strides.w),
              StridedSize(attr.ends.c - attr.starts.c, attr.strides.c));
}

BHWDC CalculateOutputShape(const BHWDC& input, const Slice3DAttributes& attr) {
  return BHWDC(StridedSize(attr.ends.b - attr.starts.b, attr.strides.b),
               StridedSize(attr.ends.h - attr.starts.h, attr.strides.h),
               StridedSize(attr.ends.w - attr.starts.w, attr.strides.w),
               StridedSize(attr.ends.d - attr.starts.d, attr.strides.d),
               StridedSize(attr.ends.c - attr.starts.c, attr.strides.c));
}

BHWC CalculateOutputShape(const BHWC& input, const PadAttributes& attr) {
  return BHWC(attr.appended.b + attr.prepended.b + input.b,
              attr.appended.h + attr.prepended.h + input.h,
              attr.appended.w + attr.prepended.w + input.w,
              attr.appended.c + attr.prepended.c + input.c);
}

BHWDC CalculateOutputShape(const BHWDC& input, const Pad3DAttributes& attr) {
  return BHWDC(attr.appended.b + attr.prepended.b + input.b,
               attr.appended.h + attr.prepended.h + input.h,
               attr.appended.w + attr.prepended.w + input.w,
               attr.appended.d + attr.prepended.d + input.d,
               attr.appended.c + attr.prepended.c + input.c);
}

BHWC CalculateOutputShape(const BHWC& input,
                          const FullyConnectedAttributes& attr) {
  return BHWC(input.b, 1, 1, attr.weights.shape.o);
}

BHWC CalculateOutputShape(const BHWC& input, const MeanAttributes& attr) {
  const int b = attr.dims.find(Axis::BATCH) == attr.dims.end() ? input.b : 1;
  const int h = attr.dims.find(Axis::HEIGHT) == attr.dims.end() ? input.h : 1;
  const int w = attr.dims.find(Axis::WIDTH) == attr.dims.end() ? input.w : 1;
  const int c = attr.dims.find(Axis::CHANNELS) == attr.dims.end() ? input.c : 1;
  return BHWC(b, h, w, c);
}

BHWDC CalculateOutputShape(const BHWDC& input, const MeanAttributes& attr) {
  const int b = attr.dims.find(Axis::BATCH) == attr.dims.end() ? input.b : 1;
  const int h = attr.dims.find(Axis::HEIGHT) == attr.dims.end() ? input.h : 1;
  const int w = attr.dims.find(Axis::WIDTH) == attr.dims.end() ? input.w : 1;
  const int d = attr.dims.find(Axis::DEPTH) == attr.dims.end() ? input.d : 1;
  const int c = attr.dims.find(Axis::CHANNELS) == attr.dims.end() ? input.c : 1;
  return BHWDC(b, h, w, d, c);
}

absl::Status CalculateOutputShape(const std::vector<BHWC>& input,
                                  const ConcatAttributes& attr,
                                  BHWC* output_shape) {
  BHWC new_shape = input[0];
  switch (attr.axis) {
    case Axis::CHANNELS:
      for (int i = 1; i < input.size(); i++) {
        if (input[i].h != new_shape.h || input[i].w != new_shape.w ||
            input[i].b != new_shape.b) {
          return absl::InvalidArgumentError(
              "Height, Width and Batch must be the same when concatenating "
              "by channels axis");
        }
        new_shape.c += input[i].c;
      }
      break;
    case Axis::HEIGHT:
      for (int i = 1; i < input.size(); i++) {
        if (input[i].w != new_shape.w || input[i].c != new_shape.c ||
            input[i].b != new_shape.b) {
          return absl::InvalidArgumentError(
              "Channels, Width and Batch must be the same when concatenating "
              "by height axis");
        }
        new_shape.h += input[i].h;
      }
      break;
    case Axis::WIDTH:
      for (int i = 1; i < input.size(); i++) {
        if (input[i].h != new_shape.h || input[i].c != new_shape.c ||
            input[i].b != new_shape.b) {
          return absl::InvalidArgumentError(
              "Height, Channels and Batch must be the same when concatenating "
              "by width axis");
        }
        new_shape.w += input[i].w;
      }
      break;
    case Axis::BATCH:
      for (int i = 1; i < input.size(); i++) {
        if (input[i].h != new_shape.h || input[i].c != new_shape.c ||
            input[i].w != new_shape.w) {
          return absl::InvalidArgumentError(
              "Width, Height and Channels must be the same when concatenating "
              "by batch axis");
        }
        new_shape.b += input[i].b;
      }
      break;
    default:
      return absl::InvalidArgumentError("Invalid axis");
      break;
  }
  *output_shape = new_shape;
  return absl::OkStatus();
}

absl::Status CalculateOutputShape(const std::vector<BHWDC>& input,
                                  const ConcatAttributes& attr,
                                  BHWDC* output_shape) {
  BHWDC new_shape = input[0];
  switch (attr.axis) {
    case Axis::CHANNELS:
      for (int i = 1; i < input.size(); ++i) {
        if (input[i].h != new_shape.h || input[i].w != new_shape.w ||
            input[i].d != new_shape.d || input[i].b != new_shape.b) {
          return absl::InvalidArgumentError(
              "Height, Width, Batch and Depth must be the same when "
              "concatenating "
              "by channels axis");
        }
        new_shape.c += input[i].c;
      }
      break;
    case Axis::HEIGHT:
      for (int i = 1; i < input.size(); ++i) {
        if (input[i].w != new_shape.w || input[i].c != new_shape.c ||
            input[i].d != new_shape.d || input[i].b != new_shape.b) {
          return absl::InvalidArgumentError(
              "Width, Depth, Batch and Channels must be the same when "
              "concatenating "
              "by height axis");
        }
        new_shape.h += input[i].h;
      }
      break;
    case Axis::WIDTH:
      for (int i = 1; i < input.size(); ++i) {
        if (input[i].h != new_shape.h || input[i].c != new_shape.c ||
            input[i].d != new_shape.d || input[i].b != new_shape.b) {
          return absl::InvalidArgumentError(
              "Height, Depth, Batch and Channels must be the same when "
              "concatenating "
              "by width axis");
        }
        new_shape.w += input[i].w;
      }
      break;
    case Axis::DEPTH:
      for (int i = 1; i < input.size(); ++i) {
        if (input[i].w != new_shape.w || input[i].h != new_shape.h ||
            input[i].c != new_shape.c || input[i].b != new_shape.b) {
          return absl::InvalidArgumentError(
              "Width, Height, Batch and Channels must be the same when "
              "concatenating "
              "by depth axis");
        }
        new_shape.d += input[i].d;
      }
      break;
    case Axis::BATCH:
      for (int i = 1; i < input.size(); ++i) {
        if (input[i].w != new_shape.w || input[i].h != new_shape.h ||
            input[i].c != new_shape.c || input[i].d != new_shape.d) {
          return absl::InvalidArgumentError(
              "Width, Height, Depth and Channels must be the same when "
              "concatenating "
              "by batch axis");
        }
        new_shape.b += input[i].b;
      }
      break;
    default:
      return absl::InvalidArgumentError("Invalid axis");
  }
  *output_shape = new_shape;
  return absl::OkStatus();
}

Padding2D CalculateSamePadding(const BHWC& input,
                               const Convolution2DAttributes& attr) {
  return MakeSamePadding(input, attr);
}

Padding3D CalculateSamePadding(const BHWDC& input,
                               const Convolution3DAttributes& attr) {
  return MakeSamePadding(input, attr);
}

Padding2D CalculateSamePadding(const BHWC& input,
                               const ConvolutionTransposedAttributes& attr) {
  return MakeSamePadding(input, attr);
}

Padding3D CalculateSamePadding(const BHWDC& input,
                               const ConvolutionTransposed3DAttributes& attr) {
  return MakeSamePadding(input, attr);
}

Padding2D CalculateSamePadding(const BHWC& input,
                               const DepthwiseConvolution2DAttributes& attr) {
  return MakeSamePadding(input, attr);
}

Padding3D CalculateSamePadding(const BHWDC& input,
                               const DepthwiseConvolution3DAttributes& attr) {
  return MakeSamePadding(input, attr);
}

Padding2D CalculateSamePadding(const BHWC& input,
                               const Pooling2DAttributes& attr) {
  return MakeSamePadding(input, attr);
}

Padding3D CalculateSamePadding(const BHWDC& input,
                               const Pooling3DAttributes& attr) {
  return MakeSamePadding(input, attr);
}

Padding2D CalculateSamePadding(const BHWC& input,
                               const MaxUnpooling2DAttributes& attr) {
  return MakeSamePadding(input, attr);
}

Padding3D CalculateSamePadding(const BHWDC& input,
                               const MaxUnpooling3DAttributes& attr) {
  return MakeSamePadding(input, attr);
}

float CalculateResizeScale(int32_t input_size, int32_t output_size,
                           const Resize2DAttributes& attr) {
  return attr.align_corners && input_size > 1 && output_size > 1
             ? static_cast<float>(input_size - 1) / (output_size - 1)
             : static_cast<float>(input_size) / output_size;
}

float CalculateResizeScale(int32_t input_size, int32_t output_size,
                           const Resize3DAttributes& attr) {
  return attr.align_corners && input_size > 1 && output_size > 1
             ? static_cast<float>(input_size - 1) / (output_size - 1)
             : static_cast<float>(input_size) / output_size;
}

BHWC CalculateOutputShape(const BHWC& input, const Resize2DAttributes& attr) {
  return BHWC(input.b, attr.new_shape.h, attr.new_shape.w, input.c);
}

BHWDC CalculateOutputShape(const BHWDC& input, const Resize3DAttributes& attr) {
  return BHWDC(input.b, attr.new_shape.h, attr.new_shape.w, attr.new_shape.d,
               input.c);
}

BHWC CalculateOutputShape(const BHWC& input, const TransposeAttributes& attr) {
  return BHWC(input.get(attr.perm.b), input.get(attr.perm.h),
              input.get(attr.perm.w), input.get(attr.perm.c));
}

BHWDC CalculateOutputShape(const BHWDC& input,
                           const Transpose3DAttributes& attr) {
  return BHWDC(input.get(attr.perm.b), input.get(attr.perm.h),
               input.get(attr.perm.w), input.get(attr.perm.d),
               input.get(attr.perm.c));
}

FullyConnectedAttributes DequatizeFullyConnectedAttr(
    const FullyConnectedInt8Attributes& attr) {
  FullyConnectedAttributes dequant_attr;
  dequant_attr.weights.id = attr.weights.id;
  dequant_attr.weights.shape = attr.weights.shape;
  dequant_attr.weights.data.resize(
      dequant_attr.weights.shape.DimensionsProduct());
  dequant_attr.bias = attr.bias;

  // weights dequantization to float32
  for (int i = 0; i < attr.weights.data.size(); i++) {
    const int32_t val = attr.weights.data[i];
    dequant_attr.weights.data[i] = attr.scale * (val - attr.zero_point);
  }
  return dequant_attr;
}

}  // namespace gpu
}  // namespace tflite
