/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>
#include <unordered_map>

#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

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
    case OperationType::APPLY_MASK:
      return "apply_mask";
    case OperationType::BATCH_NORMALIZATION:
      return "batch_normalization";
    case OperationType::BATCH_TO_SPACE:
      return "batch_to_space";
    case OperationType::CONCAT:
      return "concat";
    case OperationType::CONST:
      return "const";
    case OperationType::CONVOLUTION_2D:
      return "convolution_2d";
    case OperationType::CONVOLUTION_TRANSPOSED:
      return "convolution_transposed";
    case OperationType::COS:
      return "cos";
    case OperationType::DEPTHWISE_CONVOLUTION:
      return "depthwise_convolution";
    case OperationType::DIV:
      return "div";
    case OperationType::FULLY_CONNECTED:
      return "fully_connected";
    case OperationType::HARD_SWISH:
      return "hard_swish";
    case OperationType::LOG:
      return "log";
    case OperationType::LSTM:
      return "lstm";
    case OperationType::MAX_UNPOOLING_2D:
      return "max_unpooling";
    case OperationType::MUL:
      return "mul";
    case OperationType::MULTIPLY_SCALAR:
      return "multiply_scalar";
    case OperationType::PAD:
      return "pad";
    case OperationType::POOLING_2D:
      return "pooling_2d";
    case OperationType::POW:
      return "pow";
    case OperationType::PRELU:
      return "prelu";
    case OperationType::RELU:
      return "relu";
    case OperationType::RESHAPE:
      return "reshape";
    case OperationType::RESIZE:
      return "resize";
    case OperationType::RSQRT:
      return "rsqrt";
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
    case OperationType::TRANSPOSE:
      return "transpose";
    case OperationType::UPSAMPLE_2D:
      return "upsample_2d";
    default:
      break;
  }
  return "unknown_operation";
}

OperationType OperationTypeFromString(const std::string& name) {
  static const auto operations =
      new std::unordered_map<std::string, OperationType>({
          {"abs", OperationType::ABS},
          {"add", OperationType::ADD},
          {"apply_mask", OperationType::APPLY_MASK},
          {"batch_normalization", OperationType::BATCH_NORMALIZATION},
          {"concat", OperationType::CONCAT},
          {"const", OperationType::CONST},
          {"convolution_2d", OperationType::CONVOLUTION_2D},
          {"convolution_transposed", OperationType::CONVOLUTION_TRANSPOSED},
          {"cos", OperationType::COS},
          {"depthwise_convolution", OperationType::DEPTHWISE_CONVOLUTION},
          {"div", OperationType::DIV},
          {"fully_connected", OperationType::FULLY_CONNECTED},
          {"hard_swish", OperationType::HARD_SWISH},
          {"log", OperationType::LOG},
          {"lstm", OperationType::LSTM},
          {"max_unpooling", OperationType::MAX_UNPOOLING_2D},
          {"mul", OperationType::MUL},
          {"multiply_scalar", OperationType::MULTIPLY_SCALAR},
          {"pad", OperationType::PAD},
          {"pooling_2d", OperationType::POOLING_2D},
          {"pow", OperationType::POW},
          {"prelu", OperationType::PRELU},
          {"relu", OperationType::RELU},
          {"resize", OperationType::RESIZE},
          {"reshape", OperationType::RESHAPE},
          {"rsqrt", OperationType::RSQRT},
          {"sigmoid", OperationType::SIGMOID},
          {"sin", OperationType::SIN},
          {"slice", OperationType::SLICE},
          {"softmax", OperationType::SOFTMAX},
          {"sqrt", OperationType::SQRT},
          {"square", OperationType::SQUARE},
          {"squared_diff", OperationType::SQUARED_DIFF},
          {"subtract", OperationType::SUB},
          {"tanh", OperationType::TANH},
          {"transpose", OperationType::TRANSPOSE},
          {"upsample_2d", OperationType::UPSAMPLE_2D},
      });
  auto op = operations->find(name);
  return op == operations->end() ? OperationType::UNKNOWN : op->second;
}

namespace {

template <typename T>
T IntegralDivideRoundUp(T n, T divisor) {
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

inline int32_t StridedSize(int32_t size, int32_t stride) {
  return stride == 0 ? -1 : IntegralDivideRoundUp(size, stride);
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

template <Axis AxisT>
int32_t CalculateSamePadding(const BHWC& input,
                             const ConvolutionTransposedAttributes& attr) {
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

BHWC CalculateOutputShape(const BHWC& input,
                          const ConvolutionTransposedAttributes& attr) {
  return BHWC(input.b, CalculateOutput<Axis::HEIGHT>(input, attr),
              CalculateOutput<Axis::WIDTH>(input, attr),
              attr.weights.shape.get<Axis::OUTPUT_CHANNELS>());
}

BHWC CalculateOutputShape(const BHWC& input,
                          const DepthwiseConvolution2DAttributes& attr) {
  return BHWC(input.b, CalculateOutput<Axis::HEIGHT>(input, attr),
              CalculateOutput<Axis::WIDTH>(input, attr),
              attr.weights.shape.get<Axis::OUTPUT_CHANNELS>() *
                  attr.weights.shape.get<Axis::INPUT_CHANNELS>());
}

BHWC CalculateOutputShape(const BHWC& input, const SliceAttributes& attr) {
  return BHWC(StridedSize(attr.ends.b - attr.starts.b, attr.strides.b),
              StridedSize(attr.ends.h - attr.starts.h, attr.strides.h),
              StridedSize(attr.ends.w - attr.starts.w, attr.strides.w),
              StridedSize(attr.ends.c - attr.starts.c, attr.strides.c));
}

BHWC CalculateOutputShape(const BHWC& input, const PadAttributes& attr) {
  return BHWC(attr.appended.b + attr.prepended.b + input.b,
              attr.appended.h + attr.prepended.h + input.h,
              attr.appended.w + attr.prepended.w + input.w,
              attr.appended.c + attr.prepended.c + input.c);
}

BHWC CalculateOutputShape(const BHWC& input,
                          const FullyConnectedAttributes& attr) {
  return BHWC(input.b, 1, 1, attr.weights.shape.o);
}

Status CalculateOutputShape(const std::vector<BHWC>& input,
                            const ConcatAttributes& attr, BHWC* output_shape) {
  BHWC new_shape = input[0];
  switch (attr.axis) {
    case Axis::CHANNELS:
      for (int i = 1; i < input.size(); i++) {
        if (input[i].h != new_shape.h || input[i].w != new_shape.w) {
          return InvalidArgumentError(
              "Height and Width must be the same when concatenating "
              "by channels axis");
        }
        new_shape.c += input[i].c;
      }
      break;
    case Axis::HEIGHT:
      for (int i = 1; i < input.size(); i++) {
        if (input[i].w != new_shape.w || input[i].c != new_shape.c) {
          return InvalidArgumentError(
              "Channels and Width must be the same when concatenating "
              "by height axis");
        }
        new_shape.h += input[i].h;
      }
      break;
    case Axis::WIDTH:
      for (int i = 1; i < input.size(); i++) {
        if (input[i].h != new_shape.h || input[i].c != new_shape.c) {
          return InvalidArgumentError(
              "Height and Channels must be the same when concatenating "
              "by width axis");
        }
        new_shape.w += input[i].w;
      }
      break;
    default:
      return InvalidArgumentError("Invalid axis");
      break;
  }
  *output_shape = new_shape;
  return OkStatus();
}

Padding2D CalculateSamePadding(const BHWC& input,
                               const Convolution2DAttributes& attr) {
  return MakeSamePadding(input, attr);
}

Padding2D CalculateSamePadding(const BHWC& input,
                               const ConvolutionTransposedAttributes& attr) {
  return MakeSamePadding(input, attr);
}

Padding2D CalculateSamePadding(const BHWC& input,
                               const DepthwiseConvolution2DAttributes& attr) {
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

float CalculateResizeScale(int32_t input_size, int32_t output_size,
                           const Upsample2DAttributes& attr) {
  return attr.align_corners && input_size > 1 && output_size > 1
             ? static_cast<float>(input_size - 1) / (output_size - 1)
             : static_cast<float>(input_size) / output_size;
}

BHWC CalculateOutputShape(const BHWC& input, const Upsample2DAttributes& attr) {
  return BHWC(input.b, attr.new_shape.h, attr.new_shape.w, input.c);
}

BHWC CalculateOutputShape(const BHWC& input, const TransposeAttributes& attr) {
  return BHWC(input.get(attr.perm.b), input.get(attr.perm.h),
              input.get(attr.perm.w), input.get(attr.perm.c));
}

}  // namespace gpu
}  // namespace tflite
