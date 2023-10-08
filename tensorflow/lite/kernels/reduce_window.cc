/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
         //
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <type_traits>

#include "tensorflow/lite/array.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace reduce_window {
namespace {

constexpr int32_t kMaxReduceWindowDims = 6;

template <int Val>
using IntCst = std::integral_constant<int, Val>;

// Reduces the elements of a tensor viewed through a strided window.
//
// This applies a reduction to a tensor by skipping over elements that are not
// in the window defined by the given shape and strides. The window is reduced
// to one element.
//
// The shape is the shape of the window. The strides are based on the actual
// tensor and the distance between window elements, counted in elements. Sparse
// windows are possible.
//
// For instance: the following window has a [2, 2] shape and [8, 3] strides.
//
// ┌──┐     ┌──┐
// │ 1│ 2  3│ 4│
// └──┘     └──┘
//   5  6  7  8    is reduced to 1 + 4 + 9 + 12 = 26
// ┌──┐     ┌──┐
// │ 9│10 11│12│
// └──┘     └──┘
//  13 14 15 16
//
// This is a recursive implementation of the strided reduction.
template <class Op, class Type>
void StridedReduce(const Type* input, const int64_t* const shape,
                   const int64_t* const strides, Type& accu, const int rank,
                   const int depth) {
  const int64_t stride = strides[depth];
  const int64_t size = shape[depth];
  if (depth + 1 == rank) {
    const Op op;
    for (int64_t i = 0; i < size; ++i) {
      accu = op(accu, *input);
      input += stride;
    }
  } else {
    for (int64_t i = 0; i < size; ++i) {
      StridedReduce<Op, Type>(input, shape, strides, accu, rank, depth + 1);
      input += stride;
    }
  }
}

// Recursively computes strided reductions using a sliding window over the given
// tensor.
//
// The window is defined using a shape and a dilation. The shape defines the
// elements that the window will let the reduction *see*. The dilation defines
// the step between window elements.
//
// For instance: the following window has a [2, 2] shape and [2, 3] dilations.
//
//    3
// ┌────┐
// ┌─┐   ┌─┐
// │X│X X│X│┐
// └─┘   └─┘│2
//  X X X X ┘
// ┌─┐   ┌─┐
// │X│X X│X│
// └─┘   └─┘
template <class Op, class Type>
void ReduceWindowImpl(const Type* input, Type* output,
                      const int64_t* const output_shape,
                      const int64_t* const output_strides,
                      const int64_t* const window_offset_strides,
                      const int64_t* const window_shape,
                      const int64_t* const window_reduce_strides,
                      const Type init, const int rank, const int depth) {
  if (depth + 1 == rank) {
    for (int32_t dim = 0; dim < output_shape[depth]; ++dim) {
      *output = init;
      StridedReduce<Op, Type>(input, window_shape, window_reduce_strides,
                              *output, rank, 0);
      input += window_offset_strides[depth];
      output += output_strides[depth];
    }
  } else {
    for (int32_t dim = 0; dim < output_shape[depth]; ++dim) {
      ReduceWindowImpl<Op, Type>(input, output, output_shape, output_strides,
                                 window_offset_strides, window_shape,
                                 window_reduce_strides, init, rank, depth + 1);
      input += window_offset_strides[depth];
      output += output_strides[depth];
    }
  }
}

std::array<int64_t, kMaxReduceWindowDims> ComputeStrides(
    const int64_t* const shape, const int64_t rank) {
  std::array<int64_t, kMaxReduceWindowDims> strides;
  strides[rank - 1] = 1;
  for (int64_t i = rank - 2; i >= 0; --i) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  return strides;
}

// Element-wise multiplication of the two operands of given size.
std::array<int64_t, kMaxReduceWindowDims> Multiply(const int64_t* const vec1,
                                                   const int64_t* const vec2,
                                                   const int64_t size) {
  std::array<int64_t, kMaxReduceWindowDims> result;
  for (int64_t i = 0; i < size; ++i) {
    result[i] = vec2[i] * vec1[i];
  }
  return result;
}

// Computes the output shape of the ReduceWindow operator.
std::array<int64_t, kMaxReduceWindowDims> ComputeOutputShape(
    const int64_t* const shape, const int64_t* const window_shape,
    const int64_t* const window_strides, const int64_t* const window_dilations,
    const int64_t rank) {
  std::array<int64_t, kMaxReduceWindowDims> dilated_window_shape;
  for (int64_t i = 0; i < rank; ++i) {
    dilated_window_shape[i] = (window_shape[i] - 1) * window_dilations[i] + 1;
  }

  std::array<int64_t, kMaxReduceWindowDims> window_range;
  for (int64_t i = 0; i < rank; ++i) {
    window_range[i] =
        (shape[i] - dilated_window_shape[i]) / window_strides[i] + 1;
  }
  return window_range;
}

template <class Op, class Type>
void ReduceWindow(const Type* const input, Type* output,
                  const int64_t* const shape, const int64_t* const window_shape,
                  const int64_t* const window_strides,
                  const int64_t* const window_dilations, const Type init,
                  const int rank) {
  const std::array<int64_t, kMaxReduceWindowDims> strides =
      ComputeStrides(shape, rank);
  const std::array<int64_t, kMaxReduceWindowDims> window_reduce_strides =
      Multiply(strides.data(), window_dilations, rank);
  const std::array<int64_t, kMaxReduceWindowDims> window_offset_strides =
      Multiply(strides.data(), window_strides, rank);
  const std::array<int64_t, kMaxReduceWindowDims> output_shape =
      ComputeOutputShape(shape, window_shape, window_strides, window_dilations,
                         rank);
  const std::array<int64_t, kMaxReduceWindowDims> output_strides =
      ComputeStrides(output_shape.data(), rank);
  ReduceWindowImpl<Op, Type>(input, output, output_shape.data(),
                             output_strides.data(),
                             window_offset_strides.data(), window_shape,
                             window_reduce_strides.data(), init, rank, 0);
}

std::array<int64_t, kMaxReduceWindowDims> AsInt64(const int32_t* data,
                                                  const int size) {
  std::array<int64_t, kMaxReduceWindowDims> res;
  std::copy_n(data, size, res.data());
  return res;
}

// Holds the tensors and operation context for convenience.
struct ReduceWindowContext {
  enum InputTensorId {
    kInput,
    kInitValue,
    kWindowShape,
    kWindowStrides,
    kWindowDilations,
    kNumInputTensors
  };
  enum OutputTensorId { kOutput, kNumOutputTensors };

  ReduceWindowContext(TfLiteContext* context, TfLiteNode* node)
      : context(context),
        node(node),
        input_tensor(GetInput(context, node, kInput)),
        init_value_tensor(GetInput(context, node, kInitValue)),
        window_shape_tensor(GetInput(context, node, kWindowShape)),
        window_strides_tensor(GetInput(context, node, kWindowStrides)),
        window_dilations_tensor(GetInput(context, node, kWindowDilations)),
        output_tensor(GetOutput(context, node, kOutput)) {}

  TfLiteContext* context;
  TfLiteNode* node;
  const TfLiteTensor* input_tensor;
  const TfLiteTensor* init_value_tensor;
  const TfLiteTensor* window_shape_tensor;
  const TfLiteTensor* window_strides_tensor;
  const TfLiteTensor* window_dilations_tensor;
  TfLiteTensor* output_tensor;
};

TfLiteStatus SetupOutputTensor(const ReduceWindowContext& ctx) {
  const int rank = ctx.input_tensor->dims->size;
  const std::array<int64_t, kMaxReduceWindowDims> input_shape =
      AsInt64(ctx.input_tensor->dims->data, rank);
  const std::array<int64_t, kMaxReduceWindowDims> output_shape_data =
      ComputeOutputShape(input_shape.data(), ctx.window_shape_tensor->data.i64,
                         ctx.window_strides_tensor->data.i64,
                         ctx.window_dilations_tensor->data.i64, rank);
  IntArrayUniquePtr output_shape =
      BuildTfLiteArray<int32_t>(rank, output_shape_data.data());
  return ctx.context->ResizeTensor(ctx.context, ctx.output_tensor,
                                   output_shape.release());
}

template <class Op>
TfLiteStatus DispatchReduceWindowType(ReduceWindowContext& ctx) {
  const int rank = ctx.input_tensor->dims->size;
  const std::array<int64_t, kMaxReduceWindowDims> input_shape =
      AsInt64(ctx.input_tensor->dims->data, rank);
#define REDUCE_WINDOW_TYPE_CASE(CPP_TYPE, TENSOR_TYPE)                        \
  case TENSOR_TYPE:                                                           \
    ReduceWindow<Op, CPP_TYPE>(                                               \
        reinterpret_cast<const CPP_TYPE*>(ctx.input_tensor->data.raw),        \
        reinterpret_cast<CPP_TYPE*>(ctx.output_tensor->data.raw),             \
        input_shape.data(), ctx.window_shape_tensor->data.i64,                \
        ctx.window_strides_tensor->data.i64,                                  \
        ctx.window_dilations_tensor->data.i64,                                \
        *reinterpret_cast<CPP_TYPE*>(ctx.init_value_tensor->data.raw), rank); \
    break;
  switch (ctx.input_tensor->type) {
    REDUCE_WINDOW_TYPE_CASE(int8_t, kTfLiteBool);
    REDUCE_WINDOW_TYPE_CASE(int8_t, kTfLiteInt8);
    REDUCE_WINDOW_TYPE_CASE(int16_t, kTfLiteInt16);
    REDUCE_WINDOW_TYPE_CASE(int32_t, kTfLiteInt32);
    REDUCE_WINDOW_TYPE_CASE(int64_t, kTfLiteInt64);
    REDUCE_WINDOW_TYPE_CASE(uint8_t, kTfLiteUInt8);
    // REDUCE_WINDOW_TYPE_CASE(uint16_t, kTfLiteUInt16);
    // REDUCE_WINDOW_TYPE_CASE(uint32_t, kTfLiteUInt32);
    // REDUCE_WINDOW_TYPE_CASE(uint64_t, kTfLiteUInt64);
    REDUCE_WINDOW_TYPE_CASE(float, kTfLiteFloat32);
    static_assert(sizeof(float) == 4,
                  "float type is expected to be 32 bit long");
    REDUCE_WINDOW_TYPE_CASE(double, kTfLiteFloat64);
    static_assert(sizeof(double) == 8,
                  "double type is expected to be 64 bit long");
    default:
      return kTfLiteError;
  }
#undef REDUCE_WINDOW_TYPE_CASE
  return kTfLiteOk;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node),
                    ReduceWindowContext::kNumInputTensors);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node),
                    ReduceWindowContext::kNumOutputTensors);
  ReduceWindowContext ctx(context, node);
  TF_LITE_ENSURE(context, IsConstantTensor(ctx.window_shape_tensor));
  TF_LITE_ENSURE(context, IsConstantTensor(ctx.window_strides_tensor));
  TF_LITE_ENSURE(context, IsConstantTensor(ctx.window_dilations_tensor));
  TF_LITE_ENSURE(context, ctx.input_tensor->dims != nullptr);
  TF_LITE_ENSURE(context, ctx.input_tensor->dims->size > 0);
  TF_LITE_ENSURE(context, ctx.input_tensor->dims->size <= kMaxReduceWindowDims);
  return SetupOutputTensor(ctx);
}

struct Max {
  template <class T>
  constexpr T operator()(const T& a, const T& b) const {
    return a >= b ? a : b;
  }
};

struct Min {
  template <class T>
  constexpr T operator()(const T& a, const T& b) const {
    return a <= b ? a : b;
  }
};

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const auto& params =
      *reinterpret_cast<TfLiteReduceWindowParams*>(node->builtin_data);
  ReduceWindowContext ctx(context, node);
  switch (params.reduce_function) {
    case TfLiteReduceWindowFunctionUnsupported:
      return kTfLiteError;
    case TfLiteReduceWindowFunctionAdd:
      return DispatchReduceWindowType<std::plus<>>(ctx);
    case TfLiteReduceWindowFunctionMul:
      return DispatchReduceWindowType<std::multiplies<>>(ctx);
    case TfLiteReduceWindowFunctionAll:
      return DispatchReduceWindowType<std::logical_and<>>(ctx);
    case TfLiteReduceWindowFunctionAny:
      return DispatchReduceWindowType<std::logical_or<>>(ctx);
    case TfLiteReduceWindowFunctionMin:
      return DispatchReduceWindowType<Min>(ctx);
    case TfLiteReduceWindowFunctionMax:
      return DispatchReduceWindowType<Max>(ctx);
  }
}

}  // namespace
}  // namespace reduce_window

TfLiteRegistration* Register_REDUCE_WINDOW() {
  static TfLiteRegistration r = {/*.init=*/nullptr, /*.free=*/nullptr,
                                 /*.prepare=*/reduce_window::Prepare,
                                 /*.invoke=*/reduce_window::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
