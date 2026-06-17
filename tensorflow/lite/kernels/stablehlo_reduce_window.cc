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
#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

#include "tensorflow/lite/array.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace ops {
namespace builtin {

namespace {
constexpr int32_t kMaxReduceWindowRank = 6;

// Reccursive implementation of a strided copy of a tensor.
void StridedCopy(const int rank, const char* input, const int64_t* input_shape,
                 const int64_t* input_strides, char* output,
                 const int64_t* output_strides, const int64_t element_size,
                 const int depth) {
  if (depth + 1 == rank) {
    for (int64_t i = 0; i < input_shape[depth]; ++i) {
      std::memcpy(output, input, element_size);
      input += input_strides[depth];
      output += output_strides[depth];
    }
  } else {
    for (int64_t i = 0; i < input_shape[depth]; ++i) {
      StridedCopy(rank, input, input_shape, input_strides, output,
                  output_strides, element_size, depth + 1);
      input += input_strides[depth];
      output += output_strides[depth];
    }
  }
}

}  // namespace

namespace dilate {
namespace {

const int64_t kTFLiteDefaultBaseDilation[kMaxReduceWindowRank] = {1, 1, 1,
                                                                  1, 1, 1};

// Computes and holds the parameters that can be precomputed for the dilation
// operation.
struct DilateData {
  DilateData() = default;

  DilateData(const int rank, const int64_t* input_shape,
             const int64_t* dilation, const int64_t element_size)
      : rank(rank), init_element_size(element_size) {
    std::copy_n(input_shape, rank, shape);
    std::copy_n(dilation, rank, base_dilations);
    ComputeOutputShapeAndSize(element_size);
    skip = std::all_of(dilation, dilation + rank,
                       [](int64_t d) { return d == 1; });
    if (skip) {
      return;
    }
    MergeTrailingDilations(element_size);
    ComputeInputStrides();
    ComputeOutputStridesAndSizes();
  }

  // Trailing dilation factors of 1 can be merged to the left.
  //
  // This optimisation artificially reduces the number of dimensions of the
  // input tensor. If a dilation factor is 1 then no padding element is added
  // between elements of the given dimension. From the innermost dimension we
  // can collapse all the adjacent dimensions that have a dilation factor
  // of 1.
  //
  // Note: this function updates input_strides[rank-1].
  void MergeTrailingDilations(int64_t element_size) {
    for (int i = rank - 2; i >= 0; --i) {
      if (base_dilations[i + 1] == 1) {
        element_size *= shape[i + 1];
        --rank;
      } else {
        break;
      }
    }
    // This can only happen if all the dilation factors are 1. It would be
    // better to just not apply the operation but we check it as a failsafe.
    if (rank == 1 && base_dilations[0] == 1) {
      element_size *= shape[0];
      shape[0] = 1;
    }
    input_strides[rank - 1] = element_size;
  }

  // Computes the input strides using the shape and the element size.
  //
  // Note the element size must be stored in `input_strides[rank-1]`.
  void ComputeInputStrides() {
    assert(input_strides[rank - 1] != 0);
    for (int i = rank - 2; i >= 0; --i) {
      input_strides[i] = shape[i + 1] * input_strides[i + 1];
    }
  }

  // Computes the output stride and the byte size for each dimension.
  //
  // The size of a dimension is not the same as the stride of the next
  // inner dimension because of the dilation.
  //
  // Note the element size must be stored in `input_strides[rank-1]`.
  void ComputeOutputStridesAndSizes() {
    output_dimension_sizes[rank - 1] = input_strides[rank - 1];
    output_strides[rank - 1] =
        base_dilations[rank - 1] * output_dimension_sizes[rank - 1];
    for (int i = rank - 2; i >= 0; --i) {
      output_dimension_sizes[i] = ((shape[i + 1] - 1) * output_strides[i + 1] +
                                   output_dimension_sizes[i + 1]);
      output_strides[i] = base_dilations[i] * output_dimension_sizes[i];
    }
  }

  void ComputeOutputShapeAndSize(const int64_t element_size) {
    output_size = element_size;
    for (int i = 0; i < rank; ++i) {
      output_shape[i] = (shape[i] - 1) * base_dilations[i] + 1;
      output_size *= output_shape[i];
    }
  }

  int64_t ElementSize() const { return input_strides[rank - 1]; }

  bool skip = true;
  int rank = 0;
  int64_t init_element_size = 0;
  int64_t shape[kMaxReduceWindowRank] = {};
  int64_t base_dilations[kMaxReduceWindowRank] = {};
  int64_t output_strides[kMaxReduceWindowRank] = {};
  int64_t output_dimension_sizes[kMaxReduceWindowRank] = {};
  int64_t input_strides[kMaxReduceWindowRank] = {};
  int64_t output_shape[kMaxReduceWindowRank] = {};
  int64_t output_size = 1;
};

// Dilates the input tensor following the parameters held in the given context.
//
// The dilation operation scatters the elements of its input into a new tensor
// according to a dilation factor for each dimension. The new tensor elements
// are initialized to 0.
//
// This operation can also be seen as adding interior padding to the tensor. In
// that case, `interior padding size = dilation factor - 1`.
//
// For instance:
//
//                        1 2 3
// A is a 3x3 tensor. A = 4 5 6
//                        7 8 9
//
// We apply a dilation of 2x3.
//
//                         1 0 0 2 0 0 3
//                         0 0 0 0 0 0 0
// B = dilate(A, [2, 3]) = 4 0 0 5 0 0 6
//                         0 0 0 0 0 0 0
//                         7 0 0 8 0 0 9
//
// More rigorously:
// - Let [s0, ..., sN] be the shape of A.
// - Let [d0, ..., dN] be the dilation factors.
//
// - The shape of B is [(s0 - 1) * d0 + 1, ..., (sN - 1) * dN + 1].
// - B(i0, ..., iN) = ┌ A(i0 / d0, ..., iN / dN)   if iX % dX == 0 for all X
//                    └ 0 otherwise.
void Dilate(const DilateData& ctx, const char* input, const char* init_value,
            char* output) {
  assert(!ctx.skip);
  // Fill the output tensor with the padding value.
  {
    std::memcpy(output, init_value, ctx.init_element_size);
    int64_t remaining_bytes = ctx.output_size - ctx.init_element_size;
    int64_t copied_bytes = ctx.init_element_size;
    while (remaining_bytes) {
      int64_t bytes = std::min(remaining_bytes, copied_bytes);
      std::memcpy(output + copied_bytes, output, bytes);
      remaining_bytes -= bytes;
      copied_bytes += bytes;
    }
  }
  // Copy the relevant input elements into the output tensor.
  StridedCopy(ctx.rank, input, ctx.shape, ctx.input_strides, output,
              ctx.output_strides, ctx.ElementSize(), 0);
}

}  // namespace
}  // namespace dilate

namespace pad {
namespace {

const int64_t kTFLiteDefaultPadding[kMaxReduceWindowRank] = {0, 0, 0, 0, 0, 0};

// Computes and holds the parameters that can be precomputed for the padding
// operation. Note that StableHLO padding treats negative values as cropping.
struct PadCropData {
  PadCropData() = default;

  PadCropData(int rank, const int64_t* dims, const int64_t* padding,
              const int64_t element_size)
      : rank(rank), element_size(element_size) {
    assert(rank > 0);
    assert(rank < kMaxReduceWindowRank);

    // Compute the output shape.
    output_size = element_size;
    for (int i = 0; i < rank; ++i) {
      output_shape[i] = dims[i] + padding[2 * i] + padding[2 * i + 1];
      output_size *= output_shape[i];
    }

    skip = std::all_of(padding, padding + 2 * rank,
                       [](int64_t v) { return v == 0; });
    if (skip) {
      return;
    }

    // Compute the strides for the input and the output tensors.
    output_strides[rank - 1] = element_size;
    input_strides[rank - 1] = element_size;
    for (int i = rank - 2; i >= 0; --i) {
      output_strides[i] = output_shape[i + 1] * output_strides[i + 1];
      input_strides[i] = dims[i + 1] * input_strides[i + 1];
    }

    // Compute the offset to apply to the pointers to take into account
    // padding.
    for (int i = 0; i < rank; ++i) {
      input_offset += std::max<int64_t>(-padding[2 * i], 0) * input_strides[i];
      output_offset += std::max<int64_t>(padding[2 * i], 0) * output_strides[i];
      cropped_input_shape[i] = dims[i] + std::min<int64_t>(padding[2 * i], 0) +
                               std::min<int64_t>(padding[2 * i + 1], 0);
    }
  }

  bool skip = true;
  int rank = 0;
  int64_t element_size = 0;
  int64_t cropped_input_shape[kMaxReduceWindowRank];
  int64_t input_strides[kMaxReduceWindowRank];
  int64_t output_shape[kMaxReduceWindowRank];
  int64_t output_strides[kMaxReduceWindowRank];
  int64_t input_offset = 0;
  int64_t output_offset = 0;
  int64_t output_size = 0;
};

// Pads and crops the input tensor following the parameters held in the given
// context.
//
// The StableHLO padding algorithm uses negative values to denote cropping.
void PadCrop(const PadCropData& ctx, const char* input, const char* init_value,
             char* output) {
  assert(!ctx.skip);
  // Fill the output tensor with the padding value.
  {
    std::memcpy(output, init_value, ctx.element_size);
    int64_t remaining_bytes = ctx.output_size - ctx.element_size;
    int64_t copied_bytes = ctx.element_size;
    while (remaining_bytes) {
      int64_t bytes = std::min(remaining_bytes, copied_bytes);
      std::memcpy(output + copied_bytes, output, bytes);
      remaining_bytes -= bytes;
      copied_bytes += bytes;
    }
  }
  // Copy the relevant input elements into the output tensor.
  StridedCopy(ctx.rank, input + ctx.input_offset, ctx.cropped_input_shape,
              ctx.input_strides, output + ctx.output_offset, ctx.output_strides,
              ctx.element_size, /*depth=*/0);
}

}  // namespace
}  // namespace pad

namespace reduce_window {
namespace {

// Reduces the elements of a tensor viewed through a strided window.
//
// This applies a reduction to a tensor by skipping over elements that are not
// in the window defined by the given shape and strides. The window is reduced
// to one element.
//
// The shape is the shape of the window. The strides are based on the actual
// tensor and the distance between window elements, counted in elements.
// Sparse windows are possible.
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

// Recursively computes strided reductions using a sliding window over the
// given tensor.
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
                              *output, rank, /*depth=*/0);
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

// Computes and holds the parameters that can be precomputed for the dilation
// operation.
struct ReduceWindowData {
  ReduceWindowData() = default;

  ReduceWindowData(const int rank, const int64_t* input_shape,
                   const int64_t* window_shape, const int64_t* window_strides,
                   const int64_t* window_dilations)
      : rank(rank),
        input_shape(input_shape),
        window_shape(window_shape),
        window_dilations(window_dilations),
        window_strides(window_strides) {
    ComputeStrides(input_strides, input_shape);
    Multiply(window_reduce_strides, input_strides, window_dilations);
    Multiply(window_offset_strides, input_strides, window_strides);
    ComputeOutputShape();
    ComputeStrides(output_strides, output_shape);
  }

  void ComputeStrides(int64_t* strides, const int64_t* const shape) {
    strides[rank - 1] = 1;
    for (int64_t i = rank - 2; i >= 0; --i) {
      strides[i] = shape[i + 1] * strides[i + 1];
    }
  }

  void Multiply(int64_t* dst, const int64_t* const vec1,
                const int64_t* const vec2) {
    for (int64_t i = 0; i < rank; ++i) {
      dst[i] = vec2[i] * vec1[i];
    }
  }

  void ComputeOutputShape() {
    int64_t dilated_window_shape[kMaxReduceWindowRank];
    for (int64_t i = 0; i < rank; ++i) {
      dilated_window_shape[i] = (window_shape[i] - 1) * window_dilations[i] + 1;
    }
    for (int64_t i = 0; i < rank; ++i) {
      if (input_shape[i] < dilated_window_shape[i]) {
        output_shape[i] = 0;
      } else {
        output_shape[i] =
            (input_shape[i] - dilated_window_shape[i]) / window_strides[i] + 1;
      }
    }
  }

  int rank = 0;
  const int64_t* input_shape;
  const int64_t* window_shape;
  const int64_t* window_dilations;
  const int64_t* window_strides;
  int64_t input_strides[kMaxReduceWindowRank] = {};
  int64_t window_offset_strides[kMaxReduceWindowRank] = {};
  int64_t window_reduce_strides[kMaxReduceWindowRank] = {};
  int64_t output_shape[kMaxReduceWindowRank] = {};
  int64_t output_strides[kMaxReduceWindowRank] = {};
};

template <class Op, class Type>
void ReduceWindow(const ReduceWindowData& ctx, const Type* const input,
                  const Type init, Type* output) {
  ReduceWindowImpl<Op, Type>(input, output, ctx.output_shape,
                             ctx.output_strides, ctx.window_offset_strides,
                             ctx.window_shape, ctx.window_reduce_strides, init,
                             ctx.rank, /*depth=*/0);
}

}  // namespace
}  // namespace reduce_window

/// Operator implementation

namespace reduce_window_op {
namespace {

// Holds the data needed throughout the node lifetime.
struct NodeData {
  // These members are only for STABLEHLO_REDUCE_WINDOW
  enum { kDilateOutput, kPadOutput, kTempTensorCount };
  int temporary_tensor_offset = -1;
  // These members are shared.
  pad::PadCropData pad_ctx;
  dilate::DilateData dilate_ctx;
  reduce_window::ReduceWindowData reduce_window_ctx;
  TfLiteReduceWindowFunction body;
};

// Holds the operation data. This is extended by the StablehloData and the
// TFLiteData classes.
//
// There are two available semantics for this op implementation.
//
// - StablehloData, that models the STABLEHLO_REDUCE_WINDOW op.
// - TFLiteData, that models the DEPRECATED initial REDUCE_WINDOW op.
struct OpData {
  OpData(TfLiteContext* context, TfLiteNode* node)
      : context(context), node(node) {}

  TfLiteContext* context;
  TfLiteNode* node;

  TfLiteType type;
  int rank;
  int64_t element_size;
  int64_t input_dims[kMaxReduceWindowRank];
  const char* input;
  const char* init_value;
  const int64_t* window_dimensions;
  const int64_t* window_strides;
  const int64_t* base_dilations;
  const int64_t* window_dilations;
  const int64_t* padding;
  char* dilate_output = nullptr;
  char* pad_output = nullptr;
  char* output;

  // Helper to resize a tensor.
  TfLiteStatus ResizeTensor(TfLiteTensor* const tensor,
                            const int64_t* const shape) {
    auto dims = BuildTfLiteArray<int32_t>(rank, shape);
    return context->ResizeTensor(context, tensor, dims.release());
  }

  // Sets the operation data type and the associated byte size.
  TfLiteStatus SetElementType(TfLiteType t) {
    type = t;
    size_t unsigned_element_size;
    TF_LITE_ENSURE_OK(context,
                      GetSizeOfType(context, type, &unsigned_element_size));
    TF_LITE_ENSURE_MSG(
        context,
        // Directly comparing the unsigned_element_size to the max value of
        // int64_t fails the -Wtautological-constant-out-of-range-compare
        // warning when building on 32 bit targets.
        sizeof(unsigned_element_size) < sizeof(int64_t) ||
            unsigned_element_size <= std::numeric_limits<int64_t>::max(),
        "The element size cannot be contained in an int64_t value.");
    element_size = unsigned_element_size;
    return kTfLiteOk;
  }

  // Factors the initialization that are common across semantics.
  //
  // Semantic is one of StablehloData or TFLiteData.
  template <class Semantic>
  TfLiteStatus InitializeBase() {
    init_value = reinterpret_cast<const char*>(
        GetInput(context, node, Semantic::kInitValue)->data.data);

    const TfLiteTensor* const input_tensor =
        GetInput(context, node, Semantic::kInput);
    SetElementType(input_tensor->type);
    rank = input_tensor->dims->size;
    std::copy_n(input_tensor->dims->data, rank, input_dims);
    input = reinterpret_cast<const char*>(input_tensor->data.data);

    TfLiteTensor* const output_tensor =
        GetOutput(context, node, Semantic::kOutput);
    output = reinterpret_cast<char*>(output_tensor->data.data);
    return kTfLiteOk;
  }
};

// Speciliazes OpData for the STABLEHLO_REDUCE_WINDOW operation.
struct StablehloData : public OpData {
  enum InputTensorId { kInput, kInitValue, kNumInputTensors };
  enum OutputTensorId { kOutput, kNumOutputTensors };

  using OpData::OpData;

  TfLiteTensor* GetTemporary(int id) {
    return tflite::GetTemporary(context, node, id);
  }

  TfLiteStatus Check() const {
    TF_LITE_ENSURE_EQ(context, NumInputs(node), kNumInputTensors);
    TF_LITE_ENSURE_EQ(context, NumOutputs(node), kNumOutputTensors);
    const TfLiteTensor* const input_tensor = GetInput(context, node, kInput);
    const TfLiteTensor* const output_tensor = GetOutput(context, node, kOutput);
    const TfLiteTensor* const init_value_tensor =
        GetInput(context, node, kInitValue);
    TF_LITE_ENSURE_EQ(context, input_tensor->type, output_tensor->type);
    TF_LITE_ENSURE_EQ(context, input_tensor->type, init_value_tensor->type);
    TF_LITE_ENSURE(context, input_tensor->dims != nullptr);
    TF_LITE_ENSURE(context, input_tensor->dims->size > 0);
    TF_LITE_ENSURE(context, input_tensor->dims->size <= kMaxReduceWindowRank);
    return kTfLiteOk;
  }

  TfLiteStatus Initialize() {
    TF_LITE_ENSURE_OK(context, InitializeBase<StablehloData>());
    const auto& params = *reinterpret_cast<TfLiteStablehloReduceWindowParams*>(
        node->builtin_data);
    window_dimensions = params.window_dimensions;
    window_strides = params.window_strides;
    base_dilations = params.base_dilations;
    window_dilations = params.window_dilations;
    padding = params.padding;
    auto AllGtThanZero = [&](const int64_t* const attr) {
      return std::all_of(attr, attr + rank, [](int64_t d) { return d > 0; });
    };
    TF_LITE_ENSURE(context, AllGtThanZero(base_dilations));
    TF_LITE_ENSURE(context, AllGtThanZero(window_dimensions));
    TF_LITE_ENSURE(context, AllGtThanZero(window_strides));
    TF_LITE_ENSURE(context, AllGtThanZero(window_dilations));

    if (node->temporaries &&
        node->temporaries->size >= NodeData::kTempTensorCount) {
      TfLiteTensor* const dilated_tensor =
          GetTemporary(NodeData::kDilateOutput);
      TfLiteTensor* const padded_tensor = GetTemporary(NodeData::kPadOutput);
      TF_LITE_ENSURE(context, dilated_tensor != nullptr);
      TF_LITE_ENSURE(context, padded_tensor != nullptr);
      // When called in Prepare, these pointers are bogus because the tensors
      // have not been resized yet. This is ok in Eval.
      dilate_output = dilated_tensor->data.raw;
      pad_output = padded_tensor->data.raw;
    }
    return kTfLiteOk;
  }

  // Sets up the temporary and output tensors and the sub-ops to dilate, pad,
  // crop and reduce.
  //
  // This should be called during Prepare.
  TfLiteStatus Setup() {
    NodeData& node_data = *reinterpret_cast<NodeData*>(node->user_data);

    TfLiteIntArrayFree(node->temporaries);
    node->temporaries = TfLiteIntArrayCreate(NodeData::kTempTensorCount);
    for (int i = 0; i < NodeData::kTempTensorCount; ++i) {
      node->temporaries->data[i] = node_data.temporary_tensor_offset + i;
    }

    node_data.body = GetBodyFunction();

    node_data.dilate_ctx =
        dilate::DilateData(rank, input_dims, base_dilations, element_size);
    node_data.pad_ctx = pad::PadCropData(
        rank, node_data.dilate_ctx.output_shape, padding, element_size);
    node_data.reduce_window_ctx = reduce_window::ReduceWindowData(
        rank, node_data.pad_ctx.output_shape, window_dimensions, window_strides,
        window_dilations);

    TfLiteTensor* const dilated_tensor = GetTemporary(NodeData::kDilateOutput);
    TfLiteTensor* const padded_tensor = GetTemporary(NodeData::kPadOutput);
    TfLiteTensor* const output_tensor = GetOutput(context, node, kOutput);
    dilated_tensor->type = type;
    dilated_tensor->allocation_type = kTfLiteArenaRw;
    padded_tensor->type = type;
    padded_tensor->allocation_type = kTfLiteArenaRw;

    TF_LITE_ENSURE_OK(context, ResizeTensor(dilated_tensor,
                                            node_data.dilate_ctx.output_shape));
    TF_LITE_ENSURE_OK(
        context, ResizeTensor(padded_tensor, node_data.pad_ctx.output_shape));
    TF_LITE_ENSURE_OK(
        context,
        ResizeTensor(output_tensor, node_data.reduce_window_ctx.output_shape));
    return kTfLiteOk;
  }

  // Inspects the subgraph associated to the STABLEHLO_REDUCE_WINDOW node to
  // find out the reduction body.
  TfLiteReduceWindowFunction GetBodyFunction() {
    const TfLiteStablehloReduceWindowParams& params =
        *reinterpret_cast<TfLiteStablehloReduceWindowParams*>(
            node->builtin_data);
    const int body_subgraph_index = params.body_subgraph_index;
    const Subgraph& parent_subgraph =
        *reinterpret_cast<Subgraph*>(context->impl_);
    const std::vector<std::unique_ptr<Subgraph>>& subgraphs =
        *parent_subgraph.GetSubgraphs();
    if (body_subgraph_index >= subgraphs.size()) {
      TF_LITE_KERNEL_LOG(
          context, "Body subgraph not found for stablehlo.reduce_window: %d.",
          body_subgraph_index);
      return TfLiteReduceWindowFunctionUnsupported;
    }
    const Subgraph& body_subgraph = *subgraphs[body_subgraph_index];
    const std::vector<int>& execution_plan =
        body_subgraph.pre_delegation_execution_plan().empty()
            ? body_subgraph.execution_plan()
            : body_subgraph.pre_delegation_execution_plan();

    if (execution_plan.size() != 1) {
      TF_LITE_KERNEL_LOG(context,
                         "Only one kernel is allowed within "
                         "stablehlo.reduce_window body. (%zu) kernels found.\n",
                         execution_plan.size());
      return TfLiteReduceWindowFunctionUnsupported;
    }
    const int body_kernel_index = execution_plan[0];
    const TfLiteRegistration& body_kernel_registration =
        body_subgraph.node_and_registration(body_kernel_index)->second;
    switch (body_kernel_registration.builtin_code) {
      case kTfLiteBuiltinAdd:
      case kTfLiteBuiltinStablehloAdd:
        return TfLiteReduceWindowFunctionAdd;
      case kTfLiteBuiltinMul:
      case kTfLiteBuiltinStablehloMultiply:
        return TfLiteReduceWindowFunctionMul;
      case kTfLiteBuiltinMaximum:
      case kTfLiteBuiltinStablehloMaximum:
        return TfLiteReduceWindowFunctionMax;
      case kTfLiteBuiltinMinimum:
      case kTfLiteBuiltinStablehloMinimum:
        return TfLiteReduceWindowFunctionMin;
      case kTfLiteBuiltinLogicalAnd:
      case kTfLiteBuiltinStablehloAnd:
        return TfLiteReduceWindowFunctionAll;
      case kTfLiteBuiltinLogicalOr:
      case kTfLiteBuiltinStablehloOr:
        return TfLiteReduceWindowFunctionAny;
      default:
        TF_LITE_KERNEL_LOG(
            context, "%s:%d unsupported reduction body builtin code: %d.\n",
            __FILE__, __LINE__, body_kernel_registration.builtin_code);
        return TfLiteReduceWindowFunctionUnsupported;
    }
  }
};

// Specializes OpData for the REDUCE_WINDOW operation.
struct TFLiteData : public OpData {
  enum InputTensorId {
    kInput,
    kInitValue,
    kWindowShape,
    kWindowStrides,
    kWindowDilations,
    kNumInputTensors
  };
  enum OutputTensorId { kOutput, kNumOutputTensors };

  using OpData::OpData;

  TfLiteStatus Check() const {
    TF_LITE_ENSURE_EQ(context, NumInputs(node), kNumInputTensors);
    TF_LITE_ENSURE_EQ(context, NumOutputs(node), kNumOutputTensors);
    const TfLiteTensor* const input_tensor = GetInput(context, node, kInput);
    const TfLiteTensor* const init_value_tensor =
        GetInput(context, node, kInitValue);
    const TfLiteTensor* const window_dimensions_tensor =
        GetInput(context, node, kWindowShape);
    const TfLiteTensor* const window_strides_tensor =
        GetInput(context, node, kWindowStrides);
    const TfLiteTensor* const window_dilations_tensor =
        GetInput(context, node, kWindowDilations);
    const TfLiteTensor* const output_tensor = GetOutput(context, node, kOutput);
    TF_LITE_ENSURE(context, IsConstantTensor(window_dimensions_tensor));
    TF_LITE_ENSURE(context, IsConstantTensor(window_strides_tensor));
    TF_LITE_ENSURE(context, IsConstantTensor(window_dilations_tensor));
    TF_LITE_ENSURE_EQ(context, input_tensor->type, output_tensor->type);
    TF_LITE_ENSURE_EQ(context, input_tensor->type, init_value_tensor->type);
    TF_LITE_ENSURE_EQ(context, window_dimensions_tensor->type, kTfLiteInt64);
    TF_LITE_ENSURE_EQ(context, window_strides_tensor->type, kTfLiteInt64);
    TF_LITE_ENSURE_EQ(context, window_dilations_tensor->type, kTfLiteInt64);
    TF_LITE_ENSURE(context, input_tensor->dims != nullptr);
    TF_LITE_ENSURE(context, input_tensor->dims->size > 0);
    TF_LITE_ENSURE(context, input_tensor->dims->size <= kMaxReduceWindowRank);

    return kTfLiteOk;
  }

  TfLiteStatus Initialize() {
    TF_LITE_ENSURE_OK(context, InitializeBase<TFLiteData>());
    window_dimensions = reinterpret_cast<const int64_t*>(
        GetInput(context, node, kWindowShape)->data.data);
    window_strides = reinterpret_cast<const int64_t*>(
        GetInput(context, node, kWindowStrides)->data.data);
    base_dilations = dilate::kTFLiteDefaultBaseDilation;
    window_dilations = reinterpret_cast<const int64_t*>(
        GetInput(context, node, kWindowDilations)->data.data);
    padding = pad::kTFLiteDefaultPadding;
    return kTfLiteOk;
  }

  TfLiteStatus Setup() {
    NodeData& node_data = *reinterpret_cast<NodeData*>(node->user_data);
    const auto& params =
        *reinterpret_cast<TfLiteReduceWindowParams*>(node->builtin_data);
    node_data.body = params.reduce_function;

    node_data.dilate_ctx.skip = true;
    node_data.pad_ctx.skip = true;
    node_data.reduce_window_ctx = reduce_window::ReduceWindowData(
        rank, input_dims, window_dimensions, window_strides, window_dilations);

    TfLiteTensor* const output_tensor = GetOutput(context, node, kOutput);
    return context->ResizeTensor(
        context, output_tensor,
        BuildTfLiteArray<int32_t>(rank,
                                  node_data.reduce_window_ctx.output_shape)
            .release());
  }
};

// Applies the sub-ops that are needed to compute the whole
// [STABLEHLO_]REDUCE_WINDOW op.
//
// The ops that aren't needed are skipped.
template <class Op, class Type>
void PadCropReduceWindow(const OpData& op_ctx) {
  NodeData& node_data = *reinterpret_cast<NodeData*>(op_ctx.node->user_data);
  const char* input = op_ctx.input;
  const int64_t* input_shape = op_ctx.input_dims;

  if (!node_data.dilate_ctx.skip) {
    dilate::Dilate(node_data.dilate_ctx, input, op_ctx.init_value,
                   op_ctx.dilate_output);
    input = op_ctx.dilate_output;
    input_shape = node_data.dilate_ctx.output_shape;
  }

  if (!node_data.pad_ctx.skip) {
    pad::PadCrop(node_data.pad_ctx, input, op_ctx.init_value,
                 op_ctx.pad_output);
    input = op_ctx.pad_output;
    input_shape = node_data.pad_ctx.output_shape;
  }

  reduce_window::ReduceWindow<Op, Type>(
      node_data.reduce_window_ctx, reinterpret_cast<const Type*>(input),
      *reinterpret_cast<const Type*>(op_ctx.init_value),
      reinterpret_cast<Type*>(op_ctx.output));
}

// Dispatches to the template implementation according to the tensor type.
template <class Op>
TfLiteStatus DispatchReduceWindowType(OpData& ctx) {
#define REDUCE_WINDOW_TYPE_CASE(CPP_TYPE, TENSOR_TYPE) \
  case TENSOR_TYPE:                                    \
    PadCropReduceWindow<Op, CPP_TYPE>(ctx);            \
    break;
  switch (ctx.type) {
    REDUCE_WINDOW_TYPE_CASE(int8_t, kTfLiteBool);
    REDUCE_WINDOW_TYPE_CASE(int8_t, kTfLiteInt8);
    REDUCE_WINDOW_TYPE_CASE(int16_t, kTfLiteInt16);
    REDUCE_WINDOW_TYPE_CASE(int32_t, kTfLiteInt32);
    REDUCE_WINDOW_TYPE_CASE(int64_t, kTfLiteInt64);
    REDUCE_WINDOW_TYPE_CASE(uint8_t, kTfLiteUInt8);
    REDUCE_WINDOW_TYPE_CASE(float, kTfLiteFloat32);
    REDUCE_WINDOW_TYPE_CASE(double, kTfLiteFloat64);
    default:
      TF_LITE_KERNEL_LOG(
          ctx.context,
          "%s:%d unsupported kernel data type (TfliteType: %d a.k.a %s).",
          __FILE__, __LINE__, ctx.type, TfLiteTypeGetName(ctx.type));
      return kTfLiteError;
  }
#undef REDUCE_WINDOW_TYPE_CASE
  return kTfLiteOk;
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

// Dispatches to the template instanciation according to the reduction body.
TfLiteStatus DispatchReduceWindowBody(OpData& ctx) {
  const NodeData& node_data = *static_cast<NodeData*>(ctx.node->user_data);
  switch (node_data.body) {
    case TfLiteReduceWindowFunctionUnsupported:
      TF_LITE_KERNEL_LOG(ctx.context, "%s:%d unsupported reduction body.\n",
                         __FILE__, __LINE__);
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
  TF_LITE_KERNEL_LOG(ctx.context, "%s:%d unhandled reduction body case.\n",
                     __FILE__, __LINE__);
  return kTfLiteError;
}

// Initializes the node's user data when the STABLEHLO_REDUCE_WINDOW sematic is
// used.
void* StablehloInit(TfLiteContext* context, const char* options,
                    size_t options_len) {
  NodeData* node_data = new NodeData();
  context->AddTensors(context, NodeData::kTempTensorCount,
                      &node_data->temporary_tensor_offset);
  return node_data;
}

void* TFLiteInit(TfLiteContext* context, const char* options,
                 size_t options_len) {
  return new NodeData();
}

// Frees the node's user data when the STABLEHLO_REDUCE_WINDOW sematic is used.
void Free(TfLiteContext* context, void* node_data) {
  delete static_cast<NodeData*>(node_data);
}

template <class Semantic>
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  Semantic ctx(context, node);
  TF_LITE_ENSURE_OK(context, ctx.Check());
  TF_LITE_ENSURE_OK(context, ctx.Initialize());
  return ctx.Setup();
}

template <class Semantic>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  Semantic ctx(context, node);
  TF_LITE_ENSURE_OK(context, ctx.Initialize());
  // Too much cropping can lead to a negative dimension.
  //
  // This never happens with the REDUCE_WINDOW (TFLiteData) semantic but since
  // that op is deprecated we don't care about the extra check.
  NodeData& node_data = *reinterpret_cast<NodeData*>(node->user_data);
  TF_LITE_ENSURE_MSG(
      context, node_data.pad_ctx.skip || node_data.pad_ctx.output_size > 0,
      "The padding specification of stablehlo.reduce_window gives an empty "
      "tensor.");
  return DispatchReduceWindowBody(ctx);
}

}  // namespace
}  // namespace reduce_window_op

TfLiteRegistration* Register_STABLEHLO_REDUCE_WINDOW() {
  static TfLiteRegistration r = {
      /*.init=*/reduce_window_op::StablehloInit,
      /*.free=*/reduce_window_op::Free,
      /*.prepare=*/reduce_window_op::Prepare<reduce_window_op::StablehloData>,
      /*.invoke=*/reduce_window_op::Eval<reduce_window_op::StablehloData>};
  return &r;
}

TfLiteRegistration* Register_REDUCE_WINDOW() {
  static TfLiteRegistration r = {
      /*.init=*/reduce_window_op::TFLiteInit,
      /*.free=*/reduce_window_op::Free,
      /*.prepare=*/reduce_window_op::Prepare<reduce_window_op::TFLiteData>,
      /*.invoke=*/reduce_window_op::Eval<reduce_window_op::TFLiteData>};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
