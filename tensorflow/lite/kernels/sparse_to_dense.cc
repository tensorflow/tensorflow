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
#include <stdint.h>

#include <vector>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace sparse_to_dense {

constexpr int kIndicesTensor = 0;
constexpr int kOutputShapeTensor = 1;
constexpr int kValueInputTensor = 2;
constexpr int kDefaultValueTensor = 3;
constexpr int kOutputTensor = 0;

constexpr int kMaxDimensions = 4;

template <typename T>
TfLiteStatus Resize(TfLiteContext* context, const TfLiteTensor* output_shape,
                    TfLiteTensor* output) {
  const int output_dimensions = NumElements(output_shape);
  TfLiteIntArray* output_shape_array = TfLiteIntArrayCreate(output_dimensions);
  for (int i = 0; i < output_dimensions; ++i) {
    output_shape_array->data[i] = GetTensorData<T>(output_shape)[i];
  }

  return context->ResizeTensor(context, output, output_shape_array);
}

TfLiteStatus CheckDimensionsMatch(TfLiteContext* context,
                                  const TfLiteTensor* indices,
                                  const TfLiteTensor* output_shape,
                                  const TfLiteTensor* values) {
  switch (NumDimensions(indices)) {
    case 0:
    case 1: {
      if (NumDimensions(values) == 0) {
        TF_LITE_ENSURE_EQ(context, NumElements(indices), NumElements(values));
      }
      TF_LITE_ENSURE_EQ(context, NumElements(output_shape), 1);
      break;
    }
    case 2: {
      TF_LITE_ENSURE_EQ(context, SizeOfDimension(indices, 1),
                        NumElements(output_shape));
      if (NumDimensions(values) == 0)
        TF_LITE_ENSURE_EQ(context, SizeOfDimension(indices, 0),
                          NumElements(values));
      break;
    }
    default:
      context->ReportError(
          context, "Wrong indices dimensions %d, should be less than 3.",
          NumDimensions(indices));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

// Convert indices into a vector of 4-d vectors.
// TODO(renjieliu): Revisit here to improve the performance, since multiple
// allocations of std::vectors will be quite slow on phones.
template <typename T>
TfLiteStatus GetIndicesVector(TfLiteContext* context,
                              const TfLiteTensor* indices,
                              const int num_indices,
                              std::vector<std::vector<T>>* indices_vector) {
  // Note because TfLite will reverse the dimensions, so pad zeros upfront.
  switch (NumDimensions(indices)) {
    case 0:
    case 1: {
      const auto indices_data = GetTensorData<T>(indices);
      for (int i = 0; i < num_indices; ++i) {
        std::vector<T> index({0, 0, 0, indices_data[i]});
        indices_vector->push_back(index);
      }
      break;
    }
    case 2: {
      const int true_dimensions = SizeOfDimension(indices, 1);
      TF_LITE_ENSURE(context, true_dimensions <= kMaxDimensions);
      for (int i = 0; i < num_indices; ++i) {
        std::vector<T> index;
        index.reserve(kMaxDimensions);
        // Fill the index with 1 up to kMaxDimensions - true_dimensions to
        // satisfy the needs for 4-dimension index.
        for (int j = 0; j < kMaxDimensions - true_dimensions; ++j) {
          index.push_back(0);
        }
        for (int j = 0; j < true_dimensions; ++j) {
          index.push_back(GetTensorData<T>(indices)[i * true_dimensions + j]);
        }

        indices_vector->push_back(index);
      }
      break;
    }
    default:
      context->ReportError(context,
                           "Indices dimensions problem, got %d dimensions",
                           NumDimensions(indices));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus ResizeOutputShape(TfLiteContext* context,
                               const TfLiteTensor* output_shape,
                               TfLiteTensor* output) {
  if (output_shape->type == kTfLiteInt32) {
    return Resize<int32_t>(context, output_shape, output);
  } else if (output_shape->type == kTfLiteInt64) {
    return Resize<int64_t>(context, output_shape, output);
  } else {
    context->ReportError(context, "Dense shape type %d not supported.",
                         output_shape->type);
    return kTfLiteError;
  }
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 4);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* indices;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kIndicesTensor, &indices));
  const TfLiteTensor* output_shape;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, kOutputShapeTensor, &output_shape));
  const TfLiteTensor* values;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kValueInputTensor, &values));
  const TfLiteTensor* default_value;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kDefaultValueTensor,
                                          &default_value));

  // TODO(renjieliu): Handle validate_indices.

  // Indices can be 0-D, 1-D or 2-D.
  TF_LITE_ASSERT(NumDimensions(indices) >= 0);
  TF_LITE_ENSURE(context, NumDimensions(indices) < 3);
  TF_LITE_ASSERT(NumDimensions(output_shape) >= 0);
  TF_LITE_ENSURE_EQ(context, NumDimensions(output_shape), 1);
  // Values can be 0-D or 1-D.
  TF_LITE_ASSERT(NumDimensions(values) >= 0);
  TF_LITE_ENSURE(context, NumDimensions(values) < 2);

  TF_LITE_ENSURE_EQ(context, NumElements(default_value), 1);

  TF_LITE_ENSURE(
      context, indices->type == kTfLiteInt32 || indices->type == kTfLiteInt64);
  TF_LITE_ENSURE(context, output_shape->type == kTfLiteInt32 ||
                              output_shape->type == kTfLiteInt64);
  TF_LITE_ENSURE(context, values->type == kTfLiteInt32 ||
                              values->type == kTfLiteInt64 ||
                              values->type == kTfLiteInt8 ||
                              values->type == kTfLiteUInt8 ||
                              values->type == kTfLiteFloat32);
  TF_LITE_ENSURE_TYPES_EQ(context, values->type, default_value->type);

  // Ensure dimensions match.
  TF_LITE_ENSURE_OK(
      context, CheckDimensionsMatch(context, indices, output_shape, values));

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  output->type = values->type;
  TF_LITE_ENSURE_EQ(context, NumDimensions(output_shape), 1);

  if (!IsConstantOrPersistentTensor(output_shape)) {
    SetTensorToDynamic(output);
    return kTfLiteOk;
  }
  return ResizeOutputShape(context, output_shape, output);
}

template <typename T, typename TI>
TfLiteStatus SparseToDenseImpl(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* indices;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kIndicesTensor, &indices));
  const TfLiteTensor* output_shape;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, kOutputShapeTensor, &output_shape));
  const TfLiteTensor* values;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kValueInputTensor, &values));
  const TfLiteTensor* default_value;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kDefaultValueTensor,
                                          &default_value));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  if (IsDynamicTensor(output)) {
    TF_LITE_ENSURE_OK(context,
                      ResizeOutputShape(context, output_shape, output));
  }

  const int num_indices = SizeOfDimension(indices, 0);
  const bool value_is_scalar = NumDimensions(values) == 0;
  std::vector<std::vector<TI>> indices_vector;
  indices_vector.reserve(num_indices);
  TF_LITE_ENSURE_OK(context, GetIndicesVector<TI>(context, indices, num_indices,
                                                  &indices_vector));
  reference_ops::SparseToDense(indices_vector, GetTensorData<T>(values),
                               *GetTensorData<T>(default_value),
                               value_is_scalar, GetTensorShape(output),
                               GetTensorData<T>(output));

  return kTfLiteOk;
}

template <typename T>
TfLiteStatus EvalForIndexType(TfLiteContext* context, TfLiteNode* node,
                              const TfLiteTensor* indices) {
  switch (indices->type) {
    case kTfLiteInt32: {
      return SparseToDenseImpl<T, int32_t>(context, node);
    }
    case kTfLiteInt64: {
      return SparseToDenseImpl<T, int64_t>(context, node);
    }
    default:
      TF_LITE_KERNEL_LOG(
          context,
          "Indice type %s is currently not supported by sparse to dense.",
          TfLiteTypeGetName(indices->type));
      return kTfLiteError;
  }
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* indices;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kIndicesTensor, &indices));
  const TfLiteTensor* values;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kValueInputTensor, &values));

  switch (values->type) {
    case kTfLiteFloat32:
      return EvalForIndexType<float>(context, node, indices);
    case kTfLiteInt32:
      return EvalForIndexType<int32_t>(context, node, indices);
    case kTfLiteInt64:
      return EvalForIndexType<int64_t>(context, node, indices);
    case kTfLiteInt8:
      return EvalForIndexType<int8_t>(context, node, indices);
    case kTfLiteUInt8:
      return EvalForIndexType<uint8_t>(context, node, indices);
    default:
      TF_LITE_KERNEL_LOG(
          context,
          "Value type %s is currently not supported by sparse to dense.",
          TfLiteTypeGetName(values->type));
      return kTfLiteError;
  }
}

}  // namespace sparse_to_dense

TfLiteRegistration* Register_SPARSE_TO_DENSE() {
  static TfLiteRegistration r = {nullptr, nullptr, sparse_to_dense::Prepare,
                                 sparse_to_dense::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
