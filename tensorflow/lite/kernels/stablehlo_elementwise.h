/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_STABLEHLO_ELEMENTWISE_H_
#define TENSORFLOW_LITE_KERNELS_STABLEHLO_ELEMENTWISE_H_

#include <cstdint>
#include <vector>

#include "Eigen/Core"  // from @eigen_archive
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {

constexpr int kInputTensor1 = 0;
constexpr int kInputTensor2 = 1;
constexpr int kOutputTensor = 0;

// Indicates the type of the computation performed by the element-wise op.
enum class ComputationType { kAdd, kSub, kMax, kMin, kMul, kAnd };

TfLiteStatus ElementwisePrepare(TfLiteContext* context, TfLiteNode* node);

template <typename DataType, ComputationType computation_type>
inline DataType ApplyComputation(DataType input1, DataType input2) {
  if (computation_type == ComputationType::kAnd) {
    if constexpr (std::is_integral<DataType>::value) {
      return input1 & input2;
    } else if constexpr (std::is_same<DataType, bool>::value) {
      return input1 && input2;
    }
  } else if (computation_type == ComputationType::kAdd) {
    return input1 + input2;
  } else if (computation_type == ComputationType::kSub) {
    return input1 - input2;
  } else if (computation_type == ComputationType::kMax) {
    return std::max(input1, input2);
  } else if (computation_type == ComputationType::kMin) {
    return std::min(input1, input2);
  } else if (computation_type == ComputationType::kMul) {
    return input1 * input2;
  }
  TFL_UNREACHABLE();
}

// Evaluates this node given the type of the elements in the output_tensor
// and the type of the elements in the input/updates vector.
template <ComputationType computation_type, typename DataType>
TfLiteStatus EvalWithType(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input_tensor1;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor1, &input_tensor1));
  const DataType* input_data1 = GetTensorData<DataType>(input_tensor1);

  const TfLiteTensor* input_tensor2;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor2, &input_tensor2));
  const DataType* input_data2 = GetTensorData<DataType>(input_tensor2);

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  DataType* output_data = GetTensorData<DataType>(output);

  const int64_t num_elements = NumElements(input_tensor1);
  for (int64_t i = 0; i < num_elements; ++i) {
    output_data[i] = ApplyComputation<DataType, computation_type>(
        input_data1[i], input_data2[i]);
  }

  return TfLiteStatus::kTfLiteOk;
}

template <ComputationType computation_type>
TfLiteStatus ElementwiseEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input_tensor1;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor1, &input_tensor1));

  TfLiteType data_type = input_tensor1->type;

  if constexpr (computation_type == ComputationType::kAnd) {
    if (data_type == kTfLiteFloat16 || data_type == kTfLiteFloat32 ||
        data_type == kTfLiteFloat64) {
      TF_LITE_KERNEL_LOG(
          context,
          "(Data Type: %s) is not supported for bitwise/logical AND.\n",
          TfLiteTypeGetName(data_type));
      return kTfLiteError;
    }
  }

  switch (data_type) {
    case kTfLiteFloat16:
      return EvalWithType<computation_type, Eigen::half>(context, node);
    case kTfLiteFloat32:
      return EvalWithType<computation_type, float>(context, node);
    case kTfLiteFloat64:
      return EvalWithType<computation_type, double>(context, node);
    case kTfLiteInt8:
      return EvalWithType<computation_type, int8_t>(context, node);
    case kTfLiteInt16:
      return EvalWithType<computation_type, int16_t>(context, node);
    case kTfLiteInt32:
      return EvalWithType<computation_type, int32_t>(context, node);
    case kTfLiteInt64:
      return EvalWithType<computation_type, int64_t>(context, node);
    case kTfLiteUInt8:
      return EvalWithType<computation_type, uint8_t>(context, node);
    case kTfLiteUInt16:
      return EvalWithType<computation_type, uint16_t>(context, node);
    case kTfLiteUInt32:
      return EvalWithType<computation_type, uint32_t>(context, node);
    case kTfLiteUInt64:
      return EvalWithType<computation_type, uint64_t>(context, node);
    case kTfLiteBool:
      return EvalWithType<computation_type, bool>(context, node);
    default:
      TF_LITE_KERNEL_LOG(context, "(Data Type: %s) currently not supported.\n",
                         TfLiteTypeGetName(data_type));
      return TfLiteStatus::kTfLiteError;
  }
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_STABLEHLO_ELEMENTWISE_H_
