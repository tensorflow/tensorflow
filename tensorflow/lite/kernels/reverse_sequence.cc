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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace reverse_sequence {
namespace {

constexpr int kInputTensor = 0;
constexpr int kSeqLengthsTensor = 1;
constexpr int kOutputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* seq_lengths = GetInput(context, node, kSeqLengthsTensor);
  TF_LITE_ENSURE_EQ(context, NumDimensions(seq_lengths), 1);

  if (input->type != kTfLiteInt32 && input->type != kTfLiteFloat32 &&
      input->type != kTfLiteUInt8 && input->type != kTfLiteInt16 &&
      input->type != kTfLiteInt64) {
    context->ReportError(context,
                         "Type '%s' is not supported by reverse_sequence.",
                         TfLiteTypeGetName(input->type));
    return kTfLiteError;
  }

  if (seq_lengths->type != kTfLiteInt32 && seq_lengths->type != kTfLiteInt64) {
    context->ReportError(
        context, "Seq_lengths type '%s' is not supported by reverse_sequence.",
        TfLiteTypeGetName(seq_lengths->type));
    return kTfLiteError;
  }

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  TfLiteIntArray* output_shape = TfLiteIntArrayCopy(input->dims);
  TF_LITE_ENSURE_EQ(context, output->type, input->type);

  return context->ResizeTensor(context, output, output_shape);
}

template <typename T, typename TS>
TfLiteStatus ReverseSequenceImpl(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* seq_lengths_tensor =
      GetInput(context, node, kSeqLengthsTensor);
  const TS* seq_lengths = GetTensorData<TS>(seq_lengths_tensor);

  auto* params =
      reinterpret_cast<TfLiteReverseSequenceParams*>(node->builtin_data);
  int seq_dim = params->seq_dim;
  int batch_dim = params->batch_dim;

  TF_LITE_ENSURE(context, seq_dim >= 0);
  TF_LITE_ENSURE(context, batch_dim >= 0);
  TF_LITE_ENSURE(context, seq_dim != batch_dim);
  TF_LITE_ENSURE(context, seq_dim < NumDimensions(input));
  TF_LITE_ENSURE(context, batch_dim < NumDimensions(input));
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(seq_lengths_tensor, 0),
                    SizeOfDimension(input, batch_dim));
  for (int i = 0; i < NumDimensions(seq_lengths_tensor); ++i) {
    TF_LITE_ENSURE(context, seq_lengths[i] <= SizeOfDimension(input, seq_dim));
  }

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  reference_ops::ReverseSequence<T, TS>(
      seq_lengths, seq_dim, batch_dim, GetTensorShape(input),
      GetTensorData<T>(input), GetTensorShape(output),
      GetTensorData<T>(output));

  return kTfLiteOk;
}

template <typename T>
TfLiteStatus ReverseSequenceHelper(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* seq_lengths_tensor =
      GetInput(context, node, kSeqLengthsTensor);
  switch (seq_lengths_tensor->type) {
    case kTfLiteInt32: {
      return ReverseSequenceImpl<T, int32_t>(context, node);
    }
    case kTfLiteInt64: {
      return ReverseSequenceImpl<T, int64_t>(context, node);
    }
    default: {
      context->ReportError(
          context,
          "Seq_lengths type '%s' is not supported by reverse_sequence.",
          TfLiteTypeGetName(seq_lengths_tensor->type));
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  switch (output->type) {
    case kTfLiteFloat32: {
      return ReverseSequenceHelper<float>(context, node);
    }
    case kTfLiteUInt8: {
      return ReverseSequenceHelper<uint8_t>(context, node);
    }
    case kTfLiteInt16: {
      return ReverseSequenceHelper<int16_t>(context, node);
    }
    case kTfLiteInt32: {
      return ReverseSequenceHelper<int32_t>(context, node);
    }
    case kTfLiteInt64: {
      return ReverseSequenceHelper<int64_t>(context, node);
    }
    default: {
      context->ReportError(context,
                           "Type '%s' is not supported by reverse_sequence.",
                           TfLiteTypeGetName(output->type));
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}  // namespace

}  // namespace
}  // namespace reverse_sequence

TfLiteRegistration* Register_REVERSE_SEQUENCE() {
  static TfLiteRegistration r = {nullptr, nullptr, reverse_sequence::Prepare,
                                 reverse_sequence::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
