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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

/*
 * The circular buffer custom operator is used to implement strided streaming
 * convolutions on TFLite Micro.  Each time this operator is invoked, it checks
 * whether or not to run, based on a predetermined stride in time.  If the op
 * runs, it inserts the input into the end of the output buffer and shifts the
 * output values towards the start of the buffer.  It discards the oldest value
 * in the output buffer.
 *
 * Input: [<input N+1]
 * Before shifting:
 * Output: [<input 1>, <input 2>, <input ...>, <input N>]
 *
 * After shifting:
 * Output: [<input 2>, <input 3>, <input ...>, <input N+1>]
 *
 * We make some assumptions in this custom operator:
 * - Input shape must be [1, 1, 1, depth]
 * - Output shape must be [1, num_slots, 1, depth]
 * - Input and output types must match.
 * - Input and output quantization params must be identical.
 */
namespace tflite {
namespace ops {
namespace micro {
namespace circular_buffer {

namespace {

// The CircularBuffer op has one input and one output tensor.
constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

// TODO(b/149795762): Add this to TfLiteStatus enum.
constexpr int kTfLiteAbort = -9;

// These fields control the stride period of a strided streaming model. This op
// returns kTfLiteAbort until cycles_until_run-- is zero.  At this time,
// cycles_until_run is reset to cycles_max.
struct OpData {
  int cycles_until_run;
  int cycles_max;
};

// These constants represent constants specific to the music detect model.
// They exist until (b/132070898) is fixed.
constexpr int kMaxOpDataSize = 7;
int op_data_counter = 0;
OpData op_data_array[kMaxOpDataSize];

}  // namespace

void Free(TfLiteContext* context, void* buffer) { op_data_counter = 0; }

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  TF_LITE_ENSURE(context, input != nullptr);
  TF_LITE_ENSURE(context, output != nullptr);
  TF_LITE_ENSURE_EQ(context, 1, output->dims->data[0]);
  TF_LITE_ENSURE_EQ(context, 1, input->dims->data[0]);
  TF_LITE_ENSURE_EQ(context, 1, input->dims->data[1]);
  TF_LITE_ENSURE_EQ(context, 1, output->dims->data[2]);
  TF_LITE_ENSURE_EQ(context, 1, input->dims->data[2]);
  TF_LITE_ENSURE_EQ(context, output->dims->data[3], input->dims->data[3]);

  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);

  // The circular buffer custom operator currently only supports int8_t.
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteInt8);

  // TODO(b/132070898): Use statically slotted OpData structures until a
  // scratch memory API is ready.
  TFLITE_DCHECK_LE(op_data_counter, kMaxOpDataSize);
  OpData* op_data = &op_data_array[op_data_counter++];
  // The last circular buffer layer (length 5) simply accumulates outputs, and
  // does not run periodically.
  // TODO(b/150001379): Move this special case logic to the tflite flatbuffer.
  if (output->dims->data[1] == 5) {
    op_data->cycles_max = 1;
  } else {
    op_data->cycles_max = 2;
  }
  op_data->cycles_until_run = op_data->cycles_max;
  node->user_data = op_data;

  return kTfLiteOk;
}

// Shifts buffer over by the output depth, and write new input to end of buffer.
// num_slots is the number of samples stored in the output buffer.
// depth is the size of each sample.
void EvalInt8(const int8_t* input, int num_slots, int depth, int8_t* output) {
  memmove(output, &output[depth], (num_slots - 1) * depth);
  memcpy(&output[(num_slots - 1) * depth], input, depth);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  int num_slots = output->dims->data[1];
  int depth = output->dims->data[3];

  if (input->type == kTfLiteInt8) {
    EvalInt8(tflite::micro::GetTensorData<int8_t>(input), num_slots, depth,
             tflite::micro::GetTensorData<int8_t>(output));
  } else {
    TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                       TfLiteTypeGetName(input->type), input->type);
    return kTfLiteError;
  }

  if (--data->cycles_until_run != 0) {
    // Signal the interpreter to end current run if the delay before op invoke
    // has not been reached.
    // TODO(b/149795762): Add kTfLiteAbort to TfLiteStatus enum.
    return static_cast<TfLiteStatus>(kTfLiteAbort);
  }

  // If prepare is ever called more than one time (for example, when testing the
  // ambient model, the interpreter is created a few times), this op data
  // counter needs to be reset so that future instances do not overrun this op
  // data array.
  op_data_counter = 0;

  data->cycles_until_run = data->cycles_max;

  return kTfLiteOk;
}

}  // namespace circular_buffer

TfLiteRegistration* Register_CIRCULAR_BUFFER() {
  static TfLiteRegistration r = {/*init=*/nullptr,
                                 /*free=*/circular_buffer::Free,
                                 /*prepare=*/circular_buffer::Prepare,
                                 /*invoke=*/circular_buffer::Eval,
                                 /*profiling_string=*/nullptr,
                                 /*builtin_code=*/0,
                                 /*custom_name=*/nullptr,
                                 /*version=*/0};
  return &r;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
