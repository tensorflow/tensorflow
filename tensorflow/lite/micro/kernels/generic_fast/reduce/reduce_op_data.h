/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_GENERIC_FAST_REDUCE_OP_DATA_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_GENERIC_FAST_REDUCE_OP_DATA_H_

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace ops {
namespace micro {
namespace reduce {

struct OpData;

typedef TfLiteStatus (*EvalVariantFptr)(TfLiteContext* context, OpData* op_data,
                                        TfLiteReducerParams* params,
                                        const TfLiteEvalTensor* input,
                                        const TfLiteEvalTensor* axis,
                                        TfLiteEvalTensor* output);

#define EVAL_FUNC(name)                                      \
  TfLiteStatus name(TfLiteContext* context, OpData* op_data, \
                    TfLiteReducerParams* params,             \
                    const TfLiteEvalTensor* input,           \
                    const TfLiteEvalTensor* axis, TfLiteEvalTensor* output);

EVAL_FUNC(ReduceFloatKeepDims);
EVAL_FUNC(ReduceFloatChangeDims);
EVAL_FUNC(ReduceInt8KeepDims);
EVAL_FUNC(ReduceInt8ChangeDims);
EVAL_FUNC(ReduceInt8ChangeDimsAndQuant);
EVAL_FUNC(ReduceUInt8KeepDims);
EVAL_FUNC(ReduceUInt8ChangeDims);
EVAL_FUNC(ReduceUInt8ChangeDimsAndQuant);
EVAL_FUNC(ReduceMaxFloat);
EVAL_FUNC(ReduceMaxInt8);

#undef EVAL_FUNC

struct OpData {
  int32_t multiplier;
  int shift;
  int temp_buffer_idx;
  int resolved_axis_idx;
  int32_t* temp_buffer;
  int input_zp;
  float input_scale;
  int32_t mean_multiplier;
  int mean_shift;
  int output_zp;
  float output_scale;
  EvalVariantFptr eval_function;
};

}  // namespace reduce
}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_GENERIC_FAST_REDUCE_OP_DATA_H_
