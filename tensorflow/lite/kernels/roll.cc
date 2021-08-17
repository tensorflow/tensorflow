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

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include <cstring>
#include <vector>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace custom {
namespace roll {
namespace {

// A helper function to extract int32_t or int64_t tensor data.
std::vector<int32_t> ExtractIntegerVector(const TfLiteTensor* t) {
  TFLITE_DCHECK(t->type == kTfLiteInt32 || t->type == kTfLiteInt64);
  const RuntimeShape& shape = GetTensorShape(t);
  std::vector<int32_t> result(shape.FlatSize());
  if (t->type == kTfLiteInt32) {
    memcpy(result.data(), t->data.raw_const, t->bytes);
  } else {
    const int64_t* data = GetTensorData<int64_t>(t);
    for (int i = 0; i < result.size(); ++i) {
      result[i] = static_cast<int32_t>(data[i]);
    }
  }
  return result;
}

template <typename T>
inline void Pool(const std::vector<int32_t>& shift_map,
                 const RuntimeShape& shape, const TfLiteTensor* input,
                 TfLiteTensor* cache, TfLiteTensor* output) {
  int stride = 1, outer_size, next_stride;
  bool in_place_rolling = false;
  for (int i = shift_map.size() - 1; i >= 0; --i, stride = next_stride) {
    next_stride = stride * shape.Dims(i);
    if (shift_map[i] == 0) continue;

    TFLITE_DCHECK_EQ(shape.FlatSize() % next_stride, 0);
    outer_size = shape.FlatSize() / next_stride;
    const TfLiteTensor* source = input;
    if (in_place_rolling) {
      SequentialTensorWriter<T> writer(output, cache);
      writer.WriteN(0, shape.FlatSize());
      source = cache;
    }
    SequentialTensorWriter<T> writer(source, output);
    for (int j = 0; j < outer_size; ++j) {
      // Copies the first stride.
      const int begin_1 =
          j * next_stride + (shape.Dims(i) - shift_map[i]) * stride;
      const int size_1 = shift_map[i] * stride;
      writer.WriteN(begin_1, size_1);
      // Copies the second stride.
      const int begin_2 = j * next_stride;
      const int size_2 = (shape.Dims(i) - shift_map[i]) * stride;
      writer.WriteN(begin_2, size_2);
    }
    in_place_rolling = true;
  }

  // Copies input to output if no rolling is needed.
  if (!in_place_rolling) {
    SequentialTensorWriter<T> writer(input, output);
    writer.WriteN(0, shape.FlatSize());
    return;
  }
}

}  // namespace

constexpr int kInputTensor = 0;
constexpr int kShiftTensor = 1;
constexpr int kAxisTensor = 2;
constexpr int kOutputTensor = 0;
constexpr int kTensorNotAllocated = -1;

struct OpData {
  // A temporary tensor to store intermediate output data when doing in-place
  // rolling.
  int cache_tensor_id = kTensorNotAllocated;
  int32_t cache_index = kTensorNotAllocated;
  bool need_cache = false;
};

TfLiteStatus AllocateTemporaryTensorsIfRequired(TfLiteContext* context,
                                                TfLiteNode* node,
                                                OpData* opdata,
                                                const TfLiteTensor* input,
                                                const TfLiteTensor* shift) {
  int temporaries_count = 0;
  opdata->need_cache = (NumElements(shift) > 1);
  if (opdata->need_cache) {
    if (opdata->cache_tensor_id == kTensorNotAllocated) {
      TF_LITE_ENSURE_OK(
          context, context->AddTensors(context, 1, &opdata->cache_tensor_id));
    }
    opdata->cache_index = temporaries_count++;
  }

  TfLiteIntArrayFree(node->temporaries);
  node->temporaries = TfLiteIntArrayCreate(temporaries_count);

  if (opdata->need_cache) {
    node->temporaries->data[opdata->cache_index] = opdata->cache_tensor_id;
    TfLiteTensor* cache;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, opdata->cache_index, &cache));
    cache->type = input->type;
    cache->allocation_type = kTfLiteArenaRw;
    TfLiteIntArray* cache_shape = TfLiteIntArrayCopy(input->dims);
    TF_LITE_ENSURE_OK(context,
                      context->ResizeTensor(context, cache, cache_shape));
  }
  return kTfLiteOk;
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* opdata = new OpData;
  return opdata;
}

void Free(TfLiteContext* context, void* buffer) {
  delete static_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  OpData* opdata = reinterpret_cast<OpData*>(node->user_data);
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  const TfLiteTensor* shift;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kShiftTensor, &shift));
  const TfLiteTensor* axis;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kAxisTensor, &axis));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  // Check tensor type.
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  TF_LITE_ENSURE(
      context, (shift->type == kTfLiteInt32) || (shift->type == kTfLiteInt64));
  TF_LITE_ENSURE(context,
                 (axis->type == kTfLiteInt32) || (axis->type == kTfLiteInt64));

  // Make sure shift and axis are scalars or 1-D tensors.
  TF_LITE_ENSURE(context,
                 (NumDimensions(shift) == 0) || (NumDimensions(shift) == 1));
  TF_LITE_ENSURE(context,
                 (NumDimensions(shift) == 0) || (NumDimensions(shift) == 1));
  TF_LITE_ENSURE_EQ(context, NumElements(shift), NumElements(axis));

  TF_LITE_ENSURE_OK(context, AllocateTemporaryTensorsIfRequired(
                                 context, node, opdata, input, shift));

  // Output shape always equals to input shape.
  TfLiteIntArray* output_shape = TfLiteIntArrayCopy(input->dims);
  return context->ResizeTensor(context, output, output_shape);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  OpData* opdata = reinterpret_cast<OpData*>(node->user_data);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  const TfLiteTensor* shift;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kShiftTensor, &shift));
  const TfLiteTensor* axis;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kAxisTensor, &axis));

  TfLiteTensor* cache = GetTemporary(context, node, opdata->cache_index);
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  // Extract the shift and axis information.
  std::vector<int32_t> shift_data = ExtractIntegerVector(shift);
  std::vector<int32_t> axis_data = ExtractIntegerVector(axis);

  // Maps from index as axis to its corresponding shift value.
  const int input_rank = NumDimensions(input);
  std::vector<int32_t> shift_map(input_rank, 0);

  // Make sure axis is in range [0, rank(input)).
  for (int i = 0; i < axis_data.size(); ++i) {
    int32_t axis_i = axis_data[i];
    if (axis_i < 0) axis_i += input_rank;
    shift_map[axis_i] += shift_data[i];
  }

  // Make sure shift is range [0, rank(input)).
  for (int i = 0; i < input_rank; ++i) {
    const int32_t input_dims_i = SizeOfDimension(input, i);
    int32_t shift_i = shift_map[i] % input_dims_i;
    if (shift_i < 0) shift_i += input_dims_i;
    shift_map[i] = shift_i;
  }

#define TF_LITE_ROLL(type) \
  Pool<type>(shift_map, GetTensorShape(input), input, cache, output);

  // The type itself doesn't matter, we only care about type size.
  switch (input->type) {
    case kTfLiteFloat32:
      TF_LITE_ROLL(float);
      break;
    case kTfLiteInt32:
      TF_LITE_ROLL(int32_t);
      break;
    case kTfLiteInt64:
      TF_LITE_ROLL(int64_t);
      break;
    case kTfLiteInt8:
      TF_LITE_ROLL(int8_t);
      break;
    case kTfLiteInt16:
      TF_LITE_ROLL(int16_t);
      break;
    case kTfLiteUInt8:
      TF_LITE_ROLL(uint8_t);
      break;
    case kTfLiteBool:
      TF_LITE_ROLL(bool);
      break;
    case kTfLiteString:
      TF_LITE_ROLL(string);
      break;
    default:
      TF_LITE_KERNEL_LOG(
          context, "Type %d is currently not supported by Slice.", input->type);
      return kTfLiteError;
  }
#undef TF_LITE_ROLL
  return kTfLiteOk;
}
}  // namespace roll

TfLiteRegistration* Register_ROLL() {
  static TfLiteRegistration r = {roll::Init, roll::Free, roll::Prepare,
                                 roll::Eval};
  return &r;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
