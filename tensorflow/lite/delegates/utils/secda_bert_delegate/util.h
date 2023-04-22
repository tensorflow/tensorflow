#ifndef TENSORFLOW_LITE_DELEGATES_UTILS_BERT_DELEGATE_BERT_DELEGATE_UTIL_H_
#define TENSORFLOW_LITE_DELEGATES_UTILS_BERT_DELEGATE_BERT_DELEGATE_UTIL_H_

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/util.h"

#include <cassert>

using namespace std;

// This file has four implementations of FullyConnected
enum KernelType {
  kReference,
  kGenericOptimized,
  kLegacyPie,  // Legacy path used by the PIE team and related clients.
};

const int kTensorNotAllocated = -1;
static constexpr size_t kMaxIm2colBufferSizeMobile = 1024 * 1024 * 1024;  // 1GB

struct OpData {
  // The scaling factor from input to output (aka the 'real multiplier') can
  // be represented as a fixed point multiplier plus a left shift.
  int32_t output_multiplier;
  int output_shift;
  // The range of the fused activation layer. For example for kNone and
  // uint8_t these would be 0 and 255.
  int32_t output_activation_min;
  int32_t output_activation_max;
  // The index of the temporary tensor where the quantized inputs are cached.
  int scratch_tensor_index;
  bool compute_row_sums = false;
  // Only used for sparse hybrid fully connected kernels.
  bool ledger_initialized;
};
constexpr int kInputTensor = 0;
constexpr int kWeightsTensor = 1;
constexpr int kBiasTensor = 2;
constexpr int kOutputTensor = 0;
constexpr int kShuffledInputWorkspaceTensor = 1;

inline TfLiteTensor* GetTensorAtIndex(const TfLiteContext* context,
                                      int tensor_index) {
  return &context->tensors[tensor_index];
}

inline TfLiteStatus GetMutableInputSafe(const TfLiteContext* context,
                                        int tensor_index,
                                        const TfLiteTensor** tensor) {
  *tensor = GetTensorAtIndex(context, tensor_index);
  return kTfLiteOk;
}

TfLiteStatus GetInputSafe(const TfLiteContext* context, int tensor_index,
                          const TfLiteTensor** tensor) {
  return GetMutableInputSafe(context, tensor_index, tensor);
}

TfLiteStatus GetOutputSafe(const TfLiteContext* context, int tensor_index,
                           TfLiteTensor** tensor) {
  *tensor = GetTensorAtIndex(context, tensor_index);
  return kTfLiteOk;
}

int Generic_Quantised_Multiplier(int x, int qm, int shift) {
#define MAX32 2147483647
#define MIN32 -2147483648

  int nshift = shift;
  int total_shift = 31 - shift;
  int64_t x_64 = x;
  int64_t quantized_multiplier_64(qm);
  int64_t one = 1;
  int64_t round = one << (total_shift - 1);  // ALU ADD + ALU SHLI
  int64_t result = x_64 * quantized_multiplier_64 + round;  // ALU ADD + ALU MUL
  result = result >> total_shift;                           // ALU SHRI
  int nresult = result;
  if (result > MAX32) result = MAX32;  // ALU MIN
  if (result < MIN32) result = MIN32;  // ALU MAX
  return static_cast<std::int32_t>(result);

#undef MAX32
#undef MIN32
}

void precal_sums(int8_t* data, int width, int depth, vector<int>& sums) {
  int w = ((width + 4 - 1) - ((width + 4 - 1) % 4));
  int d = ((depth + 16 - 1) - ((depth + 16 - 1) % 16));
  int max = width * depth;
  for (int i = 0; i < w / 4; i++) {
    int s0 = 0;
    int s1 = 0;
    int s2 = 0;
    int s3 = 0;

    for (int j = 0; j < d; j++) {
      if (j < depth) {
        int8_t w0 =
            (i * (depth * 4) + j >= max) ? 0 : data[i * (depth * 4) + j];
        int8_t w1 = (i * (depth * 4) + j + depth * 1 >= max)
                        ? 0
                        : data[i * (depth * 4) + j + depth * 1];
        int8_t w2 = (i * (depth * 4) + j + depth * 2 >= max)
                        ? 0
                        : data[i * (depth * 4) + j + depth * 2];
        int8_t w3 = (i * (depth * 4) + j + depth * 3 >= max)
                        ? 0
                        : data[i * (depth * 4) + j + depth * 3];
        int8_t g_data[] = {w3, w2, w1, w0};
        s0 += w0;
        s1 += w1;
        s2 += w2;
        s3 += w3;
      }
    }
    sums.push_back(s0);
    sums.push_back(s1);
    sums.push_back(s2);
    sums.push_back(s3);
  }
}

static TfLiteStatus AllocateTemporaryTensorsIfRequired(
    TfLiteContext* context, TfLiteNode* node, bool req_temp_out,
    int temp_out_tid, int& temp_out_id, int input_tid, int filter_tid) {

  TF_LITE_ENSURE(context, node->inputs->size >= 2);
  const TfLiteTensor* input;
  const TfLiteTensor* filter;

  GetInputSafe(context, input_tid, &input);
  GetInputSafe(context, filter_tid, &filter);
  int temporaries_count = node->temporaries->size;

  if (req_temp_out) {
    temp_out_id = temporaries_count;
    if (temp_out_tid == kTensorNotAllocated) {
      context->AddTensors(context, 1, &temp_out_tid);
    }
    ++temporaries_count;
  }

  auto temp_array = TfLiteIntArrayCreate(temporaries_count);
  for (int i = 0; i < node->temporaries->size; i++)
    temp_array->data[i] = node->temporaries->data[i];

  TfLiteIntArrayFree(node->temporaries);
  node->temporaries = temp_array;

  return kTfLiteOk;
}

// int Generic_Quantised_Multiplier(int x, int qm, int shift) {
//   int total_shift = 31 - shift;
//   std::int64_t x_64(x);
//   std::int64_t quantized_multiplier_64(qm);
//   std::int64_t round = (int64_t)1 << (total_shift - 1);
//   int64_t result = x_64 * quantized_multiplier_64 + round;
//   result = result >> total_shift;
//   return static_cast<std::int32_t>(result);
// }

#endif