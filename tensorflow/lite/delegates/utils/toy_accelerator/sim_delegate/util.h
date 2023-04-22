#ifndef TENSORFLOW_LITE_DELEGATES_UTILS_TOY_DELEGATE_TOY_DELEGATE_UTIL_H_
#define TENSORFLOW_LITE_DELEGATES_UTILS_TOY_DELEGATE_TOY_DELEGATE_UTIL_H_

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


struct OpData {
  // These fields are used in both the general 8-bit -> 8bit quantized path,
  // and the special 16-bit -> 16bit quantized path
  int input1_shift;
  int input2_shift;
  int32 output_activation_min;
  int32 output_activation_max;

  // These fields are used only in the general 8-bit -> 8bit quantized path
  int32 input1_multiplier;
  int32 input2_multiplier;
  int32 output_multiplier;
  int output_shift;
  int left_shift;
  int32 input1_offset;
  int32 input2_offset;
  int32 output_offset;

  // This parameter is used to indicate whether
  // parameter scale is power of two.
  // It is used in 16-bit -> 16-bit quantization.
  bool pot_scale_int16;
};


inline int32_t RoundingDivideByPOT(int32_t x, int exponent) {
  std::int32_t msk = (1 << exponent) - 1;
  std::int32_t sm = msk >> 1;
  std::int32_t val_3 = x >> exponent;

  std::int32_t temp_2 = x & msk;
  std::int32_t temp_3 = (x < 0) & 1;
  std::int32_t temp_4 = sm + temp_3;
  std::int32_t temp_5 = ((temp_2 > temp_4) & 1);
  std::int32_t result_32 = val_3 + temp_5;
}

inline std::int32_t SaturatingRoundingDoublingHighMul(std::int32_t a,
                                                      std::int32_t b) {
  bool overflow = a == b && a == std::numeric_limits<std::int32_t>::min();
  std::int64_t a_64(a);
  std::int64_t b_64(b);
  std::int64_t ab_64 = a_64 * b_64;
  std::int32_t nudge = ab_64 >= 0 ? (1 << 30) : (1 - (1 << 30));
  std::int32_t ab_x2_high32 =
      static_cast<std::int32_t>((ab_64 + nudge) / (1ll << 31));
  return overflow ? std::numeric_limits<std::int32_t>::max() : ab_x2_high32;
}

inline int32_t MultiplyByQuantizedMultiplierSmallerThanOneExp(
    int32_t x, int32_t quantized_multiplier, int left_shift) {
  return RoundingDivideByPOT(
      SaturatingRoundingDoublingHighMul(x, quantized_multiplier), -left_shift);
}


constexpr int kInputTensor1 = 0;
constexpr int kInputTensor2 = 1;
constexpr int kOutputTensor = 0;

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

#endif