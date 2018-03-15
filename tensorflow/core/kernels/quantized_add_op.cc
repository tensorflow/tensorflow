/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// Implements a quantized eight-bit version of the matmul operation.

#define EIGEN_USE_THREADS

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#define USE_NEON
#define QUANTIZED_ADD_USE_NEON
#include <arm_neon.h>
#endif

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/meta_support.h"
#include "tensorflow/core/kernels/quantization_utils.h"
#include "tensorflow/core/lib/core/casts.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/bcast.h"

// There are implementations for three broadcast patterns for add:
//  - Scalar * Array
//  - Array * Array
//  - Array * Shorter Array (repeated to match first)
//
// These handle a lot of common broadcast patterns, and we have NEON SIMD
// versions to accelerate performance on ARM platforms.

namespace tensorflow {
namespace {

template <class T, class Toutput>
void ScalarAddition(OpKernelContext* context, const T* full_input,
                    float full_input_min, float full_input_max,
                    int64 num_elements, T scalar_input, float scalar_input_min,
                    float scalar_input_max, float output_min, float output_max,
                    Toutput* output) {
  const Toutput scalar_in_output_range = RequantizeInNewRange<T, Toutput>(
      scalar_input, scalar_input_min, scalar_input_max, output_min, output_max);
  for (int i = 0; i < num_elements; ++i) {
    const Toutput full_input_in_output_range = RequantizeInNewRange<T, Toutput>(
        full_input[i], full_input_min, full_input_max, output_min, output_max);
    output[i] = full_input_in_output_range + scalar_in_output_range;
  }
}

#ifdef QUANTIZED_ADD_USE_NEON

template <>
void ScalarAddition(OpKernelContext* context, const quint8* full_input,
                    float full_input_min, float full_input_max,
                    int64 num_elements, quint8 scalar_input,
                    float scalar_input_min, float scalar_input_max,
                    float output_min, float output_max, qint32* output) {
  const int32 scalar_in_output_range = RequantizeInNewRange<quint8, qint32>(
      scalar_input, scalar_input_min, scalar_input_max, output_min, output_max);

  const float input_0_float =
      QuantizedToFloat<quint8>(0, full_input_min, full_input_max);
  const float input_1_float =
      QuantizedToFloat<quint8>(1, full_input_min, full_input_max);
  const int64 input_0_int64 =
      FloatToQuantizedUnclamped<qint32>(input_0_float, output_min, output_max);
  const int64 input_1_int64 =
      FloatToQuantizedUnclamped<qint32>(input_1_float, output_min, output_max);
  const int32 input_mult_int32 = input_1_int64 - input_0_int64;

  const int64 lowest_quantized =
      static_cast<int64>(Eigen::NumTraits<qint32>::lowest());
  const int64 highest_quantized =
      static_cast<int64>(Eigen::NumTraits<qint32>::highest());

  const int64x2_t input_0_64x2 = vmovq_n_s64(input_0_int64);
  const int32x2_t input_mult_32x2 = vmov_n_s32(input_mult_int32);
  const int32x4_t scalar_in_output_range_32x4 =
      vmovq_n_s32(scalar_in_output_range);
  int64 i = 0;
  for (; i < (num_elements - 7); i += 8) {
    const uint8* full_input_ptr = &(full_input->value) + i;
    const std::array<int32x4_t, 2> output_value =
        Requantize8x8To32Neon(full_input_ptr, input_0_64x2, input_mult_32x2);
    const int32x4_t result_low_32x4 =
        vaddq_s32(output_value[0], scalar_in_output_range_32x4);
    const int32x4_t result_high_32x4 =
        vaddq_s32(output_value[1], scalar_in_output_range_32x4);
    int32* output_ptr = &(output->value) + i;
    vst1q_s32(output_ptr + 0, result_low_32x4);
    vst1q_s32(output_ptr + 4, result_high_32x4);
  }
  for (; i < num_elements; ++i) {
    const int64 full_input_value = static_cast<int64>(full_input[i]);
    int64 full_input_in_output_range_64 =
        input_0_int64 + (full_input_value * input_mult_int32);
    full_input_in_output_range_64 =
        std::max(full_input_in_output_range_64, lowest_quantized);
    full_input_in_output_range_64 =
        std::min(full_input_in_output_range_64, highest_quantized);
    const int32 full_input_in_output_range =
        static_cast<int32>(full_input_in_output_range_64);
    output[i] = full_input_in_output_range + scalar_in_output_range;
  }
}

#else  // QUANTIZED_ADD_USE_NEON

template <>
void ScalarAddition(OpKernelContext* context, const quint8* full_input,
                    float full_input_min, float full_input_max,
                    int64 num_elements, quint8 scalar_input,
                    float scalar_input_min, float scalar_input_max,
                    float output_min, float output_max, qint32* output) {
  const int32 scalar_in_output_range = RequantizeInNewRange<quint8, qint32>(
      scalar_input, scalar_input_min, scalar_input_max, output_min, output_max);

  const float input_0_float =
      QuantizedToFloat<quint8>(0, full_input_min, full_input_max);
  const float input_1_float =
      QuantizedToFloat<quint8>(1, full_input_min, full_input_max);
  const int64 input_0_int64 =
      FloatToQuantizedUnclamped<qint32>(input_0_float, output_min, output_max);
  const int64 input_1_int64 =
      FloatToQuantizedUnclamped<qint32>(input_1_float, output_min, output_max);
  const int32 input_mult_int32 = input_1_int64 - input_0_int64;

  const int64 lowest_quantized =
      static_cast<int64>(Eigen::NumTraits<qint32>::lowest());
  const int64 highest_quantized =
      static_cast<int64>(Eigen::NumTraits<qint32>::highest());

  for (int i = 0; i < num_elements; ++i) {
    const int64 full_input_value = static_cast<int64>(full_input[i]);
    int64 full_input_in_output_range_64 =
        input_0_int64 + (full_input_value * input_mult_int32);
    full_input_in_output_range_64 =
        std::max(full_input_in_output_range_64, lowest_quantized);
    full_input_in_output_range_64 =
        std::min(full_input_in_output_range_64, highest_quantized);
    const int32 full_input_in_output_range =
        static_cast<int32>(full_input_in_output_range_64);
    output[i] = full_input_in_output_range + scalar_in_output_range;
  }
}

#endif  // QUANTIZED_ADD_USE_NEON

template <class T, class Toutput>
void VectorAddition(OpKernelContext* context, const T* x_data, float min_x,
                    float max_x, const T* y_data, float min_y, float max_y,
                    int64 num_elements, float output_min, float output_max,
                    Toutput* output) {
  for (int i = 0; i < num_elements; ++i) {
    const Toutput x_in_output_range = RequantizeInNewRange<T, Toutput>(
        x_data[i], min_x, max_x, output_min, output_max);
    const Toutput y_in_output_range = RequantizeInNewRange<T, Toutput>(
        y_data[i], min_y, max_y, output_min, output_max);
    output[i] = x_in_output_range + y_in_output_range;
  }
}

#ifdef QUANTIZED_ADD_USE_NEON

template <>
void VectorAddition(OpKernelContext* context, const quint8* x_data, float min_x,
                    float max_x, const quint8* y_data, float min_y, float max_y,
                    int64 num_elements, float output_min, float output_max,
                    qint32* output) {
  const float x_0_float = QuantizedToFloat<quint8>(0, min_x, max_x);
  const float x_1_float = QuantizedToFloat<quint8>(1, min_x, max_x);
  const int64 x_0_int64 =
      FloatToQuantizedUnclamped<qint32>(x_0_float, output_min, output_max);
  const int64 x_1_int64 =
      FloatToQuantizedUnclamped<qint32>(x_1_float, output_min, output_max);
  const int32 x_mult_int32 = x_1_int64 - x_0_int64;

  const float y_0_float = QuantizedToFloat<quint8>(0, min_y, max_y);
  const float y_1_float = QuantizedToFloat<quint8>(1, min_y, max_y);
  const int64 y_0_int64 =
      FloatToQuantizedUnclamped<qint32>(y_0_float, output_min, output_max);
  const int64 y_1_int64 =
      FloatToQuantizedUnclamped<qint32>(y_1_float, output_min, output_max);
  const int32 y_mult_int32 = y_1_int64 - y_0_int64;

  const int64 lowest_quantized =
      static_cast<int64>(Eigen::NumTraits<qint32>::lowest());
  const int64 highest_quantized =
      static_cast<int64>(Eigen::NumTraits<qint32>::highest());

  const int64x2_t x_0_64x2 = vmovq_n_s64(x_0_int64);
  const int32x2_t x_mult_32x2 = vmov_n_s32(x_mult_int32);

  const int64x2_t y_0_64x2 = vmovq_n_s64(y_0_int64);
  const int32x2_t y_mult_32x2 = vmov_n_s32(y_mult_int32);

  int64 i = 0;
  for (; i < (num_elements - 7); i += 8) {
    const uint8* x_ptr = &(x_data->value) + i;
    const std::array<int32x4_t, 2> x_output_value =
        Requantize8x8To32Neon(x_ptr, x_0_64x2, x_mult_32x2);
    const uint8* y_ptr = &(y_data->value) + i;
    const std::array<int32x4_t, 2> y_output_value =
        Requantize8x8To32Neon(y_ptr, y_0_64x2, y_mult_32x2);

    const int32x4_t result_low_32x4 =
        vaddq_s32(x_output_value[0], y_output_value[0]);
    const int32x4_t result_high_32x4 =
        vaddq_s32(x_output_value[1], y_output_value[1]);
    int32* output_ptr = &(output->value) + i;
    vst1q_s32(output_ptr + 0, result_low_32x4);
    vst1q_s32(output_ptr + 4, result_high_32x4);
  }

  for (; i < num_elements; ++i) {
    const int64 x_value = static_cast<int64>(x_data[i]);
    int64 x_in_output_range_64 = x_0_int64 + (x_value * x_mult_int32);
    x_in_output_range_64 = std::max(x_in_output_range_64, lowest_quantized);
    x_in_output_range_64 = std::min(x_in_output_range_64, highest_quantized);
    const int32 x_in_output_range = static_cast<int32>(x_in_output_range_64);

    const int64 y_value = static_cast<int64>(y_data[i]);
    int64 y_in_output_range_64 = y_0_int64 + (y_value * y_mult_int32);
    y_in_output_range_64 = std::max(y_in_output_range_64, lowest_quantized);
    y_in_output_range_64 = std::min(y_in_output_range_64, highest_quantized);
    const int32 y_in_output_range = static_cast<int32>(y_in_output_range_64);

    output[i] = x_in_output_range + y_in_output_range;
  }
}

#else  // QUANTIZED_ADD_USE_NEON

template <>
void VectorAddition(OpKernelContext* context, const quint8* x_data, float min_x,
                    float max_x, const quint8* y_data, float min_y, float max_y,
                    int64 num_elements, float output_min, float output_max,
                    qint32* output) {
  const float x_0_float = QuantizedToFloat<quint8>(0, min_x, max_x);
  const float x_1_float = QuantizedToFloat<quint8>(1, min_x, max_x);
  const int64 x_0_int64 =
      FloatToQuantizedUnclamped<qint32>(x_0_float, output_min, output_max);
  const int64 x_1_int64 =
      FloatToQuantizedUnclamped<qint32>(x_1_float, output_min, output_max);
  const int32 x_mult_int32 = x_1_int64 - x_0_int64;

  const float y_0_float = QuantizedToFloat<quint8>(0, min_y, max_y);
  const float y_1_float = QuantizedToFloat<quint8>(1, min_y, max_y);
  const int64 y_0_int64 =
      FloatToQuantizedUnclamped<qint32>(y_0_float, output_min, output_max);
  const int64 y_1_int64 =
      FloatToQuantizedUnclamped<qint32>(y_1_float, output_min, output_max);
  const int32 y_mult_int32 = y_1_int64 - y_0_int64;

  const int64 lowest_quantized =
      static_cast<int64>(Eigen::NumTraits<qint32>::lowest());
  const int64 highest_quantized =
      static_cast<int64>(Eigen::NumTraits<qint32>::highest());

  for (int i = 0; i < num_elements; ++i) {
    const int64 x_value = static_cast<int64>(x_data[i]);
    int64 x_in_output_range_64 = x_0_int64 + (x_value * x_mult_int32);
    x_in_output_range_64 = std::max(x_in_output_range_64, lowest_quantized);
    x_in_output_range_64 = std::min(x_in_output_range_64, highest_quantized);
    const int32 x_in_output_range = static_cast<int32>(x_in_output_range_64);

    const int64 y_value = static_cast<int64>(y_data[i]);
    int64 y_in_output_range_64 = y_0_int64 + (y_value * y_mult_int32);
    y_in_output_range_64 = std::max(y_in_output_range_64, lowest_quantized);
    y_in_output_range_64 = std::min(y_in_output_range_64, highest_quantized);
    const int32 y_in_output_range = static_cast<int32>(y_in_output_range_64);

    output[i] = x_in_output_range + y_in_output_range;
  }
}

#endif  // QUANTIZED_ADD_USE_NEON

template <class T, class Toutput>
void VectorTensorAddition(const T* vector_data, float min_vector,
                          float max_vector, int64 vector_num_elements,
                          const T* tensor_data, float min_tensor,
                          float max_tensor, int64 tensor_num_elements,
                          float output_min, float output_max, Toutput* output) {
  for (int i = 0; i < tensor_num_elements; ++i) {
    const int64 vector_i = i % vector_num_elements;
    const Toutput vector_in_output_range = RequantizeInNewRange<T, Toutput>(
        vector_data[vector_i], min_vector, max_vector, output_min, output_max);
    const Toutput tensor_in_output_range = RequantizeInNewRange<T, Toutput>(
        tensor_data[i], min_tensor, max_tensor, output_min, output_max);
    output[i] = vector_in_output_range + tensor_in_output_range;
  }
}

#ifdef QUANTIZED_ADD_USE_NEON

template <>
void VectorTensorAddition(const quint8* vector_data, float min_vector,
                          float max_vector, int64 vector_num_elements,
                          const quint8* tensor_data, float min_tensor,
                          float max_tensor, int64 tensor_num_elements,
                          float output_min, float output_max, qint32* output) {
  const float vector_0_float =
      QuantizedToFloat<quint8>(0, min_vector, max_vector);
  const float vector_1_float =
      QuantizedToFloat<quint8>(1, min_vector, max_vector);
  const int64 vector_0_int64 =
      FloatToQuantizedUnclamped<qint32>(vector_0_float, output_min, output_max);
  const int64 vector_1_int64 =
      FloatToQuantizedUnclamped<qint32>(vector_1_float, output_min, output_max);
  const int32 vector_mult_int32 = vector_1_int64 - vector_0_int64;

  const float tensor_0_float =
      QuantizedToFloat<quint8>(0, min_tensor, max_tensor);
  const float tensor_1_float =
      QuantizedToFloat<quint8>(1, min_tensor, max_tensor);
  const int64 tensor_0_int64 =
      FloatToQuantizedUnclamped<qint32>(tensor_0_float, output_min, output_max);
  const int64 tensor_1_int64 =
      FloatToQuantizedUnclamped<qint32>(tensor_1_float, output_min, output_max);
  const int32 tensor_mult_int32 = tensor_1_int64 - tensor_0_int64;

  const int64 lowest_quantized =
      static_cast<int64>(Eigen::NumTraits<qint32>::lowest());
  const int64 highest_quantized =
      static_cast<int64>(Eigen::NumTraits<qint32>::highest());

  const int64x2_t vector_0_64x2 = vmovq_n_s64(vector_0_int64);
  const int32x2_t vector_mult_32x2 = vmov_n_s32(vector_mult_int32);

  const int64x2_t tensor_0_64x2 = vmovq_n_s64(tensor_0_int64);
  const int32x2_t tensor_mult_32x2 = vmov_n_s32(tensor_mult_int32);

  for (int64 base_i = 0; base_i < tensor_num_elements;
       base_i += vector_num_elements) {
    int64 i = base_i;
    int64 vector_i = 0;
    for (; vector_i < (vector_num_elements - 7); vector_i += 8, i += 8) {
      const uint8* vector_ptr = &(vector_data->value) + vector_i;
      const std::array<int32x4_t, 2> vector_output_value =
          Requantize8x8To32Neon(vector_ptr, vector_0_64x2, vector_mult_32x2);
      const uint8* tensor_ptr = &(tensor_data->value) + i;
      const std::array<int32x4_t, 2> tensor_output_value =
          Requantize8x8To32Neon(tensor_ptr, tensor_0_64x2, tensor_mult_32x2);

      const int32x4_t result_low_32x4 =
          vaddq_s32(vector_output_value[0], tensor_output_value[0]);
      const int32x4_t result_high_32x4 =
          vaddq_s32(vector_output_value[1], tensor_output_value[1]);
      int32* output_ptr = &(output->value) + i;
      vst1q_s32(output_ptr + 0, result_low_32x4);
      vst1q_s32(output_ptr + 4, result_high_32x4);
    }
    for (; vector_i < vector_num_elements; ++vector_i, ++i) {
      const int64 vector_value = static_cast<int64>(vector_data[vector_i]);
      int64 vector_in_output_range_64 =
          vector_0_int64 + (vector_value * vector_mult_int32);
      vector_in_output_range_64 =
          std::max(vector_in_output_range_64, lowest_quantized);
      vector_in_output_range_64 =
          std::min(vector_in_output_range_64, highest_quantized);
      const int32 vector_in_output_range =
          static_cast<int32>(vector_in_output_range_64);

      const int64 tensor_value = static_cast<int64>(tensor_data[i]);
      int64 tensor_in_output_range_64 =
          tensor_0_int64 + (tensor_value * tensor_mult_int32);
      tensor_in_output_range_64 =
          std::max(tensor_in_output_range_64, lowest_quantized);
      tensor_in_output_range_64 =
          std::min(tensor_in_output_range_64, highest_quantized);
      const int32 tensor_in_output_range =
          static_cast<int32>(tensor_in_output_range_64);

      output[i] = vector_in_output_range + tensor_in_output_range;
    }
  }
}

#else  // QUANTIZED_ADD_USE_NEON

template <>
void VectorTensorAddition(const quint8* vector_data, float min_vector,
                          float max_vector, int64 vector_num_elements,
                          const quint8* tensor_data, float min_tensor,
                          float max_tensor, int64 tensor_num_elements,
                          float output_min, float output_max, qint32* output) {
  const float vector_0_float =
      QuantizedToFloat<quint8>(0, min_vector, max_vector);
  const float vector_1_float =
      QuantizedToFloat<quint8>(1, min_vector, max_vector);
  const int64 vector_0_int64 =
      FloatToQuantizedUnclamped<qint32>(vector_0_float, output_min, output_max);
  const int64 vector_1_int64 =
      FloatToQuantizedUnclamped<qint32>(vector_1_float, output_min, output_max);
  const int32 vector_mult_int32 = vector_1_int64 - vector_0_int64;

  const float tensor_0_float =
      QuantizedToFloat<quint8>(0, min_tensor, max_tensor);
  const float tensor_1_float =
      QuantizedToFloat<quint8>(1, min_tensor, max_tensor);
  const int64 tensor_0_int64 =
      FloatToQuantizedUnclamped<qint32>(tensor_0_float, output_min, output_max);
  const int64 tensor_1_int64 =
      FloatToQuantizedUnclamped<qint32>(tensor_1_float, output_min, output_max);
  const int32 tensor_mult_int32 = tensor_1_int64 - tensor_0_int64;

  const int64 lowest_quantized =
      static_cast<int64>(Eigen::NumTraits<qint32>::lowest());
  const int64 highest_quantized =
      static_cast<int64>(Eigen::NumTraits<qint32>::highest());

  for (int i = 0; i < tensor_num_elements; ++i) {
    const int64 vector_i = i % vector_num_elements;
    const int64 vector_value = static_cast<int64>(vector_data[vector_i]);
    int64 vector_in_output_range_64 =
        vector_0_int64 + (vector_value * vector_mult_int32);
    vector_in_output_range_64 =
        std::max(vector_in_output_range_64, lowest_quantized);
    vector_in_output_range_64 =
        std::min(vector_in_output_range_64, highest_quantized);
    const int32 vector_in_output_range =
        static_cast<int32>(vector_in_output_range_64);

    const int64 tensor_value = static_cast<int64>(tensor_data[i]);
    int64 tensor_in_output_range_64 =
        tensor_0_int64 + (tensor_value * tensor_mult_int32);
    tensor_in_output_range_64 =
        std::max(tensor_in_output_range_64, lowest_quantized);
    tensor_in_output_range_64 =
        std::min(tensor_in_output_range_64, highest_quantized);
    const int32 tensor_in_output_range =
        static_cast<int32>(tensor_in_output_range_64);

    output[i] = vector_in_output_range + tensor_in_output_range;
  }
}

#endif  // QUANTIZED_ADD_USE_NEON

}  // namespace

template <class T, class Toutput>
class QuantizedAddOp : public OpKernel {
 public:
  explicit QuantizedAddOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& x = context->input(0);
    const Tensor& y = context->input(1);
    const float min_x = context->input(2).flat<float>()(0);
    const float max_x = context->input(3).flat<float>()(0);
    const float min_y = context->input(4).flat<float>()(0);
    const float max_y = context->input(5).flat<float>()(0);

    BCast bcast(BCast::FromShape(x.shape()), BCast::FromShape(y.shape()));
    if (!bcast.IsValid()) {
      context->SetStatus(errors::InvalidArgument(
          "Incompatible shapes: ", x.shape().DebugString(), " vs. ",
          y.shape().DebugString()));
      return;
    }
    Tensor* z;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, BCast::ToShape(bcast.output_shape()), &z));

    // Make sure that we have valid quantization ranges for the input buffers.
    // If the difference between the min and max is negative or zero, it makes
    // it hard to do meaningful intermediate operations on the values.
    OP_REQUIRES(context, (max_x > min_x),
                errors::InvalidArgument("max_x must be larger than min_x."));
    OP_REQUIRES(context, (max_y > min_y),
                errors::InvalidArgument("max_y must be larger than min_y."));
    const T* x_data = x.flat<T>().data();
    const T* y_data = y.flat<T>().data();
    Toutput* z_data = z->flat<Toutput>().data();

    // We want the range of the output to be symmetrical around zero so that
    // adding zero leaves the result unchanged, and to contain the largest of
    // the two input values with some room to spare.
    const float smallest_min = std::min(min_x, min_y);
    const float largest_max = std::max(max_x, max_y);
    const float biggest_range =
        std::max(std::abs(smallest_min), std::abs(largest_max));
    const float output_range = (biggest_range * (1 << 14));
    const float min_z_value = -output_range;
    const float max_z_value = output_range;

    const int ndims = bcast.x_reshape().size();
    if (ndims <= 1) {
      if (x.NumElements() == 1) {
        ScalarAddition<T, Toutput>(context, y_data, min_y, max_y,
                                   y.NumElements(), x_data[0], min_x, max_x,
                                   min_z_value, max_z_value, z_data);
      } else if (y.NumElements() == 1) {
        ScalarAddition<T, Toutput>(context, x_data, min_x, max_x,
                                   x.NumElements(), y_data[0], min_y, max_y,
                                   min_z_value, max_z_value, z_data);
      } else {
        VectorAddition<T, Toutput>(context, x_data, min_x, max_x, y_data, min_y,
                                   max_y, x.NumElements(), min_z_value,
                                   max_z_value, z_data);
      }
    } else if (ndims == 2) {
      const T* vector_data;
      int64 vector_num_elements;
      float vector_min;
      float vector_max;
      const T* tensor_data;
      int64 tensor_num_elements;
      float tensor_min;
      float tensor_max;
      if (x.NumElements() < y.NumElements()) {
        vector_data = x_data;
        vector_num_elements = x.NumElements();
        vector_min = min_x;
        vector_max = max_x;
        tensor_data = y_data;
        tensor_num_elements = y.NumElements();
        tensor_min = min_y;
        tensor_max = max_y;
      } else {
        vector_data = y_data;
        vector_num_elements = y.NumElements();
        vector_min = min_y;
        vector_max = max_y;
        tensor_data = x_data;
        tensor_num_elements = x.NumElements();
        tensor_min = min_x;
        tensor_max = max_x;
      }
      VectorTensorAddition<T, Toutput>(
          vector_data, vector_min, vector_max, vector_num_elements, tensor_data,
          tensor_min, tensor_max, tensor_num_elements, min_z_value, max_z_value,
          z_data);
    } else {
      LOG(INFO) << "ndims=" << ndims;
      LOG(INFO) << "bcast.x_reshape()="
                << TensorShape(bcast.x_reshape()).DebugString();
      LOG(INFO) << "bcast.y_reshape()="
                << TensorShape(bcast.y_reshape()).DebugString();
      LOG(INFO) << "bcast.x_bcast()="
                << TensorShape(bcast.x_bcast()).DebugString();
      LOG(INFO) << "bcast.y_bcast()="
                << TensorShape(bcast.y_bcast()).DebugString();

      context->SetStatus(errors::Unimplemented(
          "Broadcast between ", context->input(0).shape().DebugString(),
          " and ", context->input(1).shape().DebugString(),
          " is not supported yet."));
      return;
    }

    Tensor* z_min = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, {}, &z_min));
    z_min->flat<float>()(0) = min_z_value;

    Tensor* z_max = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(2, {}, &z_max));
    z_max->flat<float>()(0) = max_z_value;
  }
};

REGISTER_KERNEL_BUILDER(Name("QuantizedAdd")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("T1")
                            .TypeConstraint<quint8>("T2")
                            .TypeConstraint<qint32>("Toutput"),
                        QuantizedAddOp<quint8, qint32>);

}  // namespace tensorflow
