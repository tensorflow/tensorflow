/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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
#include <arm_neon.h>
#endif

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/meta_support.h"
#include "tensorflow/core/kernels/quantization_utils.h"
#include "tensorflow/core/lib/core/casts.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/bcast.h"

namespace tensorflow {
namespace {

template <class T, class Toutput>
void ScalarMultiply(OpKernelContext* context, const T* full_input,
                    int32 full_input_offset, int64 num_elements, T scalar_input,
                    int32 scalar_input_offset, Toutput* output) {
  const int32 scalar_minus_offset =
      static_cast<int32>(scalar_input) - scalar_input_offset;
  for (int i = 0; i < num_elements; ++i) {
    output[i] = (static_cast<int32>(full_input[i]) - full_input_offset) *
                scalar_minus_offset;
  }
}

#ifdef USE_NEON

template <>
void ScalarMultiply<quint8, qint32>(OpKernelContext* context,
                                    const quint8* full_input,
                                    int32 full_input_offset, int64 num_elements,
                                    quint8 scalar_input,
                                    int32 scalar_input_offset, qint32* output) {
  const int16 scalar_minus_offset =
      static_cast<int16>(scalar_input) - scalar_input_offset;
  const int16x4_t scalar_minus_offset_16x4 = vmov_n_s16(scalar_minus_offset);
  const uint8x8_t full_input_offset_8x8 = vmov_n_u8(full_input_offset);
  // Go through the results in 16-element chunks for NEON acceleration.
  int i;
  for (i = 0; i < (num_elements - 15); i += 16) {
    // Load the tensor inputs.
    const uint8* full_input_ptr = &(full_input->value) + i;
    const uint8x16_t full_input_8x16 = vld1q_u8(full_input_ptr);

    // Break into two sets of vectors so we can do further calculations
    // easily.
    const uint8x8_t full_input_high_8x8 = vget_high_u8(full_input_8x16);
    const uint8x8_t full_input_low_8x8 = vget_low_u8(full_input_8x16);

    // Subtract off the offset value to get 16-bit results.
    const int16x8_t full_input_minus_offset_high_16x8 = vreinterpretq_s16_u16(
        vsubl_u8(full_input_high_8x8, full_input_offset_8x8));
    const int16x8_t full_input_minus_offset_low_16x8 = vreinterpretq_s16_u16(
        vsubl_u8(full_input_low_8x8, full_input_offset_8x8));

    // We have to work with 4-wide vectors, so extract them.
    const int16x4_t x_high_high_16x4 =
        vget_high_s16(full_input_minus_offset_high_16x8);
    const int16x4_t x_high_low_16x4 =
        vget_low_s16(full_input_minus_offset_high_16x8);
    const int16x4_t x_low_high_16x4 =
        vget_high_s16(full_input_minus_offset_low_16x8);
    const int16x4_t x_low_low_16x4 =
        vget_low_s16(full_input_minus_offset_low_16x8);

    // Perform the multiplication.
    const int32x4_t z_high_high_32x4 =
        vmull_s16(x_high_high_16x4, scalar_minus_offset_16x4);
    const int32x4_t z_high_low_32x4 =
        vmull_s16(x_high_low_16x4, scalar_minus_offset_16x4);
    const int32x4_t z_low_high_32x4 =
        vmull_s16(x_low_high_16x4, scalar_minus_offset_16x4);
    const int32x4_t z_low_low_32x4 =
        vmull_s16(x_low_low_16x4, scalar_minus_offset_16x4);

    // Write out the results.
    int32* output_ptr = &(output->value) + i;
    vst1q_s32(output_ptr + 0, z_low_low_32x4);
    vst1q_s32(output_ptr + 4, z_low_high_32x4);
    vst1q_s32(output_ptr + 8, z_high_low_32x4);
    vst1q_s32(output_ptr + 12, z_high_high_32x4);
  }
  // Finish up any remaining elements that weren't a multiple of 16.
  for (; i < num_elements; ++i) {
    output[i] = (static_cast<int32>(full_input[i]) - full_input_offset) *
                scalar_minus_offset;
  }
}
#endif  // USE_NEON

template <class T, class Toutput>
void VectorMultiply(OpKernelContext* context, const T* x_data, int32 offset_x,
                    const T* y_data, int32 offset_y, int64 num_elements,
                    Toutput* output) {
  for (int i = 0; i < num_elements; ++i) {
    output[i] = (static_cast<int32>(x_data[i]) - offset_x) *
                (static_cast<int32>(y_data[i]) - offset_y);
  }
}

#ifdef USE_NEON
template <>
void VectorMultiply<quint8, qint32>(OpKernelContext* context,
                                    const quint8* x_data, int32 offset_x,
                                    const quint8* y_data, int32 offset_y,
                                    int64 num_elements, qint32* output) {
  const uint8x8_t offset_x_8x8 = vmov_n_u8(offset_x);
  const uint8x8_t offset_y_8x8 = vmov_n_u8(offset_y);
  int i;
  // Go through the results in 16-element chunks for NEON acceleration.
  for (i = 0; i < (num_elements - 15); i += 16) {
    // Load the vector inputs.
    const uint8* x_data_ptr = &(x_data->value) + i;
    const uint8x16_t x_8x16 = vld1q_u8(x_data_ptr);
    const uint8* y_data_ptr = &(y_data->value) + i;
    const uint8x16_t y_8x16 = vld1q_u8(y_data_ptr);

    // Break into two sets of vectors so we can do further calculations easily.
    const uint8x8_t x_high_8x8 = vget_high_u8(x_8x16);
    const uint8x8_t x_low_8x8 = vget_low_u8(x_8x16);
    const uint8x8_t y_high_8x8 = vget_high_u8(y_8x16);
    const uint8x8_t y_low_8x8 = vget_low_u8(y_8x16);

    // Subtract off the offset values to get 16-bit results.
    const int16x8_t x_minus_offset_high_16x8 =
        vreinterpretq_s16_u16(vsubl_u8(x_high_8x8, offset_x_8x8));
    const int16x8_t x_minus_offset_low_16x8 =
        vreinterpretq_s16_u16(vsubl_u8(x_low_8x8, offset_x_8x8));
    const int16x8_t y_minus_offset_high_16x8 =
        vreinterpretq_s16_u16(vsubl_u8(y_high_8x8, offset_y_8x8));
    const int16x8_t y_minus_offset_low_16x8 =
        vreinterpretq_s16_u16(vsubl_u8(y_low_8x8, offset_y_8x8));

    // We have to work with 4-wide vectors, so extract them.
    const int16x4_t x_high_high_16x4 = vget_high_s16(x_minus_offset_high_16x8);
    const int16x4_t x_high_low_16x4 = vget_low_s16(x_minus_offset_high_16x8);
    const int16x4_t x_low_high_16x4 = vget_high_s16(x_minus_offset_low_16x8);
    const int16x4_t x_low_low_16x4 = vget_low_s16(x_minus_offset_low_16x8);
    const int16x4_t y_high_high_16x4 = vget_high_s16(y_minus_offset_high_16x8);
    const int16x4_t y_high_low_16x4 = vget_low_s16(y_minus_offset_high_16x8);
    const int16x4_t y_low_high_16x4 = vget_high_s16(y_minus_offset_low_16x8);
    const int16x4_t y_low_low_16x4 = vget_low_s16(y_minus_offset_low_16x8);

    // Perform the multiplication.
    const int32x4_t z_high_high_32x4 =
        vmull_s16(x_high_high_16x4, y_high_high_16x4);
    const int32x4_t z_high_low_32x4 =
        vmull_s16(x_high_low_16x4, y_high_low_16x4);
    const int32x4_t z_low_high_32x4 =
        vmull_s16(x_low_high_16x4, y_low_high_16x4);
    const int32x4_t z_low_low_32x4 = vmull_s16(x_low_low_16x4, y_low_low_16x4);

    // Write out the results.
    int32* output_ptr = &(output->value) + i;
    vst1q_s32(output_ptr + 0, z_low_low_32x4);
    vst1q_s32(output_ptr + 4, z_low_high_32x4);
    vst1q_s32(output_ptr + 8, z_high_low_32x4);
    vst1q_s32(output_ptr + 12, z_high_high_32x4);
  }
  for (; i < num_elements; ++i) {
    output[i] = (static_cast<int32>(x_data[i]) - offset_x) *
                (static_cast<int32>(y_data[i]) - offset_y);
  }
}
#endif  // USE_NEON

template <class T, class Toutput>
void VectorTensorMultiply(const T* vector_data, int32 vector_offset,
                          int64 vector_num_elements, const T* tensor_data,
                          int32 tensor_offset, int64 tensor_num_elements,
                          Toutput* output) {
  for (int i = 0; i < tensor_num_elements; ++i) {
    const int64 vector_i = i % vector_num_elements;
    output[i] = (static_cast<int32>(vector_data[vector_i]) - vector_offset) *
                (static_cast<int32>(tensor_data[i]) - tensor_offset);
  }
}

#ifdef USE_NEON
template <>
void VectorTensorMultiply<quint8, qint32>(
    const quint8* vector_data, int32 vector_offset, int64 vector_num_elements,
    const quint8* tensor_data, int32 tensor_offset, int64 tensor_num_elements,
    qint32* output) {
  const uint8x8_t offset_x_8x8 = vmov_n_u8(vector_offset);
  const uint8x8_t offset_y_8x8 = vmov_n_u8(tensor_offset);
  CHECK_EQ(0, tensor_num_elements % vector_num_elements);
  for (int base_i = 0; base_i < tensor_num_elements;
       base_i += vector_num_elements) {
    int i = base_i;
    const int end_i = base_i + vector_num_elements;
    // Go through the results in 16-element chunks for NEON acceleration.
    int vector_i;
    for (vector_i = 0; vector_i < (vector_num_elements - 15);
         vector_i += 16, i += 16) {
      // Load the vector inputs.
      const uint8* x_data_ptr = &(vector_data->value) + vector_i;
      const uint8x16_t x_8x16 = vld1q_u8(x_data_ptr);
      const uint8* y_data_ptr = &(tensor_data->value) + i;
      const uint8x16_t y_8x16 = vld1q_u8(y_data_ptr);

      // Break into two sets of vectors so we can do further calculations
      // easily.
      const uint8x8_t x_high_8x8 = vget_high_u8(x_8x16);
      const uint8x8_t x_low_8x8 = vget_low_u8(x_8x16);
      const uint8x8_t y_high_8x8 = vget_high_u8(y_8x16);
      const uint8x8_t y_low_8x8 = vget_low_u8(y_8x16);

      // Subtract off the offset values to get 16-bit results.
      const int16x8_t x_minus_offset_high_16x8 =
          vreinterpretq_s16_u16(vsubl_u8(x_high_8x8, offset_x_8x8));
      const int16x8_t x_minus_offset_low_16x8 =
          vreinterpretq_s16_u16(vsubl_u8(x_low_8x8, offset_x_8x8));
      const int16x8_t y_minus_offset_high_16x8 =
          vreinterpretq_s16_u16(vsubl_u8(y_high_8x8, offset_y_8x8));
      const int16x8_t y_minus_offset_low_16x8 =
          vreinterpretq_s16_u16(vsubl_u8(y_low_8x8, offset_y_8x8));

      // We have to work with 4-wide vectors, so extract them.
      const int16x4_t x_high_high_16x4 =
          vget_high_s16(x_minus_offset_high_16x8);
      const int16x4_t x_high_low_16x4 = vget_low_s16(x_minus_offset_high_16x8);
      const int16x4_t x_low_high_16x4 = vget_high_s16(x_minus_offset_low_16x8);
      const int16x4_t x_low_low_16x4 = vget_low_s16(x_minus_offset_low_16x8);
      const int16x4_t y_high_high_16x4 =
          vget_high_s16(y_minus_offset_high_16x8);
      const int16x4_t y_high_low_16x4 = vget_low_s16(y_minus_offset_high_16x8);
      const int16x4_t y_low_high_16x4 = vget_high_s16(y_minus_offset_low_16x8);
      const int16x4_t y_low_low_16x4 = vget_low_s16(y_minus_offset_low_16x8);

      // Perform the multiplication.
      const int32x4_t z_high_high_32x4 =
          vmull_s16(x_high_high_16x4, y_high_high_16x4);
      const int32x4_t z_high_low_32x4 =
          vmull_s16(x_high_low_16x4, y_high_low_16x4);
      const int32x4_t z_low_high_32x4 =
          vmull_s16(x_low_high_16x4, y_low_high_16x4);
      const int32x4_t z_low_low_32x4 =
          vmull_s16(x_low_low_16x4, y_low_low_16x4);

      // Write out the results.
      int32* output_ptr = &(output->value) + i;
      vst1q_s32(output_ptr + 0, z_low_low_32x4);
      vst1q_s32(output_ptr + 4, z_low_high_32x4);
      vst1q_s32(output_ptr + 8, z_high_low_32x4);
      vst1q_s32(output_ptr + 12, z_high_high_32x4);
    }
    for (; i < end_i; ++i, ++vector_i) {
      output[i] = (static_cast<int32>(vector_data[vector_i]) - vector_offset) *
                  (static_cast<int32>(tensor_data[i]) - tensor_offset);
    }
  }
}
#endif  // USE_NEON

}  // namespace

template <class T, class Toutput>
class QuantizedMulOp : public OpKernel {
 public:
  explicit QuantizedMulOp(OpKernelConstruction* context) : OpKernel(context) {}

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
                errors::InvalidArgument("max_x must be larger than min_a."));
    OP_REQUIRES(context, (max_y > min_y),
                errors::InvalidArgument("max_x must be larger than min_b."));
    const int32 offset_x = FloatToQuantizedUnclamped<T>(0.0f, min_x, max_x);
    const int32 offset_y = FloatToQuantizedUnclamped<T>(0.0f, min_y, max_y);
    const T* x_data = x.flat<T>().data();
    const T* y_data = y.flat<T>().data();
    Toutput* z_data = z->flat<Toutput>().data();

    const int ndims = bcast.x_reshape().size();
    if (ndims <= 1) {
      if (x.NumElements() == 1) {
        ScalarMultiply<T, Toutput>(context, y_data, offset_y, y.NumElements(),
                                   x_data[0], offset_x, z_data);
      } else if (y.NumElements() == 1) {
        ScalarMultiply<T, Toutput>(context, x_data, offset_x, x.NumElements(),
                                   y_data[0], offset_y, z_data);
      } else {
        VectorMultiply<T, Toutput>(context, x_data, offset_x, y_data, offset_y,
                                   x.NumElements(), z_data);
      }
    } else if (ndims == 2) {
      const T* vector_data;
      int64 vector_num_elements;
      int32 vector_offset;
      const T* tensor_data;
      int64 tensor_num_elements;
      int32 tensor_offset;
      if (x.NumElements() < y.NumElements()) {
        vector_data = x_data;
        vector_num_elements = x.NumElements();
        vector_offset = offset_x;
        tensor_data = y_data;
        tensor_num_elements = y.NumElements();
        tensor_offset = offset_y;
      } else {
        vector_data = y_data;
        vector_num_elements = y.NumElements();
        vector_offset = offset_y;
        tensor_data = x_data;
        tensor_num_elements = x.NumElements();
        tensor_offset = offset_x;
      }
      VectorTensorMultiply<T, Toutput>(
          vector_data, vector_offset, vector_num_elements, tensor_data,
          tensor_offset, tensor_num_elements, z_data);
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

    float min_z_value;
    float max_z_value;
    QuantizationRangeForMultiplication<T, T, Toutput>(
        min_x, max_x, min_y, max_y, &min_z_value, &max_z_value);
    Tensor* z_min = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, {}, &z_min));
    z_min->flat<float>()(0) = min_z_value;

    Tensor* z_max = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(2, {}, &z_max));
    z_max->flat<float>()(0) = max_z_value;
  }
};

REGISTER_KERNEL_BUILDER(Name("QuantizedMul")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("T1")
                            .TypeConstraint<quint8>("T2")
                            .TypeConstraint<qint32>("Toutput"),
                        QuantizedMulOp<quint8, qint32>);

}  // namespace tensorflow
