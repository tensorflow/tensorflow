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

// See docs in ../ops/parse_ops.cc.

#include <algorithm>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/bfloat16.h"
#include "tensorflow/core/platform/byte_order.h"

namespace tensorflow {

template <typename T>
class DecodeRawOp : public OpKernel {
 public:
  explicit DecodeRawOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("out_type", &out_type_));

    const bool host_is_little_endian = port::kLittleEndian;
    bool data_is_little_endian;
    OP_REQUIRES_OK(context,
                   context->GetAttr("little_endian", &data_is_little_endian));
    convert_data_endianness_ = host_is_little_endian != data_is_little_endian;
  }

  void Compute(OpKernelContext* context) override {
    const auto& input = context->input(0);
    int64_t str_size = -1;
    auto flat_in = input.flat<tstring>();
    for (int64_t i = 0; i < flat_in.size(); ++i) {
      const tstring& in_str = flat_in(i);
      if (str_size == -1) {
        str_size = in_str.size();
      } else {
        OP_REQUIRES(context, str_size == in_str.size(),
                    errors::InvalidArgument(
                        "DecodeRaw requires input strings to all be the same "
                        "size, but element ",
                        i, " has size ", str_size, " != ", in_str.size()));
      }
    }
    TensorShape out_shape = input.shape();
    if (str_size == -1 || str_size == 0) {  // Empty input
      OP_REQUIRES_OK(context, out_shape.AddDimWithStatus(0));
      Tensor* output_tensor = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output("output", out_shape,
                                                       &output_tensor));
      return;
    }
    OP_REQUIRES(
        context, str_size % sizeof(T) == 0,
        errors::InvalidArgument("Input to DecodeRaw has length ", str_size,
                                " that is not a multiple of ", sizeof(T),
                                ", the size of ", DataTypeString(out_type_)));
    const int64_t added_dim = str_size / sizeof(T);
    OP_REQUIRES_OK(context, out_shape.AddDimWithStatus(added_dim));
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output("output", out_shape, &output_tensor));
    auto out = output_tensor->flat_inner_dims<T>();
    DCHECK_EQ(flat_in.size(), out.dimensions()[0]);
    T* out_data = out.data();

    // If the data is already in the host's byte order, or if the width of the
    // output type is a single byte, we can copy the memory directly.
    if (!convert_data_endianness_ || sizeof(T) == 1) {
      for (int64_t i = 0; i < flat_in.size(); ++i) {
        const T* in_data = reinterpret_cast<const T*>(flat_in(i).data());
        memcpy(out_data, in_data, str_size);
        out_data += added_dim;
      }
    } else {
      // Otherwise, the data is not in the host's byte order, and rather than a
      // direct copy, we need to reverse the byte ordering of each element.
      int64_t element_size;
      if (out_type_ == DT_COMPLEX64 || out_type_ == DT_COMPLEX128) {
        // For Complex data type, real and imaginary parts need to be swapped
        // separately
        element_size = sizeof(T) / 2;
      } else {
        element_size = sizeof(T);
      }
      for (int64_t i = 0; i < flat_in.size(); ++i) {
        const char* in_data_bytes =
            reinterpret_cast<const char*>(flat_in(i).data());
        char* out_data_bytes = reinterpret_cast<char*>(out_data);
        const char* p = in_data_bytes;
        char* q = out_data_bytes;
        for (; p < in_data_bytes + str_size;
             p += element_size, q += element_size) {
          std::reverse_copy(p, p + element_size, q);
        }
        out_data += added_dim;
      }
    }
  }

 private:
  // True if the endianness of the data and the endianness of the host are
  // different, and the data needs conversion.
  bool convert_data_endianness_;

  // True if the input data is in little endian format.
  bool data_is_little_endian_;
  DataType out_type_;
};

#define REGISTER(type)                                                       \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("DecodeRaw").Device(DEVICE_CPU).TypeConstraint<type>("out_type"), \
      DecodeRawOp<type>)

REGISTER(Eigen::half);
REGISTER(float);
REGISTER(double);
REGISTER(int32);
REGISTER(uint16);
REGISTER(uint8);
REGISTER(int16);
REGISTER(int8);
REGISTER(int64_t);
REGISTER(bool);
REGISTER(complex64);
REGISTER(complex128);
REGISTER(bfloat16);

#undef REGISTER

}  // namespace tensorflow
