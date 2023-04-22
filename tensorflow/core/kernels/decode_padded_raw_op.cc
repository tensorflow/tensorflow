/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <algorithm>
#include <cmath>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

template <typename T>
class DecodePaddedRawOp : public OpKernel {
 public:
  explicit DecodePaddedRawOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("out_type", &out_type_));

    const bool host_is_little_endian = port::kLittleEndian;
    bool data_is_little_endian;
    OP_REQUIRES_OK(context,
                   context->GetAttr("little_endian", &data_is_little_endian));
    convert_data_endianness_ = host_is_little_endian != data_is_little_endian;
  }

  void Compute(OpKernelContext* context) override {
    const auto& input = context->input(0);
    auto flat_in = input.flat<tstring>();

    int fixed_length;
    const auto& length_input = context->input(1);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(length_input.shape()),
                errors::InvalidArgument("k must be scalar, got shape ",
                                        length_input.shape().DebugString()));
    fixed_length = length_input.scalar<int32>()();

    OP_REQUIRES(
        context, fixed_length % sizeof(T) == 0,
        errors::InvalidArgument(
            "fixed_length (", fixed_length,
            ") must be a multiple of the size of out_type (", sizeof(T), ")"));

    OP_REQUIRES(context, fixed_length > 0,
                errors::InvalidArgument("fixed_length (", fixed_length,
                                        ") must be greater than zero."));

    int width = fixed_length / sizeof(T);

    TensorShape out_shape = input.shape();
    out_shape.AddDim(width);
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output("output", out_shape, &output_tensor));

    if (flat_in.size() == 0) {  // Empty input
      return;
    }

    auto out = output_tensor->flat_inner_dims<T>();
    T* out_data = out.data();

    // Forcibly clear memory - we're going to copy variable length strings in,
    // and need to ensure that if we don't write to byte N when we copy, that
    // we're not getting random data.
    memset(out_data, 0, fixed_length * flat_in.size());

    // If the data is already in the host's byte order, or if the width of the
    // output type is a single byte (meaning the ordering doesn't matter), we
    // can copy the memory directly.
    if (!convert_data_endianness_ || sizeof(T) == 1) {
      for (int64 i = 0; i < flat_in.size(); ++i) {
        const auto to_copy =
            std::min(flat_in(i).size(), static_cast<size_t>(fixed_length));
        memcpy(out_data, flat_in(i).data(), to_copy);
        // Note: increase out_data by width since it's already of type T* so
        // each shift amount is implicitly multiplied by sizeof(T) according to
        // pointer arithmetic rules.
        out_data += width;
      }
    } else {
      // Otherwise, the data is not in the host's byte order, and rather than a
      // direct copy, we need to reverse the byte ordering of each element.
      for (int64 i = 0; i < flat_in.size(); ++i) {
        const char* in_data_bytes =
            reinterpret_cast<const char*>(flat_in(i).data());
        char* out_data_bytes = reinterpret_cast<char*>(out_data);
        const char* p_in = in_data_bytes;
        char* p_out = out_data_bytes;
        for (; p_in < in_data_bytes + fixed_length;
             p_in += sizeof(T), p_out += sizeof(T)) {
          std::reverse_copy(p_in, p_in + sizeof(T), p_out);
        }
        // Note: increase out_data by width since it's already of type T* so
        // each shift amount is implicitly multiplied by sizeof(T) according to
        // pointer arithmetic rules.
        out_data += width;
      }
    }
  }

 private:
  // True if the endianness of the data and the endianness of the host are
  // different, and the data needs conversion.
  bool convert_data_endianness_;

  // Data type of the output tensor.
  DataType out_type_;
};

#define REGISTER(type)                                           \
  REGISTER_KERNEL_BUILDER(Name("DecodePaddedRaw")                \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<type>("out_type"), \
                          DecodePaddedRawOp<type>)

REGISTER(float);
REGISTER(double);
REGISTER(int32);
REGISTER(uint16);
REGISTER(uint8);
REGISTER(int16);
REGISTER(int8);
REGISTER(int64);
REGISTER(bfloat16);

#undef REGISTER

}  // namespace tensorflow
