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
#include "tensorflow/core/platform/cpu_info.h"

namespace tensorflow {

template <typename T>
class DecodeRawOp : public OpKernel {
 public:
  explicit DecodeRawOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("little_endian", &little_endian_));
    OP_REQUIRES_OK(context, context->GetAttr("out_type", &out_type_));
  }

  void Compute(OpKernelContext* context) override {
    const auto& input = context->input(0);
    int64 str_size = -1;
    auto flat_in = input.flat<string>();
    for (int64 i = 0; i < flat_in.size(); ++i) {
      const string& in_str = flat_in(i);
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
      out_shape.AddDim(1);
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
    const int64 added_dim = str_size / sizeof(T);
    out_shape.AddDim(added_dim);
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output("output", out_shape, &output_tensor));
    auto out = output_tensor->flat_inner_dims<T>();
    DCHECK_EQ(flat_in.size(), out.dimensions()[0]);
    T* out_data = out.data();
    if (port::kLittleEndian == little_endian_ || sizeof(T) == 1) {
      for (int64 i = 0; i < flat_in.size(); ++i) {
        const T* in_data = reinterpret_cast<const T*>(flat_in(i).data());
        memcpy(out_data, in_data, str_size);
        out_data += added_dim;
      }
    } else {
      for (int64 i = 0; i < flat_in.size(); ++i) {
        const char* in_data_bytes =
            reinterpret_cast<const char*>(flat_in(i).data());
        char* out_data_bytes = reinterpret_cast<char*>(out_data);
        const char* p = in_data_bytes;
        char* q = out_data_bytes;
        for (; p < in_data_bytes + str_size; p += sizeof(T), q += sizeof(T)) {
          std::reverse_copy(p, p + sizeof(T), q);
        }
        out_data += added_dim;
      }
    }
  }

 private:
  bool little_endian_;
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
REGISTER(uint8);
REGISTER(int16);
REGISTER(int8);
REGISTER(int64);

#undef REGISTER

}  // namespace tensorflow
