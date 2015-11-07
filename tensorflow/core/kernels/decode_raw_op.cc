// See docs in ../ops/parse_ops.cc.

#include <algorithm>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/public/tensor_shape.h"

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
    int str_size = -1;
    auto flat_in = input.flat<string>();
    for (int i = 0; i < flat_in.size(); ++i) {
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
    if (str_size == -1) {  // Empty input
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
    const int added_dim = str_size / sizeof(T);
    out_shape.AddDim(added_dim);
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output("output", out_shape, &output_tensor));
    auto out = output_tensor->flat_inner_dims<T>();
    DCHECK_EQ(flat_in.size(), out.dimensions()[0]);
    OP_REQUIRES(
        context,
        little_endian_ == ::tensorflow::port::kLittleEndian || sizeof(T) == 1,
        errors::Unimplemented("Unimplemented support for little_endian=",
                              little_endian_ ? "true" : "false"));
    // Endianness matches, so just copy each string byte-for-byte.
    T* out_data = out.data();
    for (int i = 0; i < flat_in.size(); ++i) {
      const T* in_data = reinterpret_cast<const T*>(flat_in(i).data());
      memcpy(out_data, in_data, str_size);
      out_data += added_dim;
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

REGISTER(float);
REGISTER(double);
REGISTER(int32);
REGISTER(uint8);
REGISTER(int16);
REGISTER(int8);
REGISTER(int64);

#undef REGISTER

}  // namespace tensorflow
