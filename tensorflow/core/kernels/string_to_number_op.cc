// See docs in ../ops/parse_ops.cc.

#include <errno.h>
#include <string>

#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/public/status.h"
#include "tensorflow/core/public/tensor.h"

namespace tensorflow {

static constexpr char kErrorMessage[] =
    "StringToNumberOp could not correctly convert string: ";

template <typename OutputType>
class StringToNumberOp : public OpKernel {
 public:
  using OpKernel::OpKernel;

  void Compute(OpKernelContext* context) override {
    // This is not a deep copy of the input tensor; they will share the same
    // underlying storage.
    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("string_tensor", &input_tensor));
    const auto& input_flat = input_tensor->flat<string>();

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output("output", input_tensor->shape(),
                                            &output_tensor));
    auto output_flat = output_tensor->flat<OutputType>();

    for (int i = 0; i < input_flat.size(); ++i) {
      const char* s = input_flat(i).data();
      Convert(s, &output_flat(i), context);
    }
  }

 private:
  void Convert(const char* s, OutputType* output_data,
               OpKernelContext* context);
};

template <>
void StringToNumberOp<float>::Convert(const char* s, float* output_data,
                                      OpKernelContext* context) {
  OP_REQUIRES(context, strings::safe_strtof(s, output_data),
              errors::InvalidArgument(kErrorMessage, s));
}

template <>
void StringToNumberOp<int32>::Convert(const char* s, int32* output_data,
                                      OpKernelContext* context) {
  OP_REQUIRES(context, strings::safe_strto32(s, output_data),
              errors::InvalidArgument(kErrorMessage, s));
}

// Registers the currently supported output types.
#define REGISTER(type)                                           \
  REGISTER_KERNEL_BUILDER(Name("StringToNumber")                 \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<type>("out_type"), \
                          StringToNumberOp<type>)
REGISTER(float);
REGISTER(int32);
#undef REGISTER

}  // namespace tensorflow
