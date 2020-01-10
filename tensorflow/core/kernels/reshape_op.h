#ifndef TENSORFLOW_KERNELS_RESHAPE_OP_H_
#define TENSORFLOW_KERNELS_RESHAPE_OP_H_

#include <memory>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/status.h"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/public/tensor_shape.h"

namespace tensorflow {

class ReshapeOp : public OpKernel {
 public:
  explicit ReshapeOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& sizes = context->input(1);
    // Preliminary validation of sizes.
    OP_REQUIRES(context, TensorShapeUtils::IsLegacyVector(sizes.shape()),
                errors::InvalidArgument("sizes input must be 1-D, not shape ",
                                        sizes.shape().ShortDebugString()));
    const int64 num_dims = sizes.NumElements();
    OP_REQUIRES(
        context, num_dims <= 8,
        errors::InvalidArgument(num_dims, " > max 8 output dims supported"));

    // Compute the output shape.  Determine product of specified
    // dimensions, and find the index of the unspecified one.
    TensorShape shape;
    int32 product = 1;
    int unknown_index = -1;
    auto Svec = sizes.flat<int32>();
    for (int d = 0; d < num_dims; ++d) {
      const int32 size = Svec(d);
      if (size == -1) {
        OP_REQUIRES(
            context, unknown_index == -1,
            errors::InvalidArgument("only one input size may be -1, not both ",
                                    unknown_index, " and ", d));
        unknown_index = d;
        shape.AddDim(1);
      } else {
        OP_REQUIRES(context, size >= 0,
                    errors::InvalidArgument(
                        "size ", d, " must be non-negative, not ", size));
        shape.AddDim(size);
        product *= size;
      }
    }
    if (unknown_index != -1) {
      OP_REQUIRES(
          context, product > 0,
          errors::InvalidArgument("cannot infer the missing input size for "
                                  "an empty tensor unless all specified "
                                  "input sizes are non-zero"));
      const int32 missing = input.NumElements() / product;
      OP_REQUIRES(context, product * missing == input.NumElements(),
                  errors::InvalidArgument("Input has ", input.NumElements(),
                                          " values, which isn't divisible by ",
                                          product));
      shape.set_dim(unknown_index, missing);
    }
    OP_REQUIRES(context, shape.num_elements() == input.NumElements(),
                errors::InvalidArgument("Input has ", input.NumElements(),
                                        " values, which isn't the same as ",
                                        shape.num_elements()));

    // Actually produce the reshaped output.
    Tensor output(input.dtype());
    CHECK(output.CopyFrom(input, shape));
    context->set_output(0, output);
  }

  bool IsExpensive() override { return false; }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_RESHAPE_OP_H_
