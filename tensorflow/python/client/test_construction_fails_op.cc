#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/public/status.h"

namespace tensorflow {

REGISTER_OP("ConstructionFails");

class ConstructionFailsOp : public OpKernel {
 public:
  explicit ConstructionFailsOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES(ctx, false,
                errors::InvalidArgument("Failure during construction."));
  }

  void Compute(OpKernelContext* ctx) override {}
};

REGISTER_KERNEL_BUILDER(Name("ConstructionFails").Device(DEVICE_CPU),
                        ConstructionFailsOp);

}  // end namespace tensorflow
