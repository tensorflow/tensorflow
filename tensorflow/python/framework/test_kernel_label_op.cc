#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/public/status.h"

namespace tensorflow {

REGISTER_OP("KernelLabel").Output("result: string");

namespace {
enum KernelLabel { DEFAULT_LABEL, OVERLOAD_1_LABEL, OVERLOAD_2_LABEL };
}  // namespace

template <KernelLabel KL>
class KernelLabelOp : public OpKernel {
 public:
  using OpKernel::OpKernel;

  void Compute(OpKernelContext* ctx) override {
    Tensor* output;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("result", TensorShape({}), &output));
    switch (KL) {
      case DEFAULT_LABEL:
        output->scalar<string>()() = "My label is: default";
        break;
      case OVERLOAD_1_LABEL:
        output->scalar<string>()() = "My label is: overload_1";
        break;
      case OVERLOAD_2_LABEL:
        output->scalar<string>()() = "My label is: overload_2";
        break;
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("KernelLabel").Device(DEVICE_CPU),
                        KernelLabelOp<DEFAULT_LABEL>);
REGISTER_KERNEL_BUILDER(Name("KernelLabel")
                            .Device(DEVICE_CPU)
                            .Label("overload_1"),
                        KernelLabelOp<OVERLOAD_1_LABEL>);
REGISTER_KERNEL_BUILDER(Name("KernelLabel")
                            .Device(DEVICE_CPU)
                            .Label("overload_2"),
                        KernelLabelOp<OVERLOAD_2_LABEL>);

}  // end namespace tensorflow
