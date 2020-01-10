#ifndef TENSORFLOW_KERNELS_CONTROL_FLOW_OPS_H_
#define TENSORFLOW_KERNELS_CONTROL_FLOW_OPS_H_

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

// A ControlTriggerOp is similar to a NoOp. However, it always treats the input
// control edges as Live edges. Its primary use so far is in the scheduling of
// recvs, where we add ControlTrigger nodes and use them to trigger recvs. We
// allow ControlTrigger nodes to be enabled by dead nodes.
class ControlTriggerOp : public OpKernel {
 public:
  explicit ControlTriggerOp(OpKernelConstruction* context)
      : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {}
  bool IsExpensive() override { return false; }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_CONTROL_FLOW_OPS_H_
