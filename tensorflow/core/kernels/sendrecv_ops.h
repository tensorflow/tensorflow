#ifndef TENSORFLOW_KERNELS_SENDRECV_OPS_H_
#define TENSORFLOW_KERNELS_SENDRECV_OPS_H_

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

class SendOp : public OpKernel {
 public:
  explicit SendOp(OpKernelConstruction* ctx);
  void Compute(OpKernelContext* ctx) override;

 private:
  string key_prefix_;

  TF_DISALLOW_COPY_AND_ASSIGN(SendOp);
};

class RecvOp : public AsyncOpKernel {
 public:
  explicit RecvOp(OpKernelConstruction* ctx);
  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;

 private:
  string key_prefix_;

  TF_DISALLOW_COPY_AND_ASSIGN(RecvOp);
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_KERNELS_SENDRECV_OPS_H_
