#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/public/status.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

class AssertOp : public OpKernel {
 public:
  explicit AssertOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("summarize", &summarize_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& cond = ctx->input(0);
    OP_REQUIRES(ctx, TensorShapeUtils::IsLegacyScalar(cond.shape()),
                errors::InvalidArgument("In[0] should be a scalar: ",
                                        cond.shape().ShortDebugString()));

    if (cond.scalar<bool>()()) {
      return;
    }
    string msg = "assertion failed: ";
    for (int i = 1; i < ctx->num_inputs(); ++i) {
      strings::StrAppend(&msg, "[", ctx->input(i).SummarizeValue(summarize_),
                         "]");
      if (i < ctx->num_inputs() - 1) strings::StrAppend(&msg, " ");
    }
    ctx->SetStatus(errors::InvalidArgument(msg));
  }

 private:
  int32 summarize_ = 0;
};

REGISTER_KERNEL_BUILDER(Name("Assert").Device(DEVICE_CPU), AssertOp);

class PrintOp : public OpKernel {
 public:
  explicit PrintOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), call_counter_(0) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("message", &message_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("first_n", &first_n_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("summarize", &summarize_));
  }

  void Compute(OpKernelContext* ctx) override {
    if (IsRefType(ctx->input_dtype(0))) {
      ctx->forward_ref_input_to_ref_output(0, 0);
    } else {
      ctx->set_output(0, ctx->input(0));
    }
    if (first_n_ >= 0) {
      mutex_lock l(mu_);
      if (call_counter_ >= first_n_) return;
      call_counter_++;
    }
    string msg;
    strings::StrAppend(&msg, message_);
    for (int i = 1; i < ctx->num_inputs(); ++i) {
      strings::StrAppend(&msg, "[", ctx->input(i).SummarizeValue(summarize_),
                         "]");
    }
    LOG(INFO) << msg;
  }

 private:
  mutex mu_;
  int64 call_counter_ GUARDED_BY(mu_) = 0;
  int64 first_n_ = 0;
  int32 summarize_ = 0;
  string message_;
};

REGISTER_KERNEL_BUILDER(Name("Print").Device(DEVICE_CPU), PrintOp);

}  // end namespace tensorflow
