// See docs in ../ops/data_flow_ops.cc.

#include <deque>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/fifo_queue.h"
#include "tensorflow/core/kernels/queue_base.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/public/tensor_shape.h"

namespace tensorflow {

// Defines a FIFOQueueOp, which produces a Queue (specifically, one
// backed by FIFOQueue) that persists across different graph
// executions, and sessions. Running this op produces a single-element
// tensor of handles to Queues in the corresponding device.
class FIFOQueueOp : public OpKernel {
 public:
  explicit FIFOQueueOp(OpKernelConstruction* context)
      : OpKernel(context), queue_handle_set_(false) {
    OP_REQUIRES_OK(context, context->GetAttr("capacity", &capacity_));
    OP_REQUIRES_OK(context,
                   context->allocate_persistent(DT_STRING, TensorShape({2}),
                                                &queue_handle_, nullptr));
    if (capacity_ < 0) {
      capacity_ = FIFOQueue::kUnbounded;
    }
    OP_REQUIRES_OK(context,
                   context->GetAttr("component_types", &component_types_));
    OP_REQUIRES_OK(context, context->GetAttr("shapes", &component_shapes_));
  }

  ~FIFOQueueOp() override {
    // If the queue object was not shared, delete it.
    if (queue_handle_set_ && cinfo_.resource_is_private_to_kernel()) {
      TF_CHECK_OK(cinfo_.resource_manager()->Delete<QueueInterface>(
          cinfo_.container(), cinfo_.name()));
    }
  }

  void Compute(OpKernelContext* ctx) override {
    mutex_lock l(mu_);
    if (!queue_handle_set_) {
      OP_REQUIRES_OK(ctx, SetQueueHandle(ctx));
    }
    ctx->set_output_ref(0, &mu_, queue_handle_.AccessTensor(ctx));
  }

 private:
  Status SetQueueHandle(OpKernelContext* ctx) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    TF_RETURN_IF_ERROR(cinfo_.Init(ctx->resource_manager(), def()));
    QueueInterface* queue;
    auto creator = [this](QueueInterface** ret) {
      FIFOQueue* queue = new FIFOQueue(capacity_, component_types_,
                                       component_shapes_, cinfo_.name());
      *ret = queue;
      return queue->Initialize();
    };
    TF_RETURN_IF_ERROR(
        cinfo_.resource_manager()->LookupOrCreate<QueueInterface>(
            cinfo_.container(), cinfo_.name(), &queue, creator));
    core::ScopedUnref unref_me(queue);
    // Verify that the shared queue is compatible with the requested arguments.
    TF_RETURN_IF_ERROR(queue->MatchesNodeDef(def()));
    auto h = queue_handle_.AccessTensor(ctx)->flat<string>();
    h(0) = cinfo_.container();
    h(1) = cinfo_.name();
    queue_handle_set_ = true;
    return Status::OK();
  }

  int32 capacity_;
  DataTypeVector component_types_;
  std::vector<TensorShape> component_shapes_;
  ContainerInfo cinfo_;

  mutex mu_;
  PersistentTensor queue_handle_ GUARDED_BY(mu_);
  bool queue_handle_set_ GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(FIFOQueueOp);
};

REGISTER_KERNEL_BUILDER(Name("FIFOQueue").Device(DEVICE_CPU), FIFOQueueOp);

}  // namespace tensorflow
