#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/fifo_bucketed_queue.h"
#include "tensorflow/core/kernels/queue_op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class FIFOBucketedQueueOp : public QueueOp {
 public:
  explicit FIFOBucketedQueueOp(OpKernelConstruction* context)
    : QueueOp(context) {
    OP_REQUIRES_OK(context, context->GetAttr("shapes", &component_shapes_));
    OP_REQUIRES_OK(context, context->GetAttr("buckets", &buckets_));
    OP_REQUIRES_OK(context, context->GetAttr("batch_size", &batch_size_));
  }

 protected:
  CreatorCallback GetCreator() const override {
    return [this](QueueInterface** ret) {
      FIFOBucketedQueue* queue = new FIFOBucketedQueue(
          buckets_, capacity_, batch_size_, component_types_, component_shapes_,
          cinfo_.name());
      *ret = queue;
      return queue->Initialize();
    };
  }

 private:
  int buckets_;
  int batch_size_;
  std::vector<TensorShape> component_shapes_;

  TF_DISALLOW_COPY_AND_ASSIGN(FIFOBucketedQueueOp);
};

REGISTER_KERNEL_BUILDER(
    Name("FIFOBucketedQueue").Device(DEVICE_CPU), FIFOBucketedQueueOp);

}  // namespace tensorflow
