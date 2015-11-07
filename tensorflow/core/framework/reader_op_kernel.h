#ifndef TENSORFLOW_FRAMEWORK_READER_OP_KERNEL_H_
#define TENSORFLOW_FRAMEWORK_READER_OP_KERNEL_H_

#include <functional>
#include <string>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/reader_interface.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/platform/port.h"

namespace tensorflow {

// Implementation for ops providing a Reader.
class ReaderOpKernel : public OpKernel {
 public:
  explicit ReaderOpKernel(OpKernelConstruction* context);
  ~ReaderOpKernel() override;

  void Compute(OpKernelContext* context) override;

  // Must be called by descendants before the first call to Compute()
  // (typically called during construction).  factory must return a
  // ReaderInterface descendant allocated with new that ReaderOpKernel
  // will take ownership of.
  void SetReaderFactory(std::function<ReaderInterface*()> factory) {
    mutex_lock l(mu_);
    DCHECK(!have_handle_);
    factory_ = factory;
  }

 private:
  mutex mu_;
  bool have_handle_ GUARDED_BY(mu_);
  PersistentTensor handle_ GUARDED_BY(mu_);
  ContainerInfo cinfo_;
  std::function<ReaderInterface*()> factory_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_FRAMEWORK_READER_OP_KERNEL_H_
