#include "tensorflow/stream_executor/cuda/cuda_activation.h"

#include "tensorflow/stream_executor/cuda/cuda_driver.h"
#include "tensorflow/stream_executor/stream_executor.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"

namespace perftools {
namespace gputools {
namespace cuda {

CUcontext ExtractCudaContext(CUDAExecutor *cuda_exec);
CUDAExecutor *ExtractCudaExecutor(StreamExecutor *stream_exec);

ScopedActivateExecutorContext::ScopedActivateExecutorContext(
    CUDAExecutor *cuda_exec, MultiOpActivation moa)
    : cuda_exec_(cuda_exec),
      driver_scoped_activate_context_(
          new ScopedActivateContext{ExtractCudaContext(cuda_exec), moa}) {}

ScopedActivateExecutorContext::ScopedActivateExecutorContext(
    StreamExecutor *stream_exec, MultiOpActivation moa)
    : ScopedActivateExecutorContext(ExtractCudaExecutor(stream_exec), moa) {}

ScopedActivateExecutorContext::~ScopedActivateExecutorContext() {
  delete static_cast<ScopedActivateContext *>(driver_scoped_activate_context_);
}

}  // namespace cuda
}  // namespace gputools
}  // namespace perftools
