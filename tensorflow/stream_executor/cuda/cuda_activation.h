// This file contains APIs that assume a StreamExecutor is backed by CUDA.
// It reaches into the CUDA implementation to activate an underlying CUDA
// context.
//
// Having this file separate from cuda_gpu_executor.h means that dependent
// code does not also have to depend on cuda.h.

#ifndef TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_ACTIVATION_H_
#define TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_ACTIVATION_H_

#include "tensorflow/stream_executor/cuda/multi_op_activation.h"
#include "tensorflow/stream_executor/platform/port.h"

namespace perftools {
namespace gputools {

class StreamExecutor;

namespace cuda {

class CUDAExecutor;
class ScopedActivateContext;

// Activates a CUDA context within an enclosing scope.
class ScopedActivateExecutorContext {
 public:
  // Form that takes a CUDA executor implementation.
  explicit ScopedActivateExecutorContext(
      CUDAExecutor* cuda_exec, MultiOpActivation moa = MultiOpActivation::kNo);

  // Form that takes a pImpl executor and extracts a CUDA implementation --
  // fatal failure if it is not CUDA inside.
  explicit ScopedActivateExecutorContext(
      StreamExecutor* stream_exec,
      MultiOpActivation moa = MultiOpActivation::kNo);

  ~ScopedActivateExecutorContext();

 private:
  // The CUDA executor implementation whose context is activated.
  CUDAExecutor* cuda_exec_;

  // The cuda.h-using datatype that we wrap.
  ScopedActivateContext* driver_scoped_activate_context_;

  SE_DISALLOW_COPY_AND_ASSIGN(ScopedActivateExecutorContext);
};

}  // namespace cuda
}  // namespace gputools
}  // namespace perftools

#endif  // TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_ACTIVATION_H_
