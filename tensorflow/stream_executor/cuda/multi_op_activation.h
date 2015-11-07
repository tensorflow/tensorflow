#ifndef TENSORFLOW_STREAM_EXECUTOR_CUDA_MULTI_OP_ACTIVATION_H_
#define TENSORFLOW_STREAM_EXECUTOR_CUDA_MULTI_OP_ACTIVATION_H_

namespace perftools {
namespace gputools {
namespace cuda {

// Type-safe boolean wrapper: denotes whether a ScopedActivateExecutorContext
// may have other ScopedActivateExecutorContexts nested within it.
enum class MultiOpActivation { kNo = false, kYes = true };

}  // namespace cuda
}  // namespace gputools
}  // namespace perftools

#endif  // TENSORFLOW_STREAM_EXECUTOR_CUDA_MULTI_OP_ACTIVATION_H_
