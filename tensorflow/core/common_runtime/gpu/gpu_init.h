#ifndef TENSORFLOW_COMMON_RUNTIME_GPU_GPU_INIT_H_
#define TENSORFLOW_COMMON_RUNTIME_GPU_GPU_INIT_H_

namespace perftools {
namespace gputools {
class Platform;
}  // namespace gputools
}  // namespace perftools

namespace tensorflow {

// Returns the GPU machine manager singleton, creating it and
// initializing the GPUs on the machine if needed the first time it is
// called.
perftools::gputools::Platform* GPUMachineManager();

}  // namespace tensorflow

#endif  // TENSORFLOW_COMMON_RUNTIME_GPU_GPU_INIT_H_
