#include "tensorflow/stream_executor/stream_executor_internal.h"

#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/lib/stringprintf.h"

namespace perftools {
namespace gputools {
namespace internal {

// -- CUDA

StreamExecutorFactory* MakeCUDAExecutorImplementation() {
  static StreamExecutorFactory instance;
  return &instance;
}
EventFactory* MakeCUDAEventImplementation() {
  static EventFactory instance;
  return &instance;
}
StreamFactory* MakeCUDAStreamImplementation() {
  static StreamFactory instance;
  return &instance;
}
TimerFactory* MakeCUDATimerImplementation() {
  static TimerFactory instance;
  return &instance;
}
KernelFactory* MakeCUDAKernelImplementation() {
  static KernelFactory instance;
  return &instance;
}

// -- OpenCL

StreamExecutorFactory* MakeOpenCLExecutorImplementation() {
  static StreamExecutorFactory instance;
  return &instance;
}
StreamExecutorFactory* MakeOpenCLAlteraExecutorImplementation() {
  static StreamExecutorFactory instance;
  return &instance;
}
StreamFactory* MakeOpenCLStreamImplementation() {
  static StreamFactory instance;
  return &instance;
}
TimerFactory* MakeOpenCLTimerImplementation() {
  static TimerFactory instance;
  return &instance;
}
KernelFactory* MakeOpenCLKernelImplementation() {
  static KernelFactory instance;
  return &instance;
}

// -- Host

StreamExecutorFactory MakeHostExecutorImplementation;
StreamFactory MakeHostStreamImplementation;
TimerFactory MakeHostTimerImplementation;


}  // namespace internal
}  // namespace gputools
}  // namespace perftools
