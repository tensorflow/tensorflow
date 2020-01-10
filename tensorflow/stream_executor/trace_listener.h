// This file defines the StreamExecutor trace listener, used for inserting
// non-device-specific instrumentation into the StreamExecutor.
#ifndef TENSORFLOW_STREAM_EXECUTOR_TRACE_LISTENER_H_
#define TENSORFLOW_STREAM_EXECUTOR_TRACE_LISTENER_H_

#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/kernel.h"
#include "tensorflow/stream_executor/launch_dim.h"
#include "tensorflow/stream_executor/lib/status.h"

namespace perftools {
namespace gputools {

class Stream;

// Traces StreamExecutor PIMPL-level events.
// The few StreamExecutor interfaces that are synchronous have both Begin and
// Complete versions of their trace calls. Asynchronous operations only have
// Submit calls, as execution of the underlying operations is device-specific.
// As all tracing calls mirror StreamExecutor routines, documentation here is
// minimal.
//
// All calls have default implementations that perform no work; subclasses
// should override functionality of interest. Keep in mind that these routines
// are not called on a dedicated thread, so callbacks should execute quickly.
//
// Note: This API is constructed on an as-needed basis. Users should add
// support for further StreamExecutor operations as required. By enforced
// convention (see SCOPED_TRACE in stream_executor_pimpl.cc), synchronous
// tracepoints should be named NameBegin and NameComplete.
class TraceListener {
 public:
  virtual ~TraceListener() {}

  virtual void LaunchSubmit(Stream* stream, const ThreadDim& thread_dims,
                            const BlockDim& block_dims,
                            const KernelBase& kernel,
                            const std::vector<KernelArg>& args) {}

  virtual void SynchronousMemcpyH2DBegin(int64 correlation_id,
                                         const void* host_src, int64 size,
                                         DeviceMemoryBase* gpu_dst) {}
  virtual void SynchronousMemcpyH2DComplete(int64 correlation_id,
                                            const port::Status* result) {}

  virtual void SynchronousMemcpyD2HBegin(int64 correlation_id,
                                         const DeviceMemoryBase& gpu_src,
                                         int64 size, void* host_dst) {}
  virtual void SynchronousMemcpyD2HComplete(int64 correlation_id,
                                            const port::Status* result) {}

  virtual void BlockHostUntilDoneBegin(int64 correlation_id, Stream* stream) {}
  virtual void BlockHostUntilDoneComplete(int64 correlation_id, bool result) {}
};

}  // namespace gputools
}  // namespace perftools

#endif  // TENSORFLOW_STREAM_EXECUTOR_TRACE_LISTENER_H_
