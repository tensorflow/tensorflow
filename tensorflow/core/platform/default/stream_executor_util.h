#ifndef TENSORFLOW_PLATFORM_DEFAULT_STREAM_EXECUTOR_UTIL_H_
#define TENSORFLOW_PLATFORM_DEFAULT_STREAM_EXECUTOR_UTIL_H_

#include "tensorflow/stream_executor/lib/status.h"

namespace tensorflow {

namespace gpu = ::perftools::gputools;

// On the open-source platform, stream_executor currently uses
// tensorflow::Status
inline Status FromStreamExecutorStatus(
    const perftools::gputools::port::Status& s) {
  return s;
}

}  // namespace tensorflow

#endif  // TENSORFLOW_PLATFORM_DEFAULT_STREAM_EXECUTOR_UTIL_H_
