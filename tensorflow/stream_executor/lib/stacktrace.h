#ifndef TENSORFLOW_STREAM_EXECUTOR_LIB_STACKTRACE_H_
#define TENSORFLOW_STREAM_EXECUTOR_LIB_STACKTRACE_H_

#include "tensorflow/stream_executor/platform/port.h"

namespace perftools {
namespace gputools {
namespace port {

#if !defined(PLATFORM_GOOGLE)
inline string CurrentStackTrace() { return "No stack trace available"; }
#endif

}  // namespace port
}  // namespace gputools
}  // namespace perftools

#endif  // TENSORFLOW_STREAM_EXECUTOR_LIB_STACKTRACE_H_
