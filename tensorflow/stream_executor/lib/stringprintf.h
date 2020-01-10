#ifndef TENSORFLOW_STREAM_EXECUTOR_LIB_STRINGPRINTF_H_
#define TENSORFLOW_STREAM_EXECUTOR_LIB_STRINGPRINTF_H_

#include "tensorflow/core/lib/strings/stringprintf.h"

namespace perftools {
namespace gputools {
namespace port {

using tensorflow::strings::Printf;
using tensorflow::strings::Appendf;
using tensorflow::strings::Appendv;

}  // namespace port
}  // namespace gputools
}  // namespace perftools

#endif  // TENSORFLOW_STREAM_EXECUTOR_LIB_STRINGPRINTF_H_
