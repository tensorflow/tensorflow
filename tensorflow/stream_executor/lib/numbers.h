#ifndef TENSORFLOW_STREAM_EXECUTOR_LIB_NUMBERS_H_
#define TENSORFLOW_STREAM_EXECUTOR_LIB_NUMBERS_H_

#include "tensorflow/stream_executor/platform/port.h"

namespace perftools {
namespace gputools {
namespace port {

// Convert strings to floating point values.
// Leading and trailing spaces are allowed.
// Values may be rounded on over- and underflow.
bool safe_strto32(const string& str, int32* value);

}  // namespace port
}  // namespace gputools
}  // namespace perftools

#endif  // TENSORFLOW_STREAM_EXECUTOR_LIB_NUMBERS_H_
