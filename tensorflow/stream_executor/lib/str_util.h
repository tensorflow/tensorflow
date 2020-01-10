#ifndef TENSORFLOW_STREAM_EXECUTOR_LIB_STR_UTIL_H_
#define TENSORFLOW_STREAM_EXECUTOR_LIB_STR_UTIL_H_

#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/stream_executor/lib/stringpiece.h"

namespace perftools {
namespace gputools {
namespace port {

using tensorflow::str_util::Join;
using tensorflow::str_util::Split;

// Returns a copy of the input string 'str' with the given 'suffix'
// removed. If the suffix doesn't match, returns a copy of the original string.
inline string StripSuffixString(port::StringPiece str, port::StringPiece suffix) {
  if (str.ends_with(suffix)) {
    str.remove_suffix(suffix.size());
  }
  return str.ToString();
}

using tensorflow::str_util::Lowercase;
using tensorflow::str_util::Uppercase;

}  // namespace port
}  // namespace gputools
}  // namespace perftools

#endif  // TENSORFLOW_STREAM_EXECUTOR_LIB_STR_UTIL_H_
