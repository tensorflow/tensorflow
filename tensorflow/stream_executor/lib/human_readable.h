#ifndef TENSORFLOW_STREAM_EXECUTOR_LIB_HUMAN_READABLE_H_
#define TENSORFLOW_STREAM_EXECUTOR_LIB_HUMAN_READABLE_H_

#include <assert.h>
#include <limits>

#include "tensorflow/stream_executor/lib/stringprintf.h"
#include "tensorflow/stream_executor/platform/port.h"

namespace perftools {
namespace gputools {
namespace port {

class HumanReadableNumBytes {
 public:
  static string ToString(int64 num_bytes) {
    if (num_bytes == std::numeric_limits<int64>::min()) {
      // Special case for number with not representable nagation.
      return "-8E";
    }

    const char* neg_str = GetNegStr(&num_bytes);

    // Special case for bytes.
    if (num_bytes < 1024LL) {
      // No fractions for bytes.
      return port::Printf("%s%lldB", neg_str, num_bytes);
    }

    static const char units[] = "KMGTPE";  // int64 only goes up to E.
    const char* unit = units;
    while (num_bytes >= (1024LL) * (1024LL)) {
      num_bytes /= (1024LL);
      ++unit;
      assert(unit < units + sizeof(units));
    }

    return port::Printf(((*unit == 'K') ? "%s%.1f%c" : "%s%.2f%c"), neg_str,
                        num_bytes / 1024.0, *unit);
  }

 private:
  template <typename T>
  static const char* GetNegStr(T* value) {
    if (*value < 0) {
      *value = -(*value);
      return "-";
    } else {
      return "";
    }
  }
};

}  // namespace port
}  // namespace gputools
}  // namespace perftools

#endif  // TENSORFLOW_STREAM_EXECUTOR_LIB_HUMAN_READABLE_H_
