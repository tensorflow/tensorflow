#include "tensorflow/stream_executor/lib/numbers.h"

#include <stdlib.h>

namespace perftools {
namespace gputools {
namespace port {

bool safe_strto32(const char* str, int32* value) {
  char* endptr;
  *value = strtol(str, &endptr, 10);  // NOLINT
  if (endptr != str) {
    while (isspace(*endptr)) ++endptr;
  }
  return *str != '\0' && *endptr == '\0';
}

// Convert strings to floating point values.
// Leading and trailing spaces are allowed.
// Values may be rounded on over- and underflow.
bool safe_strto32(const string& str, int32* value) {
  return port::safe_strto32(str.c_str(), value);
}

}  // namespace port
}  // namespace gputools
}  // namespace perftools
