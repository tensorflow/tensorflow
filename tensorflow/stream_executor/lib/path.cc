#include "tensorflow/stream_executor/lib/path.h"
#include "tensorflow/stream_executor/lib/strcat.h"

using ::perftools::gputools::port::StringPiece;
using ::perftools::gputools::port::StrAppend;

namespace perftools {
namespace gputools {
namespace port {
namespace internal {

static bool IsAbsolutePath(port::StringPiece path) {
  return !path.empty() && path[0] == '/';
}

// For an array of paths of length count, append them all together,
// ensuring that the proper path separators are inserted between them.
string JoinPathImpl(std::initializer_list<port::StringPiece> paths) {
  string result;

  for (port::StringPiece path : paths) {
    if (path.empty()) continue;

    if (result.empty()) {
      result = path.ToString();
      continue;
    }

    if (result[result.size() - 1] == '/') {
      if (IsAbsolutePath(path)) {
        StrAppend(&result, path.substr(1));
      } else {
        StrAppend(&result, path);
      }
    } else {
      if (IsAbsolutePath(path)) {
        StrAppend(&result, path);
      } else {
        StrAppend(&result, "/", path);
      }
    }
  }

  return result;
}

}  // namespace internal
}  // namespace port
}  // namespace gputools
}  // namespace perftools
