#include "tensorflow/stream_executor/lib/process_state.h"

#include <unistd.h>

#include <memory>

namespace perftools {
namespace gputools {
namespace port {

string Hostname() {
  char hostname[1024];
  gethostname(hostname, sizeof hostname);
  hostname[sizeof hostname - 1] = 0;
  return hostname;
}

bool GetCurrentDirectory(string* dir) {
  size_t len = 128;
  std::unique_ptr<char[]> a(new char[len]);
  for (;;) {
    char* p = getcwd(a.get(), len);
    if (p != NULL) {
      *dir = p;
      return true;
    } else if (errno == ERANGE) {
      len += len;
      a.reset(new char[len]);
    } else {
      return false;
    }
  }
}

}  // namespace port
}  // namespace gputools
}  // namespace perftools
