#include "tensorflow/core/platform/tracing.h"

#include <unistd.h>

namespace tensorflow {
namespace port {

void Tracing::RegisterEvent(EventCategory id, const char* name) {
  // TODO(opensource): implement
}

void Tracing::Initialize() {}

static bool TryGetEnv(const char* name, const char** value) {
  *value = getenv(name);
  return *value != nullptr && (*value)[0] != '\0';
}

const char* Tracing::LogDir() {
  const char* dir;
  if (TryGetEnv("TEST_TMPDIR", &dir)) return dir;
  if (TryGetEnv("TMP", &dir)) return dir;
  if (TryGetEnv("TMPDIR", &dir)) return dir;
  dir = "/tmp";
  if (access(dir, R_OK | W_OK | X_OK) == 0) return dir;
  return ".";  // Default to current directory.
}

static bool DoInit() {
  Tracing::Initialize();
  return true;
}

static const bool dummy = DoInit();

}  // namespace port
}  // namespace tensorflow
