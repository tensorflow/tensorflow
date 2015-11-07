#include "tensorflow/core/platform/port.h"

#if defined(PLATFORM_GOOGLE) || defined(PLATFORM_POSIX_ANDROID) || \
    defined(PLATFORM_GOOGLE_ANDROID)
#include "testing/base/public/googletest.h"
#endif

namespace tensorflow {
namespace testing {

#if defined(PLATFORM_GOOGLE) || defined(PLATFORM_POSIX_ANDROID) || \
    defined(PLATFORM_GOOGLE_ANDROID)
string TmpDir() { return FLAGS_test_tmpdir; }
int RandomSeed() { return FLAGS_test_random_seed; }
#else
string TmpDir() {
  // 'bazel test' sets TEST_TMPDIR
  const char* env = getenv("TEST_TMPDIR");
  if (env && env[0] != '\0') {
    return env;
  }
  env = getenv("TMPDIR");
  if (env && env[0] != '\0') {
    return env;
  }
  return "/tmp";
}
int RandomSeed() {
  const char* env = getenv("TEST_RANDOM_SEED");
  int result;
  if (env && sscanf(env, "%d", &result) == 1) {
    return result;
  }
  return 301;
}
#endif

}  // namespace testing
}  // namespace tensorflow
