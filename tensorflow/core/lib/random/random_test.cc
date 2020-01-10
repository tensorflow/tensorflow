#include "tensorflow/core/lib/random/random.h"

#include <set>
#include <gtest/gtest.h>
#include "tensorflow/core/platform/port.h"

namespace tensorflow {
namespace random {
namespace {

TEST(New64Test, SanityCheck) {
  std::set<uint64> values;
  for (int i = 0; i < 1000000; i++) {
    uint64 x = New64();
    EXPECT_TRUE(values.insert(x).second) << "duplicate " << x;
  }
}

}  // namespace
}  // namespace random
}  // namespace tensorflow
