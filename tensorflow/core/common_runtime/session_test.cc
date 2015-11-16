#include "tensorflow/core/public/session.h"

#include <gtest/gtest.h>
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace {

TEST(SessionTest, InvalidTargetReturnsNull) {
  SessionOptions options;
  options.target = "invalid target";

  EXPECT_EQ(nullptr, tensorflow::NewSession(options));
}

}  // namespace
}  // namespace tensorflow
