/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/op_requires.h"

#include <optional>
#include <utility>

#include "absl/status/status.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

using ::tensorflow::testing::StatusIs;
using ::testing::Optional;

class Holder {
 public:
  explicit Holder()
      : fine_(absl::OkStatus()), foul_(absl::InternalError("test")) {}

  const absl::Status& Fine() const { return fine_; }
  const absl::Status& Foul() const { return foul_; }

 private:
  absl::Status fine_;
  absl::Status foul_;
};

struct TestContext {
 public:
  void CtxFailureWithWarning(const char* file, int line, absl::Status status) {
    stored_status.emplace(std::move(status));
  }

  friend void CheckNotInComputeAsync(TestContext* ctx, const char* msg) {}

  std::optional<absl::Status> stored_status = std::nullopt;
};

void TestFunction(TestContext& ctx, bool success, bool& reached) {
  if (success) {
    OP_REQUIRES_OK(&ctx, Holder().Fine());
  } else {
    OP_REQUIRES_OK(&ctx, Holder().Foul());
  }
  reached = true;
}

TEST(OpRequires, RequiresOkWithOkStatus) {
  TestContext ctx;
  bool reached = false;

  TestFunction(ctx, /*success=*/true, reached);
  EXPECT_FALSE(ctx.stored_status.has_value());
  EXPECT_TRUE(reached);
}

TEST(OpRequires, RequiresOkWithFailedStatus) {
  TestContext ctx;
  bool reached = false;

  TestFunction(ctx, /*success=*/false, reached);
  EXPECT_THAT(ctx.stored_status,
              Optional(StatusIs(absl::StatusCode::kInternal)));
  EXPECT_FALSE(reached);
}

void TestFunctionAsync(TestContext& ctx, bool success, bool& reached,
                       bool& handled) {
  auto done = gtl::MakeCleanup([&handled]() { handled = true; });
  if (success) {
    OP_REQUIRES_OK_ASYNC(&ctx, Holder().Fine(), done.release());
  } else {
    OP_REQUIRES_OK_ASYNC(&ctx, Holder().Foul(), done.release());
  }
  reached = true;
}

TEST(OpRequires, RequiresOkAsyncWithOkStatus) {
  TestContext ctx;
  bool reached = false;
  bool handled = false;

  TestFunctionAsync(ctx, /*success=*/true, reached, handled);
  EXPECT_FALSE(ctx.stored_status.has_value());
  EXPECT_TRUE(reached);
  EXPECT_TRUE(handled);
}

TEST(OpRequires, RequiresOkAsyncWithFailedStatus) {
  TestContext ctx;
  bool reached = false;
  bool handled = false;

  TestFunctionAsync(ctx, /*success=*/false, reached, handled);
  EXPECT_THAT(ctx.stored_status,
              Optional(StatusIs(absl::StatusCode::kInternal)));
  EXPECT_FALSE(reached);
  EXPECT_TRUE(handled);
}

}  // namespace
}  // namespace tensorflow
