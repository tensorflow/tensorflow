/* Copyright 2026 The OpenXLA Authors.

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

// GpuCompiler loads autotune results from a file at most once per process, so
// this test lives in its own binary to make sure the load actually happens.

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/autotune_cache.pb.h"
#include "xla/backends/core/autotuner/backends.pb.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/platform/env.h"
#include "xla/xla.pb.h"
#include "tsl/platform/path.h"
#include "tsl/platform/protobuf.h"

namespace xla::gpu {
namespace {

constexpr absl::string_view kHloText = R"(
HloModule t

ENTRY e {
  p0 = f16[1,16,17,3] parameter(0)
  p1 = s8[16,17,3] parameter(1)
  cp1 = f16[16,17,3] convert(p1)
  ROOT _ = f16[1,16,16] dot(p0, cp1),
    lhs_contracting_dims={2,3}, rhs_contracting_dims={1,2}
})";

class LoadAutotuneCacheTest : public HloTestBase {
 protected:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions options = HloTestBase::GetDebugOptionsForTest();
    options.set_xla_gpu_use_new_autotune_cache_format(true);
    options.set_xla_gpu_load_autotune_results_from(
        deprecated_load_autotune_results_from_);
    options.set_xla_load_autotune_cache_from(load_autotune_cache_from_);
    return options;
  }

  std::string deprecated_load_autotune_results_from_;
  std::string load_autotune_cache_from_;
};

TEST_F(LoadAutotuneCacheTest, LoadAutotuneCacheFromTakesPrecedence) {
  autotuner::AutotuneCache cache;
  auto* entry = cache.add_entries();
  entry->mutable_key()->mutable_target()->set_device("test_device");
  entry->mutable_key()->mutable_target()->set_hlo_fingerprint("test_hlo");
  entry->mutable_value()->mutable_optimal_config()->set_backend(
      autotuner::CUBLASLT_FISSION);
  std::string serialized;
  ASSERT_TRUE(tsl::protobuf::TextFormat::PrintToString(cache, &serialized));

  load_autotune_cache_from_ =
      tsl::io::JoinPath(::testing::TempDir(), "autotune_cache.txt");
  ASSERT_OK(tsl::WriteStringToFile(tsl::Env::Default(),
                                   load_autotune_cache_from_, serialized));
  // Loading from the deprecated path would fail compilation, since the file
  // does not exist.
  deprecated_load_autotune_results_from_ =
      tsl::io::JoinPath(::testing::TempDir(), "does_not_exist.txt");

  EXPECT_OK(GetOptimizedModule(kHloText).status());
}

}  // namespace
}  // namespace xla::gpu
