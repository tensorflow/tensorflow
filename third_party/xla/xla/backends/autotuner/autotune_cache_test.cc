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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "xla/backends/autotuner/autotuner_cache_interface.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/autotuner/fake_codegen_backend.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace {

AutotuneCacheContext CreateCacheContext(
    std::string explicit_version = "v1.0", int device_core_count = 108,
    std::vector<std::pair<autotuner::Backend, std::string>> backends_info = {
        {autotuner::Backend::TRITON, "triton_v1"}}) {
  stream_executor::DeviceDescription device_description;
  device_description.set_name("test_gpu");
  device_description.set_core_count(device_core_count);
  device_description.set_clock_rate_ghz(1.41);
  device_description.set_memory_bandwidth(1555000000000);
  device_description.set_l2_cache_size(41943040);

  stream_executor::CudaComputeCapability cuda_cc(8, 0);
  stream_executor::GpuComputeCapability gpu_cc(cuda_cc);
  device_description.set_gpu_compute_capability(gpu_cc);

  std::vector<std::unique_ptr<CodegenBackend>> backends;
  for (const auto& [backend, version] : backends_info) {
    backends.push_back(std::make_unique<FakeCodegenBackend>(backend, version));
  }

  return AutotuneCacheContext::Create(device_description, backends,
                                      std::move(explicit_version));
}

TEST(AutotuneCacheContextTest, Create) {
  AutotuneCacheContext context = CreateCacheContext(
      "my_explicit_version", 108, {{autotuner::Backend::TRITON, "1.2.3"}});

  EXPECT_EQ(context.device().size(), 16);
  EXPECT_NE(context.device(), "unknown");
  EXPECT_EQ(context.explicit_version(), "my_explicit_version");
  EXPECT_NE(context.codegen_version(), "");
  EXPECT_NE(context.codegen_version(), "unknown");
  EXPECT_EQ(context.per_backend_versions().at(autotuner::Backend::TRITON),
            "1.2.3");
}

TEST(AutotuneCacheContextTest, CreateWithDifferentDeviceSpec) {
  AutotuneCacheContext context1 = CreateCacheContext("v1.0", 128);
  AutotuneCacheContext context2 = CreateCacheContext("v1.0", 108);
  EXPECT_NE(context1.device(), context2.device());
  EXPECT_EQ(context1.explicit_version(), context2.explicit_version());
  EXPECT_EQ(context1.codegen_version(), context2.codegen_version());
  EXPECT_NE(context1, context2);
}

TEST(AutotuneCacheContextTest, CreateWithDifferentExplicitVersion) {
  AutotuneCacheContext context1 = CreateCacheContext("v1.0");
  AutotuneCacheContext context2 = CreateCacheContext("v1.1");
  EXPECT_EQ(context1.device(), context2.device());
  EXPECT_NE(context1.explicit_version(), context2.explicit_version());
  EXPECT_EQ(context1.codegen_version(), context2.codegen_version());
  EXPECT_NE(context1, context2);
}

TEST(AutotuneCacheContextTest, CreateWithDifferentPerBackendVersions) {
  AutotuneCacheContext context1 =
      CreateCacheContext("v1.0", 108, {{autotuner::Backend::TRITON, "1.2.3"}});
  AutotuneCacheContext context2 =
      CreateCacheContext("v1.0", 108, {{autotuner::Backend::TRITON, "1.2.4"}});
  EXPECT_EQ(context1.device(), context2.device());
  EXPECT_EQ(context1.explicit_version(), context2.explicit_version());
  EXPECT_NE(context1.codegen_version(), context2.codegen_version());
  EXPECT_NE(context1, context2);
}

}  // namespace
}  // namespace xla
