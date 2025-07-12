/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/pjrt/pjrt_compiler.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_device_description.h"

namespace xla {

namespace {
class PjRtTestTopology : public PjRtTopologyDescription {
 public:
  PjRtPlatformId platform_id() const override { return 0; }
  absl::string_view platform_name() const override { return "not_registered"; }
  absl::string_view platform_version() const override { return "test"; }
  std::vector<std::unique_ptr<const PjRtDeviceDescription>> DeviceDescriptions()
      const override {
    LOG(FATAL) << "Unused";
  }
  absl::StatusOr<std::string> Serialize() const override { return "test_topo"; }
  const absl::flat_hash_map<std::string, PjRtDeviceAttribute>& Attributes()
      const override {
    LOG(FATAL) << "Unused";
  }
  absl::StatusOr<Layout> GetDefaultLayout(
      PrimitiveType element_type,
      absl::Span<const int64_t> dims) const override {
    return Unimplemented("TestTopology does not support GetDefaultLayout");
  }
};

TEST(PjRtCompilerTest, CompilerNotRegistered) {
  PjRtTestTopology topology;

  CompileOptions options;
  XlaComputation computation;
  auto res = PjRtCompile(options, computation, topology);

  EXPECT_TRUE(absl::IsNotFound(res.status()));
}

TEST(PjRtCompilerTest, CompilerRegistered) {
  class PjRtTestTopology : public PjRtTopologyDescription {
   public:
    PjRtPlatformId platform_id() const override { return 0; }
    absl::string_view platform_name() const override { return "registered"; }
    absl::string_view platform_version() const override { return "test"; }
    std::vector<std::unique_ptr<const PjRtDeviceDescription>>
    DeviceDescriptions() const override {
      LOG(FATAL) << "Unused";
    }
    absl::StatusOr<std::string> Serialize() const override {
      return "test_topo";
    }
    const absl::flat_hash_map<std::string, PjRtDeviceAttribute>& Attributes()
        const override {
      LOG(FATAL) << "Unused";
    }
    absl::StatusOr<Layout> GetDefaultLayout(
        PrimitiveType element_type,
        absl::Span<const int64_t> dims) const override {
      return Unimplemented("TestTopology does not support GetDefaultLayout");
    }
  };
  PjRtTestTopology topology;

  class PjRtTestCompiler : public PjRtCompiler {
   public:
    absl::StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
        CompileOptions options, const XlaComputation& computation,
        const PjRtTopologyDescription& topology, PjRtClient* client) override {
      return tsl::errors::Unimplemented("test compiler!");
    }
    absl::StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
        CompileOptions options, mlir::ModuleOp module,
        const PjRtTopologyDescription& topology, PjRtClient* client) override {
      return tsl::errors::Unimplemented("test compiler!");
    }
  };
  std::unique_ptr<PjRtCompiler> compiler = std::make_unique<PjRtTestCompiler>();
  PjRtRegisterCompiler(topology.platform_name(), std::move(compiler));

  CompileOptions options;
  XlaComputation computation;
  auto res = PjRtCompile(options, computation, topology);

  EXPECT_TRUE(absl::IsUnimplemented(res.status()));
}

}  // namespace

}  // namespace xla
