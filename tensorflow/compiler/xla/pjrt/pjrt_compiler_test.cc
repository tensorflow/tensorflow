/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/pjrt/pjrt_compiler.h"

#include <memory>
#include <utility>

#include <gtest/gtest.h>
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"

namespace xla {

namespace {

TEST(PjRtCompilerTest, CompilerNotRegistered) {
  class PjRtTestTopology : public PjRtTopologyDescription {
   public:
    PjRtPlatformId platform_id() const override { return 0; }
    absl::string_view platform_name() const override {
      return "not_registered";
    }
    absl::string_view platform_version() const override { return "test"; }
  };
  PjRtTestTopology topology;

  CompileOptions options;
  XlaComputation computation;
  auto res = PjRtCompile(options, computation, topology);

  EXPECT_TRUE(tsl::errors::IsNotFound(res.status()));
}

TEST(PjRtCompilerTest, CompilerRegistered) {
  class PjRtTestTopology : public PjRtTopologyDescription {
   public:
    PjRtPlatformId platform_id() const override { return 0; }
    absl::string_view platform_name() const override { return "registered"; }
    absl::string_view platform_version() const override { return "test"; }
  };
  PjRtTestTopology topology;

  class PjRtTestCompiler : public PjRtCompiler {
   public:
    StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
        CompileOptions options, const XlaComputation& computation,
        const PjRtTopologyDescription& topology, PjRtClient* client) override {
      return tsl::errors::Unimplemented("test compiler!");
    }
    StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
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

  EXPECT_TRUE(tsl::errors::IsUnimplemented(res.status()));
}

}  // namespace

}  // namespace xla
