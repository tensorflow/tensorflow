/* Copyright 2023 The OpenXLA Authors.

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
#include <optional>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/gpu/autotuning/autotuner_util.h"
#include "xla/service/gpu/gpu_symbol_repository.h"
#include "xla/service/platform_util.h"
#include "xla/service/symbol_repository.h"
#include "xla/service/xla_compile_result.pb.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tools/xla_compile_lib.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/protobuf/error_codes.pb.h"
#include "xla/tsl/protobuf/status.pb.h"
#include "xla/util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/path.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

using ::testing::IsEmpty;
using ::testing::Not;
using ::tsl::testing::IsOkAndHolds;

class XlaCompileLibTest : public HloTestBase {
 protected:
  XlaCompileLibTest()
      : HloTestBase(*PlatformUtil::GetPlatform(std::string("GPU")),
                    GetReferencePlatform()) {}
  void SetUp() override {
    const std::string hlo_path = tsl::io::JoinPath(tsl::testing::XlaSrcRoot(),
                                                   "tools", "data", "add.hlo");
    std::string hlo;
    TF_ASSERT_OK(tsl::ReadFileToString(tsl::Env::Default(), hlo_path, &hlo));
    TF_ASSERT_OK_AND_ASSIGN(module_, ParseAndReturnVerifiedModule(hlo));
  }

  std::unique_ptr<HloModule> module_;
};

TEST_F(XlaCompileLibTest, CompilesForGpuWithDevice) {
  CompilationResult result;
  EXPECT_THAT(CompileExecutable(std::move(module_), BackendType::kGpu,
                                std::nullopt, result),
              IsOkAndHolds(Not(IsEmpty())));
  EXPECT_TRUE(result.has_hlo_module()) << result.DebugString();
}

TEST_F(XlaCompileLibTest, CompilesForGpuWithoutDevice) {
  const std::string target_config_path =
      tsl::io::JoinPath(tsl::testing::XlaSrcRoot(), "service",
                        "xla_aot_compile_test_gpu_target_config.prototxt");
  stream_executor::GpuTargetConfigProto target_config;
  TF_ASSERT_OK(tsl::ReadTextProto(tsl::Env::Default(), target_config_path,
                                  &target_config));
  CompilationResult result;
  EXPECT_THAT(CompileExecutable(std::move(module_), BackendType::kGpu,
                                std::nullopt, result),
              IsOkAndHolds(Not(IsEmpty())));
  EXPECT_TRUE(result.has_hlo_module()) << result.DebugString();
}

TEST_F(XlaCompileLibTest, MainForGpu) {
  const std::string module_file =
      tsl::io::JoinPath(tsl::testing::TmpDir(), "module.txt");
  TF_ASSERT_OK(tsl::WriteStringToFile(tsl::Env::Default(), module_file,
                                      module_->ToString()));

  const std::string output_path =
      tsl::io::JoinPath(tsl::testing::TmpDir(), "gpu_output");
  const std::string result_file =
      tsl::io::JoinPath(tsl::testing::TmpDir(), "gpu_result.pb");

  XlaCompileOptions options;
  options.module_path = module_file;
  options.output_path = output_path;
  options.platform = "gpu";
  options.result_output_file = result_file;
  options.gpu_options.use_attached_device = true;
  TF_EXPECT_OK(XlaCompileMain(options));

  CompilationResult result;
  TF_ASSERT_OK(tsl::ReadBinaryProto(tsl::Env::Default(), result_file, &result));
  EXPECT_TRUE(result.has_status());
  EXPECT_EQ(result.status().code(), tensorflow::error::OK);
}

TEST_F(XlaCompileLibTest, LoadAutotuneDataGpuDataPresentAndAutotuningEnabled) {
  gpu::AutotunerUtil::ClearAutotuneResults();

  HloModuleAndMetadata mod;
  mod.hlo_module = std::move(module_);
  auto data = std::make_unique<gpu::GpuBackendSpecificData>();

  AutotuneResults autotune_results;
  TF_ASSERT_OK(tsl::ReadTextProto(
      tsl::Env::Default(),
      tsl::io::JoinPath(tsl::testing::XlaSrcRoot(), "service", "gpu",
                        "gpu_compiler_test_autotune_db.textproto"),
      &autotune_results));
  data->autotune_results = autotune_results;
  mod.backend_specific_data = std::move(data);

  DebugOptions opts = mod.hlo_module->config().debug_options();
  opts.set_xla_gpu_autotune_level(3);
  mod.hlo_module->mutable_config().set_debug_options(opts);

  EXPECT_THAT(internal::LoadAutotuneDataFromModule(&mod, BackendType::kGpu),
              IsOkAndHolds(true));
  EXPECT_FALSE(gpu::AutotunerUtil::ResultCacheIsEmpty());
}

TEST_F(XlaCompileLibTest, LoadAutotuneDataGpuDataPresentAndAutotuningDisabled) {
  gpu::AutotunerUtil::ClearAutotuneResults();

  HloModuleAndMetadata mod;
  mod.hlo_module = std::move(module_);
  auto data = std::make_unique<gpu::GpuBackendSpecificData>();

  AutotuneResults autotune_results;
  TF_ASSERT_OK(tsl::ReadTextProto(
      tsl::Env::Default(),
      tsl::io::JoinPath(tsl::testing::XlaSrcRoot(), "service", "gpu",
                        "gpu_compiler_test_autotune_db.textproto"),
      &autotune_results));
  data->autotune_results = autotune_results;
  mod.backend_specific_data = std::move(data);

  DebugOptions opts = mod.hlo_module->config().debug_options();
  opts.set_xla_gpu_autotune_level(0);
  mod.hlo_module->mutable_config().set_debug_options(opts);

  EXPECT_THAT(internal::LoadAutotuneDataFromModule(&mod, BackendType::kGpu),
              IsOkAndHolds(false));
  EXPECT_TRUE(gpu::AutotunerUtil::ResultCacheIsEmpty());
}

TEST_F(XlaCompileLibTest,
       LoadAutotuneDataGpuDataNotPresentAndAutotuningEnabled) {
  gpu::AutotunerUtil::ClearAutotuneResults();

  HloModuleAndMetadata mod;
  mod.hlo_module = std::move(module_);

  DebugOptions opts = mod.hlo_module->config().debug_options();
  opts.set_xla_gpu_autotune_level(3);
  mod.hlo_module->mutable_config().set_debug_options(opts);

  EXPECT_THAT(internal::LoadAutotuneDataFromModule(&mod, BackendType::kGpu),
              IsOkAndHolds(false));
  EXPECT_TRUE(gpu::AutotunerUtil::ResultCacheIsEmpty());
}

TEST_F(XlaCompileLibTest,
       LoadAutotuneDataGpuDataNotPresentAndAutotuningDisabled) {
  gpu::AutotunerUtil::ClearAutotuneResults();

  HloModuleAndMetadata mod;
  mod.hlo_module = std::move(module_);

  DebugOptions opts = mod.hlo_module->config().debug_options();
  opts.set_xla_gpu_autotune_level(0);
  mod.hlo_module->mutable_config().set_debug_options(opts);

  EXPECT_THAT(internal::LoadAutotuneDataFromModule(&mod, BackendType::kGpu),
              IsOkAndHolds(false));
  EXPECT_TRUE(gpu::AutotunerUtil::ResultCacheIsEmpty());
}

}  // namespace
}  // namespace xla
