// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/litert/tools/apply_plugin.h"

#include <cstdint>
#include <memory>
#include <sstream>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/absl_check.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/core/build_stamp.h"
#include "tensorflow/lite/experimental/litert/core/dispatch_op_schema.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/test/common.h"
#include "tensorflow/lite/experimental/litert/test/matchers.h"

namespace litert::tools {
namespace {

using ::litert::internal::kLiteRtBuildStampKey;
using ::litert::internal::ParseBuildStamp;
using ::testing::HasSubstr;
using ::testing::litert::IsError;

static constexpr absl::string_view kPluginSearchPath =
    "third_party/tensorflow/lite/experimental/litert/vendors/examples";

static constexpr absl::string_view kSocManufacturer = "ExampleSocManufacturer";

static constexpr absl::string_view kSocModel = "ExampleSocModel";

absl::string_view TestModelPath() {
  static char kModelPath[512] = {};
  if (kModelPath[0] == '\0') {
    const auto model_path =
        ::litert::testing::GetTestFilePath("one_mul.tflite");
    ABSL_CHECK(model_path.size() < 512);
    model_path.copy(kModelPath, model_path.size(), 0);
  }
  return kModelPath;
}

ApplyPluginRun::Ptr MakeBaseRun(ApplyPluginRun::Cmd cmd) {
  auto run = std::make_unique<ApplyPluginRun>();
  run->cmd = cmd;
  run->lib_search_paths.push_back(kPluginSearchPath);
  run->model.emplace(TestModelPath());
  run->soc_manufacturer.emplace(kSocManufacturer);
  run->soc_models.push_back(kSocModel);
  run->outs.clear();
  return run;
}

TEST(TestApplyPluginTool, TestInfoBadConfig) {
  auto run = MakeBaseRun(ApplyPluginRun::Cmd::INFO);
  run->lib_search_paths.clear();
  EXPECT_THAT(ApplyPlugin(std::move(run)),
              IsError(kLiteRtStatusErrorInvalidToolConfig));
}

TEST(TestApplyPluginTool, TestInfo) {
  auto run = MakeBaseRun(ApplyPluginRun::Cmd::INFO);
  std::stringstream out;
  run->outs.push_back(out);
  LITERT_ASSERT_OK(ApplyPlugin(std::move(run)));
  EXPECT_THAT(out.str(),
              ::testing::HasSubstr(
                  "< LiteRtCompilerPlugin > \"ExampleSocManufacturer\" | "
                  "\"ExampleSocModel\""));
}

TEST(TestApplyPluginTool, TestNoopBadConfig) {
  auto run = MakeBaseRun(ApplyPluginRun::Cmd::NOOP);
  run->model.reset();
  EXPECT_THAT(ApplyPlugin(std::move(run)),
              IsError(kLiteRtStatusErrorInvalidToolConfig));
}

TEST(TestApplyPluginTool, TestNoop) {
  auto run = MakeBaseRun(ApplyPluginRun::Cmd::NOOP);
  std::stringstream out;
  run->outs.push_back(out);
  LITERT_ASSERT_OK(ApplyPlugin(std::move(run)));

  auto model = Model::CreateFromBuffer(
      BufferRef<uint8_t>(out.view().data(), out.view().size()));
  EXPECT_EQ(model->Get()->NumSubgraphs(), 1);
}

TEST(TestApplyPluginTool, TestPartitionBadConfig) {
  auto run = MakeBaseRun(ApplyPluginRun::Cmd::PARTITION);
  run->model.reset();
  EXPECT_THAT(ApplyPlugin(std::move(run)),
              IsError(kLiteRtStatusErrorInvalidToolConfig));
}

TEST(TestApplyPluginTool, TestPartition) {
  auto run = MakeBaseRun(ApplyPluginRun::Cmd::PARTITION);
  std::stringstream out;
  run->outs.push_back(out);
  LITERT_ASSERT_OK(ApplyPlugin(std::move(run)));
  EXPECT_FALSE(out.str().empty());
}

TEST(TestApplyPluginTool, TestCompileBadConfig) {
  auto run = MakeBaseRun(ApplyPluginRun::Cmd::COMPILE);
  run->model.reset();
  EXPECT_THAT(ApplyPlugin(std::move(run)),
              IsError(kLiteRtStatusErrorInvalidToolConfig));
}

TEST(TestApplyPluginTool, TestCompile) {
  auto run = MakeBaseRun(ApplyPluginRun::Cmd::COMPILE);
  std::stringstream out;
  run->outs.push_back(out);
  LITERT_ASSERT_OK(ApplyPlugin(std::move(run)));
  EXPECT_FALSE(out.str().empty());
  EXPECT_THAT(out.str(), HasSubstr("Partition_0_with_1_muls"));
}

TEST(TestApplyPluginTool, TestApplyBadConfig) {
  auto run = MakeBaseRun(ApplyPluginRun::Cmd::APPLY);
  run->model.reset();
  EXPECT_THAT(ApplyPlugin(std::move(run)),
              IsError(kLiteRtStatusErrorInvalidToolConfig));
}

TEST(TestApplyPluginTool, TestApply) {
  auto run = MakeBaseRun(ApplyPluginRun::Cmd::APPLY);
  std::stringstream out;
  run->outs.push_back(out);
  LITERT_ASSERT_OK(ApplyPlugin(std::move(run)));

  const auto out_str = out.str();
  BufferRef<uint8_t> serialized(out_str.data(), out_str.size());

  auto model = Model::CreateFromBuffer(serialized);
  EXPECT_EQ(model->Get()->NumSubgraphs(), 1);

  {
    auto stamp_buffer = model->Get()->FindMetadata(kLiteRtBuildStampKey);
    auto stamp = ParseBuildStamp(*stamp_buffer);
    auto [man, soc_model] = *stamp;
    EXPECT_EQ(man, kSocManufacturer);
    EXPECT_EQ(soc_model, kSocModel);
  }

  auto* op = model->Get()->MainSubgraph()->Ops().front();
  ASSERT_EQ(op->OpCode(), kLiteRtOpCodeTflCustom);

  const auto options = internal::GetDispatchOpOptions(op->CustomOptions());
  const auto& [size, offset, name] = options;
  EXPECT_EQ(name, "Partition_0");
  ASSERT_LE(offset + size, serialized.Size());

  EXPECT_THAT(serialized.StrView().substr(offset, size),
              HasSubstr("Partition_0_with_1_muls"));
}

}  // namespace
}  // namespace litert::tools
