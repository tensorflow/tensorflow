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

#include "xla/backends/gpu/target_config/cudnn_device_props.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "third_party/cudnn_frontend/include/cudnn_frontend/backend/device_properties.h"
#include "third_party/cudnn_frontend/include/cudnn_frontend_shim.h"
#include "json/json.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/test.h"

namespace xla::gpu {
namespace {

namespace se = ::stream_executor;

// Fields that BuildDeviceProperties() hardcodes (or that don't affect cuDNN
// plan selection), so they are not expected to match the live device.
absl::string_view kIgnoredFields[] = {
    "pciDeviceId",   // hardcoded to 0 in synthesized output
    "cudaDeviceId",  // hardcoded to 0 in synthesized output
    "isTccDriver",   // hardcoded to 0 in synthesized output
    "driverVer",     // populated from desc.driver_version(), live cuDNN reads
                     // its own value — encoding can differ
    "deviceName",    // not serialized by live cuDNN
    "maxThreadsPerBlock",  // not serialized by live cuDNN
    "smCountOrigin",  // TODO(b/530828838): Ask NVIDIA about whether this should
                      // be serialized.
};

// Clock-rate fields where small drift is expected because XLA stores them as
// float GHz and converts back to int KHz, while cuDNN reads int KHz directly.
absl::string_view kClockFields[] = {
    "smClockRateKHz",
    "memClockRateKHz",
};

constexpr int64_t kClockToleranceKHz = 1000;

// The device memory size depends on the driver version, so the live device may
// have slightly more memory than the spec (same tolerance as
// embedded_target_config_test).
constexpr int64_t kTotalGlobalMemToleranceBytes = 1024 * 1024 * 1024;

// True if the running cuDNN runtime supports DEVICEPROP serialization
// (added in 9.8). Older runtimes will fail set_device_id().build().
bool CudnnSupportsDeviceProps() {
  return cudnn_frontend::detail::get_backend_version() >= 90800;
}

Json::Value ParseProps(
    const std::shared_ptr<cudnn_frontend::DeviceProperties>& props) {
  std::vector<uint8_t> buf;
  auto err = props->serialize(buf);
  EXPECT_FALSE(err.is_bad()) << err.get_message();
  std::string str(buf.begin(), buf.end());
  Json::Value value;
  Json::CharReaderBuilder builder;
  std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
  std::string errors;
  EXPECT_TRUE(
      reader->parse(str.data(), str.data() + str.size(), &value, &errors))
      << errors;
  return value;
}

void StripIgnoredFields(Json::Value& v) {
  for (absl::string_view field : kIgnoredFields) {
    v.removeMember(field.data());
  }
}

void ExpectClockFieldsClose(const Json::Value& live, const Json::Value& synth) {
  for (absl::string_view field : kClockFields) {
    if (!live.isMember(field.data()) || !synth.isMember(field.data())) {
      continue;
    }
    int64_t live_val = live[field.data()].asInt64();
    int64_t synth_val = synth[field.data()].asInt64();
    EXPECT_LE(std::abs(live_val - synth_val), kClockToleranceKHz)
        << field << ": live=" << live_val << " synth=" << synth_val;
  }
}

std::string Dump(const Json::Value& v) {
  Json::StreamWriterBuilder builder;
  builder["indentation"] = "  ";
  return Json::writeString(builder, v);
}

void ExpectPropsMatch(Json::Value live_json, Json::Value synth_json,
                      const se::DeviceDescription& desc) {
  VLOG(1) << "live : " << Dump(live_json);
  VLOG(1) << "synth: " << Dump(synth_json);

  constexpr char kOversizedSharedMemoryField[] =
      "oversizedSharedMemoryPerBlock";
  const int64_t live_oversized_shared_memory =
      live_json.get(kOversizedSharedMemoryField, 0).asInt64();
  const int64_t synth_oversized_shared_memory =
      synth_json.get(kOversizedSharedMemoryField, 0).asInt64();
  const auto* cc = desc.gpu_compute_capability().cuda_compute_capability();
  const bool is_sm107 = cc && cc->major == 10 && cc->minor == 7;
  EXPECT_EQ(synth_oversized_shared_memory,
            is_sm107 ? desc.oversized_shared_memory_per_block() : 0);
  // cuDNN may omit this property or report zero.
  if (live_oversized_shared_memory != 0) {
    EXPECT_EQ(synth_oversized_shared_memory, live_oversized_shared_memory);
  }
  live_json.removeMember(kOversizedSharedMemoryField);
  synth_json.removeMember(kOversizedSharedMemoryField);

  StripIgnoredFields(live_json);
  StripIgnoredFields(synth_json);
  ExpectClockFieldsClose(live_json, synth_json);
  for (absl::string_view field : kClockFields) {
    live_json.removeMember(field.data());
    synth_json.removeMember(field.data());
  }

  constexpr char kTotalGlobalMemField[] = "totalGlobalMem";
  const int64_t live_memory = live_json[kTotalGlobalMemField].asInt64();
  const int64_t synth_memory = synth_json[kTotalGlobalMemField].asInt64();
  // The spec must not promise more memory than the device actually has.
  EXPECT_LE(synth_memory, live_memory);
  EXPECT_LE(live_memory - synth_memory, kTotalGlobalMemToleranceBytes);
  live_json.removeMember(kTotalGlobalMemField);
  synth_json.removeMember(kTotalGlobalMemField);

  EXPECT_EQ(live_json, synth_json);
}

TEST(CudnnDevicePropsTest, MatchesLiveDevice) {
  if (!CudnnSupportsDeviceProps()) {
    GTEST_SKIP() << "cuDNN runtime < 9.8 does not support DeviceProperties "
                    "JSON serialization.";
  }

  ASSERT_OK_AND_ASSIGN(std::string platform_name,
                       PlatformUtil::CanonicalPlatformName("gpu"));
  std::string name = absl::AsciiStrToUpper(platform_name);
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName(name));

  bool any_compared = false;
  for (int i = 0; i < platform->VisibleDeviceCount(); ++i) {
    ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor,
                         platform->ExecutorForDevice(i));
    const se::DeviceDescription& desc = executor->GetDeviceDescription();
    SCOPED_TRACE(absl::StrCat("device ", i, ": ", desc.name()));

    auto live = std::make_shared<cudnn_frontend::DeviceProperties>();
    auto build_err = live->set_device_id(i).build();
    ASSERT_FALSE(build_err.is_bad()) << build_err.get_message();

    ASSERT_OK_AND_ASSIGN(auto synth, BuildDeviceProperties(desc));
    ExpectPropsMatch(ParseProps(live), ParseProps(synth), desc);
    any_compared = true;
  }

  if (!any_compared) {
    GTEST_SKIP() << "No visible GPU devices to compare against.";
  }
}

// Make sure the cudnn_frontend::DeviceProperties created from our checked in
// target config still matches the one from the live device. This should catch
// cases where CuDNN requires a new field to be serialized, which may not have
// been mapped in our target config.
TEST(CudnnDevicePropsTest, SpecMatchesLiveDevice) {
  if (!CudnnSupportsDeviceProps()) {
    GTEST_SKIP() << "cuDNN runtime < 9.8 does not support DeviceProperties "
                    "JSON serialization.";
  }

  ASSERT_OK_AND_ASSIGN(std::string platform_name,
                       PlatformUtil::CanonicalPlatformName("gpu"));
  std::string name = absl::AsciiStrToUpper(platform_name);
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName(name));
  ASSERT_GT(platform->VisibleDeviceCount(), 0);
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor,
                       platform->ExecutorForDevice(0));
  const se::DeviceDescription& live_desc = executor->GetDeviceDescription();

  std::optional<se::GpuTargetConfigProto> spec;
  // GpuModel values are contiguous from 0, and GetGpuTargetConfig returns
  // NotFound past the last one.
  for (int i = 0;; ++i) {
    absl::StatusOr<se::GpuTargetConfigProto> proto =
        GetGpuTargetConfig(static_cast<GpuModel>(i));
    if (absl::IsNotFound(proto.status())) {
      break;
    }
    ASSERT_OK(proto);
    // Name alone doesn't distinguish MIG partitions from the full GPU.
    if (proto->gpu_device_info().name() == live_desc.name() &&
        proto->gpu_device_info().core_count() == live_desc.core_count()) {
      spec = *std::move(proto);
      break;
    }
  }
  if (!spec.has_value()) {
    GTEST_SKIP() << "No spec for " << live_desc.name();
  }
  SCOPED_TRACE(live_desc.name());

  ASSERT_OK_AND_ASSIGN(GpuTargetConfig spec_config,
                       GpuTargetConfig::FromProto(*spec));
  auto live = std::make_shared<cudnn_frontend::DeviceProperties>();
  auto build_err = live->set_device_id(0).build();
  ASSERT_FALSE(build_err.is_bad()) << build_err.get_message();
  ASSERT_OK_AND_ASSIGN(auto synth,
                       BuildDeviceProperties(spec_config.device_description));
  ExpectPropsMatch(ParseProps(live), ParseProps(synth),
                   spec_config.device_description);
}

}  // namespace
}  // namespace xla::gpu
