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
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "third_party/cudnn_frontend/include/cudnn_frontend.h"
#include "json/json.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"
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
};

// Clock-rate fields where small drift is expected because XLA stores them as
// float GHz and converts back to int KHz, while cuDNN reads int KHz directly.
absl::string_view kClockFields[] = {
    "smClockRateKHz",
    "memClockRateKHz",
};

constexpr int64_t kClockToleranceKHz = 1000;

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
    if (!live.isMember(field.data()) || !synth.isMember(field.data())) continue;
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

TEST(CudnnDevicePropsTest, MatchesLiveDevice) {
  if (!CudnnSupportsDeviceProps()) {
    GTEST_SKIP() << "cuDNN runtime < 9.8 does not support DeviceProperties "
                    "JSON serialization.";
  }

  std::string name =
      absl::AsciiStrToUpper(PlatformUtil::CanonicalPlatformName("gpu").value());
  TF_ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                          se::PlatformManager::PlatformWithName(name));

  bool any_compared = false;
  for (int i = 0; i < platform->VisibleDeviceCount(); ++i) {
    TF_ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor,
                            platform->ExecutorForDevice(i));
    const se::DeviceDescription& desc = executor->GetDeviceDescription();
    SCOPED_TRACE(absl::StrCat("device ", i, ": ", desc.name()));

    auto live = std::make_shared<cudnn_frontend::DeviceProperties>();
    auto build_err = live->set_device_id(i).build();
    ASSERT_FALSE(build_err.is_bad()) << build_err.get_message();

    TF_ASSERT_OK_AND_ASSIGN(auto synth, BuildDeviceProperties(desc));

    Json::Value live_json = ParseProps(live);
    Json::Value synth_json = ParseProps(synth);

    VLOG(1) << "live : " << Dump(live_json);
    VLOG(1) << "synth: " << Dump(synth_json);

    StripIgnoredFields(live_json);
    StripIgnoredFields(synth_json);
    ExpectClockFieldsClose(live_json, synth_json);
    for (absl::string_view field : kClockFields) {
      live_json.removeMember(field.data());
      synth_json.removeMember(field.data());
    }

    EXPECT_EQ(live_json, synth_json) << "live:\n"
                                     << Dump(live_json) << "\nsynth:\n"
                                     << Dump(synth_json);
    any_compared = true;
  }

  if (!any_compared) {
    GTEST_SKIP() << "No visible GPU devices to compare against.";
  }
}

}  // namespace
}  // namespace xla::gpu
