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

#include "xla/backends/profiler/plugin/plugin_tracer_impl.h"

#include <cstdint>
#include <memory>
#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "xla/backends/profiler/plugin/plugin_tracer.h"
#include "xla/backends/profiler/plugin/profiler_c_api.h"
#include "xla/backends/profiler/plugin/profiler_error.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "tsl/platform/logging.h"
#include "tsl/profiler/lib/profiler_factory.h"
#include "tsl/profiler/lib/profiler_interface.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {
using tensorflow::ProfileOptions;
using tsl::profiler::ProfilerInterface;
using tsl::profiler::XPlaneBuilder;

class PluginTracerImpl : public ProfilerInterface {
 public:
  explicit PluginTracerImpl(const ProfileOptions& options)
      : options_(options) {}

  // Starts profiling.
  absl::Status Start() override {
    LOG(INFO) << "Starting Tracer";
    return absl::OkStatus();
  }

  // Stops profiling.
  absl::Status Stop() override {
    LOG(INFO) << "Stopping Tracer";
    return absl::OkStatus();
  }

  // Saves collected profile data into XSpace.
  absl::Status CollectData(tensorflow::profiler::XSpace* space) override {
    LOG(INFO) << "Collecting data";
    tensorflow::profiler::XPlane* plane = space->add_planes();
    XPlaneBuilder builder(plane);
    builder.SetName("GpuBackendTracer");
    tensorflow::profiler::XStatMetadata* metadata =
        builder.GetOrCreateStatMetadata((int64_t)0);
    metadata->set_name("ProfileOptions");

    builder.AddStatValue(*metadata, options_.SerializeAsString());
    return absl::OkStatus();
  }

 private:
  ProfileOptions options_;
};

std::unique_ptr<ProfilerInterface> CreatePluginTracer(
    const ProfileOptions& options) {
  return std::make_unique<PluginTracerImpl>(options);
}

static auto register_test_tracer = [] {
  RegisterProfilerFactory(&CreatePluginTracer);
  return 0;
}();

TEST(PluginTracerTest, TestPluginWithPluginTracer) {
  PLUGIN_Profiler_Api api;
  api.create = &PLUGIN_Profiler_Create;
  api.start = &PLUGIN_Profiler_Start;
  api.stop = &PLUGIN_Profiler_Stop;
  api.collect_data = &PLUGIN_Profiler_CollectData;
  api.destroy = &PLUGIN_Profiler_Destroy;
  api.error_destroy = &PLUGIN_Profiler_Error_Destroy;
  api.error_message = &PLUGIN_Profiler_Error_Message;
  api.error_get_code = &PLUGIN_Profiler_Error_GetCode;
  api.struct_size = PLUGIN_Profiler_Api_STRUCT_SIZE;

  // The options would be emmited by the tracer in the XPlane Stat which can be
  // checked to see if they are the same. This ensures that the tracer is
  // correctly passing the options to the PluginTracer implementations.
  ProfileOptions options;
  options.set_repository_path("TestRepositoryPath");
  options.set_device_tracer_level(2);

  PluginTracer tracer(&api, options);

  tensorflow::profiler::XSpace xspace;
  EXPECT_TRUE(tracer.Start().ok());
  EXPECT_TRUE(tracer.Stop().ok());

  EXPECT_TRUE(tracer.CollectData(&xspace).ok());

  ASSERT_THAT(xspace.planes(), testing::SizeIs(1));
  ASSERT_THAT(xspace.planes(0).stats(), testing::SizeIs(1));

  tsl::profiler::XPlaneVisitor visitor(&xspace.planes(0));
  std::optional<tsl::profiler::XStatVisitor> stat =
      visitor.GetStat(0, *visitor.GetStatMetadata(0));

  ASSERT_TRUE(stat.has_value());
  EXPECT_EQ(stat->Name(), "ProfileOptions");
  EXPECT_EQ(stat->StrOrRefValue(), options.SerializeAsString());
}

}  // namespace profiler
}  // namespace xla
