/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/profiler/gpu/ondevice_event_exporter.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include <gtest/gtest.h>
#include "xla/tsl/profiler/backends/gpu/ondevice_event_receiver.h"
#include "xla/tsl/profiler/backends/gpu/ondevice_trace_event.h"

namespace xla {
namespace profiler {
namespace test {

namespace {

using ::tsl::profiler::GpuOnDeviceTraceEvent;
using ::tsl::profiler::GpuOnDeviceTraceEventReceiver;

TEST(GpuOnDeviceTraceEventExporterTest, SimpleInjection) {
  static constexpr int kMaxInjectionInstance = 1;
  std::unique_ptr<GpuOnDeviceTraceEventExporter> collector =
      CreateGpuOnDeviceTraceEventExporter(
          {.max_injection_instance = kMaxInjectionInstance,
           .max_pid = 10,
           .max_tid = 10},
          0, 0);
  auto* receiver = GpuOnDeviceTraceEventReceiver::GetSingleton();
  ASSERT_EQ(receiver->ActiveVersion(), 0);

  auto status_or_version =
      receiver->StartWith(collector.get(), kMaxInjectionInstance);
  ASSERT_TRUE(status_or_version.ok());
  size_t version = status_or_version.value();
  ASSERT_NE(version, 0);

  int32_t wrong_version_injection =
      receiver->StartInjectionInstance(version + 100);
  ASSERT_EQ(wrong_version_injection, 0);

  int32_t instance_id = receiver->StartInjectionInstance(version);
  ASSERT_NE(instance_id, 0);

  GpuOnDeviceTraceEvent event{
      .injection_instance_id = instance_id,
      .tag_name = "test_tag",
      .pid = 1,
      .tid = 1,
      .start_time_ns = 10,
      .duration_ps = 10000,
  };
  ASSERT_TRUE(receiver->Inject(version, std::move(event)).ok());

  GpuOnDeviceTraceEvent event_wrong_instance_id{
      .injection_instance_id = instance_id + 10,
      .tag_name = "wrong_instance_id",
      .pid = 1,
      .tid = 1,
      .start_time_ns = 20,
      .duration_ps = 10000,
  };
  auto status = receiver->Inject(version, std::move(event_wrong_instance_id));
  ASSERT_FALSE(status.ok());
  ASSERT_EQ(status.message(), "Injection instance id is out of range.");

  ASSERT_TRUE(receiver->Stop().ok());
}

}  // namespace

}  // namespace test
}  // namespace profiler
}  // namespace xla
