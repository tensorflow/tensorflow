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

#include <cstddef>
#include <cstdint>

#include "nanobind/nanobind.h"
#include "nanobind/stl/string_view.h"  // IWYU pragma: keep
#include "xla/tsl/profiler/backends/gpu/ondevice_event_receiver.h"
#include "xla/tsl/profiler/backends/gpu/ondevice_trace_event.h"

namespace nb = ::nanobind;

namespace xla {
namespace {

using ::tsl::profiler::GpuOnDeviceTraceEvent;
using ::tsl::profiler::GpuOnDeviceTraceEventReceiver;

size_t ActiveVersion() {
  return GpuOnDeviceTraceEventReceiver::GetSingleton()->ActiveVersion();
}

uint32_t StartInjectionInstance(uint32_t version) {
  return GpuOnDeviceTraceEventReceiver::GetSingleton()->StartInjectionInstance(
      version);
}

void Inject(size_t version, int32_t injection_instance_id,
            absl::string_view tag_name, uint32_t tag_id, uint32_t pid,
            uint32_t tid, int64_t start_time_ns, int64_t duration_ps) {
  return GpuOnDeviceTraceEventReceiver::GetSingleton()
      ->Inject(version,
               GpuOnDeviceTraceEvent{
                   .injection_instance_id = injection_instance_id,
                   .tag_name = tag_name,
                   .tag_id = tag_id,
                   .pid = pid,
                   .tid = tid,
                   .start_time_ns = start_time_ns,
                   .duration_ps = duration_ps,
               })
      .IgnoreError();
}

}  // namespace

NB_MODULE(_gpu_ondevice_tracing, m) {
  m.def("active_version", &ActiveVersion,
        "Check if there are active profiling session using the on-device event "
        "receiver.");
  m.def("start_injection_instance", &StartInjectionInstance, nb::arg("version"),
        "Get the Id for a new injection instance.");
  m.def("inject", &Inject, nb::arg("version"), nb::arg("injection_instance_id"),
        nb::arg("tag_name"), nb::arg("tag_id"), nb::arg("pid"), nb::arg("tid"),
        nb::arg("start_time_ns"), nb::arg("duration_ps"),
        "Injecting software generated events to the on-device event receiver.");
}

}  // namespace xla
