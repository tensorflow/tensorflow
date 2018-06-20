/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PLATFORM_DEVICE_TRACER_H_
#define TENSORFLOW_CORE_PLATFORM_DEVICE_TRACER_H_

#include <memory>

#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

class StepStatsCollector;

// 'DeviceTracer' is an interface for collecting low-level execution timings
// of hardware accelerator (e.g. GPU) computation and DMA transfers.
//
// Typical usage pattern is as follows:
//
// DeviceTracer* tracer = CreateDeviceTracer();
// if (tracer) {
//   tracer->Start();
//
//   ... perform some computations on a hardware accelerator.
//
//   tracer->Stop();
//
//   StepStats stats;
//   StepStatsCollector collector(&stats);
//   tracer->Collect(&collector);
// }
//
// Notes:
// Tracing is not supported on all plaforms.  On platforms
// with no tracing support, 'CreateDeviceTracer' will return 'nullptr'.
// On most plaforms, hardware tracing will be a system-wide activity and
// a single 'DeviceTracer' will collect activity from all devices.
// It is also common that only a single tracer may be active at any
// given time.  The 'Start' method will return an error if tracing is
// already in progress elsewhere.
//
class DeviceTracer {
 public:
  virtual ~DeviceTracer() {}

  // Start device tracing.
  // Note that only a single trace can be active, in which case this
  // methods will return an 'Unavailable' error.
  virtual Status Start() = 0;

  // Stop device tracing.
  // It is safe to call 'Stop' on a tracer which is not enabled.
  virtual Status Stop() = 0;

  // Collect trace results.  Results are added to the specified
  // StepStatsCollector.  Does not clear any existing stats.
  // It is an error to call 'Collect' while a trace is running.
  virtual Status Collect(StepStatsCollector* collector) = 0;
};

// Creates a platform-specific DeviceTracer.
// Returns 'nullptr' on platforms where tracing is not supported.
std::unique_ptr<DeviceTracer> CreateDeviceTracer();

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_DEVICE_TRACER_H_
