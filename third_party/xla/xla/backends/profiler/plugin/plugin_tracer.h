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
#ifndef XLA_BACKENDS_PROFILER_PLUGIN_PLUGIN_TRACER_H_
#define XLA_BACKENDS_PROFILER_PLUGIN_PLUGIN_TRACER_H_

#include "absl/status/status.h"
#include "xla/backends/profiler/plugin/profiler_c_api.h"
#include "tsl/profiler/lib/profiler_interface.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {

// Plugin implementation of ProfilerInterface.
//
// Thread-safety: This class is go/thread-compatible.
class PluginTracer : public tsl::profiler::ProfilerInterface {
 public:
  explicit PluginTracer(const PLUGIN_Profiler_Api* profiler_api,
                        const tensorflow::ProfileOptions& options);
  ~PluginTracer() override;

  absl::Status Start() override;

  absl::Status Stop() override;

  absl::Status CollectData(tensorflow::profiler::XSpace* space) override;

 private:
  const PLUGIN_Profiler_Api* profiler_api_;
  PLUGIN_Profiler* profiler_;
};

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_PLUGIN_PLUGIN_TRACER_H_
