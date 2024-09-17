/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/python/profiler_utils.h"

#include <functional>
#include <memory>
#include <utility>

#include "xla/backends/profiler/plugin/plugin_tracer.h"
#include "xla/backends/profiler/plugin/profiler_c_api.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_profiler_extension.h"
#include "tsl/profiler/lib/profiler_factory.h"
#include "tsl/profiler/lib/profiler_interface.h"

namespace xla {

static const PLUGIN_Profiler_Api* FindProfilerApi(const PJRT_Api* pjrt_api) {
  PJRT_Profiler_Extension* profiler_extension =
      pjrt::FindExtension<PJRT_Profiler_Extension>(
          pjrt_api, PJRT_Extension_Type::PJRT_Extension_Type_Profiler);

  if (profiler_extension == nullptr) {
    // TODO(b/342627527): return proper error when no profiler api is found.
    return nullptr;
  }
  return profiler_extension->profiler_api;
}

void RegisterProfiler(const PJRT_Api* pjrt_api) {
  const PLUGIN_Profiler_Api* profiler_api = FindProfilerApi(pjrt_api);
  std::function<std::unique_ptr<tsl::profiler::ProfilerInterface>(
      const tensorflow::ProfileOptions&)>
      create_func = [profiler_api = profiler_api](
                        const tensorflow::ProfileOptions& options) mutable {
        return std::make_unique<xla::profiler::PluginTracer>(profiler_api,
                                                             options);
      };
  tsl::profiler::RegisterProfilerFactory(std::move(create_func));
}

}  // namespace xla
