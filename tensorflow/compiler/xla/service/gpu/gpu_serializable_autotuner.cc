/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/gpu_serializable_autotuner.h"

#include <algorithm>
#include <tuple>
#include <utility>

namespace xla {
namespace gpu {

Status SerializeAutotuneResults(const AutotuneCacheMap& autotune_cache,
                                AutotuneResults* results) {
  for (const auto& [k, result] : autotune_cache) {
    const auto& [model_str, hlo] = k;
    auto& entry = *results->add_results();
    entry.set_device(model_str);
    entry.set_hlo(hlo);
    *entry.mutable_result() = result;
  }

  // Sort the results so that they're deterministic.
  std::sort(results->mutable_results()->pointer_begin(),
            results->mutable_results()->pointer_end(),
            [](const auto* a, const auto* b) {
              return std::make_pair(absl::string_view(a->device()),
                                    absl::string_view(a->hlo())) <
                     std::make_pair(absl::string_view(b->device()),
                                    absl::string_view(b->hlo()));
            });

  return OkStatus();
}

Status LoadAutotuneResults(AutotuneCacheMap& autotune_cache,
                           const AutotuneResults& results) {
  for (const auto& result : results.results()) {
    autotune_cache[std::make_tuple(result.device(), result.hlo())] =
        result.result();
  }
  return OkStatus();
}

}  // namespace gpu
}  // namespace xla
