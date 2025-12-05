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

#include "xla/backends/gpu/runtime/thunk_buffer_debug_filter.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "re2/re2.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_id.h"
#include "xla/ffi/ffi.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

namespace {

// A function that decides whether the thunk should be instrumented
// (kInstrument) or not (kSkip).
using ThunkFilter = absl::AnyInvocable<InstrumentAction(const Thunk&) const>;

// Creates a thunk filter that filters thunks by their IDs, based the allowed
// ranges passed in debug options.
ThunkFilter CreateThunkIdFilter(const DebugOptions& debug_options) {
  std::vector<std::pair<int64_t, int64_t>> thunk_id_ranges;
  for (const auto& range :
       debug_options.xla_gpu_experimental_thunk_buffer_debug_filter()
           .thunk_id_ranges()) {
    VLOG(1) << "Thunk filter: id range [" << range.first() << ", "
            << range.last() << "]";
    thunk_id_ranges.emplace_back(range.first(), range.last());
  }

  return [id_ranges = std::move(thunk_id_ranges)](const Thunk& thunk) {
    if (id_ranges.empty()) {
      return InstrumentAction::kInstrument;
    }

    const ThunkId thunk_id = thunk.thunk_info().thunk_id;
    if (absl::c_any_of(id_ranges, [&](const auto& range) {
          VLOG(2) << "Thunk filter: check ID range: " << range.first
                  << " <= " << thunk_id.value() << " <= " << range.second;
          return range.first <= thunk_id.value() &&
                 thunk_id.value() <= range.second;
        })) {
      VLOG(2) << "Thunk filter: ID matches";
      return InstrumentAction::kInstrument;
    }

    VLOG(2) << "Thunk filter: ID does not match";
    return InstrumentAction::kSkip;
  };
}

// Creates a thunk filter that filters thunks by matching their profile
// annotations against regexes configured in debug options.
ThunkFilter CreateProfileAnnotationRegexFilter(
    const DebugOptions& debug_options) {
  std::vector<std::unique_ptr<RE2>> profile_annotation_regexes;
  for (const auto& regex :
       debug_options.xla_gpu_experimental_thunk_buffer_debug_filter()
           .profile_annotation_regexes()) {
    VLOG(1) << "Thunk filter: profile annotation regex: " << regex;
    profile_annotation_regexes.push_back(std::make_unique<RE2>(regex));
  }
  return [regexes = std::move(profile_annotation_regexes)](const Thunk& thunk) {
    if (regexes.empty()) {
      return InstrumentAction::kInstrument;
    }

    const std::string& profile_annotation =
        thunk.thunk_info().profile_annotation;
    if (absl::c_any_of(regexes, [&](const auto& regex) {
          VLOG(2) << "Thunk filter: check profile annotation regex: "
                  << regex->pattern();
          return RE2::PartialMatch(profile_annotation, *regex);
        })) {
      VLOG(2) << "Thunk filter: profile annotation matches";
      return InstrumentAction::kInstrument;
    }

    VLOG(2) << "Thunk filter: profile annotation does not match";
    return InstrumentAction::kSkip;
  };
}

}  // namespace

// Creates a thunk filter that filters thunks by all the conditions configured
// in debug options.
ThunkFilter CreateThunkFilter(const DebugOptions& debug_options) {
  std::vector<ThunkFilter> filters;
  filters.push_back(CreateThunkIdFilter(debug_options));
  filters.push_back(CreateProfileAnnotationRegexFilter(debug_options));

  return [filters = std::move(filters)](const Thunk& thunk) {
    VLOG(2) << "Thunk filter: check ID " << thunk.thunk_info().thunk_id
            << ", profile annotation " << thunk.thunk_info().profile_annotation;
    if (absl::c_all_of(filters, [&](const auto& filter) {
          return filter(thunk) == InstrumentAction::kInstrument;
        })) {
      return InstrumentAction::kInstrument;
    }
    return InstrumentAction::kSkip;
  };
}

}  // namespace xla::gpu
