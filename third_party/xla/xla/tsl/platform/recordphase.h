/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

// Client library for recording action subphase timing metrics.
//
// This is **different** to the TSL profiler TraceMe functionality.
//
// Currently the public implementation of this library is a stub that does
// nothing. There is an implementation internally at Google.
//
// This library is used to record the start and end of a subphase in an action.
// A subphase is a named section of work that happens within an action.
//
// Example:
//   // Start a phase named "parse_action".
//   StartPhase("parse_action");
//   // Do some work.
//   // ...
//   // End the phase.
//   EndPhase("parse_action");
//   // Start another phase named "link_executable" which depends on
//   // (always starts after) "parse_action".
//   StartPhase("link_executable", {"parse_action"});
//   // Do some work.
//   // ...
//   // End the phase.
//   EndPhase("link_executable");
//
// The StartPhase and EndPhase methods are thread-safe.
//
// The LoadPhase and LoadAllPhases methods can be used in
// unit tests to verify the recorded phase timing information.

#ifndef XLA_TSL_PLATFORM_RECORDPHASE_H_
#define XLA_TSL_PLATFORM_RECORDPHASE_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"

namespace tsl::recordphase {
// Records the start of a phase.
// * phase_name: the name of the phase, must be unique within the namespace.
// * dependencies: the phases that must complete before this phase can start.
// phase_name and dependencies must contain only alphanumeric characters,
// dashes and underscores.
// If a phase has already started, or a dependency does not exist,
// or if there are illegal characters in the phase name or its dependencies,
// the method will log an error and fail silently.
void StartPhase(absl::string_view phase_name,
                const std::vector<absl::string_view>& dependencies = {});

// This is like StartPhase, but it generates a unique phase name (which it uses
// to invoke StartPhase) and returns it.
std::string StartPhaseUnique(
    absl::string_view phase_name,
    const std::vector<absl::string_view>& dependencies = {});

// Records the end of a phase. The phase must have been started before.
void EndPhase(absl::string_view phase_name);

// Simple RAII wrapper around StartPhase and EndPhase. Does not perform any
// additional checking.
class RecordScoped {
 public:
  explicit RecordScoped(const absl::string_view phase_name,
                        bool use_unique_phase_name = false,
                        const std::vector<absl::string_view>& dependencies = {})
      : phase_name_(phase_name) {
    if (!use_unique_phase_name) {
      StartPhase(phase_name_, dependencies);
    } else {
      phase_name_ = StartPhaseUnique(phase_name, dependencies);
    }
  }
  ~RecordScoped() { EndPhase(phase_name_); }

  absl::string_view phase_name() const { return phase_name_; }

 private:
  std::string phase_name_;
};
}  // namespace tsl::recordphase

#endif  // XLA_TSL_PLATFORM_RECORDPHASE_H_
