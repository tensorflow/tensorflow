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

#ifndef XLA_STREAM_EXECUTOR_PLATFORM_ID_H_
#define XLA_STREAM_EXECUTOR_PLATFORM_ID_H_

#include "absl/strings/string_view.h"

namespace stream_executor {

// Returns metadata about the Platforms ID.
class PlatformIdInfo {
 public:
  using NameGetter = absl::string_view (*)(const PlatformIdInfo&);

  explicit constexpr PlatformIdInfo(NameGetter name_getter)
      : name_getter_(name_getter) {}

  // Returns the platforms name, i.e. the string representation of the
  // platform ID.
  absl::string_view ToName() const { return name_getter_(*this); };

 private:
  NameGetter name_getter_;
};

// A platform ID is a unique identifier for each registered platform type -
// each platform is required to expose an ID to ensure unique registration and
// as a target against which plugins can register.
//
// The macro below is provided to help generate a [process-unique] identifier.
using PlatformId = const PlatformIdInfo*;

// Helper macro to define a plugin ID. To be used only inside plugin
// implementation files. Works by "reserving" an address/value (guaranteed to be
// unique) inside a process space.
//
// ID_VAR_NAME: The name of the variable to initialize with the platform ID.
// PLATFORM_NAME: The string name of the platform.
#define PLATFORM_DEFINE_ID(ID_VAR_NAME, PLATFORM_NAME)                         \
  namespace {                                                                  \
  constexpr ::stream_executor::PlatformIdInfo kInternalIdInfo_##PLATFORM_NAME( \
      [](const ::stream_executor::PlatformIdInfo&) -> absl::string_view {      \
        return #PLATFORM_NAME;                                                 \
      });                                                                      \
  }                                                                            \
  constexpr ::stream_executor::PlatformId ID_VAR_NAME(                         \
      &kInternalIdInfo_##PLATFORM_NAME);

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_PLATFORM_ID_H_
