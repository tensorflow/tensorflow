/* Copyright 2015 The OpenXLA Authors.

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

// Defines types and declares functions for identifying and extracting
// information about the types of platforms and supporting libraries for which
// StreamExecutor implementations exist.
#ifndef XLA_STREAM_EXECUTOR_PLATFORM_H_
#define XLA_STREAM_EXECUTOR_PLATFORM_H_

#include <cstddef>
#include <memory>
#include <string>

#include "absl/base/casts.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/device_description.h"

namespace stream_executor {

class StreamExecutor;

// An enum to represent different levels of stream priorities.
// This is to avoid platform-specific representations in abstractions.
enum class StreamPriority { Default = 0, Lowest, Highest };

// Returns a printable description of StreamPriority.
std::string StreamPriorityToString(StreamPriority priority);

// Abstract base class for a platform registered with the PlatformManager.
class Platform {
 public:
  virtual ~Platform() = default;

  // Returns metadata about the Platforms ID.
  class IdInfo {
   public:
    using NameGetter = absl::string_view (*)(const IdInfo&);

    explicit constexpr IdInfo(NameGetter name_getter)
        : name_getter_(name_getter) {}

    // Returns the platforms name, i.e. the string representation of the
    // platform ID.
    absl::string_view ToName() const { return name_getter_(*this); };

   private:
    NameGetter name_getter_;
  };

  // Non-owning, pointer-like wrapper around `IdInfo`, needed to replace the
  // previously used `void*` in a backwards compatible manner.
  // TODO: b/465773559 - After we make the changes on the TF side, remove this
  // and just use a `const IdInfo*`.
  class IdPtr {
   public:
    IdPtr() = default;
    explicit constexpr IdPtr(const IdInfo* id_info) : id_info_(id_info) {}

    constexpr IdPtr(void* ptr)  // NOLINT(google-explicit-constructor)
        : id_info_(absl::bit_cast<const IdInfo*>(ptr)) {}

    operator void*() const {  // NOLINT(google-explicit-constructor)
      return absl::bit_cast<void*>(id_info_);
    }

    friend bool operator==(const IdPtr& lhs, const IdPtr& rhs) {
      return lhs.id_info_ == rhs.id_info_;
    }
    friend bool operator!=(const IdPtr& lhs, const IdPtr& rhs) {
      return !(lhs == rhs);
    }

    friend bool operator==(const IdPtr& lhs, std::nullptr_t) {
      return lhs.id_info_ == nullptr;
    }

    friend bool operator==(std::nullptr_t, const IdPtr& rhs) {
      return rhs.id_info_ == nullptr;
    }

    friend bool operator!=(const IdPtr& lhs, std::nullptr_t) {
      return lhs.id_info_ != nullptr;
    }

    friend bool operator!=(std::nullptr_t, const IdPtr& rhs) {
      return rhs.id_info_ != nullptr;
    }

    // This is often used in hash tables.
    template <typename H>
    friend H AbslHashValue(H h, const IdPtr& id) {
      return H::combine(std::move(h), id.id_info_);
    }

    // Pointer-like access to IdInfo.
    const IdInfo* operator->() const { return id_info_; }
    const IdInfo& operator*() const { return *id_info_; }

    template <typename Sink>
    friend void AbslStringify(Sink& sink, const IdPtr& id) {
      sink.Append(id == nullptr ? "nullptr" : id->ToName());
    }

   private:
    const IdInfo* id_info_;
  };

  // A platform ID is a unique identifier for each registered platform type -
  // each platform is required to expose an ID to ensure unique registration and
  // as a target against which plugins can register.
  //
  // The macro below is provided to help generate a [process-unique] identifier.
  using Id = IdPtr;

#define PLATFORM_DEFINE_ID_IMPL_1(ID_VAR_NAME) \
  PLATFORM_DEFINE_ID_IMPL_2(ID_VAR_NAME, ID_VAR_NAME)

// Helper macro to define a plugin ID. To be used only inside plugin
// implementation files. Works by "reserving" an address/value (guaranteed to be
// unique) inside a process space.
//
// ID_VAR_NAME: The name of the variable to initialize with the platform ID.
// PLATFORM_NAME: The string name of the platform.
#define PLATFORM_DEFINE_ID_IMPL_2(ID_VAR_NAME, PLATFORM_NAME)   \
  namespace {                                                   \
  constexpr ::stream_executor::Platform::IdInfo                 \
      kInternalIdInfo_##PLATFORM_NAME(                          \
          [](const ::stream_executor::Platform::IdInfo&)        \
              -> absl::string_view { return #PLATFORM_NAME; }); \
  }                                                             \
  constexpr ::stream_executor::Platform::Id ID_VAR_NAME(        \
      &kInternalIdInfo_##PLATFORM_NAME);

#define PLATFORM_DEFINE_ID_GET_MACRO(_1, _2, NAME, ...) NAME

// Because we can't make cross cutting changes across XLA and TF, we need this
// to support the old single parameter PLATFORM_DEFINE_ID macro.
// TODO: b/455530217 - Remove this, and keep only the 2 parameter version.
#define PLATFORM_DEFINE_ID(...)                                        \
  PLATFORM_DEFINE_ID_GET_MACRO(__VA_ARGS__, PLATFORM_DEFINE_ID_IMPL_2, \
                               PLATFORM_DEFINE_ID_IMPL_1)(__VA_ARGS__)

  // Returns a key uniquely identifying this platform.
  virtual Id id() const = 0;

  // Name of this platform.
  virtual const std::string& Name() const = 0;

  // Returns the number of devices accessible on this platform.
  //
  // Note that, though these devices are visible, if there is only one userspace
  // context allowed for the device at a time and another process is using this
  // device, a call to ExecutorForDevice may return an error status.
  virtual int VisibleDeviceCount() const = 0;

  // Returns true iff the platform has been initialized.
  virtual bool Initialized() const;

  // Initializes the platform. The platform must be initialized before obtaining
  // StreamExecutor objects.
  virtual absl::Status Initialize();

  // Returns a populated DeviceDescription for the device at the given ordinal.
  // This should not require device initialization. Note that not all platforms
  // may support acquiring the DeviceDescription indirectly.
  //
  // Alternatively callers may call GetDeviceDescription() on the StreamExecutor
  // which returns a cached instance specific to the initialized StreamExecutor.
  virtual absl::StatusOr<std::unique_ptr<DeviceDescription>>
  DescriptionForDevice(int ordinal) const = 0;

  // Returns a StreamExecutor for the given ordinal if one has already been
  // created, or an error is returned if none exists.  Does not create a new
  // context with the device.
  virtual absl::StatusOr<StreamExecutor*> FindExisting(int ordinal) {
    return absl::NotFoundError("Not implemented for this platform.");
  }

  // Returns a device with the given ordinal on this platform or, if none can
  // be found with the given ordinal or there is an error in opening a context
  // to communicate with the device, an error status is returned.
  //
  // Ownership of the executor is NOT transferred to the caller --
  // the Platform owns the executors in a singleton-like fashion.
  virtual absl::StatusOr<StreamExecutor*> ExecutorForDevice(int ordinal) = 0;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_PLATFORM_H_
