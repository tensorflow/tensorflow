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

// Defines types and declares functions for identifying and extracting
// information about the types of platforms and supporting libraries for which
// StreamExecutor implementations exist.
#ifndef TENSORFLOW_STREAM_EXECUTOR_PLATFORM_H_
#define TENSORFLOW_STREAM_EXECUTOR_PLATFORM_H_

#include <map>

#include "tensorflow/stream_executor/device_description.h"
#include "tensorflow/stream_executor/device_options.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/lib/status_macros.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/plugin.h"
#include "tensorflow/stream_executor/trace_listener.h"

namespace stream_executor {

class StreamExecutor;
class DeviceDescription;

// Describes the platform for a StreamExecutor instantiation to act upon.
//
// Implementors: if you add a value here be sure to update PlatformKindString
// and CheckPlatformKindIsValid.
enum class PlatformKind {
  kInvalid,
  kCuda,
  kROCm,
  kOpenCL,
  kHost,
  kMock,
  kSize,
};

// Returns true if kind represents a valid platform capable of enqueuing items
// on a stream, but not necessarily on an accelerator device.
// Returns false for kMock and any invalid PlatformKind values.
bool PlatformIsRunnable(PlatformKind kind);

// Returns true if kind represents a valid platform capable of running kernels
// on an accelerator device. Returns false for kHost*, kMock and any invalid
// PlatformKind values.
bool PlatformIsRunnableOnDevice(PlatformKind kind);

// Returns a printable description of a PlatformKind.
std::string PlatformKindString(PlatformKind kind);

// Returns the PlatformKind corresponding to the input string; returns kInvalid
// in the case of no match.
PlatformKind PlatformKindFromString(std::string platform_string);

// Checks that kind takes on a valid value.
void CheckPlatformKindIsValid(PlatformKind kind);

// StreamExecutorConfig encapsulates the set of options for constructing a
// StreamExecutor for a given platform.
struct StreamExecutorConfig {
  // Sets members to defaults: -1 for ordinal (must be changed), and default
  // PluginConfig and DeviceOptions.
  StreamExecutorConfig();

  // Simple ordinal-setting constructor.
  explicit StreamExecutorConfig(int ordinal);

  // The ordinal of the device to be managed by the returned StreamExecutor.
  int ordinal;

  // The PluginConfig for the returned StreamExecutor.
  PluginConfig plugin_config;

  // The DeviceOptions for the returned StreamExecutor.
  DeviceOptions device_options;
};

// Abstract base class for a platform registered with the MultiPlatformManager.
class Platform {
 public:
  virtual ~Platform();

  // A platform ID is a unique identifier for each registered platform type -
  // each platform is required to expose an ID to ensure unique registration and
  // as a target against which plugins can register.
  //
  // The macro below is provided to help generate a [process-unique] identifier.
  using Id = void*;

// Helper macro to define a plugin ID. To be used only inside plugin
// implementation files. Works by "reserving" an address/value (guaranteed to be
// unique) inside a process space.
#define PLATFORM_DEFINE_ID(ID_VAR_NAME) \
  namespace {                           \
  int plugin_id_value;                  \
  }                                     \
  const ::stream_executor::Platform::Id ID_VAR_NAME = &plugin_id_value;

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

  // Initializes the platform with a custom set of options. The platform must be
  // initialized before obtaining StreamExecutor objects.  The interpretation of
  // the platform_options argument is implementation specific.  This method may
  // return an error if unrecognized options are provided.  If using
  // MultiPlatformManager, this method will be called automatically by
  // InitializePlatformWithId/InitializePlatformWithName.
  virtual port::Status Initialize(
      const std::map<std::string, std::string>& platform_options);

  // Returns a populated DeviceDescription for the device at the given ordinal.
  // This should not require device initialization. Note that not all platforms
  // may support acquiring the DeviceDescription indirectly.
  //
  // Alternatively callers may call GetDeviceDescription() on the StreamExecutor
  // which returns a cached instance specific to the initialized StreamExecutor.
  virtual port::StatusOr<std::unique_ptr<DeviceDescription>>
  DescriptionForDevice(int ordinal) const = 0;

  // Returns a device with the given ordinal on this platform with a default
  // plugin configuration or, if none can be found with the given ordinal or
  // there is an error in opening a context to communicate with the device, an
  // error status is returned.
  //
  // Ownership of the executor is NOT transferred to the caller --
  // the Platform owns the executors in a singleton-like fashion.
  virtual port::StatusOr<StreamExecutor*> ExecutorForDevice(int ordinal) = 0;

  // Returns a device or error, as above, with the specified plugins.
  //
  // Ownership of the executor is NOT transferred to the caller.
  virtual port::StatusOr<StreamExecutor*> ExecutorForDeviceWithPluginConfig(
      int ordinal, const PluginConfig& plugin_config) = 0;

  // Returns a device constructed with the options specified in "config".
  // Ownership of the executor is NOT transferred to the caller.
  virtual port::StatusOr<StreamExecutor*> GetExecutor(
      const StreamExecutorConfig& config) = 0;

  // Returns a device constructed with the options specified in "config" without
  // looking in or storing to the Platform's executor cache.
  // Ownership IS transferred to the caller.
  virtual port::StatusOr<std::unique_ptr<StreamExecutor>> GetUncachedExecutor(
      const StreamExecutorConfig& config) = 0;

  // Warning: this is a dangerous API and should be used with caution.
  //
  // Forces the platform to delete executor instances, releasing their
  // associated device contexts. There must be no held instances of the executor
  // and there must be no outstanding activity on the devices for this platform.
  //
  // This is only useful on platforms which bind a device to a single process
  // that has obtained the device context. May return UNIMPLEMENTED on platforms
  // that have no reason to destroy device contexts.
  //
  // The platform must be reinitialized after this is called.
  virtual port::Status ForceExecutorShutdown();

  // Registers a TraceListener to listen to all StreamExecutors for this
  // platform.
  // Takes ownership of listener.
  virtual void RegisterTraceListener(
      std::unique_ptr<TraceListener> listener) = 0;

  // Removes the specified TraceListener from all StreamExecutors.
  virtual void UnregisterTraceListener(TraceListener* listener) = 0;

  // Map of executor-to-executor coordinate and boolean, indicating if the first
  // executor can access the second's memory.
  using PeerAccessMap = std::map<std::pair<int, int>, bool>;

  // Returns a matrix indicating which executors can access which other
  // executors' memory.
  virtual std::unique_ptr<PeerAccessMap> GetPeerAccessMap();

  // Attempts to enable all peer-to-peer access links described by the result of
  // GetPeerAccessMap(). Note that calling this routine will force the creation
  // of a default-argument (see StreamExecutorConfig) StreamExecutor object for
  // each device ordinal in the system, should any not yet exist.
  virtual port::Status EnablePeerAccess();

 protected:
  // SE_DISALLOW_COPY_AND_ASSIGN declares a constructor, which suppresses the
  // presence of the default constructor. This statement re-enables it, which
  // simplifies subclassing.
  Platform() = default;

 private:
  SE_DISALLOW_COPY_AND_ASSIGN(Platform);
};

}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_PLATFORM_H_
