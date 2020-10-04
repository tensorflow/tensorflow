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

// This is a registration-oriented interface for multiple platforms. It will
// replace the MachineManager singleton interface, as MachineManager does not
// currently support simultaneous use of multiple platforms.
//
// Usage:
//
// In your BUILD rule, add a dependency on a platform plugin that you'd like
// to use, such as:
//
//   //third_party/tensorflow/stream_executor/cuda:cuda_platform
//   //third_party/tensorflow/stream_executor/opencl:opencl_platform
//
// This will register platform plugins that can be discovered via this
// interface. Sample API usage:
//
//   port::StatusOr<Platform*> platform_status =
//      se::MultiPlatformManager::PlatformWithName("OpenCL");
//   if (!platform_status.ok()) { ... }
//   Platform* platform = platform_status.ValueOrDie();
//   LOG(INFO) << platform->VisibleDeviceCount() << " devices visible";
//   if (platform->VisibleDeviceCount() <= 0) { return; }
//
//   for (int i = 0; i < platform->VisibleDeviceCount(); ++i) {
//     port::StatusOr<StreamExecutor*> executor_status =
//        platform->ExecutorForDevice(i);
//     if (!executor_status.ok()) {
//       LOG(INFO) << "could not retrieve executor for device ordinal " << i
//                 << ": " << executor_status.status();
//       continue;
//     }
//     LOG(INFO) << "found usable executor: " << executor_status.ValueOrDie();
//   }
//
// A few things to note:
//  - There is no standard formatting/practice for identifying the name of a
//    platform. Ideally, a platform will list its registered name in its header
//    or in other associated documentation.
//  - Platform name lookup is case-insensitive. "OpenCL" or "opencl" (or even
//    ("OpEnCl") would work correctly in the above example.
//
// And similarly, for standard interfaces (BLAS, RNG, etc.) you can add
// dependencies on support libraries, e.g.:
//
//    //third_party/tensorflow/stream_executor/cuda:pluton_blas_plugin
//    //third_party/tensorflow/stream_executor/cuda:cudnn_plugin
//    //third_party/tensorflow/stream_executor/cuda:cublas_plugin
//    //third_party/tensorflow/stream_executor/cuda:curand_plugin

#ifndef TENSORFLOW_STREAM_EXECUTOR_MULTI_PLATFORM_MANAGER_H_
#define TENSORFLOW_STREAM_EXECUTOR_MULTI_PLATFORM_MANAGER_H_

#include <functional>
#include <map>
#include <memory>
#include <vector>

#include "absl/strings/string_view.h"
#include "tensorflow/stream_executor/lib/initialize.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/platform.h"
#include "tensorflow/stream_executor/platform/port.h"

namespace stream_executor {

// Manages multiple platforms that may be present on the current machine.
class MultiPlatformManager {
 public:
  // Registers a platform object, returns an error status if the platform is
  // already registered. The associated listener, if not null, will be used to
  // trace events for ALL executors for that platform.
  // Takes ownership of platform.
  static port::Status RegisterPlatform(std::unique_ptr<Platform> platform);

  // Retrieves the platform registered with the given platform name (e.g.
  // "CUDA", "OpenCL", ...) or id (an opaque, comparable value provided by the
  // Platform's Id() method).
  //
  // If the platform has not already been initialized, it will be initialized
  // with a default set of parameters.
  //
  // If the requested platform is not registered, an error status is returned.
  // Ownership of the platform is NOT transferred to the caller --
  // the MultiPlatformManager owns the platforms in a singleton-like fashion.
  static port::StatusOr<Platform*> PlatformWithName(absl::string_view target);
  static port::StatusOr<Platform*> PlatformWithId(const Platform::Id& id);

  // Same functions as above, but allows platforms to be returned without
  // initialization if initialize_platform == false.
  static port::StatusOr<Platform*> PlatformWithName(absl::string_view target,
                                                    bool initialize_platform);
  static port::StatusOr<Platform*> PlatformWithId(const Platform::Id& id,
                                                  bool initialize_platform);

  // Retrieves the platform registered with the given platform name (e.g.
  // "CUDA", "OpenCL", ...) or id (an opaque, comparable value provided by the
  // Platform's Id() method).
  //
  // The platform will be initialized with the given options. If the platform
  // was already initialized, an error will be returned.
  //
  // If the requested platform is not registered, an error status is returned.
  // Ownership of the platform is NOT transferred to the caller --
  // the MultiPlatformManager owns the platforms in a singleton-like fashion.
  static port::StatusOr<Platform*> InitializePlatformWithName(
      absl::string_view target,
      const std::map<std::string, std::string>& options);

  static port::StatusOr<Platform*> InitializePlatformWithId(
      const Platform::Id& id,
      const std::map<std::string, std::string>& options);

  // Retrieves the platforms satisfying the given filter, i.e. returns true.
  // Returned Platforms are always initialized.
  static port::StatusOr<std::vector<Platform*>> PlatformsWithFilter(
      const std::function<bool(const Platform*)>& filter);

  static port::StatusOr<std::vector<Platform*>> PlatformsWithFilter(
      const std::function<bool(const Platform*)>& filter,
      bool initialize_platform);

  // Although the MultiPlatformManager "owns" its platforms, it holds them as
  // undecorated pointers to prevent races during program exit (between this
  // object's data and the underlying platforms (e.g., CUDA, OpenCL).
  // Because certain platforms have unpredictable deinitialization
  // times/sequences, it is not possible to strucure a safe deinitialization
  // sequence. Thus, we intentionally "leak" allocated platforms to defer
  // cleanup to the OS. This should be acceptable, as these are one-time
  // allocations per program invocation.
  // The MultiPlatformManager should be considered the owner
  // of any platforms registered with it, and leak checking should be disabled
  // during allocation of such Platforms, to avoid spurious reporting at program
  // exit.

  // Interface for a listener that gets notified at certain events.
  class Listener {
   public:
    virtual ~Listener() = default;
    // Callback that is invoked when a Platform is registered.
    virtual void PlatformRegistered(Platform* platform) = 0;
  };
  // Registers a listeners to receive notifications about certain events.
  // Precondition: No Platform has been registered yet.
  static port::Status RegisterListener(std::unique_ptr<Listener> listener);
};

}  // namespace stream_executor

// multi_platform_manager.cc will define these instances.
//
// Registering a platform:
// REGISTER_MODULE_INITIALIZER_SEQUENCE(my_platform, multi_platform_manager);
// REGISTER_MODULE_INITIALIZER_SEQUENCE(multi_platform_manager_listener,
// my_platform);
//
// Registering a listener:
// REGISTER_MODULE_INITIALIZER_SEQUENCE(my_listener,
// multi_platform_manager_listener);
DECLARE_MODULE_INITIALIZER(multi_platform_manager);
DECLARE_MODULE_INITIALIZER(multi_platform_manager_listener);

#endif  // TENSORFLOW_STREAM_EXECUTOR_MULTI_PLATFORM_MANAGER_H_
