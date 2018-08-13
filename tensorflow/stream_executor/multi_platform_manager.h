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

#include "tensorflow/stream_executor/lib/initialize.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/platform.h"
#include "tensorflow/stream_executor/platform/mutex.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/platform/thread_annotations.h"

namespace stream_executor {

// Manages multiple platforms that may be present on the current machine.
class MultiPlatformManager {
 public:
  // Registers a platform object, returns an error status if the platform is
  // already registered. The associated listener, if not null, will be used to
  // trace events for ALL executors for that platform.
  // Takes ownership of listener.
  static port::Status RegisterPlatform(std::unique_ptr<Platform> platform)
      LOCKS_EXCLUDED(platforms_mutex_);

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
  static port::StatusOr<Platform*> PlatformWithName(const string& target)
      LOCKS_EXCLUDED(platforms_mutex_);
  static port::StatusOr<Platform*> PlatformWithId(const Platform::Id& id)
      LOCKS_EXCLUDED(platforms_mutex_);

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
      const string& target, const std::map<string, string>& options)
      LOCKS_EXCLUDED(platforms_mutex_);
  static port::StatusOr<Platform*> InitializePlatformWithId(
      const Platform::Id& id, const std::map<string, string>& options)
      LOCKS_EXCLUDED(platforms_mutex_);

  // Clears the set of registered platforms, primarily used for testing.
  static void ClearPlatformRegistry() LOCKS_EXCLUDED(platforms_mutex_);

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
  using PlatformMap = std::map<string, Platform*>;

  // Provides access to the available set of platforms under a lock.
  static port::Status WithPlatforms(
      std::function<port::Status(PlatformMap*)> callback)
      LOCKS_EXCLUDED(platforms_mutex_) {
    mutex_lock lock(platforms_mutex_);
    return callback(GetPlatformMap());
  }

 private:
  using PlatformIdMap = std::map<Platform::Id, Platform*>;

  static mutex platforms_mutex_;

  // TODO(b/22689637): Clean up these two maps; make sure they coexist nicely.
  // TODO(b/22689637): Move this (whatever the final/"official" map is) to
  // plugin_regstry.h, along with the associated functionality.
  // Platform-name-to-object mapping. These platforms are registered via module
  // initializers, and linkage determines which platforms are available to a
  // given target.
  static PlatformMap* GetPlatformMap() {
    static PlatformMap* instance = new PlatformMap;
    return instance;
  }

  // Holds a Platform::Id-to-object mapping.
  // Unlike platforms_ above, this map does not own its contents.
  static PlatformIdMap* GetPlatformByIdMap() {
    static PlatformIdMap* instance = new PlatformIdMap;
    return instance;
  }

  // Looks up the platform object with the given name.  Assumes the Platforms
  // mutex is held.
  static port::StatusOr<Platform*> LookupByNameLocked(const string& target)
      EXCLUSIVE_LOCKS_REQUIRED(platforms_mutex_);

  // Looks up the platform object with the given id.  Assumes the Platforms
  // mutex is held.
  static port::StatusOr<Platform*> LookupByIdLocked(const Platform::Id& id)
      EXCLUSIVE_LOCKS_REQUIRED(platforms_mutex_);

  SE_DISALLOW_COPY_AND_ASSIGN(MultiPlatformManager);
};

}  // namespace stream_executor

// multi_platform_manager.cc will define this instance. Includers of this header
// should use
// REGISTER_MODULE_INITIALIZER_SEQUENCE(my_platform, multi_platform_manager);
DECLARE_MODULE_INITIALIZER(multi_platform_manager);

#endif  // TENSORFLOW_STREAM_EXECUTOR_MULTI_PLATFORM_MANAGER_H_
