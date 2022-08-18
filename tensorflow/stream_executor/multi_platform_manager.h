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

#include "tensorflow/compiler/xla/stream_executor/multi_platform_manager.h"

#endif  // TENSORFLOW_STREAM_EXECUTOR_MULTI_PLATFORM_MANAGER_H_
