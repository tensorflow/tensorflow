/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_SINGLE_THREADED_CPU_DEVICE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_SINGLE_THREADED_CPU_DEVICE_H_

namespace tensorflow {

class Device;
class Env;

// Returns a simple single-threaded CPU device. This can be used to run
// inexpensive computations. In particular, using this avoids initializing the
// global thread pools in LocalDevice.
//
// The returned pointer is owned by the caller.
Device* NewSingleThreadedCpuDevice(Env* env);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_SINGLE_THREADED_CPU_DEVICE_H_
