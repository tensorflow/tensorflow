/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_PLATFORM_CPU_INFO_H_
#define TENSORFLOW_PLATFORM_CPU_INFO_H_

namespace tensorflow {
namespace port {

// TODO(jeff,sanjay): Make portable
static const bool kLittleEndian = true;

// Returns an estimate of the number of schedulable CPUs for this
// process.  Usually, it's constant throughout the lifetime of a
// process, but it might change if the underlying cluster management
// software can change it dynamically.
int NumSchedulableCPUs();

}  // namespace port
}  // namespace tensorflow

#endif  // TENSORFLOW_PLATFORM_CPU_INFO_H_
