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

#ifndef TENSORFLOW_TSL_PLATFORM_INIT_MAIN_H_
#define TENSORFLOW_TSL_PLATFORM_INIT_MAIN_H_

namespace tsl {
namespace port {

void InitMain(const char* usage, int* argc, char*** argv);

}  // namespace port
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PLATFORM_INIT_MAIN_H_
