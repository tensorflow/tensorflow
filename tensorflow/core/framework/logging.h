/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_FRAMEWORK_LOGGING_H_
#define TENSORFLOW_CORE_FRAMEWORK_LOGGING_H_

#include <string>

namespace tensorflow {

namespace logging {

// Register a listener method to call on any printed messages.
// Returns true if it is successfully registered.
bool RegisterListener(void (*listener)(const char*));

// Log string to active listeners. Returns true if any listeners were
// registered.
bool LogToListeners(std::string msg, std::string end = "\n");

}  // namespace logging

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_LOGGING_H_
