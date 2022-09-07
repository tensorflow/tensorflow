/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_TSL_PLATFORM_WINDOWS_ERROR_WINDOWS_H_
#define TENSORFLOW_TSL_PLATFORM_WINDOWS_ERROR_WINDOWS_H_

// This file is here to provide a windows specific interface to error functions
// without needing to include any windows specific headers. This is intended to
// reduce conflicts induced by code needing to run on multiple operating
// systems.

#include <string>

namespace tensorflow {
namespace internal {

// WindowsGetLastErrorMessage calls GetLastError() and then formats the error
// message for reporting.
std::string WindowsGetLastErrorMessage();

// WindowsWSLGetLastErrorMessage calls GetLastError() and then formats the error
// message for reporting.
std::string WindowsWSAGetLastErrorMessage();

}  // namespace internal
}  // namespace tensorflow

#endif  // TENSORFLOW_TSL_PLATFORM_WINDOWS_ERROR_WINDOWS_H_
