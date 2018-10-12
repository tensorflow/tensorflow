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

#ifndef TENSORFLOW_CORE_PLATFORM_FILE_SYSTEM_HELPER_H_
#define TENSORFLOW_CORE_PLATFORM_FILE_SYSTEM_HELPER_H_

#include <string>
#include <vector>

#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

class FileSystem;
class Env;

namespace internal {

// Given a pattern, stores in 'results' the set of paths (in the given file
// system) that match that pattern.
//
// This helper may be used by implementations of FileSystem::GetMatchingPaths()
// in order to provide parallel scanning of subdirectories (except on iOS).
//
// Arguments:
//   fs: may not be null and will be used to identify directories and list
//       their contents.
//   env: may not be null and will be used to check if a match has been found.
//   pattern: see FileSystem::GetMatchingPaths() for details.
//   results: will be cleared and may not be null.
//
// Returns an error status if any call to 'fs' failed.
Status GetMatchingPaths(FileSystem* fs, Env* env, const string& pattern,
                        std::vector<string>* results);

}  // namespace internal
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_FILE_SYSTEM_HELPER_H_
