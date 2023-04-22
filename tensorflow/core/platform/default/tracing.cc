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

#include "tensorflow/core/platform/tracing.h"

#include <stdlib.h>

#ifndef PLATFORM_WINDOWS
#include <unistd.h>
#endif

namespace tensorflow {
namespace tracing {
namespace {
bool TryGetEnv(const char* name, const char** value) {
  *value = getenv(name);
  return *value != nullptr && (*value)[0] != '\0';
}
}  // namespace

void EventCollector::SetCurrentThreadName(const char*) {}

const char* GetLogDir() {
  const char* dir;
  if (TryGetEnv("TEST_TMPDIR", &dir)) return dir;
  if (TryGetEnv("TMP", &dir)) return dir;
  if (TryGetEnv("TMPDIR", &dir)) return dir;
#ifndef PLATFORM_WINDOWS
  dir = "/tmp";
  if (access(dir, R_OK | W_OK | X_OK) == 0) return dir;
#endif
  return ".";  // Default to current directory.
}
}  // namespace tracing
}  // namespace tensorflow
