/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include <stdio.h>

#ifdef _WIN32
#include <direct.h>
#include <stdlib.h>
#ifndef PATH_MAX
#include <windows.h>
#define PATH_MAX MAX_PATH
#endif
#define getcwd _getcwd
#else
#include <limits.h>
#include <unistd.h>
#endif

int main(int argc, char* argv[]) {
  char buf[PATH_MAX];
  if (getcwd(buf, sizeof(buf)) != nullptr) {
    printf("%s", buf);
  } else {
    return 1;
  }
  return 0;
}
