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
#include "tensorflow/lite/testing/init_tensorflow.h"

#include <cstdlib>
#include <cstring>

#include "tensorflow/core/platform/init_main.h"

namespace tflite {
void InitTensorFlow() {
  static const char* kFakeName = "fake program name";
  int argc = 1;
  char* fake_name_copy = strdup(kFakeName);
  char** argv = &fake_name_copy;
  ::tensorflow::port::InitMain(kFakeName, &argc, &argv);
  free(fake_name_copy);
}
}  // namespace tflite
