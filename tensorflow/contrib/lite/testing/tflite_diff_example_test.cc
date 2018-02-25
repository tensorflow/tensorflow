/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/contrib/lite/testing/tflite_diff_flags.h"
#include "tensorflow/contrib/lite/testing/tflite_diff_util.h"

int main(int argc, char** argv) {
  ::tflite::testing::DiffOptions options =
      ::tflite::testing::ParseTfliteDiffFlags(&argc, argv);
  for (int i = 0; i < 100; i++) {
    if (!tflite::testing::RunDiffTest(options)) {
      return 1;
    }
  }
  return 0;
}
