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
  if (options.tensorflow_model.empty()) return 1;

  int failure_count = 0;
  for (int i = 0; i < options.num_runs_per_pass; i++) {
    if (!tflite::testing::RunDiffTest(options, /*num_invocations=*/1)) {
      ++failure_count;
    }
  }
  int failures_in_first_pass = failure_count;

  if (failure_count == 0) {
    // Let's try again with num_invocations > 1 to make sure we can do multiple
    // invocations without resetting the interpreter.
    for (int i = 0; i < options.num_runs_per_pass; i++) {
      if (!tflite::testing::RunDiffTest(options, /*num_invocations=*/2)) {
        ++failure_count;
      }
    }
  }

  fprintf(stderr, "Num errors in single-inference pass: %d\n",
          failures_in_first_pass);
  fprintf(stderr, "Num errors in multi-inference pass : %d\n",
          failure_count - failures_in_first_pass);

  return failure_count != 0 ? 1 : 0;
}
