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
#include <stdio.h>

#include "tensorflow/lite/experimental/microfrontend/lib/frontend.h"
#include "tensorflow/lite/experimental/microfrontend/lib/frontend_util.h"
#include "tensorflow/lite/experimental/microfrontend/lib/frontend_io.h"

int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr,
            "%s requires exactly two parameters - the names of the header and "
            "source files to save\n");
    return 1;
  }
  struct FrontendConfig frontend_config;
  FrontendFillConfigWithDefaults(&frontend_config);

  int sample_rate = 16000;
  struct FrontendState frontend_state;
  if (!FrontendPopulateState(&frontend_config, &frontend_state, sample_rate)) {
    fprintf(stderr, "Failed to populate frontend state\n");
    FrontendFreeStateContents(&frontend_state);
    return 1;
  }

  if (!WriteFrontendStateMemmap(argv[1], argv[2], &frontend_state)) {
    fprintf(stderr, "Failed to write memmap\n");
    FrontendFreeStateContents(&frontend_state);
    return 1;
  }

  FrontendFreeStateContents(&frontend_state);
  return 0;
}
