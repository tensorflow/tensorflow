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
// NOTE: this is an example driver that converts a tflite model to TensorFlow.
// This is an example that will be integrated more tightly into tflite in
// the future.
//
// Usage: bazel run -c opt \
// tensorflow/lite/nnapi:nnapi_example -- <filename>
//
#include <dirent.h>
#include <cstdarg>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include "tensorflow/lite/nnapi/NeuralNetworksShim.h"
#include "tensorflow/lite/testing/parse_testdata.h"
#include "tensorflow/lite/testing/tflite_driver.h"

string dirname(const string& s) { return s.substr(0, s.find_last_of("/")); }

bool Interpret(const char* examples_filename, bool use_nnapi) {
  std::ifstream tflite_stream(examples_filename);
  if (!tflite_stream.is_open()) {
    fprintf(stderr, "Can't open input file.");
    return false;
  }

  printf("Use nnapi is set to: %d\n", use_nnapi);
  tflite::testing::TfLiteDriver test_driver(use_nnapi);

  test_driver.SetModelBaseDir(dirname(examples_filename));
  if (!tflite::testing::ParseAndRunTests(&tflite_stream, &test_driver)) {
    fprintf(stderr, "Results from tflite don't match.");
    return false;
  }

  return true;
}

int main(int argc, char* argv[]) {
  bool use_nnapi = true;
  if (argc == 4) {
    use_nnapi = strcmp(argv[3], "1") == 0 ? true : false;
  }
  if (argc < 3) {
    fprintf(stderr,
            "Compiled " __DATE__ __TIME__
            "\n"
            "Usage!!!: %s <tflite model> <examples to test> "
            "{ use nn api i.e. 0,1}\n",
            argv[0]);
    return 1;
  }

  string base_dir = dirname(argv[1]);
  DIR* dir = opendir(base_dir.c_str());
  if (dir == nullptr) {
    fprintf(stderr, "Can't open dir %s\n", base_dir.c_str());
    return 1;
  }
  while (struct dirent* ent = readdir(dir)) {
    string name = ent->d_name;
    if (name.rfind(".txt") == name.length() - 4) {
      printf("%s: ", name.c_str());
      if (Interpret((base_dir + "/" + name).c_str(), use_nnapi)) {
        printf(" %s\n", "OK");
      } else {
        printf(" %s\n", "FAIL");
      }
    }
  }
  closedir(dir);

  return 0;
}
