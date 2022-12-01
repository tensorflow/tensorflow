/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#include <iostream>
#include <memory>
#include <string>

#include "absl/strings/numbers.h"
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/constants.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/model_loader.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/status_codes.h"
#include "tensorflow/lite/schema/mutable/schema_generated.h"

extern "C" {

constexpr int kStdOutFd = 1;

int TfLiteJustReturnZero(int argc, char** argv) { return 0; }

int TfLiteReturnOne(int argc, char** argv) { return 1; }

int TfLiteReturnSuccess(int argc, char** argv) {
  return ::tflite::acceleration::kMinibenchmarkSuccess;
}

int TfLiteSigKillSelf(int argc, char** argv) {
  kill(getpid(), SIGKILL);
  return 1;
}

int TfLiteWriteOk(int argc, char** argv) {
  write(kStdOutFd, "ok\n", 3);
  return ::tflite::acceleration::kMinibenchmarkSuccess;
}

// Write the pid to output stream and then sleep N seconds. N is parsed from
// argv[3].
int TfLiteWritePidThenSleepNSec(int argc, char** argv) {
  std::string pid = std::to_string(getpid());
  pid.resize(::tflite::acceleration::kPidBufferLength);
  write(kStdOutFd, pid.data(), ::tflite::acceleration::kPidBufferLength);

  int sleep_sec;
  if (!absl::SimpleAtoi(argv[3], &sleep_sec)) {
    return 1;
  }
  sleep(sleep_sec);
  return ::tflite::acceleration::kMinibenchmarkSuccess;
}

int TfLiteWrite10kChars(int argc, char** argv) {
  char buffer[10000];
  memset(buffer, 'A', 10000);
  return write(kStdOutFd, buffer, 10000) == 10000
             ? ::tflite::acceleration::kMinibenchmarkSuccess
             : 1;
}

int TfLiteWriteArgs(int argc, char** argv) {
  for (int i = 3; i < argc; i++) {
    write(1, argv[i], strlen(argv[i]));
    write(1, "\n", 1);
  }
  return ::tflite::acceleration::kMinibenchmarkSuccess;
}

int TfLiteReadFromPipe(int argc, char** argv) {
  std::unique_ptr<tflite::acceleration::ModelLoader> model_loader =
      tflite::acceleration::CreateModelLoaderFromPath(argv[3]);
  if (model_loader->Init() != tflite::acceleration::kMinibenchmarkSuccess) {
    return 1;
  }
  tflite::ModelT model;
  model_loader->GetModel()->GetModel()->UnPackTo(&model);
  flatbuffers::FlatBufferBuilder fbb;
  fbb.Finish(CreateModel(fbb, &model));

  return write(kStdOutFd, fbb.GetBufferPointer(), fbb.GetSize()) ==
                 fbb.GetSize()
             ? ::tflite::acceleration::kMinibenchmarkSuccess
             : 1;
}

int TfLiteReadFromPipeInProcess(int argc, char** argv) {
  std::unique_ptr<tflite::acceleration::ModelLoader> model_loader =
      tflite::acceleration::CreateModelLoaderFromPath(argv[3]);
  return model_loader->Init();
}

}  // extern "C"
