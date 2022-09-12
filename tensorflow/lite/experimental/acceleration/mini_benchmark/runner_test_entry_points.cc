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

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/model_loader.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/status_codes.h"
#include "tensorflow/lite/schema/mutable/schema_generated.h"

extern "C" {

constexpr int kStdOutFd = 1;

int JustReturnZero(int argc, char** argv) { return 0; }

int ReturnOne(int argc, char** argv) { return 1; }

int ReturnSuccess(int argc, char** argv) {
  return ::tflite::acceleration::kMinibenchmarkSuccess;
}

int SigKill(int argc, char** argv) {
  kill(getpid(), SIGKILL);
  return 1;
}

int WriteOk(int argc, char** argv) {
  write(kStdOutFd, "ok\n", 3);
  return ::tflite::acceleration::kMinibenchmarkSuccess;
}

int Write10kChars(int argc, char** argv) {
  char buffer[10000];
  memset(buffer, 'A', 10000);
  return write(kStdOutFd, buffer, 10000) == 10000
             ? ::tflite::acceleration::kMinibenchmarkSuccess
             : 1;
}

int WriteArgs(int argc, char** argv) {
  for (int i = 3; i < argc; i++) {
    write(1, argv[i], strlen(argv[i]));
    write(1, "\n", 1);
  }
  return ::tflite::acceleration::kMinibenchmarkSuccess;
}

int ReadFromPipe(int argc, char** argv) {
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

int ReadFromPipeInProcess(int argc, char** argv) {
  std::unique_ptr<tflite::acceleration::ModelLoader> model_loader =
      tflite::acceleration::CreateModelLoaderFromPath(argv[3]);
  return model_loader->Init();
}

}  // extern "C"
