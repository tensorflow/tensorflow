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

#include <limits.h>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/c_api_experimental.h"
#include "tensorflow/c/env.h"
#include "tensorflow/c/kernels.h"

// A compute function. This will never actually get called in this test, it's
// just nice to know that it compiles.
void compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Tensor* input;
  TF_Status* s = TF_NewStatus();
  TF_GetInput(ctx, 0, &input, s);
  TF_DeleteTensor(input);

  TF_DataType type;
  TF_OpKernelContext_GetAttrType(ctx, "foobar", &type, s);

  TF_DeleteStatus(s);

}

// Exercises tensorflow's C API.
int main(int argc, char** argv) {
  TF_InitMain(argv[0], &argc, &argv);

  struct TF_StringStream* s = TF_GetLocalTempDirectories();
  const char* path;

  if (!TF_StringStreamNext(s, &path)) {
    fprintf(stderr, "TF_GetLocalTempDirectories returned no results\n");
    return 1;
  }

  char file_name[100];
  struct timeval t;
  if (gettimeofday(&t, NULL)) {
    perror("gettimeofday failed");
    return 1;
  }
  snprintf(file_name, sizeof(file_name), "test-%d-%ld.txt", getpid(), t.tv_sec);

  size_t length = 2 + strlen(path) + strlen(file_name);
  char* full_path = malloc(length);
  snprintf(full_path, length, "%s/%s", path, file_name);

  TF_WritableFileHandle* h;
  TF_Status* status = TF_NewStatus();
  TF_NewWritableFile(full_path, &h, status);
  if (TF_GetCode(status) != TF_OK) {
    fprintf(stderr, "TF_NewWritableFile failed: %s\n", TF_Message(status));
    return 1;
  }
  fprintf(stderr, "wrote %s\n", full_path);
  free(full_path);
  TF_CloseWritableFile(h, status);
  if (TF_GetCode(status) != TF_OK) {
    fprintf(stderr, "TF_CloseWritableFile failed: %s\n", TF_Message(status));
  }
  TF_StringStreamDone(s);

  TF_KernelBuilder* b =
      TF_NewKernelBuilder("SomeOp", "SomeDevice", NULL, &compute, NULL);
  TF_RegisterKernelBuilder("someKernel", b, status);

  TF_DeleteStatus(status);
  return 0;
}
