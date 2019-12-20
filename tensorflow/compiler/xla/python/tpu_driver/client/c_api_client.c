/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// Before you start, make sure c_api.so, c_api.h and and c_api_client.c are in
// the same working directory.
//
// To compile: gcc -o c_api_client c_api_client.c -ldl
// To run: sudo ./c_api_client

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

#include "c_api.h"

void* LoadAndInitializeDriver(const char* shared_lib,
                              struct TpuDriverFn* driver_fn) {
  void* handle;
  handle = dlopen("./c_api.so", RTLD_NOW);
  if (!handle) {
    fprintf(stderr, "Error: %s\n", dlerror());
    exit(EXIT_FAILURE);
  }

  PrototypeTpuDriver_Initialize* initialize_fn;
  *(void**)(&initialize_fn) = dlsym(handle, "TpuDriver_Initialize");
  initialize_fn(driver_fn);

  return handle;
}

int main(int argc, char** argv) {
  struct TpuDriverFn driver_fn;
  void* handle = LoadAndInitializeDriver("./c_api.so", &driver_fn);

  fprintf(stdout, "------ Going to Query Version ------\n");
  fprintf(stdout, "TPU Driver Version: %s\n", driver_fn.TpuDriver_Version());

  fprintf(stdout, "------ Going to Open a TPU Driver ------\n");
  struct TpuDriver* driver = driver_fn.TpuDriver_Open("local://");

  fprintf(stdout, "------ Going to Allocate a TPU Buffer ------\n");
  struct TpuBufferHandle* buffer_handle =
      driver_fn.TpuDriver_Allocate(driver, 0, 1, 32 * 1024 * 1024, 0, NULL);

  fprintf(stdout, "------ Going to Deallocate a TPU Buffer ------\n");
  struct TpuEvent* tpu_event =
      driver_fn.TpuDriver_Deallocate(driver, buffer_handle, 0, NULL);

  driver_fn.TpuDriver_FreeEvent(tpu_event);

  dlclose(handle);
  exit(EXIT_SUCCESS);
}
