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

// To compile: gcc -o c_api_client c_api_client.c -ldl
// To run, make sure c_api.so and c_api_client in the same directory, and then
//   sudo ./c_api_client

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  void* handle;
  handle = dlopen("./c_api.so", RTLD_NOW);
  if (!handle) {
    fprintf(stderr, "Error: %s\n", dlerror());
    exit(EXIT_FAILURE);
  }

  const char* (*TpuDriver_Version)(void);
  void (*TpuDriver_Initialize)(void);
  void (*TpuDriver_Open)(const char* worker);

  fprintf(stdout, "------ Going to Find Out Version ------\n");
  *(void**)(&TpuDriver_Version) = dlsym(handle, "TpuDriver_Version");
  fprintf(stdout, "TPU Driver Version: %s\n", TpuDriver_Version());

  fprintf(stdout, "------ Going to Initialize ------\n");
  *(void**)(&TpuDriver_Initialize) = dlsym(handle, "TpuDriver_Initialize");
  TpuDriver_Initialize();

  fprintf(stdout, "------ Going to Open a TPU Driver ------\n");
  *(void**)(&TpuDriver_Open) = dlsym(handle, "TpuDriver_Open");
  TpuDriver_Open("local://");

  dlclose(handle);
  exit(EXIT_SUCCESS);
}
