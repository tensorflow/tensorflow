/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifdef TENSORFLOW_USE_JEMALLOC
#include "jemalloc/jemalloc.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef TF_USE_SNAPPY
#include "snappy.h"
#endif

#include <Windows.h>
#include <shlwapi.h>

#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/demangle.h"
#include "tensorflow/core/platform/host_info.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/snappy.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace port {

void InitMain(const char* usage, int* argc, char*** argv) {}

string Hostname() {
  char name[1024];
  DWORD name_size = sizeof(name);
  name[0] = 0;
  if (::GetComputerNameA(name, &name_size)) {
    name[name_size] = 0;
  }
  return name;
}

int NumSchedulableCPUs() {
  SYSTEM_INFO system_info;
  GetSystemInfo(&system_info);
  return system_info.dwNumberOfProcessors;
}

void* AlignedMalloc(size_t size, int minimum_alignment) {
#ifdef TENSORFLOW_USE_JEMALLOC
  void* ptr = NULL;
  // posix_memalign requires that the requested alignment be at least
  // sizeof(void*). In this case, fall back on malloc which should return
  // memory aligned to at least the size of a pointer.
  const int required_alignment = sizeof(void*);
  if (minimum_alignment < required_alignment) return Malloc(size);
  int err = jemalloc_posix_memalign(&ptr, minimum_alignment, size);
  if (err != 0) {
    return NULL;
  } else {
    return ptr;
  }
#else
  return _aligned_malloc(size, minimum_alignment);
#endif
}

void AlignedFree(void* aligned_memory) {
#ifdef TENSORFLOW_USE_JEMALLOC
  jemalloc_free(aligned_memory);
#else
  _aligned_free(aligned_memory);
#endif
}

void* Malloc(size_t size) {
#ifdef TENSORFLOW_USE_JEMALLOC
  return jemalloc_malloc(size);
#else
  return malloc(size);
#endif
}

void* Realloc(void* ptr, size_t size) {
#ifdef TENSORFLOW_USE_JEMALLOC
  return jemalloc_realloc(ptr, size);
#else
  return realloc(ptr, size);
#endif
}

void Free(void* ptr) {
#ifdef TENSORFLOW_USE_JEMALLOC
  return jemalloc_free(ptr);
#else
  return free(ptr);
#endif
}

void MallocExtension_ReleaseToSystem(std::size_t num_bytes) {
  // No-op.
}

std::size_t MallocExtension_GetAllocatedSize(const void* p) { return 0; }

void AdjustFilenameForLogging(string* filename) {
  // Nothing to do
}

bool Snappy_Compress(const char* input, size_t length, string* output) {
#ifdef TF_USE_SNAPPY
  output->resize(snappy::MaxCompressedLength(length));
  size_t outlen;
  snappy::RawCompress(input, length, &(*output)[0], &outlen);
  output->resize(outlen);
  return true;
#else
  return false;
#endif
}

bool Snappy_GetUncompressedLength(const char* input, size_t length,
                                  size_t* result) {
#ifdef TF_USE_SNAPPY
  return snappy::GetUncompressedLength(input, length, result);
#else
  return false;
#endif
}

bool Snappy_Uncompress(const char* input, size_t length, char* output) {
#ifdef TF_USE_SNAPPY
  return snappy::RawUncompress(input, length, output);
#else
  return false;
#endif
}

string Demangle(const char* mangled) { return mangled; }

double NominalCPUFrequency() {
  DWORD data;
  DWORD data_size = sizeof(data);
  #pragma comment(lib, "shlwapi.lib")  // For SHGetValue().
  if (SUCCEEDED(
          SHGetValueA(HKEY_LOCAL_MACHINE,
                      "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                      "~MHz", nullptr, &data, &data_size))) {
    return data * 1e6;  // Value is MHz.
  }
  return 1.0;
}

int64 AvailableRam() {
  MEMORYSTATUSEX statex;
  statex.dwLength = sizeof(statex);
  if (GlobalMemoryStatusEx(&statex)) {
    return statex.ullAvailPhys;
  }
  return INT64_MAX;
}

}  // namespace port
}  // namespace tensorflow
