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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef TF_USE_SNAPPY
#include "snappy.h"
#endif

#include <Windows.h>
#include <processthreadsapi.h>
#include <shlwapi.h>

#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/demangle.h"
#include "tensorflow/core/platform/host_info.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/numa.h"
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

int NumTotalCPUs() {
  // TODO(ebrevdo): Make this more accurate.
  //
  // This only returns the number of processors in the current
  // processor group; which may be undercounting if you have more than 64 cores.
  // For that case, one needs to call
  // GetLogicalProcessorInformationEx(RelationProcessorCore, ...) and accumulate
  // the Size fields by iterating over the written-to buffer.  Since I can't
  // easily test this on Windows, I'm deferring this to someone who can!
  //
  // If you fix this, also consider updatig GetCurrentCPU below.
  return NumSchedulableCPUs();
}

int GetCurrentCPU() {
  // NOTE(ebrevdo): This returns the processor number within the processor
  // group on systems with >64 processors.  Therefore it doesn't necessarily map
  // naturally to an index in NumSchedulableCPUs().
  //
  // On the plus side, this number is probably guaranteed to be within
  // [0, NumTotalCPUs()) due to its incomplete implementation.
  return GetCurrentProcessorNumber();
}

bool NUMAEnabled() {
  // Not yet implemented: coming soon.
  return false;
}

int NUMANumNodes() { return 1; }

void NUMASetThreadNodeAffinity(int node) {}

int NUMAGetThreadNodeAffinity() { return kNUMANoAffinity; }

void* AlignedMalloc(size_t size, int minimum_alignment) {
  return _aligned_malloc(size, minimum_alignment);
}

void AlignedFree(void* aligned_memory) { _aligned_free(aligned_memory); }

void* Malloc(size_t size) { return malloc(size); }

void* Realloc(void* ptr, size_t size) { return realloc(ptr, size); }

void Free(void* ptr) { return free(ptr); }

void* NUMAMalloc(int node, size_t size, int minimum_alignment) {
  return AlignedMalloc(size, minimum_alignment);
}

void NUMAFree(void* ptr, size_t size) { Free(ptr); }

int NUMAGetMemAffinity(const void* addr) { return kNUMANoAffinity; }

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

int NumHyperthreadsPerCore() {
  static const int ht_per_core = tensorflow::port::CPUIDNumSMT();
  return (ht_per_core > 0) ? ht_per_core : 1;
}

}  // namespace port
}  // namespace tensorflow
