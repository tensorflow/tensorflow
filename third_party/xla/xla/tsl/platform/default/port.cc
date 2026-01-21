/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "absl/base/internal/sysinfo.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/profile_utils/cpu_utils.h"
#include "xla/tsl/platform/types.h"
#include "tsl/platform/cpu_info.h"
#include "tsl/platform/host_info.h"
#include "tsl/platform/mem.h"
#include "tsl/platform/numa.h"
#include "tsl/platform/snappy.h"

#if defined(__linux__)
#include <sched.h>
#include <sys/sysinfo.h>
#elif defined(_WIN32)
#include <winsock.h>
#else
#include <sys/syscall.h>
#endif

#if (__x86_64__ || __i386__)
#include <cpuid.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#ifdef TF_USE_SNAPPY
#include "snappy.h"
#endif
#if (defined(__APPLE__) && defined(__MACH__)) || defined(__FreeBSD__) || \
    defined(__HAIKU__)
#include <thread>
#endif

#if defined(__ANDROID__) && (defined(__i386__) || defined(__x86_64__))
#define TENSORFLOW_HAS_CXA_DEMANGLE 0
#elif (__GNUC__ >= 4 || (__GNUC__ >= 3 && __GNUC_MINOR__ >= 4)) && \
    !defined(__mips__)
#define TENSORFLOW_HAS_CXA_DEMANGLE 1
#elif defined(__clang__) && !defined(_MSC_VER)
#define TENSORFLOW_HAS_CXA_DEMANGLE 1
#else
#define TENSORFLOW_HAS_CXA_DEMANGLE 0
#endif

#if TENSORFLOW_HAS_CXA_DEMANGLE
#include <cxxabi.h>
#endif

namespace {
absl::Time GetStartupTime() {
  static absl::Time start_up_time = absl::Now();
  return start_up_time;
}
}  // namespace

namespace tsl {
namespace port {

void InitMain(const char* usage, int* argc, char*** argv) { GetStartupTime(); }

absl::Duration GetUptime() { return absl::Now() - GetStartupTime(); }

std::string Hostname() {
  char hostname[1024];
  gethostname(hostname, sizeof hostname);
  hostname[sizeof hostname - 1] = 0;
  return std::string(hostname);
}

std::string JobName() {
  const char* job_name_cs = std::getenv("TF_JOB_NAME");
  if (job_name_cs != nullptr) {
    return std::string(job_name_cs);
  }
  return "";
}

int64_t JobUid() { return -1; }

int64_t TaskId() { return -1; }

int NumSchedulableCPUs() {
#if defined(__linux__)
  for (int ncpus = 1024; ncpus < std::numeric_limits<int>::max() / 2;
       ncpus *= 2) {
    size_t setsize = CPU_ALLOC_SIZE(ncpus);
    cpu_set_t* mask = CPU_ALLOC(ncpus);
    if (!mask) break;
    if (sched_getaffinity(0, setsize, mask) == 0) {
      int result = CPU_COUNT_S(setsize, mask);
      CPU_FREE(mask);
      return result;
    }
    CPU_FREE(mask);
    if (errno != EINVAL) break;
  }
  perror("sched_getaffinity");
#endif
#if (defined(__APPLE__) && defined(__MACH__)) || defined(__FreeBSD__) || \
    defined(__HAIKU__)
  unsigned int count = std::thread::hardware_concurrency();
  if (count > 0) return static_cast<int>(count);
#endif
  const int kDefaultCores = 4;  // Semi-conservative guess
  fprintf(stderr, "can't determine number of CPU cores: assuming %d\n",
          kDefaultCores);
  return kDefaultCores;
}

int MaxParallelism() { return NumSchedulableCPUs(); }

int MaxParallelism(int numa_node) {
  if (numa_node != port::kNUMANoAffinity) {
    // Assume that CPUs are equally distributed over available NUMA nodes.
    // This may not be true, but there isn't currently a better way of
    // determining the number of CPUs specific to the requested node.
    return NumSchedulableCPUs() / port::NUMANumNodes();
  }
  return NumSchedulableCPUs();
}

int NumTotalCPUs() {
  int count = absl::base_internal::NumCPUs();
  return (count <= 0) ? kUnknownCPU : count;
}

int GetCurrentCPU() {
#if defined(__EMSCRIPTEN__)
  return sched_getcpu();
#elif defined(__linux__)
  return sched_getcpu();
  // Attempt to use cpuid on all other platforms.  If that fails, perform a
  // syscall.
#elif defined(__cpuid) && !defined(__APPLE__)
  // TODO(b/120919972): __cpuid returns invalid APIC ids on OS X.
  uint32_t eax = 0;
  uint32_t ebx = 0;
  uint32_t ecx = 0;
  uint32_t edx = 0;
  __cpuid(/*level=*/1, eax, ebx, ecx, edx);
  if ((edx & /*bit_APIC=*/(1 << 9)) != 0) {
    // EBX bits 24-31 are APIC ID
    return (ebx & 0xFF) >> 24;
  }
#elif defined(__NR_getcpu)
  unsigned int cpu;
  if (syscall(__NR_getcpu, &cpu, NULL, NULL) < 0) {
    return kUnknownCPU;
  } else {
    return static_cast<int>(cpu);
  }
#endif
  return kUnknownCPU;
}

int NumHyperthreadsPerCore() {
  static const int ht_per_core = tsl::port::CPUIDNumSMT();
  return (ht_per_core > 0) ? ht_per_core : 1;
}

bool Snappy_Compress(const char* input, size_t length, std::string* output) {
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

bool Snappy_CompressFromIOVec(const struct iovec* iov,
                              size_t uncompressed_length, std::string* output) {
#ifdef TF_USE_SNAPPY
  output->resize(snappy::MaxCompressedLength(uncompressed_length));
  size_t outlen;
  snappy::RawCompressFromIOVec(iov, uncompressed_length, &(*output)[0],
                               &outlen);
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

bool Snappy_UncompressToIOVec(const char* compressed, size_t compressed_length,
                              const struct iovec* iov, size_t iov_cnt) {
#ifdef TF_USE_SNAPPY
  return snappy::RawUncompressToIOVec(compressed, compressed_length, iov,
                                      iov_cnt);
#else
  return false;
#endif
}

static void DemangleToString(const char* mangled, std::string* out) {
  int status = 0;
  char* demangled = nullptr;
#if TENSORFLOW_HAS_CXA_DEMANGLE
  demangled = abi::__cxa_demangle(mangled, nullptr, nullptr, &status);
#endif
  if (status == 0 && demangled != nullptr) {  // Demangling succeeded.
    out->append(demangled);
    free(demangled);
  } else {
    out->append(mangled);
  }
}

std::string Demangle(const char* mangled) {
  std::string demangled;
  DemangleToString(mangled, &demangled);
  return demangled;
}

double NominalCPUFrequency() {
  return tsl::profile_utils::CpuUtils::GetCycleCounterFrequency();
}

}  // namespace port
}  // namespace tsl

namespace tsl {
namespace port {

void* AlignedMalloc(size_t size, std::align_val_t minimum_alignment) {
  const size_t alignment = static_cast<size_t>(minimum_alignment);
#if defined(__ANDROID__)
  return memalign(alignment, size);
#else  // !defined(__ANDROID__)
  // posix_memalign requires that the requested alignment be at least
  // sizeof(void*). In this case, fall back on malloc which should return
  // memory aligned to at least the size of a pointer.
  constexpr int kRequiredAlignment = sizeof(void*);
  if (alignment < kRequiredAlignment) {
    return Malloc(size);
  }
  void* ptr = nullptr;
  int err = posix_memalign(&ptr, alignment, size);
  if (err != 0) {
    return nullptr;
  }
  return ptr;
#endif
}

void AlignedFree(void* aligned_memory) { Free(aligned_memory); }

void AlignedSizedFree(void* aligned_memory, size_t size,
                      std::align_val_t alignment) {
  (void)alignment;
  (void)size;

  Free(aligned_memory);
}

void* Malloc(size_t size) { return malloc(size); }

void* Realloc(void* ptr, size_t size) { return realloc(ptr, size); }

void Free(void* ptr) { free(ptr); }

void MallocExtension_ReleaseToSystem(std::size_t num_bytes) {
  // No-op.
}

std::size_t MallocExtension_GetAllocatedSize(const void* p) {
#if !defined(__ANDROID__)
  return 0;
#else
  return malloc_usable_size(p);
#endif
}

MemoryInfo GetMemoryInfo() {
  MemoryInfo mem_info = {INT64_MAX, INT64_MAX};
#if defined(__linux__)
  struct sysinfo info;
  int err = sysinfo(&info);
  if (err == 0) {
    mem_info.free = info.freeram;
    mem_info.total = info.totalram;
  }
#endif
  return mem_info;
}

MemoryBandwidthInfo GetMemoryBandwidthInfo() {
  MemoryBandwidthInfo membw_info = {INT64_MAX};
  return membw_info;
}

IOStatistics GetIOStatistics() { return IOStatistics(); }

}  // namespace port
}  // namespace tsl
