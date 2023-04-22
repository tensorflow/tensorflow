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
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/numa.h"
#include "tensorflow/core/platform/profile_utils/cpu_utils.h"
#include "tensorflow/core/platform/snappy.h"
#include "tensorflow/core/platform/types.h"

#if defined(__linux__) && !defined(__ANDROID__)
#include <sched.h>
#include <sys/sysinfo.h>
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

#if TENSORFLOW_USE_NUMA
#include "hwloc.h"  // from @hwloc
#endif

namespace tensorflow {
namespace port {

void InitMain(const char* usage, int* argc, char*** argv) {}

string Hostname() {
  char hostname[1024];
  gethostname(hostname, sizeof hostname);
  hostname[sizeof hostname - 1] = 0;
  return string(hostname);
}

string JobName() {
  const char* job_name_cs = std::getenv("TF_JOB_NAME");
  if (job_name_cs != nullptr) {
    return string(job_name_cs);
  }
  return "";
}

int NumSchedulableCPUs() {
#if defined(__linux__) && !defined(__ANDROID__)
  cpu_set_t cpuset;
  if (sched_getaffinity(0, sizeof(cpu_set_t), &cpuset) == 0) {
    return CPU_COUNT(&cpuset);
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
#elif defined(__linux__) && !defined(__ANDROID__)
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
  static const int ht_per_core = tensorflow::port::CPUIDNumSMT();
  return (ht_per_core > 0) ? ht_per_core : 1;
}

#ifdef TENSORFLOW_USE_NUMA
namespace {
static hwloc_topology_t hwloc_topology_handle;

bool HaveHWLocTopology() {
  // One time initialization
  static bool init = []() {
    if (hwloc_topology_init(&hwloc_topology_handle)) {
      LOG(ERROR) << "Call to hwloc_topology_init() failed";
      return false;
    }
    if (hwloc_topology_load(hwloc_topology_handle)) {
      LOG(ERROR) << "Call to hwloc_topology_load() failed";
      return false;
    }
    return true;
  }();
  return init;
}

// Return the first hwloc object of the given type whose os_index
// matches 'index'.
hwloc_obj_t GetHWLocTypeIndex(hwloc_obj_type_t tp, int index) {
  hwloc_obj_t obj = nullptr;
  if (index >= 0) {
    while ((obj = hwloc_get_next_obj_by_type(hwloc_topology_handle, tp, obj)) !=
           nullptr) {
      if (obj->os_index == index) break;
    }
  }
  return obj;
}
}  // namespace
#endif  // TENSORFLOW_USE_NUMA

bool NUMAEnabled() { return (NUMANumNodes() > 1); }

int NUMANumNodes() {
#ifdef TENSORFLOW_USE_NUMA
  if (HaveHWLocTopology()) {
    int num_numanodes =
        hwloc_get_nbobjs_by_type(hwloc_topology_handle, HWLOC_OBJ_NUMANODE);
    return std::max(1, num_numanodes);
  } else {
    return 1;
  }
#else
  return 1;
#endif  // TENSORFLOW_USE_NUMA
}

void NUMASetThreadNodeAffinity(int node) {
#ifdef TENSORFLOW_USE_NUMA
  if (HaveHWLocTopology()) {
    // Find the corresponding NUMA node topology object.
    hwloc_obj_t obj = GetHWLocTypeIndex(HWLOC_OBJ_NUMANODE, node);
    if (obj) {
      hwloc_set_cpubind(hwloc_topology_handle, obj->cpuset,
                        HWLOC_CPUBIND_THREAD | HWLOC_CPUBIND_STRICT);
    } else {
      LOG(ERROR) << "Could not find hwloc NUMA node " << node;
    }
  }
#endif  // TENSORFLOW_USE_NUMA
}

int NUMAGetThreadNodeAffinity() {
  int node_index = kNUMANoAffinity;
#ifdef TENSORFLOW_USE_NUMA
  if (HaveHWLocTopology()) {
    hwloc_cpuset_t thread_cpuset = hwloc_bitmap_alloc();
    hwloc_get_cpubind(hwloc_topology_handle, thread_cpuset,
                      HWLOC_CPUBIND_THREAD);
    hwloc_obj_t obj = nullptr;
    // Return the first NUMA node whose cpuset is a (non-proper) superset of
    // that of the current thread.
    while ((obj = hwloc_get_next_obj_by_type(
                hwloc_topology_handle, HWLOC_OBJ_NUMANODE, obj)) != nullptr) {
      if (hwloc_bitmap_isincluded(thread_cpuset, obj->cpuset)) {
        node_index = obj->os_index;
        break;
      }
    }
    hwloc_bitmap_free(thread_cpuset);
  }
#endif  // TENSORFLOW_USE_NUMA
  return node_index;
}

void* AlignedMalloc(size_t size, int minimum_alignment) {
#if defined(__ANDROID__)
  return memalign(minimum_alignment, size);
#else  // !defined(__ANDROID__)
  void* ptr = nullptr;
  // posix_memalign requires that the requested alignment be at least
  // sizeof(void*). In this case, fall back on malloc which should return
  // memory aligned to at least the size of a pointer.
  const int required_alignment = sizeof(void*);
  if (minimum_alignment < required_alignment) return Malloc(size);
  int err = posix_memalign(&ptr, minimum_alignment, size);
  if (err != 0) {
    return nullptr;
  } else {
    return ptr;
  }
#endif
}

void AlignedFree(void* aligned_memory) { Free(aligned_memory); }

void* Malloc(size_t size) { return malloc(size); }

void* Realloc(void* ptr, size_t size) { return realloc(ptr, size); }

void Free(void* ptr) { free(ptr); }

void* NUMAMalloc(int node, size_t size, int minimum_alignment) {
#ifdef TENSORFLOW_USE_NUMA
  if (HaveHWLocTopology()) {
    hwloc_obj_t numa_node = GetHWLocTypeIndex(HWLOC_OBJ_NUMANODE, node);
    if (numa_node) {
      return hwloc_alloc_membind(hwloc_topology_handle, size,
                                 numa_node->nodeset, HWLOC_MEMBIND_BIND,
                                 HWLOC_MEMBIND_BYNODESET);
    } else {
      LOG(ERROR) << "Failed to find hwloc NUMA node " << node;
    }
  }
#endif  // TENSORFLOW_USE_NUMA
  return AlignedMalloc(size, minimum_alignment);
}

void NUMAFree(void* ptr, size_t size) {
#ifdef TENSORFLOW_USE_NUMA
  if (HaveHWLocTopology()) {
    hwloc_free(hwloc_topology_handle, ptr, size);
    return;
  }
#endif  // TENSORFLOW_USE_NUMA
  Free(ptr);
}

int NUMAGetMemAffinity(const void* addr) {
  int node = kNUMANoAffinity;
#ifdef TENSORFLOW_USE_NUMA
  if (HaveHWLocTopology() && addr) {
    hwloc_nodeset_t nodeset = hwloc_bitmap_alloc();
    if (!hwloc_get_area_memlocation(hwloc_topology_handle, addr, 4, nodeset,
                                    HWLOC_MEMBIND_BYNODESET)) {
      hwloc_obj_t obj = nullptr;
      while ((obj = hwloc_get_next_obj_by_type(
                  hwloc_topology_handle, HWLOC_OBJ_NUMANODE, obj)) != nullptr) {
        if (hwloc_bitmap_isincluded(nodeset, obj->nodeset)) {
          node = obj->os_index;
          break;
        }
      }
      hwloc_bitmap_free(nodeset);
    } else {
      LOG(ERROR) << "Failed call to hwloc_get_area_memlocation.";
    }
  }
#endif  // TENSORFLOW_USE_NUMA
  return node;
}

void MallocExtension_ReleaseToSystem(std::size_t num_bytes) {
  // No-op.
}

std::size_t MallocExtension_GetAllocatedSize(const void* p) { return 0; }

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

bool Snappy_UncompressToIOVec(const char* compressed, size_t compressed_length,
                              const struct iovec* iov, size_t iov_cnt) {
#ifdef TF_USE_SNAPPY
  return snappy::RawUncompressToIOVec(compressed, compressed_length, iov,
                                      iov_cnt);
#else
  return false;
#endif
}

string Demangle(const char* mangled) { return mangled; }

double NominalCPUFrequency() {
  return tensorflow::profile_utils::CpuUtils::GetCycleCounterFrequency();
}

MemoryInfo GetMemoryInfo() {
  MemoryInfo mem_info = {INT64_MAX, INT64_MAX};
#if defined(__linux__) && !defined(__ANDROID__)
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

}  // namespace port
}  // namespace tensorflow
