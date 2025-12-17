/* Copyright 2025 The OpenXLA Authors.

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

#include <cstddef>
#include <memory>
#include <type_traits>

#include "absl/base/call_once.h"
#include "absl/log/log.h"
#include "hwloc.h"
#include "tsl/platform/mem.h"
#include "tsl/platform/numa.h"

namespace tsl {
namespace port {

namespace {
hwloc_topology_t GetHWLocTopology() {
  static absl::once_flag init_once;
  static hwloc_topology_t hwloc_topology_handle = nullptr;
  absl::call_once(init_once, [] {
    if (hwloc_topology_init(&hwloc_topology_handle)) {
      LOG(ERROR) << "Call to hwloc_topology_init() failed";
      return;
    }
    if (hwloc_topology_set_flags(hwloc_topology_handle,
                                 HWLOC_TOPOLOGY_FLAG_DONT_CHANGE_BINDING)) {
      LOG(ERROR) << "Call to hwloc_topology_set_flags() failed";
      return;
    }
    if (hwloc_topology_load(hwloc_topology_handle)) {
      LOG(ERROR) << "Call to hwloc_topology_load() failed";
      return;
    }
  });
  return hwloc_topology_handle;
}

// Return the first hwloc object of the given type whose os_index
// matches 'index'.
hwloc_obj_t GetHWLocTypeIndex(hwloc_obj_type_t tp, int index) {
  auto* topology = GetHWLocTopology();
  if (!topology) {
    return nullptr;
  }

  if (index < 0) {
    return nullptr;
  }

  hwloc_obj_t obj = nullptr;
  while ((obj = hwloc_get_next_obj_by_type(topology, tp, obj)) != nullptr) {
    if (obj->os_index == index) {
      break;
    }
  }
  return obj;
}

struct HWLocBitmapDeleter {
  void operator()(hwloc_bitmap_t bitmap) const { hwloc_bitmap_free(bitmap); }
};

auto AllocateBitmap() {
  return std::unique_ptr<std::remove_pointer_t<hwloc_bitmap_t>,
                         HWLocBitmapDeleter>(hwloc_bitmap_alloc());
}
}  // namespace

bool NUMAEnabled() { return NUMANumNodes() > 1; }

int NUMANumNodes() {
  static int num_numanodes = 1;
  static absl::once_flag init_once;
  absl::call_once(init_once, [] {
    auto* topology = GetHWLocTopology();
    if (!topology) {
      return;
    }
    num_numanodes = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_NUMANODE);
    if (num_numanodes < 1) {
      LOG(ERROR) << "Unknown number of NUMA nodes (got " << num_numanodes
                 << "), assuming 1.";
      num_numanodes = 1;
    }
  });
  return num_numanodes;
}

void NUMASetThreadNodeAffinity(int node) {
  if (node == kNUMANoAffinity) {
    return;
  }

  auto* topology = GetHWLocTopology();
  if (!topology) {
    return;
  }

  // Find the corresponding NUMA node topology object.
  hwloc_obj_t obj = GetHWLocTypeIndex(HWLOC_OBJ_NUMANODE, node);
  if (!obj) {
    LOG(ERROR) << "Could not find hwloc NUMA node " << node;
    return;
  }

  if (hwloc_set_cpubind(topology, obj->cpuset,
                        HWLOC_CPUBIND_THREAD | HWLOC_CPUBIND_STRICT)) {
    LOG(ERROR).WithPerror() << "Call to hwloc_set_cpubind() failed";
  }
}

int NUMAGetThreadNodeAffinity() {
  auto* topology = GetHWLocTopology();
  if (!topology) {
    return kNUMANoAffinity;
  }

  auto thread_cpuset = AllocateBitmap();
  if (!thread_cpuset) {
    LOG(ERROR) << "Call to hwloc_bitmap_alloc() failed";
    return kNUMANoAffinity;
  }

  if (hwloc_get_cpubind(topology, thread_cpuset.get(), HWLOC_CPUBIND_THREAD)) {
    LOG(ERROR).WithPerror() << "Call to hwloc_get_cpubind() failed";
    return kNUMANoAffinity;
  }

  hwloc_obj_t obj = nullptr;
  // Return the first NUMA node whose cpuset is a (non-proper) superset of
  // that of the current thread.
  while ((obj = hwloc_get_next_obj_by_type(topology, HWLOC_OBJ_NUMANODE,
                                           obj)) != nullptr) {
    if (hwloc_bitmap_isincluded(thread_cpuset.get(), obj->cpuset)) {
      break;
    }
  }
  return obj ? obj->os_index : kNUMANoAffinity;
}

void* NUMAMalloc(int node, size_t size, int minimum_alignment) {
  if (node != kNUMANoAffinity) {
    if (auto* topology = GetHWLocTopology()) {
      hwloc_obj_t numa_node = GetHWLocTypeIndex(HWLOC_OBJ_NUMANODE, node);
      if (numa_node) {
        return hwloc_alloc_membind(topology, size, numa_node->nodeset,
                                   HWLOC_MEMBIND_BIND, HWLOC_MEMBIND_BYNODESET);
      }
      LOG(ERROR) << "Failed to find hwloc NUMA node " << node;
    }
  }
  return AlignedMalloc(size, static_cast<std::align_val_t>(minimum_alignment));
}

void NUMAFree(void* ptr, size_t size) {
  auto* topology = GetHWLocTopology();
  if (!topology) {
    ::tsl::port::Free(ptr);
    return;
  }
  hwloc_free(topology, ptr, size);
}

int NUMAGetMemAffinity(const void* ptr) {
  if (!ptr) {
    return kNUMANoAffinity;
  }

  auto* topology = GetHWLocTopology();
  if (!topology) {
    return kNUMANoAffinity;
  }

  auto nodeset = AllocateBitmap();
  if (!nodeset) {
    LOG(ERROR) << "Call to hwloc_bitmap_alloc() failed";
    return kNUMANoAffinity;
  }

  if (hwloc_get_area_memlocation(topology, ptr, 4, nodeset.get(),
                                 HWLOC_MEMBIND_BYNODESET)) {
    LOG(ERROR) << "Failed call to hwloc_get_area_memlocation.";
    return kNUMANoAffinity;
  }

  hwloc_obj_t obj = nullptr;
  while ((obj = hwloc_get_next_obj_by_type(topology, HWLOC_OBJ_NUMANODE,
                                           obj)) != nullptr) {
    if (hwloc_bitmap_isincluded(nodeset.get(), obj->nodeset)) {
      break;
    }
  }
  return obj ? obj->os_index : kNUMANoAffinity;
}

}  // namespace port
}  // namespace tsl
