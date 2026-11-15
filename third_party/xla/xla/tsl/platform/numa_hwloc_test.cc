/* Copyright 2026 The OpenXLA Authors.

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

#include "hwloc.h"
#include "xla/tsl/platform/test.h"

namespace tsl {
namespace port {

hwloc_topology_t GetHWLocTopology();

namespace {

TEST(NumaHwlocTest, TopologyFiltersOutNonNumaObjects) {
  hwloc_topology_t topology = GetHWLocTopology();
  ASSERT_NE(topology, nullptr);
  EXPECT_GE(hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_NUMANODE), 1);
  EXPECT_EQ(hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_L1CACHE), 0);
  EXPECT_EQ(hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_L2CACHE), 0);
  EXPECT_EQ(hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_L3CACHE), 0);
  EXPECT_EQ(hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_PCI_DEVICE), 0);
  EXPECT_EQ(hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_OS_DEVICE), 0);
}

}  // namespace
}  // namespace port
}  // namespace tsl
