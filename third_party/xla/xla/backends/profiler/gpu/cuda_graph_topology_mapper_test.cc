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

#include "xla/backends/profiler/gpu/cuda_graph_topology_mapper.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"

namespace xla {
namespace profiler {
namespace {

TEST(CudaGraphTopologyMapperTest, CalculateMergedSizeSimpleInline) {
  absl::flat_hash_map<uint32_t, size_t> base_sizes = {{1, 3}, {2, 2}};
  absl::flat_hash_map<uint32_t,
                      std::vector<CudaGraphTopologyMapper::ChildGraphEntry>>
      child_graphs;
  child_graphs[1].push_back({2, 1, /*is_conditional=*/false});

  absl::flat_hash_map<uint32_t, size_t> merged_sizes;
  size_t size = CudaGraphTopologyMapper::CalculateMergedSize(
      1, base_sizes, child_graphs, &merged_sizes);

  // 3 + 2 - 1 = 4
  EXPECT_EQ(size, 4);
}

TEST(CudaGraphTopologyMapperTest, ResolveMergedNodeInline) {
  absl::flat_hash_map<uint32_t, size_t> base_sizes = {{1, 3}, {2, 2}};
  absl::flat_hash_map<uint32_t,
                      std::vector<CudaGraphTopologyMapper::ChildGraphEntry>>
      child_graphs;
  child_graphs[1].push_back({2, 1, /*is_conditional=*/false});

  absl::flat_hash_map<uint32_t, size_t> merged_sizes;

  // Expected Merged indices:
  // 0 -> Parent node 0
  // 1 -> Child node 0
  // 2 -> Child node 1
  // 3 -> Parent node 2
  auto res0 = CudaGraphTopologyMapper::ResolveMergedNode(
      1, 0, base_sizes, child_graphs, &merged_sizes);
  EXPECT_EQ(res0.first, 1);
  EXPECT_EQ(res0.second, 0);

  auto res1 = CudaGraphTopologyMapper::ResolveMergedNode(
      1, 1, base_sizes, child_graphs, &merged_sizes);
  EXPECT_EQ(res1.first, 2);
  EXPECT_EQ(res1.second, 0);

  auto res2 = CudaGraphTopologyMapper::ResolveMergedNode(
      1, 2, base_sizes, child_graphs, &merged_sizes);
  EXPECT_EQ(res2.first, 2);
  EXPECT_EQ(res2.second, 1);

  auto res3 = CudaGraphTopologyMapper::ResolveMergedNode(
      1, 3, base_sizes, child_graphs, &merged_sizes);
  EXPECT_EQ(res3.first, 1);
  EXPECT_EQ(res3.second, 2);
}

TEST(CudaGraphTopologyMapperTest, ResolveMergedNodeDeepNestingInline) {
  // Parent (1) -> Child (2) -> Grandchild (3)
  absl::flat_hash_map<uint32_t, size_t> base_sizes = {{1, 3}, {2, 3}, {3, 2}};
  absl::flat_hash_map<uint32_t,
                      std::vector<CudaGraphTopologyMapper::ChildGraphEntry>>
      child_graphs;
  child_graphs[1].push_back({2, 1, /*is_conditional=*/false});
  child_graphs[2].push_back({3, 1, /*is_conditional=*/false});

  absl::flat_hash_map<uint32_t, size_t> merged_sizes;

  // Expected indices:
  // 0 -> Parent 0
  // 1 -> Child 0
  // 2 -> Grandchild 0
  // 3 -> Grandchild 1
  // 4 -> Child 2
  // 5 -> Parent 2
  auto res2 = CudaGraphTopologyMapper::ResolveMergedNode(
      1, 2, base_sizes, child_graphs, &merged_sizes);
  EXPECT_EQ(res2.first, 3);
  EXPECT_EQ(res2.second, 0);

  auto res5 = CudaGraphTopologyMapper::ResolveMergedNode(
      1, 5, base_sizes, child_graphs, &merged_sizes);
  EXPECT_EQ(res5.first, 1);
  EXPECT_EQ(res5.second, 2);
}

TEST(CudaGraphTopologyMapperTest, CalculateMergedSizeConditional) {
  absl::flat_hash_map<uint32_t, size_t> base_sizes = {{1, 3}, {2, 2}};
  absl::flat_hash_map<uint32_t,
                      std::vector<CudaGraphTopologyMapper::ChildGraphEntry>>
      child_graphs;
  child_graphs[1].push_back({2, 1, /*is_conditional=*/true});

  absl::flat_hash_map<uint32_t, size_t> merged_sizes;
  size_t size = CudaGraphTopologyMapper::CalculateMergedSize(
      1, base_sizes, child_graphs, &merged_sizes);

  // 3 + 2 = 5 (Conditional child graph nodes are appended, size is not shifted
  // by -1)
  EXPECT_EQ(size, 5);
}

TEST(CudaGraphTopologyMapperTest, ResolveMergedNodeConditional) {
  absl::flat_hash_map<uint32_t, size_t> base_sizes = {{1, 3}, {2, 2}};
  absl::flat_hash_map<uint32_t,
                      std::vector<CudaGraphTopologyMapper::ChildGraphEntry>>
      child_graphs;
  child_graphs[1].push_back({2, 1, /*is_conditional=*/true});

  absl::flat_hash_map<uint32_t, size_t> merged_sizes;

  // Expected Merged indices for Conditional Nested Graph (Hardware Ground
  // Truth): 0 -> Parent node 0 1 -> Parent node 1 2 -> Parent node 2 3 -> Child
  // node 0 4 -> Child node 1
  auto res0 = CudaGraphTopologyMapper::ResolveMergedNode(
      1, 0, base_sizes, child_graphs, &merged_sizes);
  EXPECT_EQ(res0.first, 1);
  EXPECT_EQ(res0.second, 0);

  auto res1 = CudaGraphTopologyMapper::ResolveMergedNode(
      1, 1, base_sizes, child_graphs, &merged_sizes);
  EXPECT_EQ(res1.first, 1);
  EXPECT_EQ(res1.second, 1);

  auto res2 = CudaGraphTopologyMapper::ResolveMergedNode(
      1, 2, base_sizes, child_graphs, &merged_sizes);
  EXPECT_EQ(res2.first, 1);
  EXPECT_EQ(res2.second, 2);

  auto res3 = CudaGraphTopologyMapper::ResolveMergedNode(
      1, 3, base_sizes, child_graphs, &merged_sizes);
  EXPECT_EQ(res3.first, 2);
  EXPECT_EQ(res3.second, 0);

  auto res4 = CudaGraphTopologyMapper::ResolveMergedNode(
      1, 4, base_sizes, child_graphs, &merged_sizes);
  EXPECT_EQ(res4.first, 2);
  EXPECT_EQ(res4.second, 1);
}

TEST(CudaGraphTopologyMapperTest, ResolveMergedNodeMixedInlineAndConditional) {
  absl::flat_hash_map<uint32_t, size_t> base_sizes = {{1, 3}, {2, 2}, {3, 2}};
  absl::flat_hash_map<uint32_t,
                      std::vector<CudaGraphTopologyMapper::ChildGraphEntry>>
      child_graphs;
  child_graphs[1].push_back({2, 1, /*is_conditional=*/false});
  child_graphs[1].push_back({3, 2, /*is_conditional=*/true});

  absl::flat_hash_map<uint32_t, size_t> merged_sizes;

  auto res = CudaGraphTopologyMapper::ResolveMergedNode(
      1, 4, base_sizes, child_graphs, &merged_sizes);
  EXPECT_EQ(res.first, 3);
  EXPECT_EQ(res.second, 0);
}

TEST(CudaGraphTopologyMapperTest, ResolveMergedNodeMultipleConditional) {
  absl::flat_hash_map<uint32_t, size_t> base_sizes = {{1, 3}, {2, 2}, {3, 2}};
  absl::flat_hash_map<uint32_t,
                      std::vector<CudaGraphTopologyMapper::ChildGraphEntry>>
      child_graphs;
  child_graphs[1].push_back({2, 1, /*is_conditional=*/true});
  child_graphs[1].push_back({3, 2, /*is_conditional=*/true});

  absl::flat_hash_map<uint32_t, size_t> merged_sizes;

  auto res = CudaGraphTopologyMapper::ResolveMergedNode(
      1, 5, base_sizes, child_graphs, &merged_sizes);
  EXPECT_EQ(res.first, 3);
  EXPECT_EQ(res.second, 0);
}

TEST(CudaGraphTopologyMapperTest, ResolveMergedNodeOutOfBounds) {
  absl::flat_hash_map<uint32_t, size_t> base_sizes = {{1, 3}, {2, 2}};
  absl::flat_hash_map<uint32_t,
                      std::vector<CudaGraphTopologyMapper::ChildGraphEntry>>
      child_graphs;
  child_graphs[1].push_back({2, 1, /*is_conditional=*/false});

  absl::flat_hash_map<uint32_t, size_t> merged_sizes;

  auto res = CudaGraphTopologyMapper::ResolveMergedNode(
      1, 4, base_sizes, child_graphs, &merged_sizes);
  EXPECT_EQ(res.first, 1);
  EXPECT_EQ(res.second, 4);
}

}  // namespace
}  // namespace profiler
}  // namespace xla
