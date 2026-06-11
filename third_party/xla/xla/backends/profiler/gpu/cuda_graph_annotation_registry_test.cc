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

#include <gtest/gtest.h>
#include "third_party/gpus/cuda/include/cuda.h"

#if CUDA_VERSION >= 13010

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "xla/backends/profiler/gpu/cuda_driver_graph_interface.h"
#include "xla/backends/profiler/gpu/cuda_graph_annotation_registry.h"
#include "xla/backends/profiler/gpu/cuda_graph_telemetry_listener_impl.h"
#include "xla/stream_executor/gpu/scoped_command_buffer_annotation.h"

namespace xla {
namespace profiler {
namespace {

struct DummyGraph {};
struct DummyNode {};
struct DummyExec {};

CUgraph ToCUgraph(DummyGraph* g) { return reinterpret_cast<CUgraph>(g); }
CUgraphNode ToCUgraphNode(DummyNode* n) {
  return reinterpret_cast<CUgraphNode>(n);
}
CUgraphExec ToCUgraphExec(DummyExec* e) {
  return reinterpret_cast<CUgraphExec>(e);
}

class MockCudaDriver : public CudaDriverGraphInterface {
 public:
  CUresult GetGraphId(CUgraph graph, unsigned int* id) const override {
    auto it = graph_ids_.find(graph);
    if (it == graph_ids_.end()) return CUDA_ERROR_INVALID_VALUE;
    *id = it->second;
    return CUDA_SUCCESS;
  }

  CUresult GetNodeToolsId(CUgraphNode node, uint64_t* id) const override {
    auto it = node_tools_ids_.find(node);
    if (it == node_tools_ids_.end()) return CUDA_ERROR_INVALID_VALUE;
    *id = it->second;
    return CUDA_SUCCESS;
  }

  CUresult GetExecId(CUgraphExec exec, unsigned int* id) const override {
    auto it = exec_ids_.find(exec);
    if (it == exec_ids_.end()) return CUDA_ERROR_INVALID_VALUE;
    *id = it->second;
    return CUDA_SUCCESS;
  }

  void SetGraphId(CUgraph graph, unsigned int id) { graph_ids_[graph] = id; }
  void SetNodeToolsId(CUgraphNode node, uint64_t id) {
    node_tools_ids_[node] = id;
  }
  void SetExecId(CUgraphExec exec, unsigned int id) { exec_ids_[exec] = id; }

  void Clear() {
    graph_ids_.clear();
    node_tools_ids_.clear();
    exec_ids_.clear();
  }

 private:
  absl::flat_hash_map<CUgraph, unsigned int> graph_ids_;
  absl::flat_hash_map<CUgraphNode, uint64_t> node_tools_ids_;
  absl::flat_hash_map<CUgraphExec, unsigned int> exec_ids_;
};

}  // namespace

class ScopedCudaDriverOverrideForTesting {
 public:
  explicit ScopedCudaDriverOverrideForTesting(
      const CudaDriverGraphInterface* driver)
      : old_driver_(CudaGraphAnnotationRegistry::ExchangeCudaDriver(driver)) {}
  ~ScopedCudaDriverOverrideForTesting() {
    CudaGraphAnnotationRegistry::ExchangeCudaDriver(old_driver_);
  }

 private:
  const CudaDriverGraphInterface* old_driver_;
};

class CudaGraphAnnotationRegistryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    CudaGraphAnnotationRegistry::ResetForTesting();
    driver_override_ =
        std::make_unique<ScopedCudaDriverOverrideForTesting>(&mock_driver_);
    mock_driver_.Clear();
  }
  void TearDown() override {
    driver_override_.reset();
    CudaGraphAnnotationRegistry::ResetForTesting();
  }

  MockCudaDriver mock_driver_;
  std::unique_ptr<ScopedCudaDriverOverrideForTesting> driver_override_;
};

TEST_F(CudaGraphAnnotationRegistryTest, SingleThreadedRegisterAndLookup) {
  DummyGraph g1;
  DummyNode n1, n2;

  // Setup mock IDs.
  // Template Graph 1.
  mock_driver_.SetGraphId(ToCUgraph(&g1), 1);
  // Node Tools IDs: (graph_id << 32) | relative_index
  mock_driver_.SetNodeToolsId(ToCUgraphNode(&n1), (1ULL << 32) | 0);
  mock_driver_.SetNodeToolsId(ToCUgraphNode(&n2), (1ULL << 32) | 1);

  CudaGraphAnnotationRegistry::RegisterNodeAnnotation(
      ToCUgraph(&g1), ToCUgraphNode(&n1), "op1");
  CudaGraphAnnotationRegistry::RegisterNodeAnnotation(
      ToCUgraph(&g1), ToCUgraphNode(&n2), "op2");

  // Direct lookup using template IDs (no instantiation mapping).
  EXPECT_EQ(CudaGraphAnnotationRegistry::LookupAnnotation(1, (1ULL << 32) | 0),
            "op1");
  EXPECT_EQ(CudaGraphAnnotationRegistry::LookupAnnotation(1, (1ULL << 32) | 1),
            "op2");
  // Non-existent lookup.
  EXPECT_EQ(CudaGraphAnnotationRegistry::LookupAnnotation(1, (1ULL << 32) | 2),
            "");
}

TEST_F(CudaGraphAnnotationRegistryTest, UnregisterGraph) {
  DummyGraph g1;
  DummyNode n1;

  mock_driver_.SetGraphId(ToCUgraph(&g1), 1);
  mock_driver_.SetNodeToolsId(ToCUgraphNode(&n1), (1ULL << 32) | 0);

  CudaGraphAnnotationRegistry::RegisterNodeAnnotation(
      ToCUgraph(&g1), ToCUgraphNode(&n1), "op1");

  // Verify it exists first.
  EXPECT_EQ(CudaGraphAnnotationRegistry::LookupAnnotation(1, (1ULL << 32) | 0),
            "op1");

  CudaGraphAnnotationRegistry::UnregisterGraphAnnotations(ToCUgraph(&g1));

  // Verify it is gone.
  EXPECT_EQ(CudaGraphAnnotationRegistry::LookupAnnotation(1, (1ULL << 32) | 0),
            "");
}

TEST_F(CudaGraphAnnotationRegistryTest, RegisterGraphExecAndLookupMerged) {
  // Simulate a parent graph with a nested child graph.
  // Parent Graph (Pg): 3 nodes (Pn1, Pn2 [child node], Pn3)
  // Child Graph (Cg): 2 nodes (Cn1, Cn2)
  DummyGraph Pg, Cg;
  DummyNode Pn1, Pn2, Pn3;
  DummyNode Cn1, Cn2;
  DummyExec Eg;  // Instantiated Exec Graph

  // 1. Setup mock IDs.
  mock_driver_.SetGraphId(ToCUgraph(&Pg), 1);
  mock_driver_.SetGraphId(ToCUgraph(&Cg), 2);
  mock_driver_.SetExecId(ToCUgraphExec(&Eg), 10);  // Instantiated Graph ID = 10
  // Template Tools IDs.
  mock_driver_.SetNodeToolsId(ToCUgraphNode(&Pn1), (1ULL << 32) | 0);
  mock_driver_.SetNodeToolsId(ToCUgraphNode(&Pn2),
                              (1ULL << 32) | 1);  // Child node at index 1
  mock_driver_.SetNodeToolsId(ToCUgraphNode(&Pn3), (1ULL << 32) | 2);
  mock_driver_.SetNodeToolsId(ToCUgraphNode(&Cn1), (2ULL << 32) | 0);
  mock_driver_.SetNodeToolsId(ToCUgraphNode(&Cn2), (2ULL << 32) | 1);

  // 2. Register annotations on template graphs.
  CudaGraphAnnotationRegistry::RegisterNodeAnnotation(
      ToCUgraph(&Pg), ToCUgraphNode(&Pn1), "parent_op1");
  CudaGraphAnnotationRegistry::RegisterNodeAnnotation(
      ToCUgraph(&Pg), ToCUgraphNode(&Pn3), "parent_op3");
  CudaGraphAnnotationRegistry::RegisterNodeAnnotation(
      ToCUgraph(&Cg), ToCUgraphNode(&Cn1), "child_op1");
  CudaGraphAnnotationRegistry::RegisterNodeAnnotation(
      ToCUgraph(&Cg), ToCUgraphNode(&Cn2), "child_op2");

  // 3. Register structural relations.
  CudaGraphAnnotationRegistry::RegisterGraphSize(ToCUgraph(&Pg), 3, "ModuleP");
  CudaGraphAnnotationRegistry::RegisterGraphSize(ToCUgraph(&Cg), 2, "ModuleC");
  CudaGraphAnnotationRegistry::RegisterChildGraph(
      ToCUgraph(&Pg), ToCUgraph(&Cg), ToCUgraphNode(&Pn2),
      /*is_conditional=*/true);

  // 4. Instantiate the graph.
  // This triggers Pre-computation of merged mappings.
  CudaGraphAnnotationRegistry::RegisterGraphExec(ToCUgraphExec(&Eg),
                                                 ToCUgraph(&Pg));

  // 5. Verify lookups using instantiated IDs.
  // Instantiated Graph ID = 10.
  // Merged nodes should be mapped to their templates:
  // Index 0 -> Pg index 0 (Pn1) -> "ModuleP::parent_op1"
  // Index 1 -> Pg index 1 (Pn2 - child node, no kernel annotation) -> ""
  // Index 2 -> Pg index 2 (Pn3) -> "ModuleP::parent_op3" (NOT shifted)
  // Index 3 -> Cg index 0 (Cn1) -> "ModuleC::child_op1" (appended)
  // Index 4 -> Cg index 1 (Cn2) -> "ModuleC::child_op2" (appended)

  EXPECT_EQ(
      CudaGraphAnnotationRegistry::LookupAnnotation(10, (10ULL << 32) | 0),
      "ModuleP::parent_op1");
  EXPECT_EQ(
      CudaGraphAnnotationRegistry::LookupAnnotation(10, (10ULL << 32) | 1),
      "Thunk:#hlo_op=driver_conditional_helper#");
  EXPECT_EQ(
      CudaGraphAnnotationRegistry::LookupAnnotation(10, (10ULL << 32) | 2),
      "ModuleP::parent_op3");
  EXPECT_EQ(
      CudaGraphAnnotationRegistry::LookupAnnotation(10, (10ULL << 32) | 3),
      "ModuleC::child_op1");
  EXPECT_EQ(
      CudaGraphAnnotationRegistry::LookupAnnotation(10, (10ULL << 32) | 4),
      "ModuleC::child_op2");
}

TEST_F(CudaGraphAnnotationRegistryTest,
       RegisterGraphExecAndLookupInlineMerged) {
  // Simulate a parent graph with an inline child graph.
  // Parent Graph (Pg): 3 nodes (Pn1, Pn2 [child node], Pn3)
  // Child Graph (Cg): 2 nodes (Cn1, Cn2)
  DummyGraph Pg, Cg;
  DummyNode Pn1, Pn2, Pn3;
  DummyNode Cn1, Cn2;
  DummyExec Eg;  // Instantiated Exec Graph

  // 1. Setup mock IDs.
  mock_driver_.SetGraphId(ToCUgraph(&Pg), 1);
  mock_driver_.SetGraphId(ToCUgraph(&Cg), 2);
  mock_driver_.SetExecId(ToCUgraphExec(&Eg), 10);  // Instantiated Graph ID = 10

  // Template Tools IDs.
  mock_driver_.SetNodeToolsId(ToCUgraphNode(&Pn1), (1ULL << 32) | 0);
  mock_driver_.SetNodeToolsId(ToCUgraphNode(&Pn2),
                              (1ULL << 32) | 1);  // Child node at index 1
  mock_driver_.SetNodeToolsId(ToCUgraphNode(&Pn3), (1ULL << 32) | 2);
  mock_driver_.SetNodeToolsId(ToCUgraphNode(&Cn1), (2ULL << 32) | 0);
  mock_driver_.SetNodeToolsId(ToCUgraphNode(&Cn2), (2ULL << 32) | 1);

  // 2. Register annotations on template graphs.
  CudaGraphAnnotationRegistry::RegisterNodeAnnotation(
      ToCUgraph(&Pg), ToCUgraphNode(&Pn1), "parent_op1");
  CudaGraphAnnotationRegistry::RegisterNodeAnnotation(
      ToCUgraph(&Pg), ToCUgraphNode(&Pn3), "parent_op3");
  CudaGraphAnnotationRegistry::RegisterNodeAnnotation(
      ToCUgraph(&Cg), ToCUgraphNode(&Cn1), "child_op1");
  CudaGraphAnnotationRegistry::RegisterNodeAnnotation(
      ToCUgraph(&Cg), ToCUgraphNode(&Cn2), "child_op2");

  // 3. Register structural relations.
  CudaGraphAnnotationRegistry::RegisterGraphSize(ToCUgraph(&Pg), 3, "ModuleP");
  CudaGraphAnnotationRegistry::RegisterGraphSize(ToCUgraph(&Cg), 2, "ModuleC");
  CudaGraphAnnotationRegistry::RegisterChildGraph(
      ToCUgraph(&Pg), ToCUgraph(&Cg), ToCUgraphNode(&Pn2),
      /*is_conditional=*/false);

  // 4. Instantiate the graph.
  CudaGraphAnnotationRegistry::RegisterGraphExec(ToCUgraphExec(&Eg),
                                                 ToCUgraph(&Pg));

  // 5. Verify lookups using instantiated IDs.
  // Instantiated Graph ID = 10.
  // Merged nodes should be mapped to their templates:
  // Index 0 -> Pg index 0 (Pn1) -> "ModuleP::parent_op1"
  // Index 1 -> Cg index 0 (Cn1) -> "ModuleC::child_op1"
  // Index 2 -> Cg index 1 (Cn2) -> "ModuleC::child_op2"
  // Index 3 -> Pg index 2 (Pn3) -> "ModuleP::parent_op3"

  EXPECT_EQ(
      CudaGraphAnnotationRegistry::LookupAnnotation(10, (10ULL << 32) | 0),
      "ModuleP::parent_op1");
  EXPECT_EQ(
      CudaGraphAnnotationRegistry::LookupAnnotation(10, (10ULL << 32) | 1),
      "ModuleC::child_op1");
  EXPECT_EQ(
      CudaGraphAnnotationRegistry::LookupAnnotation(10, (10ULL << 32) | 2),
      "ModuleC::child_op2");
  EXPECT_EQ(
      CudaGraphAnnotationRegistry::LookupAnnotation(10, (10ULL << 32) | 3),
      "ModuleP::parent_op3");
}

TEST_F(CudaGraphAnnotationRegistryTest,
       RegisterGraphExecAndLookupDeepNestingInline) {
  // Simulate Parent -> Child -> Grandchild (all inline)
  DummyGraph Pg, Cg, Gg;
  DummyNode Pn1, Pn2, Pn3;
  DummyNode Cn1, Cn2, Cn3;
  DummyNode Gn1, Gn2;
  DummyExec Eg;

  mock_driver_.SetGraphId(ToCUgraph(&Pg), 1);
  mock_driver_.SetGraphId(ToCUgraph(&Cg), 2);
  mock_driver_.SetGraphId(ToCUgraph(&Gg), 3);
  mock_driver_.SetExecId(ToCUgraphExec(&Eg), 10);

  mock_driver_.SetNodeToolsId(ToCUgraphNode(&Pn1), (1ULL << 32) | 0);
  mock_driver_.SetNodeToolsId(ToCUgraphNode(&Pn2),
                              (1ULL << 32) | 1);  // Child node at index 1
  mock_driver_.SetNodeToolsId(ToCUgraphNode(&Pn3), (1ULL << 32) | 2);

  mock_driver_.SetNodeToolsId(ToCUgraphNode(&Cn1), (2ULL << 32) | 0);
  mock_driver_.SetNodeToolsId(ToCUgraphNode(&Cn2),
                              (2ULL << 32) | 1);  // Grandchild node at index 1
  mock_driver_.SetNodeToolsId(ToCUgraphNode(&Cn3), (2ULL << 32) | 2);

  mock_driver_.SetNodeToolsId(ToCUgraphNode(&Gn1), (3ULL << 32) | 0);
  mock_driver_.SetNodeToolsId(ToCUgraphNode(&Gn2), (3ULL << 32) | 1);

  CudaGraphAnnotationRegistry::RegisterNodeAnnotation(
      ToCUgraph(&Pg), ToCUgraphNode(&Pn1), "parent_op1");
  CudaGraphAnnotationRegistry::RegisterNodeAnnotation(
      ToCUgraph(&Pg), ToCUgraphNode(&Pn3), "parent_op3");

  CudaGraphAnnotationRegistry::RegisterNodeAnnotation(
      ToCUgraph(&Cg), ToCUgraphNode(&Cn1), "child_op1");
  CudaGraphAnnotationRegistry::RegisterNodeAnnotation(
      ToCUgraph(&Cg), ToCUgraphNode(&Cn3), "child_op3");

  CudaGraphAnnotationRegistry::RegisterNodeAnnotation(
      ToCUgraph(&Gg), ToCUgraphNode(&Gn1), "grandchild_op1");
  CudaGraphAnnotationRegistry::RegisterNodeAnnotation(
      ToCUgraph(&Gg), ToCUgraphNode(&Gn2), "grandchild_op2");

  CudaGraphAnnotationRegistry::RegisterGraphSize(ToCUgraph(&Pg), 3, "ModuleP");
  CudaGraphAnnotationRegistry::RegisterGraphSize(ToCUgraph(&Cg), 3, "ModuleC");
  CudaGraphAnnotationRegistry::RegisterGraphSize(ToCUgraph(&Gg), 2, "ModuleG");

  CudaGraphAnnotationRegistry::RegisterChildGraph(
      ToCUgraph(&Pg), ToCUgraph(&Cg), ToCUgraphNode(&Pn2),
      /*is_conditional=*/false);
  CudaGraphAnnotationRegistry::RegisterChildGraph(
      ToCUgraph(&Cg), ToCUgraph(&Gg), ToCUgraphNode(&Cn2),
      /*is_conditional=*/false);

  CudaGraphAnnotationRegistry::RegisterGraphExec(ToCUgraphExec(&Eg),
                                                 ToCUgraph(&Pg));

  // Expected Merged indices:
  // 0 -> Pn1
  // 1 -> Cn1
  // 2 -> Gn1
  // 3 -> Gn2
  // 4 -> Cn3
  // 5 -> Pn3
  EXPECT_EQ(
      CudaGraphAnnotationRegistry::LookupAnnotation(10, (10ULL << 32) | 0),
      "ModuleP::parent_op1");
  EXPECT_EQ(
      CudaGraphAnnotationRegistry::LookupAnnotation(10, (10ULL << 32) | 2),
      "ModuleG::grandchild_op1");
  EXPECT_EQ(
      CudaGraphAnnotationRegistry::LookupAnnotation(10, (10ULL << 32) | 3),
      "ModuleG::grandchild_op2");
  EXPECT_EQ(
      CudaGraphAnnotationRegistry::LookupAnnotation(10, (10ULL << 32) | 5),
      "ModuleP::parent_op3");
}

TEST_F(CudaGraphAnnotationRegistryTest,
       TelemetryListenerInterceptsAnnotations) {
  CudaGraphTelemetryListenerImpl listener;
  DummyGraph parent_graph;
  DummyNode parent_node;
  DummyExec exec;

  mock_driver_.SetGraphId(ToCUgraph(&parent_graph), 1);
  mock_driver_.SetNodeToolsId(ToCUgraphNode(&parent_node), (1ULL << 32) | 0);
  mock_driver_.SetExecId(ToCUgraphExec(&exec), 100);

  // 1. Trigger size registration within thread-local Scoped Annotation.
  {
    stream_executor::ScopedCommandBufferAnnotation scope("ModuleP");
    listener.OnRegisterGraphSize(ToCUgraph(&parent_graph), 1);
  }

  // 2. Register node annotations.
  listener.OnRegisterNodeAnnotation(
      ToCUgraph(&parent_graph),
      reinterpret_cast<stream_executor::gpu::GpuCommandBuffer::GraphNodeHandle>(
          &parent_node),
      "parent_op");

  // 3. Register executable.
  listener.OnRegisterGraphExec(ToCUgraphExec(&exec), ToCUgraph(&parent_graph));

  // 4. Verify lookup prepends the HLO Module annotation name correctly.
  EXPECT_EQ(
      CudaGraphAnnotationRegistry::LookupAnnotation(100, (1ULL << 32) | 0),
      "ModuleP::parent_op");
}

}  // namespace profiler
}  // namespace xla

#else  // CUDA_VERSION < 13010

TEST(CudaGraphAnnotationRegistryTest, SkippedOnOlderCuda) {
  GTEST_SKIP() << "CUDA Graphs telemetry requires CUDA 13.1+";
}

#endif  // CUDA_VERSION >= 13010
