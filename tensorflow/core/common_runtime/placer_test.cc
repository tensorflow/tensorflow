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

#include "tensorflow/core/common_runtime/placer.h"

#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/graph/graph_def_builder_util.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

using ::tensorflow::test::function::GDef;
using ::tensorflow::test::function::NDef;
using FDH = ::tensorflow::FunctionDefHelper;

constexpr char kCPU[] = "/device:FakeCPU:0";
constexpr char kGPU[] = "/device:FakeGPU:0";

constexpr char kFullCPU[] = "/job:a/replica:0/task:0/device:FakeCPU:0";
constexpr char kFullGPU[] = "/job:a/replica:0/task:0/device:FakeGPU:0";

namespace {

////////////////////////////////////////////////////////////////////////////////
//
// Op, kernel, and device registrations to set up the environment.
//
// The Placer uses information about the op (input types),
// kernel (device constraints), and available devices to make
// placement decisions. To avoid depending on the full runtime, we
// define dummy implementations of these, and register them with the
// runtime.
//
////////////////////////////////////////////////////////////////////////////////

// A dummy OpKernel that is used to register ops on different devices.
class DummyOp : public OpKernel {
 public:
  explicit DummyOp(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {}
};

// A fake device that has specific device attributes, used to simulate
// the presence of a CPU or a GPU (without depending on that part of
// the runtime.
class FakeDevice : public Device {
 private:
  explicit FakeDevice(const DeviceAttributes& device_attributes)
      : Device(nullptr, device_attributes) {}

 public:
  Status Sync() override { return errors::Unimplemented("FakeDevice::Sync()"); }

  Allocator* GetAllocator(AllocatorAttributes attr) override { return nullptr; }

  static std::unique_ptr<Device> MakeCPU(const string& name) {
    DeviceAttributes device_attributes;
    device_attributes.set_name(name);
    device_attributes.set_device_type(DeviceType("FakeCPU").type());
    return std::unique_ptr<Device>(new FakeDevice(device_attributes));
  }

  static std::unique_ptr<Device> MakeGPU(const string& name) {
    DeviceAttributes device_attributes;
    device_attributes.set_name(name);
    device_attributes.set_device_type(DeviceType("FakeGPU").type());
    return std::unique_ptr<Device>(new FakeDevice(device_attributes));
  }
};

class DummyFactory : public DeviceFactory {
 public:
  Status ListPhysicalDevices(std::vector<string>* devices) override {
    return Status::OK();
  }
  Status CreateDevices(const SessionOptions& options, const string& name_prefix,
                       std::vector<std::unique_ptr<Device>>* devices) override {
    return Status::OK();
  }
};

// Device order now depends on the registration of devices, not a fixed
// value in device_set.cc.  To avoid the need to link in the real CPU and GPU
// devices into this test, we create fake devices and registrations that
// can stand-in for the real devices for the purposes of testing placement
// and ordering.
REGISTER_LOCAL_DEVICE_FACTORY("FakeCPU", DummyFactory);
REGISTER_LOCAL_DEVICE_FACTORY("FakeGPU", DummyFactory, 51);

// Register the following ops so they can be added to a Graph, and
// kernels so that they can be placed on particular device types.
REGISTER_OP("TestVariable").Output("o: Ref(float)");
REGISTER_KERNEL_BUILDER(Name("TestVariable").Device("FakeCPU"), DummyOp);
REGISTER_KERNEL_BUILDER(Name("TestVariable").Device("FakeGPU"), DummyOp);

REGISTER_OP("VariableCPU").Output("o: Ref(float)");
REGISTER_KERNEL_BUILDER(Name("VariableCPU").Device("FakeCPU"), DummyOp);

REGISTER_OP("VariableGPU").Output("o: Ref(float)");
REGISTER_KERNEL_BUILDER(Name("VariableGPU").Device("FakeGPU"), DummyOp);

REGISTER_OP("VariableNoKernels").Output("o: Ref(float)");

REGISTER_OP("TestAdd").Input("a: float").Input("b: float").Output("o: float");
REGISTER_KERNEL_BUILDER(Name("TestAdd").Device("FakeCPU"), DummyOp);
REGISTER_KERNEL_BUILDER(Name("TestAdd").Device("FakeGPU"), DummyOp);

REGISTER_OP("TestRelu").Input("i: float").Output("o: float");
REGISTER_KERNEL_BUILDER(Name("TestRelu").Device("FakeCPU"), DummyOp);
REGISTER_KERNEL_BUILDER(Name("TestRelu").Device("FakeGPU"), DummyOp);

REGISTER_OP("ReluCPU").Input("i: float").Output("o: float");
REGISTER_KERNEL_BUILDER(Name("ReluCPU").Device("FakeCPU"), DummyOp);

REGISTER_OP("ReluGPU").Input("i: float").Output("o: float");
REGISTER_KERNEL_BUILDER(Name("ReluGPU").Device("FakeGPU"), DummyOp);

REGISTER_OP("TestAssign").Input("i: Ref(float)").Input("v: float");
REGISTER_KERNEL_BUILDER(Name("TestAssign").Device("FakeCPU"), DummyOp);
REGISTER_KERNEL_BUILDER(Name("TestAssign").Device("FakeGPU"), DummyOp);

REGISTER_OP("AssignCPU").Input("i: Ref(float)").Input("v: float");
REGISTER_KERNEL_BUILDER(Name("AssignCPU").Device("FakeCPU"), DummyOp);

REGISTER_OP("AssignGPU").Input("i: Ref(float)").Input("v: float");
REGISTER_KERNEL_BUILDER(Name("AssignGPU").Device("FakeGPU"), DummyOp);

REGISTER_OP("TestInput").Output("a: float").Output("b: float");
REGISTER_KERNEL_BUILDER(Name("TestInput").Device("FakeCPU"), DummyOp);

// Op producing an output that can be placed on CPU or GPU.
REGISTER_OP("TestCPUGPUOutput").Output("a: float");
REGISTER_KERNEL_BUILDER(Name("TestCPUGPUOutput").Device("FakeCPU"), DummyOp);
REGISTER_KERNEL_BUILDER(Name("TestCPUGPUOutput").Device("FakeGPU"), DummyOp);

REGISTER_OP("TestGPUOutput").Output("a: float");
REGISTER_KERNEL_BUILDER(Name("TestGPUOutput").Device("FakeGPU"), DummyOp);

REGISTER_OP("TestDevice").Output("a: float").Output("b: float");
REGISTER_KERNEL_BUILDER(Name("TestDevice").Device("FakeGPU"), DummyOp);

REGISTER_OP("TestDeviceEnforce").Input("a: Ref(float)").Output("b: float");
REGISTER_KERNEL_BUILDER(Name("TestDeviceEnforce").Device("FakeCPU"), DummyOp);
REGISTER_KERNEL_BUILDER(Name("TestDeviceEnforce").Device("FakeGPU"), DummyOp);

REGISTER_KERNEL_BUILDER(Name("Shape").Device("FakeCPU"), DummyOp);
REGISTER_KERNEL_BUILDER(Name("Shape").Device("FakeGPU"), DummyOp);

// Op that has kernels with device priorities specified.
REGISTER_OP("TestDatasetOp").Input("a: float").Output("b: float");
REGISTER_KERNEL_BUILDER(Name("TestDatasetOp").Device("FakeCPU").Priority(2),
                        DummyOp);
REGISTER_KERNEL_BUILDER(Name("TestDatasetOp").Device("FakeGPU").Priority(1),
                        DummyOp);

////////////////////////////////////////////////////////////////////////////////
//
// A PlacerTest method has three phases:
//
// 1. Build a TensorFlow graph, with no (or partial) device assignments.
// 2. Attempt to compute a placement using the Placer.
// 3. EITHER: test that the constraints implied by the graph are respected;
//    or that an appropriate error was reported.
//
////////////////////////////////////////////////////////////////////////////////
class PlacerTest : public ::testing::Test {
 protected:
  PlacerTest() {
    // Build a set of 10 GPU and 10 CPU devices.
    // NOTE: this->local_devices_ owns the device objects;
    // this->devices_ contains borrowed pointers to the device
    // objects.
    for (int i = 0; i < 10; ++i) {
      local_devices_.emplace_back(FakeDevice::MakeCPU(
          strings::StrCat("/job:a/replica:0/task:0/device:FakeCPU:", i)));
      devices_.AddDevice(local_devices_.back().get());
      // Insert the GPUs in reverse order.
      local_devices_.emplace_back(FakeDevice::MakeGPU(
          strings::StrCat("/job:a/replica:0/task:0/device:FakeGPU:", 9 - i)));
      devices_.AddDevice(local_devices_.back().get());
    }
  }

  // Builds the given graph, and (if successful) indexes the node
  // names for use in placement, and later lookup.
  Status BuildGraph(const GraphDefBuilder& builder, Graph* out_graph) {
    TF_RETURN_IF_ERROR(GraphDefBuilderToGraph(builder, out_graph));
    nodes_by_name_.clear();
    for (Node* node : out_graph->nodes()) {
      nodes_by_name_[node->name()] = node->id();
    }
    return Status::OK();
  }

  Status BuildGraph(const GraphDef& graph_def, Graph* out_graph) {
    GraphConstructorOptions opts;
    TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(opts, graph_def, out_graph));
    nodes_by_name_.clear();
    for (Node* node : out_graph->nodes()) {
      nodes_by_name_[node->name()] = node->id();
    }
    return Status::OK();
  }

  // Invokes the Placer on "graph". If no DeviceSet is specified, the
  // placement will use the default DeviceSet (of 10 CPU and 10 GPU devices).
  //
  // REQUIRES: "*graph" was produced by the most recent call to BuildGraph.
  Status Place(Graph* graph, DeviceSet* devices, bool allow_soft_placement,
               bool log_device_placement) {
    Placer placer(graph, devices, nullptr, allow_soft_placement,
                  log_device_placement);
    return placer.Run();
  }

  Status Place(Graph* graph, DeviceSet* devices) {
    return Place(graph, devices, true, false);
  }

  Status Place(Graph* graph, bool allow_soft_placement,
               bool log_device_placement) {
    return Place(graph, &devices_, allow_soft_placement, log_device_placement);
  }

  Status Place(Graph* graph) { return Place(graph, &devices_, true, false); }

  // Returns the node in "graph" with the given name.
  //
  // REQUIRES: "graph" was produced by the most recent call to BuildGraph.
  Node* GetNodeByName(const Graph& graph, const string& name) {
    const auto search = nodes_by_name_.find(name);
    CHECK(search != nodes_by_name_.end()) << "Unknown node name: " << name;
    return graph.FindNodeId(search->second);
  }

 protected:
  std::vector<std::unique_ptr<Device>> local_devices_;
  DeviceSet devices_;
  Placer::NodeNameToIdMap nodes_by_name_;

  Status ReferenceTestHelper(const string& variable_op_type,
                             const string& assign_op_type,
                             const DeviceType& expected_device_type);
};

// Fixture that add a parameter for allow_soft_placement.
// Test cases that want to test behavior with and without soft placement
// can use this fixture instead of PlacerTest.
class SoftPlacementPlacerTest : public PlacerTest,
                                public ::testing::WithParamInterface<bool> {};

INSTANTIATE_TEST_SUITE_P(, SoftPlacementPlacerTest,
                         ::testing::Values(false, true),
                         ::testing::PrintToStringParamName());

#define EXPECT_COLOCATED(g, name_a, name_b)                         \
  do {                                                              \
    Graph& g_ = (g);                                                \
    EXPECT_EQ(GetNodeByName(g_, (name_a))->assigned_device_name(),  \
              GetNodeByName(g_, (name_b))->assigned_device_name()); \
  } while (0)

#define EXPECT_NOT_COLOCATED(g, name_a, name_b)                     \
  do {                                                              \
    Graph& g_ = (g);                                                \
    EXPECT_NE(GetNodeByName(g_, (name_a))->assigned_device_name(),  \
              GetNodeByName(g_, (name_b))->assigned_device_name()); \
  } while (0)

#define EXPECT_DEVICE_TYPE(g, name, expected_device_type)               \
  EXPECT_EQ(DeviceType(expected_device_type).type(),                    \
            devices_                                                    \
                .FindDeviceByName(                                      \
                    GetNodeByName((g), (name))->assigned_device_name()) \
                ->attributes()                                          \
                .device_type())

#define EXPECT_DEVICE_CONTAINS(g, name, device_substr) \
  EXPECT_TRUE(::tensorflow::str_util::StrContains(     \
      GetNodeByName((g), (name))->assigned_device_name(), device_substr))

// Test that a graph with no constraints will successfully assign nodes to the
// "best available" device (i.e. prefer GPU over CPU).
TEST_F(PlacerTest, TestNoConstraints) {
  Graph g(OpRegistry::Global());
  {  // Scope for temporary variables used to construct g.
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    Node* input = ops::SourceOp("TestInput", b.opts().WithName("in"));
    ops::UnaryOp("TestRelu", ops::NodeOut(input, 0), b.opts().WithName("n1"));
    ops::UnaryOp("TestRelu", ops::NodeOut(input, 1), b.opts().WithName("n2"));
    TF_EXPECT_OK(BuildGraph(b, &g));
  }

  TF_EXPECT_OK(Place(&g));
  EXPECT_DEVICE_TYPE(g, "in", "FakeCPU");
  EXPECT_DEVICE_TYPE(g, "n1", "FakeGPU");
  EXPECT_DEVICE_TYPE(g, "n2", "FakeGPU");
}

// Test that a graph with no constraints but using kernels that have a specified
// device priority will successfully assign nodes to the device with higher
// priority
TEST_F(PlacerTest, TestNoConstraintsWithPrioritizedKernels) {
  Graph g(OpRegistry::Global());
  {  // Scope for temporary variables used to construct g.
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    Node* input = ops::SourceOp("TestInput", b.opts().WithName("in"));
    ops::UnaryOp("TestDatasetOp", ops::NodeOut(input, 0),
                 b.opts().WithName("n1"));
    ops::UnaryOp("TestDatasetOp", ops::NodeOut(input, 1),
                 b.opts().WithName("n2"));
    TF_EXPECT_OK(BuildGraph(b, &g));
  }

  TF_EXPECT_OK(Place(&g));
  EXPECT_DEVICE_TYPE(g, "in", "FakeCPU");
  EXPECT_DEVICE_TYPE(g, "n1", "FakeCPU");
  EXPECT_DEVICE_TYPE(g, "n2", "FakeCPU");
}

TEST_F(PlacerTest, TestGPUInputIntoPrioritizedKernel) {
  Graph g(OpRegistry::Global());
  {
    // Scope for temp variables used to construct g.
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    Node* input = ops::SourceOp("TestGPUOutput", b.opts().WithName("in"));
    ops::UnaryOp("TestDatasetOp", ops::NodeOut(input, 0),
                 b.opts().WithName("n1"));
    TF_EXPECT_OK(BuildGraph(b, &g));
  }

  TF_EXPECT_OK(Place(&g));
  EXPECT_DEVICE_TYPE(g, "in", "FakeGPU");
  EXPECT_DEVICE_TYPE(g, "n1", "FakeCPU");
}

// Tests that a GPU kernel colocated with prioritized kernel respects it.
TEST_F(PlacerTest, TestGPUInputColocatedWithPrioritizedKernel) {
  Graph g(OpRegistry::Global());
  {
    // Scope for temp variables used to construct g.
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    Node* input = ops::SourceOp("TestGPUOutput", b.opts().WithName("in"));
    // We colocate n1 with in.
    ops::UnaryOp("TestDatasetOp", ops::NodeOut(input, 0),
                 b.opts().WithName("n1").WithAttr("_class", {"loc:@in"}));
    // We don't colocate n2 with in.
    ops::UnaryOp("TestDatasetOp", ops::NodeOut(input, 0),
                 b.opts().WithName("n2"));
    TF_EXPECT_OK(BuildGraph(b, &g));
  }

  TF_EXPECT_OK(Place(&g));
  EXPECT_DEVICE_TYPE(g, "in", "FakeGPU");
  EXPECT_DEVICE_TYPE(g, "n1", "FakeGPU");
  EXPECT_DEVICE_TYPE(g, "n2", "FakeCPU");
}

REGISTER_OP("CreateDatasetCPU").Output("o: resource");
REGISTER_KERNEL_BUILDER(Name("CreateDatasetCPU").Device("FakeCPU"), DummyOp);

REGISTER_OP("CreateDatasetSP").Output("o: resource");
REGISTER_KERNEL_BUILDER(Name("CreateDatasetSP").Device("FakeCPU").Priority(2),
                        DummyOp);
REGISTER_KERNEL_BUILDER(Name("CreateDatasetSP").Device("FakeGPU").Priority(1),
                        DummyOp);

REGISTER_OP("CreateDatasetRP").Output("o: resource");
REGISTER_KERNEL_BUILDER(Name("CreateDatasetRP").Device("FakeCPU").Priority(1),
                        DummyOp);
REGISTER_KERNEL_BUILDER(Name("CreateDatasetRP").Device("FakeGPU").Priority(2),
                        DummyOp);

REGISTER_OP("CreateDatasetNP").Output("o: resource");
REGISTER_KERNEL_BUILDER(Name("CreateDatasetNP").Device("FakeCPU"), DummyOp);
REGISTER_KERNEL_BUILDER(Name("CreateDatasetNP").Device("FakeGPU"), DummyOp);

REGISTER_OP("IteratorNP").Input("i: resource").Output("o: float");
REGISTER_KERNEL_BUILDER(Name("IteratorNP").Device("FakeCPU"), DummyOp);
REGISTER_KERNEL_BUILDER(Name("IteratorNP").Device("FakeGPU"), DummyOp);

REGISTER_OP("IteratorSP").Input("i: resource").Output("o: float");
REGISTER_KERNEL_BUILDER(Name("IteratorSP").Device("FakeCPU").Priority(2),
                        DummyOp);
REGISTER_KERNEL_BUILDER(Name("IteratorSP").Device("FakeGPU").Priority(1),
                        DummyOp);

REGISTER_OP("IteratorRP").Input("i: resource").Output("o: float");
REGISTER_KERNEL_BUILDER(Name("IteratorRP").Device("FakeCPU").Priority(1),
                        DummyOp);
REGISTER_KERNEL_BUILDER(Name("IteratorRP").Device("FakeGPU").Priority(2),
                        DummyOp);

REGISTER_OP("IteratorGPU").Input("i: resource").Output("o: float");
REGISTER_KERNEL_BUILDER(Name("IteratorGPU").Device("FakeGPU"), DummyOp);

// Test reference edges with one node having prioritized kernels and the other
// has no preference. We should respect priority here.
TEST_F(PlacerTest, TestDSWithPriority) {
  Graph g(OpRegistry::Global());
  {
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    Node* ds = ops::SourceOp("CreateDatasetSP", b.opts().WithName("ds"));
    ops::UnaryOp("IteratorNP", ops::NodeOut(ds, 0), b.opts().WithName("it"));
    TF_EXPECT_OK(BuildGraph(b, &g));
  }
  TF_EXPECT_OK(Place(&g));
  EXPECT_DEVICE_TYPE(g, "ds", "FakeCPU");
  EXPECT_DEVICE_TYPE(g, "it", "FakeCPU");
}

// Test reference edges with one node having kernels with regular priority and
// the other has no preference. We should respect priority here.
TEST_F(PlacerTest, TestDSWithGPUPriority) {
  Graph g(OpRegistry::Global());
  {
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    Node* ds = ops::SourceOp("CreateDatasetRP", b.opts().WithName("ds"));
    ops::UnaryOp("IteratorNP", ops::NodeOut(ds, 0), b.opts().WithName("it"));
    TF_EXPECT_OK(BuildGraph(b, &g));
  }
  TF_EXPECT_OK(Place(&g));
  EXPECT_DEVICE_TYPE(g, "ds", "FakeGPU");
  EXPECT_DEVICE_TYPE(g, "it", "FakeGPU");
}

// Test reference edges with one node having prioritized kernels and the other
// has no preference. We should respect priority here.
TEST_F(PlacerTest, TestITWithPriority) {
  Graph g(OpRegistry::Global());
  {
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    Node* ds = ops::SourceOp("CreateDatasetNP", b.opts().WithName("ds"));
    ops::UnaryOp("IteratorSP", ops::NodeOut(ds, 0), b.opts().WithName("it"));
    TF_EXPECT_OK(BuildGraph(b, &g));
  }
  TF_EXPECT_OK(Place(&g));
  EXPECT_DEVICE_TYPE(g, "ds", "FakeCPU");
  EXPECT_DEVICE_TYPE(g, "it", "FakeCPU");
}

// Test reference edges with one node having kernels with regular priority and
// the other has no preference. We should respect priority here.
TEST_F(PlacerTest, TestITWithGPUPriority) {
  Graph g(OpRegistry::Global());
  {
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    Node* ds = ops::SourceOp("CreateDatasetNP", b.opts().WithName("ds"));
    ops::UnaryOp("IteratorRP", ops::NodeOut(ds, 0), b.opts().WithName("it"));
    TF_EXPECT_OK(BuildGraph(b, &g));
  }
  TF_EXPECT_OK(Place(&g));
  EXPECT_DEVICE_TYPE(g, "ds", "FakeGPU");
  EXPECT_DEVICE_TYPE(g, "it", "FakeGPU");
}

// Test reference edges with one node having prioritized kernels and other node
// can only be placed on GPU. We should respect the constraint then.
TEST_F(PlacerTest, TestITGPU) {
  Graph g(OpRegistry::Global());
  {
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    Node* ds = ops::SourceOp("CreateDatasetSP", b.opts().WithName("ds"));
    ops::UnaryOp("IteratorGPU", ops::NodeOut(ds, 0), b.opts().WithName("it"));
    TF_EXPECT_OK(BuildGraph(b, &g));
  }
  TF_EXPECT_OK(Place(&g));
  EXPECT_DEVICE_TYPE(g, "ds", "FakeGPU");
  EXPECT_DEVICE_TYPE(g, "it", "FakeGPU");
}

// Test reference edges with one node having prioritized kernels and other node
// can only be placed on CPU. We should respect the constraint then.
TEST_F(PlacerTest, TestSimpleIteratorOnlyGPU) {
  Graph g(OpRegistry::Global());
  {
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    Node* ds = ops::SourceOp("CreateDatasetCPU", b.opts().WithName("ds"));
    ops::UnaryOp("IteratorRP", ops::NodeOut(ds, 0), b.opts().WithName("it"));
    TF_EXPECT_OK(BuildGraph(b, &g));
  }
  TF_EXPECT_OK(Place(&g));
  EXPECT_DEVICE_TYPE(g, "ds", "FakeCPU");
  EXPECT_DEVICE_TYPE(g, "it", "FakeCPU");
}

// Test constraints with agreeing priorities.
TEST_F(PlacerTest, TestAgreeingPriorities) {
  Graph g(OpRegistry::Global());
  {
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    Node* ds = ops::SourceOp("CreateDatasetSP", b.opts().WithName("ds"));
    ops::UnaryOp("IteratorSP", ops::NodeOut(ds, 0), b.opts().WithName("it"));
    TF_EXPECT_OK(BuildGraph(b, &g));
  }
  TF_EXPECT_OK(Place(&g));
  EXPECT_DEVICE_TYPE(g, "ds", "FakeCPU");
  EXPECT_DEVICE_TYPE(g, "it", "FakeCPU");
}

// Test constraints with agreeing regular priorities.
TEST_F(PlacerTest, TestAgreeingRegularPriorities) {
  Graph g(OpRegistry::Global());
  {
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    Node* ds = ops::SourceOp("CreateDatasetRP", b.opts().WithName("ds"));
    ops::UnaryOp("IteratorRP", ops::NodeOut(ds, 0), b.opts().WithName("it"));
    TF_EXPECT_OK(BuildGraph(b, &g));
  }
  TF_EXPECT_OK(Place(&g));
  EXPECT_DEVICE_TYPE(g, "ds", "FakeGPU");
  EXPECT_DEVICE_TYPE(g, "it", "FakeGPU");
}

// Test constraints with different priorities. In this case, we should bail
// and just revert to default.
TEST_F(PlacerTest, TestConflictingPriorities) {
  Graph g(OpRegistry::Global());
  {
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    Node* ds = ops::SourceOp("CreateDatasetSP", b.opts().WithName("ds"));
    ops::UnaryOp("IteratorRP", ops::NodeOut(ds, 0), b.opts().WithName("it"));
    TF_EXPECT_OK(BuildGraph(b, &g));
  }
  TF_EXPECT_OK(Place(&g));
  EXPECT_DEVICE_TYPE(g, "ds", "FakeGPU");
  EXPECT_DEVICE_TYPE(g, "it", "FakeGPU");
}

// Test constraints with different priorities. In this case, we should bail
// and just revert to default.
TEST_F(PlacerTest, TestConflictingPrioritiesReversed) {
  Graph g(OpRegistry::Global());
  {
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    Node* ds = ops::SourceOp("CreateDatasetRP", b.opts().WithName("ds"));
    ops::UnaryOp("IteratorSP", ops::NodeOut(ds, 0), b.opts().WithName("it"));
    TF_EXPECT_OK(BuildGraph(b, &g));
  }
  TF_EXPECT_OK(Place(&g));
  EXPECT_DEVICE_TYPE(g, "ds", "FakeGPU");
  EXPECT_DEVICE_TYPE(g, "it", "FakeGPU");
}

// Test that a graph with device type and reference constraints on
// some of the ops will successfully assign nodes to the constrained
// device, and colocate nodes with reference connections.
TEST_F(PlacerTest, TestDeviceTypeConstraints) {
  Graph g(OpRegistry::Global());
  {  // Scope for temporary variables used to construct g.
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    Node* input = ops::SourceOp("TestInput", b.opts().WithName("in"));
    Node* var_cpu = ops::SourceOp("VariableCPU", b.opts().WithName("var_cpu"));
    ops::BinaryOp("AssignCPU", var_cpu, input, b.opts().WithName("assign_cpu"));
    Node* var_gpu = ops::SourceOp("VariableGPU", b.opts().WithName("var_gpu"));
    ops::BinaryOp("AssignGPU", var_gpu, input, b.opts().WithName("assign_gpu"));
    TF_EXPECT_OK(BuildGraph(b, &g));
  }

  TF_EXPECT_OK(Place(&g));
  EXPECT_DEVICE_TYPE(g, "in", "FakeCPU");
  EXPECT_DEVICE_TYPE(g, "var_cpu", "FakeCPU");
  EXPECT_DEVICE_TYPE(g, "assign_cpu", "FakeCPU");
  EXPECT_COLOCATED(g, "var_cpu", "assign_cpu");
  EXPECT_DEVICE_TYPE(g, "var_gpu", "FakeGPU");
  EXPECT_DEVICE_TYPE(g, "assign_gpu", "FakeGPU");
  EXPECT_COLOCATED(g, "var_gpu", "assign_gpu");
}

TEST_F(PlacerTest, TestMetadataColocatedWithInput) {
  Graph g(OpRegistry::Global());
  {  // Scope for temporary variables used to construct g.
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    Node* var_cpu = ops::SourceOp("VariableCPU", b.opts().WithName("var_cpu"));

    // Normally, shape has a GPU implementation and would be placed
    // on GPU.  However, because it is a metadata operation, it is
    // placed on CPU to avoid transferring the data from CPU to GPU.
    ops::UnaryOp("Shape", var_cpu, b.opts().WithName("shape_op"));
    TF_EXPECT_OK(BuildGraph(b, &g));
  }

  TF_EXPECT_OK(Place(&g));
  EXPECT_DEVICE_TYPE(g, "var_cpu", "FakeCPU");
  EXPECT_DEVICE_TYPE(g, "shape_op", "FakeCPU");
  EXPECT_COLOCATED(g, "var_cpu", "shape_op");
}

// Heuristic A implements "Island fusing": if a node only generates
// an output and it has only one consumer, we place the node
// with its consumer.
TEST_F(PlacerTest, TestHeuristicGeneratorFollowsSingleConsumer) {
  Graph g(OpRegistry::Global());
  {  // Scope for temporary variables used to construct g.
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);

    // A variable is only on CPU
    Node* var_cpu = ops::SourceOp("VariableCPU", b.opts().WithName("var_cpu"));

    // The constant to be assigned can be on both GPU or CPU.
    //
    // Because of the heuristic, it gets placed on CPU to avoid a
    // copy.
    Node* input = ops::SourceOp("TestCPUGPUOutput", b.opts().WithName("in"));

    // The assign is bound to CPU by the reference edge.
    ops::BinaryOp("TestAssign", var_cpu, input, b.opts().WithName("assign"));

    TF_EXPECT_OK(BuildGraph(b, &g));
  }

  TF_EXPECT_OK(Place(&g));
  EXPECT_COLOCATED(g, "var_cpu", "in");
  EXPECT_COLOCATED(g, "assign", "in");
}

TEST_F(PlacerTest, TestIgnoreGeneratorHeuristicIfWrongDevice) {
  Graph g(OpRegistry::Global());
  {  // Scope for temporary variables used to construct g.
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);

    // A variable is only on CPU
    Node* var_cpu = ops::SourceOp("VariableCPU", b.opts().WithName("var_cpu"));

    // The constant to be assigned can only be on GPU.
    //
    // The heuristic to place the generator with its consumer does
    // not apply since the consumer's device is not in the list
    // of valid devices for the generator.
    Node* input = ops::SourceOp("TestGPUOutput", b.opts().WithName("in"));

    // The assign is bound to CPU by the reference edge.
    ops::BinaryOp("TestAssign", var_cpu, input, b.opts().WithName("assign"));

    TF_EXPECT_OK(BuildGraph(b, &g));
  }

  TF_EXPECT_OK(Place(&g));
  EXPECT_DEVICE_TYPE(g, "in", "FakeGPU");
  EXPECT_DEVICE_TYPE(g, "var_cpu", "FakeCPU");
  EXPECT_COLOCATED(g, "var_cpu", "assign");
}

TEST_F(PlacerTest, TestIgnoreGeneratorHeuristicIfWrongPartialDevice) {
  Graph g(OpRegistry::Global());
  {  // Scope for temporary variables used to construct g.
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);

    // A variable is only on CPU
    Node* var_cpu = ops::SourceOp("VariableCPU", b.opts().WithName("var_cpu"));

    // The constant to be assigned can be on CPU or GPU, but is explicitly
    // placed on CPU:1.
    //
    // The heuristic to place the generator with its consumer does
    // not apply since the consumer's device is not in the list
    // of valid devices for the generator.
    Node* input =
        ops::SourceOp("TestCPUGPUOutput",
                      b.opts().WithName("in").WithDevice("/device:FakeCPU:1"));

    // The assign is bound to CPU by the reference edge.
    ops::BinaryOp("TestAssign", var_cpu, input, b.opts().WithName("assign"));

    TF_EXPECT_OK(BuildGraph(b, &g));
  }

  TF_EXPECT_OK(Place(&g));
  EXPECT_DEVICE_TYPE(g, "in", "FakeCPU");
  EXPECT_DEVICE_CONTAINS(g, "in", "/device:FakeCPU:1");
  EXPECT_DEVICE_TYPE(g, "var_cpu", "FakeCPU");
  EXPECT_COLOCATED(g, "var_cpu", "assign");
  EXPECT_DEVICE_CONTAINS(g, "var_cpu", "/device:FakeCPU:0");
}

// Test that a graph with partial device specifications on the ops
// will successfully
TEST_F(PlacerTest, TestPartialSpec) {
  Graph g(OpRegistry::Global());
  {  // Scope for temporary variables used to construct g.
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    ops::SourceOp("TestInput", b.opts().WithName("in").WithDevice("/job:a"));
    ops::SourceOp("TestVariable",
                  b.opts().WithName("var").WithDevice("/job:a"));
    TF_EXPECT_OK(BuildGraph(b, &g));
  }

  TF_EXPECT_OK(Place(&g));
  EXPECT_DEVICE_TYPE(g, "in", "FakeCPU");
  EXPECT_DEVICE_CONTAINS(g, "in", "/job:a");
  EXPECT_DEVICE_TYPE(g, "var", "FakeGPU");
  EXPECT_DEVICE_CONTAINS(g, "var", "/job:a");
}

// Test that a node with a pre-assigned device is not relocated.
TEST_F(PlacerTest, TestAssignedDevicePreserved) {
  Graph g(OpRegistry::Global());
  {  // Scope for temporary variables used to construct g.
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    ops::SourceOp("TestInput", b.opts().WithName("in"));
    TF_EXPECT_OK(BuildGraph(b, &g));
  }

  GetNodeByName(g, "in")->set_assigned_device_name(
      "/job:a/replica:0/task:0/device:FakeCPU:7");

  TF_EXPECT_OK(Place(&g));
  EXPECT_EQ("/job:a/replica:0/task:0/device:FakeCPU:7",
            GetNodeByName(g, "in")->assigned_device_name());
}

// Test that a graph with partial device specifications for CPU-only ops
// will be relocated to CPU.
TEST_F(PlacerTest, TestPartialSpecGpuToCpu) {
  Graph g(OpRegistry::Global());
  {  // Scope for temporary variables used to construct g.
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    ops::SourceOp("TestInput",
                  b.opts().WithName("in").WithDevice("/device:FakeGPU:0"));
    ops::SourceOp("TestVariable",
                  b.opts().WithName("var").WithDevice("/device:FakeGPU:0"));
    TF_EXPECT_OK(BuildGraph(b, &g));
  }

  TF_EXPECT_OK(Place(&g, true, false));
  EXPECT_DEVICE_TYPE(g, "in", "FakeCPU");
  EXPECT_DEVICE_CONTAINS(g, "in", "/device:FakeCPU");
  EXPECT_DEVICE_TYPE(g, "var", "FakeGPU");
  EXPECT_DEVICE_CONTAINS(g, "var", "/device:FakeGPU:0");
}

// Test that a resource with requested device will be moved to another
// device if it is processed by an op that is not supported on requested device.
TEST_F(PlacerTest, TestResourceMove) {
  Graph g(OpRegistry::Global());
  {
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    Node* ds =
        ops::SourceOp("CreateDatasetSP",
                      b.opts().WithName("ds").WithDevice("/device:FakeCPU:0"));
    ops::UnaryOp("IteratorGPU", ops::NodeOut(ds, 0), b.opts().WithName("it"));
    TF_EXPECT_OK(BuildGraph(b, &g));
  }
  TF_EXPECT_OK(Place(&g));
  EXPECT_DEVICE_TYPE(g, "ds", "FakeGPU");
  EXPECT_DEVICE_TYPE(g, "it", "FakeGPU");
}

// Test that a node with an assigned GPU device but has not registered
// OpKernel will fail.
TEST_F(PlacerTest, TestAssignedGpuDeviceToCpuDevice) {
  Graph g(OpRegistry::Global());
  {  // Scope for temporary variables used to construct g.
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    ops::SourceOp("TestInput", b.opts().WithName("in"));
    TF_EXPECT_OK(BuildGraph(b, &g));
  }

  GetNodeByName(g, "in")->set_assigned_device_name(
      "/job:a/replica:0/task:0/device:FakeGPU:0");

  Status s = Place(&g);
  EXPECT_EQ(error::INTERNAL, s.code()) << s.ToString();
  EXPECT_TRUE(str_util::StrContains(
      s.error_message(),
      "Assigned device '/job:a/replica:0/task:0/device:FakeGPU:0' "
      "does not have registered OpKernel support for TestInput"))
      << s.ToString();
}

// Test that graphs with reference connections are correctly placed.

// Build a graph containing a Variable op of "variable_op_type" and an
// Assign op of "assign_op_type", and expect all of the ops to be
// placed on a device of type "expected_device_type".
Status PlacerTest::ReferenceTestHelper(const string& variable_op_type,
                                       const string& assign_op_type,
                                       const DeviceType& expected_device_type) {
  Graph g(OpRegistry::Global());
  {  // Scope for temporary variables used to construct g.
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    Node* input = ops::SourceOp("TestInput", b.opts().WithName("in"));
    // Build ten variable-and-assignment pairs.
    for (int i = 0; i < 10; ++i) {
      Node* var = ops::SourceOp(variable_op_type,
                                b.opts().WithName(strings::StrCat("var_", i)));
      ops::BinaryOp(assign_op_type, var, input,
                    b.opts().WithName(strings::StrCat("assign_", i)));
    }
    TF_EXPECT_OK(BuildGraph(b, &g));
  }

  TF_RETURN_IF_ERROR(Place(&g));

  for (int i = 0; i < 10; ++i) {
    EXPECT_COLOCATED(g, strings::StrCat("var_", i),
                     strings::StrCat("assign_", i));
    EXPECT_DEVICE_TYPE(g, strings::StrCat("var_", i), expected_device_type);
    EXPECT_DEVICE_TYPE(g, strings::StrCat("assign_", i), expected_device_type);
  }

  return Status::OK();
}

// Test all 2^3 combinations of Variable and Assignment op types
// (unconstrained, CPU-only, and GPU-only).
TEST_F(PlacerTest, TestReferenceConnection) {
  Status s;
  TF_EXPECT_OK(ReferenceTestHelper("TestVariable", "TestAssign", "FakeGPU"));
  TF_EXPECT_OK(ReferenceTestHelper("TestVariable", "AssignCPU", "FakeCPU"));
  TF_EXPECT_OK(ReferenceTestHelper("TestVariable", "AssignGPU", "FakeGPU"));
  TF_EXPECT_OK(ReferenceTestHelper("VariableCPU", "TestAssign", "FakeCPU"));
  TF_EXPECT_OK(ReferenceTestHelper("VariableCPU", "AssignCPU", "FakeCPU"));
  {
    Status s = ReferenceTestHelper("VariableCPU", "AssignGPU", "FakeCPU");
    EXPECT_EQ(error::INVALID_ARGUMENT, s.code());
    EXPECT_TRUE(str_util::StrContains(
        s.error_message(), "no device type supports both of those nodes"));
  }
  TF_EXPECT_OK(ReferenceTestHelper("VariableGPU", "TestAssign", "FakeGPU"));
  {
    Status s = ReferenceTestHelper("VariableGPU", "AssignCPU", "FakeCPU");
    EXPECT_EQ(error::INVALID_ARGUMENT, s.code());
    EXPECT_TRUE(str_util::StrContains(
        s.error_message(), "no device type supports both of those nodes"));
  }
  TF_EXPECT_OK(ReferenceTestHelper("VariableGPU", "AssignGPU", "FakeGPU"));
}

// Handle-using dummy variable ops.
REGISTER_OP("TestHandleVariable").Output("o: resource");
REGISTER_KERNEL_BUILDER(Name("TestHandleVariable").Device("FakeCPU"), DummyOp);
REGISTER_KERNEL_BUILDER(Name("TestHandleVariable").Device("FakeGPU"), DummyOp);

REGISTER_OP("HandleVariableCPU").Output("o: resource");
REGISTER_KERNEL_BUILDER(Name("HandleVariableCPU").Device("FakeCPU"), DummyOp);

REGISTER_OP("HandleVariableGPU").Output("o: resource");
REGISTER_KERNEL_BUILDER(Name("HandleVariableGPU").Device("FakeGPU"), DummyOp);

REGISTER_OP("TestHandleAssign").Input("i: resource").Input("v: float");
REGISTER_KERNEL_BUILDER(Name("TestHandleAssign").Device("FakeCPU"), DummyOp);
REGISTER_KERNEL_BUILDER(Name("TestHandleAssign").Device("FakeGPU"), DummyOp);

REGISTER_OP("HandleAssignCPU").Input("i: resource").Input("v: float");
REGISTER_KERNEL_BUILDER(Name("HandleAssignCPU").Device("FakeCPU"), DummyOp);

REGISTER_OP("HandleAssignGPU").Input("i: resource").Input("v: float");
REGISTER_KERNEL_BUILDER(Name("HandleAssignGPU").Device("FakeGPU"), DummyOp);

REGISTER_OP("TestTwoHandlesIn").Input("i: resource").Input("j: resource");
REGISTER_KERNEL_BUILDER(Name("TestTwoHandlesIn").Device("FakeCPU"), DummyOp);
REGISTER_KERNEL_BUILDER(Name("TestTwoHandlesIn").Device("FakeGPU"), DummyOp);

// Tests all combinations of resource handles and ops using them.
TEST_F(PlacerTest, TestResourceHandle) {
  auto handle_test = [this](const string& var_op_name,
                            const string& use_op_name, DeviceType device) {
    Graph g(OpRegistry::Global());
    {  // Scope for temporary variables used to construct g.
      GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
      Node* input = ops::SourceOp("TestInput", b.opts().WithName("in"));
      Node* var = ops::SourceOp(var_op_name, b.opts().WithName("var"));
      ops::BinaryOp(use_op_name, var, input, b.opts().WithName("assign"));
      TF_EXPECT_OK(BuildGraph(b, &g));
    }

    TF_RETURN_IF_ERROR(Place(&g));

    EXPECT_COLOCATED(g, "var", "assign");
    EXPECT_DEVICE_TYPE(g, "var", device);
    EXPECT_DEVICE_TYPE(g, "assign", device);
    return Status::OK();
  };
  TF_EXPECT_OK(
      handle_test("TestHandleVariable", "TestHandleAssign", "FakeGPU"));
  TF_EXPECT_OK(handle_test("TestHandleVariable", "HandleAssignCPU", "FakeCPU"));
  TF_EXPECT_OK(handle_test("TestHandleVariable", "HandleAssignGPU", "FakeGPU"));
  TF_EXPECT_OK(handle_test("HandleVariableCPU", "TestHandleAssign", "FakeCPU"));
  TF_EXPECT_OK(handle_test("HandleVariableCPU", "HandleAssignCPU", "FakeCPU"));
  TF_EXPECT_OK(handle_test("HandleVariableGPU", "HandleAssignGPU", "FakeGPU"));
  TF_EXPECT_OK(handle_test("HandleVariableGPU", "TestHandleAssign", "FakeGPU"));
  EXPECT_FALSE(
      handle_test("HandleVariableGPU", "HandleAssignCPU", "FakeCPU").ok());
  EXPECT_FALSE(
      handle_test("HandleVariableCPU", "HandleAssignGPU", "FakeCPU").ok());
}

TEST_F(PlacerTest, TestResourceHandlesOnDifferentDevicesFails) {
  auto handle_test = [this](bool allow_soft_placement, bool set_assigned) {
    Graph g(OpRegistry::Global());
    {  // Scope for temporary variables used to construct g.
      GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
      Node* var_cpu =
          ops::SourceOp("TestHandleVariable", b.opts().WithName("var_cpu"));
      Node* var_gpu =
          ops::SourceOp("TestHandleVariable", b.opts().WithName("var_gpu"));
      ops::BinaryOp("TestTwoHandlesIn", var_cpu, var_gpu,
                    b.opts().WithName("two_handles_in"));
      TF_EXPECT_OK(BuildGraph(b, &g));

      if (set_assigned) {
        GetNodeByName(g, "var_cpu")
            ->set_assigned_device_name(
                "/job:a/replica:0/task:0/device:FakeCPU:0");
        GetNodeByName(g, "var_gpu")
            ->set_assigned_device_name(
                "/job:a/replica:0/task:0/device:FakeGPU:0");
      } else {
        GetNodeByName(g, "var_cpu")
            ->set_requested_device("/job:a/replica:0/task:0/device:FakeCPU:0");
        GetNodeByName(g, "var_gpu")
            ->set_requested_device("/job:a/replica:0/task:0/device:FakeGPU:0");
      }
    }

    Status s = Place(&g, allow_soft_placement, true);
    EXPECT_EQ(error::INVALID_ARGUMENT, s.code()) << s.ToString();
    if (set_assigned) {
      EXPECT_TRUE(str_util::StrContains(
          s.error_message(),
          "Cannot place the graph because a reference or resource edge "
          "connects "
          "colocation groups with incompatible assigned devices: "
          "/job:a/replica:0/task:0/device:FakeGPU:0 vs "
          "/job:a/replica:0/task:0/device:FakeCPU:0"))
          << s.ToString();
    } else {
      EXPECT_TRUE(str_util::StrContains(
          s.error_message(),
          "Cannot place the graph because a reference or resource edge "
          "connects "
          "colocation groups with incompatible resource devices: "
          "/job:a/replica:0/task:0/device:FakeGPU:0 vs "
          "/job:a/replica:0/task:0/device:FakeCPU:0"))
          << s.ToString();
    }

    return Status::OK();
  };

  TF_EXPECT_OK(handle_test(false, false));
  TF_EXPECT_OK(handle_test(false, true));
  TF_EXPECT_OK(handle_test(true, false));
  TF_EXPECT_OK(handle_test(true, true));
}

// Test that an assignment of an operator to the wrong device
// is ignored when it could never be satisfied (due to reference
// edges, for example).
TEST_F(PlacerTest, TestReferenceConnectionIgnoreInfeasible) {
  Status s;
  Graph g(OpRegistry::Global());
  {
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    Node* input = ops::SourceOp(
        "TestDevice",
        b.opts().WithName("in").WithDevice("/job:a/task:0/device:FakeGPU:0"));
    Node* var =
        ops::SourceOp("TestVariable", b.opts().WithName("var_0").WithDevice(
                                          "/job:a/task:0/device:FakeGPU:0"));

    // This op is specified on CPU, but in practice will be ignored,
    // because the reference edges forces it on GPU.
    ops::BinaryOp("TestAssign", var, input,
                  b.opts().WithName("assign").WithDevice(
                      "/job:a/task:0/device:FakeCPU:0"));
    TF_EXPECT_OK(BuildGraph(b, &g));
  }

  s = Place(&g, false, false);
  TF_EXPECT_OK(s);
  EXPECT_DEVICE_TYPE(g, "var_0", "FakeGPU");
  EXPECT_DEVICE_TYPE(g, "assign", "FakeGPU");
}

// Test that an assignment of an operator to the a more specified device
// causes the device to maintain its more specific placement.
TEST_F(PlacerTest, TestReferenceConnectionMoreSpecificDestinationSourceWins) {
  Status s;
  Graph g(OpRegistry::Global());
  {
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    // Input can be on either device
    Node* input =
        ops::SourceOp("TestCPUGPUOutput",
                      b.opts().WithName("in").WithDevice("/job:a/task:0"));

    // Variable can be on either device
    Node* var = ops::SourceOp(
        "TestVariable", b.opts().WithName("var_0").WithDevice("/job:a/task:0"));

    // This op is specified on CPU and is more specific than the variable.
    // Because the variable is less specified, the variable will be
    // assigned to CPU.
    ops::BinaryOp("TestAssign", var, input,
                  b.opts().WithName("assign").WithDevice(
                      "/job:a/task:0/device:FakeCPU:0"));
    TF_EXPECT_OK(BuildGraph(b, &g));
  }

  s = Place(&g, false, false);
  TF_EXPECT_OK(s);
  EXPECT_DEVICE_TYPE(g, "var_0", "FakeCPU");
  EXPECT_DEVICE_TYPE(g, "assign", "FakeCPU");
}

// A reference connection exists between a variable and an assign,
// where the assign has a device but the variable does not.  In this
// case, the variable gets placed on the location of the assign
// operation.
TEST_F(PlacerTest, TestReferenceConnectionNoSourceDevice) {
  Status s;
  Graph g(OpRegistry::Global());
  {
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    Node* input = ops::SourceOp(
        "TestDevice",
        b.opts().WithName("in").WithDevice("/job:a/task:0/device:FakeGPU:0"));
    Node* var = ops::SourceOp("TestVariable", b.opts().WithName("var_0"));
    ops::BinaryOp("TestAssign", var, input,
                  b.opts().WithName("assign").WithDevice(
                      "/job:a/task:0/device:FakeCPU:0"));
    TF_EXPECT_OK(BuildGraph(b, &g));
  }

  s = Place(&g, false, false);
  TF_EXPECT_OK(s);
  EXPECT_DEVICE_TYPE(g, "var_0", "FakeCPU");
  EXPECT_DEVICE_TYPE(g, "assign", "FakeCPU");
}

TEST_F(PlacerTest, TestColocationGroup) {
  Graph g(OpRegistry::Global());
  {  // Scope for temporary variables used to construct g.
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    Node* input = ops::SourceOp("TestInput", b.opts().WithName("in"));
    Node* colocated_with_input = ops::UnaryOp(
        "TestRelu", input,
        b.opts().WithName("colocated_1").WithAttr("_class", {"loc:@in"}));

    // This will not be colocated with the input because TestInput is
    // only available on CPU and TestRelu will default to GPU.
    Node* not_colocated_with_input =
        ops::UnaryOp("TestRelu", input, b.opts().WithName("foo"));
    CHECK(colocated_with_input);
    CHECK(not_colocated_with_input);
    TF_EXPECT_OK(BuildGraph(b, &g));
  }

  TF_EXPECT_OK(Place(&g));
  EXPECT_COLOCATED(g, "in", "colocated_1");
  EXPECT_NOT_COLOCATED(g, "in", "foo");
}

TEST_F(PlacerTest, TestMultipleColocationGroups) {
  Graph g(OpRegistry::Global());
  {  // Scope for temporary variables used to construct g.
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    Node* input = ops::SourceOp("TestInput", b.opts().WithName("in"));
    Node* colocated_with_input = ops::UnaryOp(
        "TestRelu", input,
        b.opts().WithName("colocated_1").WithAttr("_class", {"loc:@in"}));
    Node* colocated_with_input_and_other =
        ops::UnaryOp("TestRelu", input,
                     b.opts().WithName("foo").WithAttr(
                         "_class", {"loc:@in", "loc:@colocated_1"}));
    CHECK(colocated_with_input);
    CHECK(colocated_with_input_and_other);
    TF_EXPECT_OK(BuildGraph(b, &g));
  }

  TF_EXPECT_OK(Place(&g));
  EXPECT_COLOCATED(g, "in", "colocated_1");
  EXPECT_COLOCATED(g, "in", "foo");
}

TEST_P(SoftPlacementPlacerTest, TestInvalidMultipleColocationGroups) {
  Graph g(OpRegistry::Global());
  {  // Scope for temporary variables used to construct g.
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    Node* input = ops::SourceOp("TestInput", b.opts().WithName("in"));
    Node* colocated_with_input = ops::UnaryOp(
        "ReluCPU", input,
        b.opts().WithName("colocated_1").WithAttr("_class", {"loc:@in"}));
    Node* colocated_with_input_and_other =
        ops::UnaryOp("ReluGPU", input,
                     b.opts().WithName("foo").WithAttr(
                         "_class", {"loc:@in", "loc:@colocated_1"}));
    CHECK(colocated_with_input);
    CHECK(colocated_with_input_and_other);
    TF_EXPECT_OK(BuildGraph(b, &g));
  }

  bool allow_soft_placement = GetParam();
  Status s = Place(&g, allow_soft_placement, true);
  if (allow_soft_placement) {
    EXPECT_EQ(error::OK, s.code()) << s.ToString();
    EXPECT_DEVICE_TYPE(g, "in", "FakeCPU");
    EXPECT_DEVICE_TYPE(g, "colocated_1", "FakeCPU");
    EXPECT_DEVICE_TYPE(g, "foo", "FakeGPU");
  } else {
    EXPECT_TRUE(str_util::StrContains(
        s.error_message(),
        "Cannot colocate nodes {{colocation_node foo}} and "
        "{{colocation_node in}} because no device type supports both of those "
        "nodes and the other nodes colocated with them"))
        << s.ToString();
  }
}

TEST_F(PlacerTest, TestColocationGroupWithReferenceConnections) {
  Graph g(OpRegistry::Global());
  {  // Scope for temporary variables used to construct g.
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    Node* input = ops::SourceOp("TestInput", b.opts().WithName("in"));
    Node* var1 = ops::SourceOp("VariableCPU", b.opts().WithName("var1"));
    Node* var2 = ops::SourceOp("VariableCPU", b.opts().WithName("var2"));

    // Two assigns (reference connections) with two different
    // colocation groups. Because their colocation groups all map to the
    // same device, this is a valid assignment.
    ops::BinaryOp(
        "TestAssign", var1, input,
        b.opts().WithName("assign1").WithAttr("_class", {"loc:@var1"}));
    ops::BinaryOp(
        "TestAssign", var2, input,
        b.opts().WithName("assign2").WithAttr("_class", {"loc:@var2"}));
    TF_EXPECT_OK(BuildGraph(b, &g));
  }

  TF_EXPECT_OK(Place(&g));
  EXPECT_COLOCATED(g, "in", "var1");
  EXPECT_COLOCATED(g, "in", "var2");
  EXPECT_COLOCATED(g, "var1", "assign2");
  EXPECT_COLOCATED(g, "var2", "assign1");
}

TEST_P(SoftPlacementPlacerTest,
       TestColocationGroupWithUnsatisfiableReferenceConnections) {
  Graph g(OpRegistry::Global());
  {  // Scope for temporary variables used to construct g.
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    Node* input = ops::SourceOp("TestInput", b.opts().WithName("in"));

    Node* var1 = ops::SourceOp("VariableCPU", b.opts().WithName("var1"));
    Node* var2 = ops::SourceOp("VariableCPU", b.opts().WithName("var2"));
    // Var 3 is on GPU
    Node* var3 = ops::SourceOp("VariableGPU", b.opts().WithName("var3"));

    // Two assigns (reference connections) with two different
    // colocation groups. Because their colocation groups all map to the
    // same device, this is a valid assignment.
    ops::BinaryOp(
        "TestAssign", var1, input,
        b.opts().WithName("assign1").WithAttr("_class", {"loc:@var1"}));
    ops::BinaryOp(
        "TestAssign", var2, input,
        b.opts().WithName("assign2").WithAttr("_class", {"loc:@var2"}));
    // Assign to var3, but try to use a colocation group that matches
    // the assign of var2.  This should fail because assign2 must be on CPU
    // (it has a reference edge on var2), and assign3 must be on GPU,
    // hence the conflict.
    ops::BinaryOp(
        "TestAssign", var3, input,
        b.opts().WithName("assign3").WithAttr("_class", {"loc:@var2"}));
    TF_EXPECT_OK(BuildGraph(b, &g));
  }

  bool allow_soft_placement = GetParam();
  Status s = Place(&g, allow_soft_placement, true);
  if (allow_soft_placement) {
    EXPECT_EQ(error::OK, s.code()) << s.ToString();
  } else {
    EXPECT_EQ(error::INVALID_ARGUMENT, s.code()) << s.ToString();
    EXPECT_TRUE(str_util::StrContains(
        s.error_message(),
        "Cannot colocate nodes {{colocation_node assign3}} and "
        "{{colocation_node var2}} because no device type supports both of "
        "those nodes and the other nodes colocated with them."))
        << s.ToString();
  }
}

TEST_F(PlacerTest, TestColocationAndReferenceConnections) {
  Graph g(OpRegistry::Global());
  {  // Scope for temporary variables used to construct g.
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    Node* input = ops::SourceOp("TestInput", b.opts().WithName("in"));
    for (int i = 0; i < 10; ++i) {
      // Declare ten variable and assignment pairs.
      Node* var = ops::SourceOp("TestVariable",
                                b.opts().WithName(strings::StrCat("var_", i)));
      ops::BinaryOp("TestAssign", var, input,
                    b.opts().WithName(strings::StrCat("assign_", i)));
    }
    for (int i = 10; i < 100; ++i) {
      // Create a variable colocated with some existing variable, and
      // an assignment colocated with a possibly-different variable.
      Node* var = ops::SourceOp(
          "TestVariable",
          b.opts()
              .WithName(strings::StrCat("var_", i))
              .WithAttr("_class", {strings::StrCat("loc:@var_", i % 6)}));
      ops::BinaryOp(
          "TestAssign", var, input,
          b.opts()
              .WithName(strings::StrCat("assign_", i))
              .WithAttr("_class", {strings::StrCat("loc:@assign_", i % 3)}));
    }
    TF_EXPECT_OK(BuildGraph(b, &g));
  }

  TF_EXPECT_OK(Place(&g));
  for (int i = 0; i < 10; ++i) {
    EXPECT_COLOCATED(g, strings::StrCat("var_", i),
                     strings::StrCat("assign_", i));
  }
  for (int i = 10; i < 100; ++i) {
    EXPECT_COLOCATED(g, strings::StrCat("var_", i),
                     strings::StrCat("assign_", i));
    EXPECT_COLOCATED(g, strings::StrCat("var_", i),
                     strings::StrCat("var_", i % 6));
    EXPECT_COLOCATED(g, strings::StrCat("assign_", i),
                     strings::StrCat("assign_", i % 3));
  }
}

// Test that placement fails when no devices are registered.
TEST_F(PlacerTest, TestEmptyDeviceSet) {
  Graph g(OpRegistry::Global());
  {  // Scope for temporary variables used to construct g.
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    ops::SourceOp("TestInput", b.opts().WithName("in"));
    TF_EXPECT_OK(BuildGraph(b, &g));
  }

  DeviceSet empty;

  Status s = Place(&g, &empty);
  EXPECT_TRUE(
      str_util::StrContains(s.error_message(), "No devices are registered"));
}

// Test that placement fails when the requested device forces an
// indirect constraint to be violated.
TEST_F(PlacerTest, TestHeterogeneousDeviceSetFailure) {
  Graph g(OpRegistry::Global());
  {  // Scope for temporary variables used to construct g.
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    Node* in = ops::SourceOp("TestInput", b.opts().WithName("in"));
    Node* var = ops::SourceOp("VariableGPU", b.opts().WithName("var"));
    ops::BinaryOp("TestAssign", var, in,
                  b.opts().WithName("assign").WithDevice("/job:b/task:1"));
    TF_EXPECT_OK(BuildGraph(b, &g));
  }

  DeviceSet heterogeneous;
  std::unique_ptr<Device> gpu(
      FakeDevice::MakeGPU("/job:b/replica:0/task:0/device:FakeGPU:0"));
  heterogeneous.AddDevice(gpu.get());
  std::unique_ptr<Device> cpu(
      FakeDevice::MakeCPU("/job:b/replica:0/task:1/device:FakeCPU:0"));
  heterogeneous.AddDevice(cpu.get());
  Status s = Place(&g, &heterogeneous);
  EXPECT_EQ(error::INVALID_ARGUMENT, s.code());
  EXPECT_TRUE(
      str_util::StrContains(s.error_message(),
                            "colocated with a group of nodes that required "
                            "incompatible device"));

  // The error message should contain information that indicates which
  // op types have which registered device types.
  EXPECT_TRUE(str_util::StrContains(s.error_message(), "VariableGPU: FakeGPU"))
      << s;
  EXPECT_TRUE(
      str_util::StrContains(s.error_message(), "TestAssign: FakeGPU FakeCPU"))
      << s;
}

// Test that placement fails when an unknown device is requested.
TEST_F(PlacerTest, TestUnknownDevice) {
  Graph g(OpRegistry::Global());
  {  // Scope for temporary variables used to construct g.
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    ops::SourceOp("TestInput", b.opts().WithName("in").WithDevice("/job:foo"));
    TF_EXPECT_OK(BuildGraph(b, &g));
  }

  Status s = Place(&g);
  EXPECT_EQ(error::INVALID_ARGUMENT, s.code());
  EXPECT_TRUE(str_util::StrContains(s.error_message(), "/job:foo"));
}

// Test that placement fails when the combination of partial
// constraints leads to an unknown device.
TEST_F(PlacerTest, TestUnknownMergedDevice) {
  Graph g(OpRegistry::Global());
  {  // Scope for temporary variables used to construct g.
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    ops::SourceOp("TestInput", b.opts().WithName("in").WithDevice("/job:foo"));
    TF_EXPECT_OK(BuildGraph(b, &g));
  }

  Status s = Place(&g);
  EXPECT_EQ(error::INVALID_ARGUMENT, s.code());
  EXPECT_TRUE(str_util::StrContains(s.error_message(), "/job:foo"));
}

// Test that placement fails when the previously-assigned device for a
// node is unknown.
TEST_F(PlacerTest, TestUnknownAssignedDevice) {
  Graph g(OpRegistry::Global());
  {  // Scope for temporary variables used to construct g.
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    ops::SourceOp("TestInput", b.opts().WithName("in"));
    TF_EXPECT_OK(BuildGraph(b, &g));
  }

  GetNodeByName(g, "in")->set_assigned_device_name("/job:foo");

  Status s = Place(&g);
  EXPECT_EQ(error::INTERNAL, s.code());
  EXPECT_TRUE(str_util::StrContains(
      s.error_message(),
      "Assigned device '/job:foo' does not match any device"));
}

// Test that placement fails when an op with no registered kernels is
// requested.
TEST_F(PlacerTest, TestNoKernelsRegistered) {
  Graph g(OpRegistry::Global());
  {  // Scope for temporary variables used to construct g.
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    ops::SourceOp("VariableNoKernels", b.opts().WithName("var"));
    TF_EXPECT_OK(BuildGraph(b, &g));
  }

  Status s = Place(&g);
  EXPECT_EQ(error::INVALID_ARGUMENT, s.code());
  EXPECT_TRUE(
      str_util::StrContains(s.error_message(),
                            "No OpKernel was registered to support Op "
                            "'VariableNoKernels' used by {{node var}}"));
  EXPECT_TRUE(
      str_util::StrContains(s.error_message(), "<no registered kernels>"));
}

// Test that placement fails when a kernel is registered but no known
// device supports it.
TEST_F(PlacerTest, TestNoDevicesRegistered) {
  Graph g(OpRegistry::Global());
  {  // Scope for temporary variables used to construct g.
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    ops::SourceOp("VariableGPU", b.opts().WithName("var"));
    TF_EXPECT_OK(BuildGraph(b, &g));
  }

  DeviceSet cpu_only;
  std::unique_ptr<Device> cpu(
      FakeDevice::MakeCPU("/job:a/replica:0/task:0/device:FakeCPU:0"));
  cpu_only.AddDevice(cpu.get());

  Status s = Place(&g, &cpu_only);
  EXPECT_EQ(error::INVALID_ARGUMENT, s.code());
  EXPECT_TRUE(str_util::StrContains(s.error_message(),
                                    "No OpKernel was registered to support Op "
                                    "'VariableGPU' used by {{node var}}"));
  EXPECT_TRUE(str_util::StrContains(s.error_message(), "device='FakeGPU'"));
}

// Test that placement fails when a requested device is malformed.
TEST_F(PlacerTest, TestMalformedDeviceSpecification) {
  Graph g(OpRegistry::Global());
  {  // Scope for temporary variables used to construct g.
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    ops::SourceOp("TestInput", b.opts().WithName("in").WithDevice("/foo:bar"));
    TF_EXPECT_OK(BuildGraph(b, &g));
  }

  Status s = Place(&g);
  EXPECT_EQ(error::INVALID_ARGUMENT, s.code());
  EXPECT_TRUE(str_util::StrContains(
      s.error_message(), "Malformed device specification '/foo:bar'"));
}

// Test that placement fails when a previously-assigned device is malformed.
TEST_F(PlacerTest, TestMalformedAssignedDevice) {
  Graph g(OpRegistry::Global());
  {  // Scope for temporary variables used to construct g.
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    ops::SourceOp("TestInput", b.opts().WithName("in"));
    TF_EXPECT_OK(BuildGraph(b, &g));
  }

  GetNodeByName(g, "in")->set_assigned_device_name("/foo:bar");

  Status s = Place(&g);
  EXPECT_EQ(error::INTERNAL, s.code());
  EXPECT_TRUE(str_util::StrContains(s.error_message(),
                                    "Malformed assigned device '/foo:bar'"));
}

// Test that placement fails when a device was previously assigned to
// a node, but it does not uniquely identify a particular device.
TEST_F(PlacerTest, TestNonUniqueAssignedDevice) {
  Graph g(OpRegistry::Global());
  {  // Scope for temporary variables used to construct g.
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    ops::SourceOp("TestInput", b.opts().WithName("in"));
    TF_EXPECT_OK(BuildGraph(b, &g));
  }

  GetNodeByName(g, "in")->set_assigned_device_name("/job:a");

  Status s = Place(&g);
  EXPECT_EQ(error::INTERNAL, s.code());
  EXPECT_TRUE(str_util::StrContains(
      s.error_message(), "Assigned device '/job:a' does not match any device"));
}

// Test that ops request to be placed on non-existent devices will be relocated
// to existing device of the same type if allow_soft_placement is set.
TEST_F(PlacerTest, TestNonexistentGpuAllowSoftPlacement) {
  Graph g(OpRegistry::Global());
  {  // Scope for temporary variables used to construct g.
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    ops::SourceOp("TestDevice",
                  b.opts().WithName("in").WithDevice("/device:FakeGPU:11"));
    TF_EXPECT_OK(BuildGraph(b, &g));
  }

  TF_EXPECT_OK(Place(&g, true, false));
  EXPECT_DEVICE_CONTAINS(g, "in", "/device:FakeGPU:0");
}

// Test that ops request to be placed on non-existent devices will fail if
// allow_soft_placement is not set.
TEST_F(PlacerTest, TestNonexistentGpuNoAllowSoftPlacement) {
  Graph g(OpRegistry::Global());
  {  // Scope for temporary variables used to construct g.
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    ops::SourceOp("TestDevice",
                  b.opts().WithName("in").WithDevice("/device:FakeGPU:11"));
    TF_EXPECT_OK(BuildGraph(b, &g));
  }

  Status s = Place(&g, false, false);
  EXPECT_EQ(error::INVALID_ARGUMENT, s.code());
  EXPECT_TRUE(str_util::StrContains(s.error_message(), "/device:FakeGPU:11"));
}

// Test that the "Cannot assign a device" error message contains a format tag
// when requested.
TEST_F(PlacerTest, TestNonexistentGpuNoAllowSoftPlacementFormatTag) {
  Graph g(OpRegistry::Global());
  {  // Scope for temporary variables used to construct g.
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    ops::SourceOp("TestDevice",
                  b.opts().WithName("in").WithDevice("/device:FakeGPU:11"));
    TF_EXPECT_OK(BuildGraph(b, &g));
  }

  Status s = Place(&g, false, false);
  EXPECT_EQ(error::INVALID_ARGUMENT, s.code());
  LOG(WARNING) << s.error_message();
  EXPECT_TRUE(str_util::StrContains(s.error_message(),
                                    "Cannot assign a device for operation in"));
  EXPECT_TRUE(str_util::StrContains(s.error_message(), "{{node in}}"));
}

// Test that placement fails when a node requests an explicit device that is not
// supported by the registered kernels if allow_soft_placement is no set.
TEST_F(PlacerTest, TestUnsupportedDeviceNoAllowSoftPlacement) {
  Graph g(OpRegistry::Global());
  {  // Scope for temporary variables used to construct g.
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    ops::SourceOp("VariableGPU",
                  b.opts().WithName("var").WithDevice("/device:FakeCPU:0"));
    TF_EXPECT_OK(BuildGraph(b, &g));
  }

  Status s = Place(&g, false, false);
  EXPECT_EQ(error::INVALID_ARGUMENT, s.code()) << s.ToString();
  EXPECT_TRUE(str_util::StrContains(s.error_message(), "/device:FakeCPU:0"))
      << s.ToString();
  EXPECT_TRUE(str_util::StrContains(
      s.error_message(),
      "no supported kernel for FakeCPU devices is available"))
      << s.ToString();
}

// Test that placement fails when a node requests an explicit device that is not
// supported by the registered kernels if allow_soft_placement is no set.
TEST_F(PlacerTest, TestNonExistentDevice) {
  Graph g(OpRegistry::Global());
  {  // Scope for temporary variables used to construct g.
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    ops::SourceOp("VariableGPU",
                  b.opts().WithName("var").WithDevice("/job:foo/replica:17"));
    TF_EXPECT_OK(BuildGraph(b, &g));
  }

  Status s = Place(&g, false, false);
  EXPECT_EQ(error::INVALID_ARGUMENT, s.code());
  LOG(WARNING) << s.error_message();
  EXPECT_TRUE(str_util::StrContains(
      s.error_message(), "was explicitly assigned to /job:foo/replica:17"));
  EXPECT_TRUE(
      str_util::StrContains(s.error_message(), "but available devices"));
}

#if !GOOGLE_CUDA
// Test that we inform the user if they appear to be explicitly placing nodes
// on a GPU when CUDA is not available
TEST_F(PlacerTest, TestUseGpuWithNoCuda) {
  Graph g(OpRegistry::Global());
  {  // Scope for temporary variables used to construct g.
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    ops::SourceOp("VariableGPU",
                  b.opts().WithName("var").WithDevice("/device:gpu:0"));
    TF_EXPECT_OK(BuildGraph(b, &g));
  }

  Status s = Place(&g, false, false);
  EXPECT_EQ(error::INVALID_ARGUMENT, s.code());
  LOG(WARNING) << s.error_message();
  EXPECT_TRUE(str_util::StrContains(
      s.error_message(),
      "The requested device appears to be a GPU, but CUDA is not enabled."));
}
#endif

TEST_F(PlacerTest, TestUnsupportedDeviceAllowSoftPlacement) {
  Graph g(OpRegistry::Global());
  {  // Scope for temporary variables used to construct g.
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    ops::SourceOp("TestInput",  // has only CPU kernel
                  b.opts().WithName("a").WithDevice("/device:FakeGPU:0"));
    TF_EXPECT_OK(BuildGraph(b, &g));
  }

  TF_EXPECT_OK(Place(&g, true, false));
}

// Test that a graph with device type and reference constraints on
// some of the ops will successfully assign nodes to the constrained
// device, and colocate nodes with reference connections.
TEST_F(PlacerTest, TestDeviceTypeConstraintsAllowSoftPlacement) {
  Graph g(OpRegistry::Global());
  {  // Scope for temporary variables used to construct g.
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    // var_gpu has ref output and runs on GPU.
    // force_gpu takes var_gpu and requested CPU.
    // Verify that both are placed on GPU.
    Node* var_gpu = ops::SourceOp("VariableGPU", b.opts().WithName("var_gpu"));
    ops::UnaryOp(
        "TestDeviceEnforce", var_gpu,
        b.opts().WithName("force_gpu").WithDevice("/device:FakeCPU:0"));
    // var_cpu has ref output and runs on CPU.
    // force_cpu takes var_cpu and requested GPU.
    // Verify that both are placed on CPU.
    Node* var_cpu = ops::SourceOp("VariableCPU", b.opts().WithName("var_cpu"));
    ops::UnaryOp(
        "TestDeviceEnforce", var_cpu,
        b.opts().WithName("force_cpu").WithDevice("/device:FakeGPU:0"));
    TF_EXPECT_OK(BuildGraph(b, &g));
  }

  TF_EXPECT_OK(Place(&g, true, false));
  EXPECT_DEVICE_TYPE(g, "var_gpu", "FakeGPU");
  EXPECT_DEVICE_TYPE(g, "force_gpu", "FakeGPU");
  EXPECT_COLOCATED(g, "var_gpu", "force_gpu");
  EXPECT_DEVICE_TYPE(g, "var_cpu", "FakeCPU");
  EXPECT_DEVICE_TYPE(g, "force_cpu", "FakeCPU");
  EXPECT_COLOCATED(g, "var_cpu", "force_cpu");
}

// Test that placement fails when two nodes have a reference connection
// constraint, and each node requires a mutually incompatible device.
TEST_F(PlacerTest, TestUnsatisfiableConstraintWithReferenceConnections) {
  Graph g(OpRegistry::Global());
  {  // Scope for temporary variables used to construct g.
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
    Node* var = ops::SourceOp("VariableGPU", b.opts().WithName("var"));
    Node* input = ops::SourceOp("TestInput", b.opts().WithName("in"));
    ops::BinaryOp("AssignCPU", var, input, b.opts().WithName("assign"));
    TF_EXPECT_OK(BuildGraph(b, &g));
  }

  Status s = Place(&g);
  EXPECT_EQ(error::INVALID_ARGUMENT, s.code());
  EXPECT_TRUE(str_util::StrContains(s.error_message(),
                                    "Cannot colocate nodes {{colocation_node "
                                    "var}} and {{colocation_node assign}}"));
}

// Test that a generator node follows its consumers (where there are several
// consumer nodes on the same devices).
TEST_F(PlacerTest, TestGeneratorNodeFollowsConsumerNode) {
  Graph g(OpRegistry::Global());
  {  // Scope for temporary variables used to construct g.
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);

    // A variable is only on CPU
    Node* var1_cpu =
        ops::SourceOp("VariableCPU", b.opts().WithName("var1_cpu"));
    Node* var2_cpu =
        ops::SourceOp("VariableCPU", b.opts().WithName("var2_cpu"));

    // The constant to be assigned can be on both GPU or CPU.
    //
    // Because of the heuristic, it gets placed on CPU to avoid a
    // copy.
    Node* input = ops::SourceOp("TestCPUGPUOutput", b.opts().WithName("in"));

    // The assigns are bound to CPU by the reference edge.
    ops::BinaryOp("TestAssign", var1_cpu, input, b.opts().WithName("assign1"));
    ops::BinaryOp("TestAssign", var2_cpu, input, b.opts().WithName("assign2"));

    TF_EXPECT_OK(BuildGraph(b, &g));
  }

  TF_EXPECT_OK(Place(&g));
  EXPECT_COLOCATED(g, "var1_cpu", "in");
  EXPECT_COLOCATED(g, "assign1", "in");
  EXPECT_COLOCATED(g, "var2_cpu", "in");
  EXPECT_COLOCATED(g, "assign2", "in");
}

// Test that a generator node does not follow its consumers (where there are
// several consumers on different devices).
TEST_F(PlacerTest, TestGeneratorNodeDoesntFollowNonColocatedConsumers) {
  Graph g(OpRegistry::Global());
  {  // Scope for temporary variables used to construct g.
    GraphDefBuilder b(GraphDefBuilder::kFailImmediately);

    // A variable is only on CPU
    Node* var1_cpu =
        ops::SourceOp("VariableCPU", b.opts().WithName("var1_cpu"));
    Node* var2_cpu =
        ops::SourceOp("VariableCPU", b.opts().WithName("var2_cpu"));

    // The constant to be assigned can be on both GPU or CPU.
    //
    // Because of the heuristic, it ought to be on the GPU (cannot be
    // co-located with both consumers, so goes to the 'standard' place)
    Node* input = ops::SourceOp("TestCPUGPUOutput", b.opts().WithName("in"));

    // The assigns are bound to CPU by the reference edge.
    ops::BinaryOp("TestAssign", var1_cpu, input, b.opts().WithName("assign1"));
    ops::BinaryOp("TestAssign", var2_cpu, input, b.opts().WithName("assign2"));

    TF_EXPECT_OK(BuildGraph(b, &g));

    GetNodeByName(g, "var1_cpu")
        ->set_assigned_device_name("/job:a/replica:0/task:0/device:FakeCPU:1");

    GetNodeByName(g, "var2_cpu")
        ->set_assigned_device_name("/job:a/replica:0/task:0/device:FakeCPU:2");
  }

  TF_EXPECT_OK(Place(&g));
  EXPECT_COLOCATED(g, "assign1", "var1_cpu");
  EXPECT_COLOCATED(g, "assign2", "var2_cpu");
  EXPECT_DEVICE_TYPE(g, "in", "FakeGPU");
}

REGISTER_KERNEL_BUILDER(Name("_Arg").Device("FakeCPU"), DummyOp);
REGISTER_KERNEL_BUILDER(Name("_Arg").Device("FakeGPU"), DummyOp);
REGISTER_KERNEL_BUILDER(Name("_Retval").Device("FakeCPU"), DummyOp);
REGISTER_KERNEL_BUILDER(Name("_Retval").Device("FakeGPU"), DummyOp);
REGISTER_KERNEL_BUILDER(Name("Identity").Device("FakeCPU"), DummyOp);
REGISTER_KERNEL_BUILDER(Name("Identity").Device("FakeGPU"), DummyOp);
REGISTER_KERNEL_BUILDER(Name("Const").Device("FakeCPU"), DummyOp);
REGISTER_KERNEL_BUILDER(Name("Const").Device("FakeGPU"), DummyOp);
REGISTER_KERNEL_BUILDER(Name("Mul").Device("FakeCPU"), DummyOp);
REGISTER_KERNEL_BUILDER(Name("Mul").Device("FakeGPU"), DummyOp);
REGISTER_KERNEL_BUILDER(Name("Add").Device("FakeCPU"), DummyOp);
REGISTER_KERNEL_BUILDER(Name("Add").Device("FakeGPU"), DummyOp);

TEST_P(SoftPlacementPlacerTest,
       RequestedDeviceOnResourceGeneratorIsTreatedAsAssigned) {
  /*
   *    a:RES:GPU  b:RES:CPU
   *       |         |
   *       |         |
   *       v         v
   *      id1       id2
   *     @loc:id2
   */
  FunctionDef func = test::function::ResourceOutput();
  GraphDef graph = GDef(
      {
          NDef("a", "_Arg", {}, {{"T", DT_RESOURCE}}, kGPU),
          NDef("b", "_Arg", {}, {{"T", DT_RESOURCE}}, kCPU),
          NDef("id1", "Identity", {"a"},
               {{"T", DT_RESOURCE},
                {"_class", gtl::ArraySlice<string>({"loc:@id2"})}}),
          NDef("id2", "Identity", {"b"}, {{"T", DT_RESOURCE}}),
      },
      // FunctionLib
      {func});

  Graph g(OpRegistry::Global());
  TF_ASSERT_OK(BuildGraph(graph, &g));

  bool allow_soft_placement = GetParam();
  Status s = Place(&g, allow_soft_placement, true);
  if (allow_soft_placement) {
    EXPECT_EQ(error::OK, s.code()) << s.ToString();
    EXPECT_DEVICE_TYPE(g, "a", "FakeGPU");
    EXPECT_DEVICE_TYPE(g, "id1", "FakeGPU");
    EXPECT_DEVICE_TYPE(g, "b", "FakeCPU");
    EXPECT_DEVICE_TYPE(g, "id2", "FakeCPU");
  } else {
    EXPECT_EQ(error::INVALID_ARGUMENT, s.code());
    EXPECT_TRUE(str_util::StrContains(
        s.error_message(),
        "Cannot colocate nodes {{colocation_node id2}} and {{colocation_node "
        "id1}}: Cannot merge devices with incompatible types: "
        "'/device:FakeCPU:0' and '/device:FakeGPU:0'"))
        << s.ToString();
  }
}

TEST_F(PlacerTest, RequestedDeviceCanBeOverridden) {
  /*
   *     a:RES      b:RES
   *       |         |
   *     id_a:GPU   id_b:CPU
   *       |         |
   *       v         v
   *      id1       id2
   *     @loc:id2
   */
  FunctionDef func = test::function::ResourceOutput();
  GraphDef graph = GDef(
      {
          NDef("a", "_Arg", {}, {{"T", DT_RESOURCE}}),
          NDef("b", "_Arg", {}, {{"T", DT_RESOURCE}}),
          NDef("id_a", "Identity", {"a"}, {{"T", DT_RESOURCE}}, kGPU),
          NDef("id_b", "Identity", {"b"}, {{"T", DT_RESOURCE}}, kCPU),
          NDef("id1", "Identity", {"id_a"},
               {{"T", DT_RESOURCE},
                {"_class", gtl::ArraySlice<string>({"loc:@id2"})}}),
          NDef("id2", "Identity", {"id_b"}, {{"T", DT_RESOURCE}}),
      },
      // FunctionLib
      {func});

  Graph g(OpRegistry::Global());
  TF_ASSERT_OK(BuildGraph(graph, &g));
  TF_ASSERT_OK(Place(&g));

  // All should be colocated
  EXPECT_COLOCATED(g, "a", "b");
  EXPECT_COLOCATED(g, "id_a", "id_b");
  EXPECT_COLOCATED(g, "id1", "id2");
  EXPECT_COLOCATED(g, "a", "id_a");
  EXPECT_COLOCATED(g, "a", "id1");
}

TEST_F(PlacerTest, AssignedDeviceOfColocatedNodeIsRespected) {
  /*
   *     a:float (assigned to CPU)
   *       |
   *       v
   *     iter (has only GPU kernel)
   */
  GraphDef graph = GDef({
      NDef("a", "_Arg", {}, {{"T", DT_RESOURCE}}),
      NDef("iter", "IteratorGPU", {"a"}),
  });

  Graph g(OpRegistry::Global());
  TF_ASSERT_OK(BuildGraph(graph, &g));
  GetNodeByName(g, "a")->set_assigned_device_name(kFullCPU);
  Status s = Place(&g);
  EXPECT_EQ(error::INVALID_ARGUMENT, s.code()) << s.ToString();
  EXPECT_TRUE(
      str_util::StrContains(s.error_message(),
                            "{{colocation_node iter}} was colocated with a "
                            "group of nodes that required incompatible device "
                            "'/job:a/replica:0/task:0/device:FakeCPU:0'"))
      << s.ToString();
}

TEST_P(SoftPlacementPlacerTest,
       AssignedDevicesAreNotOverriddenDueToResourcesAndColocation) {
  /*
   *     a:RES      b:RES
   *       |         |
   *     id_a:GPU   id_b:CPU
   *       |         |
   *       v         v
   *      id1       id2
   *     @loc:id2
   */
  FunctionDef func = test::function::ResourceOutput();
  GraphDef graph = GDef(
      {
          NDef("a", "_Arg", {}, {{"T", DT_RESOURCE}}),
          NDef("b", "_Arg", {}, {{"T", DT_RESOURCE}}),
          NDef("id_a", "Identity", {"a"}, {{"T", DT_RESOURCE}}),
          NDef("id_b", "Identity", {"b"}, {{"T", DT_RESOURCE}}),
          NDef("id1", "Identity", {"id_a"},
               {{"T", DT_RESOURCE},
                {"_class", gtl::ArraySlice<string>({"loc:@id2"})}}),
          NDef("id2", "Identity", {"id_b"}, {{"T", DT_RESOURCE}}),
      },
      // FunctionLib
      {func});

  Graph g(OpRegistry::Global());
  TF_ASSERT_OK(BuildGraph(graph, &g));
  GetNodeByName(g, "id_a")->set_assigned_device_name(kFullGPU);
  GetNodeByName(g, "id_b")->set_assigned_device_name(kFullCPU);

  bool allow_soft_placement = GetParam();

  Status s = Place(&g, allow_soft_placement, false);
  if (allow_soft_placement) {
    EXPECT_EQ(error::OK, s.code()) << s.ToString();
    EXPECT_DEVICE_TYPE(g, "a", "FakeGPU");
    EXPECT_DEVICE_TYPE(g, "id_a", "FakeGPU");
    EXPECT_DEVICE_TYPE(g, "id1", "FakeGPU");
    EXPECT_DEVICE_TYPE(g, "b", "FakeCPU");
    EXPECT_DEVICE_TYPE(g, "id_b", "FakeCPU");
    EXPECT_DEVICE_TYPE(g, "id2", "FakeCPU");
  } else {
    EXPECT_EQ(error::INVALID_ARGUMENT, s.code());
    EXPECT_TRUE(str_util::StrContains(
        s.error_message(),
        "Cannot colocate nodes {{colocation_node id2}} and {{colocation_node "
        "id1}}: Cannot merge devices with incompatible types: "
        "'/job:a/replica:0/task:0/device:FakeCPU:0' and "
        "'/job:a/replica:0/task:0/device:FakeGPU:0'"))
        << s.ToString();
  }
}

}  // namespace
}  // namespace tensorflow
