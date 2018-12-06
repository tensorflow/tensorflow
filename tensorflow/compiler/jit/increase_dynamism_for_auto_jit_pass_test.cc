/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/increase_dynamism_for_auto_jit_pass.h"

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/compiler/jit/node_matchers.h"
#include "tensorflow/compiler/jit/xla_cluster_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace {

using ::testing::_;
using testing::matchers::AssignedDevice;
using testing::matchers::Attr;
using testing::matchers::Const;
using testing::matchers::CtrlDeps;
using testing::matchers::Inputs;
using testing::matchers::Name;
using testing::matchers::NodeWith;
using testing::matchers::Op;
using testing::matchers::Out;

// A fake device used to populate a DeviceSet.
class FakeDevice : public Device {
 public:
  explicit FakeDevice(const DeviceAttributes& device_attributes)
      : Device(nullptr, device_attributes) {}

  Status Sync() override { return errors::Unimplemented("FakeDevice::Sync()"); }

  Allocator* GetAllocator(AllocatorAttributes attr) override { return nullptr; }

  static std::unique_ptr<Device> Make(const string& name, const string& type) {
    DeviceAttributes device_attributes;
    device_attributes.set_name(name);
    device_attributes.set_device_type(DeviceType(type).type());
    return absl::make_unique<FakeDevice>(device_attributes);
  }
};

const char* kHostName = "/job:worker/replica:0/task:0/device:CPU:0";
const char* kDeviceName = "/job:worker/replica:0/task:0/device:GPU:0";

Status IncreaseDynamismForAutoJit(const Scope& s,
                                  std::unique_ptr<Graph>* result) {
  std::vector<std::unique_ptr<Device>> devices;
  devices.push_back(FakeDevice::Make(kDeviceName, DEVICE_GPU));
  devices.push_back(FakeDevice::Make(kHostName, DEVICE_CPU));

  std::unique_ptr<DeviceSet> device_set(new DeviceSet());
  for (auto& device : devices) {
    device_set->AddDevice(device.get());
  }

  auto graph = absl::make_unique<Graph>(OpRegistry::Global());
  SessionOptions session_options;
  session_options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_global_jit_level(OptimizerOptions::ON_2);
  GraphOptimizationPassOptions options;
  options.graph = &graph;
  options.device_set = device_set.get();
  options.session_options = &session_options;

  // Scope::ToGraph seems to drop assigned devices, probably because it goes
  // through a GraphDef.  So explicitly maintain the device assignment.
  std::unordered_map<string, string> assigned_device_names;
  for (Node* n : s.graph()->nodes()) {
    assigned_device_names[n->name()] = n->assigned_device_name();
  }
  TF_RETURN_IF_ERROR(s.ToGraph(graph.get()));
  for (Node* n : graph->nodes()) {
    n->set_assigned_device_name(assigned_device_names[n->name()]);
  }

  IncreaseDynamismForAutoJitPass rewriter;
  TF_RETURN_IF_ERROR(rewriter.Run(options));
  *result = std::move(graph);
  return Status::OK();
}

TEST(SliceToDynamicSliceRewriteTest, Basic) {
  Scope root = Scope::NewRootScope()
                   .ExitOnError()
                   .WithAssignedDevice(kDeviceName)
                   .WithXlaCluster("cluster_0");

  Output input = ops::Placeholder(root.WithOpName("input"), DT_FLOAT);
  Output begin = ops::Placeholder(root.WithOpName("begin"), DT_INT32);
  Output size = ops::Const(root.WithOpName("size"), {-1, 500});
  Output slice = ops::Slice(root.WithOpName("slice"), input, begin, size);

  std::unique_ptr<Graph> result;
  TF_ASSERT_OK(IncreaseDynamismForAutoJit(root, &result));

  const int64 zero_64 = 0;
  const int32 zero_32 = 0;
  const int64 one_64 = 1;

  auto m_input = Out(NodeWith(Op("Placeholder"), Name("input")));
  auto m_begin_s64 = Out(NodeWith(
      Op("Cast"), Inputs(Out(NodeWith(Op("Placeholder"), Name("begin"))))));
  auto m_input_shape = Out(NodeWith(Op("Shape"), Inputs(m_input)));
  auto m_slice_size_0 = Out(NodeWith(
      Op("Sub"), AssignedDevice(kHostName),
      Inputs(
          Out(NodeWith(Op("Slice"), AssignedDevice(kHostName),
                       Inputs(m_input_shape, Const(zero_64), Const(one_64)))),
          Out(NodeWith(Op("Slice"), AssignedDevice(kHostName),
                       Inputs(m_begin_s64, Const(zero_64), Const(one_64)))))));
  auto m_dynamic_slice_size = Out(NodeWith(
      Op("ConcatV2"), AssignedDevice(kHostName),
      Inputs(m_slice_size_0, Const(static_cast<int64>(500)), Const(zero_32))));

  std::vector<string> compile_time_constant_inputs;
  compile_time_constant_inputs.push_back("size");
  auto m_dynamic_slice = NodeWith(
      Op("Slice"), AssignedDevice(kDeviceName),
      Attr(kXlaCompileTimeConstantInputsAttr, compile_time_constant_inputs),
      Inputs(m_input, m_begin_s64, m_dynamic_slice_size));

  Node* static_shaped_slice = testing::FindNodeByName(
      result.get(), "slice/static_shaped_slice/static_shaped_slice");
  ASSERT_NE(static_shaped_slice, nullptr);
  EXPECT_THAT(static_shaped_slice, m_dynamic_slice);
}

TEST(SliceToDynamicSliceRewriteTest, SliceFromVector) {
  Scope root = Scope::NewRootScope()
                   .ExitOnError()
                   .WithAssignedDevice(kDeviceName)
                   .WithXlaCluster("cluster_0");

  Output input = ops::Placeholder(root.WithOpName("input"), DT_FLOAT);
  Output begin = ops::Placeholder(root.WithOpName("begin"), DT_INT32);
  Output size = ops::Const(root.WithOpName("size"), {-1});
  Output slice = ops::Slice(root.WithOpName("slice"), input, begin, size);

  std::unique_ptr<Graph> result;
  TF_ASSERT_OK(IncreaseDynamismForAutoJit(root, &result));

  Node* static_shaped_slice = testing::FindNodeByName(
      result.get(), "slice/static_shaped_slice/static_shaped_slice");
  EXPECT_NE(static_shaped_slice, nullptr);
  EXPECT_THAT(result->nodes(), Not(Contains(NodeWith(Op("ConcatV2")))));
}

TEST(SliceToDynamicSliceRewriteTest, ControlDependencePreserved) {
  Scope root = Scope::NewRootScope()
                   .ExitOnError()
                   .WithAssignedDevice(kDeviceName)
                   .WithXlaCluster("cluster_0");

  Output input = ops::Placeholder(root.WithOpName("input"), DT_FLOAT);
  Output begin = ops::Placeholder(root.WithOpName("begin"), DT_INT32);
  Output size = ops::Const(root.WithOpName("size"), {-1, 500});
  Output control_pred = ops::Placeholder(root.WithOpName("control"), DT_BOOL);
  Output slice = ops::Slice(root.WithOpName("slice"), input, begin, size);
  root.graph()->AddControlEdge(control_pred.node(), slice.node());

  std::unique_ptr<Graph> result;
  TF_ASSERT_OK(IncreaseDynamismForAutoJit(root, &result));

  Node* static_shaped_slice = testing::FindNodeByName(
      result.get(), "slice/static_shaped_slice/static_shaped_slice");
  ASSERT_NE(static_shaped_slice, nullptr);
  EXPECT_THAT(static_shaped_slice,
              NodeWith(Op("Slice"),
                       CtrlDeps(NodeWith(Op("Placeholder"), Name("control")))));
}

int64 ToInt64(int v) { return static_cast<int64>(v); }

TEST(SliceToDynamicSliceRewriteTest, Int64Indices) {
  Scope root = Scope::NewRootScope()
                   .ExitOnError()
                   .WithAssignedDevice(kDeviceName)
                   .WithXlaCluster("cluster_0");

  Output input = ops::Placeholder(root.WithOpName("input"), DT_FLOAT);
  Output begin = ops::Placeholder(root.WithOpName("begin"), DT_INT64);
  Output size =
      ops::Const(root.WithOpName("size"), {ToInt64(-1), ToInt64(500)});
  Output slice = ops::Slice(root.WithOpName("slice"), input, begin, size);

  std::unique_ptr<Graph> result;
  TF_ASSERT_OK(IncreaseDynamismForAutoJit(root, &result));

  EXPECT_THAT(result->nodes(), Not(Contains(NodeWith(Op("Cast")))));
}

TEST(SliceToDynamicSliceRewriteTest, DontRewriteInvalidSlice) {
  Scope root = Scope::NewRootScope()
                   .ExitOnError()
                   .WithAssignedDevice(kDeviceName)
                   .WithXlaCluster("cluster_0");

  Output input = ops::Placeholder(root.WithOpName("input"), DT_FLOAT);
  Output begin = ops::Placeholder(root.WithOpName("begin"), DT_INT32);

  // The shape refiner throws an error if we use a bogus constant value for
  // size.  So we first use a Placeholder to placate the shape refiner, and
  // later replace it with a bogus constant.
  Output size_placeholder =
      ops::Placeholder(root.WithOpName("size_placeholder"), DT_INT32);
  Output slice =
      ops::Slice(root.WithOpName("slice"), input, begin, size_placeholder);

  Output size = ops::Const(root.WithOpName("size"), {-8, 500});
  TF_ASSERT_OK(root.graph()->UpdateEdge(/*new_src=*/size.node(),
                                        /*new_src_index=*/0,
                                        /*dst=*/slice.node(), /*dst_index=*/2));

  std::unique_ptr<Graph> result;
  TF_ASSERT_OK(IncreaseDynamismForAutoJit(root, &result));

  EXPECT_THAT(result->nodes(),
              Not(Contains(NodeWith(Op("Slice"),
                                    Attr(kXlaCompileTimeConstantInputsAttr)))));
}

TEST(SliceToDynamicSliceRewriteTest, DontRewriteUnclusteredSlice) {
  Scope root =
      Scope::NewRootScope().ExitOnError().WithAssignedDevice(kDeviceName);

  Output input = ops::Placeholder(root.WithOpName("input"), DT_FLOAT);
  Output begin = ops::Placeholder(root.WithOpName("begin"), DT_INT32);
  Output size = ops::Const(root.WithOpName("size"), {-1, 500});
  Output slice = ops::Slice(root.WithOpName("slice"), input, begin, size);

  std::unique_ptr<Graph> result;
  TF_ASSERT_OK(IncreaseDynamismForAutoJit(root, &result));

  EXPECT_THAT(result->nodes(),
              Not(Contains(NodeWith(Op("Slice"),
                                    Attr(kXlaCompileTimeConstantInputsAttr)))));
}

TEST(SliceToDynamicSliceRewriteTest, DontRewriteSliceWithNonConstSize) {
  Scope root = Scope::NewRootScope()
                   .ExitOnError()
                   .WithAssignedDevice(kDeviceName)
                   .WithXlaCluster("cluster_0");

  Output input = ops::Placeholder(root.WithOpName("input"), DT_FLOAT);
  Output begin = ops::Placeholder(root.WithOpName("begin"), DT_INT64);
  Output size = ops::Placeholder(root.WithOpName("size"), DT_INT64);
  Output slice = ops::Slice(root.WithOpName("slice"), input, begin, size);

  std::unique_ptr<Graph> result;
  TF_ASSERT_OK(IncreaseDynamismForAutoJit(root, &result));

  EXPECT_THAT(result->nodes(),
              Not(Contains(NodeWith(Op("Slice"),
                                    Attr(kXlaCompileTimeConstantInputsAttr)))));
}

TEST(SliceToDynamicSliceRewriteTest, ScalarSlice) {
  Scope root = Scope::NewRootScope()
                   .ExitOnError()
                   .WithAssignedDevice(kDeviceName)
                   .WithXlaCluster("cluster_0");

  Output input = ops::Placeholder(root.WithOpName("input"), DT_FLOAT);
  Output begin = ops::Placeholder(root.WithOpName("begin"), DT_INT64);
  Output size = ops::Const<int64>(root.WithOpName("size"), {});
  Output slice = ops::Slice(root.WithOpName("slice"), input, begin, size);

  std::unique_ptr<Graph> result;
  TF_ASSERT_OK(IncreaseDynamismForAutoJit(root, &result));

  Node* static_shaped_slice = testing::FindNodeByName(
      result.get(), "slice/static_shaped_slice/static_shaped_slice");
  ASSERT_NE(static_shaped_slice, nullptr);
  EXPECT_THAT(static_shaped_slice,
              NodeWith(Op("Slice"), Attr(kXlaCompileTimeConstantInputsAttr),
                       Inputs(_, _, Out(NodeWith(Name(size.node()->name()))))));
}

TEST(SliceToDynamicSliceRewriteTest, IndicesNotVector) {
  Scope root = Scope::NewRootScope()
                   .ExitOnError()
                   .WithAssignedDevice(kDeviceName)
                   .WithXlaCluster("cluster_0");

  auto ToInt64 = [](int v) { return static_cast<int64>(v); };

  Output input = ops::Placeholder(root.WithOpName("input"), DT_FLOAT);
  Output begin = ops::Placeholder(root.WithOpName("begin"), DT_INT64);

  // The C++ node bindings immediately error out when we try construct a bogus
  // slice so we first use a placeholder to construct the Slice and then replace
  // the input.
  Output size_placeholder = ops::Placeholder(root.WithOpName("size"), DT_INT64);
  Output slice =
      ops::Slice(root.WithOpName("slice"), input, begin, size_placeholder);

  Output size =
      ops::Const(root.WithOpName("size"), {{ToInt64(-1)}, {ToInt64(500)}});
  TF_ASSERT_OK(root.graph()->UpdateEdge(size.node(), 0, slice.node(), 2));

  std::unique_ptr<Graph> result;
  TF_ASSERT_OK(IncreaseDynamismForAutoJit(root, &result));

  EXPECT_THAT(result->nodes(),
              Not(Contains(NodeWith(Op("Slice"),
                                    Attr(kXlaCompileTimeConstantInputsAttr)))));
}

TEST(SliceToDynamicSliceRewriteTest, SliceWithSliceInput) {
  Scope root = Scope::NewRootScope()
                   .ExitOnError()
                   .WithAssignedDevice(kDeviceName)
                   .WithXlaCluster("cluster_0");

  Output input = ops::Placeholder(root.WithOpName("input"), DT_FLOAT);
  Output begin = ops::Placeholder(root.WithOpName("begin"), DT_INT32);
  Output size_a = ops::Const(root.WithOpName("size_a"), {-1, 500});
  Output slice = ops::Slice(root.WithOpName("slice"), input, begin, size_a);

  Output size_b = ops::Const(root.WithOpName("size_a"), {-1, 200});
  Output slice_with_slice_input = ops::Slice(
      root.WithOpName("slice_with_slice_input"), slice, begin, size_b);

  std::unique_ptr<Graph> result;
  TF_ASSERT_OK(IncreaseDynamismForAutoJit(root, &result));

  Node* static_shaped_slice = testing::FindNodeByName(
      result.get(),
      "slice_with_slice_input/static_shaped_slice/static_shaped_slice");
  ASSERT_NE(static_shaped_slice, nullptr);
  EXPECT_EQ(static_shaped_slice->output_type(0), DT_FLOAT)
      << "Expected DT_FLOAT, was "
      << DataType_Name(static_shaped_slice->output_type(0));
  EXPECT_THAT(
      static_shaped_slice,
      NodeWith(
          Op("Slice"),
          Inputs(Out(NodeWith(
                     Op("Slice"),
                     Name("slice/static_shaped_slice/static_shaped_slice"))),
                 _, _)));
}

TEST(SliceToDynamicSliceRewriteTest, SliceWithSliceBegin) {
  Scope root = Scope::NewRootScope()
                   .ExitOnError()
                   .WithAssignedDevice(kDeviceName)
                   .WithXlaCluster("cluster_0");

  Output input_float =
      ops::Placeholder(root.WithOpName("input_float"), DT_FLOAT);
  Output input_i64 = ops::Placeholder(root.WithOpName("input_i64"), DT_INT64);

  Output begin_begin =
      ops::Placeholder(root.WithOpName("begin_begin"), DT_INT32);
  Output begin_size = ops::Const(root.WithOpName("begin_size"), {-1});
  Output begin =
      ops::Slice(root.WithOpName("begin"), input_i64, begin_begin, begin_size);

  Output size =
      ops::Const(root.WithOpName("size"), {ToInt64(-1), ToInt64(200)});
  Output slice_with_slice_begin = ops::Slice(
      root.WithOpName("slice_with_slice_begin"), input_float, begin, size);

  std::unique_ptr<Graph> result;
  TF_ASSERT_OK(IncreaseDynamismForAutoJit(root, &result));

  Node* static_shaped_slice = testing::FindNodeByName(
      result.get(),
      "slice_with_slice_begin/static_shaped_slice/static_shaped_slice");
  ASSERT_NE(static_shaped_slice, nullptr);
  EXPECT_EQ(static_shaped_slice->output_type(0), DT_FLOAT)
      << "Expected DT_FLOAT, was "
      << DataType_Name(static_shaped_slice->output_type(0));
  EXPECT_THAT(
      static_shaped_slice,
      NodeWith(
          Op("Slice"),
          Inputs(_,
                 Out(NodeWith(
                     Op("Slice"),
                     Name("begin/static_shaped_slice/static_shaped_slice"))),
                 _)));
}
}  // namespace
}  // namespace tensorflow
