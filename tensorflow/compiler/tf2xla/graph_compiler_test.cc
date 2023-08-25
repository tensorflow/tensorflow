/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/graph_compiler.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/compiler/tf2xla/graph_compiler_util.h"
#include "tensorflow/compiler/tf2xla/tf2xla.pb.h"
#include "tensorflow/compiler/tf2xla/xla_compilation_device.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/monitoring/cell_reader.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace {

using ::tensorflow::monitoring::testing::CellReader;

constexpr char kOpCompilationFailureStreamz[] =
    "/tensorflow/core/tf2xla/graph_compilation_failed_op_count";

class DummyOp : public XlaOpKernel {
 public:
  explicit DummyOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}
  void Compile(XlaOpKernelContext* ctx) override {}
};

REGISTER_KERNEL_BUILDER(Name("NoOp").Device(DEVICE_DEFAULT), DummyOp);
REGISTER_KERNEL_BUILDER(Name("NoOp").Device("XLA_TPU_JIT"), DummyOp);
REGISTER_KERNEL_BUILDER(Name("NoOp").Device("XLA_CPU_JIT"), DummyOp);

class MockAlwaysFailsOp : public XlaOpKernel {
 public:
  explicit MockAlwaysFailsOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}
  void Compile(XlaOpKernelContext* ctx) override {
    ctx->CtxFailure(__FILE__, __LINE__, errors::InvalidArgument("MockBroken"));
  }
};

REGISTER_OP("MockAlwaysFails")
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
A test only Op that always fails to compile.
)doc");

REGISTER_KERNEL_BUILDER(Name("MockAlwaysFails").Device(DEVICE_DEFAULT),
                        MockAlwaysFailsOp);
REGISTER_KERNEL_BUILDER(Name("MockAlwaysFails").Device("XLA_CPU_JIT"),
                        MockAlwaysFailsOp);
REGISTER_KERNEL_BUILDER(Name("MockAlwaysFails").Device("XLA_TPU_JIT"),
                        MockAlwaysFailsOp);
REGISTER_XLA_OP(Name("MockAlwaysFails").CompilationOnly(), MockAlwaysFailsOp);

class GraphCompilerTest : public ::testing::Test {
 public:
  void SetUp() override {
    device_ = new tensorflow::XlaCompilationDevice(
        tensorflow::SessionOptions(), tensorflow::DeviceType("XLA_TPU_JIT"));
    device_mgr_ = std::make_unique<StaticDeviceMgr>(absl::WrapUnique(device_));
  }

  Status RunGraphCompiler(Graph& graph) {
    ProcessFunctionLibraryRuntime runtime(
        device_mgr_.get(), Env::Default(), nullptr, TF_GRAPH_DEF_VERSION,
        &graph.flib_def(), OptimizerOptions());

    xla::XlaBuilder builder("test_builder");
    XlaCompiler::Options options;
    options.device_type = "XLA_TPU_JIT";

    XlaCompiler xla_compiler(options);

    // Resource cleanup is messy, see the LINT.ThenChange for comments.
    // LINT.IfChange
    XlaContext* xla_context = new XlaContext(&xla_compiler, &builder, &graph);
    core::ScopedUnref context_unref(xla_context);
    xla_context->Ref();

    auto step_container =
        std::make_unique<ScopedStepContainer>(0, [this](const string& name) {
          Status status = this->device_->resource_manager()->Cleanup(name);
        });
    auto container_status = step_container->Create(
        device_->resource_manager(), XlaContext::kXlaContextResourceName,
        xla_context);

    GraphCompiler graph_compiler(
        device_, &graph, runtime.GetFLR(device_->name()), step_container.get());

    return graph_compiler.Compile();
    // LINT.ThenChange(//tensorflow/compiler/tf2xla/xla_compiler.cc:ExecuteGraph)
  }

 protected:
  XlaCompilationDevice* device_;  // Owned by device_mgr_
  std::unique_ptr<StaticDeviceMgr> device_mgr_;
};

TEST_F(GraphCompilerTest, CompilesGraph) {
  Graph graph(OpRegistry::Global());

  EXPECT_TRUE(RunGraphCompiler(graph).ok());
}

TEST_F(GraphCompilerTest, RecordsStreamzFailedCompilationNode) {
  Graph graph(OpRegistry::Global());
  Node* mock_fail;
  ASSERT_TRUE(NodeBuilder("mock_fail", "MockAlwaysFails")
                  .Finalize(&graph, &mock_fail)
                  .ok());
  graph.AddControlEdge(graph.source_node(), mock_fail);
  graph.AddControlEdge(mock_fail, graph.sink_node());

  CellReader<int64_t> op_reader(kOpCompilationFailureStreamz);

  EXPECT_FALSE(RunGraphCompiler(graph).ok());

  EXPECT_EQ(op_reader.Delta("MockAlwaysFails"), 1);
}

}  // namespace
}  // namespace tensorflow
