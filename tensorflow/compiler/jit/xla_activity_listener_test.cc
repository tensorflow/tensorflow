/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/xla_activity_listener.h"

#include <cstdlib>

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/list_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/core/common_runtime/direct_session.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

class TestListener : public XlaActivityListener {
 public:
  Status Listen(
      const XlaAutoClusteringActivity& auto_clustering_activity) override {
    auto_clustering_activity_ = auto_clustering_activity;
    return OkStatus();
  }

  Status Listen(
      const XlaJitCompilationActivity& jit_compilation_activity) override {
    jit_compilation_activity_ = jit_compilation_activity;
    return OkStatus();
  }

  Status Listen(const XlaOptimizationRemark& optimization_remark) override {
    return OkStatus();
  }

  ~TestListener() override {}

  const XlaAutoClusteringActivity& auto_clustering_activity() const {
    return auto_clustering_activity_;
  }
  const XlaJitCompilationActivity& jit_compilation_activity() const {
    return jit_compilation_activity_;
  }

 private:
  XlaAutoClusteringActivity auto_clustering_activity_;
  XlaJitCompilationActivity jit_compilation_activity_;
};

class XlaActivityListenerTest : public ::testing::Test {
 protected:
  XlaActivityListenerTest() {
    auto listener = absl::make_unique<TestListener>();
    listener_ = listener.get();
    RegisterXlaActivityListener(std::move(listener));
  }

  TestListener* listener() const { return listener_; }

 private:
  TestListener* listener_;
};

GraphDef CreateGraphDef() {
  Scope root = Scope::NewRootScope().ExitOnError().WithAssignedDevice(
      "/job:localhost/replica:0/task:0/device:CPU:0");
  Output a = ops::Placeholder(root.WithOpName("A"), DT_FLOAT);
  for (int i = 0; i < 5; i++) {
    a = ops::MatMul(root.WithOpName(absl::StrCat("matmul_", i)), a, a);
    a = ops::Add(root.WithOpName(absl::StrCat("add_", i)), a, a);
  }

  GraphDef graph_def;
  root.graph()->ToGraphDef(&graph_def);
  return graph_def;
}

TEST_F(XlaActivityListenerTest, Test) {
  GraphDef graph_def = CreateGraphDef();
  SessionOptions options;
  options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_global_jit_level(OptimizerOptions::ON_2);
  std::unique_ptr<Session> session(NewSession(options));

  TF_ASSERT_OK(session->Create(graph_def));

  std::vector<std::string> output_names = {std::string("add_4:0")};

  Tensor tensor_2x2(DT_FLOAT, TensorShape({2, 2}));
  for (int i = 0; i < 4; i++) {
    tensor_2x2.matrix<float>()(i / 2, i % 2) = 5 * i;
  }

  Tensor tensor_3x3(DT_FLOAT, TensorShape({3, 3}));
  for (int i = 0; i < 9; i++) {
    tensor_3x3.matrix<float>()(i / 3, i % 3) = 5 * i;
  }

  std::vector<std::pair<string, Tensor>> inputs_2x2 = {{"A", tensor_2x2}};

  std::vector<Tensor> outputs;
  TF_ASSERT_OK(session->Run(inputs_2x2, output_names, /*target_node_names=*/{},
                            &outputs));

  XlaAutoClusteringActivity expected_auto_clustering_activity;
  protobuf::TextFormat::ParseFromString(
      R"(global_jit_level: ON_2
cpu_global_jit_enabled: true
summary {
  unclustered_node_count: 4
  clustered_node_count: 14
  clusters {
    name: "cluster_0"
    size: 14
    op_histogram {
      op: "Add"
      count: 1
    }
    op_histogram {
      op: "Const"
      count: 4
    }
    op_histogram {
      op: "MatMul"
      count: 5
    }
    op_histogram {
      op: "Mul"
      count: 4
    }
  }
  unclustered_op_histogram {
    op: "NoOp"
    count: 2
  }
  unclustered_op_histogram {
    op: "_Arg"
    count: 1
  }
  unclustered_op_histogram {
    op: "_Retval"
    count: 1
  }
}
)",
      &expected_auto_clustering_activity);
  EXPECT_EQ(listener()->auto_clustering_activity().DebugString(),
            expected_auto_clustering_activity.DebugString());

  EXPECT_EQ(listener()->jit_compilation_activity().cluster_name(), "cluster_0");
  EXPECT_EQ(listener()->jit_compilation_activity().compile_count(), 1);

  int64_t first_compile_time =
      listener()->jit_compilation_activity().compile_time_us();
  EXPECT_GT(first_compile_time, 0);
  EXPECT_EQ(listener()->jit_compilation_activity().cumulative_compile_time_us(),
            first_compile_time);

  std::vector<std::pair<string, Tensor>> inputs_3x3 = {{"A", tensor_3x3}};

  outputs.clear();
  for (int i = 0; i < 3; i++) {
    TF_ASSERT_OK(session->Run(inputs_3x3, output_names,
                              /*target_node_names=*/{}, &outputs));
  }

  EXPECT_EQ(listener()->jit_compilation_activity().cluster_name(), "cluster_0");
  EXPECT_EQ(listener()->jit_compilation_activity().compile_count(), 2);

  EXPECT_GT(listener()->jit_compilation_activity().compile_time_us(), 0);
  EXPECT_EQ(listener()->jit_compilation_activity().cumulative_compile_time_us(),
            first_compile_time +
                listener()->jit_compilation_activity().compile_time_us());
}

}  // namespace
}  // namespace tensorflow

int main(int argc, char** argv) {
  tensorflow::GetMarkForCompilationPassFlags()->tf_xla_cpu_global_jit = true;
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
