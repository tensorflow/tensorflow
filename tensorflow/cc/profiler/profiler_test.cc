/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/platform/test.h"

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/profiler/profiler.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace tfprof {

class ProfilerTest : public ::testing::Test {
 protected:
  ProfilerTest() {}
};

GraphDef CreateGraphDef() {
  Scope root = Scope::NewRootScope();

  auto a = ops::Const<float>(root, {{3, 2}, {-1, 0}});

  auto x = ops::Const(root.WithOpName("x"), {{1.f}, {1.f}});

  auto y = ops::MatMul(root.WithOpName("y"), a, x);

  auto y2 = ops::Square(root, y);

  auto y2_sum = ops::Sum(root, y2, 0);

  auto y_norm = ops::Sqrt(root, y2_sum);

  auto y_div = ops::Div(root.WithOpName("y_normalized"), y, y_norm);

  GraphDef def;
  TF_CHECK_OK(root.ToGraphDef(&def));

  return def;
}

Options Default() {
  Options opts(1000,       /* max_depth */
               0,          /* min_bytes */
               0,          /* min_peak_bytes */
               0,          /* min_residual_bytes */
               0,          /* min_output_bytes */
               0,          /* min_micros */
               0,          /* min_accelerator_micros */
               0,          /* min_cpu_micros */
               0,          /* min_params */
               0,          /* min_float_ops */
               0,          /* min_occurrence */
               0,          /* step */
               "name",     /* order_by */
               {".*"},     /* account_type_regexes */
               {".*"},     /* start_name_regexes */
               {},         /* trim_name_regexes */
               {".*"}, {}, /* hide_name_regexes */
               false,      /* account_displayed_op_only */
               {"micros"}, /* select */
               {"none"},   /* output_type */
               {});
  return opts;
}

template <typename T>
const T* ExtractNode(const T& pb, const string& name) {
  if (pb.name() == name) {
    return &pb;
  }
  for (const T& c : pb.children()) {
    const T* ret = ExtractNode(c, name);
    if (ret) return ret;
  }
  return nullptr;
}

TEST_F(ProfilerTest, Basics) {
  SessionOptions options;
  options.config.set_allow_soft_placement(true);
  std::unique_ptr<Session> session(NewSession(options));
  GraphDef def = CreateGraphDef();
  if (options.target.empty()) {
    graph::SetDefaultDevice("/gpu:0", &def);
  }

  TF_CHECK_OK(session->Create(def));

  Tensor x(DT_FLOAT, TensorShape({2, 1}));
  auto x_flat = x.flat<float>();
  x_flat.setRandom();
  Eigen::Tensor<float, 0, Eigen::RowMajor> inv_norm =
      x_flat.square().sum().sqrt().inverse();
  x_flat = x_flat * inv_norm();

  std::vector<Tensor> outputs;
  RunOptions run_options;
  run_options.set_trace_level(RunOptions::FULL_TRACE);
  RunMetadata run_metadata;
  outputs.clear();

  Profiler profiler(def);
  for (int i = 0; i < 2; ++i) {
    TF_CHECK_OK(session->Run(run_options, {{"x", x}}, {"y:0", "y_normalized:0"},
                             {}, &outputs, &run_metadata));
    profiler.AddStep(i, run_metadata);
    CHECK_EQ(size_t{2}, outputs.size());
  }

  std::vector<DeviceAttributes> resp;
  TF_CHECK_OK(session->ListDevices(&resp));
  bool has_gpu = false;
  for (const auto& dev : resp) {
    if (dev.device_type() == "GPU") {
      has_gpu = true;
    }
  }

  GraphNodeProto ret = profiler.ProfileNameScope(Default());
  const GraphNodeProto* matmul = ExtractNode(ret, "y");
  EXPECT_TRUE(matmul);
  EXPECT_GT(matmul->exec_micros(), 0);
  if (has_gpu) {
    EXPECT_GT(matmul->accelerator_exec_micros(), 0);
  } else {
    EXPECT_EQ(matmul->accelerator_exec_micros(), 0);
  }
  const GraphNodeProto* square = ExtractNode(ret, "Square");
  EXPECT_TRUE(square);
  EXPECT_GT(square->exec_micros(), 0);
  if (has_gpu) {
    EXPECT_GT(square->accelerator_exec_micros(), 0);
  } else {
    EXPECT_EQ(square->accelerator_exec_micros(), 0);
  }

  Options opts2 = Default();
  opts2.output_type = "timeline";
  string timeline_file = io::JoinPath(testing::TmpDir(), "timeline");
  opts2.output_options["outfile"] = timeline_file;
  GraphNodeProto ret2 = profiler.ProfileGraph(opts2);
  string s;
  TF_CHECK_OK(ReadFileToString(Env::Default(), timeline_file + "_0", &s));
  EXPECT_TRUE(s.find("Square") != s.npos);

  MultiGraphNodeProto ret3 = profiler.ProfileOperations(Default());
  const MultiGraphNodeProto* matmul2 = ExtractNode(ret3, "MatMul");
  EXPECT_TRUE(matmul2);
  EXPECT_GT(matmul2->exec_micros(), 0);
  if (has_gpu) {
    EXPECT_GT(matmul2->accelerator_exec_micros(), 0);
  } else {
    EXPECT_EQ(matmul2->accelerator_exec_micros(), 0);
  }

  TF_CHECK_OK(session->Close());
}

}  // namespace tfprof
}  // namespace tensorflow
