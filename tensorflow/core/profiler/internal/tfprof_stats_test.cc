/* Copyright 2016 The TensorFlow Authors All Rights Reserved.

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

#include "tensorflow/core/profiler/internal/tfprof_stats.h"

#include <utility>

#include "tensorflow/c/checkpoint_reader.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/profiler/internal/tfprof_constants.h"
#include "tensorflow/core/profiler/internal/tfprof_utils.h"
#include "tensorflow/core/profiler/tfprof_log.pb.h"
#include "tensorflow/core/profiler/tfprof_options.h"
#include "tensorflow/core/profiler/tfprof_output.pb.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {
namespace tfprof {
class TFProfStatsTest : public ::testing::Test {
 protected:
  TFProfStatsTest() {
    string graph_path =
        io::JoinPath(testing::TensorFlowSrcRoot(),
                     "core/profiler/internal/testdata/graph.pbtxt");
    std::unique_ptr<tensorflow::GraphDef> graph_pb(new tensorflow::GraphDef());
    TF_CHECK_OK(
        ReadProtoFile(Env::Default(), graph_path, graph_pb.get(), false));

    std::unique_ptr<tensorflow::RunMetadata> run_meta_pb(
        new tensorflow::RunMetadata());
    string run_meta_path =
        io::JoinPath(testing::TensorFlowSrcRoot(),
                     "core/profiler/internal/testdata/run_meta");
    TF_CHECK_OK(
        ReadProtoFile(Env::Default(), run_meta_path, run_meta_pb.get(), true));

    std::unique_ptr<OpLogProto> op_log_pb(new OpLogProto());
    string op_log_path =
        io::JoinPath(testing::TensorFlowSrcRoot(),
                     "core/profiler/internal/testdata/tfprof_log");
    TF_CHECK_OK(ReadBinaryProto(Env::Default(), op_log_path, op_log_pb.get()));

    string ckpt_path = io::JoinPath(testing::TensorFlowSrcRoot(),
                                    "core/profiler/internal/testdata/ckpt");
    TF_Status* status = TF_NewStatus();
    std::unique_ptr<checkpoint::CheckpointReader> ckpt_reader(
        new checkpoint::CheckpointReader(ckpt_path, status));
    CHECK(TF_GetCode(status) == TF_OK);
    TF_DeleteStatus(status);

    tf_stats_.reset(new TFStats(std::move(graph_pb), std::move(run_meta_pb),
                                std::move(op_log_pb), std::move(ckpt_reader)));
    tf_stats_->BuildAllViews();
  }

  string TestToFromProto(const string& cmd, const Options& opts) {
    string profile_file = io::JoinPath(testing::TmpDir(), "profile");
    tf_stats_->WriteProfile(profile_file);
    TFStats new_stats(profile_file, nullptr);
    new_stats.BuildAllViews();
    return new_stats.ShowGraphNode(cmd, opts).DebugString();
  }

  std::unique_ptr<TFStats> tf_stats_;
};

TEST_F(TFProfStatsTest, CustomOpType) {
  Options opts(3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, "name",
               {kTrainableVarType},  // accout_type_regexes
               {".*"}, {""}, {".*"}, {""}, false,
               {"params", "bytes", "micros", "float_ops"}, "", {});
  const GraphNodeProto& root = tf_stats_->ShowGraphNode("scope", opts);

  GraphNodeProto expected;
  CHECK(protobuf::TextFormat::ParseFromString(
      "name: \"_TFProfRoot\"\ntotal_exec_micros: 13\ntotal_requested_bytes: "
      "2560\ntotal_parameters: 451\nchildren {\n  name: \"DW\"\n  exec_micros: "
      "2\n  requested_bytes: 1280\n  parameters: 162\n  total_exec_micros: 2\n "
      " total_requested_bytes: 1280\n  total_parameters: 162\n  devices: "
      "\"/job:localhost/replica:0/task:0/gpu:0\"\n  cpu_exec_micros: 2\n  "
      "total_cpu_exec_micros: 2\n  run_count: 1\n  total_run_count: 1\n  "
      "total_definition_count: 1\n  peak_bytes: 1280\n  residual_bytes: 1280\n "
      " output_bytes: 1280\n  total_peak_bytes: 1280\n  total_residual_bytes: "
      "1280\n  total_output_bytes: 1280\n}\nchildren {\n  name: \"DW2\"\n  "
      "exec_micros: 11\n  requested_bytes: 1280\n  parameters: 288\n  "
      "total_exec_micros: 11\n  total_requested_bytes: 1280\n  "
      "total_parameters: 288\n  devices: "
      "\"/job:localhost/replica:0/task:0/gpu:0\"\n  cpu_exec_micros: 11\n  "
      "total_cpu_exec_micros: 11\n  run_count: 1\n  total_run_count: 1\n  "
      "total_definition_count: 1\n  peak_bytes: 1280\n  residual_bytes: 1280\n "
      " output_bytes: 1280\n  total_peak_bytes: 1280\n  total_residual_bytes: "
      "1280\n  total_output_bytes: 1280\n}\nchildren {\n  name: \"ScalarW\"\n  "
      "parameters: 1\n  total_parameters: 1\n  total_definition_count: "
      "1\n}\ntotal_cpu_exec_micros: 13\ntotal_run_count: "
      "2\ntotal_definition_count: 3\ntotal_peak_bytes: "
      "2560\ntotal_residual_bytes: 2560\ntotal_output_bytes: 2560\n",
      &expected));
  EXPECT_EQ(expected.DebugString(), root.DebugString());

  EXPECT_EQ(root.DebugString(), TestToFromProto("scope", opts));
}

TEST_F(TFProfStatsTest, CheckPointOpType) {
  Options opts(3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, "name",
               {kCkptVarType},  // accout_type_regexes
               {".*"}, {""}, {".*"}, {""}, false,
               {"params", "bytes", "micros", "float_ops"}, "", {});
  const GraphNodeProto& root = tf_stats_->ShowGraphNode("scope", opts);

  GraphNodeProto expected;
  CHECK(protobuf::TextFormat::ParseFromString(
      "name: \"_TFProfRoot\"\ntotal_exec_micros: 13\ntotal_requested_bytes: "
      "2560\ntotal_parameters: 451\nchildren {\n  name: \"DW\"\n  exec_micros: "
      "2\n  requested_bytes: 1280\n  parameters: 162\n  total_exec_micros: 2\n "
      " total_requested_bytes: 1280\n  total_parameters: 162\n  devices: "
      "\"/job:localhost/replica:0/task:0/gpu:0\"\n  cpu_exec_micros: 2\n  "
      "total_cpu_exec_micros: 2\n  run_count: 1\n  total_run_count: 1\n  "
      "total_definition_count: 1\n  peak_bytes: 1280\n  residual_bytes: 1280\n "
      " output_bytes: 1280\n  total_peak_bytes: 1280\n  total_residual_bytes: "
      "1280\n  total_output_bytes: 1280\n}\nchildren {\n  name: \"DW2\"\n  "
      "exec_micros: 11\n  requested_bytes: 1280\n  parameters: 288\n  "
      "total_exec_micros: 11\n  total_requested_bytes: 1280\n  "
      "total_parameters: 288\n  devices: "
      "\"/job:localhost/replica:0/task:0/gpu:0\"\n  cpu_exec_micros: 11\n  "
      "total_cpu_exec_micros: 11\n  run_count: 1\n  total_run_count: 1\n  "
      "total_definition_count: 1\n  peak_bytes: 1280\n  residual_bytes: 1280\n "
      " output_bytes: 1280\n  total_peak_bytes: 1280\n  total_residual_bytes: "
      "1280\n  total_output_bytes: 1280\n}\nchildren {\n  name: \"ScalarW\"\n  "
      "parameters: 1\n  total_parameters: 1\n  total_definition_count: "
      "1\n}\ntotal_cpu_exec_micros: 13\ntotal_run_count: "
      "2\ntotal_definition_count: 3\ntotal_peak_bytes: "
      "2560\ntotal_residual_bytes: 2560\ntotal_output_bytes: 2560\n",
      &expected));
  EXPECT_EQ(expected.DebugString(), root.DebugString());

  EXPECT_EQ(root.DebugString(), TestToFromProto("scope", opts));
}

TEST_F(TFProfStatsTest, TestGraph) {
  Options opts(100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, "name", {".*"},
               {"DW/Initializer/random_normal/mul"},  // start_name_regexes
               {""}, {".*"}, {""}, false,
               {"params", "bytes", "micros", "float_ops"}, "", {});
  const GraphNodeProto& root = tf_stats_->ShowGraphNode("graph", opts);

  GraphNodeProto expected;
  CHECK(protobuf::TextFormat::ParseFromString(
      "name: \"_TFProfRoot\"\ntotal_exec_micros: 4945\ntotal_requested_bytes: "
      "30464\ntotal_parameters: 451\nchildren {\n  name: "
      "\"DW/Initializer/random_normal/mul\"\n  children {\n    name: "
      "\"DW/Initializer/random_normal/RandomStandardNormal\"\n    children {\n "
      "     name: \"DW/Initializer/random_normal/shape\"\n      "
      "total_definition_count: 1\n    }\n    input_shapes {\n      key: 0\n    "
      "  value {\n        dim {\n          size: 4\n        }\n      }\n    "
      "}\n    total_definition_count: 2\n  }\n  children {\n    name: "
      "\"DW/Initializer/random_normal/stddev\"\n    total_definition_count: "
      "1\n  }\n  input_shapes {\n    key: 0\n    value {\n      dim {\n        "
      "size: 3\n      }\n      dim {\n        size: 3\n      }\n      dim {\n  "
      "      size: 3\n      }\n      dim {\n        size: 6\n      }\n    }\n  "
      "}\n  input_shapes {\n    key: 1\n    value {\n      dim {\n        "
      "size: 1\n      }\n    }\n  }\n  total_definition_count: "
      "4\n}\ntotal_float_ops: 10440\ntotal_accelerator_exec_micros: "
      "404\ntotal_cpu_exec_micros: 4541\ntotal_run_count: "
      "6\ntotal_definition_count: 32\ntotal_peak_bytes: "
      "25856\ntotal_residual_bytes: 3840\ntotal_output_bytes: 4864\n",
      &expected));
  EXPECT_EQ(expected.DebugString(), root.DebugString());

  EXPECT_EQ(root.DebugString(), TestToFromProto("graph", opts));
}

TEST_F(TFProfStatsTest, TestFloatOps) {
  Options opts(10, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1, "name", {".*"}, {".*"},
               {""}, {".*"}, {""}, false, {"float_ops"}, "", {});
  const GraphNodeProto& root = tf_stats_->ShowGraphNode("scope", opts);

  GraphNodeProto expected;
  CHECK(protobuf::TextFormat::ParseFromString(
      "name: \"_TFProfRoot\"\ntotal_exec_micros: 4945\ntotal_requested_bytes: "
      "30464\ntotal_parameters: 451\nchildren {\n  name: \"Conv2D\"\n  "
      "exec_micros: 4292\n  requested_bytes: 18176\n  total_exec_micros: "
      "4292\n  total_requested_bytes: 18176\n  devices: "
      "\"/job:localhost/replica:0/task:0/gpu:0\"\n  float_ops: 5832\n  "
      "total_float_ops: 5832\n  input_shapes {\n    key: 0\n    value {\n      "
      "dim {\n        size: 2\n      }\n      dim {\n        size: 6\n      "
      "}\n      dim {\n        size: 6\n      }\n      dim {\n        size: "
      "3\n      }\n    }\n  }\n  input_shapes {\n    key: 1\n    value {\n     "
      " dim {\n        size: 3\n      }\n      dim {\n        size: 3\n      "
      "}\n      dim {\n        size: 3\n      }\n      dim {\n        size: "
      "6\n      }\n    }\n  }\n  accelerator_exec_micros: 226\n  "
      "cpu_exec_micros: 4066\n  total_accelerator_exec_micros: 226\n  "
      "total_cpu_exec_micros: 4066\n  run_count: 1\n  total_run_count: 1\n  "
      "total_definition_count: 1\n  peak_bytes: 14592\n  residual_bytes: 768\n "
      " output_bytes: 768\n  total_peak_bytes: 14592\n  total_residual_bytes: "
      "768\n  total_output_bytes: 768\n}\nchildren {\n  name: \"Conv2D_1\"\n  "
      "exec_micros: 597\n  requested_bytes: 9728\n  total_exec_micros: 597\n  "
      "total_requested_bytes: 9728\n  devices: "
      "\"/job:localhost/replica:0/task:0/gpu:0\"\n  float_ops: 4608\n  "
      "total_float_ops: 4608\n  input_shapes {\n    key: 0\n    value {\n      "
      "dim {\n        size: 2\n      }\n      dim {\n        size: 3\n      "
      "}\n      dim {\n        size: 3\n      }\n      dim {\n        size: "
      "6\n      }\n    }\n  }\n  input_shapes {\n    key: 1\n    value {\n     "
      " dim {\n        size: 2\n      }\n      dim {\n        size: 2\n      "
      "}\n      dim {\n        size: 6\n      }\n      dim {\n        size: "
      "12\n      }\n    }\n  }\n  accelerator_exec_micros: 178\n  "
      "cpu_exec_micros: 419\n  total_accelerator_exec_micros: 178\n  "
      "total_cpu_exec_micros: 419\n  run_count: 1\n  total_run_count: 1\n  "
      "total_definition_count: 1\n  peak_bytes: 8704\n  residual_bytes: 512\n  "
      "output_bytes: 512\n  total_peak_bytes: 8704\n  total_residual_bytes: "
      "512\n  total_output_bytes: 512\n}\ntotal_float_ops: "
      "10440\ntotal_accelerator_exec_micros: 404\ntotal_cpu_exec_micros: "
      "4541\ntotal_run_count: 6\ntotal_definition_count: 35\ntotal_peak_bytes: "
      "25856\ntotal_residual_bytes: 3840\ntotal_output_bytes: 4864\n",
      &expected));
  EXPECT_EQ(expected.DebugString(), root.DebugString());

  EXPECT_EQ(root.DebugString(), TestToFromProto("scope", opts));
}

TEST_F(TFProfStatsTest, TestAccountShownNameOnly) {
  Options opts(100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, "name", {".*"}, {".*"},
               {""}, {"Conv2D_1"},  // show_name_regexes.
               {""}, true,          // account_displayed_op_only.
               {"params"}, "", {});
  const GraphNodeProto& root = tf_stats_->ShowGraphNode("scope", opts);

  GraphNodeProto expected;
  CHECK(protobuf::TextFormat::ParseFromString(
      "name: \"_TFProfRoot\"\ntotal_exec_micros: 597\ntotal_requested_bytes: "
      "9728\nchildren {\n  name: \"Conv2D_1\"\n  exec_micros: 597\n  "
      "requested_bytes: 9728\n  total_exec_micros: 597\n  "
      "total_requested_bytes: 9728\n  devices: "
      "\"/job:localhost/replica:0/task:0/gpu:0\"\n  float_ops: 4608\n  "
      "total_float_ops: 4608\n  input_shapes {\n    key: 0\n    value {\n      "
      "dim {\n        size: 2\n      }\n      dim {\n        size: 3\n      "
      "}\n      dim {\n        size: 3\n      }\n      dim {\n        size: "
      "6\n      }\n    }\n  }\n  input_shapes {\n    key: 1\n    value {\n     "
      " dim {\n        size: 2\n      }\n      dim {\n        size: 2\n      "
      "}\n      dim {\n        size: 6\n      }\n      dim {\n        size: "
      "12\n      }\n    }\n  }\n  accelerator_exec_micros: 178\n  "
      "cpu_exec_micros: 419\n  total_accelerator_exec_micros: 178\n  "
      "total_cpu_exec_micros: 419\n  run_count: 1\n  total_run_count: 1\n  "
      "total_definition_count: 1\n  peak_bytes: 8704\n  residual_bytes: 512\n  "
      "output_bytes: 512\n  total_peak_bytes: 8704\n  total_residual_bytes: "
      "512\n  total_output_bytes: 512\n}\ntotal_float_ops: "
      "4608\ntotal_accelerator_exec_micros: 178\ntotal_cpu_exec_micros: "
      "419\ntotal_run_count: 1\ntotal_definition_count: 2\ntotal_peak_bytes: "
      "8704\ntotal_residual_bytes: 512\ntotal_output_bytes: 512\n",
      &expected));
  EXPECT_EQ(expected.DebugString(), root.DebugString());

  EXPECT_EQ(root.DebugString(), TestToFromProto("scope", opts));
}

TEST_F(TFProfStatsTest, TestShowTensorValue) {
  Options opts(10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, "name", {".*"}, {".*"},
               {""}, {"DW"}, {""}, false,
               {"tensor_value"},  // Show tensor value from checkpoint.
               "", {});
  const GraphNodeProto& root = tf_stats_->ShowGraphNode("scope", opts);
  GraphNodeProto expected;
  CHECK(protobuf::TextFormat::ParseFromString(
      "name: \"_TFProfRoot\"\ntotal_exec_micros: 4945\ntotal_requested_bytes: "
      "30464\ntotal_parameters: 451\nchildren {\n  name: \"DW\"\n  "
      "exec_micros: 2\n  requested_bytes: 1280\n  parameters: 162\n  "
      "total_exec_micros: 2\n  total_requested_bytes: 1280\n  "
      "total_parameters: 162\n  devices: "
      "\"/job:localhost/replica:0/task:0/gpu:0\"\n  tensor_value {\n    dtype: "
      "DT_FLOAT\n    value_double: -0.000534315\n    value_double: "
      "-0.00089602\n    value_double: -0.000417239\n    value_double: "
      "0.00041444\n    value_double: 0.000780691\n    value_double: "
      "-0.000559057\n    value_double: -0.000234623\n    value_double: "
      "0.00013393\n    value_double: -0.00187574\n    value_double: "
      "0.000785666\n    value_double: 0.000673294\n    value_double: "
      "0.000653368\n    value_double: 0.000924489\n    value_double: "
      "-0.000318373\n    value_double: -0.000385202\n    value_double: "
      "-7.92661e-05\n    value_double: 2.70287e-05\n    value_double: "
      "0.00152302\n    value_double: 8.04435e-05\n    value_double: "
      "-0.00058102\n    value_double: 0.000244291\n    value_double: "
      "-0.000438045\n    value_double: -0.000110199\n    value_double: "
      "0.000731663\n    value_double: -0.0012326\n    value_double: "
      "0.00064065\n    value_double: -0.00135203\n    value_double: "
      "-6.42784e-05\n    value_double: -0.0011857\n    value_double: "
      "-0.000487383\n    value_double: 3.41493e-05\n    value_double: "
      "-0.00158447\n    value_double: 0.00168448\n    value_double: "
      "0.00160946\n    value_double: -0.000600483\n    value_double: "
      "0.000650259\n    value_double: -0.00109938\n    value_double: "
      "-0.000842166\n    value_double: -0.0022673\n    value_double: "
      "-0.00101941\n    value_double: -0.0011169\n    value_double: "
      "-0.0013557\n    value_double: -1.46354e-05\n    value_double: "
      "-1.05487e-05\n    value_double: -0.00092014\n    value_double: "
      "0.00272874\n    value_double: 5.13942e-05\n    value_double: "
      "-0.00223472\n    value_double: -0.000250875\n    value_double: "
      "-0.00180747\n    value_double: -0.00234714\n    value_double: "
      "-0.00113523\n    value_double: -0.00112635\n    value_double: "
      "-0.000843118\n    value_double: -6.84256e-05\n    value_double: "
      "0.000243336\n    value_double: 0.00119151\n    value_double: "
      "0.00131022\n    value_double: 0.000768038\n    value_double: "
      "-8.90095e-05\n    value_double: -0.000626427\n    value_double: "
      "-7.0617e-05\n    value_double: -0.0021988\n    value_double: "
      "-0.00221544\n    value_double: -0.000393118\n    value_double: "
      "0.000159464\n    value_double: -0.000874746\n    value_double: "
      "-0.00131239\n    value_double: -0.00135747\n    value_double: "
      "-0.00179753\n    value_double: -0.00101005\n    value_double: "
      "-0.000107518\n    value_double: -0.000616882\n    value_double: "
      "-0.000360923\n    value_double: -0.00026896\n    value_double: "
      "-0.000142548\n    value_double: 0.000577227\n    value_double: "
      "0.000536027\n    value_double: 0.00126907\n    value_double: "
      "-0.00122712\n    value_double: -3.60499e-05\n    value_double: "
      "0.000151026\n    value_double: 0.00107658\n    value_double: "
      "0.00116475\n    value_double: -0.00145312\n    value_double: "
      "0.000233326\n    value_double: -0.00020198\n    value_double: "
      "0.00179029\n    value_double: 0.00150048\n    value_double: "
      "-0.000884775\n    value_double: 0.000409188\n    value_double: "
      "2.97176e-05\n    value_double: -0.000506118\n    value_double: "
      "-2.33992e-05\n    value_double: -0.00037212\n    value_double: "
      "0.000862773\n    value_double: 0.00174046\n    value_double: "
      "-0.000240207\n    value_double: 0.000663976\n    value_double: "
      "-0.00134747\n    value_double: 0.00115585\n    value_double: "
      "0.000555869\n    value_double: 0.00176722\n    value_double: "
      "-0.000518409\n    value_double: 0.00101051\n    value_double: "
      "0.000129399\n    value_double: -0.000916389\n    value_double: "
      "-0.00137693\n    value_double: -0.00152412\n    value_double: "
      "7.32515e-05\n    value_double: -0.000190811\n    value_double: "
      "-0.000158692\n    value_double: -5.7791e-05\n    value_double: "
      "0.000671785\n    value_double: -0.00152924\n    value_double: "
      "0.00117314\n    value_double: -0.000384202\n    value_double: "
      "0.00176709\n    value_double: -0.000181703\n    value_double: "
      "-0.000460994\n    value_double: 0.000643716\n    value_double: "
      "4.76719e-05\n    value_double: -0.00101037\n    value_double: "
      "0.00159621\n    value_double: 0.00186758\n    value_double: "
      "0.00100001\n    value_double: -0.00121831\n    value_double: "
      "0.00132231\n    value_double: 0.0013511\n    value_double: 0.00106659\n "
      "   value_double: 0.00018091\n    value_double: 0.00155925\n    "
      "value_double: 4.26087e-05\n    value_double: 0.000243264\n    "
      "value_double: -0.0017202\n    value_double: -0.000218897\n    "
      "value_double: 0.00118693\n    value_double: 0.00258909\n    "
      "value_double: 0.000641913\n    value_double: -0.0013211\n    "
      "value_double: -0.00171943\n    value_double: 0.00089151\n    "
      "value_double: -0.00114969\n    value_double: -0.000196331\n    "
      "value_double: 0.00109994\n    value_double: 0.000302616\n    "
      "value_double: 0.000675812\n    value_double: 0.00112222\n    "
      "value_double: 0.000516456\n    value_double: 0.00133357\n    "
      "value_double: 0.000298491\n    value_double: 0.00145934\n    "
      "value_double: -0.00159102\n    value_double: -0.000819061\n    "
      "value_double: 0.000120583\n    value_double: 0.0006108\n    "
      "value_double: 0.00124132\n    value_double: 0.000764859\n    "
      "value_double: 0.000374641\n    value_double: -0.00149603\n    "
      "value_double: -0.000317367\n    value_double: -0.000417829\n  }\n  "
      "cpu_exec_micros: 2\n  total_cpu_exec_micros: 2\n  run_count: 1\n  "
      "total_run_count: 1\n  total_definition_count: 10\n  peak_bytes: 1280\n  "
      "residual_bytes: 1280\n  output_bytes: 1280\n  total_peak_bytes: 1280\n  "
      "total_residual_bytes: 1280\n  total_output_bytes: "
      "1280\n}\ntotal_float_ops: 10440\ntotal_accelerator_exec_micros: "
      "404\ntotal_cpu_exec_micros: 4541\ntotal_run_count: "
      "6\ntotal_definition_count: 35\ntotal_peak_bytes: "
      "25856\ntotal_residual_bytes: 3840\ntotal_output_bytes: 4864\n",
      &expected));
  EXPECT_EQ(expected.DebugString(), root.DebugString());
}

}  // namespace tfprof
}  // namespace tensorflow
