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

#include "tensorflow/contrib/tfprof/tools/tfprof/internal/tfprof_stats.h"

#include <utility>

#include "tensorflow/c/checkpoint_reader.h"
#include "tensorflow/contrib/tfprof/tools/tfprof/internal/tfprof_constants.h"
#include "tensorflow/contrib/tfprof/tools/tfprof/internal/tfprof_options.h"
#include "tensorflow/contrib/tfprof/tools/tfprof/internal/tfprof_utils.h"
#include "tensorflow/contrib/tfprof/tools/tfprof/tfprof_log.pb.h"
#include "tensorflow/contrib/tfprof/tools/tfprof/tfprof_output.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {
namespace tfprof {
class TFProfStatsTest : public ::testing::Test {
 protected:
  TFProfStatsTest() {
    string graph_path = io::JoinPath(
        testing::TensorFlowSrcRoot(),
        "contrib/tfprof/tools/tfprof/internal/testdata/graph.pbtxt");
    std::unique_ptr<tensorflow::GraphDef> graph_pb(new tensorflow::GraphDef());
    TF_CHECK_OK(ReadGraphDefText(Env::Default(), graph_path, graph_pb.get()));

    std::unique_ptr<tensorflow::RunMetadata> run_meta_pb(
        new tensorflow::RunMetadata());
    string run_meta_path =
        io::JoinPath(testing::TensorFlowSrcRoot(),
                     "contrib/tfprof/tools/tfprof/internal/testdata/run_meta");
    TF_CHECK_OK(
        ReadBinaryProto(Env::Default(), run_meta_path, run_meta_pb.get()));

    std::unique_ptr<OpLog> op_log_pb(new OpLog());
    string op_log_path = io::JoinPath(
        testing::TensorFlowSrcRoot(),
        "contrib/tfprof/tools/tfprof/internal/testdata/tfprof_log");
    TF_CHECK_OK(ReadBinaryProto(Env::Default(), op_log_path, op_log_pb.get()));

    string ckpt_path =
        io::JoinPath(testing::TensorFlowSrcRoot(),
                     "contrib/tfprof/tools/tfprof/internal/testdata/ckpt");
    TF_Status* status = TF_NewStatus();
    std::unique_ptr<checkpoint::CheckpointReader> ckpt_reader(
        new checkpoint::CheckpointReader(ckpt_path, status));
    CHECK(TF_GetCode(status) == TF_OK);
    TF_DeleteStatus(status);

    tf_stats_.reset(new TFStats(std::move(graph_pb), std::move(run_meta_pb),
                                std::move(op_log_pb), std::move(ckpt_reader)));
  }

  std::unique_ptr<TFStats> tf_stats_;
};

TEST_F(TFProfStatsTest, CustomOpType) {
  Options opts(3, 0, 0, 0, 0, {".*"}, "name",
               {kTrainableVarType},  // accout_type_regexes
               {".*"}, {""}, {".*"}, {""}, false,
               {"params", "bytes", "micros", "float_ops", "num_hidden_ops"},
               false);
  const TFProfNode& root = tf_stats_->PrintGraph("scope", opts);

  TFProfNode expected;
  CHECK(protobuf::TextFormat::ParseFromString(
      "name: \"_TFProfRoot\"\nexec_micros: 0\nrequested_bytes: "
      "0\ntotal_exec_micros: 0\ntotal_requested_bytes: 1800\ntotal_parameters: "
      "450\nchildren {\n  name: \"DW\"\n  exec_micros: 0\n  requested_bytes: "
      "648\n  parameters: 162\n  total_exec_micros: 0\n  "
      "total_requested_bytes: 648\n  total_parameters: 162\n  device: "
      "\"/job:localhost/replica:0/task:0/cpu:0\"\n  float_ops: 0\n  "
      "total_float_ops: 0\n}\nchildren {\n  name: \"DW2\"\n  exec_micros: 0\n  "
      "requested_bytes: 1152\n  parameters: 288\n  total_exec_micros: 0\n  "
      "total_requested_bytes: 1152\n  total_parameters: 288\n  device: "
      "\"/job:localhost/replica:0/task:0/cpu:0\"\n  float_ops: 0\n  "
      "total_float_ops: 0\n}\nfloat_ops: 0\ntotal_float_ops: 0\n",
      &expected));
  EXPECT_EQ(expected.DebugString(), root.DebugString());
}

TEST_F(TFProfStatsTest, CheckPointOpType) {
  Options opts(
      3, 0, 0, 0, 0, {".*"}, "name", {kCkptVarType},  // accout_type_regexes
      {".*"}, {""}, {".*"}, {""}, false,
      {"params", "bytes", "micros", "float_ops", "num_hidden_ops"}, false);
  const TFProfNode& root = tf_stats_->PrintGraph("scope", opts);

  TFProfNode expected;
  CHECK(protobuf::TextFormat::ParseFromString(
      "name: \"_TFProfRoot\"\nexec_micros: 0\nrequested_bytes: "
      "0\ntotal_exec_micros: 0\ntotal_requested_bytes: 1800\ntotal_parameters: "
      "450\nchildren {\n  name: \"DW\"\n  exec_micros: 0\n  requested_bytes: "
      "648\n  parameters: 162\n  total_exec_micros: 0\n  "
      "total_requested_bytes: 648\n  total_parameters: 162\n  device: "
      "\"/job:localhost/replica:0/task:0/cpu:0\"\n  float_ops: 0\n  "
      "total_float_ops: 0\n}\nchildren {\n  name: \"DW2\"\n  exec_micros: 0\n  "
      "requested_bytes: 1152\n  parameters: 288\n  total_exec_micros: 0\n  "
      "total_requested_bytes: 1152\n  total_parameters: 288\n  device: "
      "\"/job:localhost/replica:0/task:0/cpu:0\"\n  float_ops: 0\n  "
      "total_float_ops: 0\n}\nfloat_ops: 0\ntotal_float_ops: 0\n",
      &expected));
  EXPECT_EQ(expected.DebugString(), root.DebugString());
}

TEST_F(TFProfStatsTest, TestGraph) {
  Options opts(100, 0, 10000, 0, 0, {".*"}, "name", {".*"},
               {"cost.*"},  // start_name_regexes
               {""}, {".*"}, {""}, false,
               {"params", "bytes", "micros", "float_ops", "num_hidden_ops"},
               false);
  const TFProfNode& root = tf_stats_->PrintGraph("graph", opts);

  TFProfNode expected;
  CHECK(protobuf::TextFormat::ParseFromString(
      "name: \"_TFProfRoot\"\nexec_micros: 0\nrequested_bytes: 0\ninputs: "
      "0\ntotal_exec_micros: 0\ntotal_requested_bytes: 0\ntotal_parameters: "
      "0\ntotal_inputs: 0\nfloat_ops: 0\ntotal_float_ops: 0\n",
      &expected));
  EXPECT_EQ(expected.DebugString(), root.DebugString());
}

TEST_F(TFProfStatsTest, TestFloatOps) {
  Options opts(10, 0, 0, 0, 1, {".*"}, "name", {".*"}, {".*"}, {""}, {".*"},
               {""}, false, {"float_ops"}, false);
  const TFProfNode& root = tf_stats_->PrintGraph("scope", opts);

  TFProfNode expected;
  CHECK(protobuf::TextFormat::ParseFromString(
      "name: \"_TFProfRoot\"\nexec_micros: 0\nrequested_bytes: "
      "0\ntotal_exec_micros: 11\ntotal_requested_bytes: "
      "5280\ntotal_parameters: 450\nchildren {\n  name: \"Conv2D\"\n  "
      "exec_micros: 0\n  requested_bytes: 432\n  total_exec_micros: 0\n  "
      "total_requested_bytes: 432\n  total_parameters: 0\n  device: "
      "\"/job:localhost/replica:0/task:0/cpu:0\"\n  float_ops: 5832\n  "
      "total_float_ops: 5832\n}\nchildren {\n  name: \"Conv2D_1\"\n  "
      "exec_micros: 10\n  requested_bytes: 384\n  total_exec_micros: 10\n  "
      "total_requested_bytes: 384\n  total_parameters: 0\n  device: "
      "\"/job:localhost/replica:0/task:0/cpu:0\"\n  float_ops: 4608\n  "
      "total_float_ops: 4608\n}\nfloat_ops: 0\ntotal_float_ops: 10440\n",
      &expected));
  EXPECT_EQ(expected.DebugString(), root.DebugString());
}

TEST_F(TFProfStatsTest, TestAccountShownNameOnly) {
  Options opts(100, 0, 0, 0, 0, {".*"}, "name", {".*"}, {".*"}, {""},
               {"unit_2_1.*DW"},  // show_name_regexes.
               {""}, true,        // account_displayed_op_only.
               {"params"}, false);
  const TFProfNode& root = tf_stats_->PrintGraph("scope", opts);

  TFProfNode expected;
  CHECK(protobuf::TextFormat::ParseFromString(
      "name: \"_TFProfRoot\"\nexec_micros: 0\nrequested_bytes: "
      "0\ntotal_exec_micros: 0\ntotal_requested_bytes: 0\ntotal_parameters: "
      "0\nfloat_ops: 0\ntotal_float_ops: 0\n",
      &expected));
  EXPECT_EQ(expected.DebugString(), root.DebugString());
}

TEST_F(TFProfStatsTest, TestShowTensorValue) {
  Options opts(10, 0, 0, 0, 0, {".*"}, "name", {".*"}, {".*"}, {""},
               {"unit_1_0.*gamma"}, {""}, false,
               {"tensor_value"},  // Show tensor value from checkpoint.
               false);
  const TFProfNode& root = tf_stats_->PrintGraph("scope", opts);
  TFProfNode expected;
  CHECK(protobuf::TextFormat::ParseFromString(
      "name: \"_TFProfRoot\"\nexec_micros: 0\nrequested_bytes: "
      "0\ntotal_exec_micros: 11\ntotal_requested_bytes: "
      "5280\ntotal_parameters: 450\nfloat_ops: 0\ntotal_float_ops: 10440\n",
      &expected));
  EXPECT_EQ(expected.DebugString(), root.DebugString());
}

}  // namespace tfprof
}  // namespace tensorflow
