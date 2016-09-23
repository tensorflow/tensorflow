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
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {
namespace tfprof {
class TFProfShowTest : public ::testing::Test {
 protected:
  TFProfShowTest() {
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

TEST_F(TFProfShowTest, DumpScopeMode) {
  string dump_file = io::JoinPath(testing::TmpDir(), "dump");
  Options opts(5, 0, 0, 0, 0, {".*"}, "name",
               {"Variable"},  // accout_type_regexes
               {".*"}, {""}, {".*"}, {""}, false,
               {"params", "bytes", "micros", "float_ops", "num_hidden_ops"},
               false, dump_file);
  tf_stats_->PrintGraph("scope", opts);

  string dump_str;
  TF_CHECK_OK(ReadFileToString(Env::Default(), dump_file, &dump_str));
  EXPECT_EQ(
      "_TFProfRoot (--/450 params, --/0 flops, --/1.80KB, --/0us)\n  DW "
      "(3x3x3x6, 162/162 params, 0/0 flops, 648B/648B, 0us/0us)\n  DW2 "
      "(2x2x6x12, 288/288 params, 0/0 flops, 1.15KB/1.15KB, 0us/0us)\n",
      dump_str);
}

}  // namespace tfprof
}  // namespace tensorflow
