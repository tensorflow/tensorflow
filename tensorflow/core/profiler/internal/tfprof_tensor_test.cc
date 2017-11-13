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

#include "tensorflow/c/checkpoint_reader.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/profiler/internal/tfprof_options.h"
#include "tensorflow/core/profiler/internal/tfprof_stats.h"
#include "tensorflow/core/profiler/internal/tfprof_utils.h"
#include "tensorflow/core/profiler/tfprof_log.pb.h"
#include "tensorflow/core/profiler/tfprof_output.pb.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {
namespace tfprof {
class TFProfTensorTest : public ::testing::Test {
 protected:
  TFProfTensorTest() {
    string graph_path =
        io::JoinPath(testing::TensorFlowSrcRoot(),
                     "core/profiler/internal/testdata/graph.pbtxt");
    std::unique_ptr<tensorflow::GraphDef> graph_pb(new tensorflow::GraphDef());
    TF_CHECK_OK(
        ReadProtoFile(Env::Default(), graph_path, graph_pb.get(), false));

    std::unique_ptr<tensorflow::RunMetadata> run_meta_pb;
    std::unique_ptr<OpLogProto> op_log_pb;

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

  std::unique_ptr<TFStats> tf_stats_;
};

TEST_F(TFProfTensorTest, Basics) {
  Options opts(3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, "name", {"VariableV2"},
               {".*"}, {""}, {".*"}, {""}, false,
               {"tensor_value"},  // show the tensor value.
               "", {});
  const GraphNodeProto& root = tf_stats_->ShowGraphNode("scope", opts);

  GraphNodeProto expected;
  EXPECT_EQ(root.children(0).name(), "DW");
  EXPECT_GT(root.children(0).tensor_value().value_double_size(), 10);
  EXPECT_EQ(root.children(1).name(), "DW2");
  EXPECT_GT(root.children(1).tensor_value().value_double_size(), 10);
  EXPECT_EQ(root.children(2).name(), "ScalarW");
  EXPECT_EQ(root.children(2).tensor_value().value_double_size(), 1);
}

}  // namespace tfprof
}  // namespace tensorflow
