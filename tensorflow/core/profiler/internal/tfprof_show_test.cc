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
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/profiler/internal/tfprof_constants.h"
#include "tensorflow/core/profiler/internal/tfprof_options.h"
#include "tensorflow/core/profiler/internal/tfprof_utils.h"
#include "tensorflow/core/profiler/tfprof_log.pb.h"
#include "tensorflow/core/profiler/tfprof_output.pb.h"

namespace tensorflow {
namespace tfprof {
class TFProfShowTest : public ::testing::Test {
 protected:
  TFProfShowTest() {
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

    std::unique_ptr<OpLog> op_log_pb(new OpLog());
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

  std::unique_ptr<TFStats> tf_stats_;
};

TEST_F(TFProfShowTest, DumpScopeMode) {
  string dump_file = io::JoinPath(testing::TmpDir(), "dump");
  Options opts(5, 0, 0, 0, 0, 0, -1, "name",
               {"VariableV2"},  // accout_type_regexes
               {".*"}, {""}, {".*"}, {""}, false,
               {"params", "bytes", "micros", "float_ops"}, "file",
               {{"outfile", dump_file}});
  tf_stats_->ShowGraphNode("scope", opts);

  string dump_str;
  TF_CHECK_OK(ReadFileToString(Env::Default(), dump_file, &dump_str));
  EXPECT_EQ(
      "node name | # parameters | # float_ops | output bytes | total execution "
      "time | accelerator execution time | cpu execution time\n_TFProfRoot "
      "(--/370 params, --/0 flops, --/1.48KB, --/5us, --/0us, --/5us)\n  "
      "conv2d (--/140 params, --/0 flops, --/560B, --/2us, --/0us, --/2us)\n   "
      " conv2d/bias (5, 5/5 params, 0/0 flops, 20B/20B, 1us/1us, 0us/0us, "
      "1us/1us)\n    conv2d/kernel (3x3x3x5, 135/135 params, 0/0 flops, "
      "540B/540B, 1us/1us, 0us/0us, 1us/1us)\n  conv2d_1 (--/230 params, --/0 "
      "flops, --/920B, --/3us, --/0us, --/3us)\n    conv2d_1/bias (5, 5/5 "
      "params, 0/0 flops, 20B/20B, 1us/1us, 0us/0us, 1us/1us)\n    "
      "conv2d_1/kernel (3x3x5x5, 225/225 params, 0/0 flops, 900B/900B, "
      "2us/2us, 0us/0us, 2us/2us)\n",
      dump_str);
}

TEST_F(TFProfShowTest, DumpAcceleratorAndCPUMicros) {
  string dump_file = io::JoinPath(testing::TmpDir(), "dump");
  Options opts(
      5, 0, 0, 0, 0, 0, -1, "cpu_micros", {".*"},  // accout_type_regexes
      {".*"}, {""}, {".*"}, {""}, false, {"accelerator_micros", "cpu_micros"},
      "file", {{"outfile", dump_file}});
  tf_stats_->ShowGraphNode("scope", opts);

  string dump_str;
  TF_CHECK_OK(ReadFileToString(Env::Default(), dump_file, &dump_str));
  EXPECT_EQ(
      "node name | accelerator execution time | cpu execution "
      "time\n_TFProfRoot (--/0us, --/97us)\n  conv2d (0us/0us, 0us/76us)\n    "
      "conv2d/convolution (0us/0us, 60us/60us)\n      conv2d/convolution/Shape "
      "(0us/0us, 0us/0us)\n      conv2d/convolution/dilation_rate (0us/0us, "
      "0us/0us)\n    conv2d/BiasAdd (0us/0us, 12us/12us)\n    conv2d/bias "
      "(0us/0us, 1us/2us)\n      conv2d/bias/Assign (0us/0us, 0us/0us)\n      "
      "conv2d/bias/Initializer (0us/0us, 0us/0us)\n        "
      "conv2d/bias/Initializer/Const (0us/0us, 0us/0us)\n      "
      "conv2d/bias/read (0us/0us, 1us/1us)\n    conv2d/kernel (0us/0us, "
      "1us/2us)\n      conv2d/kernel/Assign (0us/0us, 0us/0us)\n      "
      "conv2d/kernel/Initializer (0us/0us, 0us/0us)\n        "
      "conv2d/kernel/Initializer/random_uniform (0us/0us, 0us/0us)\n      "
      "conv2d/kernel/read (0us/0us, 1us/1us)\n  conv2d_2 (0us/0us, 0us/15us)\n "
      "   conv2d_2/convolution (0us/0us, 13us/13us)\n      "
      "conv2d_2/convolution/Shape (0us/0us, 0us/0us)\n      "
      "conv2d_2/convolution/dilation_rate (0us/0us, 0us/0us)\n    "
      "conv2d_2/BiasAdd (0us/0us, 2us/2us)\n  conv2d_1 (0us/0us, 0us/5us)\n    "
      "conv2d_1/bias (0us/0us, 1us/2us)\n      conv2d_1/bias/Assign (0us/0us, "
      "0us/0us)\n      conv2d_1/bias/Initializer (0us/0us, 0us/0us)\n        "
      "conv2d_1/bias/Initializer/Const (0us/0us, 0us/0us)\n      "
      "conv2d_1/bias/read (0us/0us, 1us/1us)\n    conv2d_1/kernel (0us/0us, "
      "2us/3us)\n      conv2d_1/kernel/Assign (0us/0us, 0us/0us)\n      "
      "conv2d_1/kernel/Initializer (0us/0us, 0us/0us)\n        "
      "conv2d_1/kernel/Initializer/random_uniform (0us/0us, 0us/0us)\n      "
      "conv2d_1/kernel/read (0us/0us, 1us/1us)\n  init (0us/0us, 0us/0us)\n  "
      "save (0us/0us, 0us/0us)\n    save/Assign (0us/0us, 0us/0us)\n    "
      "save/Assign_1 (0us/0us, 0us/0us)\n    save/Assign_2 (0us/0us, "
      "0us/0us)\n    save/Assign_3 (0us/0us, 0us/0us)\n    save/Const "
      "(0us/0us, 0us/0us)\n    save/RestoreV2 (0us/0us, 0us/0us)\n      "
      "save/RestoreV2/shape_and_slices (0us/0us, 0us/0us)\n      "
      "save/RestoreV2/tensor_names (0us/0us, 0us/0us)\n    save/RestoreV2_1 "
      "(0us/0us, 0us/0us)\n      save/RestoreV2_1/shape_and_slices (0us/0us, "
      "0us/0us)\n      save/RestoreV2_1/tensor_names (0us/0us, 0us/0us)\n    "
      "save/RestoreV2_2 (0us/0us, 0us/0us)\n      "
      "save/RestoreV2_2/shape_and_slices (0us/0us, 0us/0us)\n      "
      "save/RestoreV2_2/tensor_names (0us/0us, 0us/0us)\n    save/RestoreV2_3 "
      "(0us/0us, 0us/0us)\n      save/RestoreV2_3/shape_and_slices (0us/0us, "
      "0us/0us)\n      save/RestoreV2_3/tensor_names (0us/0us, 0us/0us)\n    "
      "save/SaveV2 (0us/0us, 0us/0us)\n      save/SaveV2/shape_and_slices "
      "(0us/0us, 0us/0us)\n      save/SaveV2/tensor_names (0us/0us, 0us/0us)\n "
      "   save/control_dependency (0us/0us, 0us/0us)\n    save/restore_all "
      "(0us/0us, 0us/0us)\n  zeros (0us/0us, 1us/1us)\n",
      dump_str);
}

TEST_F(TFProfShowTest, DumpOpMode) {
  string dump_file = io::JoinPath(testing::TmpDir(), "dump");
  Options opts(
      5, 0, 0, 0, 0, 4, -1, "params", {".*"},  // accout_type_regexes
      {".*"}, {""}, {".*"}, {""}, false,
      {"params", "bytes", "micros", "float_ops", "occurrence", "input_shapes"},
      "file", {{"outfile", dump_file}});
  tf_stats_->ShowMultiGraphNode("op", opts);

  string dump_str;
  TF_CHECK_OK(ReadFileToString(Env::Default(), dump_file, &dump_str));
  EXPECT_EQ(
      "nodename|outputbytes|totalexecutiontime|acceleratorexecutiontime|"
      "cpuexecutiontime|#parameters|#float_ops|opoccurrence(run|defined)|"
      "inputshapes\nVariableV21.48KB(100.00%,17.10%),5us(100.00%,5.15%),0us(0."
      "00%,0.00%),5us(100.00%,5.15%),370params(100.00%,100.00%),0float_ops(100."
      "00%,0.00%),4|4\n\ninput_type:\t(run*4|defined*4)\texec_time:"
      "5us\n\nAssign0B(0.00%,0.00%),0us(94.85%,0.00%),0us(0.00%,0.00%),0us(94."
      "85%,0.00%),0params(0.00%,0.00%),0float_ops(100.00%,0.00%),0|8\n\ninput_"
      "type:0:unknown,\t1:unknown\t(run*0|defined*8)\texec_time:0us\n\nConst1."
      "54KB(58.87%,17.74%),1us(80.41%,1.03%),0us(0.00%,0.00%),1us(80.41%,1.03%)"
      ",0params(0.00%,0.00%),0float_ops(98.49%,0.00%),1|24\n\ninput_type:\t("
      "run*1|defined*24)\texec_time:1us\n\n",
      StringReplace(dump_str, " ", ""));
}
}  // namespace tfprof
}  // namespace tensorflow
