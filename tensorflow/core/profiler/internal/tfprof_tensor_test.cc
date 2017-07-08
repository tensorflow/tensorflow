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
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/profiler/internal/tfprof_options.h"
#include "tensorflow/core/profiler/internal/tfprof_stats.h"
#include "tensorflow/core/profiler/internal/tfprof_utils.h"
#include "tensorflow/core/profiler/tfprof_log.pb.h"
#include "tensorflow/core/profiler/tfprof_output.pb.h"

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
    std::unique_ptr<OpLog> op_log_pb;

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
  Options opts(3, 0, 0, 0, 0, 0, -1, "name", {"VariableV2"}, {".*"}, {""},
               {".*"}, {""}, false, {"tensor_value"},  // show the tensor value.
               "", {});
  const TFGraphNodeProto& root = tf_stats_->ShowGraphNode("scope", opts);

  TFGraphNodeProto expected;
  CHECK(protobuf::TextFormat::ParseFromString(
      "name: \"_TFProfRoot\"\nexec_micros: 0\nrequested_bytes: "
      "0\ntotal_exec_micros: 0\ntotal_requested_bytes: 0\ntotal_parameters: "
      "370\nchildren {\n  name: \"conv2d\"\n  exec_micros: 0\n  "
      "requested_bytes: 0\n  total_exec_micros: 0\n  total_requested_bytes: "
      "0\n  total_parameters: 140\n  children {\n    name: \"conv2d/bias\"\n   "
      " exec_micros: 0\n    requested_bytes: 0\n    parameters: 5\n    "
      "total_exec_micros: 0\n    total_requested_bytes: 0\n    "
      "total_parameters: 5\n    float_ops: 0\n    total_float_ops: 0\n    "
      "tensor_value {\n      dtype: DT_FLOAT\n      value_double: 0\n      "
      "value_double: 0\n      value_double: 0\n      value_double: 0\n      "
      "value_double: 0\n    }\n    accelerator_exec_micros: 0\n    "
      "cpu_exec_micros: 0\n    total_accelerator_exec_micros: 0\n    "
      "total_cpu_exec_micros: 0\n    run_count: 0\n    total_run_count: 0\n    "
      "total_definition_count: 1\n  }\n  children {\n    name: "
      "\"conv2d/kernel\"\n    exec_micros: 0\n    requested_bytes: 0\n    "
      "parameters: 135\n    total_exec_micros: 0\n    total_requested_bytes: "
      "0\n    total_parameters: 135\n    float_ops: 0\n    total_float_ops: "
      "0\n    tensor_value {\n      dtype: DT_FLOAT\n      value_double: "
      "-0.113138\n      value_double: 0.261431\n      value_double: 0.215777\n "
      "     value_double: 0.24135\n      value_double: -0.113195\n      "
      "value_double: -0.212639\n      value_double: -0.0907301\n      "
      "value_double: 0.0221634\n      value_double: 0.21821\n      "
      "value_double: 0.22715\n      value_double: -0.108698\n      "
      "value_double: 0.240911\n      value_double: -0.138626\n      "
      "value_double: -0.144752\n      value_double: -0.00962037\n      "
      "value_double: 0.0971008\n      value_double: 0.00264764\n      "
      "value_double: -0.272929\n      value_double: 0.0129845\n      "
      "value_double: 0.0466554\n      value_double: -0.229184\n      "
      "value_double: 0.153576\n      value_double: -0.169218\n      "
      "value_double: -0.112991\n      value_double: 0.205739\n      "
      "value_double: 0.257844\n      value_double: 0.107455\n      "
      "value_double: -0.207914\n      value_double: 0.15211\n      "
      "value_double: 0.277932\n      value_double: 0.145986\n      "
      "value_double: -0.0883989\n      value_double: 0.167506\n      "
      "value_double: 0.10237\n      value_double: 0.0542143\n      "
      "value_double: 0.0334378\n      value_double: 0.159489\n      "
      "value_double: 0.246583\n      value_double: 0.0154283\n      "
      "value_double: 0.0872411\n      value_double: -0.25732\n      "
      "value_double: 0.0499355\n      value_double: 0.0266221\n      "
      "value_double: 0.088801\n      value_double: -0.0794552\n      "
      "value_double: -0.00383255\n      value_double: -0.165267\n      "
      "value_double: 0.0271328\n      value_double: 0.0729822\n      "
      "value_double: 0.200795\n      value_double: 0.100276\n      "
      "value_double: 0.285254\n      value_double: -0.171945\n      "
      "value_double: -0.0187411\n      value_double: -0.218729\n      "
      "value_double: 0.233753\n      value_double: 0.109184\n      "
      "value_double: 0.247875\n      value_double: -0.224632\n      "
      "value_double: 0.0940739\n      value_double: 0.00663087\n      "
      "value_double: -0.075786\n      value_double: -0.179992\n      "
      "value_double: -0.276016\n      value_double: 0.261207\n      "
      "value_double: -0.0658191\n      value_double: -0.0747132\n      "
      "value_double: -0.0839638\n      value_double: -0.0825393\n      "
      "value_double: 0.0915958\n      value_double: -0.195425\n      "
      "value_double: -0.255836\n      value_double: -0.08745\n      "
      "value_double: -0.181623\n      value_double: -0.235936\n      "
      "value_double: 0.0205423\n      value_double: 0.185447\n      "
      "value_double: -0.0691599\n      value_double: -0.0451089\n      "
      "value_double: -0.153922\n      value_double: -0.0279411\n      "
      "value_double: 0.148915\n      value_double: -0.018026\n      "
      "value_double: -0.144903\n      value_double: 0.0370046\n      "
      "value_double: 0.0764987\n      value_double: 0.0586488\n      "
      "value_double: -0.222919\n      value_double: 0.0238447\n      "
      "value_double: -0.106012\n      value_double: -0.102202\n      "
      "value_double: -0.159347\n      value_double: -0.0232876\n      "
      "value_double: 0.109855\n      value_double: -0.141833\n      "
      "value_double: 0.1376\n      value_double: -0.12413\n      value_double: "
      "-0.208968\n      value_double: 0.0758635\n      value_double: "
      "-0.217672\n      value_double: -0.20153\n      value_double: "
      "-0.195414\n      value_double: -0.18549\n      value_double: "
      "0.00298014\n      value_double: -0.279283\n      value_double: "
      "0.200084\n      value_double: -0.0968328\n      value_double: -0.243\n  "
      "    value_double: 0.239319\n      value_double: -0.236288\n      "
      "value_double: 0.169477\n      value_double: 0.126673\n      "
      "value_double: 0.182215\n      value_double: -0.028243\n      "
      "value_double: 0.282762\n      value_double: -0.165548\n      "
      "value_double: -0.0641245\n      value_double: -0.186382\n      "
      "value_double: 0.0329038\n      value_double: 0.271848\n      "
      "value_double: 0.084653\n      value_double: -0.108163\n      "
      "value_double: 0.247094\n      value_double: 0.192687\n      "
      "value_double: 0.171922\n      value_double: -0.187649\n      "
      "value_double: 0.251253\n      value_double: 0.272077\n      "
      "value_double: 0.19068\n      value_double: 0.220352\n      "
      "value_double: -0.255741\n      value_double: 0.110853\n      "
      "value_double: 0.146625\n      value_double: 0.167754\n      "
      "value_double: 0.249554\n    }\n    accelerator_exec_micros: 0\n    "
      "cpu_exec_micros: 0\n    total_accelerator_exec_micros: 0\n    "
      "total_cpu_exec_micros: 0\n    run_count: 0\n    total_run_count: 0\n    "
      "total_definition_count: 1\n  }\n  float_ops: 0\n  total_float_ops: 0\n  "
      "accelerator_exec_micros: 0\n  cpu_exec_micros: 0\n  "
      "total_accelerator_exec_micros: 0\n  total_cpu_exec_micros: 0\n  "
      "run_count: 0\n  total_run_count: 0\n  total_definition_count: "
      "3\n}\nchildren {\n  name: \"conv2d_1\"\n  exec_micros: 0\n  "
      "requested_bytes: 0\n  total_exec_micros: 0\n  total_requested_bytes: "
      "0\n  total_parameters: 230\n  children {\n    name: \"conv2d_1/bias\"\n "
      "   exec_micros: 0\n    requested_bytes: 0\n    parameters: 5\n    "
      "total_exec_micros: 0\n    total_requested_bytes: 0\n    "
      "total_parameters: 5\n    float_ops: 0\n    total_float_ops: 0\n    "
      "tensor_value {\n      dtype: DT_FLOAT\n      value_double: 0\n      "
      "value_double: 0\n      value_double: 0\n      value_double: 0\n      "
      "value_double: 0\n    }\n    accelerator_exec_micros: 0\n    "
      "cpu_exec_micros: 0\n    total_accelerator_exec_micros: 0\n    "
      "total_cpu_exec_micros: 0\n    run_count: 0\n    total_run_count: 0\n    "
      "total_definition_count: 1\n  }\n  children {\n    name: "
      "\"conv2d_1/kernel\"\n    exec_micros: 0\n    requested_bytes: 0\n    "
      "parameters: 225\n    total_exec_micros: 0\n    total_requested_bytes: "
      "0\n    total_parameters: 225\n    float_ops: 0\n    total_float_ops: "
      "0\n    tensor_value {\n      dtype: DT_FLOAT\n      value_double: "
      "-0.00170514\n      value_double: 0.138601\n      value_double: "
      "-0.224822\n      value_double: -0.0848449\n      value_double: "
      "0.170551\n      value_double: 0.147666\n      value_double: "
      "-0.0570606\n      value_double: -0.132805\n      value_double: "
      "-0.172013\n      value_double: 0.249707\n      value_double: 0.149734\n "
      "     value_double: 0.0365986\n      value_double: -0.0923146\n      "
      "value_double: -0.17745\n      value_double: -0.169978\n      "
      "value_double: -0.173298\n      value_double: -0.110407\n      "
      "value_double: 0.1469\n      value_double: 0.0419576\n      "
      "value_double: 0.0391093\n      value_double: -0.137381\n      "
      "value_double: 0.212642\n      value_double: -0.067034\n      "
      "value_double: -0.0727709\n      value_double: -0.0276531\n      "
      "value_double: 0.218212\n      value_double: 0.0596479\n      "
      "value_double: -0.0468102\n      value_double: -0.0250467\n      "
      "value_double: -0.20391\n      value_double: -0.233801\n      "
      "value_double: 0.135615\n      value_double: -0.182124\n      "
      "value_double: 0.254205\n      value_double: 0.0819146\n      "
      "value_double: -0.146696\n      value_double: -0.20095\n      "
      "value_double: -0.250555\n      value_double: -0.226406\n      "
      "value_double: 0.0421331\n      value_double: 0.0361264\n      "
      "value_double: -0.188558\n      value_double: -0.0222711\n      "
      "value_double: -0.128226\n      value_double: -0.148305\n      "
      "value_double: -0.137598\n      value_double: -0.041647\n      "
      "value_double: -0.0574933\n      value_double: 0.122506\n      "
      "value_double: 0.0415936\n      value_double: 0.244957\n      "
      "value_double: 0.00372121\n      value_double: -0.139939\n      "
      "value_double: 0.250411\n      value_double: -0.23848\n      "
      "value_double: -0.0717569\n      value_double: -0.00884159\n      "
      "value_double: 0.135616\n      value_double: -0.0493895\n      "
      "value_double: 0.254308\n      value_double: -0.181419\n      "
      "value_double: -0.114829\n      value_double: -0.172638\n      "
      "value_double: 0.06984\n      value_double: -0.086704\n      "
      "value_double: 0.168515\n      value_double: -0.152275\n      "
      "value_double: -0.230775\n      value_double: -0.254366\n      "
      "value_double: -0.115397\n      value_double: 0.0418207\n      "
      "value_double: -0.199607\n      value_double: -0.167001\n      "
      "value_double: -0.187238\n      value_double: 0.0196097\n      "
      "value_double: 0.201653\n      value_double: -0.143758\n      "
      "value_double: 0.167187\n      value_double: -0.129141\n      "
      "value_double: 0.230154\n      value_double: -0.119968\n      "
      "value_double: -0.121843\n      value_double: -0.0118565\n      "
      "value_double: 0.0285747\n      value_double: -0.0593699\n      "
      "value_double: -0.175214\n      value_double: -0.211524\n      "
      "value_double: 0.167042\n      value_double: -0.216357\n      "
      "value_double: -0.0218886\n      value_double: -0.244211\n      "
      "value_double: 0.175301\n      value_double: 0.0654932\n      "
      "value_double: -0.0419763\n      value_double: -0.103275\n      "
      "value_double: -0.0848433\n      value_double: -0.0845421\n      "
      "value_double: -0.00269318\n      value_double: -0.145978\n      "
      "value_double: -0.217061\n      value_double: -0.0937043\n      "
      "value_double: 0.235796\n      value_double: -0.0893372\n      "
      "value_double: 0.000827968\n      value_double: 0.0172743\n      "
      "value_double: -0.234205\n      value_double: -0.0867703\n      "
      "value_double: 0.131704\n      value_double: 0.134143\n      "
      "value_double: -0.162257\n      value_double: -0.129706\n      "
      "value_double: 0.0763288\n      value_double: 0.156988\n      "
      "value_double: 0.220033\n      value_double: -0.179884\n      "
      "value_double: 0.066697\n      value_double: 0.212322\n      "
      "value_double: -0.0961226\n      value_double: -0.11223\n      "
      "value_double: 0.249944\n      value_double: 0.115673\n      "
      "value_double: -0.100203\n      value_double: 0.125645\n      "
      "value_double: -0.256104\n      value_double: 0.0996534\n      "
      "value_double: 0.167306\n      value_double: -0.00700775\n      "
      "value_double: 0.242145\n      value_double: 0.088406\n      "
      "value_double: 0.0975334\n      value_double: -0.0309525\n      "
      "value_double: -0.0422794\n      value_double: 0.20739\n      "
      "value_double: 0.113992\n      value_double: 0.253818\n      "
      "value_double: -0.0857835\n      value_double: 0.223902\n      "
      "value_double: 0.10291\n      value_double: 0.103091\n      "
      "value_double: -0.177502\n      value_double: -0.0258242\n      "
      "value_double: -0.130567\n      value_double: -0.15999\n      "
      "value_double: -0.101484\n      value_double: 0.0188813\n      "
      "value_double: 0.160626\n      value_double: 0.0467491\n      "
      "value_double: 0.193634\n      value_double: -0.0910993\n      "
      "value_double: 0.0440249\n      value_double: -0.255389\n      "
      "value_double: -0.240244\n      value_double: -0.213171\n      "
      "value_double: 0.175978\n      value_double: -0.0251202\n      "
      "value_double: 0.0943941\n      value_double: -0.196194\n      "
      "value_double: 0.163395\n      value_double: -0.010777\n      "
      "value_double: -0.0626751\n      value_double: -0.246234\n      "
      "value_double: 0.0662063\n      value_double: 0.120589\n      "
      "value_double: 0.237322\n      value_double: 0.0849243\n      "
      "value_double: -0.066591\n      value_double: 0.0512236\n      "
      "value_double: -0.144309\n      value_double: -0.235415\n      "
      "value_double: -0.0565311\n      value_double: 0.0882529\n      "
      "value_double: -0.215923\n      value_double: -0.0873292\n      "
      "value_double: -0.0691103\n      value_double: -0.00238678\n      "
      "value_double: 0.147789\n      value_double: -0.124451\n      "
      "value_double: 0.205044\n      value_double: -0.0596834\n      "
      "value_double: 0.0268479\n      value_double: 0.0857448\n      "
      "value_double: -0.0923855\n      value_double: -0.0960547\n      "
      "value_double: 0.169869\n      value_double: 0.16988\n      "
      "value_double: -0.032271\n      value_double: -0.120731\n      "
      "value_double: -0.199086\n      value_double: 0.181199\n      "
      "value_double: 0.00897732\n      value_double: -0.257469\n      "
      "value_double: -0.135556\n      value_double: -0.149663\n      "
      "value_double: -0.00990398\n      value_double: 0.221165\n      "
      "value_double: 0.0327134\n      value_double: -0.0392821\n      "
      "value_double: -0.0614503\n      value_double: 0.246602\n      "
      "value_double: -0.171692\n      value_double: -0.150835\n      "
      "value_double: -0.13854\n      value_double: -0.244668\n      "
      "value_double: 0.0790781\n      value_double: 0.212678\n      "
      "value_double: 0.0782059\n      value_double: -0.177888\n      "
      "value_double: -0.165914\n      value_double: -0.164251\n      "
      "value_double: 0.165007\n      value_double: 0.239615\n      "
      "value_double: -0.217642\n      value_double: -0.219843\n      "
      "value_double: 0.0828398\n      value_double: 0.00272235\n      "
      "value_double: -0.0323662\n      value_double: -0.255953\n      "
      "value_double: 0.237298\n      value_double: -0.0896481\n      "
      "value_double: -0.0605349\n      value_double: 0.231679\n      "
      "value_double: -0.123842\n      value_double: 0.0858642\n      "
      "value_double: 0.23111\n      value_double: 0.0491742\n    }\n    "
      "accelerator_exec_micros: 0\n    cpu_exec_micros: 0\n    "
      "total_accelerator_exec_micros: 0\n    total_cpu_exec_micros: 0\n    "
      "run_count: 0\n    total_run_count: 0\n    total_definition_count: 1\n  "
      "}\n  float_ops: 0\n  total_float_ops: 0\n  accelerator_exec_micros: 0\n "
      " cpu_exec_micros: 0\n  total_accelerator_exec_micros: 0\n  "
      "total_cpu_exec_micros: 0\n  run_count: 0\n  total_run_count: 0\n  "
      "total_definition_count: 3\n}\nfloat_ops: 0\ntotal_float_ops: "
      "0\naccelerator_exec_micros: 0\ncpu_exec_micros: "
      "0\ntotal_accelerator_exec_micros: 0\ntotal_cpu_exec_micros: "
      "0\nrun_count: 0\ntotal_run_count: 0\ntotal_definition_count: 6\n",
      &expected));
  EXPECT_EQ(expected.DebugString(), root.DebugString());
}

}  // namespace tfprof
}  // namespace tensorflow
