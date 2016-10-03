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
#include "tensorflow/contrib/tfprof/tools/tfprof/internal/tfprof_options.h"
#include "tensorflow/contrib/tfprof/tools/tfprof/internal/tfprof_stats.h"
#include "tensorflow/contrib/tfprof/tools/tfprof/internal/tfprof_utils.h"
#include "tensorflow/contrib/tfprof/tools/tfprof/tfprof_log.pb.h"
#include "tensorflow/contrib/tfprof/tools/tfprof/tfprof_output.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {
namespace tfprof {
class TFProfTensorTest : public ::testing::Test {
 protected:
  TFProfTensorTest() {
    string graph_path = io::JoinPath(
        testing::TensorFlowSrcRoot(),
        "contrib/tfprof/tools/tfprof/internal/testdata/graph.pbtxt");
    std::unique_ptr<tensorflow::GraphDef> graph_pb(new tensorflow::GraphDef());
    TF_CHECK_OK(ReadGraphDefText(Env::Default(), graph_path, graph_pb.get()));

    std::unique_ptr<tensorflow::RunMetadata> run_meta_pb;
    std::unique_ptr<OpLog> op_log_pb;

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

TEST_F(TFProfTensorTest, Basics) {
  Options opts(3, 0, 0, 0, 0, {".*"}, "name", {"Variable"}, {".*"}, {""},
               {".*"}, {""}, false, {"tensor_value"},  // show the tensor value.
               false);
  const TFProfNode& root = tf_stats_->PrintGraph("scope", opts);

  TFProfNode expected;
  CHECK(protobuf::TextFormat::ParseFromString(
      "name: \"_TFProfRoot\"\nexec_micros: 0\nrequested_bytes: "
      "0\ntotal_exec_micros: 0\ntotal_requested_bytes: 0\ntotal_parameters: "
      "450\nchildren {\n  name: \"DW\"\n  exec_micros: 0\n  requested_bytes: "
      "0\n  parameters: 162\n  total_exec_micros: 0\n  total_requested_bytes: "
      "0\n  total_parameters: 162\n  float_ops: 0\n  total_float_ops: 0\n  "
      "tensor_value {\n    dtype: DT_FLOAT\n    value_double: -0.00117808\n    "
      "value_double: -0.000709941\n    value_double: -0.00174816\n    "
      "value_double: -0.000495372\n    value_double: 0.000243039\n    "
      "value_double: -0.000126313\n    value_double: -0.000663929\n    "
      "value_double: -0.000495198\n    value_double: -0.000893934\n    "
      "value_double: -0.00179659\n    value_double: 0.000408874\n    "
      "value_double: -0.00120166\n    value_double: -0.00109484\n    "
      "value_double: -0.000200362\n    value_double: 0.000726721\n    "
      "value_double: -0.000277568\n    value_double: 0.00180584\n    "
      "value_double: 0.000997271\n    value_double: -0.00185987\n    "
      "value_double: -0.00113401\n    value_double: -0.000528852\n    "
      "value_double: -0.000197412\n    value_double: 1.32871e-05\n    "
      "value_double: -0.000285896\n    value_double: -0.000428898\n    "
      "value_double: -0.000424633\n    value_double: 2.15488e-05\n    "
      "value_double: 0.00149753\n    value_double: -0.000884576\n    "
      "value_double: -0.0013795\n    value_double: -0.000650125\n    "
      "value_double: 0.00191612\n    value_double: 4.71838e-05\n    "
      "value_double: 0.000400201\n    value_double: 0.00239555\n    "
      "value_double: -0.00177706\n    value_double: -0.000781899\n    "
      "value_double: -0.00145247\n    value_double: 0.0020025\n    "
      "value_double: 0.000597419\n    value_double: 0.00135456\n    "
      "value_double: 0.0015876\n    value_double: -0.000993568\n    "
      "value_double: 0.0006509\n    value_double: -0.000894533\n    "
      "value_double: -0.00129322\n    value_double: 0.0003859\n    "
      "value_double: 0.000415186\n    value_double: -0.000439212\n    "
      "value_double: 0.000442138\n    value_double: 0.00212353\n    "
      "value_double: 0.000702953\n    value_double: 0.000713424\n    "
      "value_double: -0.000304877\n    value_double: -9.17046e-05\n    "
      "value_double: -0.000801103\n    value_double: 0.000304854\n    "
      "value_double: -0.00070527\n    value_double: -0.00106408\n    "
      "value_double: -0.000909906\n    value_double: -4.49183e-05\n    "
      "value_double: 0.000104172\n    value_double: -0.000438067\n    "
      "value_double: -0.000317689\n    value_double: -0.000769914\n    "
      "value_double: -0.00157729\n    value_double: 0.000220733\n    "
      "value_double: 0.00107268\n    value_double: -0.000186449\n    "
      "value_double: -0.000807328\n    value_double: 0.000456308\n    "
      "value_double: -0.000593729\n    value_double: -0.000954873\n    "
      "value_double: -0.000268676\n    value_double: 9.06328e-05\n    "
      "value_double: -0.000323473\n    value_double: -0.000628768\n    "
      "value_double: 0.000664985\n    value_double: 0.0020999\n    "
      "value_double: -0.000932228\n    value_double: -0.00203203\n    "
      "value_double: 0.000565405\n    value_double: 0.000167899\n    "
      "value_double: 0.00054897\n    value_double: 0.000612407\n    "
      "value_double: -0.000619301\n    value_double: 0.00169361\n    "
      "value_double: -0.000188057\n    value_double: 0.000267652\n    "
      "value_double: -0.00127341\n    value_double: -0.000218836\n    "
      "value_double: -0.000431722\n    value_double: 5.41867e-05\n    "
      "value_double: 0.000296628\n    value_double: 0.000819415\n    "
      "value_double: -0.000758993\n    value_double: -0.000114477\n    "
      "value_double: 6.29219e-05\n    value_double: 0.000726988\n    "
      "value_double: -0.00135974\n    value_double: 2.28447e-05\n    "
      "value_double: 0.00120547\n    value_double: -0.00136907\n    "
      "value_double: -0.00140188\n    value_double: 0.000201145\n    "
      "value_double: -0.000774109\n    value_double: 0.000798465\n    "
      "value_double: -0.00131861\n    value_double: 3.08996e-05\n    "
      "value_double: -0.000637026\n    value_double: 0.00228975\n    "
      "value_double: -0.000633757\n    value_double: -0.00116047\n    "
      "value_double: 7.66039e-05\n    value_double: 2.09167e-06\n    "
      "value_double: -0.000296448\n    value_double: 0.000206795\n    "
      "value_double: 0.000674405\n    value_double: -0.000722742\n    "
      "value_double: -9.32443e-05\n    value_double: -0.00170917\n    "
      "value_double: -0.000505279\n    value_double: 0.000628132\n    "
      "value_double: -0.00145929\n    value_double: 0.00106077\n    "
      "value_double: -0.000796743\n    value_double: 0.000498275\n    "
      "value_double: -0.0002914\n    value_double: -0.00230622\n    "
      "value_double: -9.42872e-05\n    value_double: 0.000200359\n    "
      "value_double: -0.00305027\n    value_double: -0.0016218\n    "
      "value_double: 0.00137126\n    value_double: -0.00215436\n    "
      "value_double: -0.000743827\n    value_double: -0.00090007\n    "
      "value_double: -0.000762207\n    value_double: -0.000149951\n    "
      "value_double: -0.0013102\n    value_double: 0.00165781\n    "
      "value_double: 0.000343809\n    value_double: -0.000826069\n    "
      "value_double: -4.67404e-05\n    value_double: 0.0023931\n    "
      "value_double: 0.00165338\n    value_double: -0.00050529\n    "
      "value_double: 0.000178771\n    value_double: -0.000858287\n    "
      "value_double: -0.00157031\n    value_double: -0.00165846\n    "
      "value_double: -0.000713672\n    value_double: 0.00014357\n    "
      "value_double: 0.00203632\n    value_double: -0.0010973\n    "
      "value_double: -9.89852e-05\n    value_double: 0.000558808\n    "
      "value_double: 0.00087211\n    value_double: 0.000661239\n    "
      "value_double: 0.000389605\n    value_double: 0.00060653\n    "
      "value_double: -0.000330104\n  }\n}\nchildren {\n  name: \"DW2\"\n  "
      "exec_micros: 0\n  requested_bytes: 0\n  parameters: 288\n  "
      "total_exec_micros: 0\n  total_requested_bytes: 0\n  total_parameters: "
      "288\n  float_ops: 0\n  total_float_ops: 0\n  tensor_value {\n    dtype: "
      "DT_FLOAT\n    value_double: 0.000704577\n    value_double: "
      "0.000127421\n    value_double: 0.00105952\n    value_double: "
      "0.000423765\n    value_double: -0.00025461\n    value_double: "
      "-0.000857203\n    value_double: 0.000693494\n    value_double: "
      "0.000282214\n    value_double: 0.00106185\n    value_double: "
      "-0.000836552\n    value_double: -0.00116766\n    value_double: "
      "0.000733674\n    value_double: -0.000669601\n    value_double: "
      "-0.000275175\n    value_double: -0.000428215\n    value_double: "
      "-0.000495715\n    value_double: -0.000125887\n    value_double: "
      "-0.000715204\n    value_double: -0.00108936\n    value_double: "
      "0.000738267\n    value_double: 0.000376081\n    value_double: "
      "0.00191442\n    value_double: 0.001423\n    value_double: -0.00093811\n "
      "   value_double: -5.91421e-05\n    value_double: -0.000221507\n    "
      "value_double: -0.000104555\n    value_double: -0.00069682\n    "
      "value_double: -0.000278325\n    value_double: -0.00122748\n    "
      "value_double: -0.00112411\n    value_double: -0.000440511\n    "
      "value_double: -0.000392247\n    value_double: -0.000419606\n    "
      "value_double: -0.00167063\n    value_double: -0.000988578\n    "
      "value_double: -0.00040159\n    value_double: 0.00238918\n    "
      "value_double: -0.000892898\n    value_double: -0.000875976\n    "
      "value_double: 0.00154401\n    value_double: -0.000719911\n    "
      "value_double: 0.000753941\n    value_double: -0.000119961\n    "
      "value_double: -0.000305115\n    value_double: 9.97947e-05\n    "
      "value_double: -0.00128908\n    value_double: -0.000584184\n    "
      "value_double: -0.000734685\n    value_double: -0.00146612\n    "
      "value_double: 0.000670802\n    value_double: 0.000924219\n    "
      "value_double: -0.000154409\n    value_double: 0.000198231\n    "
      "value_double: -0.000340742\n    value_double: -0.00159646\n    "
      "value_double: -1.19382e-05\n    value_double: 0.00165203\n    "
      "value_double: 0.0017085\n    value_double: -0.000199614\n    "
      "value_double: 0.000529526\n    value_double: 0.000769364\n    "
      "value_double: 0.00135369\n    value_double: 0.00132873\n    "
      "value_double: 0.000451174\n    value_double: 0.000255218\n    "
      "value_double: 0.00102891\n    value_double: -0.00160068\n    "
      "value_double: 0.000324269\n    value_double: -0.000492347\n    "
      "value_double: 0.000925301\n    value_double: 0.00281998\n    "
      "value_double: -0.000826404\n    value_double: -0.000602903\n    "
      "value_double: 0.00126559\n    value_double: 0.000924364\n    "
      "value_double: -9.19827e-05\n    value_double: -5.59275e-05\n    "
      "value_double: 0.00107971\n    value_double: -9.91756e-05\n    "
      "value_double: 0.000864708\n    value_double: 0.00121747\n    "
      "value_double: 0.00146338\n    value_double: 0.000186883\n    "
      "value_double: -0.00168195\n    value_double: -0.00062029\n    "
      "value_double: 0.000658127\n    value_double: 0.00115682\n    "
      "value_double: -0.00178359\n    value_double: 0.000685606\n    "
      "value_double: -0.000503373\n    value_double: -0.000312999\n    "
      "value_double: 0.000335383\n    value_double: -1.08597e-05\n    "
      "value_double: -8.2499e-05\n    value_double: -0.000469726\n    "
      "value_double: -0.00170868\n    value_double: 0.000118957\n    "
      "value_double: -0.000460736\n    value_double: -5.56372e-05\n    "
      "value_double: -0.00110148\n    value_double: 0.00059123\n    "
      "value_double: 0.000386339\n    value_double: -0.00139967\n    "
      "value_double: -0.000835664\n    value_double: 0.00103421\n    "
      "value_double: -0.00104296\n    value_double: -0.000687497\n    "
      "value_double: 1.1338e-05\n    value_double: 0.00176484\n    "
      "value_double: 0.000531523\n    value_double: -0.000986387\n    "
      "value_double: -0.00114152\n    value_double: 0.000256744\n    "
      "value_double: 0.000228425\n    value_double: 0.00116583\n    "
      "value_double: 0.0002726\n    value_double: -0.00100828\n    "
      "value_double: -0.000950376\n    value_double: -0.00229074\n    "
      "value_double: -0.000348272\n    value_double: -0.000526032\n    "
      "value_double: -0.000133703\n    value_double: 0.000310979\n    "
      "value_double: -0.00199278\n    value_double: -0.000874469\n    "
      "value_double: -0.000631466\n    value_double: 0.0010534\n    "
      "value_double: 0.00134646\n    value_double: -0.00172743\n    "
      "value_double: 0.00131031\n    value_double: -0.000697506\n    "
      "value_double: 0.000286747\n    value_double: 0.000140759\n    "
      "value_double: 0.000568707\n    value_double: 0.000108177\n    "
      "value_double: -0.00207337\n    value_double: -0.00138146\n    "
      "value_double: 0.000483162\n    value_double: -0.00167096\n    "
      "value_double: -0.000465813\n    value_double: 0.00067724\n    "
      "value_double: 2.08388e-05\n    value_double: -0.00203279\n    "
      "value_double: 7.8429e-05\n    value_double: 0.00161337\n    "
      "value_double: -0.000269005\n    value_double: 0.000217822\n    "
      "value_double: 0.000599886\n    value_double: 0.000317549\n    "
      "value_double: 0.00146597\n    value_double: -0.00210947\n    "
      "value_double: -0.000823917\n    value_double: -6.83766e-05\n    "
      "value_double: 0.000656085\n    value_double: 0.000117134\n    "
      "value_double: -0.000390405\n    value_double: 2.39565e-05\n    "
      "value_double: 0.00104837\n    value_double: -0.000563671\n    "
      "value_double: 0.000634073\n    value_double: -0.000554531\n    "
      "value_double: 0.000677971\n    value_double: -0.000596207\n    "
      "value_double: -0.00103335\n    value_double: 0.000645199\n    "
      "value_double: 0.00162195\n    value_double: 0.000239246\n    "
      "value_double: 0.00113519\n    value_double: 0.000787431\n    "
      "value_double: -0.000471688\n    value_double: -0.000216625\n    "
      "value_double: -0.000537156\n    value_double: 0.000551816\n    "
      "value_double: 0.00094337\n    value_double: -0.000708127\n    "
      "value_double: 0.000956955\n    value_double: -0.000904936\n    "
      "value_double: -0.000424413\n    value_double: 0.000106455\n    "
      "value_double: -0.000443952\n    value_double: 0.000185436\n    "
      "value_double: 0.000944397\n    value_double: -0.000760572\n    "
      "value_double: 0.000560002\n    value_double: 4.09886e-05\n    "
      "value_double: -0.00075076\n    value_double: -0.000701856\n    "
      "value_double: -0.000234851\n    value_double: -0.000131515\n    "
      "value_double: -0.000761718\n    value_double: -0.000267808\n    "
      "value_double: -0.00039682\n    value_double: 0.000542953\n    "
      "value_double: -0.000817685\n    value_double: 0.00103851\n    "
      "value_double: -0.000427176\n    value_double: 0.000517784\n    "
      "value_double: -0.000823552\n    value_double: -0.000742637\n    "
      "value_double: 0.000529213\n    value_double: -0.000372805\n    "
      "value_double: 1.85745e-05\n    value_double: 0.00139891\n    "
      "value_double: -0.000128417\n    value_double: -0.000404316\n    "
      "value_double: -0.000671571\n    value_double: 0.000490311\n    "
      "value_double: -0.00118493\n    value_double: -0.000897118\n    "
      "value_double: 0.000939601\n    value_double: 0.000376399\n    "
      "value_double: 0.0014709\n    value_double: 0.000134806\n    "
      "value_double: -0.000294469\n    value_double: -0.000569142\n    "
      "value_double: 0.00127266\n    value_double: -0.00140936\n    "
      "value_double: 0.000870083\n    value_double: 0.000287246\n    "
      "value_double: 0.000537685\n    value_double: 0.000125569\n    "
      "value_double: 0.000360276\n    value_double: -0.000186268\n    "
      "value_double: 0.0011141\n    value_double: -0.000605185\n    "
      "value_double: -0.0016281\n    value_double: -0.000552758\n    "
      "value_double: -0.000196755\n    value_double: -0.00265188\n    "
      "value_double: 0.000480997\n    value_double: 0.00018776\n    "
      "value_double: -0.00199234\n    value_double: 0.000959982\n    "
      "value_double: 0.00040334\n    value_double: -0.000693596\n    "
      "value_double: 0.00157678\n    value_double: -0.00134499\n    "
      "value_double: 0.00121909\n    value_double: -0.000328734\n    "
      "value_double: 0.000148554\n    value_double: -0.000209509\n    "
      "value_double: -0.000266303\n    value_double: -0.00134084\n    "
      "value_double: 5.21371e-05\n    value_double: 0.0005329\n    "
      "value_double: -0.000168858\n    value_double: -0.00074875\n    "
      "value_double: 0.000959397\n    value_double: -0.00159476\n    "
      "value_double: -0.000368838\n    value_double: 0.0006077\n    "
      "value_double: -0.00117243\n    value_double: -0.00146013\n    "
      "value_double: 0.00031519\n    value_double: -0.000167911\n    "
      "value_double: 0.000482571\n    value_double: -0.000752268\n    "
      "value_double: -0.00042363\n    value_double: 0.00121219\n    "
      "value_double: -0.000208159\n    value_double: 0.000128531\n    "
      "value_double: -0.000406308\n    value_double: -0.000242663\n    "
      "value_double: -3.96673e-05\n    value_double: 0.00144854\n    "
      "value_double: -0.000787328\n    value_double: -0.000401958\n    "
      "value_double: 0.00114091\n    value_double: -0.000739546\n    "
      "value_double: 0.000483236\n    value_double: -0.000916945\n    "
      "value_double: -0.00129577\n    value_double: -0.00186504\n    "
      "value_double: 0.000806804\n    value_double: -0.000152251\n    "
      "value_double: 0.000662576\n    value_double: -0.000533236\n    "
      "value_double: 0.00151019\n    value_double: 0.00127805\n    "
      "value_double: 0.00115399\n    value_double: -0.00130876\n    "
      "value_double: 2.99457e-06\n    value_double: 0.000820777\n    "
      "value_double: 0.000878393\n    value_double: -0.000562642\n    "
      "value_double: -0.00070442\n    value_double: -0.00066277\n  "
      "}\n}\nfloat_ops: 0\ntotal_float_ops: 0\n",
      &expected));
  EXPECT_EQ(expected.DebugString(), root.DebugString());
}

}  // namespace tfprof
}  // namespace tensorflow
