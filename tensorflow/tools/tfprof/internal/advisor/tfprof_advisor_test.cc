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

#include "tensorflow/tools/tfprof/internal/advisor/tfprof_advisor.h"

#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace tfprof {

class TFProfAdvisorTest : public ::testing::Test {
 protected:
  TFProfAdvisorTest() {
    stats_.reset(new TFStats(std::unique_ptr<GraphDef>(new GraphDef()), nullptr,
                             nullptr, nullptr));

    stats_->AddNodeForTest(
        "n1", CreateNode("n1", "Conv2D", {{"data_format", "NHWC"}}, 10, 2));
    stats_->AddNodeForTest("n2", CreateNode("n2", "Conv2D", {}, 20, 2));
    advisor_.reset(new Advisor(stats_.get()));
  }

  std::unique_ptr<TFGraphNode> CreateNode(const string& name,
                                          const string& type,
                                          std::map<string, string> attrs,
                                          int64 start_miros,
                                          int64 end_rel_micros) {
    node_defs_.push_back(std::unique_ptr<NodeDef>(new NodeDef()));
    NodeDef* def = node_defs_.back().get();

    def->set_name(name);
    def->set_op(type);
    for (const auto& attr : attrs) {
      (*def->mutable_attr())[attr.first].set_s(attr.second);
    }
    std::unique_ptr<TFGraphNode> node(new TFGraphNode(def));

    NodeExecStats node_stat;
    node_stat.set_all_start_micros(start_miros);
    node_stat.set_op_end_rel_micros(end_rel_micros);
    node->AddStepStat(0, "/job:localhost/replica:0/task:0/gpu:0", node_stat);
    node->AddStepStat(0, "/job:localhost/replica:0/task:0/gpu:0:stream:all",
                      node_stat);
    node->AddStepStat(0, "/job:localhost/replica:0/task:0/gpu:0:stream:0",
                      node_stat);
    return node;
  }

  std::unique_ptr<TFStats> stats_;
  std::unique_ptr<Advisor> advisor_;
  std::vector<std::unique_ptr<NodeDef>> node_defs_;
};

TEST_F(TFProfAdvisorTest, Basics) {
  std::map<string, std::vector<string>> reports = advisor_->Advise();
  EXPECT_TRUE(reports.find("AcceleratorUtilizationChecker") != reports.end());
  EXPECT_TRUE(reports.find("OperationChecker") != reports.end());
}

TEST_F(TFProfAdvisorTest, OperationChecker) {
  std::map<string, std::vector<string>> reports = advisor_->Advise();
  EXPECT_EQ(reports["OperationChecker"].size(), 1);
  EXPECT_TRUE(StringPiece(reports["OperationChecker"][0]).contains("NCHW"));
}

TEST_F(TFProfAdvisorTest, UtilizationChecker) {
  std::map<string, std::vector<string>> reports = advisor_->Advise();
  EXPECT_EQ(reports["AcceleratorUtilizationChecker"].size(), 1);
  EXPECT_TRUE(StringPiece(reports["AcceleratorUtilizationChecker"][0])
                  .contains("low utilization"));
}

}  // namespace tfprof
}  // namespace tensorflow
