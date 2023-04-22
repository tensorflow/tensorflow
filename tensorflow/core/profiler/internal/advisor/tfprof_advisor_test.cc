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

#include "tensorflow/core/profiler/internal/advisor/tfprof_advisor.h"

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
        0, CreateNode("n1", "Conv2D", {{"data_format", "NHWC"}}, 0, 10, 2));
    stats_->AddNodeForTest(0, CreateNode("n2", "Conv2D", {}, 0, 20, 2));
    stats_->BuildAllViews();
    advisor_.reset(new Advisor(stats_.get()));
  }

  std::unique_ptr<TFGraphNode> CreateNode(const string& name,
                                          const string& type,
                                          std::map<string, string> attrs,
                                          int64_t step, int64_t start_miros,
                                          int64_t end_rel_micros) {
    node_defs_.push_back(std::unique_ptr<NodeDef>(new NodeDef()));
    NodeDef* def = node_defs_.back().get();

    def->set_name(name);
    def->set_op(type);
    for (const auto& attr : attrs) {
      (*def->mutable_attr())[attr.first].set_s(attr.second);
    }
    std::unique_ptr<TFGraphNode> node(new TFGraphNode(def, -1, nullptr));

    NodeExecStats node_stat;
    node_stat.set_all_start_micros(start_miros);
    node_stat.set_op_end_rel_micros(end_rel_micros);
    node->AddStepStat(step, "/job:localhost/replica:0/task:0/device:GPU:0",
                      node_stat);
    node->AddStepStat(step,
                      "/job:localhost/replica:0/task:0/device:GPU:0:stream:all",
                      node_stat);
    node->AddStepStat(step,
                      "/job:localhost/replica:0/task:0/device:GPU:0:stream:0",
                      node_stat);
    return node;
  }

  std::unique_ptr<TFStats> stats_;
  std::unique_ptr<Advisor> advisor_;
  std::vector<std::unique_ptr<NodeDef>> node_defs_;
};

TEST_F(TFProfAdvisorTest, Basics) {
  AdvisorOptionsProto options = Advisor::DefaultOptions();
  AdviceProto advice = advisor_->Advise(options);
  EXPECT_TRUE(advice.checkers().find(kCheckers[0]) != advice.checkers().end());
  EXPECT_TRUE(advice.checkers().find(kCheckers[1]) != advice.checkers().end());
  EXPECT_TRUE(advice.checkers().find(kCheckers[2]) != advice.checkers().end());
}

TEST_F(TFProfAdvisorTest, OperationChecker) {
  AdvisorOptionsProto options;
  (*options.mutable_checkers())[kCheckers[1]];
  AdviceProto advice = advisor_->Advise(options);
  EXPECT_EQ(advice.checkers().at(kCheckers[1]).reports_size(), 1);
  EXPECT_TRUE(
      absl::StrContains(advice.checkers().at(kCheckers[1]).reports(0), "NCHW"));
}

TEST_F(TFProfAdvisorTest, UtilizationChecker) {
  AdvisorOptionsProto options;
  (*options.mutable_checkers())[kCheckers[0]];
  AdviceProto advice = advisor_->Advise(options);
  EXPECT_EQ(advice.checkers().at(kCheckers[0]).reports_size(), 1);
  EXPECT_TRUE(absl::StrContains(advice.checkers().at(kCheckers[0]).reports(0),
                                "low utilization"));
}

TEST_F(TFProfAdvisorTest, ExpensiveOperationChecker) {
  AdvisorOptionsProto options;
  (*options.mutable_checkers())[kCheckers[2]];
  AdviceProto advice = advisor_->Advise(options);
  EXPECT_TRUE(absl::StrContains(advice.checkers().at(kCheckers[2]).reports(0),
                                "top 1 operation type: Conv2D"));
}

}  // namespace tfprof
}  // namespace tensorflow
