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

#include "tensorflow/core/util/stat_summarizer.h"

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace {

TEST(StatSummarizerTest, ExtractsOpTypes) {
  // GraphDef for a single constant name 'myconstant'
  const std::string graph_def_str(R"EOF(
node {
  name: "myconstant"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
versions {
  producer: 21
}
  )EOF");
  GraphDef graph_def;
  ASSERT_TRUE(protobuf::TextFormat::ParseFromString(graph_def_str, &graph_def));

  std::unique_ptr<Session> session(NewSession(SessionOptions()));
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(graph_def));

  RunOptions run_options;
  run_options.set_trace_level(RunOptions::FULL_TRACE);

  RunMetadata run_metadata;
  std::vector<Tensor> outputs;
  TF_ASSERT_OK(session->Run(run_options, {}, {"myconstant:0"}, {}, &outputs,
                            &run_metadata));

  StatSummarizer stats(graph_def);
  stats.ProcessStepStats(run_metadata.step_stats());

  const std::string output = stats.GetOutputString();
  const std::string by_node_type = stats.GetStatsByNodeType();

  // output should contain both the node type and node name.
  ASSERT_TRUE(output.find("Const") != std::string::npos) << output;
  ASSERT_TRUE(output.find("myconstant") != std::string::npos) << output;
  // stats by node type should include the type.
  ASSERT_TRUE(by_node_type.find("Const") != std::string::npos) << by_node_type;
}

}  // namespace
}  // namespace tensorflow
