/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/utils/grappler_test.h"
#include <memory>
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace grappler {

std::vector<Tensor> GrapplerTest::EvaluateNodes(
    const GraphDef& graph, const std::vector<string>& node_names) {
  SessionOptions options;
  std::unique_ptr<tensorflow::Session> session(NewSession(options));
  TF_CHECK_OK(session->Create(graph));
  RunOptions run_options;
  std::vector<Tensor> output_tensors;
  TF_CHECK_OK(session->Run(run_options, {}, node_names, node_names,
                           &output_tensors, nullptr));
  TF_CHECK_OK(session->Close());
  return output_tensors;
}

}  // namespace grappler
}  // namespace tensorflow
