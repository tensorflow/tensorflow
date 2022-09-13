/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2tensorrt/experimental/utils/session_utils.h"

#include <string>
#include <vector>

#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/tf2tensorrt/common/utils.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT

namespace tensorflow {
namespace tensorrt {

tensorflow::SessionOptions GetSessionConfig() {
  // We also need to disable constant folding because we already ran constant
  // folding and may have prevented quantization operation folding on purpose.
  tensorflow::SessionOptions opts;
  auto* rewriter_opts =
      opts.config.mutable_graph_options()->mutable_rewrite_options();
  rewriter_opts->set_experimental_disable_folding_quantization_emulation(true);

  // It seems  that we need to disable the optimizer entirely to prevent the
  // folding.
  rewriter_opts->set_disable_meta_optimizer(true);
  return opts;
}

Status ImportGraphDefToSession(Session* session, const GraphDef& graph_def,
                               const string& prefix, bool ignore_existing) {
  ImportGraphDefOptions opts;
  opts.prefix = prefix;
  Graph graph(OpRegistry::Global());
  TF_RETURN_IF_ERROR(ImportGraphDef(opts, graph_def, &graph, nullptr));
  GraphDef new_graph_def;
  graph.ToGraphDef(&new_graph_def);
  auto status = session->Extend(new_graph_def);
  // If `ignore_existing` is true, don't return the invalid argument
  // error from calling Extend.
  if (!status.ok() && !(ignore_existing && errors::IsInvalidArgument(status))) {
    return status;
  }
  return OkStatus();
}

Status RunSession(Session* session, const std::vector<std::string>& input_names,
                  const std::vector<std::string>& output_names,
                  const std::vector<Tensor>& input_tensors,
                  std::string prefix) {
  TRT_ENSURE(!input_names.empty());
  TRT_ENSURE(!output_names.empty());
  TRT_ENSURE(!input_tensors.empty());
  TRT_ENSURE(input_tensors.size() == input_names.size());

  std::vector<std::pair<std::string, tensorflow::Tensor>> input_pairs;
  std::vector<std::string> prefixed_output_names;
  auto prefixed_name = [](std::string prefix, std::string name) {
    return prefix.size() > 0 ? absl::StrJoin({prefix, name}, "/") : name;
  };
  for (int i = 0; i < input_names.size(); i++) {
    input_pairs.push_back(
        {prefixed_name(prefix, input_names.at(i)), input_tensors.at(i)});
  }
  for (int i = 0; i < output_names.size(); i++) {
    prefixed_output_names.push_back(prefixed_name(prefix, output_names.at(i)));
  }
  std::vector<tensorflow::Tensor> output_tensors;
  for (int i = 0; i < output_names.size(); i++) {
    output_tensors.push_back({});
  }
  TF_RETURN_IF_ERROR(
      session->Run(input_pairs, prefixed_output_names, {}, &output_tensors));
  return OkStatus();
}

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT