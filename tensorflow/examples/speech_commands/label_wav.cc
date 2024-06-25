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

#include <fstream>
#include <vector>

#include "absl/status/status.h"
#include "xla/tsl/util/command_line_flags.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tsl/platform/env.h"
#include "tsl/platform/status.h"
#include "tsl/platform/types.h"

// These are all common classes it's handy to reference with no namespace.
using tensorflow::Flag;
using tensorflow::int32;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::Tensor;
using tensorflow::tstring;

namespace {

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status LoadGraph(const string& graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session) {
  tensorflow::GraphDef graph_def;
  Status load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                        graph_file_name, "'");
  }
  session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
  Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return absl::OkStatus();
}

// Takes a file name, and loads a list of labels from it, one per line, and
// returns a vector of the strings.
Status ReadLabelsFile(const string& file_name, std::vector<string>* result) {
  std::ifstream file(file_name);
  if (!file) {
    return tensorflow::errors::NotFound("Labels file ", file_name,
                                        " not found.");
  }
  result->clear();
  string line;
  while (std::getline(file, line)) {
    result->push_back(line);
  }
  return absl::OkStatus();
}

// Analyzes the output of the graph to retrieve the highest scores and
// their positions in the tensor.
void GetTopLabels(const std::vector<Tensor>& outputs, int how_many_labels,
                  Tensor* out_indices, Tensor* out_scores) {
  const Tensor& unsorted_scores_tensor = outputs[0];
  auto unsorted_scores_flat = unsorted_scores_tensor.flat<float>();
  std::vector<std::pair<int, float>> scores;
  scores.reserve(unsorted_scores_flat.size());
  for (int i = 0; i < unsorted_scores_flat.size(); ++i) {
    scores.push_back(std::pair<int, float>({i, unsorted_scores_flat(i)}));
  }
  std::sort(scores.begin(), scores.end(),
            [](const std::pair<int, float>& left,
               const std::pair<int, float>& right) {
              return left.second > right.second;
            });
  scores.resize(how_many_labels);
  Tensor sorted_indices(tensorflow::DT_INT32, {how_many_labels});
  Tensor sorted_scores(tensorflow::DT_FLOAT, {how_many_labels});
  for (int i = 0; i < scores.size(); ++i) {
    sorted_indices.flat<int>()(i) = scores[i].first;
    sorted_scores.flat<float>()(i) = scores[i].second;
  }
  *out_indices = sorted_indices;
  *out_scores = sorted_scores;
}

}  // namespace

int main(int argc, char* argv[]) {
  string wav = "";
  string graph = "";
  string labels = "";
  string input_name = "wav_data";
  string output_name = "labels_softmax";
  int32_t how_many_labels = 3;
  std::vector<Flag> flag_list = {
      Flag("wav", &wav, "audio file to be identified"),
      Flag("graph", &graph, "model to be executed"),
      Flag("labels", &labels, "path to file containing labels"),
      Flag("input_name", &input_name, "name of input node in model"),
      Flag("output_name", &output_name, "name of output node in model"),
      Flag("how_many_labels", &how_many_labels, "number of results to show"),
  };
  string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    LOG(ERROR) << usage;
    return -1;
  }

  // We need to call this to set up global state for TensorFlow.
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    return -1;
  }

  // First we load and initialize the model.
  std::unique_ptr<tensorflow::Session> session;
  Status load_graph_status = LoadGraph(graph, &session);
  if (!load_graph_status.ok()) {
    LOG(ERROR) << load_graph_status;
    return -1;
  }

  std::vector<string> labels_list;
  Status read_labels_status = ReadLabelsFile(labels, &labels_list);
  if (!read_labels_status.ok()) {
    LOG(ERROR) << read_labels_status;
    return -1;
  }

  string wav_string;
  Status read_wav_status = tensorflow::ReadFileToString(
      tensorflow::Env::Default(), wav, &wav_string);
  if (!read_wav_status.ok()) {
    LOG(ERROR) << read_wav_status;
    return -1;
  }
  Tensor wav_tensor(tensorflow::DT_STRING, tensorflow::TensorShape({}));
  wav_tensor.scalar<tstring>()() = wav_string;

  // Actually run the audio through the model.
  std::vector<Tensor> outputs;
  Status run_status =
      session->Run({{input_name, wav_tensor}}, {output_name}, {}, &outputs);
  if (!run_status.ok()) {
    LOG(ERROR) << "Running model failed: " << run_status;
    return -1;
  }

  Tensor indices;
  Tensor scores;
  GetTopLabels(outputs, how_many_labels, &indices, &scores);
  tensorflow::TTypes<float>::Flat scores_flat = scores.flat<float>();
  tensorflow::TTypes<int32>::Flat indices_flat = indices.flat<int32>();
  for (int pos = 0; pos < how_many_labels; ++pos) {
    const int label_index = indices_flat(pos);
    const float score = scores_flat(pos);
    LOG(INFO) << labels_list[label_index] << " (" << label_index
              << "): " << score;
  }

  return 0;
}
