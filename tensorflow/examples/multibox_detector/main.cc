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

#include <setjmp.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

// These are all common classes it's handy to reference with no namespace.
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;
using tensorflow::uint8;

// Takes a file name, and loads a list of comma-separated box priors from it,
// one per line, and returns a vector of the values.
Status ReadLocationsFile(const string& file_name, std::vector<float>* result,
                         size_t* found_label_count) {
  std::ifstream file(file_name);
  if (!file) {
    return tensorflow::errors::NotFound("Labels file ", file_name,
                                        " not found.");
  }
  result->clear();
  string line;
  while (std::getline(file, line)) {
    std::vector<string> string_tokens = tensorflow::str_util::Split(line, ',');
    result->reserve(string_tokens.size());
    for (const string& string_token : string_tokens) {
      float number;
      CHECK(tensorflow::strings::safe_strtof(string_token, &number));
      result->push_back(number);
    }
  }
  *found_label_count = result->size();
  return absl::OkStatus();
}

// Given an image file name, read in the data, try to decode it as an image,
// resize it to the requested size, and then scale the values as desired.
Status ReadTensorFromImageFile(const string& file_name, const int input_height,
                               const int input_width, const float input_mean,
                               const float input_std,
                               std::vector<Tensor>* out_tensors) {
  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  string input_name = "file_reader";
  string original_name = "identity";
  string output_name = "normalized";
  auto file_reader =
      tensorflow::ops::ReadFile(root.WithOpName(input_name), file_name);
  // Now try to figure out what kind of file it is and decode it.
  const int wanted_channels = 3;
  tensorflow::Output image_reader;
  if (tensorflow::str_util::EndsWith(file_name, ".png")) {
    image_reader = DecodePng(root.WithOpName("png_reader"), file_reader,
                             DecodePng::Channels(wanted_channels));
  } else if (tensorflow::str_util::EndsWith(file_name, ".gif")) {
    image_reader = DecodeGif(root.WithOpName("gif_reader"), file_reader);
  } else {
    // Assume if it's neither a PNG nor a GIF then it must be a JPEG.
    image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
                              DecodeJpeg::Channels(wanted_channels));
  }

  // Also return identity so that we can know the original dimensions and
  // optionally save the image out with bounding boxes overlaid.
  auto original_image = Identity(root.WithOpName(original_name), image_reader);

  // Now cast the image data to float so we can do normal math on it.
  auto float_caster = Cast(root.WithOpName("float_caster"), original_image,
                           tensorflow::DT_FLOAT);
  // The convention for image ops in TensorFlow is that all images are expected
  // to be in batches, so that they're four-dimensional arrays with indices of
  // [batch, height, width, channel]. Because we only have a single image, we
  // have to add a batch dimension of 1 to the start with ExpandDims().
  auto dims_expander = ExpandDims(root, float_caster, 0);

  // Bilinearly resize the image to fit the required dimensions.
  auto resized = ResizeBilinear(
      root, dims_expander,
      Const(root.WithOpName("size"), {input_height, input_width}));
  // Subtract the mean and divide by the scale.
  Div give_me_a_name(root.WithOpName(output_name),
                     Sub(root, resized, {input_mean}), {input_std});

  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensor.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  TF_RETURN_IF_ERROR(
      session->Run({}, {output_name, original_name}, {}, out_tensors));
  return absl::OkStatus();
}

Status SaveImage(const Tensor& tensor, const string& file_path) {
  LOG(INFO) << "Saving image to " << file_path;
  CHECK(tensorflow::str_util::EndsWith(file_path, ".png"))
      << "Only saving of png files is supported.";

  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  string encoder_name = "encode";
  string output_name = "file_writer";

  tensorflow::Output image_encoder =
      EncodePng(root.WithOpName(encoder_name), tensor);
  tensorflow::ops::WriteFile file_saver = tensorflow::ops::WriteFile(
      root.WithOpName(output_name), file_path, image_encoder);

  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  std::vector<Tensor> outputs;
  TF_RETURN_IF_ERROR(session->Run({}, {}, {output_name}, &outputs));

  return absl::OkStatus();
}

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

// Analyzes the output of the MultiBox graph to retrieve the highest scores and
// their positions in the tensor, which correspond to individual box detections.
Status GetTopDetections(const std::vector<Tensor>& outputs, int how_many_labels,
                        Tensor* indices, Tensor* scores) {
  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  string output_name = "top_k";
  TopK give_me_a_name(root.WithOpName(output_name), outputs[0],
                      how_many_labels);
  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensors.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  // The TopK node returns two outputs, the scores and their original indices,
  // so we have to append :0 and :1 to specify them both.
  std::vector<Tensor> out_tensors;
  TF_RETURN_IF_ERROR(session->Run({}, {output_name + ":0", output_name + ":1"},
                                  {}, &out_tensors));
  *scores = out_tensors[0];
  *indices = out_tensors[1];
  return absl::OkStatus();
}

// Converts an encoded location to an actual box placement with the provided
// box priors.
void DecodeLocation(const float* encoded_location, const float* box_priors,
                    float* decoded_location) {
  bool non_zero = false;
  for (int i = 0; i < 4; ++i) {
    const float curr_encoding = encoded_location[i];
    non_zero = non_zero || curr_encoding != 0.0f;

    const float mean = box_priors[i * 2];
    const float std_dev = box_priors[i * 2 + 1];

    float currentLocation = curr_encoding * std_dev + mean;

    currentLocation = std::max(currentLocation, 0.0f);
    currentLocation = std::min(currentLocation, 1.0f);
    decoded_location[i] = currentLocation;
  }

  if (!non_zero) {
    LOG(WARNING) << "No non-zero encodings; check log for inference errors.";
  }
}

float DecodeScore(float encoded_score) {
  return 1 / (1 + std::exp(-encoded_score));
}

void DrawBox(const int image_width, const int image_height, int left, int top,
             int right, int bottom, tensorflow::TTypes<uint8>::Flat* image) {
  tensorflow::TTypes<uint8>::Flat image_ref = *image;

  top = std::max(0, std::min(image_height - 1, top));
  bottom = std::max(0, std::min(image_height - 1, bottom));

  left = std::max(0, std::min(image_width - 1, left));
  right = std::max(0, std::min(image_width - 1, right));

  for (int i = 0; i < 3; ++i) {
    uint8 val = i == 2 ? 255 : 0;
    for (int x = left; x <= right; ++x) {
      image_ref((top * image_width + x) * 3 + i) = val;
      image_ref((bottom * image_width + x) * 3 + i) = val;
    }
    for (int y = top; y <= bottom; ++y) {
      image_ref((y * image_width + left) * 3 + i) = val;
      image_ref((y * image_width + right) * 3 + i) = val;
    }
  }
}

// Given the output of a model run, and the name of a file containing the labels
// this prints out the top five highest-scoring values.
Status PrintTopDetections(const std::vector<Tensor>& outputs,
                          const string& labels_file_name,
                          const int num_boxes,
                          const int num_detections,
                          const string& image_file_name,
                          Tensor* original_tensor) {
  std::vector<float> locations;
  size_t label_count;
  Status read_labels_status =
      ReadLocationsFile(labels_file_name, &locations, &label_count);
  if (!read_labels_status.ok()) {
    LOG(ERROR) << read_labels_status;
    return read_labels_status;
  }
  CHECK_EQ(label_count, num_boxes * 8);

  const int how_many_labels =
      std::min(num_detections, static_cast<int>(label_count));
  Tensor indices;
  Tensor scores;
  TF_RETURN_IF_ERROR(
      GetTopDetections(outputs, how_many_labels, &indices, &scores));

  tensorflow::TTypes<float>::Flat scores_flat = scores.flat<float>();

  tensorflow::TTypes<int32>::Flat indices_flat = indices.flat<int32>();

  const Tensor& encoded_locations = outputs[1];
  auto locations_encoded = encoded_locations.flat<float>();

  LOG(INFO) << original_tensor->DebugString();
  const int image_width = original_tensor->shape().dim_size(1);
  const int image_height = original_tensor->shape().dim_size(0);

  tensorflow::TTypes<uint8>::Flat image_flat = original_tensor->flat<uint8>();

  LOG(INFO) << "===== Top " << how_many_labels << " Detections ======";
  for (int pos = 0; pos < how_many_labels; ++pos) {
    const int label_index = indices_flat(pos);
    const float score = scores_flat(pos);

    float decoded_location[4];
    DecodeLocation(&locations_encoded(label_index * 4),
                   &locations[label_index * 8], decoded_location);

    float left = decoded_location[0] * image_width;
    float top = decoded_location[1] * image_height;
    float right = decoded_location[2] * image_width;
    float bottom = decoded_location[3] * image_height;

    LOG(INFO) << "Detection " << pos << ": "
              << "L:" << left << " "
              << "T:" << top << " "
              << "R:" << right << " "
              << "B:" << bottom << " "
              << "(" << label_index << ") score: " << DecodeScore(score);

    DrawBox(image_width, image_height, left, top, right, bottom, &image_flat);
  }

  if (!image_file_name.empty()) {
    return SaveImage(*original_tensor, image_file_name);
  }
  return absl::OkStatus();
}

int main(int argc, char* argv[]) {
  // These are the command-line flags the program can understand.
  // They define where the graph and input data is located, and what kind of
  // input the model expects. If you train your own model, or use something
  // other than multibox_model you'll need to update these.
  string image =
      "tensorflow/examples/multibox_detector/data/surfers.jpg";
  string graph =
      "tensorflow/examples/multibox_detector/data/"
      "multibox_model.pb";
  string box_priors =
      "tensorflow/examples/multibox_detector/data/"
      "multibox_location_priors.txt";
  int32_t input_width = 224;
  int32_t input_height = 224;
  int32_t input_mean = 128;
  int32_t input_std = 128;
  int32_t num_detections = 5;
  int32_t num_boxes = 784;
  string input_layer = "ResizeBilinear";
  string output_location_layer = "output_locations/Reshape";
  string output_score_layer = "output_scores/Reshape";
  string root_dir = "";
  string image_out = "";

  std::vector<Flag> flag_list = {
      Flag("image", &image, "image to be processed"),
      Flag("image_out", &image_out,
           "location to save output image, if desired"),
      Flag("graph", &graph, "graph to be executed"),
      Flag("box_priors", &box_priors, "name of file containing box priors"),
      Flag("input_width", &input_width, "resize image to this width in pixels"),
      Flag("input_height", &input_height,
           "resize image to this height in pixels"),
      Flag("input_mean", &input_mean, "scale pixel values to this mean"),
      Flag("input_std", &input_std, "scale pixel values to this std deviation"),
      Flag("num_detections", &num_detections,
           "number of top detections to return"),
      Flag("num_boxes", &num_boxes,
           "number of boxes defined by the location file"),
      Flag("input_layer", &input_layer, "name of input layer"),
      Flag("output_location_layer", &output_location_layer,
           "name of location output layer"),
      Flag("output_score_layer", &output_score_layer,
           "name of score output layer"),
      Flag("root_dir", &root_dir,
           "interpret image and graph file names relative to this directory"),
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
  string graph_path = tensorflow::io::JoinPath(root_dir, graph);
  Status load_graph_status = LoadGraph(graph_path, &session);
  if (!load_graph_status.ok()) {
    LOG(ERROR) << load_graph_status;
    return -1;
  }

  // Get the image from disk as a float array of numbers, resized and normalized
  // to the specifications the main graph expects.
  std::vector<Tensor> image_tensors;
  string image_path = tensorflow::io::JoinPath(root_dir, image);

  Status read_tensor_status =
      ReadTensorFromImageFile(image_path, input_height, input_width, input_mean,
                              input_std, &image_tensors);
  if (!read_tensor_status.ok()) {
    LOG(ERROR) << read_tensor_status;
    return -1;
  }
  const Tensor& resized_tensor = image_tensors[0];

  // Actually run the image through the model.
  std::vector<Tensor> outputs;
  Status run_status =
      session->Run({{input_layer, resized_tensor}},
                   {output_score_layer, output_location_layer}, {}, &outputs);
  if (!run_status.ok()) {
    LOG(ERROR) << "Running model failed: " << run_status;
    return -1;
  }

  Status print_status = PrintTopDetections(outputs, box_priors, num_boxes,
                                           num_detections, image_out,
                                           &image_tensors[1]);

  if (!print_status.ok()) {
    LOG(ERROR) << "Running print failed: " << print_status;
    return -1;
  }
  return 0;
}
