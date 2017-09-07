// Copyright 2015 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#import "BenchmarkViewController.h"

#include <pthread.h>
#include <sys/time.h>
#include <unistd.h>
#include <fstream>
#include <queue>
#include <sstream>
#include <string>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/stat_summarizer.h"

#include "ios_image_load.h"

NSString* RunInferenceOnImage();

namespace {
class IfstreamInputStream : public ::google::protobuf::io::CopyingInputStream {
 public:
  explicit IfstreamInputStream(const std::string& file_name)
      : ifs_(file_name.c_str(), std::ios::in | std::ios::binary) {}
  ~IfstreamInputStream() { ifs_.close(); }

  int Read(void* buffer, int size) {
    if (!ifs_) {
      return -1;
    }
    ifs_.read(static_cast<char*>(buffer), size);
    return (int)ifs_.gcount();
  }

 private:
  std::ifstream ifs_;
};
}  // namespace

@interface BenchmarkViewController ()
@end

@implementation BenchmarkViewController {
}

- (IBAction)getUrl:(id)sender {
  NSString* inference_result = RunInferenceOnImage();
  self.urlContentTextView.text = inference_result;
}

@end

// Returns the top N confidence values over threshold in the provided vector,
// sorted by confidence in descending order.
static void GetTopN(
    const Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>,
                           Eigen::Aligned>& prediction,
    const int num_results, const float threshold,
    std::vector<std::pair<float, int>>* top_results) {
  // Will contain top N results in ascending order.
  std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>,
                      std::greater<std::pair<float, int>>>
      top_result_pq;

  long count = prediction.size();
  for (int i = 0; i < count; ++i) {
    const float value = prediction(i);

    // Only add it if it beats the threshold and has a chance at being in
    // the top N.
    if (value < threshold) {
      continue;
    }

    top_result_pq.push(std::pair<float, int>(value, i));

    // If at capacity, kick the smallest value out.
    if (top_result_pq.size() > num_results) {
      top_result_pq.pop();
    }
  }

  // Copy to output vector and reverse into descending order.
  while (!top_result_pq.empty()) {
    top_results->push_back(top_result_pq.top());
    top_result_pq.pop();
  }
  std::reverse(top_results->begin(), top_results->end());
}

bool PortableReadFileToProto(const std::string& file_name,
                             ::google::protobuf::MessageLite* proto) {
  ::google::protobuf::io::CopyingInputStreamAdaptor stream(
      new IfstreamInputStream(file_name));
  stream.SetOwnsCopyingStream(true);
  // TODO(jiayq): the following coded stream is for debugging purposes to allow
  // one to parse arbitrarily large messages for MessageLite. One most likely
  // doesn't want to put protobufs larger than 64MB on Android, so we should
  // eventually remove this and quit loud when a large protobuf is passed in.
  ::google::protobuf::io::CodedInputStream coded_stream(&stream);
  // Total bytes hard limit / warning limit are set to 1GB and 512MB
  // respectively.
  coded_stream.SetTotalBytesLimit(1024LL << 20, 512LL << 20);
  return proto->ParseFromCodedStream(&coded_stream);
}

NSString* FilePathForResourceName(NSString* name, NSString* extension) {
  NSString* file_path =
      [[NSBundle mainBundle] pathForResource:name ofType:extension];
  if (file_path == NULL) {
    LOG(FATAL) << "Couldn't find '" << [name UTF8String] << "."
               << [extension UTF8String] << "' in bundle.";
  }
  return file_path;
}

// A utility function to get the current time in seconds, for simple profiling.
double time() {
  timeval t;
  gettimeofday(&t, nullptr);
  return t.tv_sec + 1e-6 * t.tv_usec;
}

// Runs the session with profiling enabled, and prints out details of the time
// that each node in the graph takes to the debug log.
tensorflow::Status BenchmarkInference(
    tensorflow::Session* session,
    const std::vector<std::pair<tensorflow::string, tensorflow::Tensor>> inputs,
    const std::vector<tensorflow::string>& output_layer_names,
    std::vector<tensorflow::Tensor>* output_layers,
    tensorflow::StatSummarizer* stat_summarizer, double* average_time) {
  tensorflow::Status run_status;
  const int iterations_count = 20;
  double total_time = 0.0;
  tensorflow::RunOptions run_options;
  run_options.set_trace_level(tensorflow::RunOptions::FULL_TRACE);
  tensorflow::RunMetadata run_metadata;
  for (int iteration = 0; iteration < (iterations_count + 1); ++iteration) {
    const double start_time = time();
    run_status = session->Run(run_options, inputs, output_layer_names, {},
                              output_layers, &run_metadata);
    const double end_time = time();
    if (iteration != 0) {
      total_time += end_time - start_time;
    }
    if (!run_status.ok()) {
      LOG(ERROR) << "Running model failed: " << run_status;
      tensorflow::LogAllRegisteredKernels();
      return run_status;
    }
  }
  assert(run_metadata.has_step_stats());
  const tensorflow::StepStats& step_stats = run_metadata.step_stats();
  stat_summarizer->ProcessStepStats(step_stats);
  stat_summarizer->PrintStepStats();

  *average_time = total_time / iterations_count;
  NSLog(@"Took %f seconds", *average_time);

  return tensorflow::Status::OK();
}

NSString* RunInferenceOnImage() {
  tensorflow::SessionOptions options;

  tensorflow::Session* session_pointer = nullptr;
  tensorflow::Status session_status =
      tensorflow::NewSession(options, &session_pointer);
  if (!session_status.ok()) {
    std::string status_string = session_status.ToString();
    return [NSString
        stringWithFormat:@"Session create failed - %s", status_string.c_str()];
  }
  std::unique_ptr<tensorflow::Session> session(session_pointer);
  LOG(INFO) << "Session created.";

  tensorflow::GraphDef tensorflow_graph;
  LOG(INFO) << "Graph created.";

  NSString* network_path =
      FilePathForResourceName(@"tensorflow_inception_graph", @"pb");
  PortableReadFileToProto([network_path UTF8String], &tensorflow_graph);

  LOG(INFO) << "Creating session.";
  tensorflow::Status s = session->Create(tensorflow_graph);
  if (!s.ok()) {
    LOG(ERROR) << "Could not create TensorFlow Graph: " << s;
    return @"";
  }

  // Read the label list
  NSString* labels_path =
      FilePathForResourceName(@"imagenet_comp_graph_label_strings", @"txt");
  std::vector<std::string> label_strings;
  std::ifstream t;
  t.open([labels_path UTF8String]);
  std::string line;
  while (t) {
    std::getline(t, line);
    label_strings.push_back(line);
  }
  t.close();

  // Read the Grace Hopper image.
  NSString* image_path = FilePathForResourceName(@"grace_hopper", @"jpg");
  int image_width;
  int image_height;
  int image_channels;
  std::vector<tensorflow::uint8> image_data = LoadImageFromFile(
      [image_path UTF8String], &image_width, &image_height, &image_channels);
  const int wanted_width = 224;
  const int wanted_height = 224;
  const int wanted_channels = 3;
  const float input_mean = 117.0f;
  const float input_std = 1.0f;
  assert(image_channels >= wanted_channels);
  tensorflow::Tensor image_tensor(
      tensorflow::DT_FLOAT,
      tensorflow::TensorShape(
          {1, wanted_height, wanted_width, wanted_channels}));
  auto image_tensor_mapped = image_tensor.tensor<float, 4>();
  tensorflow::uint8* in = image_data.data();
  float* out = image_tensor_mapped.data();
  for (int y = 0; y < wanted_height; ++y) {
    const int in_y = (y * image_height) / wanted_height;
    tensorflow::uint8* in_row = in + (in_y * image_width * image_channels);
    float* out_row = out + (y * wanted_width * wanted_channels);
    for (int x = 0; x < wanted_width; ++x) {
      const int in_x = (x * image_width) / wanted_width;
      tensorflow::uint8* in_pixel = in_row + (in_x * image_channels);
      float* out_pixel = out_row + (x * wanted_channels);
      for (int c = 0; c < wanted_channels; ++c) {
        out_pixel[c] = (in_pixel[c] - input_mean) / input_std;
      }
    }
  }
  tensorflow::string input_layer = "input";
  tensorflow::string output_layer = "output";
  std::vector<tensorflow::Tensor> outputs;
  tensorflow::StatSummarizer stat_summarizer(tensorflow_graph);
  double average_time = 0.0;
  BenchmarkInference(session.get(), {{input_layer, image_tensor}},
                     {output_layer}, &outputs, &stat_summarizer, &average_time);
  NSString* result =
      [NSString stringWithFormat:@"Average time: %.4f seconds \n\n", average_time];

  tensorflow::Tensor* output = &outputs[0];
  const int kNumResults = 5;
  const float kThreshold = 0.1f;
  std::vector<std::pair<float, int>> top_results;
  GetTopN(output->flat<float>(), kNumResults, kThreshold, &top_results);

  std::stringstream ss;
  ss.precision(3);
  for (const auto& result : top_results) {
    const float confidence = result.first;
    const int index = result.second;

    ss << index << " " << confidence << " ";

    // Write out the result as a string
    if (index < label_strings.size()) {
      // just for safety: theoretically, the output is under 1000 unless there
      // is some numerical issues leading to a wrong prediction.
      ss << label_strings[index];
    } else {
      ss << "Prediction: " << index;
    }

    ss << "\n";
  }

  LOG(INFO) << "Predictions: " << ss.str();

  tensorflow::string predictions = ss.str();
  result = [NSString stringWithFormat:@"%@ - %s", result, predictions.c_str()];

  return result;
}
