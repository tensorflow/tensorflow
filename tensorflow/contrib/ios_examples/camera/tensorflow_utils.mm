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

#import <Foundation/Foundation.h>

#include "tensorflow_utils.h"

#include <fstream>
#include <pthread.h>
#include <unistd.h>
#include <queue>
#include <sstream>
#include <string>

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"
#include "google/protobuf/message_lite.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"


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
      return ifs_.gcount();
    }
    
  private:
    std::ifstream ifs_;
  };
}  // namespace

// Returns the top N confidence values over threshold in the provided vector,
// sorted by confidence in descending order.
void GetTopN(const Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>,
             Eigen::Aligned>& prediction, const int num_results,
             const float threshold,
             std::vector<std::pair<float, int> >* top_results) {
  // Will contain top N results in ascending order.
  std::priority_queue<std::pair<float, int>,
  std::vector<std::pair<float, int> >,
  std::greater<std::pair<float, int> > > top_result_pq;
  
  const int count = prediction.size();
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
  ::google::protobuf::io::CodedInputStream coded_stream(&stream);
  // Total bytes hard limit / warning limit are set to 1GB and 512MB
  // respectively.
  coded_stream.SetTotalBytesLimit(1024LL << 20, 512LL << 20);
  return proto->ParseFromCodedStream(&coded_stream);
}

NSString* FilePathForResourceName(NSString* name, NSString* extension) {
  NSString* file_path = [[NSBundle mainBundle] pathForResource:name ofType:extension];
  if (file_path == NULL) {
    LOG(FATAL) << "Couldn't find '" << [name UTF8String] << "."
    << [extension UTF8String] << "' in bundle.";
    return nullptr;
  }
  return file_path;
}

tensorflow::Status LoadModel(NSString* file_name, NSString* file_type,
                             std::unique_ptr<tensorflow::Session>* session) {
  tensorflow::SessionOptions options;
  
  tensorflow::Session* session_pointer = nullptr;
  tensorflow::Status session_status = tensorflow::NewSession(options, &session_pointer);
  if (!session_status.ok()) {
    LOG(ERROR) << "Could not create Tensorflow Session: " << session_status;
    return session_status;
  }
  session->reset(session_pointer);
  LOG(INFO) << "Session created.";
  
  tensorflow::GraphDef tensorflow_graph;
  LOG(INFO) << "Graph created.";
  
  NSString* model_path = FilePathForResourceName(file_name, file_type);
  if (!model_path) {
    LOG(ERROR) << "Failed to find model proto at" << [file_name UTF8String]
               << [file_type UTF8String];
    return tensorflow::errors::NotFound([file_name UTF8String],
                                        [file_type UTF8String]);
  }
  const bool read_proto_succeeded = PortableReadFileToProto(
    [model_path UTF8String], &tensorflow_graph);
  if (!read_proto_succeeded) {
    LOG(ERROR) << "Failed to load model proto from" << [model_path UTF8String];
    return tensorflow::errors::NotFound([model_path UTF8String]);
  }
  
  LOG(INFO) << "Creating session.";
  tensorflow::Status create_status = (*session)->Create(tensorflow_graph);
  if (!create_status.ok()) {
    LOG(ERROR) << "Could not create Tensorflow Graph: " << create_status;
    return create_status;
  }
  
  return tensorflow::Status::OK();
}

tensorflow::Status LoadLabels(NSString* file_name, NSString* file_type,
                                std::vector<std::string>* label_strings) {
  // Read the label list
  NSString* labels_path = FilePathForResourceName(file_name, file_type);
  if (!labels_path) {
    LOG(ERROR) << "Failed to find model proto at" << [file_name UTF8String]
    << [file_type UTF8String];
    return tensorflow::errors::NotFound([file_name UTF8String],
                                        [file_type UTF8String]);
  }
  std::ifstream t;
  t.open([labels_path UTF8String]);
  std::string line;
  while(t){
    std::getline(t, line);
    label_strings->push_back(line);
  }
  t.close();
  return tensorflow::Status::OK();
}