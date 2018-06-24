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

#include "ignite_dataset_iterator.h"
#include "ignite_client.h"

namespace ignite {

IgniteDataset::IgniteDataset(tensorflow::OpKernelContext* ctx, std::string host, tensorflow::int32 port, bool local, tensorflow::int32 part) 
  : GraphDatasetBase(ctx),
    host_(host),
    port_(port),
    local_(local),
    part_(part) {}

std::unique_ptr<tensorflow::IteratorBase> IgniteDataset::MakeIteratorInternal(const tensorflow::string& prefix) const {
  return std::unique_ptr<tensorflow::IteratorBase>(new IgniteDatasetIterator({this, tensorflow::strings::StrCat(prefix, "::Kafka")}));
}

const tensorflow::DataTypeVector& IgniteDataset::output_dtypes() const {
  static tensorflow::DataTypeVector* dtypes = new tensorflow::DataTypeVector({tensorflow::DT_INT32, tensorflow::DT_INT32, tensorflow::DT_DOUBLE});
  return *dtypes;
}

const std::vector<tensorflow::PartialTensorShape>& IgniteDataset::output_shapes() const {
  static std::vector<tensorflow::PartialTensorShape>* shapes =new std::vector<tensorflow::PartialTensorShape>({{}, {}, {784}});
  return *shapes;
}

tensorflow::string IgniteDataset::DebugString() const { 
  return "KafkaDatasetOp::Dataset"; 
}

tensorflow::Status IgniteDataset::AsGraphDefInternal(DatasetGraphDefBuilder* b, tensorflow::Node** output) const {
  return tensorflow::Status::OK();
}

} // namespace ignite