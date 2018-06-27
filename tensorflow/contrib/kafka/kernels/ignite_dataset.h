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

#include "tensorflow/core/framework/dataset.h"

namespace ignite {

class IgniteDataset : public tensorflow::GraphDatasetBase {
 public:
  IgniteDataset(tensorflow::OpKernelContext* ctx, std::string cache_name, std::string host, tensorflow::int32 port, bool local, tensorflow::int32 part, tensorflow::int32 page_size, std::vector<tensorflow::int32> schema, std::vector<tensorflow::int32> permutation);
  std::unique_ptr<tensorflow::IteratorBase> MakeIteratorInternal(const tensorflow::string& prefix) const override;
  const tensorflow::DataTypeVector& output_dtypes() const override;
  const std::vector<tensorflow::PartialTensorShape>& output_shapes() const override;
  tensorflow::string DebugString() const override;

 protected:
  tensorflow::Status AsGraphDefInternal(DatasetGraphDefBuilder* b, tensorflow::Node** output) const override;

 private:
  const std::string cache_name_;
  const std::string host_;
  const tensorflow::int32 port_;
  const bool local_;
  const tensorflow::int32 part_;
  const tensorflow::int32 page_size_;
  const std::vector<tensorflow::int32> schema_;
  const std::vector<tensorflow::int32> permutation_;
};

} // namespace ignite
