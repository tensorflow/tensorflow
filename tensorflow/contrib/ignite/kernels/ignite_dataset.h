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

#include "tensorflow/core/framework/dataset.h"

namespace tensorflow {

class IgniteDataset : public DatasetBase {
 public:
  IgniteDataset(OpKernelContext* ctx, std::string cache_name, std::string host,
                int32 port, bool local, int32 part, int32 page_size,
                std::string username, std::string password,
                std::string certfile, std::string keyfile,
                std::string cert_password, std::vector<int32> schema,
                std::vector<int32> permutation);
  ~IgniteDataset();
  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override;
  const DataTypeVector& output_dtypes() const override;
  const std::vector<PartialTensorShape>& output_shapes() const override;
  string DebugString() const override;

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override;

 private:
  const std::string cache_name_;
  const std::string host_;
  const int32 port_;
  const bool local_;
  const int32 part_;
  const int32 page_size_;
  const std::string username_;
  const std::string password_;
  const std::string certfile_;
  const std::string keyfile_;
  const std::string cert_password_;
  const std::vector<int32> schema_;
  const std::vector<int32> permutation_;

  DataTypeVector dtypes_;
  std::vector<PartialTensorShape> shapes_;

  void SchemaToTypes();
  void SchemaToShapes();
};

}  // namespace tensorflow
