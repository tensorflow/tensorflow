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

#ifndef TENSORFLOW_CONTRIB_IGNITE_KERNELS_DATASET_IGNITE_DATASET_H_
#define TENSORFLOW_CONTRIB_IGNITE_KERNELS_DATASET_IGNITE_DATASET_H_

#include "tensorflow/core/framework/dataset.h"

namespace tensorflow {

class IgniteDataset : public DatasetBase {
 public:
  IgniteDataset(OpKernelContext* ctx, string cache_name, string host,
                int32 port, bool local, int32 part, int32 page_size,
                string username, string password, string certfile,
                string keyfile, string cert_password, std::vector<int32> schema,
                std::vector<int32> permutation, DataTypeVector dtypes,
                std::vector<PartialTensorShape> shapes);
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
  const string cache_name_;
  const string host_;
  const int32 port_;
  const bool local_;
  const int32 part_;
  const int32 page_size_;
  const string username_;
  const string password_;
  const string certfile_;
  const string keyfile_;
  const string cert_password_;
  const std::vector<int32> schema_;
  const std::vector<int32> permutation_;
  const DataTypeVector dtypes_;
  const std::vector<PartialTensorShape> shapes_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_IGNITE_KERNELS_DATASET_IGNITE_DATASET_H_
