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

#include "tensorflow/contrib/ignite/kernels/dataset/ignite_dataset_iterator.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

IgniteDataset::IgniteDataset(OpKernelContext* ctx, string cache_name,
                             string host, int32 port, bool local, int32 part,
                             int32 page_size, string username, string password,
                             string certfile, string keyfile,
                             string cert_password, std::vector<int32> schema,
                             std::vector<int32> permutation,
                             DataTypeVector dtypes,
                             std::vector<PartialTensorShape> shapes)
    : DatasetBase(DatasetContext(ctx)),
      cache_name_(std::move(cache_name)),
      host_(std::move(host)),
      port_(port),
      local_(local),
      part_(part),
      page_size_(page_size),
      username_(std::move(username)),
      password_(std::move(password)),
      certfile_(std::move(certfile)),
      keyfile_(std::move(keyfile)),
      cert_password_(std::move(cert_password)),
      schema_(std::move(schema)),
      permutation_(std::move(permutation)),
      dtypes_(dtypes),
      shapes_(shapes) {
  LOG(INFO) << "Ignite Dataset created [cache_name='" << cache_name_
            << "', host='" << host_ << "', port=" << port_
            << ", local=" << local_ << ", part=" << part_
            << ", page_size=" << page_size_ << ", username='" << username_
            << "', certfile='" << certfile_ << "', keyfile='"
            << keyfile_ + "']";
}

IgniteDataset::~IgniteDataset() { LOG(INFO) << "Ignite Dataset destroyed"; }

std::unique_ptr<IteratorBase> IgniteDataset::MakeIteratorInternal(
    const string& prefix) const {
  return std::unique_ptr<IteratorBase>(new IgniteDatasetIterator(
      {this, strings::StrCat(prefix, "::Ignite")}, std::move(this->host_),
      this->port_, std::move(this->cache_name_), this->local_, this->part_,
      this->page_size_, std::move(this->username_), std::move(this->password_),
      std::move(this->certfile_), std::move(this->keyfile_),
      std::move(this->cert_password_), std::move(this->schema_),
      std::move(this->permutation_)));
}

const DataTypeVector& IgniteDataset::output_dtypes() const { return dtypes_; }

const std::vector<PartialTensorShape>& IgniteDataset::output_shapes() const {
  return shapes_;
}

string IgniteDataset::DebugString() const { return "IgniteDatasetOp::Dataset"; }

Status IgniteDataset::AsGraphDefInternal(SerializationContext* ctx,
                                         DatasetGraphDefBuilder* b,
                                         Node** output) const {
  return errors::Unimplemented(
      "IgniteDataset does not support 'AsGraphDefInternal'");
}

}  // namespace tensorflow
