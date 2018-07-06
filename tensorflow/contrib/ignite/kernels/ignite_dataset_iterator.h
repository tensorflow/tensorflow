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

#include "ignite_binary_object_parser.h"
#include "ignite_client.h"
#include "ignite_dataset.h"

namespace ignite {

class IgniteDatasetIterator
    : public tensorflow::DatasetIterator<IgniteDataset> {
 public:
  IgniteDatasetIterator(const Params& params, std::string host,
                                 tensorflow::int32 port, std::string cache_name,
                                 bool local, tensorflow::int32 part,
                                 tensorflow::int32 page_size,
                                 std::vector<tensorflow::int32> schema,
                                 std::vector<tensorflow::int32> permutation);
  ~IgniteDatasetIterator();
  tensorflow::Status GetNextInternal(
      tensorflow::IteratorContext* ctx,
      std::vector<tensorflow::Tensor>* out_tensors,
      bool* end_of_sequence) override;

 protected:
  tensorflow::Status SaveInternal(
      tensorflow::IteratorStateWriter* writer) override;
  tensorflow::Status RestoreInternal(
      tensorflow::IteratorContext* ctx,
      tensorflow::IteratorStateReader* reader) override;

 private:
  Client client;
  BinaryObjectParser parser;

  const std::string cache_name;
  const bool local;
  const tensorflow::int32 part;
  const tensorflow::int32 page_size;
  const std::vector<tensorflow::int32> schema;
  const std::vector<tensorflow::int32> permutation;

  std::unique_ptr<char> page;
  char* ptr;
  int remainder;
  bool last_page;
  long cursor_id;

  tensorflow::Status EstablishConnection();
  tensorflow::Status CloseConnection();
  tensorflow::Status Handshake();
  tensorflow::Status ScanQuery();
  tensorflow::Status LoadNextPage();
  int JavaHashCode(std::string str);
};

}  // namespace ignite
