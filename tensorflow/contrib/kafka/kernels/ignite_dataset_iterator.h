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

#include "ignite_dataset.h"
#include "ignite_client.h"
// #include "ignite_binary_object_parser.h"

namespace ignite {

class IgniteDatasetIterator : public tensorflow::DatasetIterator<IgniteDataset> {
 public:
  explicit IgniteDatasetIterator(const Params& params, std::string host, tensorflow::int32 port, std::string cache_name, bool local, tensorflow::int32 part, std::vector<tensorflow::int32> schema, std::vector<tensorflow::int32> permutation);
  ~IgniteDatasetIterator();
  tensorflow::Status GetNextInternal(tensorflow::IteratorContext* ctx, std::vector<tensorflow::Tensor>* out_tensors, bool* end_of_sequence) override;

 protected:
  tensorflow::Status SaveInternal(tensorflow::IteratorStateWriter* writer) override;
  tensorflow::Status RestoreInternal(tensorflow::IteratorContext* ctx, tensorflow::IteratorStateReader* reader) override;

 private:
  void Handshake();
  int JavaHashCode(std::string str);

  Client client_;
  std::string cache_name_;
  bool local_;
  tensorflow::int32 part_;
  std::vector<tensorflow::int32> schema_;
  std::vector<tensorflow::int32> permutation_;

  char* data;
  int remainder;
  bool last_page;
  long cursor_id;
};

} // namespace ignite
