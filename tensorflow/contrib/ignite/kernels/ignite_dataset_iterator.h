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

#include "ignite_binary_object_parser.h"
#include "ignite_client.h"
#include "ignite_dataset.h"

namespace tensorflow {

class IgniteDatasetIterator : public DatasetIterator<IgniteDataset> {
 public:
  IgniteDatasetIterator(const Params& params, std::string host, int32 port,
                        std::string cache_name, bool local, int32 part,
                        int32 page_size, std::string username,
                        std::string password, std::string certfile,
                        std::string keyfile, std::string cert_password,
                        std::vector<int32> schema,
                        std::vector<int32> permutation);
  ~IgniteDatasetIterator();
  Status GetNextInternal(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                         bool* end_of_sequence) override;

 protected:
  Status SaveInternal(IteratorStateWriter* writer) override;
  Status RestoreInternal(IteratorContext* ctx,
                         IteratorStateReader* reader) override;

 private:
  std::unique_ptr<Client> client_;
  BinaryObjectParser parser_;

  const std::string cache_name_;
  const bool local_;
  const int32 part_;
  const int32 page_size_;
  const std::string username_;
  const std::string password_;
  const std::vector<int32> schema_;
  const std::vector<int32> permutation_;

  int32_t remainder_;
  int64_t cursor_id_;
  bool last_page_;

  std::unique_ptr<uint8_t> page_;
  uint8_t* ptr_;

  Status EstablishConnection();
  Status CloseConnection();
  Status Handshake();
  Status ScanQuery();
  Status LoadNextPage();
  int32_t JavaHashCode(std::string str) const;
};

constexpr uint8_t null_val = 101;
constexpr uint8_t string_val = 9;
constexpr uint8_t protocol_major_version = 1;
constexpr uint8_t protocol_minor_version = 1;
constexpr uint8_t protocol_patch_version = 0;
constexpr int16_t scan_query_opcode = 2000;
constexpr int16_t load_next_page_opcode = 2001;
constexpr int16_t close_connection_opcode = 0;

}  // namespace tensorflow
