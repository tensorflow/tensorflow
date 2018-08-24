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
#include "ignite_dataset.h"

#ifndef IGNITE_CLIENT_H
#define IGNITE_CLIENT_H
#include "ignite_client.h"
#endif

namespace ignite {

class IgniteDatasetIterator
    : public tensorflow::DatasetIterator<IgniteDataset> {
 public:
  IgniteDatasetIterator(const Params& params, std::string host,
                        tensorflow::int32 port, std::string cache_name,
                        bool local, tensorflow::int32 part,
                        tensorflow::int32 page_size, std::string username,
                        std::string password, std::string certfile,
                        std::string keyfile, std::string cert_password,
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
  std::unique_ptr<Client> client;
  BinaryObjectParser parser;

  const std::string cache_name;
  const bool local;
  const tensorflow::int32 part;
  const tensorflow::int32 page_size;
  const std::string username;
  const std::string password;
  const std::vector<tensorflow::int32> schema;
  const std::vector<tensorflow::int32> permutation;

  int32_t remainder;
  int64_t cursor_id;
  bool last_page;

  std::unique_ptr<uint8_t> page;
  uint8_t* ptr;

  tensorflow::Status EstablishConnection();
  tensorflow::Status CloseConnection();
  tensorflow::Status Handshake();
  tensorflow::Status ScanQuery();
  tensorflow::Status LoadNextPage();
  int32_t JavaHashCode(std::string str);
};

constexpr uint8_t null_val = 101;
constexpr uint8_t string_val = 9;
constexpr uint8_t protocol_major_version = 1;
constexpr uint8_t protocol_minor_version = 1;
constexpr uint8_t protocol_patch_version = 0;
constexpr int16_t scan_query_opcode = 2000;
constexpr int16_t load_next_page_opcode = 2001;
constexpr int16_t close_connection_opcode = 0;

}  // namespace ignite
