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

#ifndef TENSORFLOW_CONTRIB_IGNITE_KERNELS_DATASET_IGNITE_DATASET_ITERATOR_H_
#define TENSORFLOW_CONTRIB_IGNITE_KERNELS_DATASET_IGNITE_DATASET_ITERATOR_H_

#include "tensorflow/contrib/ignite/kernels/client/ignite_client.h"
#include "tensorflow/contrib/ignite/kernels/dataset/ignite_binary_object_parser.h"
#include "tensorflow/contrib/ignite/kernels/dataset/ignite_dataset.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {

class IgniteDatasetIterator : public DatasetIterator<IgniteDataset> {
 public:
  IgniteDatasetIterator(const Params& params, string host, int32 port,
                        string cache_name, bool local, int32 part,
                        int32 page_size, string username, string password,
                        string certfile, string keyfile, string cert_password,
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
  Status GetNextInternalWithValidState(IteratorContext* ctx,
                                       std::vector<Tensor>* out_tensors,
                                       bool* end_of_sequence);

  Status EstablishConnection();
  Status CloseConnection();
  Status Handshake();
  Status ScanQuery();
  Status LoadNextPage();
  Status ReceivePage(int32_t page_size);
  Status CheckTypes(const std::vector<int32_t>& types);
  int32_t JavaHashCode(string str) const;

  std::unique_ptr<Client> client_;
  BinaryObjectParser parser_;

  const string cache_name_;
  const bool local_;
  const int32 part_;
  const int32 page_size_;
  const string username_;
  const string password_;
  const std::vector<int32> schema_;
  const std::vector<int32> permutation_;

  int32_t remainder_;
  int64_t cursor_id_;
  bool last_page_;

  bool valid_state_;

  mutex mutex_;

  std::unique_ptr<uint8_t> page_;
  uint8_t* ptr_;
};

constexpr uint8_t kNullVal = 101;
constexpr uint8_t kStringVal = 9;
constexpr uint8_t kProtocolMajorVersion = 1;
constexpr uint8_t kProtocolMinorVersion = 1;
constexpr uint8_t kProtocolPatchVersion = 0;
constexpr int16_t kScanQueryOpcode = 2000;
constexpr int16_t kLoadNextPageOpcode = 2001;
constexpr int16_t kCloseConnectionOpcode = 0;
constexpr int32_t kScanQueryReqLength = 25;
constexpr int32_t kScanQueryResHeaderLength = 25;
constexpr int32_t kLoadNextPageReqLength = 18;
constexpr int32_t kLoadNextPageResHeaderLength = 17;
constexpr int32_t kCloseConnectionReqLength = 18;
constexpr int32_t kHandshakeReqDefaultLength = 8;
constexpr int32_t kMinResLength = 12;

}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_IGNITE_KERNELS_DATASET_IGNITE_DATASET_ITERATOR_H_
