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
#include "tensorflow/core/platform/logging.h"

//#include <time.h>

namespace ignite {

IgniteDatasetIterator::IgniteDatasetIterator(
    const Params& params, std::string host, tensorflow::int32 port,
    std::string cache_name, bool local, tensorflow::int32 part,
    tensorflow::int32 page_size, std::vector<tensorflow::int32> schema,
    std::vector<tensorflow::int32> permutation)
    : tensorflow::DatasetIterator<IgniteDataset>(params),
      client_(Client(host, port)),
      cache_name_(cache_name),
      local_(local),
      part_(part),
      page_size_(page_size),
      schema_(schema),
      permutation_(permutation),
      remainder(-1),
      last_page(false) {
  LOG(INFO) << "Ignite Dataset Iterator created";
  client_.Connect();
  Handshake();
}

IgniteDatasetIterator::~IgniteDatasetIterator() { 
  client_.Disconnect(); 
  LOG(INFO) << "Ignite Dataset Iterator destroyed";
}

tensorflow::Status IgniteDatasetIterator::GetNextInternal(
    tensorflow::IteratorContext* ctx,
    std::vector<tensorflow::Tensor>* out_tensors, bool* end_of_sequence) {
  if (remainder == 0 && last_page) {
    *end_of_sequence = true;
    return tensorflow::Status::OK();
  } else {
    if (remainder == -1) ScanQuery();
    if (remainder == 0) LoadNextPage();

    char* initial_ptr = ptr;
    std::vector<int> types;
    std::vector<tensorflow::Tensor> tensors;

    ptr = parser.Parse(ptr, &tensors, &types);  // Parse key
    ptr = parser.Parse(ptr, &tensors, &types);  // Parse val
    remainder -= (ptr - initial_ptr);

    out_tensors->resize(tensors.size());
    for (int i = 0; i < tensors.size(); i++)
      (*out_tensors)[permutation_[i]] = std::move(tensors[i]);

    *end_of_sequence = false;
    return tensorflow::Status::OK();
  }

  *end_of_sequence = true;
  return tensorflow::Status::OK();
}

tensorflow::Status IgniteDatasetIterator::SaveInternal(
    tensorflow::IteratorStateWriter* writer) {
  return tensorflow::Status::OK();
}

tensorflow::Status IgniteDatasetIterator::RestoreInternal(
    tensorflow::IteratorContext* ctx, tensorflow::IteratorStateReader* reader) {
  return tensorflow::Status::OK();
}

void IgniteDatasetIterator::Handshake() {
  client_.WriteInt(8);
  client_.WriteByte(1);
  client_.WriteShort(1);
  client_.WriteShort(0);
  client_.WriteShort(0);
  client_.WriteByte(2);

  int handshake_res_len = client_.ReadInt();
  char handshake_res = client_.ReadByte();

  if (handshake_res != 1) {
    LOG(ERROR) << "Handshake error (status " << handshake_res << ")";
  }
}

void IgniteDatasetIterator::ScanQuery() {
  client_.WriteInt(25);                         // Message length
  client_.WriteShort(2000);                     // Operation code
  client_.WriteLong(0);                         // Request ID
  client_.WriteInt(JavaHashCode(cache_name_));  // Cache name
  client_.WriteByte(0);                         // Flags
  client_.WriteByte(101);                       // Filter object
  client_.WriteInt(page_size_);                 // Cursor page size
  client_.WriteInt(part_);                      // Partition to query
  client_.WriteByte(local_);                    // Local flag

  int res_len = client_.ReadInt();
  long req_id = client_.ReadLong();
  int status = client_.ReadInt();

  if (status != 0) {
    LOG(ERROR) << "Scan Query error (status " << status << ")";
  }

  cursor_id = client_.ReadLong();
  int row_cnt = client_.ReadInt();

  remainder = res_len - 25;
  page = std::unique_ptr<char>(new char[remainder]);
  ptr = page.get();
  // clock_t start = clock();
  client_.ReadData(ptr, remainder);
  // clock_t stop = clock();

  // double size_in_mb = 1.0 * remainder / 1024 / 1024;
  // double time_in_s = (stop - start) / (double) CLOCKS_PER_SEC;
  // std::cout << "Page size " << size_in_mb << " Mb, time " << time_in_s * 1000
  // <<  " ms download speed " << size_in_mb / time_in_s << " Mb/sec" <<
  // std::endl;
  last_page = !client_.ReadByte();
}

void IgniteDatasetIterator::LoadNextPage() {
  client_.WriteInt(18);          // Message length
  client_.WriteShort(2001);      // Operation code
  client_.WriteLong(0);          // Request ID
  client_.WriteLong(cursor_id);  // Cursor ID

  int res_len = client_.ReadInt();
  long req_id = client_.ReadLong();
  int status = client_.ReadInt();

  if (status != 0) {
    LOG(ERROR) << "Query Next Page error (status " << status << ")";
  }

  int row_cnt = client_.ReadInt();

  remainder = res_len - 17;
  page = std::unique_ptr<char>(new char[remainder]);
  ptr = page.get();
  // clock_t start = clock();
  client_.ReadData(ptr, remainder);
  // clock_t stop = clock();

  // double size_in_mb = 1.0 * remainder / 1024 / 1024;
  // double time_in_s = (stop - start) / (double) CLOCKS_PER_SEC;
  // std::cout << "Page size " << size_in_mb << " Mb, time " << time_in_s * 1000
  // <<  " ms download speed " << size_in_mb / time_in_s << " Mb/sec" <<
  // std::endl;
  last_page = !client_.ReadByte();
}

int IgniteDatasetIterator::JavaHashCode(std::string str) {
  int h = 0;
  for (char& c : str) {
    h = 31 * h + c;
  }
  return h;
}

}  // namespace ignite
