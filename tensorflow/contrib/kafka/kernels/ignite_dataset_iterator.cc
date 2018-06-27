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
#include "ignite_dataset_iterator.h"

namespace ignite {

IgniteDatasetIterator::IgniteDatasetIterator(const Params& params, std::string host, tensorflow::int32 port, std::string cache_name, bool local, tensorflow::int32 part, tensorflow::int32 page_size, std::vector<tensorflow::int32> schema, std::vector<tensorflow::int32> permutation) : tensorflow::DatasetIterator<IgniteDataset>(params),
 client_(Client(host, port)),
 cache_name_(cache_name),
 local_(local),
 part_(part),
 page_size_(page_size),
 schema_(schema),
 permutation_(permutation),
 remainder(-1),
 last_page(false) {
  client_.Connect();
  std::cout << "Client connected!" << std::endl;
  Handshake();
 }

IgniteDatasetIterator::~IgniteDatasetIterator() {
  client_.Disconnect();
  std::cout << "Client disconnected!" << std::endl;
}

tensorflow::Status IgniteDatasetIterator::GetNextInternal(tensorflow::IteratorContext* ctx, std::vector<tensorflow::Tensor>* out_tensors, bool* end_of_sequence) {
  if (remainder == 0 && last_page) {
    *end_of_sequence = true;
    return tensorflow::Status::OK();
  }
  else {
    if (remainder == 0) {
      // query next page
      client_.WriteInt(18);
      client_.WriteShort(2001); 
      client_.WriteLong(0); // Request id
      client_.WriteLong(cursor_id); // Cursor ID

      int res_len = client_.ReadInt();
      long req_id = client_.ReadLong();
      int status = client_.ReadInt();

      if (status != 0) {
        std::cout << "Query next page status error\n";
      }

      int row_cnt = client_.ReadInt();

      remainder = res_len - 17;
      data = (char*) malloc(remainder);
      client_.ReadData(data, remainder);
      last_page = !client_.ReadByte();
    }
    if (remainder == -1) {
      // ---------- Scan Query ---------- //
      client_.WriteInt(25); // Message length
      client_.WriteShort(2000); // Operation code
      client_.WriteLong(0); // Request id
      client_.WriteInt(JavaHashCode(cache_name_));
      client_.WriteByte(0); // Some flags...
      client_.WriteByte(101); // Filter object (NULL).
      client_.WriteInt(page_size_); // Cursor page size
      client_.WriteInt(part_); // Partition to query
      client_.WriteByte(local_); // Local flag

      int res_len = client_.ReadInt();
      long req_id = client_.ReadLong();
      int status = client_.ReadInt();

      if (status != 0) {
        std::cout << "Scan Query status error\n";
      }

      cursor_id = client_.ReadLong();
      int row_cnt = client_.ReadInt();
      
      remainder = res_len - 25;
      data = (char*) malloc(remainder);
      client_.ReadData(data, remainder);
      last_page = !client_.ReadByte();
    }

    char* initial_ptr = data;
    std::vector<int>* types = new std::vector<int>();
    std::vector<tensorflow::Tensor>* tensors = new std::vector<tensorflow::Tensor>();

    BinaryObjectParser parser;
    // Parse key 
    data = parser.Parse(data, tensors, types);
    // Parse val
    data = parser.Parse(data, tensors, types);

    remainder -= (data - initial_ptr);

    out_tensors->resize(tensors->size());

    for (int i = 0; i < tensors->size(); i++) {
      int idx = permutation_[i];
      auto a = (*tensors)[i];
      (*out_tensors)[idx] = a;
    }

    *end_of_sequence = false;
    return tensorflow::Status::OK();
  }


  *end_of_sequence = true;
  return tensorflow::Status::OK();
}

tensorflow::Status IgniteDatasetIterator::SaveInternal(tensorflow::IteratorStateWriter* writer) {
  return tensorflow::Status::OK();
}

tensorflow::Status IgniteDatasetIterator::RestoreInternal(tensorflow::IteratorContext* ctx, tensorflow::IteratorStateReader* reader) {
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

  if (handshake_res == 1) {
    std::cout << "Handshake passed\n";
  }
  else {
    std::cout << "Handshake error!\n";
  }
}

int IgniteDatasetIterator::JavaHashCode(std::string str) {
  int h = 0;
  for (char &c : str) {
    h = 31 * h + c;
  }
  return h;
}

} // namespace ignite
