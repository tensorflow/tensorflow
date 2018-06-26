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
// #include "ignite_client.h"

namespace ignite {

IgniteDatasetIterator::IgniteDatasetIterator(const Params& params, std::string host, tensorflow::int32 port, std::string cache_name, bool local, tensorflow::int32 part, std::vector<tensorflow::int32> schema) : tensorflow::DatasetIterator<IgniteDataset>(params),
 client_(Client(host, port)),
 cache_name_(cache_name),
 local_(local),
 part_(part),
 schema_(schema),
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
  if (reminder == 0) {
  	if (last_page) {
  		*end_of_sequence = true;
  		return tensorflow::Status::OK();
  	}
  	else {
  		// query next page 
  		*end_of_sequence = false;
  		return tensorflow::Status::OK();
  	}
  }
  else {
    if (reminder == -1) {
      // ---------- Scan Query ---------- //
      client->WriteInt(25); // Message length
      client->WriteShort(2000); // Operation code
      client->WriteLong(42); // Request id
      client->WriteInt(JavaHashCode(cache_name));
      client->WriteByte(0); // Some flags...
      client->WriteByte(101); // Filter object (NULL).
      client->WriteInt(1); // Cursor page size
      client->WriteInt(-1); // Partition to query
      client->WriteByte(0); // Local flag

      int res_len = ReadInt();
      long req_id = ReadLong();
      int status = ReadInt();

      if (status != 0) {
        std::cout << "Scan Query status error\n";
      }

      cursor_id = client->ReadLong();
      int row_cnt = client->ReadInt();
      
      std::cout << "Row count: " << row_cnt << std::endl;

      remainder = res_len - 25;
      data = (char*) malloc(remainder);
      client->ReadData(data, remainder);

      next_page = client->ReadByte() != 0;
    }

    srd::cout << "Remainder: " << remainder << std::endl;

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
