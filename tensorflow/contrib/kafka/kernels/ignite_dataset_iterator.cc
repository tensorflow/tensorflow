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

IgniteDatasetIterator::IgniteDatasetIterator(const Params& params) : tensorflow::DatasetIterator<IgniteDataset>(params) {
	// this->client = client;	  	
	// this->cache_name = cache_name;
	// this->local = local;
	// this->part = part;
	// this->reminder = -1;
	// this->last_page = false;
	// this->parser = new IgniteBinaryObjectParser();
}

IgniteDatasetIterator::~IgniteDatasetIterator() {
	// delete parser;
}

tensorflow::Status IgniteDatasetIterator::GetNextInternal(tensorflow::IteratorContext* ctx, std::vector<tensorflow::Tensor>* out_tensors, bool* end_of_sequence) {
  // if (reminder == -1) {
  // 	// first query
  // 	client->Connect();
  // 	Handshake();

  // 	// ---------- Scan Query ---------- //
	 //  client->WriteInt(25); // Message length
	 //  client->WriteShort(2000); // Operation code
	 //  client->WriteLong(42); // Request id
	 //  client->WriteInt(JavaHashCode(cache_name));
	 //  client->WriteByte(0); // Some flags...
	 //  client->WriteByte(101); // Filter object (NULL).
	 //  client->WriteInt(1); // Cursor page size
	 //  client->WriteInt(-1); // Partition to query
	 //  client->WriteByte(0); // Local flag

	 //  int res_len = ReadInt();
	 //  long req_id = ReadLong();
	 //  int status = ReadInt();

	 //  if (status != 0) {
	 //  	std::cout << "Scan Query status error\n";
	 //  }

	 //  cursor_id = client->ReadLong();
	 //  int row_cnt = client->ReadInt();
	  
	 //  int data_len = res_len - 8 - 4 - 8 - 4 - 1;

	 //  char* data = (char*) malloc(data_len);
	 //  client->ReadData(data, data_len);

	 //  next_page = client->ReadByte() != 0;

	 //  parser->Parse(data, types, out_tensors);
	 //  parser->Parse(data, types, out_tensors);
  // }
  // else if (reminder == 0) {
  // 	if (last_page) {
  // 		*end_of_sequence = true;
  // 		return tensorflow::Status::OK();
  // 	}
  // 	else {
  // 		// query next page 

  // 		*end_of_sequence = false;
  // 		return tensorflow::Status::OK();
  // 	}
  // }
  *end_of_sequence = true;
  return tensorflow::Status::OK();
}

tensorflow::Status IgniteDatasetIterator::SaveInternal(tensorflow::IteratorStateWriter* writer) {
  return tensorflow::Status::OK();
}

tensorflow::Status IgniteDatasetIterator::RestoreInternal(tensorflow::IteratorContext* ctx, tensorflow::IteratorStateReader* reader) {
  return tensorflow::Status::OK();
}

// bool IgniteDatasetIterator::Handshake() {
//   WriteInt(8);
//   WriteByte(1);
//   WriteShort(1);
//   WriteShort(0);
//   WriteShort(0);
//   WriteByte(2);

//   int handshake_res_len = ReadInt();
//   char handshake_res = ReadByte();

//   if (handshake_res == 1) {
//   	return true;
//   }
//   else {
//   	std::cout << "Handshake error!\n";
//   	return false;
//   }
// }

// int IgniteDatasetIterator::JavaHashCode(std::string str) {
//   int h = 0;
//   for (char &c : str) {
//     h = 31 * h + c;
//   }
//   return h;
// }

} // namespace ignite