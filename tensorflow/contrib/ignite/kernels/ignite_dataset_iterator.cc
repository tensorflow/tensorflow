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

#include <time.h>

namespace ignite {

IgniteDatasetIterator::IgniteDatasetIterator(
    const Params& params, std::string host, tensorflow::int32 port,
    std::string cache_name, bool local, tensorflow::int32 part,
    tensorflow::int32 page_size, std::vector<tensorflow::int32> schema,
    std::vector<tensorflow::int32> permutation)
    : tensorflow::DatasetIterator<IgniteDataset>(params),
      client(Client(host, port)),
      cache_name(cache_name),
      local(local),
      part(part),
      page_size(page_size),
      schema(schema),
      permutation(permutation),
      remainder(-1),
      cursor_id(-1),
      last_page(false) {
  LOG(INFO) << "Ignite Dataset Iterator created";
}

IgniteDatasetIterator::~IgniteDatasetIterator() { 
  tensorflow::Status status = CloseConnection();
  if (!status.ok())
    LOG(ERROR) << status.ToString();

  LOG(INFO) << "Ignite Dataset Iterator destroyed";
}

tensorflow::Status IgniteDatasetIterator::EstablishConnection() {
  if (!client.IsConnected()) {
     tensorflow::Status status = client.Connect();
     if (!status.ok())
       return status;

     status = Handshake();
     if (!status.ok()) {
       tensorflow::Status disconnect_status = client.Disconnect();
       if (!disconnect_status.ok())
         LOG(ERROR) << disconnect_status.ToString();

       return status;
     }
   }

  return tensorflow::Status::OK();
}

tensorflow::Status IgniteDatasetIterator::CloseConnection() {
  if (cursor_id != -1) {
    tensorflow::Status conn_status = EstablishConnection();
    if (!conn_status.ok())
      return conn_status;

    client.WriteInt(18);         // Message length
    client.WriteShort(0);        // Operation code
    client.WriteLong(0);         // Request ID
    client.WriteLong(cursor_id); // Resource ID

    int res_len = client.ReadInt();
    if (res_len < 12)
      return tensorflow::errors::Internal("Close Resource Response is corrupted");

    long req_id = client.ReadLong();
    int status = client.ReadInt();

    if (status != 0) {
      char err_msg_header = client.ReadByte();
      if (err_msg_header == 9) {
        int err_msg_length = client.ReadInt();
        char* err_msg = new char[err_msg_length];
        client.ReadData(err_msg, err_msg_length);
        return tensorflow::errors::Internal("Close Resource Error [status=", status, ", message=", std::string(err_msg, err_msg_length), "]");
      }
      return tensorflow::errors::Internal("Close Resource Error [status=", status, "]");
    }

    LOG(INFO) << "Query Cursor " << cursor_id << " is closed";

    cursor_id = -1;

    return client.Disconnect();
  }
  
  return client.IsConnected() ? client.Disconnect() : tensorflow::Status::OK();
}

tensorflow::Status IgniteDatasetIterator::GetNextInternal(
    tensorflow::IteratorContext* ctx,
    std::vector<tensorflow::Tensor>* out_tensors, bool* end_of_sequence) {
  if (remainder == 0 && last_page) {
    LOG(INFO) << "Query Cursor " << cursor_id << " is closed";

    cursor_id = -1;
    *end_of_sequence = true;
    return tensorflow::Status::OK();
  } else {
    tensorflow::Status status = EstablishConnection();
    if (!status.ok())
      return status;

    if (remainder == -1 || remainder == 0) {
      tensorflow::Status status = remainder == -1 ? ScanQuery() : LoadNextPage();
      if (!status.ok())
        return status;
    }

    char* initial_ptr = ptr;
    std::vector<int> types;
    std::vector<tensorflow::Tensor> tensors;

    ptr = parser.Parse(ptr, &tensors, &types);  // Parse key
    ptr = parser.Parse(ptr, &tensors, &types);  // Parse val
    remainder -= (ptr - initial_ptr);

    out_tensors->resize(tensors.size());
    for (int i = 0; i < tensors.size(); i++)
      (*out_tensors)[permutation[i]] = std::move(tensors[i]);

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

tensorflow::Status IgniteDatasetIterator::Handshake() {
  client.WriteInt(8);
  client.WriteByte(1);
  client.WriteShort(1);
  client.WriteShort(0);
  client.WriteShort(0);
  client.WriteByte(2);

  int handshake_res_len = client.ReadInt();
  char handshake_res = client.ReadByte();

  if (handshake_res != 1) {
    return tensorflow::errors::Internal("Handshake Error [status=", handshake_res, "]");
  }

  return tensorflow::Status::OK();
}

tensorflow::Status IgniteDatasetIterator::ScanQuery() {
  client.WriteInt(25);                         // Message length
  client.WriteShort(2000);                     // Operation code
  client.WriteLong(0);                         // Request ID
  client.WriteInt(JavaHashCode(cache_name));   // Cache name
  client.WriteByte(0);                         // Flags
  client.WriteByte(101);                       // Filter object
  client.WriteInt(page_size);                  // Cursor page size
  client.WriteInt(part);                       // Partition to query
  client.WriteByte(local);                     // Local flag

  int res_len = client.ReadInt();
  if (res_len < 12)
    return tensorflow::errors::Internal("Scan Query Response is corrupted");

  long req_id = client.ReadLong();
  int status = client.ReadInt();

  if (status != 0) {
    char err_msg_header = client.ReadByte();
    if (err_msg_header == 9) {
      int err_msg_length = client.ReadInt();
      char* err_msg = new char[err_msg_length];
      client.ReadData(err_msg, err_msg_length);
      return tensorflow::errors::Internal("Scan Query Error [status=", status, ", message=", std::string(err_msg, err_msg_length), "]");
    } 
    return tensorflow::errors::Internal("Scan Query Error [status=", status, "]");
  }

  cursor_id = client.ReadLong();
 
  LOG(INFO) << "Query Cursor " << cursor_id << " is opened";

  int row_cnt = client.ReadInt();

  remainder = res_len - 25;
  page = std::unique_ptr<char>(new char[remainder]);
  ptr = page.get();
  clock_t start = clock();
  client.ReadData(ptr, remainder);
  clock_t stop = clock();

  double size_in_mb = 1.0 * remainder / 1024 / 1024;
  double time_in_s = (stop - start) / (double) CLOCKS_PER_SEC;
  LOG(INFO) << "Page size " << size_in_mb << " Mb, time " << time_in_s * 1000
  <<  " ms download speed " << size_in_mb / time_in_s << " Mb/sec";

  last_page = !client.ReadByte();
  
  return tensorflow::Status::OK();
}

tensorflow::Status IgniteDatasetIterator::LoadNextPage() {
  client.WriteInt(18);          // Message length
  client.WriteShort(2001);      // Operation code
  client.WriteLong(0);          // Request ID
  client.WriteLong(cursor_id);  // Cursor ID

  int res_len = client.ReadInt();
  if (res_len < 12)
    return tensorflow::errors::Internal("Load Next Page Response is corrupted");

  long req_id = client.ReadLong();
  int status = client.ReadInt();

  if (status != 0) {
    char err_msg_header = client.ReadByte();
    if (err_msg_header == 9) {
      int err_msg_length = client.ReadInt();
      char* err_msg = new char[err_msg_length];
      client.ReadData(err_msg, err_msg_length);
      return tensorflow::errors::Internal("Load Next Page Error [status=", status, ", message=", std::string(err_msg, err_msg_length), "]");
    }
    return tensorflow::errors::Internal("Load Next Page Error [status=", status, "]");
  }

  int row_cnt = client.ReadInt();

  remainder = res_len - 17;
  page = std::unique_ptr<char>(new char[remainder]);
  ptr = page.get();
  clock_t start = clock();
  client.ReadData(ptr, remainder);
  clock_t stop = clock();

  double size_in_mb = 1.0 * remainder / 1024 / 1024;
  double time_in_s = (stop - start) / (double) CLOCKS_PER_SEC;
  LOG(INFO) << "Page size " << size_in_mb << " Mb, time " << time_in_s * 1000
  <<  " ms download speed " << size_in_mb / time_in_s << " Mb/sec";

  last_page = !client.ReadByte();

  return tensorflow::Status::OK();
}

int IgniteDatasetIterator::JavaHashCode(std::string str) {
  int h = 0;
  for (char& c : str) {
    h = 31 * h + c;
  }
  return h;
}

}  // namespace ignite
