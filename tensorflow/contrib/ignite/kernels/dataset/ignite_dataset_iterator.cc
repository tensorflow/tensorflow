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

#include "tensorflow/contrib/ignite/kernels/dataset/ignite_dataset_iterator.h"

#include "tensorflow/contrib/ignite/kernels/client/ignite_plain_client.h"
#include "tensorflow/contrib/ignite/kernels/client/ignite_ssl_wrapper.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

IgniteDatasetIterator::IgniteDatasetIterator(
    const Params& params, string host, int32 port, string cache_name,
    bool local, int32 part, int32 page_size, string username, string password,
    string certfile, string keyfile, string cert_password,
    std::vector<int32> schema, std::vector<int32> permutation)
    : DatasetIterator<IgniteDataset>(params),
      cache_name_(std::move(cache_name)),
      local_(local),
      part_(part),
      page_size_(page_size),
      username_(std::move(username)),
      password_(std::move(password)),
      schema_(std::move(schema)),
      permutation_(std::move(permutation)),
      remainder_(-1),
      cursor_id_(-1),
      last_page_(false),
      valid_state_(true) {
  Client* p_client = new PlainClient(std::move(host), port, false);

  if (certfile.empty())
    client_ = std::unique_ptr<Client>(p_client);
  else
    client_ = std::unique_ptr<Client>(
        new SslWrapper(std::unique_ptr<Client>(p_client), std::move(certfile),
                       std::move(keyfile), std::move(cert_password), false));

  LOG(INFO) << "Ignite Dataset Iterator created";
}

IgniteDatasetIterator::~IgniteDatasetIterator() {
  Status status = CloseConnection();
  if (!status.ok()) LOG(ERROR) << status.ToString();

  LOG(INFO) << "Ignite Dataset Iterator destroyed";
}

Status IgniteDatasetIterator::GetNextInternal(IteratorContext* ctx,
                                              std::vector<Tensor>* out_tensors,
                                              bool* end_of_sequence) {
  mutex_lock l(mutex_);

  if (valid_state_) {
    Status status =
        GetNextInternalWithValidState(ctx, out_tensors, end_of_sequence);

    if (!status.ok()) valid_state_ = false;

    return status;
  }

  return errors::Unknown("Iterator is invalid");
}

Status IgniteDatasetIterator::SaveInternal(IteratorStateWriter* writer) {
  return errors::Unimplemented(
      "Iterator for IgniteDataset does not support 'SaveInternal'");
}

Status IgniteDatasetIterator::RestoreInternal(IteratorContext* ctx,
                                              IteratorStateReader* reader) {
  return errors::Unimplemented(
      "Iterator for IgniteDataset does not support 'RestoreInternal')");
}

Status IgniteDatasetIterator::GetNextInternalWithValidState(
    IteratorContext* ctx, std::vector<Tensor>* out_tensors,
    bool* end_of_sequence) {
  if (remainder_ == 0 && last_page_) {
    cursor_id_ = -1;
    *end_of_sequence = true;

    return Status::OK();
  } else {
    TF_RETURN_IF_ERROR(EstablishConnection());

    if (remainder_ == -1) {
      TF_RETURN_IF_ERROR(ScanQuery());
    } else if (remainder_ == 0) {
      TF_RETURN_IF_ERROR(LoadNextPage());
    }

    uint8_t* initial_ptr = ptr_;
    std::vector<Tensor> tensors;
    std::vector<int32_t> types;

    TF_RETURN_IF_ERROR(parser_.Parse(&ptr_, &tensors, &types));  // Parse key
    TF_RETURN_IF_ERROR(parser_.Parse(&ptr_, &tensors, &types));  // Parse val

    remainder_ -= (ptr_ - initial_ptr);

    TF_RETURN_IF_ERROR(CheckTypes(types));

    for (size_t i = 0; i < tensors.size(); i++)
      out_tensors->push_back(tensors[permutation_[i]]);

    *end_of_sequence = false;

    return Status::OK();
  }

  *end_of_sequence = true;

  return Status::OK();
}

Status IgniteDatasetIterator::EstablishConnection() {
  if (!client_->IsConnected()) {
    TF_RETURN_IF_ERROR(client_->Connect());

    Status status = Handshake();
    if (!status.ok()) {
      Status disconnect_status = client_->Disconnect();
      if (!disconnect_status.ok()) LOG(ERROR) << disconnect_status.ToString();

      return status;
    }
  }

  return Status::OK();
}

Status IgniteDatasetIterator::CloseConnection() {
  if (cursor_id_ != -1 && !last_page_) {
    TF_RETURN_IF_ERROR(EstablishConnection());

    TF_RETURN_IF_ERROR(client_->WriteInt(kCloseConnectionReqLength));
    TF_RETURN_IF_ERROR(client_->WriteShort(kCloseConnectionOpcode));
    TF_RETURN_IF_ERROR(client_->WriteLong(0));           // Request ID
    TF_RETURN_IF_ERROR(client_->WriteLong(cursor_id_));  // Resource ID

    int32_t res_len;
    TF_RETURN_IF_ERROR(client_->ReadInt(&res_len));
    if (res_len < kMinResLength)
      return errors::Unknown("Close Resource Response is corrupted");

    int64_t req_id;
    TF_RETURN_IF_ERROR(client_->ReadLong(&req_id));
    int32_t status;
    TF_RETURN_IF_ERROR(client_->ReadInt(&status));
    if (status != 0) {
      uint8_t err_msg_header;
      TF_RETURN_IF_ERROR(client_->ReadByte(&err_msg_header));
      if (err_msg_header == kStringVal) {
        int32_t err_msg_length;
        TF_RETURN_IF_ERROR(client_->ReadInt(&err_msg_length));

        uint8_t* err_msg_c = new uint8_t[err_msg_length];
        auto clean = gtl::MakeCleanup([err_msg_c] { delete[] err_msg_c; });
        TF_RETURN_IF_ERROR(client_->ReadData(err_msg_c, err_msg_length));
        string err_msg(reinterpret_cast<char*>(err_msg_c), err_msg_length);

        return errors::Unknown("Close Resource Error [status=", status,
                               ", message=", err_msg, "]");
      }
      return errors::Unknown("Close Resource Error [status=", status, "]");
    }

    cursor_id_ = -1;

    return client_->Disconnect();
  } else {
    LOG(INFO) << "Query Cursor " << cursor_id_ << " is already closed";
  }

  return client_->IsConnected() ? client_->Disconnect() : Status::OK();
}

Status IgniteDatasetIterator::Handshake() {
  int32_t msg_len = kHandshakeReqDefaultLength;

  if (username_.empty())
    msg_len += 1;
  else
    msg_len += 5 + username_.length();  // 1 byte header, 4 bytes length.

  if (password_.empty())
    msg_len += 1;
  else
    msg_len += 5 + password_.length();  // 1 byte header, 4 bytes length.

  TF_RETURN_IF_ERROR(client_->WriteInt(msg_len));
  TF_RETURN_IF_ERROR(client_->WriteByte(1));
  TF_RETURN_IF_ERROR(client_->WriteShort(kProtocolMajorVersion));
  TF_RETURN_IF_ERROR(client_->WriteShort(kProtocolMinorVersion));
  TF_RETURN_IF_ERROR(client_->WriteShort(kProtocolPatchVersion));
  TF_RETURN_IF_ERROR(client_->WriteByte(2));
  if (username_.empty()) {
    TF_RETURN_IF_ERROR(client_->WriteByte(kNullVal));
  } else {
    TF_RETURN_IF_ERROR(client_->WriteByte(kStringVal));
    TF_RETURN_IF_ERROR(client_->WriteInt(username_.length()));
    TF_RETURN_IF_ERROR(
        client_->WriteData(reinterpret_cast<const uint8_t*>(username_.c_str()),
                           username_.length()));
  }

  if (password_.empty()) {
    TF_RETURN_IF_ERROR(client_->WriteByte(kNullVal));
  } else {
    TF_RETURN_IF_ERROR(client_->WriteByte(kStringVal));
    TF_RETURN_IF_ERROR(client_->WriteInt(password_.length()));
    TF_RETURN_IF_ERROR(
        client_->WriteData(reinterpret_cast<const uint8_t*>(password_.c_str()),
                           password_.length()));
  }

  int32_t handshake_res_len;
  TF_RETURN_IF_ERROR(client_->ReadInt(&handshake_res_len));
  uint8_t handshake_res;
  TF_RETURN_IF_ERROR(client_->ReadByte(&handshake_res));

  if (handshake_res != 1) {
    int16_t serv_ver_major;
    TF_RETURN_IF_ERROR(client_->ReadShort(&serv_ver_major));
    int16_t serv_ver_minor;
    TF_RETURN_IF_ERROR(client_->ReadShort(&serv_ver_minor));
    int16_t serv_ver_patch;
    TF_RETURN_IF_ERROR(client_->ReadShort(&serv_ver_patch));
    uint8_t header;
    TF_RETURN_IF_ERROR(client_->ReadByte(&header));

    if (header == kStringVal) {
      int32_t length;
      TF_RETURN_IF_ERROR(client_->ReadInt(&length));

      uint8_t* err_msg_c = new uint8_t[length];
      auto clean = gtl::MakeCleanup([err_msg_c] { delete[] err_msg_c; });
      TF_RETURN_IF_ERROR(client_->ReadData(err_msg_c, length));
      string err_msg(reinterpret_cast<char*>(err_msg_c), length);

      return errors::Unknown("Handshake Error [result=", handshake_res,
                             ", version=", serv_ver_major, ".", serv_ver_minor,
                             ".", serv_ver_patch, ", message='", err_msg, "']");
    } else if (header == kNullVal) {
      return errors::Unknown("Handshake Error [result=", handshake_res,
                             ", version=", serv_ver_major, ".", serv_ver_minor,
                             ".", serv_ver_patch, "]");
    } else {
      return errors::Unknown("Handshake Error [result=", handshake_res,
                             ", version=", serv_ver_major, ".", serv_ver_minor,
                             ".", serv_ver_patch, "]");
    }
  }

  return Status::OK();
}

Status IgniteDatasetIterator::ScanQuery() {
  TF_RETURN_IF_ERROR(client_->WriteInt(kScanQueryReqLength));
  TF_RETURN_IF_ERROR(client_->WriteShort(kScanQueryOpcode));
  TF_RETURN_IF_ERROR(client_->WriteLong(0));  // Request ID
  TF_RETURN_IF_ERROR(
      client_->WriteInt(JavaHashCode(cache_name_)));  // Cache name
  TF_RETURN_IF_ERROR(client_->WriteByte(0));          // Flags
  TF_RETURN_IF_ERROR(client_->WriteByte(kNullVal));   // Filter object
  TF_RETURN_IF_ERROR(client_->WriteInt(page_size_));  // Cursor page size
  TF_RETURN_IF_ERROR(client_->WriteInt(part_));       // part_ition to query
  TF_RETURN_IF_ERROR(client_->WriteByte(local_));     // local_ flag

  uint64 wait_start = Env::Default()->NowMicros();
  int32_t res_len;
  TF_RETURN_IF_ERROR(client_->ReadInt(&res_len));
  int64_t wait_stop = Env::Default()->NowMicros();

  LOG(INFO) << "Scan Query waited " << (wait_stop - wait_start) / 1000 << " ms";

  if (res_len < kMinResLength)
    return errors::Unknown("Scan Query Response is corrupted");

  int64_t req_id;
  TF_RETURN_IF_ERROR(client_->ReadLong(&req_id));

  int32_t status;
  TF_RETURN_IF_ERROR(client_->ReadInt(&status));

  if (status != 0) {
    uint8_t err_msg_header;
    TF_RETURN_IF_ERROR(client_->ReadByte(&err_msg_header));

    if (err_msg_header == kStringVal) {
      int32_t err_msg_length;
      TF_RETURN_IF_ERROR(client_->ReadInt(&err_msg_length));

      uint8_t* err_msg_c = new uint8_t[err_msg_length];
      auto clean = gtl::MakeCleanup([err_msg_c] { delete[] err_msg_c; });
      TF_RETURN_IF_ERROR(client_->ReadData(err_msg_c, err_msg_length));
      string err_msg(reinterpret_cast<char*>(err_msg_c), err_msg_length);

      return errors::Unknown("Scan Query Error [status=", status,
                             ", message=", err_msg, "]");
    }
    return errors::Unknown("Scan Query Error [status=", status, "]");
  }

  TF_RETURN_IF_ERROR(client_->ReadLong(&cursor_id_));

  int32_t row_cnt;
  TF_RETURN_IF_ERROR(client_->ReadInt(&row_cnt));

  int32_t page_size = res_len - kScanQueryResHeaderLength;

  return ReceivePage(page_size);
}

Status IgniteDatasetIterator::LoadNextPage() {
  TF_RETURN_IF_ERROR(client_->WriteInt(kLoadNextPageReqLength));
  TF_RETURN_IF_ERROR(client_->WriteShort(kLoadNextPageOpcode));
  TF_RETURN_IF_ERROR(client_->WriteLong(0));           // Request ID
  TF_RETURN_IF_ERROR(client_->WriteLong(cursor_id_));  // Cursor ID

  uint64 wait_start = Env::Default()->NowMicros();
  int32_t res_len;
  TF_RETURN_IF_ERROR(client_->ReadInt(&res_len));
  uint64 wait_stop = Env::Default()->NowMicros();

  LOG(INFO) << "Load Next Page waited " << (wait_stop - wait_start) / 1000
            << " ms";

  if (res_len < kMinResLength)
    return errors::Unknown("Load Next Page Response is corrupted");

  int64_t req_id;
  TF_RETURN_IF_ERROR(client_->ReadLong(&req_id));

  int32_t status;
  TF_RETURN_IF_ERROR(client_->ReadInt(&status));

  if (status != 0) {
    uint8_t err_msg_header;
    TF_RETURN_IF_ERROR(client_->ReadByte(&err_msg_header));

    if (err_msg_header == kStringVal) {
      int32_t err_msg_length;
      TF_RETURN_IF_ERROR(client_->ReadInt(&err_msg_length));

      uint8_t* err_msg_c = new uint8_t[err_msg_length];
      auto clean = gtl::MakeCleanup([err_msg_c] { delete[] err_msg_c; });
      TF_RETURN_IF_ERROR(client_->ReadData(err_msg_c, err_msg_length));
      string err_msg(reinterpret_cast<char*>(err_msg_c), err_msg_length);

      return errors::Unknown("Load Next Page Error [status=", status,
                             ", message=", err_msg, "]");
    }
    return errors::Unknown("Load Next Page Error [status=", status, "]");
  }

  int32_t row_cnt;
  TF_RETURN_IF_ERROR(client_->ReadInt(&row_cnt));

  int32_t page_size = res_len - kLoadNextPageResHeaderLength;

  return ReceivePage(page_size);
}

Status IgniteDatasetIterator::ReceivePage(int32_t page_size) {
  remainder_ = page_size;
  page_ = std::unique_ptr<uint8_t>(new uint8_t[remainder_]);
  ptr_ = page_.get();

  uint64 start = Env::Default()->NowMicros();
  TF_RETURN_IF_ERROR(client_->ReadData(ptr_, remainder_));
  uint64 stop = Env::Default()->NowMicros();

  double size_in_mb = 1.0 * remainder_ / 1024 / 1024;
  double time_in_s = 1.0 * (stop - start) / 1000 / 1000;
  LOG(INFO) << "Page size " << size_in_mb << " Mb, time " << time_in_s * 1000
            << " ms download speed " << size_in_mb / time_in_s << " Mb/sec";

  uint8_t last_page_b;
  TF_RETURN_IF_ERROR(client_->ReadByte(&last_page_b));

  last_page_ = !last_page_b;

  return Status::OK();
}

Status IgniteDatasetIterator::CheckTypes(const std::vector<int32_t>& types) {
  if (schema_.size() != types.size())
    return errors::Unknown("Object has unexpected schema");

  for (size_t i = 0; i < schema_.size(); i++) {
    if (schema_[i] != types[permutation_[i]])
      return errors::Unknown("Object has unexpected schema");
  }

  return Status::OK();
}

int32_t IgniteDatasetIterator::JavaHashCode(string str) const {
  int32_t h = 0;
  for (char& c : str) {
    h = 31 * h + c;
  }
  return h;
}

}  // namespace tensorflow
