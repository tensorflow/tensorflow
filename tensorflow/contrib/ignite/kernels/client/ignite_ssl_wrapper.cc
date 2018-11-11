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

#include "tensorflow/contrib/ignite/kernels/client/ignite_ssl_wrapper.h"

#include <openssl/err.h>
#include <openssl/ssl.h>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

static int PasswordCb(char *buf, int size, int rwflag, void *password) {
  strncpy(buf, (char *)(password), size);
  buf[size - 1] = '\0';
  return (strlen(buf));
}

SslWrapper::SslWrapper(std::shared_ptr<Client> client, string certfile,
                       string keyfile, string cert_password, bool big_endian)
    : Client(big_endian),
      client_(client),
      certfile_(std::move(certfile)),
      keyfile_(std::move(keyfile)),
      cert_password_(std::move(cert_password)),
      ctx_(nullptr),
      ssl_(nullptr) {}

SslWrapper::~SslWrapper() {
  if (IsConnected()) {
    Status status = Disconnect();
    if (!status.ok()) LOG(WARNING) << status.ToString();
  }

  if (ctx_ != nullptr) {
    SSL_CTX_free(ctx_);
    ctx_ = nullptr;
  }

  if (ssl_ != nullptr) {
    SSL_free(ssl_);
    ssl_ = nullptr;
  }
}

Status SslWrapper::InitSslContext() {
  OpenSSL_add_all_algorithms();
  SSL_load_error_strings();

  ctx_ = SSL_CTX_new(SSLv23_method());
  if (ctx_ == NULL) return errors::Internal("Couldn't create SSL context");

  SSL_CTX_set_default_passwd_cb(ctx_, PasswordCb);
  SSL_CTX_set_default_passwd_cb_userdata(ctx_, (void *)cert_password_.c_str());

  if (SSL_CTX_use_certificate_chain_file(ctx_, certfile_.c_str()) != 1)
    return errors::Internal("Couldn't load cetificate chain (file '", certfile_,
                            "')");

  string private_key_file = keyfile_.empty() ? certfile_ : keyfile_;
  if (SSL_CTX_use_PrivateKey_file(ctx_, private_key_file.c_str(),
                                  SSL_FILETYPE_PEM) != 1)
    return errors::Internal("Couldn't load private key (file '",
                            private_key_file, "')");

  return Status::OK();
}

Status SslWrapper::Connect() {
  if (ctx_ == NULL) {
    TF_RETURN_IF_ERROR(InitSslContext());
  }

  ssl_ = SSL_new(ctx_);
  if (ssl_ == NULL)
    return errors::Internal("Failed to establish SSL connection");

  TF_RETURN_IF_ERROR(client_->Connect());

  SSL_set_fd(ssl_, client_->GetSocketDescriptor());
  if (SSL_connect(ssl_) != 1)
    return errors::Internal("Failed to establish SSL connection");

  LOG(INFO) << "SSL connection established";

  return Status::OK();
}

Status SslWrapper::Disconnect() {
  SSL_free(ssl_);
  ssl_ = nullptr;

  LOG(INFO) << "SSL connection closed";

  return client_->Disconnect();
}

bool SslWrapper::IsConnected() { return client_->IsConnected(); }

int SslWrapper::GetSocketDescriptor() { return client_->GetSocketDescriptor(); }

Status SslWrapper::ReadData(uint8_t *buf, const int32_t length) {
  int received = 0;

  while (received < length) {
    int res = SSL_read(ssl_, buf, length - received);

    if (res < 0)
      return errors::Internal("Error occurred while reading from SSL socket: ",
                              res);

    if (res == 0) return errors::Internal("Server closed SSL connection");

    received += res;
    buf += res;
  }

  return Status::OK();
}

Status SslWrapper::WriteData(const uint8_t *buf, const int32_t length) {
  int sent = 0;

  while (sent < length) {
    int res = SSL_write(ssl_, buf, length - sent);

    if (res < 0)
      return errors::Internal("Error occurred while writing into socket: ",
                              res);

    sent += res;
    buf += res;
  }

  return Status::OK();
}

}  // namespace tensorflow
