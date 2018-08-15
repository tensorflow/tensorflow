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

#include "ignite_ssl_wrapper.h"

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/lib/core/errors.h"

// #include <map>
// #include <iostream>
#include <openssl/ssl.h>
#include <openssl/err.h>

namespace ignite {

static int PasswordCb(char *buf, int size, int rwflag, void *password) {
  strncpy(buf, (char *)(password), size);
  buf[size - 1] = '\0';
  return(strlen(buf));
}

SslWrapper::SslWrapper(std::shared_ptr<Client> client, std::string certfile, std::string keyfile, std::string cert_password) :
  client(client),
  certfile(certfile),
  keyfile(keyfile),
  cert_password(cert_password),
  ctx(NULL) {}

SslWrapper::~SslWrapper() {
  if (IsConnected()) {
    tensorflow::Status status = Disconnect();
    if (!status.ok())
      LOG(WARNING) << status.ToString();
  }

  if (ctx != NULL) {
    SSL_CTX_free(ctx);
    ctx = NULL;
  }
}

tensorflow::Status SslWrapper::InitSslContext() {
  OpenSSL_add_all_algorithms();
  SSL_load_error_strings();

  ctx = SSL_CTX_new(SSLv23_method());
  if (ctx == NULL)
    return tensorflow::errors::Internal("Couldn't create SSL context");

  SSL_CTX_set_default_passwd_cb(ctx, PasswordCb);
  SSL_CTX_set_default_passwd_cb_userdata(ctx, (void*)cert_password.c_str());

  if (SSL_CTX_use_certificate_chain_file(ctx, certfile.c_str()) != 1)
    return tensorflow::errors::Internal("Couldn't load cetificate chain (file '", certfile, "')");

  std::string private_key_file = keyfile.empty() ? certfile : keyfile;
  if (SSL_CTX_use_PrivateKey_file(ctx, private_key_file.c_str(), SSL_FILETYPE_PEM) != 1)
    return tensorflow::errors::Internal("Couldn't load private key (file '", private_key_file, "')");

  return tensorflow::Status::OK();
}

tensorflow::Status SslWrapper::Connect() {
  tensorflow::Status status;

  if (ctx == NULL) {
    status = InitSslContext();
    if (!status.ok())
      return status;
  }

  ssl = SSL_new(ctx);
  if (ssl == NULL)
    return tensorflow::errors::Internal("Failed to establish SSL connection");

  status = client->Connect();
  if (!status.ok())
    return status;

  SSL_set_fd(ssl, client->GetSocketDescriptor());
  if (SSL_connect(ssl) != 1)
    return tensorflow::errors::Internal("Failed to establish SSL connection");

  LOG(INFO) << "SSL connection established";

  return tensorflow::Status::OK();
}

tensorflow::Status SslWrapper::Disconnect() {
  SSL_free(ssl);

  LOG(INFO) << "SSL connection closed";

  return client->Disconnect();
}

bool SslWrapper::IsConnected() {
  return client->IsConnected();
}

int SslWrapper::GetSocketDescriptor() {
  return client->GetSocketDescriptor();
}

char SslWrapper::ReadByte() {
  char res;
  int a = SSL_read(ssl, &res, 1);
  return res;
}

short SslWrapper::ReadShort() {
  short res;
  int a = SSL_read(ssl, &res, 2);
  return res;
}

int SslWrapper::ReadInt() {
  int res;
  int a = SSL_read(ssl, &res, 4);
  return res;
}

long SslWrapper::ReadLong() {
  long res;
  int a = SSL_read(ssl, &res, 8);
  return res;
}

void SslWrapper::ReadData(char *buf, int length) {
  int recieved = 0;

  while (recieved < length) {
    int res = SSL_read(ssl, buf, length - recieved);

    

    recieved += res;
    buf += res;
  }
}

void SslWrapper::WriteByte(char data) { 
  int res = 0;
  while (res <= 0) {
    res = SSL_write(ssl, &data, 1); 
  }
}

void SslWrapper::WriteShort(short data) { 
  int res = 0;
  while (res <= 0) {
    res = SSL_write(ssl, &data, 2); 
  }
}

void SslWrapper::WriteInt(int data) { 
  int res = 0;
  while (res <= 0) {
    res = SSL_write(ssl, &data, 4);
  }
}

void SslWrapper::WriteLong(long data) {
  int res = 0;
  while (res <= 0) {
    res = SSL_write(ssl, &data, 8); 
  }
}

void SslWrapper::WriteData(char *buf, int length) { 
  int res = 0;
  while (res <= 0) {
    res = SSL_write(ssl, buf, length); 
  }
}

}  // namespace ignite
