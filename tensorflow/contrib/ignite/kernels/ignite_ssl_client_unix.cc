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

#include "ignite_ssl_client.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <map>

#include <iostream>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/lib/core/errors.h"

#include <openssl/ssl.h>
#include <openssl/err.h>

namespace ignite {

static int PasswordCb(char *buf, int size, int rwflag, void *password) {
  strncpy(buf, (char *)(password), size);
  buf[size - 1] = '\0';
  return(strlen(buf));
}

SslClient::SslClient(std::string host, int port, std::string certfile, std::string keyfile, std::string cert_password) :
  host(host),
  port(port),
  certfile(certfile),
  keyfile(keyfile),
  cert_password(cert_password),
  sock(-1),
  ctx(NULL) {}

SslClient::~SslClient() {
  if (ctx != NULL) {
    SSL_CTX_free(ctx);
    ctx = NULL;
  }
}

tensorflow::Status SslClient::InitSslContext() {
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

tensorflow::Status SslClient::Connect() {
  if (sock == -1) {
    sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == -1)
      return tensorflow::errors::Internal("Failed not create socket");
  }

  struct sockaddr_in server;

  if (inet_addr(host.c_str()) == -1) {
    struct hostent *he;
    struct in_addr **addr_list;

    if ((he = gethostbyname(host.c_str())) == NULL)
      return tensorflow::errors::Internal("Failed to resolve hostname \"", host, "\"");

    addr_list = (struct in_addr **)he->h_addr_list;
    for (int i = 0; addr_list[i] != NULL; i++) {
      server.sin_addr = *addr_list[i];
      break;
    }
  } else {
    server.sin_addr.s_addr = inet_addr(host.c_str());
  }

  server.sin_family = AF_INET;
  server.sin_port = htons(port);

  if (ctx == NULL) {
    tensorflow::Status status = InitSslContext();
    if (!status.ok())
      return status;
  }

  if (connect(sock, (struct sockaddr *)&server, sizeof(server)) < 0) 
    return tensorflow::errors::Internal("Failed to connect to \"", host, ":", port, "\"");

  ssl = SSL_new(ctx);
  if (ssl == NULL)
    return tensorflow::errors::Internal("Failed to establish SSL connection to \"", host, ":", port, "\"");

  SSL_set_fd(ssl, sock);
  if (SSL_connect(ssl) != 1)
    return tensorflow::errors::Internal("Failed to establish SSL connection to \"", host, ":", port, "\"");

  LOG(INFO) << "Connection to \"" << host << ":" << port << "\" established";

  return tensorflow::Status::OK();
}

tensorflow::Status SslClient::Disconnect() {
  int close_res = close(sock);
  SSL_free(ssl);
  sock = -1;

  LOG(INFO) << "Connection to \"" << host << ":" << port << "\" is closed";
 
  return close_res == 0 ? tensorflow::Status::OK() : tensorflow::errors::Internal("Failed to correctly close connection");
}

bool SslClient::IsConnected() {
  return sock != -1;
}

char SslClient::ReadByte() {
  char res;
  int a = SSL_read(ssl, &res, 1);
  return res;
}

short SslClient::ReadShort() {
  short res;
  int a = SSL_read(ssl, &res, 2);
  return res;
}

int SslClient::ReadInt() {
  int res;
  int a = SSL_read(ssl, &res, 4);
  return res;
}

long SslClient::ReadLong() {
  long res;
  int a = SSL_read(ssl, &res, 8);
  return res;
}

void SslClient::ReadData(char *buf, int length) {
  int recieved = 0;
  while (recieved < length) {
    int res = SSL_read(ssl, buf, length - recieved);
    recieved += res;
    buf += res;
  }
}

void SslClient::WriteByte(char data) { 
  int res = 0;
  while (res <= 0) {
    res = SSL_write(ssl, &data, 1); 
  }
}

void SslClient::WriteShort(short data) { 
  int res = 0;
  while (res <= 0) {
    res = SSL_write(ssl, &data, 2); 
  }
}

void SslClient::WriteInt(int data) { 
  int res = 0;
  while (res <= 0) {
    res = SSL_write(ssl, &data, 4);
  }
}

void SslClient::WriteLong(long data) {
  int res = 0;
  while (res <= 0) {
    res = SSL_write(ssl, &data, 8); 
  }
}

void SslClient::WriteData(char *buf, int length) { 
  int res = 0;
  while (res <= 0) {
    res = SSL_write(ssl, buf, length); 
  }
}

}  // namespace ignite
