/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_PJRT_DISTRIBUTED_MTLS_H_
#define XLA_PJRT_DISTRIBUTED_MTLS_H_

#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/security/server_credentials.h"

// Utilities for building gRPC mutual-TLS (mTLS) channel and server
// credentials from PEM files. With mTLS, the client and the server each
// present a certificate during the TLS handshake and verify the other's
// against a trusted CA, so the connection is encrypted and both peers are
// authenticated.

namespace xla {

struct MtlsConfig {
  // PEM-encoded identity certificate chain, private key, and trusted root
  // bundle used to verify the peer. All three are required. The files are
  // re-read every `cert_refresh_interval_seconds`, so certificate rotation is
  // picked up without restarting the process.
  std::string cert_file;
  std::string key_file;
  std::string ca_file;

  // Optional additional peer check applied on top of standard X.509 chain
  // verification against `ca_file`.
  //
  // If empty (the default), the client uses gRPC's standard hostname
  // verification: the name the client dials must match a DNS or IP SAN of
  // the server's certificate. The server performs chain verification only.
  //
  // If non-empty, both client and server instead accept the peer iff any URI
  // SAN of its leaf certificate starts with this prefix, e.g. for
  // SPIFFE-style identities whose certificates carry only URI SANs, or when
  // the server is dialed by an IP address its certificate does not name.
  // Must end with '/' (e.g. "spiffe://example.org/", not
  // "spiffe://example.org"), since otherwise "spiffe://example.org.evil/..."
  // would also match; a prefix without the trailing '/' is rejected.
  std::string peer_uri_prefix;

  // How often the credential files are re-read.
  unsigned int cert_refresh_interval_seconds = 600;
};

// Builds mutual-TLS channel credentials for a client from `config`.
absl::StatusOr<std::shared_ptr<::grpc::ChannelCredentials>>
GetMtlsClientCredentials(const MtlsConfig& config);

// Builds mutual-TLS server credentials from `config`.
absl::StatusOr<std::shared_ptr<::grpc::ServerCredentials>>
GetMtlsServerCredentials(const MtlsConfig& config);

}  // namespace xla

#endif  // XLA_PJRT_DISTRIBUTED_MTLS_H_
