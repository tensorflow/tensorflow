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

#include "xla/pjrt/distributed/mtls.h"

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "grpc/grpc_security_constants.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/security/tls_certificate_provider.h"
#include "grpcpp/security/tls_certificate_verifier.h"
#include "grpcpp/security/tls_credentials_options.h"
#include "grpcpp/support/status.h"
#include "grpcpp/support/string_ref.h"
#include "xla/tsl/platform/env.h"

namespace xla {
namespace {

// An ExternalCertificateVerifier that accepts the peer iff any URI SAN of
// the peer's leaf certificate starts with `prefix`.
class UriSanPrefixVerifier
    : public grpc::experimental::ExternalCertificateVerifier {
 public:
  explicit UriSanPrefixVerifier(std::string prefix)
      : prefix_(std::move(prefix)) {}

  bool Verify(grpc::experimental::TlsCustomVerificationCheckRequest* request,
              std::function<void(grpc::Status)> callback,
              grpc::Status* sync_status) override {
    std::vector<grpc::string_ref> uri_names = request->uri_names();
    for (const grpc::string_ref& uri_name : uri_names) {
      if (absl::StartsWith(absl::string_view(uri_name.data(), uri_name.size()),
                           prefix_)) {
        *sync_status = grpc::Status::OK;
        return true;
      }
    }
    *sync_status = grpc::Status(
        grpc::StatusCode::UNAUTHENTICATED,
        absl::StrCat(
            "TLS peer verification failed: no URI SAN with required "
            "prefix \"",
            prefix_, "\"; peer URI SANs: [",
            absl::StrJoin(
                uri_names, ", ",
                [](std::string* out, const grpc::string_ref& uri_name) {
                  absl::StrAppend(
                      out, absl::string_view(uri_name.data(), uri_name.size()));
                }),
            "]"));
    return true;  // Verification completed synchronously.
  }

  void Cancel(
      grpc::experimental::TlsCustomVerificationCheckRequest* request) override {
    // Verify() completes synchronously, so there is never a pending check to
    // cancel.
  }

 private:
  const std::string prefix_;
};

// Best-effort fail-fast check so that a missing or unreadable credential
// file surfaces as an immediate, descriptive error instead of an opaque
// handshake failure. The authoritative reads (including periodic re-reads
// for rotation) happen inside FileWatcherCertificateProvider.
absl::Status ValidateReadableFile(absl::string_view role,
                                  const std::string& path) {
  if (path.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("mTLS configuration is missing the ", role, " file"));
  }
  std::string contents;
  absl::Status status =
      tsl::ReadFileToString(tsl::Env::Default(), path, &contents);
  if (!status.ok()) {
    return absl::Status(status.code(),
                        absl::StrCat("Cannot open mTLS ", role, " file ", path,
                                     ": ", status.message()));
  }
  return absl::OkStatus();
}

absl::Status ValidateMtlsConfig(const MtlsConfig& config) {
  if (!config.peer_uri_prefix.empty() &&
      !absl::EndsWith(config.peer_uri_prefix, "/")) {
    return absl::InvalidArgumentError(
        absl::StrCat("mTLS peer_uri_prefix must end with '/', e.g. "
                     "\"spiffe://example.org/\", to avoid accepting "
                     "\"spiffe://example.org.evil/...\". Got \"",
                     config.peer_uri_prefix, "\""));
  }
  ABSL_RETURN_IF_ERROR(
      ValidateReadableFile("certificate (cert_file)", config.cert_file));
  ABSL_RETURN_IF_ERROR(
      ValidateReadableFile("private key (key_file)", config.key_file));
  ABSL_RETURN_IF_ERROR(ValidateReadableFile("root CA (ca_file)", config.ca_file));
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::shared_ptr<::grpc::ChannelCredentials>>
GetMtlsClientCredentials(const MtlsConfig& config) {
  ABSL_RETURN_IF_ERROR(ValidateMtlsConfig(config));
  auto cert_provider =
      std::make_shared<grpc::experimental::FileWatcherCertificateProvider>(
          /*private_key_path=*/config.key_file,
          /*identity_certificate_path=*/config.cert_file,
          /*root_cert_path=*/config.ca_file,
          /*refresh_interval_sec=*/config.cert_refresh_interval_seconds);
  grpc::experimental::TlsChannelCredentialsOptions options;
  options.set_root_certificate_provider(cert_provider);
  options.set_identity_certificate_provider(std::move(cert_provider));
  options.set_verify_server_certs(true);
  if (!config.peer_uri_prefix.empty()) {
    // Replaces gRPC's default HostNameCertificateVerifier.
    options.set_certificate_verifier(
        grpc::experimental::ExternalCertificateVerifier::Create<
            UriSanPrefixVerifier>(config.peer_uri_prefix));
    options.set_check_call_host(false);
  }
  LOG(INFO) << "gRPC mTLS client credentials are used (peer_uri_prefix: \""
            << config.peer_uri_prefix << "\").";
  return grpc::experimental::TlsCredentials(options);
}

absl::StatusOr<std::shared_ptr<::grpc::ServerCredentials>>
GetMtlsServerCredentials(const MtlsConfig& config) {
  ABSL_RETURN_IF_ERROR(ValidateMtlsConfig(config));
  auto cert_provider =
      std::make_shared<grpc::experimental::FileWatcherCertificateProvider>(
          /*private_key_path=*/config.key_file,
          /*identity_certificate_path=*/config.cert_file,
          /*root_cert_path=*/config.ca_file,
          /*refresh_interval_sec=*/config.cert_refresh_interval_seconds);
  ABSL_ASSIGN_OR_RETURN(
      grpc::experimental::TlsServerCredentialsOptions options,
      grpc::experimental::TlsServerCredentialsOptions::Create(cert_provider));
  options.set_root_certificate_provider(std::move(cert_provider));
  // Mutual TLS: the client must present a certificate that verifies against
  // the configured root CA.
  options.set_cert_request_type(
      GRPC_SSL_REQUEST_AND_REQUIRE_CLIENT_CERTIFICATE_AND_VERIFY);
  if (!config.peer_uri_prefix.empty()) {
    options.set_certificate_verifier(
        grpc::experimental::ExternalCertificateVerifier::Create<
            UriSanPrefixVerifier>(config.peer_uri_prefix));
  }
  LOG(INFO) << "gRPC mTLS server credentials are used (peer_uri_prefix: \""
            << config.peer_uri_prefix << "\").";
  return grpc::experimental::TlsServerCredentials(options);
}

}  // namespace xla
