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

#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/pjrt/distributed/client.h"
#include "xla/pjrt/distributed/distributed.h"
#include "xla/pjrt/distributed/service.h"
#include "xla/tsl/platform/test.h"
#include "tsl/platform/path.h"

namespace xla {
namespace {

// The fixtures in testdata/mtls (see generate.sh there): `client`, `server`
// and `multi` carry URI SANs under kGoodPrefix and chain to ca1; `evil`
// chains to ca1 but its URI SAN is outside the prefix; `nouri` chains to ca1
// and has only DNS:localhost; `outcast` is under the prefix but chains to
// ca2.
constexpr absl::string_view kGoodPrefix = "spiffe://example.org/coord/";

std::string TestDataPath(absl::string_view filename) {
  return tsl::io::JoinPath(tsl::testing::XlaSrcRoot(),
                           "pjrt/distributed/testdata/mtls", filename);
}

MtlsConfig MakeConfig(absl::string_view identity,
                      absl::string_view peer_uri_prefix = "",
                      absl::string_view ca = "ca1") {
  MtlsConfig config;
  config.cert_file = TestDataPath(absl::StrCat(identity, ".pem"));
  config.key_file = TestDataPath(absl::StrCat(identity, ".key"));
  config.ca_file = TestDataPath(absl::StrCat(ca, ".pem"));
  config.peer_uri_prefix = std::string(peer_uri_prefix);
  return config;
}

class MtlsTest : public ::testing::Test {
 protected:
  // Starts a single-node coordination service with mTLS server credentials
  // built from `config`.
  void StartService(const MtlsConfig& config) {
    port_ = tsl::testing::PickUnusedPortOrDie();
    CoordinationServiceImpl::Options options;
    options.num_nodes = 1;
    ASSERT_OK_AND_ASSIGN(options.credentials, GetMtlsServerCredentials(config));
    ASSERT_OK_AND_ASSIGN(service_, GetDistributedRuntimeService(
                                       absl::StrCat("[::]:", port_), options));
  }

  // Connects a client with mTLS credentials built from `config` to the
  // service, dialing it as `host`, and shuts it down again. A rejected TLS
  // handshake surfaces as Connect() timing out, so the timeout is kept short;
  // 100ms is plenty for a local server.
  absl::Status Connect(const MtlsConfig& config,
                       absl::string_view host = "127.0.0.1") {
    DistributedRuntimeClient::Options options;
    options.node_id = 0;
    options.init_timeout = absl::Milliseconds(100);
    options.extra_error_propagation_time = absl::ZeroDuration();
    options.missed_heartbeat_callback = [](const absl::Status&) {};
    ABSL_ASSIGN_OR_RETURN(options.credentials, GetMtlsClientCredentials(config));
    std::shared_ptr<DistributedRuntimeClient> client =
        GetDistributedRuntimeClient(absl::StrCat(host, ":", port_), options);
    ABSL_RETURN_IF_ERROR(client->Connect());
    return client->Shutdown();
  }

  int port_ = 0;
  std::unique_ptr<DistributedRuntimeService> service_;
};

// ---- peer URI prefix check on both sides ----

TEST_F(MtlsTest, GoodClientAndPrefixIsAccepted) {
  StartService(MakeConfig("server", kGoodPrefix));
  EXPECT_OK(Connect(MakeConfig("client", kGoodPrefix)));
}

TEST_F(MtlsTest, ClientRejectsServerOutsideItsPrefix) {
  StartService(MakeConfig("server", kGoodPrefix));
  // The same certificates connect in GoodClientAndPrefixIsAccepted; only the
  // client's prefix differs here, so the client-side prefix check is the only
  // possible rejector.
  EXPECT_THAT(Connect(MakeConfig("client", "spiffe://other/")),
              absl_testing::StatusIs(absl::StatusCode::kDeadlineExceeded));
}

TEST_F(MtlsTest, ServerRejectsInChainIntruder) {
  StartService(MakeConfig("server", kGoodPrefix));
  // `evil` chains to the trusted CA but its URI SAN is outside the prefix:
  // only the prefix check can reject it (InChainIntruderIsAcceptedWithoutPrefix
  // shows it is accepted by a server without one).
  EXPECT_THAT(Connect(MakeConfig("evil", kGoodPrefix)),
              absl_testing::StatusIs(absl::StatusCode::kDeadlineExceeded));
}

TEST_F(MtlsTest, ServerRejectsWrongCaClient) {
  StartService(MakeConfig("server", kGoodPrefix));
  EXPECT_THAT(Connect(MakeConfig("outcast", kGoodPrefix)),
              absl_testing::StatusIs(absl::StatusCode::kDeadlineExceeded));
}

TEST_F(MtlsTest, MultiSanLeafMatchesOnAnyUriSan) {
  StartService(MakeConfig("server", kGoodPrefix));
  // `multi`'s first URI SAN is outside the prefix, its second inside.
  EXPECT_OK(Connect(MakeConfig("multi", kGoodPrefix)));
}

TEST_F(MtlsTest, ZeroUriSanLeafIsRejected) {
  StartService(MakeConfig("server", kGoodPrefix));
  // `nouri` carries only a DNS SAN: the prefix check fails closed.
  EXPECT_THAT(Connect(MakeConfig("nouri", kGoodPrefix)),
              absl_testing::StatusIs(absl::StatusCode::kDeadlineExceeded));
}

TEST_F(MtlsTest, HostnameVerificationRejectsUriSanOnlyServer) {
  StartService(MakeConfig("server", kGoodPrefix));
  // The server certificate carries only a URI SAN and is dialed by IP, so a
  // client using the default hostname verification (no prefix) cannot
  // succeed -- the deployment shape that motivates peer_uri_prefix.
  EXPECT_THAT(Connect(MakeConfig("client")),
              absl_testing::StatusIs(absl::StatusCode::kDeadlineExceeded));
}

TEST_F(MtlsTest, PrefixWithoutTrailingSlashIsRejected) {
  // "spiffe://example.org" would also match "spiffe://example.org.evil/...",
  // so a prefix must end with '/'.
  MtlsConfig config = MakeConfig("server", "spiffe://example.org");
  EXPECT_THAT(GetMtlsServerCredentials(config),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(GetMtlsClientCredentials(config),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

// ---- default (no prefix) on both sides ----
// `nouri` carries DNS:localhost, so clients using the default hostname
// verification must dial "localhost".

TEST_F(MtlsTest, DnsSanServerDialedByNameIsAccepted) {
  StartService(MakeConfig("nouri"));
  EXPECT_OK(Connect(MakeConfig("client"), "localhost"));
  // The same server dialed by IP address does not match DNS:localhost.
  EXPECT_THAT(Connect(MakeConfig("client"), "127.0.0.1"),
              absl_testing::StatusIs(absl::StatusCode::kDeadlineExceeded));
}

TEST_F(MtlsTest, InChainIntruderIsAcceptedWithoutPrefix) {
  StartService(MakeConfig("nouri"));
  // Isolates the prefix rejection in ServerRejectsInChainIntruder above: the
  // same `evil` identity passes the server's plain chain verification.
  EXPECT_OK(Connect(MakeConfig("evil"), "localhost"));
}

TEST_F(MtlsTest, ZeroUriSanLeafIsAcceptedWithoutPrefix) {
  StartService(MakeConfig("nouri"));
  EXPECT_OK(Connect(MakeConfig("nouri"), "localhost"));
}

TEST_F(MtlsTest, WrongCaClientIsRejected) {
  StartService(MakeConfig("nouri"));
  // Mutual TLS still requires the client to chain to the server's root.
  EXPECT_THAT(Connect(MakeConfig("outcast"), "localhost"),
              absl_testing::StatusIs(absl::StatusCode::kDeadlineExceeded));
}

// ---- asymmetric trust roots ----

TEST_F(MtlsTest, ClientSideChainCheckIsEnforced) {
  // The server's identity chains to ca1 but it trusts ca2 for client
  // certificates, so the ca2-signed `outcast` (whose URI SAN is under the
  // prefix) passes the SERVER side in both cases -- isolating the
  // CLIENT-side chain check as the only possible rejector in the second case.
  StartService(MakeConfig("server", kGoodPrefix, /*ca=*/"ca2"));
  EXPECT_OK(Connect(MakeConfig("outcast", kGoodPrefix, /*ca=*/"ca1")));
  EXPECT_THAT(Connect(MakeConfig("outcast", kGoodPrefix, /*ca=*/"ca2")),
              absl_testing::StatusIs(absl::StatusCode::kDeadlineExceeded));
}

}  // namespace
}  // namespace xla
