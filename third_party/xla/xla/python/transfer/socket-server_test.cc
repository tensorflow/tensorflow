/* Copyright 2025 The OpenXLA Authors.

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
#include "xla/python/transfer/socket-server.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/python/transfer/event_loop.h"
#include "xla/python/transfer/streaming.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/test.h"

namespace aux {
namespace {

TEST(ServerTest, Basic) {
  sockaddr_in6 addr;
  memset(&addr, 0, sizeof(sockaddr_in6));
  addr.sin6_family = AF_INET6;
  auto local_factory = BulkTransportFactory::CreateLocal();
  auto servera = std::make_shared<SocketServer>();
  CHECK_OK(servera->Start(SocketAddress(addr), local_factory));
  auto serverb = std::make_shared<SocketServer>();
  CHECK_OK(serverb->Start(SocketAddress(addr), local_factory));

  std::string msg("secret message");
  uint64_t uuid = 5678;
  int buffer_id = 0;
  serverb->AwaitPull(uuid, PullTable::MakeStringEntry({msg}));

  auto [s, cd] = ChunkDestination::MakeStringDest();
  auto conn = servera->Connect(serverb->addr());
  conn->Pull(uuid, buffer_id, std::move(cd));

  CHECK_EQ(s.Await().value(), msg);
  absl::SleepFor(absl::Seconds(2));
  conn = {};
}

TEST(ServerTest, DelayedConnect) {
  auto addra = SocketAddress::Parse("0.0.0.0:0").value();
  int port = tsl::testing::PickUnusedPortOrDie();
  auto addrb = SocketAddress::Parse(absl::StrCat("0.0.0.0:", port)).value();
  auto local_factory = BulkTransportFactory::CreateLocal();
  auto servera = std::make_shared<SocketServer>();
  CHECK_OK(servera->Start(addra, local_factory));

  uint64_t uuid = 5678;
  int buffer_id = 0;

  auto [s, cd] = ChunkDestination::MakeStringDest();
  auto conn = servera->Connect(addrb);
  conn->Pull(uuid, buffer_id, std::move(cd));

  auto serverb = std::make_shared<SocketServer>();
  CHECK_OK(serverb->Start(addrb, local_factory));
  std::string msg("secret message");
  serverb->AwaitPull(uuid, PullTable::MakeStringEntry({msg}));

  CHECK_EQ(s.Await().value(), msg);
  absl::SleepFor(absl::Seconds(2));
  conn = {};
}

}  // namespace
}  // namespace aux
