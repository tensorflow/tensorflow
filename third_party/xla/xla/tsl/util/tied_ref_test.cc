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

#include "xla/tsl/util/tied_ref.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "xla/tsl/platform/test.h"

namespace tsl {
namespace {

// A Connection tracks its own construction/destruction via an external counter,
// so tests can verify when tied values are created and released.
class Connection {
 public:
  explicit Connection(int32_t* alive_count) : alive_count_(alive_count) {
    ++(*alive_count_);
  }
  ~Connection() { --(*alive_count_); }

 private:
  int32_t* alive_count_;
};

// A second tracked type used by TiedAny tests to verify that heterogeneous
// types can be tied in a single container.
struct Widget {
  explicit Widget(int32_t* alive_count) : alive_count_(alive_count) {
    ++(*alive_count_);
  }
  ~Widget() { --(*alive_count_); }
  int32_t* alive_count_;
};

// A Session is a long-lived entity that creates multiple Connections during its
// lifetime. Connections are tied to the session and expire when the session is
// destroyed or replaced.
class Session : private Tied<Connection> {
 public:
  TiedRef<Connection> Connect() {
    return Tie(std::make_unique<Connection>(&alive_connections_));
  }

  int32_t alive_connections() const { return alive_connections_; }

 private:
  int32_t alive_connections_ = 0;
};

TEST(TiedRefTest, ConnectionLocksWhileSessionAlive) {
  Session session;
  TiedRef<Connection> ref = session.Connect();

  std::shared_ptr<Connection> conn = ref.Lock();
  ASSERT_NE(conn, nullptr);
  EXPECT_EQ(session.alive_connections(), 1);
  EXPECT_FALSE(ref.Expired());
}

TEST(TiedRefTest, ConnectionExpiresWhenSessionDestroyed) {
  TiedRef<Connection> ref;
  {
    Session session;
    ref = session.Connect();
    EXPECT_EQ(session.alive_connections(), 1);
    EXPECT_FALSE(ref.Expired());
  }
  EXPECT_TRUE(ref.Expired());
  EXPECT_EQ(ref.Lock(), nullptr);
}

TEST(TiedRefTest, ConnectionReleasedWhenTiedRefDestroyed) {
  Session session;
  {
    TiedRef<Connection> ref = session.Connect();
    EXPECT_EQ(session.alive_connections(), 1);
  }
  // TiedRef destroyed while session is alive — connection eagerly released.
  EXPECT_EQ(session.alive_connections(), 0);
}

TEST(TiedRefTest, MultipleConnections) {
  Session session;
  TiedRef<Connection> ref1 = session.Connect();
  TiedRef<Connection> ref2 = session.Connect();

  EXPECT_EQ(session.alive_connections(), 2);
  EXPECT_NE(ref1.Lock(), nullptr);
  EXPECT_NE(ref2.Lock(), nullptr);
}

TEST(TiedRefTest, CachedRefsDetectSessionReplacement) {
  std::vector<TiedRef<Connection>> cache;

  std::unique_ptr<Session> session = std::make_unique<Session>();
  cache.push_back(session->Connect());
  cache.push_back(session->Connect());
  EXPECT_EQ(session->alive_connections(), 2);

  // Replace the session — old tied connections expire.
  session = std::make_unique<Session>();
  for (TiedRef<Connection>& ref : cache) {
    EXPECT_TRUE(ref.Expired());
  }

  // New connections from the replacement are alive.
  cache.clear();
  cache.push_back(session->Connect());
  EXPECT_EQ(session->alive_connections(), 1);
  EXPECT_FALSE(cache[0].Expired());
}

TEST(TiedRefTest, UntieReturnsOwnershipAndExpiresRef) {
  Session session;
  TiedRef<Connection> ref = session.Connect();
  EXPECT_EQ(session.alive_connections(), 1);

  std::shared_ptr<Connection> conn = std::move(ref).Untie();
  ASSERT_NE(conn, nullptr);

  // The ref is now expired, but the connection is still alive because the
  // caller holds the shared_ptr.
  EXPECT_TRUE(ref.Expired());
  EXPECT_EQ(ref.Lock(), nullptr);
  EXPECT_EQ(session.alive_connections(), 1);

  // Dropping the shared_ptr destroys the connection.
  conn.reset();
  EXPECT_EQ(session.alive_connections(), 0);
}

TEST(TiedRefTest, UntieExpiredRefReturnsNullptr) {
  TiedRef<Connection> ref;
  {
    Session session;
    ref = session.Connect();
  }
  EXPECT_TRUE(ref.Expired());
  EXPECT_EQ(std::move(ref).Untie(), nullptr);
}

TEST(TiedRefTest, UntieDefaultRefReturnsNullptr) {
  TiedRef<Connection> ref;
  EXPECT_EQ(std::move(ref).Untie(), nullptr);
}

TEST(TiedRefTest, UntieThenDestroySessionDoesNotDoubleDestroy) {
  Session session;
  TiedRef<Connection> ref = session.Connect();

  std::shared_ptr<Connection> conn = std::move(ref).Untie();
  ASSERT_NE(conn, nullptr);
  EXPECT_EQ(session.alive_connections(), 1);
}

TEST(TiedRefTest, DefaultTiedRefIsExpired) {
  TiedRef<Connection> ref;
  EXPECT_TRUE(ref.Expired());
  EXPECT_EQ(ref.Lock(), nullptr);
}

TEST(TiedRefTest, TieNullptrReturnsExpiredRef) {
  Tied<Connection> tied;
  TiedRef<Connection> ref = tied.Tie(nullptr);
  EXPECT_TRUE(ref.Expired());
  EXPECT_EQ(ref.Lock(), nullptr);
}

TEST(TiedAnyTest, TieAndLockMultipleTypes) {
  TiedAny container;
  int32_t alive_connections = 0;
  int32_t alive_widgets = 0;

  TiedRef<Connection> conn_ref =
      container.Tie(std::make_unique<Connection>(&alive_connections));
  TiedRef<Widget> widget_ref =
      container.Tie(std::make_unique<Widget>(&alive_widgets));

  EXPECT_EQ(alive_connections, 1);
  EXPECT_EQ(alive_widgets, 1);
  EXPECT_NE(conn_ref.Lock(), nullptr);
  EXPECT_NE(widget_ref.Lock(), nullptr);
}

TEST(TiedAnyTest, AllRefsExpireWhenContainerDestroyed) {
  TiedRef<Connection> conn_ref;
  TiedRef<Widget> widget_ref;
  int32_t alive_connections = 0;
  int32_t alive_widgets = 0;

  {
    TiedAny container;
    conn_ref = container.Tie(std::make_unique<Connection>(&alive_connections));
    widget_ref = container.Tie(std::make_unique<Widget>(&alive_widgets));
    EXPECT_EQ(alive_connections, 1);
    EXPECT_EQ(alive_widgets, 1);
  }

  EXPECT_TRUE(conn_ref.Expired());
  EXPECT_TRUE(widget_ref.Expired());
  EXPECT_EQ(alive_connections, 0);
  EXPECT_EQ(alive_widgets, 0);
}

TEST(TiedAnyTest, EagerReleaseOnRefDestruction) {
  TiedAny container;
  int32_t alive_connections = 0;

  {
    TiedRef<Connection> ref =
        container.Tie(std::make_unique<Connection>(&alive_connections));
    EXPECT_EQ(alive_connections, 1);
  }

  EXPECT_EQ(alive_connections, 0);
}

}  // namespace
}  // namespace tsl
