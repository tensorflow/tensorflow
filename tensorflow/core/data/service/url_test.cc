/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/data/service/url.h"

#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace data {
namespace {

TEST(URLTest, ParseUrl) {
  URL url("localhost");
  EXPECT_EQ(url.host(), "localhost");
  EXPECT_FALSE(url.has_port());
}

TEST(URLTest, ParseUrlWithProtocol) {
  URL url("http://localhost");
  EXPECT_EQ(url.host(), "http://localhost");
  EXPECT_FALSE(url.has_port());
}

TEST(URLTest, ParseUrlWithPort) {
  URL url("localhost:1234");
  EXPECT_EQ(url.host(), "localhost");
  EXPECT_TRUE(url.has_port());
  EXPECT_EQ(url.port(), "1234");
}

TEST(URLTest, ParseUrlWithProtocolAndPort) {
  URL url("http://localhost:1234");
  EXPECT_EQ(url.host(), "http://localhost");
  EXPECT_TRUE(url.has_port());
  EXPECT_EQ(url.port(), "1234");
}

TEST(URLTest, ParseUrlWithProtocolAndDynamicPort) {
  URL url("http://localhost:%port%");
  EXPECT_EQ(url.host(), "http://localhost");
  EXPECT_TRUE(url.has_port());
  EXPECT_EQ(url.port(), "%port%");
}

TEST(URLTest, ParseBorgAddress) {
  URL url("/worker/task/0");
  EXPECT_EQ(url.host(), "/worker/task/0");
  EXPECT_FALSE(url.has_port());
}

TEST(URLTest, ParseBorgAddressWithCustomProtocol) {
  URL url("worker:/worker/task/0");
  EXPECT_EQ(url.host(), "worker:/worker/task/0");
  EXPECT_FALSE(url.has_port());
}

TEST(URLTest, ParseBorgAddressWithNamedPort) {
  URL url("/worker/task/0:worker");
  EXPECT_EQ(url.host(), "/worker/task/0");
  EXPECT_TRUE(url.has_port());
  EXPECT_EQ(url.port(), "worker");
}

TEST(URLTest, ParseBorgAddressWithDynamicPort) {
  URL url("/worker/task/0:%port%");
  EXPECT_EQ(url.host(), "/worker/task/0");
  EXPECT_TRUE(url.has_port());
  EXPECT_EQ(url.port(), "%port%");
}

TEST(URLTest, ParseBorgAddressWithDynamicNamedPort) {
  URL url("/worker/task/0:%port_worker%");
  EXPECT_EQ(url.host(), "/worker/task/0");
  EXPECT_TRUE(url.has_port());
  EXPECT_EQ(url.port(), "%port_worker%");
}

TEST(URLTest, ParseIPv4Address) {
  URL url("127.0.0.1");
  EXPECT_EQ(url.host(), "127.0.0.1");
  EXPECT_FALSE(url.has_port());
}

TEST(URLTest, ParseIPv4AddressWithPort) {
  URL url("127.0.0.1:8000");
  EXPECT_EQ(url.host(), "127.0.0.1");
  EXPECT_TRUE(url.has_port());
  EXPECT_EQ(url.port(), "8000");
}

TEST(URLTest, ParseIPv6Address) {
  URL url("[::1]");
  EXPECT_EQ(url.host(), "[::1]");
  EXPECT_FALSE(url.has_port());
}

TEST(URLTest, ParseIPv6AddressWithProtocol) {
  URL url("http://[::1]");
  EXPECT_EQ(url.host(), "http://[::1]");
  EXPECT_FALSE(url.has_port());
}

TEST(URLTest, ParseIPv6AddressWithPort) {
  URL url("[::1]:23456");
  EXPECT_EQ(url.host(), "[::1]");
  EXPECT_TRUE(url.has_port());
  EXPECT_EQ(url.port(), "23456");
}

TEST(URLTest, ParseIPv6AddressWithProtocolAndPort) {
  URL url("http://[::1]:23456");
  EXPECT_EQ(url.host(), "http://[::1]");
  EXPECT_TRUE(url.has_port());
  EXPECT_EQ(url.port(), "23456");
}

TEST(URLTest, ParseIPv6AddressWithProtocolAndDynamicPort) {
  URL url("http://[::1]:%port_name%");
  EXPECT_EQ(url.host(), "http://[::1]");
  EXPECT_TRUE(url.has_port());
  EXPECT_EQ(url.port(), "%port_name%");
}

TEST(URLTest, ParseNonLocalIPv6Address) {
  URL url("http://ipv6:[1080:0:0:0:8:800:200C:417A]");
  EXPECT_EQ(url.host(), "http://ipv6:[1080:0:0:0:8:800:200C:417A]");
  EXPECT_FALSE(url.has_port());
}

TEST(URLTest, ParseNonLocalIPv6AddressWithNamedPort) {
  URL url("http://ipv6:[1080:0:0:0:8:800:200C:417A]:worker");
  EXPECT_EQ(url.host(), "http://ipv6:[1080:0:0:0:8:800:200C:417A]");
  EXPECT_TRUE(url.has_port());
  EXPECT_EQ(url.port(), "worker");
}

TEST(URLTest, ParseEmptyIPv6Address) {
  URL url("http://ipv6:[]");
  EXPECT_EQ(url.host(), "http://ipv6:[]");
  EXPECT_FALSE(url.has_port());
}

TEST(URLTest, ParseEmptyAddress) {
  URL url("");
  EXPECT_EQ(url.host(), "");
  EXPECT_FALSE(url.has_port());
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
