/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/platform/cloud/gcs_dns_cache.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

class TestHttpRequest : public HttpRequest {
 public:
  void SetUri(const string& uri) override {}
  void SetRange(uint64 start, uint64 end) override {}
  void AddHeader(const string& name, const string& value) override {}
  void AddResolveOverride(const string& hostname, int64 port,
                          const string& ip_addr) override {
    EXPECT_EQ(port, 443) << "Unexpected port set for hostname: " << hostname;
    auto itr = resolve_overrides_.find(hostname);
    EXPECT_EQ(itr, resolve_overrides_.end())
        << "Hostname " << hostname << "already in map: " << itr->second;

    resolve_overrides_.insert(
        std::map<string, string>::value_type(hostname, ip_addr));
  }

  void AddAuthBearerHeader(const string& auth_token) override {}

  void SetDeleteRequest() override {}

  Status SetPutFromFile(const string& body_filepath, size_t offset) override {
    return Status::OK();
  }
  void SetPutEmptyBody() override {}
  void SetPostFromBuffer(const char* buffer, size_t size) override {}
  void SetPostEmptyBody() override {}
  void SetResultBuffer(std::vector<char>* out_buffer) override {}
  void SetResultBufferDirect(char* buffer, size_t size) override {}
  size_t GetResultBufferDirectBytesTransferred() override { return 0; }

  string GetResponseHeader(const string& name) const override { return ""; }
  uint64 GetResponseCode() const override { return 0; }
  Status Send() override { return Status::OK(); }
  string EscapeString(const string& str) override { return ""; }

  void SetTimeouts(uint32 connection, uint32 inactivity,
                   uint32 total) override {}

  std::map<string, string> resolve_overrides_;
};

// Friend class for testing.
//
// It is written this way (as opposed to using FRIEND_TEST) to avoid a
// non-test-time dependency on gunit.
class GcsDnsCacheTest : public ::testing::Test {
 protected:
  void ResolveNameTest() {
    auto response = GcsDnsCache::ResolveName("www.googleapis.com");
    EXPECT_LT(1, response.size()) << str_util::Join(response, ", ");
  }

  void AnnotateRequestTest() {
    GcsDnsCache d;
    {
      mutex_lock l(d.mu_);
      d.started_ = true;  // Avoid creating a thread.
      d.addresses_ = {{"192.168.1.1"}, {"172.134.1.1"}};
    }

    TestHttpRequest req;
    d.AnnotateRequest(&req);
    EXPECT_EQ("192.168.1.1", req.resolve_overrides_["www.googleapis.com"]);
    EXPECT_EQ("172.134.1.1", req.resolve_overrides_["storage.googleapis.com"]);
  }

  void SuccessfulCleanupTest() {
    // Create a DnsCache object, start the worker thread, ensure it cleans up in
    // a timely manner.
    GcsDnsCache d;
    TestHttpRequest req;
    d.AnnotateRequest(&req);
  }
};

// This sends a DNS name resolution request, thus it is flaky.
// TEST_F(GcsDnsCacheTest, ResolveName) { ResolveNameTest(); }

TEST_F(GcsDnsCacheTest, AnnotateRequest) { AnnotateRequestTest(); }

TEST_F(GcsDnsCacheTest, SuccessfulCleanup) { SuccessfulCleanupTest(); }

}  // namespace tensorflow
