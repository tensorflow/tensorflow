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

#include "tensorflow/core/distributed_runtime/server_lib.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

class TestServerFactory : public ServerFactory {
 public:
  bool AcceptsOptions(const ServerDef& server_def) override {
    return server_def.protocol() == "test_protocol";
  }

  absl::Status NewServer(
      const ServerDef& server_def, const Options& options,
      std::unique_ptr<ServerInterface>* out_server) override {
    return absl::OkStatus();
  }
};

TEST(ServerLibTest, NewServerFactoryAccepts) {
  ServerFactory::Register("TEST_SERVER", new TestServerFactory());
  ServerDef server_def;
  server_def.set_protocol("test_protocol");
  std::unique_ptr<ServerInterface> server;
  TF_EXPECT_OK(NewServer(server_def, &server));
}

TEST(ServerLibTest, NewServerNoFactoriesAccept) {
  ServerDef server_def;
  server_def.set_protocol("fake_protocol");
  std::unique_ptr<ServerInterface> server;
  absl::Status s = NewServer(server_def, &server);
  ASSERT_NE(s, absl::OkStatus());
  EXPECT_TRUE(absl::StrContains(
      s.message(), "No server factory registered for the given ServerDef"));
  EXPECT_TRUE(
      absl::StrContains(s.message(), "The available server factories are: ["));
}

}  // namespace tensorflow
