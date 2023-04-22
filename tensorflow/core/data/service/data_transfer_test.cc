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
#include "tensorflow/core/data/service/data_transfer.h"

#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace data {
namespace {

class TestDataTransferServer : public DataTransferServer {
 public:
  explicit TestDataTransferServer(bool* called) : called_(called) {}
  Status Start() override {
    *called_ = true;
    return Status::OK();
  }
  int get_port() override { return 0; }

 private:
  bool* called_;
};

TEST(DataTransferTest, RegisterDataTransferServerBuilder) {
  bool called = false;
  DataTransferServer::Register("test", [&called](auto _) {
    return std::make_shared<TestDataTransferServer>(&called);
  });

  std::shared_ptr<DataTransferServer> server;
  TF_ASSERT_OK(DataTransferServer::Build("test", {}, &server));
  EXPECT_FALSE(called);

  TF_ASSERT_OK(server->Start());
  EXPECT_TRUE(called);
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
