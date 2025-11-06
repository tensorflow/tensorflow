/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/distributed_runtime/rpc/eager/grpc_eager_client.h"

#include <memory>

#include "absl/status/status.h"
#include "xla/tsl/protobuf/error_codes.pb.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_channel.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/blocking_counter.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace eager {

TEST(GrpcEagerClientCache, TestGetClientThreadSafety) {
  GrpcChannelSpec spec;
  TF_ASSERT_OK(spec.AddHostPortsJob("worker", {{0, "a:1"},
                                               {1, "b:2"},
                                               {2, "c:3"},
                                               {3, "d:4"},
                                               {4, "e:5"},
                                               {5, "f:6"}}));
  ChannelCreationFunction channel_func =
      ConvertToChannelCreationFunction(NewHostPortGrpcChannel);
  auto channel_cache = std::shared_ptr<GrpcChannelCache>(
      NewGrpcChannelCache(spec, channel_func));
  std::unique_ptr<EagerClientCache> client_cache(
      NewGrpcEagerClientCache(channel_cache));
  const int num_calls = 10;
  BlockingCounter counter(num_calls);

  for (int i = 0; i < num_calls; i++) {
    Env::Default()->SchedClosure([&client_cache, i, &counter]() {
      string target = absl::StrCat("/job:worker/replica:0/task:", i);
      core::RefCountPtr<EagerClient> eager_client;
      absl::Status s = client_cache->GetClient(target, &eager_client);
      // With 6 tasks added to the job, querying client for 0--5 should be OK,
      // and querying client for 6+ should give invalid argument error.
      error::Code expected_code = i <= 5 ? error::OK : error::INVALID_ARGUMENT;
      EXPECT_EQ(expected_code, s.code());
      counter.DecrementCount();
    });
  }
  counter.Wait();
}

}  // namespace eager
}  // namespace tensorflow
