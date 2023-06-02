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

#include "tensorflow/tsl/distributed_runtime/rpc/grpc_channel.h"

#include <string>
#include <vector>

#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/platform/strcat.h"
#include "tensorflow/tsl/platform/test.h"
#include "tensorflow/tsl/protobuf/rpc_options.pb.h"
#include "tensorflow/tsl/util/device_name_utils.h"

namespace tsl {
#define IsSameAddrSp DeviceNameUtils::IsSameAddressSpace

TEST(GrpcChannelTest, IsSameAddressSpace) {
  // Same.
  EXPECT_TRUE(IsSameAddrSp("/job:mnist/replica:10/task:10/cpu:0",
                           "/job:mnist/replica:10/task:10/cpu:1"));
  EXPECT_TRUE(IsSameAddrSp("/job:mnist/replica:10/task:10/cpu:0",
                           "/job:mnist/replica:10/task:10/device:GPU:2"));
  EXPECT_TRUE(IsSameAddrSp("/job:mnist/replica:10/task:10",
                           "/job:mnist/replica:10/task:10/device:GPU:2"));
  EXPECT_TRUE(IsSameAddrSp("/job:mnist/replica:10/task:10/cpu:1",
                           "/job:mnist/replica:10/task:10"));

  // Different.
  EXPECT_FALSE(IsSameAddrSp("/job:mnist/replica:10/task:9/cpu:0",
                            "/job:mnist/replica:10/task:10/cpu:0"));
  EXPECT_FALSE(IsSameAddrSp("/job:mnist/replica:9/task:10/cpu:0",
                            "/job:mnist/replica:10/task:10/cpu:0"));
  EXPECT_FALSE(IsSameAddrSp("/job:MNIST/replica:10/task:10/cpu:0",
                            "/job:mnist/replica:10/task:10/cpu:0"));

  // Invalid names.
  EXPECT_FALSE(IsSameAddrSp("random_invalid_target", "random_invalid_target"));
  EXPECT_FALSE(IsSameAddrSp("/job:/replica:10/task:10/cpu:0",
                            "/job:/replica:10/task:10/cpu:1"));
  EXPECT_FALSE(IsSameAddrSp("/job:mnist/replica:xx/task:10/cpu:0",
                            "/job:mnist/replica:xx/task:10/cpu:1"));
  EXPECT_FALSE(IsSameAddrSp("/job:mnist/replica:10/task:yy/cpu:0",
                            "/job:mnist/replica:10/task:yy/cpu:1"));
}

TEST(GrpcChannelTest, HostPorts) {
  GrpcChannelSpec spec;
  TF_ASSERT_OK(spec.AddHostPortsJob("mnist", {{0, "a:1"},
                                              {1, "b:2"},
                                              {2, "c:3"},
                                              {3, "d:4"},
                                              {4, "e:5"},
                                              {5, "f:6"}}));
  ChannelCreationFunction channel_func =
      ConvertToChannelCreationFunction(NewHostPortGrpcChannel);
  std::unique_ptr<GrpcChannelCache> cc(
      NewGrpcChannelCache(spec, channel_func, tensorflow::RPCOptions()));

  EXPECT_EQ(nullptr, cc->FindWorkerChannel("invalid_target"));
  EXPECT_EQ(nullptr, cc->FindWorkerChannel("/job:other/replica:0/task:0"));
  EXPECT_EQ(nullptr, cc->FindWorkerChannel("/job:mnist/replica:0/task:6"));

  {
    // NOTE(mrry): The gRPC channel doesn't expose the target, so we
    // can't compare it for equality.
    auto a_1_1 = cc->FindWorkerChannel("/job:mnist/replica:0/task:0");
    auto a_1_2 = cc->FindWorkerChannel("/job:mnist/replica:0/task:0");

    auto d_4_1 = cc->FindWorkerChannel("/job:mnist/replica:0/task:3");
    auto d_4_2 = cc->FindWorkerChannel("/job:mnist/replica:0/task:3");

    auto e_5_1 = cc->FindWorkerChannel("/job:mnist/replica:0/task:4");
    auto e_5_2 = cc->FindWorkerChannel("/job:mnist/replica:0/task:4");

    EXPECT_EQ(a_1_1.get(), a_1_2.get());
    EXPECT_EQ(d_4_1.get(), d_4_2.get());
    EXPECT_EQ(e_5_1.get(), e_5_2.get());

    EXPECT_NE(a_1_1.get(), d_4_2.get());
    EXPECT_NE(a_1_1.get(), e_5_2.get());
    EXPECT_NE(d_4_1.get(), e_5_2.get());
  }

  {
    std::vector<string> workers;
    cc->ListWorkers(&workers);
    EXPECT_EQ(
        std::vector<string>(
            {"/job:mnist/replica:0/task:0", "/job:mnist/replica:0/task:1",
             "/job:mnist/replica:0/task:2", "/job:mnist/replica:0/task:3",
             "/job:mnist/replica:0/task:4", "/job:mnist/replica:0/task:5"}),
        workers);
  }

  {
    std::vector<string> workers;
    cc->ListWorkersInJob("mnist", &workers);
    EXPECT_EQ(
        std::vector<string>(
            {"/job:mnist/replica:0/task:0", "/job:mnist/replica:0/task:1",
             "/job:mnist/replica:0/task:2", "/job:mnist/replica:0/task:3",
             "/job:mnist/replica:0/task:4", "/job:mnist/replica:0/task:5"}),
        workers);
  }

  {
    std::vector<string> workers;
    cc->ListWorkersInJob("other", &workers);
    EXPECT_TRUE(workers.empty());
  }
}

TEST(GrpcChannelTest, HostPortsMultiChannelPerTarget) {
  GrpcChannelSpec spec;
  TF_EXPECT_OK(
      spec.AddHostPortsJob("mnist", {{0, "a:1"}, {1, "b:2"}, {2, "c:3"}}));
  ChannelCreationFunction channel_func =
      ConvertToChannelCreationFunction(NewHostPortGrpcChannel);
  tensorflow::RPCOptions rpc_options;
  rpc_options.set_num_channels_per_target(4);
  std::unique_ptr<GrpcChannelCache> cc(
      NewGrpcChannelCache(spec, channel_func, rpc_options));

  EXPECT_EQ(nullptr, cc->FindWorkerChannel("invalid_target"));
  EXPECT_EQ(nullptr, cc->FindWorkerChannel("/job:other/replica:0/task:0"));
  EXPECT_EQ(nullptr, cc->FindWorkerChannel("/job:mnist/replica:0/task:3"));

  {
    // NOTE(mrry): The gRPC channel doesn't expose the target, so we
    // can't compare it for equality.
    std::vector<SharedGrpcChannelPtr> a_1_channels, b_2_channels, c_3_channels;

    for (int i = 0; i < 10; i++) {
      a_1_channels.push_back(
          cc->FindWorkerChannel("/job:mnist/replica:0/task:0"));
      b_2_channels.push_back(
          cc->FindWorkerChannel("/job:mnist/replica:0/task:1"));
      c_3_channels.push_back(
          cc->FindWorkerChannel("/job:mnist/replica:0/task:2"));
    }

    // Same channel every 4 calls.
    for (int i = 0; i < 6; i++) {
      EXPECT_EQ(a_1_channels[i].get(), a_1_channels[i + 4].get());
      EXPECT_EQ(b_2_channels[i].get(), b_2_channels[i + 4].get());
      EXPECT_EQ(c_3_channels[i].get(), c_3_channels[i + 4].get());
    }

    // Other channels not equal
    for (int i = 0; i < 6; i++) {
      for (int j = 1; j < 4; j++) {
        EXPECT_NE(a_1_channels[i].get(), a_1_channels[i + j].get());
        EXPECT_NE(b_2_channels[i].get(), b_2_channels[i + j].get());
        EXPECT_NE(c_3_channels[i].get(), c_3_channels[i + j].get());
      }
    }

    // Cross Channels never equal
    for (int i = 0; i < 6; i++) {
      for (int j = 0; j < 6; j++) {
        EXPECT_NE(a_1_channels[i].get(), b_2_channels[j].get());
        EXPECT_NE(a_1_channels[i].get(), c_3_channels[j].get());
        EXPECT_NE(b_2_channels[i].get(), c_3_channels[j].get());
      }
    }
  }

  {
    std::vector<string> workers;
    cc->ListWorkers(&workers);
    EXPECT_EQ(std::vector<string>({"/job:mnist/replica:0/task:0",
                                   "/job:mnist/replica:0/task:1",
                                   "/job:mnist/replica:0/task:2"}),
              workers);
  }

  {
    std::vector<string> workers;
    cc->ListWorkersInJob("mnist", &workers);
    EXPECT_EQ(std::vector<string>({"/job:mnist/replica:0/task:0",
                                   "/job:mnist/replica:0/task:1",
                                   "/job:mnist/replica:0/task:2"}),
              workers);
  }

  {
    std::vector<string> workers;
    cc->ListWorkersInJob("other", &workers);
    EXPECT_TRUE(workers.empty());
  }
}

TEST(GrpcChannelTest, HostPortsMultiGrpcMultiChannelPerTarget) {
  GrpcChannelSpec spec;
  TF_EXPECT_OK(
      spec.AddHostPortsJob("mnist", {{0, "a:1"}, {1, "b:2"}, {2, "c:3"}}));
  TF_EXPECT_OK(
      spec.AddHostPortsJob("mnist2", {{0, "a:1"}, {1, "b:2"}, {2, "c:3"}}));
  ChannelCreationFunction channel_func =
      ConvertToChannelCreationFunction(NewHostPortGrpcChannel);
  tensorflow::RPCOptions rpc_options;
  rpc_options.set_num_channels_per_target(4);
  std::unique_ptr<GrpcChannelCache> cc(
      NewGrpcChannelCache(spec, channel_func, rpc_options));

  EXPECT_EQ(nullptr, cc->FindWorkerChannel("invalid_target"));
  EXPECT_EQ(nullptr, cc->FindWorkerChannel("/job:other/replica:0/task:0"));
  EXPECT_EQ(nullptr, cc->FindWorkerChannel("/job:mnist/replica:0/task:3"));
  EXPECT_NE(nullptr, cc->FindWorkerChannel("/job:mnist2/replica:0/task:0"));

  {
    // NOTE(mrry): The gRPC channel doesn't expose the target, so we
    // can't compare it for equality.
    std::vector<SharedGrpcChannelPtr> a_1_channels, b_2_channels, c_3_channels;

    for (int i = 0; i < 10; i++) {
      a_1_channels.push_back(
          cc->FindWorkerChannel("/job:mnist/replica:0/task:0"));
      b_2_channels.push_back(
          cc->FindWorkerChannel("/job:mnist/replica:0/task:1"));
      c_3_channels.push_back(
          cc->FindWorkerChannel("/job:mnist2/replica:0/task:0"));
    }

    // Same channel every 4 calls.
    for (int i = 0; i < 6; i++) {
      EXPECT_EQ(a_1_channels[i].get(), a_1_channels[i + 4].get());
      EXPECT_EQ(b_2_channels[i].get(), b_2_channels[i + 4].get());
      EXPECT_EQ(c_3_channels[i].get(), c_3_channels[i + 4].get());
    }

    // Other channels not equal
    for (int i = 0; i < 6; i++) {
      for (int j = 1; j < 4; j++) {
        EXPECT_NE(a_1_channels[i].get(), a_1_channels[i + j].get());
        EXPECT_NE(b_2_channels[i].get(), b_2_channels[i + j].get());
        EXPECT_NE(c_3_channels[i].get(), c_3_channels[i + j].get());
      }
    }

    // Cross Channels never equal
    for (int i = 0; i < 6; i++) {
      for (int j = 0; j < 6; j++) {
        EXPECT_NE(a_1_channels[i].get(), b_2_channels[j].get());
        EXPECT_NE(a_1_channels[i].get(), c_3_channels[j].get());
        EXPECT_NE(b_2_channels[i].get(), c_3_channels[j].get());
      }
    }
  }

  {
    std::vector<string> workers;
    cc->ListWorkers(&workers);
    EXPECT_EQ(
        std::vector<string>(
            {"/job:mnist/replica:0/task:0", "/job:mnist/replica:0/task:1",
             "/job:mnist/replica:0/task:2", "/job:mnist2/replica:0/task:0",
             "/job:mnist2/replica:0/task:1", "/job:mnist2/replica:0/task:2"}),
        workers);
  }

  {
    std::vector<string> workers, workers2;
    cc->ListWorkersInJob("mnist", &workers);
    EXPECT_EQ(std::vector<string>({"/job:mnist/replica:0/task:0",
                                   "/job:mnist/replica:0/task:1",
                                   "/job:mnist/replica:0/task:2"}),
              workers);
    cc->ListWorkersInJob("mnist2", &workers2);
    EXPECT_EQ(std::vector<string>({"/job:mnist2/replica:0/task:0",
                                   "/job:mnist2/replica:0/task:1",
                                   "/job:mnist2/replica:0/task:2"}),
              workers2);
  }

  {
    std::vector<string> workers;
    cc->ListWorkersInJob("other", &workers);
    EXPECT_TRUE(workers.empty());
  }
}

TEST(GrpcChannelTest, SparseHostPorts) {
  GrpcChannelSpec spec;
  TF_EXPECT_OK(
      spec.AddHostPortsJob("mnist", {{0, "a:1"}, {3, "d:4"}, {4, "e:5"}}));
  ChannelCreationFunction channel_func =
      ConvertToChannelCreationFunction(NewHostPortGrpcChannel);
  std::unique_ptr<GrpcChannelCache> cc(
      NewGrpcChannelCache(spec, channel_func, tensorflow::RPCOptions()));

  EXPECT_EQ(nullptr, cc->FindWorkerChannel("invalid_target"));
  EXPECT_EQ(nullptr, cc->FindWorkerChannel("/job:other/replica:0/task:0"));
  EXPECT_EQ(nullptr, cc->FindWorkerChannel("/job:mnist/replica:0/task:1"));
  EXPECT_EQ(nullptr, cc->FindWorkerChannel("/job:mnist/replica:0/task:2"));
  EXPECT_EQ(nullptr, cc->FindWorkerChannel("/job:mnist/replica:0/task:5"));

  {
    // NOTE(mrry): The gRPC channel doesn't expose the target, so we
    // can't compare it for equality.
    auto a_1_1 = cc->FindWorkerChannel("/job:mnist/replica:0/task:0");
    auto a_1_2 = cc->FindWorkerChannel("/job:mnist/replica:0/task:0");

    LOG(WARNING) << " Getting task 3";
    auto d_4_1 = cc->FindWorkerChannel("/job:mnist/replica:0/task:3");
    auto d_4_2 = cc->FindWorkerChannel("/job:mnist/replica:0/task:3");

    LOG(WARNING) << " Getting task 4";
    auto e_5_1 = cc->FindWorkerChannel("/job:mnist/replica:0/task:4");
    auto e_5_2 = cc->FindWorkerChannel("/job:mnist/replica:0/task:4");

    EXPECT_EQ(a_1_1.get(), a_1_2.get());
    EXPECT_EQ(d_4_1.get(), d_4_2.get());
    EXPECT_EQ(e_5_1.get(), e_5_2.get());

    EXPECT_NE(a_1_1.get(), d_4_2.get());
    EXPECT_NE(a_1_1.get(), e_5_2.get());
    EXPECT_NE(d_4_1.get(), e_5_2.get());
  }

  {
    std::vector<string> workers;
    cc->ListWorkers(&workers);
    std::sort(workers.begin(), workers.end());
    EXPECT_EQ(std::vector<string>({"/job:mnist/replica:0/task:0",
                                   "/job:mnist/replica:0/task:3",
                                   "/job:mnist/replica:0/task:4"}),
              workers);
  }

  {
    std::vector<string> workers;
    cc->ListWorkersInJob("mnist", &workers);
    EXPECT_EQ(std::vector<string>({"/job:mnist/replica:0/task:0",
                                   "/job:mnist/replica:0/task:3",
                                   "/job:mnist/replica:0/task:4"}),
              workers);
  }

  {
    std::vector<string> workers;
    cc->ListWorkersInJob("other", &workers);
    EXPECT_TRUE(workers.empty());
  }
}

TEST(GrpcChannelTest, NewHostPortGrpcChannelValidation) {
  SharedGrpcChannelPtr mock_ptr;

  EXPECT_TRUE(NewHostPortGrpcChannel("127.0.0.1:2222", /*rpc_options=*/nullptr,
                                     &mock_ptr)
                  .ok());
  EXPECT_TRUE(NewHostPortGrpcChannel("example.com:2222",
                                     /*rpc_options=*/nullptr, &mock_ptr)
                  .ok());
  EXPECT_TRUE(NewHostPortGrpcChannel("fqdn.example.com.:2222",
                                     /*rpc_options=*/nullptr, &mock_ptr)
                  .ok());
  EXPECT_TRUE(NewHostPortGrpcChannel("[2002:a9c:258e::]:2222",
                                     /*rpc_options=*/nullptr, &mock_ptr)
                  .ok());
  EXPECT_TRUE(
      NewHostPortGrpcChannel("[::]:2222", /*rpc_options=*/nullptr, &mock_ptr)
          .ok());

  EXPECT_FALSE(NewHostPortGrpcChannel("example.com/abc:2222",
                                      /*rpc_options=*/nullptr, &mock_ptr)
                   .ok());
  EXPECT_FALSE(NewHostPortGrpcChannel("127.0.0.1:2222/",
                                      /*rpc_options=*/nullptr, &mock_ptr)
                   .ok());
  EXPECT_FALSE(NewHostPortGrpcChannel(
                   "example.com/abc:", /*rpc_options=*/nullptr, &mock_ptr)
                   .ok());
  EXPECT_FALSE(
      NewHostPortGrpcChannel("[::]/:2222", /*rpc_options=*/nullptr, &mock_ptr)
          .ok());
  EXPECT_FALSE(
      NewHostPortGrpcChannel("[::]:2222/", /*rpc_options=*/nullptr, &mock_ptr)
          .ok());
  EXPECT_FALSE(
      NewHostPortGrpcChannel("[::]:", /*rpc_options=*/nullptr, &mock_ptr).ok());

  EXPECT_TRUE(
      NewHostPortGrpcChannel("/bns/example", /*rpc_options=*/nullptr, &mock_ptr)
          .ok());
}

}  // namespace tsl
