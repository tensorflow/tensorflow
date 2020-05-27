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

#include "tensorflow/compiler/xla/python/outfeed_receiver.h"

#include <memory>

#include "absl/synchronization/mutex.h"
#include "tensorflow/compiler/xla/client/executable_build_options.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/pjrt/cpu_device.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/test.h"

namespace xla {

namespace {

Status CompileAndExecute(XlaBuilder* builder, XlaOp root, int device_id,
                         PjRtClient* client) {
  XlaComputation computation = builder->Build(root).ValueOrDie();

  CompileOptions compile_options;
  compile_options.executable_build_options.set_num_replicas(1);
  compile_options.executable_build_options.set_num_partitions(1);
  DeviceAssignment device_assignment(1, 1);
  device_assignment(0, 0) = device_id;
  compile_options.executable_build_options.set_device_assignment(
      device_assignment);

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<PjRtExecutable> executable,
      PjRtExecutable::Compile(computation, client, std::move(compile_options)));
  ExecuteOptions execute_options;
  TF_ASSIGN_OR_RETURN(std::vector<std::unique_ptr<PjRtBuffer>> output_buffers,
                      executable->Execute({}, execute_options));
  return Status::OK();
}

// Accumulates the received data.
class Accumulator {
 public:
  struct Data {
    uint32_t consumer_id;
    std::shared_ptr<Literal> data;
  };

  void Receive(uint32_t consumer_id, std::shared_ptr<Literal> data) {
    absl::MutexLock lock(&mutex_);
    received_.push_back(Data{consumer_id, data});
  }

  std::vector<Data> received() {
    absl::MutexLock lock(&mutex_);
    return received_;
  }

 private:
  absl::Mutex mutex_;
  std::vector<Data> received_ TF_GUARDED_BY(mutex_);
};

TEST(OutfeedReceiverTest, ReceiveOutfeedSimple) {
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<PjRtClient> cpu_client,
                          GetCpuClient(true));
  std::vector<std::shared_ptr<PjRtClient>> clients{cpu_client};

  auto receiver = absl::make_unique<Accumulator>();
  OutfeedReceiver::Callback callback =
      [&receiver](Device* device, std::shared_ptr<PjRtClient> client,
                  uint32_t consumer_id, std::shared_ptr<Literal> data) {
        receiver->Receive(consumer_id, data);
      };
  auto outfeed_receiver =
      std::make_shared<OutfeedReceiver>(callback, clients, 128);
  outfeed_receiver->Start();

  XlaBuilder builder("execute_test_outfeed");
  constexpr int consumer_id0 = 5;
  const Shape shape0 = ShapeUtil::MakeShape(U32, {16});
  XlaOp data = Iota(&builder, shape0, 0);
  XlaOp send = outfeed_receiver
                   ->AddOutfeedToBuilder(&builder, CreateToken(&builder),
                                         consumer_id0, {data})
                   .ValueOrDie();
  EXPECT_TRUE(CompileAndExecute(&builder, send, 0, cpu_client.get()).ok());

  // Shutdown the receiver, to force it to wait to deliver the callbacks.
  outfeed_receiver = nullptr;
  std::vector<Accumulator::Data> received = receiver->received();
  EXPECT_EQ(1, received.size());
  EXPECT_EQ(consumer_id0, received[0].consumer_id);
  EXPECT_EQ(ShapeUtil::MakeTupleShape({shape0}), received[0].data->shape());
}

TEST(OutfeedReceiverTest, ReceiveOutfeedTwoComputations) {
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<PjRtClient> cpu_client,
                          GetCpuClient(true));
  std::vector<std::shared_ptr<PjRtClient>> clients{cpu_client};

  auto receiver = absl::make_unique<Accumulator>();
  OutfeedReceiver::Callback callback =
      [&receiver](Device* device, std::shared_ptr<PjRtClient> client,
                  uint32_t consumer_id, std::shared_ptr<Literal> data) {
        receiver->Receive(consumer_id, data);
      };
  auto outfeed_receiver =
      std::make_shared<OutfeedReceiver>(callback, clients, 128);
  outfeed_receiver->Start();

  XlaBuilder builder0("execute_test_outfeed_0");
  constexpr int consumer_id0 = 5;
  const Shape shape0 = ShapeUtil::MakeShape(U32, {16});
  XlaOp data0 = Iota(&builder0, shape0, 0);
  XlaOp send0 = outfeed_receiver
                    ->AddOutfeedToBuilder(&builder0, CreateToken(&builder0),
                                          consumer_id0, {data0})
                    .ValueOrDie();
  EXPECT_TRUE(CompileAndExecute(&builder0, send0, 0, cpu_client.get()).ok());

  XlaBuilder builder1("execute_test_outfeed_1");
  constexpr int consumer_id1 = 6;
  const Shape shape1 = ShapeUtil::MakeShape(U32, {128});
  XlaOp data1 = Iota(&builder1, shape1, 0);
  XlaOp send1 = outfeed_receiver
                    ->AddOutfeedToBuilder(&builder1, CreateToken(&builder1),
                                          consumer_id1, {data1})
                    .ValueOrDie();
  EXPECT_TRUE(CompileAndExecute(&builder1, send1, 0, cpu_client.get()).ok());

  // Shutdown the receiver, to force it to wait to deliver the callbacks.
  outfeed_receiver = nullptr;
  std::vector<Accumulator::Data> received = receiver->received();
  EXPECT_EQ(2, received.size());
  EXPECT_EQ(consumer_id0, received[0].consumer_id);
  EXPECT_EQ(ShapeUtil::MakeTupleShape({shape0}), received[0].data->shape());
  EXPECT_EQ(consumer_id1, received[1].consumer_id);
  EXPECT_EQ(ShapeUtil::MakeTupleShape({shape1}), received[1].data->shape());
}

TEST(OutfeedReceiverTest, ReceiveOutfeedTwoOutfeed) {
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<PjRtClient> cpu_client,
                          GetCpuClient(true));
  std::vector<std::shared_ptr<PjRtClient>> clients{cpu_client};

  auto receiver = absl::make_unique<Accumulator>();
  OutfeedReceiver::Callback callback =
      [&receiver](Device* device, std::shared_ptr<PjRtClient> client,
                  uint32_t consumer_id, std::shared_ptr<Literal> data) {
        receiver->Receive(consumer_id, data);
      };
  auto outfeed_receiver =
      std::make_shared<OutfeedReceiver>(callback, clients, 128);
  outfeed_receiver->Start();

  XlaBuilder builder("execute_test_outfeed");
  constexpr int consumer_id0 = 5;
  const Shape shape0 = ShapeUtil::MakeShape(U32, {16});
  XlaOp data0 = Iota(&builder, shape0, 0);
  XlaOp send0 = outfeed_receiver
                    ->AddOutfeedToBuilder(&builder, CreateToken(&builder),
                                          consumer_id0, {data0})
                    .ValueOrDie();

  constexpr int consumer_id1 = 6;
  const Shape shape1 = ShapeUtil::MakeShape(U32, {128});
  XlaOp data1 = Iota(&builder, shape1, 0);
  XlaOp send1 =
      outfeed_receiver
          ->AddOutfeedToBuilder(&builder, send0, consumer_id1, {data1})
          .ValueOrDie();
  EXPECT_TRUE(CompileAndExecute(&builder, send1, 0, cpu_client.get()).ok());

  // Shutdown the receiver, to force it to wait to deliver the callbacks.
  outfeed_receiver = nullptr;
  std::vector<Accumulator::Data> received = receiver->received();
  EXPECT_EQ(2, received.size());
  EXPECT_EQ(consumer_id0, received[0].consumer_id);
  EXPECT_EQ(ShapeUtil::MakeTupleShape({shape0}), received[0].data->shape());
  EXPECT_EQ(consumer_id1, received[1].consumer_id);
  EXPECT_EQ(ShapeUtil::MakeTupleShape({shape1}), received[1].data->shape());
}

TEST(OutfeedReceiverTest, DifferentShapeForConsumerIdError) {
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<PjRtClient> cpu_client,
                          GetCpuClient(true));
  std::vector<std::shared_ptr<PjRtClient>> clients{cpu_client};

  auto receiver = absl::make_unique<Accumulator>();
  OutfeedReceiver::Callback callback =
      [&receiver](Device* device, std::shared_ptr<PjRtClient> client,
                  uint32_t consumer_id, std::shared_ptr<Literal> data) {
        receiver->Receive(consumer_id, data);
      };
  auto outfeed_receiver =
      std::make_shared<OutfeedReceiver>(callback, clients, 128);
  outfeed_receiver->Start();

  XlaBuilder builder("execute_test_outfeed");
  constexpr int consumer_id0 = 5;
  const Shape shape0 = ShapeUtil::MakeShape(U32, {16});
  XlaOp data0 = Iota(&builder, shape0, 0);
  XlaOp send0 = outfeed_receiver
                    ->AddOutfeedToBuilder(&builder, CreateToken(&builder),
                                          consumer_id0, {data0})
                    .ValueOrDie();

  const Shape shape1 = ShapeUtil::MakeShape(U32, {128});
  XlaOp data1 = Iota(&builder, shape1, 0);
  // A different shape for the same consumer ID.
  StatusOr<XlaOp> send1 = outfeed_receiver->AddOutfeedToBuilder(
      &builder, send0, consumer_id0, {data1});
  EXPECT_FALSE(send1.ok());
  EXPECT_THAT(send1.status().ToString(),
              testing::HasSubstr("does not match previous shape element_type"));
}

TEST(OutfeedReceiverTest, InvalidConsumerIdError) {
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<PjRtClient> cpu_client,
                          GetCpuClient(true));
  std::vector<std::shared_ptr<PjRtClient>> clients{cpu_client};

  auto receiver = absl::make_unique<Accumulator>();
  OutfeedReceiver::Callback callback =
      [&receiver](Device* device, std::shared_ptr<PjRtClient> client,
                  uint32_t consumer_id, std::shared_ptr<Literal> data) {
        receiver->Receive(consumer_id, data);
      };
  auto outfeed_receiver =
      std::make_shared<OutfeedReceiver>(callback, clients, 128);
  outfeed_receiver->Start();

  XlaBuilder builder("execute_test_outfeed");
  const Shape shape0 = ShapeUtil::MakeShape(U32, {16});
  XlaOp data0 = Iota(&builder, shape0, 0);
  StatusOr<XlaOp> send0 = outfeed_receiver->AddOutfeedToBuilder(
      &builder, CreateToken(&builder), 0, {data0});

  EXPECT_FALSE(send0.ok());
  EXPECT_THAT(send0.status().ToString(),
              testing::HasSubstr("Consumer ID cannot be a reserved value"));
}

}  // namespace

}  // namespace xla
