/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/rendezvous.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {

TEST(RendezvousTest, Key) {
  const string key = Rendezvous::CreateKey(
      "/job:mnist/replica:1/task:2/CPU:0", 7890,
      "/job:mnist/replica:1/task:2/device:GPU:0", "var0", FrameAndIter(0, 0));
  EXPECT_EQ(key,
            "/job:mnist/replica:1/task:2/CPU:0;"
            "0000000000001ed2;"  // 7890 = 0x1ed2
            "/job:mnist/replica:1/task:2/device:GPU:0;"
            "var0;"
            "0:0");
  Rendezvous::ParsedKey parsed;
  TF_EXPECT_OK(Rendezvous::ParseKey(key, &parsed));
  EXPECT_EQ(parsed.src_device, "/job:mnist/replica:1/task:2/CPU:0");
  EXPECT_EQ(parsed.src_incarnation, 7890);
  EXPECT_EQ(parsed.src.type, "CPU");
  EXPECT_EQ(parsed.dst_device, "/job:mnist/replica:1/task:2/device:GPU:0");
  EXPECT_EQ(parsed.dst.type, "GPU");

  EXPECT_FALSE(Rendezvous::ParseKey("foo;bar;baz", &parsed).ok());
  EXPECT_FALSE(Rendezvous::ParseKey("/job:mnist/replica:1/task:2/CPU:0;"
                                    "/job:mnist/replica:1/task:2/device:GPU:0;",
                                    &parsed)
                   .ok());
  EXPECT_FALSE(
      Rendezvous::ParseKey(strings::StrCat(key, ";", key), &parsed).ok());
}

class LocalRendezvousTest : public ::testing::Test {
 public:
  LocalRendezvousTest() : threads_(Env::Default(), "test", 16) {
    rendez_ = NewLocalRendezvous();
  }

  ~LocalRendezvousTest() override { rendez_->Unref(); }

  void SchedClosure(std::function<void()> fn) {
    threads_.Schedule(std::move(fn));
  }

  Rendezvous* rendez_;

 private:
  thread::ThreadPool threads_;
};

// string -> Tensor<string>
Tensor V(const string& content) {
  Tensor tensor(DT_STRING, TensorShape({}));
  tensor.scalar<tstring>()() = content;
  return tensor;
}

// Tensor<string> -> string
string V(const Tensor& tensor) {
  CHECK_EQ(tensor.dtype(), DT_STRING);
  CHECK(TensorShapeUtils::IsScalar(tensor.shape()));
  return tensor.scalar<tstring>()();
}

Rendezvous::ParsedKey MakeKey(const string& name) {
  string s = Rendezvous::CreateKey("/job:mnist/replica:1/task:2/CPU:0", 7890,
                                   "/job:mnist/replica:1/task:2/device:GPU:0",
                                   name, FrameAndIter(0, 0));
  Rendezvous::ParsedKey k;
  TF_EXPECT_OK(Rendezvous::ParseKey(s, &k));
  return k;
}

const Rendezvous::ParsedKey& KeyFoo() {
  static auto* key = new Rendezvous::ParsedKey(MakeKey("foo"));
  return *key;
}

const Rendezvous::ParsedKey& KeyBar() {
  static auto* key = new Rendezvous::ParsedKey(MakeKey("bar"));
  return *key;
}

TEST_F(LocalRendezvousTest, SendRecv) {
  Rendezvous::Args args;
  TF_ASSERT_OK(rendez_->Send(KeyFoo(), args, V("hello"), false));
  Tensor val(DT_STRING);
  bool is_dead = false;
  TF_ASSERT_OK(rendez_->Recv(KeyFoo(), args, &val, &is_dead));
  EXPECT_EQ("hello", V(val));
}

TEST_F(LocalRendezvousTest, RecvSend) {
  SchedClosure([this]() {
    Env::Default()->SleepForMicroseconds(10000);
    Rendezvous::Args args;
    TF_ASSERT_OK(rendez_->Send(KeyFoo(), args, V("hello"), false));
  });
  Tensor val(DT_STRING);
  bool is_dead = false;
  Rendezvous::Args args;
  TF_ASSERT_OK(rendez_->Recv(KeyFoo(), args, &val, &is_dead));
  EXPECT_EQ("hello", V(val));
}

TEST_F(LocalRendezvousTest, PingPong) {
  SchedClosure([this]() {
    Tensor t(DT_STRING);
    bool is_dead = false;
    Rendezvous::Args args;
    TF_ASSERT_OK(rendez_->Recv(KeyFoo(), args, &t, &is_dead));
    TF_ASSERT_OK(rendez_->Send(KeyBar(), args, t, is_dead));
  });
  Env::Default()->SleepForMicroseconds(1000000);
  Tensor val(DT_STRING);
  bool val_dead = false;
  Rendezvous::Args args;
  TF_ASSERT_OK(rendez_->Send(KeyFoo(), args, V("secret msg"), val_dead));
  TF_ASSERT_OK(rendez_->Recv(KeyBar(), args, &val, &val_dead));
  EXPECT_EQ("secret msg", V(val));
}

TEST_F(LocalRendezvousTest, CancelBeforeRecv) {
  auto* cm = new CancellationManager();
  Tensor val(DT_STRING);
  bool is_dead = false;
  Rendezvous::Args args;
  args.cancellation_manager = cm;
  cm->StartCancel();
  auto s = rendez_->Recv(KeyFoo(), args, &val, &is_dead);
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(errors::IsCancelled(s));
  EXPECT_EQ("RecvAsync is cancelled.", s.error_message());
  delete cm;
}

TEST_F(LocalRendezvousTest, CancelAfterRecv) {
  auto* cm = new CancellationManager();
  Notification n;
  SchedClosure([cm, &n]() {
    Env::Default()->SleepForMicroseconds(10000);
    cm->StartCancel();
    n.Notify();
  });
  Tensor val(DT_STRING);
  bool is_dead = false;
  Rendezvous::Args args;
  args.cancellation_manager = cm;
  auto s = rendez_->Recv(KeyFoo(), args, &val, &is_dead);
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(errors::IsCancelled(s));
  EXPECT_EQ("RecvAsync is cancelled.", s.error_message());
  n.WaitForNotification();
  delete cm;
}

TEST_F(LocalRendezvousTest, CancelEmptyQueue) {
  auto* cm = new CancellationManager();
  Notification n;
  SchedClosure([this, cm, &n]() {
    Env::Default()->SleepForMicroseconds(10000);
    Rendezvous::Args args;
    TF_ASSERT_OK(rendez_->Send(KeyFoo(), args, V("hello"), false));
    cm->StartCancel();
    n.Notify();
  });
  Tensor val(DT_STRING);
  bool is_dead = false;
  Rendezvous::Args args;
  args.cancellation_manager = cm;
  TF_ASSERT_OK(rendez_->Recv(KeyFoo(), args, &val, &is_dead));
  EXPECT_EQ("hello", V(val));
  n.WaitForNotification();
  delete cm;
}

TEST_F(LocalRendezvousTest, CancelMultiple) {
  auto* cm = new CancellationManager();
  SchedClosure([this, cm]() {
    Env::Default()->SleepForMicroseconds(10000);
    Rendezvous::Args args;
    cm->StartCancel();
    TF_ASSERT_OK(rendez_->Send(KeyFoo(), args, V("hello"), false));
    TF_ASSERT_OK(rendez_->Send(KeyFoo(), args, V("hello"), false));
  });
  Tensor val(DT_STRING);
  Rendezvous::Args args;
  Rendezvous::Args args_with_cancellation;
  args_with_cancellation.cancellation_manager = cm;
  Notification n0;
  Notification n1;
  Notification n2;
  Notification n3;
  Status s0;
  Status s1;
  Status s2;
  Status s3;

  rendez_->RecvAsync(
      KeyFoo(), args,
      [&n0, &s0](const Status& s, const Rendezvous::Args& send_args,
                 const Rendezvous::Args& recv_args, const Tensor& v,
                 const bool dead) {
        s0.Update(s);
        n0.Notify();
      });
  rendez_->RecvAsync(
      KeyFoo(), args_with_cancellation,
      [&n1, &s1](const Status& s, const Rendezvous::Args& send_args,
                 const Rendezvous::Args& recv_args, const Tensor& v,
                 const bool dead) {
        s1.Update(s);
        n1.Notify();
      });
  rendez_->RecvAsync(
      KeyFoo(), args,
      [&n2, &s2](const Status& s, const Rendezvous::Args& send_args,
                 const Rendezvous::Args& recv_args, const Tensor& v,
                 const bool dead) {
        s2.Update(s);
        n2.Notify();
      });
  rendez_->RecvAsync(
      KeyFoo(), args_with_cancellation,
      [&n3, &s3](const Status& s, const Rendezvous::Args& send_args,
                 const Rendezvous::Args& recv_args, const Tensor& v,
                 const bool dead) {
        s3.Update(s);
        n3.Notify();
      });
  n0.WaitForNotification();
  n1.WaitForNotification();
  n2.WaitForNotification();
  n3.WaitForNotification();
  TF_ASSERT_OK(s0);
  TF_ASSERT_OK(s2);
  EXPECT_FALSE(s1.ok());
  EXPECT_FALSE(s3.ok());

  delete cm;
}

// A simple structure that behaves a bit like a blocking counter.  The
// user that decrements counter to 0 does done.Notify(), and the main
// thread waits for done to be notified.
struct BlockingState {
  mutex lock;
  int counter = 0;
  Notification done;
};

TEST_F(LocalRendezvousTest, RandomSendRecv) {
  // We are scheduling 2*N closures in the this->threads_, which is
  // configured with only 16 threads. Furthermore, because the
  // threadpool may execute the closures in an arbitrary order, we
  // must use RecvAsync below. Otherwise, blocking Recv() may run
  // before all the Send() and deadlock.
  static const int N = 100;
  random::PhiloxRandom philox(testing::RandomSeed(), 17);
  random::SimplePhilox rnd(&philox);
  BlockingState state;
  state.counter = N;
  for (int i = 0; i < N; ++i) {
    int micros = 100 + rnd.Uniform(1000);
    SchedClosure([this, i, micros]() {
      Env::Default()->SleepForMicroseconds(micros);
      Rendezvous::Args args;
      TF_ASSERT_OK(rendez_->Send(MakeKey(strings::StrCat(i)), args,
                                 V(strings::StrCat(i)), false));
    });
    auto recv_done = [this, &state, i](const Status& status,
                                       const Rendezvous::Args& sender_args,
                                       const Rendezvous::Args& recver_args,
                                       const Tensor& val, const bool val_dead) {
      EXPECT_EQ(strings::StrCat(i), V(val));
      bool done = false;
      {
        mutex_lock l(state.lock);
        state.counter--;
        if (state.counter == 0) {
          done = true;
        }
      }
      if (done) {
        state.done.Notify();
      }
    };
    micros = 100 + rnd.Uniform(1000);
    SchedClosure([this, i, micros, recv_done]() {
      Env::Default()->SleepForMicroseconds(micros);
      rendez_->RecvAsync(MakeKey(strings::StrCat(i)), Rendezvous::Args(),
                         recv_done);
    });
  }

  state.done.WaitForNotification();
}

void RandomSleep() {
  if (std::rand() % 10 == 0) {
    Env::Default()->SleepForMicroseconds(1000);
  }
}

TEST_F(LocalRendezvousTest, MultiSends) {
  static const int N = 100;
  const auto& key_foo = KeyFoo();
  Rendezvous::Args args;
  SchedClosure([=]() {
    for (int i = 0; i < N; ++i) {
      TF_ASSERT_OK(rendez_->Send(key_foo, args, V(strings::StrCat(i)), false));
      RandomSleep();
    }
  });
  Tensor val;
  bool val_dead;
  for (int i = 0; i < N; ++i) {
    TF_ASSERT_OK(rendez_->Recv(key_foo, args, &val, &val_dead));
    RandomSleep();
  }
}

TEST_F(LocalRendezvousTest, RecvAbort) {
  rendez_->Ref();
  SchedClosure([this]() {
    rendez_->StartAbort(errors::Aborted(""));  // abort
    rendez_->Unref();
  });
  Tensor val(DT_STRING);
  bool val_dead = false;
  Rendezvous::Args args;
  Status status = rendez_->Recv(KeyFoo(), args, &val, &val_dead);
  EXPECT_TRUE(errors::IsAborted(status));
}

// Similar to RecvAbort. But this test case ensures the main thread
// Recv() call happens after StartAbort().
TEST_F(LocalRendezvousTest, RecvSleepAbort) {
  rendez_->Ref();
  SchedClosure([this]() {
    Env::Default()->SleepForMicroseconds(1000000);
    rendez_->StartAbort(errors::Aborted(""));  // abort
    rendez_->Unref();
  });
  Tensor val(DT_STRING);
  bool val_dead = false;
  Rendezvous::Args args;
  Status status = rendez_->Recv(KeyFoo(), args, &val, &val_dead);
  EXPECT_TRUE(errors::IsAborted(status));
}

TEST_F(LocalRendezvousTest, AbortThenRecvOrSend) {
  rendez_->StartAbort(errors::Aborted(""));
  Tensor val(DT_STRING);
  bool val_dead = false;
  Rendezvous::Args args;
  EXPECT_TRUE(errors::IsAborted(rendez_->Send(KeyFoo(), args, val, val_dead)));
  EXPECT_TRUE(
      errors::IsAborted(rendez_->Recv(KeyFoo(), args, &val, &val_dead)));
}

class DummyDeviceContext : public DeviceContext {
 public:
  explicit DummyDeviceContext(int stream_id) : stream_id_(stream_id) {}
  ~DummyDeviceContext() override {}
  int stream_id() const { return stream_id_; }

  void CopyTensorInSameDevice(const Tensor* input_tensor, Device* device,
                              Tensor* output_tensor,
                              StatusCallback done) const override {
    done(Status::OK());
  }

 private:
  const int stream_id_;
};

TEST_F(LocalRendezvousTest, TransferDummyDeviceContext) {
  Rendezvous::Args args;
  args.device_context = new DummyDeviceContext(123);

  TF_ASSERT_OK(rendez_->Send(KeyFoo(), args, V("hello"), false));

  Notification n;
  Rendezvous::Args args1;
  args1.device_context = new DummyDeviceContext(1);
  rendez_->RecvAsync(
      KeyFoo(), args1,
      [&n](const Status& s, const Rendezvous::Args& send_args,
           const Rendezvous::Args& recv_args, const Tensor& val, bool is_dead) {
        CHECK_EQ(123, dynamic_cast<const DummyDeviceContext*>(
                          send_args.device_context)
                          ->stream_id());
        n.Notify();
      });

  n.WaitForNotification();
  args.device_context->Unref();
  args1.device_context->Unref();
}

void BM_SendRecv(::testing::benchmark::State& state) {
  Rendezvous* rendez = NewLocalRendezvous();
  Tensor orig = V("val");
  Tensor val(DT_STRING, TensorShape({}));
  bool is_dead = false;
  Rendezvous::Args args;

  for (auto s : state) {
    TF_CHECK_OK(rendez->Send(KeyFoo(), args, orig, is_dead));
    TF_CHECK_OK(rendez->Recv(KeyFoo(), args, &val, &is_dead));
  }
  CHECK_EQ(V(val), V(orig));

  rendez->Unref();
}
BENCHMARK(BM_SendRecv);

void BM_RecvSend(::testing::benchmark::State& state) {
  Rendezvous* rendez = NewLocalRendezvous();
  Tensor orig = V("val");
  Tensor val(DT_STRING, TensorShape({}));
  bool is_dead = false;
  Rendezvous::Args args;

  for (auto s : state) {
    bool received = false;
    rendez->RecvAsync(
        KeyFoo(), args,
        [&val, &received](const Status& /*s*/,
                          const Rendezvous::Args& /*send_args*/,
                          const Rendezvous::Args& /*recv_args*/,
                          const Tensor& tensor, bool /*is_dead*/) {
          val = tensor;
          received = true;
        });
    TF_CHECK_OK(rendez->Send(KeyFoo(), args, orig, is_dead));
    CHECK(received);
  }
  CHECK_EQ(V(val), V(orig));

  rendez->Unref();
}
BENCHMARK(BM_RecvSend);

void BM_PingPong(::testing::benchmark::State& state) {
  const int messages_count = state.range(0);
  auto* cm = new CancellationManager();
  thread::ThreadPool* pool = new thread::ThreadPool(Env::Default(), "test", 1);

  // Benchmark loop
  // In each iteration:
  // The main thread sends "foo" for messages_count times and receives "bar"
  // for messages_count times.  The other thread sends "bar" for
  // messages_count times and receives "foo" for messages_count times.
  for (auto s : state) {
    Rendezvous* rendez = NewLocalRendezvous();
    pool->Schedule([rendez, messages_count]() {
      Tensor bar = V("bar");
      Tensor foo(DT_STRING, TensorShape({}));
      bool is_dead = false;
      Rendezvous::Args args;
      for (int i = 0; i < messages_count; ++i) {
        TF_CHECK_OK(rendez->Recv(KeyFoo(), args, &foo, &is_dead));
        TF_CHECK_OK(rendez->Send(KeyBar(), args, bar, is_dead));
      }
      CHECK_EQ("foo", V(foo));
    });
    Tensor foo = V("foo");
    Tensor bar(DT_STRING, TensorShape({}));
    bool is_dead = false;
    Rendezvous::Args args;
    args.cancellation_manager = cm;
    for (int i = 0; i < messages_count; ++i) {
      TF_CHECK_OK(rendez->Send(KeyFoo(), args, foo, is_dead));
      TF_CHECK_OK(rendez->Recv(KeyBar(), args, &bar, &is_dead));
    }
    CHECK_EQ("bar", V(bar));
  }
  state.SetItemsProcessed(messages_count * state.iterations());
  delete pool;
  delete cm;
}
BENCHMARK(BM_PingPong)->Arg(100)->Arg(200)->Arg(300);

}  // namespace
}  // namespace tensorflow
