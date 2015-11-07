#include "tensorflow/core/framework/rendezvous.h"

#include <gtest/gtest.h>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/env.h"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/public/tensor_shape.h"

namespace tensorflow {

TEST(RendezvousTest, Key) {
  const string key = Rendezvous::CreateKey(
      "/job:mnist/replica:1/task:2/CPU:0", 7890,
      "/job:mnist/replica:1/task:2/GPU:0", "var0", FrameAndIter(0, 0));
  EXPECT_EQ(key,
            "/job:mnist/replica:1/task:2/CPU:0;"
            "0000000000001ed2;"  // 7890 = 0x1ed2
            "/job:mnist/replica:1/task:2/GPU:0;"
            "var0;"
            "0:0");
  Rendezvous::ParsedKey parsed;
  EXPECT_OK(Rendezvous::ParseKey(key, &parsed));
  EXPECT_EQ(parsed.src_device, "/job:mnist/replica:1/task:2/CPU:0");
  EXPECT_EQ(parsed.src_incarnation, 7890);
  EXPECT_EQ(parsed.src.type, "CPU");
  EXPECT_EQ(parsed.dst_device, "/job:mnist/replica:1/task:2/GPU:0");
  EXPECT_EQ(parsed.dst.type, "GPU");

  EXPECT_FALSE(Rendezvous::ParseKey("foo;bar;baz", &parsed).ok());
  EXPECT_FALSE(Rendezvous::ParseKey("/job:mnist/replica:1/task:2/CPU:0;"
                                    "/job:mnist/replica:1/task:2/GPU:0;",
                                    &parsed)
                   .ok());
  EXPECT_FALSE(
      Rendezvous::ParseKey(strings::StrCat(key, ";", key), &parsed).ok());
}

class LocalRendezvousTest : public ::testing::Test {
 public:
  LocalRendezvousTest()
      : threads_(new thread::ThreadPool(Env::Default(), "test", 16)) {
    rendez_ = NewLocalRendezvous();
  }

  ~LocalRendezvousTest() override {
    rendez_->Unref();
    delete threads_;
  }

  void SchedClosure(std::function<void()> fn) { threads_->Schedule(fn); }

  Rendezvous* rendez_;

 private:
  thread::ThreadPool* threads_;
};

// string -> Tensor<string>
Tensor V(const string& content) {
  Tensor tensor(DT_STRING, TensorShape({}));
  tensor.scalar<string>()() = content;
  return tensor;
}

// Tensor<string> -> string
string V(const Tensor& tensor) {
  CHECK_EQ(tensor.dtype(), DT_STRING);
  CHECK(TensorShapeUtils::IsScalar(tensor.shape()));
  return tensor.scalar<string>()();
}

TEST_F(LocalRendezvousTest, SendRecv) {
  Rendezvous::Args args;
  ASSERT_OK(rendez_->Send("foo", args, V("hello"), false));
  EXPECT_TRUE(errors::IsAborted(rendez_->Send("foo", args, V("hello"), false)));
  Tensor val(DT_STRING);
  bool is_dead = false;
  ASSERT_OK(rendez_->Recv("foo", args, &val, &is_dead));
  EXPECT_EQ("hello", V(val));
}

TEST_F(LocalRendezvousTest, RecvSend) {
  SchedClosure([this]() {
    Env::Default()->SleepForMicroseconds(10000);
    Rendezvous::Args args;
    ASSERT_OK(rendez_->Send("foo", args, V("hello"), false));
  });
  Tensor val(DT_STRING);
  bool is_dead = false;
  Rendezvous::Args args;
  ASSERT_OK(rendez_->Recv("foo", args, &val, &is_dead));
  EXPECT_EQ("hello", V(val));
}

TEST_F(LocalRendezvousTest, DuplicateWaiterRecv) {
  SchedClosure([this]() {
    Tensor t(DT_STRING);
    bool is_dead = false;
    Rendezvous::Args args;
    ASSERT_OK(rendez_->Recv("foo", args, &t, &is_dead));
    ASSERT_OK(rendez_->Send("bar", args, t, is_dead));
  });
  Env::Default()->SleepForMicroseconds(1000000);
  Tensor val(DT_STRING);
  bool val_dead = false;
  Rendezvous::Args args;
  EXPECT_TRUE(errors::IsAborted(rendez_->Recv("foo", args, &val, &val_dead)));
  ASSERT_OK(rendez_->Send("foo", args, V("secret msg"), val_dead));
  ASSERT_OK(rendez_->Recv("bar", args, &val, &val_dead));
  EXPECT_EQ("secret msg", V(val));
}

TEST_F(LocalRendezvousTest, DuplicateSerialRecv) {
  SchedClosure([this]() {
    Tensor t(DT_STRING);
    bool is_dead = false;
    Rendezvous::Args args;
    ASSERT_OK(rendez_->Recv("foo", args, &t, &is_dead));
    ASSERT_OK(rendez_->Send("bar", args, t, is_dead));
  });
  Env::Default()->SleepForMicroseconds(1000000);
  Tensor val(DT_STRING);
  bool val_dead = false;
  Rendezvous::Args args;
  ASSERT_OK(rendez_->Send("foo", args, V("secret msg"), val_dead));
  ASSERT_OK(rendez_->Recv("bar", args, &val, &val_dead));
  EXPECT_EQ("secret msg", V(val));
  EXPECT_TRUE(errors::IsAborted(rendez_->Recv("foo", args, &val, &val_dead)));
}

// A simple structure that behaves a bit like a blocking counter.  The
// user that decrements counter to 0 does done.Notify(), and the main
// thread waits for done to be notified.
struct BlockingState {
  mutex lock;
  int counter;
  Notification done;
};

TEST_F(LocalRendezvousTest, RandomSendRecv) {
  static const int N = 1000;
  BlockingState state;
  state.counter = N;
  for (int i = 0; i < N; ++i) {
    SchedClosure([this, i]() {
      random::PhiloxRandom philox(testing::RandomSeed() + i, 17);
      random::SimplePhilox rnd(&philox);
      Env::Default()->SleepForMicroseconds(1000 + rnd.Uniform(10000));
      Rendezvous::Args args;
      ASSERT_OK(rendez_->Send(strings::StrCat(i), args, V(strings::StrCat(i)),
                              false));
    });
    SchedClosure([this, &state, i]() {
      random::PhiloxRandom philox(testing::RandomSeed() + N + i, 17);
      random::SimplePhilox rnd(&philox);
      Env::Default()->SleepForMicroseconds(1000 + rnd.Uniform(10000));
      Tensor val(DT_STRING);
      bool val_dead = false;
      Rendezvous::Args args;
      ASSERT_OK(rendez_->Recv(strings::StrCat(i), args, &val, &val_dead));
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
    });
  }

  state.done.WaitForNotification();
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
  Status status = rendez_->Recv("foo", args, &val, &val_dead);
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
  Status status = rendez_->Recv("foo", args, &val, &val_dead);
  EXPECT_TRUE(errors::IsAborted(status));
}

TEST_F(LocalRendezvousTest, AbortThenRecvOrSend) {
  rendez_->StartAbort(errors::Aborted(""));
  Tensor val(DT_STRING);
  bool val_dead = false;
  Rendezvous::Args args;
  EXPECT_TRUE(errors::IsAborted(rendez_->Send("foo", args, val, val_dead)));
  EXPECT_TRUE(errors::IsAborted(rendez_->Recv("foo", args, &val, &val_dead)));
}

class DummyDeviceContext : public DeviceContext {
 public:
  explicit DummyDeviceContext(int stream_id) : stream_id_(stream_id) {}
  ~DummyDeviceContext() override {}
  int stream_id() const { return stream_id_; }

 private:
  const int stream_id_;
};

TEST_F(LocalRendezvousTest, TransferDummyDeviceContext) {
  Rendezvous::Args args;
  args.device_context = new DummyDeviceContext(123);

  ASSERT_OK(rendez_->Send("foo", args, V("hello"), false));

  Notification n;
  Rendezvous::Args args1;
  args1.device_context = new DummyDeviceContext(1);
  rendez_->RecvAsync("foo", args1, [&n](const Status& s,
                                        const Rendezvous::Args& send_args,
                                        const Rendezvous::Args& recv_args,
                                        const Tensor& val, bool is_dead) {
    CHECK_EQ(123,
             dynamic_cast<const DummyDeviceContext*>(send_args.device_context)
                 ->stream_id());
    n.Notify();
  });

  n.WaitForNotification();
  args.device_context->Unref();
  args1.device_context->Unref();
}

static void BM_SendRecv(int iters) {
  Rendezvous* rendez = NewLocalRendezvous();
  Tensor orig = V("val");
  Tensor val(DT_STRING, TensorShape({}));
  bool is_dead = false;
  Rendezvous::Args args;
  Status s;
  if (iters > 0) {
    while (iters--) {
      s = rendez->Send("foo", args, orig, is_dead);
      s = rendez->Recv("foo", args, &val, &is_dead);
    }
    CHECK_EQ(V(val), V(orig));
  }
  rendez->Unref();
}
BENCHMARK(BM_SendRecv);

static void BM_RecvSend(int iters) {
  thread::ThreadPool* pool = new thread::ThreadPool(Env::Default(), "test", 1);

  // The main thread sends "foo" for iters/2 times and receives "bar"
  // for iters/2 times.  The other thread sends "bar" for iters/2
  // times and receives "foo" for iters/2 times.
  Rendezvous* rendez = NewLocalRendezvous();
  pool->Schedule([rendez, iters]() {
    Tensor bar = V("bar");
    Tensor foo(DT_STRING, TensorShape({}));
    bool is_dead = false;
    Rendezvous::Args args;
    Status s;
    for (int i = 0; i < iters / 2; ++i) {
      s = rendez->Recv("foo", args, &foo, &is_dead);
      s = rendez->Send("bar", args, bar, is_dead);
    }
    CHECK_EQ("foo", V(foo));
  });
  Tensor foo = V("foo");
  Tensor bar(DT_STRING, TensorShape({}));
  bool is_dead = false;
  Rendezvous::Args args;
  Status s;
  for (int i = 0; i < iters / 2; ++i) {
    s = rendez->Send("foo", args, foo, is_dead);
    s = rendez->Recv("bar", args, &bar, &is_dead);
  }
  CHECK_EQ("bar", V(bar));
  delete pool;
}
BENCHMARK(BM_RecvSend);

}  // namespace tensorflow
