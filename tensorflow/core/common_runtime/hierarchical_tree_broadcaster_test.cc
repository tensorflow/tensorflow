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
#include "tensorflow/core/common_runtime/hierarchical_tree_broadcaster.h"

#include <algorithm>

#include "absl/memory/memory.h"
#include "tensorflow/core/common_runtime/base_collective_executor.h"
#include "tensorflow/core/common_runtime/collective_rma_local.h"
#include "tensorflow/core/common_runtime/collective_test_util.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/device_resolver_local.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/common_runtime/test_collective_executor_mgr.h"
#include "tensorflow/core/common_runtime/threadpool_device.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/unbounded_work_queue.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace {

// The test harness won't allow a mixture of fixture and non-fixture
// tests in one file, so this is a trivial fixture for tests that don't
// need the heavy-weight HierarchicalTreeBroadcasterTest fixture.
class TrivialTest : public ::testing::Test {
 protected:
  TrivialTest() {}
};

// Tests of static TreeSendTo() and TreeRecvFrom() functions.
// D = number of devices
// S = source rank
// R = tested rank
// RF = receive-from rank
// ST = send_to rank vector
#define DEF_TL_TEST(D, S, R, RF, ST)                                  \
  TEST_F(TrivialTest, TreeLinks_##D##Devs_##S##Source_##R##Rank) {    \
    auto* cp = new CollectiveParams();                                \
    core::ScopedUnref unref(cp);                                      \
    cp->group.group_size = D;                                         \
    cp->instance.impl_details.subdiv_source_rank = {S};               \
    cp->instance.impl_details.subdiv_permutations.push_back(          \
        std::vector<int>(D, 0));                                      \
    cp->subdiv_rank = {R};                                            \
    cp->is_source = (S == R);                                         \
    EXPECT_EQ(RF, HierarchicalTreeBroadcaster::TreeRecvFrom(*cp, 0)); \
    std::vector<int> expected = ST;                                   \
    std::vector<int> send_to;                                         \
    HierarchicalTreeBroadcaster::TreeSendTo(*cp, 0, &send_to);        \
    ASSERT_EQ(expected.size(), send_to.size());                       \
    for (int i = 0; i < expected.size(); ++i) {                       \
      EXPECT_EQ(expected[i], send_to[i]);                             \
    }                                                                 \
  }

#define V(...) std::vector<int>({__VA_ARGS__})

//          D  S  R  RF  ST
// 2 device cases
DEF_TL_TEST(2, 0, 0, -1, V(1))
DEF_TL_TEST(2, 1, 0, 1, V())
DEF_TL_TEST(2, 0, 1, 0, V())
DEF_TL_TEST(2, 1, 1, -1, V(0))
// 3 device cases
DEF_TL_TEST(3, 0, 0, -1, V(1, 2))
DEF_TL_TEST(3, 0, 1, 0, V())
DEF_TL_TEST(3, 0, 2, 0, V())
DEF_TL_TEST(3, 1, 0, 1, V(2))
DEF_TL_TEST(3, 1, 1, -1, V(0))
DEF_TL_TEST(3, 1, 2, 0, V())
DEF_TL_TEST(3, 2, 0, 2, V())
DEF_TL_TEST(3, 2, 1, 2, V())
DEF_TL_TEST(3, 2, 2, -1, V(0, 1))
// 4 device cases
DEF_TL_TEST(4, 0, 0, -1, V(1, 2))
DEF_TL_TEST(4, 0, 1, 0, V(3))
DEF_TL_TEST(4, 0, 2, 0, V())
DEF_TL_TEST(4, 0, 3, 1, V())
DEF_TL_TEST(4, 1, 0, 1, V(2, 3))
DEF_TL_TEST(4, 1, 1, -1, V(0))
DEF_TL_TEST(4, 1, 2, 0, V())
DEF_TL_TEST(4, 1, 3, 0, V())
DEF_TL_TEST(4, 2, 0, 2, V(3))
DEF_TL_TEST(4, 2, 1, 2, V())
DEF_TL_TEST(4, 2, 2, -1, V(0, 1))
DEF_TL_TEST(4, 2, 3, 0, V())
DEF_TL_TEST(4, 3, 0, 3, V(2))
DEF_TL_TEST(4, 3, 1, 3, V())
DEF_TL_TEST(4, 3, 2, 0, V())
DEF_TL_TEST(4, 3, 3, -1, V(0, 1))
// 8 device cases
//          D  S  R  RF  ST
DEF_TL_TEST(8, 0, 0, -1, V(1, 2))
DEF_TL_TEST(8, 0, 1, 0, V(3, 4))
DEF_TL_TEST(8, 0, 2, 0, V(5, 6))
DEF_TL_TEST(8, 0, 3, 1, V(7))
DEF_TL_TEST(8, 0, 4, 1, V())
DEF_TL_TEST(8, 0, 5, 2, V())
DEF_TL_TEST(8, 0, 6, 2, V())
DEF_TL_TEST(8, 0, 7, 3, V())
DEF_TL_TEST(8, 7, 0, 7, V(2, 3))
DEF_TL_TEST(8, 7, 1, 7, V(4, 5))
DEF_TL_TEST(8, 7, 2, 0, V(6))
DEF_TL_TEST(8, 7, 3, 0, V())
DEF_TL_TEST(8, 7, 4, 1, V())
DEF_TL_TEST(8, 7, 5, 1, V())
DEF_TL_TEST(8, 7, 6, 2, V())
DEF_TL_TEST(8, 7, 7, -1, V(0, 1))
#undef DEF_TL_TEST
#undef V

class HierarchicalTreeBroadcasterTest : public ::testing::Test {
 protected:
  void Init(int num_workers, int num_devices, DataType dtype,
            const TensorShape& shape, const DeviceType& device_type,
            int fail_after) {
    test_env_ = CreateCollectiveTestEnv(num_workers, num_devices, device_type);
    test_env_->remote_access->set_fail_after(fail_after);
    for (int wi = 0; wi < num_workers; ++wi) {
      for (int di = 0; di < num_devices; ++di) {
        int rank = wi * num_devices + di;
        instances_.push_back(std::make_unique<DeviceInstance>(
            rank, dtype, shape, test_env_.get()));
      }
    }
  }

  typedef std::function<void(Tensor*)> InitFunc;

  void Broadcast(bool forward_input) {
    VLOG(2) << "#instances=" << instances_.size();
    std::atomic<int> done(0);
    for (auto& di : instances_) {
      SchedClosure([&di, forward_input, &done] {
        di->DoBroadcast(forward_input);
        ++done;
      });
    }
    while (done < instances_.size()) {
      Env::Default()->SleepForMicroseconds(1000);
    }
  }

  template <typename T>
  void RunTest(DataType dtype, const DeviceType& device_type, int num_workers,
               int num_devices, int tensor_len, int fail_after,
               bool forward_input) {
    Init(num_workers, num_devices, dtype, TensorShape({tensor_len}),
         device_type, fail_after);

    // Initialize each instance tensor with distinct values.
    for (int di = 0; di < instances_.size(); ++di) {
      instances_[di]->InitTensor([di](Tensor* t) {
        for (size_t i = 0; i < t->NumElements(); ++i) {
          // The cast is necessary to prevent clang-tidy from insisting
          // that a faster non-open source function be substituted.
          float value = pow(10, static_cast<double>(di)) * i;
          t->flat<T>()(i) = value;
        }
      });
    }

    Tensor expected = instances_[0]->input_tensor_;
    Broadcast(forward_input);
    // At this point all of the ops have terminated.
    for (int di = 0; di < instances_.size(); ++di) {
      if (!instances_[di]->status_.ok()) {
        ASSERT_GT(fail_after, 0);
        ASSERT_NE(
            instances_[di]->status_.error_message().find("Deliberate failure"),
            string::npos);
        ++failure_count_;
        continue;
      }
      test::ExpectTensorEqual<T>(expected, instances_[di]->output_tensor_);
    }

    // Note that the order of operations during broadcast is
    // non-deterministic and unlike the reduce case some Ops in the
    // instance may succeed while others fail, even if a transmission
    // failure occurs early in the operation chain.  So, when an abort
    // is specified we need to verify that at least one Op fails with
    // the expected status and any Op that succeeds yields the correct
    // value.
    if (fail_after > 0) {
      EXPECT_GT(failure_count_, 0);
    }
  }

  class DeviceInstance {
   public:
    DeviceInstance(int rank, DataType dtype, const TensorShape& shape,
                   CollectiveTestEnv* test_env)
        : test_env_(test_env), input_tensor_(dtype, shape) {
      col_params_ =
          CreateCollectiveParams(*test_env_, rank, "HierarchicalTreeBroadcast",
                                 BROADCAST_COLLECTIVE, dtype, shape);
      // In the test we always broadcast from rank 0.
      col_params_->is_source = (rank == 0);
      col_params_->source_rank = 0;
      string dev_name = col_params_->group.members[rank].device.name();
      TF_CHECK_OK(test_env_->device_mgr->LookupDevice(dev_name, &device_))
          << "Couldn't find device " << dev_name
          << " existing devices: " << test_env_->device_mgr->DebugString();
    }

    void InitTensor(const InitFunc& f) { f(&input_tensor_); }

    void DoBroadcast(bool forward_input) {
      if (forward_input) {
        output_tensor_ = input_tensor_;
      } else {
        output_tensor_ = Tensor(input_tensor_.dtype(), input_tensor_.shape());
      }
      status_ = RunCollective(test_env_, col_params_.get(), device_,
                              &input_tensor_, &output_tensor_);
    }

    CollectiveTestEnv* test_env_;
    Tensor input_tensor_;
    Tensor output_tensor_;
    Device* device_;
    core::RefCountPtr<CollectiveParams> col_params_;
    Status status_;
  };  // class DeviceInstance

  std::unique_ptr<CollectiveTestEnv> test_env_;
  std::vector<std::unique_ptr<DeviceInstance>> instances_;
  int failure_count_ = 0;
};

class HierarchicalTreeBroadcasterInitParamsTest : public ::testing::Test {
 protected:
  void RunSubdivPermsTest(
      CollectiveParams* cp,
      const std::vector<std::vector<int>>& expected_subdiv_perms,
      const std::vector<int>& expected_subdiv_rank,
      const std::vector<int>& expected_subdiv_source_rank) {
    cp->instance.impl_details.subdiv_permutations.clear();
    cp->subdiv_rank.clear();
    cp->instance.impl_details.subdiv_source_rank.clear();
    // Create a stub broadcaster only for testing param initialization.
    HierarchicalTreeBroadcaster* broadcaster = new HierarchicalTreeBroadcaster;
    core::ScopedUnref unref(broadcaster);
    TF_CHECK_OK(broadcaster->InitializeCollectiveParams(cp));
    EXPECT_EQ(expected_subdiv_perms,
              cp->instance.impl_details.subdiv_permutations);
    EXPECT_EQ(expected_subdiv_rank, cp->subdiv_rank);
    EXPECT_EQ(expected_subdiv_source_rank,
              cp->instance.impl_details.subdiv_source_rank);
  }
};

TEST_F(HierarchicalTreeBroadcasterInitParamsTest,
       InitializeParams1Task8Device) {
  const int kNumDevsPerWorker = 8;
  const int kNumWorkers = 1;
  auto test_env =
      CreateCollectiveTestEnv(kNumWorkers, kNumDevsPerWorker, DEVICE_CPU);
  auto cp =
      CreateCollectiveParams(*test_env, /*rank*/ 0, "HierarchicalTreeBroadcast",
                             BROADCAST_COLLECTIVE, DT_FLOAT, TensorShape({1}));

  // source 0 device 0
  cp->source_rank = 0;
  cp->default_rank = 0;
  RunSubdivPermsTest(cp.get(), {{0, 1, 2, 3, 4, 5, 6, 7}}, {0}, {0});

  // source 2 device 2
  cp->source_rank = 2;
  cp->default_rank = 2;
  RunSubdivPermsTest(cp.get(), {{0, 1, 2, 3, 4, 5, 6, 7}}, {2}, {2});

  // source 2 device 0
  cp->source_rank = 2;
  cp->default_rank = 0;
  RunSubdivPermsTest(cp.get(), {{0, 1, 2, 3, 4, 5, 6, 7}}, {0}, {2});
}

TEST_F(HierarchicalTreeBroadcasterInitParamsTest,
       InitializeParams4Tasks8Device) {
  const int kNumDevsPerWorker = 8;
  const int kNumWorkers = 4;
  auto test_env =
      CreateCollectiveTestEnv(kNumWorkers, kNumDevsPerWorker, DEVICE_CPU);
  auto cp =
      CreateCollectiveParams(*test_env, /*rank*/ 0, "HierarchicalTreeBroadcast",
                             BROADCAST_COLLECTIVE, DT_FLOAT, TensorShape({1}));

  // source 0 device 0
  cp->source_rank = 0;
  cp->default_rank = 0;
  RunSubdivPermsTest(cp.get(),
                     {{0, 8, 16, 24},
                      {0, 1, 2, 3, 4, 5, 6, 7},
                      {8, 9, 10, 11, 12, 13, 14, 15},
                      {16, 17, 18, 19, 20, 21, 22, 23},
                      {24, 25, 26, 27, 28, 29, 30, 31}},
                     {0, 0, -1, -1, -1}, {0, 0, 0, 0, 0});

  // source 2 device 0
  cp->source_rank = 2;
  cp->default_rank = 0;
  RunSubdivPermsTest(cp.get(),
                     {{2, 8, 16, 24},
                      {0, 1, 2, 3, 4, 5, 6, 7},
                      {8, 9, 10, 11, 12, 13, 14, 15},
                      {16, 17, 18, 19, 20, 21, 22, 23},
                      {24, 25, 26, 27, 28, 29, 30, 31}},
                     {-1, 0, -1, -1, -1}, {0, 2, 0, 0, 0});

  // source 9 device 9
  cp->source_rank = 9;
  cp->default_rank = 9;
  RunSubdivPermsTest(cp.get(),
                     {{0, 9, 16, 24},
                      {0, 1, 2, 3, 4, 5, 6, 7},
                      {8, 9, 10, 11, 12, 13, 14, 15},
                      {16, 17, 18, 19, 20, 21, 22, 23},
                      {24, 25, 26, 27, 28, 29, 30, 31}},
                     {1, -1, 1, -1, -1}, {1, 0, 1, 0, 0});
}

TEST_F(HierarchicalTreeBroadcasterInitParamsTest,
       InitializeParams4TasksVariableDevice) {
  auto* cp = new CollectiveParams();
  core::ScopedUnref unref(cp);
  int num_tasks = 4;
  cp->group.device_type = DeviceType("GPU");
  cp->group.num_tasks = num_tasks;
  cp->group.group_size = 0;
  cp->instance.type = BROADCAST_COLLECTIVE;
  cp->instance.impl_details.collective_name = "HierarchicalTreeBroadcast";
  std::vector<int> dev_per_task = {4, 4, 6, 8};
  for (int ti = 0; ti < cp->group.num_tasks; ti++) {
    string task_name = strings::StrCat("/job:worker/replica:0/task:", ti);
    for (int di = 0; di < dev_per_task[ti]; di++) {
      CollGroupMember member;
      member.device.set_name(strings::StrCat(task_name, "/device:GPU:", di));
      member.task = task_name;
      cp->group.members.push_back(member);
      cp->group.group_size++;
    }
  }

  // source 0 device 0
  cp->source_rank = 0;
  cp->default_rank = 0;
  RunSubdivPermsTest(cp,
                     {{0, 4, 8, 14},
                      {0, 1, 2, 3},
                      {4, 5, 6, 7},
                      {8, 9, 10, 11, 12, 13},
                      {14, 15, 16, 17, 18, 19, 20, 21}},
                     {0, 0, -1, -1, -1}, {0, 0, 0, 0, 0});

  // source 2 device 0
  cp->source_rank = 2;
  cp->default_rank = 0;
  RunSubdivPermsTest(cp,
                     {{2, 4, 8, 14},
                      {0, 1, 2, 3},
                      {4, 5, 6, 7},
                      {8, 9, 10, 11, 12, 13},
                      {14, 15, 16, 17, 18, 19, 20, 21}},
                     {-1, 0, -1, -1, -1}, {0, 2, 0, 0, 0});

  // source 9 device 5
  cp->source_rank = 9;
  cp->default_rank = 5;
  RunSubdivPermsTest(cp,
                     {{0, 4, 9, 14},
                      {0, 1, 2, 3},
                      {4, 5, 6, 7},
                      {8, 9, 10, 11, 12, 13},
                      {14, 15, 16, 17, 18, 19, 20, 21}},
                     {-1, -1, 1, -1, -1}, {2, 0, 0, 1, 0});
}

// TODO(b/113171733): change to use TEST_P.
// Tests of full broadcast algorithm, with different device and
// data types.
// B = data element type
// T = device type
// W = number of workers
// D = number of devices per worker
// L = tensor length
// A = abort after count
// F = forward input
#define DEF_TEST(B, T, W, D, L, A, F)                                      \
  TEST_F(HierarchicalTreeBroadcasterTest,                                  \
         DaTy##B##_DevTy##T##_Wkr##W##_Dev##D##_Len##L##_Abt##A##_Fw##F) { \
    DataType dtype = DT_##B;                                               \
    switch (dtype) {                                                       \
      case DT_BOOL: {                                                      \
        RunTest<bool>(dtype, DEVICE_##T, W, D, L, A, F);                   \
      } break;                                                             \
      case DT_FLOAT: {                                                     \
        RunTest<float>(dtype, DEVICE_##T, W, D, L, A, F);                  \
      } break;                                                             \
      case DT_DOUBLE: {                                                    \
        RunTest<double>(dtype, DEVICE_##T, W, D, L, A, F);                 \
      } break;                                                             \
      case DT_INT32: {                                                     \
        RunTest<int32>(dtype, DEVICE_##T, W, D, L, A, F);                  \
      } break;                                                             \
      case DT_INT64: {                                                     \
        RunTest<int64_t>(dtype, DEVICE_##T, W, D, L, A, F);                \
      } break;                                                             \
      default:                                                             \
        LOG(FATAL) << "Unimplemented";                                     \
    }                                                                      \
  }

#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
//       B      T    W  D  L  A  F
DEF_TEST(FLOAT, CPU, 1, 2, 1, 0, false)
DEF_TEST(FLOAT, CPU, 1, 2, 1001, 0, true)
DEF_TEST(FLOAT, CPU, 2, 1, 128, 0, false)
DEF_TEST(FLOAT, CPU, 2, 4, 128, 0, true)
DEF_TEST(FLOAT, CPU, 2, 8, 4095, 0, false)
DEF_TEST(FLOAT, CPU, 4, 4, 1045991, 0, true)

DEF_TEST(BOOL, CPU, 1, 4, 1, 0, false)
DEF_TEST(BOOL, CPU, 2, 4, 1, 0, false)
DEF_TEST(BOOL, CPU, 2, 4, 1001, 0, false)

DEF_TEST(DOUBLE, CPU, 2, 4, 128, 0, false)
DEF_TEST(INT32, CPU, 2, 4, 128, 0, true)
DEF_TEST(INT64, CPU, 2, 4, 128, 0, false)

// Failure cases
DEF_TEST(FLOAT, CPU, 2, 4, 128, 1, true)
DEF_TEST(FLOAT, CPU, 2, 4, 128, 5, false)
#endif

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Can only set W=1 for GPU tests.
//       B      T    W  D  L  A  F
DEF_TEST(FLOAT, GPU, 1, 2, 1, 0, true)
DEF_TEST(FLOAT, GPU, 1, 2, 33, 0, false)
DEF_TEST(FLOAT, GPU, 1, 3, 64, 0, true)
DEF_TEST(FLOAT, GPU, 1, 8, 1001, 0, false)
DEF_TEST(FLOAT, GPU, 1, 8, 4095, 0, true)
DEF_TEST(FLOAT, GPU, 1, 8, 1045991, 0, false)

DEF_TEST(BOOL, GPU, 1, 4, 1, 0, false)
DEF_TEST(BOOL, GPU, 1, 4, 1001, 0, false)

DEF_TEST(DOUBLE, GPU, 1, 8, 1001, 0, true)
DEF_TEST(INT64, GPU, 1, 8, 1001, 0, false)

// Failure cases
DEF_TEST(FLOAT, GPU, 1, 8, 128, 6, true)
#endif

}  // namespace
}  // namespace tensorflow
