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
#include "tensorflow/core/common_runtime/ring_reducer.h"

#include <algorithm>

#include "absl/memory/memory.h"
#include "tensorflow/core/common_runtime/base_collective_executor.h"
#include "tensorflow/core/common_runtime/collective_rma_local.h"
#include "tensorflow/core/common_runtime/collective_test_util.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/device_resolver_local.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/common_runtime/test_collective_executor_mgr.h"
#include "tensorflow/core/common_runtime/threadpool_device.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/unbounded_work_queue.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

std::unique_ptr<OpKernel> GetKernel(const NodeDef& node,
                                    const DeviceType& device_type,
                                    DeviceBase* device) {
  Status status;
  std::unique_ptr<OpKernel> k = CreateOpKernel(
      device_type, device, device->GetAllocator(AllocatorAttributes()), node,
      TF_GRAPH_DEF_VERSION, &status);
  if (!status.ok()) {
    LOG(FATAL) << status;
  }
  return k;
}

std::unique_ptr<OpKernel> GetAdd(DataType dtype, const DeviceType& device_type,
                                 DeviceBase* device) {
  NodeDef node_def;
  NodeDefBuilder builder("add_node", "Add");
  TF_CHECK_OK(builder.Attr("T", dtype)
                  .Input(FakeInput(dtype))
                  .Input(FakeInput(dtype))
                  .Finalize(&node_def));
  return GetKernel(node_def, device_type, device);
}

std::unique_ptr<OpKernel> GetDiv(DataType dtype, const DeviceType& device_type,
                                 DeviceBase* device) {
  NodeDef node_def;
  NodeDefBuilder builder("add_node", "Div");
  TF_CHECK_OK(builder.Attr("T", dtype)
                  .Input(FakeInput(dtype))
                  .Input(FakeInput(dtype))
                  .Finalize(&node_def));
  return GetKernel(node_def, device_type, device);
}

class RingReducerTest : public ::testing::Test {
 protected:
  void Init(int num_workers, int num_devices, DataType dtype,
            const TensorShape& shape, const DeviceType& device_type,
            int num_subdivs, int fail_after) {
    test_env_ = CreateCollectiveTestEnv(num_workers, num_devices, device_type);
    test_env_->remote_access->set_fail_after(fail_after);
    for (int wi = 0; wi < num_workers; ++wi) {
      for (int di = 0; di < num_devices; ++di) {
        int rank = wi * num_devices + di;
        instances_.push_back(std::make_unique<DeviceInstance>(
            rank, num_subdivs, dtype, shape, test_env_.get()));
      }
    }
  }

  void Reduce(int fail_after) {
    std::atomic<int> done(0);
    for (auto& di : instances_) {
      SchedClosure([&di, &done] {
        di->DoReduce();
        ++done;
      });
      if (fail_after > 0) {
        // Stagger the op execution starts.
        Env::Default()->SleepForMicroseconds(100);
      }
    }
    while (done < static_cast<int>(instances_.size())) {
      Env::Default()->SleepForMicroseconds(1000);
    }
  }

  template <typename T>
  void RunTest(DataType dtype, const DeviceType& device_type, int num_workers,
               int num_devices, int num_subdivs, int tensor_len,
               int fail_after) {
    Init(num_workers, num_devices, dtype, TensorShape({tensor_len}),
         device_type, num_subdivs, fail_after);
    std::vector<T> expected(tensor_len);
    for (int di = 0; di < static_cast<int>(instances_.size()); ++di) {
      instances_[di]->InitTensor([&expected, dtype, di](Tensor* t) {
        for (size_t i = 0; i < t->NumElements(); ++i) {
          // The cast is necessary to prevent clang-tidy from insisting
          // that a faster non-open source function be substituted.
          float value = pow(10, static_cast<double>(di)) * i;
          if (dtype == DT_INT32 || dtype == DT_INT64) {
            value = di * 10 + i;
          }
          t->flat<T>()(i) = static_cast<T>(value);
          expected[i] += static_cast<T>(value);
        }
      });
    }
    Reduce(fail_after);
    if (fail_after > 0) {
      // Confirm that every device terminated with the expected error status.
      for (int di = 0; di < static_cast<int>(instances_.size()); ++di) {
        EXPECT_NE(instances_[di]->status_.message().find("Deliberate failure"),
                  string::npos);
      }
    } else {
      // Confirm that every device computed the same correct reduction value.
      for (int i = 0; i < tensor_len; ++i) {
        expected[i] /= static_cast<T>(num_workers * num_devices);
      }
      for (int di = 0; di < static_cast<int>(instances_.size()); ++di) {
        TF_EXPECT_OK(instances_[di]->status_);
        test::ExpectTensorEqual<T>(test::AsTensor<T>(expected),
                                   instances_[di]->tensor());
      }
    }
  }

  class DeviceInstance {
   public:
    DeviceInstance(int rank, int num_subdivs, DataType dtype,
                   const TensorShape& shape, CollectiveTestEnv* test_env)
        : test_env_(test_env), tensor_(dtype, shape) {
      col_params_ = CreateCollectiveParams(*test_env_, rank, "RingReduce",
                                           REDUCTION_COLLECTIVE, dtype, shape);
      if (num_subdivs > 0) {
        col_params_->instance.impl_details.subdiv_offsets =
            GenerateEvenSubdivOffsets(test_env->num_devices_per_worker,
                                      num_subdivs);
      }
      string dev_name = col_params_->group.members[rank].device.name();
      TF_CHECK_OK(test_env_->device_mgr->LookupDevice(dev_name, &device_))
          << "Couldn't find device " << dev_name
          << " existing devices: " << test_env_->device_mgr->DebugString();
      merge_op_ = GetAdd(col_params_->instance.data_type,
                         test_env_->device_type, device_);
      final_op_ = GetDiv(col_params_->instance.data_type,
                         test_env_->device_type, device_);
      col_params_->merge_op = merge_op_.get();
      col_params_->final_op = final_op_.get();
    }

    void InitTensor(const std::function<void(Tensor*)>& init_f) {
      init_f(&tensor_);
    }

    void DoReduce() {
      status_ = RunCollective(test_env_, col_params_.get(), device_, &tensor_,
                              &tensor_);
    }

    const Tensor& tensor() { return tensor_; }

    CollectiveTestEnv* test_env_;
    Tensor tensor_;
    Device* device_;
    core::RefCountPtr<CollectiveParams> col_params_;
    std::unique_ptr<OpKernel> merge_op_;
    std::unique_ptr<OpKernel> final_op_;
    Status status_;
  };

  std::unique_ptr<CollectiveTestEnv> test_env_;
  std::vector<std::unique_ptr<DeviceInstance>> instances_;
  mutex mu_;
  int32 reduce_counter_ TF_GUARDED_BY(mu_) = 0;
};

class RingReducerInitParamsTest : public ::testing::Test {
 protected:
  void RunSubdivPermsTest(
      CollectiveParams* cp,
      const std::vector<std::vector<int>>& expected_subdiv_perms,
      const std::vector<int>& expected_subdiv_rank) {
    cp->instance.impl_details.subdiv_permutations.clear();
    cp->subdiv_rank.clear();
    // Create a stub ring reducer only for testing param initialization.
    core::RefCountPtr<RingReducer> reducer(new RingReducer());
    TF_CHECK_OK(reducer->InitializeCollectiveParams(cp));
    EXPECT_EQ(expected_subdiv_perms,
              cp->instance.impl_details.subdiv_permutations);
    EXPECT_EQ(expected_subdiv_rank, cp->subdiv_rank);
    reducer->group_size_tensor_ready_.Notify();  // To unblock destructor.
  }
};

TEST_F(RingReducerInitParamsTest, SpecifiedSubdivs) {
  const int kNumDevsPerWorker = 8;
  const int kNumWorkers = 3;
  auto test_env =
      CreateCollectiveTestEnv(kNumWorkers, kNumDevsPerWorker, DEVICE_CPU);
  auto cp =
      CreateCollectiveParams(*test_env, /*rank*/ 0, "RingReduce",
                             REDUCTION_COLLECTIVE, DT_FLOAT, TensorShape({1}));

  cp->default_rank = 0;
  cp->instance.impl_details.subdiv_offsets = {0, 4};
  RunSubdivPermsTest(cp.get(),
                     {{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                       12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
                      {4, 5, 6,  7,  0,  1,  2,  3,  12, 13, 14, 15,
                       8, 9, 10, 11, 20, 21, 22, 23, 16, 17, 18, 19}},
                     {0, 4});

  cp->instance.impl_details.subdiv_offsets = {0, -4};
  RunSubdivPermsTest(cp.get(),
                     {{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                       12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
                      {3,  2,  1,  0,  7,  6,  5,  4,  11, 10, 9,  8,
                       15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20}},
                     {0, 3});

  cp->default_rank = 3;
  cp->instance.impl_details.subdiv_offsets = {3, -3};
  RunSubdivPermsTest(cp.get(),
                     {{3,  4, 5, 6,  7,  0,  1,  2,  11, 12, 13, 14,
                       15, 8, 9, 10, 19, 20, 21, 22, 23, 16, 17, 18},
                      {4, 3,  2,  1,  0,  7,  6,  5,  12, 11, 10, 9,
                       8, 15, 14, 13, 20, 19, 18, 17, 16, 23, 22, 21}},
                     {0, 1});
}

TEST_F(RingReducerInitParamsTest, AutomaticSubdivs) {
  const int kNumDevsPerWorker = 8;
  const int kNumWorkers = 3;
  const int kNumDevs = kNumDevsPerWorker * kNumWorkers;
  auto test_env =
      CreateCollectiveTestEnv(kNumWorkers, kNumDevsPerWorker, DEVICE_CPU);
  auto cp =
      CreateCollectiveParams(*test_env, /*rank*/ 0, "RingReduce",
                             REDUCTION_COLLECTIVE, DT_FLOAT, TensorShape({1}));

  // Test automatic generation of subdiv offsets.
  cp->default_rank = 0;
  cp->instance.impl_details.subdiv_offsets.clear();
  cp->instance.impl_details.max_subdivs_per_device = 0;
  RunSubdivPermsTest(cp.get(),
                     {{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                       12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}},
                     {0});

  // Set shape so that with 2 subdivs chunk_size is 3 MiB.  This should cause 2
  // offsets, {0, -4}, to be generated.
  {
    int num_subdivs = 2;
    int num_chunks = kNumDevs * num_subdivs;
    size_t chunk_size = 3 * 1048576;  // 3 MB
    size_t tensor_size = chunk_size * num_chunks;
    cp->instance.shape = TensorShape(
        {static_cast<int64_t>(tensor_size / DataTypeSize(DT_FLOAT))});
  }
  cp->instance.impl_details.subdiv_offsets.clear();
  RunSubdivPermsTest(cp.get(),
                     {{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                       12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
                      {3,  2,  1,  0,  7,  6,  5,  4,  11, 10, 9,  8,
                       15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20}},
                     {0, 3});
}

TEST_F(RingReducerInitParamsTest, AutomaticSubdivUpperBound) {
  const int kNumDevsPerWorker = 1;
  const int kNumWorkers = 4;
  auto test_env =
      CreateCollectiveTestEnv(kNumWorkers, kNumDevsPerWorker, DEVICE_CPU);
  auto cp =
      CreateCollectiveParams(*test_env, /*rank*/ 0, "RingReduce",
                             REDUCTION_COLLECTIVE, DT_FLOAT, TensorShape({1}));

  cp->default_rank = 0;
  cp->instance.impl_details.subdiv_offsets.clear();
  cp->instance.impl_details.max_subdivs_per_device = 0;
  cp->instance.shape = TensorShape({104857600 / DataTypeSize(DT_FLOAT)});
  RunSubdivPermsTest(cp.get(), {{0, 1, 2, 3}, {0, 1, 2, 3}}, {0, 0});
}

TEST_F(RingReducerInitParamsTest, AutomaticSubdivIgnoresMaxNumSubdivs) {
  const int kNumDevsPerWorker = 1;
  const int kNumWorkers = 4;
  auto test_env =
      CreateCollectiveTestEnv(kNumWorkers, kNumDevsPerWorker, DEVICE_CPU);
  auto cp =
      CreateCollectiveParams(*test_env, /*rank*/ 0, "RingReduce",
                             REDUCTION_COLLECTIVE, DT_FLOAT, TensorShape({1}));

  cp->default_rank = 0;
  // When subdiv_offsets is present it will override automatic generation of
  // offsets even when max_subdivs_per_device is present.
  // cp->instance.impl_details.subdiv_offsets.clear();
  cp->instance.impl_details.max_subdivs_per_device = 4;
  cp->instance.shape = TensorShape({104857600 / DataTypeSize(DT_FLOAT)});
  RunSubdivPermsTest(cp.get(), {{0, 1, 2, 3}}, {0});

  cp->default_rank = 0;
  // subdiv_offsets cleared, max_subdivs_per_device = 4 takes effect.
  cp->instance.impl_details.subdiv_offsets.clear();
  cp->instance.impl_details.max_subdivs_per_device = 4;
  cp->instance.shape = TensorShape({104857600 / DataTypeSize(DT_FLOAT)});
  RunSubdivPermsTest(cp.get(),
                     {{0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}},
                     {0, 0, 0, 0});
}

TEST_F(RingReducerInitParamsTest, AutomaticSubdivUsesDefault) {
  const int kNumDevsPerWorker = 1;
  const int kNumWorkers = 4;
  auto test_env =
      CreateCollectiveTestEnv(kNumWorkers, kNumDevsPerWorker, DEVICE_CPU);
  auto cp =
      CreateCollectiveParams(*test_env, /*rank*/ 0, "RingReduce",
                             REDUCTION_COLLECTIVE, DT_FLOAT, TensorShape({1}));

  cp->default_rank = 0;
  // When subdiv_offsets is NOT present and max_subdivs_per_device has a
  // == 0 value, the default setting of 2 is used.
  cp->instance.impl_details.subdiv_offsets.clear();
  cp->instance.impl_details.max_subdivs_per_device = 0;
  cp->instance.shape = TensorShape({104857600 / DataTypeSize(DT_FLOAT)});
  RunSubdivPermsTest(cp.get(), {{0, 1, 2, 3}, {0, 1, 2, 3}}, {0, 0});
}

TEST_F(RingReducerInitParamsTest, AutomaticSubdivDisabled) {
  const int kNumDevsPerWorker = 1;
  const int kNumWorkers = 4;
  auto test_env =
      CreateCollectiveTestEnv(kNumWorkers, kNumDevsPerWorker, DEVICE_CPU);
  auto cp =
      CreateCollectiveParams(*test_env, /*rank*/ 0, "RingReduce",
                             REDUCTION_COLLECTIVE, DT_FLOAT, TensorShape({1}));

  cp->default_rank = 0;
  // When subdiv_offsets is NOT present and max_subdivs_per_device = -1 no
  // subidivision should be done. (old behavior)
  cp->instance.impl_details.subdiv_offsets.clear();
  cp->instance.impl_details.max_subdivs_per_device = -1;
  cp->instance.shape = TensorShape({104857600 / DataTypeSize(DT_FLOAT)});
  RunSubdivPermsTest(cp.get(), {{0, 1, 2, 3}}, {0});
}

// TODO(b/113171733): change to use TEST_P.
#define DEF_TEST(B, T, W, D, S, L, A)                                         \
  TEST_F(RingReducerTest,                                                     \
         DaTy##B##_DevTy##T##_Wkr##W##_Dev##D##_Sdiv##S##_Len##L##_Abrt##A) { \
    DataType dtype = DT_##B;                                                  \
    switch (dtype) {                                                          \
      case DT_FLOAT: {                                                        \
        RunTest<float>(dtype, DEVICE_##T, W, D, S, L, A);                     \
      } break;                                                                \
      case DT_DOUBLE: {                                                       \
        RunTest<double>(dtype, DEVICE_##T, W, D, S, L, A);                    \
      } break;                                                                \
      case DT_BFLOAT16: {                                                     \
        RunTest<tensorflow::bfloat16>(dtype, DEVICE_##T, W, D, S, L, A);      \
      } break;                                                                \
      case DT_INT32: {                                                        \
        RunTest<int32>(dtype, DEVICE_##T, W, D, S, L, A);                     \
      } break;                                                                \
      case DT_INT64: {                                                        \
        RunTest<int64_t>(dtype, DEVICE_##T, W, D, S, L, A);                   \
      } break;                                                                \
      default:                                                                \
        LOG(FATAL) << "Unimplemented";                                        \
    }                                                                         \
  }

#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
// Success tests
DEF_TEST(FLOAT, CPU, 1, 2, 1, 1, 0)
DEF_TEST(FLOAT, CPU, 1, 2, 1, 2, 0)
DEF_TEST(FLOAT, CPU, 1, 2, 1, 8, 0)
DEF_TEST(FLOAT, CPU, 1, 2, 1, 16, 0)
DEF_TEST(FLOAT, CPU, 1, 2, 1, 1001, 0)
DEF_TEST(FLOAT, CPU, 2, 4, 1, 128, 0)
DEF_TEST(FLOAT, CPU, 2, 8, 1, 1001, 0)
DEF_TEST(FLOAT, CPU, 2, 8, 1, 4096, 0)
DEF_TEST(FLOAT, CPU, 2, 8, 1, 9408, 0)
DEF_TEST(FLOAT, CPU, 2, 8, 3, 4095, 0)
DEF_TEST(FLOAT, CPU, 2, 8, 3, 1045991, 0)
DEF_TEST(FLOAT, CPU, 4, 4, 4, 1045991, 0)
DEF_TEST(DOUBLE, CPU, 1, 2, 1, 1001, 0)
DEF_TEST(DOUBLE, CPU, 2, 8, 3, 4095, 0)
DEF_TEST(BFLOAT16, CPU, 1, 2, 1, 8, 0)
DEF_TEST(BFLOAT16, CPU, 2, 8, 3, 16, 0)
DEF_TEST(INT32, CPU, 1, 2, 1, 1001, 0)
DEF_TEST(INT32, CPU, 2, 8, 3, 4095, 0)
DEF_TEST(INT64, CPU, 1, 2, 1, 1001, 0)
DEF_TEST(INT64, CPU, 2, 8, 3, 4095, 0)

// Failure tests
DEF_TEST(FLOAT, CPU, 2, 8, 1, 9408, 1)
DEF_TEST(FLOAT, CPU, 2, 8, 1, 9408, 7)
DEF_TEST(FLOAT, CPU, 2, 8, 2, 9408, 11)
#endif

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// GPU tests.  So long as the device names are all in a single tasks we
// bypass inter-worker routing code and can fake multiple GPUs with a single
// GPU, from the perspective of the RingReducer logic.  So these tests
// are all single-worker.
DEF_TEST(FLOAT, GPU, 1, 2, 1, 1, 0)
DEF_TEST(FLOAT, GPU, 1, 2, 1, 2, 0)
DEF_TEST(FLOAT, GPU, 1, 2, 1, 8, 0)
DEF_TEST(FLOAT, GPU, 1, 2, 1, 16, 0)
DEF_TEST(FLOAT, GPU, 1, 2, 1, 1001, 0)
DEF_TEST(FLOAT, GPU, 1, 8, 1, 1001, 0)
DEF_TEST(FLOAT, GPU, 1, 8, 1, 4096, 0)
DEF_TEST(FLOAT, GPU, 1, 8, 3, 4095, 0)
DEF_TEST(FLOAT, GPU, 1, 8, 3, 1045991, 0)
DEF_TEST(FLOAT, GPU, 1, 4, 4, 1045991, 0)
DEF_TEST(DOUBLE, GPU, 1, 2, 1, 1001, 0)
// INT32 values are never on the GPU.
// DEF_TEST(INT32, GPU, 1, 2, 1, 1001, 0)
DEF_TEST(INT64, GPU, 1, 2, 1, 1001, 0)

// Failure tests
DEF_TEST(FLOAT, GPU, 1, 8, 1, 9408, 2)
DEF_TEST(FLOAT, GPU, 1, 8, 2, 9408, 5)
#endif

}  // namespace tensorflow
