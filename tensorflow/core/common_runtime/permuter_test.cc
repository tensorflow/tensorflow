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
#include "tensorflow/core/common_runtime/permuter.h"

#include <algorithm>

#include "absl/memory/memory.h"
#include "absl/types/span.h"
#include "tensorflow/core/common_runtime/base_collective_executor.h"
#include "tensorflow/core/common_runtime/collective_rma_local.h"
#include "tensorflow/core/common_runtime/collective_test_util.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/device_resolver_local.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/common_runtime/test_collective_executor_mgr.h"
#include "tensorflow/core/common_runtime/threadpool_device.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/unbounded_work_queue.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace {

class PermuterTest : public ::testing::Test {
 protected:
  void Init(int num_workers, int num_devices,
            const std::vector<int>& permutation, DataType dtype,
            const TensorShape& shape, const DeviceType& device_type,
            int fail_after) {
    test_env_ = CreateCollectiveTestEnv(num_workers, num_devices, device_type);
    test_env_->remote_access->set_fail_after(fail_after);
    for (int wi = 0; wi < num_workers; ++wi) {
      for (int di = 0; di < num_devices; ++di) {
        int rank = wi * num_devices + di;
        instances_.push_back(absl::make_unique<DeviceInstance>(
            rank, permutation, dtype, shape, test_env_.get()));
      }
    }
  }

  typedef std::function<void(Tensor*)> InitFunc;

  void Permute(int fail_after) {
    std::atomic<int> done(0);
    for (auto& di : instances_) {
      SchedClosure([&di, &done] {
        di->DoPermute();
        ++done;
      });
      if (fail_after > 0) {
        // Stagger the op execution starts.
        Env::Default()->SleepForMicroseconds(100);
      }
    }
    while (done < instances_.size()) {
      Env::Default()->SleepForMicroseconds(1000);
    }
  }

  template <typename T>
  void RunTest(DataType dtype, const DeviceType& device_type, int num_workers,
               int num_devices, int tensor_len, int fail_after) {
    std::vector<int> permutation(num_workers * num_devices);
    std::iota(permutation.begin(), permutation.end(), 0);
    // Generate a permutation by permuting every two instances.
    // E.g. [0,1] becomes [1,0]
    //      [0,1,2,3] becomes [1,0,3,2]
    for (int i = 0; i < permutation.size(); i += 2) {
      // If the total number of instances is odd,
      // swap the last instance with the first.
      // E.g. [0,1,2] becomes [2,0,1]
      if (permutation.size() == i + 1) {
        std::swap(permutation[i], permutation[0]);
        continue;
      }
      std::next_permutation(permutation.begin() + i,
                            permutation.begin() + i + 2);
    }
    Init(num_workers, num_devices, permutation, dtype,
         TensorShape({tensor_len}), device_type, fail_after);
    gtl::InlinedVector<T, 4> expected(tensor_len * num_devices * num_workers,
                                      0.0);
    // Initialize each instance tensor with distinct values.
    for (int di = 0; di < instances_.size(); ++di) {
      instances_[di]->InitTensor(
          [&permutation, &expected, di, tensor_len](Tensor* t) {
            for (size_t i = 0; i < t->NumElements(); ++i) {
              // The cast is necessary to prevent clang-tidy from insisting
              // that a faster non-open source function be substituted.
              float value = pow(10, static_cast<double>(di)) * i;
              t->flat<T>()(i) = value;
              expected[permutation[di] * tensor_len + i] = value;
            }
          });
    }

    Permute(fail_after);

    // At this point all of the ops have terminated.
    for (int di = 0; di < instances_.size(); ++di) {
      if (!instances_[di]->status_.ok()) {
        ASSERT_GT(fail_after, 0);
        ASSERT_NE(
            instances_[di]->status_.error_message().find("Deliberate failure"),
            string::npos);
        continue;
      }
      TF_EXPECT_OK(instances_[di]->status_);
      test::ExpectTensorEqual<T>(
          test::AsTensor<T>(
              absl::MakeSpan(expected).subspan(di * tensor_len, tensor_len)),
          instances_[di]->output_tensor_);
    }
  }

  class DeviceInstance {
   public:
    DeviceInstance(int rank, std::vector<int> permutation, DataType dtype,
                   const TensorShape& shape, CollectiveTestEnv* test_env)
        : test_env_(test_env),
          input_tensor_(dtype, shape),
          output_tensor_(dtype, shape) {
      col_params_ = CreateCollectiveParams(*test_env_, rank, "Permute",
                                           PERMUTE_COLLECTIVE, dtype, shape);
      col_params_->instance.permutation = std::move(permutation);
      col_params_->instance.devices = col_params_->group.device_names;
      string dev_name = col_params_->group.device_names[rank];
      TF_CHECK_OK(test_env_->device_mgr->LookupDevice(dev_name, &device_))
          << "Couldn't find device " << dev_name
          << " existing devices: " << test_env_->device_mgr->DebugString();
    }

    void InitTensor(const InitFunc& f) { f(&input_tensor_); }

    void DoPermute() {
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
};

// TODO(b/113171733): change to use TEST_P.
// Tests of full permute algorithm, with different device and
// data types.
// B = data element type
// T = device type
// W = number of workers
// D = number of devices per worker
// L = tensor length
// A = abort after count
#define DEF_TEST(B, T, W, D, L, A)                                  \
  TEST_F(PermuterTest,                                              \
         DaTy##B##_DevTy##T##_Wkr##W##_Dev##D##_Len##L##_Abrt##A) { \
    DataType dtype = DT_##B;                                        \
    switch (dtype) {                                                \
      case DT_BOOL: {                                               \
        RunTest<bool>(dtype, DEVICE_##T, W, D, L, A);               \
      } break;                                                      \
      case DT_FLOAT: {                                              \
        RunTest<float>(dtype, DEVICE_##T, W, D, L, A);              \
      } break;                                                      \
      case DT_DOUBLE: {                                             \
        RunTest<double>(dtype, DEVICE_##T, W, D, L, A);             \
      } break;                                                      \
      case DT_INT32: {                                              \
        RunTest<int32>(dtype, DEVICE_##T, W, D, L, A);              \
      } break;                                                      \
      case DT_INT64: {                                              \
        RunTest<int64>(dtype, DEVICE_##T, W, D, L, A);              \
      } break;                                                      \
      default:                                                      \
        LOG(FATAL) << "Unimplemented";                              \
    }                                                               \
  }

#if !(GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
//       B      T    W  D  L  A
DEF_TEST(FLOAT, CPU, 1, 2, 1, 0)
DEF_TEST(FLOAT, CPU, 1, 3, 3, 0)
DEF_TEST(FLOAT, CPU, 1, 7, 3, 0)
DEF_TEST(FLOAT, CPU, 1, 2, 1001, 0)
DEF_TEST(FLOAT, CPU, 2, 2, 3, 0)
DEF_TEST(FLOAT, CPU, 2, 1, 128, 0)
DEF_TEST(FLOAT, CPU, 2, 4, 128, 0)
DEF_TEST(FLOAT, CPU, 2, 8, 4095, 0)
DEF_TEST(FLOAT, CPU, 4, 4, 1045991, 0)

DEF_TEST(BOOL, CPU, 1, 4, 1, 0)
DEF_TEST(BOOL, CPU, 2, 4, 1, 0)
DEF_TEST(BOOL, CPU, 2, 4, 1001, 0)

DEF_TEST(DOUBLE, CPU, 2, 4, 128, 0)
DEF_TEST(INT32, CPU, 2, 4, 128, 0)
DEF_TEST(INT64, CPU, 2, 4, 128, 0)

// Failure cases
DEF_TEST(FLOAT, CPU, 1, 2, 1, 1)
DEF_TEST(FLOAT, CPU, 2, 4, 128, 1)
DEF_TEST(FLOAT, CPU, 2, 4, 128, 5)
#endif

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Can only set W=1 for GPU tests.
//       B      T    W  D  L  A
DEF_TEST(FLOAT, GPU, 1, 2, 1, 0)
DEF_TEST(FLOAT, GPU, 1, 7, 3, 0)
DEF_TEST(FLOAT, GPU, 1, 2, 33, 0)
DEF_TEST(FLOAT, GPU, 1, 3, 64, 0)
DEF_TEST(FLOAT, GPU, 1, 8, 1001, 0)
DEF_TEST(FLOAT, GPU, 1, 8, 4095, 0)
DEF_TEST(FLOAT, GPU, 1, 8, 1045991, 0)

DEF_TEST(BOOL, GPU, 1, 4, 1, 0)
DEF_TEST(BOOL, GPU, 1, 4, 1001, 0)

DEF_TEST(DOUBLE, GPU, 1, 8, 1001, 0)
DEF_TEST(INT64, GPU, 1, 8, 1001, 0)

// Failure cases
DEF_TEST(FLOAT, GPU, 1, 8, 128, 6)
#endif

}  // namespace
}  // namespace tensorflow
