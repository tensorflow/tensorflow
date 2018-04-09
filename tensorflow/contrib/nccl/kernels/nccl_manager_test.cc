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

#ifdef GOOGLE_CUDA

#include <algorithm>
#include <random>
#include <vector>

#include "tensorflow/contrib/nccl/kernels/nccl_manager.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/gpu/gpu_device.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

static std::vector<BaseGPUDevice*> GetGPUDevices() {
  std::vector<Device*> devices;
  SessionOptions session_options;
  session_options.config.mutable_gpu_options()
      ->set_per_process_gpu_memory_fraction(0.1);
  session_options.env = Env::Default();
  Status s = DeviceFactory::GetFactory(DEVICE_GPU)
                 ->AddDevices(session_options, "", &devices);
  TF_CHECK_OK(s);
  std::vector<BaseGPUDevice*> gpus;
  for (Device* d : devices) {
    if (d->device_type() == "GPU") {
      gpus.push_back(static_cast<BaseGPUDevice*>(d));
    } else {
      delete d;
    }
  }
  return gpus;
}

template <typename Scalar>
class NcclManagerTest : public ::testing::Test {
 public:
  // A single all-reduce to apply.
  struct TestCase {
    string key;
    std::vector<Tensor> ins;
    std::vector<Tensor> outs;
    Tensor expected;

    mutex mu;
    Status final_status;
    int num_completed = 0;
  };

  static void SetUpTestCase() {
    setenv("NCCL_DEBUG", "INFO", 1 /* replace */);
    devices_ = new std::vector<BaseGPUDevice*>(GetGPUDevices());
    CHECK(!devices_->empty());
    LOG(ERROR) << "Running test with " << devices_->size() << " gpus";
  }

  static void TearDownTestCase() {
    for (auto device : *devices_) delete device;
    delete devices_;
  }

  TestCase* MakeTestCase(int num_ranks, ncclRedOp_t reduction_op,
                         TensorShape shape, float value_offset) {
    TestCase* test_case = new TestCase();
    test_case->expected = Tensor(data_type_, shape);
    if (reduction_op == ncclProd) {
      test::FillFn<Scalar>(&test_case->expected,
                           [](int) { return static_cast<Scalar>(1); });
    } else if (reduction_op == ncclSum) {
      test::FillFn<Scalar>(&test_case->expected,
                           [](int) { return static_cast<Scalar>(0); });
    } else if (reduction_op == ncclMax) {
      test::FillFn<Scalar>(&test_case->expected, [](int) { return -max_; });
    } else if (reduction_op == ncclMin) {
      test::FillFn<Scalar>(&test_case->expected, [](int) { return max_; });
    } else {
      LOG(FATAL) << "Invalid reduction_op " << reduction_op;
    }

    float value_scale = 0.01;  // Small scale to avoid fp16 overflow.
    for (int rank = 0; rank < num_ranks; ++rank) {
      auto* device = GetDevice(rank);
      auto* stream = device->tensorflow_gpu_device_info()->stream;

      Tensor in_cpu(data_type_, shape);
      test::FillFn<Scalar>(&in_cpu, [&](int index) {
        return static_cast<Scalar>((index + 1) * value_scale + value_offset);
      });
      for (int j = 0; j < shape.num_elements(); ++j) {
        auto in_val = in_cpu.flat<Scalar>()(j);
        auto out_expr = test_case->expected.template flat<Scalar>();
        if (reduction_op == ncclProd) {
          out_expr(j) = out_expr(j) * in_val;
        } else if (reduction_op == ncclSum) {
          out_expr(j) = out_expr(j) + in_val;
        } else if (reduction_op == ncclMax) {
          if (in_val > out_expr(j)) {
            out_expr(j) = in_val;
          }
        } else if (reduction_op == ncclMin) {
          if (in_val < out_expr(j)) {
            out_expr(j) = in_val;
          }
        }
      }

      value_scale *= 10;
      test_case->ins.emplace_back(GpuAllocator(device), data_type_, shape);
      test_case->outs.emplace_back(GpuAllocator(device), data_type_, shape);

      const Tensor& in_gpu = test_case->ins.back();
      auto in_gpu_mem = AsDeviceMemory(in_gpu.flat<Scalar>().data());
      stream->ThenMemcpy(&in_gpu_mem, in_cpu.flat<Scalar>().data(),
                         in_cpu.TotalBytes());
    }
    return test_case;
  }

  void VerifyResults(const string& case_label, TestCase* test_case) {
    // Wait for the done callback to be called.
    {
      test_case->mu.lock();
      while (test_case->num_completed != test_case->outs.size()) {
        test_case->mu.unlock();
        Env::Default()->SleepForMicroseconds(10);
        test_case->mu.lock();
      }
      test_case->mu.unlock();
    }
    // Copy memory to host and verify.
    for (int rank = 0; rank < test_case->outs.size(); ++rank) {
      auto* device = GetDevice(rank);
      auto* stream = device->tensorflow_gpu_device_info()->stream;
      const Tensor& out_gpu = test_case->outs[rank];
      Tensor out_cpu(data_type_, out_gpu.shape());
      auto out_gpu_mem = AsDeviceMemory(out_gpu.flat<Scalar>().data());
      stream->ThenMemcpy(out_cpu.flat<Scalar>().data(), out_gpu_mem,
                         out_cpu.TotalBytes());
      SE_ASSERT_OK(stream->BlockHostUntilDone());
      test::ExpectTensorNear<Scalar>(test_case->expected, out_cpu, 0.01);
    }
  }

  NcclManager::DoneCallback CreateDoneCallback(TestCase* test_case) {
    return [this, test_case](Status s) {
      mutex_lock l(test_case->mu);
      ++test_case->num_completed;
      test_case->final_status.Update(s);
    };
  }

  static BaseGPUDevice* GetDevice(size_t rank) {
    return devices_->at(rank % devices_->size());
  }

 private:
  static Allocator* GpuAllocator(BaseGPUDevice* device) {
    return device->GetStepAllocator(AllocatorAttributes(),
                                    nullptr /* step_resource_manager */);
  }

  static perftools::gputools::DeviceMemory<Scalar> AsDeviceMemory(
      const Scalar* cuda_memory) {
    perftools::gputools::DeviceMemoryBase wrapped(
        const_cast<Scalar*>(cuda_memory));
    perftools::gputools::DeviceMemory<Scalar> typed(wrapped);
    return typed;
  }

 private:
  static std::vector<BaseGPUDevice*>* devices_;
  static const DataType data_type_;
  static const Scalar max_;
};

template <typename Scalar>
std::vector<BaseGPUDevice*>* NcclManagerTest<Scalar>::devices_ = nullptr;
template <typename Scalar>
const DataType NcclManagerTest<Scalar>::data_type_ =
    DataTypeToEnum<Scalar>::value;
template <typename Scalar>
const Scalar NcclManagerTest<Scalar>::max_ =
    Eigen::NumTraits<Scalar>::highest();

// Instantiate tests for float and half.
using TypeList = ::testing::Types<float, Eigen::half>;
TYPED_TEST_CASE(NcclManagerTest, TypeList);

// Test basic sum reduction.
TYPED_TEST(NcclManagerTest, BasicSumReduction) {
  const int num_ranks = 3;

  for (int op = 0; op < 4; ++op) {
    ncclRedOp_t reduction_op = static_cast<ncclRedOp_t>(op);
    std::unique_ptr<typename TestFixture::TestCase> test_case(
        this->MakeTestCase(num_ranks, reduction_op, TensorShape({2, 3}), 0.0f));
    for (int rank = 0; rank < num_ranks; ++rank) {
      auto* device = this->GetDevice(rank);
      auto* event_mgr = device->tensorflow_gpu_device_info()->event_mgr;
      auto* stream = device->tensorflow_gpu_device_info()->stream;
      NcclManager::instance()->AddToAllReduce(
          num_ranks, "allreduce", reduction_op, device->executor(),
          device->gpu_id(), event_mgr, stream, &test_case->ins[rank],
          &test_case->outs[rank], this->CreateDoneCallback(test_case.get()));
    }

    LOG(ERROR) << "Verifying results";
    this->VerifyResults("test_case", test_case.get());
  }
}

// Same as the Basic test, but with multiple threads launching parts of many
// reductions.
//
// Testing the multi-rank execution is currently reduced as it can hang when run
// with num_ranks > devices->size(), for some GPUs (e.g. K20m).
// To test the higher settings, increase num_ranks,
// num_collectives_per_iteration and time_limit_micros.
TYPED_TEST(NcclManagerTest, MultipleCallers) {
  const int num_ranks = 1;                      // 2;
  const int num_collectives_per_iteration = 1;  // 1000;
  const int num_threads = 3;
  const int time_limit_micros = 1;  // 60 * 30 * 1000 * 1000;

  int64 start = Env::Default()->NowMicros();
  srand(Env::Default()->NowMicros());

  for (;;) {
    std::vector<std::pair<int, int>> case_and_rank;
    std::vector<std::unique_ptr<typename TestFixture::TestCase>> test_cases;
    for (int i = 0; i < num_collectives_per_iteration; ++i) {
      test_cases.emplace_back(this->MakeTestCase(
          num_ranks, ncclSum, TensorShape({100, i % 5 + 1, i % 3 + 1}),
          1.1f * i));
      for (int j = 0; j < num_ranks; ++j) {
        case_and_rank.emplace_back(i, j);
      }
    }

    for (int rank = 0; rank < num_ranks; ++rank) {
      auto* device = this->GetDevice(rank);
      auto* stream = device->tensorflow_gpu_device_info()->stream;
      SE_ASSERT_OK(stream->BlockHostUntilDone());
    }

    std::shuffle(case_and_rank.begin(), case_and_rank.end(),
                 std::mt19937(std::random_device()()));

    mutex mu;  // guards case_and_rank.
    std::unique_ptr<thread::ThreadPool> pool(
        new thread::ThreadPool(Env::Default(), "test", num_threads));
    const int to_schedule = case_and_rank.size();
    for (int i = 0; i < to_schedule; ++i) {
      auto fn = [&]() {
        int rank;
        int test_num;
        {
          mutex_lock l(mu);
          test_num = case_and_rank.back().first;
          rank = case_and_rank.back().second;
          case_and_rank.pop_back();
        }
        auto* device = this->GetDevice(rank);
        auto* event_mgr = device->tensorflow_gpu_device_info()->event_mgr;
        auto* stream = device->tensorflow_gpu_device_info()->stream;
        typename TestFixture::TestCase* test_case = test_cases[test_num].get();
        NcclManager::instance()->AddToAllReduce(
            num_ranks, strings::StrCat("allreduce", test_num), ncclSum,
            device->executor(), device->gpu_id(), event_mgr, stream,
            &test_case->ins[rank], &test_case->outs[rank],
            this->CreateDoneCallback(test_case));
      };
      pool->Schedule(fn);
    }
    pool.reset();  // wait for all work to be scheduled.

    LOG(ERROR) << "Verifying results for " << num_collectives_per_iteration
               << " collectives";
    for (int i = 0; i < test_cases.size(); ++i) {
      this->VerifyResults(strings::StrCat("collective", i),
                          test_cases[i].get());
    }

    int64 delta = Env::Default()->NowMicros() - start;
    if (delta > time_limit_micros) {
      LOG(ERROR) << "Ran for " << delta << " quitting";
      break;
    }
  }
}

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
