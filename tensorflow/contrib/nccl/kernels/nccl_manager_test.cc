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

class NcclManagerTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    setenv("NCCL_DEBUG", "INFO", 1 /* replace */);
    devices = new std::vector<BaseGPUDevice*>(GetGPUDevices());
    CHECK(!devices->empty());
    LOG(ERROR) << "Running test with " << devices->size() << " gpus";
  }
  static void TearDownTestCase() {
    for (auto device : *devices) delete device;
    delete devices;
  }

  static Allocator* gpu_allocator(BaseGPUDevice* device) {
    return device->GetStepAllocator(AllocatorAttributes(),
                                    nullptr /* step_resource_manager */);
  }

  static std::vector<BaseGPUDevice*>* devices;

  template <typename Scalar>
  perftools::gputools::DeviceMemory<Scalar> AsDeviceMemory(
      const Scalar* cuda_memory) {
    perftools::gputools::DeviceMemoryBase wrapped(
        const_cast<Scalar*>(cuda_memory));
    perftools::gputools::DeviceMemory<Scalar> typed(wrapped);
    return typed;
  }

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

  TestCase* MakeTestCase(int num_ranks, ncclRedOp_t reduction_op,
                         TensorShape shape, float value_offset) {
    TestCase* test_case = new TestCase();
    test_case->expected = Tensor(DT_FLOAT, shape);
    if (reduction_op == ncclProd) {
      test::FillFn<float>(&test_case->expected, [](int) { return 1; });
    } else if (reduction_op == ncclSum) {
      test::FillFn<float>(&test_case->expected, [](int) { return 0; });
    } else if (reduction_op == ncclMax) {
      test::FillFn<float>(&test_case->expected, [](int) {
        return -1 * std::numeric_limits<float>::max();
      });
    } else if (reduction_op == ncclMin) {
      test::FillFn<float>(&test_case->expected, [](int) {
        return std::numeric_limits<float>::max();
      });
    } else {
      LOG(FATAL) << "Invalid reduction_op " << reduction_op;
    }

    int mult = 1;
    for (int i = 0; i < num_ranks; ++i) {
      auto* device = devices->at(i % devices->size());
      auto* stream = device->tensorflow_gpu_device_info()->stream;

      Tensor in_cpu(DT_FLOAT, shape);
      test::FillFn<float>(&in_cpu, [mult, value_offset](int index) {
        return value_offset + (index + 1) * mult;
      });
      for (int j = 0; j < shape.num_elements(); ++j) {
        auto in_val = in_cpu.flat<float>()(j);
        auto out_expr = test_case->expected.flat<float>();
        if (reduction_op == ncclProd) {
          out_expr(j) *= in_val;
        } else if (reduction_op == ncclSum) {
          out_expr(j) += in_val;
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

      mult *= 10;
      test_case->ins.emplace_back(gpu_allocator(device), DT_FLOAT, shape);
      test_case->outs.emplace_back(gpu_allocator(device), DT_FLOAT, shape);

      const Tensor& in_gpu = test_case->ins.back();
      auto in_gpu_mem = AsDeviceMemory(in_gpu.flat<float>().data());
      stream->ThenMemcpy(&in_gpu_mem, in_cpu.flat<float>().data(),
                         in_cpu.TotalBytes());
    }
    return test_case;
  }

  NcclManager::DoneCallback CreateDoneCallback(TestCase* test_case) {
    return [this, test_case](Status s) {
      mutex_lock l(test_case->mu);
      ++test_case->num_completed;
      test_case->final_status.Update(s);
    };
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
    for (int i = 0; i < test_case->outs.size(); ++i) {
      auto* device = devices->at(i % devices->size());
      auto* stream = device->tensorflow_gpu_device_info()->stream;
      const Tensor& out_gpu = test_case->outs[i];
      Tensor out_cpu(DT_FLOAT, out_gpu.shape());
      auto out_gpu_mem = AsDeviceMemory(out_gpu.flat<float>().data());
      stream->ThenMemcpy(out_cpu.flat<float>().data(), out_gpu_mem,
                         out_cpu.TotalBytes());
      stream->BlockHostUntilDone();
      test::ExpectTensorEqual<float>(test_case->expected, out_cpu);
    }
  }
};
std::vector<BaseGPUDevice*>* NcclManagerTest::devices = nullptr;

// Test basic sum reduction.
TEST_F(NcclManagerTest, BasicSumReduction) {
  const int num_ranks = 3;

  for (int op = 0; op < 4; ++op) {
    ncclRedOp_t reduction_op = static_cast<ncclRedOp_t>(op);
    std::unique_ptr<TestCase> test_case(
        MakeTestCase(num_ranks, reduction_op, TensorShape({2, 3}), 0));
    for (int device_num = 0; device_num < num_ranks; ++device_num) {
      auto* device = devices->at(device_num % devices->size());
      auto* event_mgr = device->tensorflow_gpu_device_info()->event_mgr;
      auto* stream = device->tensorflow_gpu_device_info()->stream;
      NcclManager::instance()->AddToAllReduce(
          num_ranks, "allreduce", reduction_op, device->executor(),
          device->gpu_id(), event_mgr, stream, &test_case->ins[device_num],
          &test_case->outs[device_num], CreateDoneCallback(test_case.get()));
    }

    LOG(ERROR) << "Verifying results";
    VerifyResults("test_case", test_case.get());
  }
}

// Same as the Basic test, but with multiple threads launching parts of many
// reductions.
//
// Testing the multi-rank execution is currently reduced as it can hang when run
// with num_ranks > devices->size(), for some GPUs (e.g. K20m).
// To test the higher settings, increase num_ranks,
// num_collectives_per_iteration and time_limit_micros.
TEST_F(NcclManagerTest, MultipleCallers) {
  const int num_ranks = 1;                      // 2;
  const int num_collectives_per_iteration = 1;  // 1000;
  const int num_threads = 3;
  const int time_limit_micros = 1;  // 60 * 30 * 1000 * 1000;

  int64 start = Env::Default()->NowMicros();
  srand(Env::Default()->NowMicros());

  for (;;) {
    std::vector<std::pair<int, int>> case_and_device_num;
    std::vector<std::unique_ptr<TestCase>> test_cases;
    for (int i = 0; i < num_collectives_per_iteration; ++i) {
      test_cases.emplace_back(
          MakeTestCase(num_ranks, ncclSum,
                       TensorShape({100, i % 5 + 1, i % 3 + 1}), i + 0.1 * i));
      for (int j = 0; j < num_ranks; ++j) {
        case_and_device_num.emplace_back(i, j);
      }
    }

    for (int i = 0; i < num_ranks; ++i) {
      auto* device = devices->at(i % devices->size());
      auto* stream = device->tensorflow_gpu_device_info()->stream;
      stream->BlockHostUntilDone();
    }

    std::random_shuffle(case_and_device_num.begin(), case_and_device_num.end());

    mutex mu;  // guards case_and_device_num.
    std::unique_ptr<thread::ThreadPool> pool(
        new thread::ThreadPool(Env::Default(), "test", num_threads));
    const int to_schedule = case_and_device_num.size();
    for (int i = 0; i < to_schedule; ++i) {
      auto fn = [&]() {
        int device_num;
        int test_num;
        {
          mutex_lock l(mu);
          test_num = case_and_device_num.back().first;
          device_num = case_and_device_num.back().second;
          case_and_device_num.pop_back();
        }
        auto* device = devices->at(device_num % devices->size());
        auto* event_mgr = device->tensorflow_gpu_device_info()->event_mgr;
        auto* stream = device->tensorflow_gpu_device_info()->stream;
        TestCase* test_case = test_cases[test_num].get();
        NcclManager::instance()->AddToAllReduce(
            num_ranks, strings::StrCat("allreduce", test_num), ncclSum,
            device->executor(), device->gpu_id(), event_mgr, stream,
            &test_case->ins[device_num], &test_case->outs[device_num],
            CreateDoneCallback(test_case));
      };
      pool->Schedule(fn);
    }
    pool.reset();  // wait for all work to be scheduled.

    LOG(ERROR) << "Verifying results for " << num_collectives_per_iteration
               << " collectives";
    for (int i = 0; i < test_cases.size(); ++i) {
      VerifyResults(strings::StrCat("collective", i), test_cases[i].get());
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
