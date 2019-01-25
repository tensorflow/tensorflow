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

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/gpu/gpu_device.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/nccl/nccl_manager.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

static std::vector<std::unique_ptr<BaseGPUDevice>> GetGPUDevices() {
  std::vector<std::unique_ptr<Device>> devices;
  SessionOptions session_options;
  session_options.config.mutable_gpu_options()
      ->set_per_process_gpu_memory_fraction(0.1);
  session_options.env = Env::Default();
  Status s = DeviceFactory::GetFactory(DEVICE_GPU)
                 ->AddDevices(session_options, "", &devices);
  TF_CHECK_OK(s);
  std::vector<std::unique_ptr<BaseGPUDevice>> gpus;
  for (std::unique_ptr<Device>& device : devices) {
    if (device->device_type() == "GPU") {
      // If `device_type()` is GPU, this `Device` is guaranteed to be a
      // `BaseGPUDevice`, which is a subclass of `Device`.
      gpus.emplace_back(static_cast<BaseGPUDevice*>(device.release()));
    }
  }
  return gpus;
}

template <typename Scalar>
class NcclManagerTest : public ::testing::Test {
 public:
  // A single all-reduce to apply.
  struct TestCase {
    std::vector<Tensor> ins;
    std::vector<Tensor> outs;
    Tensor expected;

    mutex mu;
    Status final_status;
    int num_completed = 0;
  };

  static void SetUpTestCase() {
    setenv("NCCL_DEBUG", "INFO", 1 /* replace */);
    setenv("NCCL_LAUNCH_MODE", "PARALLEL", 1 /* replace */);
    devices_ = new std::vector<std::unique_ptr<BaseGPUDevice>>(GetGPUDevices());
    LOG(INFO) << "Running test with " << devices_->size() << " gpus";
  }

  static int32 NumGPUs() { return static_cast<int32>(devices_->size()); }

  static void TearDownTestCase() { delete devices_; }

  TestCase* MakeTestCase(int num_nodes, int num_ranks_per_node,
                         ncclRedOp_t reduction_op, TensorShape shape,
                         float value_offset) {
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
    for (int node = 0; node < num_nodes; ++node) {
      for (int local_rank = 0; local_rank < num_ranks_per_node; ++local_rank) {
        auto* device = GetDevice(local_rank);
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
    }

    return test_case;
  }

  // Waits for the done callback to be called for each participant.
  void WaitForTestCompletion(TestCase* test_case) {
    test_case->mu.lock();
    while (test_case->num_completed != test_case->outs.size()) {
      test_case->mu.unlock();
      Env::Default()->SleepForMicroseconds(10);
      test_case->mu.lock();
    }
    test_case->mu.unlock();
  }

  void VerifyResults(TestCase* test_case) {
    WaitForTestCompletion(test_case);
    TF_ASSERT_OK(test_case->final_status);
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
      test::ExpectClose(test_case->expected, out_cpu);
    }
  }

  void VerifyError(TestCase* test_case) {
    WaitForTestCompletion(test_case);
    LOG(INFO) << test_case->final_status;
    EXPECT_EQ(test_case->final_status.code(), error::INTERNAL);
  }

  NcclManager::DoneCallback CreateDoneCallback(TestCase* test_case) {
    return [this, test_case](Status s) {
      mutex_lock l(test_case->mu);
      ++test_case->num_completed;
      test_case->final_status.Update(s);
    };
  }

  static BaseGPUDevice* GetDevice(size_t rank) {
    return devices_->at(rank % devices_->size()).get();
  }

 private:
  static Allocator* GpuAllocator(BaseGPUDevice* device) {
    return device->GetAllocator(AllocatorAttributes());
  }

  static se::DeviceMemory<Scalar> AsDeviceMemory(const Scalar* cuda_memory) {
    se::DeviceMemoryBase wrapped(const_cast<Scalar*>(cuda_memory));
    se::DeviceMemory<Scalar> typed(wrapped);
    return typed;
  }

 private:
  static std::vector<std::unique_ptr<BaseGPUDevice>>* devices_;
  static const DataType data_type_;
  static const Scalar max_;
};

template <typename Scalar>
std::vector<std::unique_ptr<BaseGPUDevice>>* NcclManagerTest<Scalar>::devices_ =
    nullptr;
template <typename Scalar>
const DataType NcclManagerTest<Scalar>::data_type_ =
    DataTypeToEnum<Scalar>::value;
template <typename Scalar>
const Scalar NcclManagerTest<Scalar>::max_ =
    Eigen::NumTraits<Scalar>::highest();

// Instantiate tests for float and double.
using TypeList = ::testing::Types<float, double>;
TYPED_TEST_SUITE(NcclManagerTest, TypeList);

// Test basic sum reduction.
TYPED_TEST(NcclManagerTest, BasicSumReduction) {
  const int num_ranks = 4;

  for (int op = 0; op < 4; ++op) {
    ncclRedOp_t reduction_op = static_cast<ncclRedOp_t>(op);
    std::unique_ptr<typename TestFixture::TestCase> test_case(
        this->MakeTestCase(/*num_nodes=*/1, num_ranks, reduction_op,
                           TensorShape({2, 3}), 0.0f));
    for (int rank = 0; rank < num_ranks; ++rank) {
      auto* device = this->GetDevice(rank);
      VLOG(2) << "rank " << rank << " device " << device->name();
      auto* event_mgr = device->tensorflow_gpu_device_info()->event_mgr;
      auto* stream = device->tensorflow_gpu_device_info()->stream;
      auto participant = absl::make_unique<NcclManager::Participant>(
          device->executor(), stream, event_mgr, device->gpu_id(),
          &test_case->ins[rank], &test_case->outs[rank], /*global_rank=*/-1,
          this->CreateDoneCallback(test_case.get()));
      NcclManager::instance()->AddToAllReduce(
          std::move(participant),
          {"allreduce", /*num_local_devices=*/num_ranks,
           /*num_global_devices=*/num_ranks, /*communicator_key=*/""},
          reduction_op);
    }

    LOG(INFO) << "Verifying results";
    this->VerifyResults(test_case.get());
  }
}

// Same as the Basic test, but with multiple threads launching parts of many
// reductions.
//
// To run test longer, increase num_ranks, num_collectives_per_iteration and
// time_limit_micros.
TYPED_TEST(NcclManagerTest, MultipleCallers) {
  const int num_ranks = 4;
  const int num_collectives_per_iteration = 10;
  const int num_threads = num_ranks * 2;
  const int time_limit_micros = 1 * 1000 * 1000;  // 1 second

  int64 start = Env::Default()->NowMicros();
  srand(Env::Default()->NowMicros());

  for (;;) {
    std::vector<std::pair<int, int>> case_and_rank;
    std::vector<std::unique_ptr<typename TestFixture::TestCase>> test_cases;
    for (int i = 0; i < num_collectives_per_iteration; ++i) {
      test_cases.emplace_back(this->MakeTestCase(
          /*num_nodes=*/1, num_ranks, ncclSum,
          TensorShape({100, i % 5 + 1, i % 3 + 1}), 1.1f * i));
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
        auto participant = absl::make_unique<NcclManager::Participant>(
            device->executor(), stream, event_mgr, device->gpu_id(),
            &test_case->ins[rank], &test_case->outs[rank], /*global_rank=*/-1,
            this->CreateDoneCallback(test_case));
        NcclManager::instance()->AddToAllReduce(
            std::move(participant),
            {strings::StrCat("allreduce", test_num),
             /*num_local_devices=*/num_ranks,
             /*num_global_devices=*/num_ranks,
             /*communicator_key=*/""},
            ncclSum);
      };
      pool->Schedule(fn);
    }
    pool.reset();  // wait for all work to be scheduled.

    VLOG(2) << "Verifying results for " << num_collectives_per_iteration
            << " collectives";
    for (int i = 0; i < test_cases.size(); ++i) {
      this->VerifyResults(test_cases[i].get());
    }

    int64 delta = Env::Default()->NowMicros() - start;
    if (delta > time_limit_micros) {
      LOG(INFO) << "Ran for " << delta << " microsecs, now quitting";
      break;
    }
  }
}

// Multi-node NCCL tests.

TEST(NcclManagerTest, CommunicatorKey) {
  const string communicator_key =
      NcclManager::instance()->GenerateCommunicatorKey();
  EXPECT_EQ(communicator_key.size(), NCCL_UNIQUE_ID_BYTES);
}

// This test creates `num_nodes` NcclManagers to simulate a multi-node
// environment.  It works on a single node and reuse GPUs.  It enqueues NCCL ops
// on separate stream per rank.
TYPED_TEST(NcclManagerTest, MultiNode) {
  const int num_nodes = 2;
  const int num_ranks_per_node = 4;
  const int num_global_ranks = num_nodes * num_ranks_per_node;
  std::vector<NcclManager> nccl_managers(num_nodes);
  const string collective_key = "allreduce";
  // The NcclManagers in this test synchronize in real-time, so we need to run
  // each node's code in a separate thread.
  // Specifically, the call to ncclGroupEnd() after calling ncclCommInitRank
  // waits for all communicators before returning.
  thread::ThreadPool pool(Env::Default(), "test_multi_node_nccl", num_nodes);

  // First, initialize the communicator_key used for this collective.
  const string communicator_key = nccl_managers[0].GenerateCommunicatorKey();

  for (int op = 0; op < 4; ++op) {
    ncclRedOp_t reduction_op = static_cast<ncclRedOp_t>(op);
    std::unique_ptr<typename TestFixture::TestCase> test_case(
        this->MakeTestCase(num_nodes, num_ranks_per_node, reduction_op,
                           TensorShape({2, 3}), 0.0f));
    for (int node = 0; node < num_nodes; ++node) {
      auto node_fn = [this, node, &nccl_managers, &communicator_key,
                      &collective_key, reduction_op, &test_case] {
        for (int local_rank = 0; local_rank < num_ranks_per_node;
             ++local_rank) {
          auto* device = this->GetDevice(local_rank);
          auto* event_mgr = device->tensorflow_gpu_device_info()->event_mgr;
          auto* stream = device->tensorflow_gpu_device_info()->stream;
          const int global_rank = node * num_ranks_per_node + local_rank;
          auto participant = absl::make_unique<NcclManager::Participant>(
              device->executor(), stream, event_mgr, device->gpu_id(),
              &test_case->ins[global_rank], &test_case->outs[global_rank],
              global_rank, this->CreateDoneCallback(test_case.get()));
          nccl_managers[node].AddToAllReduce(
              std::move(participant),
              {collective_key, num_ranks_per_node, num_global_ranks,
               communicator_key},
              reduction_op);
          VLOG(1) << "AddToAllReduce node " << node << " global_rank "
                  << global_rank;
        }

        // Signal collective ready to launch at this node.
        nccl_managers[node].SignalMultiNodeReady(collective_key);
      };
      pool.Schedule(node_fn);
    }

    VLOG(2) << "Verifying results";
    this->VerifyResults(test_case.get());
  }
}

// Checks that we return error status if a collective_key is used for different
// types of collectives, e.g. a reduction and a broadcast.
TYPED_TEST(NcclManagerTest, ConsistentCollectiveType) {
  const int num_ranks = 2;

  std::unique_ptr<typename TestFixture::TestCase> test_case(this->MakeTestCase(
      1 /* num_nodes */, num_ranks, ncclSum, TensorShape({2, 3}), 0.0f));
  for (int rank = 0; rank < num_ranks; ++rank) {
    auto* device = this->GetDevice(rank);
    auto* event_mgr = device->tensorflow_gpu_device_info()->event_mgr;
    auto* stream = device->tensorflow_gpu_device_info()->stream;
    auto participant = absl::make_unique<NcclManager::Participant>(
        device->executor(), stream, event_mgr, device->gpu_id(),
        &test_case->ins[rank], &test_case->outs[rank], /*global_rank=*/-1,
        this->CreateDoneCallback(test_case.get()));
    if (rank == 0) {
      NcclManager::instance()->AddToAllReduce(std::move(participant),
                                              {"bad_coll_type",
                                               /*num_local_devices=*/num_ranks,
                                               /*num_global_devices=*/num_ranks,
                                               /*communicator_key=*/""},
                                              ncclSum);
    } else {
      NcclManager::instance()->AddBroadcastSend(
          std::move(participant), {"bad_coll_type",
                                   /*num_local_devices=*/num_ranks,
                                   /*num_global_devices=*/num_ranks,
                                   /*communicator_key=*/""});
    }
  }

  this->VerifyError(test_case.get());
}

// Checks that we return error status if different communicator_key is passed to
// same collective.
TYPED_TEST(NcclManagerTest, ConsistentCommunicatorKey) {
  const int num_ranks = 2;

  std::unique_ptr<typename TestFixture::TestCase> test_case(this->MakeTestCase(
      1 /* num_nodes */, num_ranks, ncclSum, TensorShape({2, 3}), 0.0f));
  for (int rank = 0; rank < num_ranks; ++rank) {
    auto* device = this->GetDevice(rank);
    auto* event_mgr = device->tensorflow_gpu_device_info()->event_mgr;
    auto* stream = device->tensorflow_gpu_device_info()->stream;
    auto participant = absl::make_unique<NcclManager::Participant>(
        device->executor(), stream, event_mgr, device->gpu_id(),
        &test_case->ins[rank], &test_case->outs[rank], /*global_rank=*/-1,
        this->CreateDoneCallback(test_case.get()));
    NcclManager::instance()->AddToAllReduce(
        std::move(participant),
        {"bad_coll_type",
         /*num_local_devices=*/num_ranks,
         /*num_global_devices=*/num_ranks,
         rank == 0 ? "" : NcclManager::instance()->GenerateCommunicatorKey()},
        ncclSum);
  }

  this->VerifyError(test_case.get());
}

// Checks that we return error status if the number of devices is inconsistent
// across multiple participants of a collective.
TYPED_TEST(NcclManagerTest, ConsistentNumberOfDevices) {
  const int num_ranks = 2;

  std::unique_ptr<typename TestFixture::TestCase> test_case(this->MakeTestCase(
      1 /* num_nodes */, num_ranks, ncclSum, TensorShape({2, 3}), 0.0f));
  for (int rank = 0; rank < num_ranks; ++rank) {
    auto* device = this->GetDevice(rank);
    auto* event_mgr = device->tensorflow_gpu_device_info()->event_mgr;
    auto* stream = device->tensorflow_gpu_device_info()->stream;
    int num_devices = rank == 0 ? num_ranks : num_ranks + 1;
    auto participant = absl::make_unique<NcclManager::Participant>(
        device->executor(), stream, event_mgr, device->gpu_id(),
        &test_case->ins[rank], &test_case->outs[rank], /*global_rank=*/-1,
        this->CreateDoneCallback(test_case.get()));
    NcclManager::instance()->AddToAllReduce(std::move(participant),
                                            {"bad_coll_type",
                                             /*num_local_devices=*/num_devices,
                                             /*num_global_devices=*/num_devices,
                                             /*communicator_key=*/""},
                                            ncclSum);
  }

  this->VerifyError(test_case.get());
}  // namespace tensorflow

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
