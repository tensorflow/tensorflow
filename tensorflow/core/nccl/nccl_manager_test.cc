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

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include <algorithm>
#include <random>
#include <vector>

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/gpu/gpu_device.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/nccl/nccl_manager.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/unbounded_work_queue.h"

namespace tensorflow {

static std::vector<std::unique_ptr<BaseGPUDevice>> GetGPUDevices() {
  std::vector<std::unique_ptr<Device>> devices;
  TF_CHECK_OK(DeviceFactory::GetFactory(DEVICE_GPU)
                  ->AddDevices(SessionOptions(), "", &devices));
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
    TestCase(int num_nodes, int num_ranks_per_node)
        : num_nodes(num_nodes), num_ranks_per_node(num_ranks_per_node) {}
    std::vector<Tensor> ins;
    std::vector<Tensor> outs;
    Tensor expected;
    const int num_nodes;
    const int num_ranks_per_node;

    mutex mu;
    Status final_status;
    int num_completed TF_GUARDED_BY(mu) = 0;
    condition_variable done_cv;
  };

  static void SetUpTestSuite() {
    setenv("NCCL_DEBUG", "INFO", 1 /* replace */);
    setenv("NCCL_LAUNCH_MODE", "PARALLEL", 1 /* replace */);
    devices_ = new std::vector<std::unique_ptr<BaseGPUDevice>>(GetGPUDevices());
    VLOG(1) << "Running test with " << devices_->size() << " gpus";
    if (devices_->size() <= 1) {
      LOG(FATAL) << "Cannot run NCCL test without multiple GPUs";
    }
    work_queue_ = new UnboundedWorkQueue(Env::Default(), "nccl_manager_test");
  }

  void SetUp() override {
    ASSERT_GT(devices_->size(), 0) << "No GPUs found";
    ASSERT_NE(work_queue_, nullptr);
  }

  static int32 NumGPUs() { return static_cast<int32>(devices_->size()); }

  // Let N = #GPUs.  When N is even, num_nodes=2 and num_ranks_per_node=N/2.
  // When N is odd, num_nodes=2 and num_ranks_per_node=(N-1)/2.
  static void PopulateMultiNodeParams(int* num_nodes, int* num_ranks_per_node) {
    const auto num_gpus = NumGPUs();
    CHECK_GT(num_gpus, 1);
    *num_nodes = 2;
    if (num_gpus % 2 == 0) {
      *num_ranks_per_node = num_gpus / 2;
    } else {
      *num_ranks_per_node = (num_gpus - 1) / 2;
    }
  }

  static void TearDownTestSuite() {
    delete devices_;
    delete work_queue_;
  }

  TestCase* MakeReductionTestCase(int num_nodes, int num_ranks_per_node,
                                  ncclRedOp_t reduction_op, TensorShape shape,
                                  float value_offset) {
    TestCase* test_case = new TestCase(num_nodes, num_ranks_per_node);
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
        auto* device = GetDevice(num_ranks_per_node, node, local_rank);
        auto* stream = device->tensorflow_accelerator_device_info()->stream;

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
        TF_CHECK_OK(stream->Memcpy(&in_gpu_mem, in_cpu.flat<Scalar>().data(),
                                   in_cpu.TotalBytes()));
      }
    }

    return test_case;
  }

  TestCase* MakeGatherTestCase(int num_nodes, int num_ranks_per_node,
                               TensorShape in_shape, TensorShape out_shape) {
    TestCase* test_case = new TestCase(num_nodes, num_ranks_per_node);
    test_case->expected = Tensor(data_type_, out_shape);
    test::FillFn<Scalar>(&test_case->expected,
                         [](int) { return static_cast<Scalar>(0); });

    float value_scale = 0.01;  // Small scale to avoid fp16 overflow.
    for (int node = 0; node < num_nodes; ++node) {
      for (int i = 0; i < num_ranks_per_node; ++i) {
        auto* device = GetDevice(num_ranks_per_node, node, i);
        auto* stream = device->tensorflow_accelerator_device_info()->stream;

        Tensor in_cpu(data_type_, in_shape);
        test::FillFn<Scalar>(&in_cpu, [&](int index) {
          return static_cast<Scalar>((index + 1) * value_scale);
        });
        // Starting index for this rank's tensor in the all-gathered output.
        int32_t gather_idx =
            (node * num_ranks_per_node + i) * in_shape.num_elements();
        for (int j = 0; j < in_shape.num_elements(); ++j) {
          auto in_val = in_cpu.flat<Scalar>()(j);
          auto out_expr = test_case->expected.template flat<Scalar>();
          out_expr(gather_idx + j) = in_val;
        }

        value_scale *= 10;
        test_case->ins.emplace_back(GpuAllocator(device), data_type_, in_shape);
        test_case->outs.emplace_back(GpuAllocator(device), data_type_,
                                     out_shape);

        const Tensor& in_gpu = test_case->ins.back();
        auto in_gpu_mem = AsDeviceMemory(in_gpu.flat<Scalar>().data());
        TF_CHECK_OK(stream->Memcpy(&in_gpu_mem, in_cpu.flat<Scalar>().data(),
                                   in_cpu.TotalBytes()));
      }
    }

    return test_case;
  }

  // Make a broadcast test which broadcasts a tensor with shape `shape` from
  // `src_node`, `src_rank` to all other ranks.
  // If `in_place` is true, input and output are the same for the source,
  // otherwise they are tensors backed by different buffers.
  TestCase* MakeBroadcastTestCase(int num_nodes, int num_ranks_per_node,
                                  TensorShape shape, int src_node, int src_rank,
                                  bool in_place) {
    TestCase* test_case = new TestCase(num_nodes, num_ranks_per_node);
    test_case->expected = Tensor(data_type_, shape);
    test::FillFn<Scalar>(&test_case->expected,
                         [](int) { return static_cast<Scalar>(1); });

    for (int node = 0; node < num_nodes; ++node) {
      for (int local_rank = 0; local_rank < num_ranks_per_node; ++local_rank) {
        auto* device = GetDevice(num_ranks_per_node, node, local_rank);
        if (node == src_node && local_rank == src_rank) {
          test_case->ins.emplace_back(GpuAllocator(device), data_type_, shape);
          if (in_place) {
            test_case->outs.emplace_back(test_case->ins.back());
          } else {
            test_case->outs.emplace_back(GpuAllocator(device), data_type_,
                                         shape);
          }
          Tensor in_cpu(data_type_, shape);
          test::FillFn<Scalar>(&in_cpu,
                               [](int) { return static_cast<Scalar>(1); });
          const Tensor& in_gpu = test_case->ins.back();
          auto in_gpu_mem = AsDeviceMemory(in_gpu.flat<Scalar>().data());
          auto* stream = device->tensorflow_accelerator_device_info()->stream;
          TF_CHECK_OK(stream->Memcpy(&in_gpu_mem, in_cpu.flat<Scalar>().data(),
                                     in_cpu.TotalBytes()));
        } else {
          test_case->ins.emplace_back(Tensor());
          test_case->outs.emplace_back(GpuAllocator(device), data_type_, shape);
        }
      }
    }

    return test_case;
  }

  // Waits for the done callback to be called for each participant.
  void WaitForTestCompletion(TestCase* test_case) {
    mutex_lock l(test_case->mu);
    while (test_case->num_completed != test_case->outs.size()) {
      test_case->done_cv.wait(l);
    }
  }

  void VerifyResults(TestCase* test_case) {
    WaitForTestCompletion(test_case);
    TF_ASSERT_OK(test_case->final_status);
    // Copy memory to host and verify.
    for (int node = 0; node < test_case->num_nodes; ++node) {
      for (int local_rank = 0; local_rank < test_case->num_ranks_per_node;
           ++local_rank) {
        auto* device =
            GetDevice(test_case->num_ranks_per_node, node, local_rank);
        auto* stream = device->tensorflow_accelerator_device_info()->stream;
        const int global_rank =
            GlobalRank(test_case->num_ranks_per_node, node, local_rank);
        const Tensor& out_gpu = test_case->outs[global_rank];
        Tensor out_cpu(data_type_, out_gpu.shape());
        auto out_gpu_mem = AsDeviceMemory(out_gpu.flat<Scalar>().data());
        TF_CHECK_OK(stream->Memcpy(out_cpu.flat<Scalar>().data(), out_gpu_mem,
                                   out_cpu.TotalBytes()));
        TF_ASSERT_OK(stream->BlockHostUntilDone());
        VLOG(1) << "Verifying rank " << global_rank << " expected shape "
                << test_case->expected.shape() << " out shape "
                << out_cpu.shape();
        test::ExpectClose(test_case->expected, out_cpu);
      }
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
      test_case->final_status.Update(s);
      if (++test_case->num_completed == test_case->outs.size()) {
        test_case->done_cv.notify_one();
      }
    };
  }

  struct NodeState {
    NcclManager nccl_manager;
    std::atomic<int> launched{0};
  };

  void RunMultiNodeAllReduceTest(const int num_nodes,
                                 const int num_ranks_per_node) {
    std::vector<NodeState> node_states(num_nodes);
    RunMultiNodeAllReduceTest(node_states, num_ranks_per_node);
  }

  void RunMultiNodeAllReduceTest(std::vector<NodeState>& node_states,
                                 const int num_ranks_per_node) {
    const int num_nodes = node_states.size();
    const int num_global_ranks = num_nodes * num_ranks_per_node;
    const string collective_key = "allreduce";
    // The NcclManagers in this test synchronize in real-time, so we need to run
    // each node's code in a separate thread.
    // Specifically, the call to ncclGroupEnd() after calling ncclCommInitRank
    // waits for all communicators before returning.

    // First, initialize the communicator_key used for this collective.
    const string communicator_key =
        node_states[0].nccl_manager.GenerateCommunicatorKey();

    for (int op = 0; op < 4; ++op) {
      ncclRedOp_t reduction_op = static_cast<ncclRedOp_t>(op);
      std::unique_ptr<TestCase> test_case(
          this->MakeReductionTestCase(num_nodes, num_ranks_per_node,
                                      reduction_op, TensorShape({2, 3}), 0.0f));
      for (int node = 0; node < num_nodes; ++node) {
        auto node_fn = [this, node, num_ranks_per_node, num_global_ranks,
                        &node_states, &communicator_key, &collective_key,
                        reduction_op, &test_case] {
          for (int local_rank = 0; local_rank < num_ranks_per_node;
               ++local_rank) {
            auto* device = GetDevice(num_ranks_per_node, node, local_rank);
            auto* info = device->tensorflow_accelerator_device_info();
            auto* stream = device->tensorflow_accelerator_device_info()->stream;
            const int global_rank =
                GlobalRank(num_ranks_per_node, node, local_rank);
            auto participant = absl::make_unique<NcclManager::Participant>(
                device->executor(), stream, info, &test_case->ins[global_rank],
                &test_case->outs[global_rank], global_rank,
                this->CreateDoneCallback(test_case.get()));
            node_states[node].nccl_manager.AddToAllReduce(
                std::move(participant),
                {collective_key, num_ranks_per_node, num_global_ranks,
                 communicator_key, /*source_rank=*/-1},
                reduction_op);
            VLOG(1) << "AddToAllReduce node " << node << " global_rank "
                    << global_rank;
          }

          // Signal collective ready to launch at this node.
          node_states[node].nccl_manager.SignalMultiNodeReady(collective_key);
        };
        this->work_queue_->Schedule(node_fn);
      }

      VLOG(2) << "Verifying results";
      this->VerifyResults(test_case.get());
    }
  }

  void RunMultiNodeBroadcastTest(const int num_nodes,
                                 const int num_ranks_per_node,
                                 const int src_node, const int src_local_rank,
                                 const bool in_place) {
    const int num_global_ranks = num_nodes * num_ranks_per_node;
    const int src_global_rank = src_node * num_ranks_per_node + src_local_rank;
    const string collective_key = "broadcast";
    std::vector<NodeState> node_states(num_nodes);
    const string communicator_key =
        node_states[0].nccl_manager.GenerateCommunicatorKey();
    std::unique_ptr<TestCase> test_case(this->MakeBroadcastTestCase(
        num_nodes, num_ranks_per_node, TensorShape({5, 6}), src_node,
        src_local_rank, in_place));
    for (int node = 0; node < num_nodes; ++node) {
      for (int local_rank = 0; local_rank < num_ranks_per_node; ++local_rank) {
        // Launch each rank in a separate thread to test concurrent,
        // randomly-ordered calls into NcclManager.
        auto rank_fn = [this, node, num_ranks_per_node, num_global_ranks,
                        src_global_rank, local_rank, &node_states,
                        &collective_key, &communicator_key, &test_case]() {
          auto* device = GetDevice(num_ranks_per_node, node, local_rank);
          auto* info = device->tensorflow_accelerator_device_info();
          auto* stream = device->tensorflow_accelerator_device_info()->stream;
          const int global_rank =
              GlobalRank(num_ranks_per_node, node, local_rank);
          auto* input = global_rank == src_global_rank
                            ? &test_case->ins[global_rank]
                            : nullptr;
          auto* output = test_case->outs[global_rank].NumElements() == 0
                             ? nullptr
                             : &test_case->outs[global_rank];
          auto participant = absl::make_unique<NcclManager::Participant>(
              device->executor(), stream, info, input, output, global_rank,
              this->CreateDoneCallback(test_case.get()));
          if (global_rank == src_global_rank) {
            node_states[node].nccl_manager.AddBroadcastSend(
                std::move(participant),
                {collective_key, num_ranks_per_node, num_global_ranks,
                 communicator_key, src_global_rank});
          } else {
            node_states[node].nccl_manager.AddBroadcastRecv(
                std::move(participant),
                {collective_key, num_ranks_per_node, num_global_ranks,
                 communicator_key, src_global_rank});
          }

          if (++node_states[node].launched == num_ranks_per_node) {
            // Signal collective ready to launch at this node.
            node_states[node].nccl_manager.SignalMultiNodeReady(collective_key);
          }
        };
        this->work_queue_->Schedule(std::move(rank_fn));
      }
    }

    VLOG(2) << "Verifying results";
    this->VerifyResults(test_case.get());
  }

  static int GlobalRank(int num_ranks_per_node, int node, int local_rank) {
    return node * num_ranks_per_node + local_rank;
  }

  static BaseGPUDevice* GetDevice(int num_ranks_per_node, int node,
                                  int local_rank) {
    const int device_idx = GlobalRank(num_ranks_per_node, node, local_rank);
    CHECK_LT(device_idx, devices_->size());
    return (*devices_)[device_idx].get();
  }

  static UnboundedWorkQueue* work_queue_;

 private:
  static Allocator* GpuAllocator(BaseGPUDevice* device) {
    return device->GetAllocator(AllocatorAttributes());
  }

  static se::DeviceMemory<Scalar> AsDeviceMemory(const Scalar* cuda_memory) {
    se::DeviceMemoryBase wrapped(const_cast<Scalar*>(cuda_memory));
    se::DeviceMemory<Scalar> typed(wrapped);
    return typed;
  }

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
template <typename Scalar>
UnboundedWorkQueue* NcclManagerTest<Scalar>::work_queue_ = nullptr;

// Instantiate tests for float and double.
using TypeList = ::testing::Types<float, double>;
TYPED_TEST_SUITE(NcclManagerTest, TypeList);

// Test basic sum reduction.
TYPED_TEST(NcclManagerTest, BasicSumReduction) {
  const int num_ranks = this->NumGPUs();

  for (int op = 0; op < 4; ++op) {
    ncclRedOp_t reduction_op = static_cast<ncclRedOp_t>(op);
    std::unique_ptr<typename TestFixture::TestCase> test_case(
        this->MakeReductionTestCase(/*num_nodes=*/1, num_ranks, reduction_op,
                                    TensorShape({2, 3}), 0.0f));
    for (int rank = 0; rank < num_ranks; ++rank) {
      auto* device = this->GetDevice(num_ranks, /*node=*/0, rank);
      VLOG(2) << "rank " << rank << " device " << device->name();
      auto* info = device->tensorflow_accelerator_device_info();
      auto* stream = device->tensorflow_accelerator_device_info()->stream;
      auto participant = absl::make_unique<NcclManager::Participant>(
          device->executor(), stream, info, &test_case->ins[rank],
          &test_case->outs[rank], /*global_rank=*/-1,
          this->CreateDoneCallback(test_case.get()));
      NcclManager::instance()->AddToAllReduce(
          std::move(participant),
          {"allreduce", /*num_local_devices=*/num_ranks,
           /*num_global_devices=*/num_ranks, /*communicator_key=*/"",
           /*source_rank=*/-1},
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
  const int num_ranks = this->NumGPUs();
  const int num_collectives_per_iteration = 10;
  const int time_limit_micros = 1 * 1000 * 1000;  // 1 second

  int64_t start = Env::Default()->NowMicros();
  srand(Env::Default()->NowMicros());

  for (;;) {
    std::vector<std::pair<int, int>> case_and_rank;
    std::vector<std::unique_ptr<typename TestFixture::TestCase>> test_cases;
    for (int i = 0; i < num_collectives_per_iteration; ++i) {
      test_cases.emplace_back(this->MakeReductionTestCase(
          /*num_nodes=*/1, num_ranks, ncclSum,
          TensorShape({100, i % 5 + 1, i % 3 + 1}), 1.1f * i));
      for (int j = 0; j < num_ranks; ++j) {
        case_and_rank.emplace_back(i, j);
      }
    }

    for (int rank = 0; rank < num_ranks; ++rank) {
      auto* device = this->GetDevice(num_ranks, /*node=*/0, rank);
      auto* stream = device->tensorflow_accelerator_device_info()->stream;
      TF_ASSERT_OK(stream->BlockHostUntilDone());
    }

    std::shuffle(case_and_rank.begin(), case_and_rank.end(),
                 std::mt19937(std::random_device()()));

    mutex mu;  // guards case_and_rank.
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
        auto* device = this->GetDevice(num_ranks, /*node=*/0, rank);
        auto* info = device->tensorflow_accelerator_device_info();
        auto* stream = device->tensorflow_accelerator_device_info()->stream;
        typename TestFixture::TestCase* test_case = test_cases[test_num].get();
        auto participant = absl::make_unique<NcclManager::Participant>(
            device->executor(), stream, info, &test_case->ins[rank],
            &test_case->outs[rank], /*global_rank=*/-1,
            this->CreateDoneCallback(test_case));
        NcclManager::instance()->AddToAllReduce(
            std::move(participant),
            {strings::StrCat("allreduce", test_num),
             /*num_local_devices=*/num_ranks,
             /*num_global_devices=*/num_ranks,
             /*communicator_key=*/"", /*source_rank=*/-1},
            ncclSum);
      };
      this->work_queue_->Schedule(fn);
    }

    VLOG(2) << "Verifying results for " << num_collectives_per_iteration
            << " collectives";
    for (int i = 0; i < test_cases.size(); ++i) {
      this->VerifyResults(test_cases[i].get());
    }

    int64_t delta = Env::Default()->NowMicros() - start;
    if (delta > time_limit_micros) {
      LOG(INFO) << "Ran for " << delta << " microsecs, now quitting";
      break;
    }
  }
}

// Test basic all-gather.
TYPED_TEST(NcclManagerTest, BasicAllGather) {
  const int num_ranks = this->NumGPUs();
  for (int i = 0; i < num_ranks; ++i) {
    std::unique_ptr<typename TestFixture::TestCase> test_case(
        this->MakeGatherTestCase(/*num_nodes=*/1, num_ranks,
                                 TensorShape({2, 3}),
                                 TensorShape({2 * num_ranks, 3})));
    for (int rank = 0; rank < num_ranks; ++rank) {
      auto* device = this->GetDevice(num_ranks, /*node=*/0, rank);
      VLOG(2) << "rank " << rank << " device " << device->name();
      auto* info = device->tensorflow_accelerator_device_info();
      auto* stream = device->tensorflow_accelerator_device_info()->stream;
      auto participant = absl::make_unique<NcclManager::Participant>(
          device->executor(), stream, info, &test_case->ins[rank],
          &test_case->outs[rank], rank,
          this->CreateDoneCallback(test_case.get()));
      NcclManager::instance()->AddToAllGather(
          std::move(participant),
          {"allgather", /*num_local_devices=*/num_ranks,
           /*num_global_devices=*/num_ranks, /*communicator_key=*/"",
           /*source_rank=*/-1});
    }

    LOG(INFO) << "Verifying results";
    this->VerifyResults(test_case.get());
  }
}

// Test basic broadcast.
TYPED_TEST(NcclManagerTest, BasicBroadcast) {
  this->RunMultiNodeBroadcastTest(/*num_nodes=*/1,
                                  /*num_ranks_per_node=*/this->NumGPUs(),
                                  /*src_node=*/0, /*src_local_rank=*/0,
                                  /*in_place=*/false);
}

// Test in-place broadcast.
TYPED_TEST(NcclManagerTest, InPlaceBroadcast) {
  this->RunMultiNodeBroadcastTest(/*num_nodes=*/1,
                                  /*num_ranks_per_node=*/this->NumGPUs(),
                                  /*src_node=*/0, /*src_local_rank=*/0,
                                  /*in_place=*/true);
}

// Test broadcast with increasing ranks.
TYPED_TEST(NcclManagerTest, BroadcastWithDifferentRanks) {
  for (int num_ranks = 1; num_ranks <= this->NumGPUs(); ++num_ranks) {
    const int src_rank = static_cast<int>(random::New64() % num_ranks);
    for (int in_place_idx = 0; in_place_idx <= 1; ++in_place_idx) {
      const bool in_place = in_place_idx == 0;
      this->RunMultiNodeBroadcastTest(/*num_nodes=*/1, num_ranks,
                                      /*src_node=*/0, src_rank, in_place);
    }
  }
}

// Multi-node NCCL tests.

TEST(NcclManagerTest, CommunicatorKey) {
  const string communicator_key =
      NcclManager::instance()->GenerateCommunicatorKey();
  EXPECT_EQ(communicator_key.size(), NCCL_UNIQUE_ID_BYTES);
}

#if !TENSORFLOW_USE_ROCM
// ROCm platform currently does not support simulating a multi-node
// environment, on a single node with multiple GPUS. So tests that rely
// upon such simulation need to be skipped on the ROCm platform

// This test creates `num_nodes` NcclManagers to simulate a multi-node
// environment.  It works on a single node with multiple GPUs.  It enqueues NCCL
// kernels on separate stream per rank.
TYPED_TEST(NcclManagerTest, MultiNode) {
  int num_nodes;
  int num_ranks_per_node;
  this->PopulateMultiNodeParams(&num_nodes, &num_ranks_per_node);
  VLOG(1) << "Calling RunMultiNodeAllReduceTest with num_nodes=" << num_nodes
          << " and num_ranks_per_node=" << num_ranks_per_node;
  this->RunMultiNodeAllReduceTest(num_nodes, num_ranks_per_node);
}
#endif

// Tests that specifying `communicator_key` with a single node NCCL collective
// works well.
TYPED_TEST(NcclManagerTest, MultiNodeSingle) {
  this->RunMultiNodeAllReduceTest(/*num_nodes=*/1,
                                  /*num_ranks_per_node=*/this->NumGPUs());
}

#if !TENSORFLOW_USE_ROCM
// ROCm platform currently does not support simulating a multi-node
// environment, on a single node with multiple GPUS. So tests that rely
// upon such simulation need to be skipped on the ROCm platform

// Multi-node broadcast.
TYPED_TEST(NcclManagerTest, MultiNodeBroadcast) {
  int num_nodes;
  int num_ranks_per_node;
  this->PopulateMultiNodeParams(&num_nodes, &num_ranks_per_node);
  VLOG(1) << "Calling RunMultiNodeBroadcastTest with num_nodes=" << num_nodes
          << " and num_ranks_per_node=" << num_ranks_per_node;
  this->RunMultiNodeBroadcastTest(num_nodes, num_ranks_per_node,
                                  /*src_node=*/0, /*src_local_rank=*/0,
                                  /*in_place=*/true);
}
#endif

// Checks that we return error status if a collective_key is used for different
// types of collectives, e.g.a reduction and a broadcast.
TYPED_TEST(NcclManagerTest, ConsistentCollectiveType) {
  const int num_ranks = 2;

  std::unique_ptr<typename TestFixture::TestCase> test_case(
      this->MakeReductionTestCase(/*num_nodes=*/1, num_ranks, ncclSum,
                                  TensorShape({2, 3}), 0.0f));
  for (int rank = 0; rank < num_ranks; ++rank) {
    auto* device = this->GetDevice(num_ranks, /*node=*/0, rank);
    auto* info = device->tensorflow_accelerator_device_info();
    auto* stream = device->tensorflow_accelerator_device_info()->stream;
    auto participant = absl::make_unique<NcclManager::Participant>(
        device->executor(), stream, info, &test_case->ins[rank],
        &test_case->outs[rank], /*global_rank=*/-1,
        this->CreateDoneCallback(test_case.get()));
    if (rank == 0) {
      NcclManager::instance()->AddToAllReduce(std::move(participant),
                                              {"bad_coll_type",
                                               /*num_local_devices=*/num_ranks,
                                               /*num_global_devices=*/num_ranks,
                                               /*communicator_key=*/"",
                                               /*source_rank=*/-1},
                                              ncclSum);
    } else {
      NcclManager::instance()->AddBroadcastSend(
          std::move(participant),
          {"bad_coll_type",
           /*num_local_devices=*/num_ranks,
           /*num_global_devices=*/num_ranks,
           /*communicator_key=*/"", /*source_rank=*/-1});
    }
  }

  this->VerifyError(test_case.get());
}

// Checks that we return error status if different communicator_key is passed to
// same collective.
TYPED_TEST(NcclManagerTest, ConsistentCommunicatorKey) {
  const int num_ranks = 2;

  std::unique_ptr<typename TestFixture::TestCase> test_case(
      this->MakeReductionTestCase(/*num_nodes=*/1, num_ranks, ncclSum,
                                  TensorShape({2, 3}), 0.0f));
  for (int rank = 0; rank < num_ranks; ++rank) {
    auto* device = this->GetDevice(num_ranks, /*node=*/0, rank);
    auto* info = device->tensorflow_accelerator_device_info();
    auto* stream = device->tensorflow_accelerator_device_info()->stream;
    auto participant = absl::make_unique<NcclManager::Participant>(
        device->executor(), stream, info, &test_case->ins[rank],
        &test_case->outs[rank], /*global_rank=*/-1,
        this->CreateDoneCallback(test_case.get()));
    NcclManager::instance()->AddToAllReduce(
        std::move(participant),
        {"bad_coll_type",
         /*num_local_devices=*/num_ranks,
         /*num_global_devices=*/num_ranks,
         rank == 0 ? "" : NcclManager::instance()->GenerateCommunicatorKey(),
         /*source_rank=*/-1},
        ncclSum);
  }

  this->VerifyError(test_case.get());
}

// Checks that we return error status if the number of devices is inconsistent
// across multiple participants of a collective.
TYPED_TEST(NcclManagerTest, ConsistentNumberOfDevices) {
  const int num_ranks = 2;

  std::unique_ptr<typename TestFixture::TestCase> test_case(
      this->MakeReductionTestCase(/*num_nodes=*/1, num_ranks, ncclSum,
                                  TensorShape({2, 3}), 0.0f));
  for (int rank = 0; rank < num_ranks; ++rank) {
    auto* device = this->GetDevice(num_ranks, /*node=*/0, rank);
    auto* info = device->tensorflow_accelerator_device_info();
    auto* stream = device->tensorflow_accelerator_device_info()->stream;
    int num_devices = rank == 0 ? num_ranks : num_ranks + 1;
    auto participant = absl::make_unique<NcclManager::Participant>(
        device->executor(), stream, info, &test_case->ins[rank],
        &test_case->outs[rank], /*global_rank=*/-1,
        this->CreateDoneCallback(test_case.get()));
    NcclManager::instance()->AddToAllReduce(std::move(participant),
                                            {"bad_coll_type",
                                             /*num_local_devices=*/num_devices,
                                             /*num_global_devices=*/num_devices,
                                             /*communicator_key=*/"",
                                             /*source_rank=*/-1},
                                            ncclSum);
  }

  this->VerifyError(test_case.get());
}

// Checks that we return error status if a broadcast does not have source.
TYPED_TEST(NcclManagerTest, BroadcastNoSource) {
  const int num_ranks = 2;

  std::unique_ptr<typename TestFixture::TestCase> test_case(
      this->MakeBroadcastTestCase(/*num_nodes=*/1, num_ranks,
                                  TensorShape({2, 3}), /*src_node=*/-1,
                                  /*src_rank=*/-1, false));
  for (int rank = 0; rank < num_ranks; ++rank) {
    auto* device = this->GetDevice(num_ranks, /*node=*/0, rank);
    auto* info = device->tensorflow_accelerator_device_info();
    auto* stream = device->tensorflow_accelerator_device_info()->stream;
    auto participant = absl::make_unique<NcclManager::Participant>(
        device->executor(), stream, info, nullptr, &test_case->outs[rank], rank,
        this->CreateDoneCallback(test_case.get()));
    NcclManager::instance()->AddBroadcastRecv(std::move(participant),
                                              {"bcast_no_send",
                                               /*num_local_devices=*/num_ranks,
                                               /*num_global_devices=*/num_ranks,
                                               /*communicator_key=*/"",
                                               /*source_rank=*/-1});
  }

  this->VerifyError(test_case.get());
}

// Checks that we return error status if a broadcast has multiple sends.
TYPED_TEST(NcclManagerTest, BroadcastMultipleSends) {
  const int num_ranks = 2;

  std::unique_ptr<typename TestFixture::TestCase> test_case(
      this->MakeBroadcastTestCase(/*num_nodes=*/1, num_ranks,
                                  TensorShape({2, 3}), /*src_node=*/-1,
                                  /*src_rank=*/-1, false));
  for (int rank = 0; rank < num_ranks; ++rank) {
    auto* device = this->GetDevice(num_ranks, /*node=*/0, rank);
    auto* info = device->tensorflow_accelerator_device_info();
    auto* stream = device->tensorflow_accelerator_device_info()->stream;
    auto participant = absl::make_unique<NcclManager::Participant>(
        device->executor(), stream, info, &test_case->outs[rank],
        &test_case->outs[rank], rank,
        this->CreateDoneCallback(test_case.get()));
    NcclManager::instance()->AddBroadcastSend(std::move(participant),
                                              {"bcast_multiple_send",
                                               /*num_local_devices=*/num_ranks,
                                               /*num_global_devices=*/num_ranks,
                                               /*communicator_key=*/"",
                                               /*source_rank=*/-1});
  }

  this->VerifyError(test_case.get());
}

// Checks that we return error status if a broadcast has inconsistent source
// ranks.
TYPED_TEST(NcclManagerTest, BroadcastInconsistentSource) {
  const int num_ranks = 2;

  std::unique_ptr<typename TestFixture::TestCase> test_case(
      this->MakeBroadcastTestCase(/*num_nodes=*/1, num_ranks,
                                  TensorShape({2, 3}), /*src_node=*/-1,
                                  /*src_rank=*/-1, false));
  for (int rank = 0; rank < num_ranks; ++rank) {
    auto* device = this->GetDevice(num_ranks, /*node=*/0, rank);
    auto* info = device->tensorflow_accelerator_device_info();
    auto* stream = device->tensorflow_accelerator_device_info()->stream;
    auto participant = absl::make_unique<NcclManager::Participant>(
        device->executor(), stream, info, &test_case->outs[rank],
        &test_case->outs[rank], rank,
        this->CreateDoneCallback(test_case.get()));
    NcclManager::instance()->AddBroadcastRecv(std::move(participant),
                                              {"bcast_inconsistent_source",
                                               /*num_local_devices=*/num_ranks,
                                               /*num_global_devices=*/num_ranks,
                                               /*communicator_key=*/"",
                                               /*source_rank=*/rank});
  }

  this->VerifyError(test_case.get());
}

#if !TENSORFLOW_USE_ROCM
// ROCm platform currently does not support simulating a multi-node
// environment, on a single node with multiple GPUS. So tests that rely
// upon such simulation need to be skipped on the ROCm platform

TYPED_TEST(NcclManagerTest, AbortThenReset) {
  using NodeState = typename TestFixture::NodeState;
  using TestCase = typename TestFixture::TestCase;
  const int num_nodes = 2;
  std::vector<NodeState> nodes(num_nodes);
  // First do a normal all-reduce to simulate the case when there're
  // multiple communicators.
  this->RunMultiNodeAllReduceTest(nodes, /* num_ranks_per_node */ 1);

  const string collective_key = "allreduce";
  ncclRedOp_t reduction_op = static_cast<ncclRedOp_t>(0);
  auto node_fn = [&](TestCase* test_case, int node,
                     const string& communicator_key) {
    auto* device = this->GetDevice(/* num_ranks_per_node */ 1, node,
                                   /* local_rank */ 0);
    auto* info = device->tensorflow_accelerator_device_info();
    auto* stream = device->tensorflow_accelerator_device_info()->stream;
    auto participant = absl::make_unique<NcclManager::Participant>(
        device->executor(), stream, info, &test_case->ins[node],
        &test_case->outs[node], /* global_rank */ node,
        this->CreateDoneCallback(test_case));
    nodes[node].nccl_manager.AddToAllReduce(
        std::move(participant),
        {collective_key, /* num_local_devices */ 1,
         /* num_global_devices */ num_nodes, communicator_key,
         /*source_rank=*/-1},
        reduction_op);
    nodes[node].nccl_manager.SignalMultiNodeReady(collective_key);
  };

  // Use a new communicator_key, which uses a new set of ncclComm underneath.
  string communicator_key = nodes[0].nccl_manager.GenerateCommunicatorKey();
  // Do a normal all-reduce with this communicator key to initialize ncclComm.
  // This is because ncclCommInitRank waits for all ranks and is blocking.
  {
    std::unique_ptr<typename TestFixture::TestCase> test_case(
        this->MakeReductionTestCase(
            /* num_nodes */ num_nodes, /* num_ranks_per_node */ 1, reduction_op,
            TensorShape({2, 3}), 0.0f));
    for (int i = 0; i < num_nodes; ++i) {
      this->work_queue_->Schedule(
          [&node_fn, &test_case, i, communicator_key]() {
            node_fn(test_case.get(), i, communicator_key);
          });
    }
    this->VerifyResults(test_case.get());
  }

  // A hanging all-reduce.
  ASSERT_GT(num_nodes, 1);
  std::unique_ptr<typename TestFixture::TestCase> test_case(
      this->MakeReductionTestCase(
          /* num_nodes */ num_nodes, /* num_ranks_per_node */ 1, reduction_op,
          TensorShape({2, 3}), 0.0f));
  node_fn(test_case.get(), 0, communicator_key);
  Env::Default()->SleepForMicroseconds(1000000);
  for (auto& node : nodes) {
    node.nccl_manager.StartAbort(errors::Unavailable("peer down"));
  }
  {
    mutex_lock l(test_case->mu);
    while (test_case->num_completed != 1) {
      test_case->done_cv.wait(l);
    }
  }

  // Reset the aborted NcclManager and then run another all-reduce with the
  // resetted NcclManagers.
  for (auto& node : nodes) {
    node.nccl_manager.Reset();
  }
  // Regenerate the communicator_key, because this is needed to create new
  // communicators.
  communicator_key = nodes[0].nccl_manager.GenerateCommunicatorKey();
  {
    std::unique_ptr<typename TestFixture::TestCase> test_case(
        this->MakeReductionTestCase(
            /* num_nodes */ num_nodes, /* num_ranks_per_node */ 1, reduction_op,
            TensorShape({2, 3}), 0.0f));
    for (int i = 0; i < num_nodes; ++i) {
      this->work_queue_->Schedule(
          [&node_fn, &test_case, i, communicator_key]() {
            node_fn(test_case.get(), i, communicator_key);
          });
    }
    this->VerifyResults(test_case.get());
  }
}

#endif

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
