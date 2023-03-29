/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include <algorithm>

#include "absl/memory/memory.h"
#include "tensorflow/core/common_runtime/base_collective_executor.h"
#include "tensorflow/core/common_runtime/collective_test_util.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/device_resolver_local.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/common_runtime/test_collective_executor_mgr.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/unbounded_work_queue.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
static constexpr int kStepId = 10;

std::unique_ptr<OpKernel> GetKernel(const NodeDef& node, DeviceBase* device) {
  Status status;
  std::unique_ptr<OpKernel> k = CreateOpKernel(
      DEVICE_GPU, device, device->GetAllocator(AllocatorAttributes()), node,
      TF_GRAPH_DEF_VERSION, &status);
  if (!status.ok()) LOG(FATAL) << status;
  return k;
}

std::unique_ptr<OpKernel> GetAdd(DeviceBase* device) {
  NodeDef node_def;
  NodeDefBuilder builder("add_node", "Add");
  TF_CHECK_OK(builder.Attr("T", DT_FLOAT)
                  .Input(FakeInput(DT_FLOAT))
                  .Input(FakeInput(DT_FLOAT))
                  .Finalize(&node_def));
  return GetKernel(node_def, device);
}

std::unique_ptr<OpKernel> GetDiv(DeviceBase* device) {
  NodeDef node_def;
  NodeDefBuilder builder("add_node", "Div");
  TF_CHECK_OK(builder.Attr("T", DT_FLOAT)
                  .Input(FakeInput(DT_FLOAT))
                  .Input(FakeInput(DT_FLOAT))
                  .Finalize(&node_def));
  return GetKernel(node_def, device);
}

class NcclTestBase : public ::testing::Test {
 protected:
  class DeviceInstance;

  NcclTestBase(CollectiveType collective_type, const string& collective_name)
      : collective_type_(collective_type), collective_name_(collective_name) {}

  void Init(const int num_ranks) {
    setenv("NCCL_DEBUG", "INFO", 1 /* replace */);
    setenv("NCCL_LAUNCH_MODE", "PARALLEL", 1 /* replace */);
    test_env_ = CreateCollectiveTestEnv(/*num_workers*/ 1,
                                        /*num_devices_per_worker*/ num_ranks,
                                        DEVICE_GPU, /*use_nccl=*/true);
    for (int rank = 0; rank < num_ranks; ++rank) {
      instances_.push_back(std::make_unique<DeviceInstance>(
          rank, collective_name_, collective_type_, test_env_.get()));
    }
  }

  // Initialize `input` tensor at rank `rank`.
  virtual void InitInput(Tensor* input, const int rank) = 0;

  // Initialize `expected` output at `current_rank` out of `num_ranks` ranks.
  virtual void InitExpected(std::vector<float>* expected,
                            const int tensor_length, const int current_rank,
                            const int num_ranks) = 0;

  // Initialize device `di` specific to the collective op.
  virtual void InitDevice(DeviceInstance* di) = 0;

  // Run collective op on device `di`.
  virtual void RunCollectiveOnDevice(DeviceInstance* di) = 0;

  void RunCollective() {
    std::atomic<int> done(0);
    for (const auto& instance : instances_) {
      DeviceInstance* di = instance.get();
      InitDevice(di);
      SchedClosure([this, di, &done] {
        RunCollectiveOnDevice(di);
        ++done;
      });
    }
    while (done < static_cast<int>(instances_.size())) {
      Env::Default()->SleepForMicroseconds(1000);
    }
  }

  void RunTest(int num_ranks, int input_length) {
    Init(num_ranks);
    if (num_ranks > test_env_->device_mgr->NumDevices()) {
      LOG(WARNING) << "Skipping test because required " << num_ranks
                   << " GPUs but found " << test_env_->device_mgr->NumDevices();
      return;
    }
    for (int rank = 0; rank < num_ranks; ++rank) {
      instances_[rank]->InitTensor(
          DT_FLOAT, TensorShape({input_length}),
          [this, rank](Tensor* t) { InitInput(t, rank); });
    }
    RunCollective();
    // Check output.
    for (int rank = 0; rank < instances_.size(); ++rank) {
      std::vector<float> expected;
      InitExpected(&expected, input_length, rank, num_ranks);
      if (VLOG_IS_ON(3)) {
        string str_buf;
        for (const auto& x : expected) {
          strings::StrAppend(&str_buf, " ", x);
        }
        VLOG(3) << "Expected output " << str_buf;
      }

      TF_ASSERT_OK(instances_[rank]->status_);
      VLOG(2) << "rank " << rank << " output " << &instances_[rank]->output_
              << " buf " << DMAHelper::base(&instances_[rank]->output_);
      VLOG(3) << "rank " << rank << " got output tensor "
              << instances_[rank]->output_.DebugString(
                     instances_[rank]->output_.NumElements());
      test::ExpectTensorEqual<float>(test::AsTensor<float>(expected),
                                     instances_[rank]->output_);
    }
  }

  class DeviceInstance {
   public:
    DeviceInstance(int rank, const string& collective_name,
                   CollectiveType collective_type, CollectiveTestEnv* test_env)
        : test_env_(test_env) {  // TODO(tmorris): tensor_?
      col_params_ =
          CreateCollectiveParams(*test_env_, rank, collective_name,
                                 collective_type, DT_FLOAT, TensorShape());
      string device_name = col_params_->group.members[rank].device.name();
      TF_CHECK_OK(test_env_->device_mgr->LookupDevice(device_name, &device_))
          << "Could not find device " << device_name << " existing devices "
          << test_env_->device_mgr->DebugString();
      merge_op_ = GetAdd(device_);
      final_op_ = GetDiv(device_);
    }

    void InitTensor(DataType dtype, const TensorShape& shape,
                    const std::function<void(Tensor*)>& init_f) {
      input_ = Tensor(dtype, shape);
      init_f(&input_);
      if (VLOG_IS_ON(3)) {
        VLOG(3) << "input tensor " << input_.DebugString(shape.num_elements());
      } else {
        VLOG(2) << "input tensor " << input_.DebugString();
      }
    }

    void RunReduce() {
      output_ = input_;
      status_ = tensorflow::RunCollective(test_env_, col_params_.get(), device_,
                                          &input_, &output_);
    }

    void RunReduceScatter() {
      // Allocate output. We can't reuse the input because output has a
      // different shape.
      auto output_shape = input_.shape();
      output_shape.set_dim(
          0, output_shape.dim_size(0) / col_params_->group.group_size);
      output_ = Tensor(DT_FLOAT, output_shape);
      status_ = tensorflow::RunCollective(test_env_, col_params_.get(), device_,
                                          &input_, &output_);
    }

    void RunBroadcast() {
      output_ = input_;
      status_ = tensorflow::RunCollective(test_env_, col_params_.get(), device_,
                                          &input_, &output_);
    }

    void RunGather() {
      // Allocate output. We can't reuse the input because output has a
      // different shape.
      auto output_shape = input_.shape();
      output_shape.set_dim(
          0, output_shape.dim_size(0) * col_params_->group.group_size);
      output_ = Tensor(DT_FLOAT, output_shape);
      status_ = tensorflow::RunCollective(test_env_, col_params_.get(), device_,
                                          &input_, &output_);
    }

    void RunAllToAll() {
      // Allocate output. We can't reuse the input because NCCL does not support
      // in-place all-to-all.
      output_ = Tensor(DT_FLOAT, input_.shape());
      status_ = tensorflow::RunCollective(test_env_, col_params_.get(), device_,
                                          &input_, &output_);
    }

    CollectiveTestEnv* test_env_;
    Tensor input_;
    Tensor output_;
    Device* device_;
    core::RefCountPtr<CollectiveParams> col_params_;
    std::unique_ptr<OpKernel> merge_op_;
    std::unique_ptr<OpKernel> final_op_;
    Status status_;
  };

  CollectiveType collective_type_;
  const string collective_name_;
  std::vector<std::unique_ptr<DeviceInstance>> instances_;
  mutex mu_;
  int32 op_counter_ TF_GUARDED_BY(mu_) = 0;
  std::unique_ptr<CollectiveTestEnv> test_env_;
};

class NcclReducerTest : public NcclTestBase {
 protected:
  NcclReducerTest()
      : NcclTestBase(/*collective_type=*/REDUCTION_COLLECTIVE,
                     /*collective_name=*/"NcclReduce") {}
  ~NcclReducerTest() override = default;

  void InitInput(Tensor* input, const int rank) override {
    for (size_t i = 0; i < input->NumElements(); ++i) {
      float value = pow(10, rank) * i;
      input->flat<float>()(i) = value;
    }
  }

  void InitExpected(std::vector<float>* expected, const int tensor_length,
                    const int current_rank, const int num_ranks) override {
    expected->resize(tensor_length);
    for (int i = 0; i < tensor_length; ++i) {
      float expected_sum = 0.0;
      for (int rank = 0; rank < num_ranks; ++rank) {
        float value = pow(10, rank) * i;
        expected_sum += value;
      }
      (*expected)[i] = expected_sum / num_ranks;
    }
  }

  void InitDevice(DeviceInstance* di) override {
    di->col_params_->merge_op = di->merge_op_.get();
    di->col_params_->final_op = di->final_op_.get();
  }

  void RunCollectiveOnDevice(DeviceInstance* di) override { di->RunReduce(); }
};

class NcclReduceScattererTest : public NcclTestBase {
 protected:
  NcclReduceScattererTest()
      : NcclTestBase(/*collective_type=*/REDUCE_SCATTER_COLLECTIVE,
                     /*collective_name=*/"NcclReduceScatter") {}
  ~NcclReduceScattererTest() override = default;

  void InitInput(Tensor* input, const int rank) override {
    for (size_t i = 0; i < input->NumElements(); ++i) {
      float value = pow(10, rank) * i;
      input->flat<float>()(i) = value;
    }
  }

  void InitExpected(std::vector<float>* expected, const int tensor_length,
                    const int current_rank, const int num_ranks) override {
    const int output_length = tensor_length / num_ranks;
    expected->resize(output_length);
    for (int i = 0; i < output_length; ++i) {
      float expected_sum = 0.0;
      for (int rank = 0; rank < num_ranks; ++rank) {
        float value = pow(10, rank) * (i + current_rank * output_length);
        expected_sum += value;
      }
      (*expected)[i] = expected_sum / num_ranks;
    }
  }

  void InitDevice(DeviceInstance* di) override {
    di->col_params_->merge_op = di->merge_op_.get();
    di->col_params_->final_op = di->final_op_.get();
  }

  void RunCollectiveOnDevice(DeviceInstance* di) override {
    di->RunReduceScatter();
  }
};

class NcclBroadcasterTest : public NcclTestBase {
 protected:
  NcclBroadcasterTest()
      : NcclTestBase(/*collective_type=*/BROADCAST_COLLECTIVE,
                     /*collective_name=*/"NcclBroadcast") {}
  ~NcclBroadcasterTest() override = default;

  void InitInput(Tensor* input, const int rank) override {
    bool source = rank == source_rank_;
    for (size_t i = 0; i < input->NumElements(); ++i) {
      input->flat<float>()(i) = source ? static_cast<float>(i) : -1.0;
    }
  }

  void InitExpected(std::vector<float>* expected, const int tensor_length,
                    const int current_rank, const int num_ranks) override {
    expected->resize(tensor_length);
    for (int i = 0; i < tensor_length; ++i) {
      (*expected)[i] = i;
    }
  }

  void InitDevice(DeviceInstance* di) override {
    di->col_params_->source_rank = source_rank_;
    di->col_params_->is_source = di->col_params_->default_rank == source_rank_;
  }

  void RunCollectiveOnDevice(DeviceInstance* di) override {
    di->RunBroadcast();
  }

  int source_rank_ = 0;
};

class NcclGathererTest : public NcclTestBase {
 protected:
  NcclGathererTest()
      : NcclTestBase(/*collective_type=*/GATHER_COLLECTIVE,
                     /*collective_name=*/"NcclGather") {}
  ~NcclGathererTest() override = default;

  void InitInput(Tensor* input, const int rank) override {
    for (size_t i = 0; i < input->NumElements(); ++i) {
      float value = pow(10, rank) * i;
      input->flat<float>()(i) = value;
    }
  }

  void InitExpected(std::vector<float>* expected, const int tensor_length,
                    const int current_rank, const int num_ranks) override {
    expected->resize(tensor_length * num_ranks, -1);
    for (int rank = 0, i = 0; rank < num_ranks; ++rank) {
      for (int j = 0; j < tensor_length; ++j, ++i) {
        (*expected)[i] = pow(10, rank) * j;
      }
    }
  }

  void InitDevice(DeviceInstance* di) override {}

  void RunCollectiveOnDevice(DeviceInstance* di) override { di->RunGather(); }

  int source_rank_ = 0;
};

class NcclAllToAllTest : public NcclTestBase {
 protected:
  NcclAllToAllTest()
      : NcclTestBase(/*collective_type=*/ALL_TO_ALL_COLLECTIVE,
                     /*collective_name=*/"NcclAllToAll") {}
  ~NcclAllToAllTest() override = default;

  void InitInput(Tensor* input, const int rank) override {
    for (size_t i = 0; i < input->NumElements(); ++i) {
      float value = rank * input->NumElements() + i;
      input->flat<float>()(i) = value;
    }
  }

  void InitExpected(std::vector<float>* expected, const int tensor_length,
                    const int current_rank, const int num_ranks) override {
    expected->resize(tensor_length);
    // Each rank will have num_ranks parts of size part_size from each rank.
    const int part_size = tensor_length / num_ranks;
    for (int rank = 0, i = 0; rank < num_ranks; ++rank) {
      for (int j = 0; j < part_size; ++j, ++i) {
        const int part_index = current_rank + rank * num_ranks;
        (*expected)[i] = part_index * part_size + j;
      }
    }
  }

  void InitDevice(DeviceInstance* di) override {}

  void RunCollectiveOnDevice(DeviceInstance* di) override { di->RunAllToAll(); }

  int source_rank_ = 0;
};

TEST_F(NcclReducerTest, Test2Dev16Len) {
  RunTest(/*num_ranks=*/2, /*tensor_length=*/16);
}
TEST_F(NcclReducerTest, Test4Dev16Len) {
  RunTest(/*num_ranks=*/4, /*tensor_length=*/16);
}
TEST_F(NcclReducerTest, Test8Dev16Len) {
  RunTest(/*num_ranks=*/8, /*tensor_length=*/16);
}
TEST_F(NcclReducerTest, Test8Dev128Len) {
  RunTest(/*num_ranks=*/8, /*tensor_length=*/128);
}
TEST_F(NcclReducerTest, Test8Dev1048576Len) {
  RunTest(/*num_ranks=*/8, /*tensor_length=*/1048576);
}

TEST_F(NcclBroadcasterTest, Test2Dev16LenSrc0) {
  RunTest(/*num_ranks=*/2, /*tensor_length=*/16);
}
TEST_F(NcclBroadcasterTest, Test4Dev16LenSrc1) {
  source_rank_ = 1;
  RunTest(/*num_ranks=*/4, /*tensor_length=*/16);
}
TEST_F(NcclBroadcasterTest, Test8Dev16LenSrc7) {
  source_rank_ = 7;
  RunTest(/*num_ranks=*/8, /*tensor_length=*/16);
}
TEST_F(NcclBroadcasterTest, Test8Dev128LenSrc0) {
  RunTest(/*num_ranks=*/8, /*tensor_length=*/128);
}
TEST_F(NcclBroadcasterTest, Test8Dev1048576LenSrc0) {
  RunTest(/*num_ranks=*/8, /*tensor_length=*/1048576);
}

TEST_F(NcclGathererTest, Test2Dev16Len) {
  RunTest(/*num_ranks=*/2, /*tensor_length=*/16);
}
TEST_F(NcclGathererTest, Test4Dev16Len) {
  RunTest(/*num_ranks=*/4, /*tensor_length=*/16);
}
TEST_F(NcclGathererTest, Test8Dev16Len) {
  RunTest(/*num_ranks=*/8, /*tensor_length=*/16);
}
TEST_F(NcclGathererTest, Test8Dev128Len) {
  RunTest(/*num_ranks=*/8, /*tensor_length=*/128);
}
TEST_F(NcclGathererTest, Test8Dev1048576Len) {
  RunTest(/*num_ranks=*/8, /*tensor_length=*/1048576);
}

TEST_F(NcclReduceScattererTest, Test2Dev16Len) {
  RunTest(/*num_ranks=*/2, /*tensor_length=*/16);
}
TEST_F(NcclReduceScattererTest, Test4Dev16Len) {
  RunTest(/*num_ranks=*/4, /*tensor_length=*/16);
}
TEST_F(NcclReduceScattererTest, Test8Dev16Len) {
  RunTest(/*num_ranks=*/8, /*tensor_length=*/16);
}
TEST_F(NcclReduceScattererTest, Test8Dev128Len) {
  RunTest(/*num_ranks=*/8, /*tensor_length=*/128);
}
TEST_F(NcclReduceScattererTest, Test8Dev1048576Len) {
  RunTest(/*num_ranks=*/8, /*tensor_length=*/1048576);
}

TEST_F(NcclAllToAllTest, Test2Dev16Len) {
  RunTest(/*num_ranks=*/2, /*tensor_length=*/16);
}
TEST_F(NcclAllToAllTest, Test4Dev16Len) {
  RunTest(/*num_ranks=*/4, /*tensor_length=*/16);
}
TEST_F(NcclAllToAllTest, Test8Dev16Len) {
  RunTest(/*num_ranks=*/8, /*tensor_length=*/16);
}
TEST_F(NcclAllToAllTest, Test8Dev128Len) {
  RunTest(/*num_ranks=*/8, /*tensor_length=*/128);
}
TEST_F(NcclAllToAllTest, Test8Dev1048576Len) {
  RunTest(/*num_ranks=*/8, /*tensor_length=*/1048576);
}

}  // namespace tensorflow

#endif
