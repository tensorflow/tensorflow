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

#include "tensorflow/core/framework/device_attributes.pb.h"
#ifdef GOOGLE_CUDA

#include "tensorflow/core/kernels/collective_nccl.h"

#include <algorithm>

#include "absl/memory/memory.h"
#include "tensorflow/core/common_runtime/base_collective_executor.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
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
#include "tensorflow/core/kernels/collective_nccl_broadcaster.h"
#include "tensorflow/core/kernels/collective_nccl_gatherer.h"
#include "tensorflow/core/kernels/collective_nccl_reducer.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/nccl/collective_communicator.h"
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
      : collective_type_(collective_type),
        collective_name_(collective_name),
        nccl_communicator_(MaybeCreateNcclCommunicator(config_proto_)),
        work_queue_(std::make_shared<UnboundedWorkQueue>(
            Env::Default(), "collective_executor")),
        col_exec_(nullptr),
        col_params_(nullptr) {}

  ~NcclTestBase() override {
    if (col_exec_) col_exec_->Unref();
    if (col_params_) col_params_->Unref();
  }

  void SetUp() {
    std::vector<std::unique_ptr<Device>> all_devices;
    TF_CHECK_OK(DeviceFactory::GetFactory(DEVICE_GPU)
                    ->AddDevices(SessionOptions(), "", &all_devices));
    for (std::unique_ptr<Device>& d : all_devices) {
      if (d->device_type() == "GPU") {
        gpus_.emplace_back(std::move(d));
      }
    }
  }

  void Init(const int num_ranks, const int instance_key) {
    setenv("NCCL_DEBUG", "INFO", 1 /* replace */);
    setenv("NCCL_LAUNCH_MODE", "PARALLEL", 1 /* replace */);
    std::vector<std::unique_ptr<Device>> local_devices;
    std::vector<string> device_names;
    CHECK_LE(num_ranks, gpus_.size());
    for (int rank = 0; rank < num_ranks; ++rank) {
      local_devices.emplace_back(std::move(gpus_[rank]));
    }
    int num_gpus = local_devices.size();
    for (const auto& device : local_devices) {
      device_names.push_back(device->name());
      VLOG(2) << device->name();
    }
    if (!dev_mgr_)
      dev_mgr_ = std::make_unique<StaticDeviceMgr>(std::move(local_devices));
    col_exec_ =
        new BaseCollectiveExecutor(&col_exec_mgr_, /*remote_access=*/nullptr,
                                   kStepId, dev_mgr_.get(), work_queue_);

    // Initialize collective params.
    col_params_ = new CollectiveParams();
    col_params_->name = "test_nccl_collective_op";
    const int group_key = num_ranks;
    col_params_->group.group_key = group_key;
    col_params_->group.device_type = DEVICE_GPU;
    col_params_->group.group_size = num_ranks;
    col_params_->instance.instance_key = instance_key;
    col_params_->instance.type = collective_type_;
    col_params_->instance.data_type = DT_FLOAT;
    col_params_->instance.impl_details.collective_name = collective_name_;
    const string task_name = "/job:worker/replica:0/task:0";
    col_params_->group.num_devices_per_task[task_name] = num_ranks;
    for (int rank = 0; rank < num_ranks; ++rank) {
      CollGroupMember member;
      member.device.set_name(device_names[rank % num_gpus]);
      col_params_->group.members.push_back(member);
    }
    for (int rank = 0; rank < num_ranks; ++rank) {
      instances_.push_back(std::make_unique<DeviceInstance>(
          rank, col_params_->group.members[rank].device.name(), this));
    }
  }

  // Initialize `input` tensor at rank `rank`.
  virtual void InitInput(Tensor* input, const int rank) = 0;

  // Initialize `expected` output at all `num_ranks` ranks.
  virtual void InitExpected(std::vector<float>* expected,
                            const int tensor_length, const int num_ranks) = 0;

  // Initialize device `di` specific to the collective op.
  virtual void InitDevice(DeviceInstance* di) = 0;

  // Run collective op on device `di`.
  virtual void RunCollectiveOnDevice(DeviceInstance* di) = 0;

  void RunCollective() {
    int done = 0;
    mutex done_mu;
    condition_variable done_cv;
    for (const auto& instance : instances_) {
      DeviceInstance* di = instance.get();
      InitDevice(di);
      SchedClosure([this, di, &done, &done_mu, &done_cv] {
        RunCollectiveOnDevice(di);
        mutex_lock l(done_mu);
        ++done;
        done_cv.notify_all();
      });
    }

    mutex_lock l(done_mu);
    while (done < instances_.size()) done_cv.wait(l);
  }

  void RunTest(int num_ranks, int input_length, int instance_key) {
    if (num_ranks > gpus_.size()) {
      LOG(WARNING) << "Skipping test because required " << num_ranks
                   << " GPUs but found " << gpus_.size();
      return;
    }
    Init(num_ranks, instance_key);
    std::vector<float> expected;
    InitExpected(&expected, input_length, num_ranks);
    if (VLOG_IS_ON(3)) {
      string str_buf;
      for (const auto& x : expected) {
        strings::StrAppend(&str_buf, " ", x);
      }
      VLOG(3) << "Expected output " << str_buf;
    }
    for (int rank = 0; rank < num_ranks; ++rank) {
      DeviceInstance* instance = instances_[rank].get();
      instance->InitTensor(DT_FLOAT, TensorShape({input_length}),
                           [this, rank](Tensor* t) { InitInput(t, rank); });
    }
    RunCollective();
    // Confirm that every rank computed the same correct value.
    for (int rank = 0; rank < instances_.size(); ++rank) {
      TF_ASSERT_OK(instances_[rank]->status_);
      Tensor* output = &instances_[rank]->output_;
      const int output_length = output->NumElements();
      VLOG(2) << "rank " << rank << " output " << output << " buf "
              << DMAHelper::base(output);
      Tensor actual(DT_FLOAT, TensorShape({output_length}));
      Device* dev = instances_[rank]->device_;
      auto* dev_info = dev->tensorflow_accelerator_device_info();
      TF_CHECK_OK(dev_info->default_context->CopyDeviceTensorToCPUSync(
          output, /*tensor_name=*/"", dev, &actual));
      VLOG(3) << "rank " << rank << " got output tensor "
              << actual.DebugString(output_length);
      for (int i = 0; i < output_length; ++i) {
        EXPECT_FLOAT_EQ(expected[i], actual.template flat<float>()(i))
            << "Mismatch at rank " << rank << " index " << i;
      }
    }
  }

  std::unique_ptr<OpKernel> GetCollectiveReduceOpKernel(
      const CollectiveParams& params, Tensor* input, DeviceBase* device) {
    mutex_lock l(mu_);
    NodeDef node_def;
    NodeDefBuilder builder(strings::StrCat("collective_reduce_", op_counter_++),
                           "CollectiveReduce");
    TF_CHECK_OK(
        builder.Attr("T", params.instance.data_type)
            .Attr("merge_op", "Add")
            .Attr("final_op", "Div")
            .Attr("group_size", params.group.group_size)
            .Attr("group_key", params.group.group_key)
            .Attr("instance_key", params.instance.instance_key)
            .Attr("subdiv_offsets", params.instance.impl_details.subdiv_offsets)
            .Input(FakeInput(params.instance.data_type))
            .Finalize(&node_def));
    return GetKernel(node_def, device);
  }

  class DeviceInstance {
   public:
    DeviceInstance(int rank, const string& device_name, NcclTestBase* parent)
        : parent_(parent),
          device_name_(device_name),
          rank_(rank),
          col_params_(new CollectiveParams()) {
      TF_CHECK_OK(parent_->dev_mgr_->LookupDevice(device_name_, &device_))
          << "Could not find device " << device_name_ << " existing devices "
          << parent_->dev_mgr_->DebugString();
      merge_op_ = GetAdd(device_);
      final_op_ = GetDiv(device_);
      col_params_->name = parent_->col_params_->name;
      col_params_->default_rank = rank;
      col_params_->group = parent_->col_params_->group;
      col_params_->instance = parent->col_params_->instance;
    }

    ~DeviceInstance() { col_params_->Unref(); }

    void InitTensor(DataType dtype, const TensorShape& shape,
                    const std::function<void(Tensor*)>& init_f) {
      input_ =
          Tensor(device_->GetAllocator(AllocatorAttributes()), dtype, shape);
      Tensor cpu_tensor(dtype, shape);
      init_f(&cpu_tensor);
      if (VLOG_IS_ON(3)) {
        VLOG(3) << "input tensor "
                << cpu_tensor.DebugString(shape.num_elements());
      } else {
        VLOG(2) << "input tensor " << cpu_tensor.DebugString();
      }
      auto* dev_info = device_->tensorflow_accelerator_device_info();
      TF_CHECK_OK(dev_info->default_context->CopyCPUTensorToDeviceSync(
          &cpu_tensor, device_, &input_));
    }

    void PrepareDeviceContext(OpKernelContext::Params* params) {
      params->step_id = kStepId;
      params->device = device_;
      DeviceContext* dev_ctx = nullptr;
      auto* dev_info = device_->tensorflow_accelerator_device_info();
      if (dev_info) {
        dev_ctx = dev_info->default_context;
        dev_ctx->Ref();
      } else {
        dev_ctx = new DeviceContext;
      }
      params->op_device_context = dev_ctx;
    }

    void RunReduce() {
      // Prepare an OpKernelContext.
      OpKernelContext::Params op_params;
      PrepareDeviceContext(&op_params);

      // Prepare inputs and outputs to OpKernel.
      gtl::InlinedVector<TensorValue, 4> inputs;
      inputs.push_back(TensorValue(&input_));
      op_params.inputs = inputs;
      gtl::InlinedVector<AllocatorAttributes, 4> input_aa(
          {AllocatorAttributes()});
      op_params.input_alloc_attrs = input_aa;
      int forward_from = 0;
      op_params.forward_from_array = &forward_from;
      AllocatorAttributes generic_alloc_attr;
      op_params.output_attr_array = &generic_alloc_attr;
      std::unique_ptr<OpKernel> op =
          parent_->GetCollectiveReduceOpKernel(*col_params_, &input_, device_);
      op_params.op_kernel = op.get();
      OpKernelContext ctx(&op_params, 1);
      // We never actually execute the kernel, so we need to do the output
      // allocation it would do, ourselves.
      Tensor* output_tensor_ptr = nullptr;
      TF_CHECK_OK(ctx.forward_input_or_allocate_output({0}, 0, input_.shape(),
                                                       &output_tensor_ptr));
      CHECK_EQ(output_tensor_ptr, ctx.mutable_output(0));

      // Run the all-reduce.
      string exec_key =
          strings::StrCat(col_params_->instance.instance_key, ":0:0");
      auto* reducer = new NcclReducer();
      auto col_ctx = std::make_shared<CollectiveContext>(
          parent_->col_exec_, parent_->nccl_communicator_.get(),
          parent_->dev_mgr_.get(),
          /*OpKernelContext=*/&ctx, &op_params, col_params_, exec_key, kStepId,
          /*input=*/&input_, /*output=*/&input_);
      TF_CHECK_OK(reducer->InitializeCollectiveContext(col_ctx));
      Notification note;
      reducer->Run([this, &note](Status s) {
        status_ = s;
        note.Notify();
      });
      note.WaitForNotification();
      if (status_.ok()) {
        CHECK(output_.CopyFrom(*ctx.mutable_output(0), input_.shape()));
      }

      reducer->Unref();
      op_params.op_device_context->Unref();
    }

    void RunBroadcast() {
      VLOG(2) << "RunBroadcast name " << parent_->collective_name_ << " rank "
              << col_params_->default_rank;
      // Prepare an OpKernelContext.
      OpKernelContext::Params op_params;
      PrepareDeviceContext(&op_params);
      OpKernelContext ctx(&op_params, 1);

      // Run broadcast.
      string exec_key =
          strings::StrCat(col_params_->instance.instance_key, ":0:0");
      auto* broadcaster = new NcclBroadcaster();
      auto col_ctx = std::make_shared<CollectiveContext>(
          parent_->col_exec_, parent_->nccl_communicator_.get(),
          parent_->dev_mgr_.get(),
          /*OpKernelContext=*/&ctx, &op_params, col_params_, exec_key, kStepId,
          /*input=*/col_params_->is_source ? &input_ : nullptr,
          /*output=*/&input_);
      TF_CHECK_OK(broadcaster->InitializeCollectiveContext(col_ctx));
      Notification note;
      broadcaster->Run([this, &note](Status s) {
        status_ = s;
        note.Notify();
      });
      note.WaitForNotification();
      if (status_.ok()) {
        CHECK(output_.CopyFrom(input_, input_.shape()));
      }

      broadcaster->Unref();
      op_params.op_device_context->Unref();
    }

    void RunGather() {
      VLOG(2) << "RunGather name " << parent_->collective_name_ << " rank "
              << col_params_->default_rank;
      // Prepare an OpKernelContext.
      OpKernelContext::Params op_params;
      PrepareDeviceContext(&op_params);
      OpKernelContext ctx(&op_params, 1);

      // Allocate output.  We can't reuse the input because output has a
      // different shape.
      auto output_shape = input_.shape();
      output_shape.set_dim(
          0, output_shape.dim_size(0) * col_params_->group.group_size);
      output_ = Tensor(device_->GetAllocator(AllocatorAttributes()), DT_FLOAT,
                       output_shape);

      // Run gather.
      string exec_key =
          strings::StrCat(col_params_->instance.instance_key, ":0:0");
      auto* gatherer = new NcclGatherer();
      auto col_ctx = std::make_shared<CollectiveContext>(
          parent_->col_exec_, parent_->nccl_communicator_.get(),
          parent_->dev_mgr_.get(),
          /*OpKernelContext=*/&ctx, &op_params, col_params_, exec_key, kStepId,
          /*input=*/&input_,
          /*output=*/&output_);
      TF_CHECK_OK(gatherer->InitializeCollectiveContext(col_ctx));
      Notification note;
      gatherer->Run([this, &note](Status s) {
        status_ = s;
        note.Notify();
      });
      note.WaitForNotification();

      gatherer->Unref();
      op_params.op_device_context->Unref();
    }

    NcclTestBase* parent_;
    string device_name_;
    int rank_;
    Tensor input_;
    Tensor output_;
    Device* device_;
    CollectiveParams* col_params_;
    std::unique_ptr<OpKernel> merge_op_;
    std::unique_ptr<OpKernel> final_op_;
    Status status_;
  };

  CollectiveType collective_type_;
  const string collective_name_;
  std::vector<std::unique_ptr<tensorflow::Device>> gpus_;
  TestCollectiveExecutorMgr col_exec_mgr_;
  ConfigProto config_proto_;
  std::unique_ptr<NcclCommunicatorInterface> nccl_communicator_;
  std::shared_ptr<UnboundedWorkQueue> work_queue_;
  CollectiveExecutor* col_exec_;
  std::unique_ptr<DeviceMgr> dev_mgr_;
  std::vector<std::unique_ptr<DeviceInstance>> instances_;
  CollectiveParams* col_params_;
  mutex mu_;
  int32 op_counter_ TF_GUARDED_BY(mu_) = 0;
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
                    const int num_ranks) override {
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
                    const int num_ranks) override {
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
                    const int num_ranks) override {
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

TEST_F(NcclReducerTest, Test2Dev16Len) {
  RunTest(/*num_ranks=*/2, /*tensor_length=*/16, /*instance_key=*/23);
}
TEST_F(NcclReducerTest, Test4Dev16Len) {
  RunTest(/*num_ranks=*/4, /*tensor_length=*/16, /*instance_key=*/23);
}
TEST_F(NcclReducerTest, Test8Dev16Len) {
  RunTest(/*num_ranks=*/8, /*tensor_length=*/16, /*instance_key=*/23);
}
TEST_F(NcclReducerTest, Test8Dev128Len) {
  RunTest(/*num_ranks=*/8, /*tensor_length=*/128, /*instance_key=*/23);
}
TEST_F(NcclReducerTest, Test8Dev1045991Len) {
  RunTest(/*num_ranks=*/8, /*tensor_length=*/1048576, /*instance_key=*/23);
}

TEST_F(NcclBroadcasterTest, Test2Dev16LenSrc0) {
  RunTest(/*num_ranks=*/2, /*tensor_length=*/16, /*instance_key=*/23);
}
TEST_F(NcclBroadcasterTest, Test4Dev16LenSrc1) {
  source_rank_ = 1;
  RunTest(/*num_ranks=*/4, /*tensor_length=*/16, /*instance_key=*/23);
}
TEST_F(NcclBroadcasterTest, Test8Dev16LenSrc7) {
  source_rank_ = 7;
  RunTest(/*num_ranks=*/8, /*tensor_length=*/16, /*instance_key=*/23);
}
TEST_F(NcclBroadcasterTest, Test8Dev128LenSrc0) {
  RunTest(/*num_ranks=*/8, /*tensor_length=*/128, /*instance_key=*/24);
}
TEST_F(NcclBroadcasterTest, Test8Dev1045991LenSrc0) {
  RunTest(/*num_ranks=*/8, /*tensor_length=*/1048576, /*instance_key=*/23);
}

TEST_F(NcclGathererTest, Test2Dev16Len) {
  RunTest(/*num_ranks=*/2, /*tensor_length=*/16, /*instance_key=*/23);
}
TEST_F(NcclGathererTest, Test4Dev16Len) {
  RunTest(/*num_ranks=*/4, /*tensor_length=*/16, /*instance_key=*/23);
}
TEST_F(NcclGathererTest, Test8Dev16Len) {
  RunTest(/*num_ranks=*/8, /*tensor_length=*/16, /*instance_key=*/23);
}
TEST_F(NcclGathererTest, Test8Dev128Len) {
  RunTest(/*num_ranks=*/8, /*tensor_length=*/128, /*instance_key=*/24);
}
TEST_F(NcclGathererTest, Test8Dev1045991Len) {
  RunTest(/*num_ranks=*/8, /*tensor_length=*/1048576, /*instance_key=*/23);
}

}  // namespace tensorflow

#endif
