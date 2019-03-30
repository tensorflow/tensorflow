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

#ifdef GOOGLE_CUDA

#include "tensorflow/core/kernels/collective_nccl_reducer.h"

#include <algorithm>
#include "absl/memory/memory.h"
#include "tensorflow/core/common_runtime/base_collective_executor.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/device_resolver_local.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/common_runtime/test_collective_executor_mgr.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
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

class NcclReducerTest : public ::testing::Test {
 protected:
  ~NcclReducerTest() override {
    if (col_exec_) col_exec_->Unref();
  }

  void InitGPUDevices() {
    std::vector<std::unique_ptr<Device>> all_devices;
    SessionOptions session_options;
    session_options.config.mutable_gpu_options()
        ->set_per_process_gpu_memory_fraction(0.1);
    session_options.env = Env::Default();
    Status s = DeviceFactory::GetFactory(DEVICE_GPU)
                   ->AddDevices(session_options, "", &all_devices);
    TF_CHECK_OK(s);
    for (std::unique_ptr<Device>& d : all_devices) {
      if (d->device_type() == "GPU") {
        gpus_.emplace_back(std::move(d));
      }
    }
  }

  void Init(int num_ranks) {
    setenv("NCCL_DEBUG", "INFO", 1 /* replace */);
    setenv("NCCL_LAUNCH_MODE", "PARALLEL", 1 /* replace */);
    InitGPUDevices();
    std::vector<std::unique_ptr<Device>> local_devices;
    std::vector<string> device_names;
    for (int rank = 0; rank < num_ranks; ++rank) {
      if (rank < gpus_.size()) {
        local_devices.emplace_back(std::move(gpus_[rank]));
      }
    }
    int num_gpus = local_devices.size();
    for (const auto& device : local_devices) {
      device_names.push_back(device->name());
      VLOG(2) << device->name();
    }
    if (!dev_mgr_) dev_mgr_.reset(new DeviceMgr(std::move(local_devices)));
    col_exec_ = new BaseCollectiveExecutor(
        &col_exec_mgr_, /*remote_access=*/nullptr, kStepId, dev_mgr_.get(),
        /*gpu_ring_order=*/nullptr);

    // Initialize collective params.
    col_params_.name = "test_nccl_collective_op";
    const int group_key = 5;
    col_params_.group.group_key = group_key;
    col_params_.group.device_type = DEVICE_GPU;
    col_params_.group.group_size = num_ranks;
    const int instance_key = 23;
    col_params_.instance.instance_key = instance_key;
    col_params_.instance.type = REDUCTION_COLLECTIVE;
    col_params_.instance.data_type = DT_FLOAT;
    col_params_.instance.impl_details.collective_name = "NcclReduce";
    const string task_name = "/job:worker/replica:0/task:0";
    col_params_.instance.num_devices_per_task[task_name] = num_ranks;
    for (int rank = 0; rank < num_ranks; ++rank) {
      col_params_.instance.device_names.push_back(
          device_names[rank % num_gpus]);
      col_params_.instance.task_names.push_back(task_name);
    }
    for (int rank = 0; rank < num_ranks; ++rank) {
      instances_.push_back(absl::make_unique<DeviceInstance>(
          rank, col_params_.instance.device_names[rank], this));
    }
  }

  void Reduce() {
    int done = 0;
    mutex done_mu;
    condition_variable done_cv;
    for (const auto& instance : instances_) {
      DeviceInstance* di = instance.get();
      SchedClosure([di, &done, &done_mu, &done_cv] {
        di->DoReduce();
        mutex_lock l(done_mu);
        ++done;
        done_cv.notify_all();
      });
    }

    mutex_lock l(done_mu);
    while (done < instances_.size()) done_cv.wait(l);
  }

  void RunTest(int num_ranks, int tensor_length) {
    Init(num_ranks);
    std::vector<float> expected(tensor_length, 0.0);
    for (int rank = 0; rank < num_ranks; ++rank) {
      DeviceInstance* instance = instances_[rank].get();
      instance->InitTensor(DT_FLOAT, TensorShape({tensor_length}),
                           [&expected, rank](Tensor* t) {
                             for (size_t i = 0; i < t->NumElements(); ++i) {
                               float value = pow(10, rank) * i;
                               t->flat<float>()(i) = value;
                               expected[i] += value;
                             }
                           });
    }
    Reduce();
    // Confirm that every rank computed the same correct value.
    for (int i = 0; i < tensor_length; ++i) {
      expected[i] /= num_ranks;
    }
    for (int rank = 0; rank < instances_.size(); ++rank) {
      TF_ASSERT_OK(instances_[rank]->status_);
      Tensor* dev_tensor = &instances_[rank]->tensor_;
      Tensor actual(DT_FLOAT, TensorShape({tensor_length}));
      Notification note;
      Device* dev = instances_[rank]->device_;
      auto* dev_info = dev->tensorflow_gpu_device_info();
      dev_info->default_context->CopyDeviceTensorToCPU(
          dev_tensor, /*tensor_name=*/"", dev, &actual,
          [&note](const Status&) { note.Notify(); });
      note.WaitForNotification();
      for (int i = 0; i < tensor_length; ++i) {
        EXPECT_FLOAT_EQ(expected[i], actual.template flat<float>()(i))
            << "Mismatch at rank " << rank << " index " << i;
      }
    }
  }

  std::unique_ptr<OpKernel> GetCollectiveReduce(const CollectiveParams& params,
                                                Tensor* input,
                                                DeviceBase* device) {
    mutex_lock l(mu_);
    NodeDef node_def;
    NodeDefBuilder builder(
        strings::StrCat("collective_reduce_", reduce_counter_++),
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
    DeviceInstance(int rank, const string& device_name, NcclReducerTest* parent)
        : parent_(parent), device_name_(device_name), rank_(rank) {
      TF_CHECK_OK(parent_->dev_mgr_->LookupDevice(device_name_, &device_))
          << "Could not find device " << device_name_ << " existing devices "
          << parent_->dev_mgr_->DebugString();
      col_params_.name = parent_->col_params_.name;
      col_params_.default_rank = rank;
      col_params_.group.group_key = parent_->col_params_.group.group_key;
      col_params_.group.device_type = parent_->col_params_.group.device_type;
      col_params_.group.group_size = parent_->col_params_.group.group_size;
      col_params_.instance = parent->col_params_.instance;
    }

    void InitTensor(DataType dtype, const TensorShape& shape,
                    const std::function<void(Tensor*)>& init_f) {
      tensor_ =
          Tensor(device_->GetAllocator(AllocatorAttributes()), dtype, shape);
      Tensor cpu_tensor(dtype, shape);
      init_f(&cpu_tensor);
      VLOG(2) << "cpu_tensor " << cpu_tensor.DebugString();
      auto* dev_info = device_->tensorflow_gpu_device_info();
      Notification note;
      dev_info->default_context->CopyCPUTensorToDevice(
          &cpu_tensor, device_, &tensor_,
          [&note](const Status&) { note.Notify(); });
      note.WaitForNotification();
    }

    void DoReduce() {
      col_params_.merge_op = GetAdd(device_);
      col_params_.final_op = GetDiv(device_);

      // Prepare an OpKernelContext.
      OpKernelContext::Params op_params;
      op_params.step_id = kStepId;
      op_params.device = device_;
      gtl::InlinedVector<TensorValue, 4> inputs;
      inputs.push_back(TensorValue(&tensor_));
      op_params.inputs = &inputs;
      gtl::InlinedVector<AllocatorAttributes, 4> input_aa(
          {AllocatorAttributes()});
      op_params.input_alloc_attrs = &input_aa;
      gtl::InlinedVector<DeviceContext*, 4> input_dc;
      DeviceContext* dev_ctx = nullptr;
      auto* dev_info = device_->tensorflow_gpu_device_info();
      if (dev_info) {
        dev_ctx = dev_info->default_context;
        dev_ctx->Ref();
      } else {
        dev_ctx = new DeviceContext;
      }
      input_dc.push_back(dev_ctx);
      op_params.input_device_contexts = &input_dc;
      op_params.op_device_context = dev_ctx;
      int forward_from = 0;
      op_params.forward_from_array = &forward_from;
      AllocatorAttributes generic_alloc_attr;
      op_params.output_attr_array = &generic_alloc_attr;
      std::unique_ptr<OpKernel> op =
          parent_->GetCollectiveReduce(col_params_, &tensor_, device_);
      op_params.op_kernel = op.get();
      OpKernelContext ctx(&op_params, 1);

      // We never actually execute the kernel, so we need to do the output
      // allocation it would do, ourselves.
      Tensor* output_tensor_ptr = nullptr;
      TF_CHECK_OK(ctx.forward_input_or_allocate_output({0}, 0, tensor_.shape(),
                                                       &output_tensor_ptr));
      CHECK_EQ(output_tensor_ptr, ctx.mutable_output(0));

      // Prepare a NcclReducer instance.
      string exec_key =
          strings::StrCat(col_params_.instance.instance_key, ":0:0");
      NcclReducer reducer;
      CollectiveContext col_ctx(parent_->col_exec_, parent_->dev_mgr_.get(),
                                &ctx, &op_params, col_params_, exec_key,
                                kStepId, &tensor_, &tensor_);
      TF_CHECK_OK(reducer.InitializeCollectiveContext(&col_ctx));

      // Run the all-reduce.
      reducer.Run([this](Status s) { status_ = s; });
      if (status_.ok()) {
        CHECK(tensor_.CopyFrom(*ctx.mutable_output(0), tensor_.shape()));
      }

      dev_ctx->Unref();
    }

    NcclReducerTest* parent_;
    string device_name_;
    int rank_;
    Tensor tensor_;
    Device* device_;
    CollectiveParams col_params_;
    Status status_;
  };

  std::vector<std::unique_ptr<tensorflow::Device>> gpus_;
  TestCollectiveExecutorMgr col_exec_mgr_;
  CollectiveExecutor* col_exec_;
  std::unique_ptr<DeviceMgr> dev_mgr_;
  std::vector<std::unique_ptr<DeviceInstance>> instances_;
  CollectiveParams col_params_;
  mutex mu_;
  int32 reduce_counter_ GUARDED_BY(mu_) = 0;
};

TEST_F(NcclReducerTest, Test2Dev16Len) { RunTest(2, 16); }
TEST_F(NcclReducerTest, Test4Dev16Len) { RunTest(4, 16); }
TEST_F(NcclReducerTest, Test8Dev16Len) { RunTest(8, 16); }
TEST_F(NcclReducerTest, Test8Dev128Len) { RunTest(8, 128); }
TEST_F(NcclReducerTest, Test8Dev1045991Len) { RunTest(8, 1048576); }

}  // namespace tensorflow

#endif
