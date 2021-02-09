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
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/unbounded_work_queue.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

// Wraps CollectiveRemoteAccessLocal with the ability to return an
// error status to the N'th action.
class FailTestRMA : public CollectiveRemoteAccessLocal {
 public:
  FailTestRMA(const DeviceMgr* dev_mgr, DeviceResolverInterface* dev_resolver,
              int64 step_id, int fail_after)
      : CollectiveRemoteAccessLocal(dev_mgr, dev_resolver, step_id),
        fail_after_(fail_after) {}

  bool MaybeFail(const StatusCallback& done) {
    bool fail_now = false;
    {
      mutex_lock l(mu_);
      if (fail_after_ > 0) {
        fail_now = (--fail_after_ == 0);
      }
    }
    if (fail_now) {
      done(errors::Internal("Deliberate failure"));
      return true;
    }
    return false;
  }

  void RecvFromPeer(const string& peer_device, const string& peer_task,
                    bool peer_is_local, const string& key, Device* to_device,
                    DeviceContext* to_device_ctx,
                    const AllocatorAttributes& to_alloc_attr, Tensor* to_tensor,
                    const DeviceLocality& client_locality,
                    int dev_to_dev_stream_index,
                    CancellationManager* cancellation_manager,
                    const StatusCallback& done) override {
    if (MaybeFail(done)) return;
    CollectiveRemoteAccessLocal::RecvFromPeer(
        peer_device, peer_task, peer_is_local, key, to_device, to_device_ctx,
        to_alloc_attr, to_tensor, client_locality, dev_to_dev_stream_index,
        cancellation_manager, done);
  }

  void PostToPeer(const string& peer_device, const string& peer_task,
                  const string& key, Device* from_device,
                  DeviceContext* from_device_ctx,
                  const AllocatorAttributes& from_alloc_attr,
                  const Tensor* from_tensor,
                  const DeviceLocality& client_locality,
                  CancellationManager* cancellation_manager,
                  const StatusCallback& done) override {
    if (MaybeFail(done)) return;
    CollectiveRemoteAccessLocal::PostToPeer(
        peer_device, peer_task, key, from_device, from_device_ctx,
        from_alloc_attr, from_tensor, client_locality, cancellation_manager,
        done);
  }

  mutex mu_;
  int fail_after_ TF_GUARDED_BY(mu_);
};

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

static int64 kStepId = 123;

class RingReducerTest : public ::testing::Test {
 protected:
  RingReducerTest()
      : device_type_(DEVICE_CPU), col_exec_(nullptr), col_params_(nullptr) {}

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  void InitGPUDevices() {
    auto device_factory = DeviceFactory::GetFactory("GPU");
    CHECK(device_factory);
    SessionOptions options;
    Status s = device_factory->CreateDevices(
        options, "/job:worker/replica:0/task:0", &gpu_devices_);
    CHECK(s.ok());
  }
#endif

  ~RingReducerTest() override {
    stop_ = true;
    for (auto i : instances_) delete i;
    if (col_exec_) col_exec_->Unref();
    if (col_params_) col_params_->Unref();
  }

  void Init(int num_workers, int num_devices, DataType dtype,
            const DeviceType& device_type, int num_subdivs, int fail_after) {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    InitGPUDevices();
#endif
    device_type_ = device_type;
    std::vector<std::unique_ptr<Device>> local_devices;
    SessionOptions sess_opts;
    sess_opts.env = Env::Default();
    Bytes mem_limit(4 << 20);
    DeviceLocality dev_locality;
    for (int wi = 0; wi < num_workers; ++wi) {
      for (int di = 0; di < num_devices; ++di) {
        if (device_type == DEVICE_CPU) {
          string dev_name =
              strings::StrCat("/job:worker/replica:0/task:", wi, "/cpu:", di);
          local_devices.push_back(absl::make_unique<ThreadPoolDevice>(
              sess_opts, dev_name, mem_limit, dev_locality, cpu_allocator()));
        } else if (device_type == DEVICE_GPU && !gpu_devices_.empty()) {
          int dev_idx = (wi * num_devices) + di;
          if (dev_idx >= static_cast<int>(gpu_devices_.size())) {
            LOG(INFO) << "dev_mgr has access to limited GPUs, reusing for more "
                         "than one ring node.";
          } else {
            local_devices.push_back(std::move(gpu_devices_[dev_idx]));
          }
        } else {
          LOG(FATAL) << "Unsupported device_type " << device_type;
        }
      }
    }
    if (!dev_mgr_ || device_type == DEVICE_CPU) {
      LOG(INFO) << "resetting dev_mgr for " << local_devices.size()
                << " devices: ";
      dev_mgr_ = absl::make_unique<StaticDeviceMgr>(std::move(local_devices));
    }
    if (!gpu_ring_order_) {
      gpu_ring_order_ = absl::make_unique<string>();
    }
    dev_resolver_ = absl::make_unique<DeviceResolverLocal>(dev_mgr_.get());
    work_queue_ = std::make_shared<UnboundedWorkQueue>(Env::Default(), "test");
    rma_ = new FailTestRMA(dev_mgr_.get(), dev_resolver_.get(), kStepId,
                           fail_after);
    col_exec_ = new BaseCollectiveExecutor(&col_exec_mgr_, rma_, kStepId,
                                           dev_mgr_.get(),
                                           gpu_ring_order_.get(), work_queue_);
    col_params_ = new CollectiveParams();
    col_params_->name = "test_collective";
    static const int kGroupKey = 5;
    col_params_->group.group_key = kGroupKey;
    col_params_->group.device_type = device_type;
    col_params_->group.group_size = num_workers * num_devices;
    static const int kInstanceKey = 17;
    col_params_->instance.instance_key = kInstanceKey;
    col_params_->instance.impl_details.subdiv_offsets.clear();
    col_params_->instance.type = REDUCTION_COLLECTIVE;
    col_params_->instance.impl_details.collective_name = "RingReduce";
    col_params_->instance.data_type = dtype;
    col_params_->instance.impl_details.subdiv_permutations.resize(num_subdivs);
    col_params_->subdiv_rank.resize(num_subdivs);
    int subdiv_stride = num_devices / num_subdivs;
    for (int sdi = 0; sdi < num_subdivs; ++sdi) {
      col_params_->instance.impl_details.subdiv_offsets.push_back(
          sdi * subdiv_stride);
      col_params_->subdiv_rank[sdi] = sdi * subdiv_stride;
    }

    // Set up a local device ring order that's not just 0,1,2...
    std::vector<int> local_ring_order;
    for (int di = 0; di < num_devices; ++di) {
      local_ring_order.push_back(di);
    }
    for (int di = 0; di < num_devices; ++di) {
      bool is_odd = ((di % 2) == 1);
      int other = (di + (is_odd ? 7 : 3)) % num_devices;
      if (di == other) continue;
      iter_swap(local_ring_order.begin() + di,
                local_ring_order.begin() + other);
    }
    string lro_buf;
    for (auto d : local_ring_order) strings::StrAppend(&lro_buf, d, ", ");
    VLOG(1) << "local_ring_order " << lro_buf;

    // Set up all of the fake device contexts.
    for (int wi = 0; wi < num_workers; ++wi) {
      string task_name = strings::StrCat("/job:worker/replica:0/task:", wi);
      col_params_->group.num_devices_per_task[task_name] = num_devices;
      for (int di = 0; di < num_devices; ++di) {
        string dev_name = strings::StrCat(task_name, "/cpu:", di);
        if (device_type == DEVICE_GPU) {
          dev_name =
              strings::StrCat(task_name, "/gpu:", di % gpu_devices_.size());
        }
        col_params_->group.device_names.push_back(dev_name);
        col_params_->group.task_names.push_back(task_name);
        // Normally each device would set is_local to its own perspective but
        // this test runs in a single process so is_local is always true.
        col_params_->task.is_local.push_back(true);
        for (int sdi = 0; sdi < num_subdivs; ++sdi) {
          int rotated_di =
              (di + col_params_->instance.impl_details.subdiv_offsets[sdi]) %
              num_devices;
          col_params_->instance.impl_details.subdiv_permutations[sdi].push_back(
              wi * num_devices + local_ring_order[rotated_di]);
        }
      }
    }
    for (int wi = 0; wi < num_workers; ++wi) {
      for (int di = 0; di < num_devices; ++di) {
        int rank = wi * num_devices + di;
        instances_.push_back(new DeviceInstance(
            rank, col_params_->group.device_names[rank], device_type_, this));
      }
    }
  }

  void Reduce(int fail_after) {
    std::atomic<int> done(0);
    for (auto di : instances_) {
      SchedClosure([di, &done] {
        di->DoReduce();
        ++done;
      });
      if (fail_after > 0) {
        // Stagger the op execution starts.
        Env::Default()->SleepForMicroseconds(100);
      }
    }
    while (done < static_cast<int>(instances_.size())) {
      if (stop_) break;
      Env::Default()->SleepForMicroseconds(1000);
    }
  }

  template <typename T>
  void RunTest(DataType dtype, const DeviceType& device_type, int num_workers,
               int num_devices, int num_subdivs, int tensor_len,
               int fail_after) {
    Init(num_workers, num_devices, dtype, device_type, num_subdivs, fail_after);
    std::vector<T> expected(tensor_len, 0.0);
    for (int di = 0; di < static_cast<int>(instances_.size()); ++di) {
      DeviceInstance* instance = instances_[di];
      instance->InitTensor(
          dtype, TensorShape({tensor_len}), [&expected, dtype, di](Tensor* t) {
            for (size_t i = 0; i < t->NumElements(); ++i) {
              // The cast is necessary to prevent clang-tidy from insisting
              // that a faster non-open source function be substituted.
              float value = pow(10, static_cast<double>(di)) * i;
              if (dtype == DT_INT32 || dtype == DT_INT64) {
                value = di * 10 + i;
              }
              t->flat<T>()(i) = static_cast<T>(value);
              expected[i] += value;
            }
          });
    }
    Reduce(fail_after);
    if (fail_after > 0) {
      // Confirm that every device terminated with the expected error status.
      for (int di = 0; di < static_cast<int>(instances_.size()); ++di) {
        EXPECT_NE(
            instances_[di]->status_.error_message().find("Deliberate failure"),
            string::npos);
      }
    } else {
      // Confirm that every device computed the same correct reduction value.
      for (int i = 0; i < tensor_len; ++i) {
        expected[i] /= (num_workers * num_devices);
      }
      for (int di = 0; di < static_cast<int>(instances_.size()); ++di) {
        TF_EXPECT_OK(instances_[di]->status_);
        Tensor* inst = &instances_[di]->tensor_;
        CHECK(inst);
        Tensor actual(dtype, TensorShape({tensor_len}));
        if (device_type_ == DEVICE_CPU) {
          CHECK(actual.CopyFrom(*inst, inst->shape()));
          VLOG(1) << "actual " << actual.SummarizeValue(100);
        } else if (device_type_ == DEVICE_GPU) {
          Device* dev = instances_[di]->device_;
          auto* dev_info = dev->tensorflow_gpu_device_info();
          CHECK(dev_info);
          CHECK(dev_info->default_context
                    ->CopyDeviceTensorToCPUSync(inst, "" /*tensor_name*/, dev,
                                                &actual)
                    .ok());
        }

        auto alias = actual.template unaligned_flat<T>();
        for (int i = 0; i < tensor_len; ++i) {
          switch (dtype) {
            case DT_FLOAT:
              EXPECT_FLOAT_EQ(expected[i], alias(i))
                  << "Mismatch at device " << di << " index " << i;
              break;
            case DT_DOUBLE:
              EXPECT_DOUBLE_EQ(expected[i], alias(i))
                  << "Mismatch at device " << di << " index " << i;
              break;
            case DT_INT32:
            case DT_INT64:
              EXPECT_EQ(expected[i], alias(i))
                  << "Mismatch at device " << di << " index " << i;
              break;
            default:
              LOG(FATAL) << "unimplemented";
          }
        }
      }
    }
  }

  std::unique_ptr<OpKernel> GetCollectiveReduce(const CollectiveParams& params,
                                                Tensor* input,
                                                const DeviceType& device_type,
                                                DeviceBase* device) {
    mutex_lock l(mu_);
    NodeDef node_def;
    NodeDefBuilder builder(
        strings::StrCat("collective_reduce_", reduce_counter_++),
        "CollectiveReduce");
    TF_CHECK_OK(
        builder.Attr("T", params.instance.data_type)
            .Attr("merge_op", "Add")
            .Attr("final_op", "Id")
            .Attr("group_size", params.group.group_size)
            .Attr("group_key", params.group.group_key)
            .Attr("instance_key", params.instance.instance_key)
            .Attr("subdiv_offsets", params.instance.impl_details.subdiv_offsets)
            .Input(FakeInput(params.instance.data_type))
            .Finalize(&node_def));
    return GetKernel(node_def, device_type, device);
  }

  void RunSubdivPermsTest(
      CollectiveParams* cp,
      const std::vector<std::vector<int>>& expected_subdiv_perms,
      const std::vector<int>& expected_subdiv_rank) {
    col_exec_ = nullptr;
    cp->instance.impl_details.subdiv_permutations.clear();
    cp->subdiv_rank.clear();
    // Create a stub ring reducer only for testing param initialization.
    RingReducer* reducer = new RingReducer;
    core::ScopedUnref unref(reducer);
    TF_CHECK_OK(reducer->InitializeCollectiveParams(cp));
    EXPECT_EQ(expected_subdiv_perms,
              cp->instance.impl_details.subdiv_permutations);
    EXPECT_EQ(expected_subdiv_rank, cp->subdiv_rank);
    reducer->group_size_tensor_ready_.Notify();  // To unblock destructor.
  }

  class DeviceInstance {
   public:
    DeviceInstance(int rank, const string& dev_name,
                   const DeviceType& device_type, RingReducerTest* parent)
        : parent_(parent),
          dev_name_(dev_name),
          device_type_(device_type),
          rank_(rank),
          col_params_(new CollectiveParams()) {
      TF_CHECK_OK(parent_->dev_mgr_->LookupDevice(dev_name, &device_))
          << "Couldn't find device " << dev_name
          << " existing devices: " << parent_->dev_mgr_->DebugString();
      col_params_->name = parent_->col_params_->name;
      col_params_->group = parent_->col_params_->group;
      col_params_->instance = parent->col_params_->instance;
      col_params_->task.is_local = parent_->col_params_->task.is_local;
      col_params_->subdiv_rank = parent_->col_params_->subdiv_rank;

      int num_subdivs = static_cast<int>(col_params_->subdiv_rank.size());
      int group_size = col_params_->group.group_size;
      CHECK_EQ(group_size,
               static_cast<int>(col_params_->group.device_names.size()));
      // Id of this device is at rank position in first subdiv perm.
      int my_device_id =
          col_params_->instance.impl_details.subdiv_permutations[0][rank];
      col_params_->default_rank = my_device_id;
      // Set rank for all other subdivs by finding that device_id.
      for (int sdi = 0; sdi < num_subdivs; ++sdi) {
        for (int r = 0; r < static_cast<int>(col_params_->instance.impl_details
                                                 .subdiv_permutations[sdi]
                                                 .size());
             ++r) {
          if (my_device_id ==
              col_params_->instance.impl_details.subdiv_permutations[sdi][r]) {
            col_params_->subdiv_rank[sdi] = r;
            break;
          }
        }
      }
    }

    ~DeviceInstance() { col_params_->Unref(); }

    void InitTensor(DataType dtype, const TensorShape& shape,
                    const std::function<void(Tensor*)>& init_f) {
      tensor_ =
          Tensor(device_->GetAllocator(AllocatorAttributes()), dtype, shape);
      if (device_type_ == DEVICE_CPU) {
        init_f(&tensor_);
      } else if (device_type_ == DEVICE_GPU) {
        Tensor cpu_tensor(dtype, shape);
        init_f(&cpu_tensor);
        auto* dev_info = device_->tensorflow_gpu_device_info();
        CHECK(dev_info);
        CHECK(dev_info->default_context
                  ->CopyCPUTensorToDeviceSync(&cpu_tensor, device_, &tensor_)
                  .ok());
      } else {
        LOG(FATAL) << "Unsupported device_type " << device_type_;
      }
    }

    void DoReduce() {
      merge_op_ =
          GetAdd(col_params_->instance.data_type, device_type_, device_);
      final_op_ =
          GetDiv(col_params_->instance.data_type, device_type_, device_);
      col_params_->merge_op = merge_op_.get();
      col_params_->final_op = final_op_.get();

      // Prepare an OpKernelContext.
      OpKernelContext::Params op_params;
      op_params.step_id = kStepId;
      op_params.device = device_;
      op_params.cancellation_manager = &parent_->cancellation_manager_;
      gtl::InlinedVector<TensorValue, 4> inputs;
      inputs.push_back(TensorValue(&tensor_));
      op_params.inputs = &inputs;
      gtl::InlinedVector<AllocatorAttributes, 4> input_aa(
          {AllocatorAttributes()});
      op_params.input_alloc_attrs = &input_aa;
      DeviceContext* dev_ctx = nullptr;
      auto* dev_info = device_->tensorflow_gpu_device_info();
      if (dev_info) {
        dev_ctx = dev_info->default_context;
        dev_ctx->Ref();
      } else {
        dev_ctx = new DeviceContext;
      }
      op_params.op_device_context = dev_ctx;
      int forward_from = 0;
      op_params.forward_from_array = &forward_from;
      AllocatorAttributes generic_alloc_attr;
      op_params.output_attr_array = &generic_alloc_attr;
      std::unique_ptr<OpKernel> op = parent_->GetCollectiveReduce(
          *col_params_, &tensor_, DEVICE_CPU, device_);
      op_params.op_kernel = op.get();
      OpKernelContext ctx(&op_params, 1);

      // We never actually execute the kernel, so we need to do the output
      // allocation it would do, ourselves.
      Tensor* output_tensor_ptr = nullptr;
      TF_CHECK_OK(ctx.forward_input_or_allocate_output({0}, 0, tensor_.shape(),
                                                       &output_tensor_ptr));
      CHECK_EQ(output_tensor_ptr, ctx.mutable_output(0));

      // Prepare a RingReducer instance.
      string exec_key =
          strings::StrCat(col_params_->instance.instance_key, ":0:0");
      RingReducer* reducer = new RingReducer;
      core::ScopedUnref unref(reducer);
      auto col_ctx = std::make_shared<CollectiveContext>(
          parent_->col_exec_, /*nccl_communicator*/ nullptr,
          parent_->dev_mgr_.get(), &ctx, &op_params, col_params_, exec_key,
          kStepId, &tensor_, &tensor_);
      TF_CHECK_OK(reducer->InitializeCollectiveContext(col_ctx));

      // Run the all-reduce.
      reducer->Run([this](Status s) { status_ = s; });
      if (status_.ok()) {
        CHECK(tensor_.CopyFrom(*ctx.mutable_output(0), tensor_.shape()));
      }

      dev_ctx->Unref();
    }

    const Tensor& tensor() { return tensor_; }

    RingReducerTest* parent_;
    string dev_name_;
    DeviceType device_type_;
    int rank_;
    Tensor tensor_;
    Device* device_;
    CollectiveParams* col_params_;
    std::unique_ptr<OpKernel> merge_op_;
    std::unique_ptr<OpKernel> final_op_;
    std::unique_ptr<CollectiveAdapter> ca_;
    std::unique_ptr<OpKernelContext> ctx_;
    Status status_;
  };

  bool stop_ = false;
  DeviceType device_type_;
  TestCollectiveExecutorMgr col_exec_mgr_;
  CollectiveExecutor* col_exec_;
  CollectiveRemoteAccessLocal* rma_;
  std::unique_ptr<DeviceResolverLocal> dev_resolver_;
  std::shared_ptr<UnboundedWorkQueue> work_queue_;
  std::vector<DeviceInstance*> instances_;
  CollectiveParams* col_params_;
  std::vector<std::unique_ptr<tensorflow::Device>> gpu_devices_;
  std::unique_ptr<tensorflow::DeviceMgr> dev_mgr_;
  std::unique_ptr<string> gpu_ring_order_;
  mutex mu_;
  int32 reduce_counter_ TF_GUARDED_BY(mu_) = 0;
  CancellationManager cancellation_manager_;
};

CollectiveParams* SetUpCollectiveParams(const int num_devs_per_task,
                                        const int num_tasks) {
  auto cp = new CollectiveParams();
  const int kNumDevs = num_devs_per_task * num_tasks;
  cp->group.group_key = 1;
  cp->group.group_size = kNumDevs;
  cp->group.device_type = DeviceType("GPU");
  cp->group.num_tasks = num_tasks;
  cp->instance.instance_key = 3;
  cp->instance.type = REDUCTION_COLLECTIVE;
  cp->instance.data_type = DataType(DT_FLOAT);
  cp->instance.shape = TensorShape({kNumDevs});
  cp->instance.impl_details.collective_name = "RingReduce";
  cp->instance.impl_details.subdiv_offsets.push_back(0);
  cp->is_source = false;
  for (int i = 0; i < kNumDevs; ++i) {
    int task_id = i / num_devs_per_task;
    int dev_id = i % num_devs_per_task;
    string task_name = strings::StrCat("/job:worker/replica:0/task:", task_id);
    string device_name = strings::StrCat(task_name, "/device:GPU:", dev_id);
    cp->group.task_names.push_back(task_name);
    cp->group.device_names.push_back(device_name);
  }
  return cp;
}

TEST_F(RingReducerTest, InitializeParams) {
  const int kNumDevsPerTask = 8;
  const int kNumTasks = 3;
  CollectiveParams* cp = SetUpCollectiveParams(kNumDevsPerTask, kNumTasks);
  core::ScopedUnref unref(cp);

  cp->default_rank = 0;
  cp->instance.impl_details.subdiv_offsets = {0, 4};
  RunSubdivPermsTest(cp,
                     {{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                       12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
                      {4, 5, 6,  7,  0,  1,  2,  3,  12, 13, 14, 15,
                       8, 9, 10, 11, 20, 21, 22, 23, 16, 17, 18, 19}},
                     {0, 4});

  cp->instance.impl_details.subdiv_offsets = {0, -4};
  RunSubdivPermsTest(cp,
                     {{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                       12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
                      {3,  2,  1,  0,  7,  6,  5,  4,  11, 10, 9,  8,
                       15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20}},
                     {0, 3});

  cp->default_rank = 3;
  cp->instance.impl_details.subdiv_offsets = {3, -3};
  RunSubdivPermsTest(cp,
                     {{3,  4, 5, 6,  7,  0,  1,  2,  11, 12, 13, 14,
                       15, 8, 9, 10, 19, 20, 21, 22, 23, 16, 17, 18},
                      {4, 3,  2,  1,  0,  7,  6,  5,  12, 11, 10, 9,
                       8, 15, 14, 13, 20, 19, 18, 17, 16, 23, 22, 21}},
                     {0, 1});
}

TEST_F(RingReducerTest, AutomaticSubdivs) {
  const int kNumDevsPerTask = 8;
  const int kNumTasks = 3;
  const int kNumDevs = kNumDevsPerTask * kNumTasks;
  CollectiveParams* cp = SetUpCollectiveParams(kNumDevsPerTask, kNumTasks);
  core::ScopedUnref unref(cp);

  // Test automatic generation of subdiv offsets.
  cp->default_rank = 0;
  cp->instance.impl_details.subdiv_offsets.clear();
  RunSubdivPermsTest(cp, {{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                           12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}},
                     {0});

  // Set shape so that with 2 subdivs chunk_size is 3 MiB.  This should cause 2
  // offsets, {0, -4}, to be generated.
  {
    int num_subdivs = 2;
    int num_chunks = kNumDevs * num_subdivs;
    size_t chunk_size = 3 * 1048576;  // 3 MB
    size_t tensor_size = chunk_size * num_chunks;
    cp->instance.shape =
        TensorShape({static_cast<int64>(tensor_size / DataTypeSize(DT_FLOAT))});
  }
  cp->instance.impl_details.subdiv_offsets.clear();
  RunSubdivPermsTest(cp,
                     {{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                       12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
                      {3,  2,  1,  0,  7,  6,  5,  4,  11, 10, 9,  8,
                       15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20}},
                     {0, 3});
}

TEST_F(RingReducerTest, AutomaticSubdivUpperBound) {
  const int kNumDevsPerTask = 1;
  const int kNumTasks = 4;
  CollectiveParams* cp = SetUpCollectiveParams(kNumDevsPerTask, kNumTasks);
  core::ScopedUnref unref(cp);

  cp->default_rank = 0;
  cp->instance.impl_details.subdiv_offsets.clear();
  cp->instance.shape = TensorShape({104857600 / DataTypeSize(DT_FLOAT)});
  RunSubdivPermsTest(cp, {{0, 1, 2, 3}, {0, 1, 2, 3}}, {0, 0});
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
      case DT_INT32: {                                                        \
        RunTest<int32>(dtype, DEVICE_##T, W, D, S, L, A);                     \
      } break;                                                                \
      case DT_INT64: {                                                        \
        RunTest<int64>(dtype, DEVICE_##T, W, D, S, L, A);                     \
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
