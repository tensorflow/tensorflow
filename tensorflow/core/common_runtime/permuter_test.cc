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
#include "tensorflow/core/common_runtime/base_collective_executor.h"
#include "tensorflow/core/common_runtime/collective_rma_local.h"
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
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/unbounded_work_queue.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace {

static int64 kStepId = 123;

// Wraps CollectiveRemoteAccessLocal with the ability to return an
// error status to the N'th action.
// TODO(b/113171733): factor out of this file and ring_reducer_test.cc
// into a single common source.
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
      auto error = errors::Internal("Deliberate failure");
      LOG(INFO) << "triggering failure " << error;
      SchedNonBlockingClosureAfter(
          1000, [this, error] { buf_rendezvous()->StartAbort(error); });
      done(error);
      return true;
    }
    return false;
  }

  void RecvFromPeer(const string& peer_device, const string& peer_task,
                    bool peer_is_local, const string& key, Device* to_device,
                    DeviceContext* to_device_ctx,
                    const AllocatorAttributes& to_alloc_attr, Tensor* to_tensor,
                    const DeviceLocality& client_locality, int stream_index,
                    CancellationManager* cancellation_manager,
                    const StatusCallback& done) override {
    if (MaybeFail(done)) return;
    CollectiveRemoteAccessLocal::RecvFromPeer(
        peer_device, peer_task, peer_is_local, key, to_device, to_device_ctx,
        to_alloc_attr, to_tensor, client_locality, stream_index,
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

class PermuterTest : public ::testing::Test {
 protected:
  PermuterTest()
      : device_type_(DEVICE_CPU), col_exec_(nullptr), col_params_(nullptr) {}

  ~PermuterTest() override {
    stop_ = true;
    for (auto i : instances_) delete i;
    if (col_exec_) col_exec_->Unref();
    if (col_params_) col_params_->Unref();
  }

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

  void Init(int num_workers, int num_devices_per_worker, DataType dtype,
            const DeviceType& device_type, int fail_after) {
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
      for (int di = 0; di < num_devices_per_worker; ++di) {
        if (device_type == DEVICE_CPU) {
          string dev_name = strings::StrCat("/job:worker/replica:0/task:", wi,
                                            "/device:CPU:", di);
          local_devices.push_back(absl::make_unique<ThreadPoolDevice>(
              sess_opts, dev_name, mem_limit, dev_locality, cpu_allocator()));
        } else if (device_type == DEVICE_GPU && !gpu_devices_.empty()) {
          int dev_idx = (wi * num_devices_per_worker) + di;
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
    col_params_->instance.data_type = dtype;
    static const int kInstanceKey = 18;
    col_params_->instance.instance_key = kInstanceKey;
    col_params_->group.device_type = device_type;
    col_params_->instance.type = PERMUTE_COLLECTIVE;

    // Set up all the fake device contexts.
    for (int wi = 0; wi < num_workers; wi++) {
      for (int di = 0; di < num_devices_per_worker; di++) {
        string task_name = strings::StrCat("/job:worker/replica:0/task:", wi);
        string dev_name;
        if (device_type == DEVICE_GPU) {
          dev_name = strings::StrCat(task_name, "/device:GPU:0");
        } else {
          dev_name = strings::StrCat(task_name, "/device:CPU:", di);
        }
        col_params_->group.device_names.push_back(dev_name);
        col_params_->instance.devices.push_back(dev_name);
        int default_rank = wi * num_devices_per_worker + di;
        permutation_.push_back(default_rank);
        col_params_->group.task_names.push_back(task_name);
        col_params_->task.is_local.push_back(true);
      }
    }

    // Generate a permutation by permuting every two instances.
    // E.g. [0,1] becomes [1,0]
    //      [0,1,2,3] becomes [1,0,3,2]
    for (int i = 0; i < permutation_.size(); i += 2) {
      // If the total number of instances is odd,
      // swap the last instance with the first.
      // E.g. [0,1,2] becomes [2,0,1]
      if (permutation_.size() == i + 1) {
        std::swap(permutation_[i], permutation_[0]);
        continue;
      }
      std::next_permutation(permutation_.begin() + i,
                            permutation_.begin() + i + 2);
    }
    col_params_->instance.permutation = permutation_;

    for (int wi = 0; wi < num_workers; wi++) {
      for (int di = 0; di < num_devices_per_worker; di++) {
        int default_rank = wi * num_devices_per_worker + di;
        instances_.push_back(new DeviceInstance(
            default_rank, col_params_->group.device_names[default_rank],
            device_type, this));
      }
    }
  }

  typedef std::function<void(Tensor*)> InitFunc;

  void Permute(int fail_after) {
    std::atomic<int> done(0);
    for (auto di : instances_) {
      SchedClosure([di, &done] {
        di->DoPermute();
        ++done;
      });
      if (fail_after > 0) {
        // Stagger the op execution starts.
        Env::Default()->SleepForMicroseconds(100);
      }
    }
    while (done < instances_.size()) {
      if (stop_) break;
      Env::Default()->SleepForMicroseconds(1000);
    }
  }

  template <typename T>
  void RunTest(DataType dtype, const DeviceType& device_type, int num_workers,
               int num_devices, int tensor_len, int fail_after) {
    Init(num_workers, num_devices, dtype, device_type, fail_after);
    std::vector<T> expected(tensor_len * num_devices * num_workers, 0.0);
    // Initialize each instance tensor with distinct values.
    for (int di = 0; di < instances_.size(); ++di) {
      DeviceInstance* instance = instances_[di];
      instance->InitTensor(
          dtype, TensorShape({tensor_len}),
          [this, &expected, di, tensor_len](Tensor* t) {
            for (size_t i = 0; i < t->NumElements(); ++i) {
              // The cast is necessary to prevent clang-tidy from insisting
              // that a faster non-open source function be substituted.
              float value = pow(10, static_cast<double>(di)) * i;
              t->flat<T>()(i) = value;
              expected[permutation_[di] * tensor_len + i] = value;
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
      Tensor* inst = &instances_[di]->tensor_output_;
      Tensor actual(dtype, TensorShape({tensor_len}));
      if (device_type_ == DEVICE_CPU) {
        CHECK(actual.CopyFrom(*inst, inst->shape()));
      } else if (device_type_ == DEVICE_GPU) {
        Device* dev = instances_[di]->device_;
        auto* dev_info = dev->tensorflow_gpu_device_info();
        CHECK(dev_info);
        TF_CHECK_OK(dev_info->default_context->CopyDeviceTensorToCPUSync(
            inst, "" /*tensor_name*/, dev, &actual));
      }
      for (int i = 0; i < tensor_len; ++i) {
        switch (dtype) {
          case DT_FLOAT:
            EXPECT_FLOAT_EQ(expected[(di * tensor_len) + i],
                            actual.template flat<T>()(i))
                << "Mismatch at device " << di << " index " << i;
            break;
          case DT_DOUBLE:
            EXPECT_DOUBLE_EQ(expected[(di * tensor_len) + i],
                             actual.template flat<T>()(i))
                << "Mismatch at device " << di << " index " << i;
            break;
          case DT_BOOL:
          case DT_INT32:
          case DT_INT64:
            EXPECT_EQ(expected[(di * tensor_len) + i],
                      actual.template flat<T>()(i))
                << "Mismatch at device " << di << " index " << i;
            break;
          default:
            LOG(FATAL) << "unimplemented";
        }
      }
      //  }
    }
  }

  class DeviceInstance {
   public:
    DeviceInstance(int rank, const string& dev_name,
                   const DeviceType& device_type, PermuterTest* parent)
        : parent_(parent),
          dev_name_(dev_name),
          device_type_(device_type),
          rank_(rank),
          col_params_(new CollectiveParams()) {
      TF_CHECK_OK(parent_->dev_mgr_->LookupDevice(dev_name, &device_));
      col_params_->name = parent_->col_params_->name;
      col_params_->instance.data_type =
          parent_->col_params_->instance.data_type;
      col_params_->instance.instance_key =
          parent_->col_params_->instance.instance_key;
      col_params_->group.device_type = parent_->col_params_->group.device_type;
      col_params_->group.device_names =
          parent_->col_params_->group.device_names;
      col_params_->instance.devices = parent_->col_params_->instance.devices;
      col_params_->instance.permutation =
          parent->col_params_->instance.permutation;
      col_params_->group.task_names = parent_->col_params_->group.task_names;
      col_params_->task.is_local = parent_->col_params_->task.is_local;
      CHECK_EQ(col_params_->instance.devices.size(),
               col_params_->group.device_names.size());
      // Default rank is order in device_names.
      col_params_->default_rank = rank;
    }

    ~DeviceInstance() { col_params_->Unref(); }

    void InitTensor(DataType dtype, const TensorShape& shape,
                    const InitFunc& f) {
      tensor_input_ =
          Tensor(device_->GetAllocator(AllocatorAttributes()), dtype, shape);
      tensor_output_ =
          Tensor(device_->GetAllocator(AllocatorAttributes()), dtype, shape);
      if (device_type_ == DEVICE_CPU) {
        f(&tensor_input_);
      } else if (device_type_ == DEVICE_GPU) {
        Tensor cpu_tensor(dtype, shape);
        f(&cpu_tensor);
        // Notification notification;
        auto* dev_info = device_->tensorflow_gpu_device_info();
        CHECK(dev_info);
        TF_CHECK_OK(dev_info->default_context->CopyCPUTensorToDeviceSync(
            &cpu_tensor, device_, &tensor_input_));
      } else {
        LOG(FATAL) << "Unsupported device_type " << device_type_;
      }
    }

    void DoPermute() {
      // Prepare an OpKernelContext.
      OpKernelContext::Params op_params;
      op_params.step_id = parent_->step_id_;
      op_params.device = device_;
      op_params.cancellation_manager = &parent_->cancellation_manager_;
      gtl::InlinedVector<TensorValue, 4> inputs;
      inputs.push_back(TensorValue(&tensor_input_));
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
      AllocatorAttributes generic_alloc_attr;
      op_params.output_attr_array = &generic_alloc_attr;
      OpKernelContext ctx(&op_params, 1);

      // Prepare a Permuter instance.
      string exec_key =
          strings::StrCat(col_params_->instance.instance_key, ":0:0");
      Permuter* permuter = new Permuter;
      core::ScopedUnref unref(permuter);
      auto col_ctx = std::make_shared<CollectiveContext>(
          parent_->col_exec_, /*nccl_communicator*/ nullptr,
          parent_->dev_mgr_.get(), &ctx, &op_params, col_params_, exec_key,
          kStepId, &tensor_input_, &tensor_output_);
      TF_CHECK_OK(permuter->InitializeCollectiveContext(col_ctx));
      Notification note;
      // Run the permute.
      permuter->Run([this, &note](Status s) {
        status_ = s;
        note.Notify();
      });
      note.WaitForNotification();
      dev_ctx->Unref();
    }

    PermuterTest* parent_;
    string dev_name_;
    DeviceType device_type_ = DEVICE_CPU;
    int rank_;
    Tensor tensor_input_;
    Tensor tensor_output_;
    Device* device_;
    CollectiveParams* col_params_;
    Status status_;
  };  // class DeviceInstance

  bool stop_ = false;
  int64 step_id_ = kStepId;
  DeviceType device_type_;
  TestCollectiveExecutorMgr col_exec_mgr_;
  CollectiveExecutor* col_exec_ = nullptr;
  CollectiveRemoteAccessLocal* rma_;
  std::unique_ptr<DeviceResolverLocal> dev_resolver_;
  std::shared_ptr<UnboundedWorkQueue> work_queue_;
  std::vector<DeviceInstance*> instances_;
  CollectiveParams* col_params_;
  std::vector<std::unique_ptr<tensorflow::Device>> gpu_devices_;
  std::unique_ptr<tensorflow::DeviceMgr> dev_mgr_;
  std::unique_ptr<string> gpu_ring_order_;
  mutex mu_;
  int permute_counter_ TF_GUARDED_BY(mu_) = 0;
  std::vector<int> permutation_;
  CancellationManager cancellation_manager_;
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
#define DEF_TEST(B, T, W, D, L, A)                                            \
  TEST_F(PermuterTest,                                                        \
         DaTy##B##_DevTy##T##_Wkr##W##_Dev##D##_Sdiv##S##_Len##L##_Abrt##A) { \
    DataType dtype = DT_##B;                                                  \
    switch (dtype) {                                                          \
      case DT_BOOL: {                                                         \
        RunTest<bool>(dtype, DEVICE_##T, W, D, L, A);                         \
      } break;                                                                \
      case DT_FLOAT: {                                                        \
        RunTest<float>(dtype, DEVICE_##T, W, D, L, A);                        \
      } break;                                                                \
      case DT_DOUBLE: {                                                       \
        RunTest<double>(dtype, DEVICE_##T, W, D, L, A);                       \
      } break;                                                                \
      case DT_INT32: {                                                        \
        RunTest<int32>(dtype, DEVICE_##T, W, D, L, A);                        \
      } break;                                                                \
      case DT_INT64: {                                                        \
        RunTest<int64>(dtype, DEVICE_##T, W, D, L, A);                        \
      } break;                                                                \
      default:                                                                \
        LOG(FATAL) << "Unimplemented";                                        \
    }                                                                         \
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
