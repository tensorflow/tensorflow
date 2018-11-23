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
#include "tensorflow/core/common_runtime/hierarchical_tree_broadcaster.h"

#include <algorithm>
#include "tensorflow/core/common_runtime/base_collective_executor.h"
#include "tensorflow/core/common_runtime/collective_rma_local.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/device_resolver_local.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
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
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace {

static int64 kStepId = 123;

// The test harness won't allow a mixture of fixture and non-fixture
// tests in one file, so this is a trival fixture for tests that don't
// need the heavy-weight HierarchicalTreeBroadcasterTest fixture.
class TrivialTest : public ::testing::Test {
 protected:
  TrivialTest() {}
};

// Tests of static TreeSendTo() and TreeRecvFrom() functions.
// D = number of devices
// S = source rank
// R = tested rank
// RF = receive-from rank
// ST = send_to rank vector
#define DEF_TL_TEST(D, S, R, RF, ST)                                 \
  TEST_F(TrivialTest, TreeLinks_##D##Devs_##S##Source_##R##Rank) {   \
    CollectiveParams cp;                                             \
    cp.group.group_size = D;                                         \
    cp.instance.impl_details.subdiv_source_rank = {S};               \
    cp.instance.impl_details.subdiv_permutations.push_back(          \
        std::vector<int>(D, 0));                                     \
    cp.subdiv_rank = {R};                                            \
    cp.is_source = (S == R);                                         \
    EXPECT_EQ(RF, HierarchicalTreeBroadcaster::TreeRecvFrom(cp, 0)); \
    std::vector<int> expected = ST;                                  \
    std::vector<int> send_to;                                        \
    HierarchicalTreeBroadcaster::TreeSendTo(cp, 0, &send_to);        \
    ASSERT_EQ(expected.size(), send_to.size());                      \
    for (int i = 0; i < expected.size(); ++i) {                      \
      EXPECT_EQ(expected[i], send_to[i]);                            \
    }                                                                \
  }

#define V(...) std::vector<int>({__VA_ARGS__})

//          D  S  R  RF  ST
// 2 device cases
DEF_TL_TEST(2, 0, 0, -1, V(1))
DEF_TL_TEST(2, 1, 0, 1, V())
DEF_TL_TEST(2, 0, 1, 0, V())
DEF_TL_TEST(2, 1, 1, -1, V(0))
// 3 device cases
DEF_TL_TEST(3, 0, 0, -1, V(1, 2))
DEF_TL_TEST(3, 0, 1, 0, V())
DEF_TL_TEST(3, 0, 2, 0, V())
DEF_TL_TEST(3, 1, 0, 1, V(2))
DEF_TL_TEST(3, 1, 1, -1, V(0))
DEF_TL_TEST(3, 1, 2, 0, V())
DEF_TL_TEST(3, 2, 0, 2, V())
DEF_TL_TEST(3, 2, 1, 2, V())
DEF_TL_TEST(3, 2, 2, -1, V(0, 1))
// 4 device cases
DEF_TL_TEST(4, 0, 0, -1, V(1, 2))
DEF_TL_TEST(4, 0, 1, 0, V(3))
DEF_TL_TEST(4, 0, 2, 0, V())
DEF_TL_TEST(4, 0, 3, 1, V())
DEF_TL_TEST(4, 1, 0, 1, V(2, 3))
DEF_TL_TEST(4, 1, 1, -1, V(0))
DEF_TL_TEST(4, 1, 2, 0, V())
DEF_TL_TEST(4, 1, 3, 0, V())
DEF_TL_TEST(4, 2, 0, 2, V(3))
DEF_TL_TEST(4, 2, 1, 2, V())
DEF_TL_TEST(4, 2, 2, -1, V(0, 1))
DEF_TL_TEST(4, 2, 3, 0, V())
DEF_TL_TEST(4, 3, 0, 3, V(2))
DEF_TL_TEST(4, 3, 1, 3, V())
DEF_TL_TEST(4, 3, 2, 0, V())
DEF_TL_TEST(4, 3, 3, -1, V(0, 1))
// 8 device cases
//          D  S  R  RF  ST
DEF_TL_TEST(8, 0, 0, -1, V(1, 2))
DEF_TL_TEST(8, 0, 1, 0, V(3, 4))
DEF_TL_TEST(8, 0, 2, 0, V(5, 6))
DEF_TL_TEST(8, 0, 3, 1, V(7))
DEF_TL_TEST(8, 0, 4, 1, V())
DEF_TL_TEST(8, 0, 5, 2, V())
DEF_TL_TEST(8, 0, 6, 2, V())
DEF_TL_TEST(8, 0, 7, 3, V())
DEF_TL_TEST(8, 7, 0, 7, V(2, 3))
DEF_TL_TEST(8, 7, 1, 7, V(4, 5))
DEF_TL_TEST(8, 7, 2, 0, V(6))
DEF_TL_TEST(8, 7, 3, 0, V())
DEF_TL_TEST(8, 7, 4, 1, V())
DEF_TL_TEST(8, 7, 5, 1, V())
DEF_TL_TEST(8, 7, 6, 2, V())
DEF_TL_TEST(8, 7, 7, -1, V(0, 1))
#undef DEF_TL_TEST
#undef V

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
                    const StatusCallback& done) override {
    if (MaybeFail(done)) return;
    CollectiveRemoteAccessLocal::RecvFromPeer(
        peer_device, peer_task, peer_is_local, key, to_device, to_device_ctx,
        to_alloc_attr, to_tensor, client_locality, stream_index, done);
  }

  void PostToPeer(const string& peer_device, const string& peer_task,
                  const string& key, Device* from_device,
                  DeviceContext* from_device_ctx,
                  const AllocatorAttributes& from_alloc_attr,
                  const Tensor* from_tensor,
                  const DeviceLocality& client_locality,
                  const StatusCallback& done) override {
    if (MaybeFail(done)) return;
    CollectiveRemoteAccessLocal::PostToPeer(
        peer_device, peer_task, key, from_device, from_device_ctx,
        from_alloc_attr, from_tensor, client_locality, done);
  }

  mutex mu_;
  int fail_after_ GUARDED_BY(mu_);
};

class HierarchicalTreeBroadcasterTest : public ::testing::Test {
 protected:
  HierarchicalTreeBroadcasterTest() : device_type_(DEVICE_CPU) {}

  ~HierarchicalTreeBroadcasterTest() override {
    stop_ = true;
    for (auto i : instances_) delete i;
    if (col_exec_) col_exec_->Unref();
  }

#ifdef GOOGLE_CUDA
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
#ifdef GOOGLE_CUDA
    InitGPUDevices();
#endif
    VLOG(2) << "num_workers=" << num_workers
            << " num_devices_per_worker=" << num_devices_per_worker;
    int total_num_devices = num_workers * num_devices_per_worker;
    device_type_ = device_type;
    std::vector<Device*> local_devices;
    SessionOptions sess_opts;
    sess_opts.env = Env::Default();
    Bytes mem_limit(4 << 20);
    DeviceLocality dev_locality;
    for (int wi = 0; wi < num_workers; ++wi) {
      for (int di = 0; di < num_devices_per_worker; ++di) {
        if (device_type == DEVICE_CPU) {
          string dev_name = strings::StrCat("/job:worker/replica:0/task:", wi,
                                            "/device:CPU:", di);
          local_devices.push_back(new ThreadPoolDevice(
              sess_opts, dev_name, mem_limit, dev_locality, cpu_allocator()));
        } else if (device_type == DEVICE_GPU && !gpu_devices_.empty()) {
          int dev_idx = (wi * num_devices_per_worker) + di;
          if (dev_idx >= static_cast<int>(gpu_devices_.size())) {
            LOG(INFO) << "dev_mgr has access to limited GPUs, reusing for more "
                         "than one ring node.";
          } else {
            local_devices.push_back(gpu_devices_[dev_idx]);
          }
        } else {
          LOG(FATAL) << "Unsupported device_type " << device_type;
        }
      }
    }
    if (!dev_mgr_ || device_type == DEVICE_CPU) {
      dev_mgr_.reset(new DeviceMgr(local_devices));
    }
    if (!gpu_ring_order_) gpu_ring_order_.reset(new string());
    dev_resolver_.reset(new DeviceResolverLocal(dev_mgr_.get()));
    rma_ = new FailTestRMA(dev_mgr_.get(), dev_resolver_.get(), kStepId,
                           fail_after);
    col_exec_ = new BaseCollectiveExecutor(
        &col_exec_mgr_, rma_, kStepId, dev_mgr_.get(), gpu_ring_order_.get());
    col_params_.name = "test_collective";
    col_params_.instance.data_type = dtype;
    static const int kGroupKey = 6;
    col_params_.group.group_key = kGroupKey;
    static const int kInstanceKey = 18;
    col_params_.instance.instance_key = kInstanceKey;
    col_params_.group.device_type = device_type;
    col_params_.group.group_size = num_workers * num_devices_per_worker;
    col_params_.instance.impl_details.subdiv_offsets.clear();
    col_params_.instance.type = BROADCAST_COLLECTIVE;

    int num_subdivs = num_workers + (num_workers > 1 ? 1 : 0);
    VLOG(2) << "#subdiv=" << num_subdivs;
    col_params_.instance.impl_details.subdiv_permutations.resize(num_subdivs);
    col_params_.subdiv_rank.resize(num_subdivs);

    // Inter-machine broadcast.
    int subdiv_i = 0;
    if (num_workers > 1) {
      col_params_.instance.impl_details.subdiv_permutations[subdiv_i].resize(
          total_num_devices, -1);
      for (int i = 0, rank = 0; i < total_num_devices; i++) {
        if (i % num_devices_per_worker == 0) {
          col_params_.instance.impl_details
              .subdiv_permutations[subdiv_i][rank] = i;
          rank++;
        }
      }
      if (VLOG_IS_ON(2)) {
        string sp_buf;
        for (int p :
             col_params_.instance.impl_details.subdiv_permutations[subdiv_i])
          strings::StrAppend(&sp_buf, p, ", ");
        VLOG(2) << "subdiv_i=" << subdiv_i << " perm=" << sp_buf;
      }
      subdiv_i++;
    }
    // Intra-machine broadcast.
    for (int i = 0; subdiv_i < num_subdivs; i++, subdiv_i++) {
      col_params_.instance.impl_details.subdiv_permutations[subdiv_i].resize(
          total_num_devices, -1);
      int perm_i_base = i * num_devices_per_worker;
      VLOG(2) << "subdiv_i=" << subdiv_i << " i=" << i
              << " perm_i_base=" << perm_i_base << " subdiv_perms.size="
              << col_params_.instance.impl_details.subdiv_permutations.size();
      // subdiv for worker i.
      for (int j = perm_i_base, rank = 0;
           j < perm_i_base + num_devices_per_worker; j++, rank++) {
        col_params_.instance.impl_details.subdiv_permutations[subdiv_i][rank] =
            j;
      }
      if (VLOG_IS_ON(2)) {
        string sp_buf;
        for (int p :
             col_params_.instance.impl_details.subdiv_permutations[subdiv_i])
          strings::StrAppend(&sp_buf, p, ", ");
        VLOG(2) << "subdiv_i=" << subdiv_i << " perm=" << sp_buf;
      }
    }

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
        VLOG(2) << "dev=" << dev_name;
        col_params_.instance.device_names.push_back(dev_name);
        col_params_.instance.task_names.push_back(task_name);
        col_params_.task.is_local.push_back(true);
      }
    }
    for (int wi = 0; wi < num_workers; wi++) {
      for (int di = 0; di < num_devices_per_worker; di++) {
        int default_rank = wi * num_devices_per_worker + di;
        instances_.push_back(new DeviceInstance(
            default_rank, col_params_.instance.device_names[default_rank],
            device_type, this));
      }
    }
  }

  typedef std::function<void(Tensor*)> InitFunc;

  void Broadcast(bool forward_input) {
    VLOG(2) << "#instances=" << instances_.size();
    std::atomic<int> done(0);
    for (auto di : instances_) {
      SchedClosure([di, forward_input, &done] {
        di->DoBroadcast(forward_input);
        ++done;
      });
    }
    while (done < instances_.size()) {
      if (stop_) break;
      Env::Default()->SleepForMicroseconds(1000);
    }
  }

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

  std::unique_ptr<OpKernel> GetCollectiveBcastSend(
      const CollectiveParams& params, Tensor* input,
      const DeviceType& device_type, DeviceBase* device) {
    mutex_lock l(mu_);
    NodeDef node_def;
    NodeDefBuilder builder(
        strings::StrCat("collective_bcast_send_", bcast_send_counter_++),
        "CollectiveBcastSend");
    TF_CHECK_OK(builder.Attr("T", input->dtype())
                    .Attr("group_size", params.group.group_size)
                    .Attr("group_key", params.group.group_key)
                    .Attr("instance_key", params.instance.instance_key)
                    .Attr("shape", input->shape())
                    .Input(FakeInput(params.instance.data_type))
                    .Finalize(&node_def));
    return GetKernel(node_def, device_type, device);
  }

  std::unique_ptr<OpKernel> GetCollectiveBcastRecv(
      const CollectiveParams& params, const TensorShape& shape,
      const DeviceType& device_type, DeviceBase* device) {
    mutex_lock l(mu_);
    NodeDef node_def;
    NodeDefBuilder builder(
        strings::StrCat("collective_bcast_recv_", bcast_recv_counter_++),
        "CollectiveBcastRecv");
    TF_CHECK_OK(builder.Attr("T", params.instance.data_type)
                    .Attr("group_size", params.group.group_size)
                    .Attr("group_key", params.group.group_key)
                    .Attr("instance_key", params.instance.instance_key)
                    .Attr("shape", shape)
                    .Finalize(&node_def));
    return GetKernel(node_def, device_type, device);
  }

  template <typename T>
  void RunTest(DataType dtype, const DeviceType& device_type, int num_workers,
               int num_devices, int tensor_len, int fail_after,
               bool forward_input) {
    Init(num_workers, num_devices, dtype, device_type, fail_after);

    // Initialize each instance tensor with distinct values.
    for (int di = 0; di < instances_.size(); ++di) {
      DeviceInstance* instance = instances_[di];
      instance->InitTensor(
          dtype, TensorShape({tensor_len}), [di, dtype](Tensor* t) {
            for (size_t i = 0; i < t->NumElements(); ++i) {
              // The cast is necessary to prevent clang-tidy from insisting
              // that a faster non-open source function be substituted.
              float value = pow(10, static_cast<double>(di)) * i;
              t->flat<T>()(i) = value;
            }
          });
    }

    // Copy the expected value from the broadcast source tensor
    std::vector<T> expected(tensor_len, 0.0);
    const CollectiveParams& cp = instances_[0]->col_params_;
    int broadcast_dev_id =
        cp.instance.impl_details.subdiv_permutations
            [0][cp.instance.impl_details.subdiv_source_rank[0]];
    const Tensor* t = &instances_[broadcast_dev_id]->tensor_;
    Tensor cpu_copy(dtype, TensorShape({tensor_len}));
    if (device_type == DEVICE_GPU) {
      Notification notification;
      Device* dev = instances_[broadcast_dev_id]->device_;
      auto* dev_info = dev->tensorflow_gpu_device_info();
      CHECK(dev_info);
      dev_info->default_context->CopyDeviceTensorToCPU(
          t, "" /*tensor_name*/, dev, &cpu_copy,
          [this, &notification](Status s) {
            TF_CHECK_OK(s);
            notification.Notify();
          });
      notification.WaitForNotification();
      t = &cpu_copy;
    }
    for (size_t i = 0; i < t->NumElements(); ++i) {
      expected[i] = t->flat<T>()(i);
    }

    Broadcast(forward_input);

    // At this point all of the ops have terminated.
    for (int di = 0; di < instances_.size(); ++di) {
      if (!instances_[di]->status_.ok()) {
        ASSERT_GT(fail_after, 0);
        ASSERT_EQ(instances_[di]->status_.error_message(),
                  "Deliberate failure");
        mutex_lock l(mu_);
        ++failure_count_;
        continue;
      }
      Tensor* inst = &instances_[di]->tensor_;
      Tensor actual(dtype, TensorShape({tensor_len}));
      if (device_type_ == DEVICE_CPU) {
        CHECK(actual.CopyFrom(*inst, inst->shape()));
      } else if (device_type_ == DEVICE_GPU) {
        Notification notification;
        Device* dev = instances_[di]->device_;
        auto* dev_info = dev->tensorflow_gpu_device_info();
        CHECK(dev_info);
        dev_info->default_context->CopyDeviceTensorToCPU(
            inst, "" /*tensor_name*/, dev, &actual,
            [this, &notification](Status s) {
              TF_CHECK_OK(s);
              notification.Notify();
            });
        notification.WaitForNotification();
      }
      for (int i = 0; i < tensor_len; ++i) {
        switch (dtype) {
          case DT_FLOAT:
            EXPECT_FLOAT_EQ(expected[i], actual.template flat<T>()(i))
                << "Mismatch at device " << di << " index " << i;
            break;
          case DT_DOUBLE:
            EXPECT_DOUBLE_EQ(expected[i], actual.template flat<T>()(i))
                << "Mismatch at device " << di << " index " << i;
            break;
          case DT_INT32:
          case DT_INT64:
            EXPECT_EQ(expected[i], actual.template flat<T>()(i))
                << "Mismatch at device " << di << " index " << i;
            break;
          default:
            LOG(FATAL) << "unimplemented";
        }
      }
    }

    // Note that the order of operations during broadcast is
    // non-deterministic and unlike the reduce case some Ops in the
    // instance may succeed while others fail, even if a transmission
    // failure occurs early in the operation chain.  So, when an abort
    // is specified we need to verify that at least one Op fails with
    // the expected status and any Op that succeeds yeilds the correct
    // value.
    if (fail_after > 0) {
      mutex_lock l(mu_);
      EXPECT_GT(failure_count_, 0);
    }
  }

  void RunSubdivPermsTest(
      CollectiveParams* cp,
      const std::vector<std::vector<int>>& expected_subdiv_perms,
      const std::vector<int>& expected_subdiv_rank,
      const std::vector<int>& expected_subdiv_source_rank) {
    col_exec_ = nullptr;
    cp->instance.impl_details.subdiv_permutations.clear();
    cp->subdiv_rank.clear();
    cp->instance.impl_details.subdiv_source_rank.clear();
    // Create a stub broadcaster only for testing param initialization.
    HierarchicalTreeBroadcaster broadcaster;
    TF_CHECK_OK(broadcaster.InitializeCollectiveParams(cp));
    EXPECT_EQ(expected_subdiv_perms,
              cp->instance.impl_details.subdiv_permutations);
    EXPECT_EQ(expected_subdiv_rank, cp->subdiv_rank);
    EXPECT_EQ(expected_subdiv_source_rank,
              cp->instance.impl_details.subdiv_source_rank);
  }

  void PrepColParamsForSubdivPermsTest(CollectiveParams* cp, int num_tasks,
                                       int num_gpus) {
    cp->group.device_type = DeviceType("GPU");
    cp->group.num_tasks = num_tasks;
    cp->group.group_size = num_tasks * num_gpus;
    cp->instance.type = BROADCAST_COLLECTIVE;
    cp->instance.impl_details.collective_name = "HierarchicalTreeBroadcast";
    for (int ti = 0; ti < num_tasks; ti++) {
      string task_name = strings::StrCat("/job:worker/replica:0/task:", ti);
      for (int di = 0; di < num_gpus; di++) {
        string dev_name = strings::StrCat(task_name, "/device:GPU:", di);
        cp->instance.task_names.push_back(task_name);
        cp->instance.device_names.push_back(dev_name);
      }
    }
  }

  class DeviceInstance {
   public:
    DeviceInstance(int rank, const string& dev_name,
                   const DeviceType& device_type,
                   HierarchicalTreeBroadcasterTest* parent)
        : parent_(parent),
          dev_name_(dev_name),
          device_type_(device_type),
          rank_(rank) {
      TF_CHECK_OK(parent_->dev_mgr_->LookupDevice(dev_name, &device_));
      col_params_.name = parent_->col_params_.name;
      col_params_.instance.data_type = parent_->col_params_.instance.data_type;
      col_params_.group.group_key = parent_->col_params_.group.group_key;
      col_params_.instance.instance_key =
          parent_->col_params_.instance.instance_key;
      col_params_.group.device_type = parent_->col_params_.group.device_type;
      col_params_.group.group_size = parent_->col_params_.group.group_size;
      col_params_.instance.device_names =
          parent_->col_params_.instance.device_names;
      col_params_.instance.task_names =
          parent_->col_params_.instance.task_names;
      col_params_.task.is_local = parent_->col_params_.task.is_local;
      col_params_.instance.impl_details.subdiv_permutations =
          parent_->col_params_.instance.impl_details.subdiv_permutations;
      col_params_.subdiv_rank = parent_->col_params_.subdiv_rank;

      int group_size = col_params_.group.group_size;
      CHECK_EQ(group_size, col_params_.instance.device_names.size());
      // Default rank is order in device_names.
      col_params_.default_rank = rank;

      auto& impl = col_params_.instance.impl_details;
      size_t num_subdivs = impl.subdiv_permutations.size();
      impl.subdiv_source_rank.resize(num_subdivs, 0);
      col_params_.subdiv_rank.resize(num_subdivs);
      for (size_t si = 0; si < num_subdivs; si++) {
        int perm_rank = -1;
        for (int i = 0; i < group_size; i++) {
          if (rank == impl.subdiv_permutations[si][i]) {
            perm_rank = i;
            break;
          }
        }
        col_params_.subdiv_rank[si] = perm_rank;
      }
      string rank_buf;
      for (int r : col_params_.subdiv_rank) {
        strings::StrAppend(&rank_buf, r, ", ");
      }
      VLOG(1) << "default=" << rank << " subdiv_ranks=" << rank_buf;

      col_params_.is_source =
          col_params_.subdiv_rank[0] == impl.subdiv_source_rank[0];
    }

    void InitTensor(DataType dtype, const TensorShape& shape,
                    const InitFunc& f) {
      tensor_ =
          Tensor(device_->GetAllocator(AllocatorAttributes()), dtype, shape);
      if (device_type_ == DEVICE_CPU) {
        f(&tensor_);
      } else if (device_type_ == DEVICE_GPU) {
        Tensor cpu_tensor(dtype, shape);
        f(&cpu_tensor);
        Notification notification;
        auto* dev_info = device_->tensorflow_gpu_device_info();
        CHECK(dev_info);
        dev_info->default_context->CopyCPUTensorToDevice(
            &cpu_tensor, device_, &tensor_, [this, &notification](Status s) {
              TF_CHECK_OK(s);
              notification.Notify();
            });
        notification.WaitForNotification();
      } else {
        LOG(FATAL) << "Unsupported device_type " << device_type_;
      }
    }

    void DoBroadcast(bool forward_input) {
      // Prepare an OpKernelContext.
      OpKernelContext::Params op_params;
      op_params.step_id = parent_->step_id_;
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
      int forward_from[] = {OpKernelContext::Params::kNeverForward};
      if (forward_input) forward_from[0] = 0;
      if (col_params_.is_source) {
        op_params.forward_from_array = &forward_from[0];
      }
      AllocatorAttributes generic_alloc_attr;
      op_params.output_attr_array = &generic_alloc_attr;
      std::unique_ptr<OpKernel> op =
          col_params_.is_source
              ? parent_->GetCollectiveBcastSend(col_params_, &tensor_,
                                                DEVICE_CPU, device_)
              : parent_->GetCollectiveBcastRecv(col_params_, tensor_.shape(),
                                                DEVICE_CPU, device_);
      op_params.op_kernel = op.get();
      OpKernelContext ctx(&op_params, 1);

      Tensor* output_tensor_ptr = nullptr;
      if (col_params_.is_source) {
        TF_CHECK_OK(ctx.forward_input_or_allocate_output(
            {0}, 0, tensor_.shape(), &output_tensor_ptr));
      } else {
        TF_CHECK_OK(
            ctx.allocate_output(0, tensor_.shape(), &output_tensor_ptr));
      }
      CHECK_EQ(output_tensor_ptr, ctx.mutable_output(0));
      const Tensor* input_tensor_ptr =
          col_params_.is_source ? &tensor_ : nullptr;

      // Prepare a Broadcaster instance.
      string exec_key =
          strings::StrCat(col_params_.instance.instance_key, ":0:0");
      HierarchicalTreeBroadcaster broadcaster;
      CollectiveContext col_ctx(parent_->col_exec_, parent_->dev_mgr_.get(),
                                &ctx, &op_params, col_params_, exec_key,
                                kStepId, input_tensor_ptr, output_tensor_ptr);
      TF_CHECK_OK(broadcaster.InitializeCollectiveContext(&col_ctx));

      // Run the broadcast.
      broadcaster.Run([this](Status s) { status_ = s; });
      if (status_.ok()) {
        CHECK(tensor_.CopyFrom(*ctx.mutable_output(0), tensor_.shape()));
      }

      dev_ctx->Unref();
    }

    HierarchicalTreeBroadcasterTest* parent_;
    string dev_name_;
    DeviceType device_type_ = DEVICE_CPU;
    int rank_;
    Tensor tensor_;
    Device* device_;
    CollectiveParams col_params_;
    Status status_;
  };  // class DeviceInstance

  bool stop_ = false;
  int64 step_id_ = kStepId;
  int broadcast_dev_id_ = 0;
  DeviceType device_type_;
  TestCollectiveExecutorMgr col_exec_mgr_;
  CollectiveExecutor* col_exec_ = nullptr;
  CollectiveRemoteAccessLocal* rma_;
  std::unique_ptr<DeviceResolverLocal> dev_resolver_;
  std::vector<DeviceInstance*> instances_;
  CollectiveParams col_params_;
  std::vector<tensorflow::Device*> gpu_devices_;
  std::unique_ptr<tensorflow::DeviceMgr> dev_mgr_;
  std::unique_ptr<string> gpu_ring_order_;
  mutex mu_;
  int bcast_recv_counter_ GUARDED_BY(mu_) = 0;
  int bcast_send_counter_ GUARDED_BY(mu_) = 0;
  int failure_count_ GUARDED_BY(mu_) = 0;
};

TEST_F(HierarchicalTreeBroadcasterTest, InitializeParams1Task8GPU) {
  CollectiveParams cp;
  PrepColParamsForSubdivPermsTest(&cp, 1, 8);

  // source 0 device 0
  cp.source_rank = 0;
  cp.default_rank = 0;
  RunSubdivPermsTest(&cp, {{0, 1, 2, 3, 4, 5, 6, 7}}, {0}, {0});

  // source 2 device 2
  cp.source_rank = 2;
  cp.default_rank = 2;
  RunSubdivPermsTest(&cp, {{0, 1, 2, 3, 4, 5, 6, 7}}, {2}, {2});

  // source 2 device 0
  cp.source_rank = 2;
  cp.default_rank = 0;
  RunSubdivPermsTest(&cp, {{0, 1, 2, 3, 4, 5, 6, 7}}, {0}, {2});
}

TEST_F(HierarchicalTreeBroadcasterTest, InitializeParams4Tasks8GPU) {
  CollectiveParams cp;
  PrepColParamsForSubdivPermsTest(&cp, 4, 8);

  // source 0 device 0
  cp.source_rank = 0;
  cp.default_rank = 0;
  RunSubdivPermsTest(&cp,
                     {{0, 8, 16, 24},
                      {0, 1, 2, 3, 4, 5, 6, 7},
                      {8, 9, 10, 11, 12, 13, 14, 15},
                      {16, 17, 18, 19, 20, 21, 22, 23},
                      {24, 25, 26, 27, 28, 29, 30, 31}},
                     {0, 0, -1, -1, -1}, {0, 0, 0, 0, 0});

  // source 2 device 0
  cp.source_rank = 2;
  cp.default_rank = 0;
  RunSubdivPermsTest(&cp,
                     {{2, 8, 16, 24},
                      {0, 1, 2, 3, 4, 5, 6, 7},
                      {8, 9, 10, 11, 12, 13, 14, 15},
                      {16, 17, 18, 19, 20, 21, 22, 23},
                      {24, 25, 26, 27, 28, 29, 30, 31}},
                     {-1, 0, -1, -1, -1}, {0, 2, 0, 0, 0});

  // source 9 device 9
  cp.source_rank = 9;
  cp.default_rank = 9;
  RunSubdivPermsTest(&cp,
                     {{0, 9, 16, 24},
                      {0, 1, 2, 3, 4, 5, 6, 7},
                      {8, 9, 10, 11, 12, 13, 14, 15},
                      {16, 17, 18, 19, 20, 21, 22, 23},
                      {24, 25, 26, 27, 28, 29, 30, 31}},
                     {1, -1, 1, -1, -1}, {1, 0, 1, 0, 0});
}

TEST_F(HierarchicalTreeBroadcasterTest, InitializeParams4TasksVariableGPU) {
  CollectiveParams cp;
  int num_tasks = 4;
  cp.group.device_type = DeviceType("GPU");
  cp.group.num_tasks = num_tasks;
  cp.group.group_size = 0;
  cp.instance.type = BROADCAST_COLLECTIVE;
  cp.instance.impl_details.collective_name = "HierarchicalTreeBroadcast";
  std::vector<int> dev_per_task = {4, 4, 6, 8};
  for (int ti = 0; ti < cp.group.num_tasks; ti++) {
    string task_name = strings::StrCat("/job:worker/replica:0/task:", ti);
    for (int di = 0; di < dev_per_task[ti]; di++) {
      string dev_name = strings::StrCat(task_name, "/device:GPU:", di);
      cp.instance.task_names.push_back(task_name);
      cp.instance.device_names.push_back(dev_name);
      cp.group.group_size++;
    }
  }

  // source 0 device 0
  cp.source_rank = 0;
  cp.default_rank = 0;
  RunSubdivPermsTest(&cp,
                     {{0, 4, 8, 14},
                      {0, 1, 2, 3},
                      {4, 5, 6, 7},
                      {8, 9, 10, 11, 12, 13},
                      {14, 15, 16, 17, 18, 19, 20, 21}},
                     {0, 0, -1, -1, -1}, {0, 0, 0, 0, 0});

  // source 2 device 0
  cp.source_rank = 2;
  cp.default_rank = 0;
  RunSubdivPermsTest(&cp,
                     {{2, 4, 8, 14},
                      {0, 1, 2, 3},
                      {4, 5, 6, 7},
                      {8, 9, 10, 11, 12, 13},
                      {14, 15, 16, 17, 18, 19, 20, 21}},
                     {-1, 0, -1, -1, -1}, {0, 2, 0, 0, 0});

  // source 9 device 5
  cp.source_rank = 9;
  cp.default_rank = 5;
  RunSubdivPermsTest(&cp,
                     {{0, 4, 9, 14},
                      {0, 1, 2, 3},
                      {4, 5, 6, 7},
                      {8, 9, 10, 11, 12, 13},
                      {14, 15, 16, 17, 18, 19, 20, 21}},
                     {-1, -1, 1, -1, -1}, {2, 0, 0, 1, 0});
}

// TODO(b/113171733): change to use TEST_P.
// Tests of full broadcast algorithm, with different device and
// data types.
// B = data element type
// T = device type
// W = number of workers
// D = number of devices per worker
// L = tensor length
// A = abort after count
#define DEF_TEST(B, T, W, D, L, A, F)                                      \
  TEST_F(HierarchicalTreeBroadcasterTest,                                  \
         DaTy##B##_DevTy##T##_Wkr##W##_Dev##D##_Len##L##_Abt##A##_Fw##F) { \
    DataType dtype = DT_##B;                                               \
    switch (dtype) {                                                       \
      case DT_FLOAT: {                                                     \
        RunTest<float>(dtype, DEVICE_##T, W, D, L, A, F);                  \
      } break;                                                             \
      case DT_DOUBLE: {                                                    \
        RunTest<double>(dtype, DEVICE_##T, W, D, L, A, F);                 \
      } break;                                                             \
      case DT_INT32: {                                                     \
        RunTest<int32>(dtype, DEVICE_##T, W, D, L, A, F);                  \
      } break;                                                             \
      case DT_INT64: {                                                     \
        RunTest<int64>(dtype, DEVICE_##T, W, D, L, A, F);                  \
      } break;                                                             \
      default:                                                             \
        LOG(FATAL) << "Unimplemented";                                     \
    }                                                                      \
  }

#ifndef GOOGLE_CUDA
//       B      T    W  D  L  A  F
DEF_TEST(FLOAT, CPU, 1, 2, 1, 0, false)
DEF_TEST(FLOAT, CPU, 1, 2, 1001, 0, true)
DEF_TEST(FLOAT, CPU, 2, 1, 128, 0, false)
DEF_TEST(FLOAT, CPU, 2, 4, 128, 0, true)
DEF_TEST(FLOAT, CPU, 2, 8, 4095, 0, false)
DEF_TEST(FLOAT, CPU, 4, 4, 1045991, 0, true)

DEF_TEST(DOUBLE, CPU, 2, 4, 128, 0, false)
DEF_TEST(INT32, CPU, 2, 4, 128, 0, true)
DEF_TEST(INT64, CPU, 2, 4, 128, 0, false)

// Failure cases
DEF_TEST(FLOAT, CPU, 2, 4, 128, 1, true)
DEF_TEST(FLOAT, CPU, 2, 4, 128, 5, false)
#endif

#ifdef GOOGLE_CUDA
// Can only set W=1 for GPU tests.
//       B      T    W  D  L  A  F
DEF_TEST(FLOAT, GPU, 1, 2, 1, 0, true)
DEF_TEST(FLOAT, GPU, 1, 2, 33, 0, false)
DEF_TEST(FLOAT, GPU, 1, 3, 64, 0, true)
DEF_TEST(FLOAT, GPU, 1, 8, 1001, 0, false)
DEF_TEST(FLOAT, GPU, 1, 8, 4095, 0, true)
DEF_TEST(FLOAT, GPU, 1, 8, 1045991, 0, false)

DEF_TEST(DOUBLE, GPU, 1, 8, 1001, 0, true)
DEF_TEST(INT64, GPU, 1, 8, 1001, 0, false)

// Failure cases
DEF_TEST(FLOAT, GPU, 1, 8, 128, 6, true)
#endif

}  // namespace
}  // namespace tensorflow
