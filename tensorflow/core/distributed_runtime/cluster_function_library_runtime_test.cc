/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/distributed_runtime/cluster_function_library_runtime.h"

#include <map>
#include <memory>

#include "absl/synchronization/notification.h"
#include "tensorflow/core/common_runtime/function_testlib.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_channel.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_testlib.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_cache.h"
#include "tensorflow/core/distributed_runtime/worker_session.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/equal_graph_def.h"

namespace tensorflow {

class ClusterFunctionLibraryRuntimeTest : public ::testing::Test {
 public:
  ClusterFunctionLibraryRuntimeTest() {
    SessionOptions options;
    TF_CHECK_OK(test::TestCluster::MakeTestCluster(
        test::TestClusterConfig().Options(options).Jobs(
            {test::TestJob{"localhost", 2}}),
        &cluster_));
    GrpcChannelSpec spec;

    std::map<int, string> host_ports;
    int i = 0;
    for (const auto& target : cluster_->targets("localhost")) {
      host_ports[i++] = target;
    }

    TF_CHECK_OK(spec.AddHostPortsJob("localhost", host_ports));
    ChannelCreationFunction channel_func =
        ConvertToChannelCreationFunction(NewHostPortGrpcChannel);
    grpc_worker_env_.reset(CreateGrpcWorkerEnv());
    std::shared_ptr<GrpcChannelCache> channel_cache(
        NewGrpcChannelCache(spec, channel_func));
    std::unique_ptr<WorkerCacheInterface> worker_cache(
        NewGrpcWorkerCache(channel_cache, grpc_worker_env_.get()));

    worker_session_ = std::make_unique<WorkerSession>(
        "cluster_test_session", "/job:localhost/replica:0/task:0",
        std::move(worker_cache), std::unique_ptr<DeviceMgr>(),
        std::unique_ptr<GraphMgr>(), nullptr,
        [](WorkerSession* worker_session, bool called,
           DeviceMgr* remote_device_mgr) { return nullptr; });

    cluster_flr_ = std::make_unique<ClusterFunctionLibraryRuntime>(
        worker_session_.get(), true, nullptr);
  }

  absl::Status ConstructFunctionGraphHelper(
      const OpDef& sig, test::function::Attrs attrs,
      const FunctionLibraryRuntime::InstantiateOptions& options,
      const FunctionLibraryDefinition& lib_def, GraphDef* g,
      std::vector<string>* send_keys, std::vector<string>* recv_keys) {
    return ClusterFunctionLibraryRuntime::ConstructFunctionGraph(
        sig, attrs, options, lib_def, g, send_keys, recv_keys);
  }

  void Instantiate(const string& function_name,
                   const FunctionLibraryDefinition& lib_def,
                   test::function::Attrs attrs,
                   const FunctionLibraryRuntime::InstantiateOptions& options,
                   FunctionLibraryRuntime::LocalHandle* local_handle,
                   FunctionLibraryRuntime::DoneCallback done) {
    cluster_flr_->Instantiate(function_name, lib_def, attrs, options,
                              local_handle, done);
  }

  absl::Status InstantiateAndRun(
      const string& function_name, const FunctionLibraryDefinition& lib_def,
      test::function::Attrs attrs,
      const FunctionLibraryRuntime::InstantiateOptions& options,
      const std::vector<Tensor>& args, std::vector<Tensor*> rets) {
    FunctionLibraryRuntime::LocalHandle handle;
    absl::Status status;
    absl::Notification instantiate_done;
    cluster_flr_->Instantiate(
        function_name, lib_def, attrs, options, &handle,
        [&status, &instantiate_done](const absl::Status& s) {
          status = s;
          instantiate_done.Notify();
        });
    instantiate_done.WaitForNotification();
    if (!status.ok()) {
      return status;
    }

    absl::Notification done;
    FunctionLibraryRuntime::Options opts;
    std::vector<Tensor> out;
    cluster_flr_->Run(opts, handle, args, &out,
                      [&status, &done](const absl::Status& s) {
                        status = s;
                        done.Notify();
                      });
    done.WaitForNotification();
    if (!status.ok()) {
      return status;
    }
    CHECK_EQ(rets.size(), out.size());
    for (size_t i = 0; i < rets.size(); ++i) {
      *rets[i] = out[i];
    }

    return absl::OkStatus();
  }

 protected:
  std::unique_ptr<test::TestCluster> cluster_;
  std::unique_ptr<WorkerSession> worker_session_;
  std::unique_ptr<ClusterFunctionLibraryRuntime> cluster_flr_;
  std::unique_ptr<GrpcWorkerEnv> grpc_worker_env_;
};

TEST_F(ClusterFunctionLibraryRuntimeTest, ConstructFunctionGraph) {
  GraphDef actual;
  std::vector<string> send_keys, recv_keys;
  FunctionDefLibrary proto;
  *(proto.add_function()) = test::function::Swap();
  FunctionLibraryDefinition lib_def(OpRegistry::Global(), proto);

  FunctionLibraryRuntime::InstantiateOptions instantiate_opts;
  instantiate_opts.target = "/job:a/replica:0/task:0/device:CPU:0";
  TF_CHECK_OK(ConstructFunctionGraphHelper(
      test::function::Swap().signature(), {{"T", DT_FLOAT}}, instantiate_opts,
      lib_def, &actual, &send_keys, &recv_keys));
  GraphDef expected;
  protobuf::TextFormat::ParseFromString(R"(
node {
  name: "_recv_i0_0"
  op: "_Recv"
  device: "/job:a/replica:0/task:0/device:CPU:0"
  attr {
    key: "client_terminated"
    value {
      b: true
    }
  }
  attr {
    key: "recv_device"
    value {
      s: "/job:a/replica:0/task:0/device:CPU:0"
    }
  }
  attr {
    key: "send_device"
    value {
      s: "/job:a/replica:0/task:0/device:CPU:0"
    }
  }
  attr {
    key: "send_device_incarnation"
    value {
      i: 1
    }
  }
  attr {
    key: "tensor_name"
    value {
      s: "i0"
    }
  }
  attr {
    key: "tensor_type"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "_recv_i1_1"
  op: "_Recv"
  device: "/job:a/replica:0/task:0/device:CPU:0"
  attr {
    key: "client_terminated"
    value {
      b: true
    }
  }
  attr {
    key: "recv_device"
    value {
      s: "/job:a/replica:0/task:0/device:CPU:0"
    }
  }
  attr {
    key: "send_device"
    value {
      s: "/job:a/replica:0/task:0/device:CPU:0"
    }
  }
  attr {
    key: "send_device_incarnation"
    value {
      i: 1
    }
  }
  attr {
    key: "tensor_name"
    value {
      s: "i1"
    }
  }
  attr {
    key: "tensor_type"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Func/Swap/input/_0"
  op: "Identity"
  input: "_recv_i0_0"
  device: "/job:a/replica:0/task:0/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Func/Swap/input/_1"
  op: "Identity"
  input: "_recv_i1_1"
  device: "/job:a/replica:0/task:0/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Swap/o0"
  op: "Identity"
  input: "Func/Swap/input/_1"
  device: "/job:a/replica:0/task:0/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Swap/o1"
  op: "Identity"
  input: "Func/Swap/input/_0"
  device: "/job:a/replica:0/task:0/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Func/Swap/output/_2"
  op: "Identity"
  input: "Swap/o0"
  device: "/job:a/replica:0/task:0/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Func/Swap/output/_3"
  op: "Identity"
  input: "Swap/o1"
  device: "/job:a/replica:0/task:0/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "_send_o0_0"
  op: "_Send"
  input: "Func/Swap/output/_2"
  device: "/job:a/replica:0/task:0/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "client_terminated"
    value {
      b: true
    }
  }
  attr {
    key: "recv_device"
    value {
      s: "/job:a/replica:0/task:0/device:CPU:0"
    }
  }
  attr {
    key: "send_device"
    value {
      s: "/job:a/replica:0/task:0/device:CPU:0"
    }
  }
  attr {
    key: "send_device_incarnation"
    value {
      i: 1
    }
  }
  attr {
    key: "tensor_name"
    value {
      s: "o0"
    }
  }
}
node {
  name: "_send_o1_1"
  op: "_Send"
  input: "Func/Swap/output/_3"
  device: "/job:a/replica:0/task:0/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "client_terminated"
    value {
      b: true
    }
  }
  attr {
    key: "recv_device"
    value {
      s: "/job:a/replica:0/task:0/device:CPU:0"
    }
  }
  attr {
    key: "send_device"
    value {
      s: "/job:a/replica:0/task:0/device:CPU:0"
    }
  }
  attr {
    key: "send_device_incarnation"
    value {
      i: 1
    }
  }
  attr {
    key: "tensor_name"
    value {
      s: "o1"
    }
  }
}
)",
                                        &expected);
  TF_EXPECT_GRAPH_EQ(expected, actual);
}

// Disabling the following two tests since there seem to be some issues with
// GRPC bringing up multiple processes as sub-processes.
// More info at: https://github.com/grpc/grpc/issues/10142.
// TODO(rohanj): Enable tests when the grpc bug is fixed.
TEST_F(ClusterFunctionLibraryRuntimeTest, DISABLED_InstantiateAndRun) {
  FunctionDefLibrary proto;
  *(proto.add_function()) = test::function::XTimesTwoInt32();
  FunctionLibraryDefinition lib_def(OpRegistry::Global(), proto);
  FunctionLibraryRuntime::InstantiateOptions instantiate_opts;
  instantiate_opts.target = "/job:localhost/replica:0/task:1/cpu:0";

  Tensor y;
  auto x = test::AsTensor<int32>({1, 2, 3, 4});
  TF_EXPECT_OK(InstantiateAndRun("XTimesTwoInt32", lib_def, {},
                                 instantiate_opts, {x}, {&y}));
  test::ExpectTensorEqual<int32>(y, test::AsTensor<int32>({2, 4, 6, 8}));
}

TEST_F(ClusterFunctionLibraryRuntimeTest,
       DISABLED_InstantiateAndRunAttrSubstitution) {
  FunctionDefLibrary proto;
  *(proto.add_function()) = test::function::Swap();
  FunctionLibraryDefinition lib_def(OpRegistry::Global(), proto);
  FunctionLibraryRuntime::InstantiateOptions instantiate_opts;
  instantiate_opts.target = "/job:localhost/replica:0/task:1/cpu:0";
  Tensor y1, y2;
  auto x1 = test::AsTensor<float>({1, 2, 3, 4});
  auto x2 = test::AsTensor<float>({4, 3, 2, 1});
  TF_EXPECT_OK(InstantiateAndRun("Swap", lib_def, {{"T", DT_FLOAT}},
                                 instantiate_opts, {x1, x2}, {&y1, &y2}));
  test::ExpectTensorEqual<float>(y1, test::AsTensor<float>({4, 3, 2, 1}));
  test::ExpectTensorEqual<float>(y2, test::AsTensor<float>({1, 2, 3, 4}));
}

}  // namespace tensorflow
