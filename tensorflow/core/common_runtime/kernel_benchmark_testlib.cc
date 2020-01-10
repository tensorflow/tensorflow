#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/framework/op_segment.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session_options.h"

#if defined(PLATFORM_GOOGLE)
DECLARE_bool(brain_gpu_use_bfc_allocator);
#else
extern bool FLAGS_brain_gpu_use_bfc_allocator;
#endif

namespace tensorflow {
namespace test {

Benchmark::Benchmark(const string& device, Graph* g,
                     const SessionOptions* options, Graph* init) {
  RequireDefaultOps();

  FLAGS_brain_gpu_use_bfc_allocator = true;

  SessionOptions default_options;
  if (!options) {
    options = &default_options;
  }

  testing::StopTiming();
  string t = str_util::Uppercase(device);
  device_ =
      DeviceFactory::NewDevice(t, *options, "/job:localhost/replica:0/task:0");
  CHECK(device_) << "Could not create a " << device << " device";

  pool_ = new thread::ThreadPool(options->env, "blocking",
                                 port::NumSchedulableCPUs());

  auto runner = [this](std::function<void()> closure) {
    pool_->Schedule(closure);
  };

  rendez_ = NewLocalRendezvous();

  if (init) {
    Executor* init_exec;
    TF_CHECK_OK(NewLocalExecutor(
        {
            device_, nullptr, false,
            [this](const NodeDef& ndef, OpKernel** kernel) {
              return CreateNonCachedKernel(device_, nullptr, ndef, kernel);
            },
            [](OpKernel* kernel) { DeleteNonCachedKernel(kernel); },
        },
        init, &init_exec));
    Executor::Args args;
    args.rendezvous = rendez_;
    args.runner = runner;
    TF_CHECK_OK(init_exec->Run(args));
    delete init_exec;
  }

  TF_CHECK_OK(NewLocalExecutor(
      {
          device_,
          nullptr,
          false,
          [this](const NodeDef& ndef, OpKernel** kernel) {
            return CreateNonCachedKernel(device_, nullptr, ndef, kernel);
          },
          [](OpKernel* kernel) { DeleteNonCachedKernel(kernel); },
      },
      g, &exec_));
}

Benchmark::~Benchmark() {
  if (device_) {
    rendez_->Unref();
    delete exec_;
    delete device_;
    delete pool_;
  }
}

void Benchmark::Run(int iters) { RunWithArgs({}, {}, iters); }

string GetRendezvousKey(const Node* node) {
  string send_device;
  TF_CHECK_OK(GetNodeAttr(node->def(), "send_device", &send_device));
  string recv_device;
  TF_CHECK_OK(GetNodeAttr(node->def(), "recv_device", &recv_device));
  string tensor_name;
  TF_CHECK_OK(GetNodeAttr(node->def(), "tensor_name", &tensor_name));
  uint64 send_device_incarnation;
  TF_CHECK_OK(GetNodeAttr(node->def(), "send_device_incarnation",
                          reinterpret_cast<int64*>(&send_device_incarnation)));
  return Rendezvous::CreateKey(send_device, send_device_incarnation,
                               recv_device, tensor_name, FrameAndIter(0, 0));
}

void Benchmark::RunWithArgs(
    const std::vector<std::pair<const Node*, Tensor>>& inputs,
    const std::vector<const Node*>& outputs, int iters) {
  if (device_) {
    // Gets inputs' and outputs' rendezvous keys.
    std::vector<std::pair<string, Tensor>> in;
    for (const auto& p : inputs) {
      in.push_back({GetRendezvousKey(p.first), p.second});
    }
    std::vector<string> out;
    for (const auto& n : outputs) {
      out.push_back(GetRendezvousKey(n));
    }
    Tensor unused;  // In benchmark, we don't care the return value.
    bool is_dead;

    // Warm up
    Executor::Args args;
    args.rendezvous = rendez_;
    args.runner = [this](std::function<void()> closure) {
      pool_->Schedule(closure);
    };
    for (int i = 0; i < 3; ++i) {
      for (const auto& p : in) {
        rendez_->Send(p.first, Rendezvous::Args(), p.second, false);
      }
      TF_CHECK_OK(exec_->Run(args));
      for (const string& key : out) {
        rendez_->Recv(key, Rendezvous::Args(), &unused, &is_dead);
      }
    }
    TF_CHECK_OK(device_->Sync());

    testing::StartTiming();
    while (iters-- > 0) {
      for (const auto& p : in) {
        rendez_->Send(p.first, Rendezvous::Args(), p.second, false);
      }
      TF_CHECK_OK(exec_->Run(args));
      for (const string& key : out) {
        rendez_->Recv(key, Rendezvous::Args(), &unused, &is_dead);
      }
    }

    TF_CHECK_OK(device_->Sync());
    testing::StopTiming();
  }
}

}  // end namespace test
}  // end namespace tensorflow
