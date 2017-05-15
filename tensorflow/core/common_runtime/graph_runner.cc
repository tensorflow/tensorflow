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

#include "tensorflow/core/common_runtime/graph_runner.h"

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/memory_types.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

namespace {

std::unique_ptr<Device> GetCPUDevice(Env* env) {
  std::vector<Device*> devices;
  SessionOptions session_options;
  session_options.env = env;
  Status s = DeviceFactory::GetFactory(DEVICE_CPU)
                 ->CreateDevices(session_options, "", &devices);
  if (s.ok() && !devices.empty()) {
    return std::unique_ptr<Device>(devices[0]);
  }
  return nullptr;
}

// A simple rendezvous class.
// Assumes a single sender and a single receiver, no duplicate sends, and no
// sends of dead tensors.
class SimpleRendezvous : public Rendezvous {
 public:
  explicit SimpleRendezvous() {}

  Status Send(const ParsedKey& parsed, const Args& send_args, const Tensor& val,
              const bool is_dead) override {
    if (is_dead) {
      return errors::Internal("Send of a dead tensor");
    }

    mutex_lock l(mu_);
    string edge_name = parsed.edge_name.ToString();
    if (table_.count(edge_name) > 0) {
      return errors::Internal("Send of an already sent tensor");
    }
    table_[edge_name] = val;
    return Status::OK();
  }

  void RecvAsync(const ParsedKey& parsed, const Args& recv_args,
                 DoneCallback done) override {
    Tensor tensor;
    Status status = Status::OK();
    {
      string key = parsed.edge_name.ToString();
      mutex_lock l(mu_);
      if (table_.count(key) <= 0) {
        status = errors::Internal("Did not find key ", key);
      } else {
        tensor = table_[key];
      }
    }
    done(status, Args{}, recv_args, tensor, false);
  }

  void StartAbort(const Status& status) override {}

 private:
  typedef std::unordered_map<string, Tensor> Table;

  mutex mu_;
  Table table_ GUARDED_BY(mu_);
};

}  // namespace

GraphRunner::GraphRunner(Env* env) : cpu_device_(GetCPUDevice(env)) {}

GraphRunner::~GraphRunner() {}

Status GraphRunner::Run(Graph* graph, FunctionLibraryRuntime* function_library,
                        const NamedTensorList& inputs,
                        const std::vector<string>& output_names,
                        std::vector<Tensor>* outputs) {
  if (cpu_device_ == nullptr) {
    return errors::NotFound("Cannot find a device for GraphRunner.");
  }

  // TODO(vrv): Instead of copying the entire graph, consider modifying
  // the existing graph, and then removing those removed edges.
  // prior to returning.
  std::unique_ptr<Graph> graph_to_run(new Graph(graph->op_registry()));
  CopyGraph(*graph, graph_to_run.get());

  SimpleRendezvous* rendez = new SimpleRendezvous;
  core::ScopedUnref rendez_unref(rendez);

  // Extract the input names and keys, and feed in the inputs.
  std::vector<string> input_names;
  for (const auto& in : inputs) {
    const string& tensor_name = in.first;
    input_names.emplace_back(tensor_name);
    string full_key = Rendezvous::CreateKey("/cpu:0", 1, "/cpu:1", tensor_name,
                                            FrameAndIter(0, 0));
    Rendezvous::ParsedKey parsed;
    TF_RETURN_IF_ERROR(Rendezvous::ParseKey(full_key, &parsed));
    TF_RETURN_IF_ERROR(rendez->Send(parsed, Rendezvous::Args(), in.second,
                                    false /* is_dead */));
  }

  // Call RewriteGraphForExecution
  subgraph::RewriteGraphMetadata metadata;
  TF_RETURN_IF_ERROR(subgraph::RewriteGraphForExecution(
      graph_to_run.get(), input_names, output_names, {} /* target nodes */,
      cpu_device_->attributes(), false /* use_function_convention */,
      &metadata));

  // Create the local executor and the Rendezvous for fetching back the
  // constants.

  // Run operators on the local thread. We should not need concurrency here; we
  // should not be running expensive operators.
  auto runner = [](Executor::Args::Closure c) { c(); };

  // Take ownership and pass to NewLocalExecutor
  Graph* g = graph_to_run.release();

  LocalExecutorParams params;
  // The ownership of the output tensors are bound to this device's lifetime.
  params.device = cpu_device_.get();
  params.function_library = function_library;
  params.create_kernel = [this, g](const NodeDef& ndef, OpKernel** kernel) {
    return CreateNonCachedKernel(cpu_device_.get(), nullptr, ndef,
                                 g->versions().producer(), kernel);
  };
  params.delete_kernel = [](OpKernel* kernel) { delete kernel; };

  Executor* executor;
  TF_RETURN_IF_ERROR(NewLocalExecutor(params, g, &executor));
  std::unique_ptr<Executor> executor_unref(executor);

  Executor::Args args;
  // NOTE: we could take a step id as an argument, but currently
  // there is no need since we never trace the running of a graph
  // called via this method.
  args.step_id = LogMemory::CONSTANT_FOLDING_STEP_ID;
  args.runner = runner;
  args.rendezvous = rendez;

  // Run the graph.
  TF_RETURN_IF_ERROR(executor->Run(args));

  outputs->resize(output_names.size());
  for (size_t i = 0; i < output_names.size(); ++i) {
    const string& output_key = Rendezvous::CreateKey(
        "/cpu:0", 1, "/cpu:1", output_names[i], FrameAndIter(0, 0));
    Rendezvous::ParsedKey parsed;
    TF_RETURN_IF_ERROR(Rendezvous::ParseKey(output_key, &parsed));
    bool is_dead;
    Tensor output_tensor;
    TF_RETURN_IF_ERROR(
        rendez->Recv(parsed, Rendezvous::Args(), &output_tensor, &is_dead));
    // Does a deep copy so that ownership of the tensor isn't tied to the
    // allocator of the cpu device we created above. The allocator could be
    // deleted along with the device.
    (*outputs)[i] = tensor::DeepCopy(output_tensor);
  }

  return Status::OK();
}

}  // namespace tensorflow
