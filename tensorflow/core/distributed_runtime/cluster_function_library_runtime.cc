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

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/distributed_runtime/worker_session.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/protobuf/named_tensor.pb.h"
#include "tensorflow/core/protobuf/worker.pb.h"

namespace tensorflow {

/* static */
Status ClusterFunctionLibraryRuntime::ConstructFunctionGraph(
    const OpDef& sig, AttrSlice attrs,
    const FunctionLibraryRuntime::InstantiateOptions& options,
    const FunctionLibraryDefinition& flib_def, GraphDef* gdef,
    std::vector<string>* send_keys, std::vector<string>* recv_keys) {
  const string& target = options.target;
  const string& func_name = sig.name();
  const FunctionDef* func_def = flib_def.Find(sig.name());
  if (func_def == nullptr) {
    return errors::InvalidArgument("Function ", func_name,
                                   " not found in flib_def.");
  }

  // Build a smaller flib_def containing only the functions used by the given
  // function, plus that function itself.
  FunctionLibraryDefinition pruned_flib_def =
      flib_def.ReachableDefinitions(*func_def);
  TF_RETURN_IF_ERROR(pruned_flib_def.CopyFunctionDefFrom(func_name, flib_def));

  Graph g(pruned_flib_def);

  std::vector<Node*> input_nodes;
  input_nodes.reserve(sig.input_arg_size());

  // Construct recv nodes for each input argument.
  int i = 0;
  for (const auto& in : sig.input_arg()) {
    // Resolve the input type.
    bool is_type_list;
    DataTypeVector dtypes;
    TF_RETURN_IF_ERROR(ArgNumType(attrs, in, &is_type_list, &dtypes));
    // TODO(rohanj): Handle list and variadic number of attrs. Here and below.
    if (is_type_list || dtypes.size() > 1) {
      return errors::Unimplemented("Input arg: ", in.name(),
                                   " has a list type or variadic number of "
                                   "attrs. Currently unsupported.");
    }

    auto input_node_builder =
        NodeDefBuilder(strings::StrCat("_recv_", in.name(), "_", i), "_Recv")
            .Attr("tensor_type", dtypes[0])
            .Attr("tensor_name", in.name())
            .Attr("send_device", target)
            .Attr("recv_device", target)
            .Attr("send_device_incarnation", 1)
            .Attr("client_terminated", true)
            .Device(target);

    Node* input_node;
    TF_RETURN_IF_ERROR(
        NodeBuilder(input_node_builder).Finalize(&g, &input_node));
    input_nodes.push_back(input_node);

    // src_incarnation = 1 works because the transfer is across the same device.
    // TODO(rohanj): Find the src_incarnation for the remote device and set it.
    const string& key = Rendezvous::CreateKey(
        target, 1 /* src_incarnation */, target, in.name(), FrameAndIter(0, 0));
    send_keys->push_back(key);
    ++i;
  }

  NodeDef function_node_def;
  function_node_def.set_name(func_name);
  function_node_def.set_op(func_name);
  i = 0;
  function_node_def.set_device(target);
  for (const auto& p : attrs) {
    (*function_node_def.mutable_attr())[p.first] = p.second;
  }
  Status status;
  Node* function_node = g.AddNode(std::move(function_node_def), &status);
  TF_RETURN_IF_ERROR(status);
  for (size_t i = 0; i < input_nodes.size(); ++i) {
    g.AddEdge(input_nodes[i], 0, function_node, i);
  }

  // Construct output nodes for each output.
  i = 0;
  for (const auto& out : sig.output_arg()) {
    // Resolve the output type.
    bool is_type_list;
    DataTypeVector dtypes;
    TF_RETURN_IF_ERROR(ArgNumType(attrs, out, &is_type_list, &dtypes));
    // TODO(rohanj): Handle list and variadic number of attrs. Here and below.
    if (is_type_list || dtypes.size() > 1) {
      return errors::Unimplemented("Output arg: ", out.name(),
                                   " has a list type or variadic number of "
                                   "attrs. Currently unsupported.");
    }

    auto output_node_builder =
        NodeDefBuilder(strings::StrCat("_send_", out.name(), "_", i), "_Send")
            .Input(func_name, i, dtypes[0])
            .Attr("tensor_name", out.name())
            .Attr("send_device", target)
            .Attr("recv_device", target)
            .Attr("send_device_incarnation", 1)
            .Attr("client_terminated", true)
            .Device(target);

    Node* output_node;
    TF_RETURN_IF_ERROR(
        NodeBuilder(output_node_builder).Finalize(&g, &output_node));

    g.AddEdge(function_node, i, output_node, 0);

    const string& key =
        Rendezvous::CreateKey(target, 1 /* src_incarnation */, target,
                              out.name(), FrameAndIter(0, 0));
    recv_keys->push_back(key);
    ++i;
  }

  // Inline function node into the graph.
  InlineFunctionBodyOptions inline_options;
  inline_options.inlined_function_body_placer =
      InlinedFunctionBodyPlacer::SingleDevice();
  // When the remote call is a partition of a multi-device function, and the
  // Send/Recv nodes depend on the frame names in the original graph, we must
  // retain the original frame names. Since the graph contains a single function
  // call, we do not need to add a unique prefix to frame names inside the
  // inlined graph.
  inline_options.uniquify_frame_names = false;
  std::unique_ptr<FunctionBody> function_body;
  TF_RETURN_IF_ERROR(FunctionDefToBodyHelper(*func_def, attrs, &pruned_flib_def,
                                             &function_body));
  TF_RETURN_IF_ERROR(InlineFunctionBody(pruned_flib_def, &g, function_node,
                                        function_body.get(), inline_options));

  g.ToGraphDef(gdef);

  // Since we have inlined `function_node`, we can prune its function definition
  // from the library.
  *(gdef->mutable_library()) = flib_def.ReachableDefinitions(*gdef).ToProto();

  return Status::OK();
}

ClusterFunctionLibraryRuntime::~ClusterFunctionLibraryRuntime() {
  for (auto& function_data : function_data_) {
    worker_session_->worker_cache()->ReleaseWorker(function_data.target,
                                                   function_data.wi);
  }
}

void ClusterFunctionLibraryRuntime::Instantiate(
    const string& function_name, const FunctionLibraryDefinition& lib_def,
    AttrSlice attrs, const FunctionLibraryRuntime::InstantiateOptions& options,
    FunctionLibraryRuntime::LocalHandle* handle,
    FunctionLibraryRuntime::DoneCallback done) {
  auto target = options.target;
  VLOG(1) << "CFLR::Instantiate: " << function_name << " on " << target
          << " (this: " << this << ")";
  WorkerInterface* wi =
      worker_session_->worker_cache()->GetOrCreateWorker(target);

  if (wi == nullptr) {
    std::vector<string> workers;
    worker_session_->worker_cache()->ListWorkers(&workers);
    done(errors::InvalidArgument(
        "Could not find worker with target: ", target,
        " Available workers: ", absl::StrJoin(workers, ", ")));
    return;
  }

  // Make RPC and obtain a graph handle.
  GraphDef gdef;
  auto* send_keys = new std::vector<string>;
  auto* recv_keys = new std::vector<string>;
  auto construct_graph_fn = [&](const FunctionLibraryDefinition* lib_def) {
    const FunctionDef* fdef = lib_def->Find(function_name);
    const OpDef& sig = fdef->signature();
    TF_RETURN_IF_ERROR(ConstructFunctionGraph(sig, attrs, options, *lib_def,
                                              &gdef, send_keys, recv_keys));
    return Status::OK();
  };
  Status s;
  if (options.lib_def) {
    s = construct_graph_fn(options.lib_def);
  } else {
    s = construct_graph_fn(&lib_def);
  }
  if (!s.ok()) {
    done(s);
    return;
  }

  auto* req = new RegisterGraphRequest;
  req->set_session_handle(worker_session_->session_name());
  req->set_create_worker_session_called(create_worker_session_called_);
  *req->mutable_graph_def() = std::move(gdef);
  req->mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_do_function_inlining(true);
  auto* resp = new RegisterGraphResponse;

  wi->RegisterGraphAsync(
      req, resp,
      [this, handle, req, resp, wi, function_name, target, send_keys, recv_keys,
       done](const Status& status) {
        if (status.ok()) {
          mutex_lock l(mu_);
          *handle = function_data_.size();
          function_data_.push_back(FunctionData(resp->graph_handle(), target,
                                                wi, *send_keys, *recv_keys));
          VLOG(1) << "CFLR::Instantiate: [Success] " << function_name << " on "
                  << target << " (this: " << this << ")"
                  << " with handle: " << *handle;
        }
        done(status);
        delete recv_keys;
        delete send_keys;
        delete req;
        delete resp;
      });
}

void ClusterFunctionLibraryRuntime::Run(
    const FunctionLibraryRuntime::Options& opts,
    FunctionLibraryRuntime::LocalHandle handle, gtl::ArraySlice<Tensor> args,
    std::vector<Tensor>* rets, FunctionLibraryRuntime::DoneCallback done) {
  FunctionData* function_data = nullptr;
  {
    mutex_lock l(mu_);
    CHECK_LE(handle, function_data_.size());
    function_data = &function_data_[handle];
  }

  WorkerInterface* wi = function_data->wi;

  if (wi == nullptr) {
    done(errors::Internal("Could not find worker"));
    return;
  }

  RunGraphRequest* req = new RunGraphRequest;
  req->set_session_handle(worker_session_->session_name());
  req->set_create_worker_session_called(create_worker_session_called_);
  req->set_graph_handle(function_data->graph_handle);
  req->set_step_id(opts.step_id);
  int i = 0;
  for (const auto& send_key : function_data->send_keys) {
    NamedTensorProto* send = req->add_send();
    send->set_name(send_key);
    args[i].AsProtoTensorContent(send->mutable_tensor());
    i++;
  }
  const std::vector<string>& recv_keys = function_data->recv_keys;
  for (const auto& recv_key : recv_keys) {
    req->add_recv_key(recv_key);
  }

  RunGraphResponse* resp = new RunGraphResponse();
  CallOptions* call_options = new CallOptions();
  wi->RunGraphAsync(
      call_options, req, resp,
      [call_options, req, resp, rets, recv_keys, done](const Status& status) {
        Status* local_status = new Status(status);
        auto cleanup =
            gtl::MakeCleanup([call_options, req, resp, local_status, done] {
              done(*local_status);
              delete call_options;
              delete req;
              delete resp;
              delete local_status;
            });
        if (!local_status->ok()) {
          return;
        }
        std::map<string, TensorProto*> mapped_recvs;
        for (auto& recv : *resp->mutable_recv()) {
          mapped_recvs[recv.name()] = recv.mutable_tensor();
        }

        for (const auto& recv_key : recv_keys) {
          TensorProto* tp = mapped_recvs[recv_key];
          if (tp == nullptr) {
            local_status->Update(
                errors::Internal("Could not find key: ", recv_key));
            return;
          }
          Tensor t;
          if (t.FromProto(*tp)) {
            rets->push_back(t);
          } else {
            local_status->Update(errors::Internal(
                "Could not convert tensor proto: ", tp->DebugString()));
            return;
          }
        }
      });
}

void ClusterFunctionLibraryRuntime::CleanUp(
    uint64 step_id, FunctionLibraryRuntime::LocalHandle handle,
    FunctionLibraryRuntime::DoneCallback done) {
  FunctionData* function_data = nullptr;
  {
    mutex_lock l(mu_);
    DCHECK_LE(handle, function_data_.size());
    function_data = &function_data_[handle];
  }

  WorkerInterface* wi = function_data->wi;

  if (wi == nullptr) {
    done(errors::Internal("Could not find worker"));
    return;
  }
  CleanupGraphRequest* cleanup_req = new CleanupGraphRequest;
  cleanup_req->set_step_id(step_id);
  CleanupGraphResponse* cleanup_resp = new CleanupGraphResponse;
  wi->CleanupGraphAsync(
      cleanup_req, cleanup_resp,
      [cleanup_req, cleanup_resp, done](const Status& cleanup_status) {
        done(cleanup_status);
        delete cleanup_req;
        delete cleanup_resp;
      });
}

}  // namespace tensorflow
