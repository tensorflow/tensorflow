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

#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/protobuf/named_tensor.pb.h"

namespace tensorflow {

/* static */
Status ClusterFunctionLibraryRuntime::ConstructFunctionGraph(
    const OpDef& sig, AttrSlice attrs,
    const FunctionLibraryRuntime::InstantiateOptions& options, GraphDef* g,
    std::vector<string>* send_keys, std::vector<string>* recv_keys) {
  const string& target = options.target;
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

    NodeDef* input_node = g->add_node();
    TF_RETURN_IF_ERROR(
        NodeDefBuilder(strings::StrCat("_recv_", in.name(), "_", i), "_Recv")
            .Attr("tensor_type", dtypes[0])
            .Attr("tensor_name", in.name())
            .Attr("send_device", target)
            .Attr("recv_device", target)
            .Attr("send_device_incarnation", 1)
            .Attr("client_terminated", true)
            .Device(target)
            .Finalize(input_node));
    // src_incarnation = 1 works because the transfer is across the same device.
    // TODO(rohanj): Find the src_incarnation for the remote device and set it.
    const string& key = Rendezvous::CreateKey(
        target, 1 /* src_incarnation */, target, in.name(), FrameAndIter(0, 0));
    send_keys->push_back(key);
    ++i;
  }

  NodeDef* function_node = g->add_node();
  function_node->set_name(sig.name());
  function_node->set_op(sig.name());
  i = 0;
  for (const auto& in : sig.input_arg()) {
    function_node->add_input(strings::StrCat("_recv_", in.name(), "_", i));
    ++i;
  }
  function_node->set_device(target);
  for (const auto& p : attrs) {
    (*function_node->mutable_attr())[p.first] = p.second;
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

    NodeDef* output_node = g->add_node();
    TF_RETURN_IF_ERROR(
        NodeDefBuilder(strings::StrCat("_send_", out.name(), "_", i), "_Send")
            .Input(sig.name(), i, dtypes[0])
            .Attr("tensor_name", out.name())
            .Attr("send_device", target)
            .Attr("recv_device", target)
            .Attr("send_device_incarnation", 1)
            .Attr("client_terminated", true)
            .Device(target)
            .Finalize(output_node));
    const string& key =
        Rendezvous::CreateKey(target, 1 /* src_incarnation */, target,
                              out.name(), FrameAndIter(0, 0));
    recv_keys->push_back(key);
    ++i;
  }
  return Status::OK();
}

ClusterFunctionLibraryRuntime::~ClusterFunctionLibraryRuntime() {
  for (auto& function_data : function_data_) {
    worker_session_->worker_cache->ReleaseWorker(function_data.target,
                                                 function_data.wi);
  }
}

Status ClusterFunctionLibraryRuntime::Instantiate(
    const string& function_name, const FunctionLibraryDefinition& lib_def,
    AttrSlice attrs, const FunctionLibraryRuntime::InstantiateOptions& options,
    FunctionLibraryRuntime::LocalHandle* handle) {
  WorkerInterface* wi =
      worker_session_->worker_cache->CreateWorker(options.target);

  if (wi == nullptr) {
    std::vector<string> workers;
    worker_session_->worker_cache->ListWorkers(&workers);
    return errors::InvalidArgument(
        "Could not find worker with target: ", options.target,
        " Available workers: ", str_util::Join(workers, ", "));
  }

  // Make RPC and obtain a graph handle.
  const FunctionDef* fdef = lib_def.Find(function_name);
  const OpDef& sig = fdef->signature();
  GraphDef gdef;
  std::vector<string> send_keys, recv_keys;
  TF_RETURN_IF_ERROR(ConstructFunctionGraph(sig, attrs, options, &gdef,
                                            &send_keys, &recv_keys));
  *gdef.mutable_library() = lib_def.ToProto();

  RegisterGraphRequest req;
  req.set_session_handle(worker_session_->session_name);
  *req.mutable_graph_def() = gdef;
  req.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_do_function_inlining(true);
  RegisterGraphResponse resp;
  TF_RETURN_IF_ERROR(wi->RegisterGraph(&req, &resp));

  mutex_lock l(mu_);
  *handle = function_data_.size();
  function_data_.push_back(FunctionData(resp.graph_handle(), options.target, wi,
                                        send_keys, recv_keys));
  return Status::OK();
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
  req->set_session_handle(worker_session_->session_name);
  req->set_graph_handle(function_data->graph_handle);
  // Borrowed from master_session.cc
  const uint64 step_id = (random::New64() & ((1uLL << 56) - 1)) | (1uLL << 56);
  req->set_step_id(step_id);
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
        if (!status.ok()) {
          done(status);
          delete call_options;
          delete req;
          delete resp;
          return;
        }
        std::map<string, TensorProto*> mapped_recvs;
        for (auto& recv : *resp->mutable_recv()) {
          mapped_recvs[recv.name()] = recv.mutable_tensor();
        }

        for (const auto& recv_key : recv_keys) {
          TensorProto* tp = mapped_recvs[recv_key];
          if (tp == nullptr) {
            done(errors::Internal("Could not find key: ", recv_key));
            delete call_options;
            delete req;
            delete resp;
            return;
          }
          Tensor t;
          if (t.FromProto(*tp)) {
            rets->push_back(t);
          } else {
            done(errors::Internal("Could not convert tensor proto: ",
                                  tp->DebugString()));
            delete call_options;
            delete req;
            delete resp;
            return;
          }
        }
        done(status);
        delete call_options;
        delete req;
        delete resp;
      });
}

}  // namespace tensorflow
