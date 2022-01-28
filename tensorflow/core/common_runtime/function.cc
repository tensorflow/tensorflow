/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/function.h"

#include <deque>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/common_runtime/executor_factory.h"
#include "tensorflow/core/common_runtime/gradients.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/graph_optimizer.h"
#include "tensorflow/core/common_runtime/inline_function_utils.h"
#include "tensorflow/core/common_runtime/memory_types.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/common_runtime/single_threaded_executor.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function_handle_cache.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/control_flow.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/optimizer_cse.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/profiler/lib/connected_traceme.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/protobuf/config.pb.h"

// See core/kernels/function_ops.cc for related kernels.

namespace tensorflow {

// A few string constant used throughout this module.
static constexpr const char* const kArgOp = FunctionLibraryDefinition::kArgOp;
static constexpr const char* const kDeviceArgOp =
    FunctionLibraryDefinition::kDeviceArgOp;
static constexpr const char* const kRetOp = FunctionLibraryDefinition::kRetOp;
static constexpr const char* const kDeviceRetOp =
    FunctionLibraryDefinition::kDeviceRetOp;
static constexpr const char* const kGradientOp =
    FunctionLibraryDefinition::kGradientOp;
static constexpr const char* const kNodeLabel = "Func";
static constexpr const char* const kFuncAttr =
    FunctionLibraryDefinition::kFuncAttr;

// Represents the index-th output of a node.
struct Endpoint {
  Node* node;
  int index;

  // Returns the string name represents this endpoint.
  string name() const {
    if (index == 0) {
      return node->name();
    } else {
      return strings::StrCat(node->name(), ":", index);
    }
  }

  DataType dtype() const { return node->output_type(index); }
};

struct EndpointHash {
  uint64 operator()(const Endpoint& x) const {
    return Hash64(reinterpret_cast<const char*>(&x.node), sizeof(Node*),
                  x.index);
  }
};

struct EndpointEq {
  bool operator()(const Endpoint& x, const Endpoint& y) const {
    return (x.node == y.node) && (x.index == y.index);
  }
};

// The following Add* routines are used to add a few graph nodes while
// functions are transformed.
static Node* AddArg(Graph* g, DataType dtype, int index) {
  DCHECK_LT(0, dtype);
  DCHECK_LT(dtype, DT_FLOAT_REF);
  NodeDef ndef;
  ndef.set_name(g->NewName(kNodeLabel));
  ndef.set_op(kArgOp);
  AddNodeAttr("T", dtype, &ndef);
  AddNodeAttr("index", index, &ndef);
  Status s;
  Node* ret = g->AddNode(ndef, &s);
  TF_CHECK_OK(s);
  return ret;
}

static Node* AddRet(Graph* g, Endpoint input, int index) {
  DCHECK_LT(0, input.dtype());
  DCHECK_LT(input.dtype(), DT_FLOAT_REF);
  NodeDef ndef;
  ndef.set_name(g->NewName(kNodeLabel));
  ndef.set_op(kRetOp);
  ndef.add_input(input.name());
  AddNodeAttr("T", input.dtype(), &ndef);
  AddNodeAttr("index", index, &ndef);
  Status s;
  Node* ret = g->AddNode(ndef, &s);
  TF_CHECK_OK(s);
  g->AddEdge(input.node, input.index, ret, 0);
  return ret;
}

// FunctionLibraryRuntime implementation that forwards all the function calls to
// the base runtime implementation, and only overrides FunctionLibraryDefinition
// in calls to Instantiate (if caller doesn't provide the
// InstantiateOptions::lib_def option).
//
// When the function library runtime (FunctionLibraryRuntimeImpl specifically)
// instantiates a function into a Graph object, it also creates an Executor for
// it. That executor has a pointer to the function library runtime instance,
// that is used to instantiate all nested function calls.
//
// The function library definition used to instantiate the function must be
// preserved in the Executor's function library runtime.
//
// IMPORTANT: This runtime is intended for use only in executors created for
// functions instantiated into a graph in FunctionLibraryRuntimeImpl.
class FunctionLibraryRuntimeOverlay : public FunctionLibraryRuntime {
 public:
  FunctionLibraryRuntimeOverlay(FunctionLibraryRuntime* base_flr,
                                const FunctionLibraryDefinition* lib_def)
      : base_flr_(base_flr), lib_def_(lib_def) {}
  ~FunctionLibraryRuntimeOverlay() override;

  Status Instantiate(const string& function_name, AttrSlice attrs,
                     const InstantiateOptions& options,
                     Handle* handle) override;

  Status ReleaseHandle(Handle handle) override;

  const FunctionBody* GetFunctionBody(Handle h) override;

  Status GetRetTypes(Handle h, DataTypeVector* ret_types) override;

  void Run(const Options& opts, Handle handle, gtl::ArraySlice<Tensor> args,
           std::vector<Tensor>* rets, DoneCallback done) override;

  void Run(const Options& opts, Handle handle, CallFrameInterface* call_frame,
           DoneCallback done) override;

  Status RunSync(Options opts, Handle handle, gtl::ArraySlice<Tensor> args,
                 std::vector<Tensor>* rets) override;

  Status RunSync(Options opts, Handle handle,
                 CallFrameInterface* frame) override;

  Status CreateKernel(const std::shared_ptr<const NodeProperties>& props,
                      OpKernel** kernel) override;

  bool IsStateful(const string& function_name) const override;

  const FunctionLibraryDefinition* GetFunctionLibraryDefinition()
      const override;

  Env* env() override;
  const ConfigProto* const config_proto() override;
  Device* device() override;
  const Device* device() const override;
  std::function<void(std::function<void()>)>* runner() override;
  const DeviceMgr* device_mgr() const override;

  string DebugString(Handle handle) override;
  int graph_def_version() const override;

  Status Clone(std::unique_ptr<FunctionLibraryDefinition>* out_lib_def,
               std::unique_ptr<ProcessFunctionLibraryRuntime>* out_pflr,
               FunctionLibraryRuntime** out_flr,
               bool skip_flib_def = false) override;

 private:
  FunctionLibraryRuntime* base_flr_;          // not owned
  const FunctionLibraryDefinition* lib_def_;  // not owned
};

FunctionLibraryRuntimeOverlay::~FunctionLibraryRuntimeOverlay() = default;

Status FunctionLibraryRuntimeOverlay::Instantiate(
    const string& function_name, AttrSlice attrs,
    const InstantiateOptions& options, Handle* handle) {
  // We automatically set the `lib_def` option for all instantiations, if the
  // caller doesn't set this option explicitly.
  if (!options.lib_def && lib_def_) {
    InstantiateOptions options_copy = options;
    options_copy.lib_def = lib_def_;
    return base_flr_->Instantiate(function_name, attrs, options_copy, handle);
  } else {
    return base_flr_->Instantiate(function_name, attrs, options, handle);
  }
}

Status FunctionLibraryRuntimeOverlay::ReleaseHandle(Handle handle) {
  return base_flr_->ReleaseHandle(handle);
}

const FunctionBody* FunctionLibraryRuntimeOverlay::GetFunctionBody(Handle h) {
  return base_flr_->GetFunctionBody(h);
}

Status FunctionLibraryRuntimeOverlay::GetRetTypes(Handle h,
                                                  DataTypeVector* ret_types) {
  return base_flr_->GetRetTypes(h, ret_types);
}

void FunctionLibraryRuntimeOverlay::Run(const Options& opts, Handle handle,
                                        gtl::ArraySlice<Tensor> args,
                                        std::vector<Tensor>* rets,
                                        DoneCallback done) {
  base_flr_->Run(opts, handle, args, rets, std::move(done));
}

void FunctionLibraryRuntimeOverlay::Run(const Options& opts, Handle handle,
                                        CallFrameInterface* call_frame,
                                        DoneCallback done) {
  base_flr_->Run(opts, handle, call_frame, std::move(done));
}

Status FunctionLibraryRuntimeOverlay::RunSync(Options opts, Handle handle,
                                              gtl::ArraySlice<Tensor> args,
                                              std::vector<Tensor>* rets) {
  return base_flr_->RunSync(std::move(opts), handle, args, rets);
}

Status FunctionLibraryRuntimeOverlay::RunSync(Options opts, Handle handle,
                                              CallFrameInterface* call_frame) {
  return base_flr_->RunSync(std::move(opts), handle, call_frame);
}

Status FunctionLibraryRuntimeOverlay::CreateKernel(
    const std::shared_ptr<const NodeProperties>&, OpKernel**) {
  // We don't have access to base_lib_def_ in base function library runtime (aka
  // FunctionLibraryRuntimeImpl), so to make sure we do not create a kernel with
  // the wrong lib_def we just disable creation of new kernels through overlays.
  //
  // When we call Instantiate from the base runtime with the lib_def option,
  // the base runtime implementation is responsible for correctly passing it
  // through to all kernel constructions.
  return errors::Internal(
      "Overlay function library runtime doesn't support kernel creation.");
}

bool FunctionLibraryRuntimeOverlay::IsStateful(
    const string& function_name) const {
  // Important: we do not forward lookup to the base FLR.
  const OpDef* op_def;
  const Status s = lib_def_->LookUpOpDef(function_name, &op_def);
  return s.ok() && op_def->is_stateful();
}

Env* FunctionLibraryRuntimeOverlay::env() { return base_flr_->env(); }

const ConfigProto* const FunctionLibraryRuntimeOverlay::config_proto() {
  return base_flr_->config_proto();
}

Device* FunctionLibraryRuntimeOverlay::device() { return base_flr_->device(); }

const Device* FunctionLibraryRuntimeOverlay::device() const {
  return base_flr_->device();
}

std::function<void(std::function<void()>)>*
FunctionLibraryRuntimeOverlay::runner() {
  return base_flr_->runner();
}

const DeviceMgr* FunctionLibraryRuntimeOverlay::device_mgr() const {
  return base_flr_->device_mgr();
}

const FunctionLibraryDefinition*
FunctionLibraryRuntimeOverlay::GetFunctionLibraryDefinition() const {
  return lib_def_ ? lib_def_ : base_flr_->GetFunctionLibraryDefinition();
}

string FunctionLibraryRuntimeOverlay::DebugString(Handle handle) {
  return base_flr_->DebugString(handle);
}

int FunctionLibraryRuntimeOverlay::graph_def_version() const {
  return base_flr_->graph_def_version();
}

Status FunctionLibraryRuntimeOverlay::Clone(
    std::unique_ptr<FunctionLibraryDefinition>* out_lib_def,
    std::unique_ptr<ProcessFunctionLibraryRuntime>* out_pflr,
    FunctionLibraryRuntime** out_flr, bool skip_flib_def) {
  // NOTE(ezhulenev): The cloned FunctionLibraryRuntime will be missing the
  // FunctionLibraryDefinition override, but that's ok because we anyway do not
  // copy / clone instantiated items from the base FLR.
  return base_flr_->Clone(out_lib_def, out_pflr, out_flr, skip_flib_def);
}

class FunctionLibraryRuntimeImpl : public FunctionLibraryRuntime {
 public:
  FunctionLibraryRuntimeImpl(const DeviceMgr* dmgr, Env* env,
                             const ConfigProto* config, Device* device,
                             int graph_def_version,
                             const FunctionLibraryDefinition* lib_def,
                             thread::ThreadPool* default_thread_pool,
                             const OptimizerOptions& optimizer_options,
                             const SessionMetadata* session_metadata,
                             ProcessFunctionLibraryRuntime* parent);

  ~FunctionLibraryRuntimeImpl() override;

  Status Instantiate(const string& function_name, AttrSlice attrs,
                     const InstantiateOptions& options,
                     Handle* handle) override;

  Status ReleaseHandle(Handle handle) override;

  const FunctionBody* GetFunctionBody(Handle handle) override;

  Status GetRetTypes(Handle handle, DataTypeVector* ret_types) override;

  Status CreateKernel(const std::shared_ptr<const NodeProperties>& props,
                      OpKernel** kernel) override;

  void Run(const Options& opts, Handle handle, gtl::ArraySlice<Tensor> args,
           std::vector<Tensor>* rets, DoneCallback done) override;
  void Run(const Options& opts, Handle handle, CallFrameInterface* frame,
           DoneCallback done) override;
  Status RunSync(Options opts, Handle handle, gtl::ArraySlice<Tensor> args,
                 std::vector<Tensor>* rets) override;
  Status RunSync(Options opts, Handle handle,
                 CallFrameInterface* call_frame) override;

  bool IsStateful(const string& function) const override;

  const FunctionLibraryDefinition* GetFunctionLibraryDefinition()
      const override {
    return base_lib_def_;
  }

  Device* device() override { return device_; }
  const Device* device() const override { return device_; }

  std::function<void(std::function<void()>)>* runner() override {
    return &default_runner_;
  }

  const DeviceMgr* device_mgr() const override { return device_mgr_; }
  Env* env() override { return env_; }
  const ConfigProto* const config_proto() override { return config_; }
  int graph_def_version() const override { return graph_def_version_; }

  string DebugString(Handle h) override;

  Status Clone(std::unique_ptr<FunctionLibraryDefinition>* out_lib_def,
               std::unique_ptr<ProcessFunctionLibraryRuntime>* out_pflr,
               FunctionLibraryRuntime** out_flr,
               bool skip_flib_def = false) override;

 private:
  typedef FunctionLibraryRuntimeImpl ME;

  const DeviceMgr* const device_mgr_;
  Device* const device_;
  Env* const env_;
  const ConfigProto* const config_;
  const int graph_def_version_;
  const FunctionLibraryDefinition* const base_lib_def_;
  GraphOptimizer optimizer_;
  const SessionMetadata* const session_metadata_;
  Executor::Args::Runner default_runner_;
  const string device_name_;

  std::function<Status(const string&, const OpDef**)> get_func_sig_;
  std::function<Status(const std::shared_ptr<const NodeProperties>&,
                       OpKernel**)>
      create_kernel_;

  mutable mutex mu_;

  int next_handle_ TF_GUARDED_BY(mu_);

  // The instantiated and transformed function is encoded as a Graph
  // object, and an executor is created for the graph.
  struct Item {
    uint64 instantiation_counter = 0;
    std::unique_ptr<const Graph> graph = nullptr;
    const FunctionLibraryDefinition* lib_def = nullptr;  // Not owned.
    FunctionBody* func_graph = nullptr;
    Executor* exec = nullptr;
    FunctionLibraryRuntimeOverlay* overlay_flr = nullptr;
    string executor_type;
    bool allow_small_function_optimizations = false;
    bool allow_control_flow_sync_execution = false;

    ~Item() {
      delete this->func_graph;
      delete this->exec;
      delete this->overlay_flr;
    }
  };
  std::unique_ptr<absl::flat_hash_map<Handle, std::unique_ptr<Item>>> items_
      TF_GUARDED_BY(mu_);
  std::unique_ptr<FunctionHandleCache> function_handle_cache_;
  ProcessFunctionLibraryRuntime* parent_ = nullptr;  // not owned.

  // Overloads the CreateKernel method, providing a FunctionLibraryRuntime
  // to use for kernel creation and execution. In particular, this method can
  // accept a FunctionLibraryRuntimeOverlay that overlays a different
  // FunctionLibraryDefinition.
  Status CreateKernel(const std::shared_ptr<const NodeProperties>& props,
                      FunctionLibraryRuntime* flr, OpKernel** kernel);
  Status FunctionDefToBody(const FunctionDef& fdef, AttrSlice attrs,
                           const FunctionLibraryDefinition* lib_def,
                           std::unique_ptr<FunctionBody>* fbody);
  Status CreateItem(Item** item);
  Status GetOrCreateItem(LocalHandle local_handle, Item** item);
  Status InstantiateSymbolicGradient(const NameAttrList& func,
                                     const FunctionLibraryDefinition* lib_def,
                                     std::unique_ptr<FunctionBody>* g_body);
  bool IsLocalTarget(const InstantiateOptions& options) const;
  AttrValueMap FixAttrs(const AttrSlice& attrs);
  void RunRemote(const Options& opts, Handle handle,
                 gtl::ArraySlice<Tensor> args, std::vector<Tensor>* rets,
                 Item* item, DoneCallback done);

  Status PrepareRunSync(
      Handle handle, Options* run_opts, Item** out_item,
      std::unique_ptr<PrivateIntraProcessRendezvous>* out_rendezvous);

  void ExecutorArgsFromOptions(const FunctionLibraryRuntime::Options& run_opts,
                               CallFrameInterface* frame,
                               Executor::Args* exec_args);

  TF_DISALLOW_COPY_AND_ASSIGN(FunctionLibraryRuntimeImpl);
};

FunctionLibraryRuntimeImpl::FunctionLibraryRuntimeImpl(
    const DeviceMgr* dmgr, Env* env, const ConfigProto* config, Device* device,
    int graph_def_version, const FunctionLibraryDefinition* lib_def,
    thread::ThreadPool* default_thread_pool,
    const OptimizerOptions& optimizer_options,
    const SessionMetadata* session_metadata,
    ProcessFunctionLibraryRuntime* parent)
    : device_mgr_(dmgr),
      device_(device),
      env_(env),
      config_(config),
      graph_def_version_(graph_def_version),
      base_lib_def_(lib_def),
      optimizer_(optimizer_options),
      session_metadata_(session_metadata),
      default_runner_(nullptr),
      device_name_(device_ == nullptr
                       ? ProcessFunctionLibraryRuntime::kDefaultFLRDevice
                       : device_->name()),
      next_handle_(0),
      items_(absl::make_unique<
             absl::flat_hash_map<Handle, std::unique_ptr<Item>>>()),
      function_handle_cache_(absl::make_unique<FunctionHandleCache>(this)),
      parent_(parent) {
  get_func_sig_ = [this](const string& op, const OpDef** sig) {
    return base_lib_def_->LookUpOpDef(op, sig);
  };
  create_kernel_ = [this](const std::shared_ptr<const NodeProperties>& props,
                          OpKernel** kernel) {
    return CreateKernel(props, kernel);
  };
  thread::ThreadPool* pool = nullptr;
  if (device_ != nullptr) {
    pool = device_->tensorflow_device_thread_pool();
  }
  if (pool == nullptr) {
    pool = default_thread_pool;
  }
  if (pool != nullptr) {
    default_runner_ = [pool](Executor::Args::Closure c) {
      pool->Schedule(std::move(c));
    };
  }
}

FunctionLibraryRuntimeImpl::~FunctionLibraryRuntimeImpl() {
  // Deleting the items_ list will delete all the function handles registered in
  // this object. A function may contains a few sub-functions which have also
  // been registered in this object. Deleting the parent function will call
  // ReleaseHandle in this class again for each of the sub-functions. These
  // circular calls may cause segfault since the items_ may have already been
  // partially deleted when releasing handles of sub-functions. Explicitly
  // release items_ here and check it in ReleaseHandle to avoid this.
  items_.reset();
}

// An asynchronous op kernel which executes an instantiated function
// defined in a library.
class CallOp : public AsyncOpKernel {
 public:
  CallOp(FunctionLibraryRuntime::Handle handle, OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx), handle_(handle) {}

  ~CallOp() override {
    // TODO(iga): Release the cached handle_
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    FunctionLibraryRuntime* lib = ctx->function_library();
    OP_REQUIRES_ASYNC(ctx, lib != nullptr,
                      errors::Internal("No function library is provided."),
                      done);
    FunctionLibraryRuntime::Options opts;
    opts.rendezvous = ctx->rendezvous();
    opts.cancellation_manager = ctx->cancellation_manager();
    opts.step_container = ctx->step_container();
    opts.stats_collector = ctx->stats_collector();
    opts.runner = ctx->runner();
    opts.run_all_kernels_inline = ctx->run_all_kernels_inline();
    opts.collective_executor = ctx->collective_executor();
    std::vector<Tensor> args;
    args.reserve(ctx->num_inputs());
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      args.push_back(ctx->input(i));
    }
    std::vector<Tensor>* rets = new std::vector<Tensor>;
    profiler::TraceMe trace_me(
        [&] {
          return absl::StrCat("CallOp #parent_step_id=", ctx->step_id(),
                              ",function_step_id=", opts.step_id, "#");
        },
        /*level=*/2);
    lib->Run(opts, handle_, args, rets,
             [ctx, done, rets](const Status& status) {
               if (!status.ok()) {
                 ctx->SetStatus(status);
               } else {
                 const int ret_size = static_cast<int>(rets->size());
                 CHECK_EQ(ret_size, ctx->num_outputs());
                 for (int i = 0; i < ret_size; ++i) {
                   ctx->set_output(i, (*rets)[i]);
                 }
               }
               delete rets;
               done();
             });
  }

 private:
  FunctionLibraryRuntime::Handle handle_;

  TF_DISALLOW_COPY_AND_ASSIGN(CallOp);
};

const FunctionBody* FunctionLibraryRuntimeImpl::GetFunctionBody(Handle h) {
  LocalHandle local_handle = parent_->GetHandleOnDevice(device_name_, h);
  if (local_handle == kInvalidLocalHandle) {
    LOG(ERROR) << "Could not find Handle: " << h
               << " on device: " << device_name_;
    return nullptr;
  }

  tf_shared_lock l(mu_);
  auto iter = items_->find(local_handle);
  CHECK(iter != items_->end());
  return iter->second->func_graph;
}

Status FunctionLibraryRuntimeImpl::GetRetTypes(Handle h,
                                               DataTypeVector* ret_types) {
  if (parent_->IsMultiDevice(h)) {
    return parent_->GetRetTypes(h, ret_types);
  }
  LocalHandle local_handle = parent_->GetHandleOnDevice(device_name_, h);
  if (local_handle == kInvalidLocalHandle) {
    return errors::InvalidArgument("Handle ", h, " not found.");
  }
  const FunctionBody* fbody = GetFunctionBody(h);
  *ret_types = fbody->ret_types;
  return Status::OK();
}

Status FunctionLibraryRuntimeImpl::CreateKernel(
    const std::shared_ptr<const NodeProperties>& props, OpKernel** kernel) {
  return CreateKernel(props, this, kernel);
}

Status FunctionLibraryRuntimeImpl::CreateKernel(
    const std::shared_ptr<const NodeProperties>& props,
    FunctionLibraryRuntime* flr, OpKernel** kernel) {
  // If a custom kernel creator is given, try that.
  Status s;
  const CustomKernelCreator* custom_kernel_creator =
      GetDefaultCustomKernelCreator();
  if (custom_kernel_creator &&
      custom_kernel_creator->CanCreateKernel(*flr, props)) {
    std::unique_ptr<OpKernel> ret;
    s = custom_kernel_creator->CreateKernel(flr, props, &ret);
    if (s.ok()) {
      *kernel = ret.release();
    } else {
      VLOG(2) << "Custom creator error: " << s;
    }
    return s;
  }

  const FunctionLibraryDefinition* lib_def =
      flr->GetFunctionLibraryDefinition();
  if (lib_def->Find(props->node_def.op()) == nullptr) {
    // A primitive operation. Creates the registered kernel.
    return CreateNonCachedKernel(device_, flr, props, graph_def_version_,
                                 kernel);
  }

  // Try to instantiate this function for the func/attr. Maybe it's
  // cached already.
  InstantiateOptions options;
  if (lib_def != base_lib_def_) {
    options.lib_def = lib_def;
  }
  Handle handle;
  TF_RETURN_IF_ERROR(Instantiate(props->node_def.op(),
                                 AttrSlice(&props->node_def.attr()), options,
                                 &handle));

  const FunctionBody* fbody = GetFunctionBody(handle);
  CHECK_NOTNULL(fbody);

  // TODO(zhifengc): For now, we assume int32 and resources are always on host
  // memory and other types are always on device memory. We should do type
  // inference over function body to derive the correct input/output memory
  // types.
  MemoryTypeVector input_memory_types;
  for (const auto& t : fbody->arg_types) {
    input_memory_types.push_back(MTypeFromDType(t));
  }
  MemoryTypeVector output_memory_types;
  for (const auto& t : fbody->ret_types) {
    output_memory_types.push_back(MTypeFromDType(t));
  }

  // Constructs a CallOp kernel for running the instantiated function.
  auto device_type = DeviceType(device_->attributes().device_type());
  auto new_props = std::make_shared<NodeProperties>(
      &fbody->fdef.signature(), props->node_def, fbody->arg_types,
      fbody->ret_types);
  OpKernelConstruction construction(
      device_type, device_, device_->GetAllocator(AllocatorAttributes()), flr,
      device_->resource_manager(), props, input_memory_types,
      output_memory_types, graph_def_version_, &s);
  if (s.ok()) {
    *kernel = new CallOp(handle, &construction);
  }
  return s;
}

Status FunctionLibraryRuntimeImpl::FunctionDefToBody(
    const FunctionDef& fdef, AttrSlice attrs,
    const FunctionLibraryDefinition* lib_def,
    std::unique_ptr<FunctionBody>* fbody) {
  if (lib_def == base_lib_def_) {
    return FunctionDefToBodyHelper(fdef, attrs, lib_def, get_func_sig_, fbody);
  } else {
    auto get_func_sig = [lib_def](const string& op, const OpDef** sig) {
      return lib_def->LookUpOpDef(op, sig);
    };
    return FunctionDefToBodyHelper(fdef, attrs, lib_def, get_func_sig, fbody);
  }
}

Status FunctionLibraryRuntimeImpl::InstantiateSymbolicGradient(
    const NameAttrList& func, const FunctionLibraryDefinition* lib_def,
    std::unique_ptr<FunctionBody>* g_body) {
  const FunctionDef* fdef = lib_def->Find(func.name());
  if (fdef == nullptr) {
    // f is a primitive op.
    gradient::Creator creator;
    TF_RETURN_IF_ERROR(gradient::GetOpGradientCreator(func.name(), &creator));
    if (creator == nullptr) {
      return errors::InvalidArgument("No gradient is defined for ",
                                     func.name());
    }
    FunctionDef grad_fdef;
    // TODO(josh11b): Should filter out the attrs from func that aren't used
    // by the gradient function.
    TF_RETURN_IF_ERROR(creator(AttrSlice(&func.attr()), &grad_fdef));
    TF_RETURN_IF_ERROR(
        FunctionDefToBody(grad_fdef, AttrSlice(&func.attr()), lib_def, g_body));
  } else {
    // f is a user-defined function.
    InstantiateOptions options;
    if (lib_def != base_lib_def_) {
      options.lib_def = lib_def;
    }
    Handle f_handle;
    TF_RETURN_IF_ERROR(
        Instantiate(func.name(), AttrSlice(&func.attr()), options, &f_handle));
    const FunctionBody* f_body = GetFunctionBody(f_handle);
    CHECK_NOTNULL(f_body);
    *g_body = SymbolicGradient(*f_body);
  }
  return Status::OK();
}

bool FunctionLibraryRuntimeImpl::IsLocalTarget(
    const InstantiateOptions& options) const {
  if (device_ == nullptr) return true;
  if (options.target.empty()) return true;
  if (options.is_multi_device_function) return false;
  Device* target_device;
  if (!device_mgr_->LookupDevice(options.target, &target_device).ok()) {
    VLOG(1) << "Not instantiating function in FLR because failed to "
            << "find device " << options.target << " in device manager";
    return false;
  }
  if (target_device != device_) {
    VLOG(1) << "Not instantiating function in FLR because target device "
            << options.target
            << " is different from FLR's device: " << device_->DebugString();
    return false;
  }
  return true;
}

Status FunctionLibraryRuntimeImpl::Instantiate(
    const string& function_name, AttrSlice attrs,
    const InstantiateOptions& options, Handle* handle) {
  if (!IsLocalTarget(options)) {
    return parent_->Instantiate(function_name, attrs, options, handle);
  }

  if (options.use_function_cache) {
    InstantiateOptions options_copy(options);
    options_copy.use_function_cache = false;
    return function_handle_cache_->Instantiate(function_name, attrs,
                                               options_copy, handle);
  }

  // Since this is a local target, ensure that the local `device_name_` appears
  // in the canonical key.
  InstantiateOptions options_copy(options);
  options_copy.target = device_name_;
  const string key = Canonicalize(function_name, attrs, options_copy);

  {
    mutex_lock l(mu_);
    *handle = parent_->GetHandle(key);
    if (*handle != kInvalidHandle) {
      FunctionLibraryRuntime::LocalHandle handle_on_device =
          parent_->GetHandleOnDevice(device_name_, *handle);
      if (handle_on_device == kInvalidLocalHandle) {
        return errors::Internal("LocalHandle not found for handle ", *handle,
                                ".");
      }
      auto item_handle = items_->find(handle_on_device);
      if (item_handle == items_->end()) {
        return errors::Internal("LocalHandle ", handle_on_device,
                                " for handle ", *handle,
                                " not found in items.");
      }
      ++item_handle->second->instantiation_counter;
      return Status::OK();
    }
  }

  const FunctionLibraryDefinition* lib_def =
      options.lib_def ? options.lib_def : base_lib_def_;
  std::unique_ptr<FunctionBody> fbody;
  if (function_name == kGradientOp) {
    const AttrValue* f = attrs.Find(kFuncAttr);
    if (f == nullptr) {
      return errors::InvalidArgument("SymbolicGradient is missing attr: f");
    }
    const auto& func = f->func();
    if (func.name() == kGradientOp) {
      return errors::InvalidArgument("Can't take gradient of SymbolicGradient");
    }
    const string grad = lib_def->FindGradient(func.name());
    if (!grad.empty()) {
      return Instantiate(grad, AttrSlice(&func.attr()), options, handle);
    }
    TF_RETURN_IF_ERROR(InstantiateSymbolicGradient(func, lib_def, &fbody));
  } else {
    const FunctionDef* fdef = lib_def->Find(function_name);
    if (fdef == nullptr) {
      return errors::NotFound("Function ", function_name, " is not defined.");
    }
    TF_RETURN_IF_ERROR(FunctionDefToBody(*fdef, attrs, lib_def, &fbody));
  }

  LocalHandle local_handle;
  {
    mutex_lock l(mu_);
    *handle = parent_->GetHandle(key);
    if (*handle != kInvalidHandle) {
      local_handle = parent_->GetHandleOnDevice(device_name_, *handle);
      ++(*items_)[local_handle]->instantiation_counter;
    } else {
      *handle = parent_->AddHandle(key, device_name_, next_handle_);
      Item* item = new Item;
      item->func_graph = fbody.release();
      item->instantiation_counter = 1;
      item->executor_type = ExecutorType(options, attrs);
      item->allow_small_function_optimizations =
          options.allow_small_function_optimizations;
      item->allow_control_flow_sync_execution =
          options.allow_control_flow_sync_execution;
      if (options.lib_def) {
        item->overlay_flr =
            new FunctionLibraryRuntimeOverlay(this, options.lib_def);
      }
      local_handle = next_handle_++;
      items_->emplace(local_handle, std::unique_ptr<Item>(item));
    }
  }

  if (options.create_kernels_eagerly) {
    Item* item;
    TF_RETURN_IF_ERROR(GetOrCreateItem(local_handle, &item));
  }

  return Status::OK();
}

Status FunctionLibraryRuntimeImpl::ReleaseHandle(Handle handle) {
  LocalHandle h = parent_->GetHandleOnDevice(device_name_, handle);
  if (h == kInvalidLocalHandle) {
    return parent_->ReleaseHandle(handle);
  }
  std::unique_ptr<Item> item_to_delete;
  Status parent_status;
  {
    mutex_lock l(mu_);
    // Return directly if all items has already been released.
    if (items_ == nullptr) return Status::OK();

    auto it = items_->find(h);
    if (it == items_->end()) {
      return errors::Internal(
          "Inconsistent FunctionLibraryRuntime. Expected to find an item for "
          "handle ",
          h, " but found none");
    }
    std::unique_ptr<Item>& item = it->second;
    --item->instantiation_counter;
    if (item->instantiation_counter == 0) {
      // We don't simply erase h's item because that would trigger
      // item destruction while holding mu_. Item destruction can
      // trigger graph destruction. If the graph contains kernels like
      // CallOp or PartitionCallOp, their destructors will release cached
      // function handles, resulting in deadlock here.
      item_to_delete = std::move(item);
      items_->erase(h);
      parent_status = parent_->RemoveHandle(handle);
    }
  }
  return parent_status;
}

namespace {

// Removes all stateless nodes that do not contribute to a return
// value from the function body. Unlike `RemoveDeadNodes()`, which is
// triggered by `OptimizerOptions.do_function_inlining`, this pass
// ignores the SINK node, from which (by definition) all nodes are
// reverse reachable, and preserves all nodes that are reachable from
// control output nodes.
//
// TODO(ezhulenev, skyewm): Function body should not have special treatment of
// stateful ops, graph should encode nodes that must execute with `control_ret`
// and `control_output`.
void PruneFunctionBody(const FunctionDef& fdef, Graph* g) {
  VLOG(2) << "Pruning function body: function_name=" << fdef.signature().name();

  // `control_ret` nodes must be always executed.
  std::unordered_set<StringPiece, StringPieceHasher> control_ret_nodes;
  for (const auto& control_ret : fdef.control_ret()) {
    control_ret_nodes.insert(control_ret.second);
  }

  std::unordered_set<const Node*> nodes;
  for (auto n : g->nodes()) {
    // NOTE(mrry): "_Retval" nodes are stateful, and so will be added
    // to the seed set of `nodes`. "_Arg" nodes are also stateful, but we
    // specifically exclude them as seeds, to avoid unconditionally executing
    // unused argument nodes (e.g. in a function like `lambda x, y: y`).
    // TODO(mrry): Investigate whether the `n->IsControlFlow()` test is
    // still needed. It would be preferable to prune entire loops and/or
    // conditionals if they are not used in the graph.
    if (n->IsControlFlow() ||
        (n->op_def().is_stateful() && n->type_string() != kArgOp) ||
        (control_ret_nodes.find(n->name()) != control_ret_nodes.end())) {
      nodes.insert(n);
    }
  }
  bool changed = PruneForReverseReachability(g, std::move(nodes));
  if (changed) {
    FixupSourceAndSinkEdges(g);
  }
}

}  // namespace

Status FunctionLibraryRuntimeImpl::CreateItem(Item** item) {
  const FunctionBody* fbody;
  FunctionLibraryRuntime* flr;
  string executor_type;
  {
    tf_shared_lock l(mu_);
    fbody = (*item)->func_graph;
    flr = (*item)->overlay_flr
              ? static_cast<FunctionLibraryRuntime*>((*item)->overlay_flr)
              : static_cast<FunctionLibraryRuntime*>(this);
    executor_type = (*item)->executor_type;
  }
  const FunctionLibraryDefinition* lib_def =
      flr->GetFunctionLibraryDefinition();
  auto g = absl::make_unique<Graph>(lib_def);
  CopyGraph(*fbody->graph, g.get());

  PruneFunctionBody(fbody->fdef, g.get());
  optimizer_.Optimize(this, env(), device(), &g, GraphOptimizer::Options());
  TF_RETURN_IF_ERROR(EnsureMemoryTypes(DeviceType(device()->device_type()),
                                       device()->name(), g.get()));

  // Creates an executor based on the g. This must be done without
  // holding mu_ because create_kernel_ calls back into the library.
  LocalExecutorParams params;
  params.device = device_;
  params.function_library = flr;
  params.allow_control_flow_sync_execution =
      (*item)->allow_control_flow_sync_execution;
  if (flr == this) {
    params.create_kernel = create_kernel_;
  } else {
    params.create_kernel =
        [this, flr](const std::shared_ptr<const NodeProperties>& props,
                    OpKernel** kernel) {
          return CreateKernel(props, flr, kernel);
        };
  }
  params.delete_kernel = [](OpKernel* kernel) {
    DeleteNonCachedKernel(kernel);
  };
  params.session_metadata = session_metadata_;
  std::unique_ptr<Executor> exec;

  // When the instantiation options request small function optimizations, all
  // graphs which are safe for synchronous execution will set this flag to true:
  if ((*item)->allow_small_function_optimizations && executor_type.empty()) {
    executor_type = "SINGLE_THREADED_EXECUTOR";
  }

  metrics::IncrementTestCounter("flr_executor",
                                (executor_type == "SINGLE_THREADED_EXECUTOR")
                                    ? "single_threaded"
                                    : "default");

  TF_RETURN_IF_ERROR(NewExecutor(executor_type, params, *g, &exec));
  {
    // Guard item since it is already inserted in items_.
    mutex_lock l(mu_);
    if ((*item)->exec == nullptr) {
      (*item)->graph = std::move(g);
      (*item)->exec = exec.release();
    }
  }
  return Status::OK();
}

Status FunctionLibraryRuntimeImpl::GetOrCreateItem(LocalHandle local_handle,
                                                   Item** item) {
  {
    tf_shared_lock l(mu_);
    auto iter = items_->find(local_handle);
    if (iter == items_->end()) {
      return errors::Internal("Local function handle ", local_handle,
                              " is not valid. Likely an internal error.");
    }
    *item = iter->second.get();
    if ((*item)->exec != nullptr) {
      return Status::OK();
    }
  }
  // NOTE: We need to call CreateItem out of mu_ because creating an
  // executor needs to call CreateKernel.
  return CreateItem(item);
}

void FunctionLibraryRuntimeImpl::ExecutorArgsFromOptions(
    const FunctionLibraryRuntime::Options& run_opts, CallFrameInterface* frame,
    Executor::Args* exec_args) {
  // Inherit the step_id from the caller.
  exec_args->step_id = run_opts.step_id;
  exec_args->rendezvous = run_opts.rendezvous;
  exec_args->stats_collector = run_opts.stats_collector;
  exec_args->cancellation_manager = run_opts.cancellation_manager;
  exec_args->step_container = run_opts.step_container;
  if (run_opts.runner) {
    exec_args->runner = *run_opts.runner;
  } else {
    exec_args->runner = default_runner_;
  }
  exec_args->collective_executor = run_opts.collective_executor;
  exec_args->call_frame = frame;
  exec_args->run_all_kernels_inline = run_opts.run_all_kernels_inline;
  exec_args->user_intra_op_threadpool = run_opts.user_intra_op_threadpool;
  exec_args->coordination_service_agent = run_opts.coordination_service_agent;
}

void FunctionLibraryRuntimeImpl::RunRemote(const Options& opts, Handle handle,
                                           gtl::ArraySlice<Tensor> args,
                                           std::vector<Tensor>* rets,
                                           Item* item, DoneCallback done) {
  string target_device = parent_->GetDeviceName(handle);
  string source_device = opts.source_device;
  RendezvousInterface* rendezvous = opts.rendezvous;
  DeviceContext* device_context;
  Status s = parent_->GetDeviceContext(target_device, &device_context);
  if (!s.ok()) {
    done(s);
    return;
  }
  int64_t src_incarnation, target_incarnation;
  s = parent_->GetDeviceIncarnation(source_device, &src_incarnation);
  s.Update(parent_->GetDeviceIncarnation(target_device, &target_incarnation));
  if (!s.ok()) {
    done(s);
    return;
  }

  const FunctionBody* fbody = GetFunctionBody(handle);
  FunctionCallFrame* frame =
      new FunctionCallFrame(fbody->arg_types, fbody->ret_types);
  Executor::Args* exec_args = new Executor::Args;
  ExecutorArgsFromOptions(opts, frame, exec_args);

  std::vector<AllocatorAttributes> args_alloc_attrs, rets_alloc_attrs;
  args_alloc_attrs.reserve(fbody->arg_types.size());
  rets_alloc_attrs.reserve(fbody->ret_types.size());
  // Note: Functions assume that int32's are always on host memory.
  for (const auto& arg_type : fbody->arg_types) {
    AllocatorAttributes arg_alloc_attrs;
    if (MTypeFromDType(arg_type) == HOST_MEMORY) {
      arg_alloc_attrs.set_on_host(true);
    }
    args_alloc_attrs.push_back(arg_alloc_attrs);
  }
  for (const auto& ret_type : fbody->ret_types) {
    AllocatorAttributes ret_alloc_attrs;
    if (MTypeFromDType(ret_type) == HOST_MEMORY) {
      ret_alloc_attrs.set_on_host(true);
    }
    rets_alloc_attrs.push_back(ret_alloc_attrs);
  }

  bool allow_dead_tensors = opts.allow_dead_tensors;

  // The ProcFLR sends the arguments to the function from the source_device to
  // the target_device. So here we receive those arguments. Similarly, when the
  // computation is done and stored in *rets, we send the return values back
  // to the source_device (caller) so that the ProcFLR can receive them later.
  std::vector<Tensor>* remote_args = new std::vector<Tensor>;
  ProcessFunctionLibraryRuntime::ReceiveTensorsAsync(
      source_device, target_device, "arg_", src_incarnation, args.size(),
      device_context, args_alloc_attrs, rendezvous, remote_args,
      [frame, remote_args, item, source_device, target_device,
       target_incarnation, rendezvous, device_context, rets, done, exec_args,
       rets_alloc_attrs, allow_dead_tensors](const Status& status) {
        Status s = status;
        if (s.ok()) {
          s = frame->SetArgs(*remote_args);
        }
        if (!s.ok()) {
          delete frame;
          delete remote_args;
          delete exec_args;
          done(s);
          return;
        }
        item->exec->RunAsync(
            *exec_args,
            [frame, rets, done, source_device, target_device,
             target_incarnation, rendezvous, device_context, remote_args,
             rets_alloc_attrs, allow_dead_tensors](const Status& status) {
              Status s = status;
              if (s.ok()) {
                s = frame->ConsumeRetvals(rets, allow_dead_tensors);
              }
              delete frame;
              if (!s.ok()) {
                delete remote_args;
                done(s);
                return;
              }
              s = ProcessFunctionLibraryRuntime::SendTensors(
                  target_device, source_device, "ret_", target_incarnation,
                  *rets, device_context, rets_alloc_attrs, rendezvous);
              delete remote_args;
              done(s);
            });
        delete exec_args;
      });
}

void FunctionLibraryRuntimeImpl::Run(const Options& opts, Handle handle,
                                     gtl::ArraySlice<Tensor> args,
                                     std::vector<Tensor>* rets,
                                     DoneCallback done) {
  if (opts.cancellation_manager && opts.cancellation_manager->IsCancelled()) {
    done(errors::Cancelled("Function was cancelled before it was started"));
    return;
  }
  Options run_opts = opts;
  if (opts.create_rendezvous) {
    auto* rendezvous = new PrivateIntraProcessRendezvous(device_mgr_);
    run_opts.rendezvous = rendezvous;
    run_opts.create_rendezvous = false;
    done = [done = std::move(done), rendezvous](const Status& status) mutable {
      delete rendezvous;
      done(status);
    };
  }

  LocalHandle local_handle = parent_->GetHandleOnDevice(device_name_, handle);
  if (local_handle == kInvalidLocalHandle) {
    parent_->Run(run_opts, handle, args, rets, done);
    return;
  }

  if (run_opts.runner == nullptr) {
    run_opts.runner = &default_runner_;
  }
  DCHECK(run_opts.runner != nullptr);

  Item* item = nullptr;
  Status s = GetOrCreateItem(local_handle, &item);
  if (!s.ok()) {
    done(s);
    return;
  }

  if (run_opts.remote_execution) {
    // NOTE(mrry): `RunRemote()` will set `exec_args->call_frame` for us.
    RunRemote(run_opts, handle, args, rets, item, std::move(done));
    return;
  }

  const FunctionBody* fbody = GetFunctionBody(handle);
  FunctionCallFrame* frame =
      new FunctionCallFrame(fbody->arg_types, fbody->ret_types);
  s = frame->SetArgs(args);
  if (!s.ok()) {
    delete frame;
    done(s);
    return;
  }

  profiler::TraceMeProducer activity(
      // To TraceMeConsumers in ExecutorState::Process/Finish.
      [&opts] {
        return profiler::TraceMeEncode("FunctionRun",
                                       {{"id", opts.step_id}, {"_r", 1}});
      },
      profiler::ContextType::kTfExecutor, opts.step_id,
      profiler::TraceMeLevel::kInfo);

  Executor::Args exec_args;
  ExecutorArgsFromOptions(run_opts, frame, &exec_args);

  bool allow_dead_tensors = run_opts.allow_dead_tensors;
  item->exec->RunAsync(
      // Executor args
      exec_args,
      // Done callback.
      [frame, rets, done, allow_dead_tensors](const Status& status) {
        Status s = status;
        if (s.ok()) {
          s = frame->ConsumeRetvals(rets, allow_dead_tensors);
        }
        delete frame;
        done(s);
      });
}

void FunctionLibraryRuntimeImpl::Run(const Options& opts, Handle handle,
                                     CallFrameInterface* frame,
                                     DoneCallback done) {
  if (opts.cancellation_manager && opts.cancellation_manager->IsCancelled()) {
    done(errors::Cancelled(""));
    return;
  }

  Options run_opts = opts;
  if (opts.create_rendezvous) {
    auto* rendezvous = new PrivateIntraProcessRendezvous(device_mgr_);
    run_opts.rendezvous = rendezvous;
    run_opts.create_rendezvous = false;
    done = [done = std::move(done), rendezvous](const Status& status) mutable {
      delete rendezvous;
      done(status);
    };
  }

  LocalHandle local_handle = parent_->GetHandleOnDevice(
      device_name_, handle, /*include_multi_device=*/true);
  if (local_handle == kInvalidLocalHandle) {
    parent_->Run(run_opts, handle, frame, done);
    return;
  }

  if (opts.remote_execution) {
    // NOTE(mrry): This bit is only set for a local function when `parent_`
    // calls back into this class, and the current implementation of
    // `ProcessFunctionLibraryRuntime` currently always uses the vector-based
    // `args`/`rets` interface.
    done(errors::Unimplemented("Remote calling with CallFrameInterface"));
    return;
  }

  Item* item = nullptr;
  Status s = GetOrCreateItem(local_handle, &item);
  if (!s.ok()) {
    done(s);
    return;
  }
  if (run_opts.runner == nullptr) {
    run_opts.runner = &default_runner_;
  }
  DCHECK(run_opts.runner != nullptr);

  profiler::TraceMeProducer activity(
      // To TraceMeConsumers in ExecutorState::Process/Finish.
      [&opts] {
        return profiler::TraceMeEncode("FunctionRun",
                                       {{"id", opts.step_id}, {"_r", 1}});
      },
      profiler::ContextType::kTfExecutor, opts.step_id,
      profiler::TraceMeLevel::kInfo);

  Executor::Args exec_args;
  ExecutorArgsFromOptions(run_opts, frame, &exec_args);
  item->exec->RunAsync(exec_args, std::move(done));
}

Status FunctionLibraryRuntimeImpl::PrepareRunSync(
    Handle handle, Options* run_opts, Item** out_item,
    std::unique_ptr<PrivateIntraProcessRendezvous>* out_rendezvous) {
  if (run_opts->cancellation_manager &&
      run_opts->cancellation_manager->IsCancelled()) {
    return errors::Cancelled("");
  }

  if (run_opts->remote_execution) {
    // NOTE(mrry): This bit is only set for a local function when `parent_`
    // calls back into this class, and the current implementation of
    // `ProcessFunctionLibraryRuntime` currently always uses the asynchronous
    // Run() method.
    return errors::Unimplemented("Remote calling with RunSync()");
  }

  if (run_opts->create_rendezvous) {
    *out_rendezvous =
        absl::make_unique<PrivateIntraProcessRendezvous>(device_mgr_);
    run_opts->rendezvous = out_rendezvous->get();
    run_opts->create_rendezvous = false;
  }

  LocalHandle local_handle = parent_->GetHandleOnDevice(
      device_name_, handle, /*include_multi_device=*/true);
  if (local_handle == kInvalidLocalHandle) {
    *out_item = nullptr;
    return Status::OK();
  }

  TF_RETURN_IF_ERROR(GetOrCreateItem(local_handle, out_item));

  if (run_opts->runner == nullptr) {
    run_opts->runner = &default_runner_;
  }
  DCHECK(run_opts->runner != nullptr);

  return Status::OK();
}

Status FunctionLibraryRuntimeImpl::RunSync(Options opts, Handle handle,
                                           gtl::ArraySlice<Tensor> args,
                                           std::vector<Tensor>* rets) {
  Item* item = nullptr;
  std::unique_ptr<PrivateIntraProcessRendezvous> rendezvous;
  TF_RETURN_IF_ERROR(PrepareRunSync(handle, &opts, &item, &rendezvous));
  if (item == nullptr) {
    return parent_->RunSync(opts, handle, args, rets);
  }

  Executor::Args exec_args;
  const FunctionBody* fbody = GetFunctionBody(handle);
  FunctionCallFrame frame(fbody->arg_types, fbody->ret_types);
  TF_RETURN_IF_ERROR(frame.SetArgs(args));
  ExecutorArgsFromOptions(opts, &frame, &exec_args);

  TF_RETURN_IF_ERROR(item->exec->Run(exec_args));
  return frame.ConsumeRetvals(rets, opts.allow_dead_tensors);
}

Status FunctionLibraryRuntimeImpl::RunSync(Options opts, Handle handle,
                                           CallFrameInterface* call_frame) {
  Item* item = nullptr;
  std::unique_ptr<PrivateIntraProcessRendezvous> rendezvous;
  TF_RETURN_IF_ERROR(PrepareRunSync(handle, &opts, &item, &rendezvous));
  if (item == nullptr) {
    return parent_->RunSync(opts, handle, call_frame);
  }

  Executor::Args exec_args;
  ExecutorArgsFromOptions(opts, call_frame, &exec_args);
  return item->exec->Run(exec_args);
}

bool FunctionLibraryRuntimeImpl::IsStateful(const string& func) const {
  const OpDef* op_def;
  const Status s = base_lib_def_->LookUpOpDef(func, &op_def);
  return s.ok() && op_def->is_stateful();
}

string FunctionLibraryRuntimeImpl::DebugString(Handle handle) {
  Item* item = nullptr;
  LocalHandle local_handle = parent_->GetHandleOnDevice(device_name_, handle);
  Status s = GetOrCreateItem(local_handle, &item);
  if (s.ok()) {
    if (item->graph) {
      return tensorflow::DebugString(item->graph.get());
    } else {
      return tensorflow::DebugString(item->func_graph->graph);
    }
  } else {
    return s.ToString();
  }
}

Status FunctionLibraryRuntimeImpl::Clone(
    std::unique_ptr<FunctionLibraryDefinition>* out_lib_def,
    std::unique_ptr<ProcessFunctionLibraryRuntime>* out_pflr,
    FunctionLibraryRuntime** out_flr, bool skip_flib_def) {
  TF_RETURN_IF_ERROR(parent_->Clone(env_, graph_def_version_,
                                    optimizer_.options(), out_lib_def, out_pflr,
                                    skip_flib_def));
  *out_flr = (*out_pflr)->GetFLR(device_->name());
  if (*out_flr != nullptr) {
    return Status::OK();
  } else {
    return errors::Internal("Cloning FunctionLibraryRuntime failed.");
  }
}

namespace {

struct CustomCreatorSingleton {
  mutex mu;
  CustomKernelCreator* custom_creator = nullptr;

  void Set(CustomKernelCreator* cb) {
    mutex_lock l(mu);
    custom_creator = cb;
  }

  CustomKernelCreator* Get() {
    mutex_lock l(mu);
    return custom_creator;
  }
};

CustomCreatorSingleton* GetCustomCreatorSingleton() {
  static CustomCreatorSingleton* ccs = new CustomCreatorSingleton;
  return ccs;
}

}  // namespace

const CustomKernelCreator* GetDefaultCustomKernelCreator() {
  return GetCustomCreatorSingleton()->Get();
}

void RegisterDefaultCustomKernelCreator(CustomKernelCreator* c) {
  GetCustomCreatorSingleton()->Set(c);
}

std::unique_ptr<FunctionLibraryRuntime> NewFunctionLibraryRuntime(
    const DeviceMgr* device_mgr, Env* env, const ConfigProto* config,
    Device* device, int graph_def_version,
    const FunctionLibraryDefinition* lib_def, thread::ThreadPool* thread_pool,
    const OptimizerOptions& optimizer_options,
    const SessionMetadata* session_metadata,
    ProcessFunctionLibraryRuntime* parent) {
  return std::unique_ptr<FunctionLibraryRuntime>(new FunctionLibraryRuntimeImpl(
      device_mgr, env, config, device, graph_def_version, lib_def, thread_pool,
      optimizer_options, session_metadata, parent));
}

class SymbolicGradientHelper {
 public:
  explicit SymbolicGradientHelper(const FunctionBody& f) : fbody_(&f) {}
  ~SymbolicGradientHelper() = default;

  std::unique_ptr<FunctionBody> Compute();

 private:
  const FunctionBody* fbody_;

  // Makes a copy of fbody_ in gbody.
  void Copy(FunctionBody* gbody);

  TF_DISALLOW_COPY_AND_ASSIGN(SymbolicGradientHelper);
};

void SymbolicGradientHelper::Copy(FunctionBody* gbody) {
  const Graph& src = *(fbody_->graph);
  gbody->graph = new Graph(src.op_registry());
  Graph* dst = gbody->graph;

  std::vector<Node*> node_map(src.num_node_ids());

  // Copy just the fdef attributes (copy '_noinline' and other similar flags to
  // the gradient function body).
  *(gbody->fdef.mutable_attr()) = fbody_->fdef.attr();

  // Copy the nodes.
  node_map[src.source_node()->id()] = dst->source_node();
  node_map[src.sink_node()->id()] = dst->sink_node();
  for (Node* n : src.op_nodes()) {
    node_map[n->id()] = dst->CopyNode(n);
  }

  // Copy the edges.
  for (const Edge* e : src.edges()) {
    Node* src_copy = node_map[e->src()->id()];
    Node* dst_copy = node_map[e->dst()->id()];
    dst->AddEdge(src_copy, e->src_output(), dst_copy, e->dst_input());
  }

  // Save inputs in copied graph.
  CHECK_EQ(fbody_->arg_types.size(), fbody_->arg_nodes.size());
  gbody->arg_types = fbody_->arg_types;
  for (std::size_t i = 0; i < fbody_->arg_nodes.size(); ++i) {
    gbody->arg_nodes.push_back(node_map[fbody_->arg_nodes[i]->id()]);
  }

  // Save outputs in copied graph.
  CHECK_EQ(fbody_->ret_types.size(), fbody_->ret_nodes.size());
  gbody->ret_types = fbody_->ret_types;
  for (std::size_t i = 0; i < fbody_->ret_nodes.size(); ++i) {
    gbody->ret_nodes.push_back(node_map[fbody_->ret_nodes[i]->id()]);
  }
}

std::unique_ptr<FunctionBody> SymbolicGradientHelper::Compute() {
  FunctionBody* gbody = new FunctionBody;
  Copy(gbody);  // copy fbody_ into gbody.

  Graph* g = gbody->graph;

  const int num_y = static_cast<int>(gbody->ret_nodes.size());

  // Populate 'y_node_outputs_' with node function body outputs.
  // Populate 'y_grad_nodes' with initial gradient nodes for each return node
  // of the original function body (these will be 'arg' nodes in the function
  // gradient body).
  std::vector<NodeOut> y_node_outputs;
  y_node_outputs.reserve(num_y);
  std::vector<NodeOut> y_grad_node_outputs;
  y_grad_node_outputs.reserve(num_y);
  for (int i = 0; i < num_y; ++i) {
    Node* y = gbody->ret_nodes[i];
    y_node_outputs.push_back({y, 0});
    DCHECK_EQ(y->type_string(), kRetOp);
    const DataType dtype = y->input_type(0);
    const int index = static_cast<int>(gbody->arg_nodes.size());
    Node* dy = AddArg(g, dtype, index);
    gbody->arg_types.push_back(dtype);
    gbody->arg_nodes.push_back(dy);
    y_grad_node_outputs.push_back({dy, 0});
  }

  // Populate 'x_nodes' with function args (excluding 'y_grad_node_outputs').
  const size_t num_x = fbody_->arg_nodes.size();
  std::vector<NodeOut> x_node_outputs;
  x_node_outputs.reserve(num_x);
  for (size_t i = 0; i < fbody_->arg_nodes.size(); ++i) {
    x_node_outputs.push_back({gbody->arg_nodes[i], 0});
  }

  // Call AddSymbolicGradients which will add nodes to graph 'g' that
  // compute the function gradient (adding an entry in 'x_grad_node_outputs'
  // for each node in 'x_node_outputs').
  std::vector<NodeOut> x_grad_node_outputs;
  TF_CHECK_OK(AddSymbolicGradients(y_node_outputs, x_node_outputs,
                                   y_grad_node_outputs, &x_grad_node_outputs,
                                   g));

  // Remove the old return nodes from the function body.
  for (Node* n : gbody->ret_nodes) {
    g->RemoveNode(n);
  }
  gbody->ret_types = fbody_->arg_types;
  // TODO(apassos): use the right dtype for gradients of  resource variables
  for (int i = 0; i < gbody->ret_types.size(); ++i) {
    if (gbody->ret_types[i] == DT_RESOURCE) {
      gbody->ret_types[i] = DT_FLOAT;
    }
  }
  gbody->ret_nodes.clear();
  // Add new return nodes to the function gradient body for each node
  // in 'x_grad_nodes'.
  const int arg_types_size = static_cast<int>(fbody_->arg_types.size());
  for (int i = 0; i < arg_types_size; ++i) {
    Endpoint grad = {x_grad_node_outputs[i].node, x_grad_node_outputs[i].index};
    Node* ret = AddRet(g, grad, i);
    gbody->ret_nodes.push_back(ret);
  }

  return std::unique_ptr<FunctionBody>(gbody);
}

std::unique_ptr<FunctionBody> SymbolicGradient(const FunctionBody& f) {
  return SymbolicGradientHelper(f).Compute();
}

}  // end namespace tensorflow
