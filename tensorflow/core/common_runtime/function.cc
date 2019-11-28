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
#include "tensorflow/core/common_runtime/graph_optimizer.h"
#include "tensorflow/core/common_runtime/memory_types.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/control_flow.h"
#include "tensorflow/core/graph/gradients.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/optimizer_cse.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/macros.h"
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
static Node* AddNoOp(StringPiece name, Graph* g) {
  NodeDef ndef;
  ndef.set_name(g->NewName(absl::StrCat(kNodeLabel, "/", name)));
  ndef.set_op("NoOp");
  Status s;
  Node* ret = g->AddNode(ndef, &s);
  TF_CHECK_OK(s);
  return ret;
}

static Node* AddIdentity(StringPiece name, Graph* g, Endpoint input) {
  DCHECK_LT(0, input.dtype());
  NodeDef ndef;
  ndef.set_name(g->NewName(absl::StrCat(kNodeLabel, "/", name)));
  ndef.set_op("Identity");
  ndef.add_input(input.name());
  AddNodeAttr("T", BaseType(input.dtype()), &ndef);
  Status s;
  Node* ret = g->AddNode(ndef, &s);
  TF_CHECK_OK(s);
  g->AddEdge(input.node, input.index, ret, 0);
  return ret;
}

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

  Status CreateKernel(const NodeDef& ndef, OpKernel** kernel) override;

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

Status FunctionLibraryRuntimeOverlay::CreateKernel(const NodeDef&, OpKernel**) {
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
                             const CustomKernelCreator* custom_kernel_creator,
                             const SessionMetadata* session_metadata,
                             ProcessFunctionLibraryRuntime* parent);

  ~FunctionLibraryRuntimeImpl() override;

  Status Instantiate(const string& function_name, AttrSlice attrs,
                     const InstantiateOptions& options,
                     Handle* handle) override;

  Status ReleaseHandle(Handle handle) override;

  const FunctionBody* GetFunctionBody(Handle handle) override;

  Status GetRetTypes(Handle handle, DataTypeVector* ret_types) override;

  Status CreateKernel(const NodeDef& ndef, OpKernel** kernel) override;

  void Run(const Options& opts, Handle handle, gtl::ArraySlice<Tensor> args,
           std::vector<Tensor>* rets, DoneCallback done) override;
  void Run(const Options& opts, Handle handle, CallFrameInterface* frame,
           DoneCallback done) override;

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
  const CustomKernelCreator* custom_kernel_creator_;
  const SessionMetadata* const session_metadata_;
  Executor::Args::Runner default_runner_;
  const string device_name_;

  std::function<Status(const string&, const OpDef**)> get_func_sig_;
  std::function<Status(const NodeDef&, OpKernel**)> create_kernel_;

  mutable mutex mu_;

  int next_handle_ GUARDED_BY(mu_);

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
    Executor::RendezvousFactory rendezvous_factory = nullptr;

    ~Item() {
      delete this->func_graph;
      delete this->exec;
      delete this->overlay_flr;
    }
  };
  std::unique_ptr<std::unordered_map<Handle, std::unique_ptr<Item>>> items_
      GUARDED_BY(mu_);

  ProcessFunctionLibraryRuntime* parent_ = nullptr;  // not owned.

  // Overloads the CreateKernel method, providing a FunctionLibraryRuntime
  // to use for kernel creation and execution. In particular, this method can
  // accept a FunctionLibraryRuntimeOverlay that overlays a different
  // FunctionLibraryDefinition.
  Status CreateKernel(const NodeDef& ndef, FunctionLibraryRuntime* flr,
                      OpKernel** kernel);
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
    const CustomKernelCreator* custom_kernel_creator,
    const SessionMetadata* session_metadata,
    ProcessFunctionLibraryRuntime* parent)
    : device_mgr_(dmgr),
      device_(device),
      env_(env),
      config_(config),
      graph_def_version_(graph_def_version),
      base_lib_def_(lib_def),
      optimizer_(optimizer_options),
      custom_kernel_creator_(custom_kernel_creator),
      session_metadata_(session_metadata),
      default_runner_(nullptr),
      device_name_(device_ == nullptr
                       ? ProcessFunctionLibraryRuntime::kDefaultFLRDevice
                       : device_->name()),
      next_handle_(0),
      items_(new std::unordered_map<Handle, std::unique_ptr<Item>>),
      parent_(parent) {
  get_func_sig_ = [this](const string& op, const OpDef** sig) {
    return base_lib_def_->LookUpOpDef(op, sig);
  };
  create_kernel_ = [this](const NodeDef& ndef, OpKernel** kernel) {
    return CreateKernel(ndef, kernel);
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

Status FunctionLibraryRuntimeImpl::CreateKernel(const NodeDef& ndef,
                                                OpKernel** kernel) {
  return CreateKernel(ndef, this, kernel);
}

Status FunctionLibraryRuntimeImpl::CreateKernel(const NodeDef& ndef,
                                                FunctionLibraryRuntime* flr,
                                                OpKernel** kernel) {
  // If a custom kernel creator is given, try that.
  Status s;
  if (custom_kernel_creator_ != nullptr &&
      custom_kernel_creator_->CanCreateKernel(*this, ndef)) {
    std::unique_ptr<OpKernel> ret;
    s = custom_kernel_creator_->CreateKernel(this, ndef, &ret);
    if (s.ok()) {
      *kernel = ret.release();
    } else {
      VLOG(2) << "Custom creator error: " << s;
    }
    return s;
  }

  const FunctionLibraryDefinition* lib_def =
      flr->GetFunctionLibraryDefinition();
  if (lib_def->Find(ndef.op()) == nullptr) {
    // A primitive operation. Creates the registered kernel.
    return CreateNonCachedKernel(device_, flr, ndef, graph_def_version_,
                                 kernel);
  }

  // Try to instantiate this function for the func/attr. Maybe it's
  // cached already.
  InstantiateOptions options;
  if (lib_def != base_lib_def_) {
    options.lib_def = lib_def;
  }
  Handle handle;
  TF_RETURN_IF_ERROR(
      Instantiate(ndef.op(), AttrSlice(&ndef.attr()), options, &handle));

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
  OpKernelConstruction construction(
      device_type, device_, device_->GetAllocator(AllocatorAttributes()), &ndef,
      &fbody->fdef.signature(), flr, fbody->arg_types, input_memory_types,
      fbody->ret_types, output_memory_types, graph_def_version_, &s);
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
      if (options.lib_def) {
        item->overlay_flr =
            new FunctionLibraryRuntimeOverlay(this, options.lib_def);
      }
      item->rendezvous_factory = [](const int64, const DeviceMgr* device_mgr,
                                    Rendezvous** r) {
        *r = new IntraProcessRendezvous(device_mgr);
        return Status::OK();
      };
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

void DumpGraph(StringPiece label, const Graph* g) {
  // TODO(zhifengc): Change Graph to record #nodes.
  VLOG(2) << "Graph " << label << " #nodes " << g->num_nodes() << " #edges "
          << g->num_edges();
  if (VLOG_IS_ON(5)) {
    for (const auto& line : str_util::Split(DebugString(g), '\n')) {
      VLOG(5) << "|| " << line;
    }
  }
}

void OptimizeGraph(FunctionLibraryRuntime* lib, std::unique_ptr<Graph>* g,
                   const GraphOptimizer::Options& graph_optimizer_options) {
  OptimizerOptions opts;
  opts.set_do_common_subexpression_elimination(true);
  opts.set_do_function_inlining(true);
  opts.set_do_constant_folding(true);
  GraphOptimizer optimizer(opts);
  optimizer.Optimize(lib, lib->env(), lib->device(), g,
                     graph_optimizer_options);
}

void OptimizeGraph(FunctionLibraryRuntime* lib, std::unique_ptr<Graph>* g) {
  OptimizeGraph(lib, g, GraphOptimizer::Options());
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
  std::unique_ptr<Graph> g(new Graph(lib_def));
  CopyGraph(*fbody->graph, g.get());

  PruneFunctionBody(fbody->fdef, g.get());
  optimizer_.Optimize(this, env(), device(), &g, /*shape_map=*/nullptr);
  TF_RETURN_IF_ERROR(EnsureMemoryTypes(DeviceType(device()->device_type()),
                                       device()->name(), g.get()));

  // Creates an executor based on the g. This must be done without
  // holding mu_ because create_kernel_ calls back into the library.
  LocalExecutorParams params;
  params.device = device_;
  params.function_library = flr;
  if (flr == this) {
    params.create_kernel = create_kernel_;
  } else {
    params.create_kernel = [this, flr](const NodeDef& ndef, OpKernel** kernel) {
      return CreateKernel(ndef, flr, kernel);
    };
  }
  params.delete_kernel = [](OpKernel* kernel) {
    DeleteNonCachedKernel(kernel);
  };
  params.rendezvous_factory = (*item)->rendezvous_factory;
  params.session_metadata = session_metadata_;
  std::unique_ptr<Executor> exec;
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
  int64 src_incarnation, target_incarnation;
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
    Rendezvous* rendezvous = new IntraProcessRendezvous(device_mgr_);
    run_opts.rendezvous = rendezvous;
    run_opts.create_rendezvous = false;
    done = [done = std::move(done), rendezvous](const Status& status) {
      rendezvous->Unref();
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
    Rendezvous* rendezvous = new IntraProcessRendezvous(device_mgr_);
    run_opts.rendezvous = rendezvous;
    run_opts.create_rendezvous = false;
    done = [done = std::move(done), rendezvous](const Status& status) {
      rendezvous->Unref();
      done(status);
    };
  }

  LocalHandle local_handle = parent_->GetHandleOnDevice(device_name_, handle);
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

  Executor::Args exec_args;
  ExecutorArgsFromOptions(run_opts, frame, &exec_args);
  item->exec->RunAsync(exec_args, std::move(done));
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
  TF_RETURN_IF_ERROR(parent_->Clone(
      env_, graph_def_version_, optimizer_.options(), custom_kernel_creator_,
      out_lib_def, out_pflr, skip_flib_def));
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
    const CustomKernelCreator* custom_kernel_creator,
    const SessionMetadata* session_metadata,
    ProcessFunctionLibraryRuntime* parent) {
  return std::unique_ptr<FunctionLibraryRuntime>(new FunctionLibraryRuntimeImpl(
      device_mgr, env, config, device, graph_def_version, lib_def, thread_pool,
      optimizer_options, custom_kernel_creator, session_metadata, parent));
}

bool RemoveDeadNodes(Graph* g) {
  VLOG(2) << "Removing dead nodes";
  std::unordered_set<const Node*> nodes;
  for (auto n : g->nodes()) {
    if (n->IsSource() || n->IsSink() || n->IsControlFlow() ||
        n->op_def().is_stateful()) {
      nodes.insert(n);
    }
  }
  return PruneForReverseReachability(g, std::move(nodes));
}

namespace {
// If 'edges' contains only 1 non-control edge, returns it. Otherwise,
// returns a nullptr.
const Edge* GetTheOnlyDataEdge(const EdgeSet& edges) {
  const Edge* ret = nullptr;
  for (const Edge* e : edges) {
    if (e->IsControlEdge() || ret) {
      // Don't touch it if there is a control edge.
      return nullptr;
    }
    if (IsRefType(e->src()->output_type(e->src_output()))) {
      // Don't touch it if the identity node is effectively de-reffing
      // a ref.
      return nullptr;
    }
    if (IsRecv(e->src()) || IsSwitch(e->src())) {
      // Don't touch it if the identity is introduced for control flow.
      // Recv disables all its successors if it receives a dead signal.
      // When Recv has an outgoing control edge, the current executor
      // would not disable the destination. The current solution (see
      // graph_partition.cc) is to add an identity after Recv and change
      // the control edge to be from this identity node. So the identity
      // can't be removed.
      return nullptr;
    }
    ret = e;
  }
  return ret;
}
}  // end namespace

bool RemoveIdentityNodes(Graph* g) {
  VLOG(2) << "Removing identity nodes";
  bool removed_any = false;
  gtl::InlinedVector<Node*, 8> matches;
  for (Node* n : g->nodes()) {
    if (!n->IsIdentity()) continue;
    if (!GetTheOnlyDataEdge(n->in_edges())) continue;

    // Some identity nodes are used as sink nodes to give names to output
    // tensors. These nodes are not going to be executed unless they are in the
    // fetch set. But if they are in the fetch set we don't want to remove them.
    if (n->out_edges().empty()) continue;

    matches.push_back(n);
  }
  if (!matches.empty()) {
    for (Node* n : matches) {
      const Edge* in = GetTheOnlyDataEdge(n->in_edges());
      for (const Edge* out : n->out_edges()) {
        if (out->IsControlEdge()) {
          g->AddControlEdge(in->src(), out->dst());
        } else {
          g->AddEdge(in->src(), in->src_output(), out->dst(), out->dst_input());
        }
      }
      VLOG(2) << "Remove Identity: " << n->DebugString();
      g->RemoveNode(n);
      removed_any = true;
    }
  }
  return removed_any;
}

bool RemoveListArrayConverter(Graph* g) {
  VLOG(2) << "Removing list array converter";
  gtl::InlinedVector<Node*, 8> matches;
  for (Node* n : g->nodes()) {
    if ((n->type_string() == "_ListToArray") ||
        (n->type_string() == "_ArrayToList")) {
      matches.push_back(n);
    }
  }
  bool removed_any = false;
  if (!matches.empty()) {
    for (Node* n : matches) {
      if (n->num_inputs() != n->num_outputs()) {
        continue;  // Not expected. Skip.
      }
      gtl::InlinedVector<Node*, 8> identity_nodes(n->num_inputs(), nullptr);

      const auto no_op = [&](StringPiece name) -> Node* {
        return AddNoOp(absl::StrCat(n->name(), "/", name), g);
      };

      const auto identity = [&](StringPiece name, Endpoint input) -> Node* {
        Node* node = AddIdentity(absl::StrCat(n->name(), "/", name), g, input);
        node->set_requested_device(input.node->def().device());
        return node;
      };

      // Process input edges first.
      Node* input_control_node = nullptr;
      for (const Edge* e : n->in_edges()) {
        if (e->IsControlEdge()) {
          if (input_control_node == nullptr) {
            // If node "n" has any control dependencies, adds a no-op
            // node (input_control_node) which the additional Identity
            // nodes depends on and the input_control_node depends on
            // the node "n"s control dependencies.
            input_control_node = no_op("input_control_node");
          }
          g->AddControlEdge(e->src(), input_control_node);
        } else {
          const int index = e->dst_input();
          Node** id_node = &identity_nodes[index];
          if (*id_node != nullptr) {
            LOG(ERROR)
                << "RemoveListArrayConverter unexpected duplicated input: "
                << e->dst_input();
            return removed_any;
          }
          *id_node = identity("input", {e->src(), e->src_output()});
        }
      }

      // If node "n" has any control dependencies, the added identity
      // nodes should have control dependencies on input_control_node.
      if (input_control_node != nullptr) {
        for (Node* id : identity_nodes) {
          g->AddControlEdge(input_control_node, id);
        }
      }

      Node* output_control_node = nullptr;
      for (const Edge* e : n->out_edges()) {
        if (e->IsControlEdge()) {
          if (output_control_node == nullptr) {
            // If node "n" is control-depended upon by other nodes,
            // adds a no-op node (output_control_node) which those
            // nodes will depend on and output_control_node depends on
            // all Identity nodes.
            output_control_node = no_op("output_control_node");
          }
          g->AddControlEdge(output_control_node, e->dst());
        } else {
          Node* id_node = identity_nodes[e->src_output()];
          if (id_node == nullptr) {
            LOG(ERROR) << "RemoveListArrayConverter unexpected missing input: "
                       << e->src_output();
            return removed_any;
          }
          CHECK(id_node);
          g->AddEdge(id_node, 0, e->dst(), e->dst_input());
        }
      }

      // If any nodes have control dependencies on node "n", those
      // nodes should have control dependencies on
      // output_control_node.
      if (output_control_node != nullptr) {
        for (Node* id : identity_nodes) {
          g->AddControlEdge(id, output_control_node);
        }
      }

      g->RemoveNode(n);
      removed_any = true;
    }
  }
  return removed_any;
}

Status NameAndAttrsFromFunctionCall(const NodeDef& call_def,
                                    NameAttrList* function) {
  if (call_def.op() == "PartitionedCall" ||
      call_def.op() == "StatefulPartitionedCall") {
    TF_RETURN_IF_ERROR(GetNodeAttr(call_def, "f", function));
  } else {
    function->set_name(call_def.op());
    *function->mutable_attr() = call_def.attr();
  }
  return Status::OK();
}

Status InstantiateFunctionCall(const NodeDef& call_def,
                               FunctionLibraryRuntime* flr,
                               FunctionLibraryRuntime::Handle* handle) {
  NameAttrList function;
  TF_RETURN_IF_ERROR(NameAndAttrsFromFunctionCall(call_def, &function));
  return flr->Instantiate(function.name(), AttrSlice(&function.attr()), handle);
}

namespace {

std::vector<string> InputDevices(const Node& caller) {
  std::vector<string> input_devices(caller.in_edges().size());
  std::vector<string> input_tensors(caller.in_edges().size());

  for (const Edge* edge : caller.in_edges()) {
    if (edge->IsControlEdge()) continue;
    const string& input_device = edge->src()->has_assigned_device_name()
                                     ? edge->src()->assigned_device_name()
                                     : edge->src()->requested_device();
    input_devices[edge->dst_input()] = input_device;
    input_tensors[edge->dst_input()] =
        absl::StrCat(edge->src()->name(), ":", edge->src_output());
  }

  if (VLOG_IS_ON(4)) {
    VLOG(4) << "Function instantiation input devices:";
    for (int i = 0; i < input_devices.size(); ++i) {
      if (input_tensors[i].empty()) continue;  // skip control edges
      VLOG(4) << "    [index " << i << "]"
              << " device: " << input_devices[i]
              << " (input: " << input_tensors[i] << ")";
    }
  }

  return input_devices;
}

// Place input nodes on the same device as the correspinding caller input
// node. Do not specify any placement for all other nodes.
class DefaultFunctionBodyPlacer : public InlinedFunctionBodyPlacer {
 public:
  explicit DefaultFunctionBodyPlacer(const Node& caller)
      : input_devices_(InputDevices(caller)) {}

  absl::optional<string> InputNodeDevice(int input_index) const override {
    return input_devices_[input_index];
  }
  absl::optional<string> OutputNodeDevice(int output_index) const override {
    return absl::nullopt;
  }
  bool ColocateInputOutputIdentities() const override { return false; }
  absl::optional<string> ControlNodeDevice() const override {
    return absl::nullopt;
  }
  absl::optional<string> BodyNodeDevice(const NodeDef& ndef) const override {
    return absl::nullopt;
  }

 private:
  const std::vector<string> input_devices_;
};

// Place all nodes on the same device as caller node.
class SingleDeviceFunctionBodyPlacer : public InlinedFunctionBodyPlacer {
 public:
  explicit SingleDeviceFunctionBodyPlacer(const Node& caller)
      : caller_device_(caller.def().device()) {}

  absl::optional<string> InputNodeDevice(int input_index) const override {
    return caller_device_;
  }
  absl::optional<string> OutputNodeDevice(int output_index) const override {
    return caller_device_;
  }
  bool ColocateInputOutputIdentities() const override { return false; }
  absl::optional<string> ControlNodeDevice() const override {
    return caller_device_;
  }
  absl::optional<string> BodyNodeDevice(const NodeDef& ndef) const override {
    return caller_device_;
  }

 private:
  const string caller_device_;
};

// Place input nodes on the same device as the correspinding caller input
// node. Do not place output node. Place control nodes on the same device as
// caller node. For all function body nodes overrides job, replica and task
// parts of the device assignment to match function caller node.
class MultiDeviceFunctionBodyPlacer : public InlinedFunctionBodyPlacer {
 public:
  explicit MultiDeviceFunctionBodyPlacer(const Node& caller)
      : caller_device_(caller.def().device()),
        input_devices_(InputDevices(caller)) {
    has_parsed_caller_device_ =
        DeviceNameUtils::ParseFullName(caller_device_, &caller_parsed_device_);
  }

  absl::optional<string> InputNodeDevice(int input_index) const override {
    return input_devices_[input_index];
  }
  absl::optional<string> OutputNodeDevice(int output_index) const override {
    return absl::nullopt;
  }
  bool ColocateInputOutputIdentities() const override { return true; }
  absl::optional<string> ControlNodeDevice() const override {
    return caller_device_;
  }
  absl::optional<string> BodyNodeDevice(const NodeDef& ndef) const override {
    // TODO(ezhulenev): If function would have been instantiated as a
    // multi-device function and executed via FunctionLibraryRuntime, it could
    // be potentially placed on any available device. However there are multiple
    // tests relying on this assumption. Fix them, and remove this line.
    if (ndef.device().empty()) return caller_device_;

    if (!has_parsed_caller_device_) return ndef.device();

    DeviceNameUtils::ParsedName ndef_parsed_device;
    if (!DeviceNameUtils::ParseFullName(ndef.device(), &ndef_parsed_device))
      return ndef.device();

    if (caller_parsed_device_.has_job) {
      ndef_parsed_device.has_job = caller_parsed_device_.has_job;
      ndef_parsed_device.job = caller_parsed_device_.job;
    }

    if (caller_parsed_device_.has_replica) {
      ndef_parsed_device.has_replica = caller_parsed_device_.has_replica;
      ndef_parsed_device.replica = caller_parsed_device_.replica;
    }

    if (caller_parsed_device_.has_task) {
      ndef_parsed_device.has_task = caller_parsed_device_.has_task;
      ndef_parsed_device.task = caller_parsed_device_.task;
    }
    return DeviceNameUtils::ParsedNameToString(ndef_parsed_device);
  }

 private:
  string caller_device_;
  bool has_parsed_caller_device_;
  DeviceNameUtils::ParsedName caller_parsed_device_;
  std::vector<string> input_devices_;
};

}  // namespace

std::unique_ptr<InlinedFunctionBodyPlacer>
InlinedFunctionBodyPlacer::DefaultPlacer(const Graph& graph,
                                         const Node& caller) {
  VLOG(3) << "Create default placer for inlined function body.";
  return absl::make_unique<DefaultFunctionBodyPlacer>(caller);
}

std::unique_ptr<InlinedFunctionBodyPlacer>
InlinedFunctionBodyPlacer::SingleDevicePlacer(const Graph& graph,
                                              const Node& caller) {
  VLOG(3) << "Create single device placer for inlined function body.";
  return absl::make_unique<SingleDeviceFunctionBodyPlacer>(caller);
}

std::unique_ptr<InlinedFunctionBodyPlacer>
InlinedFunctionBodyPlacer::MultiDevicePlacer(const Graph& graph,
                                             const Node& caller) {
  VLOG(3) << "Create multi device placer for inlined function body.";
  return absl::make_unique<MultiDeviceFunctionBodyPlacer>(caller);
}

namespace {

Status ValidateNoInline(const FunctionBody* fbody) {
  const auto attr = AttrSlice(&fbody->fdef.attr());
  bool noinline = false;
  if (TryGetNodeAttr(attr, kNoInlineAttr, &noinline) && noinline) {
    return errors::InvalidArgument(
        "Can't inline function marked with '_noinline'");
  }
  return Status::OK();
}

using OutputControlSrc = InlineFunctionBodyOptions::OutputControlSource;

// Propagate the debug info of `nodes` in function `func` to the `target` node.
// If the debug info of any node is missing, its node name and function name
// is used.
void PropagateDebugInfoToNode(const string& func,
                              const std::vector<const Node*>& nodes,
                              NodeDef* target) {
  if (nodes.empty() || target->has_experimental_debug_info()) {
    return;
  }
  for (const Node* node : nodes) {
    const auto& node_def = node->def();
    if (node_def.has_experimental_debug_info()) {
      target->mutable_experimental_debug_info()->MergeFrom(
          node_def.experimental_debug_info());
    } else {
      target->mutable_experimental_debug_info()->add_original_node_names(
          node_def.name());
      target->mutable_experimental_debug_info()->add_original_func_names(func);
    }
  }
}
}  // namespace

string InlineFunctionBodyOptions::DebugString() const {
  const auto true_false = [](bool b) { return b ? "true" : "false"; };

  const auto keep_caller_node_str = [this]() -> string {
    switch (keep_caller_node) {
      case KeepCallerNode::kDoNotKeep:
        return "DoNotKeep";
      case KeepCallerNode::kFetchable:
        return "Fetchable";
      case KeepCallerNode::kTargetable:
        return "Targetable";
    }
  };

  return absl::StrCat(
      "disable_inlining=", true_false(disable_inlining),
      ", ignore_noinline=", true_false(ignore_noinline),
      ", inline_impl_selection_group_functions=",
      true_false(inline_impl_selection_group_functions),
      ", keep_caller_node=", keep_caller_node_str(), ", output_control_src=",
      output_control_src == OutputControlSrc::kDataOutputs ? "DataOutputs"
                                                           : "ControlOutputs",
      ", inlined_function_body_placer=", inlined_function_body_placer.name,
      ", uniquify_frame_names=", true_false(uniquify_frame_names));
}

Status ValidateInlining(const Node* node, const FunctionBody* fbody,
                        const InlineFunctionBodyOptions& options) {
  // TODO(ezhulenev): Currently common_runtime function inlining can't guarantee
  // that all side-effectful ops will be executed after inlining. See Grappler
  // function_optimizer for details. Unify all function inlining mechanism.
  // Do not inline if `!fbody->control_ret_nodes.empty()`.

  const auto num_node_inputs = static_cast<size_t>(node->num_inputs());
  const auto num_node_outputs = static_cast<size_t>(node->num_outputs());

  if (num_node_inputs != fbody->arg_types.size() ||
      num_node_inputs != fbody->arg_nodes.size()) {
    return errors::InvalidArgument(
        "Node inputs do not match function arguments: inputs=", num_node_inputs,
        " arg_types=", fbody->arg_types.size(),
        " arg_nodes=", fbody->arg_nodes.size());
  }

  if (num_node_outputs != fbody->ret_types.size() ||
      num_node_outputs != fbody->ret_nodes.size()) {
    return errors::InvalidArgument(
        "Node outputs do not match function returns: outputs=",
        num_node_outputs, " ret_types=", fbody->ret_types.size(),
        " ret_nodes=", fbody->ret_nodes.size());
  }

  for (int i = 0; i < node->num_inputs(); ++i) {
    if (node->input_type(i) != fbody->arg_types[i]) {
      return errors::InvalidArgument(
          "Node input type doesn't match function argument type: ",
          node->input_type(i), " != ", fbody->arg_types[i], " @ index=", i);
    }
  }
  for (int i = 0; i < node->num_outputs(); ++i) {
    if (node->output_type(i) != fbody->ret_types[i]) {
      return errors::InvalidArgument(
          "Node output type doesn't match function return type: ",
          node->output_type(i), " != ", fbody->ret_types[i], " @ index=", i);
    }
  }

  if (options.disable_inlining) {
    return errors::InvalidArgument(
        "Function inlining explicitly disabled by 'options.disable_inlining'");
  }

  if (!options.inline_impl_selection_group_functions) {
    bool is_impl_selection_group_function =
        fbody->fdef.attr().find("api_implements") != fbody->fdef.attr().end();
    if (is_impl_selection_group_function) {
      return errors::InvalidArgument(
          "Inlining of implementation selection group function ",
          fbody->fdef.signature().name(),
          " is disabled by options.inline_impl_selection_group_functions");
    }
  }

  if (!options.ignore_noinline) {
    TF_RETURN_IF_ERROR(ValidateNoInline(fbody));
  }

  return Status::OK();
}

// Function inlining must preserve function execution semantics with regards to
// side-effects visibility. Tensorflow in Eager mode has an automatic control
// dependencies tracking mechanism, which enforces well-defined execution order
// of all side-effects. Any other frontend (e.g. Swift) must produce graphs
// following the same rules, to ensure that function inlining works correctly.
//
// IMPORTANT: Currently we do not have a true notion of "side-effectful" node,
// we assume that all stateful nodes might have side-effects, though it's not
// true in practice, e.g. `ReadVariableOp` doesn't have an observable
// side-effect.
//
// Automatic control dependency rules in Tensorflow 2.0 (python in eager mode):
//
// 1) When a function has a resource (DT_RESOURCE data type) input argument it
//   "captures" the mutable resource.  This is implemented by automatically
//    adding a incoming control edge from the previous side-effectful op
//    touching that resource, and an outgoing control edge to the next
//    side-effectful op using the same resource. This serializes the mutations
//    of the resource to make graph execution deterministic.
//
// 2) All stateful ops inside a function body are guaranteed to execute in
//    program order, this is achieved by adding control edges between stateful
//    ops at graph construction time. Stateful ops (or ops that must execute)
//    should be in the function control return set. Having a data edge to the
//    regular function output might be not enough, because after function
//    inlining it might happen that data output is unused.
//
// 3) Furthermore, all ops accepting the same resource as an input are
//    guaranteed to run in program order. This is also done by adding control
//    edges at graph construction time. The last op touching the resource
//    must be in a control return set, which will guarantee that all side
//    effects to the resource will happen before function completion.
//
// Function inlining must preserve side-effect visibility:
//
// 1) All side-effects to the captured resources, that happened before function
//    call must be visible to the function body nodes using that resources.
//
// 2) All side-effects to the captured resources, that happened inside function
//    body, must be visible to every op/function using that resource after the
//    function call completed.
//
// To guarantee that these properties are preserved after inlining we:
//
// 1) Create "input_control_node" NoOp. Function call node incoming control
//    edges will be forwarded *to* this node. Function inputs (Identity nodes)
//    will have a control edge *from* this node. If function body has nodes
//    without inputs, they will have a control edge *from* this node.
//
// 2) Create "output_control_node" NoOp. All nodes that have incoming control
//    edge *from* the function call node, will be forwarded to this node.
//
//    We have two options for choosing which nodes will have a control edge *to*
//    the "output control node":
//       a) control returns            (`control_ret` field in FunctionDef)
//       b) data returns               (`ret` field in FunctionDef)
//
//    We do a) for multi-device function calls in Tensorflow v2 and b)
//    for the rest for compatibility with Tensorflow v1.
//
//    Following the automatic control dependencies tracking rules, a node that
//    has an incoming control edge from the function call node is dependent on
//    the side-effects happening inside the function body. The output control
//    node will guarantee side-effects execution order.
//
//    If function call node doesn't have an outgoing control edge, it means that
//    no one is interested in observing side-effects that might have happened.
//
// Function inlining might leave the graph in partially-placed state. Function
// inlining caller must call Placer to guarantee that all nodes are placed.
//
// Function inlining with `options.override_device=true` will leave graph in
// fully placed state, by overriding all inlined nodes devices with the caller
// node device, but it will make functions always single-device. These functions
// after inlining will not be able to handle resources on multiple devices. This
// is currently acceptable for XLA use cases (XLA cluster is always executed on
// a single device).
//
// TODO(ezhulenev): Documentation above is ahead of implementation below.
Status InlineFunctionBody(const FunctionLibraryDefinition& flib_def, Graph* g,
                          Node* caller, const FunctionBody* fbody,
                          const InlineFunctionBodyOptions& options) {
  VLOG(3) << "Inline function call: " << SummarizeNode(*caller) << " ["
          << options.DebugString() << "]";

  Status validation = ValidateInlining(caller, fbody, options);
  if (!validation.ok()) {
    return errors::Internal("Inlining mismatch: ", validation.error_message());
  }

  // Placer is responsible for assigning devices for all nodes that we will add
  // to the graph.
  const std::unique_ptr<InlinedFunctionBodyPlacer> placer =
      options.inlined_function_body_placer.get(*g, *caller);

  // We can't possibly introduce a duplicate control edge during function
  // inlining, so we skip this check in calls to the 'g->AddControlEdge(...)'.
  static constexpr bool kDoNotCheckDuplicates = true;

  // ------------------------------------------------------------------------ //
  // Helper functions to create `NoOp` and `Identity` nodes for auxiliary
  // control nodes and inlined function inputs and outputs.

  // Add a NoOp node for function control inputs/outputs.
  const auto no_op = [&](StringPiece name) -> Node* {
    Node* node = AddNoOp(absl::StrCat(caller->name(), "/", name), g);
    const absl::optional<string> device = placer->ControlNodeDevice();
    if (device.has_value()) node->set_requested_device(*device);
    return node;
  };

  // Add an Identity node for function input.
  const auto input_identity = [&](StringPiece name, Endpoint input,
                                  int index) -> Node* {
    Node* node = AddIdentity(absl::StrCat(caller->name(), "/", name), g, input);
    const absl::optional<string> device = placer->InputNodeDevice(index);
    if (device.has_value()) node->set_requested_device(*device);
    bool colocate_identity = placer->ColocateInputOutputIdentities();
    if (colocate_identity) {
      node->AddAttr(kColocationAttrName,
                    std::vector<string>{absl::StrCat(kColocationGroupPrefix,
                                                     input.node->name())});
    }
    return node;
  };

  // Add an Identity node for function output.
  const auto output_identity = [&](StringPiece name, Endpoint input,
                                   int index) -> Node* {
    Node* node = AddIdentity(absl::StrCat(caller->name(), "/", name), g, input);
    const absl::optional<string> device = placer->OutputNodeDevice(index);
    if (device.has_value()) node->set_requested_device(*device);
    bool colocate_identity = placer->ColocateInputOutputIdentities();
    if (colocate_identity) {
      node->AddAttr(kColocationAttrName,
                    std::vector<string>{absl::StrCat(kColocationGroupPrefix,
                                                     input.node->name())});
    }
    return node;
  };

  // ------------------------------------------------------------------------ //
  // Input edges. For data edges coming into "caller", we first compute the
  // <src>:<src_output> for the i-th input in "inputs".
  // If "caller" has any input control dependencies, we add a NoOp
  // node "input_control_node", which depends on "caller"'s control inputs.
  std::vector<Endpoint> inputs(caller->num_inputs());
  Node* input_control_node = nullptr;
  for (const Edge* e : caller->in_edges()) {
    if (e->IsControlEdge()) {
      if (input_control_node == nullptr) {
        input_control_node = no_op("input_control_node");
      }
      g->AddControlEdge(e->src(), input_control_node, kDoNotCheckDuplicates);
    } else {
      inputs[e->dst_input()] = {e->src(), e->src_output()};
    }
  }
  if (input_control_node != nullptr) {
    VLOG(3) << "Created input control node: " << input_control_node->name();
  }

  // ------------------------------------------------------------------------ //
  // Duplicate fbody->graph into 'g'.  First, we copy the nodes of
  // fbody->graph into 'g' except the source and sink nodes.  We copy
  // edges among nodes in 'fbody->graph'.
  //
  // If 'x' is a node in fbody->graph and its copy in 'g' is 'y', we
  // remember 'y' in node_map[x->id()].
  std::vector<Node*> node_map(fbody->graph->num_node_ids());
  for (Node* n : fbody->graph->op_nodes()) {
    NodeDef ndef = n->def();

    // Maybe override requested node device assignment.
    const absl::optional<string> device = placer->BodyNodeDevice(ndef);
    if (device.has_value()) ndef.set_device(*device);

    // Add inlined function name to inlined node debug information.
    PropagateDebugInfoToNode(fbody->fdef.signature().name(), {n}, &ndef);

    // Add the function node name as a prefix:
    //  1) to node name to avoid collisions
    //  2) to frame name to avoid multiple LoopCond nodes in one frame
    //  3) to colocation attribute
    const string prefix = strings::StrCat(caller->name(), "/");
    TF_RETURN_IF_ERROR(AddPrefixAndSuffixToNode(prefix, /*suffix=*/"", &ndef,
                                                options.uniquify_frame_names));

    Status added_node;
    Node* clone = g->AddNode(ndef, &added_node);
    TF_CHECK_OK(added_node);
    node_map[n->id()] = clone;

    // If there is an input control node, and one of:
    // a) the node has no data or control inputs, or
    // b) the node is a function call (including SymbolicGradient),
    //    then add a control edge from the input control node to the clone (only
    //    if it does not already have a control input).
    //
    // We must not execute any nodes if the original function call would not
    // have executed. This is especially critical when the function call is
    // inside a control-flow construct like tf.cond(). Case (a) ensures that
    // such nodes do not run.
    //
    // The purpose of case (b) is to ensure that instances of case (a) created
    // by further inlining steps also receive the control dependency.
    //
    // This edge is required to transfer execution frame down to all function
    // body nodes of inlined nested function calls.
    if (input_control_node) {
      const auto is_input_edge = [](const Edge* e) -> bool {
        return !e->src()->IsSource();
      };
      const auto is_control_edge = [](const Edge* e) -> bool {
        return !e->src()->IsSource() && e->IsControlEdge();
      };

      // Forward execution frame if:
      //
      // a) The node has no data or control inputs.
      // b) OR the node is a function call without control inputs (control edge
      //    will be used in nested function inlining to forward execution frame
      //    to constants inside the function body).
      //
      // c) Do not forward control frame to function argument nodes, they will
      //    be connected to the corresponding function input later.
      const bool forward_execution_frame =
          (absl::c_none_of(n->in_edges(), is_input_edge) ||       // (a)
           (n->IsFunctionCall() &&                                // (b)
            absl::c_none_of(n->in_edges(), is_control_edge))) &&  //
          !n->IsArg();                                            // (c)

      if (forward_execution_frame) {
        VLOG(4) << "Add control edge from input control node to: "
                << clone->name();
        g->AddControlEdge(input_control_node, clone, kDoNotCheckDuplicates);
      }
    }
  }
  for (const Edge* e : fbody->graph->edges()) {
    if (e->src()->IsSource() || e->src()->IsSink() || e->dst()->IsSource() ||
        e->dst()->IsSink()) {
      continue;
    }
    Node* src_copy = node_map[e->src()->id()];
    Node* dst_copy = node_map[e->dst()->id()];
    g->AddEdge(src_copy, e->src_output(), dst_copy, e->dst_input());
  }

  // ------------------------------------------------------------------------ //
  // Connect input edges.
  //
  // We create one Identity node for each input. Then, we connect inputs[i] to
  // the i-th identity node added. The nodes that previously connected
  // to the j-th output of i-th arg node are reconnected to the i-th
  // identity node.
  //
  // The added identity nodes depend on "input_control_node".
  VLOG(4) << "Add input Identity nodes for each function argument:";
  for (std::size_t i = 0; i < fbody->arg_nodes.size(); ++i) {
    Node* arg = node_map[fbody->arg_nodes[i]->id()];
    Node* n = input_identity("input", inputs[i], i);
    VLOG(4) << "    [index " << i << "] "
            << fbody->fdef.signature().input_arg(i).name() << " as "
            << n->name() << " (input: " << inputs[i].name()
            << ", requested_device: " << n->requested_device() << ")";

    if (input_control_node) {
      g->AddControlEdge(input_control_node, n, kDoNotCheckDuplicates);
    }
    for (const Edge* e : arg->out_edges()) {
      if (e->IsControlEdge()) {
        g->AddControlEdge(n, e->dst(), kDoNotCheckDuplicates);
      } else {
        g->AddEdge(n, 0, e->dst(), e->dst_input());
      }
    }
    node_map[fbody->arg_nodes[i]->id()] = n;
    g->RemoveNode(arg);  // 'arg' is disconnected.
  }

  // ------------------------------------------------------------------------ //
  // Connect output edges.
  //
  // For i-th return node in fbody->graph, we add in "g" an identity node
  // (outputs[i-th]). We then reconnect every incoming edge into the i-th return
  // node to the added identity node.
  //
  // For every data edge coming out of "callee"s i-th output, we reconnect it to
  // the i-th identity added above.
  //
  // If "callee" is control-depended upon by any other nodes, we add a NoOp node
  // "output_control_node". "output_control_node" depends on all identity nodes
  // added above or on all control return nodes (controlled by
  // `options.output_control_src` value). And nodes previously depend on
  // "callee" is changed to depend on "output_control_node".
  //
  // If `keep_node_fetchable` is `true` we always add an output control node, to
  // guarantee that executing a fetchable node will execute all side-effects.
  VLOG(4) << "Add output Identity nodes for each function output argument:";
  std::vector<Node*> outputs(caller->num_outputs());
  for (std::size_t i = 0; i < fbody->ret_nodes.size(); ++i) {
    Node* ret = node_map[fbody->ret_nodes[i]->id()];
    Endpoint data;  // Data input for the ret node.
    for (const Edge* e : ret->in_edges()) {
      if (!e->IsControlEdge()) {
        data = {e->src(), e->src_output()};
        break;
      }
    }
    CHECK(data.node != nullptr);
    Node* n = output_identity("output", data, i);
    outputs[i] = n;
    VLOG(4) << "    [index " << i << "] "
            << fbody->fdef.signature().output_arg(i).name() << " as "
            << n->name() << " (ret: " << data.node->name() << ":" << data.index
            << ", requested_device: " << n->requested_device() << ")";
    for (const Edge* e : ret->in_edges()) {
      if (e->IsControlEdge()) {
        g->AddControlEdge(e->src(), n, kDoNotCheckDuplicates);
      }
    }
    g->RemoveNode(ret);  // 'ret' is disconnected.
  }

  Node* output_control_node = nullptr;
  const bool has_control_outputs = absl::c_any_of(
      caller->out_edges(), [](const Edge* e) { return e->IsControlEdge(); });

  using KeepCallerNode = InlineFunctionBodyOptions::KeepCallerNode;
  const bool keep_caller_node =
      options.keep_caller_node == KeepCallerNode::kFetchable ||
      options.keep_caller_node == KeepCallerNode::kTargetable;

  if (has_control_outputs || keep_caller_node) {
    output_control_node = no_op("output_control_node");
    VLOG(4) << "Add output control node: " << output_control_node->name();
    if (options.output_control_src == OutputControlSrc::kDataOutputs) {
      for (Node* n : outputs) {
        VLOG(4) << "    [data output] add control edge from: " << n->name();
        g->AddControlEdge(n, output_control_node, kDoNotCheckDuplicates);
      }
    } else {
      for (Node* fbody_node : fbody->control_ret_nodes) {
        Node* n = node_map[fbody_node->id()];
        VLOG(4) << "    [control output] add control edge from: " << n->name();
        g->AddControlEdge(n, output_control_node, kDoNotCheckDuplicates);
      }
    }
  }

  // We can't leave output control node without incoming control edges, because
  // in this case outgoing control edge will loose execution frame information.
  // We connect input_control_node and output_control_node with a control edge
  // to forward execution frame to the controlled nodes. Above we add a control
  // edge to all function calls inside function body, to guarantee that we will
  // always have input_control_node when we need it.
  if (output_control_node && output_control_node->in_edges().empty()) {
    if (input_control_node) {
      VLOG(4)
          << "Add add a control edge between input and output control nodes: "
          << input_control_node->name() << " to "
          << output_control_node->name();
      g->AddControlEdge(input_control_node, output_control_node,
                        kDoNotCheckDuplicates);
    } else {
      VLOG(4) << "Function inlining potentially dropped execution frame "
                 "information from outgoing control edges.";
    }
  }

  for (const Edge* e : caller->out_edges()) {
    if (e->IsControlEdge()) {
      g->AddControlEdge(output_control_node, e->dst(), kDoNotCheckDuplicates);
    } else {
      g->AddEdge(outputs[e->src_output()], 0, e->dst(), e->dst_input());
    }
  }

  // ------------------------------------------------------------------------ //
  // Add an IdentityN or NoOp node in-place of caller node to keep `caller`
  // fetchable or targetable.

  if (keep_caller_node) {
    std::vector<NodeBuilder::NodeOut> output_tensors;
    absl::c_transform(outputs, std::back_inserter(output_tensors),
                      [](Node* n) { return NodeBuilder::NodeOut(n, 0); });

    Node* caller_substitute_node;
    if (options.keep_caller_node == KeepCallerNode::kTargetable ||
        output_tensors.empty()) {
      // IdentityN node must have at least one data input. If function has no
      // data outputs, we can't keep it fetchable.
      TF_CHECK_OK(NodeBuilder(caller->name(), "NoOp")
                      .Device(caller->requested_device())
                      .ControlInput(output_control_node)
                      .Finalize(g, &caller_substitute_node));

    } else if (options.keep_caller_node == KeepCallerNode::kFetchable) {
      TF_CHECK_OK(NodeBuilder(caller->name(), "IdentityN")
                      .Device(caller->requested_device())
                      .Input(output_tensors)
                      .ControlInput(output_control_node)
                      .Finalize(g, &caller_substitute_node));
    }
  }

  // ------------------------------------------------------------------------ //
  // 'caller' is replaced with inlined function body nodes and maybe IdentityN
  // to keep it fetchable.
  VLOG(3) << "Successfully inlined function call node: " << caller->name();
  g->RemoveNode(caller);

  return Status::OK();
}

bool IsFunctionCall(const FunctionLibraryDefinition& lib_def,
                    const Node& node) {
  return node.IsFunctionCall();
}

bool ExpandInlineFunctions(FunctionLibraryRuntime* lib, Graph* graph,
                           const ExpandInlineFunctionsOptions& options) {
  std::vector<std::pair<Node*, const FunctionBody*>> candidates;

  const FunctionLibraryDefinition* fld = lib->GetFunctionLibraryDefinition();

  for (Node* node : graph->nodes()) {
    // Skip nodes that are not function calls or SymbolicGradient calls.
    if (!IsFunctionCall(*lib->GetFunctionLibraryDefinition(), *node)) {
      continue;
    }
    // Skip function calls that marked noinline.
    bool noinline;
    if (fld->GetAttr(*node, kNoInlineAttr, &noinline).ok() && noinline) {
      VLOG(3) << "noinline: " << SummarizeNode(*node);
      continue;
    }
    FunctionLibraryRuntime::Handle handle;
    Status s = InstantiateFunctionCall(node->def(), lib, &handle);
    if (!s.ok()) {
      LOG(ERROR) << "Failed to instantiate a function:  " << s.error_message();
      continue;
    }
    const FunctionBody* fbody = lib->GetFunctionBody(handle);
    CHECK_NOTNULL(fbody);
    candidates.emplace_back(node, fbody);
  }

  bool inlined_any = false;
  for (const auto& p : candidates) {
    Status inlined = InlineFunctionBody(*fld, graph, p.first, p.second,
                                        p.first->IsPartitionedCall()
                                            ? options.multi_device_options
                                            : options.native_options);
    if (inlined.ok()) {
      inlined_any = true;
    } else {
      VLOG(1) << "Failed to inline function call: node=" << p.first->name()
              << " error=" << inlined.error_message();
    }
  }

  // TODO(ezhulenev): Release handles for inlined function calls.

  return inlined_any;
}

string NewName(const Node* n, bool pretty) {
  if (pretty) {
    return strings::StrCat(n->type_string(), n->id());
  } else {
    return strings::StrCat("n", n->id());
  }
}

// TODO(zhifengc): Maybe this should be the default Graph::AsGraphDef.
// and stash the original NodeDef name as an attr for documentation
// purpose.
void ToGraphDef(const Graph* g, GraphDef* gdef, bool pretty) {
  // We visit nodes in forward topological sort order, which is a
  // possible execution order of the graph.
  gtl::InlinedVector<const Edge*, 4> inputs;
  gdef->Clear();
  *gdef->mutable_versions() = g->versions();

  std::vector<Node*> start_nodes;
  for (Node* n : g->nodes()) {
    if (n->out_edges().empty()) {
      start_nodes.push_back(n);
    }
  }

  ReverseDFSFrom(*g, start_nodes, nullptr, [gdef, pretty, &inputs](Node* n) {
    if (!n->IsOp()) return;
    NodeDef* ndef = gdef->add_node();
    ndef->set_name(NewName(n, pretty));
    ndef->set_op(n->type_string());
    for (const auto& attr : n->attrs()) {
      (*ndef->mutable_attr())[attr.first] = attr.second;
    }

    if (!n->assigned_device_name().empty()) {
      ndef->set_device(n->assigned_device_name());
    } else {
      ndef->set_device(n->requested_device());
    }

    inputs.clear();
    inputs.resize(n->num_inputs());
    for (const Edge* e : n->in_edges()) {
      if (e->IsControlEdge()) {
        inputs.push_back(e);
      } else {
        if (inputs[e->dst_input()] == nullptr) {
          inputs[e->dst_input()] = e;
        } else {
          LOG(WARNING) << "Malformed graph node. multiple input edges: "
                       << n->DebugString();
        }
      }
    }
    // node->name() is merely NodeDef::name, which are not guaranteed
    // to be unique and stable after optimization rewrites. Therefore,
    // we use "n<node id>" instead.
    for (const Edge* e : inputs) {
      if (e == nullptr) {
        ndef->add_input("unknown");
        continue;
      }
      const string srcname = NewName(e->src(), pretty);
      if (!e->src()->IsOp()) {
      } else if (e->IsControlEdge()) {
        ndef->add_input(strings::StrCat("^", srcname));
      } else if (e->src_output() == 0) {
        ndef->add_input(srcname);
      } else {
        ndef->add_input(strings::StrCat(srcname, ":", e->src_output()));
      }
    }
  });
}

string DebugString(const Graph* g) {
  GraphDef gdef;
  ToGraphDef(g, &gdef);
  return DebugString(gdef);
}

FunctionBody::FunctionBody(const FunctionDef& f, DataTypeSlice arg_t,
                           DataTypeSlice ret_t, Graph* g)
    : fdef(f),
      graph(g),
      arg_types(arg_t.begin(), arg_t.end()),
      ret_types(ret_t.begin(), ret_t.end()) {
  // 1. Find regular Arg/Ret nodes.
  this->arg_nodes.resize(arg_types.size());
  this->ret_nodes.resize(ret_types.size());
  for (Node* n : this->graph->op_nodes()) {
    gtl::InlinedVector<Node*, 4>* node_vec;
    if (n->type_string() == kRetOp || n->type_string() == kDeviceRetOp) {
      node_vec = &this->ret_nodes;
    } else if (n->type_string() == kArgOp || n->type_string() == kDeviceArgOp) {
      node_vec = &this->arg_nodes;
    } else {
      continue;
    }
    int index;
    TF_CHECK_OK(GetNodeAttr(n->attrs(), "index", &index));
    CHECK_LE(0, index);
    CHECK_LT(index, node_vec->size());
    (*node_vec)[index] = n;
  }
  // 2. Find ControlRet nodes that must be always executed.
  std::unordered_set<StringPiece, StringPieceHasher> control_ret_node_names;
  for (const auto& control_ret : fdef.control_ret()) {
    control_ret_node_names.insert(control_ret.second);
  }
  this->control_ret_nodes.reserve(control_ret_node_names.size());
  for (Node* n : this->graph->op_nodes()) {
    if (control_ret_node_names.count(n->name()) > 0) {
      this->control_ret_nodes.push_back(n);
    }
  }
}

FunctionBody::~FunctionBody() { delete this->graph; }

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

Status FunctionDefToBodyHelper(
    const FunctionDef& fdef, const AttrSlice& attrs,
    const FunctionLibraryDefinition* const lib_def,
    const std::function<Status(const string&, const OpDef**)>& get_func_sig,
    std::unique_ptr<FunctionBody>* fbody) {
  // Instantiates the function template into a graph def.
  InstantiationResult result;
  TF_RETURN_IF_ERROR(InstantiateFunction(fdef, attrs, get_func_sig, &result));

  std::unique_ptr<Graph> graph(new Graph(lib_def));
  GraphConstructorOptions opts;
  opts.allow_internal_ops = true;
  opts.expect_device_spec = false;
  TF_RETURN_IF_ERROR(ConvertNodeDefsToGraph(opts, result.nodes, graph.get()));

  // Call BuildControlFlowInfo to validate that this function body has
  // well-formed control flow.
  std::vector<ControlFlowInfo> dummy;
  TF_RETURN_IF_ERROR(BuildControlFlowInfo(graph.get(), &dummy));

  *fbody = absl::make_unique<FunctionBody>(fdef, result.arg_types,
                                           result.ret_types, graph.release());
  return Status::OK();
}

Status FunctionDefToBodyHelper(const FunctionDef& fdef, const AttrSlice& attrs,
                               const FunctionLibraryDefinition* lib_def,
                               std::unique_ptr<FunctionBody>* fbody) {
  const auto get_func_sig = [&lib_def](const string& op, const OpDef** sig) {
    return lib_def->LookUpOpDef(op, sig);
  };
  return FunctionDefToBodyHelper(fdef, attrs, lib_def, get_func_sig, fbody);
}

}  // end namespace tensorflow
