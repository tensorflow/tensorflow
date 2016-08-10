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

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/common_runtime/graph_optimizer.h"
#include "tensorflow/core/common_runtime/memory_types.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/gradients.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/optimizer_cse.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

// A few string constant used throughout this module.
static const char* const kArgOp = "_Arg";
static const char* const kRetOp = "_Retval";
static const char* const kGradientOp = "SymbolicGradient";
static const char* const kNodeLabel = "Func";
static const char* const kFuncAttr = "f";
static const char* const kNoinlineAttr = "noinline";

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
static Node* AddNoOp(Graph* g) {
  NodeDef ndef;
  ndef.set_name(g->NewName(kNodeLabel));
  ndef.set_op("NoOp");
  Status s;
  Node* ret = g->AddNode(ndef, &s);
  TF_CHECK_OK(s);
  return ret;
}

static Node* AddIdentity(Graph* g, Endpoint input) {
  DCHECK_LT(0, input.dtype());
  DCHECK_LT(input.dtype(), DT_FLOAT_REF);
  NodeDef ndef;
  ndef.set_name(g->NewName(kNodeLabel));
  ndef.set_op("Identity");
  ndef.add_input(input.name());
  AddNodeAttr("T", input.dtype(), &ndef);
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

class ArgOp : public OpKernel {
 public:
  explicit ArgOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("index", &index_));
  }

  void Compute(OpKernelContext* ctx) override {
    auto frame = ctx->call_frame();
    OP_REQUIRES(ctx, frame != nullptr, errors::Internal("no call frame"));
    Tensor val;
    OP_REQUIRES_OK(ctx, frame->GetArg(index_, &val));
    OP_REQUIRES(ctx, val.dtype() == dtype_,
                errors::InvalidArgument(
                    "Type mismatch: actual ", DataTypeString(val.dtype()),
                    " vs. expect ", DataTypeString(dtype_)));
    ctx->set_output(0, val);
  }

 private:
  int index_;
  DataType dtype_;

  TF_DISALLOW_COPY_AND_ASSIGN(ArgOp);
};

class RetvalOp : public OpKernel {
 public:
  explicit RetvalOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("index", &index_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& val = ctx->input(0);
    OP_REQUIRES(ctx, val.dtype() == dtype_,
                errors::InvalidArgument(
                    "Type mismatch: actual ", DataTypeString(val.dtype()),
                    " vs. expect ", DataTypeString(dtype_)));
    auto frame = ctx->call_frame();
    OP_REQUIRES(ctx, frame != nullptr, errors::Internal("no call frame"));
    OP_REQUIRES_OK(ctx, frame->SetRetval(index_, val));
  }

 private:
  int index_;
  DataType dtype_;

  TF_DISALLOW_COPY_AND_ASSIGN(RetvalOp);
};

REGISTER_KERNEL_BUILDER(Name("_Arg").Device(DEVICE_CPU), ArgOp);
REGISTER_KERNEL_BUILDER(Name("_Retval").Device(DEVICE_CPU), RetvalOp);

#define REGISTER_GPU_KERNELS(type)                                       \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("_Arg").Device(DEVICE_GPU).TypeConstraint<type>("T"), ArgOp); \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("_Retval").Device(DEVICE_GPU).TypeConstraint<type>("T"), RetvalOp);
REGISTER_GPU_KERNELS(Eigen::half);
REGISTER_GPU_KERNELS(float);
REGISTER_GPU_KERNELS(double);
#undef REGISTER_GPU_KERNELS

REGISTER_KERNEL_BUILDER(Name("_Arg")
                            .Device(DEVICE_GPU)
                            .HostMemory("output")
                            .TypeConstraint<int32>("T"),
                        ArgOp);
REGISTER_KERNEL_BUILDER(Name("_Retval")
                            .Device(DEVICE_GPU)
                            .HostMemory("input")
                            .TypeConstraint<int32>("T"),
                        RetvalOp);

class PassOn : public OpKernel {
 public:
  explicit PassOn(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    OP_REQUIRES(ctx, ctx->num_inputs() == ctx->num_outputs(),
                errors::Internal("#inputs != #outputs : ", ctx->num_inputs(),
                                 " vs. ", ctx->num_outputs()));
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      ctx->set_output(i, ctx->input(i));
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("_ListToArray").Device(DEVICE_CPU), PassOn);
REGISTER_KERNEL_BUILDER(Name("_ArrayToList").Device(DEVICE_CPU), PassOn);

#define REGISTER_GPU_KERNELS(type)                                       \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("_ListToArray").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      PassOn);                                                           \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("_ArrayToList").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      PassOn);

REGISTER_GPU_KERNELS(Eigen::half);
REGISTER_GPU_KERNELS(float);
REGISTER_GPU_KERNELS(double);

#undef REGISTER_GPU_KERNELS

REGISTER_KERNEL_BUILDER(Name("_ListToArray")
                            .Device(DEVICE_GPU)
                            .HostMemory("output")
                            .TypeConstraint<int32>("T"),
                        PassOn);
REGISTER_KERNEL_BUILDER(Name("_ArrayToList")
                            .Device(DEVICE_GPU)
                            .HostMemory("input")
                            .TypeConstraint<int32>("T"),
                        PassOn);

static const FunctionLibraryRuntime::Handle kInvalidHandle = -1;

class FunctionLibraryRuntimeImpl : public FunctionLibraryRuntime {
 public:
  FunctionLibraryRuntimeImpl(const DeviceMgr* dmgr, Device* device,
                             int graph_def_version,
                             const FunctionLibraryDefinition* lib_def,
                             const OptimizerOptions& optimizer_options);

  ~FunctionLibraryRuntimeImpl() override;

  Status Instantiate(const string& function_name,
                     const InstantiateAttrValueMap& attrs,
                     Handle* handle) override;

  const FunctionBody* GetFunctionBody(Handle handle) override;

  Status CreateKernel(const NodeDef& ndef, OpKernel** kernel) override;

  void Run(const Options& opts, Handle handle, gtl::ArraySlice<Tensor> args,
           std::vector<Tensor>* rets, DoneCallback done) override;

  bool IsStateful(const string& function) override;

  const FunctionLibraryDefinition* GetFunctionLibraryDefinition()
      const override {
    return lib_def_;
  }

  Device* device() override { return device_; }

 private:
  typedef FunctionLibraryRuntimeImpl ME;

  const DeviceMgr* const device_mgr_;
  Device* const device_;
  const int graph_def_version_;
  const FunctionLibraryDefinition* const lib_def_;
  GraphOptimizer optimizer_;
  std::function<Status(const string&, const OpDef**)> get_func_sig_;
  std::function<Status(const NodeDef&, OpKernel**)> create_kernel_;

  mutable mutex mu_;

  // Maps function instantiation to a handle. The key is a
  // canonicalized representation of the function name and
  // instantiation attrs. The handle is an index into the items_.
  std::unordered_map<string, Handle> table_ GUARDED_BY(mu_);

  // func_graphs_ never shrinks or reorders its members.
  std::vector<FunctionBody*> func_graphs_ GUARDED_BY(mu_);

  // The instantiated and transformed function is encoded as a Graph
  // object, and an executor is created for the graph.
  struct Item : public core::RefCounted {
    Executor* exec = nullptr;

    ~Item() override { delete this->exec; }
  };
  std::vector<Item*> items_;

  Status FunctionDefToBody(const FunctionDef& fdef,
                           const InstantiateAttrValueMap& attrs,
                           FunctionBody** fbody);
  Status CreateItem(Handle handle, Item** item);
  Status GetOrCreateItem(Handle handle, Item** item);
  Status InstantiateSymbolicGradient(const NameAttrList& func,
                                     FunctionBody** g_body);

  TF_DISALLOW_COPY_AND_ASSIGN(FunctionLibraryRuntimeImpl);
};

FunctionLibraryRuntimeImpl::FunctionLibraryRuntimeImpl(
    const DeviceMgr* dmgr, Device* device, int graph_def_version,
    const FunctionLibraryDefinition* lib_def,
    const OptimizerOptions& optimizer_options)
    : device_mgr_(dmgr),
      device_(device),
      graph_def_version_(graph_def_version),
      lib_def_(lib_def),
      optimizer_(optimizer_options) {
  get_func_sig_ = [this](const string& op, const OpDef** sig) {
    return lib_def_->LookUpOpDef(op, sig);
  };
  create_kernel_ = [this](const NodeDef& ndef, OpKernel** kernel) {
    return CreateKernel(ndef, kernel);
  };
}

FunctionLibraryRuntimeImpl::~FunctionLibraryRuntimeImpl() {
  for (FunctionBody* p : func_graphs_) delete p;
  for (Item* item : items_)
    if (item) item->Unref();
}

// An asynchronous op kernel which executes an instantiated function
// defined in a library.
class CallOp : public AsyncOpKernel {
 public:
  CallOp(FunctionLibraryRuntime::Handle handle, OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx), handle_(handle) {}

  ~CallOp() override {}

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    FunctionLibraryRuntime* lib = ctx->function_library();
    OP_REQUIRES_ASYNC(ctx, lib != nullptr,
                      errors::Internal("No function library is provided."),
                      done);
    FunctionLibraryRuntime::Options opts;
    opts.step_id = ctx->step_id();
    opts.runner = ctx->runner();
    std::vector<Tensor> args;
    args.reserve(ctx->num_inputs());
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      args.push_back(ctx->input(i));
    }
    std::vector<Tensor>* rets = new std::vector<Tensor>;
    lib->Run(opts, handle_, args, rets,
             [ctx, done, rets](const Status& status) {
               if (!status.ok()) {
                 ctx->SetStatus(status);
               } else {
                 CHECK_EQ(rets->size(), ctx->num_outputs());
                 for (size_t i = 0; i < rets->size(); ++i) {
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

class SymbolicGradientOp : public AsyncOpKernel {
 public:
  SymbolicGradientOp(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx), handle_(kInvalidHandle) {}

  ~SymbolicGradientOp() override {}

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    FunctionLibraryRuntime* lib = ctx->function_library();
    OP_REQUIRES_ASYNC(ctx, lib != nullptr,
                      errors::Internal("No function library is provided."),
                      done);

    OP_REQUIRES_OK_ASYNC(
        ctx, lib->Instantiate(kGradientOp, def().attr(), &handle_), done);

    FunctionLibraryRuntime::Options opts;
    opts.step_id = ctx->step_id();
    opts.runner = ctx->runner();
    std::vector<Tensor> args;
    args.reserve(ctx->num_inputs());
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      args.push_back(ctx->input(i));
    }
    std::vector<Tensor>* rets = new std::vector<Tensor>;
    lib->Run(
        opts, handle_, args, rets, [ctx, done, rets](const Status& status) {
          if (!status.ok()) {
            ctx->SetStatus(status);
          } else if (rets->size() != ctx->num_outputs()) {
            ctx->SetStatus(errors::InvalidArgument(
                "SymGrad expects to return ", ctx->num_outputs(),
                " tensor(s), but get ", rets->size(), " tensor(s) instead."));
          } else {
            for (size_t i = 0; i < rets->size(); ++i) {
              ctx->set_output(i, (*rets)[i]);
            }
          }
          delete rets;
          done();
        });
  }

 private:
  FunctionLibraryRuntime::Handle handle_;

  TF_DISALLOW_COPY_AND_ASSIGN(SymbolicGradientOp);
};

REGISTER_KERNEL_BUILDER(Name(kGradientOp).Device(DEVICE_CPU),
                        SymbolicGradientOp);
REGISTER_KERNEL_BUILDER(Name(kGradientOp).Device(DEVICE_GPU),
                        SymbolicGradientOp);

const FunctionBody* FunctionLibraryRuntimeImpl::GetFunctionBody(Handle h) {
  mutex_lock l(mu_);
  CHECK_LE(static_cast<Handle>(0), h);
  CHECK_LT(h, func_graphs_.size());
  return func_graphs_[h];
}

Status FunctionLibraryRuntimeImpl::CreateKernel(const NodeDef& ndef,
                                                OpKernel** kernel) {
  if (lib_def_->Find(ndef.op()) == nullptr) {
    return CreateNonCachedKernel(device_, this, ndef, graph_def_version_,
                                 kernel);
  }

  // Try to instantiate this function for the func/attr. Maybe its
  // cached already.
  Handle handle;
  TF_RETURN_IF_ERROR(Instantiate(ndef.op(), ndef.attr(), &handle));

  const FunctionBody* fbody = GetFunctionBody(handle);
  CHECK_NOTNULL(fbody);

  // TODO(zhifengc): For now, we assume int32 is always on host memory
  // and other types are always on device memory. We should do type
  // inference over function body to derive the correct input/output
  // memory types.
  MemoryTypeVector input_memory_types;
  for (const auto& t : fbody->arg_types) {
    input_memory_types.push_back(t == DT_INT32 ? HOST_MEMORY : DEVICE_MEMORY);
  }
  MemoryTypeVector output_memory_types;
  for (const auto& t : fbody->ret_types) {
    output_memory_types.push_back(t == DT_INT32 ? HOST_MEMORY : DEVICE_MEMORY);
  }

  // Constructs a CallOp kernel for running the instantiated function.
  Status s;
  auto device_type = DeviceType(device_->attributes().device_type());
  OpKernelConstruction construction(
      device_type, device_, device_->GetAllocator(AllocatorAttributes()), &ndef,
      &fbody->fdef.signature(), this, fbody->arg_types, input_memory_types,
      fbody->ret_types, output_memory_types, graph_def_version_, &s);
  *kernel = new CallOp(handle, &construction);
  if (!s.ok()) {
    delete kernel;
  }
  return s;
}

Status FunctionLibraryRuntimeImpl::FunctionDefToBody(
    const FunctionDef& fdef, const InstantiateAttrValueMap& attrs,
    FunctionBody** fbody) {
  // Instantiates the function template into a graph def.
  InstantiationResult result;
  TF_RETURN_IF_ERROR(InstantiateFunction(fdef, attrs, get_func_sig_, &result));

  Graph* graph = new Graph(lib_def_);
  GraphConstructorOptions opts;
  opts.allow_internal_ops = true;
  opts.expect_device_spec = false;
  Status s = ConvertGraphDefToGraph(opts, result.gdef, graph);
  if (!s.ok()) {
    delete graph;
  } else {
    *fbody = new FunctionBody(fdef, result.arg_types, result.ret_types, graph);
  }
  return s;
}

Status FunctionLibraryRuntimeImpl::InstantiateSymbolicGradient(
    const NameAttrList& func, FunctionBody** g_body) {
  const FunctionDef* fdef = lib_def_->Find(func.name());
  if (fdef == nullptr) {
    // f is a primitive op.
    gradient::Creator creator;
    TF_RETURN_IF_ERROR(gradient::GetOpGradientCreator(func.name(), &creator));
    if (creator == nullptr) {
      return errors::InvalidArgument("No gradient is defined for ",
                                     func.name());
    }
    FunctionDef grad_fdef;
    TF_RETURN_IF_ERROR(creator(AttrSlice(&func.attr()), &grad_fdef));
    TF_RETURN_IF_ERROR(FunctionDefToBody(grad_fdef, func.attr(), g_body));
  } else {
    // f is a user-defined function.
    Handle f_handle;
    TF_RETURN_IF_ERROR(Instantiate(func.name(), func.attr(), &f_handle));
    const FunctionBody* f_body = GetFunctionBody(f_handle);
    CHECK_NOTNULL(f_body);
    *g_body = SymbolicGradient(*f_body);
  }
  return Status::OK();
}

Status FunctionLibraryRuntimeImpl::Instantiate(
    const string& function_name, const InstantiateAttrValueMap& attrs,
    Handle* handle) {
  const string key = Canonicalize(function_name, attrs);
  {
    mutex_lock l(mu_);
    *handle = gtl::FindWithDefault(table_, key, kInvalidHandle);
    if (*handle != kInvalidHandle) {
      return Status::OK();
    }
  }

  Status s;
  FunctionBody* fbody = nullptr;
  if (function_name == kGradientOp) {
    const AttrValue* f = gtl::FindOrNull(attrs, kFuncAttr);
    if (f == nullptr) {
      return errors::InvalidArgument("SymbolicGradient is missing attr: f");
    }
    const auto& func = f->func();
    if (func.name() == kGradientOp) {
      return errors::InvalidArgument("Can't take gradient of SymbolicGradient");
    }
    const string grad = lib_def_->FindGradient(func.name());
    if (!grad.empty()) {
      return Instantiate(grad, func.attr(), handle);
    }
    TF_RETURN_IF_ERROR(InstantiateSymbolicGradient(func, &fbody));
  } else {
    const FunctionDef* fdef = lib_def_->Find(function_name);
    if (fdef == nullptr) {
      return errors::NotFound("Function ", function_name, " is not defined.");
    }
    TF_RETURN_IF_ERROR(FunctionDefToBody(*fdef, attrs, &fbody));
  }

  {
    mutex_lock l(mu_);
    *handle = gtl::FindWithDefault(table_, key, kInvalidHandle);
    if (*handle != kInvalidHandle) {
      delete fbody;
    } else {
      *handle = func_graphs_.size();
      table_.insert({key, *handle});
      func_graphs_.push_back(fbody);
      items_.resize(func_graphs_.size());
    }
  }
  return Status::OK();
}

void DumpGraph(StringPiece label, const Graph* g) {
  // TODO(zhifengc): Change Graph to record #nodes.
  VLOG(1) << "Graph " << label << " #nodes " << g->num_nodes() << " #edges "
          << g->edges().size();
  if (VLOG_IS_ON(2)) {
    for (const auto& line : str_util::Split(DebugString(g), '\n')) {
      VLOG(2) << "|| " << line;
    }
  }
}

void OptimizeGraph(FunctionLibraryRuntime* lib, Graph** g) {
  OptimizerOptions opts;
  opts.set_do_common_subexpression_elimination(true);
  opts.set_do_function_inlining(true);
  opts.set_do_constant_folding(true);
  GraphOptimizer optimizer(opts);
  optimizer.Optimize(lib, lib->device(), g);
}

Status FunctionLibraryRuntimeImpl::CreateItem(Handle handle, Item** item) {
  const FunctionBody* fbody = GetFunctionBody(handle);
  CHECK_NOTNULL(fbody);
  Graph* g = new Graph(lib_def_);
  CopyGraph(*fbody->graph, g);

  optimizer_.Optimize(this, device(), &g);
  auto s = EnsureMemoryTypes(DeviceType(device()->device_type()),
                             device()->name(), g);
  if (!s.ok()) {
    delete g;
    return Status::OK();
  }

  // Creates an executor based on the g.  This must be done without
  // holding mu_ because create_kernel_ calls back into the library.
  LocalExecutorParams params;
  params.device = device_;
  params.function_library = this;
  params.create_kernel = create_kernel_;
  params.delete_kernel = [](OpKernel* kernel) {
    DeleteNonCachedKernel(kernel);
  };
  Executor* exec;
  TF_RETURN_IF_ERROR(NewLocalExecutor(params, g, &exec));

  *item = new Item;
  (*item)->exec = exec;
  return Status::OK();
}

Status FunctionLibraryRuntimeImpl::GetOrCreateItem(Handle handle, Item** item) {
  {
    mutex_lock l(mu_);
    if (handle >= items_.size()) {
      return errors::NotFound("Function handle ", handle,
                              " is not valid. Likely an internal error.");
    }
    *item = items_[handle];
    if (*item != nullptr) {
      (*item)->Ref();
      return Status::OK();
    }
  }
  // NOTE: We need to call CreateItem out of mu_ because creating an
  // executor needs to call CreateKernel.
  TF_RETURN_IF_ERROR(CreateItem(handle, item));

  {
    mutex_lock l(mu_);
    if (items_[handle] == nullptr) {
      // Install *item in items_.
      items_[handle] = *item;
      (*item)->Ref();
    }
  }
  return Status::OK();
}

void FunctionLibraryRuntimeImpl::Run(const Options& opts, Handle handle,
                                     gtl::ArraySlice<Tensor> args,
                                     std::vector<Tensor>* rets,
                                     DoneCallback done) {
  if (opts.cancellation_manager && opts.cancellation_manager->IsCancelled()) {
    return done(errors::Cancelled(""));
  }
  const FunctionBody* fbody = GetFunctionBody(handle);
  FunctionCallFrame* frame =
      new FunctionCallFrame(fbody->arg_types, fbody->ret_types);
  Status s = frame->SetArgs(args);
  if (!s.ok()) {
    delete frame;
    return done(s);
  }
  Item* item = nullptr;
  s = GetOrCreateItem(handle, &item);
  if (!s.ok()) {
    delete frame;
    return done(s);
  }
  DCHECK(opts.runner != nullptr);

  Executor::Args exec_args;
  // Inherit the step_id from the caller.
  exec_args.step_id = opts.step_id;
  exec_args.call_frame = frame;
  exec_args.cancellation_manager = opts.cancellation_manager;
  exec_args.runner = *opts.runner;
  // TODO(zhifengc): we can avoid creating rendez here if we know
  // there is no send/recv nodes in the graph.
  auto* rendez = new IntraProcessRendezvous(device_mgr_);
  exec_args.rendezvous = rendez;
  item->exec->RunAsync(
      // Executor args
      exec_args,
      // Done callback.
      [item, frame, rets, rendez, done](const Status& status) {
        item->Unref();
        rendez->Unref();
        Status s = status;
        if (s.ok()) {
          s = frame->GetRetvals(rets);
        }
        delete frame;
        done(s);
      });
}

bool FunctionLibraryRuntimeImpl::IsStateful(const string& func) {
  const OpDef* op_def;
  const Status s = lib_def_->LookUpOpDef(func, &op_def);
  return s.ok() && op_def->is_stateful();
}

FunctionLibraryRuntime* NewFunctionLibraryRuntime(
    const DeviceMgr* dmgr, Device* device, int graph_def_version,
    const FunctionLibraryDefinition* lib_def,
    const OptimizerOptions& optimizer_options) {
  return new FunctionLibraryRuntimeImpl(dmgr, device, graph_def_version,
                                        lib_def, optimizer_options);
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
    if (n->IsIdentity() && GetTheOnlyDataEdge(n->in_edges())) {
      matches.push_back(n);
    }
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

      // Process input edges first.
      Node* input_control_node = nullptr;
      for (const Edge* e : n->in_edges()) {
        if (e->IsControlEdge()) {
          if (input_control_node == nullptr) {
            // If node "n" has any control dependencies, adds a no-op
            // node (input_control_node) which the additional Identity
            // nodes depends on and the input_control_node depends on
            // the node "n"s control dependencies.
            input_control_node = AddNoOp(g);
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
          *id_node = AddIdentity(g, {e->src(), e->src_output()});
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
            output_control_node = AddNoOp(g);
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

// Returns true iff the function '*fbody' can be inlined at 'node'
// based on the type signature of 'node' and 'fbody'.
static bool ValidateInlining(const Node* node, const FunctionBody* fbody) {
  if (static_cast<size_t>(node->num_inputs()) != fbody->arg_types.size()) {
    return false;
  }
  if (static_cast<size_t>(node->num_inputs()) != fbody->arg_nodes.size()) {
    return false;
  }
  if (static_cast<size_t>(node->num_outputs()) != fbody->ret_types.size()) {
    return false;
  }
  if (static_cast<size_t>(node->num_outputs()) != fbody->ret_nodes.size()) {
    return false;
  }
  for (int i = 0; i < node->num_inputs(); ++i) {
    if (node->input_type(i) != fbody->arg_types[i]) return false;
  }
  for (int i = 0; i < node->num_outputs(); ++i) {
    if (node->output_type(i) != fbody->ret_types[i]) return false;
  }
  return true;
}

// Given a "caller" in "graph", which is a function call of a function
// to "fbody". Replaces the "caller" with fbody->graph and connects
// edges properly.
static void InlineFunctionBody(Graph* g, Node* caller,
                               const FunctionBody* fbody) {
  if (!ValidateInlining(caller, fbody)) {
    LOG(WARNING) << "Inlining mismatch: " << caller->DebugString() << " vs. "
                 << DebugString(fbody->graph);
    return;
  }

  // Duplicate fbody->graph into 'g'.  First, we copy the nodes of
  // fbody->graph into 'g' except the source and sink nodes.  We copy
  // edges among nodes in 'fbody->graph'.
  //
  // If 'x' is a node in fbody->graph and its copy in 'g' is 'y', we
  // remember 'y' in node_map[x->id()].
  std::vector<Node*> node_map(fbody->graph->num_node_ids());
  Status s;
  for (Node* n : fbody->graph->nodes()) {
    if (n->IsSource() || n->IsSink()) continue;
    CHECK(n->IsOp());
    NodeDef ndef = n->def();
    ndef.set_name(strings::StrCat(caller->name(), "/", ndef.name()));
    node_map[n->id()] = g->AddNode(ndef, &s);
    TF_CHECK_OK(s);
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

  // Connect input edges.
  //
  // For data edges coming into "caller", we first compute the
  // <src>:<src_output> for the i-th input in "inputs". We create one
  // Identity node for each input. Then, we connect inputs[i] to to
  // the i-th identity node added. The nodes that previously connects
  // to the j-th output of i-th arg node are reconnected to th i-th
  // identity node.
  //
  // If "caller" has any input control dependencies, we add a NoOp
  // node "input_control_node". This "input_control_node" depends on
  // what "caller" depends on, and the added identity nodes depend on
  // "input_control_node".
  std::vector<Endpoint> inputs(caller->num_inputs());
  Node* input_control_node = nullptr;
  for (const Edge* e : caller->in_edges()) {
    if (e->IsControlEdge()) {
      if (input_control_node == nullptr) {
        input_control_node = AddNoOp(g);
      }
      g->AddControlEdge(e->src(), input_control_node);
    } else {
      inputs[e->dst_input()] = {e->src(), e->src_output()};
    }
  }
  for (std::size_t i = 0; i < fbody->arg_nodes.size(); ++i) {
    Node* arg = node_map[fbody->arg_nodes[i]->id()];
    Node* n = AddIdentity(g, inputs[i]);
    if (input_control_node) {
      g->AddControlEdge(input_control_node, n);
    }
    for (const Edge* e : arg->out_edges()) {
      if (e->IsControlEdge()) {
        g->AddControlEdge(n, e->dst());
      } else {
        g->AddEdge(n, 0, e->dst(), e->dst_input());
      }
    }
    node_map[fbody->arg_nodes[i]->id()] = n;
    g->RemoveNode(arg);  // 'arg' is disconnected.
  }

  // Connect output edges.
  //
  // For i-th return node in fbody->graph, we add in "g" an identity
  // node (outputs[i-th]). We then reconnect every incoming edge into
  // the i-th return node to the added identity node.
  //
  // For every data edge coming out of "callee"s i-th output, we
  // reconnect it to the i-th identity added above.
  //
  // If "callee" is control-depended upon by any other nodes, we add a
  // NoOp node "output_control_node". "output_control_node" depends on
  // all identity nodes added above. And nodes previously depend on
  // "callee" is changed to depend on "output_control_node".
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
    Node* n = AddIdentity(g, data);
    outputs[i] = n;
    for (const Edge* e : ret->in_edges()) {
      if (e->IsControlEdge()) {
        g->AddControlEdge(e->src(), n);
      }
    }
    g->RemoveNode(ret);  // 'ret' is disconnected.
  }
  Node* output_control_node = nullptr;
  for (const Edge* e : caller->out_edges()) {
    if (e->IsControlEdge()) {
      if (output_control_node == nullptr) {
        output_control_node = AddNoOp(g);
        for (Node* n : outputs) {
          g->AddControlEdge(n, output_control_node);
        }
      }
      g->AddControlEdge(output_control_node, e->dst());
    } else {
      g->AddEdge(outputs[e->src_output()], 0, e->dst(), e->dst_input());
    }
  }
  g->RemoveNode(caller);  // 'caller' is replaced with inlined nodes.
}

// Given a node's NodeDef, returns false iff the node explicitly
// specified noinline. This gives ExpandInlineFunctions a heuristic to
// decide whether to inline the function.
bool ShouldInline(const NodeDef& ndef) {
  bool noinline = false;
  if (GetNodeAttr(ndef, kNoinlineAttr, &noinline).ok()) {
    // If the node specifies attribute 'noinlne', returns accordingly.
    return !noinline;
  }
  if (ndef.op() != kGradientOp) {
    // If the op is not SymbolicGradient, we should be free to decide
    // whether to inline or not.
    return true;
  }
  // If the node is a SymbolicGradient, we use the forward
  // function's attribute 'noinline' instead.
  const NameAttrList* forward_func_attrs;
  Status s =
      GetNodeAttr(AttrSlice(&ndef.attr()), kFuncAttr, &forward_func_attrs);
  if (!s.ok()) {
    // The node def is malformed (missing attribute 'f'), we'll just
    // continue and the runtime will error out.
    return false;
  }
  s = GetNodeAttr(AttrSlice(&forward_func_attrs->attr()), kNoinlineAttr,
                  &noinline);
  if (!s.ok()) {
    // The forward function doesn't specify 'noinline' attr, we should
    // be free to decide.
    return true;
  }
  // Otherwise, make inline decision according to the attr.
  return !noinline;
}

bool ExpandInlineFunctions(FunctionLibraryRuntime* lib, Graph* graph) {
  std::vector<std::pair<Node*, const FunctionBody*>> candidates;
  for (Node* node : graph->nodes()) {
    VLOG(3) << "Expanding " << node->DebugString();
    if (!ShouldInline(node->def())) {
      VLOG(3) << "noinline: " << node->DebugString();
      continue;
    }
    FunctionLibraryRuntime::Handle handle;
    Status s =
        lib->Instantiate(node->type_string(), node->def().attr(), &handle);
    if (!s.ok()) {
      // Either "node" is a primitive op, or the instantiation failed.
      if (errors::IsNotFound(s)) {
        VLOG(3) << "ExpandInlineFunctions " << s;
      } else {
        LOG(ERROR) << "ExpandInlineFunctions " << s;
      }
      continue;
    }
    const FunctionBody* fbody = lib->GetFunctionBody(handle);
    CHECK_NOTNULL(fbody);
    candidates.push_back({node, fbody});
  }
  for (const auto& p : candidates) {
    InlineFunctionBody(graph, p.first, p.second);
  }
  return !candidates.empty();
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
  std::vector<int> pending(g->num_node_ids());
  std::deque<const Node*> ready;
  for (const Node* n : g->nodes()) {
    pending[n->id()] = n->in_edges().size();
    if (pending[n->id()] == 0) ready.push_back(n);
  }
  gtl::InlinedVector<const Edge*, 4> inputs;
  gdef->Clear();
  gdef->mutable_versions()->CopyFrom(g->versions());
  while (!ready.empty()) {
    const Node* n = ready.front();
    ready.pop_front();
    for (const Edge* e : n->out_edges()) {
      const Node* next = e->dst();
      if (--pending[next->id()] == 0) {
        ready.push_back(next);
      }
    }
    if (!n->IsOp()) continue;
    NodeDef* ndef = gdef->add_node();
    ndef->set_name(NewName(n, pretty));
    ndef->set_op(n->type_string());
    *(ndef->mutable_attr()) = n->def().attr();
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
      const string srcname = NewName(e->src(), pretty);
      if (e == nullptr) {
        ndef->add_input("unknown");
      } else if (!e->src()->IsOp()) {
      } else if (e->IsControlEdge()) {
        ndef->add_input(strings::StrCat("^", srcname));
      } else if (e->src_output() == 0) {
        ndef->add_input(srcname);
      } else {
        ndef->add_input(strings::StrCat(srcname, ":", e->src_output()));
      }
    }
  }
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
  this->arg_nodes.resize(arg_types.size());
  this->ret_nodes.resize(ret_types.size());
  for (Node* n : this->graph->nodes()) {
    gtl::InlinedVector<Node*, 4>* node_vec;
    if (n->type_string() == kRetOp) {
      node_vec = &this->ret_nodes;
    } else if (n->type_string() == kArgOp) {
      node_vec = &this->arg_nodes;
    } else {
      continue;
    }
    int index;
    TF_CHECK_OK(GetNodeAttr(n->def(), "index", &index));
    CHECK_LE(0, index);
    CHECK_LT(index, node_vec->size());
    (*node_vec)[index] = n;
  }
}

FunctionBody::~FunctionBody() { delete this->graph; }

class SymbolicGradientHelper {
 public:
  explicit SymbolicGradientHelper(const FunctionBody& f) : fbody_(&f) {}

  ~SymbolicGradientHelper() { delete gbody_; }

  FunctionBody* Compute();

 private:
  const FunctionBody* fbody_;
  FunctionBody* gbody_ = nullptr;

  // Makes a copy of fbody_ in gbody_.
  void Copy();

  TF_DISALLOW_COPY_AND_ASSIGN(SymbolicGradientHelper);
};

void SymbolicGradientHelper::Copy() {
  const Graph& src = *(fbody_->graph);
  gbody_->graph = new Graph(src.op_registry());
  Graph* dst = gbody_->graph;

  std::vector<Node*> node_map(src.num_node_ids());

  // Copy the nodes.
  node_map[src.source_node()->id()] = dst->source_node();
  node_map[src.sink_node()->id()] = dst->sink_node();
  for (Node* n : src.nodes()) {
    if (n->IsSource() || n->IsSink()) continue;
    CHECK(n->IsOp());
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
  gbody_->arg_types = fbody_->arg_types;
  for (std::size_t i = 0; i < fbody_->arg_nodes.size(); ++i) {
    gbody_->arg_nodes.push_back(node_map[fbody_->arg_nodes[i]->id()]);
  }

  // Save outputs in copied graph.
  CHECK_EQ(fbody_->ret_types.size(), fbody_->ret_nodes.size());
  gbody_->ret_types = fbody_->ret_types;
  for (std::size_t i = 0; i < fbody_->ret_nodes.size(); ++i) {
    gbody_->ret_nodes.push_back(node_map[fbody_->ret_nodes[i]->id()]);
  }
}

FunctionBody* SymbolicGradientHelper::Compute() {
  CHECK(gbody_ == nullptr);
  gbody_ = new FunctionBody;

  // Copy fbody_ into gbody_.
  Copy();

  Graph* g = gbody_->graph;

  const int num_y = gbody_->ret_nodes.size();

  // Populate 'y_node_outputs_' with node function body outputs.
  // Populate 'y_grad_nodes' with initial gradient nodes for each return node of
  // the original function body (these will be 'arg' nodes in the function
  // gradient body).
  std::vector<NodeOut> y_node_outputs;
  y_node_outputs.reserve(num_y);
  std::vector<NodeOut> y_grad_node_outputs;
  y_grad_node_outputs.reserve(num_y);
  for (int i = 0; i < num_y; ++i) {
    Node* y = gbody_->ret_nodes[i];
    y_node_outputs.push_back({y, 0});
    DCHECK_EQ(y->type_string(), kRetOp);
    const DataType dtype = y->input_type(0);
    const int index = gbody_->arg_nodes.size();
    Node* dy = AddArg(g, dtype, index);
    gbody_->arg_types.push_back(dtype);
    gbody_->arg_nodes.push_back(dy);
    y_grad_node_outputs.push_back({dy, 0});
  }

  // Populate 'x_nodes' with function args (excluding 'y_grad_node_outputs').
  const int num_x = fbody_->arg_nodes.size();
  std::vector<NodeOut> x_node_outputs;
  x_node_outputs.reserve(num_x);
  for (size_t i = 0; i < fbody_->arg_nodes.size(); ++i) {
    x_node_outputs.push_back({gbody_->arg_nodes[i], 0});
  }

  // Call AddSymbolicGradients which will add nodes to graph 'g' that
  // compute the function gradient (adding an entry in 'x_grad_node_outputs' for
  // each node in 'x_node_outputs').
  std::vector<NodeOut> x_grad_node_outputs;
  TF_CHECK_OK(AddSymbolicGradients(y_node_outputs, x_node_outputs,
                                   y_grad_node_outputs, &x_grad_node_outputs,
                                   g));

  // Remove the old return nodes from the function body.
  for (Node* n : gbody_->ret_nodes) {
    g->RemoveNode(n);
  }
  gbody_->ret_types = fbody_->arg_types;
  gbody_->ret_nodes.clear();
  // Add new return nodes to the function gradient body for each node
  // in 'x_grad_nodes'.
  for (size_t i = 0; i < fbody_->arg_types.size(); ++i) {
    Endpoint grad = {x_grad_node_outputs[i].node, x_grad_node_outputs[i].index};
    Node* ret = AddRet(g, grad, i);
    gbody_->ret_nodes.push_back(ret);
  }

  auto ret = gbody_;
  gbody_ = nullptr;
  return ret;
}

FunctionBody* SymbolicGradient(const FunctionBody& f) {
  return SymbolicGradientHelper(f).Compute();
}

}  // end namespace tensorflow
