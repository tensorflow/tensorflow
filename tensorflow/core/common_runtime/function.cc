#include "tensorflow/core/common_runtime/function.h"

#include <deque>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/optimizer_cse.h"
#include "tensorflow/core/lib/gtl/map_util.h"

namespace tensorflow {

// A few string constant used throughout this module.
static const char* const kArgOp = "_Arg";
static const char* const kRetOp = "_Retval";
static const char* const kGradientOp = "SymbolicGradient";
static const char* const kNodeLabel = "Func";

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

static Node* AddZerosLike(Graph* g, Endpoint input) {
  DCHECK_LT(0, input.dtype());
  DCHECK_LT(input.dtype(), DT_FLOAT_REF);
  NodeDef ndef;
  ndef.set_name(g->NewName(kNodeLabel));
  ndef.set_op("ZerosLike");
  ndef.add_input(input.name());
  AddNodeAttr("T", input.dtype(), &ndef);
  Status s;
  Node* ret = g->AddNode(ndef, &s);
  TF_CHECK_OK(s);
  g->AddEdge(input.node, input.index, ret, 0);
  return ret;
}

static Node* AddSymGrad(Graph* g, Node* n, gtl::ArraySlice<Endpoint> grads) {
  const int num_x = n->num_inputs();
  const int num_y = n->num_outputs();
  CHECK_EQ(num_y, grads.size());

  NodeDef ndef;
  ndef.set_name(g->NewName(kNodeLabel));
  ndef.set_op(kGradientOp);

  // The gradient node should have num_x + num_y inputs.
  std::vector<Endpoint> n_inputs(num_x);
  for (const Edge* e : n->in_edges()) {
    if (e->IsControlEdge()) continue;
    n_inputs[e->dst_input()] = {e->src(), e->src_output()};
  }
  DataTypeVector in_types;
  for (const Endpoint& ep : n_inputs) {
    ndef.add_input(ep.name());
    in_types.push_back(ep.dtype());
  }
  for (const Endpoint& ep : grads) {
    ndef.add_input(ep.name());
    in_types.push_back(ep.dtype());
  }
  CHECK_EQ(ndef.input_size(), num_x + num_y);

  AddNodeAttr("Tin", in_types, &ndef);

  // The gradient node's outputs have the same types as the node 'n's
  // inputs.
  AddNodeAttr("Tout", n->input_types(), &ndef);
  NameAttrList func;
  func.set_name(n->type_string());
  *(func.mutable_attr()) = n->def().attr();
  AddNodeAttr("f", func, &ndef);
  Status s;
  Node* ret = g->AddNode(ndef, &s);
  TF_CHECK_OK(s);
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

REGISTER_KERNEL_BUILDER(Name("_Arg").Device(DEVICE_CPU), ArgOp);
REGISTER_KERNEL_BUILDER(Name("_Arg").Device(DEVICE_GPU), ArgOp);

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

REGISTER_KERNEL_BUILDER(Name("_Retval").Device(DEVICE_CPU), RetvalOp);
REGISTER_KERNEL_BUILDER(Name("_Retval").Device(DEVICE_GPU), RetvalOp);

static const FunctionLibraryRuntime::Handle kInvalidHandle = -1;

class FunctionLibraryRuntimeImpl : public FunctionLibraryRuntime {
 public:
  FunctionLibraryRuntimeImpl(Device* device, Runner runner,
                             const FunctionLibraryDefinition* lib_def);

  ~FunctionLibraryRuntimeImpl() override;

  Status Instantiate(const string& function_name,
                     const InstantiateAttrValueMap& attrs,
                     Handle* handle) override;

  const FunctionBody* GetFunctionBody(Handle handle) override;

  Status CreateKernel(const NodeDef& ndef, OpKernel** kernel) override;

  void Run(const Options& opts, Handle handle, gtl::ArraySlice<Tensor> args,
           std::vector<Tensor>* rets, DoneCallback done) override;

  bool IsDefined(const string& function_name) override;

 private:
  typedef FunctionLibraryRuntimeImpl ME;

  Device* const device_;
  Runner runner_ = nullptr;
  const FunctionLibraryDefinition* const lib_def_;
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
  Status InstantiateSymbolicGradient(const InstantiateAttrValueMap& attrs,
                                     FunctionBody** g_body);

  TF_DISALLOW_COPY_AND_ASSIGN(FunctionLibraryRuntimeImpl);
};

FunctionLibraryRuntimeImpl::FunctionLibraryRuntimeImpl(
    Device* device, Runner runner, const FunctionLibraryDefinition* lib_def)
    : device_(device), runner_(runner), lib_def_(lib_def) {
  get_func_sig_ = [this](const string& op, const OpDef** sig) {
    Status s;
    *sig = lib_def_->LookUp(op, &s);
    return s;
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

const FunctionBody* FunctionLibraryRuntimeImpl::GetFunctionBody(Handle h) {
  mutex_lock l(mu_);
  CHECK_LE(0, h);
  CHECK_LT(h, func_graphs_.size());
  return func_graphs_[h];
}

Status FunctionLibraryRuntimeImpl::CreateKernel(const NodeDef& ndef,
                                                OpKernel** kernel) {
  if (ndef.op() != kGradientOp && (lib_def_->Find(ndef.op()) == nullptr)) {
    return CreateNonCachedKernel(device_, this, ndef, kernel);
  }

  // Try to instantiate this function for the func/attr. Maybe its
  // cached already.
  Handle handle;
  TF_RETURN_IF_ERROR(Instantiate(ndef.op(), ndef.attr(), &handle));

  const FunctionBody* fbody = GetFunctionBody(handle);
  CHECK_NOTNULL(fbody);

  // Constructs a CallOp kernel for running the instantiated function.
  Status s;
  auto device_type = DeviceType(device_->attributes().device_type());
  OpKernelConstruction construction(
      device_type, device_, device_->GetAllocator(AllocatorAttributes()), &ndef,
      &fbody->fdef.signature(), this, fbody->arg_types, fbody->ret_types, &s);
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
    const InstantiateAttrValueMap& attrs, FunctionBody** g_body) {
  const AttrValue* f = gtl::FindOrNull(attrs, "f");
  if (f == nullptr) {
    return errors::InvalidArgument("SymbolicGradient is missing attr: f");
  }
  const auto& func = f->func();
  const FunctionDef* fdef = lib_def_->Find(func.name());
  if (fdef == nullptr) {
    // f is a primitve op.
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
    TF_RETURN_IF_ERROR(InstantiateSymbolicGradient(attrs, &fbody));
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

static void DumpGraph(const char* label, const Graph* g) {
  if (VLOG_IS_ON(1)) {
    LOG(INFO) << label << ": " << std::endl << DebugString(g);
  }
}

static void SimplifyGraph(Graph* g) {
  if (RemoveListArrayConverter(g)) {
    DumpGraph("RemoveListArrayConverter", g);
  }
  bool changed;
  do {
    changed = false;
    if (RemoveDeadNodes(g)) {
      changed = true;
      DumpGraph("RemoveDeadNodes", g);
    }
    if (RemoveIdentityNodes(g)) {
      changed = true;
      DumpGraph("RemoveIdentityNodes", g);
    }
    FixupSourceAndSinkEdges(g);
    OptimizeCSE(g, nullptr);
    DumpGraph("OptimizeCSE", g);
  } while (changed);
}

void OptimizeGraph(FunctionLibraryRuntime* lib, Graph** g) {
  DumpGraph("Initial", *g);
  const int kNumInlineRounds = 10;
  for (int i = 0; i < kNumInlineRounds; ++i) {
    if (!ExpandInlineFunctions(lib, *g)) break;
    DumpGraph("ExpandInlineFunctions", *g);
    SimplifyGraph(*g);
  }

  // Makes a copy so that we densify node ids.
  Graph* copy = new Graph((*g)->op_registry());
  CopyGraph(**g, copy);
  delete *g;
  *g = copy;
  DumpGraph("ReCopy", *g);
}

Status FunctionLibraryRuntimeImpl::CreateItem(Handle handle, Item** item) {
  const FunctionBody* fbody = GetFunctionBody(handle);
  CHECK_NOTNULL(fbody);
  Graph* g = new Graph(lib_def_);
  CopyGraph(*fbody->graph, g);
  OptimizeGraph(this, &g);

  // Creates an executor based on the g.  This must be done without
  // holding mu_ because create_kernel_ calls back into the library.
  LocalExecutorParams params;
  params.device = device_;
  params.function_library = this;
  params.has_control_flow = false;
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
  Executor::Args exec_args;
  exec_args.call_frame = frame;
  exec_args.cancellation_manager = opts.cancellation_manager;
  exec_args.runner = runner_;
  item->exec->RunAsync(
      // Executor args
      exec_args,
      // Done callback.
      [item, frame, rets, done](const Status& status) {
        item->Unref();
        Status s = status;
        if (s.ok()) {
          s = frame->GetRetvals(rets);
        }
        delete frame;
        done(s);
      });
}

bool FunctionLibraryRuntimeImpl::IsDefined(const string& function_name) {
  return lib_def_->Find(function_name) != nullptr;
}

FunctionLibraryRuntime* NewFunctionLibraryRuntime(
    Device* device, Runner runner, const FunctionLibraryDefinition* lib_def) {
  return new FunctionLibraryRuntimeImpl(device, runner, lib_def);
}

bool RemoveDeadNodes(Graph* g) {
  std::vector<bool> visited(g->num_node_ids(), false);
  visited[Graph::kSourceId] = true;
  visited[Graph::kSinkId] = true;
  std::deque<Node*> q;
  for (auto n : g->nodes()) {
    if (n->op_def().is_stateful()) {
      visited[n->id()] = true;
    } else if (n->type_string() == kArgOp) {
      visited[n->id()] = true;
    } else if (n->type_string() == kRetOp) {
      visited[n->id()] = true;
      q.push_back(n);
    }
  }
  while (!q.empty()) {
    const Node* n = q.front();
    q.pop_front();
    visited[n->id()] = true;
    for (auto e : n->in_edges()) {
      q.push_back(e->src());
    }
  }
  bool removed_any = false;
  for (Node* n : g->nodes()) {
    if (!visited[n->id()]) {
      g->RemoveNode(n);
      removed_any = true;
    }
  }
  return removed_any;
}

namespace {
// If 'edges' contains only 1 non-control edge, returns it. Otherwise,
// returns a nullptr.
const Edge* GetTheOnlyDataEdge(const EdgeSet& edges) {
  const Edge* ret = nullptr;
  for (const Edge* e : edges) {
    if (e->IsControlEdge() || ret) return nullptr;
    ret = e;
  }
  return ret;
}
}  // end namespace

bool RemoveIdentityNodes(Graph* g) {
  bool removed_any = false;
  gtl::InlinedVector<Node*, 8> matches;
  for (Node* n : g->nodes()) {
    if ((n->type_string() == "Identity") && GetTheOnlyDataEdge(n->in_edges())) {
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
      g->RemoveNode(n);
      removed_any = true;
    }
  }
  return removed_any;
}

bool RemoveListArrayConverter(Graph* g) {
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
  for (Node* n : fbody->graph->nodes()) {
    if (n->IsSource() || n->IsSink()) continue;
    CHECK(n->IsOp());
    node_map[n->id()] = g->CopyNode(n);
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
  std::vector<Node*> outputs(caller->num_inputs());
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

bool ExpandInlineFunctions(FunctionLibraryRuntime* lib, Graph* graph) {
  std::vector<std::pair<Node*, const FunctionBody*>> candidates;
  for (Node* node : graph->nodes()) {
    VLOG(3) << "Expanding " << node->DebugString();
    FunctionLibraryRuntime::Handle handle;
    Status s =
        lib->Instantiate(node->type_string(), node->def().attr(), &handle);
    if (!s.ok()) {
      // Either "node" is a primitive op, or the instantiation failed.
      if (errors::IsNotFound(s)) {
        VLOG(2) << "ExpandInlineFunctions " << s;
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

// TODO(zhifengc): Maybe this should be the default Graph::AsGraphDef.
// and stash the original NodeDef name as an attr for documentation
// purpose.
static void ToGraphDef(const Graph* g, GraphDef* gdef) {
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
    ndef->set_name(strings::StrCat("n", n->id()));
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
      if (e == nullptr) {
        ndef->add_input("unknown");
      } else if (!e->src()->IsOp()) {
      } else if (e->IsControlEdge()) {
        ndef->add_input(strings::StrCat("^n", e->src()->id()));
      } else if (e->src_output() == 0) {
        ndef->add_input(strings::StrCat("n", e->src()->id()));
      } else {
        ndef->add_input(
            strings::StrCat("n", e->src()->id(), ":", e->src_output()));
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

  // A vector of output endpoints which represents backpropagated
  // gradients
  typedef std::vector<Endpoint> BackpropedGradients;

  // backprops_ is a map from an output endpoint to its accumulated
  // gradients.  When an output endpoint has accumulated all its
  // gradients, we add a node which sums them up.
  std::unordered_map<Endpoint, BackpropedGradients, EndpointHash, EndpointEq>
      backprops_;

  // pending[i] is count-down counter for i-th node's expected
  // backprops.  When pending[i] becomes zero, we collected all
  // backprop gradients for all output endpoint of the ith-node.
  std::vector<int> pending_;

  // 'ready' keeps track of nodes that have been completely
  // backpropped. Initially, for every output y of the function f, we
  // add dy as an input of the the gradient function.
  std::deque<Node*> ready_;

  // Makes a copy of fbody_ in gbody_.
  void Copy();

  // Initialize pending_ and ready_.
  void InitBackprop();

  // In the original function body, there is a forward edge from 'src'
  // to 'dst', when the backprop algorithm constructs the node
  // 'dst_grad' which computes the gradient, we need to propagate it
  // to 'src'.
  void BackpropAlongEdge(const Endpoint& dst_grad, const Endpoint& src);
  void BackpropZerosAlongEdge(const Endpoint& src);

  Endpoint SumGradients(const Endpoint& src);

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

void SymbolicGradientHelper::BackpropAlongEdge(const Endpoint& dst_grad,
                                               const Endpoint& src) {
  CHECK_NOTNULL(src.node);
  auto iter = backprops_.find(src);
  if (iter != backprops_.end()) {
    auto* grads = &iter->second;
    grads->push_back(dst_grad);
    if (--pending_[src.node->id()] == 0) {
      ready_.push_back(src.node);
    }
  }
}

void SymbolicGradientHelper::BackpropZerosAlongEdge(const Endpoint& src) {
  CHECK_NOTNULL(src.node);
  auto iter = backprops_.find(src);
  if (iter != backprops_.end()) {
    if (--pending_[src.node->id()] == 0) {
      ready_.push_back(src.node);
    }
  }
}

void SymbolicGradientHelper::InitBackprop() {
  Graph* g = gbody_->graph;
  pending_.resize(g->num_node_ids(), 0);
  {
    backprops_.clear();
    std::unordered_set<Node*> visited;
    std::deque<Node*> queue;
    for (Node* n : gbody_->arg_nodes) {
      queue.push_back(n);
    }

    // Going forward to figure out which endpoints need backprop-ed.
    // A node's endpoints need to be backprop-ed only if one of the
    // arg node can reach the node via data edges.
    while (!queue.empty()) {
      Node* n = queue.front();
      queue.pop_front();
      visited.insert(n);
      for (int i = 0; i < n->num_outputs(); ++i) {
        backprops_[{n, i}].clear();
      }
      int num_expected_backprops = 0;
      for (const Edge* e : n->out_edges()) {
        if (e->IsControlEdge()) continue;
        ++num_expected_backprops;
        if (visited.find(e->dst()) == visited.end()) {
          queue.push_back(e->dst());
        }
      }
      pending_[n->id()] = num_expected_backprops;
    }
  }

  {
    const int num_y = gbody_->ret_nodes.size();
    for (int i = 0; i < num_y; ++i) {
      Node* y = gbody_->ret_nodes[i];
      DCHECK_EQ(y->type_string(), kRetOp);
      const DataType dtype = y->input_type(0);
      const int index = gbody_->arg_nodes.size();
      Node* dy = AddArg(g, dtype, index);
      gbody_->arg_types.push_back(dtype);
      gbody_->arg_nodes.push_back(dy);

      // What's the input to y?
      Endpoint y_in{nullptr, 0};
      for (const Edge* e : y->in_edges()) {
        if (!e->IsControlEdge()) {
          y_in = {e->src(), e->src_output()};
          break;
        }
      }
      CHECK_NOTNULL(y_in.node);
      BackpropAlongEdge({dy, 0}, y_in);
    }
  }
}

Endpoint SymbolicGradientHelper::SumGradients(const Endpoint& src) {
  Graph* g = gbody_->graph;
  const DataType dtype = src.dtype();
  auto iter = backprops_.find(src);
  CHECK(iter != backprops_.end());
  const auto& grads = iter->second;
  if (grads.empty()) {
    // Nothing propagated back. The best we can come up is zeros.
    Node* zero_like = AddZerosLike(g, src);
    return {zero_like, 0};
  }
  if (grads.size() == 1) {
    // Just one backprop edge.
    return grads[0];
  }
  // Otherwise, adds backprop-ed gradients.
  NodeDef ndef;
  ndef.set_name(g->NewName(kNodeLabel));
  ndef.set_op("AddN");  // N-way Add
  for (const Endpoint& ep : grads) {
    ndef.add_input(ep.name());
  }
  AddNodeAttr("N", static_cast<int64>(grads.size()), &ndef);
  AddNodeAttr("T", dtype, &ndef);
  Status s;
  Node* add = gbody_->graph->AddNode(ndef, &s);
  TF_CHECK_OK(s);
  for (size_t i = 0; i < grads.size(); ++i) {
    const Endpoint& ep = grads[i];
    g->AddEdge(ep.node, ep.index, add, i);
  }
  return {add, 0};
}

static bool IsPrimitiveOpWithNoGrad(const string& func) {
  gradient::Creator creator;
  Status s = gradient::GetOpGradientCreator(func, &creator);
  return s.ok() && (creator == nullptr);
}

FunctionBody* SymbolicGradientHelper::Compute() {
  CHECK(gbody_ == nullptr);
  gbody_ = new FunctionBody;

  // Copy fbody_ into gbody_.
  Copy();

  // Initialize backprops.
  InitBackprop();

  // Backward propagation.
  gtl::InlinedVector<Endpoint, 8> dy;
  Graph* g = gbody_->graph;
  while (!ready_.empty()) {
    // n has collected all gradients.
    Node* n = ready_.front();
    ready_.pop_front();

    if (n->type_string() == kArgOp) {
      // We'll handle the _Arg node after backprop is done.
      continue;
    }

    // "n" has num_x inputs and num_y outputs.
    const int num_x = n->num_inputs();
    const int num_y = n->num_outputs();

    // dy[i] is the sum of i-th output's backpropped gradients.
    dy.clear();
    dy.resize(num_y, {nullptr, 0});
    for (int i = 0; i < num_y; ++i) {
      dy[i] = SumGradients({n, i});
    }

    if (IsPrimitiveOpWithNoGrad(n->type_string())) {
      // No grad defined for this op.  Backprops zeros along the in
      // edges.
      for (const Edge* e : n->in_edges()) {
        if (e->IsControlEdge()) continue;
        BackpropZerosAlongEdge({e->src(), e->src_output()});
      }
      continue;
    }

    // Adds a gradient node with num_x + num_y inputs and num_x
    // outputs.
    Node* grad = AddSymGrad(g, n, dy);
    for (const Edge* e : n->in_edges()) {
      if (e->IsControlEdge()) continue;
      g->AddEdge(e->src(), e->src_output(), grad, e->dst_input());
    }
    for (int i = 0; i < num_y; ++i) {
      g->AddEdge(dy[i].node, dy[i].index, grad, num_x + i);
    }

    // Backprops along the in edges.
    for (const Edge* e : n->in_edges()) {
      if (e->IsControlEdge()) continue;
      BackpropAlongEdge({grad, e->dst_input()}, {e->src(), e->src_output()});
    }
  }

  // The gradient's retval nodes.
  for (Node* n : gbody_->ret_nodes) {
    g->RemoveNode(n);
  }
  gbody_->ret_types = fbody_->arg_types;
  gbody_->ret_nodes.clear();
  for (size_t i = 0; i < fbody_->arg_types.size(); ++i) {
    Endpoint grad = SumGradients({gbody_->arg_nodes[i], 0});
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
