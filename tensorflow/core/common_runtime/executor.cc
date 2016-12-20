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

#include "tensorflow/core/common_runtime/executor.h"

#include <atomic>
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/common_runtime/costmodel_manager.h"
#include "tensorflow/core/common_runtime/pending_counts.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/control_flow.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_segment.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_reference.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/edgeset.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/gtl/manual_constructor.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/tensor_slice_reader_cache.h"

namespace tensorflow {
namespace {

// 1-D, 0 element tensor.
static const Tensor* const kEmptyTensor = new Tensor;

bool IsInitializationOp(const Node* node) {
  return node->op_def().allows_uninitialized_input();
}

// Sets the timeline_label field of *node_stats, using data from *node.
// Returns true iff the node is a transfer node.
// TODO(tucker): merge with the DetailText function in session.cc
// in a common location.
bool SetTimelineLabel(const Node* node, NodeExecStats* node_stats) {
  bool is_transfer_node = false;
  string memory;
  for (auto& all : node_stats->memory()) {
    int64 tot = all.total_bytes();
    if (tot >= 0.1 * 1048576.0) {
      int64 peak = all.peak_bytes();
      if (peak > 0) {
        memory =
            strings::StrCat(memory, "[", all.allocator_name(),
                            strings::Printf(" %.1fMB %.1fMB] ", tot / 1048576.0,
                                            peak / 1048576.0));
      } else {
        memory = strings::StrCat(memory, "[", all.allocator_name(),
                                 strings::Printf(" %.1fMB] ", tot / 1048576.0));
      }
    }
  }
  const NodeDef& def = node->def();
  string text = "";
  if (IsSend(node)) {
    string tensor_name;
    TF_CHECK_OK(GetNodeAttr(def, "tensor_name", &tensor_name));
    string recv_device;
    TF_CHECK_OK(GetNodeAttr(def, "recv_device", &recv_device));
    text = strings::StrCat(memory, def.name(), " = ", def.op(), "(",
                           tensor_name, " @", recv_device);
    is_transfer_node = true;
  } else if (IsRecv(node)) {
    string tensor_name;
    TF_CHECK_OK(GetNodeAttr(def, "tensor_name", &tensor_name));
    string send_device;
    TF_CHECK_OK(GetNodeAttr(def, "send_device", &send_device));
    text = strings::StrCat(memory, def.name(), " = ", def.op(), "(",
                           tensor_name, " @", send_device);
    is_transfer_node = true;
  } else {
    text = strings::StrCat(
        memory, def.name(), " = ", def.op(), "(",
        str_util::Join(
            std::vector<StringPiece>(def.input().begin(), def.input().end()),
            ", "),
        ")");
  }
  node_stats->set_timeline_label(text);
  return is_transfer_node;
}

// Helper routines for collecting step stats.
namespace nodestats {
inline int64 NowInUsec() { return Env::Default()->NowMicros(); }

void SetScheduled(NodeExecStats* nt, int64 t) { nt->set_scheduled_micros(t); }

void SetAllStart(NodeExecStats* nt) { nt->set_all_start_micros(NowInUsec()); }

void SetOpStart(NodeExecStats* nt) {
  DCHECK_NE(nt->all_start_micros(), 0);
  nt->set_op_start_rel_micros(NowInUsec() - nt->all_start_micros());
}

void SetOpEnd(NodeExecStats* nt) {
  DCHECK_NE(nt->all_start_micros(), 0);
  nt->set_op_end_rel_micros(NowInUsec() - nt->all_start_micros());
}

void SetAllEnd(NodeExecStats* nt) {
  DCHECK_NE(nt->all_start_micros(), 0);
  nt->set_all_end_rel_micros(NowInUsec() - nt->all_start_micros());
}

void SetOutput(NodeExecStats* nt, int slot, const Tensor* v) {
  DCHECK(v);
  NodeOutput* no = nt->add_output();
  no->set_slot(slot);
  v->FillDescription(no->mutable_tensor_description());
}

void SetMemory(NodeExecStats* nt, OpKernelContext* ctx) {
  for (const auto& allocator_pair : ctx->wrapped_allocators()) {
    AllocatorMemoryUsed* memory = nt->add_memory();
    // retrieving the sizes from the wrapped allocator removes the
    // executor's reference to it, so allocator_pair.second must not
    // be dereferenced again after this statement
    auto sizes = allocator_pair.second->GetSizesAndUnRef();
    memory->set_allocator_name(allocator_pair.first->Name());
    int tb = sizes.first;
    memory->set_total_bytes(tb);
    if (allocator_pair.first->TracksAllocationSizes()) {
      memory->set_peak_bytes(sizes.second);
    }
  }
}

void SetReferencedTensors(NodeExecStats* nt,
                          const TensorReferenceVector& tensors) {
  // be careful not to increment the reference count on any tensor
  // while recording the information
  for (size_t i = 0; i < tensors.size(); ++i) {
    AllocationDescription* description = nt->add_referenced_tensor();
    tensors.at(i).FillDescription(description);
  }
}

}  // namespace nodestats

struct NodeItem {
  // A graph node.
  const Node* node = nullptr;

  // The kernel for this node.
  OpKernel* kernel = nullptr;

  bool kernel_is_expensive = false;  // True iff kernel->IsExpensive()
  bool kernel_is_async = false;      // True iff kernel->AsAsync() != nullptr
  bool is_merge = false;             // True iff IsMerge(node)

  // Cached values of node->num_inputs() and node->num_outputs(), to
  // avoid levels of indirection.
  int num_inputs;
  int num_outputs;

  // ExecutorImpl::tensors_[input_start] is the 1st positional input
  // for this node.
  int input_start = 0;

  // ExecutorImpl::output_attrs_[output_attr_start] is the 1st
  // positional attribute for the 0th output of this node.
  int output_attr_start = 0;

  DataType input_type(int i) const {
    DCHECK_LT(i, num_inputs);
    return (i < 4) ? inlined_input_type[i] : node->input_type(i);
  }
  DataType output_type(int i) const {
    DCHECK_LT(i, num_outputs);
    return (i < 4) ? inlined_output_type[i] : node->output_type(i);
  }
  // Cache first 4 input and output types to reduce levels of indirection
  DataType inlined_input_type[4];
  DataType inlined_output_type[4];
};

typedef gtl::InlinedVector<TensorValue, 4> TensorValueVec;
typedef gtl::InlinedVector<DeviceContext*, 4> DeviceContextVec;
typedef gtl::InlinedVector<AllocatorAttributes, 4> AllocatorAttributeVec;

class ExecutorImpl : public Executor {
 public:
  ExecutorImpl(const LocalExecutorParams& p, const Graph* g)
      : params_(p), graph_(g) {
    CHECK(p.create_kernel != nullptr);
    CHECK(p.delete_kernel != nullptr);
  }

  ~ExecutorImpl() override {
    for (int i = 0; i < graph_->num_node_ids(); i++) {
      params_.delete_kernel(nodes_[i].kernel);
    }
    for (auto fiter : frame_info_) {
      delete fiter.second;
    }
    delete[] frame_local_ids_;
    delete[] nodes_;
    delete graph_;
  }

  Status Initialize();

  // Infer memory allocation attributes of a node n's output,
  // based on its use node dst.  Note that dst might not be directly
  // connected to n by a single edge, but might be a downstream
  // consumer of n's output by reference.  *attr is updated with any
  // necessary attributes.
  Status InferAllocAttr(const Node* n, const Node* dst,
                        const DeviceNameUtils::ParsedName& local_dev_name,
                        AllocatorAttributes* attr);

  // Process all Nodes in the current graph, attempting to infer the
  // memory allocation attributes to be used wherever they may allocate
  // a tensor buffer.
  Status SetAllocAttrs();

  void RunAsync(const Args& args, DoneCallback done) override;

 private:
  friend class ExecutorState;

  struct ControlFlowInfo {
    gtl::FlatMap<string, int, HashStr> frame_name_to_size;
    std::vector<string> frame_names;
  };

  struct FrameInfo {
    FrameInfo()
        : input_count(0),
          total_inputs(0),
          pending_counts(nullptr),
          nodes(nullptr) {}

    // The total number of inputs to a frame.
    int input_count;

    // The total number of input tensors of a frame.
    // == sum(nodes[*].num_inputs()) where nodes are the nodes in the frame.
    int total_inputs;

    // Each frame has its own PendingCounts only for the nodes in the frame.
    PendingCounts* pending_counts;  // Owned

    // The nodes in a frame. Used only for debugging.
    std::vector<const Node*>* nodes;  // Owned

    ~FrameInfo() {
      delete pending_counts;
      delete nodes;
    }
  };

  static Status BuildControlFlowInfo(const Graph* graph,
                                     ControlFlowInfo* cf_info);
  void InitializePending(const Graph* graph, const ControlFlowInfo& cf_info);

  FrameInfo* EnsureFrameInfo(const string& fname) {
    auto slot = &frame_info_[fname];
    if (*slot == nullptr) {
      *slot = new FrameInfo;
    }
    return *slot;
  }

  // Owned.
  LocalExecutorParams params_;
  const Graph* graph_;
  NodeItem* nodes_ = nullptr;     // array of size "graph_.num_node_ids()"
  int total_output_tensors_ = 0;  // == sum(nodes_[*].num_outputs())

  // A cached value of params_
  bool device_record_tensor_accesses_ = false;

  // Root nodes (with no in edges) that should form the initial ready queue
  std::vector<const Node*> root_nodes_;

  std::vector<AllocatorAttributes> output_attrs_;

  // Mapping from frame name to static information about the frame.
  // TODO(yuanbyu): We could cache it along with the graph so to avoid
  // the overhead of constructing it for each executor instance.
  gtl::FlatMap<string, FrameInfo*, HashStr> frame_info_;

  // Mapping from a node's id to its index in the PendingCounts of the
  // frame the node belongs to.
  int* frame_local_ids_ = nullptr;  // Owned

  TF_DISALLOW_COPY_AND_ASSIGN(ExecutorImpl);
};

Status ExecutorImpl::Initialize() {
  const int num_nodes = graph_->num_node_ids();
  delete[] nodes_;
  nodes_ = new NodeItem[num_nodes];

  total_output_tensors_ = 0;

  // Build the information about frames in this subgraph.
  ControlFlowInfo cf_info;
  BuildControlFlowInfo(graph_, &cf_info);

  // Cache this value so we make this virtual function call once, rather
  // that O(# steps * # nodes per step) times.
  device_record_tensor_accesses_ =
      params_.device->RequiresRecordingAccessedTensors();

  for (auto& it : cf_info.frame_name_to_size) {
    EnsureFrameInfo(it.first)->nodes = new std::vector<const Node*>;
  }
  frame_local_ids_ = new int[num_nodes];
  gtl::FlatMap<string, int, HashStr> frame_count;

  // Preprocess every node in the graph to create an instance of op
  // kernel for each node.
  for (const Node* n : graph_->nodes()) {
    const int id = n->id();
    const string& frame_name = cf_info.frame_names[id];
    FrameInfo* frame_info = EnsureFrameInfo(frame_name);

    // See if this node is a root node, and if so, add to root_nodes_.
    const int num_in_edges = n->in_edges().size();
    if (num_in_edges == 0) {
      root_nodes_.push_back(n);
    }

    NodeItem* item = &nodes_[id];
    item->node = n;
    item->num_inputs = n->num_inputs();
    item->num_outputs = n->num_outputs();

    for (int i = 0; i < std::min(4, item->num_inputs); i++) {
      item->inlined_input_type[i] = n->input_type(i);
    }
    for (int i = 0; i < std::min(4, item->num_outputs); i++) {
      item->inlined_output_type[i] = n->output_type(i);
    }

    item->input_start = frame_info->total_inputs;
    frame_info->total_inputs += n->num_inputs();

    item->output_attr_start = total_output_tensors_;
    total_output_tensors_ += n->num_outputs();

    Status s = params_.create_kernel(n->def(), &item->kernel);
    if (!s.ok()) {
      item->kernel = nullptr;
      s = AttachDef(s, n->def());
      LOG(ERROR) << "Executor failed to create kernel. " << s;
      return s;
    }
    CHECK(item->kernel);
    item->kernel_is_expensive = item->kernel->IsExpensive();
    item->kernel_is_async = (item->kernel->AsAsync() != nullptr);
    item->is_merge = IsMerge(n);

    // Initialize static information about the frames in the graph.
    frame_local_ids_[id] = frame_count[frame_name]++;
    frame_info->nodes->push_back(n);
    if (IsEnter(n)) {
      string enter_name;
      TF_RETURN_IF_ERROR(GetNodeAttr(n->def(), "frame_name", &enter_name));
      EnsureFrameInfo(enter_name)->input_count++;
    }
  }

  // Initialize PendingCounts only after frame_local_ids_ is initialized.
  InitializePending(graph_, cf_info);

  return SetAllocAttrs();
}

Status ExecutorImpl::SetAllocAttrs() {
  Status s;
  Device* device = params_.device;
  DeviceNameUtils::ParsedName local_dev_name = device->parsed_name();

  output_attrs_.resize(total_output_tensors_);
  for (const Node* n : graph_->nodes()) {
    NodeItem* item = &nodes_[n->id()];
    const int base_index = item->output_attr_start;
    // Examine the out edges of each node looking for special use
    // cases that may affect memory allocation attributes.
    for (auto e : n->out_edges()) {
      const int index = e->src_output();
      AllocatorAttributes attr;
      s = InferAllocAttr(n, e->dst(), local_dev_name, &attr);
      if (!s.ok()) return s;
      if (attr.value != 0) {
        if (!e->IsControlEdge()) {
          output_attrs_[base_index + index].Merge(attr);
        }
      }
    }

    for (int out = 0; out < n->num_outputs(); out++) {
      OpKernel* op_kernel = item->kernel;
      DCHECK_LT(out, op_kernel->output_memory_types().size());
      bool on_host = op_kernel->output_memory_types()[out] == HOST_MEMORY;
      AllocatorAttributes h;
      h.set_on_host(on_host);
      output_attrs_[base_index + out].Merge(h);
    }
  }
  return s;
}

Status ExecutorImpl::InferAllocAttr(
    const Node* n, const Node* dst,
    const DeviceNameUtils::ParsedName& local_dev_name,
    AllocatorAttributes* attr) {
  Status s;
  // Note that it's possible for *n to be a Recv and *dst to be a Send,
  // so these two cases are not mutually exclusive.
  if (IsRecv(n)) {
    string src_name;
    s = GetNodeAttr(n->def(), "send_device", &src_name);
    if (!s.ok()) return s;
    DeviceNameUtils::ParsedName parsed_src_name;
    if (!DeviceNameUtils::ParseFullName(src_name, &parsed_src_name)) {
      s = errors::Internal("Bad send_device attr '", src_name, "' in node ",
                           n->name());
      return s;
    }
    if (!DeviceNameUtils::IsSameAddressSpace(parsed_src_name, local_dev_name)) {
      // Value is going to be the sink of an RPC.
      attr->set_nic_compatible(true);
      VLOG(2) << "node " << n->name() << " is the sink of an RPC in";
    } else if ((local_dev_name.type == "CPU" || n->IsHostRecv()) &&
               parsed_src_name.type != "CPU") {
      // Value is going to be the sink of a local DMA from GPU to CPU (or other
      // types of accelerators).
      attr->set_gpu_compatible(true);
      VLOG(2) << "node " << n->name() << " is the sink of a gpu->cpu copy";
    } else {
      VLOG(2) << "default alloc case local type " << local_dev_name.type
              << " remote type " << parsed_src_name.type;
    }
  }
  if (IsSend(dst)) {
    string dst_name;
    s = GetNodeAttr(dst->def(), "recv_device", &dst_name);
    if (!s.ok()) return s;
    DeviceNameUtils::ParsedName parsed_dst_name;
    if (!DeviceNameUtils::ParseFullName(dst_name, &parsed_dst_name)) {
      s = errors::Internal("Bad recv_device attr '", dst_name, "' in node ",
                           n->name());
      return s;
    }
    if (!DeviceNameUtils::IsSameAddressSpace(parsed_dst_name, local_dev_name)) {
      // Value is going to be the source of an RPC.
      attr->set_nic_compatible(true);
      VLOG(2) << "node " << n->name() << " is the source of an RPC out";
    } else if ((local_dev_name.type == "CPU" || dst->IsHostSend()) &&
               parsed_dst_name.type != "CPU") {
      // Value is going to be the source of a local DMA from CPU to GPU (or
      // other types of accelerators).
      // Note that this does not cover the case where the allocation of the
      // output tensor is not generated by the src: n.
      attr->set_gpu_compatible(true);
      VLOG(2) << "node " << n->name() << " is the source of a cpu->gpu copy";
    } else {
      VLOG(2) << "default alloc case local type " << local_dev_name.type
              << " remote type " << parsed_dst_name.type;
    }
  } else if (dst->type_string() == "ToFloat") {
    for (auto e : dst->out_edges()) {
      s = InferAllocAttr(n, e->dst(), local_dev_name, attr);
      if (!s.ok()) return s;
    }
  }
  return s;
}

// The state associated with one invocation of ExecutorImpl::Run.
// ExecutorState dispatches nodes when they become ready and keeps
// track of how many predecessors of a node have not done (pending_).
class ExecutorState {
 public:
  ExecutorState(const Executor::Args& args, ExecutorImpl* impl);
  ~ExecutorState();

  void RunAsync(Executor::DoneCallback done);

 private:
  // Either a tensor pointer (pass-by-reference) or a tensor (pass-by-value).
  // TODO(yuanbyu): A better way to do "has_value"?
  struct Entry {
    Entry() {}
    Entry(const Entry& other)
        : ref(other.ref),
          ref_mu(other.ref_mu),
          has_value(other.has_value),
          val_field_is_set(other.val_field_is_set),
          alloc_attr(other.alloc_attr),
          device_context(other.device_context) {
      if (val_field_is_set) {
        val.Init(*other.val);
      }
    }
    ~Entry() {
      if (val_field_is_set) val.Destroy();
    }

    Entry& operator=(const Entry& other) {
      if (val_field_is_set) {
        val.Destroy();
      }
      ref = other.ref;
      ref_mu = other.ref_mu;
      has_value = other.has_value;
      val_field_is_set = other.val_field_is_set;
      alloc_attr = other.alloc_attr;
      device_context = other.device_context;
      if (val_field_is_set) {
        val.Init(*other.val);
      }
      return *this;
    }

    // Clears the <val> field.
    void ClearVal() {
      if (val_field_is_set) {
        val.Destroy();
        val_field_is_set = false;
      }
    }

    // A tensor value, if val_field_is_set.
    ManualConstructor<Tensor> val;

    Tensor* ref = nullptr;    // A tensor reference.
    mutex* ref_mu = nullptr;  // mutex for *ref if ref is not nullptr.

    // Whether the value exists, either in <val> or <ref>.
    bool has_value = false;

    bool val_field_is_set = false;

    // The attributes of the allocator that creates the tensor.
    AllocatorAttributes alloc_attr;

    // Every entry carries an optional DeviceContext containing
    // Device-specific information about how the Tensor was produced.
    DeviceContext* device_context = nullptr;
  };

  // Contains a value for [node->id()] for the device context assigned by the
  // device at the beginning of a step.
  DeviceContextMap device_context_map_;

  struct TaggedNode;
  typedef gtl::InlinedVector<TaggedNode, 8> TaggedNodeSeq;
  typedef gtl::InlinedVector<Entry, 4> EntryVector;

  struct IterationState {
    explicit IterationState(const PendingCounts* pending_counts,
                            int total_input_tensors)
        : input_tensors(new Entry[total_input_tensors]),
          outstanding_ops(0),
          outstanding_frame_count(0),
          counts_(pending_counts->num_nodes()) {
      counts_.InitializeFrom(*pending_counts);
    }

    // The state of an iteration.

    // One copy per iteration. For iteration k, i-th node's j-th input is in
    // input_tensors[k][impl_->nodes[i].input_start + j]. An entry is either
    // a tensor pointer (pass-by-reference) or a tensor (pass-by-value).
    //
    // NOTE: No need to protect input_tensors[i] by any locks because it
    // is resized once. Each element of tensors_ is written once by the
    // source node of an edge and is cleared by the destination of the same
    // edge. The latter node is never run concurrently with the former node.
    Entry* input_tensors;

    // The number of outstanding ops for each iteration.
    int outstanding_ops;

    // The number of outstanding frames for each iteration.
    int outstanding_frame_count;
    int pending(int id) { return counts_.pending(id); }
    int decrement_pending(int id, int v) {
      return counts_.decrement_pending(id, v);
    }
    // Mark a merge node as live
    // REQUIRES: Node corresponding to "id" is a merge node
    void mark_live(int id) { counts_.mark_live(id); }
    // Mark a node to show that processing has started.
    void mark_started(int id) { counts_.mark_started(id); }
    // Mark a node to show that processing has completed.
    void mark_completed(int id) { counts_.mark_completed(id); }
    PendingCounts::NodeState node_state(int id) {
      return counts_.node_state(id);
    }

    int dead_count(int id) { return counts_.dead_count(id); }
    void increment_dead_count(int id) { counts_.increment_dead_count(id); }

    ~IterationState() { delete[] input_tensors; }

   private:
    PendingCounts counts_;
  };

  struct FrameState {
    explicit FrameState(const ExecutorImpl* impl, int parallel_iters)
        : executor(impl),
          max_parallel_iterations(parallel_iters),
          num_outstanding_iterations(1) {}

    // A new frame is created for each loop. Execution starts at iteration 0.
    // When a value at iteration 0 passes through a NextIteration node,
    // iteration 1 is created and starts running. Note that iteration 0 may
    // still be running so multiple iterations may run in parallel. The
    // frame maintains the state of iterations in several data structures
    // such as pending_count and input_tensors. When iteration 0 completes,
    // we garbage collect the state of iteration 0.
    //
    // A frame instance is considered "done" and can be garbage collected
    // if all its inputs have entered and all its iterations are "done".
    //
    // A frame manages the live iterations of an iterative computation.
    // Iteration i is considered "done" when there are no outstanding ops,
    // frames at iteration i are done, all recvs for this iteration are
    // completed, and iteration i-1 is done. For iteration 0, we instead
    // wait for there to be no more pending inputs of the frame.
    //
    // Frames and iterations are garbage collected once they are done.
    // The state we need to keep around is highly dependent on the
    // parallelism enabled by the scheduler. We may want to have the
    // scheduler dynamically control the outstanding number of live
    // parallel frames and iterations. To reduce the state space, the
    // scheduler might want to schedule ops in inner frames first and
    // lower iterations first.
    //
    // This frame state is mostly initialized lazily on demand so we
    // don't introduce unnecessary overhead.

    // The executor the frame is in.
    const ExecutorImpl* executor = nullptr;

    // The name of this frame, which is the concatenation of its parent
    // frame name, the iteration of the parent frame when this frame was
    // created, and the value of the attr 'frame_name'.
    string frame_name;

    // The unique id for this frame. Generated by fingerprinting
    // frame_name.
    uint64 frame_id;

    // The iteration id of its parent frame when this frame is created.
    // -1 if there is no parent frame. The frame_name/parent_iter pair
    // uniquely identifies this FrameState.
    int64 parent_iter = -1;

    // The FrameState of its parent frame.
    FrameState* parent_frame = nullptr;

    // The maximum allowed number of parallel iterations.
    const int max_parallel_iterations;

    // The number of inputs this frame is still waiting.
    int num_pending_inputs = 0;

    // The highest iteration number we have reached so far in this frame.
    int64 iteration_count GUARDED_BY(mu) = 0;

    // The number of outstanding iterations.
    int num_outstanding_iterations GUARDED_BY(mu) = 1;

    // The active iteration states of this frame.
    gtl::InlinedVector<IterationState*, 12> iterations;

    // The NextIteration nodes to enter a new iteration. If the number of
    // outstanding iterations reaches the limit, we will defer the start of
    // the next iteration until the number of outstanding iterations falls
    // below the limit.
    std::vector<std::pair<const Node*, Entry>> next_iter_roots GUARDED_BY(mu);

    // The values of the loop invariants for this loop. They are added into
    // this list as they "enter" the frame. When a loop invariant enters,
    // we make it available to all active iterations. When the frame starts
    // a new iteration, we make all the current loop invariants available
    // to the new iteration.
    std::vector<std::pair<const Node*, Entry>> inv_values GUARDED_BY(mu);

    // The list of dead exit nodes for the current highest iteration. We
    // will only "execute" the dead exits of the final iteration.
    std::vector<const Node*> dead_exits GUARDED_BY(mu);

    // Static information specific to this frame.
    PendingCounts* pending_counts = nullptr;
    int total_input_tensors = 0;
    std::vector<const Node*>* nodes = nullptr;

    // Lock ordering: ExecutorState.mu_ < mu.
    mutex mu;

    void InitializeFrameInfo(const string& enter_name) {
      auto it_frame_info = executor->frame_info_.find(enter_name);
      DCHECK(it_frame_info != executor->frame_info_.end());
      ExecutorImpl::FrameInfo* finfo = it_frame_info->second;
      pending_counts = finfo->pending_counts;
      total_input_tensors = finfo->total_inputs;
      num_pending_inputs = finfo->input_count;
      nodes = finfo->nodes;
    }

    inline IterationState* GetIteration(int64 iter)
        EXCLUSIVE_LOCKS_REQUIRED(mu) {
      int index = iter % iterations.size();
      return iterations[index];
    }

    inline void SetIteration(int64 iter, IterationState* state)
        EXCLUSIVE_LOCKS_REQUIRED(mu) {
      int index = iter % iterations.size();
      DCHECK(state == nullptr || iterations[index] == nullptr);
      iterations[index] = state;
    }

    // Decrement the outstanding op count and clean up the iterations in the
    // frame. Return true iff the execution of the frame is done.
    inline bool DecrementOutstandingOps(int64 iter, TaggedNodeSeq* ready) {
      mutex_lock l(mu);
      GetIteration(iter)->outstanding_ops--;
      return CleanupIterations(iter, ready);
    }

    // Returns true if the computation in the frame is completed.
    inline bool IsFrameDone() EXCLUSIVE_LOCKS_REQUIRED(mu) {
      return (num_pending_inputs == 0 && num_outstanding_iterations == 0);
    }

    // Returns true if the iteration of the frame is completed.
    bool IsIterationDone(int64 iter) EXCLUSIVE_LOCKS_REQUIRED(mu);

    // Increments the iteration id. If this is a new iteration, initialize it.
    void IncrementIteration(TaggedNodeSeq* ready) EXCLUSIVE_LOCKS_REQUIRED(mu);

    // Activate all the deferred NextIteration nodes in a new iteration.
    void ActivateNexts(int64 iter, TaggedNodeSeq* ready)
        EXCLUSIVE_LOCKS_REQUIRED(mu);

    // Activate all the current loop invariants in a new iteration.
    void ActivateLoopInvs(int64 iter, TaggedNodeSeq* ready)
        EXCLUSIVE_LOCKS_REQUIRED(mu);

    // Add a new loop invariant and make it available to all active iterations.
    void AddLoopInv(const Node* node, const Entry& value, TaggedNodeSeq* ready)
        EXCLUSIVE_LOCKS_REQUIRED(mu);

    // Activate the successors of a node.
    void ActivateNodes(const Node* node, const bool is_dead, int64 iter,
                       const EntryVector& outputs, TaggedNodeSeq* ready)
        EXCLUSIVE_LOCKS_REQUIRED(mu);

    // Cleanup iterations of this frame starting from iteration iter.
    bool CleanupIterations(int64 iter, TaggedNodeSeq* ready)
        EXCLUSIVE_LOCKS_REQUIRED(mu);

    ~FrameState() {
      for (size_t i = 0; i < iterations.size(); ++i) {
        delete iterations[i];
        iterations[i] = nullptr;
      }
    }
  };

  // A tagged node: <frame*, iter, node*>.
  struct TaggedNode {
    const Node* node = nullptr;
    FrameState* input_frame = nullptr;
    int64 input_iter = -1;
    bool is_dead = false;

    TaggedNode(const Node* t_node, FrameState* in_frame, int64 in_iter,
               bool dead) {
      node = t_node;
      input_frame = in_frame;
      input_iter = in_iter;
      is_dead = dead;
    }
  };

  // A drop-in replacement for std::deque<TaggedNode>.  We typically don't
  // have that many nodes in the ready queue, so we just use a vector and
  // don't free up memory from the queue as we consume nodes.
  class TaggedNodeReadyQueue {
   public:
    TaggedNodeReadyQueue() : front_index_(0) {}

    void push_back(TaggedNode node) { ready_.push_back(node); }
    TaggedNode front() const {
      DCHECK_LT(front_index_, ready_.size());
      return ready_[front_index_];
    }
    void pop_front() {
      DCHECK_LT(front_index_, ready_.size());
      front_index_++;
      if ((front_index_ == ready_.size()) || (front_index_ > 16384)) {
        if (front_index_ == ready_.size()) {
          ready_.clear();
        } else {
          // Lots of unused entries at beginning of vector: move everything down
          // to start of vector.
          ready_.erase(ready_.begin(), ready_.begin() + front_index_);
        }
        front_index_ = 0;
      }
    }
    bool empty() const { return ready_.empty(); }
    const TaggedNode* begin() const { return ready_.begin() + front_index_; }
    const TaggedNode* end() const { return ready_.end(); }

   private:
    gtl::InlinedVector<TaggedNode, 16> ready_;
    int front_index_;
  };

  struct AsyncState;

  const bool vlog_;  // true if VLOG_IS_ON(1). Used to check vlog cheaply.

  // true if LogMemory::IsEnabled(). Used to check memory enabled cheaply.
  const bool log_memory_;

  int64 step_id_;
  // Not owned.
  Rendezvous* rendezvous_;
  SessionState* session_state_;
  TensorStore* tensor_store_;
  // Step-local container.
  ScopedStepContainer* step_container_;
  StepStatsCollector* stats_collector_;
  // QUESTION: Make it a checkpoint::TensorSliceReaderCacheWrapper
  // instead of a pointer?  (avoids having to delete).
  checkpoint::TensorSliceReaderCacheWrapper* slice_reader_cache_;
  FunctionCallFrame* call_frame_;
  const ExecutorImpl* impl_;
  CancellationManager* cancellation_manager_;
  Executor::Args::Runner runner_;
  bool sync_on_finish_;

  // Owned.

  // A flag that is set on error after the frame state has been
  // dumped for diagnostic purposes.
  bool dumped_on_error_ = false;

  // The root frame in which the execution of this step is started.
  FrameState* root_frame_;

  // Invoked when the execution finishes.
  Executor::DoneCallback done_cb_;

  std::atomic_int_fast32_t num_outstanding_ops_;

  mutex mu_;
  Status status_ GUARDED_BY(mu_);

  // Mapping from frame name to outstanding frames. A new frame is created
  // at some iteration of an active frame. So the unique key for the new
  // child frame is composed of the name of the parent frame, the iteration
  // number at which the parent frame is creating the new frame, and the
  // name of the new frame from nodedef.
  gtl::FlatMap<string, FrameState*, HashStr> outstanding_frames_
      GUARDED_BY(mu_);

  // The unique name of a frame.
  inline string MakeFrameName(FrameState* frame, int64 iter_id, string name) {
    return strings::StrCat(frame->frame_name, ";", iter_id, ";", name);
  }

  // Find an existing or create a new child frame in the frame 'frame' at
  // iteration 'iter'.
  void FindOrCreateChildFrame(FrameState* frame, int64 iter, const Node* node,
                              FrameState** child);

  // Delete a frame. Called when the frame is done.
  void DeleteFrame(FrameState* frame, TaggedNodeSeq* ready);

  // Cleanup frames and iterations starting from frame/iter. Called when
  // a child frame is done.
  void CleanupFramesIterations(FrameState* frame, int64 iter,
                               TaggedNodeSeq* ready);

  // Process a ready node in current thread.
  void Process(TaggedNode node, int64 scheduled_usec);

  // Before invoking item->kernel, fills in its "inputs".
  Status PrepareInputs(const NodeItem& item, Entry* first_input,
                       TensorValueVec* inputs,
                       DeviceContextVec* input_device_contexts,
                       AllocatorAttributeVec* input_alloc_attrs,
                       bool* is_input_dead);

  // After item->kernel computation is done, processes its outputs.
  Status ProcessOutputs(const NodeItem& item, OpKernelContext* ctx,
                        EntryVector* outputs, NodeExecStats* stats);

  // After processing the outputs, propagates the outputs to their dsts.
  void PropagateOutputs(const TaggedNode& tagged_node,
                        const EntryVector& outputs, TaggedNodeSeq* ready);

  // "node" just finishes. Takes ownership of "stats". Returns true if
  // execution has completed.
  bool NodeDone(const Status& s, const Node* node, const TaggedNodeSeq& ready,
                NodeExecStats* stats, TaggedNodeReadyQueue* inline_ready);

  // Schedule all the expensive nodes in 'ready', and put all the inexpensive
  // nodes in 'ready' into 'inline_ready'.
  void ScheduleReady(const TaggedNodeSeq& ready,
                     TaggedNodeReadyQueue* inline_ready);

  // For debugging/logging only.
  inline void MaybeMarkCompleted(FrameState* frame, int64 iter, int64 id);

  // Provide debugging output about an outstanding node in the executor.
  void DumpPendingNodeState(const int node_id, const Entry* input_vector,
                            bool show_nodes_with_no_ready_inputs);
  void DumpActiveNodeState(const int node_id, const Entry* input_vector);

  // Provide debugging output about an outstanding iteration in the executor.
  void DumpIterationState(const FrameState* frame, IterationState* iteration);

  // Provide debugging output of the state of the executor.
  void DumpState();
  const Tensor* GetTensorValueForDump(const Entry& input);

  // Clean up when this executor is done.
  void Finish();

  // A standalone routine for this expression so that we can express
  // that we don't want thread safety analysis on this reference (it's
  // safe to do without the lock because the iterations array never
  // resizes and this particular iteration's array element will not
  // be changed out from under us because the iteration is still alive).
  Entry* GetInputTensors(FrameState* input_frame,
                         int64 input_iter) const NO_THREAD_SAFETY_ANALYSIS {
    return input_frame->GetIteration(input_iter)->input_tensors;
  }
};

ExecutorState::ExecutorState(const Executor::Args& args, ExecutorImpl* impl)
    : vlog_(VLOG_IS_ON(1)),
      log_memory_(LogMemory::IsEnabled()),
      step_id_(args.step_id),
      rendezvous_(args.rendezvous),
      session_state_(args.session_state),
      tensor_store_(args.tensor_store),
      step_container_(args.step_container),
      stats_collector_(args.stats_collector),
      slice_reader_cache_(new checkpoint::TensorSliceReaderCacheWrapper),
      call_frame_(args.call_frame),
      impl_(impl),
      cancellation_manager_(args.cancellation_manager),
      runner_(args.runner),
      sync_on_finish_(args.sync_on_finish),
      num_outstanding_ops_(0) {
  // We start the entire execution in iteration 0 of the root frame
  // so let us create the root frame and the state for iteration 0.
  // We assume root_frame_->frame_name.empty().
  root_frame_ = new FrameState(impl_, 1);
  root_frame_->frame_id = 0;  // must be 0
  root_frame_->InitializeFrameInfo(root_frame_->frame_name);

  // Initialize iteration 0.
  root_frame_->iterations.resize(root_frame_->max_parallel_iterations);
  root_frame_->iterations[0] = new IterationState(
      root_frame_->pending_counts, root_frame_->total_input_tensors);

  outstanding_frames_.insert({root_frame_->frame_name, root_frame_});
}

ExecutorState::~ExecutorState() {
  for (auto name_frame : outstanding_frames_) {
    delete name_frame.second;
  }
  for (auto it : device_context_map_) {
    it->Unref();
  }
  delete slice_reader_cache_;
}

Status ExecutorImpl::BuildControlFlowInfo(const Graph* g,
                                          ControlFlowInfo* cf_info) {
  const int num_nodes = g->num_node_ids();
  cf_info->frame_names.resize(num_nodes);
  std::vector<Node*> parent_nodes;
  parent_nodes.resize(num_nodes);
  std::vector<bool> visited;
  visited.resize(num_nodes);

  string frame_name;
  std::deque<Node*> ready;

  // Initialize with the root nodes.
  for (Node* n : g->nodes()) {
    if (n->in_edges().empty()) {
      visited[n->id()] = true;
      ++cf_info->frame_name_to_size[frame_name];
      ready.push_back(n);
    }
  }

  while (!ready.empty()) {
    Node* curr_node = ready.front();
    int curr_id = curr_node->id();
    ready.pop_front();

    Node* parent = nullptr;
    if (IsEnter(curr_node)) {
      // Enter a child frame.
      TF_RETURN_IF_ERROR(
          GetNodeAttr(curr_node->def(), "frame_name", &frame_name));
      parent = curr_node;
    } else if (IsExit(curr_node)) {
      // Exit to the parent frame.
      parent = parent_nodes[curr_id];
      frame_name = cf_info->frame_names[parent->id()];
      parent = parent_nodes[parent->id()];
    } else {
      parent = parent_nodes[curr_id];
      frame_name = cf_info->frame_names[curr_id];
    }

    for (const Edge* out_edge : curr_node->out_edges()) {
      Node* out = out_edge->dst();
      int out_id = out->id();

      // Add to ready queue if not visited.
      bool is_visited = visited[out_id];
      if (!is_visited) {
        ready.push_back(out);
        visited[out_id] = true;

        // Process the node 'out'.
        cf_info->frame_names[out_id] = frame_name;
        parent_nodes[out_id] = parent;
        ++cf_info->frame_name_to_size[frame_name];
      }
    }
  }

  return Status::OK();
}

void ExecutorImpl::InitializePending(const Graph* graph,
                                     const ControlFlowInfo& cf_info) {
  for (auto& it : cf_info.frame_name_to_size) {
    PendingCounts* counts = new PendingCounts(it.second);
    EnsureFrameInfo(it.first)->pending_counts = counts;
    // Make sure everything is initialized
    for (int id = 0; id < it.second; id++) {
      counts->set_initial_count(id, 0, 0);
    }
  }
  for (const Node* n : graph->nodes()) {
    const int id = n->id();
    const int pending_id = frame_local_ids_[id];
    const int num_in_edges = n->in_edges().size();
    int initial_count;
    if (IsMerge(n)) {
      // merge waits all control inputs so we initialize the pending
      // count to be the number of control edges.
      int32 num_control_edges = 0;
      for (const Edge* edge : n->in_edges()) {
        if (edge->IsControlEdge()) {
          num_control_edges++;
        }
      }
      // Use bit 0 to indicate if we are waiting for a ready live data input.
      initial_count = 1 + (num_control_edges << 1);
    } else {
      initial_count = num_in_edges;
    }
    const string& name = cf_info.frame_names[id];
    PendingCounts* counts = EnsureFrameInfo(name)->pending_counts;
    counts->set_initial_count(pending_id, initial_count, num_in_edges);
  }
}

void ExecutorState::RunAsync(Executor::DoneCallback done) {
  const Graph* graph = impl_->graph_;
  TaggedNodeSeq ready;

  // Ask the device to fill in the device context map.
  Device* device = impl_->params_.device;
  Status fill_status = device->FillContextMap(graph, &device_context_map_);
  if (!fill_status.ok()) {
    done(fill_status);
    return;
  }

  // Initialize the ready queue.
  for (const Node* n : impl_->root_nodes_) {
    DCHECK_EQ(n->in_edges().size(), 0);
    ready.push_back(TaggedNode{n, root_frame_, 0, false});
  }
  if (ready.empty()) {
    done(Status::OK());
  } else {
    num_outstanding_ops_ = ready.size();
    root_frame_->iterations[0]->outstanding_ops = ready.size();
    done_cb_ = done;
    // Schedule to run all the ready ops in thread pool.
    ScheduleReady(ready, nullptr);
  }
}

// State kept alive for executing an asynchronous node in another
// thread.  NOTE: We need to make a copy of p.input,
// p.input_device_contexts, and p.input_alloc_attrs for asynchronous
// kernels because OpKernelContext methods like input_type(i) needs
// the param points to valid input type vector. It's not an issue for
// sync kernels because these vectors are kept on the stack.
struct ExecutorState::AsyncState {
  AsyncState(const OpKernelContext::Params& p, const TaggedNode& _tagged_node,
             const NodeItem& _item, Entry* _first_input, NodeExecStats* _stats)
      : saved_inputs(*p.inputs),
        saved_input_device_contexts(*p.input_device_contexts),
        saved_input_alloc_attrs(*p.input_alloc_attrs),
        params(p),
        tagged_node(_tagged_node),
        item(_item),
        first_input(_first_input),
        // ParamsButClearingEigenGPUDevice does equivalent of
        //   params.eigen_gpu_device = nullptr;
        ctx(ParamsButClearingEigenGPUDevice(&params), item.num_outputs),
        stats(_stats) {
    params.inputs = &saved_inputs;
    params.input_device_contexts = &saved_input_device_contexts;
    params.input_alloc_attrs = &saved_input_alloc_attrs;
  }

  TensorValueVec saved_inputs;
  DeviceContextVec saved_input_device_contexts;
  AllocatorAttributeVec saved_input_alloc_attrs;
  OpKernelContext::Params params;
  TaggedNode tagged_node;
  NodeItem item;
  Entry* first_input;
  OpKernelContext ctx;
  NodeExecStats* stats;

 private:
  OpKernelContext::Params* ParamsButClearingEigenGPUDevice(
      OpKernelContext::Params* p) {
    // Ensure OpKernelContext constructor will make a new eigen GPU device if
    // necessary.
    p->eigen_gpu_device = nullptr;  // Force allocation
    return p;
  }
};

void ExecutorState::Process(TaggedNode tagged_node, int64 scheduled_usec) {
  const NodeItem* nodes = impl_->nodes_;
  TaggedNodeSeq ready;
  TaggedNodeReadyQueue inline_ready;

  // Parameters passed to OpKernel::Compute.
  TensorValueVec inputs;
  DeviceContextVec input_device_contexts;
  AllocatorAttributeVec input_alloc_attrs;

  OpKernelContext::Params params;
  params.step_id = step_id_;
  Device* device = impl_->params_.device;
  params.device = device;
  params.log_memory = log_memory_;
  params.record_tensor_accesses = impl_->device_record_tensor_accesses_;
  params.rendezvous = rendezvous_;
  params.session_state = session_state_;
  params.tensor_store = tensor_store_;
  params.cancellation_manager = cancellation_manager_;
  params.call_frame = call_frame_;
  params.function_library = impl_->params_.function_library;
  params.resource_manager = device->resource_manager();
  params.step_container = step_container_;
  params.slice_reader_cache = slice_reader_cache_;
  params.inputs = &inputs;
  params.input_device_contexts = &input_device_contexts;
  params.input_alloc_attrs = &input_alloc_attrs;
  params.runner = &runner_;

  Status s;
  NodeExecStats* stats = nullptr;
  EntryVector outputs;
  bool completed = false;
  inline_ready.push_back(tagged_node);
  while (!inline_ready.empty()) {
    tagged_node = inline_ready.front();
    inline_ready.pop_front();
    const Node* node = tagged_node.node;
    FrameState* input_frame = tagged_node.input_frame;
    int64 input_iter = tagged_node.input_iter;
    const int id = node->id();
    const NodeItem& item = nodes[id];

    // TODO(misard) Replace with a finer-grain enabling flag once we
    // add better optional debugging support.
    if (vlog_ && VLOG_IS_ON(1)) {
      int pending_id = impl_->frame_local_ids_[id];
      mutex_lock l(input_frame->mu);
      input_frame->GetIteration(input_iter)->mark_started(pending_id);
    }

    // Set the device_context for this node id, if it exists.
    if (id < device_context_map_.size()) {
      params.op_device_context = device_context_map_[id];
    }

    params.track_allocations = false;
    stats = nullptr;
    if (stats_collector_ && !tagged_node.is_dead) {
      // track allocations if and only if we are collecting statistics
      params.track_allocations = true;
      stats = new NodeExecStats;
      stats->set_node_name(node->name());
      nodestats::SetScheduled(stats, scheduled_usec);
      nodestats::SetAllStart(stats);
    }

    if (vlog_) {
      VLOG(1) << "Process node: " << id << " step " << params.step_id << " "
              << SummarizeNodeDef(node->def());
    }

    Entry* input_tensors = GetInputTensors(input_frame, input_iter);
    Entry* first_input = input_tensors + item.input_start;
    outputs.clear();

    TensorReferenceVector accessed_tensors;
    DeviceContext* device_context = nullptr;
    // Only execute this node if it is not dead or it is a send/recv
    // transfer node. For transfer nodes, we need to propagate the "dead"
    // bit even when the node is dead.
    bool launched_asynchronously = false;
    if (tagged_node.is_dead && !IsTransferNode(node)) {
      outputs.resize(item.num_outputs);
    } else {
      // Prepares inputs.
      bool is_input_dead = false;
      s = PrepareInputs(item, first_input, &inputs, &input_device_contexts,
                        &input_alloc_attrs, &is_input_dead);
      if (!s.ok()) {
        // Clear inputs.
        int num_inputs = item.num_inputs;
        for (int i = 0; i < num_inputs; ++i) {
          (first_input + i)->ClearVal();
        }
        MaybeMarkCompleted(input_frame, input_iter, id);
        // Continue to process the nodes in 'inline_ready'.
        completed = NodeDone(s, item.node, ready, stats, &inline_ready);
        continue;
      }

      // Set up compute params.
      OpKernel* op_kernel = item.kernel;
      params.op_kernel = op_kernel;
      params.frame_iter = FrameAndIter(input_frame->frame_id, input_iter);
      params.is_input_dead = is_input_dead;
      params.output_attr_array =
          gtl::vector_as_array(&impl_->output_attrs_) + item.output_attr_start;

      if (item.kernel_is_async) {
        // Asynchronous computes.
        AsyncOpKernel* async = item.kernel->AsAsync();
        DCHECK(async != nullptr);
        launched_asynchronously = true;
        AsyncState* state =
            new AsyncState(params, tagged_node, item, first_input, stats);

        auto done = [this, state]() {
          Device* device = impl_->params_.device;
          NodeExecStats* stats = state->stats;      // Shorthand
          Entry* first_input = state->first_input;  // Shorthand

          if (vlog_) {
            VLOG(2) << this << " Async kernel done: "
                    << SummarizeNodeDef(state->item.node->def());
          }
          if (stats) nodestats::SetOpEnd(stats);
          EntryVector outputs;
          Status s = ProcessOutputs(state->item, &state->ctx, &outputs, stats);
          if (stats) nodestats::SetMemory(stats, &state->ctx);
          // Clears inputs.
          const int num_inputs = state->item.num_inputs;
          for (int i = 0; i < num_inputs; ++i) {
            (first_input + i)->ClearVal();
          }
          FrameState* input_frame = state->tagged_node.input_frame;
          const int64 input_iter = state->tagged_node.input_iter;
          const int id = state->tagged_node.node->id();
          MaybeMarkCompleted(input_frame, input_iter, id);
          TaggedNodeSeq ready;
          if (s.ok()) {
            PropagateOutputs(state->tagged_node, outputs, &ready);
          }
          outputs.clear();
          if (s.ok() && impl_->device_record_tensor_accesses_) {
            // Get the list of all tensors accessed during the execution
            TensorReferenceVector accessed;
            state->ctx.retrieve_accessed_tensors(&accessed);
            if (stats) nodestats::SetReferencedTensors(stats, accessed);
            // callee takes ownership of the vector
            device->ConsumeListOfAccessedTensors(state->ctx.op_device_context(),
                                                 accessed);
          }
          bool completed = NodeDone(s, state->item.node, ready, stats, nullptr);
          delete state;
          if (completed) Finish();
        };
        if (stats) nodestats::SetOpStart(stats);
        device->ComputeAsync(async, &state->ctx, done);
      } else {
        // Synchronous computes.
        OpKernelContext ctx(&params, item.num_outputs);
        if (stats) nodestats::SetOpStart(stats);
        device->Compute(CHECK_NOTNULL(op_kernel), &ctx);
        if (stats) nodestats::SetOpEnd(stats);

        s = ProcessOutputs(item, &ctx, &outputs, stats);
        if (s.ok() && impl_->device_record_tensor_accesses_) {
          // Get the list of all tensors accessed during the execution
          ctx.retrieve_accessed_tensors(&accessed_tensors);
          device_context = ctx.op_device_context();
        }
        if (stats) nodestats::SetMemory(stats, &ctx);
      }
    }

    if (!launched_asynchronously) {
      // Clears inputs.
      const int num_inputs = item.num_inputs;
      for (int i = 0; i < num_inputs; ++i) {
        (first_input + i)->ClearVal();
      }
      MaybeMarkCompleted(input_frame, input_iter, id);
      // Propagates outputs.
      if (s.ok()) {
        PropagateOutputs(tagged_node, outputs, &ready);
      }
      outputs.clear();
      if (!accessed_tensors.empty()) {
        if (stats) nodestats::SetReferencedTensors(stats, accessed_tensors);
        // device_context is set above in synchronous computes
        device->ConsumeListOfAccessedTensors(device_context, accessed_tensors);
      }
      if (stats) {
        scheduled_usec = nodestats::NowInUsec();
      }
      // Postprocess.
      completed = NodeDone(s, item.node, ready, stats, &inline_ready);
    }
  }  // while !inline_ready.empty()

  // This thread of computation is done if completed = true.
  if (completed) Finish();
}

Status ExecutorState::PrepareInputs(const NodeItem& item, Entry* first_input,
                                    TensorValueVec* inputs,
                                    DeviceContextVec* input_device_contexts,
                                    AllocatorAttributeVec* input_alloc_attrs,
                                    bool* is_input_dead) {
  const Node* node = item.node;

  inputs->clear();
  inputs->resize(item.num_inputs);
  input_device_contexts->clear();
  input_device_contexts->resize(item.num_inputs);
  input_alloc_attrs->clear();
  input_alloc_attrs->resize(item.num_inputs);

  *is_input_dead = false;

  bool is_merge = item.is_merge;
  for (int i = 0; i < item.num_inputs; ++i) {
    const bool expect_ref = IsRefType(item.input_type(i));
    Entry* entry = first_input + i;
    (*input_device_contexts)[i] = entry->device_context;
    (*input_alloc_attrs)[i] = entry->alloc_attr;

    // i-th input.
    TensorValue* inp = &(*inputs)[i];

    // Only merge and transfer nodes can have no-value inputs.
    if (!entry->has_value) {
      if (!is_merge) {
        DCHECK(IsTransferNode(node));
        DCHECK(!entry->val_field_is_set);
        entry->has_value = true;
        entry->val_field_is_set = true;
        entry->val.Init(*kEmptyTensor);
        inp->tensor = entry->val.get();
        *is_input_dead = true;
      }
      continue;
    }
    if (entry->ref == nullptr) {
      if (expect_ref) {
        return AttachDef(
            errors::InvalidArgument(i, "-th input expects a ref type"),
            item.kernel->def());
      }
      inp->tensor = entry->val.get();
    } else {
      if (!entry->ref->IsInitialized() && !IsInitializationOp(item.node)) {
        return AttachDef(
            errors::FailedPrecondition("Attempting to use uninitialized value ",
                                       item.kernel->def().input(i)),
            item.kernel->def());
      }
      if (expect_ref) {
        inp->mutex_if_ref = entry->ref_mu;
        inp->tensor = entry->ref;
      } else {
        // Automatically deref the tensor ref when the op expects a
        // tensor but is given a ref to a tensor.  Need to deref it
        // under the mutex.
        {
          mutex_lock l(*(entry->ref_mu));
          DCHECK(!entry->val_field_is_set);
          entry->val.Init(*entry->ref);
          entry->val_field_is_set = true;
        }
        entry->ref = nullptr;
        entry->ref_mu = nullptr;

        inp->tensor = entry->val.get();
      }
    }
  }
  return Status::OK();
}

Status ExecutorState::ProcessOutputs(const NodeItem& item, OpKernelContext* ctx,
                                     EntryVector* outputs,
                                     NodeExecStats* stats) {
  const Node* node = item.node;
  DCHECK_EQ(0, outputs->size());
  outputs->resize(item.num_outputs);

  Status s = ctx->status();
  if (!s.ok()) {
    s = AttachDef(s, item.kernel->def());
    // TODO(misard) Replace with a finer-grain enabling flag once we
    // add better optional debugging support.
    if (vlog_ && VLOG_IS_ON(1)) {
      LOG(WARNING) << this << " Compute status: " << s;
      DumpState();
    }
    return s;
  }

  // Get the device_context for this node id, if it exists.
  DeviceContext* device_context = nullptr;
  if (node->id() < device_context_map_.size()) {
    device_context = device_context_map_[node->id()];
  }

  // Experimental: debugger (tfdb) access to intermediate node completion.
  if (item.num_outputs == 0 && impl_->params_.node_outputs_cb != nullptr) {
    // If the node has no output, invoke the callback with output slot set to
    // -1, signifying that this is a no-output node.
    impl_->params_.node_outputs_cb(item.node->name(), -1, nullptr, false, ctx);
  }

  for (int i = 0; i < item.num_outputs; ++i) {
    TensorValue val = ctx->release_output(i);
    if (*ctx->is_output_dead() || val.tensor == nullptr) {
      // Unless it's a Switch or a Recv, the node must produce a
      // tensor value at i-th output.
      if (!IsSwitch(node) && !IsRecv(node)) {
        s.Update(errors::Internal("Missing ", i, "-th output from ",
                                  SummarizeNodeDef(node->def())));
      }
    } else {
      Entry* out = &((*outputs)[i]);

      // Set the device context of the output entry.
      out->device_context = device_context;

      // Set the allocator attributes of the output entry.
      out->alloc_attr = ctx->output_alloc_attr(i);

      // Sanity check of output tensor types.
      DataType dtype = val->dtype();
      if (val.is_ref()) dtype = MakeRefType(dtype);
      if (dtype == item.output_type(i)) {
        if (stats && val.tensor->IsInitialized()) {
          nodestats::SetOutput(stats, i, val.tensor);
        }
        if (val.is_ref()) {
          out->has_value = true;
          out->ref = val.tensor;
          out->ref_mu = val.mutex_if_ref;
          if (log_memory_) {
            Tensor to_log;
            {
              // Dereference the tensor under the lock.
              mutex_lock l(*out->ref_mu);
              to_log = *out->ref;
            }
            LogMemory::RecordTensorOutput(ctx->op_kernel().name(),
                                          ctx->step_id(), i, to_log);
          }

          // Experimental: debugger (tfdb) access to intermediate node outputs.
          if (impl_->params_.node_outputs_cb != nullptr) {
            impl_->params_.node_outputs_cb(item.node->name(), i, out->ref, true,
                                           ctx);
          }
        } else {
          // NOTE that std::move is used here, so val.tensor goes to
          // uninitialized state (val.tensor->IsInitialized return false).
          DCHECK(!out->val_field_is_set);
          out->has_value = true;
          out->val_field_is_set = true;
          out->val.Init(std::move(*val.tensor));
          if (log_memory_) {
            LogMemory::RecordTensorOutput(ctx->op_kernel().name(),
                                          ctx->step_id(), i, *out->val);
          }

          // Experimental: debugger access to intermediate node outputs.
          if (impl_->params_.node_outputs_cb != nullptr) {
            impl_->params_.node_outputs_cb(item.node->name(), i, out->val.get(),
                                           false, ctx);
          }
        }
      } else {
        s.Update(errors::Internal("Output ", i, " of type ",
                                  DataTypeString(dtype),
                                  " does not match declared output type ",
                                  DataTypeString(item.output_type(i)),
                                  " for node ", SummarizeNodeDef(node->def())));
      }
    }
    if (!val.is_ref()) {
      // If OpKernelContext returns outputs via pass-by-value, we
      // don't need this trouble.
      delete val.tensor;
    }
  }
  return s;
}

void ExecutorState::PropagateOutputs(const TaggedNode& tagged_node,
                                     const EntryVector& outputs,
                                     TaggedNodeSeq* ready) {
  const Node* node = tagged_node.node;
  FrameState* input_frame = tagged_node.input_frame;
  int64 input_iter = tagged_node.input_iter;
  const bool is_dead = tagged_node.is_dead;

  // Propagates outputs along out edges, and puts newly ready nodes
  // into the ready queue.
  ready->clear();
  bool is_frame_done = false;
  FrameState* output_frame = input_frame;
  int64 output_iter = input_iter;

  if (IsEnter(node)) {
    bool is_constant;
    Status s = GetNodeAttr(node->def(), "is_constant", &is_constant);
    DCHECK(s.ok()) << s;
    FindOrCreateChildFrame(input_frame, input_iter, node, &output_frame);
    output_iter = 0;
    {
      mutex_lock l(output_frame->mu);
      if (is_constant) {
        // Propagate to all active iterations if this is a loop invariant.
        output_frame->AddLoopInv(node, outputs[0], ready);
      } else {
        output_frame->ActivateNodes(node, is_dead, output_iter, outputs, ready);
      }
      output_frame->num_pending_inputs--;
    }
    is_frame_done = input_frame->DecrementOutstandingOps(input_iter, ready);
  } else if (IsExit(node)) {
    if (is_dead) {
      mutex_lock l(input_frame->mu);
      // Stop and remember this node if it is a dead exit.
      if (input_iter == input_frame->iteration_count) {
        input_frame->dead_exits.push_back(node);
      }
      input_frame->GetIteration(input_iter)->outstanding_ops--;
      is_frame_done = input_frame->CleanupIterations(input_iter, ready);
    } else {
      output_frame = input_frame->parent_frame;
      output_iter = input_frame->parent_iter;
      {
        mutex_lock l(output_frame->mu);
        output_frame->ActivateNodes(node, is_dead, output_iter, outputs, ready);
      }
      is_frame_done = input_frame->DecrementOutstandingOps(input_iter, ready);
    }
  } else {
    mutex_lock l(input_frame->mu);
    if (IsNextIteration(node)) {
      if (is_dead) {
        // Stop the deadness propagation.
        output_frame = nullptr;
      } else {
        if (input_iter == input_frame->iteration_count &&
            input_frame->num_outstanding_iterations ==
                input_frame->max_parallel_iterations) {
          // Reached the maximum for parallel iterations.
          input_frame->next_iter_roots.push_back({node, outputs[0]});
          output_frame = nullptr;
        } else {
          // If this is a new iteration, start it.
          if (input_iter == input_frame->iteration_count) {
            input_frame->IncrementIteration(ready);
          }
          output_iter = input_iter + 1;
        }
      }
    }
    if (output_frame != nullptr) {
      // This is the case when node is not Enter, Exit, or NextIteration.
      DCHECK(input_frame == output_frame);
      output_frame->ActivateNodes(node, is_dead, output_iter, outputs, ready);
    }
    input_frame->GetIteration(input_iter)->outstanding_ops--;
    is_frame_done = input_frame->CleanupIterations(input_iter, ready);
  }

  // At this point, this node is completely done. We also know if the
  // completion of this node makes its frame completed.
  if (is_frame_done) {
    FrameState* parent_frame = input_frame->parent_frame;
    int64 parent_iter = input_frame->parent_iter;
    DeleteFrame(input_frame, ready);
    if (parent_frame != nullptr) {
      // The completion of frame may cause completions in its parent frame.
      // So clean things up recursively.
      CleanupFramesIterations(parent_frame, parent_iter, ready);
    }
  }
}

bool ExecutorState::NodeDone(const Status& s, const Node* node,
                             const TaggedNodeSeq& ready, NodeExecStats* stats,
                             TaggedNodeReadyQueue* inline_ready) {
  if (stats) {
    nodestats::SetAllEnd(stats);
    if (!SetTimelineLabel(node, stats)) {
      // Only record non-transfer nodes.
      stats_collector_->Save(impl_->params_.device->name(), stats);
    } else {
      delete stats;
    }
  }

  Rendezvous* captured_rendezvous = nullptr;  // Will be set on error.
  if (!s.ok()) {
    // Some error happened. This thread of computation is done.
    mutex_lock l(mu_);
    if (status_.ok()) {
      captured_rendezvous = rendezvous_;
      if (captured_rendezvous) captured_rendezvous->Ref();
      status_ = s;
    }
  }
  if (captured_rendezvous) {
    // If we captured the rendezvous_ pointer, we are in an error condition.
    // Use captured_rendezvous, in case "this" is deleted by another thread.
    TRACEPRINTF("StartAbort: %s", s.ToString().c_str());
    captured_rendezvous->StartAbort(s);
    captured_rendezvous->Unref();
  }

  bool completed = false;
  int ready_size = ready.size();
  if (ready_size == 0 || !s.ok()) {
    completed = (num_outstanding_ops_.fetch_sub(1) == 1);
  } else if (ready_size > 1) {
    num_outstanding_ops_.fetch_add(ready_size - 1, std::memory_order_relaxed);
  }

  // Schedule the ready nodes in 'ready'.
  if (s.ok()) {
    ScheduleReady(ready, inline_ready);
  }
  return completed;
}

void ExecutorState::ScheduleReady(const TaggedNodeSeq& ready,
                                  TaggedNodeReadyQueue* inline_ready) {
  if (ready.empty()) return;

  int64 scheduled_usec = 0;
  if (stats_collector_) {
    scheduled_usec = nodestats::NowInUsec();
  }
  if (inline_ready == nullptr) {
    // Schedule to run all the ready ops in thread pool.
    for (auto& tagged_node : ready) {
      runner_([=]() { Process(tagged_node, scheduled_usec); });
    }
    return;
  }
  const NodeItem* nodes = impl_->nodes_;
  const TaggedNode* curr_expensive_node = nullptr;
  for (auto& tagged_node : ready) {
    const NodeItem& item = nodes[tagged_node.node->id()];
    if (tagged_node.is_dead || !item.kernel_is_expensive) {
      // Inline this inexpensive node.
      inline_ready->push_back(tagged_node);
    } else {
      if (curr_expensive_node) {
        // Dispatch to another thread since there is plenty of work to
        // do for this thread.
        runner_(std::bind(&ExecutorState::Process, this, *curr_expensive_node,
                          scheduled_usec));
      }
      curr_expensive_node = &tagged_node;
    }
  }
  if (curr_expensive_node) {
    if (inline_ready->empty()) {
      // Tail recursion optimization
      inline_ready->push_back(*curr_expensive_node);
    } else {
      // There are inline nodes to run already. We dispatch this expensive
      // node to other thread.
      runner_(std::bind(&ExecutorState::Process, this, *curr_expensive_node,
                        scheduled_usec));
    }
  }
}

inline void ExecutorState::MaybeMarkCompleted(FrameState* frame, int64 iter,
                                              int64 node_id) {
  // TODO(misard) Replace with a finer-grain enabling flag once we
  // add better optional debugging support.
  if (vlog_ && VLOG_IS_ON(1)) {
    int pending_id = impl_->frame_local_ids_[node_id];
    mutex_lock l(frame->mu);
    frame->GetIteration(iter)->mark_completed(pending_id);
  }
}

const Tensor* ExecutorState::GetTensorValueForDump(const Entry& input) {
  if (!input.has_value) {
    return kEmptyTensor;
  } else if (input.ref == nullptr) {
    return input.val.get();
  } else {
    return input.ref;
  }
}

void ExecutorState::DumpPendingNodeState(
    const int node_id, const Entry* input_vector,
    const bool show_nodes_with_no_ready_inputs) {
  const NodeItem& node_item = impl_->nodes_[node_id];
  const Node& node = *node_item.node;
  const int input_base = node_item.input_start;
  if (!show_nodes_with_no_ready_inputs) {
    bool has_ready_input = false;
    for (int i = 0; i < node.num_inputs(); ++i) {
      const Entry& input = input_vector[input_base + i];
      const Tensor* tensor = GetTensorValueForDump(input);
      if (tensor->IsInitialized()) {
        has_ready_input = true;
        break;
      }
    }
    if (!has_ready_input) {
      return;
    }
  }
  LOG(WARNING) << "    Pending Node: " << node.DebugString();
  for (int i = 0; i < node.num_inputs(); ++i) {
    const Entry& input = input_vector[input_base + i];
    const Tensor* tensor = GetTensorValueForDump(input);
    if (tensor->IsInitialized()) {
      LOG(WARNING) << "      Input " << i << ": "
                   << strings::StrCat(
                          "Tensor<type: ", DataTypeString(tensor->dtype()),
                          " shape: ", tensor->shape().DebugString(), ">");
    } else {
      LOG(WARNING) << "      Input " << i << ": not present";
    }
  }
}

void ExecutorState::DumpActiveNodeState(const int node_id,
                                        const Entry* input_vector) {
  const NodeItem& node_item = impl_->nodes_[node_id];
  const Node& node = *node_item.node;
  LOG(WARNING) << "    Active Node: " << node.DebugString();
  const int input_base = node_item.input_start;
  for (int i = 0; i < node.num_inputs(); ++i) {
    const Entry& input = input_vector[input_base + i];
    const Tensor* tensor = GetTensorValueForDump(input);
    if (tensor->IsInitialized()) {
      LOG(WARNING) << "      Input " << i << ": "
                   << strings::StrCat(
                          "Tensor<type: ", DataTypeString(tensor->dtype()),
                          " shape: ", tensor->shape().DebugString(), ">");
    } else {
      LOG(WARNING) << "      Input " << i << ": not present";
    }
  }
}

void ExecutorState::DumpIterationState(const FrameState* frame,
                                       IterationState* iteration) {
  const std::vector<const Node*>* nodes = frame->nodes;
  // Dump any waiting nodes that are holding on to tensors.
  for (const Node* node : *nodes) {
    int node_id = node->id();
    int pending_id = impl_->frame_local_ids_[node_id];
    if (iteration->node_state(pending_id) == PendingCounts::PENDING_NOTREADY ||
        iteration->node_state(pending_id) == PendingCounts::PENDING_READY) {
      DumpPendingNodeState(node_id, iteration->input_tensors, false);
    }
  }
  // Then the active nodes.
  for (const Node* node : *nodes) {
    int node_id = node->id();
    int pending_id = impl_->frame_local_ids_[node_id];
    if (iteration->node_state(pending_id) == PendingCounts::STARTED) {
      DumpActiveNodeState(pending_id, iteration->input_tensors);
    }
  }
  // Show all input tensors in use.
  int total_input_tensors = frame->total_input_tensors;
  size_t total_bytes = 0;
  for (int i = 0; i < total_input_tensors; ++i) {
    const Entry& input = iteration->input_tensors[i];
    const Tensor* tensor = GetTensorValueForDump(input);
    if (tensor->IsInitialized()) {
      LOG(WARNING) << "    Input " << i << ": "
                   << strings::StrCat("Tensor<type: ",
                                      DataTypeString(tensor->dtype()),
                                      " shape: ", tensor->shape().DebugString(),
                                      ", bytes: ", tensor->TotalBytes(), ">");
      total_bytes += tensor->TotalBytes();
    }
  }
  LOG(WARNING) << "    Total bytes " << total_bytes;
}

void ExecutorState::DumpState() {
  mutex_lock l(mu_);
  if (!dumped_on_error_) {
    LOG(WARNING) << "Dumping state";
    for (auto& frame : outstanding_frames_) {
      LOG(WARNING) << frame.first;
      FrameState* frame_state = frame.second;
      mutex_lock frame_lock(frame_state->mu);
      for (IterationState* iteration : frame_state->iterations) {
        LOG(WARNING) << "  Iteration:";
        DumpIterationState(frame_state, iteration);
      }
    }
    dumped_on_error_ = true;
  }
}

void ExecutorState::Finish() {
  mu_.lock();
  auto status = status_;
  auto done_cb = std::move(done_cb_);
  auto runner = std::move(runner_);
  mu_.unlock();
  if (sync_on_finish_ && status.ok()) {
    // Block until the device has finished all queued operations. For
    // devices like GPUs that continue to execute Ops after their Compute
    // methods have completed, this ensures that control is not returned to
    // the user until the step (and its side-effects) has actually completed.
    status = impl_->params_.device->Sync();
  }
  delete this;
  CHECK(done_cb != nullptr);
  runner([=]() { done_cb(status); });
}

void ExecutorState::FindOrCreateChildFrame(FrameState* frame, int64 iter,
                                           const Node* node,
                                           FrameState** child) {
  // Get the child frame name.
  string enter_name;
  Status s = GetNodeAttr(node->def(), "frame_name", &enter_name);
  DCHECK(s.ok()) << s;
  const string child_name = MakeFrameName(frame, iter, enter_name);

  {
    mutex_lock executor_lock(mu_);
    auto it = outstanding_frames_.find(child_name);
    if (it != outstanding_frames_.end()) {
      *child = it->second;
      return;
    }
  }

  // Need to create a new frame instance.
  // Note that this new frame instance is created without any locks.
  if (vlog_) VLOG(2) << "Create frame: " << child_name;

  int parallel_iters;
  s = GetNodeAttr(node->def(), "parallel_iterations", &parallel_iters);
  DCHECK(s.ok()) << s;
  FrameState* temp = new FrameState(impl_, parallel_iters);
  temp->frame_name = child_name;
  temp->frame_id = Hash64(child_name);
  temp->parent_frame = frame;
  temp->parent_iter = iter;
  temp->InitializeFrameInfo(enter_name);

  // 'iterations' is a fixed-length circular buffer.
  temp->iterations.resize(temp->max_parallel_iterations + 1);
  // Initialize iteration 0.
  temp->iterations[0] =
      new IterationState(temp->pending_counts, temp->total_input_tensors);

  {
    mutex_lock executor_lock(mu_);
    auto it = outstanding_frames_.find(child_name);
    if (it != outstanding_frames_.end()) {
      *child = it->second;
    } else {
      mutex_lock frame_lock(frame->mu);
      frame->GetIteration(iter)->outstanding_frame_count++;
      outstanding_frames_[child_name] = temp;
      *child = temp;
      temp = nullptr;
    }
  }
  delete temp;  // Not used so delete it.
}

void ExecutorState::DeleteFrame(FrameState* frame, TaggedNodeSeq* ready) {
  // First, propagate dead_exits (if any) to the parent frame.
  FrameState* parent_frame = frame->parent_frame;
  int64 parent_iter = frame->parent_iter;
  if (parent_frame != nullptr) {
    const int* pending_ids = impl_->frame_local_ids_;
    mutex_lock paranet_frame_lock(parent_frame->mu);
    // Propagate all the dead exits to the parent frame.
    for (const Node* node : frame->dead_exits) {
      auto parent_iter_state = parent_frame->GetIteration(parent_iter);
      for (const Edge* e : node->out_edges()) {
        const Node* dst_node = e->dst();
        const int dst_pending_id = pending_ids[dst_node->id()];

        // TODO(yuanbyu): We don't need this if we require the subgraph
        // given to an executor not to contain a sink node.
        if (dst_node->IsSink()) continue;

        bool dst_dead = true;
        bool dst_ready = false;
        // We know this is a dead input to dst.
        if (IsMerge(dst_node)) {
          if (e->IsControlEdge()) {
            parent_iter_state->decrement_pending(dst_pending_id, 2);
            int count = parent_iter_state->pending(dst_pending_id);
            int dead_cnt = parent_iter_state->dead_count(dst_pending_id);
            dst_dead = (dead_cnt == dst_node->num_inputs());
            dst_ready = (count == 0) || ((count == 1) && dst_dead);
          } else {
            parent_iter_state->increment_dead_count(dst_pending_id);
            const int dead_cnt = parent_iter_state->dead_count(dst_pending_id);
            dst_dead = (dead_cnt == dst_node->num_inputs());
            dst_ready =
                (parent_iter_state->pending(dst_pending_id) == 1) && dst_dead;
          }
        } else {
          parent_iter_state->increment_dead_count(dst_pending_id);
          dst_ready =
              (parent_iter_state->decrement_pending(dst_pending_id, 1) == 0);
        }
        if (dst_ready) {
          ready->push_back(
              TaggedNode(dst_node, parent_frame, parent_iter, dst_dead));
          parent_iter_state->outstanding_ops++;
        }
      }
    }
  }

  // Delete the frame.
  const string& frame_name = frame->frame_name;
  if (vlog_) VLOG(2) << "Delete frame " << frame_name;
  {
    mutex_lock executor_lock(mu_);
    outstanding_frames_.erase(frame_name);
  }
  delete frame;
}

void ExecutorState::CleanupFramesIterations(FrameState* frame, int64 iter,
                                            TaggedNodeSeq* ready) {
  bool is_frame_done = false;
  {
    mutex_lock frame_lock(frame->mu);
    frame->GetIteration(iter)->outstanding_frame_count--;
    is_frame_done = frame->CleanupIterations(iter, ready);
  }
  if (is_frame_done) {
    FrameState* parent_frame = frame->parent_frame;
    int64 parent_iter = frame->parent_iter;
    DeleteFrame(frame, ready);
    if (parent_frame != nullptr) {
      // The completion of frame may cause completions in its parent frame.
      // So clean things up recursively.
      CleanupFramesIterations(parent_frame, parent_iter, ready);
    }
  }
}

void ExecutorState::FrameState::ActivateNodes(const Node* node,
                                              const bool is_dead, int64 iter,
                                              const EntryVector& outputs,
                                              TaggedNodeSeq* ready) {
  const NodeItem* nodes = executor->nodes_;
  const int* pending_ids = executor->frame_local_ids_;
  IterationState* iter_state = GetIteration(iter);
  for (const Edge* e : node->out_edges()) {
    const Node* dst_node = e->dst();
    const int dst_id = dst_node->id();
    const int dst_pending_id = pending_ids[dst_id];
    const int src_slot = e->src_output();

    // TODO(yuanbyu): We don't need this if we require the subgraph
    // given to an executor not to contain a sink node.
    if (dst_node->IsSink()) continue;

    bool dst_dead = false;
    bool dst_ready = false;
    // True iff this input for dst is needed. We only set this input for
    // dst if this flag is true. This is needed to make the thread safety
    // analysis happy.
    bool dst_need_input = !e->IsControlEdge();
    if (IsMerge(dst_node)) {
      // A merge node is ready if all control inputs have arrived and either
      // a) a live data input becomes available or b) all data inputs are dead.
      // For Merge, pending's LSB is set iff a live data input has arrived.
      if (e->IsControlEdge()) {
        iter_state->decrement_pending(dst_pending_id, 2);
        int count = iter_state->pending(dst_pending_id);
        int dead_cnt = iter_state->dead_count(dst_pending_id);
        dst_dead = (dead_cnt == dst_node->num_inputs());
        dst_ready = (count == 0) || ((count == 1) && dst_dead);
      } else {
        if (outputs[src_slot].has_value) {
          // This is a live data input.
          int count = iter_state->pending(dst_pending_id);
          iter_state->mark_live(dst_pending_id);
          // Only the first live edge sets the input and (potentially)
          // triggers execution. The low bit of count is set if and
          // only if no live input has been used yet (mark_live clears
          // it). The node should be started if and only if this is
          // the first live input and there are no pending control
          // edges, i.e. count == 1.
          dst_ready = (count == 1);
          dst_need_input = ((count & 0x1) == 1);
        } else {
          // This is a dead data input. Note that dst_node is dead if node is
          // a dead enter. We need this to handle properly a while loop on
          // the untaken branch of a conditional.
          // TODO(yuanbyu): This is a bit hacky, but a good solution for now.
          iter_state->increment_dead_count(dst_pending_id);
          const int dead_cnt = iter_state->dead_count(dst_pending_id);
          dst_dead = (dead_cnt == dst_node->num_inputs()) || IsEnter(node);
          dst_ready = (iter_state->pending(dst_pending_id) == 1) && dst_dead;
          dst_need_input = false;
        }
      }
    } else {
      // A non-merge node is ready if all its inputs are ready. We wait
      // for all inputs to come in even if we know the node is dead. This
      // ensures that all input tensors get cleaned up.
      if (is_dead || (!e->IsControlEdge() && !outputs[src_slot].has_value)) {
        iter_state->increment_dead_count(dst_pending_id);
      }
      dst_dead = iter_state->dead_count(dst_pending_id) > 0;
      dst_ready = (iter_state->decrement_pending(dst_pending_id, 1) == 0);
    }

    if (dst_need_input) {
      const NodeItem& dst_item = nodes[dst_id];
      const int dst_slot = e->dst_input();
      Entry* input_tensors = iter_state->input_tensors;
      int dst_loc = dst_item.input_start + dst_slot;
      input_tensors[dst_loc] = outputs[src_slot];
    }

    // Add dst to the ready queue if it's ready
    if (dst_ready) {
      dst_dead = dst_dead && !IsControlTrigger(dst_node);
      ready->push_back(TaggedNode(dst_node, this, iter, dst_dead));
      iter_state->outstanding_ops++;
    }
  }
}

void ExecutorState::FrameState::ActivateNexts(int64 iter,
                                              TaggedNodeSeq* ready) {
  // Propagate the deferred NextIteration nodes to the new iteration.
  for (auto& node_entry : next_iter_roots) {
    const Node* node = node_entry.first;
    const Entry& entry = node_entry.second;
    const bool is_dead = !entry.has_value;
    ActivateNodes(node, is_dead, iter, {entry}, ready);
  }
  next_iter_roots.clear();
}

void ExecutorState::FrameState::ActivateLoopInvs(int64 iter,
                                                 TaggedNodeSeq* ready) {
  // Propagate loop invariants to the new iteration.
  for (auto& node_entry : inv_values) {
    const Node* node = node_entry.first;
    const Entry& entry = node_entry.second;
    const bool is_dead = !entry.has_value;
    ActivateNodes(node, is_dead, iter, {entry}, ready);
  }
}

void ExecutorState::FrameState::AddLoopInv(const Node* node, const Entry& entry,
                                           TaggedNodeSeq* ready) {
  // Store this value.
  inv_values.push_back({node, entry});

  // Make this value available to all iterations.
  bool is_dead = !entry.has_value;
  for (int i = 0; i <= iteration_count; ++i) {
    ActivateNodes(node, is_dead, i, {entry}, ready);
  }
}

bool ExecutorState::FrameState::IsIterationDone(int64 iter) {
  IterationState* iter_state = GetIteration(iter);
  if (iter_state->outstanding_ops == 0 &&
      iter_state->outstanding_frame_count == 0) {
    if (iter == 0) {
      // The enclosing frame has no pending input.
      return num_pending_inputs == 0;
    } else {
      // The preceding iteration is deleted (and therefore done).
      return (GetIteration(iter - 1) == nullptr);
    }
  }
  return false;
}

void ExecutorState::FrameState::IncrementIteration(TaggedNodeSeq* ready) {
  iteration_count++;
  int64 next_iter = iteration_count;

  // Initialize the next iteration.
  IterationState* iter_state =
      new IterationState(pending_counts, total_input_tensors);
  SetIteration(next_iter, iter_state);
  num_outstanding_iterations++;
  dead_exits.clear();

  // Activate the successors of the deferred roots in the new iteration.
  ActivateNexts(next_iter, ready);

  // Activate the loop invariants in the new iteration.
  ActivateLoopInvs(next_iter, ready);
}

bool ExecutorState::FrameState::CleanupIterations(int64 iter,
                                                  TaggedNodeSeq* ready) {
  int64 curr_iter = iter;
  while (curr_iter <= iteration_count && IsIterationDone(curr_iter)) {
    // Delete the iteration curr_iter.
    delete GetIteration(curr_iter);
    SetIteration(curr_iter, nullptr);
    --num_outstanding_iterations;
    ++curr_iter;

    // When one iteration is completed, we check for deferred iteration,
    // and start it if there is one.
    if (!next_iter_roots.empty()) {
      IncrementIteration(ready);
    }
  }
  return IsFrameDone();
}

void ExecutorImpl::RunAsync(const Args& args, DoneCallback done) {
  (new ExecutorState(args, this))->RunAsync(done);
}

}  // end namespace

Status NewLocalExecutor(const LocalExecutorParams& params, const Graph* graph,
                        Executor** executor) {
  ExecutorImpl* impl = new ExecutorImpl(params, graph);
  Status s = impl->Initialize();
  if (s.ok()) {
    *executor = impl;
  } else {
    delete impl;
  }
  return s;
}

Status CreateNonCachedKernel(Device* device, FunctionLibraryRuntime* flib,
                             const NodeDef& ndef, int graph_def_version,
                             OpKernel** kernel) {
  auto device_type = DeviceType(device->attributes().device_type());
  auto allocator = device->GetAllocator(AllocatorAttributes());
  return CreateOpKernel(device_type, device, allocator, flib, ndef,
                        graph_def_version, kernel);
}

void DeleteNonCachedKernel(OpKernel* kernel) { delete kernel; }

}  // end namespace tensorflow
