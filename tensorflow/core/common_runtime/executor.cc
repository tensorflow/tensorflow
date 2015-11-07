#include "tensorflow/core/common_runtime/executor.h"

#include <atomic>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <deque>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/control_flow.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_segment.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/edgeset.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/public/tensor.h"
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

void SetOutput(NodeExecStats* nt, int slot, AllocationType allocation_type,
               const Tensor* v) {
  DCHECK(v);
  NodeOutput* no = nt->add_output();
  no->set_slot(slot);
  no->set_allocation_type(allocation_type);
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
}  // namespace nodestats

struct NodeItem {
  // A graph node.
  const Node* node = nullptr;

  // The kernel for this node.
  OpKernel* kernel = nullptr;

  // ExecutorImpl::tensors_[input_start] is the 1st positional input
  // for this node.
  int input_start = 0;
};

// Map from std::pair<node_id, output_index> to attributes.
struct pairhash {
 public:
  template <typename T, typename U>
  std::size_t operator()(const std::pair<T, U>& x) const {
    return std::hash<T>()(x.first) ^ std::hash<U>()(x.second);
  }
};
typedef std::unordered_map<std::pair<int, int>, AllocatorAttributes, pairhash>
    DevAttrMap;

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
    for (NodeItem& item : nodes_) {
      params_.delete_kernel(item.kernel);
    }
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
  friend class SimpleExecutorState;

  // Owned.
  LocalExecutorParams params_;
  const Graph* graph_;
  std::vector<NodeItem> nodes_;  // nodes_.size == graph_.num_node_ids().
  int total_tensors_ = 0;        // total_tensors_ = sum(nodes_[*].num_inputs())

  // The number of inputs for each frame in this graph. This is static
  // information of the graph.
  std::unordered_map<string, int> frame_input_count_;

  DevAttrMap alloc_attr_;

  TF_DISALLOW_COPY_AND_ASSIGN(ExecutorImpl);
};

Status ExecutorImpl::Initialize() {
  const int num_nodes = graph_->num_node_ids();
  nodes_.resize(num_nodes);

  Status s;
  total_tensors_ = 0;

  // Preprocess every node in the graph to create an instance of op
  // kernel for each node;
  for (const Node* n : graph_->nodes()) {
    const int id = n->id();
    NodeItem* item = &nodes_[id];
    item->node = n;
    item->input_start = total_tensors_;
    total_tensors_ += n->num_inputs();
    s = params_.create_kernel(n->def(), &item->kernel);
    if (!s.ok()) {
      s = AttachDef(s, n->def());
      LOG(ERROR) << "Executor failed to create kernel. " << s;
      break;
    }
    CHECK(item->kernel);

    // Initialize static information about the frames in the graph.
    if (IsEnter(n)) {
      string frame_name;
      s = GetNodeAttr(n->def(), "frame_name", &frame_name);
      if (!s.ok()) return s;
      ++frame_input_count_[frame_name];
    }
  }
  if (params_.has_control_flow) {
    VLOG(2) << "Graph has control flow.";
  }
  if (!s.ok()) return s;
  return SetAllocAttrs();
}

Status ExecutorImpl::SetAllocAttrs() {
  Status s;
  Device* device = params_.device;
  DeviceNameUtils::ParsedName local_dev_name = device->parsed_name();

  for (const Node* n : graph_->nodes()) {
    // Examine the out edges of each node looking for special use
    // cases that may affect memory allocation attributes.
    for (auto e : n->out_edges()) {
      AllocatorAttributes attr;
      s = InferAllocAttr(n, e->dst(), local_dev_name, &attr);
      if (!s.ok()) return s;
      if (attr.value != 0) {
        VLOG(2) << "node " << n->name() << " gets attr " << attr.value
                << " for output " << e->src_output();
        alloc_attr_[std::make_pair(n->id(), e->src_output())].Merge(attr);
      } else {
        VLOG(2) << "default output attr for node " << n->name() << " output "
                << e->src_output();
      }
    }
  }
  return s;
}

Status ExecutorImpl::InferAllocAttr(
    const Node* n, const Node* dst,
    const DeviceNameUtils::ParsedName& local_dev_name,
    AllocatorAttributes* attr) {
  Status s;
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
    } else if (local_dev_name.type == "CPU" && parsed_dst_name.type == "GPU") {
      // Value is going to be the source of a local DMA from CPU to GPU.
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

// The state associated with one invokation of ExecutorImpl::Run.
// ExecutorState dispatches nodes when they become ready and keeps
// track of how many predecessors of a node have not done (pending_).
class ExecutorState {
 public:
  ExecutorState(const Executor::Args& args, ExecutorImpl* impl);
  ~ExecutorState();

  void RunAsync(Executor::DoneCallback done);

 private:
  typedef ExecutorState ME;

  // Either a tensor pointer (pass-by-reference) or a tensor (pass-by-value).
  // TODO(yuanbyu): A better way to do "has_value"?
  struct Entry {
    Tensor val = *kEmptyTensor;  // A tensor value.
    Tensor* ref = nullptr;       // A tensor reference.
    mutex* ref_mu = nullptr;     // mutex for *ref if ref is not nullptr.
    bool has_value = false;      // Whether the value exists

    // Every entry carries an optional DeviceContext containing
    // Device-specific information about how the Tensor was produced.
    DeviceContext* device_context = nullptr;

    // The attributes of the allocator that creates the tensor.
    AllocatorAttributes alloc_attr;
  };

  // Contains a map from node id to the DeviceContext object that was
  // assigned by the device at the beginning of a step.
  DeviceContextMap device_context_map_;

  struct IterationState {
    // The state of an iteration.

    // The pending count for each graph node. One copy per iteration.
    // Iteration i can be garbage collected when it is done.
    // TODO(yuanbyu): This vector currently has size of the number of nodes
    // in this partition. This is not efficient if the subgraph for the frame
    // is only a small subset of the partition. We should make the vector
    // size to be only the size of the frame subgraph.
    std::vector<int>* pending_count;

    // The dead input count for each graph node. One copy per iteration.
    std::vector<int>* dead_count;

    // One copy per iteration. For iteration k, i-th node's j-th input is in
    // input_tensors[k][impl_->nodes[i].input_start + j]. An entry is either
    // a tensor pointer (pass-by-reference) or a tensor (pass-by-value).
    //
    // NOTE: No need to protect input_tensors[i] by any locks because it
    // is resized once. Each element of tensors_ is written once by the
    // source node of an edge and is cleared by the destination of the same
    // edge. The latter node is never run concurrently with the former node.
    std::vector<Entry>* input_tensors;

    // The number of outstanding ops for each iteration.
    int outstanding_ops;

    // The number of outstanding frames for each iteration.
    int outstanding_frame_count;

    ~IterationState() {
      delete pending_count;
      delete dead_count;
      delete input_tensors;
    }
  };

  struct FrameState {
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

    // The highest iteration number we have reached so far in this frame.
    int64 iteration_count = 0;

    // The number of inputs this frame is still waiting.
    int num_pending_inputs = 0;

    // The number of outstanding iterations.
    int num_outstanding_iterations = 0;

    // The maximum allowed number of parallel iterations.
    int max_parallel_iterations = 1;

    // The iteration states of this frame.
    std::vector<IterationState*> iterations;

    // The NextIteration nodes to enter a new iteration. If the number of
    // outstanding iterations reaches the limit, we will defer the start of
    // the next iteration until the number of outstanding iterations falls
    // below the limit.
    std::vector<std::pair<const Node*, Entry>> next_iter_roots;

    // The values of the loop invariants for this loop. They are added into
    // this list as they "enter" the frame. When a loop invariant enters,
    // we make it available to all active iterations. When the frame starts
    // a new iteration, we make all the current loop invariants available
    // to the new iteration.
    std::vector<std::pair<const Node*, Entry>> inv_values;

    // The list of dead exit nodes for the current highest iteration. We
    // will only "execute" the dead exits of the final iteration.
    std::vector<const Node*> dead_exits;

    IterationState* GetIteration(int64 iter) {
      int index = iter % iterations.size();
      return iterations[index];
    }

    void SetIteration(int64 iter, IterationState* state) {
      int index = iter % iterations.size();
      iterations[index] = state;
    }

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

  typedef gtl::InlinedVector<TaggedNode, 8> TaggedNodeSeq;
  typedef gtl::InlinedVector<Entry, 4> EntryVector;

  // Not owned.
  Rendezvous* rendezvous_;
  StepStatsCollector* stats_collector_;
  // QUESTION: Make it a checkpoint::TensorSliceReaderCacheWrapper instead of a
  // pointer?  (avoids having to delete).
  checkpoint::TensorSliceReaderCacheWrapper* slice_reader_cache_;
  FunctionCallFrame* call_frame_;
  const ExecutorImpl* impl_;
  CancellationManager* cancellation_manager_;
  Executor::Args::Runner runner_;

  // Owned.

  // Step-local resource manager.
  ResourceMgr step_resource_manager_;

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
  std::unordered_map<string, FrameState*> outstanding_frames_ GUARDED_BY(mu_);

  // The unique name of a frame.
  inline string MakeFrameName(FrameState* frame, int64 iter_id, string name) {
    return strings::StrCat(frame->frame_name, ";", iter_id, ";", name);
  }

  // Initialize the pending count for a graph.
  static void InitializePending(const Graph* graph, std::vector<int>* pending);

  // Find an existing or create a new child frame in the frame 'frame' at
  // iteration 'iter'.
  void FindOrCreateChildFrame(FrameState* frame, int64 iter, const Node* node,
                              FrameState** child) EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Increments the iteration id. If this is a new iteration, initialize it.
  void IncrementIteration(FrameState* frame, TaggedNodeSeq* ready)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Returns true if the computation in the frame is completed.
  bool IsFrameDone(FrameState* frame) EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Returns true if the iteration of the frame is completed.
  bool IsIterationDone(FrameState* frame, int64 iter)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Get the output frame/iter of a node. Create new frame/iteration if
  // needed. If there are dead roots for the new iteration, we need to
  // "execute" them so ad them to the ready queue. Returns true if
  // we need to check for the completion of output frame/iter.
  bool SetOutputFrameIter(const TaggedNode& tagged_node,
                          const EntryVector& outputs, FrameState** frame,
                          int64* iter, TaggedNodeSeq* ready)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Cleanup frames and iterations
  void CleanupFramesIterations(FrameState* frame, int64 iter,
                               TaggedNodeSeq* ready)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Activate all the deferred NextIteration nodes in a new iteration.
  void ActivateNexts(FrameState* frame, int64 iter, TaggedNodeSeq* ready)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Activate all the current loop invariants in a new iteration.
  void ActivateLoopInvs(FrameState* frame, int64 iter, TaggedNodeSeq* ready)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Add a new loop invariant and make it available to all active iterations.
  void AddLoopInv(FrameState* frame, const Node* node, const Entry& value,
                  TaggedNodeSeq* ready) EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Activate the successors of a node.
  void ActivateNode(const Node* node, const bool is_dead, FrameState* frame,
                    int64 iter, const EntryVector& outputs,
                    TaggedNodeSeq* ready) EXCLUSIVE_LOCKS_REQUIRED(mu_);

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
                NodeExecStats* stats, std::deque<TaggedNode>* inline_ready);

  // Call Process() on all nodes in 'inline_ready'.
  void ProcessInline(const std::deque<TaggedNode>& inline_ready);

  // Schedule all the expensive nodes in 'ready', and put all the inexpensive
  // nodes in 'ready' into 'inline_ready'.
  void ScheduleReady(const TaggedNodeSeq& ready,
                     std::deque<TaggedNode>* inline_ready);

  // One thread of control finishes.
  void Finish();
};

ExecutorState::ExecutorState(const Executor::Args& args, ExecutorImpl* impl)
    : rendezvous_(args.rendezvous),
      stats_collector_(args.stats_collector),
      slice_reader_cache_(new checkpoint::TensorSliceReaderCacheWrapper),
      call_frame_(args.call_frame),
      impl_(impl),
      cancellation_manager_(args.cancellation_manager),
      runner_(args.runner),
      num_outstanding_ops_(0) {
  // We start the entire execution in iteration 0 of the root frame
  // so let us create the root frame and the state for iteration 0.
  // Initialize the frame.
  root_frame_ = new FrameState;
  root_frame_->frame_name = "_root";  // assume to be unique
  root_frame_->frame_id = 0;          // must be 0
  root_frame_->num_pending_inputs = 0;
  root_frame_->num_outstanding_iterations = 1;
  root_frame_->max_parallel_iterations = 1;  // enough for root frame
  root_frame_->iterations.resize(root_frame_->max_parallel_iterations);

  VLOG(2) << "Create frame: " << root_frame_->frame_name;

  // Initialize the iteration.
  IterationState* iter_state = new IterationState;
  root_frame_->iterations[0] = iter_state;
  iter_state->outstanding_ops = 0;
  iter_state->outstanding_frame_count = 0;
  iter_state->pending_count = new std::vector<int>;
  iter_state->dead_count = new std::vector<int>(impl->graph_->num_node_ids());
  iter_state->input_tensors = new std::vector<Entry>(impl_->total_tensors_);

  // Initialize the executor state.
  outstanding_frames_.insert({root_frame_->frame_name, root_frame_});
}

ExecutorState::~ExecutorState() {
  for (auto name_frame : outstanding_frames_) {
    delete name_frame.second;
  }

  for (auto it : device_context_map_) {
    it.second->Unref();
  }

  delete slice_reader_cache_;
}

void ExecutorState::InitializePending(const Graph* graph,
                                      std::vector<int>* pending) {
  pending->resize(graph->num_node_ids());
  for (const Node* n : graph->nodes()) {
    const int id = n->id();
    const int num_in_edges = n->in_edges().size();
    if (IsMerge(n)) {
      // merge waits all control inputs so we initialize the pending
      // count to be the number of control edges.
      int32 num_control_edges = 0;
      for (const Edge* edge : n->in_edges()) {
        if (edge->IsControlEdge()) {
          num_control_edges++;
        }
      }
      // Use bit 0 to indicate if there is a ready live data input.
      (*pending)[id] = num_control_edges << 1;
    } else {
      (*pending)[id] = num_in_edges;
    }
  }
}

void ExecutorState::RunAsync(Executor::DoneCallback done) {
  const Graph* graph = impl_->graph_;
  TaggedNodeSeq ready;

  {
    // Initialize the executor state. We grab the mutex here just to
    // keep the thread safety analysis happy.
    mutex_lock l(mu_);
    std::vector<int>* pending = root_frame_->iterations[0]->pending_count;
    InitializePending(graph, pending);
  }

  // Ask the device to fill in the device context map.
  Device* device = impl_->params_.device;
  device->FillContextMap(graph, &device_context_map_);

  // Initialize the ready queue.
  for (const Node* n : graph->nodes()) {
    const int num_in_edges = n->in_edges().size();
    if (num_in_edges == 0) {
      ready.push_back(TaggedNode{n, root_frame_, 0, false});
    }
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

namespace {

// This function is provided for use by OpKernelContext when allocating
// the index'th output of node.  It provides access to the
// AllocatorAttributes computed during initialization to determine in
// which memory region the tensor should be allocated.
AllocatorAttributes OutputAttributes(const DevAttrMap* attr_map,
                                     const Node* node,
                                     const OpKernel* op_kernel, int index) {
  DCHECK_GE(index, 0);

  AllocatorAttributes attr;
  int nid = node->id();
  const auto& iter = attr_map->find(std::make_pair(nid, index));
  if (iter != attr_map->end()) {
    attr = iter->second;
    VLOG(2) << "nondefault attr " << attr.value << " for node " << node->name()
            << " output " << index;
  } else {
    VLOG(2) << "default attr for node " << node->name() << " output " << index;
  }

  DCHECK_LT(index, op_kernel->output_memory_types().size());
  bool on_host = op_kernel->output_memory_types()[index] == HOST_MEMORY;
  attr.set_on_host(on_host);
  return attr;
}

// Helpers to make a copy of 'p' and makes a copy of the input type
// vector and the device context vector.
//
// NOTE: We need to make a copy of p.input for asynchronous kernel
// because OpKernelContext methods like input_type(i) needs the param
// points to valid input type vector. It's not an issue for sync
// kernels because the type vector is kept on the stack.
OpKernelContext::Params* CopyParams(const OpKernelContext::Params& p) {
  OpKernelContext::Params* ret = new OpKernelContext::Params;
  *ret = p;
  ret->inputs = new TensorValueVec(*p.inputs);
  ret->input_device_contexts = new DeviceContextVec(*p.input_device_contexts);
  ret->input_alloc_attrs = new AllocatorAttributeVec(*p.input_alloc_attrs);
  return ret;
}

// Helpers to delete 'p' and copies made by CopyParams.
void DeleteParams(OpKernelContext::Params* p) {
  delete p->inputs;
  delete p->input_device_contexts;
  delete p->input_alloc_attrs;
  delete p;
}

}  // namespace

void ExecutorState::Process(TaggedNode tagged_node, int64 scheduled_usec) {
  const std::vector<NodeItem>& nodes = impl_->nodes_;
  TaggedNodeSeq ready;
  std::deque<TaggedNode> inline_ready;

  // Parameters passed to OpKernel::Compute.
  TensorValueVec inputs;
  DeviceContextVec input_device_contexts;
  AllocatorAttributeVec input_alloc_attrs;

  OpKernelContext::Params params;
  Device* device = impl_->params_.device;
  params.device = device;
  // track allocations if and only if we are collecting statistics
  params.track_allocations = (stats_collector_ != nullptr);
  params.rendezvous = rendezvous_;
  params.cancellation_manager = cancellation_manager_;
  params.call_frame = call_frame_;
  params.function_library = impl_->params_.function_library;
  params.resource_manager = device->resource_manager();
  params.step_resource_manager = &step_resource_manager_;
  params.slice_reader_cache = slice_reader_cache_;
  params.inputs = &inputs;
  params.input_device_contexts = &input_device_contexts;
  params.input_alloc_attrs = &input_alloc_attrs;

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

    // Set the device_context for this node id, if it exists.
    auto dc_it = device_context_map_.find(id);
    if (dc_it != device_context_map_.end()) {
      params.op_device_context = dc_it->second;
    }

    if (stats_collector_) {
      stats = new NodeExecStats;
      stats->set_node_name(node->name());
      nodestats::SetScheduled(stats, scheduled_usec);
      nodestats::SetAllStart(stats);
    }

    VLOG(1) << "Process node: " << id << " " << SummarizeNodeDef(node->def());

    std::vector<Entry>* input_tensors;
    {
      // Need the lock because the iterations vector could be resized by
      // another thread.
      mutex_lock l(mu_);
      input_tensors = input_frame->GetIteration(input_iter)->input_tensors;
    }
    Entry* first_input = input_tensors->data() + item.input_start;
    outputs.clear();
    outputs.resize(node->num_outputs());

    // Only execute this node if it is not dead or it is a send/recv
    // transfer node. For transfer nodes, we need to propagate the "dead"
    // bit even when the node is dead.
    AsyncOpKernel* async = nullptr;
    if (!tagged_node.is_dead || IsTransferNode(node)) {
      // Prepares inputs.
      bool is_input_dead = false;
      s = PrepareInputs(item, first_input, &inputs, &input_device_contexts,
                        &input_alloc_attrs, &is_input_dead);
      if (!s.ok()) {
        // Continue to process the nodes in 'inline_ready'.
        completed = NodeDone(s, item.node, ready, stats, &inline_ready);
        continue;
      }

      // Set up compute params.
      OpKernel* op_kernel = item.kernel;
      params.op_kernel = op_kernel;
      params.frame_iter = FrameAndIter(input_frame->frame_id, input_iter);
      params.is_input_dead = is_input_dead;
      params.output_alloc_attr = [this, node, op_kernel](int index) {
        return OutputAttributes(&impl_->alloc_attr_, node, op_kernel, index);
      };

      async = op_kernel->AsAsync();
      if (async) {
        // Asynchronous computes.
        auto pcopy = CopyParams(params);
        auto ctx = new OpKernelContext(*pcopy);
        auto done = [this, tagged_node, item, first_input, ctx, stats,
                     pcopy]() {
          VLOG(2) << this << " Async kernel done: "
                  << SummarizeNodeDef(item.node->def());
          if (stats_collector_) nodestats::SetOpEnd(stats);
          EntryVector outputs;
          Status s = ProcessOutputs(item, ctx, &outputs, stats);
          if (stats_collector_) nodestats::SetMemory(stats, ctx);
          // Clears inputs.
          int num_inputs = tagged_node.node->num_inputs();
          for (int i = 0; i < num_inputs; ++i) {
            (first_input + i)->val = *kEmptyTensor;
          }
          TaggedNodeSeq ready;
          if (s.ok()) {
            PropagateOutputs(tagged_node, outputs, &ready);
          }
          // Schedule to run all the ready ops in thread pool.
          bool completed = NodeDone(s, item.node, ready, stats, nullptr);
          delete ctx;
          DeleteParams(pcopy);
          if (completed) Finish();
        };
        if (stats_collector_) nodestats::SetOpStart(stats);
        device->ComputeAsync(async, ctx, done);
      } else {
        // Synchronous computes.
        OpKernelContext ctx(params);
        if (stats_collector_) nodestats::SetOpStart(stats);
        device->Compute(CHECK_NOTNULL(op_kernel), &ctx);
        if (stats_collector_) nodestats::SetOpEnd(stats);

        // Processes outputs.
        s = ProcessOutputs(item, &ctx, &outputs, stats);
        if (stats_collector_) nodestats::SetMemory(stats, &ctx);
      }
    }

    if (!async) {
      // Clears inputs.
      int num_inputs = node->num_inputs();
      for (int i = 0; i < num_inputs; ++i) {
        (first_input + i)->val = *kEmptyTensor;
      }
      // Propagates outputs.
      if (s.ok()) {
        PropagateOutputs(tagged_node, outputs, &ready);
      }
      if (stats_collector_) {
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
  inputs->resize(node->num_inputs());
  input_device_contexts->clear();
  input_device_contexts->resize(node->num_inputs());
  input_alloc_attrs->clear();
  input_alloc_attrs->resize(node->num_inputs());

  *is_input_dead = false;

  bool is_merge = IsMerge(node);
  for (int i = 0; i < node->num_inputs(); ++i) {
    const bool expect_ref = IsRefType(node->input_type(i));
    Entry* entry = first_input + i;
    (*input_device_contexts)[i] = entry->device_context;
    (*input_alloc_attrs)[i] = entry->alloc_attr;

    // i-th input.
    TensorValue* inp = &(*inputs)[i];

    // Only merge and transfer nodes can have no-value inputs.
    if (!entry->has_value) {
      if (!is_merge) {
        DCHECK(IsTransferNode(node));
        inp->tensor = &entry->val;
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
      inp->tensor = &entry->val;
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
          entry->val = *entry->ref;
        }
        inp->tensor = &entry->val;
      }
    }
  }
  return Status::OK();
}

Status ExecutorState::ProcessOutputs(const NodeItem& item, OpKernelContext* ctx,
                                     EntryVector* outputs,
                                     NodeExecStats* stats) {
  const Node* node = item.node;
  outputs->clear();
  outputs->resize(node->num_outputs());

  Status s = ctx->status();
  if (!s.ok()) {
    s = AttachDef(s, item.kernel->def());
    LOG(WARNING) << this << " Compute status: " << s;
    return s;
  }

  // Get the device_context for this node id, if it exists.
  DeviceContext* device_context = nullptr;
  auto dc_it = device_context_map_.find(node->id());
  if (dc_it != device_context_map_.end()) {
    device_context = dc_it->second;
  }

  for (int i = 0; i < node->num_outputs(); ++i) {
    TensorValue val = ctx->release_output(i);
    // Only Switch and Recv nodes can generate new dead outputs
    if (*ctx->is_output_dead() || val.tensor == nullptr) {
      DCHECK(IsSwitch(node) || IsRecv(node));
    } else {
      Entry* out = &((*outputs)[i]);
      out->has_value = true;

      // Set the device context of the output entry.
      out->device_context = device_context;

      // Set the allocator attributes of the output entry.
      out->alloc_attr = ctx->output_alloc_attr(i);

      // Sanity check of output tensor types.
      DataType dtype = val->dtype();
      if (val.is_ref()) dtype = MakeRefType(dtype);
      if (dtype == node->output_type(i)) {
        if (val.is_ref()) {
          out->ref = val.tensor;
          out->ref_mu = val.mutex_if_ref;
        } else {
          out->val = *val.tensor;
        }
        if (stats_collector_ && val.tensor->IsInitialized()) {
          nodestats::SetOutput(stats, i, ctx->output_allocation_type(i),
                               val.tensor);
        }
      } else {
        s.Update(errors::Internal("Output ", i, " of type ",
                                  DataTypeString(dtype),
                                  " does not match declared output type ",
                                  DataTypeString(node->output_type(i)),
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
  FrameState* input_frame = tagged_node.input_frame;
  int64 input_iter = tagged_node.input_iter;

  // Propagates outputs along out edges, and puts newly ready nodes
  // into the ready queue.
  ready->clear();

  {
    FrameState* output_frame = input_frame;
    int64 output_iter = input_iter;

    mutex_lock l(mu_);
    // Sets the output_frame and output_iter of node.
    bool maybe_completed = SetOutputFrameIter(
        tagged_node, outputs, &output_frame, &output_iter, ready);
    if (output_frame != nullptr) {
      // Continue to process the out nodes:
      ActivateNode(tagged_node.node, tagged_node.is_dead, output_frame,
                   output_iter, outputs, ready);
    }

    // At this point, this node is completely done.
    input_frame->GetIteration(input_iter)->outstanding_ops--;
    CleanupFramesIterations(input_frame, input_iter, ready);

    // The execution of a node such as Enter may cause the completion of
    // output_frame:output_iter, so perform cleanup if output_frame:output_iter
    // is indeed completed.
    if (maybe_completed) {
      CleanupFramesIterations(output_frame, output_iter, ready);
    }
  }
}

void ExecutorState::ActivateNode(const Node* node, const bool is_dead,
                                 FrameState* output_frame, int64 output_iter,
                                 const EntryVector& outputs,
                                 TaggedNodeSeq* ready) {
  const std::vector<NodeItem>& nodes = impl_->nodes_;
  IterationState* output_iter_state = output_frame->GetIteration(output_iter);
  std::vector<int>* pending = output_iter_state->pending_count;
  std::vector<int>* dead_count = output_iter_state->dead_count;
  for (const Edge* e : node->out_edges()) {
    const Node* dst_node = e->dst();
    const int dst_id = dst_node->id();
    const int src_slot = e->src_output();

    bool dst_dead = false;
    bool dst_ready = false;
    bool dst_need_input = !e->IsControlEdge();
    if (IsMerge(dst_node)) {
      // A merge node is ready if a) all control edges are enabled and a
      // live data input becomes available, or b) all control edges are
      // enabled and all data inputs are dead.
      if (e->IsControlEdge()) {
        (*pending)[dst_id] -= 2;
        int count = (*pending)[dst_id];
        dst_dead = ((*dead_count)[dst_id] == dst_node->num_inputs());
        dst_ready = (count == 1) || ((count == 0) && dst_dead);
      } else {
        if (outputs[src_slot].has_value) {
          // This is a live data input.
          int count = (*pending)[dst_id];
          (*pending)[dst_id] |= 0x1;
          dst_ready = (count == 0);
        } else {
          // This is a dead data input.
          ++(*dead_count)[dst_id];
          dst_dead = ((*dead_count)[dst_id] == dst_node->num_inputs());
          dst_ready = ((*pending)[dst_id] == 0) && dst_dead;
        }
        // This input for dst is not needed if !dst_ready. We suppress the
        // propagation to make the thread safety analysis happy.
        dst_need_input = dst_ready;
      }
    } else {
      // A non-merge node is ready if all its inputs are ready. We wait
      // for all inputs to come in even if we know the node is dead. This
      // ensures that all input tensors get cleaned up.
      if (is_dead || (!e->IsControlEdge() && !outputs[src_slot].has_value)) {
        ++(*dead_count)[dst_id];
      }
      dst_dead = (*dead_count)[dst_id] > 0;
      dst_ready = (--(*pending)[dst_id] == 0);
    }

    if (dst_need_input) {
      const NodeItem& dst_item = nodes[dst_id];
      const int dst_slot = e->dst_input();
      std::vector<Entry>* input_tensors = output_iter_state->input_tensors;
      int dst_loc = dst_item.input_start + dst_slot;
      (*input_tensors)[dst_loc] = outputs[src_slot];
    }

    // Add dst to the ready queue if it's ready
    if (dst_ready) {
      dst_dead = dst_dead && !IsControlTrigger(dst_node);
      ready->push_back(
          TaggedNode(dst_node, output_frame, output_iter, dst_dead));
      output_iter_state->outstanding_ops++;
    }
  }
}

void ExecutorState::ActivateNexts(FrameState* frame, int64 iter,
                                  TaggedNodeSeq* ready) {
  // Propagate the deferred NextIteration nodes to the new iteration.
  for (auto& node_entry : frame->next_iter_roots) {
    const Node* node = node_entry.first;
    const Entry& entry = node_entry.second;
    const bool is_dead = !entry.has_value;
    ActivateNode(node, is_dead, frame, iter, {entry}, ready);
  }
  frame->next_iter_roots.clear();
}

void ExecutorState::ActivateLoopInvs(FrameState* frame, int64 iter,
                                     TaggedNodeSeq* ready) {
  // Propagate loop invariants to the new iteration.
  for (auto& node_entry : frame->inv_values) {
    const Node* node = node_entry.first;
    const Entry& entry = node_entry.second;
    const bool is_dead = !entry.has_value;
    ActivateNode(node, is_dead, frame, iter, {entry}, ready);
  }
}

void ExecutorState::AddLoopInv(FrameState* frame, const Node* node,
                               const Entry& entry, TaggedNodeSeq* ready) {
  // Store this value.
  frame->inv_values.push_back({node, entry});

  // Make this value available to all iterations.
  bool is_dead = !entry.has_value;
  for (int i = 1; i <= frame->iteration_count; ++i) {
    ActivateNode(node, is_dead, frame, i, {entry}, ready);
  }
}

bool ExecutorState::NodeDone(const Status& s, const Node* node,
                             const TaggedNodeSeq& ready, NodeExecStats* stats,
                             std::deque<TaggedNode>* inline_ready) {
  if (stats_collector_) {
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

void ExecutorState::ProcessInline(const std::deque<TaggedNode>& inline_ready) {
  if (inline_ready.empty()) return;
  int64 scheduled_usec = 0;
  if (stats_collector_) {
    scheduled_usec = nodestats::NowInUsec();
  }
  for (auto& tagged_node : inline_ready) {
    Process(tagged_node, scheduled_usec);
  }
}

void ExecutorState::ScheduleReady(const TaggedNodeSeq& ready,
                                  std::deque<TaggedNode>* inline_ready) {
  if (ready.empty()) return;

  int64 scheduled_usec = 0;
  if (stats_collector_) {
    scheduled_usec = nodestats::NowInUsec();
  }
  if (inline_ready == nullptr) {
    // Schedule to run all the ready ops in thread pool.
    for (auto& tagged_node : ready) {
      runner_(std::bind(&ME::Process, this, tagged_node, scheduled_usec));
    }
    return;
  }
  const std::vector<NodeItem>& nodes = impl_->nodes_;
  const TaggedNode* curr_expensive_node = nullptr;
  for (auto& tagged_node : ready) {
    const NodeItem& item = nodes[tagged_node.node->id()];
    if (tagged_node.is_dead || !item.kernel->IsExpensive()) {
      // Inline this inexpensive node.
      inline_ready->push_back(tagged_node);
    } else {
      if (curr_expensive_node) {
        // Dispatch to another thread since there is plenty of work to
        // do for this thread.
        runner_(std::bind(&ME::Process, this, *curr_expensive_node,
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
      runner_(
          std::bind(&ME::Process, this, *curr_expensive_node, scheduled_usec));
    }
  }
}

void ExecutorState::Finish() {
  mu_.lock();
  auto status = status_;
  auto done_cb = done_cb_;
  auto runner = runner_;
  mu_.unlock();
  delete this;
  CHECK(done_cb != nullptr);
  runner([done_cb, status]() { done_cb(status); });
}

bool ExecutorState::IsFrameDone(FrameState* frame) {
  return (frame->num_pending_inputs == 0 &&
          frame->num_outstanding_iterations == 0);
}

bool ExecutorState::IsIterationDone(FrameState* frame, int64 iter) {
  IterationState* iter_state = frame->GetIteration(iter);
  if (iter_state->outstanding_ops == 0 &&
      iter_state->outstanding_frame_count == 0) {
    if (iter == 0) {
      // The enclosing frame has no pending input.
      return frame->num_pending_inputs == 0;
    } else {
      // The preceding iteration is deleted (and therefore done).
      return (frame->GetIteration(iter - 1) == nullptr);
    }
  }
  return false;
}

void ExecutorState::FindOrCreateChildFrame(FrameState* frame, int64 iter,
                                           const Node* node,
                                           FrameState** child) {
  // Get the child frame name.
  string enter_name;
  Status s = GetNodeAttr(node->def(), "frame_name", &enter_name);
  CHECK(s.ok()) << s;
  const string child_name = MakeFrameName(frame, iter, enter_name);

  auto it = outstanding_frames_.find(child_name);
  if (it != outstanding_frames_.end()) {
    *child = it->second;
  } else {
    // Need to create a new frame instance.
    VLOG(2) << "Create frame: " << child_name;

    FrameState* temp = new FrameState;
    temp->frame_name = child_name;
    temp->frame_id = Hash64(child_name);
    temp->parent_frame = frame;
    temp->parent_iter = iter;
    s = GetNodeAttr(node->def(), "parallel_iterations",
                    &temp->max_parallel_iterations);
    CHECK(s.ok()) << s;
    // 'iterations' is a fixed-length circular buffer.
    temp->iterations.resize(temp->max_parallel_iterations + 1);
    IterationState* iter_state = new IterationState;
    temp->iterations[0] = iter_state;

    iter_state->outstanding_ops = 0;
    iter_state->outstanding_frame_count = 0;
    iter_state->pending_count = new std::vector<int>;
    InitializePending(impl_->graph_, iter_state->pending_count);
    iter_state->dead_count =
        new std::vector<int>(impl_->graph_->num_node_ids());
    iter_state->input_tensors = new std::vector<Entry>(impl_->total_tensors_);

    auto frame_pending = impl_->frame_input_count_.find(enter_name);
    DCHECK(frame_pending != impl_->frame_input_count_.end());
    temp->num_pending_inputs = frame_pending->second;
    temp->num_outstanding_iterations = 1;
    *child = temp;

    frame->GetIteration(iter)->outstanding_frame_count++;
    outstanding_frames_[child_name] = temp;
  }
}

void ExecutorState::IncrementIteration(FrameState* frame,
                                       TaggedNodeSeq* ready) {
  frame->iteration_count++;
  int64 next_iter = frame->iteration_count;

  VLOG(2) << "Create iteration: [" << frame->frame_name << ", " << next_iter
          << "]";

  IterationState* iter_state = new IterationState;
  frame->SetIteration(next_iter, iter_state);
  frame->num_outstanding_iterations++;
  frame->dead_exits.clear();

  iter_state->outstanding_ops = 0;
  iter_state->outstanding_frame_count = 0;
  iter_state->pending_count = new std::vector<int>;
  InitializePending(impl_->graph_, iter_state->pending_count);
  iter_state->dead_count = new std::vector<int>(impl_->graph_->num_node_ids());
  iter_state->input_tensors = new std::vector<Entry>(impl_->total_tensors_);

  // Activate the successors of the deferred roots in the new iteration.
  ActivateNexts(frame, next_iter, ready);

  // Activate the loop invariants in the new iteration.
  ActivateLoopInvs(frame, next_iter, ready);
}

bool ExecutorState::SetOutputFrameIter(const TaggedNode& tagged_node,
                                       const EntryVector& outputs,
                                       FrameState** output_frame,
                                       int64* output_iter,
                                       TaggedNodeSeq* ready) {
  const Node* node = tagged_node.node;
  FrameState* input_frame = tagged_node.input_frame;
  int64 input_iter = tagged_node.input_iter;
  bool is_dead = tagged_node.is_dead;
  bool is_enter = IsEnter(node);

  if (is_enter) {
    FindOrCreateChildFrame(input_frame, input_iter, node, output_frame);
    // Propagate if this is a loop invariant.
    bool is_constant;
    Status s = GetNodeAttr(node->def(), "is_constant", &is_constant);
    CHECK(s.ok()) << s;
    if (is_constant) {
      AddLoopInv(*output_frame, node, outputs[0], ready);
    }
    --(*output_frame)->num_pending_inputs;
    *output_iter = 0;
  } else if (IsExit(node)) {
    if (is_dead) {
      // Stop and remember this node if it is a dead exit.
      if (input_iter == input_frame->iteration_count) {
        input_frame->dead_exits.push_back(node);
      }
      *output_frame = nullptr;
    } else {
      *output_frame = input_frame->parent_frame;
      *output_iter = input_frame->parent_iter;
    }
  } else if (IsNextIteration(node)) {
    if (is_dead) {
      // Stop the deadness propagation
      *output_frame = nullptr;
    } else {
      if (input_iter == input_frame->iteration_count &&
          input_frame->num_outstanding_iterations ==
              input_frame->max_parallel_iterations) {
        // Reached the maximum for parallel iterations.
        input_frame->next_iter_roots.push_back({node, outputs[0]});
        *output_frame = nullptr;
      } else {
        // If this is a new iteration, start it.
        if (input_iter == input_frame->iteration_count) {
          IncrementIteration(input_frame, ready);
        }
        *output_iter = input_iter + 1;
      }
    }
  }
  return is_enter;
}

void ExecutorState::CleanupFramesIterations(FrameState* frame, int64 iter,
                                            TaggedNodeSeq* ready) {
  int64 curr_iter = iter;
  while (curr_iter <= frame->iteration_count &&
         IsIterationDone(frame, curr_iter)) {
    // Delete the iteration curr_iter
    VLOG(2) << "Delete iteration [" << frame->frame_name << ", " << curr_iter
            << "].";

    delete frame->GetIteration(curr_iter);
    frame->SetIteration(curr_iter, nullptr);
    --frame->num_outstanding_iterations;
    ++curr_iter;

    // If there is a deferred iteration, start it.
    if (frame->next_iter_roots.size() > 0) {
      IncrementIteration(frame, ready);
    }
  }

  if (IsFrameDone(frame)) {
    FrameState* parent_frame = frame->parent_frame;
    int64 parent_iter = frame->parent_iter;

    // Propagate all the dead exits to the parent frame.
    for (const Node* node : frame->dead_exits) {
      auto parent_iter_state = parent_frame->GetIteration(parent_iter);
      std::vector<int>* pending = parent_iter_state->pending_count;
      std::vector<int>* dead_count = parent_iter_state->dead_count;
      for (const Edge* e : node->out_edges()) {
        const Node* dst_node = e->dst();
        const int dst_id = dst_node->id();

        bool dst_dead = true;
        bool dst_ready = false;
        // We know this is a dead input to dst
        if (IsMerge(dst_node)) {
          if (e->IsControlEdge()) {
            (*pending)[dst_id] -= 2;
            int count = (*pending)[dst_id];
            dst_dead = ((*dead_count)[dst_id] == dst_node->num_inputs());
            dst_ready = (count == 1) || ((count == 0) && dst_dead);
          } else {
            ++(*dead_count)[dst_id];
            dst_dead = ((*dead_count)[dst_id] == dst_node->num_inputs());
            dst_ready = ((*pending)[dst_id] == 0) && dst_dead;
          }
        } else {
          ++(*dead_count)[dst_id];
          dst_ready = (--(*pending)[dst_id] == 0);
        }
        if (dst_ready) {
          ready->push_back(
              TaggedNode(dst_node, parent_frame, parent_iter, dst_dead));
          parent_iter_state->outstanding_ops++;
        }
      }
    }

    // Delete the frame
    const string& frame_name = frame->frame_name;
    VLOG(2) << "Delete frame " << frame_name;
    outstanding_frames_.erase(frame_name);
    delete frame;

    // Cleanup recursively
    if (parent_frame != nullptr) {
      parent_frame->GetIteration(parent_iter)->outstanding_frame_count--;
      CleanupFramesIterations(parent_frame, parent_iter, ready);
    }
  }
}

// When ExecutorImpl graph has no control flow nodes,
// SimpleExecutorState is used instead of ExecutorState.  It maintains
// fewer internal state and is convenient for experimenting with async
// op kernels.
class SimpleExecutorState {
 public:
  SimpleExecutorState(const Executor::Args& args, ExecutorImpl* impl);
  ~SimpleExecutorState() {
    for (auto it : device_context_map_) {
      it.second->Unref();
    }
    delete slice_reader_cache_;
  }
  void RunAsync(Executor::DoneCallback done);

 private:
  typedef SimpleExecutorState ME;

  // Not owned.
  Rendezvous* rendezvous_;
  StepStatsCollector* stats_collector_;
  checkpoint::TensorSliceReaderCacheWrapper* slice_reader_cache_;
  FunctionCallFrame* call_frame_;
  const ExecutorImpl* impl_;
  CancellationManager* cancellation_manager_;
  Executor::Args::Runner runner_;

  // Owned.

  // i-th node's j-th input is in tensors_[impl_->nodes[i].input_start
  // + j].  The output is either a tensor pointer (pass-by-reference)
  // or a tensor (pass-by-value).
  //
  // NOTE: Not protected by mu_ because tensors_ is resized once. Each
  // element of tensors_ is written once by the source node of an edge
  // and is cleared by the destination of the same edge. The latter
  // node is never run concurrently with the former node.
  struct Entry {
    Tensor val = *kEmptyTensor;  // A tensor value.
    Tensor* ref = nullptr;       // A tensor reference.
    mutex* ref_mu = nullptr;     // mutex for *ref if ref is not nullptr.

    // Every entry carries an optional DeviceContext containing
    // Device-specific information about how the Tensor was produced.
    DeviceContext* device_context = nullptr;

    // The attributes of the allocator that creates the tensor.
    AllocatorAttributes alloc_attr;
  };

  // Contains a map from node id to the DeviceContext object that was
  // assigned by the device at the beginning of a step.
  DeviceContextMap device_context_map_;

  std::vector<Entry> input_tensors_;

  // Step-local resource manager.
  ResourceMgr step_resource_manager_;

  // Invoked when the execution finishes.
  Executor::DoneCallback done_cb_;

  // How many active threads of computation are being used.  Same as
  // the number of pending Process() functions.
  std::atomic_int_fast32_t num_active_;

  mutex mu_;
  Status status_ GUARDED_BY(mu_);

  // i-th kernel is still waiting for pending[i] inputs.
  class CountDown {
   public:
    CountDown() : v_(0) {}
    void Set(int32 v) { v_.store(v); }
    bool Dec() {
      return v_.load(std::memory_order_acquire) == 1 || v_.fetch_sub(1) == 1;
    }

   private:
    std::atomic_int_fast32_t v_;
  };
  std::vector<CountDown> pending_;

  // Process Node identified by "id" in current thread. "scheduled_usec"
  // indicates when the node becomes ready and gets scheduled.
  void Process(int id, int64 scheduled_usec);

  // Before invoking item->kernel, fills in its "inputs".
  Status PrepareInputs(const NodeItem& item, TensorValueVec* inputs,
                       DeviceContextVec* input_device_contexts);

  // After item->kernel computation is done, processes its outputs
  // and returns nodes that become "ready".
  typedef gtl::InlinedVector<int, 8> ReadyNodeIds;
  Status ProcessOutputs(const NodeItem& item, OpKernelContext* ctx,
                        ReadyNodeIds* ready, NodeExecStats* stats);

  // "node" just finishes. Takes ownership of "stats". Returns true if
  // execution has completed.
  bool NodeDone(const Status& s, const Node* node, const ReadyNodeIds& ready,
                NodeExecStats* stats, std::deque<int>* inline_ready);

  // Call Process() on all nodes in 'inline_ready'.
  void ProcessInline(const std::deque<int>& inline_ready);

  // Schedule all the expensive nodes in 'ready', and put all the inexpensive
  // nodes in 'ready' into 'inline_ready'.
  void ScheduleReady(const ReadyNodeIds& ready, std::deque<int>* inline_ready);

  // One thread of control finishes.
  void Finish();

  TF_DISALLOW_COPY_AND_ASSIGN(SimpleExecutorState);
};

SimpleExecutorState::SimpleExecutorState(const Executor::Args& args,
                                         ExecutorImpl* impl)
    : rendezvous_(args.rendezvous),
      stats_collector_(args.stats_collector),
      slice_reader_cache_(new checkpoint::TensorSliceReaderCacheWrapper),
      call_frame_(args.call_frame),
      impl_(impl),
      cancellation_manager_(args.cancellation_manager),
      runner_(args.runner),
      num_active_(0),
      pending_(impl_->nodes_.size()) {}

void SimpleExecutorState::ProcessInline(const std::deque<int>& inline_ready) {
  if (inline_ready.empty()) return;
  int64 scheduled_usec = 0;
  if (stats_collector_) {
    scheduled_usec = nodestats::NowInUsec();
  }
  for (int id : inline_ready) {
    Process(id, scheduled_usec);
  }
}

void SimpleExecutorState::ScheduleReady(const ReadyNodeIds& ready,
                                        std::deque<int>* inline_ready) {
  if (ready.empty()) return;

  int64 scheduled_usec = 0;
  if (stats_collector_) {
    scheduled_usec = nodestats::NowInUsec();
  }
  if (inline_ready == nullptr) {
    // Schedule to run all the ready ops in thread pool.
    for (auto id : ready) {
      runner_(std::bind(&ME::Process, this, id, scheduled_usec));
    }
    return;
  }
  const std::vector<NodeItem>& nodes = impl_->nodes_;
  int curr_expensive_node = -1;
  for (auto id : ready) {
    if (!nodes[id].kernel->IsExpensive()) {
      // Inline this inexpensive node.
      inline_ready->push_back(id);
    } else {
      if (curr_expensive_node != -1) {
        // Dispatch to another thread since there is plenty of work to
        // do for this thread.
        runner_(
            std::bind(&ME::Process, this, curr_expensive_node, scheduled_usec));
      }
      curr_expensive_node = id;
    }
  }
  if (curr_expensive_node != -1) {
    if (inline_ready->empty()) {
      // Tail recursion optimization
      inline_ready->push_back(curr_expensive_node);
    } else {
      // There are inline nodes to run already. We dispatch this expensive
      // node to other thread.
      runner_(
          std::bind(&ME::Process, this, curr_expensive_node, scheduled_usec));
    }
  }
}

void SimpleExecutorState::RunAsync(Executor::DoneCallback done) {
  const Graph* graph = impl_->graph_;
  ReadyNodeIds ready;

  // Ask the device to fill in the device context map.
  Device* device = impl_->params_.device;
  device->FillContextMap(graph, &device_context_map_);

  for (const Node* n : graph->nodes()) {
    const int id = n->id();
    const int num_in_edges = n->in_edges().size();
    pending_[id].Set(num_in_edges);
    if (num_in_edges == 0) {
      ready.push_back(id);
    }
  }
  if (ready.empty()) {
    done(Status::OK());
  } else {
    num_active_ = ready.size();
    done_cb_ = done;
    input_tensors_.resize(impl_->total_tensors_);
    // Schedule to run all the ready ops in thread pool.
    ScheduleReady(ready, nullptr);
  }
}

Status SimpleExecutorState::PrepareInputs(
    const NodeItem& item, TensorValueVec* inputs,
    DeviceContextVec* input_device_contexts) {
  const Node* node = item.node;

  inputs->clear();
  inputs->resize(node->num_inputs());
  input_device_contexts->clear();
  input_device_contexts->resize(node->num_inputs());

  for (int i = 0; i < node->num_inputs(); ++i) {
    const bool expect_ref = IsRefType(node->input_type(i));
    Entry* entry = input_tensors_.data() + item.input_start + i;
    (*input_device_contexts)[i] = entry->device_context;

    // i-th input.
    TensorValue* inp = &(*inputs)[i];

    if (entry->ref == nullptr) {
      if (expect_ref) {
        return AttachDef(
            errors::InvalidArgument(i, "-th input expects a ref type"),
            item.kernel->def());
      }
      inp->tensor = &entry->val;
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
          entry->val = *entry->ref;
        }
        inp->tensor = &entry->val;
      }
    }
  }
  return Status::OK();
}

void SimpleExecutorState::Process(int id, int64 scheduled_usec) {
  const std::vector<NodeItem>& nodes = impl_->nodes_;
  ReadyNodeIds ready;
  std::deque<int> inline_ready;

  // Parameters passed to OpKernel::Compute.
  TensorValueVec inputs;
  DeviceContextVec input_device_contexts;

  OpKernelContext::Params params;
  Device* device = impl_->params_.device;
  params.device = device;
  // track allocations if and only if we are collecting statistics
  params.track_allocations = (stats_collector_ != nullptr);
  params.rendezvous = rendezvous_;
  params.cancellation_manager = cancellation_manager_;
  params.call_frame = call_frame_;
  params.function_library = impl_->params_.function_library;
  params.resource_manager = device->resource_manager();
  params.step_resource_manager = &step_resource_manager_;
  params.slice_reader_cache = slice_reader_cache_;
  params.inputs = &inputs;
  params.input_device_contexts = &input_device_contexts;
  params.frame_iter = FrameAndIter(0, 0);

  Status s;
  NodeExecStats* stats = nullptr;
  bool completed = false;
  inline_ready.push_back(id);
  while (!inline_ready.empty()) {
    id = inline_ready.front();
    inline_ready.pop_front();
    const NodeItem& item = nodes[id];
    const Node* node = item.node;

    // Set the device_context for this node id, if it exists.
    auto dc_it = device_context_map_.find(id);
    if (dc_it != device_context_map_.end()) {
      params.op_device_context = dc_it->second;
    }

    if (stats_collector_) {
      stats = new NodeExecStats;
      stats->set_node_name(node->name());
      nodestats::SetScheduled(stats, scheduled_usec);
      nodestats::SetAllStart(stats);
    }

    VLOG(1) << "Process node: " << id << " " << SummarizeNodeDef(node->def());

    // Prepares inputs.
    s = PrepareInputs(item, &inputs, &input_device_contexts);
    if (!s.ok()) {
      // Continue to process the nodes in 'inline_ready'.
      completed = NodeDone(s, item.node, ready, stats, &inline_ready);
      continue;
    }

    OpKernel* op_kernel = item.kernel;
    params.op_kernel = op_kernel;
    params.output_alloc_attr = [this, node, op_kernel](int index) {
      return OutputAttributes(&impl_->alloc_attr_, node, op_kernel, index);
    };

    // Asynchronous computes.
    AsyncOpKernel* async = op_kernel->AsAsync();
    if (async) {
      auto pcopy = CopyParams(params);
      auto ctx = new OpKernelContext(*pcopy);
      auto done = [this, item, ctx, stats, pcopy]() {
        VLOG(2) << this
                << " Async kernel done: " << SummarizeNodeDef(item.node->def());
        if (stats_collector_) nodestats::SetOpEnd(stats);
        ReadyNodeIds ready;
        Status s = ProcessOutputs(item, ctx, &ready, stats);
        if (stats_collector_) nodestats::SetMemory(stats, ctx);
        // Schedule to run all the ready ops in thread pool.
        bool completed = NodeDone(s, item.node, ready, stats, nullptr);
        delete ctx;
        DeleteParams(pcopy);
        if (completed) Finish();
      };
      if (stats_collector_) nodestats::SetOpStart(stats);
      device->ComputeAsync(async, ctx, done);
    } else {
      // Synchronous computes.
      OpKernelContext ctx(params);
      if (stats_collector_) nodestats::SetOpStart(stats);
      device->Compute(CHECK_NOTNULL(op_kernel), &ctx);
      if (stats_collector_) nodestats::SetOpEnd(stats);

      s = ProcessOutputs(item, &ctx, &ready, stats);
      if (stats_collector_) nodestats::SetMemory(stats, &ctx);
      if (stats_collector_) {
        scheduled_usec = nodestats::NowInUsec();
      }
      completed = NodeDone(s, node, ready, stats, &inline_ready);
    }
  }  // while !inline_ready.empty()

  // This thread of computation is done if completed = true.
  if (completed) Finish();
}

bool SimpleExecutorState::NodeDone(const Status& s, const Node* node,
                                   const ReadyNodeIds& ready,
                                   NodeExecStats* stats,
                                   std::deque<int>* inline_ready) {
  if (stats_collector_) {
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
    completed = (num_active_.fetch_sub(1) == 1);
  } else if (ready_size > 1) {
    num_active_.fetch_add(ready_size - 1, std::memory_order_relaxed);
  }

  // Schedule the ready nodes in 'ready'.
  if (s.ok()) {
    ScheduleReady(ready, inline_ready);
  }
  return completed;
}

void SimpleExecutorState::Finish() {
  mu_.lock();
  auto ret = status_;
  auto done_cb = done_cb_;
  auto runner = runner_;
  mu_.unlock();
  delete this;
  CHECK(done_cb != nullptr);
  runner([done_cb, ret]() { done_cb(ret); });
}

Status SimpleExecutorState::ProcessOutputs(const NodeItem& item,
                                           OpKernelContext* ctx,
                                           ReadyNodeIds* ready,
                                           NodeExecStats* stats) {
  Status s = ctx->status();
  if (!s.ok()) {
    s = AttachDef(s, item.kernel->def());
    LOG(WARNING) << this << " Compute status: " << s;
    return s;
  }

  // Processes outputs.
  gtl::InlinedVector<Entry, 4> outputs;
  const Node* node = item.node;
  outputs.resize(node->num_outputs());

  // Get the device_context for this node id, if it exists.
  DeviceContext* device_context = nullptr;
  auto dc_it = device_context_map_.find(node->id());
  if (dc_it != device_context_map_.end()) {
    device_context = dc_it->second;
  }

  for (int i = 0; i < node->num_outputs(); ++i) {
    TensorValue val = ctx->release_output(i);
    // Sanity check of output tensor types.
    DataType dtype = val->dtype();
    if (val.is_ref()) dtype = MakeRefType(dtype);
    if (dtype == node->output_type(i)) {
      Entry* out = &(outputs[i]);
      if (val.is_ref()) {
        out->ref = val.tensor;
        out->ref_mu = val.mutex_if_ref;
      } else {
        out->val = *val.tensor;
      }

      // Set the device context of the output entry.
      out->device_context = device_context;

      // Set the allocator attributes of the output entry.
      out->alloc_attr = ctx->output_alloc_attr(i);

      if (stats_collector_ && val.tensor->IsInitialized()) {
        nodestats::SetOutput(stats, i, ctx->output_allocation_type(i),
                             val.tensor);
      }
    } else {
      s.Update(
          errors::Internal("Output ", i, " of type ", DataTypeString(dtype),
                           " does not match declared output type ",
                           DataTypeString(node->output_type(i)),
                           " for operation ", SummarizeNodeDef(node->def())));
    }
    if (!val.is_ref()) {
      // If OpKernelContext returns outputs via pass-by-value, we
      // don't need this trouble.
      delete val.tensor;
    }
  }
  if (!s.ok()) return s;

  // Clears inputs.
  for (int i = 0; i < node->num_inputs(); ++i) {
    input_tensors_[item.input_start + i].val = *kEmptyTensor;
  }

  // Propagates outputs along out edges.
  ready->clear();
  const std::vector<NodeItem>& nodes = impl_->nodes_;
  for (const Edge* e : node->out_edges()) {
    const int src_slot = e->src_output();
    const int dst_id = e->dst()->id();
    const NodeItem& dst_item = nodes[dst_id];
    if (!e->IsControlEdge()) {
      const int dst_slot = e->dst_input();
      input_tensors_[dst_item.input_start + dst_slot] = outputs[src_slot];
    }
    if (pending_[dst_id].Dec()) {
      ready->push_back(dst_id);
    }
  }
  return Status::OK();
}

// NOTE(yuanbyu): Use the executor that supports control flow by default.
const bool use_control_flow_executor = true;
void ExecutorImpl::RunAsync(const Args& args, DoneCallback done) {
  if (params_.has_control_flow || use_control_flow_executor) {
    (new ExecutorState(args, this))->RunAsync(done);
  } else {
    (new SimpleExecutorState(args, this))->RunAsync(done);
  }
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
                             const NodeDef& ndef, OpKernel** kernel) {
  auto device_type = DeviceType(device->attributes().device_type());
  auto allocator = device->GetAllocator(AllocatorAttributes());
  return CreateOpKernel(device_type, device, allocator, flib, ndef, kernel);
}

void DeleteNonCachedKernel(OpKernel* kernel) { delete kernel; }

Status CreateCachedKernel(Device* device, const string& session,
                          FunctionLibraryRuntime* flib, const NodeDef& ndef,
                          OpKernel** kernel) {
  auto op_seg = device->op_segment();
  auto create_fn = [device, flib, &ndef](OpKernel** kernel) {
    return CreateNonCachedKernel(device, flib, ndef, kernel);
  };
  return op_seg->FindOrCreate(session, ndef.name(), kernel, create_fn);
}

// Deletes "kernel".
void DeleteCachedKernel(Device* device, const string& session,
                        OpKernel* kernel) {
  // Do nothing.
}

}  // end namespace tensorflow
