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
#include "tensorflow/core/common_runtime/executor_factory.h"
#include "tensorflow/core/common_runtime/pending_counts.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/collective.h"
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
#include "tensorflow/core/lib/gtl/flatset.h"
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
bool SetTimelineLabel(const Node* node, NodeExecStatsWrapper* stats) {
  bool is_transfer_node = false;
  if (!stats) {
    return is_transfer_node;
  }
  string memory;
  for (auto& all : stats->stats()->memory()) {
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
  const AttrSlice attrs = node->attrs();
  string text;
  if (IsSend(node)) {
    string tensor_name;
    TF_CHECK_OK(GetNodeAttr(attrs, "tensor_name", &tensor_name));
    string recv_device;
    TF_CHECK_OK(GetNodeAttr(attrs, "recv_device", &recv_device));
    text = strings::StrCat(memory, node->name(), " = ", node->type_string(),
                           "(", tensor_name, " @", recv_device);
    is_transfer_node = true;
  } else if (IsRecv(node)) {
    string tensor_name;
    TF_CHECK_OK(GetNodeAttr(attrs, "tensor_name", &tensor_name));
    string send_device;
    TF_CHECK_OK(GetNodeAttr(attrs, "send_device", &send_device));
    text = strings::StrCat(memory, node->name(), " = ", node->type_string(),
                           "(", tensor_name, " @", send_device);
    is_transfer_node = true;
  } else {
    text =
        strings::StrCat(memory, node->name(), " = ", node->type_string(), "(",
                        str_util::Join(node->requested_inputs(), ", "), ")");
  }
  stats->stats()->set_timeline_label(text);
  return is_transfer_node;
}

// Helper routines for collecting step stats.
namespace nodestats {
inline int64 NowInUsec() { return Env::Default()->NowMicros(); }

void SetScheduled(NodeExecStatsWrapper* stats, int64 t) {
  if (!stats) return;
  stats->stats()->set_scheduled_micros(t);
}

void SetAllStart(NodeExecStatsWrapper* stats) {
  if (!stats) return;
  stats->stats()->set_all_start_micros(NowInUsec());
}

void SetOpStart(NodeExecStatsWrapper* stats) {
  if (!stats) return;
  NodeExecStats* nt = stats->stats();
  DCHECK_NE(nt->all_start_micros(), 0);
  nt->set_op_start_rel_micros(NowInUsec() - nt->all_start_micros());
}

void SetOpEnd(NodeExecStatsWrapper* stats) {
  if (!stats) return;
  NodeExecStats* nt = stats->stats();
  DCHECK_NE(nt->all_start_micros(), 0);
  nt->set_op_end_rel_micros(NowInUsec() - nt->all_start_micros());
}

void SetAllEnd(NodeExecStatsWrapper* stats) {
  if (!stats) return;
  NodeExecStats* nt = stats->stats();
  DCHECK_NE(nt->all_start_micros(), 0);
  nt->set_all_end_rel_micros(NowInUsec() - nt->all_start_micros());
}

void SetOutput(NodeExecStatsWrapper* stats, int slot, const Tensor* v) {
  if (!stats) return;
  DCHECK(v);
  NodeOutput* no = stats->stats()->add_output();
  no->set_slot(slot);
  v->FillDescription(no->mutable_tensor_description());
}

void SetMemory(NodeExecStatsWrapper* stats, OpKernelContext* ctx) {
  if (!stats) return;

  for (const auto& allocator_pair : ctx->wrapped_allocators()) {
    stats->AddAllocation(allocator_pair.first, allocator_pair.second);
  }
  auto* ms = stats->stats()->mutable_memory_stats();
  ms->set_temp_memory_size(ctx->temp_memory_allocated());
  for (const auto& alloc_id : ctx->persistent_alloc_ids()) {
    ms->mutable_persistent_tensor_alloc_ids()->Add(alloc_id);
  }
  ms->set_persistent_memory_size(ctx->persistent_memory_allocated());
}

void SetReferencedTensors(NodeExecStatsWrapper* stats,
                          const TensorReferenceVector& tensors) {
  if (!stats) return;
  // be careful not to increment the reference count on any tensor
  // while recording the information
  for (size_t i = 0; i < tensors.size(); ++i) {
    AllocationDescription* description =
        stats->stats()->add_referenced_tensor();
    tensors.at(i).FillDescription(description);
  }
}

}  // namespace nodestats

class ExecutorImpl;
class GraphView;

struct EdgeInfo {
  int dst_id;
  int output_slot : 31;
  // true if this is the last info for output_slot in the EdgeInfo list.
  bool is_last : 1;
  int input_slot;
};

struct NodeItem {
  NodeItem() {}

  // A graph node.
  const Node* node = nullptr;

  // The kernel for this node.
  OpKernel* kernel = nullptr;

  bool kernel_is_expensive : 1;  // True iff kernel->IsExpensive()
  bool kernel_is_async : 1;      // True iff kernel->AsAsync() != nullptr
  bool is_merge : 1;             // True iff IsMerge(node)
  bool is_enter : 1;             // True iff IsEnter(node)
  bool is_exit : 1;              // True iff IsExit(node)
  bool is_control_trigger : 1;   // True iff IsControlTrigger(node)
  bool is_sink : 1;              // True iff IsSink(node)
  // True iff IsEnter(node) || IsExit(node) || IsNextIteration(node)
  bool is_enter_exit_or_next_iter : 1;

  // Cached values of node->num_inputs() and node->num_outputs(), to
  // avoid levels of indirection.
  int num_inputs;
  int num_outputs;

  // ExecutorImpl::tensors_[input_start] is the 1st positional input
  // for this node.
  int input_start = 0;

  // Number of output edges.
  size_t num_output_edges;

  PendingCounts::Handle pending_id;

  const EdgeInfo* output_edge_list() const { return output_edge_base(); }

  // ith output edge.
  const EdgeInfo& output_edge(int i) const {
    DCHECK_GE(i, 0);
    DCHECK_LT(i, num_output_edges);
    return output_edge_base()[i];
  }

  DataType input_type(int i) const {
    DCHECK_LT(i, num_inputs);
    return static_cast<DataType>(input_type_base()[i]);
  }
  DataType output_type(int i) const {
    DCHECK_LT(i, num_outputs);
    return static_cast<DataType>(output_type_base()[i]);
  }

  // Return array of per-output allocator attributes.
  const AllocatorAttributes* output_attrs() const { return output_attr_base(); }

  // Return array of expected input index from which each output should
  // be forwarded:
  // kNeverForward (-2) for DO NOT FORWARD (must allocate).
  // kNoReservation (-1) for no expected forwarding.
  // 0... for forward from that input.
  const int* forward_from() const { return forward_from_base(); }

 private:
  friend class GraphView;

  // Variable length section starts immediately after *this
  // (uint8 is enough for DataType).
  //   EdgeInfo            out_edges[num_out_edges];
  //   AllocatorAttributes output_attr[num_outputs];
  //   int                 forward_from[num_outputs];
  //   uint8               input_type[num_inputs];
  //   uint8               output_type[num_outputs];

  // Return pointer to variable length section.
  char* var() const {
    return const_cast<char*>(reinterpret_cast<const char*>(this) +
                             sizeof(NodeItem));
  }

  EdgeInfo* output_edge_base() const {
    return reinterpret_cast<EdgeInfo*>(var());
  }
  AllocatorAttributes* output_attr_base() const {
    return reinterpret_cast<AllocatorAttributes*>(var() + sizeof(EdgeInfo) *
                                                              num_output_edges);
  }
  int* forward_from_base() const {
    return reinterpret_cast<int*>(var() + sizeof(EdgeInfo) * num_output_edges +
                                  sizeof(AllocatorAttributes) * num_outputs);
  }
  uint8* input_type_base() const {
    return reinterpret_cast<uint8*>(
        var() + sizeof(EdgeInfo) * num_output_edges +
        sizeof(AllocatorAttributes) * num_outputs + sizeof(int) * num_outputs);
  }
  uint8* output_type_base() const {
    return reinterpret_cast<uint8*>(
        var() + sizeof(EdgeInfo) * num_output_edges +
        sizeof(AllocatorAttributes) * num_outputs + sizeof(int) * num_outputs +
        sizeof(uint8) * num_inputs);
  }

  TF_DISALLOW_COPY_AND_ASSIGN(NodeItem);
};

typedef gtl::InlinedVector<TensorValue, 4> TensorValueVec;
typedef gtl::InlinedVector<DeviceContext*, 4> DeviceContextVec;
typedef gtl::InlinedVector<AllocatorAttributes, 4> AllocatorAttributeVec;

// Immutable view of a Graph organized for efficient execution.
class GraphView {
 public:
  GraphView() : space_(nullptr) {}
  ~GraphView();

  void Initialize(const Graph* g);
  Status SetAllocAttrs(const Graph* g, const Device* device);
  void SetScopedAllocatorAttrs(const std::vector<const Node*>& sa_nodes);

  NodeItem* node(size_t id) const {
    DCHECK_GE(id, 0);
    DCHECK_LT(id, num_nodes_);
    uint32 offset = node_offsets_[id];
    return ((offset == kuint32max)
                ? nullptr
                : reinterpret_cast<NodeItem*>(space_ + node_offsets_[id]));
  }

 private:
  char* InitializeNode(char* ptr, const Node* n);
  size_t NodeItemBytes(const Node* n);

  int32 num_nodes_ = 0;
  uint32* node_offsets_ = nullptr;  // array of size "graph_.num_node_ids()"
  // node_offsets_[id] holds the byte offset for node w/ "id" in space_

  char* space_;  // NodeItem objects are allocated here

  TF_DISALLOW_COPY_AND_ASSIGN(GraphView);
};

class ExecutorImpl : public Executor {
 public:
  ExecutorImpl(const LocalExecutorParams& p, std::unique_ptr<const Graph> g)
      : params_(p), graph_(std::move(g)), gview_() {
    CHECK(p.create_kernel != nullptr);
    CHECK(p.delete_kernel != nullptr);
  }

  ~ExecutorImpl() override {
    for (int i = 0; i < graph_->num_node_ids(); i++) {
      NodeItem* item = gview_.node(i);
      if (item != nullptr) {
        params_.delete_kernel(item->kernel);
      }
    }
    for (auto fiter : frame_info_) {
      delete fiter.second;
    }
  }

  Status Initialize();

  // Process all Nodes in the current graph, attempting to infer the
  // memory allocation attributes to be used wherever they may allocate
  // a tensor buffer.
  Status SetAllocAttrs();

  void RunAsync(const Args& args, DoneCallback done) override;

 private:
  friend class ExecutorState;

  struct ControlFlowInfo {
    gtl::FlatSet<string> unique_frame_names;
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

    // Used to determine the next place to allocate space in the
    // pending_counts data structure we'll eventually construct
    PendingCounts::Layout pending_counts_layout;

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
  std::unique_ptr<const Graph> graph_;
  GraphView gview_;

  // A cached value of params_
  bool device_record_tensor_accesses_ = false;

  // Root nodes (with no in edges) that should form the initial ready queue
  std::vector<const Node*> root_nodes_;

  // Mapping from frame name to static information about the frame.
  // TODO(yuanbyu): We could cache it along with the graph so to avoid
  // the overhead of constructing it for each executor instance.
  gtl::FlatMap<string, FrameInfo*> frame_info_;

  TF_DISALLOW_COPY_AND_ASSIGN(ExecutorImpl);
};

// Infer memory allocation attributes of a node n's output,
// based on its use node dst.  Note that dst might not be directly
// connected to n by a single edge, but might be a downstream
// consumer of n's output by reference.  *attr is updated with any
// necessary attributes.
Status InferAllocAttr(const Node* n, const Node* dst,
                      const DeviceNameUtils::ParsedName& local_dev_name,
                      AllocatorAttributes* attr);

GraphView::~GraphView() {
  static_assert(std::is_trivially_destructible<AllocatorAttributes>::value,
                "Update code if AllocatorAttributes gains a destructor");
  static_assert(std::is_trivially_destructible<EdgeInfo>::value,
                "Update code if EdgeInfo gains a destructor");
  for (int i = 0; i < num_nodes_; i++) {
    NodeItem* n = node(i);
    if (n != nullptr) {
      n->NodeItem::~NodeItem();
      // Memory for "n" itself is held in space_ & gets cleaned up below
    }
  }
  delete[] node_offsets_;
  delete[] space_;
}

size_t GraphView::NodeItemBytes(const Node* n) {
  const size_t num_output_edges = n->out_edges().size();
  const int num_inputs = n->num_inputs();
  const int num_outputs = n->num_outputs();

  // Compute number of bytes needed for NodeItem and variable length data.
  // We do not subtract sizeof(var) since num_inputs/num_outputs might
  // both be zero.
  const size_t raw_bytes =
      sizeof(NodeItem)                             // Fixed
      + num_output_edges * sizeof(EdgeInfo)        // output_edges[...]
      + num_outputs * sizeof(AllocatorAttributes)  // output_attr[...]
      + num_outputs * sizeof(int)                  // forward_from[num_outputs]
      + num_inputs * sizeof(uint8)                 // input_type[num_inputs]
      + num_outputs * sizeof(uint8);               // output_type[num_outputs]
  static constexpr size_t kItemAlignment = sizeof(NodeItem*);
  static_assert(kItemAlignment % alignof(NodeItem) == 0,
                "NodeItem must be aligned with kItemAlignment");
  static_assert(kItemAlignment % alignof(EdgeInfo) == 0,
                "EdgeInfo must be aligned with kItemAlignment");
  static_assert(kItemAlignment % alignof(AllocatorAttributes) == 0,
                "AllocatorAttributes must be aligned with kItemAlignment");
  static_assert(sizeof(NodeItem) % alignof(EdgeInfo) == 0,
                "NodeItem must be aligned with EdgeInfo");
  static_assert(sizeof(NodeItem) % alignof(AllocatorAttributes) == 0,
                "NodeItem must be aligned with AllocatorAttributes");
  static_assert(sizeof(EdgeInfo) % alignof(AllocatorAttributes) == 0,
                "EdgeInfo must be aligned with AllocatorAttributes");
  const size_t bytes =
      ((raw_bytes + kItemAlignment - 1) / kItemAlignment) * kItemAlignment;
  return bytes;
}

char* GraphView::InitializeNode(char* ptr, const Node* n) {
  const int id = n->id();
  CHECK(node_offsets_[id] == kuint32max);  // Initial value in constructor

  const size_t bytes = NodeItemBytes(n);
  constexpr size_t kItemAlignment = sizeof(NodeItem*);
  CHECK_EQ(reinterpret_cast<uintptr_t>(ptr) % kItemAlignment, 0);
  NodeItem* item = reinterpret_cast<NodeItem*>(ptr);

  // We store a 32-bit offset relative to the beginning of space_, so that we
  // only need an array of 32-bit values to map from node id to the NodeItem*,
  // (versus 64 bits on most machines if we just stored an array of NodeItem*
  // pointers). Casting to int64 is needed on 32bit CPU to avoid comparing
  // values as "int" vs "size_t" in CHECK_LE.
  CHECK_LE(static_cast<int64>(ptr - space_), kuint32max);
  const uint32 offset = static_cast<uint32>(ptr - space_);
  node_offsets_[id] = offset;
  ptr += bytes;

  const size_t num_output_edges = n->out_edges().size();
  const int num_inputs = n->num_inputs();
  const int num_outputs = n->num_outputs();

  new (item) NodeItem();
  item->num_inputs = num_inputs;
  item->num_outputs = num_outputs;
  item->num_output_edges = num_output_edges;

  // Fill output edges.
  // Keep track of the last EdgeInfo in the EdgeInfo array that references
  // a given output slot.  For all but the last, we need to do a copy of the
  // Tensor when propagating results downstream in the graph, but for the
  // last one, we can just do a move of the Tensor object to propagate it.
  gtl::InlinedVector<EdgeInfo*, 4> last_indices(num_outputs, nullptr);
  EdgeInfo* dst_edge = item->output_edge_base();
  for (auto e : n->out_edges()) {
    dst_edge->dst_id = e->dst()->id();
    CHECK_LE(e->src_output(), 0x3FFFFFFF);  // Must fit in 31 bits
    dst_edge->output_slot = e->src_output();
    dst_edge->is_last = false;
    const int output_slot = dst_edge->output_slot;
    if (output_slot >= 0) {
      last_indices[output_slot] = dst_edge;
    }
    dst_edge->input_slot = e->dst_input();
    dst_edge++;
  }
  for (EdgeInfo* edge_info : last_indices) {
    if (edge_info != nullptr) {
      edge_info->is_last = true;
    }
  }

  AllocatorAttributes* output_attrs = item->output_attr_base();
  for (int i = 0; i < num_outputs; i++) {
    new (&output_attrs[i]) AllocatorAttributes();
  }

  DCHECK_LT(DataType_MAX, 255);  // Must fit in uint8
  uint8* input_types = item->input_type_base();
  for (int i = 0; i < num_inputs; i++) {
    input_types[i] = static_cast<uint8>(n->input_type(i));
    DCHECK_EQ(item->input_type(i), n->input_type(i));
  }

  // Check ScopedAllocatorAttrs and forward_from.  Also assign output_types.
  {
    std::vector<int> forward_input;
    Status fwd_status =
        GetNodeAttr(n->attrs(), "_forward_input", &forward_input);
    std::vector<int> scoped_allocator_attrs;
    Status sa_status =
        GetNodeAttr(n->attrs(), "_scoped_allocator", &scoped_allocator_attrs);

    int* forward_from = item->forward_from_base();
    uint8* output_types = item->output_type_base();
    for (int i = 0; i < num_outputs; ++i) {
      output_types[i] = static_cast<uint8>(n->output_type(i));
      DCHECK_EQ(item->output_type(i), n->output_type(i));

      forward_from[i] = OpKernelContext::Params::kNoReservation;
      if (sa_status.ok()) {
        for (int j = 0; j < scoped_allocator_attrs.size(); j += 2) {
          if (scoped_allocator_attrs[j] == i) {
            // This output slot must be explicitly allocated from a
            // ScopedAllocator.
            forward_from[i] = OpKernelContext::Params::kNeverForward;
            DCHECK_EQ(output_attrs[i].scope_id, 0);
            output_attrs[i].scope_id = scoped_allocator_attrs[j + 1];
          }
        }
      }
      if (fwd_status.ok() &&
          forward_from[i] == OpKernelContext::Params::kNoReservation) {
        DCHECK_EQ(forward_input.size() % 2, 0);
        for (int j = 0; j < forward_input.size(); j += 2) {
          if (forward_input[j + 1] == i) {
            DCHECK_EQ(forward_from[i], OpKernelContext::Params::kNoReservation);
            forward_from[i] = forward_input[j];
            break;
          }
        }
      }
    }
  }

  return ptr;
}

void GraphView::Initialize(const Graph* g) {
  CHECK(node_offsets_ == nullptr);
  const int num_nodes = g->num_node_ids();
  num_nodes_ = num_nodes;
  size_t total_bytes = 0;
  for (const Node* n : g->nodes()) {
    total_bytes += NodeItemBytes(n);
  }

  node_offsets_ = new uint32[num_nodes];
  for (int i = 0; i < num_nodes; i++) {
    node_offsets_[i] = kuint32max;
  }

  space_ = new char[total_bytes];  // NodeItem objects are allocated here
  char* ptr = space_;
  for (const Node* n : g->nodes()) {
    ptr = InitializeNode(ptr, n);
  }
  CHECK_EQ(ptr, space_ + total_bytes);
}

void GetMaxPendingCounts(const Node* n, size_t* max_pending,
                         size_t* max_dead_count) {
  const size_t num_in_edges = n->in_edges().size();
  size_t initial_count;
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

  *max_pending = initial_count;
  *max_dead_count = num_in_edges;
}

Status ExecutorImpl::Initialize() {
  gview_.Initialize(graph_.get());

  // Build the information about frames in this subgraph.
  ControlFlowInfo cf_info;
  TF_RETURN_IF_ERROR(BuildControlFlowInfo(graph_.get(), &cf_info));

  // Cache this value so we make this virtual function call once, rather
  // that O(# steps * # nodes per step) times.
  device_record_tensor_accesses_ =
      params_.device->RequiresRecordingAccessedTensors();

  for (auto& it : cf_info.unique_frame_names) {
    EnsureFrameInfo(it)->nodes = new std::vector<const Node*>;
  }

  // Preprocess every node in the graph to create an instance of op
  // kernel for each node.
  for (const Node* n : graph_->nodes()) {
    const int id = n->id();
    const string& frame_name = cf_info.frame_names[id];
    FrameInfo* frame_info = EnsureFrameInfo(frame_name);

    // See if this node is a root node, and if so, add to root_nodes_.
    if (n->in_edges().empty()) {
      root_nodes_.push_back(n);
    }

    NodeItem* item = gview_.node(id);
    item->node = n;

    item->input_start = frame_info->total_inputs;
    frame_info->total_inputs += n->num_inputs();

    Status s = params_.create_kernel(n->def(), &item->kernel);
    if (!s.ok()) {
      item->kernel = nullptr;
      s = AttachDef(s, *n);
      LOG(ERROR) << "Executor failed to create kernel. " << s;
      return s;
    }
    CHECK(item->kernel);
    item->kernel_is_expensive = item->kernel->IsExpensive();
    item->kernel_is_async = (item->kernel->AsAsync() != nullptr);
    item->is_merge = IsMerge(n);
    item->is_enter = IsEnter(n);
    item->is_exit = IsExit(n);
    item->is_control_trigger = IsControlTrigger(n);
    item->is_sink = IsSink(n);
    item->is_enter_exit_or_next_iter =
        (IsEnter(n) || IsExit(n) || IsNextIteration(n));

    // Compute the maximum values we'll store for this node in the
    // pending counts data structure, and allocate a handle in
    // that frame's pending counts data structure that has enough
    // space to store these maximal count values.
    size_t max_pending, max_dead;
    GetMaxPendingCounts(n, &max_pending, &max_dead);
    item->pending_id =
        frame_info->pending_counts_layout.CreateHandle(max_pending, max_dead);

    // Initialize static information about the frames in the graph.
    frame_info->nodes->push_back(n);
    if (IsEnter(n)) {
      string enter_name;
      TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "frame_name", &enter_name));
      EnsureFrameInfo(enter_name)->input_count++;
    }
  }

  // Initialize PendingCounts only after item->pending_id is initialized for
  // all nodes.
  InitializePending(graph_.get(), cf_info);

  return gview_.SetAllocAttrs(graph_.get(), params_.device);
}

// If a Node has been marked to use a ScopedAllocator x for output i, then
// sc_attr will contain the subsequence (i, x) at an even offset.  This function
// extracts and transfers that ScopedAllocator id to alloc_attr.  For now, we
// only allow one ScopedAllocator use per Node.
bool ExtractScopedAllocatorAttr(const std::vector<int>& sc_attr,
                                int output_index,
                                AllocatorAttributes* alloc_attr) {
  DCHECK_LE(2, sc_attr.size());
  for (int i = 0; i < sc_attr.size(); i += 2) {
    if (sc_attr[i] == output_index) {
      CHECK_EQ(alloc_attr->scope_id, 0);
      alloc_attr->scope_id = sc_attr[i + 1];
      return true;
    }
  }
  return false;
}

void GraphView::SetScopedAllocatorAttrs(
    const std::vector<const Node*>& sa_nodes) {
  for (const Node* sa : sa_nodes) {
    NodeItem* sa_item = node(sa->id());
    AllocatorAttributes* sa_attrs = sa_item->output_attr_base();
    // Control edges out of the ScopedAllocator should be use instances, but may
    // include a few other nodes.
    for (const auto& e : sa->out_edges()) {
      if (!e->IsControlEdge()) {
        continue;
      }
      Node* use_node = e->dst();
      NodeItem* item = node(use_node->id());
      AllocatorAttributes* use_attrs = item->output_attr_base();
      std::vector<int> scoped_allocator_attrs;
      Status s = GetNodeAttr(use_node->attrs(), "_scoped_allocator",
                             &scoped_allocator_attrs);
      if (!s.ok()) {
        VLOG(2) << "Failed to find expected ScopedAllocator attr on "
                << use_node->name();
        continue;
      }
      // There can be more than one output using ScopedAllocation, but this
      // analysis assumes they use the same ScopedAllocator.
      for (const auto& e : use_node->out_edges()) {
        if (!e->IsControlEdge()) {
          AllocatorAttributes attr;
          if (ExtractScopedAllocatorAttr(scoped_allocator_attrs,
                                         e->src_output(), &attr)) {
            // Set the scope_id on this use instance node.
            (use_attrs + e->src_output())->Merge(attr);
            // Propagate the other attributes of this node back to the SA node.
            attr = *(use_attrs + e->src_output());
            attr.scope_id = 0;
            sa_attrs->Merge(attr);
          }
        }
      }
    }
  }
}

Status GraphView::SetAllocAttrs(const Graph* g, const Device* device) {
  Status s;
  DeviceNameUtils::ParsedName local_dev_name = device->parsed_name();

  std::vector<const Node*> scoped_allocator_instances;
  for (const Node* n : g->nodes()) {
    NodeItem* item = node(n->id());
    AllocatorAttributes* attrs = item->output_attr_base();
    if (IsScopedAllocator(n)) {
      scoped_allocator_instances.push_back(n);
    }

    // Examine the out edges of each node looking for special use
    // cases that may affect memory allocation attributes.
    for (const auto& e : n->out_edges()) {
      if (!e->IsControlEdge()) {
        AllocatorAttributes attr;
        s = InferAllocAttr(n, e->dst(), local_dev_name, &attr);
        if (!s.ok()) return s;
        if (attr.value != 0 || attr.scope_id != 0) {
          attrs[e->src_output()].Merge(attr);
        }
      }
    }

    for (int out = 0; out < n->num_outputs(); out++) {
      const OpKernel* op_kernel = item->kernel;
      DCHECK_LT(out, op_kernel->output_memory_types().size());
      bool on_host = op_kernel->output_memory_types()[out] == HOST_MEMORY;
      if (on_host) {
        AllocatorAttributes h;
        h.set_on_host(on_host);
        attrs[out].Merge(h);
      }
    }
  }
  SetScopedAllocatorAttrs(scoped_allocator_instances);
  return s;
}

Status InferAllocAttr(const Node* n, const Node* dst,
                      const DeviceNameUtils::ParsedName& local_dev_name,
                      AllocatorAttributes* attr) {
  Status s;
  // Note that it's possible for *n to be a Recv and *dst to be a Send,
  // so these two cases are not mutually exclusive.
  if (IsRecv(n)) {
    string src_name;
    s = GetNodeAttr(n->attrs(), "send_device", &src_name);
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
      // Value is going to be the sink of a local DMA from GPU to CPU (or
      // other types of accelerators).
      attr->set_gpu_compatible(true);
      VLOG(2) << "node " << n->name() << " is the sink of a gpu->cpu copy";
    } else {
      VLOG(2) << "default alloc case local type " << local_dev_name.type
              << " remote type " << parsed_src_name.type;
    }
  }
  if (IsSend(dst)) {
    string dst_name;
    s = GetNodeAttr(dst->attrs(), "recv_device", &dst_name);
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
  }
  if (n->IsCollective()) {
    // We'll make the sweeping assumption that any collective op is going
    // to be involved in network i/o.
    attr->set_nic_compatible(true);
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

    Entry& operator=(Entry&& other) {
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
        val.Init(std::move(*other.val));
      }
      return *this;
    }

    // Clears the <val> field.
    void ClearVal() {
      if (val_field_is_set) {
        val.Destroy();
        val_field_is_set = false;
        has_value = false;
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
          counts_(*pending_counts) {  // Initialize with copy of *pending_counts
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
    size_t outstanding_ops;

    // The number of outstanding frames for each iteration.
    int outstanding_frame_count;
    int pending(PendingCounts::Handle h) { return counts_.pending(h); }
    int decrement_pending(PendingCounts::Handle h, int v) {
      return counts_.decrement_pending(h, v);
    }
    // Mark a merge node as live
    // REQUIRES: Node corresponding to "h" is a merge node
    void mark_live(PendingCounts::Handle h) { counts_.mark_live(h); }
    // Mark a node to show that processing has started.
    void mark_started(PendingCounts::Handle h) { counts_.mark_started(h); }
    // Mark a node to show that processing has completed.
    void mark_completed(PendingCounts::Handle h) { counts_.mark_completed(h); }
    PendingCounts::NodeState node_state(PendingCounts::Handle h) {
      return counts_.node_state(h);
    }

    int dead_count(PendingCounts::Handle h) { return counts_.dead_count(h); }
    void increment_dead_count(PendingCounts::Handle h) {
      counts_.increment_dead_count(h);
    }
    void adjust_for_activation(PendingCounts::Handle h, bool increment_dead,
                               int* pending_result, int* dead_result) {
      counts_.adjust_for_activation(h, increment_dead, pending_result,
                                    dead_result);
    }

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

    // Lock ordering: ExecutorState.mu_ < mu;
    // during structured traversal: parent_frame->mu < mu.
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
      size_t index = iter % iterations.size();
      return iterations[index];
    }

    inline void SetIteration(int64 iter, IterationState* state)
        EXCLUSIVE_LOCKS_REQUIRED(mu) {
      size_t index = iter % iterations.size();
      DCHECK(state == nullptr || iterations[index] == nullptr);
      iterations[index] = state;
    }

    // Decrement the outstanding op count and clean up the iterations in the
    // frame. Return true iff the execution of the frame is done.
    inline bool DecrementOutstandingOps(const GraphView* gview, int64 iter,
                                        TaggedNodeSeq* ready) {
      mutex_lock l(mu);
      return DecrementOutstandingOpsLocked(gview, iter, ready);
    }

    // Decrement the outstanding op count and clean up the iterations in the
    // frame. Return true iff the execution of the frame is done.
    inline bool DecrementOutstandingOpsLocked(const GraphView* gview,
                                              int64 iter, TaggedNodeSeq* ready)
        EXCLUSIVE_LOCKS_REQUIRED(mu) {
      IterationState* istate = GetIteration(iter);
      istate->outstanding_ops--;
      if (istate->outstanding_ops != 0) {
        return false;
      } else {
        return CleanupIterations(gview, iter, ready);
      }
    }

    // Returns true if the computation in the frame is completed.
    inline bool IsFrameDone() EXCLUSIVE_LOCKS_REQUIRED(mu) {
      return (num_pending_inputs == 0 && num_outstanding_iterations == 0);
    }

    // Returns true if the iteration of the frame is completed.
    bool IsIterationDone(int64 iter) EXCLUSIVE_LOCKS_REQUIRED(mu);

    // Increments the iteration id. If this is a new iteration, initialize it.
    void IncrementIteration(const GraphView* gview, TaggedNodeSeq* ready)
        EXCLUSIVE_LOCKS_REQUIRED(mu);

    // Activate all the deferred NextIteration nodes in a new iteration.
    void ActivateNexts(const GraphView* gview, int64 iter, TaggedNodeSeq* ready)
        EXCLUSIVE_LOCKS_REQUIRED(mu);

    // Activate all the current loop invariants in a new iteration.
    void ActivateLoopInvs(const GraphView* gview, int64 iter,
                          TaggedNodeSeq* ready) EXCLUSIVE_LOCKS_REQUIRED(mu);

    // Add a new loop invariant and make it available to all active
    // iterations.
    void AddLoopInv(const NodeItem* item, const Entry& value,
                    TaggedNodeSeq* ready) EXCLUSIVE_LOCKS_REQUIRED(mu);

    // Activate the successors of a node. Contents of *outputs are left in an
    // indeterminate state after returning from this method.
    void ActivateNodes(const NodeItem* item, const bool is_dead, int64 iter,
                       EntryVector* outputs, TaggedNodeSeq* ready)
        EXCLUSIVE_LOCKS_REQUIRED(mu);

    // Cleanup iterations of this frame starting from iteration iter.
    bool CleanupIterations(const GraphView* gview, int64 iter,
                           TaggedNodeSeq* ready) EXCLUSIVE_LOCKS_REQUIRED(mu);

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
          // Lots of unused entries at beginning of vector: move everything
          // down to start of vector.
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
  CollectiveExecutor* collective_executor_ = nullptr;
  SessionState* session_state_;
  TensorStore* tensor_store_;
  // Step-local container.
  ScopedStepContainer* step_container_;
  StepStatsCollector* stats_collector_;
  // QUESTION: Make it a checkpoint::TensorSliceReaderCacheWrapper
  // instead of a pointer?  (avoids having to delete).
  checkpoint::TensorSliceReaderCacheWrapper* slice_reader_cache_;
  CallFrameInterface* call_frame_;
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
  gtl::FlatMap<string, FrameState*> outstanding_frames_ GUARDED_BY(mu_);

  // The unique name of a frame.
  inline string MakeFrameName(FrameState* frame, int64 iter_id,
                              const string& name) {
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
                        EntryVector* outputs, NodeExecStatsWrapper* stats);

  // After processing the outputs, propagates the outputs to their dsts.
  // Contents of *outputs are left in an indeterminate state after
  // returning from this method.
  void PropagateOutputs(const TaggedNode& tagged_node, const NodeItem* item,
                        EntryVector* outputs, TaggedNodeSeq* ready);

  // "node" just finishes. Takes ownership of "stats". Returns true if
  // execution has completed.
  bool NodeDone(const Status& s, const Node* node, const TaggedNodeSeq& ready,
                NodeExecStatsWrapper* stats,
                TaggedNodeReadyQueue* inline_ready);

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
      collective_executor_(args.collective_executor),
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
      cf_info->unique_frame_names.insert(frame_name);
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
          GetNodeAttr(curr_node->attrs(), "frame_name", &frame_name));
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
      const int out_id = out->id();

      // Add to ready queue if not visited.
      bool is_visited = visited[out_id];
      if (!is_visited) {
        ready.push_back(out);
        visited[out_id] = true;

        // Process the node 'out'.
        cf_info->frame_names[out_id] = frame_name;
        parent_nodes[out_id] = parent;
        cf_info->unique_frame_names.insert(frame_name);
      }
    }
  }

  return Status::OK();
}

void ExecutorImpl::InitializePending(const Graph* graph,
                                     const ControlFlowInfo& cf_info) {
  for (auto& it : cf_info.unique_frame_names) {
    FrameInfo* finfo = EnsureFrameInfo(it);
    PendingCounts* counts = new PendingCounts(finfo->pending_counts_layout);
    DCHECK_EQ(finfo->pending_counts, nullptr);
    finfo->pending_counts = counts;
  }
  for (const Node* n : graph->nodes()) {
    const int id = n->id();
    const string& name = cf_info.frame_names[id];
    size_t max_pending, max_dead;
    GetMaxPendingCounts(n, &max_pending, &max_dead);
    const NodeItem* item = gview_.node(id);
    PendingCounts* counts = EnsureFrameInfo(name)->pending_counts;
    counts->set_initial_count(item->pending_id, max_pending);
  }
}

void ExecutorState::RunAsync(Executor::DoneCallback done) {
  const Graph* graph = impl_->graph_.get();
  TaggedNodeSeq ready;

  // Ask the device to fill in the device context map.
  Device* device = impl_->params_.device;
  const Status fill_status =
      device->FillContextMap(graph, &device_context_map_);
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
    done_cb_ = std::move(done);
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
             const NodeItem* _item, Entry* _first_input,
             NodeExecStatsWrapper* _stats)
      : saved_inputs(*p.inputs),
        saved_input_device_contexts(*p.input_device_contexts),
        saved_input_alloc_attrs(*p.input_alloc_attrs),
        params(p),
        tagged_node(_tagged_node),
        item(_item),
        first_input(_first_input),
        // ParamsButClearingEigenGPUDevice does equivalent of
        //   params.eigen_gpu_device = nullptr;
        ctx(ParamsButClearingEigenGPUDevice(&params), item->num_outputs),
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
  const NodeItem* item;
  Entry* first_input;
  OpKernelContext ctx;
  NodeExecStatsWrapper* stats;

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
  const GraphView& gview = impl_->gview_;
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
  params.collective_executor = collective_executor_;
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
  params.stats_collector = stats_collector_;

  Status s;
  NodeExecStatsWrapper* stats = nullptr;
  EntryVector outputs;
  bool completed = false;
  inline_ready.push_back(tagged_node);
  while (!inline_ready.empty()) {
    tagged_node = inline_ready.front();
    inline_ready.pop_front();
    const Node* node = tagged_node.node;
    FrameState* input_frame = tagged_node.input_frame;
    const int64 input_iter = tagged_node.input_iter;
    const int id = node->id();
    const NodeItem& item = *gview.node(id);

    // TODO(misard) Replace with a finer-grain enabling flag once we
    // add better optional debugging support.
    if (vlog_ && VLOG_IS_ON(1)) {
      mutex_lock l(input_frame->mu);
      input_frame->GetIteration(input_iter)->mark_started(item.pending_id);
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
      stats = new NodeExecStatsWrapper;
      stats->stats()->set_node_name(node->name());
      nodestats::SetScheduled(stats, scheduled_usec);
      nodestats::SetAllStart(stats);
    }

    if (vlog_) {
      VLOG(1) << "Process node: " << id << " step " << params.step_id << " "
              << SummarizeNode(*node) << " is dead: " << tagged_node.is_dead
              << " device: " << device->name();
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
      params.output_attr_array = item.output_attrs();
      params.forward_from_array = item.forward_from();

      if (item.kernel_is_async) {
        // Asynchronous computes.
        AsyncOpKernel* async = item.kernel->AsAsync();
        DCHECK(async != nullptr);
        launched_asynchronously = true;
        AsyncState* state =
            new AsyncState(params, tagged_node, &item, first_input, stats);

        auto done = [this, state]() {
          Device* device = impl_->params_.device;
          NodeExecStatsWrapper* stats = state->stats;  // Shorthand
          Entry* first_input = state->first_input;     // Shorthand

          nodestats::SetOpEnd(stats);
          EntryVector outputs;
          Status s = ProcessOutputs(*state->item, &state->ctx, &outputs, stats);
          nodestats::SetMemory(stats, &state->ctx);
          if (vlog_) {
            VLOG(2) << "Async kernel done: " << state->item->node->id()
                    << " step " << step_id_ << " "
                    << SummarizeNode(*state->item->node)
                    << " is dead: " << state->tagged_node.is_dead
                    << " device: " << device->name();
          }

          // Clears inputs.
          const int num_inputs = state->item->num_inputs;
          for (int i = 0; i < num_inputs; ++i) {
            (first_input + i)->ClearVal();
          }
          FrameState* input_frame = state->tagged_node.input_frame;
          const int64 input_iter = state->tagged_node.input_iter;
          const int id = state->tagged_node.node->id();
          MaybeMarkCompleted(input_frame, input_iter, id);
          TaggedNodeSeq ready;
          if (s.ok()) {
            PropagateOutputs(state->tagged_node, state->item, &outputs, &ready);
          }
          outputs.clear();
          if (s.ok() && impl_->device_record_tensor_accesses_) {
            // Get the list of all tensors accessed during the execution
            TensorReferenceVector accessed;
            state->ctx.retrieve_accessed_tensors(&accessed);
            nodestats::SetReferencedTensors(stats, accessed);
            // callee takes ownership of the vector
            device->ConsumeListOfAccessedTensors(state->ctx.op_device_context(),
                                                 accessed);
          }
          const bool completed =
              NodeDone(s, state->item->node, ready, stats, nullptr);
          delete state;
          if (completed) Finish();
        };
        nodestats::SetOpStart(stats);
        device->ComputeAsync(async, &state->ctx, done);
      } else {
        // Synchronous computes.
        OpKernelContext ctx(&params, item.num_outputs);
        nodestats::SetOpStart(stats);
        device->Compute(CHECK_NOTNULL(op_kernel), &ctx);
        nodestats::SetOpEnd(stats);
        s = ProcessOutputs(item, &ctx, &outputs, stats);
        if (s.ok() && impl_->device_record_tensor_accesses_) {
          // Get the list of all tensors accessed during the execution
          ctx.retrieve_accessed_tensors(&accessed_tensors);
          device_context = ctx.op_device_context();
        }
        nodestats::SetMemory(stats, &ctx);
      }
    }

    if (!launched_asynchronously) {
      if (vlog_) {
        VLOG(2) << "Synchronous kernel done: " << id << " step "
                << params.step_id << " " << SummarizeNode(*node)
                << " is dead: " << tagged_node.is_dead
                << " device: " << device->name();
      }

      // Clears inputs.
      const int num_inputs = item.num_inputs;
      for (int i = 0; i < num_inputs; ++i) {
        (first_input + i)->ClearVal();
      }
      MaybeMarkCompleted(input_frame, input_iter, id);
      // Propagates outputs.
      if (s.ok()) {
        PropagateOutputs(tagged_node, &item, &outputs, &ready);
      }
      outputs.clear();
      if (!accessed_tensors.empty()) {
        nodestats::SetReferencedTensors(stats, accessed_tensors);
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
        DCHECK(IsTransferNode(node)) << node->name() << " - input " << i;
        DCHECK(!entry->val_field_is_set) << node->name() << " - input " << i;
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
      {
        mutex_lock ml(*entry->ref_mu);
        if (!entry->ref->IsInitialized() && !IsInitializationOp(item.node)) {
          return AttachDef(errors::FailedPrecondition(
                               "Attempting to use uninitialized value ",
                               item.kernel->requested_input(i)),
                           item.kernel->def());
        }
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
        // The dtype of entry->ref could have been changed by another operation
        // that ran after the operation that "produced" it executed, so
        // re-validate that the type of the dereferenced tensor matches the
        // expected input type.
        if (item.input_type(i) != inp->tensor->dtype()) {
          return AttachDef(
              errors::InvalidArgument(
                  i, "-th input expects type ",
                  DataTypeString(item.input_type(i)),
                  " but automatically dereferenced input tensor has type ",
                  DataTypeString(inp->tensor->dtype())),
              item.kernel->def());
        }
      }
    }
  }
  return Status::OK();
}

Status ExecutorState::ProcessOutputs(const NodeItem& item, OpKernelContext* ctx,
                                     EntryVector* outputs,
                                     NodeExecStatsWrapper* stats) {
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
    if (s.code() == error::RESOURCE_EXHAUSTED) {
      if (stats_collector_) {
        string err = stats_collector_->ReportAllocsOnResourceExhausted(
            s.error_message());
        s = Status(s.code(), strings::StrCat(s.error_message(), err));
      } else {
        s = Status(
            s.code(),
            strings::StrCat(
                s.error_message(),
                "\nHint: If you want to see a list of allocated tensors when "
                "OOM happens, add report_tensor_allocations_upon_oom "
                "to RunOptions for current allocation info.\n"));
      }
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
    s.Update(impl_->params_.node_outputs_cb(item.node->name(), -1, nullptr,
                                            false, ctx));
  }

  for (int i = 0; i < item.num_outputs; ++i) {
    const TensorValue val = ctx->release_output(i);
    if (*ctx->is_output_dead() || val.tensor == nullptr) {
      // Unless it's a Switch or a Recv, the node must produce a
      // tensor value at i-th output.
      if (!IsSwitch(node) && !IsRecv(node)) {
        s.Update(errors::Internal("Missing ", i, "-th output from ",
                                  SummarizeNode(*node)));
      }
    } else {
      Entry* out = &((*outputs)[i]);

      // Set the device context of the output entry.
      out->device_context = device_context;

      // Set the allocator attributes of the output entry.
      out->alloc_attr = ctx->output_alloc_attr(i);

      // Sanity check of output tensor types.
      DataType dtype;
      if (val.is_ref()) {
        mutex_lock ml(*val.mutex_if_ref);
        dtype = MakeRefType(val->dtype());
      } else {
        dtype = val->dtype();
      }
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

          // Experimental: debugger (tfdb) access to intermediate node
          // outputs.
          if (impl_->params_.node_outputs_cb != nullptr) {
            s.Update(impl_->params_.node_outputs_cb(item.node->name(), i,
                                                    out->ref, true, ctx));
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
            s.Update(impl_->params_.node_outputs_cb(
                item.node->name(), i, out->val.get(), false, ctx));
          }
        }
      } else {
        s.Update(errors::Internal("Output ", i, " of type ",
                                  DataTypeString(dtype),
                                  " does not match declared output type ",
                                  DataTypeString(item.output_type(i)),
                                  " for node ", SummarizeNode(*node)));
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
                                     const NodeItem* item, EntryVector* outputs,
                                     TaggedNodeSeq* ready) {
  const Node* node = tagged_node.node;
  FrameState* input_frame = tagged_node.input_frame;
  const int64 input_iter = tagged_node.input_iter;
  const bool is_dead = tagged_node.is_dead;

  // Propagates outputs along out edges, and puts newly ready nodes
  // into the ready queue.
  ready->clear();
  bool is_frame_done = false;
  FrameState* output_frame = input_frame;
  int64 output_iter = input_iter;

  if (!item->is_enter_exit_or_next_iter) {
    // Fast path for nodes types that don't need special handling
    DCHECK_EQ(input_frame, output_frame);
    // Normal path for most nodes
    mutex_lock l(input_frame->mu);
    output_frame->ActivateNodes(item, is_dead, output_iter, outputs, ready);
    is_frame_done = input_frame->DecrementOutstandingOpsLocked(
        &impl_->gview_, input_iter, ready);
  } else if (item->is_enter) {
    bool is_constant;
    const Status s = GetNodeAttr(node->attrs(), "is_constant", &is_constant);
    DCHECK(s.ok()) << s;
    FindOrCreateChildFrame(input_frame, input_iter, node, &output_frame);
    output_iter = 0;
    {
      const NodeItem* item = impl_->gview_.node(node->id());
      mutex_lock l(output_frame->mu);
      if (is_constant) {
        // Propagate to all active iterations if this is a loop invariant.
        output_frame->AddLoopInv(item, (*outputs)[0], ready);
      } else {
        output_frame->ActivateNodes(item, is_dead, output_iter, outputs, ready);
      }
      output_frame->num_pending_inputs--;
    }
    is_frame_done =
        input_frame->DecrementOutstandingOps(&impl_->gview_, input_iter, ready);
  } else if (item->is_exit) {
    if (is_dead) {
      mutex_lock l(input_frame->mu);
      // Stop and remember this node if it is a dead exit.
      if (input_iter == input_frame->iteration_count) {
        input_frame->dead_exits.push_back(node);
      }
      is_frame_done = input_frame->DecrementOutstandingOpsLocked(
          &impl_->gview_, input_iter, ready);
    } else {
      output_frame = input_frame->parent_frame;
      output_iter = input_frame->parent_iter;
      {
        mutex_lock l(output_frame->mu);
        output_frame->ActivateNodes(item, is_dead, output_iter, outputs, ready);
      }
      is_frame_done = input_frame->DecrementOutstandingOps(&impl_->gview_,
                                                           input_iter, ready);
    }
  } else {
    DCHECK(IsNextIteration(node));
    mutex_lock l(input_frame->mu);
    if (is_dead) {
      // Stop the deadness propagation.
      output_frame = nullptr;
    } else {
      if (input_iter == input_frame->iteration_count &&
          input_frame->num_outstanding_iterations ==
              input_frame->max_parallel_iterations) {
        // Reached the maximum for parallel iterations.
        input_frame->next_iter_roots.push_back({node, (*outputs)[0]});
        output_frame = nullptr;
      } else {
        // If this is a new iteration, start it.
        if (input_iter == input_frame->iteration_count) {
          input_frame->IncrementIteration(&impl_->gview_, ready);
        }
        output_iter = input_iter + 1;
      }
    }
    if (output_frame != nullptr) {
      // This is the case when node is not Enter, Exit, or NextIteration.
      DCHECK(input_frame == output_frame);
      output_frame->ActivateNodes(item, is_dead, output_iter, outputs, ready);
    }
    is_frame_done = input_frame->DecrementOutstandingOpsLocked(
        &impl_->gview_, input_iter, ready);
  }

  // At this point, this node is completely done. We also know if the
  // completion of this node makes its frame completed.
  if (is_frame_done) {
    FrameState* parent_frame = input_frame->parent_frame;
    const int64 parent_iter = input_frame->parent_iter;
    DeleteFrame(input_frame, ready);
    if (parent_frame != nullptr) {
      // The completion of frame may cause completions in its parent frame.
      // So clean things up recursively.
      CleanupFramesIterations(parent_frame, parent_iter, ready);
    }
  }
}

bool ExecutorState::NodeDone(const Status& s, const Node* node,
                             const TaggedNodeSeq& ready,
                             NodeExecStatsWrapper* stats,
                             TaggedNodeReadyQueue* inline_ready) {
  nodestats::SetAllEnd(stats);
  if (stats_collector_ != nullptr && !SetTimelineLabel(node, stats)) {
    // Only record non-transfer nodes.
    // Transfers 'stats' ownership to 'stats_collector_'.
    stats_collector_->Save(impl_->params_.device->name(), stats);
  } else if (stats) {
    delete stats;
  }

  bool abort_run = false;
  if (!s.ok()) {
    // Some error happened. This thread of computation is done.
    mutex_lock l(mu_);
    if (status_.ok()) {
      abort_run = true;
      status_ = s;
    }
  }
  if (abort_run) {
    TRACEPRINTF("StartAbort: %s", s.ToString().c_str());
    if (rendezvous_) {
      rendezvous_->StartAbort(s);
    }
    if (collective_executor_) {
      collective_executor_->StartAbort(s);
    }
    if (cancellation_manager_) {
      cancellation_manager_->StartCancel();
    }
  }

  bool completed = false;
  const size_t ready_size = ready.size();
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
  const GraphView& gview = impl_->gview_;
  const TaggedNode* curr_expensive_node = nullptr;
  for (auto& tagged_node : ready) {
    const NodeItem& item = *gview.node(tagged_node.node->id());
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
    const NodeItem* item = impl_->gview_.node(node_id);
    mutex_lock l(frame->mu);
    frame->GetIteration(iter)->mark_completed(item->pending_id);
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
  const NodeItem& node_item = *impl_->gview_.node(node_id);
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
  const NodeItem& node_item = *impl_->gview_.node(node_id);
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
    const int node_id = node->id();
    PendingCounts::Handle pending_id = impl_->gview_.node(node_id)->pending_id;
    if (iteration->node_state(pending_id) == PendingCounts::PENDING_NOTREADY ||
        iteration->node_state(pending_id) == PendingCounts::PENDING_READY) {
      DumpPendingNodeState(node_id, iteration->input_tensors, false);
    }
  }
  // Then the active nodes.
  for (const Node* node : *nodes) {
    const int node_id = node->id();
    PendingCounts::Handle pending_id = impl_->gview_.node(node_id)->pending_id;
    if (iteration->node_state(pending_id) == PendingCounts::STARTED) {
      DumpActiveNodeState(node_id, iteration->input_tensors);
    }
  }
  // Show all input tensors in use.
  const int total_input_tensors = frame->total_input_tensors;
  size_t total_bytes = 0;
  for (int i = 0; i < total_input_tensors; ++i) {
    const Entry& input = iteration->input_tensors[i];
    const Tensor* tensor = GetTensorValueForDump(input);
    if (tensor->IsInitialized()) {
      LOG(WARNING) << "    Input " << i << ": "
                   << strings::StrCat(
                          "Tensor<type: ", DataTypeString(tensor->dtype()),
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
  Status s = GetNodeAttr(node->attrs(), "frame_name", &enter_name);
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
  s = GetNodeAttr(node->attrs(), "parallel_iterations", &parallel_iters);
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
  const int64 parent_iter = frame->parent_iter;
  if (parent_frame != nullptr) {
    mutex_lock parent_frame_lock(parent_frame->mu);
    // Propagate all the dead exits to the parent frame.
    mutex_lock this_frame_lock(frame->mu);
    for (const Node* node : frame->dead_exits) {
      auto parent_iter_state = parent_frame->GetIteration(parent_iter);
      for (const Edge* e : node->out_edges()) {
        const Node* dst_node = e->dst();

        const auto dst_pending_id =
            impl_->gview_.node(dst_node->id())->pending_id;

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
          if (IsControlTrigger(dst_node)) dst_dead = false;
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
    is_frame_done = frame->CleanupIterations(&impl_->gview_, iter, ready);
  }
  if (is_frame_done) {
    FrameState* parent_frame = frame->parent_frame;
    const int64 parent_iter = frame->parent_iter;
    DeleteFrame(frame, ready);
    if (parent_frame != nullptr) {
      // The completion of frame may cause completions in its parent frame.
      // So clean things up recursively.
      CleanupFramesIterations(parent_frame, parent_iter, ready);
    }
  }
}

void ExecutorState::FrameState::ActivateNodes(const NodeItem* item,
                                              const bool is_dead, int64 iter,
                                              EntryVector* outputs,
                                              TaggedNodeSeq* ready) {
  const GraphView& gview = executor->gview_;
  IterationState* iter_state = GetIteration(iter);
  const size_t num_output_edges = item->num_output_edges;
  const EdgeInfo* edges = item->output_edge_list();
  Entry* input_tensors = iter_state->input_tensors;
  for (size_t out_index = 0; out_index < num_output_edges; out_index++) {
    const EdgeInfo& e = edges[out_index];
    const int dst_id = e.dst_id;
    const NodeItem* dst_item = gview.node(dst_id);
    const PendingCounts::Handle dst_pending_id = dst_item->pending_id;
    const int src_slot = e.output_slot;

    // TODO(yuanbyu): We don't need this if we require the subgraph
    // given to an executor not to contain a sink node.
    if (dst_item->is_sink) continue;

    bool dst_dead = false;
    bool dst_ready = false;
    // True iff this input for dst is needed. We only set this input for
    // dst if this flag is true. This is needed to make the thread safety
    // analysis happy.
    const bool is_control_edge = (src_slot == Graph::kControlSlot);
    bool dst_need_input = !is_control_edge;
    if (dst_item->is_merge) {
      // A merge node is ready if all control inputs have arrived and either
      // a) a live data input becomes available or b) all data inputs are
      // dead. For Merge, pending's LSB is set iff a live data input has
      // arrived.
      if (is_control_edge) {
        iter_state->decrement_pending(dst_pending_id, 2);
        int count = iter_state->pending(dst_pending_id);
        int dead_cnt = iter_state->dead_count(dst_pending_id);
        dst_dead = (dead_cnt == dst_item->num_inputs);
        dst_ready = (count == 0) || ((count == 1) && dst_dead);
      } else {
        if ((*outputs)[src_slot].has_value) {
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
          // TODO(yuanbyu): This is a bit hacky, but a good solution for
          // now.
          iter_state->increment_dead_count(dst_pending_id);
          const int dead_cnt = iter_state->dead_count(dst_pending_id);
          dst_dead = (dead_cnt == dst_item->num_inputs) || item->is_enter;
          dst_ready = (iter_state->pending(dst_pending_id) == 1) && dst_dead;
          dst_need_input = false;
        }
      }
    } else {
      const bool increment_dead =
          (is_dead || (!is_control_edge && !(*outputs)[src_slot].has_value));
      int pending, dead;
      iter_state->adjust_for_activation(dst_pending_id, increment_dead,
                                        &pending, &dead);
      dst_dead = (dead > 0);
      dst_ready = (pending == 0);
    }

    if (dst_need_input) {
      const int dst_slot = e.input_slot;
      const int dst_loc = dst_item->input_start + dst_slot;
      if (e.is_last) {
        input_tensors[dst_loc] = std::move((*outputs)[src_slot]);
      } else {
        input_tensors[dst_loc] = (*outputs)[src_slot];
      }
    }

    // Add dst to the ready queue if it's ready
    if (dst_ready) {
      if (dst_item->is_control_trigger) dst_dead = false;
      ready->push_back(TaggedNode(dst_item->node, this, iter, dst_dead));
      iter_state->outstanding_ops++;
    }
  }
}

void ExecutorState::FrameState::ActivateNexts(const GraphView* gview,
                                              int64 iter,
                                              TaggedNodeSeq* ready) {
  // Propagate the deferred NextIteration nodes to the new iteration.
  for (auto& node_entry : next_iter_roots) {
    const Node* node = node_entry.first;
    const Entry& entry = node_entry.second;
    const bool is_dead = !entry.has_value;
    const NodeItem* item = gview->node(node->id());
    EntryVector outputs{entry};
    ActivateNodes(item, is_dead, iter, &outputs, ready);
  }
  next_iter_roots.clear();
}

void ExecutorState::FrameState::ActivateLoopInvs(const GraphView* gview,
                                                 int64 iter,
                                                 TaggedNodeSeq* ready) {
  // Propagate loop invariants to the new iteration.
  for (auto& node_entry : inv_values) {
    const Node* node = node_entry.first;
    const Entry& entry = node_entry.second;
    const bool is_dead = !entry.has_value;
    const NodeItem* item = gview->node(node->id());
    EntryVector outputs{entry};
    ActivateNodes(item, is_dead, iter, &outputs, ready);
  }
}

void ExecutorState::FrameState::AddLoopInv(const NodeItem* item,
                                           const Entry& entry,
                                           TaggedNodeSeq* ready) {
  // Store this value.
  inv_values.push_back({item->node, entry});

  // Make this value available to all iterations.
  const bool is_dead = !entry.has_value;
  for (int i = 0; i <= iteration_count; ++i) {
    EntryVector outputs{entry};
    ActivateNodes(item, is_dead, i, &outputs, ready);
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

void ExecutorState::FrameState::IncrementIteration(const GraphView* gview,
                                                   TaggedNodeSeq* ready) {
  iteration_count++;
  const int64 next_iter = iteration_count;

  // Initialize the next iteration.
  IterationState* iter_state =
      new IterationState(pending_counts, total_input_tensors);
  SetIteration(next_iter, iter_state);
  num_outstanding_iterations++;
  dead_exits.clear();

  // Activate the successors of the deferred roots in the new iteration.
  ActivateNexts(gview, next_iter, ready);

  // Activate the loop invariants in the new iteration.
  ActivateLoopInvs(gview, next_iter, ready);
}

bool ExecutorState::FrameState::CleanupIterations(const GraphView* gview,
                                                  int64 iter,
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
      IncrementIteration(gview, ready);
    }
  }
  return IsFrameDone();
}

void ExecutorImpl::RunAsync(const Args& args, DoneCallback done) {
  (new ExecutorState(args, this))->RunAsync(std::move(done));
}

}  // namespace

Status NewLocalExecutor(const LocalExecutorParams& params,
                        std::unique_ptr<const Graph> graph,
                        Executor** executor) {
  ExecutorImpl* impl = new ExecutorImpl(params, std::move(graph));
  const Status s = impl->Initialize();
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
  const auto device_type = DeviceType(device->attributes().device_type());
  auto allocator = device->GetAllocator(AllocatorAttributes());
  return CreateOpKernel(device_type, device, allocator, flib, ndef,
                        graph_def_version, kernel);
}

void DeleteNonCachedKernel(OpKernel* kernel) { delete kernel; }

namespace {

class DefaultExecutorRegistrar {
 public:
  DefaultExecutorRegistrar() {
    Factory* factory = new Factory;
    ExecutorFactory::Register("", factory);
    ExecutorFactory::Register("DEFAULT", factory);
  }

 private:
  class Factory : public ExecutorFactory {
    Status NewExecutor(const LocalExecutorParams& params,
                       std::unique_ptr<const Graph> graph,
                       std::unique_ptr<Executor>* out_executor) override {
      Executor* ret = nullptr;
      TF_RETURN_IF_ERROR(NewLocalExecutor(params, std::move(graph), &ret));
      out_executor->reset(ret);
      return Status::OK();
    }
  };
};
static DefaultExecutorRegistrar registrar;

}  // namespace

}  // namespace tensorflow
