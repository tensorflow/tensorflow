#include "tensorflow/core/graph/costmodel.h"

#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace {
const Microseconds kDefaultTimeEstimate(1);
const Microseconds kMinTimeEstimate(1);
}  // namespace

void CostModel::SuppressInfrequent() {
  // Find the median of the non-zero counts, and use half of its value
  // as the cutoff for a "normal" execution mode node.
  if (count_.empty()) return;
  std::vector<int32> non_zero;
  for (auto v : count_) {
    if (v > 0) non_zero.push_back(v);
  }
  const size_t sz = non_zero.size();
  if (sz > 0) {
    std::nth_element(non_zero.begin(), non_zero.begin() + sz / 2,
                     non_zero.end());
    int32 median_value = non_zero[sz / 2];
    min_count_ = median_value / 2;
    VLOG(1) << "num non_zero vals: " << non_zero.size() << " median_value "
            << median_value;
  } else {
    min_count_ = 1;
  }
}

void CostModel::MergeFromLocal(const Graph& g, const CostModel& cm) {
  CHECK(is_global_);
  CHECK(!cm.is_global());
  for (const Node* n : g.nodes()) {
    const int local_id = cm.Id(n);
    const int global_id = Id(n);
    if (local_id < 0 || global_id < 0) continue;
    Ensure(global_id);
    count_[global_id] += cm.count_[local_id];
    time_[global_id] += cm.time_[local_id];
    int num_slots = cm.slot_bytes_[local_id].size();
    if (num_slots > 0) {
      if (slot_bytes_[global_id].size() == 0) {
        slot_bytes_[global_id].resize(num_slots);
      } else {
        CHECK_EQ(num_slots, slot_bytes_[global_id].size());
      }
      for (int s = 0; s < num_slots; ++s) {
        slot_bytes_[global_id][s] += cm.slot_bytes_[local_id][s];
      }
    }
  }
}

void CostModel::MergeFromGlobal(const CostModel& cm) {
  CHECK(is_global_);
  CHECK_EQ(true, cm.is_global());
  const int num_nodes = cm.count_.size();
  Ensure(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    count_[i] += cm.count_[i];
    time_[i] += cm.time_[i];
    int num_slots = cm.slot_bytes_[i].size();
    if (num_slots > 0) {
      if (slot_bytes_[i].size() == 0) {
        slot_bytes_[i].resize(num_slots);
      } else {
        CHECK_EQ(num_slots, slot_bytes_[i].size());
      }
      for (int s = 0; s < num_slots; ++s) {
        slot_bytes_[i][s] += cm.slot_bytes_[i][s];
      }
    }
  }
}

void CostModel::MergeFromStats(const NodeNameToCostIdMap& map,
                               const StepStats& ss) {
  CHECK(is_global_);
  for (auto& ds : ss.dev_stats()) {
    for (auto& ns : ds.node_stats()) {
      NodeNameToCostIdMap::const_iterator iter = map.find(ns.node_name());
      // We don't keep stats for nodes not in the global graph, i.e.
      // copy/send/recv nodes, feed/fetch, etc.
      if (iter == map.end()) continue;
      int32 global_id = iter->second;
      Ensure(global_id);
      int64 elapsed_micros = ns.op_end_rel_micros() - ns.op_start_rel_micros();
      count_[global_id]++;
      time_[global_id] += elapsed_micros;
      for (auto& no : ns.output()) {
        int si = no.slot();
        if (static_cast<size_t>(si) >= slot_bytes_[global_id].size()) {
          slot_bytes_[global_id].resize(1 + si);
        }
        slot_bytes_[global_id][si] +=
            no.tensor_description().allocation_description().requested_bytes();
      }
    }
  }
}

void CostModel::Ensure(int id) {
  if (slot_bytes_.size() <= static_cast<size_t>(id)) {
    slot_bytes_.resize(id + 1);
    count_.resize(id + 1);
    time_.resize(id + 1);
  }
}

void CostModel::SetNumOutputs(const Node* node, int num_outputs) {
  const int id = Id(node);
  if (id < 0) return;
  Ensure(id);
  auto perslot = &slot_bytes_[id];
  if (perslot->size() > 0) {
    CHECK_EQ(num_outputs, perslot->size()) << "Cannot resize slot_bytes, node="
                                           << node->name();
  } else {
    perslot->resize(num_outputs, Bytes(-1));
  }
}

void CostModel::RecordCount(const Node* node, int count) {
  const int id = Id(node);
  if (id < 0) return;
  CHECK_LT(id, slot_bytes_.size());
  count_[id] += count;
}

int32 CostModel::TotalCount(const Node* node) const {
  const int id = Id(node);
  if (id < 0) return 0;
  return (static_cast<size_t>(id) < slot_bytes_.size()) ? count_[id] : 0;
}

void CostModel::RecordSize(const Node* node, int slot, Bytes bytes) {
  const int id = Id(node);
  if (id < 0) return;
  CHECK_LT(id, slot_bytes_.size());
  auto perslot = &slot_bytes_[id];
  CHECK_LT(slot, perslot->size());
  auto v = &(*perslot)[slot];
  if (*v >= 0) {
    *v += bytes;
  } else {
    *v = bytes;
  }
}

Bytes CostModel::TotalBytes(const Node* node, int slot) const {
  const int id = Id(node);
  if (id < 0 || static_cast<size_t>(id) >= slot_bytes_.size() ||
      slot_bytes_[id].size() <= static_cast<size_t>(slot)) {
    return Bytes(0);
  }
  return slot_bytes_[id][slot];
}

Bytes CostModel::SizeEstimate(const Node* node, int slot) const {
  int32 count = TotalCount(node);
  if (count < min_count_) return Bytes(0);
  return TotalBytes(node, slot) / std::max(1, TotalCount(node));
}

void CostModel::RecordTime(const Node* node, Microseconds time) {
  const int id = Id(node);
  if (id < 0) return;
  DCHECK(node->IsOp()) << node->DebugString();
  Ensure(id);
  time_[id] += time;
}

Microseconds CostModel::TotalTime(const Node* node) const {
  DCHECK(node->IsOp()) << node->DebugString();
  const int id = Id(node);
  if (id < 0 || static_cast<size_t>(id) >= time_.size() ||
      time_[id] < Microseconds(0)) {
    return Microseconds(0);
  }
  return time_[id];
}

Microseconds CostModel::TimeEstimate(const Node* node) const {
  int32 count = TotalCount(node);
  if (count <= min_count_) return kMinTimeEstimate;
  return std::max(kMinTimeEstimate, TotalTime(node) / std::max(1, count));
}

void CostModel::CheckInitialized(const Graph& graph) const {
  for (const Node* n : graph.nodes()) {
    if (n->IsOp()) {
      CHECK(static_cast<size_t>(n->id()) < time_.size() &&
            time_[n->id()] >= Microseconds(0))
          << ": no time estimate for " << n->DebugString();

      CHECK(static_cast<size_t>(n->id()) < slot_bytes_.size())
          << ": no size estimate for " << n->DebugString();
      const auto& perslot = slot_bytes_[n->id()];
      for (size_t i = 0; i < perslot.size(); i++) {
        CHECK_GE(perslot[i], Bytes(0)) << ": no size estimate for output# " << i
                                       << " of " << n->DebugString();
      }
    }
  }
}

Microseconds CostModel::CopyTimeEstimate(Bytes b, double network_latency_millis,
                                         double estimated_gbps) {
  // TODO(jeff,sanjay): estimate cost based on bandwidth along the
  // communication path and the type of transport we are using between
  // devices.
  //
  // We assume the copy time follows a linear model:
  //    copy_time = copy_bytes / rate + min_time
  int64 copy_bytes = b.value();
  const double bytes_per_usec = estimated_gbps * 1000.0 / 8;
  const double min_micros = network_latency_millis * 1000.0;
  return Microseconds(
      static_cast<int64>(copy_bytes / bytes_per_usec + min_micros));
}

Microseconds CostModel::ComputationTimeEstimate(int64 math_ops) {
  // TODO(jeff,sanjay): Eventually we should pass in the type of device
  // (GPU vs. CPU) and use that to affect the estimate.

  // We estimate the microseconds using that value.  We divide
  // by 1000 to convert the madd number into microseconds (assuming
  // roughly 1000 madds per microsecond (~1 GHz for one core)).
  return Microseconds(math_ops / 1000);
}

// ----------------------------------------------------------------------------
// InitCostModel
// ----------------------------------------------------------------------------

namespace {

static void AddNodesToCostModel(const Graph& g, CostModel* cost_model) {
  for (Node* n : g.nodes()) {
    const int num_outputs = n->num_outputs();
    cost_model->SetNumOutputs(n, num_outputs);
    for (int output = 0; output < num_outputs; output++) {
      // Set up an initial bogus estimate for the node's outputs
      cost_model->RecordSize(n, output, Bytes(1));
    }
  }
}

static void AssignSizes(const Graph& g, CostModel* cost_model) {
  for (const Edge* e : g.edges()) {
    // Skip if it is a control edge.
    if (e->IsControlEdge()) {
      continue;
    }
    Node* src = e->src();

    // TODO(josh11b): Get an estimate from the Op
    Bytes size(1);
    cost_model->RecordSize(src, e->src_output(), size);
  }
}

// This generates an extremely simple initial guess for the
// computation cost of each node. For ordinary Ops, its value should quickly
// be wiped out by the real runtime measurements.  For other Ops we don't
// actually generate measurements, so suppression of infrequent Ops ends up
// giving them 0 costs.  So, this is not of much consequence except perhaps
// in tests.
static Microseconds TimeEstimateForNode(CostModel* cost_model, Node* n) {
  CHECK(n->IsOp());
  VLOG(2) << "Node " << n->id() << ": " << n->name()
          << " type_string: " << n->type_string();
  if (IsConstant(n) || IsVariable(n)) {
    return Microseconds(0);
  }
  return kDefaultTimeEstimate;
}

static void EstimateComputationCosts(const Graph& g, CostModel* cost_model) {
  for (Node* n : g.nodes()) {
    if (!n->IsOp()) continue;
    cost_model->RecordTime(n, TimeEstimateForNode(cost_model, n));
  }
}

}  // namespace

void CostModel::InitFromGraph(const Graph& g) {
  AddNodesToCostModel(g, this);
  AssignSizes(g, this);
  EstimateComputationCosts(g, this);
  CheckInitialized(g);
}

void CostModel::WriteToLog() {
  LOG(INFO) << " min_count_=" << min_count_;
  for (size_t i = 0; i < count_.size(); ++i) {
    LOG(INFO) << "Node " << i << " count " << count_[i] << " total time "
              << time_[i] << " avg time "
              << (time_[i] / (std::max(1, count_[i])));
  }
}

}  // namespace tensorflow
