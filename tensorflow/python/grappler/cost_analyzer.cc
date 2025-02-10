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

#include "tensorflow/python/grappler/cost_analyzer.h"

#include <iomanip>
#include "tensorflow/core/grappler/costs/utils.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace grappler {

CostAnalyzer::CostAnalyzer(const GrapplerItem& item, Cluster* cluster,
                           const string& suffix)
    : item_(&item),
      measure_estimator_(cluster, 10, 0),
      analytical_estimator_(cluster, /*use_static_shapes=*/false,
                            /*use_aggressive_shape_inference=*/true),
      suffix_(suffix) {}

absl::Status CostAnalyzer::GenerateReport(std::ostream& os,
                                          bool per_node_report, bool verbose) {
  GatherCosts();
  PreprocessCosts();
  AnalyzeCosts();
  PrintAnalysis(os, per_node_report, verbose);
  return absl::OkStatus();
}

void CostAnalyzer::PredictCosts(CostEstimator* cost_estimator,
                                CostGraphDef* cost_graph, int64_t* total_time) {
  TF_CHECK_OK(cost_estimator->Initialize(*item_));
  RunMetadata run_metadata;
  Costs costs;
  const absl::Status status =
      cost_estimator->PredictCosts(item_->graph, &run_metadata, &costs);
  if (cost_graph) {
    cost_graph->Swap(run_metadata.mutable_cost_graph());
  }
  *total_time = costs.execution_time.count();
  if (!status.ok()) {
    LOG(ERROR) << "Could not estimate the cost for item " << item_->id << ": "
               << status.message();
    return;
  }
}

void CostAnalyzer::GatherCosts() {
  CostGraphDef cost_graph_measured;
  PredictCosts(&measure_estimator_, &cost_graph_measured,
               &total_time_measured_);
  VLOG(1) << "Graph size: " << item_->graph.node_size();
  VLOG(1) << "cost_graph_measured size: " << cost_graph_measured.node_size();

  CostGraphDef cost_graph_analytical;
  PredictCosts(&analytical_estimator_, &cost_graph_analytical,
               &total_time_analytical_);
  VLOG(1) << "cost_graph_analytical size: "
          << cost_graph_analytical.node_size();

  CostGraphDef cost_graph_analytical_filtered;
  CostGraphDef cost_graph_measured_filtered;
  std::map<string, const CostGraphDef_Node*> measured_nodes;
  for (const auto& node : cost_graph_measured.node()) {
    measured_nodes[node.name()] = &node;
  }
  for (const auto& node : cost_graph_analytical.node()) {
    auto it = measured_nodes.find(node.name());
    // Filter the nodes that are not the cost nodes returned by
    // MeasuringCostEstimator.
    if (it == measured_nodes.end()) {
      continue;
    }
    auto added_node_analytical = cost_graph_analytical_filtered.add_node();
    auto added_node_measured = cost_graph_measured_filtered.add_node();
    *added_node_analytical = node;
    *added_node_measured = *(it->second);
  }
  VLOG(1) << "cost_graph_analytical_filtered size: "
          << cost_graph_analytical_filtered.node_size();

  // TODO(yaozhang): add a test to make sure that op_perf_analytical_ and
  // op_perf_ cover the same set of nodes.
  op_perf_analytical_ = CostGraphToOpPerformanceData(
      cost_graph_analytical_filtered, item_->graph);
  op_perf_ =
      CostGraphToOpPerformanceData(cost_graph_measured_filtered, item_->graph);
}

void CostAnalyzer::PreprocessCosts() {
  for (int i = 0; i < op_perf_.op_performance_size(); i++) {
    OpPerformance* perf = op_perf_.mutable_op_performance(i);
    const OpPerformance& analytical = op_perf_analytical_.op_performance(i);
    perf->set_compute_time(analytical.compute_time());
    perf->set_memory_time(analytical.memory_time());
    double measured_cost = perf->compute_cost();

    double analytical_compute_cost = analytical.compute_time();
    if (analytical_compute_cost == 0) {
      // Negative infinity indicates unavailable data.
      perf->set_compute_efficiency(-INFINITY);
    } else {
      perf->set_compute_efficiency(analytical_compute_cost / measured_cost);
    }

    double analytical_memory_cost = analytical.memory_time();
    if (analytical_memory_cost == 0) {
      // Negative infinity indicates unavailable data.
      perf->set_memory_efficiency(-INFINITY);
    } else {
      perf->set_memory_efficiency(analytical_memory_cost / measured_cost);
    }
  }
}

void CostAnalyzer::SortOpsByTime(std::map<string, OpPerfSummary> ops) {
  for (const auto& op : ops) {
    ops_.push_back(op.second);
  }
  struct CompareByTime {
    bool operator()(const OpPerfSummary& a, const OpPerfSummary& b) const {
      return a.time > b.time;
    }
  };
  std::stable_sort(ops_.begin(), ops_.end(), CompareByTime());
}

void CostAnalyzer::AnalyzeCosts() {
  std::map<string, OpPerfSummary> ops;
  for (const auto& op_perf : op_perf_.op_performance()) {
    string op_name = op_perf.op().op();
    ops[op_name].count++;
    ops[op_name].time += op_perf.compute_cost();
    ops[op_name].compute_time += op_perf.compute_time();
    ops[op_name].memory_time += op_perf.memory_time();
    ops[op_name].time_upper += op_perf.compute_time() + op_perf.memory_time();
    ops[op_name].time_lower +=
        std::max(op_perf.compute_time(), op_perf.memory_time());
    ops[op_name].name = op_name;
  }
  SortOpsByTime(ops);

  total_time_measured_serialized_ = 0;
  total_time_analytical_upper_ = 0;
  total_time_analytical_lower_ = 0;
  for (const auto& op : ops_) {
    total_time_measured_serialized_ += op.time;
    total_time_analytical_upper_ += op.time_upper;
    total_time_analytical_lower_ += op.time_lower;
  }
}

void CostAnalyzer::PrintAnalysis(std::ostream& os, bool per_node_report,
                                 bool verbose) const {
  os << std::endl;
  os << std::left << std::setw(50)
     << "Total time measured in ns (serialized): " << std::right
     << std::setw(20) << total_time_measured_serialized_ << std::endl;
  os << std::left << std::setw(50)
     << "Total time measured in ns (actual): " << std::right << std::setw(20)
     << total_time_measured_ << std::endl;
  os << std::left << std::setw(50)
     << "Total time analytical in ns (upper bound): " << std::right
     << std::setw(20) << total_time_analytical_upper_ << std::endl;
  os << std::left << std::setw(50)
     << "Total time analytical in ns (lower bound): " << std::right
     << std::setw(20) << total_time_analytical_lower_ << std::endl;
  double efficiency_upper = static_cast<double>(total_time_analytical_upper_) /
                            static_cast<double>(total_time_measured_);
  os << std::left << std::setw(50)
     << "Overall efficiency (analytical upper/actual): " << std::right
     << std::setw(20) << efficiency_upper << std::endl;
  double efficiency_lower = static_cast<double>(total_time_analytical_lower_) /
                            static_cast<double>(total_time_measured_);
  os << std::left << std::setw(50)
     << "Overall efficiency (analytical lower/actual): " << std::right
     << std::setw(20) << efficiency_lower << std::endl;
  os << std::endl;

  int width = 35;
  int width_narrow = 15;
  int width_wide = 20;
  os << std::setw(width + 1) << "Op,";
  os << std::setw(width_narrow + 1) << "Count,";
  os << std::setw(width_wide + 1) << "Measured time (ns),";
  os << std::setw(width_narrow + 2) << "Time percent,";
  os << std::setw(width_narrow + 2) << "Acc percent,";
  os << std::setw(width_wide + 1) << "Analytical upper,";
  os << std::setw(width_wide + 1) << "Analytical lower,";
  os << std::setw(width_narrow + 2) << "Overall eff";
  os << std::setw(width_narrow + 2) << "Compute eff";
  os << std::setw(width_narrow + 2) << "Memory eff" << std::endl;
  float acc_percent = 0;
  for (const auto& op : ops_) {
    double percent = static_cast<double>(op.time) /
                     static_cast<double>(total_time_measured_serialized_);
    double eff =
        static_cast<double>(op.time_upper) / static_cast<double>(op.time);
    double compute_eff =
        static_cast<double>(op.compute_time) / static_cast<double>(op.time);
    double memory_eff =
        static_cast<double>(op.memory_time) / static_cast<double>(op.time);
    os << std::setw(width) << op.name << ",";
    os << std::setw(width_narrow) << op.count << ",";
    os << std::setw(width_wide) << op.time << ",";
    os << std::setw(width_narrow) << std::setprecision(2) << percent * 100
       << "%,";
    acc_percent += percent;
    os << std::setw(width_narrow) << std::setprecision(2) << acc_percent * 100
       << "%,";
    os << std::setw(width_wide) << op.time_upper << ",";
    os << std::setw(width_wide) << op.time_lower << ",";
    os << std::setw(width_narrow) << std::setprecision(2) << eff * 100 << "%,";
    os << std::setw(width_narrow) << std::setprecision(2) << compute_eff * 100
       << "%,";
    os << std::setw(width_narrow) << std::setprecision(2) << memory_eff * 100
       << "%,";
    os << std::endl;
  }
  os << std::endl;

  if (per_node_report) {
    if (verbose) {
      os << "Below is the full per-node report:" << std::endl;
      os << op_perf_.DebugString();
    } else {
      os << "Below is the per-node report summary:" << std::endl;
      int width = 35;
      int width_narrow = 15;
      int width_wide = 20;
      os << std::setw(width + 1) << "Op,";
      os << std::setw(width_wide + 1) << "Measured time (ns),";
      os << std::setw(width_wide + 1) << "Compute time (ns),";
      os << std::setw(width_wide + 1) << "Memory time (ns),";
      os << std::setw(width_narrow + 2) << "Compute eff,";
      os << std::setw(width_narrow + 2) << "Memory eff,";
      os << "    Inputs" << std::endl;
      for (int i = 0; i < op_perf_.op_performance_size(); i++) {
        const auto& perf = op_perf_.op_performance(i);
        string op_name = perf.op().op();
        os << std::setw(width) << op_name << ",";
        os << std::setw(width_wide) << perf.compute_cost() << ",";
        os << std::setw(width_wide) << perf.compute_time() << ",";
        os << std::setw(width_wide) << perf.memory_time() << ",";
        os << std::setw(width_narrow) << std::setprecision(2)
           << perf.compute_efficiency() * 100 << "%,";
        os << std::setw(width_narrow) << std::setprecision(2)
           << perf.memory_efficiency() * 100 << "%,";
        os << "    [";
        for (int j = 0; j < perf.op().inputs_size(); j++) {
          const auto& shape = perf.op().inputs(j).shape();
          if (shape.dim_size() > 0) {
            os << "(";
            std::vector<int> dims;
            for (int k = 0; k < shape.dim_size(); k++) {
              os << shape.dim(k).size();
              if (k < shape.dim_size() - 1) {
                os << ", ";
              }
            }
            os << ")";
            if (j < perf.op().inputs_size() - 1) {
              os << ", ";
            }
          }
        }
        os << "]" << std::endl;
      }
      os << std::endl;
    }
  }
}
}  // end namespace grappler
}  // end namespace tensorflow
