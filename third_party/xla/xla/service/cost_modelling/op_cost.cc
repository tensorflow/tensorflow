/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/cost_modelling/op_cost.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla {
namespace {

// Used in LOG(INFO) statements for analysis logging.
constexpr absl::string_view kLoggingAnalysisId = "COST_LOGGING";

}  // namespace

CostMetricId CostMetricId::LatencySeconds(const HloInstruction& instruction) {
  return CostMetricId(MetricType::kLatencySeconds, instruction, std::nullopt,
                      std::nullopt);
}

CostMetricId CostMetricId::ComputeSeconds(const HloInstruction& instruction) {
  return CostMetricId(MetricType::kComputeSeconds, instruction, std::nullopt,
                      std::nullopt);
}

CostMetricId CostMetricId::OperandBytesAccessed(
    const HloInstruction& instruction, int64_t operand_num,
    const ShapeIndex& shape_index) {
  return CostMetricId(MetricType::kOperandBytesAccessed, instruction,
                      operand_num, shape_index);
}

CostMetricId CostMetricId::OutputBytesAccessed(
    const HloInstruction& instruction, const ShapeIndex& shape_index) {
  return CostMetricId(MetricType::kOutputBytesAccessed, instruction,
                      std::nullopt, shape_index);
}

CostMetricId CostMetricId::TotalBytesAccessed(
    const HloInstruction& instruction) {
  return CostMetricId(MetricType::kTotalBytesAccessed, instruction,
                      std::nullopt, std::nullopt);
}

std::vector<std::string> CostMetricId::LoggingColumnNames() {
  return {
      "metric_id",        "metric_type", "module_name", "instruction_name",
      "instruction_type", "operand_num", "shape_index",
  };
}

bool CostMetricId::operator==(const CostMetricId& other) const {
  return MakeTuple() == other.MakeTuple();
}

int64_t CostMetricId::operand_num() const {
  CHECK(operand_num_.has_value());
  return *operand_num_;
}

const ShapeIndex& CostMetricId::shape_index() const {
  CHECK(shape_index_.has_value());
  return *shape_index_;
}

std::vector<std::string> CostMetricId::LoggingColumns() const {
  return {Identifier(),         MetricTypeName(),
          ModuleName(),         std::string(instruction_->name()),
          InstructionTypeStr(), OperandNumStr(),
          ShapeIndexStr()};
}

std::string CostMetricId::ToString() const {
  return absl::StrCat(
      "<type=", MetricTypeName(), ",computation=", ComputationName(),
      ",instruction=", instruction_->name(), ",operand_num=", OperandNumStr(),
      ",shape_index=", ShapeIndexStr(), ">");
}

CostMetricId::CostMetricId(MetricType type, const HloInstruction& instruction,
                           std::optional<int64_t> operand_num,
                           std::optional<ShapeIndex> shape_index)
    : type_(type),
      instruction_(&instruction),
      operand_num_(operand_num),
      shape_index_(std::move(shape_index)) {}

std::string CostMetricId::Identifier() const {
  std::string result;

  absl::Base64Escape(
      absl::StrJoin({absl::StrCat(static_cast<uint8_t>(type_)), ModuleName(),
                     absl::StrCat(instruction_->unique_id()), OperandNumStr(),
                     ShapeIndexStr()},
                    ","),
      &result);

  return result;
}

std::string CostMetricId::MetricTypeName() const {
  switch (type_) {
    case MetricType::kLatencySeconds:
      return "latency-seconds";
    case MetricType::kComputeSeconds:
      return "compute-seconds";
    case MetricType::kOperandBytesAccessed:
      return "operand-bytes-accessed";
    case MetricType::kOutputBytesAccessed:
      return "output-bytes-accessed";
    case MetricType::kTotalBytesAccessed:
      return "total-bytes-accessed";
  }
}

std::string CostMetricId::ModuleName() const {
  if (instruction_->GetModule()) {
    return instruction_->GetModule()->name();
  }
  return "-";
}

std::string CostMetricId::ComputationName() const {
  if (instruction_->parent()) {
    return std::string(instruction_->parent()->name());
  }
  return "-";
}

std::string CostMetricId::InstructionTypeStr() const {
  if (instruction_->opcode() == HloOpcode::kCustomCall) {
    return absl::StrCat(HloOpcodeString(instruction_->opcode()), "-",
                        instruction_->custom_call_target());
  }

  if (instruction_->opcode() == HloOpcode::kFusion) {
    return absl::StrCat(HloOpcodeString(instruction_->opcode()), "-",
                        ::xla::ToString(instruction_->fusion_kind()));
  }

  return std::string(HloOpcodeString(instruction_->opcode()));
}

std::string CostMetricId::OperandNumStr() const {
  if (operand_num_.has_value()) {
    return absl::StrCat(*operand_num_);
  }
  return "-";
}

std::string CostMetricId::ShapeIndexStr() const {
  if (shape_index_.has_value()) {
    return shape_index_->ToString();
  }
  return "-";
}

CostMetricId::Tuple CostMetricId::MakeTuple() const {
  return std::make_tuple(type_, instruction_, operand_num_, shape_index_);
}

CostValue CostValue::MakeNotFound() { return CostValue(Type::kNotFound, 0.0); }

CostValue CostValue::MakeError() { return CostValue(Type::kError, 0.0); }

CostValue CostValue::MakeValue(double value) {
  return CostValue(Type::kOk, value);
}

bool CostValue::operator==(const CostValue& other) const {
  return MakeTuple() == other.MakeTuple();
}

double CostValue::value() const {
  CHECK(type_ == Type::kOk);
  return value_;
}

std::string CostValue::ToString() const {
  switch (type_) {
    case Type::kNotFound:
      return "nf";
    case Type::kError:
      return "err";
    case Type::kOk:
      return absl::StrCat(value_);
  }
}

OpCostManager::CalculationNode::Result::Result(bool track_calculation_details)
    : track_calculation_details_(track_calculation_details) {}

OpCostManager::CalculationNode::Result::Result(bool track_calculation_details,
                                               absl::string_view calculator,
                                               CostValue value)
    : track_calculation_details_(track_calculation_details) {
  AddCalculatorResult(calculator, value, /*set_final_value=*/value.IsOk());
}

std::string OpCostManager::CalculationNode::Result::ToString() const {
  std::string str;
  if (HasValue()) {
    absl::StrAppend(&str, "Result(value=", Value(), ", source=", ValueSource());
  } else {
    absl::StrAppend(&str, "Result(value=missing");
  }

  if (track_calculation_details_) {
    absl::StrAppend(&str, ", calculator_value_map=[",
                    absl::StrJoin(calculator_to_value_map_, ", ",
                                  [](std::string* out, const auto& pair) {
                                    absl::StrAppend(out, pair.first, ": ",
                                                    pair.second.ToString());
                                  }),
                    "]");
  }

  absl::StrAppend(&str, ")");

  return str;
}

bool OpCostManager::CalculationNode::Result::Merge(const Result& other,
                                                   bool merge_final_value) {
  bool final_value_added = false;
  if (merge_final_value && !HasValue() && other.HasValue()) {
    AddCalculatorResult(other.ValueSource(),
                        CostValue::MakeValue(other.Value()),
                        /*set_final_value*/ true);
    final_value_added = true;
  }

  for (const auto& [calculator, value] : other.calculator_to_value_map_) {
    if (final_value_added && other.ValueSource() == calculator) {
      // We already added this calculator above.
      continue;
    }
    AddCalculatorResult(calculator, value, /*set_final_value=*/false);
  }

  return final_value_added;
}

bool OpCostManager::CalculationNode::Result::HasValue() const {
  return value_.has_value();
}

double OpCostManager::CalculationNode::Result::Value() const {
  CHECK(value_.has_value());
  return *value_;
}

std::string OpCostManager::CalculationNode::Result::ValueSource() const {
  CHECK(value_source_.has_value());
  return *value_source_;
}

std::string OpCostManager::CalculationNode::Result::GetCalculatorResult(
    absl::string_view calculator_name) const {
  auto it = calculator_to_value_map_.find(calculator_name);
  if (it == calculator_to_value_map_.end()) {
    return "na";
  }
  return it->second.ToString();
}

const OpCostManager::CalculationNode::Result::CalculatorMapTy&
OpCostManager::CalculationNode::Result::GetCalculatorValueMap() const {
  return calculator_to_value_map_;
}

void OpCostManager::CalculationNode::Result::AddCalculatorResult(
    absl::string_view calculator_name, CostValue value, bool set_final_value) {
  if (set_final_value) {
    value_ = value.value();
    value_source_ = calculator_name;
  }

  if (!track_calculation_details_) {
    return;
  }

  CHECK(calculator_to_value_map_.insert({std::string(calculator_name), value})
            .second);
}

namespace {

// Implementation for leaf calculation nodes.
class CalculationLeaf : public OpCostManager::CalculationNode {
 public:
  // If enable_cache is true, the leaf node will cache the MetricCalculators
  // it creates per HLO instruction.
  CalculationLeaf(absl::string_view name,
                  std::unique_ptr<OpCostCalculator> op_cost_calculator,
                  bool enable_cache)
      : name_(name),
        op_cost_calculator_(std::move(op_cost_calculator)),
        enable_cache_(enable_cache) {}

  ~CalculationLeaf() override = default;

  Result GetMetricValue(bool track_calculation_details,
                        const CostMetricId& metric_id) override {
    MetricCalculator* metric_calculator = nullptr;

    // Check the calculator cost cache.
    if (enable_cache_) {
      auto it = cached_costs_.find(&metric_id.instruction());
      if (it != cached_costs_.end()) {
        metric_calculator = it->second.get();
        VLOG(4) << "Found op cost for instruction "
                << metric_id.instruction().name() << " in the " << name_
                << " cache";
      }
    }

    // If we didn't find an op cost in the cache, calculate it, and update the
    // cache (if enabled).
    std::unique_ptr<MetricCalculator> metric_calculator_storage;
    if (!metric_calculator) {
      metric_calculator_storage =
          op_cost_calculator_->CreateMetricCalculator(metric_id.instruction());
      metric_calculator = metric_calculator_storage.get();
      if (enable_cache_) {
        CHECK(cached_costs_
                  .insert({&metric_id.instruction(),
                           std::move(metric_calculator_storage)})
                  .second);
        VLOG(4) << "Added op cost for instruction "
                << metric_id.instruction().name() << " to the " << name_
                << " cache";
      }
    }

    // Get the CostValue.
    CostValue cost_value = metric_calculator->Calculate(metric_id);

    VLOG(2) << name_ << " calculated a value of " << cost_value.ToString()
            << " for " << metric_id.ToString();

    return Result(track_calculation_details, name_, cost_value);
  }

  absl::string_view Name() const override { return name_; }

  std::vector<std::string> LeafCalculatorNames() const override {
    return {name_};
  }

 private:
  std::string name_;
  std::unique_ptr<OpCostCalculator> op_cost_calculator_;
  bool enable_cache_;
  absl::flat_hash_map<const HloInstruction*, std::unique_ptr<MetricCalculator>>
      cached_costs_;
};

// Implementation for delegation calculation nodes.
class DelegationCalculationNode : public OpCostManager::CalculationNode {
 public:
  DelegationCalculationNode(
      absl::string_view name,
      std::vector<std::unique_ptr<OpCostManager::CalculationNode>> children,
      DelegationOrderFn delegation_order_fn)
      : name_(name), children_(std::move(children)) {
    if (delegation_order_fn) {
      delegation_order_fn_ = std::move(delegation_order_fn);
    } else {
      size_t num_children = children_.size();
      delegation_order_fn_ = [num_children](const HloInstruction& instruction,
                                            bool enable_analysis_logging) {
        DelegationInfo delegation_info;
        delegation_info.order.reserve(num_children);
        for (CalculatorIndex i = 0; i < num_children; ++i) {
          delegation_info.order.push_back(i);
        }
        return delegation_info;
      };
    }
  }

  ~DelegationCalculationNode() override = default;

  Result GetMetricValue(bool track_calculation_details,
                        const CostMetricId& metric_id) override {
    DelegationInfo delegation_info = delegation_order_fn_(
        metric_id.instruction(),
        /*enable_analysis_logging=*/track_calculation_details);
    Result final_result(track_calculation_details);
    for (CalculatorIndex calculator_index : delegation_info.order) {
      CHECK_LT(calculator_index, children_.size());
      VLOG(3) << name_ << " delegating to "
              << children_[calculator_index]->Name() << " to compute "
              << metric_id.ToString();
      if (final_result.Merge(children_[calculator_index]->GetMetricValue(
                                 track_calculation_details, metric_id),
                             /*merge_final_value=*/true)) {
        VLOG(3) << name_ << " selecting the value from "
                << children_[calculator_index]->Name() << " for metric "
                << metric_id.ToString();
        if (!track_calculation_details) {
          break;
        }
      }
    }

    // Go through the remaining calculators for logging purposes.
    if (track_calculation_details) {
      for (CalculatorIndex calculator_index :
           delegation_info.additional_calculators_to_log) {
        CHECK_LT(calculator_index, children_.size());
        VLOG(3) << name_ << " asking " << children_[calculator_index]->Name()
                << " to compute " << metric_id.ToString()
                << " for analysis logging";
        final_result.Merge(children_[calculator_index]->GetMetricValue(
                               track_calculation_details, metric_id),
                           /*merge_final_value=*/false);
      }
    }

    return final_result;
  }

  absl::string_view Name() const override { return name_; }

  std::vector<std::string> LeafCalculatorNames() const override {
    std::vector<std::string> result;
    for (const auto& child : children_) {
      std::vector<std::string> child_names = child->LeafCalculatorNames();
      result.insert(result.end(), child_names.begin(), child_names.end());
    }
    return result;
  }

 private:
  DelegationCalculationNode() = delete;

  std::string name_;
  std::vector<std::unique_ptr<OpCostManager::CalculationNode>> children_;
  DelegationOrderFn delegation_order_fn_;
};

}  // namespace

std::unique_ptr<OpCostManager::CalculationNode>
OpCostManager::CalculationNode::CreateLeaf(
    absl::string_view name, std::unique_ptr<OpCostCalculator> calculator,
    bool enable_cache) {
  return std::make_unique<CalculationLeaf>(name, std::move(calculator),
                                           enable_cache);
}

std::unique_ptr<OpCostManager::CalculationNode>
OpCostManager::CalculationNode::CreateDelegationNode(
    absl::string_view name,
    std::vector<std::unique_ptr<OpCostManager::CalculationNode>> children,
    DelegationOrderFn delegation_order_fn) {
  return std::make_unique<DelegationCalculationNode>(
      name, std::move(children), std::move(delegation_order_fn));
}

OpCostManager::OpCostManager(Options options,
                             std::unique_ptr<CalculationNode> root)
    : options_(std::move(options)),
      root_(std::move(root)),
      leaf_calculator_names_([&]() {
        std::vector<std::string> calculator_names =
            root_->LeafCalculatorNames();
        absl::c_sort(calculator_names);
        absl::string_view previous = "";
        for (const std::string& calculator_name : calculator_names) {
          CHECK_NE(calculator_name, previous);
          previous = calculator_name;
        }
        return calculator_names;
      }()) {
  LOG_IF(INFO, options_.enable_analysis_logging) << AnalysisLoggingColumns();
}

double OpCostManager::LatencySeconds(const HloInstruction& instruction) {
  return GetMetricValue(CostMetricId::LatencySeconds(instruction));
}

double OpCostManager::ComputeSeconds(const HloInstruction& instruction) {
  return GetMetricValue(CostMetricId::ComputeSeconds(instruction));
}

double OpCostManager::OperandBytesAccessed(const HloInstruction& instruction,
                                           int64_t operand_num,
                                           const ShapeIndex& shape_index) {
  return GetMetricValue(CostMetricId::OperandBytesAccessed(
      instruction, operand_num, shape_index));
}

double OpCostManager::OutputBytesAccessed(const HloInstruction& instruction,
                                          const ShapeIndex& shape_index) {
  return GetMetricValue(
      CostMetricId::OutputBytesAccessed(instruction, shape_index));
}

double OpCostManager::TotalBytesAccessed(const HloInstruction& instruction) {
  return GetMetricValue(CostMetricId::TotalBytesAccessed(instruction));
}

double OpCostManager::GetMetricValue(const CostMetricId& metric_id) {
  // Check the metric cache.
  if (options_.enable_cache) {
    auto it = metric_cache_.find(metric_id);
    if (it != metric_cache_.end()) {
      VLOG(4) << "Found cost for " << metric_id.ToString()
              << " in the OpCostManager cache";
      VLOG(1) << "Cost for " << metric_id.ToString() << " is " << it->second;
      return it->second;
    }
  }

  VLOG(3) << "OpCostManager delegating to " << root_->Name() << " to compute "
          << metric_id.ToString();
  CalculationNode::Result result =
      root_->GetMetricValue(options_.enable_analysis_logging, metric_id);
  // If users don't want to crash, they should register a calculator that
  // computes a default cost.
  LOG_IF(FATAL, !result.HasValue())
      << "Unable to compute a cost for " << metric_id.ToString();
  if (options_.enable_cache) {
    metric_cache_[metric_id] = result.Value();
    VLOG(4) << "Added cost for " << metric_id.ToString()
            << " to the OpCostManager cache";
  }

  LOG_IF(INFO, options_.enable_analysis_logging)
      << AnalysisLoggingLine(metric_id, result);

  VLOG(1) << "Cost for " << metric_id.ToString() << " is " << result.Value();
  return result.Value();
}

std::string OpCostManager::AnalysisLoggingColumns() const {
  std::vector<std::string> columns = CostMetricId::LoggingColumnNames();
  columns.push_back("selected_calculator");
  columns.insert(columns.end(), leaf_calculator_names_.begin(),
                 leaf_calculator_names_.end());

  return absl::StrCat(kLoggingAnalysisId, ": ", absl::StrJoin(columns, "\t"));
}

std::string OpCostManager::AnalysisLoggingLine(
    const CostMetricId& metric_id,
    const CalculationNode::Result& result) const {
  std::vector<std::string> columns = metric_id.LoggingColumns();
  columns.push_back(result.HasValue() ? result.ValueSource() : "na");
  for (const std::string& calculator_name : leaf_calculator_names_) {
    columns.push_back(result.GetCalculatorResult(calculator_name));
  }
  return absl::StrCat(kLoggingAnalysisId, ": ", absl::StrJoin(columns, "\t"));
}

HloCostAnalysisWithAcceptState::HloCostAnalysisWithAcceptState(
    std::unique_ptr<HloCostAnalysis> cost_analysis)
    : cost_analysis_storage_(std::move(cost_analysis)),
      cost_analysis_(*cost_analysis_storage_) {}

HloCostAnalysisWithAcceptState::HloCostAnalysisWithAcceptState(
    HloCostAnalysis& cost_analysis)
    : cost_analysis_(cost_analysis) {}

HloCostAnalysis& HloCostAnalysisWithAcceptState::cost_analysis(
    const HloInstruction& instruction) {
  if (!accepted_entry_computation_) {
    CHECK(instruction.GetModule());
    absl::Status status =
        instruction.GetModule()->entry_computation()->Accept(&cost_analysis_);
    LOG_IF(FATAL, !status.ok())
        << "Computation "
        << instruction.GetModule()->entry_computation()->name()
        << " failed to accept HloCostAnalysis: " << status;
    accepted_entry_computation_ = true;
  }

  return cost_analysis_;
}

namespace {

class HloCostAnalysisMetricCalculator : public MetricCalculator {
 public:
  explicit HloCostAnalysisMetricCalculator(const HloCostAnalysis& cost_analysis)
      : cost_analysis_(cost_analysis) {}

  ~HloCostAnalysisMetricCalculator() override = default;

  CostValue Calculate(const CostMetricId& metric_id) override {
    switch (metric_id.type()) {
      case CostMetricId::MetricType::kLatencySeconds: {
        std::vector<double> latencies = {
            // Min latency;
            cost_analysis_.min_latency_seconds(HloCostAnalysis::kFlopsKey),
            // Latency.
            cost_analysis_.optimal_seconds(metric_id.instruction())};
        return CostValue::MakeValue(*absl::c_max_element(latencies));
      }
      case CostMetricId::MetricType::kComputeSeconds: {
        std::vector<double> latencies = {
            // Min latency;
            cost_analysis_.min_latency_seconds(HloCostAnalysis::kFlopsKey),
            // Standard compute latency.
            static_cast<double>(
                cost_analysis_.flop_count(metric_id.instruction())) /
                static_cast<double>(
                    cost_analysis_.per_second_rate(HloCostAnalysis::kFlopsKey)),
            // Transcendental compute latency.
            static_cast<double>(
                cost_analysis_.transcendental_count(metric_id.instruction())) /
                static_cast<double>(cost_analysis_.per_second_rate(
                    HloCostAnalysis::kTranscendentalsKey))};
        return CostValue::MakeValue(*absl::c_max_element(latencies));
      }
      case CostMetricId::MetricType::kOperandBytesAccessed: {
        return CostValue::MakeValue(
            static_cast<double>(cost_analysis_.operand_bytes_accessed(
                metric_id.instruction(), metric_id.operand_num(),
                metric_id.shape_index())));
      }
      case CostMetricId::MetricType::kOutputBytesAccessed: {
        return CostValue::MakeValue(
            static_cast<double>(cost_analysis_.output_bytes_accessed(
                metric_id.instruction(), metric_id.shape_index())));
      }
      case CostMetricId::MetricType::kTotalBytesAccessed: {
        return CostValue::MakeValue(static_cast<double>(
            cost_analysis_.bytes_accessed(metric_id.instruction())));
      }
    };
  }

 private:
  HloCostAnalysisMetricCalculator() = delete;

  const HloCostAnalysis& cost_analysis_;
};

class HloCostAnalysisOpCostCalculator : public OpCostCalculator {
 public:
  explicit HloCostAnalysisOpCostCalculator(
      HloCostAnalysisWithAcceptState& cost_analysis_wrapper)
      : cost_analysis_wrapper_(cost_analysis_wrapper) {}

  ~HloCostAnalysisOpCostCalculator() override = default;

  std::unique_ptr<MetricCalculator> CreateMetricCalculator(
      const HloInstruction& instruction) override {
    return std::make_unique<HloCostAnalysisMetricCalculator>(
        cost_analysis_wrapper_.cost_analysis(instruction));
  }

 private:
  HloCostAnalysisOpCostCalculator() = delete;

  HloCostAnalysisWithAcceptState& cost_analysis_wrapper_;
};

}  // namespace

std::unique_ptr<OpCostCalculator> CreateHloCostAnalysisCalculator(
    HloCostAnalysisWithAcceptState& cost_analysis_wrapper) {
  return std::make_unique<HloCostAnalysisOpCostCalculator>(
      cost_analysis_wrapper);
}

namespace {

class MetricCalculatorWithPostProcessedCostValues : public MetricCalculator {
 public:
  MetricCalculatorWithPostProcessedCostValues(
      std::unique_ptr<MetricCalculator> initial_metric_calculator,
      absl::AnyInvocable<CostValue(const CostMetricId& metric_id,
                                   CostValue cost_value)>& post_process_fn)
      : initial_metric_calculator_(std::move(initial_metric_calculator)),
        post_process_fn_(post_process_fn) {}

  CostValue Calculate(const CostMetricId& metric_id) override {
    return post_process_fn_(metric_id,
                            initial_metric_calculator_->Calculate(metric_id));
  }

 private:
  std::unique_ptr<MetricCalculator> initial_metric_calculator_;
  absl::AnyInvocable<CostValue(const CostMetricId& metric_id,
                               CostValue cost_value)>& post_process_fn_;
};

class OpCostCalculatorWithPostProcessedCostValues : public OpCostCalculator {
 public:
  OpCostCalculatorWithPostProcessedCostValues(
      std::unique_ptr<OpCostCalculator> initial_calculator,
      absl::AnyInvocable<CostValue(const CostMetricId& metric_id,
                                   CostValue cost_value)>
          post_process_fn)
      : initial_calculator_(std::move(initial_calculator)),
        post_process_fn_(std::move(post_process_fn)) {}

  ~OpCostCalculatorWithPostProcessedCostValues() override = default;

  std::unique_ptr<MetricCalculator> CreateMetricCalculator(
      const HloInstruction& instruction) override {
    return std::make_unique<MetricCalculatorWithPostProcessedCostValues>(
        initial_calculator_->CreateMetricCalculator(instruction),
        post_process_fn_);
  }

 protected:
  OpCostCalculatorWithPostProcessedCostValues() = default;

  std::unique_ptr<OpCostCalculator> initial_calculator_;
  absl::AnyInvocable<CostValue(const CostMetricId& metric_id,
                               CostValue cost_value)>
      post_process_fn_;
};

}  // namespace

std::unique_ptr<OpCostCalculator> CreateCalculatorWithPostProcessedCostValues(
    std::unique_ptr<OpCostCalculator> initial_calculator,
    absl::AnyInvocable<CostValue(const CostMetricId& metric_id,
                                 CostValue cost_value)>
        post_process_fn) {
  return std::make_unique<OpCostCalculatorWithPostProcessedCostValues>(
      std::move(initial_calculator), std::move(post_process_fn));
}

namespace {

CostValue DefaultTotalBytesAccessed(const HloInstruction& instruction,
                                    MetricCalculator& metric_calculator) {
  CostValue result = CostValue::MakeValue(0.0);
  auto update_result = [&result](const Shape& subshape, CostValue next_cost) {
    if (!result.IsOk()) {
      return;
    }
    if (next_cost.IsNotFound()) {
      result = CostValue::MakeNotFound();
      return;
    }
    if (next_cost.IsError()) {
      result = CostValue::MakeError();
      return;
    }
    result = CostValue::MakeValue(result.value() + next_cost.value());
  };

  for (int64_t operand_num = 0; operand_num < instruction.operand_count();
       ++operand_num) {
    const HloInstruction& operand = *instruction.operand(operand_num);
    ShapeUtil::ForEachSubshape(
        operand.shape(), [&](const Shape& subshape, const ShapeIndex& index) {
          if (subshape.IsTuple()) {
            return;
          }
          update_result(subshape, metric_calculator.Calculate(
                                      CostMetricId::OperandBytesAccessed(
                                          instruction, operand_num, index)));
        });
  }
  ShapeUtil::ForEachSubshape(instruction.shape(), [&](const Shape& subshape,
                                                      const ShapeIndex& index) {
    if (subshape.IsTuple()) {
      return;
    }
    update_result(subshape,
                  metric_calculator.Calculate(
                      CostMetricId::OutputBytesAccessed(instruction, index)));
  });

  return result;
}

class MetricCalculatorWithDefaultTotalBytesAccessed : public MetricCalculator {
 public:
  explicit MetricCalculatorWithDefaultTotalBytesAccessed(
      std::unique_ptr<MetricCalculator> initial_metric_calculator)
      : initial_metric_calculator_(std::move(initial_metric_calculator)) {}

  CostValue Calculate(const CostMetricId& metric_id) override {
    if (metric_id.type() == CostMetricId::MetricType::kTotalBytesAccessed) {
      return DefaultTotalBytesAccessed(metric_id.instruction(),
                                       *initial_metric_calculator_);
    }
    return initial_metric_calculator_->Calculate(metric_id);
  }

 private:
  std::unique_ptr<MetricCalculator> initial_metric_calculator_;
};

class OpCostCalculatorWithDefaultTotalBytesAccessed : public OpCostCalculator {
 public:
  explicit OpCostCalculatorWithDefaultTotalBytesAccessed(
      std::unique_ptr<OpCostCalculator> initial_calculator)
      : initial_calculator_(std::move(initial_calculator)) {}

  ~OpCostCalculatorWithDefaultTotalBytesAccessed() override = default;

  std::unique_ptr<MetricCalculator> CreateMetricCalculator(
      const HloInstruction& instruction) override {
    return std::make_unique<MetricCalculatorWithDefaultTotalBytesAccessed>(
        initial_calculator_->CreateMetricCalculator(instruction));
  }

 protected:
  OpCostCalculatorWithDefaultTotalBytesAccessed() = default;

  std::unique_ptr<OpCostCalculator> initial_calculator_;
};

}  // namespace

std::unique_ptr<OpCostCalculator> CreateCalculatorWithDefaultTotalBytesAccessed(
    std::unique_ptr<OpCostCalculator> initial_calculator) {
  return std::make_unique<OpCostCalculatorWithDefaultTotalBytesAccessed>(
      std::move(initial_calculator));
}

}  // namespace xla
