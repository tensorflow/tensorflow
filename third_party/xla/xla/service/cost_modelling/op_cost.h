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

#ifndef XLA_SERVICE_COST_MODELLING_OP_COST_H_
#define XLA_SERVICE_COST_MODELLING_OP_COST_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <tuple>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/shape_util.h"

// This file introduces a simplified cost interface for use in passes like
// Memory Space Assignment.
//
// In addition to the simplified interface, this file provides the following
// common cost functionality:
// * Delegation between different cost implementations
// * Support for allowing cost implementations to be partial
// * Caching of cost values
// * Logging of cost values
//
// The design of the simplified cost interface is as follows:
// * CostMetricId: uniquely identifies a metric we want to compute, e.g.,
//   latency of a particular HLO instruction.
// * CostValue: the value assigned to a metric.
// * MetricCalculator: a function that takes a CostMetricId and returns a
//   CostValue, for a given instruction.
// * OpCostCalculator: a function that takes an HLO instruction and returns a
//   MetricCalculator (computing metrics for that instruction).
//   - We use a 2 layer approach (i.e., OpCostCalculator and MetricCalculator)
//     to support cases where we compute intermediate computations for an HLO
//     that can be refined into specific metric values.
// * OpCostManager: a class that manages the computation of costs. it supports
//   delegation, partial cost implementation, caching, and logging.

namespace xla {

// A unique identifier for a cost metric. For example, the latency of a
// particular HLO instruction.
class CostMetricId {
 public:
  // Supported metric types.
  enum class MetricType : std::uint8_t {
    kLatencySeconds,
    kComputeSeconds,
    kOperandBytesAccessed,
    kOutputBytesAccessed,
    kTotalBytesAccessed,
  };

  // Factory constructors, one for each type of metric.
  static CostMetricId LatencySeconds(const HloInstruction& instruction);
  static CostMetricId ComputeSeconds(const HloInstruction& instruction);
  static CostMetricId OperandBytesAccessed(const HloInstruction& instruction,
                                           int64_t operand_num,
                                           const ShapeIndex& shape_index);
  static CostMetricId OutputBytesAccessed(const HloInstruction& instruction,
                                          const ShapeIndex& shape_index);
  static CostMetricId TotalBytesAccessed(const HloInstruction& instruction);

  // The names of logging columns that correspond to values output by
  // LoggingColumns(). For use with LOG(INFO) type logging. For example, the
  // return columns include "metric_id", "module_name", ...
  static std::vector<std::string> LoggingColumnNames();

  CostMetricId(const CostMetricId& other) = default;
  CostMetricId& operator=(const CostMetricId& other) = default;

  bool operator==(const CostMetricId& other) const;

  template <typename H>
  friend H AbslHashValue(H h, const CostMetricId& id) {
    return H::combine(std::move(h), id.MakeTuple());
  }

  MetricType type() const { return type_; }

  // The instruction the metric is calculated for.
  const HloInstruction& instruction() const { return *instruction_; }

  // REQUIRES: operand_num_.has_value().
  int64_t operand_num() const;

  // REQUIRES: shape_index_.has_value().
  const ShapeIndex& shape_index() const;

  // Values suitable for logging analysis via LOG(INFO).
  std::vector<std::string> LoggingColumns() const;

  // Suitable for errors, warnings, and debugging.
  std::string ToString() const;

 private:
  using Tuple = std::tuple<MetricType, const HloInstruction*,
                           std::optional<int64_t>, std::optional<ShapeIndex>>;

  CostMetricId() = delete;
  CostMetricId(MetricType type, const HloInstruction& instruction,
               std::optional<int64_t> operand_num,
               std::optional<ShapeIndex> shape_index);

  std::string Identifier() const;
  std::string MetricTypeName() const;
  std::string ModuleName() const;
  std::string ComputationName() const;
  std::string InstructionTypeStr() const;
  std::string OperandNumStr() const;
  std::string ShapeIndexStr() const;

  // Returns a tuple of private data members for use in equality and hashing.
  Tuple MakeTuple() const;

  MetricType type_;
  const HloInstruction* instruction_;
  // Null unless type_ is kOperandBytesAccessed.
  std::optional<int64_t> operand_num_ = std::nullopt;
  // Null unless type_ is kOperandBytesAccessed or kOutputBytesAccessed.
  std::optional<ShapeIndex> shape_index_ = std::nullopt;
};

// A value assigned to a cost metric.
class CostValue {
 public:
  // Not found should be used for cases where the cost is not implemented.
  static CostValue MakeNotFound();
  // An error should be used for cases where there is a problem computing a
  // cost.
  static CostValue MakeError();
  static CostValue MakeValue(double value);

  bool operator==(const CostValue& other) const;

  bool IsOk() const { return type_ == Type::kOk; }
  bool IsNotFound() const { return type_ == Type::kNotFound; }
  bool IsError() const { return type_ == Type::kError; }

  // REQUIRES: IsOk().
  double value() const;

  // Suitable for logging analysis for debugging.
  std::string ToString() const;
  friend std::ostream& operator<<(std::ostream& os, const CostValue& value) {
    return os << value.ToString();
  }

 private:
  enum class Type : std::uint8_t { kNotFound, kError, kOk };
  using DataTuple = std::tuple<Type, double>;

  CostValue() = default;
  CostValue(Type type, double value) : type_(type), value_(value) {}

  DataTuple MakeTuple() const { return DataTuple(type_, value_); }

  Type type_ = Type::kNotFound;
  double value_ = 0.0;
};

// A calculator that computes the values of cost metrics, for a given HLO
// instruction.
class MetricCalculator {
 public:
  virtual ~MetricCalculator() = default;

  virtual CostValue Calculate(const CostMetricId& metric_id) {
    return CostValue::MakeError();
  }

 protected:
  MetricCalculator() = default;
};

// A calculator that creates a MetricCalculator, for a given HLO instruction.
class OpCostCalculator {
 public:
  virtual ~OpCostCalculator() = default;

  virtual std::unique_ptr<MetricCalculator> CreateMetricCalculator(
      const HloInstruction& instruction) = 0;

 protected:
  OpCostCalculator() = default;
};

// A manager that computes the values of cost metrics.
class OpCostManager {
 public:
  // Options for the OpCostManager.
  struct Options {
    // If true, the OpCostManager will cache CostValues for CostMetricIds.
    bool enable_cache = false;

    // Enables extra info logging that is suitable for analysis.
    bool enable_analysis_logging = false;
  };

  // Costs are calculated using a tree structure of CalculationNodes. Leaf nodes
  // wrap an OpCostCalculator, and calculate costs. Non-leaf nodes delegate
  // calculation to their children.
  //
  // The names of all nodes passed to an OpCostManager should be unique.
  class CalculationNode {
   public:
    // The type used to index the children of a delegation node.
    using CalculatorIndex = size_t;

    // A map from the names of leaf calculators to the values they computed for
    // a given metric.
    using LeafCalculatorValueMap = absl::flat_hash_map<std::string, CostValue>;

    // Delegation nodes delegate calculation to their children. A
    // DelegationOrderFn returns DelegationInfo, describing the order of
    // delegation. This can be set differently per instruction type.
    struct DelegationInfo {
      // The order a delegation node should delegate to its children.
      std::vector<CalculatorIndex> order;

      // When analysis logging is enabled, we may want to log the costs of
      // additional children calculators, not just the calculator whose value we
      // choose. The indices of these additional calculators are stored here.
      std::vector<CalculatorIndex> additional_calculators_to_log;
    };
    using DelegationOrderFn = absl::AnyInvocable<DelegationInfo(
        const HloInstruction& instruction, bool enable_analysis_logging)>;

    // Creates a leaf node.
    //
    // If enable_cache is true, the leaf node will cache the MetricCalculators
    // it creates per HLO instruction.
    static std::unique_ptr<CalculationNode> CreateLeaf(
        absl::string_view name, std::unique_ptr<OpCostCalculator> calculator,
        bool enable_cache);

    // Creates a delegation node.
    //
    // If delegation_order_fn is nullptr, the node will delegate to its children
    // in the order they are placed in children.
    static std::unique_ptr<CalculationNode> CreateDelegationNode(
        absl::string_view name,
        std::vector<std::unique_ptr<CalculationNode>> children,
        DelegationOrderFn delegation_order_fn = nullptr);

    virtual ~CalculationNode() = default;

    virtual std::optional<double> GetMetricValue(
        const CostMetricId& metric_id,
        LeafCalculatorValueMap* calculator_value_map) = 0;

    virtual absl::string_view Name() const = 0;

    // Returns the names of leaf calculators at or below the node (in the tree).
    // Leaf calculator names are used to uniquely identify the costs associated
    // with a leaf node. They are also used to as additional column names in
    // analysis logging.
    virtual std::vector<std::string> LeafCalculatorNames() const = 0;

   protected:
    CalculationNode() = default;
  };

  OpCostManager(Options options, std::unique_ptr<CalculationNode> root);

  double LatencySeconds(const HloInstruction& instruction);

  double ComputeSeconds(const HloInstruction& instruction);

  double OperandBytesAccessed(const HloInstruction& instruction,
                              int64_t operand_num,
                              const ShapeIndex& shape_index);

  double OutputBytesAccessed(const HloInstruction& instruction,
                             const ShapeIndex& shape_index);

  double TotalBytesAccessed(const HloInstruction& instruction);

 private:
  OpCostManager() = delete;

  // Returns the final value for a given metric.
  double GetMetricValue(const CostMetricId& metric_id);

  // Returns the list of logging column names.
  std::string AnalysisLoggingColumns() const;

  // Returns an analysis logging line for a metric with the specified computed
  // costs.
  std::string AnalysisLoggingLine(
      const CostMetricId& metric_id,
      const CalculationNode::LeafCalculatorValueMap& calculator_costs) const;

  Options options_;
  std::unique_ptr<CalculationNode> root_;
  std::vector<std::string> leaf_calculator_names_;

  // Caching.
  absl::flat_hash_map<CostMetricId, double> metric_cache_;
};

// A wrapper around HloCostAnalysis that calls
// HloModule::entry_computation()->Accept(cost_analysis) as needed.
class HloCostAnalysisWithAcceptState {
 public:
  explicit HloCostAnalysisWithAcceptState(
      std::unique_ptr<HloCostAnalysis> cost_analysis);

  explicit HloCostAnalysisWithAcceptState(HloCostAnalysis& cost_analysis);

  HloCostAnalysis& cost_analysis(const HloInstruction& instruction);

 private:
  HloCostAnalysisWithAcceptState() = delete;

  std::unique_ptr<HloCostAnalysis> cost_analysis_storage_;
  HloCostAnalysis& cost_analysis_;
  bool accepted_entry_computation_ = false;
};

// Creates an OpCostCalculator that uses HloCostAnalysis.
//
// REQUIRES:
// - cost_analysis must outlive the return OpCostCalculator.
std::unique_ptr<OpCostCalculator> CreateHloCostAnalysisCalculator(
    HloCostAnalysisWithAcceptState& cost_analysis_wrapper);

// Creates an OpCostCalculator whose initial values are computed by
// initial_calculator, before being post-processed by the specified
// post_process_fn.
std::unique_ptr<OpCostCalculator> CreateCalculatorWithPostProcessedCostValues(
    std::unique_ptr<OpCostCalculator> initial_calculator,
    absl::AnyInvocable<CostValue(const CostMetricId& metric_id,
                                 CostValue cost_value)>
        post_process_fn);

// Creates an OpCostCalculator that returns the values resulting from initial
// calculator, except in the case of TotalByteAccessed. For TotalByteAccessed,
// the calculator returns the sum of the operand and output bytes accessed.
std::unique_ptr<OpCostCalculator> CreateCalculatorWithDefaultTotalBytesAccessed(
    std::unique_ptr<OpCostCalculator> initial_calculator);

}  // namespace xla

#endif  // XLA_SERVICE_COST_MODELLING_OP_COST_H_
