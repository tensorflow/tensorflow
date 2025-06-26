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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/map_util.h"
#include "xla/service/cost_modelling/op_cost_test_utils.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace {

using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::UnorderedElementsAre;

class MetricCalculatorFromMap : public MetricCalculator {
 public:
  explicit MetricCalculatorFromMap(
      const absl::flat_hash_map<CostMetricId, CostValue>& metric_map)
      : metric_map_(metric_map) {}
  ~MetricCalculatorFromMap() override = default;

  CostValue Calculate(const CostMetricId& metric_id) override {
    CostValue default_value = CostValue::MakeNotFound();
    return FindOrDefault(metric_map_, metric_id, default_value);
  }

 protected:
  MetricCalculatorFromMap() = default;

  const absl::flat_hash_map<CostMetricId, CostValue>& metric_map_;
};

class OpCostCalculatorFromMap : public OpCostCalculator {
 public:
  explicit OpCostCalculatorFromMap(
      absl::flat_hash_map<CostMetricId, CostValue> metric_map)
      : metric_map_(std::move(metric_map)) {}
  ~OpCostCalculatorFromMap() override = default;

  std::unique_ptr<MetricCalculator> CreateMetricCalculator(
      const HloInstruction& instruction) override {
    return std::make_unique<MetricCalculatorFromMap>(metric_map_);
  }

 protected:
  OpCostCalculatorFromMap() = default;

  const absl::flat_hash_map<CostMetricId, CostValue> metric_map_;
};

std::unique_ptr<OpCostCalculator> CreateOpCostCalculatorFromMap(
    absl::flat_hash_map<CostMetricId, CostValue> metric_map) {
  return std::make_unique<OpCostCalculatorFromMap>(std::move(metric_map));
}

class OpCostTest : public HloHardwareIndependentTestBase {
 protected:
  using CalculatorValues = std::vector<std::pair<std::string, CostValue>>;

  void SetUp() override {
    HloHardwareIndependentTestBase::SetUp();

    constexpr absl::string_view kHloModule = R"(
    HloModule mymodule

    ENTRY accumulated_all_reduce {
      p0 = (f32[10,10], f32[10,10]) parameter(0)
      p1 = (f32[10,10], f32[10,10]) parameter(1)

      gte0 = f32[10,10] get-tuple-element(p0), index=0
      gte1 = f32[10,10] get-tuple-element(p1), index=1
      tuple0 = (f32[10,10], f32[10,10]) tuple(gte0, gte1)

      gte2 = f32[10,10] get-tuple-element(p0), index=1
      gte3 = f32[10,10] get-tuple-element(p1), index=0
      add0 = f32[10,10] add(gte2, gte3)
      tuple1 = (f32[10,10], f32[10,10]) tuple(gte2, add0)

      ROOT result = ((f32[10,10], f32[10,10]), (f32[10,10], f32[10,10])) tuple(tuple0, tuple1)
    })";
    TF_ASSERT_OK_AND_ASSIGN(module_, ParseAndReturnVerifiedModule(kHloModule));

    add0_ =
        module_->entry_computation()->root_instruction()->operand(1)->operand(
            1);
    ASSERT_EQ(add0_->name(), "add0");
    tuple1_ = module_->entry_computation()->root_instruction()->operand(1);
    ASSERT_EQ(tuple1_->name(), "tuple1");
    result_ = module_->entry_computation()->root_instruction();
    ASSERT_EQ(result_->name(), "result");
  }

  std::unique_ptr<HloModule> module_;
  const HloInstruction* add0_;
  const HloInstruction* tuple1_;
  const HloInstruction* result_;
};

// Creates a leaf node that:
// * returns 100.0 s for instruction's latency.
// * returns 50.0 bytes for operand 0, shape {1} of instruction.
// * returns an error for instruction's compute time.
// * returns not found for all other metrics.
std::unique_ptr<OpCostManager::CalculationNode> CreateTestLeaf(
    const HloInstruction& instruction, bool enable_cache) {
  return OpCostManager::CalculationNode::CreateLeaf(
      "leaf-calculator",
      CreateOpCostCalculatorFromMap({
          {CostMetricId::LatencySeconds(instruction),
           CostValue::MakeValue(100.0)},
          {CostMetricId::ComputeSeconds(instruction), CostValue::MakeError()},
          {CostMetricId::OperandBytesAccessed(instruction, /*operand_num=*/0,
                                              ShapeIndex({1})),
           CostValue::MakeValue(50.0)},
      }),
      enable_cache);
}

// Implements the test for the LeafNodeNoCache and LeafNodeWithCache tests.
void RunLeafNodeCacheTest(const HloInstruction& instruction,
                          bool enable_cache) {
  std::unique_ptr<OpCostManager::CalculationNode> node =
      CreateTestLeaf(instruction, enable_cache);

  // We should repeatably get the same results.
  for (int i = 0; i < 2; ++i) {
    EXPECT_THAT(node->GetMetricValue(/*track_calculation_details=*/false,
                                     CostMetricId::LatencySeconds(instruction)),
                HasCalculationValue(100.0, "leaf-calculator"));
    EXPECT_THAT(node->GetMetricValue(/*track_calculation_details=*/false,
                                     CostMetricId::ComputeSeconds(instruction)),
                MissingCalculationValue());
    EXPECT_THAT(node->GetMetricValue(
                    /*track_calculation_details=*/false,
                    CostMetricId::OperandBytesAccessed(
                        instruction, /*operand_num=*/0, ShapeIndex({0}))),
                MissingCalculationValue());
    EXPECT_THAT(node->GetMetricValue(
                    /*track_calculation_details=*/false,
                    CostMetricId::OperandBytesAccessed(
                        instruction, /*operand_num=*/0, ShapeIndex({1}))),
                HasCalculationValue(50.0, "leaf-calculator"));
  }

  EXPECT_EQ(node->Name(), "leaf-calculator");
  EXPECT_THAT(node->LeafCalculatorNames(), ElementsAre("leaf-calculator"));
}

TEST_F(OpCostTest, LeafNodeNoCache) {
  RunLeafNodeCacheTest(*result_, /*enable_cache=*/false);
}

TEST_F(OpCostTest, LeafNodeWithCache) {
  RunLeafNodeCacheTest(*result_, /*enable_cache=*/true);
}

// Tests that we can pass a map to the leaf calculator, to collect calculated
// values.
TEST_F(OpCostTest, LeafNodeWithValueMap) {
  std::unique_ptr<OpCostManager::CalculationNode> node =
      CreateTestLeaf(*result_, /*enable_cache=*/false);

  EXPECT_THAT(node->GetMetricValue(/*track_calculation_details=*/true,
                                   CostMetricId::LatencySeconds(*result_)),
              AllOf(HasCalculationValue(100.0, "leaf-calculator"),
                    HasCalculatorMapValues(CalculatorValues(
                        {{"leaf-calculator", CostValue::MakeValue(100.0)}}))));
  EXPECT_THAT(node->GetMetricValue(/*track_calculation_details=*/true,
                                   CostMetricId::ComputeSeconds(*result_)),
              AllOf(MissingCalculationValue(),
                    HasCalculatorMapValues(CalculatorValues(
                        {{"leaf-calculator", CostValue::MakeError()}}))));
}

// Creates a delegation node that delegates between leaf0, leaf1, leaf2, and
// leaf3. If no delegation_order_fn is provided, the delegation order is leaf0,
// ... leaf3.
// * leaf0:
//   * latency: 100
//   * compute: error
//   * else: not found
// * leaf1:
//   * latency: 200
//   * compute: 10
//   * else: not found
// * leaf2:
//   * latency: 300
//   * compute: 20
//   * else: not found
// * leaf3:
//   * latency: 400
//   * else: not found
std::unique_ptr<OpCostManager::CalculationNode> CreateTestDelegationNode(
    const HloInstruction& instruction,
    OpCostManager::CalculationNode::DelegationOrderFn delegation_order_fn =
        nullptr) {
  std::vector<std::unique_ptr<OpCostManager::CalculationNode>> children;
  children.push_back(OpCostManager::CalculationNode::CreateLeaf(
      "leaf0",
      CreateOpCostCalculatorFromMap({
          {CostMetricId::LatencySeconds(instruction),
           CostValue::MakeValue(100.0)},
          {CostMetricId::ComputeSeconds(instruction), CostValue::MakeError()},
      }),
      /*enable_cache=*/false));
  children.push_back(OpCostManager::CalculationNode::CreateLeaf(
      "leaf1",
      CreateOpCostCalculatorFromMap({
          {CostMetricId::LatencySeconds(instruction),
           CostValue::MakeValue(200.0)},
          {CostMetricId::ComputeSeconds(instruction), CostValue::MakeValue(10)},
      }),
      /*enable_cache=*/false));
  children.push_back(OpCostManager::CalculationNode::CreateLeaf(
      "leaf2",
      CreateOpCostCalculatorFromMap({
          {CostMetricId::LatencySeconds(instruction),
           CostValue::MakeValue(300.0)},
          {CostMetricId::ComputeSeconds(instruction), CostValue::MakeValue(20)},
      }),
      /*enable_cache=*/false));
  children.push_back(OpCostManager::CalculationNode::CreateLeaf(
      "leaf3",
      CreateOpCostCalculatorFromMap({
          {CostMetricId::LatencySeconds(instruction),
           CostValue::MakeValue(400.0)},
      }),
      /*enable_cache=*/false));

  return OpCostManager::CalculationNode::CreateDelegationNode(
      "delegation-node", std::move(children), std::move(delegation_order_fn));
}

TEST_F(OpCostTest, DelegationNodeDefaultOrder) {
  std::unique_ptr<OpCostManager::CalculationNode> node =
      CreateTestDelegationNode(*result_);

  // We should repeatably get the same results.
  for (int i = 0; i < 2; ++i) {
    // The default delegation order is 0, 1, 2, 3, so the latency comes from
    // leaf0.
    EXPECT_THAT(node->GetMetricValue(/*track_calculation_details=*/false,
                                     CostMetricId::LatencySeconds(*result_)),
                HasCalculationValue(100.0, "leaf0"));
    // The default delegation order is 0, 1, 2, 3. Since leaf0 returns an error
    // for compute, the compute comes from leaf1.
    EXPECT_THAT(node->GetMetricValue(/*track_calculation_details=*/false,
                                     CostMetricId::ComputeSeconds(*result_)),
                HasCalculationValue(10.0, "leaf1"));
    // None of the calculators compute a value for operand bytes accessed.
    EXPECT_THAT(node->GetMetricValue(
                    /*track_calculation_details=*/false,
                    CostMetricId::OperandBytesAccessed(
                        *result_, /*operand_num=*/0, ShapeIndex({1}))),
                MissingCalculationValue());
  }

  EXPECT_EQ(node->Name(), "delegation-node");
  EXPECT_THAT(node->LeafCalculatorNames(),
              UnorderedElementsAre("leaf0", "leaf1", "leaf2", "leaf3"));
}

TEST_F(OpCostTest, DelegationNodeCustomDelegationFn) {
  OpCostManager::CalculationNode::DelegationOrderFn delegation_order_fn =
      [](const HloInstruction& instruction, bool enable_analysis_logging) {
        OpCostManager::CalculationNode::DelegationInfo result;
        result.order = {3, 0, 2};

        return result;
      };

  std::unique_ptr<OpCostManager::CalculationNode> node =
      CreateTestDelegationNode(*result_, std::move(delegation_order_fn));

  // We should repeatably get the same results.
  for (int i = 0; i < 2; ++i) {
    // The delegation order is 3, 0, 2, so the latency comes from leaf3.
    EXPECT_THAT(node->GetMetricValue(/*track_calculation_details=*/false,
                                     CostMetricId::LatencySeconds(*result_)),
                HasCalculationValue(400.0, "leaf3"));

    // The delegation order is 3, 0, 2. Since leaf3 returns not found for
    // compute, and leaf0 returns an error for compute, the compute comes from
    // leaf2.
    EXPECT_THAT(node->GetMetricValue(/*track_calculation_details=*/false,
                                     CostMetricId::ComputeSeconds(*result_)),
                HasCalculationValue(20.0, "leaf2"));

    // None of the calculators compute a value for operand bytes accessed.
    EXPECT_THAT(
        node->GetMetricValue(/*track_calculation_details=*/false,
                             CostMetricId::OperandBytesAccessed(
                                 *result_, /*operand_num=*/0, ShapeIndex({1}))),
        MissingCalculationValue());
  }

  EXPECT_EQ(node->Name(), "delegation-node");
  EXPECT_THAT(node->LeafCalculatorNames(),
              UnorderedElementsAre("leaf0", "leaf1", "leaf2", "leaf3"));
}

TEST_F(OpCostTest, DelegationNodeWithValueMap) {
  OpCostManager::CalculationNode::DelegationOrderFn delegation_order_fn =
      [](const HloInstruction& instruction, bool enable_analysis_logging) {
        OpCostManager::CalculationNode::DelegationInfo result;
        result.order = {3, 0};
        result.additional_calculators_to_log = {1, 2};

        return result;
      };

  std::unique_ptr<OpCostManager::CalculationNode> node =
      CreateTestDelegationNode(*result_, std::move(delegation_order_fn));

  // The delegation order is 3, 0, so the latency comes from leaf3.
  EXPECT_THAT(node->GetMetricValue(/*track_calculation_details=*/true,
                                   CostMetricId::LatencySeconds(*result_)),
              AllOf(HasCalculationValue(400.0, "leaf3"),
                    HasCalculatorMapValues(CalculatorValues({
                        {"leaf0", CostValue::MakeValue(100.0)},
                        {"leaf1", CostValue::MakeValue(200.0)},
                        {"leaf2", CostValue::MakeValue(300.0)},
                        {"leaf3", CostValue::MakeValue(400.0)},
                    }))));

  // The delegation order is 3, 0, but neither of those leaves calculate a value
  // for compute.
  EXPECT_THAT(node->GetMetricValue(/*track_calculation_details=*/true,
                                   CostMetricId::ComputeSeconds(*result_)),
              AllOf(MissingCalculationValue(),
                    HasCalculatorMapValues(CalculatorValues({
                        {"leaf0", CostValue::MakeError()},
                        {"leaf1", CostValue::MakeValue(10.0)},
                        {"leaf2", CostValue::MakeValue(20.0)},
                        {"leaf3", CostValue::MakeNotFound()},
                    }))));

  // None of the calculators compute a value for operand bytes accessed.
  EXPECT_THAT(
      node->GetMetricValue(/*track_calculation_details=*/true,
                           CostMetricId::OperandBytesAccessed(
                               *result_, /*operand_num=*/0, ShapeIndex({1}))),
      AllOf(MissingCalculationValue(), HasCalculatorMapValues(CalculatorValues({
                                           {"leaf0", CostValue::MakeNotFound()},
                                           {"leaf1", CostValue::MakeNotFound()},
                                           {"leaf2", CostValue::MakeNotFound()},
                                           {"leaf3", CostValue::MakeNotFound()},
                                       }))));
}

TEST_F(OpCostTest, DelegationNodeDifferentOrdersForDifferentInstructions) {
  std::vector<std::unique_ptr<OpCostManager::CalculationNode>> children;
  children.push_back(OpCostManager::CalculationNode::CreateLeaf(
      "leaf0",
      CreateOpCostCalculatorFromMap({
          {CostMetricId::LatencySeconds(*result_), CostValue::MakeValue(100.0)},
          {CostMetricId::LatencySeconds(*tuple1_), CostValue::MakeValue(100.0)},
      }),
      /*enable_cache=*/false));
  children.push_back(OpCostManager::CalculationNode::CreateLeaf(
      "leaf1",
      CreateOpCostCalculatorFromMap({
          {CostMetricId::LatencySeconds(*result_), CostValue::MakeValue(200.0)},
          {CostMetricId::LatencySeconds(*tuple1_), CostValue::MakeValue(200.0)},
      }),
      /*enable_cache=*/false));

  OpCostManager::CalculationNode::DelegationOrderFn delegation_order_fn =
      [&](const HloInstruction& instruction, bool enable_analysis_logging) {
        OpCostManager::CalculationNode::DelegationInfo result;
        if (&instruction == tuple1_) {
          result.order = {1, 0};
        } else {
          result.order = {0, 1};
        }

        return result;
      };

  std::unique_ptr<OpCostManager::CalculationNode> node =
      OpCostManager::CalculationNode::CreateDelegationNode(
          "delegation-node", std::move(children),
          std::move(delegation_order_fn));

  // For the result_ instruction, the delegation order is 0, 1, so the latency
  // comes from leaf0.
  EXPECT_THAT(node->GetMetricValue(/*track_calculation_details=*/false,
                                   CostMetricId::LatencySeconds(*result_)),
              HasCalculationValue(100.0, "leaf0"));

  // For the tuple1_ instruction, the delegation order is 1, 0, so the latency
  // comes from leaf1.
  EXPECT_THAT(node->GetMetricValue(/*track_calculation_details=*/false,
                                   CostMetricId::LatencySeconds(*tuple1_)),
              HasCalculationValue(200.0, "leaf1"));
}

// Implements the test for the OpCostManagerNoCache and OpCostManagerWithCache
// tests.
void RunOpCostManagerTest(const HloInstruction& instruction, bool enable_cache,
                          bool enable_analysis_logging) {
  std::vector<std::unique_ptr<OpCostManager::CalculationNode>> children;
  children.push_back(OpCostManager::CalculationNode::CreateLeaf(
      "leaf0",
      CreateOpCostCalculatorFromMap({
          {CostMetricId::LatencySeconds(instruction),
           CostValue::MakeValue(100.0)},
          {CostMetricId::ComputeSeconds(instruction),
           CostValue::MakeValue(10.0)},
      }),
      enable_cache));

  children.push_back(OpCostManager::CalculationNode::CreateLeaf(
      "leaf1",
      CreateOpCostCalculatorFromMap({
          {CostMetricId::LatencySeconds(instruction),
           CostValue::MakeValue(200.0)},
          {CostMetricId::ComputeSeconds(instruction),
           CostValue::MakeValue(20.0)},
          {CostMetricId::OperandBytesAccessed(instruction, /*operand_num=*/0,
                                              ShapeIndex({1})),
           CostValue::MakeValue(20.0)},
          {CostMetricId::OutputBytesAccessed(instruction, ShapeIndex({1, 1})),
           CostValue::MakeValue(40.0)},
          {CostMetricId::TotalBytesAccessed(instruction),
           CostValue::MakeValue(2000.0)},
      }),
      enable_cache));

  OpCostManager op_cost_manager(
      {enable_cache, enable_analysis_logging},
      OpCostManager::CalculationNode::CreateDelegationNode(
          "delegation-node", std::move(children)));

  EXPECT_EQ(op_cost_manager.LatencySeconds(instruction), 100.0);
  EXPECT_EQ(op_cost_manager.ComputeSeconds(instruction), 10.0);
  EXPECT_EQ(
      op_cost_manager.OperandBytesAccessed(instruction,
                                           /*operand_num=*/0, ShapeIndex({1})),
      20.0);
  EXPECT_EQ(
      op_cost_manager.OutputBytesAccessed(instruction, ShapeIndex({1, 1})),
      40.0);
  EXPECT_EQ(op_cost_manager.TotalBytesAccessed(instruction), 2000.0);
}

TEST_F(OpCostTest, OpCostManagerNoCache) {
  RunOpCostManagerTest(*result_, /*enable_cache=*/false,
                       /*enable_analysis_logging=*/false);
}

TEST_F(OpCostTest, OpCostManagerWithCache) {
  RunOpCostManagerTest(*result_, /*enable_cache=*/true,
                       /*enable_analysis_logging=*/false);
}

TEST_F(OpCostTest, OpCostManagerWithAnalysisLogging) {
  RunOpCostManagerTest(*result_, /*enable_cache=*/true,
                       /*enable_analysis_logging=*/true);
}

TEST_F(OpCostTest, HloCostAnalysisWithAcceptState) {
  auto hlo_cost_analysis_for_wrapper = std::make_unique<HloCostAnalysis>();
  TF_EXPECT_OK(module_->entry_computation()->Accept(
      hlo_cost_analysis_for_wrapper.get()));
  HloCostAnalysisWithAcceptState hlo_cost_analysis_wrapper(
      std::move(hlo_cost_analysis_for_wrapper));

  HloCostAnalysis hlo_cost_analysis;
  TF_EXPECT_OK(module_->entry_computation()->Accept(&hlo_cost_analysis));

  // It should be ok to keep getting cost analysis results, without crashing for
  // calling Accept() multiple times.
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(
        hlo_cost_analysis_wrapper.cost_analysis(*add0_).flop_count(*add0_),
        hlo_cost_analysis.flop_count(*add0_));
  }
}

TEST_F(OpCostTest, CreateHloCostAnalysisCalculator) {
  HloCostAnalysisWithAcceptState hlo_cost_analysis_wrapper(
      std::make_unique<HloCostAnalysis>());

  std::unique_ptr<OpCostCalculator> op_cost_calculator =
      CreateHloCostAnalysisCalculator(hlo_cost_analysis_wrapper);

  HloCostAnalysis hlo_cost_analysis;
  TF_EXPECT_OK(module_->entry_computation()->Accept(&hlo_cost_analysis));

  EXPECT_EQ(op_cost_calculator->CreateMetricCalculator(*add0_)->Calculate(
                CostMetricId::LatencySeconds(*add0_)),
            CostValue::MakeValue(hlo_cost_analysis.optimal_seconds(*add0_)));
}

TEST_F(OpCostTest, CalculatorWithPostProcessedValues) {
  std::unique_ptr<OpCostCalculator> op_cost_calculator =
      CreateCalculatorWithPostProcessedCostValues(
          CreateOpCostCalculatorFromMap({
              {CostMetricId::LatencySeconds(*result_),
               CostValue::MakeValue(8.0)},
              {CostMetricId::ComputeSeconds(*result_),
               CostValue::MakeValue(4.0)},
          }),
          [](const CostMetricId& metric_id, CostValue cost_value) {
            if (metric_id.type() == CostMetricId::MetricType::kLatencySeconds) {
              return CostValue::MakeValue(cost_value.value() * 2.0);
            }
            return cost_value;
          });
  EXPECT_EQ(op_cost_calculator->CreateMetricCalculator(*result_)->Calculate(
                CostMetricId::LatencySeconds(*result_)),
            CostValue::MakeValue(16.0));
  EXPECT_EQ(op_cost_calculator->CreateMetricCalculator(*result_)->Calculate(
                CostMetricId::ComputeSeconds(*result_)),
            CostValue::MakeValue(4.0));
}

TEST_F(OpCostTest, CalculatorWithDefaultTotalBytesAccessed) {
  std::unique_ptr<OpCostCalculator> op_cost_calculator =
      CreateOpCostCalculatorFromMap({
          {CostMetricId::OperandBytesAccessed(*result_, /*operand_num=*/0,
                                              ShapeIndex({0})),
           CostValue::MakeValue(10.0)},
          {CostMetricId::OperandBytesAccessed(*result_, /*operand_num=*/0,
                                              ShapeIndex({1})),
           CostValue::MakeValue(10.0)},
          {CostMetricId::OperandBytesAccessed(*result_, /*operand_num=*/1,
                                              ShapeIndex({0})),
           CostValue::MakeValue(10.0)},
          {CostMetricId::OperandBytesAccessed(*result_, /*operand_num=*/1,
                                              ShapeIndex({1})),
           CostValue::MakeValue(10.0)},
          {CostMetricId::OutputBytesAccessed(*result_, ShapeIndex({0, 0})),
           CostValue::MakeValue(10.0)},
          {CostMetricId::OutputBytesAccessed(*result_, ShapeIndex({0, 1})),
           CostValue::MakeValue(10.0)},
          {CostMetricId::OutputBytesAccessed(*result_, ShapeIndex({1, 0})),
           CostValue::MakeValue(10.0)},
          {CostMetricId::OutputBytesAccessed(*result_, ShapeIndex({1, 1})),
           CostValue::MakeValue(10.0)},
          {CostMetricId::TotalBytesAccessed(*result_),
           CostValue::MakeValue(2000.0)},
      });
  EXPECT_EQ(op_cost_calculator->CreateMetricCalculator(*result_)->Calculate(
                CostMetricId::TotalBytesAccessed(*result_)),
            CostValue::MakeValue(2000.0));

  std::unique_ptr<OpCostCalculator>
      op_cost_calculator_with_default_total_bytes_accessed =
          CreateCalculatorWithDefaultTotalBytesAccessed(
              std::move(op_cost_calculator));
  EXPECT_EQ(op_cost_calculator_with_default_total_bytes_accessed
                ->CreateMetricCalculator(*result_)
                ->Calculate(CostMetricId::TotalBytesAccessed(*result_)),
            CostValue::MakeValue(80.0));
}

}  // namespace
}  // namespace xla
