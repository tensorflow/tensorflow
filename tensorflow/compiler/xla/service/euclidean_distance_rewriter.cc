// License TODO ....
#include "tensorflow/compiler/xla/service/euclidean_distance_rewriter.h"

#include "absl/algorithm/container.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/status_macros.h"

namespace xla {

namespace {

namespace m = match;

class EuclideanDistanceRewriterVisitor : public DfsHloRewriteVisitor {
 public:
  explicit EuclideanDistanceRewriterVisitor() {}

  // MatchDistanceMatrix checks that
  bool MatchDistanceMatrix(HloInstruction* reduce, HloInstruction** x,
                           HloInstruction** y, bool* is_sub,
                           std::vector<int64_t>* lhs_reduce_dims,
                           std::vector<int64_t>* rhs_reduce_dims,
                           std::vector<int64_t>* lhs_broadcast_dims,
                           std::vector<int64_t>* rhs_broadcast_dims,
                           std::vector<int64_t>* lhs_batch_dims,
                           std::vector<int64_t>* rhs_batch_dims);

  Status HandleReduce(HloInstruction* reduce) override;
};

}  // namespace

bool EuclideanDistanceRewriterVisitor::MatchDistanceMatrix(
    HloInstruction* reduce, HloInstruction** lhs, HloInstruction** rhs,
    bool* is_sub, std::vector<int64_t>* lhs_reduce_dims,
    std::vector<int64_t>* rhs_reduce_dims,
    std::vector<int64_t>* lhs_broadcast_dims,
    std::vector<int64_t>* rhs_broadcast_dims,
    std::vector<int64_t>* lhs_batch_dims,
    std::vector<int64_t>* rhs_batch_dims) {
  HloInstruction* core_expr = nullptr;
  HloInstruction* reduce_operand = nullptr;
  HloInstruction* reduce_init = nullptr;

  if (!Match(reduce,
             m::Reduce(m::Op(&reduce_operand), m::Constant(&reduce_init)))) {
    return false;
  }
  auto is_pow_based = reduce_operand->opcode() == HloOpcode::kPower;
  auto is_mul_based = reduce_operand->opcode() == HloOpcode::kMultiply &&
                      reduce_operand->operand(0) == reduce_operand->operand(1);
  if (is_mul_based) {
    core_expr = reduce_operand->mutable_operand(0);
  } else if (is_pow_based) {
    HloInstruction* power_const = nullptr;
    if (!Match(reduce_operand,
               m::Power(m::Op(&core_expr),
                        m::Broadcast(m::Constant(&power_const))))) {
      return false;
    }
    if (ShapeUtil::ElementsIn(power_const->shape()) != 1 ||
        !power_const->literal().Get<float>({0}) == 2.0) {
      return false;
    }
  }

  // Check add or sub
  HloInstruction* lhs_broadcast = nullptr;
  HloInstruction* rhs_broadcast = nullptr;
  if (Match(core_expr, m::Add(m::Op(&lhs_broadcast), m::Op(&rhs_broadcast)))) {
    *is_sub = false;
  } else if (Match(core_expr,
                   m::Subtract(m::Op(&lhs_broadcast), m::Op(&rhs_broadcast)))) {
    *is_sub = true;
  } else {
    return false;
  }

  // Check the constants are correct
  if (ShapeUtil::ElementsIn(reduce_init->shape()) != 1 ||
      !reduce_init->literal().IsZero({0})) {
    return false;
  }

  // Check broadcast
  if (!Match(lhs_broadcast, m::Broadcast(m::Op(lhs))) ||
      !Match(rhs_broadcast, m::Broadcast(m::Op(rhs)))) {
    return false;
  }

  Shape prod_shape = reduce->shape();
  Shape reduce_input_shape = reduce_operand->shape();
  auto reduce_dims = reduce->dimensions();
  auto lhs_broadcast_dims_orig = lhs_broadcast->dimensions();
  auto rhs_broadcast_dims_orig = rhs_broadcast->dimensions();
  auto dimension_indices_intersection = [](const auto& reference,
                                           const auto& dimensions) {
    auto matched_count = 0;
    std::vector<int64_t> output;
    for (auto i = 0; i < reference.size(); ++i) {
      for (auto j = 0; j < dimensions.size(); ++j) {
        if (reference[i] == dimensions[j]) {
          output.push_back(i);
          ++matched_count;
        }
      }
    }
    return std::make_tuple(matched_count == dimensions.size(), output);
  };

  bool lhs_has_reduced_dims;
  bool rhs_has_reduced_dims;
  std::tie(lhs_has_reduced_dims, *lhs_reduce_dims) =
      dimension_indices_intersection(lhs_broadcast_dims_orig, reduce_dims);
  std::tie(rhs_has_reduced_dims, *rhs_reduce_dims) =
      dimension_indices_intersection(rhs_broadcast_dims_orig, reduce_dims);

  if (!lhs_has_reduced_dims || !rhs_has_reduced_dims) {
    // Add failure message here
    return false;
  }

  auto update_broadcast_dimensions_after_reduce =
      [](const auto& broadcast_dimensions, const auto& reduce_dimensions) {
        std::vector<int64_t> updated_dimensions;
        absl::c_remove_copy_if(
            broadcast_dimensions, std::back_inserter(updated_dimensions),
            [&reduce_dimensions](const auto& value) {
              return absl::c_linear_search(reduce_dimensions, value);
            });
        std::vector<int64_t> updated_dimensions_origin = updated_dimensions;
        for (auto reduce_dim : reduce_dimensions) {
          for (auto i = 0; i < updated_dimensions.size(); ++i) {
            if (updated_dimensions_origin[i] > reduce_dim) {
              updated_dimensions[i] -= 1;
            }
          }
        }
        return updated_dimensions;
      };

  *lhs_broadcast_dims = update_broadcast_dimensions_after_reduce(
      lhs_broadcast_dims_orig, reduce_dims);
  *rhs_broadcast_dims = update_broadcast_dimensions_after_reduce(
      rhs_broadcast_dims_orig, reduce_dims);

  std::vector<int64_t> lhs_unique_broadcast_dims;
  std::vector<int64_t> rhs_unique_broadcast_dims;

  // Add LHS dot dimensions
  for (auto dim = 0; dim < lhs_broadcast_dims->size(); ++dim) {
    if (!absl::c_linear_search(*lhs_reduce_dims, dim)) {
      auto lhs_dim = lhs_broadcast_dims->at(dim);
      if (absl::c_linear_search(*rhs_broadcast_dims, lhs_dim)) {
        lhs_batch_dims->push_back(dim);
      } else {
        lhs_unique_broadcast_dims.push_back(lhs_dim);
      }
    }
  }

  // Add RHS dot dimensions
  for (auto dim = 0; dim < rhs_broadcast_dims->size(); ++dim) {
    if (!absl::c_linear_search(*rhs_reduce_dims, dim)) {
      auto rhs_dim = rhs_broadcast_dims->at(dim);
      if (absl::c_linear_search(*lhs_broadcast_dims, rhs_dim)) {
        rhs_batch_dims->push_back(dim);
      } else {
        rhs_unique_broadcast_dims.push_back(rhs_dim);
      }
    }
  }

  auto lhs_batch_size = lhs_batch_dims->size();
  auto rhs_batch_size = rhs_batch_dims->size();

  if (lhs_batch_size != rhs_batch_size) {
    return false;
  }

  if (lhs_batch_size > 0) {
    auto lhs_batch_max = absl::c_max_element(*lhs_batch_dims);
    auto rhs_batch_max = absl::c_max_element(*rhs_batch_dims);

    if (*lhs_batch_max != (lhs_batch_dims->size() - 1)) {
      return false;
    }

    if (*rhs_batch_max != (rhs_batch_dims->size() - 1)) {
      return false;
    }
  }

  auto lhs_broadcast_max = absl::c_max_element(lhs_unique_broadcast_dims);
  auto lhs_broadcast_min = absl::c_min_element(lhs_unique_broadcast_dims);

  auto rhs_broadcast_max = absl::c_max_element(rhs_unique_broadcast_dims);
  auto rhs_broadcast_min = absl::c_min_element(rhs_unique_broadcast_dims);

  // Check the correct order for dot products
  if (*lhs_broadcast_min > *rhs_broadcast_max) {
    std::swap(lhs_broadcast_max, rhs_broadcast_max);
    std::swap(lhs_broadcast_min, rhs_broadcast_min);
    std::swap(lhs_unique_broadcast_dims, rhs_unique_broadcast_dims);
    std::swap(*lhs_broadcast_dims, *rhs_broadcast_dims);
    std::swap(*lhs_reduce_dims, *rhs_reduce_dims);
    std::swap(*lhs, *rhs);
  }

  // Check that broadcast dimensions are not interleaving.
  if (*lhs_broadcast_max > *rhs_broadcast_min) {
    return false;
  }

  return true;
}

Status EuclideanDistanceRewriterVisitor::HandleReduce(HloInstruction* reduce) {
  HloInstruction *lhs, *rhs;
  bool is_sub;
  std::vector<int64_t> lhs_reduce_dims;
  std::vector<int64_t> rhs_reduce_dims;
  std::vector<int64_t> lhs_broadcast_dims;
  std::vector<int64_t> rhs_broadcast_dims;
  std::vector<int64_t> lhs_batch_dims;
  std::vector<int64_t> rhs_batch_dims;

  if (!MatchDistanceMatrix(reduce, &lhs, &rhs, &is_sub, &lhs_reduce_dims,
                           &rhs_reduce_dims, &lhs_broadcast_dims,
                           &rhs_broadcast_dims, &lhs_batch_dims,
                           &rhs_batch_dims)) {
    return OkStatus();
  }

  HloComputation* comp = reduce->parent();
  Shape lhs_shape = lhs->shape();
  Shape rhs_shape = rhs->shape();

  // constants
  auto type = lhs_shape.element_type();
  auto zero_literal = LiteralUtil::CreateR0(0)
                          .ConvertToShape(ShapeUtil::MakeShape(type, {}))
                          .ValueOrDie();
  HloInstruction* zero = comp->AddInstruction(
      HloInstruction::CreateConstant(std::move(zero_literal)));

  // Squares
  HloInstruction* lhs_squared = comp->AddInstruction(
      HloInstruction::CreateBinary(lhs_shape, HloOpcode::kMultiply, lhs, lhs));
  HloInstruction* rhs_squared = comp->AddInstruction(
      HloInstruction::CreateBinary(rhs_shape, HloOpcode::kMultiply, rhs, rhs));

  // Reduce the squares

  auto lhs_reduce_shape = ShapeUtil::FilterDimensions(
      [&lhs_reduce_dims](auto dim) {
        return !absl::c_linear_search(lhs_reduce_dims, dim);
      },
      lhs_shape);
  auto rhs_reduce_shape = ShapeUtil::FilterDimensions(
      [&rhs_reduce_dims](auto dim) {
        return !absl::c_linear_search(rhs_reduce_dims, dim);
      },
      rhs_shape);

  HloInstruction* lhs_reduced = comp->AddInstruction(
      HloInstruction::CreateReduce(lhs_reduce_shape, lhs_squared, zero,
                                   lhs_reduce_dims, reduce->to_apply()));
  HloInstruction* rhs_reduced = comp->AddInstruction(
      HloInstruction::CreateReduce(rhs_reduce_shape, rhs_squared, zero,
                                   rhs_reduce_dims, reduce->to_apply()));

  // Start constructing an outer product
  // LHS and RHS outer products

  DotDimensionNumbers dnums;

  // Add LHS dot dimensions
  for (auto dim : lhs_reduce_dims) {
    dnums.add_lhs_contracting_dimensions(dim);
  }
  for (auto dim : lhs_batch_dims) {
    dnums.add_lhs_batch_dimensions(dim);
  }

  // Add RHS dot dimensions
  for (auto dim : rhs_reduce_dims) {
    dnums.add_rhs_contracting_dimensions(dim);
  }
  for (auto dim : rhs_batch_dims) {
    dnums.add_rhs_batch_dimensions(dim);
  }

  PrecisionConfig precision_config;
  precision_config.mutable_operand_precision()->Resize(
      2, PrecisionConfig::DEFAULT);
  Shape prod_shape = reduce->shape();
  HloInstruction* prod = comp->AddInstruction(
      HloInstruction::CreateDot(prod_shape, lhs, rhs, dnums, precision_config));

  auto two_literal = LiteralUtil::CreateR0(2)
                         .ConvertToShape(ShapeUtil::MakeShape(type, {}))
                         .ValueOrDie();
  HloInstruction* two = comp->AddInstruction(
      HloInstruction::CreateConstant(std::move(two_literal)));
  HloInstruction* two_broadcast = comp->AddInstruction(
      HloInstruction::CreateBroadcast(prod_shape, two, {}));

  HloInstruction* two_prod = comp->AddInstruction(HloInstruction::CreateBinary(
      prod_shape, HloOpcode::kMultiply, two_broadcast, prod));

  // Final steps to broadcast LHS and RHS inputs to the outer product shape

  HloInstruction* lhs_broadcast =
      comp->AddInstruction(HloInstruction::CreateBroadcast(
          prod_shape, lhs_reduced, lhs_broadcast_dims));

  HloInstruction* rhs_broadcast =
      comp->AddInstruction(HloInstruction::CreateBroadcast(
          prod_shape, rhs_reduced, rhs_broadcast_dims));

  auto comp_code = HloOpcode::kAdd;
  if (is_sub) {
    comp_code = HloOpcode::kSubtract;
  }

  HloInstruction* lhs_rhs_sum =
      comp->AddInstruction(HloInstruction::CreateBinary(
          prod_shape, HloOpcode::kAdd, lhs_broadcast, rhs_broadcast));

  HloInstruction* replacement =
      comp->AddInstruction(HloInstruction::CreateBinary(prod_shape, comp_code,
                                                        lhs_rhs_sum, two_prod));

  return ReplaceInstruction(reduce, replacement);
}

StatusOr<bool> EuclideanDistanceRewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  LOG(INFO) << "Running algebraic rewriter for '" << module->name() << "'";
  EuclideanDistanceRewriterVisitor visitor;
  return visitor.RunOnModule(module, execution_threads);
}

}  // namespace xla