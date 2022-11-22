

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MCO_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MCO_H_

#include <stack>

#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/util.h"
namespace xla {

// A pass which performs matrix chain optimization. The pass
// iterates over the instructions in topological order to detect matrix chain
// and then performa a MCO algorithm to produce high efficient matrix chain
// solutions
class HloMCO : public HloModulePass {
 public:
  explicit HloMCO(bool only_fusion_computations = false)
      : only_fusion_computations_(only_fusion_computations) {}
  ~HloMCO() override = default;
  absl::string_view name() const override { return "mco"; }

  // Run MCO on the given module. Returns whether the module was changed
  // (matrix chains were found and optimized).
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  StatusOr<bool> ChainOptimize(
      HloComputation* computation,
      absl::flat_hash_map<HloInstruction*, std::vector<HloInstruction*>>&
          chain_map,
      absl::flat_hash_map<HloInstruction*, HloInstruction*>&
          reduce_one_vector_to_orig_init_val);
  StatusOr<HloInstruction*> ComputeOptimalChainOrder(
      HloInstruction* root, std::vector<HloInstruction*>& chain,
      absl::flat_hash_map<HloInstruction*, HloInstruction*>&
          reduce_one_vector_to_orig_init_val);
  StatusOr<HloInstruction*> ConstructOptimalChain(
      HloInstruction* orig_root, std::vector<std::vector<int64_t>>& solution,
      std::vector<HloInstruction*>& chain_instructions,
      absl::flat_hash_map<HloInstruction*, HloInstruction*>&
          reduce_one_vector_to_orig_init_val);
  Status ConstructOptimalChainHelper(
      HloInstruction* orig_root, std::vector<std::vector<int64_t>>& solution,
      std::vector<HloInstruction*>& chain_instructions, int64_t start_index,
      int64_t end_index, std::vector<HloInstruction*>& subgraph_stack,
      absl::flat_hash_map<HloInstruction*, HloInstruction*>&
          reduce_one_vector_to_orig_init_val);
  const PrecisionConfig& precision_config() const { return precision_config_; }
  Status SetPrecisionConfig(PrecisionConfig& precision_config) {
    precision_config_ = precision_config;
  }

 private:
  const bool only_fusion_computations_;
  // Information used to communicate to the implementation about the algorithm
  // used to produce results. See the documentation on precision_config().
  PrecisionConfig precision_config_;
};

class ChainRecorder : public DfsHloVisitorWithDefault {
 public:
  ChainRecorder(HloInstruction* input_chain_root)
      : chain_root(input_chain_root) {}
  Status DefaultAction(HloInstruction* hlo) override { return OkStatus(); }
  Status Preprocess(HloInstruction* hlo) override;
  Status FinishVisit(HloInstruction* hlo) override { return OkStatus(); }
  const absl::flat_hash_map<HloInstruction*, std::vector<HloInstruction*>>
  GetChainMap() {
    return chain_map;
  }
  int64_t GetChainLength(HloInstruction* root) {
    return chain_map[root].size();
  }
  std::vector<HloInstruction*> GetChain(HloInstruction* root) {
    return chain_map[root];
  }
  Status RemoveChain(HloInstruction* root) {
    chain_map.erase(root);
    return OkStatus();
  }

 private:
  // Each kv pair is (chain_root_ptr, vector<chain_instruction_ptr>)
  absl::flat_hash_map<HloInstruction*, std::vector<HloInstruction*>> chain_map;
  HloInstruction* chain_root;
};

class MatrixChainDetector : public DfsHloVisitorWithDefault {
 public:
  MatrixChainDetector() {}

  const absl::flat_hash_map<HloInstruction*, std::vector<HloInstruction*>>
  GetChainMap() {
    return chain_map;
  }
  Status DefaultAction(HloInstruction* hlo) override { return OkStatus(); }
  Status Preprocess(HloInstruction* hlo) override;
  static bool CheckRealDot(HloInstruction* hlo);

  Status FinishVisit(HloInstruction* hlo) override { return OkStatus(); }
  Status DetectMatrixChain(HloInstruction* chain_root);

 private:
  // Each kv pair is (chain_root_ptr, vector<chain_instruction_ptr>)
  absl::flat_hash_map<HloInstruction*, std::vector<HloInstruction*>> chain_map;
};

// a post-order visitor which convert einsum to regular matrix/vector product
// convert reduce_sum to regular m-v dot and unfold transpose
class EinSumReduceSumConverter : public DfsHloRewriteVisitor {
 public:
  Status HandleDot(HloInstruction* dot) override;
  Status HandleReduce(HloInstruction* reduce) override;
  Status TransposeSinker(std::stack<HloInstruction*> trans_stack);
  bool IsReduceSumDot(const HloInstruction* hlo);

  bool DimsAreIndexes(HloInstruction* inst) {
    for (auto i = 0; i < inst->dimensions().size(); i++) {
      if (i != inst->dimensions(i)) {
        return false;
      }
    }
    return true;
  }

  bool IsTrivialTranspose(HloInstruction* transpose) {
    return DimsAreIndexes(transpose);
  }
  absl::flat_hash_map<HloInstruction*, HloInstruction*>&
  GetReduceOneVectorSet() {
    return reduce_one_vector_to_orig_init_val;
  }

 private:
  absl::flat_hash_map<HloInstruction*, HloInstruction*>
      reduce_one_vector_to_orig_init_val;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MCO_H_
