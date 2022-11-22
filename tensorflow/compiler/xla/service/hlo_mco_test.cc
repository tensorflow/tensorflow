#include "tensorflow/compiler/xla/service/hlo_mco.h"

#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
#include "tensorflow/compiler/xla/service/hlo_cse.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_graph_dumper.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/compiler/xla/service/transpose_folding.h"

namespace xla {
namespace {
class HloMCOTest : public HloTestBase {
 protected:
  HloMCOTest() {}
};

std::string dot_dir = "~/test_output/";

void DumpToFileInDirImpl(std::string dir, std::string filename,
                         std::string contents) {
  // const string& dir = opts.dump_to;
  std::cout << "Dumping " << filename << " to " << dir;

  tensorflow::Env* env = tensorflow::Env::Default();
  // Two threads can race to observe the absence of the dump directory and
  // simultaneously try to create it, causing the "losing" thread to get a
  // "directory already exists" error.  We can work around this by checking
  // again whether the dir exists.
  if (!env->IsDirectory(dir).ok()) {
    auto status = env->RecursivelyCreateDir(dir);
    if (!status.ok() && !env->IsDirectory(dir).ok()) {
      LOG(ERROR) << "Could not create directory " << dir
                 << " for dumping XLA debug data: " << status;
      return;
    }
  }

  string file_path = tensorflow::io::JoinPath(dir, SanitizeFileName(filename));
  auto status = tensorflow::WriteStringToFile(env, file_path, contents);
  if (!status.ok()) {
    LOG(ERROR) << "Could not write XLA debug data to " << file_path << ": "
               << status;
  }
}

TEST_F(HloMCOTest, PureOptimalMatrixChain) {
  // Test matrix chain only consists of matrices and the chain is already
  // optimal
  auto builder = HloComputation::Builder(TestName());
  const std::string hlo_text = R"(
HloModule PureOptimalMatrixChain
main{
  %A = f32[10,20]{1,0} parameter(0), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %B = f32[20,30]{1,0} parameter(1), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %dot1 = f32[10,30]{1,0} dot(f32[10,20]{1,0} %A, f32[20,30]{1,0} %B), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="matmul"}
  %C = f32[30,40]{1,0} parameter(2), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %dot2 = f32[10,40]{1,0} dot(f32[10,30]{1,0} %dot1, f32[30,40]{1,0} %C), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="matmul_1"}
  %D = f32[40,30]{1,0} parameter(3), parameter_replication={false}, metadata={op_name="XLA_Args"}
  ROOT %dot3 = f32[10,30]{1,0} dot(f32[10,40]{1,0} %dot2, f32[40,30]{1,0} %D), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="matmul_2"}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_text));
  
  std::string filename = TestName() + "_before_opotimization";
  // DebugOptions debug_options
  auto render_graph = [&](RenderedGraphFormat format) {
    StatusOr<string> rendered_graph =
        RenderGraph(*m->entry_computation(),
                    /*label=*/filename, m->config().debug_options(), format);
    if (rendered_graph.ok()) {
      return std::move(rendered_graph).ValueOrDie();
    }
    return absl::StrFormat("Error rendering graph: %s",
                           rendered_graph.status().ToString());
  };
  std::cout << "Start Dumping " << filename << " to " << dot_dir;
  DumpToFileInDirImpl(dot_dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));
  HloMCO pass;
  ASSERT_TRUE(pass.Run(m.get()).ValueOrDie());
  AlgebraicSimplifierOptions default_options_;
  AlgebraicSimplifier simplifier(default_options_);
  simplifier.Run(m.get()).ValueOrDie();
  TransposeFolding transpose_folding;
  transpose_folding.Run(m.get()).ValueOrDie();
  HloDCE dce;
  RunHloPass(&dce, m.get());
  HloCSE cse(/*is_layout_sensitive=*/false);
  cse.Run(m.get()).ValueOrDie();
  HloInstruction* root = m->entry_computation()->root_instruction();
  printf("After opotimization:\n %f\n", m->ToString().c_str());

  filename = TestName() + "_after_opotimization";
  std::cout << "Start Dumping " << filename << " to " << dot_dir;
  DumpToFileInDirImpl(dot_dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));
}

TEST_F(HloMCOTest, PureMatrixChain) {
  // Test matrix chain only consists of matrices
  auto builder = HloComputation::Builder(TestName());
  const std::string hlo_text = R"(
HloModule PureMatrixChain
main{
  %A = f32[40,20]{1,0} parameter(0)
  %B = f32[20,30]{1,0} parameter(1)
  %dot1 = f32[40,30]{1,0} dot(f32[40,20]{1,0} %A, f32[20,30]{1,0} %B), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %C = f32[30,10]{1,0} parameter(2), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %dot2 = f32[40,10]{1,0} dot(f32[40,30]{1,0} %dot1, f32[30,10]{1,0} %C), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %D = f32[10,30]{1,0} parameter(3)
  ROOT %dot3 = f32[40,30]{1,0} dot(f32[40,10]{1,0} %dot2, f32[10,30]{1,0} %D), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_text));
  
  std::string filename = TestName() + "_before_opotimization";
  // DebugOptions debug_options
  auto render_graph = [&](RenderedGraphFormat format) {
    StatusOr<string> rendered_graph =
        RenderGraph(*m->entry_computation(),
                    /*label=*/filename, m->config().debug_options(), format);
    if (rendered_graph.ok()) {
      return std::move(rendered_graph).ValueOrDie();
    }
    return absl::StrFormat("Error rendering graph: %s",
                           rendered_graph.status().ToString());
  };
  std::cout << "Start Dumping " << filename << " to " << dot_dir;
  DumpToFileInDirImpl(dot_dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));
  HloMCO pass;
  ASSERT_TRUE(pass.Run(m.get()).ValueOrDie());
  AlgebraicSimplifierOptions default_options_;
  AlgebraicSimplifier simplifier(default_options_);
  simplifier.Run(m.get()).ValueOrDie();
  TransposeFolding transpose_folding;
  transpose_folding.Run(m.get()).ValueOrDie();
  HloDCE dce;
  RunHloPass(&dce, m.get());
  HloCSE cse(/*is_layout_sensitive=*/false);
  cse.Run(m.get()).ValueOrDie();
  HloInstruction* root = m->entry_computation()->root_instruction();
  printf("After opotimization:\n %f\n", m->ToString().c_str());

  filename = TestName() + "_after_opotimization";
  std::cout << "Start Dumping " << filename << " to " << dot_dir;
  DumpToFileInDirImpl(dot_dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));
}

TEST_F(HloMCOTest, MatrixChainAsSubgraph) {
  // Test opotimization in graph which contain a matrix cahin as a subgraph
  auto builder = HloComputation::Builder(TestName());
  const std::string hlo_text = R"(
HloModule MatrixChainAsSubgraph
main{
  %A = f32[40,20]{1,0} parameter(0)
  %B = f32[20,30]{1,0} parameter(1)
  %dot1 = f32[40,30]{1,0} dot(f32[40,20]{1,0} %A, f32[20,30]{1,0} %B), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %C = f32[30,10]{1,0} parameter(2), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %dot2 = f32[40,10]{1,0} dot(f32[40,30]{1,0} %dot1, f32[30,10]{1,0} %C), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %D = f32[10,30]{1,0} parameter(3)
  %dot3 = f32[40,30]{1,0} dot(f32[40,10]{1,0} %dot2, f32[10,30]{1,0} %D), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %E = f32[40,30]{1,0} parameter(4)
  ROOT %add = f32[40,30] add(%E, %dot3)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_text));
  
  std::string filename = TestName() + "_before_opotimization";
  // DebugOptions debug_options
  auto render_graph = [&](RenderedGraphFormat format) {
    StatusOr<string> rendered_graph =
        RenderGraph(*m->entry_computation(),
                    /*label=*/filename, m->config().debug_options(), format);
    if (rendered_graph.ok()) {
      return std::move(rendered_graph).ValueOrDie();
    }
    return absl::StrFormat("Error rendering graph: %s",
                           rendered_graph.status().ToString());
  };
  std::cout << "Start Dumping " << filename << " to " << dot_dir;
  DumpToFileInDirImpl(dot_dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));
  HloMCO pass;
  ASSERT_TRUE(pass.Run(m.get()).ValueOrDie());
  AlgebraicSimplifierOptions default_options_;
  AlgebraicSimplifier simplifier(default_options_);
  simplifier.Run(m.get()).ValueOrDie();
  TransposeFolding transpose_folding;
  transpose_folding.Run(m.get()).ValueOrDie();
  HloDCE dce;
  RunHloPass(&dce, m.get());
  HloCSE cse(/*is_layout_sensitive=*/false);
  cse.Run(m.get()).ValueOrDie();
  HloInstruction* root = m->entry_computation()->root_instruction();
  printf("After opotimization:\n %f\n", m->ToString().c_str());

  filename = TestName() + "_after_opotimization";
  std::cout << "Start Dumping " << filename << " to " << dot_dir;
  DumpToFileInDirImpl(dot_dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));
}
TEST_F(HloMCOTest, MatrixVectorChainAsSubgraph) {
  // Test opotimization in graph which contain a matrix cahin as a subgraph
  auto builder = HloComputation::Builder(TestName());
  const std::string hlo_text = R"(
HloModule MatrixVectorChainAsSubgraph
main{
  %A = f32[40,20]{1,0} parameter(0), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %B = f32[20,30]{1,0} parameter(1), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %dot1 = f32[40,30]{1,0} dot(f32[40,20]{1,0} %A, f32[20,30]{1,0} %B), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="matmul"}
  %C = f32[30,10]{1,0} parameter(2), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %dot5 = f32[40,10]{1,0} dot(f32[40,30]{1,0} %dot1, f32[30,10]{1,0} %C), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="matmul"}
  %D = f32[10]{0} parameter(3), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %dot2 = f32[40]{0} dot(f32[40,10]{1,0} %dot5, f32[10]{0} %D), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="Einsum" op_name="einsum/Einsum"}
  %E = f32[40]{0} parameter(4), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %dot3 = f32[] dot(f32[40]{0} %dot2, f32[40]{0} %E), lhs_contracting_dims={0}, rhs_contracting_dims={0}, metadata={op_type="Einsum" op_name="einsum_1/Einsum"}
  %F = f32[] parameter(5), parameter_replication={false}, metadata={op_name="XLA_Args"}
  ROOT %add1 = f32[] add(f32[] %dot3, f32[] %F), metadata={op_type="AddV2" op_name="add"}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_text));
  
  std::string filename = TestName() + "_before_opotimization";
  // DebugOptions debug_options
  auto render_graph = [&](RenderedGraphFormat format) {
    StatusOr<string> rendered_graph =
        RenderGraph(*m->entry_computation(),
                    /*label=*/filename, m->config().debug_options(), format);
    if (rendered_graph.ok()) {
      return std::move(rendered_graph).ValueOrDie();
    }
    return absl::StrFormat("Error rendering graph: %s",
                           rendered_graph.status().ToString());
  };
  std::cout << "Start Dumping " << filename << " to " << dot_dir;
  DumpToFileInDirImpl(dot_dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));
  HloMCO pass;
  ASSERT_TRUE(pass.Run(m.get()).ValueOrDie());
  // AlgebraicSimplifierOptions default_options_;
  // AlgebraicSimplifier simplifier(default_options_);
  // simplifier.Run(m.get()).ValueOrDie();
  // TransposeFolding transpose_folding;
  // transpose_folding.Run(m.get()).ValueOrDie();
  HloDCE dce;
  RunHloPass(&dce, m.get());
  HloCSE cse(/*is_layout_sensitive=*/false);
  cse.Run(m.get()).ValueOrDie();
  HloInstruction* root = m->entry_computation()->root_instruction();
  printf("After opotimization:\n %f\n", m->ToString().c_str());

  filename = TestName() + "_after_opotimization";
  std::cout << "Start Dumping " << filename << " to " << dot_dir;
  DumpToFileInDirImpl(dot_dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));
}

TEST_F(HloMCOTest, ReusedSameSubMatrixAfterOptimizationChain) {
  // Test opotimization in graph which contain a matrix cahin as a subgraph
  // and a sub-chain of the graph is reused, but after optimization the
  // sub-chain result doesn't change
  auto builder = HloComputation::Builder(TestName());
  const std::string hlo_text = R"(
HloModule ReusedSameSubMatrixAfterOptimizationChain
main{
  %A = f32[40,20]{1,0} parameter(0)
  %B = f32[20,30]{1,0} parameter(1)
  %dot1 = f32[40,30]{1,0} dot(f32[40,20]{1,0} %A, f32[20,30]{1,0} %B), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %C = f32[30,10]{1,0} parameter(2), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %dot2 = f32[40,10]{1,0} dot(f32[40,30]{1,0} %dot1, f32[30,10]{1,0} %C), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %D = f32[10,30]{1,0} parameter(3)
  %dot3 = f32[40,30]{1,0} dot(f32[40,10]{1,0} %dot2, f32[10,30]{1,0} %D), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %E = f32[10,30]{1,0} parameter(4)
  %dot5 = f32[40,30]{1,0} dot(f32[40,10]{1,0} %dot2, f32[10,30]{1,0} %E), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %F = f32[40,30]{1,0} parameter(5)
  %add1 = f32[40,30] add(%F, %dot5)
  ROOT %add2 = f32[40,30] add(%dot3, %add1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_text));
  
  std::string filename = TestName() + "_before_opotimization";
  // DebugOptions debug_options
  auto render_graph = [&](RenderedGraphFormat format) {
    StatusOr<string> rendered_graph =
        RenderGraph(*m->entry_computation(),
                    /*label=*/filename, m->config().debug_options(), format);
    if (rendered_graph.ok()) {
      return std::move(rendered_graph).ValueOrDie();
    }
    return absl::StrFormat("Error rendering graph: %s",
                           rendered_graph.status().ToString());
  };
  std::cout << "Start Dumping " << filename << " to " << dot_dir;
  DumpToFileInDirImpl(dot_dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));
  HloMCO pass;
  ASSERT_TRUE(pass.Run(m.get()).ValueOrDie());
  AlgebraicSimplifierOptions default_options_;
  AlgebraicSimplifier simplifier(default_options_);
  simplifier.Run(m.get()).ValueOrDie();
  TransposeFolding transpose_folding;
  transpose_folding.Run(m.get()).ValueOrDie();
  HloDCE dce;
  RunHloPass(&dce, m.get());
  HloCSE cse(/*is_layout_sensitive=*/false);
  cse.Run(m.get()).ValueOrDie();
  HloInstruction* root = m->entry_computation()->root_instruction();
  printf("After opotimization:\n %f\n", m->ToString().c_str());

  filename = TestName() + "_after_opotimization";
  std::cout << "Start Dumping " << filename << " to " << dot_dir;
  DumpToFileInDirImpl(dot_dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));
}

TEST_F(HloMCOTest, ReusedDiffSubMatrixAfterOptimizationChain) {
  // Test opotimization in graph which contain a matrix cahin as a subgraph
  // and a sub-chain of the graph is reused, but after optimization the
  // sub-chain result change
  auto builder = HloComputation::Builder(TestName());
  const std::string hlo_text = R"(
HloModule ReusedDiffSubMatrixAfterOptimizationChain
main{
  %A = f32[40,20]{1,0} parameter(0)
  %B = f32[20,30]{1,0} parameter(1)
  %dot1 = f32[40,30]{1,0} dot(f32[40,20]{1,0} %A, f32[20,30]{1,0} %B), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %C = f32[30,10]{1,0} parameter(2), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %dot2 = f32[40,10]{1,0} dot(f32[40,30]{1,0} %dot1, f32[30,10]{1,0} %C), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %D = f32[10,30]{1,0} parameter(3)
  %dot3 = f32[40,30]{1,0} dot(f32[40,10]{1,0} %dot2, f32[10,30]{1,0} %D), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %E = f32[30,30]{1,0} parameter(4)
  %dot5 = f32[40,30]{1,0} dot(f32[40,30]{1,0} %dot1, f32[30,30]{1,0} %E), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %F = f32[40,30]{1,0} parameter(5)
  %add1 = f32[40,30] add(%F, %dot5)
  ROOT %add2 = f32[40,30] add(%dot3, %add1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_text));
  
  std::string filename = TestName() + "_before_opotimization";
  // DebugOptions debug_options
  auto render_graph = [&](RenderedGraphFormat format) {
    StatusOr<string> rendered_graph =
        RenderGraph(*m->entry_computation(),
                    /*label=*/filename, m->config().debug_options(), format);
    if (rendered_graph.ok()) {
      return std::move(rendered_graph).ValueOrDie();
    }
    return absl::StrFormat("Error rendering graph: %s",
                           rendered_graph.status().ToString());
  };
  std::cout << "Start Dumping " << filename << " to " << dot_dir;
  DumpToFileInDirImpl(dot_dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));
  HloMCO pass;
  ASSERT_TRUE(pass.Run(m.get()).ValueOrDie());
  AlgebraicSimplifierOptions default_options_;
  AlgebraicSimplifier simplifier(default_options_);
  simplifier.Run(m.get()).ValueOrDie();
  TransposeFolding transpose_folding;
  transpose_folding.Run(m.get()).ValueOrDie();
  HloDCE dce;
  RunHloPass(&dce, m.get());
  HloCSE cse(/*is_layout_sensitive=*/false);
  cse.Run(m.get()).ValueOrDie();
  HloInstruction* root = m->entry_computation()->root_instruction();
  printf("After opotimization:\n %f\n", m->ToString().c_str());

  filename = TestName() + "_after_opotimization";
  std::cout << "Start Dumping " << filename << " to " << dot_dir;
  DumpToFileInDirImpl(dot_dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));
}
TEST_F(HloMCOTest, ComplexChainWithRewrittenTranspose) {
  // Test opotimization in graph which rewrites transpose op to dot op with
  // contract dimensions{lhs=0,rhs=1}
  auto builder = HloComputation::Builder(TestName());
  const std::string hlo_text = R"(
HloModule ComplexChainWithRewrittenTranspose
main{
  %A = f32[40,20]{1,0} parameter(0), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %B = f32[20,30]{1,0} parameter(1), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %dot1 = f32[40,30]{1,0} dot(f32[40,20]{1,0} %A, f32[20,30]{1,0} %B), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="matmul"}
  %C = f32[30,10]{1,0} parameter(2), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %dot2 = f32[40,10]{1,0} dot(f32[40,30]{1,0} %dot1, f32[30,10]{1,0} %C), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="matmul_1"}
  %H = f32[10,10]{1,0} parameter(7), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %D = f32[30,10]{1,0} parameter(3), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %G = f32[10,10]{1,0} parameter(6), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %dot3 = f32[30,10]{1,0} dot(f32[30,10]{1,0} %D, f32[10,10]{1,0} %G), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="matmul_2"}
  %dot4 = f32[10,30]{1,0} dot(f32[10,10]{1,0} %H, f32[30,10]{1,0} %dot3), lhs_contracting_dims={0}, rhs_contracting_dims={1}, metadata={op_type="Transpose" op_name="transpose"}
  %dot5 = f32[40,30]{1,0} dot(f32[40,10]{1,0} %dot2, f32[10,30]{1,0} %dot4), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="matmul_4"}
  %E = f32[10,30]{1,0} parameter(4), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %dot6 = f32[40,30]{1,0} dot(f32[40,10]{1,0} %dot2, f32[10,30]{1,0} %E), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="matmul_7"}
  %F = f32[40,30]{1,0} parameter(5), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %add1 = f32[40,30]{1,0} add(f32[40,30]{1,0} %dot6, f32[40,30]{1,0} %F), metadata={op_type="AddV2" op_name="add"}
  ROOT %add2 = f32[40,30]{1,0} add(f32[40,30]{1,0} %dot5, f32[40,30]{1,0} %add1), metadata={op_type="AddV2" op_name="add_1"}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_text));
  
  std::string filename = TestName() + "_before_opotimization";
  // DebugOptions debug_options
  auto render_graph = [&](RenderedGraphFormat format) {
    StatusOr<string> rendered_graph =
        RenderGraph(*m->entry_computation(),
                    /*label=*/filename, m->config().debug_options(), format);
    if (rendered_graph.ok()) {
      return std::move(rendered_graph).ValueOrDie();
    }
    return absl::StrFormat("Error rendering graph: %s",
                           rendered_graph.status().ToString());
  };
  std::cout << "Start Dumping " << filename << " to " << dot_dir;
  DumpToFileInDirImpl(dot_dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));
  HloMCO pass;
  ASSERT_TRUE(pass.Run(m.get()).ValueOrDie());
  AlgebraicSimplifierOptions default_options_;
  AlgebraicSimplifier simplifier(default_options_);
  simplifier.Run(m.get()).ValueOrDie();
  TransposeFolding transpose_folding;
  transpose_folding.Run(m.get()).ValueOrDie();
  HloDCE dce;
  RunHloPass(&dce, m.get());
  HloCSE cse(/*is_layout_sensitive=*/false);
  cse.Run(m.get()).ValueOrDie();
  HloInstruction* root = m->entry_computation()->root_instruction();
  printf("After opotimization:\n %f\n", m->ToString().c_str());

  filename = TestName() + "_after_opotimization";
  std::cout << "Start Dumping " << filename << " to " << dot_dir;
  DumpToFileInDirImpl(dot_dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));
}
TEST_F(HloMCOTest, ComplexChainWithTranspose) {
  // Test opotimization in graph which rewrites transpose op to dot op with
  // contract dimensions{lhs=0,rhs=1}
  auto builder = HloComputation::Builder(TestName());
  const std::string hlo_text = R"(
HloModule ComplexChainWithTranspose
main{
  %A = f32[40,20]{1,0} parameter(0), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %B = f32[20,30]{1,0} parameter(1), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %dot1 = f32[40,30]{1,0} dot(f32[40,20]{1,0} %A, f32[20,30]{1,0} %B), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="matmul"}
  %C = f32[30,10]{1,0} parameter(2), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %dot2 = f32[40,10]{1,0} dot(f32[40,30]{1,0} %dot1, f32[30,10]{1,0} %C), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="matmul_1"}
  %H = f32[10,10]{1,0} parameter(7), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %D = f32[30,10]{1,0} parameter(3), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %G = f32[10,10]{1,0} parameter(6), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %dot3 = f32[30,10]{1,0} dot(f32[30,10]{1,0} %D, f32[10,10]{1,0} %G), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="matmul_2"}
  %dot4 = f32[30,10]{1,0} dot(f32[30,10]{1,0} %dot3, f32[10,10]{1,0} %H ), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="matmul_3"}
  %transpose = f32[10,30]{0,1} transpose(f32[30,10]{1,0} %dot4), dimensions={1,0}, metadata={op_type="Transpose" op_name="transpose"}
  %dot5 = f32[40,30]{1,0} dot(f32[40,10]{1,0} %dot2, f32[10,30]{0,1} %transpose), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="matmul_4"}
  %E = f32[10,30]{1,0} parameter(4), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %dot6 = f32[40,30]{1,0} dot(f32[40,10]{1,0} %dot2, f32[10,30]{1,0} %E), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="matmul_7"}
  %F = f32[40,30]{1,0} parameter(5), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %add1 = f32[40,30]{1,0} add(f32[40,30]{1,0} %dot6, f32[40,30]{1,0} %F), metadata={op_type="AddV2" op_name="add"}
  ROOT %add2 = f32[40,30]{1,0} add(f32[40,30]{1,0} %dot5, f32[40,30]{1,0} %add1), metadata={op_type="AddV2" op_name="add_1"}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_text));
  
  std::string filename = TestName() + "_before_opotimization";
  // DebugOptions debug_options
  auto render_graph = [&](RenderedGraphFormat format) {
    StatusOr<string> rendered_graph =
        RenderGraph(*m->entry_computation(),
                    /*label=*/filename, m->config().debug_options(), format);
    if (rendered_graph.ok()) {
      return std::move(rendered_graph).ValueOrDie();
    }
    return absl::StrFormat("Error rendering graph: %s",
                           rendered_graph.status().ToString());
  };
  std::cout << "Start Dumping " << filename << " to " << dot_dir;
  DumpToFileInDirImpl(dot_dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));
  HloMCO pass;
  ASSERT_TRUE(pass.Run(m.get()).ValueOrDie());
  AlgebraicSimplifierOptions default_options_;
  AlgebraicSimplifier simplifier(default_options_);
  simplifier.Run(m.get()).ValueOrDie();
  TransposeFolding transpose_folding;
  transpose_folding.Run(m.get()).ValueOrDie();
  HloDCE dce;
  RunHloPass(&dce, m.get());
  HloCSE cse(/*is_layout_sensitive=*/false);
  cse.Run(m.get()).ValueOrDie();
  HloInstruction* root = m->entry_computation()->root_instruction();
  printf("After opotimization:\n %f\n", m->ToString().c_str());

  filename = TestName() + "_after_opotimization";
  std::cout << "Start Dumping " << filename << " to " << dot_dir;
  DumpToFileInDirImpl(dot_dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));
}

TEST_F(HloMCOTest, NestedRewrittenTransposeChain) {
  // Test opotimization in graph which rewrites transpose op to dot op with
  // contract dimensions{lhs=0,rhs=1}
  auto builder = HloComputation::Builder(TestName());
  const std::string hlo_text = R"(
HloModule NestedRewrittenTransposeChain
main{
  %arg5.6 = f32[5,10]{1,0} parameter(5), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg4.5 = f32[20,5]{1,0} parameter(4), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %dot = f32[10,20]{1,0} dot(f32[5,10]{1,0} %arg5.6, f32[20,5]{1,0} %arg4.5), lhs_contracting_dims={0}, rhs_contracting_dims={1}, metadata={op_type="Transpose" op_name="transpose"}
  %arg3.4 = f32[30,10]{1,0} parameter(3), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %dot.1 = f32[20,30]{1,0} dot(f32[10,20]{1,0} %dot, f32[30,10]{1,0} %arg3.4), lhs_contracting_dims={0}, rhs_contracting_dims={1}, metadata={op_type="Transpose" op_name="transpose_1"}
  %arg0.1 = f32[40,20]{1,0} parameter(0), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg1.2 = f32[20,30]{1,0} parameter(1), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %dot.13 = f32[40,30]{1,0} dot(f32[40,20]{1,0} %arg0.1, f32[20,30]{1,0} %arg1.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="matmul_2"}
  %arg2.3 = f32[30,20]{1,0} parameter(2), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %dot.15 = f32[40,20]{1,0} dot(f32[40,30]{1,0} %dot.13, f32[30,20]{1,0} %arg2.3), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="matmul_3"}
  ROOT %dot.2 = f32[30,40]{1,0} dot(f32[20,30]{1,0} %dot.1, f32[40,20]{1,0} %dot.15), lhs_contracting_dims={0}, rhs_contracting_dims={1}, metadata={op_type="Transpose" op_name="transpose_2"}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_text));
  
  std::string filename = TestName() + "_before_opotimization";
  // DebugOptions debug_options
  auto render_graph = [&](RenderedGraphFormat format) {
    StatusOr<string> rendered_graph =
        RenderGraph(*m->entry_computation(),
                    /*label=*/filename, m->config().debug_options(), format);
    if (rendered_graph.ok()) {
      return std::move(rendered_graph).ValueOrDie();
    }
    return absl::StrFormat("Error rendering graph: %s",
                           rendered_graph.status().ToString());
  };
  std::cout << "Start Dumping " << filename << " to " << dot_dir;
  DumpToFileInDirImpl(dot_dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));
  HloMCO pass;
  ASSERT_TRUE(pass.Run(m.get()).ValueOrDie());
  AlgebraicSimplifierOptions default_options_;
  AlgebraicSimplifier simplifier(default_options_);
  simplifier.Run(m.get()).ValueOrDie();
  TransposeFolding transpose_folding;
  transpose_folding.Run(m.get()).ValueOrDie();
  HloDCE dce;
  RunHloPass(&dce, m.get());
  HloCSE cse(/*is_layout_sensitive=*/false);
  cse.Run(m.get()).ValueOrDie();
  HloInstruction* root = m->entry_computation()->root_instruction();
  printf("After opotimization:\n %f\n", m->ToString().c_str());

  filename = TestName() + "_after_opotimization";
  std::cout << "Start Dumping " << filename << " to " << dot_dir;
  DumpToFileInDirImpl(dot_dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));
}

TEST_F(HloMCOTest, OptimalMatrixVectorTransDotChain) {
  // Test opotimization in graph which rewrites transpose op to dot op with
  // contract dimensions{lhs=0,rhs=1}
  auto builder = HloComputation::Builder(TestName());
  const std::string hlo_text = R"(
HloModule OptimalMatrixVectorTransDotChain
main{
  %D = f32[10,30]{1,0} parameter(3), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %A = f32[40,20]{1,0} parameter(0), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %B = f32[20]{0} parameter(1), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %dot.9 = f32[40]{0} dot(f32[40,20]{1,0} %A, f32[20]{0} %B), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="Einsum" op_name="einsum/Einsum"}
  %C = f32[40,30]{1,0} parameter(2), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %dot.11 = f32[30]{0} dot(f32[40]{0} %dot.9, f32[40,30]{1,0} %C), lhs_contracting_dims={0}, rhs_contracting_dims={0}, metadata={op_type="Einsum" op_name="einsum_1/Einsum"}
  ROOT %dot.13 = f32[10]{0} dot(f32[10,30]{1,0} %D, f32[30]{0} %dot.11), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="Einsum" op_name="einsum_2/Einsum"}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_text));
  
  std::string filename = TestName() + "_before_opotimization";
  // DebugOptions debug_options
  auto render_graph = [&](RenderedGraphFormat format) {
    StatusOr<string> rendered_graph =
        RenderGraph(*m->entry_computation(),
                    /*label=*/filename, m->config().debug_options(), format);
    if (rendered_graph.ok()) {
      return std::move(rendered_graph).ValueOrDie();
    }
    return absl::StrFormat("Error rendering graph: %s",
                           rendered_graph.status().ToString());
  };
  std::cout << "Start Dumping " << filename << " to " << dot_dir;
  DumpToFileInDirImpl(dot_dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));
  HloMCO pass;
  ASSERT_TRUE(pass.Run(m.get()).ValueOrDie());
  // AlgebraicSimplifierOptions default_options_;
  // AlgebraicSimplifier simplifier(default_options_);
  // simplifier.Run(m.get()).ValueOrDie();
  // TransposeFolding transpose_folding;
  // transpose_folding.Run(m.get()).ValueOrDie();
  HloDCE dce;
  RunHloPass(&dce, m.get());
  HloCSE cse(/*is_layout_sensitive=*/false);
  cse.Run(m.get()).ValueOrDie();
  HloInstruction* root = m->entry_computation()->root_instruction();
  printf("After opotimization:\n %f\n", m->ToString().c_str());

  filename = TestName() + "_after_opotimization";
  std::cout << "Start Dumping " << filename << " to " << dot_dir;
  DumpToFileInDirImpl(dot_dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));
}

TEST_F(HloMCOTest, MatrixVectorTransDotChain) {
  // Test opotimization in graph which rewrites transpose op to dot op with
  // contract dimensions{lhs=0,rhs=1}
  auto builder = HloComputation::Builder(TestName());
  const std::string hlo_text = R"(
HloModule MatrixVectorTransDotChain

%Sum-reduction.11 (x.12: f32[], y.13: f32[]) -> f32[] {
  %x.12 = f32[] parameter(0)
  %y.13 = f32[] parameter(1)
  ROOT %add.14 = f32[] add(f32[] %x.12, f32[] %y.13)
}

main{
  %arg3.4 = f32[10,30]{1,0} parameter(3), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg2.3 = f32[40,30]{1,0} parameter(2), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %dot = f32[10,40]{1,0} dot(f32[10,30]{1,0} %arg3.4, f32[40,30]{1,0} %arg2.3), lhs_contracting_dims={1}, rhs_contracting_dims={1}, metadata={op_type="MatMul" op_name="matmul"}
  %arg0.1 = f32[40,20]{1,0} parameter(0), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg1.2 = f32[20]{0} parameter(1), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %dot.9 = f32[40]{0} dot(f32[40,20]{1,0} %arg0.1, f32[20]{0} %arg1.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="Einsum" op_name="einsum/Einsum"}
  ROOT %dot.14 = f32[10]{0} dot(f32[10,40]{1,0} %dot, f32[40]{0} %dot.9), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="Einsum" op_name="einsum_1/Einsum"}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_text));
  
  std::string filename = TestName() + "_before_opotimization";
  // DebugOptions debug_options
  auto render_graph = [&](RenderedGraphFormat format) {
    StatusOr<string> rendered_graph =
        RenderGraph(*m->entry_computation(),
                    /*label=*/filename, m->config().debug_options(), format);
    if (rendered_graph.ok()) {
      return std::move(rendered_graph).ValueOrDie();
    }
    return absl::StrFormat("Error rendering graph: %s",
                           rendered_graph.status().ToString());
  };
  std::cout << "Start Dumping " << filename << " to " << dot_dir;
  DumpToFileInDirImpl(dot_dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));
  HloMCO pass;
  ASSERT_TRUE(pass.Run(m.get()).ValueOrDie());
  // AlgebraicSimplifierOptions default_options_;
  // AlgebraicSimplifier simplifier(default_options_);
  // simplifier.Run(m.get()).ValueOrDie();
  // TransposeFolding transpose_folding;
  // transpose_folding.Run(m.get()).ValueOrDie();
  HloDCE dce;
  RunHloPass(&dce, m.get());
  HloCSE cse(/*is_layout_sensitive=*/false);
  cse.Run(m.get()).ValueOrDie();
  HloInstruction* root = m->entry_computation()->root_instruction();
  printf("After opotimization:\n %f\n", m->ToString().c_str());

  filename = TestName() + "_after_opotimization";
  std::cout << "Start Dumping " << filename << " to " << dot_dir;
  DumpToFileInDirImpl(dot_dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));
}

TEST_F(HloMCOTest, ReduceSumDotChain) {
  // Test opotimization in graph which rewrites transpose op to dot op with
  // contract dimensions{lhs=0,rhs=1}
  auto builder = HloComputation::Builder(TestName());
  const std::string hlo_text = R"(
HloModule ReduceSumDotChain

%Sum-reduction.11 (x.12: f32[], y.13: f32[]) -> f32[] {
  %x.12 = f32[] parameter(0)
  %y.13 = f32[] parameter(1)
  ROOT %add.14 = f32[] add(f32[] %x.12, f32[] %y.13)
}

main{
  %A = f32[40,20]{1,0} parameter(0), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %B = f32[20,30]{1,0} parameter(1), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %C = f32[30,10]{1,0} parameter(2), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %dot.52 = f32[20,10]{1,0} dot(f32[20,30]{1,0} %B, f32[30,10]{1,0} %C), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %dot.53 = f32[40,10]{1,0} dot(f32[40,20]{1,0} %A, f32[20,10]{1,0} %dot.52), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %constant.9 = f32[] constant(0), metadata={op_type="Sum" op_name="Sum" source_file="/vol/bitbucket/ya321/codes/gambit/try.py" source_line=117}
  ROOT %reduce.15 = f32[40]{0} reduce(f32[40,10]{1,0} %dot.53, f32[] %constant.9), dimensions={1}, to_apply=%Sum-reduction.11, metadata={op_type="Sum" op_name="Sum" source_file="/vol/bitbucket/ya321/codes/gambit/try.py" source_line=117}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_text));
  
  std::string filename = TestName() + "_before_opotimization";
  // DebugOptions debug_options
  auto render_graph = [&](RenderedGraphFormat format) {
    StatusOr<string> rendered_graph =
        RenderGraph(*m->entry_computation(),
                    /*label=*/filename, m->config().debug_options(), format);
    if (rendered_graph.ok()) {
      return std::move(rendered_graph).ValueOrDie();
    }
    return absl::StrFormat("Error rendering graph: %s",
                           rendered_graph.status().ToString());
  };
  std::cout << "Start Dumping " << filename << " to " << dot_dir;
  DumpToFileInDirImpl(dot_dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));
  HloMCO pass;
  ASSERT_TRUE(pass.Run(m.get()).ValueOrDie());
  // AlgebraicSimplifierOptions default_options_;
  // AlgebraicSimplifier simplifier(default_options_);
  // simplifier.Run(m.get()).ValueOrDie();
  // TransposeFolding transpose_folding;
  // transpose_folding.Run(m.get()).ValueOrDie();
  HloDCE dce;
  RunHloPass(&dce, m.get());
  HloCSE cse(/*is_layout_sensitive=*/false);
  cse.Run(m.get()).ValueOrDie();
  HloInstruction* root = m->entry_computation()->root_instruction();
  printf("After opotimization:\n %f\n", m->ToString().c_str());

  filename = TestName() + "_after_opotimization";
  std::cout << "Start Dumping " << filename << " to " << dot_dir;
  DumpToFileInDirImpl(dot_dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));
}

TEST_F(HloMCOTest, ReduceSumOuterDotChain) {
  // Test opotimization in graph which rewrites transpose op to dot op with
  // contract dimensions{lhs=0,rhs=1}
  auto builder = HloComputation::Builder(TestName());
  const std::string hlo_text = R"(
HloModule ReduceSumOuterDotChain

%Sum-reduction.12 (x.13: f32[], y.14: f32[]) -> f32[] {
  %x.13 = f32[] parameter(0)
  %y.14 = f32[] parameter(1)
  ROOT %add.15 = f32[] add(f32[] %x.13, f32[] %y.14)
}

main (arg0.1: f32[40,20], arg1.2: f32[20,30], arg2.3: f32[30,10], arg3.4: f32[10]) -> f32[40,10] {
  %arg0.1 = f32[40,20]{1,0} parameter(0), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg1.2 = f32[20,30]{1,0} parameter(1), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %arg2.3 = f32[30,10]{1,0} parameter(2), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %dot = f32[20,10]{1,0} dot(f32[20,30]{1,0} %arg1.2, f32[30,10]{1,0} %arg2.3), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %dot.1 = f32[40,10]{1,0} dot(f32[40,20]{1,0} %arg0.1, f32[20,10]{1,0} %dot), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="matmul_1" source_file="/vol/bitbucket/ya321/codes/gambit/try.py" source_line=118}
  %constant.10 = f32[] constant(0), metadata={op_type="Sum" op_name="Sum" source_file="/vol/bitbucket/ya321/codes/gambit/try.py" source_line=118}
  %reduce.16 = f32[40]{0} reduce(f32[40,10]{1,0} %dot.1, f32[] %constant.10), dimensions={1}, to_apply=%Sum-reduction.12, metadata={op_type="Sum" op_name="Sum" source_file="/vol/bitbucket/ya321/codes/gambit/try.py" source_line=118}
  %reshape = f32[40,1]{1,0} reshape(f32[40]{0} %reduce.16)
  %reshape.3 = f32[40]{0} reshape(f32[40,1]{1,0} %reshape)
  %arg3.4 = f32[10]{0} parameter(3), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.1 = f32[1,10]{1,0} reshape(f32[10]{0} %arg3.4)
  %reshape.4 = f32[10]{0} reshape(f32[1,10]{1,0} %reshape.1)
  ROOT %dot.3 = f32[40,10]{1,0} dot(f32[40]{0} %reshape.3, f32[10]{0} %reshape.4), lhs_contracting_dims={}, rhs_contracting_dims={}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_text));
  
  std::string filename = TestName() + "_before_opotimization";
  // DebugOptions debug_options
  auto render_graph = [&](RenderedGraphFormat format) {
    StatusOr<string> rendered_graph =
        RenderGraph(*m->entry_computation(),
                    /*label=*/filename, m->config().debug_options(), format);
    if (rendered_graph.ok()) {
      return std::move(rendered_graph).ValueOrDie();
    }
    return absl::StrFormat("Error rendering graph: %s",
                           rendered_graph.status().ToString());
  };
  std::cout << "Start Dumping " << filename << " to " << dot_dir;
  DumpToFileInDirImpl(dot_dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));
  HloMCO pass;
  ASSERT_TRUE(pass.Run(m.get()).ValueOrDie());
  // AlgebraicSimplifierOptions default_options_;
  // AlgebraicSimplifier simplifier(default_options_);
  // simplifier.Run(m.get()).ValueOrDie();
  // TransposeFolding transpose_folding;
  // transpose_folding.Run(m.get()).ValueOrDie();
  HloDCE dce;
  RunHloPass(&dce, m.get());
  HloCSE cse(/*is_layout_sensitive=*/false);
  cse.Run(m.get()).ValueOrDie();
  HloInstruction* root = m->entry_computation()->root_instruction();
  printf("After opotimization:\n %f\n", m->ToString().c_str());

  filename = TestName() + "_after_opotimization";
  std::cout << "Start Dumping " << filename << " to " << dot_dir;
  DumpToFileInDirImpl(dot_dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));
}

}  // namespace
}  // namespace xla