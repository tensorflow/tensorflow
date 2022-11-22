// License TODO ....

#include "tensorflow/compiler/xla/service/tensor_splitter.h"

#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_graph_dumper.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/path.h"

namespace xla {
namespace {

std::string splitter_dir =
    "/vol/bitbucket/ya321/codes/MscProject/test_output/tensor_splitter";

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

namespace m = match;

class TensorSplitterTest : public HloTestBase {
 protected:
  const int64_t max_size() {
    return std::get<0>(TensorSplitter::SplitSettings());
  }

  const int64_t large_dim() { return max_size() / 32 * 8 / 3; }

  const int64_t max_op_bytes_in_graph(HloInstruction* inst) {
    int64_t max_size = 0;
    max_size = std::max(max_size, ShapeUtil::ByteSizeOf(inst->shape(), 8));
    for (HloInstruction* op : inst->operands()) {
      max_size = std::max(max_size, max_op_bytes_in_graph(op));
    }
    return max_size;
  }

  const bool graph_needs_split(HloInstruction* inst) {
    return max_size() < max_op_bytes_in_graph(inst);
  }

  string replace_all_in_string(string original, string find, string replace) {
    int len = find.length();
    size_t index = 0;
    while (true) {
      index = original.find(find, index);
      if (index == std::string::npos) break;
      original.replace(index, len, replace);
      index += len;
    }
    return original;
  }
};

// Test multi argument reduce (e.g. argmax)
TEST_F(TensorSplitterTest, BasicCaseLhsAfter) {
  const string module_str = R"(
HloModule BasicCaseLhsAfter

ENTRY %BasicCaseLhs (a: f32[83333333,2], b: f32[83333333,2], v: f32[83333333]) -> f32[83333333] {
  %constant = s64[] constant(0)
  %a = f32[83333333,2]{1,0} parameter(0)
  %b = f32[83333333,2]{1,0} parameter(1)
  %v = f32[83333333]{0} parameter(2)
  %constant.1 = f32[] constant(0)
  %broadcast = f32[83333333]{0} broadcast(f32[] %constant.1), dimensions={}
  %tuple = (s64[], f32[83333333,2]{1,0}, f32[83333333,2]{1,0}, f32[83333333]{0}, f32[83333333]{0}) tuple(s64[] %constant, f32[83333333,2]{1,0} %a, f32[83333333,2]{1,0} %b, f32[83333333]{0} %v, f32[83333333]{0} %broadcast)
  %tuple.3 = ((s64[], f32[83333333,2]{1,0}, f32[83333333,2]{1,0}, f32[83333333]{0}, f32[83333333]{0})) tuple((s64[], f32[83333333,2]{1,0}, f32[83333333,2]{1,0}, f32[83333333]{0}, f32[83333333]{0}) %tuple)
  %while = ((s64[], f32[83333333,2]{1,0}, f32[83333333,2]{1,0}, f32[83333333]{0}, f32[83333333]{0})) while(((s64[], f32[83333333,2]{1,0}, f32[83333333,2]{1,0}, f32[83333333]{0}, f32[83333333]{0})) %tuple.3), condition=%tensor_splitter_dot_cond, body=%merged_while_loop_0
  %get-tuple-element.9 = (s64[], f32[83333333,2]{1,0}, f32[83333333,2]{1,0}, f32[83333333]{0}, f32[83333333]{0}) get-tuple-element(((s64[], f32[83333333,2]{1,0}, f32[83333333,2]{1,0}, f32[83333333]{0}, f32[83333333]{0})) %while), index=0
  ROOT %get-tuple-element.10 = f32[83333333]{0} get-tuple-element((s64[], f32[83333333,2]{1,0}, f32[83333333,2]{1,0}, f32[83333333]{0}, f32[83333333]{0}) %get-tuple-element.9), index=4
}

)";

  string module_with_big_dims = replace_all_in_string(
      module_str, "1000", std::to_string(large_dim() / 1000));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_with_big_dims));

  std::string filename = TestName() + "_before_opotimization";
  // DebugOptions debug_options
  auto render_graph = [&](RenderedGraphFormat format) {
    StatusOr<string> rendered_graph = RenderGraph(
        *module->entry_computation(),
        /*label=*/filename, module->config().debug_options(), format);
    if (rendered_graph.ok()) {
      return std::move(rendered_graph).ValueOrDie();
    }
    return absl::StrFormat("Error rendering graph: %s",
                           rendered_graph.status().ToString());
  };
  printf("After opotimization:\n %s\n", m->ToString().c_str()) << std::endl;
  DumpToFileInDirImpl(splitter_dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));
}

// Test the most basic case: exp(AB^T)v
TEST_F(TensorSplitterTest, BasicCaseLhs) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());

  Shape a_shape = ShapeUtil::MakeShape(F32, {large_dim(), 2});
  Shape b_shape = ShapeUtil::MakeShape(F32, {large_dim(), 2});
  Shape v_shape = ShapeUtil::MakeShape(F32, {large_dim()});
  HloInstruction* a =
      builder.AddInstruction(HloInstruction::CreateParameter(0, a_shape, "a"));
  HloInstruction* b =
      builder.AddInstruction(HloInstruction::CreateParameter(1, b_shape, "b"));
  HloInstruction* v =
      builder.AddInstruction(HloInstruction::CreateParameter(2, v_shape, "v"));

  DotDimensionNumbers dnums_ab;
  dnums_ab.add_lhs_contracting_dimensions(1);
  dnums_ab.add_rhs_contracting_dimensions(1);
  Shape ab_shape = ShapeUtil::MakeShape(F32, {large_dim(), large_dim()});
  HloInstruction* ab = builder.AddInstruction(HloInstruction::CreateDot(
      ab_shape, a, b, dnums_ab, DefaultPrecisionConfig(2)));

  HloInstruction* exp_ab = builder.AddInstruction(
      HloInstruction::CreateUnary(ab_shape, HloOpcode::kExp, ab));

  DotDimensionNumbers dnums_final;
  dnums_final.add_lhs_contracting_dimensions(1);
  dnums_final.add_rhs_contracting_dimensions(0);
  Shape final_shape = ShapeUtil::MakeShape(F32, {large_dim()});
  builder.AddInstruction(HloInstruction::CreateDot(
      final_shape, exp_ab, v, dnums_final, DefaultPrecisionConfig(2)));

  HloComputation* computation = m->AddEntryComputation(builder.Build());

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
  printf("After opotimization:\n %f\n", m->ToString().c_str()) << std::endl;
  DumpToFileInDirImpl(splitter_dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));

  EXPECT_TRUE(Match(
      computation->root_instruction(),
      m::Dot(m::Exp(m::Dot(m::Op().Is(a), m::Op().Is(b))), m::Op().Is(v))));
  EXPECT_TRUE(graph_needs_split(computation->root_instruction()));

  TensorSplitter optim;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&optim, m.get()));
  EXPECT_TRUE(result);

  EXPECT_FALSE(graph_needs_split(computation->root_instruction()));

  printf("After opotimization:\n %s\n", m->ToString().c_str());
  filename = TestName() + "_after_opotimization";
  printf("After opotimization:\n %f\n", m->ToString().c_str()) << std::endl;
  DumpToFileInDirImpl(splitter_dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));
}

// Test the most basic rhs case: exp(AB^T)v
TEST_F(TensorSplitterTest, BasicCaseRhs) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());

  Shape a_shape = ShapeUtil::MakeShape(F32, {large_dim(), 2});
  Shape b_shape = ShapeUtil::MakeShape(F32, {large_dim(), 2});
  Shape v_shape = ShapeUtil::MakeShape(F32, {large_dim()});
  HloInstruction* a =
      builder.AddInstruction(HloInstruction::CreateParameter(0, a_shape, "a"));
  HloInstruction* b =
      builder.AddInstruction(HloInstruction::CreateParameter(1, b_shape, "b"));
  HloInstruction* v =
      builder.AddInstruction(HloInstruction::CreateParameter(2, v_shape, "v"));

  DotDimensionNumbers dnums_ab;
  dnums_ab.add_lhs_contracting_dimensions(1);
  dnums_ab.add_rhs_contracting_dimensions(1);
  Shape ab_shape = ShapeUtil::MakeShape(F32, {large_dim(), large_dim()});
  HloInstruction* ab = builder.AddInstruction(HloInstruction::CreateDot(
      ab_shape, a, b, dnums_ab, DefaultPrecisionConfig(2)));

  HloInstruction* exp_ab = builder.AddInstruction(
      HloInstruction::CreateUnary(ab_shape, HloOpcode::kExp, ab));

  DotDimensionNumbers dnums_final;
  dnums_final.add_lhs_contracting_dimensions(0);
  dnums_final.add_rhs_contracting_dimensions(1);
  Shape final_shape = ShapeUtil::MakeShape(F32, {large_dim()});
  builder.AddInstruction(HloInstruction::CreateDot(
      final_shape, v, exp_ab, dnums_final, DefaultPrecisionConfig(2)));

  HloComputation* computation = m->AddEntryComputation(builder.Build());

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
  printf("After opotimization:\n %f\n", m->ToString().c_str()) << std::endl;
  DumpToFileInDirImpl(splitter_dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));

  EXPECT_TRUE(Match(
      computation->root_instruction(),
      m::Dot(m::Op().Is(v), m::Exp(m::Dot(m::Op().Is(a), m::Op().Is(b))))));
  EXPECT_TRUE(graph_needs_split(computation->root_instruction()));

  TensorSplitter optim;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&optim, m.get()));
  EXPECT_TRUE(result);

  EXPECT_FALSE(graph_needs_split(computation->root_instruction()));

  printf("After opotimization:\n %s\n", m->ToString().c_str());
  filename = TestName() + "_after_opotimization";
  printf("After opotimization:\n %f\n", m->ToString().c_str()) << std::endl;
  DumpToFileInDirImpl(splitter_dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));
}

// Self dot: outer-product -> inner-product
TEST_F(TensorSplitterTest, BasicSelfDot) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());

  Shape a_shape = ShapeUtil::MakeShape(F32, {large_dim(), 2});
  Shape b_shape = ShapeUtil::MakeShape(F32, {1000, 2});
  HloInstruction* a =
      builder.AddInstruction(HloInstruction::CreateParameter(0, a_shape, "a"));
  HloInstruction* b =
      builder.AddInstruction(HloInstruction::CreateParameter(1, b_shape, "b"));

  DotDimensionNumbers dnums_ab;
  dnums_ab.add_lhs_contracting_dimensions(1);
  dnums_ab.add_rhs_contracting_dimensions(1);
  Shape ab_shape = ShapeUtil::MakeShape(F32, {large_dim(), 1000});
  HloInstruction* ab = builder.AddInstruction(HloInstruction::CreateDot(
      ab_shape, a, b, dnums_ab, DefaultPrecisionConfig(2)));

  HloInstruction* exp_ab = builder.AddInstruction(
      HloInstruction::CreateUnary(ab_shape, HloOpcode::kExp, ab));

  DotDimensionNumbers dnums_final;
  dnums_final.add_lhs_contracting_dimensions(0);
  dnums_final.add_rhs_contracting_dimensions(0);
  Shape final_shape = ShapeUtil::MakeShape(F32, {1000, 1000});
  builder.AddInstruction(HloInstruction::CreateDot(
      final_shape, exp_ab, exp_ab, dnums_final, DefaultPrecisionConfig(2)));

  HloComputation* computation = m->AddEntryComputation(builder.Build());

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
  printf("After opotimization:\n %f\n", m->ToString().c_str()) << std::endl;
  DumpToFileInDirImpl(splitter_dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));

  EXPECT_TRUE(Match(computation->root_instruction(),
                    m::Dot(m::Exp(m::Dot(m::Op().Is(a), m::Op().Is(b))),
                           m::Exp(m::Dot(m::Op().Is(a), m::Op().Is(b))))));
  EXPECT_TRUE(graph_needs_split(computation->root_instruction()));

  TensorSplitter optim;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&optim, m.get()));
  EXPECT_TRUE(result);

  EXPECT_FALSE(graph_needs_split(computation->root_instruction()));

  printf("After opotimization:\n %s\n", m->ToString().c_str());
  filename = TestName() + "_after_opotimization";
  printf("After opotimization:\n %f\n", m->ToString().c_str()) << std::endl;
  DumpToFileInDirImpl(splitter_dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));
}

// Test the case where the to split dimension lies on the
// rhs of the source dot
TEST_F(TensorSplitterTest, BasicSplitDotOnRhs) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());

  Shape a_shape = ShapeUtil::MakeShape(F32, {large_dim(), 2});
  Shape b_shape = ShapeUtil::MakeShape(F32, {large_dim(), 2});
  Shape v_shape = ShapeUtil::MakeShape(F32, {large_dim()});
  HloInstruction* a =
      builder.AddInstruction(HloInstruction::CreateParameter(0, a_shape, "a"));
  HloInstruction* b =
      builder.AddInstruction(HloInstruction::CreateParameter(1, b_shape, "b"));
  HloInstruction* v =
      builder.AddInstruction(HloInstruction::CreateParameter(2, v_shape, "v"));

  DotDimensionNumbers dnums_ab;
  dnums_ab.add_lhs_contracting_dimensions(1);
  dnums_ab.add_rhs_contracting_dimensions(1);
  Shape ab_shape = ShapeUtil::MakeShape(F32, {large_dim(), large_dim()});
  HloInstruction* ab = builder.AddInstruction(HloInstruction::CreateDot(
      ab_shape, a, b, dnums_ab, DefaultPrecisionConfig(2)));

  HloInstruction* exp_ab = builder.AddInstruction(
      HloInstruction::CreateUnary(ab_shape, HloOpcode::kExp, ab));

  DotDimensionNumbers dnums_final;
  dnums_final.add_lhs_contracting_dimensions(0);
  dnums_final.add_rhs_contracting_dimensions(0);
  Shape final_shape = ShapeUtil::MakeShape(F32, {large_dim()});
  builder.AddInstruction(HloInstruction::CreateDot(
      final_shape, exp_ab, v, dnums_final, DefaultPrecisionConfig(2)));

  HloComputation* computation = m->AddEntryComputation(builder.Build());

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
  printf("After opotimization:\n %f\n", m->ToString().c_str()) << std::endl;
  DumpToFileInDirImpl(splitter_dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));

  EXPECT_TRUE(Match(
      computation->root_instruction(),
      m::Dot(m::Exp(m::Dot(m::Op().Is(a), m::Op().Is(b))), m::Op().Is(v))));
  EXPECT_TRUE(graph_needs_split(computation->root_instruction()));

  TensorSplitter optim;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&optim, m.get()));
  EXPECT_TRUE(result);

  EXPECT_FALSE(graph_needs_split(computation->root_instruction()));

  printf("After opotimization:\n %s\n", m->ToString().c_str());
  filename = TestName() + "_after_opotimization";
  printf("After opotimization:\n %f\n", m->ToString().c_str()) << std::endl;
  DumpToFileInDirImpl(splitter_dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));
}

// Test broadcast instructions as source
TEST_F(TensorSplitterTest, Broadcast) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());

  Shape param_shape = ShapeUtil::MakeShape(F32, {});
  HloInstruction* p = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "p"));

  Shape broadcast_shape = ShapeUtil::MakeShape(F32, {large_dim(), large_dim()});
  std::vector<int64_t> dims = {};
  HloInstruction* broadcast =
      builder.AddInstruction(HloInstruction::CreateBroadcast(
          broadcast_shape, p, absl::MakeSpan(dims)));

  Shape v_shape = ShapeUtil::MakeShape(F32, {large_dim()});
  HloInstruction* v =
      builder.AddInstruction(HloInstruction::CreateParameter(1, v_shape, "v"));

  DotDimensionNumbers dnums;
  dnums.add_lhs_contracting_dimensions(1);
  dnums.add_rhs_contracting_dimensions(0);
  HloInstruction* dot = builder.AddInstruction(HloInstruction::CreateDot(
      v_shape, broadcast, v, dnums, DefaultPrecisionConfig(2)));

  HloComputation* computation = m->AddEntryComputation(builder.Build());

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
  printf("After opotimization:\n %f\n", m->ToString().c_str()) << std::endl;
  DumpToFileInDirImpl(splitter_dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));

  EXPECT_TRUE(Match(computation->root_instruction(),
                    m::Dot(m::Broadcast(m::Op().Is(p)), m::Op().Is(v))));
  EXPECT_TRUE(graph_needs_split(computation->root_instruction()));

  TensorSplitter optim;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&optim, m.get()));
  EXPECT_TRUE(result);

  EXPECT_FALSE(graph_needs_split(computation->root_instruction()));

  printf("After opotimization:\n %s\n", m->ToString().c_str());
  filename = TestName() + "_after_opotimization";
  printf("After opotimization:\n %f\n", m->ToString().c_str()) << std::endl;
  DumpToFileInDirImpl(splitter_dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));
}

// Test broadcast instructions as source when split dim
// is a real dimension
TEST_F(TensorSplitterTest, BroadcastSplitOnOperandDim) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());

  Shape param_shape = ShapeUtil::MakeShape(F32, {large_dim()});
  HloInstruction* p = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "p"));

  Shape broadcast_shape = ShapeUtil::MakeShape(F32, {large_dim(), large_dim()});
  std::vector<int64_t> dims = {0};
  HloInstruction* broadcast =
      builder.AddInstruction(HloInstruction::CreateBroadcast(
          broadcast_shape, p, absl::MakeSpan(dims)));

  Shape v_shape = ShapeUtil::MakeShape(F32, {large_dim()});
  HloInstruction* v =
      builder.AddInstruction(HloInstruction::CreateParameter(1, v_shape, "v"));

  DotDimensionNumbers dnums;
  dnums.add_lhs_contracting_dimensions(0);
  dnums.add_rhs_contracting_dimensions(0);
  HloInstruction* dot = builder.AddInstruction(HloInstruction::CreateDot(
      v_shape, broadcast, v, dnums, DefaultPrecisionConfig(2)));

  HloComputation* computation = m->AddEntryComputation(builder.Build());

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
  printf("After opotimization:\n %f\n", m->ToString().c_str()) << std::endl;
  DumpToFileInDirImpl(splitter_dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));

  EXPECT_TRUE(Match(computation->root_instruction(),
                    m::Dot(m::Broadcast(m::Op().Is(p)), m::Op().Is(v))));
  EXPECT_TRUE(graph_needs_split(computation->root_instruction()));

  TensorSplitter optim;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&optim, m.get()));
  EXPECT_TRUE(result);

  EXPECT_FALSE(graph_needs_split(computation->root_instruction()));

  printf("After opotimization:\n %s\n", m->ToString().c_str());
  filename = TestName() + "_after_opotimization";
  printf("After opotimization:\n %f\n", m->ToString().c_str()) << std::endl;
  DumpToFileInDirImpl(splitter_dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));
}

// Test iota with iota dimension along split
TEST_F(TensorSplitterTest, IotaSplitAlongIotaDim) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());

  Shape iota_shape = ShapeUtil::MakeShape(F32, {large_dim(), large_dim()});
  Shape param_shape = ShapeUtil::MakeShape(F32, {large_dim()});

  HloInstruction* iota =
      builder.AddInstruction(HloInstruction::CreateIota(iota_shape, 0));
  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "p"));

  DotDimensionNumbers dnums;
  dnums.add_lhs_contracting_dimensions(1);
  dnums.add_rhs_contracting_dimensions(0);
  HloInstruction* dot = builder.AddInstruction(HloInstruction::CreateDot(
      param_shape, iota, param, dnums, DefaultPrecisionConfig(2)));

  HloComputation* computation = m->AddEntryComputation(builder.Build());

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
  printf("After opotimization:\n %f\n", m->ToString().c_str()) << std::endl;
  DumpToFileInDirImpl(splitter_dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));

  EXPECT_TRUE(Match(computation->root_instruction(),
                    m::Dot(m::Iota().Is(iota), m::Op().Is(param))));
  EXPECT_TRUE(graph_needs_split(computation->root_instruction()));

  TensorSplitter optim;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&optim, m.get()));
  EXPECT_TRUE(result);

  EXPECT_FALSE(graph_needs_split(computation->root_instruction()));

  printf("After opotimization:\n %s\n", m->ToString().c_str());
  filename = TestName() + "_after_opotimization";
  printf("After opotimization:\n %f\n", m->ToString().c_str()) << std::endl;
  DumpToFileInDirImpl(splitter_dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));
}

// Test iota with non-iota dimension along split
TEST_F(TensorSplitterTest, IotaSplitAlongNonIotaDim) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());

  Shape iota_shape = ShapeUtil::MakeShape(F32, {large_dim(), large_dim()});
  Shape param_shape = ShapeUtil::MakeShape(F32, {large_dim()});

  HloInstruction* iota =
      builder.AddInstruction(HloInstruction::CreateIota(iota_shape, 1));
  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "p"));

  DotDimensionNumbers dnums;
  dnums.add_lhs_contracting_dimensions(1);
  dnums.add_rhs_contracting_dimensions(0);
  HloInstruction* dot = builder.AddInstruction(HloInstruction::CreateDot(
      param_shape, iota, param, dnums, DefaultPrecisionConfig(2)));

  HloComputation* computation = m->AddEntryComputation(builder.Build());

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
  printf("After opotimization:\n %f\n", m->ToString().c_str()) << std::endl;
  DumpToFileInDirImpl(splitter_dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));

  EXPECT_TRUE(Match(computation->root_instruction(),
                    m::Dot(m::Iota().Is(iota), m::Op().Is(param))));
  EXPECT_TRUE(graph_needs_split(computation->root_instruction()));

  TensorSplitter optim;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&optim, m.get()));
  EXPECT_TRUE(result);

  EXPECT_FALSE(graph_needs_split(computation->root_instruction()));

  printf("After opotimization:\n %s\n", m->ToString().c_str());
  filename = TestName() + "_after_opotimization";
  printf("After opotimization:\n %f\n", m->ToString().c_str()) << std::endl;
  DumpToFileInDirImpl(splitter_dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));
}

// Test single argument reduce (e.g. max)
TEST_F(TensorSplitterTest, SingleOperandReduce) {
  auto m = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());
  HloComputation::Builder max_builder(TestName() + ".max");

  Shape empty_shape = ShapeUtil::MakeShape(F32, {});
  HloInstruction* x = max_builder.AddInstruction(
      HloInstruction::CreateParameter(0, empty_shape, "x"));
  HloInstruction* y = max_builder.AddInstruction(
      HloInstruction::CreateParameter(1, empty_shape, "y"));
  max_builder.AddInstruction(
      HloInstruction::CreateBinary(empty_shape, HloOpcode::kMaximum, x, y));
  HloComputation* max = m->AddEmbeddedComputation(max_builder.Build());

  Shape big_shape = ShapeUtil::MakeShape(F32, {large_dim(), large_dim()});
  HloInstruction* a =
      builder.AddInstruction(HloInstruction::CreateIota(big_shape, 0));

  HloInstruction* init = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0)));

  Shape small_shape = ShapeUtil::MakeShape(F32, {large_dim()});
  builder.AddInstruction(
      HloInstruction::CreateReduce(small_shape, a, init, {1}, max));

  HloComputation* computation = m->AddEntryComputation(builder.Build());

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
  printf("After opotimization:\n %f\n", m->ToString().c_str()) << std::endl;
  DumpToFileInDirImpl(splitter_dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));

  EXPECT_TRUE(Match(computation->root_instruction(),
                    m::Reduce(m::Op().Is(a), m::Op().Is(init))));
  EXPECT_TRUE(graph_needs_split(computation->root_instruction()));

  TensorSplitter optim;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&optim, m.get()));
  EXPECT_TRUE(result);

  EXPECT_FALSE(graph_needs_split(computation->root_instruction()));

  printf("After opotimization:\n %s\n", m->ToString().c_str());
  filename = TestName() + "_after_opotimization";
  printf("After opotimization:\n %f\n", m->ToString().c_str()) << std::endl;
  DumpToFileInDirImpl(splitter_dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));
}

// Test multi argument reduce (e.g. argmax)
TEST_F(TensorSplitterTest, MultiOperandReduce) {
  const string module_str = R"(
HloModule a_inference_arg_max_test_29__XlaMustCompile_true_config_proto___n_007_n_0...02_001_000__executor_type____.35

%minmax_func.17 (lhs_value.18: f32[], lhs_index.19: s32[], rhs_value.20: f32[], rhs_index.21: s32[]) -> (f32[], s32[]) {
  %lhs_value.18 = f32[] parameter(0)
  %rhs_value.20 = f32[] parameter(2)
  %compare.22 = pred[] compare(f32[] %lhs_value.18, f32[] %rhs_value.20), direction=GE
  %select.23 = f32[] select(pred[] %compare.22, f32[] %lhs_value.18, f32[] %rhs_value.20)
  %compare.25 = pred[] compare(f32[] %lhs_value.18, f32[] %rhs_value.20), direction=EQ
  %lhs_index.19 = s32[] parameter(1)
  %rhs_index.21 = s32[] parameter(3)
  %minimum.26 = s32[] minimum(s32[] %lhs_index.19, s32[] %rhs_index.21)
  %select.24 = s32[] select(pred[] %compare.22, s32[] %lhs_index.19, s32[] %rhs_index.21)
  %select.27 = s32[] select(pred[] %compare.25, s32[] %minimum.26, s32[] %select.24)
  ROOT %tuple.28 = (f32[], s32[]) tuple(f32[] %select.23, s32[] %select.27)
}

ENTRY %a_inference_arg_max_test_29__XlaMustCompile_true_config_proto___n_007_n_0...02_001_000__executor_type____.35 (arg0.1: f32[2000,1000], arg1.2: f32[2000,1000]) -> s64[2000,2000] {
  %arg0.1 = f32[2000,1000]{1,0} parameter(0), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.3 = f32[2000,1000]{1,0} reshape(f32[2000,1000]{1,0} %arg0.1)
  %slice.5 = f32[2000,1000]{1,0} slice(f32[2000,1000]{1,0} %reshape.3), slice={[0:2000], [0:1000]}, metadata={op_type="StridedSlice" op_name="strided_slice" source_file="xla_playground.py" source_line=224}
  %reshape.6 = f32[1,2000,1000]{2,1,0} reshape(f32[2000,1000]{1,0} %slice.5), metadata={op_type="StridedSlice" op_name="strided_slice" source_file="xla_playground.py" source_line=224}
  %reshape.9 = f32[2000,1000]{1,0} reshape(f32[1,2000,1000]{2,1,0} %reshape.6), metadata={op_type="Sub" op_name="sub" source_file="xla_playground.py" source_line=224}
  %broadcast.10 = f32[2000,2000,1000]{2,1,0} broadcast(f32[2000,1000]{1,0} %reshape.9), dimensions={1,2}, metadata={op_type="Sub" op_name="sub" source_file="xla_playground.py" source_line=224}
  %arg1.2 = f32[2000,1000]{1,0} parameter(1), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.4 = f32[2000,1000]{1,0} reshape(f32[2000,1000]{1,0} %arg1.2)
  %slice.7 = f32[2000,1000]{1,0} slice(f32[2000,1000]{1,0} %reshape.4), slice={[0:2000], [0:1000]}, metadata={op_type="StridedSlice" op_name="strided_slice_1" source_file="xla_playground.py" source_line=224}
  %reshape.8 = f32[2000,1,1000]{2,1,0} reshape(f32[2000,1000]{1,0} %slice.7), metadata={op_type="StridedSlice" op_name="strided_slice_1" source_file="xla_playground.py" source_line=224}
  %reshape.11 = f32[2000,1000]{1,0} reshape(f32[2000,1,1000]{2,1,0} %reshape.8), metadata={op_type="Sub" op_name="sub" source_file="xla_playground.py" source_line=224}
  %broadcast.12 = f32[2000,2000,1000]{2,1,0} broadcast(f32[2000,1000]{1,0} %reshape.11), dimensions={0,2}, metadata={op_type="Sub" op_name="sub" source_file="xla_playground.py" source_line=224}
  %subtract.13 = f32[2000,2000,1000]{2,1,0} subtract(f32[2000,2000,1000]{2,1,0} %broadcast.10, f32[2000,2000,1000]{2,1,0} %broadcast.12), metadata={op_type="Sub" op_name="sub" source_file="xla_playground.py" source_line=224}
  %iota.16 = s32[2000,2000,1000]{2,1,0} iota(), iota_dimension=2, metadata={op_type="ArgMax" op_name="ArgMax" source_file="xla_playground.py" source_line=225}
  %constant.14 = f32[] constant(-inf), metadata={op_type="ArgMax" op_name="ArgMax" source_file="xla_playground.py" source_line=225}
  %constant.15 = s32[] constant(0), metadata={op_type="ArgMax" op_name="ArgMax" source_file="xla_playground.py" source_line=225}
  %reduce.29 = (f32[2000,2000]{1,0}, s32[2000,2000]{1,0}) reduce(f32[2000,2000,1000]{2,1,0} %subtract.13, s32[2000,2000,1000]{2,1,0} %iota.16, f32[] %constant.14, s32[] %constant.15), dimensions={2}, to_apply=%minmax_func.17, metadata={op_type="ArgMax" op_name="ArgMax" source_file="xla_playground.py" source_line=225}
  %get-tuple-element.30 = s32[2000,2000]{1,0} get-tuple-element((f32[2000,2000]{1,0}, s32[2000,2000]{1,0}) %reduce.29), index=1, metadata={op_type="ArgMax" op_name="ArgMax" source_file="xla_playground.py" source_line=225}
  %convert.31 = s64[2000,2000]{1,0} convert(s32[2000,2000]{1,0} %get-tuple-element.30), metadata={op_type="ArgMax" op_name="ArgMax" source_file="xla_playground.py" source_line=225}
  %reshape.32 = s64[2000,2000]{1,0} reshape(s64[2000,2000]{1,0} %convert.31), metadata={op_name="XLA_Retvals"}
  %tuple.33 = (s64[2000,2000]{1,0}) tuple(s64[2000,2000]{1,0} %reshape.32), metadata={op_name="XLA_Retvals"}
  ROOT %get-tuple-element.34 = s64[2000,2000]{1,0} get-tuple-element((s64[2000,2000]{1,0}) %tuple.33), index=0, metadata={op_name="XLA_Retvals"}
}
)";

  string module_with_big_dims = replace_all_in_string(
      module_str, "1000", std::to_string(large_dim() / 1000));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_with_big_dims));

  std::string filename = TestName() + "_before_opotimization";
  // DebugOptions debug_options
  auto render_graph = [&](RenderedGraphFormat format) {
    StatusOr<string> rendered_graph = RenderGraph(
        *module->entry_computation(),
        /*label=*/filename, module->config().debug_options(), format);
    if (rendered_graph.ok()) {
      return std::move(rendered_graph).ValueOrDie();
    }
    return absl::StrFormat("Error rendering graph: %s",
                           rendered_graph.status().ToString());
  };
  printf("After opotimization:\n %f\n", m->ToString().c_str()) << std::endl;
  DumpToFileInDirImpl(splitter_dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));

  HloComputation* entry = module->entry_computation();

  EXPECT_TRUE(graph_needs_split(entry->root_instruction()));

  TensorSplitter optim;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&optim, module.get()));
  EXPECT_TRUE(result);

  EXPECT_FALSE(graph_needs_split(entry->root_instruction()));

  printf("After opotimization:\n %f\n", module->ToString().c_str());
  filename = TestName() + "_after_opotimization";
  printf("After opotimization:\n %f\n", m->ToString().c_str()) << std::endl;
  DumpToFileInDirImpl(splitter_dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));
}

// Test nested reduce
TEST_F(TensorSplitterTest, NestedReduce) {
  const string module_str = R"(
HloModule a_inference_test_simple_dist_matrix_40__XlaMustCompile_true_config_proto___n_007_n_0...02_001_000__executor_type____.34

%Sum-reduction.22 (x.23: f32[], y.24: f32[]) -> f32[] {
  %x.23 = f32[] parameter(0)
  %y.24 = f32[] parameter(1)
  ROOT %add.25 = f32[] add(f32[] %x.23, f32[] %y.24)
}

ENTRY %a_inference_test_simple_dist_matrix_40__XlaMustCompile_true_config_proto___n_007_n_0...02_001_000__executor_type____.34 (arg0.1: f32[2000,3], arg1.2: f32[2000,3], arg2.3: f32[2000,2]) -> f32[2000,2] {
  %arg0.1 = f32[2000,3]{1,0} parameter(0), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.4 = f32[2000,3]{1,0} reshape(f32[2000,3]{1,0} %arg0.1)
  %slice.7 = f32[2000,3]{1,0} slice(f32[2000,3]{1,0} %reshape.4), slice={[0:2000], [0:3]}, metadata={op_type="StridedSlice" op_name="strided_slice" source_file="xla_playground.py" source_line=142}
  %reshape.8 = f32[1,2000,3]{2,1,0} reshape(f32[2000,3]{1,0} %slice.7), metadata={op_type="StridedSlice" op_name="strided_slice" source_file="xla_playground.py" source_line=142}
  %reshape.11 = f32[2000,3]{1,0} reshape(f32[1,2000,3]{2,1,0} %reshape.8), metadata={op_type="Sub" op_name="sub" source_file="xla_playground.py" source_line=142}
  %broadcast.12 = f32[2000,2000,3]{2,1,0} broadcast(f32[2000,3]{1,0} %reshape.11), dimensions={1,2}, metadata={op_type="Sub" op_name="sub" source_file="xla_playground.py" source_line=142}
  %arg1.2 = f32[2000,3]{1,0} parameter(1), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.5 = f32[2000,3]{1,0} reshape(f32[2000,3]{1,0} %arg1.2)
  %slice.9 = f32[2000,3]{1,0} slice(f32[2000,3]{1,0} %reshape.5), slice={[0:2000], [0:3]}, metadata={op_type="StridedSlice" op_name="strided_slice_1" source_file="xla_playground.py" source_line=142}
  %reshape.10 = f32[2000,1,3]{2,1,0} reshape(f32[2000,3]{1,0} %slice.9), metadata={op_type="StridedSlice" op_name="strided_slice_1" source_file="xla_playground.py" source_line=142}
  %reshape.13 = f32[2000,3]{1,0} reshape(f32[2000,1,3]{2,1,0} %reshape.10), metadata={op_type="Sub" op_name="sub" source_file="xla_playground.py" source_line=142}
  %broadcast.14 = f32[2000,2000,3]{2,1,0} broadcast(f32[2000,3]{1,0} %reshape.13), dimensions={0,2}, metadata={op_type="Sub" op_name="sub" source_file="xla_playground.py" source_line=142}
  %subtract.15 = f32[2000,2000,3]{2,1,0} subtract(f32[2000,2000,3]{2,1,0} %broadcast.12, f32[2000,2000,3]{2,1,0} %broadcast.14), metadata={op_type="Sub" op_name="sub" source_file="xla_playground.py" source_line=142}
  %constant.16 = f32[] constant(2), metadata={op_type="Pow" op_name="pow" source_file="xla_playground.py" source_line=143}
  %broadcast.17 = f32[2000,2000,3]{2,1,0} broadcast(f32[] %constant.16), dimensions={}, metadata={op_type="Pow" op_name="pow" source_file="xla_playground.py" source_line=143}
  %power.18 = f32[2000,2000,3]{2,1,0} power(f32[2000,2000,3]{2,1,0} %subtract.15, f32[2000,2000,3]{2,1,0} %broadcast.17), metadata={op_type="Pow" op_name="pow" source_file="xla_playground.py" source_line=143}
  %convert.19 = f32[2000,2000,3]{2,1,0} convert(f32[2000,2000,3]{2,1,0} %power.18), metadata={op_type="Sum" op_name="Sum" source_file="xla_playground.py" source_line=143}
  %constant.20 = f32[] constant(0), metadata={op_type="Sum" op_name="Sum" source_file="xla_playground.py" source_line=143}
  %convert.21 = f32[] convert(f32[] %constant.20), metadata={op_type="Sum" op_name="Sum" source_file="xla_playground.py" source_line=143}
  %reduce.26 = f32[2000,2000]{1,0} reduce(f32[2000,2000,3]{2,1,0} %convert.19, f32[] %convert.21), dimensions={2}, to_apply=%Sum-reduction.22, metadata={op_type="Sum" op_name="Sum" source_file="xla_playground.py" source_line=143}
  %convert.27 = f32[2000,2000]{1,0} convert(f32[2000,2000]{1,0} %reduce.26), metadata={op_type="Sum" op_name="Sum" source_file="xla_playground.py" source_line=143}
  %exponential.28 = f32[2000,2000]{1,0} exponential(f32[2000,2000]{1,0} %convert.27), metadata={op_type="Exp" op_name="Exp" source_file="xla_playground.py" source_line=144}
  %arg2.3 = f32[2000,2]{1,0} parameter(2), parameter_replication={false}, metadata={op_name="XLA_Args"}
  %reshape.6 = f32[2000,2]{1,0} reshape(f32[2000,2]{1,0} %arg2.3)
  %dot.29 = f32[2000,2]{1,0} dot(f32[2000,2000]{1,0} %exponential.28, f32[2000,2]{1,0} %reshape.6), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="matmul" source_file="xla_playground.py" source_line=144}
  %transpose.30 = f32[2000,2]{1,0} transpose(f32[2000,2]{1,0} %dot.29), dimensions={0,1}, metadata={op_type="MatMul" op_name="matmul" source_file="xla_playground.py" source_line=144}
  %reshape.31 = f32[2000,2]{1,0} reshape(f32[2000,2]{1,0} %transpose.30), metadata={op_name="XLA_Retvals"}
  %tuple.32 = (f32[2000,2]{1,0}) tuple(f32[2000,2]{1,0} %reshape.31), metadata={op_name="XLA_Retvals"}
  ROOT %get-tuple-element.33 = f32[2000,2]{1,0} get-tuple-element((f32[2000,2]{1,0}) %tuple.32), index=0, metadata={op_name="XLA_Retvals"}
}
)";

  string module_with_big_dims =
      replace_all_in_string(module_str, "2000", std::to_string(large_dim()));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_with_big_dims));

  std::string filename = TestName() + "_before_opotimization";
  // DebugOptions debug_options
  auto render_graph = [&](RenderedGraphFormat format) {
    StatusOr<string> rendered_graph = RenderGraph(
        *module->entry_computation(),
        /*label=*/filename, module->config().debug_options(), format);
    if (rendered_graph.ok()) {
      return std::move(rendered_graph).ValueOrDie();
    }
    return absl::StrFormat("Error rendering graph: %s",
                           rendered_graph.status().ToString());
  };
  printf("After opotimization:\n %f\n", m->ToString().c_str()) << std::endl;
  DumpToFileInDirImpl(splitter_dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));

  HloComputation* entry = module->entry_computation();

  EXPECT_TRUE(graph_needs_split(entry->root_instruction()));

  TensorSplitter optim;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&optim, module.get()));
  EXPECT_TRUE(result);

  EXPECT_FALSE(graph_needs_split(entry->root_instruction()));

  printf("After opotimization:\n %f\n", module->ToString().c_str());
  filename = TestName() + "_after_opotimization";
  printf("After opotimization:\n %f\n", m->ToString().c_str()) << std::endl;
  DumpToFileInDirImpl(splitter_dir, absl::StrFormat("%s.dot", filename),
                      render_graph(RenderedGraphFormat::kDot));
}

}  // namespace
}  // namespace xla
