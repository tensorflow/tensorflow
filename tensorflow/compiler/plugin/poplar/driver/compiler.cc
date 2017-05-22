/* Copyright 2017 Graphcore Ltd
 */

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

#include <stdlib.h>
#include <fstream>

#include "tensorflow/compiler/plugin/poplar/driver/compiler.h"
#include "tensorflow/compiler/plugin/poplar/driver/executable.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitor_full.h"
#include "tensorflow/compiler/plugin/poplar/driver/outliner.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/stream_executor/executor.h"

#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
#include "tensorflow/compiler/xla/service/flatten_call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_constant_folding.h"
#include "tensorflow/compiler/xla/service/hlo_cse.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/hlo_subcomputation_unification.h"
#include "tensorflow/compiler/xla/service/inliner.h"
#include "tensorflow/compiler/xla/service/reshape_mover.h"
#include "tensorflow/compiler/xla/status_macros.h"

#include "tensorflow/stream_executor/lib/strcat.h"
#include "tensorflow/stream_executor/lib/initialize.h"

#include "tensorflow/core/lib/core/errors.h"

#include <poplar/exceptions.hpp>
#include <popstd/exceptions.hpp>
#include <xgraph_core/exceptions.hpp>

#include <popconv/codelets.hpp>
#include <poplin/codelets.hpp>
#include <popnn/codelets.hpp>
#include <popreduce/codelets.hpp>
#include <popstd/codelets.hpp>

namespace se = ::perftools::gputools;
namespace sep = ::perftools::gputools::poplarplugin;

namespace xla {
namespace poplarplugin {

class EntryVisitor : public FullVisitor {
public:
  EntryVisitor(poplar::Graph* graph,
                    CompilerResources& resources,
                    uint64 num_parameters)
          : FullVisitor(graph, resources),
            all_outputs_are_parameters(false) {}

  Status HandleParameter(HloInstruction* inst) {
    poplar::Tensor out;
    TF_ASSIGN_OR_RETURN(out, AddTensor(*graph_, inst->name(), inst->shape()));
    TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));

    graph_->createHostWrite(sep::GetCopyHandle(inst->parameter_number()), out);
    return Status::OK();
  }

  Status FinishVisit(HloInstruction* inst) {
    size_t num;
    if (ShapeUtil::IsTuple(inst->shape())) {
      num = ShapeUtil::TupleElementCount(inst->shape());
    } else {
      num = 1;
    }

    for (int64 i=0; i<num; i++) {
      poplar::Tensor out;
      TF_ASSIGN_OR_RETURN(out, FindInstructionOutput(tensor_map, inst, i));
      graph_->createHostRead(sep::GetCopyHandle(i), out);
    }

    if (inst->opcode() == HloOpcode::kParameter) {
      all_outputs_are_parameters = true;
    } else if (inst->opcode() == HloOpcode::kTuple){
      all_outputs_are_parameters = true;
      for (auto op : inst->operands()) {
        all_outputs_are_parameters &= (op->opcode() == HloOpcode::kParameter);
      }
    }

    tensor_map.clear();

    return Status::OK();
  }

  std::map<int64,int64> output_map;
  bool all_outputs_are_parameters;


};

class CallTargetFinder : public DfsHloVisitorWithDefault {
public:
  CallTargetFinder() {}

  Status DefaultAction(HloInstruction*) override { return Status::OK(); }

  Status HandleCall(HloInstruction* call) override {
    targets.insert(call->to_apply());
    return Status::OK();
  }

  std::set<HloComputation*> targets;
};

Status PoplarCompiler::RunHloOptimization(HloModule* hlo_module,
                                          HloDumper dump_hlo) {
  HloPassPipeline pipeline("IPU", dump_hlo);
  pipeline.AddPass<Inliner>();
  pipeline.AddPass<Outliner>(2);
  pipeline.AddPass<HloSubcomputationUnification>();
  pipeline.AddPass<HloCSE>(false);

  pipeline.AddPass<HloPassFix<AlgebraicSimplifier>>(false,
          [](const Shape&, const Shape&) { return false; });
  pipeline.AddPass<ReshapeMover>();
  pipeline.AddPass<HloConstantFolding>();
  pipeline.AddPass<HloCSE>(true);

  pipeline.AddPass<HloDCE>();
  return pipeline.Run(hlo_module).status();
}

StatusOr<std::unique_ptr<Executable>> PoplarCompiler::Compile(
    std::unique_ptr<HloModule> hlo_module, HloDumper dump_hlo,
    se::StreamExecutor* stream_exec) {
  TF_RET_CHECK(stream_exec != nullptr);

  VLOG(1) << "Generate graph " << hlo_module->name();

  TF_RETURN_IF_ERROR(
          RunHloOptimization(hlo_module.get(), dump_hlo));

  sep::PoplarExecutor* poplarExecutor(
          static_cast<sep::PoplarExecutor*>(stream_exec->implementation()));

  bool use_ipu_model = (getenv("TF_POPLAR_COMPILE_IPU_MODEL") != NULL);

  poplar::DeviceInfo dev_info;
  poplar::DeviceOptions dev_opts;
  dev_opts.convertFloat16 = true;

  poplar::Device dev(use_ipu_model ?
                     poplar::createIPUModelDevice(dev_info, dev_opts) :
                     poplar::createCPUDevice(dev_opts));

  poplar::Graph* graph = new poplar::Graph(dev);
  graph->addCodelets(poplarExecutor->GetPathToGraphProgFile());
  popconv::addCodelets(*graph);
  poplin::addCodelets(*graph);
  popnn::addCodelets(*graph);
  popreduce::addCodelets(*graph);
  popstd::addCodelets(*graph);

  CompilerResources resources;

  HloComputation* entry = hlo_module->entry_computation();

  // Find all Call instructions
  CallTargetFinder call_finder;
  TF_RETURN_IF_ERROR(entry->Accept(&call_finder));

  // Find subgraphs that will not be inlined and construct poplar equivalents
  for (const auto& comp : call_finder.targets) {
    if (comp != entry) {
      // If this computation is a target of a 'call' then compile
      // it and store in compiler resources
      VLOG(2) << "Compiling sub-graph " << comp->name();
    }
  }

  // Visit the graph, building up a poplar equivalent
  EntryVisitor visitor(graph, resources, entry->num_parameters());
  try {
    TF_RETURN_IF_ERROR(entry->Accept(&visitor));
  }
  catch (std::logic_error e) {
    return port::Status(port::error::UNKNOWN,
                        port::StrCat("[Poplar Compile] ",
                                     e.what()));
  }

  std::unique_ptr<poplar::Engine> engine;
  std::vector<poplar::program::Program> progs;
  progs.push_back(visitor.sequence);

  if (visitor.all_outputs_are_parameters) {
    VLOG(1) << "Skip engine compilation - all outputs are inputs";
  } else {
    try {
      VLOG(1) << "Compile engine " << hlo_module->name();

      engine.reset(new poplar::Engine(*graph, progs));
    }
    catch (std::logic_error e) {
      return port::Status(port::error::UNKNOWN,
                          port::StrCat("[Poplar Engine] ",
                                       e.what()));
    }
  }

  const char *vertex_graph = getenv("TF_POPLAR_VERTEX_GRAPH_FILENAME");
  if (vertex_graph != NULL) {
    std::ofstream stream;
    stream.open(vertex_graph);
    engine->outputVertexGraph(stream, *graph);
  }

  const char *compute_graph = getenv("TF_POPLAR_COMPUTE_GRAPH_FILENAME");
  if (compute_graph != NULL) {
    std::ofstream stream;
    stream.open(compute_graph);
    engine->outputComputeGraph(stream, *graph);
  }

  std::unique_ptr<Executable> executable;
  executable.reset(
          new PoplarExecutable(std::move(hlo_module),
                               std::move(engine),
                               std::move(visitor.output_map)));

  return std::move(executable);
}

StatusOr<std::vector<std::unique_ptr<Executable>>> PoplarCompiler::Compile(
    std::vector<std::unique_ptr<HloModule>> hlo_modules,
    HloDumper dump_hlos, std::vector<se::StreamExecutor*> stream_execs) {

  return tensorflow::errors::Unimplemented(
          "Compilation of multiple HLO modules is not supported on Poplar.");
}

StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
PoplarCompiler::CompileAheadOfTime(
    std::vector<std::unique_ptr<HloModule>> hlo_modules,
    HloDumper dump_hlo, const AotCompilationOptions& aot_options) {

  return tensorflow::errors::InvalidArgument(
      "AOT compilation not supported on Poplar");
}

int64 PoplarCompiler::ShapeSizeBytes(const Shape& shape) const {
  return ShapeUtil::ByteSizeOf(shape, sizeof(void*));
}

se::Platform::Id PoplarCompiler::PlatformId() const {
  return sep::kPoplarPlatformId;
}

}  // namespace poplarplugin
}  // namespace xla

REGISTER_MODULE_INITIALIZER(poplar_compiler, {
  xla::Compiler::RegisterCompilerFactory(sep::kPoplarPlatformId, []() {
    return xla::MakeUnique<xla::poplarplugin::PoplarCompiler>();
  });
});
