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
#include <dlfcn.h>
#include <unistd.h>

#include "tensorflow/compiler/plugin/poplar/driver/compiler.h"

#include "tensorflow/compiler/plugin/poplar/driver/allocation_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/executable.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitor_full.h"
#include "tensorflow/compiler/plugin/poplar/driver/outliner.h"
#include "tensorflow/compiler/plugin/poplar/driver/fuse_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/platform_id.h"

#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
#include "tensorflow/compiler/xla/service/batchnorm_rewriter.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/service/flatten_call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_constant_folding.h"
#include "tensorflow/compiler/xla/service/hlo_cse.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/hlo_subcomputation_unification.h"
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
#include <poprand/codelets.hpp>

namespace se = ::perftools::gputools;
namespace sep = ::perftools::gputools::poplarplugin;

namespace xla {
namespace poplarplugin {

static std::string GetPathToGraphProgFile() {
  Dl_info dlInfo;
  static const void* dummy;
  if (dladdr(&dummy, &dlInfo)) {
    std::string path(dlInfo.dli_fname);
    path = path.substr(0, path.find_last_of( '/' ) + 1);
    path = path + "../compiler/plugin/poplar/tf.gp";
    if (access(path.c_str(), R_OK) != -1) {
      return path;
    }
  }

  // This is for unit tests
  {
    char buf[256];
    getcwd(buf, 255);
    std::string path(buf);
    path = path + "/tensorflow/compiler/plugin/poplar/tf.gp";
    if (access(path.c_str(), R_OK) != -1) {
      return path;
    }
  }

  return "";
}

class EntryVisitor : public FullVisitor {
public:
  EntryVisitor(poplar::Graph* graph,
                    CompilerResources& resources,
                    uint64 num_parameters)
          : FullVisitor(graph, resources),
            parameter_shapes(num_parameters),
            all_outputs_are_parameters(false) {}

  Status HandleParameter(HloInstruction* inst) {

    parameter_shapes[inst->parameter_number()] = inst->shape();

    std::vector<Shape> shapes = FlattenedXlaShape(inst->shape());

    for (unsigned i=0; i<shapes.size(); i++) {
      poplar::Tensor out;
      TF_ASSIGN_OR_RETURN(out,
                          AddTensor(*graph_, inst, shapes[i], resources_));
      TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, i, out));

      graph_->createHostWrite(
              sep::GetInputCopyHandle(inst->parameter_number(), i), out);
    }
    return Status::OK();
  }

  Status FinishVisit(HloInstruction* inst) {
    const HloComputation* comp = inst->parent();

    auto outputs = FindInstructionOutputs(tensor_map, inst);

    for (size_t o=0; o<outputs.size(); o++) {

      // For each output, if there is an identical input, put it into the map
      for (int64 i=0; i<comp->num_parameters(); i++) {
        HloInstruction* param = comp->parameter_instruction(i);
        auto in = FindInstructionOutputs(tensor_map, param);

        // Only non-tuple inputs are considered for input<->output mapping
        if (in.size() == 1 && in[0] == outputs[o]) {
          output_map[o] = i;
        }
      }

      graph_->createHostRead(sep::GetOutputCopyHandle(o), outputs[o]);
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

  sep::OutputMap output_map;
  std::vector<Shape> parameter_shapes;

  bool all_outputs_are_parameters;
};

class CallTargetFinder : public DfsHloVisitorWithDefault {
public:
  CallTargetFinder(HloComputation* entry) {
    todo.insert(entry);
  }

  Status DefaultAction(HloInstruction*) override { return Status::OK(); }

  Status HandleCall(HloInstruction* inst) override {
    CallSiteFound(inst->to_apply(), 1);
    return Status::OK();
  }

  Status HandleWhile(HloInstruction* inst) override {
    CallSiteFound(inst->while_condition(), 2);
    CallSiteFound(inst->while_body(), 2);
    return Status::OK();
  }

  Status Run() {
    while (!todo.empty()) {
      auto it = todo.begin();
      HloComputation *comp = *it;
      todo.erase(it);
      done.insert(comp);
      TF_RETURN_IF_ERROR(comp->Accept(this));
    }
    return Status::OK();
  }

  std::map<HloComputation*,int> targets;

private:
  void CallSiteFound(HloComputation* comp, int count) {
    if (done.find(comp) == done.end()) {
      todo.insert(comp);
    }
    targets[comp] += count;
  }

  std::set<HloComputation*> todo;
  std::set<HloComputation*> done;
};

Status PoplarCompiler::RunHloOptimization(HloModule* hlo_module) {
  HloPassPipeline pipeline("IPU");
  pipeline.AddPass<BatchNormRewriter>(true, true, false);
  pipeline.AddPass<HloCSE>(false);
  pipeline.AddPass<HloPassFix<AlgebraicSimplifier>>(false,
          [](const Shape&, const Shape&) { return false; });
  pipeline.AddPass<ReshapeMover>();
  pipeline.AddPass<HloConstantFolding>();
  pipeline.AddPass<HloCSE>(true);
  pipeline.AddPass<Outliner>();
  pipeline.AddPass<HloSubcomputationUnification>();
  pipeline.AddPass<FuseOps>();
  pipeline.AddPass<HloDCE>();
  return pipeline.Run(hlo_module).status();
}

StatusOr<std::unique_ptr<Executable>> PoplarCompiler::Compile(
    std::unique_ptr<HloModule> hlo_module,
    se::StreamExecutor* stream_exec) {

  VLOG(1) << "Begin compilation of module " << hlo_module->name();

  TF_RETURN_IF_ERROR(RunHloOptimization(hlo_module.get()));

  bool use_ipu_model = (getenv("TF_POPLAR_COMPILE_IPU_MODEL") != NULL);

  poplar::DeviceInfo dev_info;
  dev_info.IPUExchangeType =
      poplar::DeviceInfo::ExchangeType::AGGRESSIVE_MULTICAST;

  poplar::DeviceOptions dev_opts;
  dev_opts.convertFloat16 = true;

  poplar::Device dev(use_ipu_model ?
                     poplar::createIPUModelDevice(dev_info, dev_opts) :
                     poplar::createCPUDevice(dev_opts));

  poplar::Graph* graph = new poplar::Graph(dev);
  graph->addCodelets(GetPathToGraphProgFile());
  popconv::addCodelets(*graph);
  poplin::addCodelets(*graph);
  popnn::addCodelets(*graph);
  popreduce::addCodelets(*graph);
  popstd::addCodelets(*graph);
  poprand::addCodelets(*graph);

  CompilerResources resources(hlo_module->config().seed());

  HloComputation* entry = hlo_module->entry_computation();

  VLOG(2) << "Running poplar call site finder";

  // Find all Call instructions
  CallTargetFinder call_finder(entry);
  TF_RETURN_IF_ERROR(call_finder.Run());

  VLOG(2) << "Running tensor allocation tracker";

  AllocationFinder finder;
  TF_RETURN_IF_ERROR(finder.CreateAllocationMap(hlo_module.get()));
  resources.tensor_allocation_map = std::move(finder.tensor_allocation_map);

  for (const auto comp : hlo_module->MakeComputationPostOrder()) {
    if (call_finder.targets.count(comp) > 0) {
      if (comp != entry && call_finder.targets.at(comp) > 1) {
        // If this computation is a target of a call or while then compile
        // it and store in compiler resources
        VLOG(1) << "Compiling sub-computation " << comp->name();
        resources.computation_map.emplace(
                std::piecewise_construct,
                std::forward_as_tuple(comp),
                std::forward_as_tuple(graph, resources, comp->num_parameters()));
        TF_RETURN_IF_ERROR(comp->Accept(&(resources.computation_map.at(comp))));
      }
    }
  }

  VLOG(1) << "Compiling main computation " << entry->name();
  XLA_VLOG_LINES(1, entry->ToString());

  EntryVisitor visitor(graph, resources, entry->num_parameters());
  try {
    TF_RETURN_IF_ERROR(entry->Accept(&visitor));
  }
  catch (std::logic_error e) {
    return port::Status(port::error::UNKNOWN,
                        port::StrCat("[Poplar Compile] ",
                                     e.what()));
  }

  std::shared_ptr<poplar::Engine> engine;
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
    graph->outputVertexGraph(stream, progs);
  }

  const char *compute_graph = getenv("TF_POPLAR_COMPUTE_GRAPH_FILENAME");
  if (compute_graph != NULL) {
    std::ofstream stream;
    stream.open(compute_graph);
    graph->outputComputeGraph(stream, progs);
  }

  hlo_module->mutable_entry_computation_layout()->SetToDefaultLayout();

  std::unique_ptr<Executable> executable;
  executable.reset(
          new PoplarExecutable(std::move(hlo_module),
                               std::move(engine),
                               std::move(visitor.output_map),
                               std::move(visitor.parameter_shapes)));

  return std::move(executable);
}

StatusOr<std::vector<std::unique_ptr<Executable>>> PoplarCompiler::Compile(
    std::vector<std::unique_ptr<HloModule>> hlo_modules,
    std::vector<se::StreamExecutor*> stream_execs) {

  return tensorflow::errors::Unimplemented(
          "Compilation of multiple HLO modules is not supported on Poplar.");
}

StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
PoplarCompiler::CompileAheadOfTime(
    std::vector<std::unique_ptr<HloModule>> hlo_modules,
    const AotCompilationOptions& aot_options) {

  return tensorflow::errors::InvalidArgument(
      "AOT compilation not supported on Poplar");
}

se::Platform::Id PoplarCompiler::PlatformId() const {
  return sep::kPoplarPlatformId;
}

HloCostAnalysis::ShapeSizeFunction
PoplarCompiler::ShapeSizeBytesFunction() const {
  return PoplarExecutable::ShapeSizeBytes;
}

}  // namespace poplarplugin
}  // namespace xla

static std::unique_ptr<xla::ComputationPlacer> CreateComputationPlacer() {
  return xla::MakeUnique<xla::ComputationPlacer>();
}

static bool RegisterComputationPlacer() {
  xla::ComputationPlacer::RegisterComputationPlacer(
          se::poplarplugin::kPoplarPlatformId,
          &CreateComputationPlacer);
  return true;
}

bool placer_registration = RegisterComputationPlacer();

REGISTER_MODULE_INITIALIZER(poplar_compiler, {
  xla::Compiler::RegisterCompilerFactory(sep::kPoplarPlatformId, []() {
    return xla::MakeUnique<xla::poplarplugin::PoplarCompiler>();
  });
});
