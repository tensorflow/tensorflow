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
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/executable.h"
#include "tensorflow/compiler/plugin/poplar/driver/executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/fuse_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/outliner.h"
#include "tensorflow/compiler/plugin/poplar/driver/platform_id.h"
#include "tensorflow/compiler/plugin/poplar/driver/scheduler.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/util.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitor_full.h"
#include "tensorflow/compiler/plugin/poplar/driver/wide_const_finder.h"

#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
#include "tensorflow/compiler/xla/service/batchnorm_expander.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/service/dot_decomposer.h"
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
#include <poplar/IPUModel.hpp>

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
    VLOG(1) << "Processing " << inst->name();

    parameter_shapes[inst->parameter_number()] = inst->shape();

    std::vector<Shape> shapes = FlattenedXlaShape(inst->shape());

    HloModule* module = inst->parent()->parent();
    ComputationLayout* layout = module->mutable_entry_computation_layout();
    const Shape& mod_shape = layout->parameter_shape(inst->parameter_number());
    std::vector<Shape> module_shapes = FlattenedXlaShape(mod_shape);

    poplar::DataTransferOptions opt;
    opt.convertHalf = true;

    for (unsigned i=0; i<shapes.size(); i++) {
      poplar::Tensor out;
      TF_ASSIGN_OR_RETURN(out,
                          AddTensor(*graph_, std::make_pair(inst,i), shapes[i],
                                    resources_));
      TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, i, out));

      if (!LayoutUtil::IsMonotonicWithDim0Major(module_shapes[i].layout())) {
        // Host tensor needs to be host layout
        out = ConvertFromDeviceLayout(module_shapes[i], out);
        non_standard_parameter_layout.insert(inst);
      }

      graph_->createHostWrite(
              sep::GetInputCopyHandle(inst->parameter_number(), i), out, opt);
    }
    return Status::OK();
  }

  Status FinishVisit(HloInstruction* inst) {
    HloComputation* comp = inst->parent();

    poplar::DataTransferOptions opt;
    opt.convertHalf = true;

    auto outputs = FindInstructionOutputs(tensor_map, inst);

    auto* layout = comp->parent()->mutable_entry_computation_layout();
    std::vector<Shape> shapes = FlattenedXlaShape(layout->result_shape());

    for (size_t o=0; o<outputs.size(); o++) {

      // For each output, if there is an identical input, put it into the map
      for (int64 i=0; i<comp->num_parameters(); i++) {
        HloInstruction* param = comp->parameter_instruction(i);
        if (non_standard_parameter_layout.count(inst) == 0) {
          auto in = FindInstructionOutputs(tensor_map, param);

          // Only non-tuple inputs are considered for input<->output mapping
          if (in.size() == 1 && in[0] == outputs[o]) {
            output_map[o] = i;
          }
        }
      }

      poplar::Tensor out = ConvertFromDeviceLayout(shapes[o], outputs[o]);
      graph_->createHostRead(sep::GetOutputCopyHandle(o), out, opt);
    }

    if (inst->opcode() == HloOpcode::kParameter) {
      all_outputs_are_parameters = true;
    } else if (inst->opcode() == HloOpcode::kTuple){
      all_outputs_are_parameters = true;
      for (auto op : inst->operands()) {
        all_outputs_are_parameters &= (op->opcode() == HloOpcode::kParameter);
      }
    }

    all_outputs_are_parameters &= (non_standard_parameter_layout.size() == 0);

    tensor_map.clear();

    return Status::OK();
  }

  sep::OutputMap output_map;
  std::vector<Shape> parameter_shapes;

  bool all_outputs_are_parameters;
  std::set<HloInstruction*> non_standard_parameter_layout;
};

static void DumpGraph(const HloComputation* comp) {
  DebugOptions debug_opts;
  debug_opts.set_xla_hlo_dump_as_graphdef(true);
  debug_opts.set_xla_hlo_graph_path("/tmp");

  hlo_graph_dumper::DumpGraph(*comp, "poplar", debug_opts, NULL, false);
}

StatusOr<std::unique_ptr<HloModule>> PoplarCompiler::RunHloPasses(
        std::unique_ptr<HloModule> module,
        perftools::gputools::StreamExecutor* executor,
        DeviceMemoryAllocator* device_allocator) {
  VLOG(1) << "Begin HloPasses: " << module->name();

  HloPassPipeline pipeline("IPU");
  pipeline.AddPass<BatchNormExpander>(true, true, true, false);
  pipeline.AddPass<DotDecomposer>();
  pipeline.AddPass<HloCSE>(false);
  pipeline.AddPass<HloPassFix<AlgebraicSimplifier>>(false,
          [](const Shape&, const Shape&) { return false; });
  pipeline.AddPass<ReshapeMover>();
  pipeline.AddPass<Inliner>();
  pipeline.AddPass<HloPassFix<AlgebraicSimplifier>>(false,
          [](const Shape&, const Shape&) { return false; });
  pipeline.AddPass<HloConstantFolding>();
  pipeline.AddPass<HloCSE>(true);
  pipeline.AddPass<WideConstFinder>();
  pipeline.AddPass<FuseOps>();
  pipeline.AddPass<Outliner>();
  pipeline.AddPass<HloSubcomputationUnification>();
  pipeline.AddPass<HloDCE>();

  bool ok;
  TF_ASSIGN_OR_RETURN(ok, pipeline.Run(module.get()));

  if (!ok) {
    VLOG(1) << "HLO module optimization returned false";
  }

  return std::move(module);
}

StatusOr<std::unique_ptr<Executable>> PoplarCompiler::RunBackend(
        std::unique_ptr<HloModule> module,
        perftools::gputools::StreamExecutor* stream_exec,
        DeviceMemoryAllocator* device_allocator) {

  VLOG(1) << "Begin compilation: " << module->name();

  bool use_ipu_model = (getenv("TF_POPLAR_COMPILE_IPU_MODEL") != NULL);

  poplar::IPUModel model;
  model.IPUExchangeType = poplar::IPUModel::ExchangeType::AGGRESSIVE_MULTICAST;

  poplar::Device dev(use_ipu_model ?
                     model.createDevice() :
                     poplar::Device::createCPUDevice());

  poplar::Graph* graph = new poplar::Graph(dev);
  graph->addCodelets(GetPathToGraphProgFile());
  popconv::addCodelets(*graph);
  poplin::addCodelets(*graph);
  popnn::addCodelets(*graph);
  popreduce::addCodelets(*graph);
  popstd::addCodelets(*graph);
  poprand::addCodelets(*graph);

  CompilerResources resources(module->config().seed() + 1);

  HloComputation* entry = module->entry_computation();

  if (getenv("TF_POPLAR_DUMP_HLO") != NULL) {
    DumpGraph(entry);
  }

  {
    AllocationFinder finder;
    TF_RETURN_IF_ERROR(finder.CreateAllocationMap(module.get()));
    resources.tensor_allocation_map = std::move(finder.tensor_allocation_map);
  }

  {
    InplaceFinder finder;
    TF_RETURN_IF_ERROR(finder.FindInplaceInstructions(module.get()));
    resources.inplace_instructions = std::move(finder.inplace_instructions);
  }

  // Set layout if there isn't one
  auto comp_layout = module->mutable_entry_computation_layout()
          ->mutable_result_layout();
  if (!comp_layout->LayoutIsSet()) {
    auto shape = entry->root_instruction()->shape();
    TF_CHECK_OK(comp_layout->CopyLayoutFromShape(shape));
  }

  VLOG(1) << "Compiling main computation " << entry->name();
  XLA_VLOG_LINES(1, entry->ToString());

  std::vector<const HloInstruction*> instruction_order;
  TF_ASSIGN_OR_RETURN(instruction_order, Scheduler::schedule(entry));

  EntryVisitor visitor(graph, resources, entry->num_parameters());
  try {
    TF_RETURN_IF_ERROR(entry->AcceptOrdered(&visitor, instruction_order));
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
      VLOG(1) << "Compile engine " << module->name();

      engine.reset(new poplar::Engine(dev, *graph, progs));
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

  std::unique_ptr<HloProfileIndexMap> profile_index_map;
  std::unique_ptr<HloProfilePrinterData> profile_printer;
  if (module->config().hlo_profiling_enabled()) {
    HloCostAnalysis cost_analysis(ShapeSizeBytesFunction());
    profile_index_map = MakeUnique<HloProfileIndexMap>(*module);
    profile_printer =
            CreateHloProfilePrinterData(*profile_index_map, cost_analysis);
  }

  std::unique_ptr<Executable> executable;
  executable.reset(
          new PoplarExecutable(std::move(module),
                               std::move(profile_printer),
                               std::move(profile_index_map),
                               std::move(engine),
                               std::move(visitor.output_map),
                               std::move(visitor.parameter_shapes)));

  return std::move(executable);
}

StatusOr<std::vector<std::unique_ptr<Executable>>> PoplarCompiler::Compile(
    std::vector<std::unique_ptr<HloModule>> hlo_modules,
    std::vector<std::vector<perftools::gputools::StreamExecutor*>> execs,
    DeviceMemoryAllocator* device_allocator) {

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

static bool InitModule() {
  xla::Compiler::RegisterCompilerFactory(se::poplarplugin::kPoplarPlatformId, []() {
    return xla::MakeUnique<xla::poplarplugin::PoplarCompiler>();
  });
  return true;
}
static bool module_initialized = InitModule();
