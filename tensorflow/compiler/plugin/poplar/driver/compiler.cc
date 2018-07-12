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

#include <dlfcn.h>
#include <stdlib.h>
#include <unistd.h>
#include <fstream>

#include "tensorflow/compiler/plugin/poplar/driver/compiler.h"

#include "tensorflow/compiler/plugin/poplar/driver/allocation_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/computation_flattener.h"
#include "tensorflow/compiler/plugin/poplar/driver/executable.h"
#include "tensorflow/compiler/plugin/poplar/driver/executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/expression_outliner.h"
#include "tensorflow/compiler/plugin/poplar/driver/fuse_ops_early.h"
#include "tensorflow/compiler/plugin/poplar/driver/fuse_ops_late.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/outliner.h"
#include "tensorflow/compiler/plugin/poplar/driver/platform_id.h"
#include "tensorflow/compiler/plugin/poplar/driver/scheduler.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/update_op_dependencies.h"
#include "tensorflow/compiler/plugin/poplar/driver/util.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitor_full.h"
#include "tensorflow/compiler/plugin/poplar/driver/while_loop_condition_simplify.h"
#include "tensorflow/compiler/plugin/poplar/driver/wide_const_finder.h"

#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
#include "tensorflow/compiler/xla/service/batchnorm_expander.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/service/dot_decomposer.h"
#include "tensorflow/compiler/xla/service/gather_expander.h"
#include "tensorflow/compiler/xla/service/hlo_constant_folding.h"
#include "tensorflow/compiler/xla/service/hlo_cse.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/hlo_subcomputation_unification.h"
#include "tensorflow/compiler/xla/service/hlo_tfgraph_builder.h"
#include "tensorflow/compiler/xla/service/inliner.h"
#include "tensorflow/compiler/xla/service/reshape_mover.h"
#include "tensorflow/compiler/xla/service/tuple_simplifier.h"
#include "tensorflow/compiler/xla/service/zero_sized_hlo_elimination.h"
#include "tensorflow/compiler/xla/status_macros.h"

#include "tensorflow/stream_executor/lib/initialize.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/strings/strcat.h"

#include <poplar/exceptions.hpp>
#include <poputil/exceptions.hpp>

#include <popconv/codelets.hpp>
#include <popnn/codelets.hpp>
#include <popops/codelets.hpp>
#include <poprand/codelets.hpp>

namespace se = ::stream_executor;

using ::tensorflow::strings::StrCat;

namespace xla {
namespace poplarplugin {

static std::string GetPathToGraphProgFile() {
  Dl_info dlInfo;
  static const void* dummy;
  if (dladdr(&dummy, &dlInfo)) {
    std::string path(dlInfo.dli_fname);
    path = path.substr(0, path.find_last_of('/') + 1);
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

static bool OkToStream(const Shape& shape) {
  if (ShapeUtil::IsTuple(shape)) {
    return false;
  }
  if (ShapeUtil::ElementsIn(shape) == 0) {
    return false;
  }
  if (ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type()) > 4) {
    return false;
  }
  return true;
}

class EntryVisitor : public FullVisitor {
 public:
  EntryVisitor(poplar::Graph& graph, CompilerResources& resources,
               uint64 num_parameters, uint64 num_outputs)
      : FullVisitor(graph, resources),
        parameter_streamed(num_parameters),
        output_streamed(num_outputs),
        all_outputs_are_parameters(false) {}

  Status HandleParameter(HloInstruction* inst) {
    VLOG(1) << "Processing " << inst->name();

    auto num_streaming = inst->parent()->num_parameters() -
                         resources_.annotations.num_resource_variables;

    parameter_streamed[inst->parameter_number()] =
        (inst->parameter_number() < num_streaming) && OkToStream(inst->shape());

    std::vector<Shape> shapes = FlattenedXlaShape(inst->shape());
    std::vector<Shape> module_shapes;

    HloModule* module = inst->parent()->parent();
    ComputationLayout* layout = module->mutable_entry_computation_layout();
    if (layout->parameter_count() > inst->parameter_number()) {
      const Shape& mod_shape =
          layout->parameter_shape(inst->parameter_number());
      module_shapes = FlattenedXlaShape(mod_shape);
    }

    for (unsigned i = 0; i < shapes.size(); i++) {
      poplar::Tensor out;
      TF_ASSIGN_OR_RETURN(out, AddTensor(graph_, std::make_pair(inst, i),
                                         shapes[i], resources_));
      TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, i, out));

      if (module_shapes.size() > i) {
        if (!LayoutUtil::IsMonotonicWithDim0Major(module_shapes[i].layout())) {
          // Host tensor needs to be host layout
          out = ConvertFromDeviceLayout(module_shapes[i], out);
          non_standard_parameter_layout.insert(inst);
        }
      }

      if (parameter_streamed[inst->parameter_number()]) {
        auto fifo = graph_.addHostToDeviceFIFO(
            GetInputCopyHandle(inst->parameter_number(), i), out.elementType(),
            out.numElements());

        sequence.add(poplar::program::Copy(fifo, out));

      } else {
        graph_.createHostWrite(GetInputCopyHandle(inst->parameter_number(), i),
                               out);
      }
    }
    return Status::OK();
  }

  Status FinishVisit(HloInstruction* inst) {
    HloComputation* comp = inst->parent();

    auto outputs = FindInstructionOutputs(tensor_map, inst);

    auto* layout = comp->parent()->mutable_entry_computation_layout();
    std::vector<Shape> shapes = FlattenedXlaShape(layout->result_shape());

    for (size_t o = 0; o < outputs.size(); o++) {
      // For each output, if there is an identical input, put it into the map
      for (int64 i = 0; i < comp->num_parameters(); i++) {
        HloInstruction* param = comp->parameter_instruction(i);
        if (non_standard_parameter_layout.count(inst) == 0) {
          auto in = FindInstructionOutputs(tensor_map, param);

          // Only non-tuple inputs are considered for input<->output mapping
          if (in.size() == 1 && in[0] == outputs[o]) {
            output_map[o] = i;
          }
        }
      }

      if (!output_streamed[o]) {
        poplar::Tensor out = ConvertFromDeviceLayout(shapes[o], outputs[o]);
        graph_.createHostRead(GetOutputCopyHandle(o), out);
      }
    }

    if (inst->opcode() == HloOpcode::kParameter) {
      all_outputs_are_parameters = true;
    } else if (inst->opcode() == HloOpcode::kTuple) {
      all_outputs_are_parameters = true;
      for (auto op : inst->operands()) {
        all_outputs_are_parameters &= (op->opcode() == HloOpcode::kParameter);
      }
    }

    all_outputs_are_parameters &= (non_standard_parameter_layout.size() == 0);

    tensor_map.clear();

    return Status::OK();
  }

  Status Postprocess(HloInstruction* inst) {
    // After processing each instruction, check if its output can be streamed
    // off the device, and arrange for a FIFO and Copy if it can be.
    if (OkToStream(inst->shape())) {
      const auto* root = inst->parent()->root_instruction();
      auto num_streaming = FlattenedXlaShape(root->shape()).size() -
                           resources_.annotations.num_resource_variables;
      if (root->opcode() == HloOpcode::kTuple) {
        for (int i = 0; i < root->operand_count(); i++) {
          if (root->operand(i) == inst) {
            if (i < num_streaming) {
              auto pair = FindTupleInputIndices(root, i);
              const auto& outputs = FindInstructionOutputs(tensor_map, inst);
              if (pair.second - pair.first == outputs.size()) {
                for (int o = 0; o < outputs.size(); o++) {
                  int64 index = pair.first + o;
                  output_streamed[index] = true;

                  HloComputation* comp = inst->parent();
                  HloModule* mod = comp->parent();
                  auto* layout = mod->mutable_entry_computation_layout();
                  auto shapes = FlattenedXlaShape(layout->result_shape());

                  poplar::Tensor out =
                      ConvertFromDeviceLayout(shapes[index], outputs[o]);

                  auto fifo = graph_.addDeviceToHostFIFO(
                      GetOutputCopyHandle(index), out.elementType(),
                      out.numElements());

                  sequence.add(poplar::program::Copy(out, fifo));
                }
              }
            }
          }
        }
      }
    }
    return Status::OK();
  }

  OutputMap output_map;
  std::vector<bool> parameter_streamed;
  std::vector<bool> output_streamed;

  bool all_outputs_are_parameters;

 private:
  std::set<HloInstruction*> non_standard_parameter_layout;
};

static std::string SerializeComputationToGraphDef(const HloComputation& comp) {
  std::string buffer;
  hlo_graph_dumper::HloTfGraphBuilder builder;
  TF_CHECK_OK(builder.AddComputation(comp));
  builder.GetGraphDef().SerializeToString(&buffer);
  return buffer;
}

StatusOr<std::unique_ptr<HloModule>> PoplarCompiler::RunHloPasses(
    std::unique_ptr<HloModule> module,
    perftools::gputools::StreamExecutor* executor,
    DeviceMemoryAllocator* device_allocator) {
  return std::move(module);
}

StatusOr<std::unique_ptr<Executable>> PoplarCompiler::RunBackend(
    std::unique_ptr<HloModule> module,
    perftools::gputools::StreamExecutor* stream_exec,
    DeviceMemoryAllocator* device_allocator) {
  VLOG(1) << "Begin compilation: " << module->name();

  if (stream_exec == nullptr) {
    return tensorflow::errors::Unknown(
        "NULL stream pointer in poplar compiler");
  }

  PoplarExecutor* poplarExecutor(
      static_cast<PoplarExecutor*>(stream_exec->implementation()));

  std::unique_ptr<HloProfileIndexMap> profile_index_map;
  std::unique_ptr<HloProfilePrinterData> profile_printer;
  if (module->config().hlo_profiling_enabled()) {
    HloCostAnalysis cost_analysis(ShapeSizeBytesFunction());
    profile_index_map = MakeUnique<HloProfileIndexMap>(*module);
    profile_printer =
        CreateHloProfilePrinterData(*profile_index_map, cost_analysis);
  }

  std::string filename;
  if (poplarExecutor->HaveExecutableCache()) {
    filename = poplarExecutor->CachedExecutableFilename(*module);

    if (poplarExecutor->HaveCachedExecutable(filename)) {
      PoplarExecutable* poplar_executable;
      TF_ASSIGN_OR_RETURN(poplar_executable,
                          PoplarExecutable::Deserialize(
                              std::move(module), std::move(profile_printer),
                              std::move(profile_index_map), filename));

      std::unique_ptr<Executable> executable;
      executable.reset(poplar_executable);

      return std::move(executable);
    }
  }

  const poplar::Device& dev = poplarExecutor->GetPoplarDevice();

  std::lock_guard<std::mutex> g(static_mu_);

  poplar::Graph graph(dev);
  graph.addCodelets(GetPathToGraphProgFile());
  popconv::addCodelets(graph);
  popnn::addCodelets(graph);
  popops::addCodelets(graph);
  poprand::addCodelets(graph);

  uint64 start_micros = tensorflow::Env::Default()->NowMicros();

  uint64 seed = module->config().seed();
  if (seed == 0) {
    seed = tensorflow::random::New64();
  }

  CompilerResources resources(seed + 1, poplarExecutor->GetRandomGenMode());
  resources.annotations.num_resource_variables =
      module->config().resource_update_count();

  {
    HloPassPipeline pipeline("IPU");
    pipeline.AddPass<BatchNormExpander>(true, true, true);
    pipeline.AddPass<GatherExpander>();
    pipeline.AddPass<DotDecomposer>();
    pipeline.AddPass<FuseOpsEarly>(resources.annotations);
    pipeline.AddPass<HloCSE>(false);
    pipeline.AddPass<HloPassFix<AlgebraicSimplifier>>(
        false, [](const Shape&, const Shape&) { return false; }, false, false);
    pipeline.AddPass<ReshapeMover>();
    pipeline.AddPass<Inliner>();
    pipeline.AddPass<HloPassFix<AlgebraicSimplifier>>(
        false, [](const Shape&, const Shape&) { return false; }, false, false);
    pipeline.AddPass<ZeroSizedHloElimination>();
    pipeline.AddPass<ComputationFlattener>();
    pipeline.AddPass<TupleSimplifier>(true);
    // pipeline.AddPass<WhileLoopSimplifier>();
    // pass.AddPass<ConditionalSimplifier>();
    pipeline.AddPass<HloConstantFolding>();
    pipeline.AddPass<HloCSE>(true);
    pipeline.AddPass<WideConstFinder>();
    pipeline.AddPass<ConvolutionClassifier>(resources.annotations);
    pipeline.AddPass<HloDCE>();
    pipeline.AddPass<FuseOpsLate>(resources.annotations);
    pipeline.AddPass<Outliner>(resources.annotations);
    pipeline.AddPass<InplaceFinder>(resources.annotations);
    pipeline.AddPass<ExpressionOutliner>(resources.annotations);
    pipeline.AddPass<UpdateOpDependenctOrdering>(resources.annotations);
    pipeline.AddPass<HloSubcomputationUnification>();
    pipeline.AddPass<WhileLoopConditionSimplify>();
    pipeline.AddPass<HloDCE>();
    pipeline.AddPass<ConvolutionClassifier>(resources.annotations);
    pipeline.AddPass<AllocationFinder>(resources.annotations);

    bool ok;
    TF_ASSIGN_OR_RETURN(ok, pipeline.Run(module.get()));
  }

  HloComputation* entry = module->entry_computation();

  if (poplarExecutor->CompilerReportingEnabled()) {
    poplarExecutor->AddEventRecord(tensorflow::IpuTraceEvent::COMPILE_BEGIN,
                                   module->name(),
                                   SerializeComputationToGraphDef(*entry), 0);
  }

  // Set layout if there isn't one
  auto comp_layout =
      module->mutable_entry_computation_layout()->mutable_result_layout();
  if (!comp_layout->LayoutIsSet()) {
    auto shape = entry->root_instruction()->shape();
    TF_CHECK_OK(comp_layout->CopyLayoutFromShape(shape));
  }

  VLOG(1) << "Compiling main computation " << entry->name();
  XLA_VLOG_LINES(1, entry->ToString());

  std::vector<const HloInstruction*> instruction_order;
  TF_ASSIGN_OR_RETURN(instruction_order, Scheduler::schedule(entry));

  uint64 num_inputs = entry->num_parameters();
  uint64 num_outputs = CountShapes(entry->root_instruction()->shape());

  EntryVisitor visitor(graph, resources, num_inputs, num_outputs);
  try {
    TF_RETURN_IF_ERROR(entry->AcceptOrdered(&visitor, instruction_order));
  } catch (std::logic_error e) {
    return tensorflow::errors::Unknown(StrCat("[Poplar Compile] ", e.what()));
  }

  std::shared_ptr<poplar::Engine> engine;
  std::vector<poplar::program::Program> progs;
  progs.push_back(visitor.sequence);

  if (visitor.all_outputs_are_parameters) {
    VLOG(1) << "Skip engine compilation - all outputs are inputs";
  } else {
    try {
      VLOG(1) << "Compile engine " << module->name();

      auto opts = poplarExecutor->GetOptionsFlags();
      engine.reset(new poplar::Engine(graph, progs, opts));
    } catch (std::logic_error e) {
      return tensorflow::errors::Unknown(StrCat("[Poplar Engine] ", e.what()));
    }
  }

  if (poplarExecutor->CompilerReportingEnabled()) {
    std::stringstream stream;

    if (engine != nullptr) {
      poplar::OptionFlags opts;
      opts.set("includeVarStorageReport", "true");

      auto rep = engine->getGraphReport(opts);
      if (poplarExecutor->CompilerReportingTextFormat()) {
        rep.printSummary(stream);
      } else {
        rep.serialize(stream, poplar::SerializationFormat::JSON);
      }
    }

    uint64 duration = tensorflow::Env::Default()->NowMicros() - start_micros;

    poplarExecutor->AddEventRecord(tensorflow::IpuTraceEvent::COMPILE_END,
                                   module->name(), stream.str(), duration);
  }

  std::unique_ptr<Executable> executable;
  PoplarExecutable* poplar_executable;
  poplar_executable = new PoplarExecutable(
      std::move(module), std::move(profile_printer),
      std::move(profile_index_map), std::move(engine),
      std::move(visitor.output_map), std::move(visitor.parameter_streamed),
      std::move(visitor.output_streamed));
  executable.reset(poplar_executable);

  if (poplarExecutor->HaveExecutableCache()) {
    if (!poplarExecutor->HaveCachedExecutable(filename)) {
      TF_RETURN_IF_ERROR(
          PoplarExecutable::Serialize(*poplar_executable, filename));
    }
  }

  return std::move(executable);
}

StatusOr<std::vector<std::unique_ptr<Executable>>> PoplarCompiler::Compile(
    std::vector<std::unique_ptr<HloModule>> modules,
    std::vector<std::vector<perftools::gputools::StreamExecutor*>> stream_execs,
    DeviceMemoryAllocator* device_allocator) {
  std::vector<std::unique_ptr<Executable>> result;
  for (size_t i = 0; i < modules.size(); i++) {
    if (stream_execs[i].size() != 1) {
      return Unimplemented("Model partitioning not implemented for Poplar");
    }

    TF_ASSIGN_OR_RETURN(modules[i],
                        RunHloPasses(std::move(modules[i]), stream_execs[i][0],
                                     device_allocator));

    TF_ASSIGN_OR_RETURN(std::unique_ptr<Executable> executable,
                        RunBackend(std::move(modules[i]), stream_execs[i][0],
                                   device_allocator));

    result.push_back(std::move(executable));
  }

  return {std::move(result)};
}

StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
PoplarCompiler::CompileAheadOfTime(
    std::vector<std::unique_ptr<HloModule>> hlo_modules,
    const AotCompilationOptions& aot_options) {
  return xla::InvalidArgument("AOT compilation not supported on Poplar");
}

se::Platform::Id PoplarCompiler::PlatformId() const {
  return kPoplarPlatformId;
}

HloCostAnalysis::ShapeSizeFunction PoplarCompiler::ShapeSizeBytesFunction()
    const {
  return PoplarExecutable::ShapeSizeBytes;
}

std::mutex PoplarCompiler::static_mu_;

}  // namespace poplarplugin
}  // namespace xla

static std::unique_ptr<xla::ComputationPlacer> CreateComputationPlacer() {
  return xla::MakeUnique<xla::ComputationPlacer>();
}

static bool RegisterComputationPlacer() {
  xla::ComputationPlacer::RegisterComputationPlacer(
      xla::poplarplugin::kPoplarPlatformId, &CreateComputationPlacer);
  return true;
}

bool placer_registration = RegisterComputationPlacer();

static bool InitModule() {
  xla::Compiler::RegisterCompilerFactory(
      xla::poplarplugin::kPoplarPlatformId,
      []() { return xla::MakeUnique<xla::poplarplugin::PoplarCompiler>(); });
  return true;
}
static bool module_initialized = InitModule();
