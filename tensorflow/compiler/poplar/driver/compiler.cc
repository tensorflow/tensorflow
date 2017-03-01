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

#include "tensorflow/compiler/poplar/driver/compiler.h"
#include "tensorflow/compiler/poplar/driver/executable.h"
#include "tensorflow/compiler/poplar/driver/ops.h"
#include "tensorflow/compiler/poplar/driver/tensor.h"
#include "tensorflow/compiler/poplar/driver/visitor_base.h"
#include "tensorflow/compiler/poplar/stream_executor/executor.h"
//
#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
#include "tensorflow/compiler/xla/service/hlo_cse.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/inliner.h"
#include "tensorflow/compiler/xla/service/hlo_subcomputation_unification.h"
#include "tensorflow/compiler/xla/status_macros.h"

#include "tensorflow/stream_executor/poplar/poplar_platform_id.h"
#include "tensorflow/stream_executor/lib/strcat.h"
#include "tensorflow/stream_executor/lib/initialize.h"

#include "tensorflow/core/lib/core/errors.h"

#include <poplar/exceptions.hpp>
#include <popnn/exceptions.hpp>

namespace se = ::perftools::gputools;
namespace sep = ::perftools::gputools::poplarplugin;

namespace xla {
namespace poplarplugin {

class PoplarMainVisitor : public PoplarBaseVisitor {
public:
  PoplarMainVisitor(poplar::Graph* graph, uint64 num_parameters)
          : PoplarBaseVisitor(graph) {
    input_copy_buffers.resize(num_parameters);
  }

  Status HandleInfeed(HloInstruction* inst) {
    LOG(INFO) << inst->ToString();
    return port::Status(port::error::UNIMPLEMENTED,
                        port::StrCat(inst->name(),
                                     " not implemented"));
  }

  Status HandleOutfeed(HloInstruction* inst) {
    LOG(INFO) << inst->ToString();
    return port::Status(port::error::UNIMPLEMENTED,
                        port::StrCat(inst->name(),
                                     " not implemented"));
  }

  Status HandleParameter(HloInstruction* inst) {
    LOG(INFO) << inst->ToString() << " " << inst->parameter_number();
    // Allocate the output tensor
    poplar::Tensor out;
    TF_ASSIGN_OR_RETURN(out, AddTensor(*graph_, inst->name(), inst->shape()));
    TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));

    // TODO (remove when possible) allocate temporary copy buffer
    char* input_buffer = new char[ShapeUtil::ByteSizeOf(inst->shape())];
    input_copy_buffers[inst->parameter_number()] = input_buffer;
    copy_in.add(poplar::program::Copy(out, input_buffer));
    return Status::OK();
  }

  Status HandleSend(HloInstruction* inst) {
    LOG(INFO) << inst->ToString();
    return port::Status(port::error::UNIMPLEMENTED,
                        port::StrCat(inst->name(),
                                     " not implemented"));
  }

  Status HandleRecv(HloInstruction* inst) {
    LOG(INFO) << inst->ToString();
    return port::Status(port::error::UNIMPLEMENTED,
                        port::StrCat(inst->name(),
                                     " not implemented"));
  }

// Invoked to inform the visitor that the traversal has completed, and that
// the root was "root".
  Status FinishVisit(HloInstruction* inst) {
    if (ShapeUtil::IsTuple(inst->shape())) {
      output_copy_buffers.resize(ShapeUtil::TupleElementCount(inst->shape()));

      for (size_t i=0; i<output_copy_buffers.size(); i++) {
        poplar::Tensor out;
        TF_ASSIGN_OR_RETURN(out, FindInstructionOutput(tensor_map, inst, i));

        const Shape& shape(ShapeUtil::GetTupleElementShape(inst->shape(), i));

        // TODO (remove when possible) allocate temporary copy buffer
        char* output_buffer = new char[ShapeUtil::ByteSizeOf(shape)];
        output_copy_buffers[i] = output_buffer;
        copy_out.add(poplar::program::Copy(output_buffer, out));
      }
    } else {
      output_copy_buffers.resize(1);

      poplar::Tensor out;
      TF_ASSIGN_OR_RETURN(out, FindInstructionOutput(tensor_map, inst, 0));

      // TODO (remove when possible) allocate temporary copy buffer
      char* output_buffer = new char[ShapeUtil::ByteSizeOf(inst->shape())];
      output_copy_buffers[0] = output_buffer;
      copy_out.add(poplar::program::Copy(output_buffer, out));
    }

    return Status::OK();
  }

  // TODO these are an artifact of the current Engine Copy programs
  // TODO remove them once there is a Copy interface that doesn't
  // TODO require specifying the buffers up front
  poplar::program::Sequence copy_in;
  poplar::program::Sequence copy_out;
  std::vector<char*> input_copy_buffers;
  std::vector<char*> output_copy_buffers;
};

Status PoplarCompiler::RunHloOptimization(HloModule* hlo_module,
                                          HloModuleConfig* module_config,
                                          HloDumper dump_hlo) {
  HloPassPipeline pipeline("IPU", dump_hlo);
  pipeline.AddPass<Inliner>();
  pipeline.AddPass<HloSubcomputationUnification>();
  pipeline.AddPass<HloCSE>(false);

  pipeline.AddPass<HloPassFix<AlgebraicSimplifier>>(false,
          [](const Shape&, const Shape&) { return false; });
  pipeline.AddPass<HloCSE>(true);

  pipeline.AddPass<HloDCE>();
  return pipeline.Run(hlo_module).status();
}

StatusOr<std::unique_ptr<Executable>> PoplarCompiler::Compile(
    std::unique_ptr<HloModule> hlo_module,
    std::unique_ptr<HloModuleConfig> module_config, HloDumper dump_hlo,
    se::StreamExecutor* stream_exec) {
  TF_RET_CHECK(stream_exec != nullptr);

  LOG(INFO) << "Compiling " << hlo_module->name();

  TF_RETURN_IF_ERROR(
          RunHloOptimization(hlo_module.get(), module_config.get(), dump_hlo));

  sep::PoplarExecutor* poplarExecutor(
          static_cast<sep::PoplarExecutor*>(stream_exec->implementation()));

  poplar::GraphProgEnv* env(poplarExecutor->GetPoplarGraphProgEnv());

  poplar::DeviceOptions opts;
  opts.convertFloat16 = true;
  poplar::Device dev(poplar::createCPUDevice(opts));

  poplar::Graph* graph = new poplar::Graph(*env, dev);

  // Visit the graph, building up a poplar equivalent
  HloComputation* entry = hlo_module->entry_computation();
  PoplarMainVisitor visitor(graph, entry->num_parameters());
  try {
    TF_RETURN_IF_ERROR(entry->Accept(&visitor));
  }
  catch (poplar::poplar_error e) {
    return port::Status(port::error::UNKNOWN,
                        port::StrCat("[Poplar Compile] ",
                                     e.what()));
  }
  catch (popnn::popnn_error e) {
    return port::Status(port::error::UNKNOWN,
                        port::StrCat("[Popnn Compile] ",
                                     e.what()));
  }

  std::unique_ptr<poplar::Engine> engine;
  std::vector<poplar::program::Program> progs;
  progs.push_back(visitor.sequence);
  progs.push_back(visitor.copy_in);
  progs.push_back(visitor.copy_out);

  try {
    engine.reset(new poplar::Engine(*graph, progs));
  }
  catch (poplar::poplar_error e) {
    return port::Status(port::error::UNKNOWN,
                        port::StrCat("[Poplar Engine] ",
                                     e.what()));
  }

  std::unique_ptr<Executable> executable;
  executable.reset(
          new PoplarExecutable(std::move(hlo_module),
                               std::move(module_config),
                               std::move(engine),
                               visitor.input_copy_buffers,
                               visitor.output_copy_buffers));

  return std::move(executable);
}

StatusOr<std::vector<std::unique_ptr<Executable>>> PoplarCompiler::Compile(
    std::vector<std::unique_ptr<HloModule>> hlo_modules,
    std::vector<std::unique_ptr<HloModuleConfig>> module_configs,
    HloDumper dump_hlos, std::vector<se::StreamExecutor*> stream_execs) {

  return tensorflow::errors::Unimplemented(
          "Compilation of multiple HLO modules is not yet supported on Poplar.");
}

StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
PoplarCompiler::CompileAheadOfTime(
    std::vector<std::unique_ptr<HloModule>> hlo_modules,
    std::vector<std::unique_ptr<HloModuleConfig>> module_configs,
    HloDumper dump_hlo, const AotCompilationOptions& aot_options) {
  TF_RET_CHECK(hlo_modules.size() == module_configs.size());

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
