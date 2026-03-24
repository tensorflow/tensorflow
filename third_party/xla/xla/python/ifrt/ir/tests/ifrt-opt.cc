/* Copyright 2023 The OpenXLA Authors.

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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include "absl/base/thread_annotations.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "xla/mlir_hlo/mhlo/IR/register.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/hlo/hlo_program.h"
#include "xla/python/ifrt/ir/atom_program_compiler.h"
#include "xla/python/ifrt/ir/conversions/mpmd/lower_to_ifrt.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_ir_program.h"
#include "xla/python/ifrt/ir/transforms/passes.h"
#include "xla/python/ifrt/mock.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/support/module_parsing.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/init_main.h"

namespace xla {
namespace ifrt {
namespace {

static constexpr int kMaxTestMethods = 1000;

class TestChildExecutableCompiler : public AtomProgramCompiler {
 public:
  TestChildExecutableCompiler() { methods_.reserve(kMaxTestMethods); }

  tsl::Future<LoadedExecutableRef> CompileXla(
      std::unique_ptr<HloProgram> hlo_program,
      xla::CompileOptions options) override ABSL_LOCKS_EXCLUDED(mu_) {
    absl::MutexLock lock(mu_);
    methods_.push_back(absl::StrCat("fake_method_", methods_.size()));
    CHECK_LT(methods_.size(), kMaxTestMethods)
        << "push_back() might have caused reallocation, which might have "
           "invalidated some method string_views.";
    auto mock_executable =
        std::make_unique<testing::NiceMock<MockLoadedExecutable>>();
    int num_devices;
    if (options.executable_build_options.has_device_assignment()) {
      num_devices =
          options.executable_build_options.device_assignment().num_elements();
    } else {
      num_devices = 1;
    }
    int num_parameters_to_propagate =
        options.executable_build_options
            .allow_spmd_sharding_propagation_to_parameters()
            .size();
    if (num_devices > 1 && num_parameters_to_propagate > 0) {
      xla::OpSharding op_sharding;
      op_sharding.set_type(xla::OpSharding::REPLICATED);
      std::vector<xla::OpSharding> parameter_shardings(
          num_parameters_to_propagate, op_sharding);
      ON_CALL(*mock_executable, GetParameterShardings())
          .WillByDefault(testing::Return(std::move(parameter_shardings)));
    }
    int num_outputs_to_propagate =
        options.executable_build_options
            .allow_spmd_sharding_propagation_to_output()
            .size();
    if (num_devices > 1 && num_outputs_to_propagate > 0) {
      // Always infer output shardings to be replicated for the lit tests.
      xla::OpSharding op_sharding;
      op_sharding.set_type(xla::OpSharding::REPLICATED);
      std::vector<xla::OpSharding> output_shardings(num_outputs_to_propagate,
                                                    op_sharding);
      ON_CALL(*mock_executable, GetOutputShardings())
          .WillByDefault(testing::Return(std::move(output_shardings)));
    }
    return mock_executable;
  }

  tsl::Future<LoadedExecutableRef> CompileMpmdReshard(
      std::vector<DType> dtypes, std::vector<Shape> shapes,
      std::vector<IfrtArrayType> in_array_types,
      std::vector<IfrtArrayType> out_array_types) override
      ABSL_LOCKS_EXCLUDED(mu_) {
    absl::MutexLock lock(mu_);
    methods_.push_back(absl::StrCat("fake_method_", methods_.size()));
    CHECK_LT(methods_.size(), kMaxTestMethods)
        << "push_back() might have caused reallocation, which might have "
           "invalidated some method string_views.";
    return std::make_unique<testing::NiceMock<MockLoadedExecutable>>();
  }

 private:
  absl::Mutex mu_;
  std::vector<std::string> methods_ ABSL_GUARDED_BY(mu_);
};

}  // namespace
}  // namespace ifrt
}  // namespace xla

int main(int argc, char** argv) {
  // Allow passing ABSL flags to ifrt-opt.
  int absl_flags_end = 1;
  bool has_absl_flags = false;
  for (int i = 1; i < argc; ++i) {
    if (absl::string_view(argv[i]) == "--") {
      absl_flags_end = i;
      argc -= (i + 1);
      has_absl_flags = true;
      break;
    }
  }
  tsl::port::InitMain((argc >= 1 ? argv[0] : ""), &absl_flags_end, &argv);
  if (has_absl_flags) {
    argc += absl_flags_end;
    argv[1] = argv[0];
    ++argv;
  }

  std::shared_ptr<xla::ifrt::AtomProgramCompiler> compiler =
      std::make_shared<xla::ifrt::TestChildExecutableCompiler>();
  std::shared_ptr<xla::ifrt::IfrtIRCompileOptions> compile_options =
      std::make_shared<xla::ifrt::IfrtIRCompileOptions>();
  compile_options->dot_graph_dump_to = "sponge";
  auto atom_executable_map =
      std::make_shared<xla::ifrt::AtomExecutableFutureMap>();
  auto bound_executable_map = std::make_shared<xla::ifrt::AtomExecutableMap>();

  mlir::registerAllPasses();
  xla::ifrt::registerIfrtPassesAndPipelines(
      compiler, compile_options, atom_executable_map, bound_executable_map);
  xla::ifrt::mpmd::RegisterLowerToIfrtPasses();
  mlir::DialectRegistry registry;
  xla::ifrt::support::InitializeMlirDialectRegistry(registry);
  // Register dialects that are only used in the MLIR lit tests.
  registry.insert<mlir::math::MathDialect, mlir::sdy::SdyDialect,
                  mlir::mpmd::MpmdDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "IFRT IR dialect driver\n", registry));
}
