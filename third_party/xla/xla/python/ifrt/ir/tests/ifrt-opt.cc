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
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "xla/mlir_hlo/mhlo/IR/register.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/hlo/hlo_program.h"
#include "xla/python/ifrt/ir/atom_program_compiler.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/transforms/passes.h"
#include "xla/python/ifrt/mock.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/support/module_parsing.h"

namespace xla {
namespace ifrt {
namespace {

static constexpr int kMaxTestMethods = 1000;

class TestChildExecutableCompiler : public AtomProgramCompiler {
 public:
  TestChildExecutableCompiler() { methods_.reserve(kMaxTestMethods); }

  absl::StatusOr<AtomProgramCompileResult> CompileXla(
      std::unique_ptr<HloProgram> hlo_program,
      xla::CompileOptions options) override ABSL_LOCKS_EXCLUDED(mu_) {
    absl::MutexLock lock(&mu_);
    methods_.push_back(absl::StrCat("fake_method_", methods_.size()));
    CHECK_LT(methods_.size(), kMaxTestMethods)
        << "push_back() might have caused reallocation, which might have "
           "invalidated some method string_views.";
    auto mock_executable =
        std::make_unique<testing::NiceMock<MockLoadedExecutable>>();
    int num_parameters_to_propagate =
        options.executable_build_options
            .allow_spmd_sharding_propagation_to_parameters()
            .size();
    if (num_parameters_to_propagate > 0) {
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
    if (num_outputs_to_propagate > 0) {
      // Always infer output shardings to be replicated for the lit tests.
      xla::OpSharding op_sharding;
      op_sharding.set_type(xla::OpSharding::REPLICATED);
      std::vector<xla::OpSharding> output_shardings(num_outputs_to_propagate,
                                                    op_sharding);
      ON_CALL(*mock_executable, GetOutputShardings())
          .WillByDefault(testing::Return(std::move(output_shardings)));
    }
    return AtomProgramCompileResult{
        /*name=*/absl::StrCat("fake_component__", methods_.back()),
        /*executable=*/std::move(mock_executable)};
  }

  absl::StatusOr<AtomProgramCompileResult> CompileMpmdReshard(
      std::vector<DType> dtypes, std::vector<Shape> shapes,
      std::vector<IfrtArrayType> in_array_types,
      std::vector<IfrtArrayType> out_array_types) override
      ABSL_LOCKS_EXCLUDED(mu_) {
    absl::MutexLock lock(&mu_);
    methods_.push_back(absl::StrCat("fake_method_", methods_.size()));
    CHECK_LT(methods_.size(), kMaxTestMethods)
        << "push_back() might have caused reallocation, which might have "
           "invalidated some method string_views.";
    auto mock_executable =
        std::make_unique<testing::NiceMock<MockLoadedExecutable>>();
    return AtomProgramCompileResult{
        /*name=*/absl::StrCat("fake_mpmd_reshard_component__", methods_.back()),
        /*executable=*/std::make_unique<MockLoadedExecutable>()};
  }

 private:
  absl::Mutex mu_;
  std::vector<std::string> methods_ ABSL_GUARDED_BY(mu_);
};

}  // namespace
}  // namespace ifrt
}  // namespace xla

int main(int argc, char** argv) {
  std::shared_ptr<xla::ifrt::AtomProgramCompiler> compiler =
      std::make_shared<xla::ifrt::TestChildExecutableCompiler>();
  auto compile_options = std::make_shared<absl::flat_hash_map<
      std::string, std::unique_ptr<xla::ifrt::CompileOptions>>>();
  std::shared_ptr<xla::ifrt::AtomExecutableMap> atom_executable_map =
      std::make_shared<xla::ifrt::AtomExecutableMap>();
  std::shared_ptr<xla::ifrt::AtomExecutableMap> bound_executable_map =
      std::make_shared<xla::ifrt::AtomExecutableMap>();

  mlir::registerAllPasses();
  xla::ifrt::RegisterIfrtPassesAndPipelines(
      compiler, compile_options, atom_executable_map, bound_executable_map);
  mlir::DialectRegistry registry;
  xla::ifrt::support::InitializeMlirDialectRegistry(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "IFRT IR dialect driver\n", registry));
}
