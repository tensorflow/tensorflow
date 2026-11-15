/* Copyright 2024 The OpenXLA Authors.

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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "stablehlo/dialect/Version.h"
#include "xla/python/ifrt/ir/ifrt_ir_program.h"
#include "xla/python/ifrt/ir/ifrt_ir_program.pb.h"
#include "xla/python/ifrt/ir/support/module_parsing.h"
#include "xla/python/ifrt/ir/version.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt/serdes.pb.h"
#include "xla/python/ifrt/serdes_test_util.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {
namespace {

using ::testing::AllOf;
using ::testing::HasSubstr;
using ::testing::Not;

std::string PrintModule(mlir::ModuleOp module) {
  std::string module_str;
  llvm::raw_string_ostream os(module_str);
  module->print(os, mlir::OpPrintingFlags().enableDebugInfo());
  return module_str;
}

struct VersionedTestProgram {
  Serialized serialized;
  IfrtIrProgramProto program_proto;
};

// Serializes a simple single-atom versioned `IfrtIRProgram` and parses its
// `IfrtIrProgramProto` payload so tests can mutate individual fields.
absl::StatusOr<VersionedTestProgram> SerializeVersionedTestProgram() {
  static constexpr absl::string_view kMlirModuleStr = R"mlir(
!array = !ifrt.array<tensor<2xi32>, #ifrt.sharding_param<1 to [0] on 1>, [0]>
module {
  func.func @main(%arg0: !array) -> !array attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @add_one::@main(%arg0) on devices [0]
        : (!array) -> !array
    return %0 : !array
  }

  module @add_one <sym_visibility = "private"> {
    func.func @main(%arg0: tensor<2xi32>) -> tensor<2xi32> {
      %0 = stablehlo.constant dense<1> : tensor<2xi32>
      %1 = stablehlo.add %arg0, %0 : tensor<2xi32>
      return %1 : tensor<2xi32>
    }
  }
}
  )mlir";

  auto context = std::make_unique<mlir::MLIRContext>();
  ABSL_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> module,
                   support::ParseMlirModuleString(kMlirModuleStr, *context));
  IfrtIRProgram program(std::move(context), std::move(module));
  ABSL_ASSIGN_OR_RETURN(
      Serialized serialized,
      Serialize(
          program,
          std::make_unique<SerializeIfrtIRProgramOptions>(
              /*ifrt_version=*/Version::getCurrentVersion().toString(),
              /*atom_program_version=*/
              mlir::vhlo::Version::getCurrentVersion().toString(),
              /*atom_program_sdy_version=*/
              mlir::sdy::SdyDialectVersion::getCurrentVersion().toString(),
              /*version_in_place=*/false)));

  IfrtIrProgramProto program_proto;
  if (!program_proto.ParseFromString(serialized.data())) {
    return absl::InternalError("Failed to parse IfrtIrProgramProto");
  }
  return VersionedTestProgram{std::move(serialized), std::move(program_proto)};
}

// Returns a versioned serialized `IfrtIRProgram` whose outer `ifrt_program`
// bytecode has been replaced with the result of calling `replace` on it.
absl::StatusOr<Serialized> SerializeWithIfrtProgram(
    absl::FunctionRef<std::string(absl::string_view)> replace) {
  ABSL_ASSIGN_OR_RETURN(VersionedTestProgram test_program,
                   SerializeVersionedTestProgram());
  const std::string bytecode(test_program.program_proto.ifrt_program());
  test_program.program_proto.set_ifrt_program(replace(bytecode));
  test_program.serialized.set_data(
      test_program.program_proto.SerializeAsCord());
  return test_program.serialized;
}

// Returns a versioned serialized `IfrtIRProgram` whose only atom program, which
// is serialized as MLIR bytecode, has been replaced with the result of calling
// `replace` on it.
absl::StatusOr<Serialized> SerializeWithAtomProgram(
    absl::FunctionRef<std::string(absl::string_view)> replace) {
  ABSL_ASSIGN_OR_RETURN(VersionedTestProgram test_program,
                   SerializeVersionedTestProgram());
  if (test_program.program_proto.atom_programs_size() != 1) {
    return absl::InternalError(
        absl::StrCat("Expected 1 atom program, got ",
                     test_program.program_proto.atom_programs_size()));
  }
  IfrtIrAtomProgramProto& atom_program =
      *test_program.program_proto.mutable_atom_programs(0);
  const std::string bytecode(atom_program.program());
  atom_program.set_program(replace(bytecode));
  test_program.serialized.set_data(
      test_program.program_proto.SerializeAsCord());
  return test_program.serialized;
}

class IfrtIRProgramSerDesTest : public testing::TestWithParam<SerDesVersion> {
 public:
  IfrtIRProgramSerDesTest() : version_(GetParam()) {}

  SerDesVersion version() const { return version_; }

 private:
  SerDesVersion version_;
};

TEST_P(IfrtIRProgramSerDesTest, RoundTrip) {
  static constexpr absl::string_view kMlirModuleStr = R"mlir(
!array = !ifrt.array<tensor<2xi32>, #ifrt.sharding_param<1 to [0] on 1>, [0]>
module {
  func.func @main(%arg0: !array) -> !array attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @add_one::@main(%arg0) on devices [0]
        : (!array) -> !array
    return %0 : !array
  }

  module @add_one {
    func.func @main(%arg0: tensor<2xi32>) -> tensor<2xi32> {
      %0 = mhlo.constant dense<1> : tensor<2xi32>
      %1 = mhlo.add %arg0, %0 : tensor<2xi32>
      return %1 : tensor<2xi32>
    }
  }
}
  )mlir";

  Serialized serialized;
  auto context = std::make_unique<mlir::MLIRContext>();
  TF_ASSERT_OK_AND_ASSIGN(
      mlir::OwningOpRef<mlir::ModuleOp> module,
      support::ParseMlirModuleString(kMlirModuleStr, *context));
  auto initial_program =
      std::make_unique<IfrtIRProgram>(std::move(context), std::move(module));

  // TODO(hyeontaek): Use `version()` to fill in
  // `SerializeIfrtIRProgramOptions::ifrt_version`.
  TF_ASSERT_OK_AND_ASSIGN(serialized,
                          Serialize(*initial_program, /*options=*/nullptr));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IfrtIRProgram> deserialized_program,
      Deserialize<IfrtIRProgram>(serialized, /*options=*/nullptr));

  EXPECT_EQ(PrintModule(initial_program->mlir_module),
            PrintModule(deserialized_program->mlir_module));
}

TEST_P(IfrtIRProgramSerDesTest, RoundTripWithUnreduced) {
  static constexpr absl::string_view kMlirModuleStr = R"mlir(
!array = !ifrt.array<tensor<2xi32>, #ifrt.sharding_param<1 to [0] on 1 unreduced [0]>, [0]>
module {
  func.func @main(%arg0: !array) -> !array attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @add_one::@main(%arg0) on devices [0]
        : (!array) -> !array
    return %0 : !array
  }

  module @add_one {
    func.func @main(%arg0: tensor<2xi32>) -> tensor<2xi32> {
      %0 = mhlo.constant dense<1> : tensor<2xi32>
      %1 = mhlo.add %arg0, %0 : tensor<2xi32>
      return %1 : tensor<2xi32>
    }
  }
}
  )mlir";

  Serialized serialized;
  auto context = std::make_unique<mlir::MLIRContext>();
  TF_ASSERT_OK_AND_ASSIGN(
      mlir::OwningOpRef<mlir::ModuleOp> module,
      support::ParseMlirModuleString(kMlirModuleStr, *context));
  auto initial_program =
      std::make_unique<IfrtIRProgram>(std::move(context), std::move(module));

  // TODO(hyeontaek): Use `version()` to fill in
  // `SerializeIfrtIRProgramOptions::ifrt_version`.
  TF_ASSERT_OK_AND_ASSIGN(serialized,
                          Serialize(*initial_program, /*options=*/nullptr));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IfrtIRProgram> deserialized_program,
      Deserialize<IfrtIRProgram>(serialized, /*options=*/nullptr));

  EXPECT_EQ(PrintModule(initial_program->mlir_module),
            PrintModule(deserialized_program->mlir_module));
}

TEST_P(IfrtIRProgramSerDesTest, VersioningRoundTrip) {
  static constexpr absl::string_view kMlirModuleStr = R"mlir(
!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
module @multiple_calls_of_same_module {
  func.func @main(%arg0: !array) -> !array attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @add_one::@main(%arg0) on devices [0,1]
        : (!array) -> !array
    %1, %ctrl_1 = ifrt.Call @add_one::@main(%0) on devices [0,1]
        : (!array) -> !array
    %2, %ctrl_2 = ifrt.Call @add_one::@main(%1) after %ctrl_1 on devices [0,1]
        : (!array) -> !array
    return %2 : !array
  }

  module @add_one <sym_visibility = "private"> {
    func.func @main(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
      %0 = stablehlo.constant dense<1> : tensor<2x2xi32>
      %1 = stablehlo.add %arg0, %0 : tensor<2x2xi32>
      return %1 : tensor<2x2xi32>
    }
  }
}
  )mlir";

  Serialized serialized;
  auto context = std::make_unique<mlir::MLIRContext>();
  TF_ASSERT_OK_AND_ASSIGN(
      mlir::OwningOpRef<mlir::ModuleOp> module,
      support::ParseMlirModuleString(kMlirModuleStr, *context));
  auto initial_program =
      std::make_unique<IfrtIRProgram>(std::move(context), std::move(module));

  // TODO(hyeontaek): Use `version()` to fill in
  // `SerializeIfrtIRProgramOptions::ifrt_version`.
  auto options = std::make_unique<SerializeIfrtIRProgramOptions>(
      Version::getCurrentVersion().toString(),
      /*vhlo_target_version=*/
      mlir::vhlo::Version::getCurrentVersion().toString(),
      /*atom_program_sdy_version=*/
      mlir::sdy::SdyDialectVersion::getCurrentVersion().toString(),
      /*version_in_place=*/false);
  TF_ASSERT_OK_AND_ASSIGN(serialized,
                          Serialize(*initial_program, std::move(options)));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IfrtIRProgram> deserialized_program,
      Deserialize<IfrtIRProgram>(serialized, /*options=*/nullptr));

  EXPECT_EQ(PrintModule(initial_program->mlir_module),
            PrintModule(deserialized_program->mlir_module));
}

TEST_P(IfrtIRProgramSerDesTest, VersioningRoundTripWithUnreduced) {
  static constexpr absl::string_view kMlirModuleStr = R"mlir(
!array = !ifrt.array<tensor<2xi32>,
                     #ifrt.sharding_param<1 to [0,1] on 1x2 unreduced [1]>, [0,1]>
module @multiple_calls_of_same_module {
  func.func @main(%arg0: !array) -> !array attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @add_one::@main(%arg0) on devices [0,1]
        : (!array) -> !array
    return %0 : !array
  }

  module @add_one <sym_visibility = "private"> {
    func.func @main(%arg0: tensor<2xi32>) -> tensor<2xi32> {
      %0 = stablehlo.constant dense<1> : tensor<2xi32>
      %1 = stablehlo.add %arg0, %0 : tensor<2xi32>
      return %1 : tensor<2xi32>
    }
  }
}
  )mlir";

  Serialized serialized;
  auto context = std::make_unique<mlir::MLIRContext>();
  TF_ASSERT_OK_AND_ASSIGN(
      mlir::OwningOpRef<mlir::ModuleOp> module,
      support::ParseMlirModuleString(kMlirModuleStr, *context));
  auto initial_program =
      std::make_unique<IfrtIRProgram>(std::move(context), std::move(module));

  // TODO(hyeontaek): Use `version()` to fill in
  // `SerializeIfrtIRProgramOptions::ifrt_version`.
  auto options = std::make_unique<SerializeIfrtIRProgramOptions>(
      Version::getCurrentVersion().toString(),
      /*vhlo_target_version=*/
      mlir::vhlo::Version::getCurrentVersion().toString(),
      /*atom_program_sdy_version=*/
      mlir::sdy::SdyDialectVersion::getCurrentVersion().toString(),
      /*version_in_place=*/false);
  TF_ASSERT_OK_AND_ASSIGN(serialized,
                          Serialize(*initial_program, std::move(options)));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IfrtIRProgram> deserialized_program,
      Deserialize<IfrtIRProgram>(serialized, /*options=*/nullptr));

  EXPECT_EQ(PrintModule(initial_program->mlir_module),
            PrintModule(deserialized_program->mlir_module));
}

TEST_P(IfrtIRProgramSerDesTest, RoundTripWithExternalContext) {
  static constexpr absl::string_view kMlirModuleStr = R"mlir(
!array = !ifrt.array<tensor<2xi32>, #ifrt.sharding_param<1 to [0] on 1>, [0]>
module {
  func.func @main(%arg0: !array) -> !array attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @add_one::@main(%arg0) on devices [0]
        : (!array) -> !array
    return %0 : !array
  }

  module @add_one {
    func.func @main(%arg0: tensor<2xi32>) -> tensor<2xi32> {
      %0 = mhlo.constant dense<1> : tensor<2xi32>
      %1 = mhlo.add %arg0, %0 : tensor<2xi32>
      return %1 : tensor<2xi32>
    }
  }
}
  )mlir";

  auto context = std::make_unique<mlir::MLIRContext>();
  ASSERT_OK_AND_ASSIGN(
      mlir::OwningOpRef<mlir::ModuleOp> module,
      support::ParseMlirModuleString(kMlirModuleStr, *context));
  auto initial_program =
      std::make_unique<IfrtIRProgram>(std::move(context), std::move(module));

  ASSERT_OK_AND_ASSIGN(Serialized serialized,
                       Serialize(*initial_program, /*options=*/nullptr));

  mlir::MLIRContext external_context;
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IfrtIRProgram> deserialized_program,
      Deserialize<IfrtIRProgram>(
          serialized, std::make_unique<DeserializeIfrtIRProgramOptions>(
                          &external_context)));

  EXPECT_FALSE(deserialized_program->OwnsMlirContext());
  EXPECT_EQ(deserialized_program->mlir_module.getContext(), &external_context);
  EXPECT_EQ(PrintModule(initial_program->mlir_module),
            PrintModule(deserialized_program->mlir_module));
}

TEST_P(IfrtIRProgramSerDesTest, VersioningRoundTripWithExternalContext) {
  static constexpr absl::string_view kMlirModuleStr = R"mlir(
!array = !ifrt.array<tensor<2x2xi32>,
                     #ifrt.sharding_param<2x1 to [0] on 2>, [0,1]>
module @multiple_calls_of_same_module {
  func.func @main(%arg0: !array) -> !array attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @add_one::@main(%arg0) on devices [0,1]
        : (!array) -> !array
    return %0 : !array
  }

  module @add_one <sym_visibility = "private"> {
    func.func @main(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
      %0 = stablehlo.constant dense<1> : tensor<2x2xi32>
      %1 = stablehlo.add %arg0, %0 : tensor<2x2xi32>
      return %1 : tensor<2x2xi32>
    }
  }
}
  )mlir";

  auto context = std::make_unique<mlir::MLIRContext>();
  ASSERT_OK_AND_ASSIGN(
      mlir::OwningOpRef<mlir::ModuleOp> module,
      support::ParseMlirModuleString(kMlirModuleStr, *context));
  auto initial_program =
      std::make_unique<IfrtIRProgram>(std::move(context), std::move(module));

  auto options = std::make_unique<SerializeIfrtIRProgramOptions>(
      /*ifrt_version=*/Version::getCurrentVersion().toString(),
      /*atom_program_version=*/
      mlir::vhlo::Version::getCurrentVersion().toString(),
      /*atom_program_sdy_version=*/
      mlir::sdy::SdyDialectVersion::getCurrentVersion().toString(),
      /*version_in_place=*/false);
  ASSERT_OK_AND_ASSIGN(Serialized serialized,
                       Serialize(*initial_program, std::move(options)));

  mlir::MLIRContext external_context;
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IfrtIRProgram> deserialized_program,
      Deserialize<IfrtIRProgram>(
          serialized, std::make_unique<DeserializeIfrtIRProgramOptions>(
                          &external_context)));

  EXPECT_FALSE(deserialized_program->OwnsMlirContext());
  EXPECT_EQ(deserialized_program->mlir_module.getContext(), &external_context);
  EXPECT_EQ(PrintModule(initial_program->mlir_module),
            PrintModule(deserialized_program->mlir_module));
}

TEST_P(IfrtIRProgramSerDesTest, VersioningOfAtomProgramInMhloShouldFail) {
  static constexpr absl::string_view kMlirModuleStr = R"mlir(
!array = !ifrt.array<tensor<2xi32>, #ifrt.sharding_param<1 to [0] on 1>, [0]>
module {
  func.func @main(%arg0: !array) -> !array attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @add_one::@main(%arg0) on devices [0]
        : (!array) -> !array
    return %0 : !array
  }

  module @add_one {
    func.func @main(%arg0: tensor<2xi32>) -> tensor<2xi32> {
      %0 = mhlo.constant dense<1> : tensor<2xi32>
      %1 = mhlo.add %arg0, %0 : tensor<2xi32>
      return %1 : tensor<2xi32>
    }
  }
}
  )mlir";

  Serialized serialized;
  auto context = std::make_unique<mlir::MLIRContext>();
  TF_ASSERT_OK_AND_ASSIGN(
      mlir::OwningOpRef<mlir::ModuleOp> module,
      support::ParseMlirModuleString(kMlirModuleStr, *context));
  auto initial_program =
      std::make_unique<IfrtIRProgram>(std::move(context), std::move(module));

  // TODO(hyeontaek): Use `version()` to fill in
  // `SerializeIfrtIRProgramOptions::ifrt_version`.
  auto options = std::make_unique<SerializeIfrtIRProgramOptions>(
      Version::getCurrentVersion().toString(),
      /*vhlo_target_version=*/
      mlir::vhlo::Version::getCurrentVersion().toString(),
      /*atom_program_sdy_version=*/
      mlir::sdy::SdyDialectVersion::getCurrentVersion().toString());
  EXPECT_THAT(
      Serialize(*initial_program, std::move(options)),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             HasSubstr("Failed to version IFRT IR program")));
}

TEST_P(IfrtIRProgramSerDesTest, DeserializationError) {
  static constexpr absl::string_view kMlirModuleStr = R"mlir(
!array = !ifrt.array<tensor<2xi32>, #ifrt.sharding_param<1 to [0] on 1>, [0]>
module {
  func.func @main(%arg0: !array) -> !array attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @add_one::@main(%arg0) on devices [0]
        : (!array) -> !array
    return %0 : !array
  }

  module @add_one {
    func.func @main(%arg0: tensor<2xi32>) -> tensor<2xi32> {
      %0 = mhlo.constant dense<1> : tensor<2xi32>
      %1 = mhlo.add %arg0, %0 : tensor<2xi32>
      return %1 : tensor<2xi32>
    }
  }
}
  )mlir";
  Serialized serialized;
  {
    auto context = std::make_unique<mlir::MLIRContext>();
    TF_ASSERT_OK_AND_ASSIGN(
        mlir::OwningOpRef<mlir::ModuleOp> module,
        support::ParseMlirModuleString(kMlirModuleStr, *context));
    auto program =
        std::make_unique<IfrtIRProgram>(std::move(context), std::move(module));
    // TODO(hyeontaek): Use `version()` to fill in
    // `SerializeIfrtIRProgramOptions::ifrt_version`.
    TF_ASSERT_OK_AND_ASSIGN(serialized,
                            Serialize(*program, /*options=*/nullptr));
  }

  serialized.set_data("invalid data");

  EXPECT_THAT(
      Deserialize<IfrtIRProgram>(serialized, /*options=*/nullptr),
      absl_testing::StatusIs(Not(absl::StatusCode::kOk),
                             HasSubstr("Failed to parse IfrtIrProgramProto")));
}

TEST_P(IfrtIRProgramSerDesTest, NonBytecodeIfrtProgramDeserializationError) {
  ASSERT_OK_AND_ASSIGN(Serialized serialized,
                       SerializeWithIfrtProgram([](absl::string_view) {
                         return std::string("module {}");
                       }));

  EXPECT_THAT(Deserialize<IfrtIRProgram>(serialized, /*options=*/nullptr),
              absl_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  AllOf(HasSubstr("Failed to parse IFRT IR module string"),
                        HasSubstr("not MLIR bytecode"))));
}

// Bytecode `IfrtIRProgram` payloads are deserialized directly from a flattened
// `absl::Cord` without copying to a null-terminated `std::string`, so these
// tests truncate the bytecode and rely on sanitizers to detect out-of-bounds
// reads.
TEST_P(IfrtIRProgramSerDesTest,
       ShortTruncatedBytecodeIfrtProgramDeserializationError) {
  // Short enough for `absl::Cord` to store inline.
  ASSERT_OK_AND_ASSIGN(Serialized serialized,
                       SerializeWithIfrtProgram([](absl::string_view bytecode) {
                         return std::string(bytecode.substr(0, 15));
                       }));

  EXPECT_THAT(Deserialize<IfrtIRProgram>(serialized, /*options=*/nullptr),
              absl_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  AllOf(HasSubstr("Failed to parse IFRT IR module string"),
                        Not(HasSubstr("not MLIR bytecode")))));
}

TEST_P(IfrtIRProgramSerDesTest,
       LongTruncatedBytecodeIfrtProgramDeserializationError) {
  // Too long for `absl::Cord` to store inline.
  ASSERT_OK_AND_ASSIGN(
      Serialized serialized,
      SerializeWithIfrtProgram([](absl::string_view bytecode) {
        return std::string(bytecode.substr(0, bytecode.size() - 1));
      }));

  EXPECT_THAT(Deserialize<IfrtIRProgram>(serialized, /*options=*/nullptr),
              absl_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  AllOf(HasSubstr("Failed to parse IFRT IR module string"),
                        Not(HasSubstr("not MLIR bytecode")))));
}

TEST_P(IfrtIRProgramSerDesTest,
       ShortNonBytecodeAtomProgramDeserializationError) {
  // Short enough for `absl::Cord` to store inline.
  ASSERT_OK_AND_ASSIGN(Serialized serialized,
                       SerializeWithAtomProgram([](absl::string_view) {
                         return std::string("module @abcdefg");
                       }));

  EXPECT_THAT(Deserialize<IfrtIRProgram>(serialized, /*options=*/nullptr),
              absl_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  AllOf(HasSubstr("Failed to deserialize atom program"),
                        HasSubstr("not MLIR bytecode"))));
}

TEST_P(IfrtIRProgramSerDesTest,
       LongNonBytecodeAtomProgramDeserializationError) {
  // Too long for `absl::Cord` to store inline.
  ASSERT_OK_AND_ASSIGN(
      Serialized serialized, SerializeWithAtomProgram([](absl::string_view) {
        return absl::StrCat("module @add_one { func.func @main() { return } } ",
                            std::string(42, ' '));
      }));

  EXPECT_THAT(Deserialize<IfrtIRProgram>(serialized, /*options=*/nullptr),
              absl_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  AllOf(HasSubstr("Failed to deserialize atom program"),
                        HasSubstr("not MLIR bytecode"))));
}

// Bytecode atom programs are deserialized directly from a flattened
// `absl::Cord` without copying to a null-terminated `std::string`, so these
// tests truncate the bytecode and rely on sanitizers to detect out-of-bounds
// reads.
TEST_P(IfrtIRProgramSerDesTest,
       ShortTruncatedBytecodeAtomProgramDeserializationError) {
  // Short enough for `absl::Cord` to store inline.
  ASSERT_OK_AND_ASSIGN(Serialized serialized,
                       SerializeWithAtomProgram([](absl::string_view bytecode) {
                         return std::string(bytecode.substr(0, 15));
                       }));

  EXPECT_THAT(Deserialize<IfrtIRProgram>(serialized, /*options=*/nullptr),
              absl_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  AllOf(HasSubstr("Failed to deserialize atom program"),
                        Not(HasSubstr("not MLIR bytecode")))));
}

TEST_P(IfrtIRProgramSerDesTest,
       LongTruncatedBytecodeAtomProgramDeserializationError) {
  // Too long for `absl::Cord` to store inline.
  ASSERT_OK_AND_ASSIGN(
      Serialized serialized,
      SerializeWithAtomProgram([](absl::string_view bytecode) {
        return std::string(bytecode.substr(0, bytecode.size() - 1));
      }));

  EXPECT_THAT(Deserialize<IfrtIRProgram>(serialized, /*options=*/nullptr),
              absl_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  AllOf(HasSubstr("Failed to deserialize atom program"),
                        Not(HasSubstr("not MLIR bytecode")))));
}

INSTANTIATE_TEST_SUITE_P(
    SerDesVersion, IfrtIRProgramSerDesTest,
    testing::ValuesIn(test_util::Week4OldOrLaterSerDesVersions()),
    [](const testing::TestParamInfo<SerDesVersion>& info) {
      return absl::StrCat(info.param.version_number().value());
    });

}  // namespace
}  // namespace ifrt
}  // namespace xla
