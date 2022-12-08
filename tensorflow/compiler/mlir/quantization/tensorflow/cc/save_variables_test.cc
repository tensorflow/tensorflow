/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/quantization/tensorflow/cc/save_variables.h"

#include <string>

#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/status_matchers.h"

namespace tensorflow {
namespace quantization {
namespace {

using ::tensorflow::test::AsTensor;
using ::tensorflow::test::ExpectEqual;
using ::testing::Not;
using ::tsl::testing::IsOk;

// This fixture simply wraps the Env and MLIRContext.
class SaveVariablesToCheckpointTest : public ::testing::Test {
 protected:
  SaveVariablesToCheckpointTest() : env_(Env::Default()) {
    ctx_.loadDialect<mlir::func::FuncDialect, mlir::TF::TensorFlowDialect,
                     mlir::tf_saved_model::TensorFlowSavedModelDialect>();
  }

  absl::StatusOr<std::string> MakeTempDir() {
    std::string tmp_dir{};
    if (!env_->LocalTempFilename(&tmp_dir)) {
      return absl::InternalError("Failed to create temp file.");
    }

    TF_CHECK_OK(env_->CreateDir(tmp_dir));
    return tmp_dir;
  }

  // Parses `module_op_str` to create a `ModuleOp`. Checks whether the created
  // module op is valid.
  mlir::OwningOpRef<mlir::ModuleOp> ParseModuleOpString(
      const absl::string_view module_op_str) {
    auto module_op_ref =
        mlir::parseSourceString<mlir::ModuleOp>(module_op_str, &ctx_);
    EXPECT_TRUE(module_op_ref);
    return module_op_ref;
  }

  Env* env_{};
  mlir::MLIRContext ctx_{};
};

TEST_F(SaveVariablesToCheckpointTest, VariableSavedToCheckpoint) {
  constexpr absl::string_view kModuleCode = R"mlir(
    module attributes {tf_saved_model.semantics} {
      "tf_saved_model.session_initializer"() {initializers = [@init_func_restore_op]} : () -> ()

      func.func @init_func_restore_op() -> () attributes {tf_saved_model.exported_names = ["restore"], tf_saved_model.initializer_type = "restore_op"} {
        %cst = "tf.Const"() {device = "", value = dense<[1.0, 2.0]> : tensor<2xf32>} : () -> tensor<2xf32>
        %0 = "tf.VarHandleOp"() {container = "", device = "/device:CPU:0", shared_name = "var_0"} : () -> tensor<!tf_type.resource<tensor<2xf32>>>
        "tf.AssignVariableOp"(%0, %cst) : (tensor<!tf_type.resource<tensor<2xf32>>>, tensor<2xf32>) -> ()
        return
      }
    }
  )mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module_op_ref =
      ParseModuleOpString(kModuleCode);

  const absl::StatusOr<std::string> checkpoint_prefix = MakeTempDir();
  EXPECT_TRUE(checkpoint_prefix.ok());

  const absl::Cleanup checkpoint_prefix_cleanup = [this, &checkpoint_prefix]() {
    int64_t undeleted_files, undeleted_dirs;
    TF_CHECK_OK(env_->DeleteRecursively(*checkpoint_prefix, &undeleted_files,
                                        &undeleted_dirs));
  };

  EXPECT_TRUE(
      SaveVariablesToCheckpoint(*checkpoint_prefix, *module_op_ref).ok());

  // Verify the saved variable.
  BundleReader bundle_reader(env_, *checkpoint_prefix);

  Tensor loaded_tensor{};
  EXPECT_TRUE(
      tsl::ToAbslStatus(bundle_reader.Lookup("var_0", &loaded_tensor)).ok());

  ExpectEqual(loaded_tensor, AsTensor<float>({1.0, 2.0}));
}

TEST_F(SaveVariablesToCheckpointTest, MultipleVariablesSavedToCheckpoint) {
  // Module's session intializer contains two variables.
  constexpr absl::string_view kModuleCode = R"mlir(
    module attributes {tf_saved_model.semantics} {
      "tf_saved_model.session_initializer"() {initializers = [@init_func_restore_op]} : () -> ()

      func.func @init_func_restore_op() -> () attributes {tf_saved_model.exported_names = ["restore"], tf_saved_model.initializer_type = "restore_op"} {
        %cst = "tf.Const"() {device = "", value = dense<[1.0, 2.0]> : tensor<2xf32>} : () -> tensor<2xf32>
        %0 = "tf.VarHandleOp"() {container = "", device = "/device:CPU:0", shared_name = "var_0"} : () -> tensor<!tf_type.resource<tensor<2xf32>>>
        "tf.AssignVariableOp"(%0, %cst) : (tensor<!tf_type.resource<tensor<2xf32>>>, tensor<2xf32>) -> ()

        %cst_0 = "tf.Const"() {device = "", value = dense<[3, 4, 5, 6]> : tensor<4xi32>} : () -> tensor<4xi32>
        %1 = "tf.VarHandleOp"() {container = "", device = "/device:CPU:0", shared_name = "var_1"} : () -> tensor<!tf_type.resource<tensor<4xi32>>>
        "tf.AssignVariableOp"(%1, %cst_0) : (tensor<!tf_type.resource<tensor<4xi32>>>, tensor<4xi32>) -> ()

        return
      }
    }
  )mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module_op_ref =
      ParseModuleOpString(kModuleCode);

  const absl::StatusOr<std::string> checkpoint_prefix = MakeTempDir();
  EXPECT_TRUE(checkpoint_prefix.ok());

  const absl::Cleanup checkpoint_prefix_cleanup = [this, &checkpoint_prefix]() {
    int64_t undeleted_files, undeleted_dirs;
    TF_CHECK_OK(env_->DeleteRecursively(*checkpoint_prefix, &undeleted_files,
                                        &undeleted_dirs));
  };

  EXPECT_TRUE(
      SaveVariablesToCheckpoint(*checkpoint_prefix, *module_op_ref).ok());

  // Verify that both variables are saved correctly.
  BundleReader bundle_reader(env_, *checkpoint_prefix);

  Tensor loaded_var_0{};
  EXPECT_TRUE(
      tsl::ToAbslStatus(bundle_reader.Lookup("var_0", &loaded_var_0)).ok());
  ExpectEqual(loaded_var_0, AsTensor<float>({1.0, 2.0}));

  Tensor loaded_var_1{};
  EXPECT_TRUE(
      tsl::ToAbslStatus(bundle_reader.Lookup("var_1", &loaded_var_1)).ok());
  ExpectEqual(loaded_var_1, AsTensor<int>({3, 4, 5, 6}));
}

TEST_F(SaveVariablesToCheckpointTest,
       NoVariablesSavedWhenNoInitializerFunction) {
  constexpr absl::string_view kModuleCode = R"mlir(
    module attributes {tf_saved_model.semantics} {
      "tf_saved_model.session_initializer"() {initializers = []} : () -> ()
    }
  )mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module_op_ref =
      ParseModuleOpString(kModuleCode);

  const absl::StatusOr<std::string> checkpoint_prefix = MakeTempDir();
  EXPECT_TRUE(checkpoint_prefix.ok());

  const absl::Cleanup checkpoint_prefix_cleanup = [this, &checkpoint_prefix]() {
    int64_t undeleted_files, undeleted_dirs;
    TF_CHECK_OK(env_->DeleteRecursively(*checkpoint_prefix, &undeleted_files,
                                        &undeleted_dirs));
  };

  EXPECT_TRUE(
      SaveVariablesToCheckpoint(*checkpoint_prefix, *module_op_ref).ok());

  // Verify that the checkpoint doesn't exist.
  BundleReader bundle_reader(env_, *checkpoint_prefix);
  EXPECT_THAT(bundle_reader.status(), Not(IsOk()));
}

TEST_F(SaveVariablesToCheckpointTest,
       NoVariablesSavedWhenNoSessionInitializerOp) {
  constexpr absl::string_view kModuleCode = R"mlir(
    module {
      func.func @my_func() -> () {
        return
      }
    }
  )mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module_op_ref =
      ParseModuleOpString(kModuleCode);

  const absl::StatusOr<std::string> checkpoint_prefix = MakeTempDir();
  EXPECT_TRUE(checkpoint_prefix.ok());

  const absl::Cleanup checkpoint_prefix_cleanup = [this, &checkpoint_prefix]() {
    int64_t undeleted_files, undeleted_dirs;
    TF_CHECK_OK(env_->DeleteRecursively(*checkpoint_prefix, &undeleted_files,
                                        &undeleted_dirs));
  };

  EXPECT_TRUE(
      SaveVariablesToCheckpoint(*checkpoint_prefix, *module_op_ref).ok());

  // Verify that the checkpoint doesn't exist.
  BundleReader bundle_reader(env_, *checkpoint_prefix);
  EXPECT_THAT(bundle_reader.status(), Not(IsOk()));
}

TEST_F(SaveVariablesToCheckpointTest,
       NoVariablesSavedWhenNoSessionInitializerOpTypeRestoreOp) {
  constexpr absl::string_view kModuleCode = R"mlir(
    module attributes {tf_saved_model.semantics} {
      "tf_saved_model.session_initializer"() {initializers = [@init_func_init_op]} : () -> ()

      func.func @init_func_init_op() -> () attributes {tf_saved_model.exported_names = ["init"], tf_saved_model.initializer_type = "init_op"} {
        %cst = "tf.Const"() {device = "", value = dense<[1.0, 2.0]> : tensor<2xf32>} : () -> tensor<2xf32>
        %0 = "tf.VarHandleOp"() {container = "", device = "/device:CPU:0", shared_name = "var_0"} : () -> tensor<!tf_type.resource<tensor<2xf32>>>
        "tf.AssignVariableOp"(%0, %cst) : (tensor<!tf_type.resource<tensor<2xf32>>>, tensor<2xf32>) -> ()
        return
      }
    }
  )mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module_op_ref =
      ParseModuleOpString(kModuleCode);

  const absl::StatusOr<std::string> checkpoint_prefix = MakeTempDir();
  EXPECT_TRUE(checkpoint_prefix.ok());

  const absl::Cleanup checkpoint_prefix_cleanup = [this, &checkpoint_prefix]() {
    int64_t undeleted_files, undeleted_dirs;
    TF_CHECK_OK(env_->DeleteRecursively(*checkpoint_prefix, &undeleted_files,
                                        &undeleted_dirs));
  };

  EXPECT_TRUE(
      SaveVariablesToCheckpoint(*checkpoint_prefix, *module_op_ref).ok());

  // Verify that the checkpoint doesn't exist.
  BundleReader bundle_reader(env_, *checkpoint_prefix);
  EXPECT_THAT(bundle_reader.status(), Not(IsOk()));
}

TEST_F(SaveVariablesToCheckpointTest, MutableVariablesNotSaved) {
  // This function includes an AssignVariableOp that does not initialize the
  // variable from a ConstOp. In this case, the variable is not saved to the
  // checkpoint.
  constexpr absl::string_view kModuleCode = R"mlir(
    module attributes {tf_saved_model.semantics} {
      "tf_saved_model.session_initializer"() {initializers = [@init_func_restore_op]} : () -> ()

      func.func @init_func_restore_op() -> () attributes {tf_saved_model.exported_names = ["init"], tf_saved_model.initializer_type = "restore_op"} {
        %cst = "tf.Const"() {device = "", value = dense<[1.0, 2.0]> : tensor<2xf32>} : () -> tensor<2xf32>
        %add = "tf.AddV2"(%cst, %cst) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
        %var_handle = "tf.VarHandleOp"() {container = "", device = "/device:CPU:0", shared_name = "var_0"} : () -> tensor<!tf_type.resource<tensor<2xf32>>>
        "tf.AssignVariableOp"(%var_handle, %add) : (tensor<!tf_type.resource<tensor<2xf32>>>, tensor<2xf32>) -> ()
        return
      }
    }
  )mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module_op_ref =
      ParseModuleOpString(kModuleCode);

  const absl::StatusOr<std::string> checkpoint_prefix = MakeTempDir();
  EXPECT_TRUE(checkpoint_prefix.ok());

  const absl::Cleanup checkpoint_prefix_cleanup = [this, &checkpoint_prefix]() {
    int64_t undeleted_files, undeleted_dirs;
    TF_CHECK_OK(env_->DeleteRecursively(*checkpoint_prefix, &undeleted_files,
                                        &undeleted_dirs));
  };

  EXPECT_TRUE(
      SaveVariablesToCheckpoint(*checkpoint_prefix, *module_op_ref).ok());

  BundleReader bundle_reader(env_, *checkpoint_prefix);
  EXPECT_THAT(bundle_reader.status(), IsOk());

  // Verify that the variable is not saved.
  EXPECT_FALSE(bundle_reader.Contains("var_0"));
}

}  // namespace
}  // namespace quantization
}  // namespace tensorflow
