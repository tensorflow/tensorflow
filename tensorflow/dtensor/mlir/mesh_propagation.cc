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

#include <optional>
#include <string>
#include <utility>

#include "absl/types/optional.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/DebugStringHelper.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/dtensor_mlir_passes.h"
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/op_utils.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"

namespace tensorflow {
namespace dtensor {

namespace {
#define GEN_PASS_DEF_DTENSORMESHPROPAGATION
#include "tensorflow/dtensor/mlir/dtensor_passes.h.inc"

// Extracts mesh of `block_arg` by parsing function argument attributes of it's
// enclosing function. Mesh is inferred either using `tf._layout` or `tf._mesh`
// attributes.
mlir::LogicalResult ExtractMeshFromBlockArgument(mlir::BlockArgument block_arg,
                                                 absl::optional<Mesh>* out) {
  auto func_op = mlir::dyn_cast_or_null<mlir::func::FuncOp>(
      block_arg.getOwner()->getParentOp());
  if (!func_op) {
    return block_arg.getOwner()->getParentOp()->emitOpError(
        "must be enclosed by a function");
  }
  auto layout_or_status = ExtractLayoutFromOperand(block_arg);
  if (!layout_or_status.ok())
    return func_op.emitOpError(layout_or_status.status().error_message());

  if (layout_or_status->has_value()) {
    out->emplace(layout_or_status->value().mesh());
    return mlir::success();
  }

  auto mesh_attr = func_op.getArgAttrOfType<mlir::StringAttr>(
      block_arg.getArgNumber(), kCustomDeviceMeshAttr);
  if (!mesh_attr) return mlir::success();

  auto mesh_from_block_arg_or_status =
      Mesh::FromString(mesh_attr.getValue().str());
  if (!mesh_from_block_arg_or_status.ok()) {
    return func_op.emitOpError(
        "Failed during mesh propagation. Op operand has invalid serialized "
        "mesh");
  }

  out->emplace(mesh_from_block_arg_or_status.value());
  return mlir::success();
}

// Extracts mesh of operation that produces `value`.
mlir::LogicalResult ExtractMeshFromOpOutput(mlir::Value value,
                                            absl::optional<Mesh>* out) {
  auto input_op = value.getDefiningOp();
  if (!input_op) return mlir::success();

  auto operand_cluster =
      llvm::dyn_cast<mlir::tf_device::ClusterOp>(value.getDefiningOp());
  if (!operand_cluster) {
    return mlir::emitError(value.getLoc())
           << "operand must be from different device cluster.";
  }

  auto mesh_or_status = ExtractDeviceMeshFromOp(operand_cluster);
  if (!mesh_or_status.ok())
    return operand_cluster.emitOpError(
        llvm::formatv("Failed during mesh propagation. {0}",
                      mesh_or_status.status().error_message()));

  auto extracted_mesh = mesh_or_status.value();
  if (extracted_mesh) *out = extracted_mesh.value();
  return mlir::success();
}

// Extracts mesh configuration from `operand`. If operand is a function
// argument, then mesh config is extracted from "tf._mesh" arg attribute of the
// corresponding func op. If operand is from a preceding op, then mesh
// configuration is extracted from the enclosing tf_device.Cluster op.
mlir::LogicalResult ExtractMeshFromOperand(
    const llvm::DenseMap<mlir::OpOperand*, std::vector<mlir::Value>>& producers,
    mlir::OpOperand* operand, absl::optional<Mesh>* out) {
  mlir::Value operand_value = operand->get();

  const auto check_and_assign_mesh =
      [](mlir::Location loc, absl::optional<Mesh>& mesh,
         absl::optional<Mesh>& operand_mesh) -> mlir::LogicalResult {
    if (mesh && !operand_mesh) {
      operand_mesh.swap(mesh);
    } else if (mesh && operand_mesh && mesh != operand_mesh) {
      return mlir::emitError(
          loc,
          "Error during mesh propagation. Found inconsistent mesh "
          "while inferring mesh from operands.");
    }
    return mlir::success();
  };

  // If `operand` is a block argument then extract mesh from `tf._mesh`
  // attribute of the corresponding function argument.
  if (auto block_arg = operand_value.dyn_cast<mlir::BlockArgument>()) {
    if (mlir::failed(ExtractMeshFromBlockArgument(block_arg, out)))
      return mlir::failure();

    if (!out->has_value()) {
      auto it = producers.find(operand);
      if (it != producers.end()) {
        auto producer_values = it->getSecond();
        absl::optional<Mesh> operand_mesh;
        for (mlir::Value producer_value : producer_values) {
          if (auto arg = producer_value.dyn_cast<mlir::BlockArgument>()) {
            absl::optional<Mesh> mesh;
            if (mlir::failed(ExtractMeshFromBlockArgument(arg, &mesh)))
              return mlir::failure();

            if (mlir::failed(check_and_assign_mesh(
                    operand->getOwner()->getLoc(), mesh, operand_mesh)))
              return mlir::failure();
          } else {
            auto input_cluster =
                producer_value.getDefiningOp()
                    ->getParentOfType<mlir::tf_device::ClusterOp>();
            auto output_from_producing_op = input_cluster.getResult(
                producer_value.cast<mlir::OpResult>().getResultNumber());

            absl::optional<Mesh> mesh;
            if (mlir::failed(
                    ExtractMeshFromOpOutput(output_from_producing_op, &mesh)))
              return mlir::failure();

            if (mlir::failed(check_and_assign_mesh(
                    operand->getOwner()->getLoc(), mesh, operand_mesh)))
              return mlir::failure();
          }
        }
        *out = operand_mesh;
      }
    }
    return mlir::success();
  }

  // If `operand` is from another operation, extract mesh from enclosing
  // tf_device.cluster op of the input operation.
  if (mlir::failed(ExtractMeshFromOpOutput(operand_value, out)))
    return mlir::failure();

  return mlir::success();
}

// Infers mesh of `cluster` from it's operands. If mesh can be inferred, all
// operands must have same mesh.
mlir::LogicalResult InferMeshFromInputs(
    const llvm::DenseMap<mlir::OpOperand*, std::vector<mlir::Value>>& producers,
    mlir::tf_device::ClusterOp cluster, absl::optional<Mesh>* mesh,
    llvm::SmallVector<mlir::OpOperand*, 8>* inputs_with_inferred_mesh) {
  auto result = mlir::success();

  // If `cluster` wraps a `tf.CopyToMesh` op, do not infer mesh from it's
  // inputs. `tf.CopyToMesh` specifies that all operations following the
  // operation is executed on target device mesh cluster specified by
  // `tf.CopyToMesh`.
  if (llvm::isa<mlir::TF::CopyToMeshOp>(&cluster.GetBody().front()))
    return result;
  if (llvm::isa<mlir::TF::CopyToMeshGradOp>(&cluster.GetBody().front())) {
    return result;
  }

  mlir::visitUsedValuesDefinedAbove(
      cluster.getBody(), cluster.getBody(), [&](mlir::OpOperand* operand) {
        if (mlir::failed(result)) return;
        absl::optional<Mesh> extracted_config;

        // If inputs to mesh is from DTensorLayout op, then use the mesh
        // extracted from the DTensorLayout op to infer the mesh of the cluster.
        if (auto layout_op =
                llvm::dyn_cast<mlir::TF::DTensorLayout>(operand->getOwner())) {
          auto mesh = layout_op.getLayout().mesh();
          extracted_config.emplace(mesh);
        } else {
          auto extract_result =
              ExtractMeshFromOperand(producers, operand, &extracted_config);
          if (mlir::failed(extract_result)) {
            result = extract_result;
            return;
          }
        }

        // DTensorDevice may create a graph with resource arguments with an
        // empty layout. These layouts of the resource values will be added
        // after layout is inferred from resource update ops. Therefore, ignore
        // DTensorLayout ops will empty layouts.
        if (!extracted_config || extracted_config->IsEmpty()) return;

        inputs_with_inferred_mesh->emplace_back(operand);
        if (mesh->has_value() && extracted_config != mesh->value()) {
          result = cluster.emitOpError(
              "failed during mesh propagation. All inputs to "
              "`tf_device.Cluster` must have same mesh configuration.");
        }

        if (!mesh->has_value()) mesh->emplace(extracted_config.value());
      });

  return result;
}

// Extracts mesh from function return attributes. If `tf._default_layout`
// attribute exists, mesh from the default layout is used. If not, mesh from
// `tf._mesh` attribute is used.
StatusOr<absl::optional<Mesh>> ExtractMeshFromFuctionOutput(
    const int output_index, mlir::func::FuncOp function) {
  absl::optional<Mesh> function_mesh;
  auto terminator = llvm::cast<mlir::func::ReturnOp>(
      function.getBody().front().getTerminator());
  TF_ASSIGN_OR_RETURN(auto layout, ExtractLayoutFromFunctionReturnAttr(
                                       terminator, output_index));

  if (layout) {
    function_mesh.emplace(layout->mesh());
    return function_mesh;
  }

  auto output_mesh_attr = function.getResultAttrOfType<mlir::StringAttr>(
      output_index, kCustomDeviceMeshAttr);
  if (output_mesh_attr) {
    TF_ASSIGN_OR_RETURN(auto mesh,
                        Mesh::FromString(output_mesh_attr.getValue().str()));
    function_mesh.emplace(std::move(mesh));
  }
  return function_mesh;
}

// Infers mesh from users of `cluster` and records the usages that were used to
// infer mesh configuration in `consumers_with_mesh`.
mlir::LogicalResult InferMeshFromConsumers(
    mlir::tf_device::ClusterOp cluster, absl::optional<Mesh>* mesh,
    llvm::SmallVector<mlir::OpOperand*, 8>* consumers_with_mesh) {
  for (auto& use_value : cluster.getOperation()->getUses()) {
    mlir::Operation* consumer = use_value.getOwner();

    // `tf.CopyToMesh` specifies that all operations following the
    // operation are executed on target device mesh cluster specified by
    // `tf.CopyToMesh`. Therefore, if `consumer` operation is `tf.CopyToMesh`
    // do not propagate mesh backwards to `cluster`.
    if (llvm::isa<mlir::TF::CopyToMeshOp>(consumer)) continue;
    if (llvm::isa<mlir::TF::CopyToMeshGradOp>(&cluster.GetBody().front()))
      continue;

    Mesh extracted_mesh;

    // If `cluster` output is output value of a function, then infer mesh using
    // function return value attribute, if it exists.
    if (auto return_op = llvm::dyn_cast<mlir::func::ReturnOp>(consumer)) {
      auto status_or_mesh = ExtractMeshFromFuctionOutput(
          use_value.getOperandNumber(),
          return_op->getParentOfType<mlir::func::FuncOp>());
      if (!status_or_mesh.ok())
        return cluster.emitOpError(status_or_mesh.status().ToString());

      auto mesh = status_or_mesh.value();
      if (mesh) extracted_mesh = *mesh;
    } else {
      // If `cluster` output is input to another cluster/op then infer mesh from
      // the consumer operation.
      auto consumer_cluster =
          consumer->getParentOfType<mlir::tf_device::ClusterOp>();
      if (!consumer_cluster) {
        return cluster.emitOpError(
            "failed to propagate mesh information. All operations must be "
            "enclosed inside a tf_device.cluster op.");
      }

      auto mesh_or_status = ExtractDeviceMeshFromOp(consumer_cluster);
      if (!mesh_or_status.ok())
        return cluster.emitOpError(mesh_or_status.status().error_message());

      auto consumer_mesh = mesh_or_status.value();
      if (!consumer_mesh) continue;

      extracted_mesh = consumer_mesh.value();
    }

    if (extracted_mesh.IsEmpty()) continue;

    if (mesh->has_value() && extracted_mesh != mesh->value()) {
      return cluster.emitOpError(
          "failed to propagate mesh information. Mesh for op is ambiguous as "
          "consumers have different mesh attributes");
    }

    consumers_with_mesh->emplace_back(&use_value);
    if (!mesh->has_value()) mesh->emplace(std::move(extracted_mesh));
  }
  return mlir::success();
}

// Infers default mesh of function given it's inputs and outputs. Function has a
// default mesh if all its inputs/outputs have valus assigned to the same mesh.
mlir::LogicalResult InferFunctionDefaultMesh(
    const llvm::DenseMap<mlir::OpOperand*, std::vector<mlir::Value>>& producers,
    mlir::func::FuncOp function, mlir::OpBuilder* builder,
    absl::optional<mlir::StringAttr>* inferred_default_mesh) {
  auto terminator = function.getCallableRegion()->front().getTerminator();
  for (auto& result_value : terminator->getOpOperands()) {
    auto result_defining_op = result_value.get().getDefiningOp();
    if (!result_defining_op) continue;

    auto result_cluster =
        llvm::cast<mlir::tf_device::ClusterOp>(result_defining_op);
    auto result_mesh =
        result_cluster->getAttrOfType<mlir::StringAttr>(kMeshAttr);
    if (!result_mesh) continue;

    if (inferred_default_mesh->has_value() &&
        inferred_default_mesh->value() != result_mesh) {
      inferred_default_mesh->reset();
      return mlir::success();
    }
    inferred_default_mesh->emplace(result_mesh);
  }

  absl::optional<Mesh> inferred_mesh_from_args;
  for (auto function_arg : function.getArguments()) {
    auto uses = function_arg.getUses();
    if (uses.empty()) {
      if (mlir::failed(ExtractMeshFromBlockArgument(function_arg,
                                                    &inferred_mesh_from_args)))
        return mlir::failure();
    } else {
      auto operand = uses.begin().getOperand();
      if (mlir::failed(ExtractMeshFromOperand(producers, operand,
                                              &inferred_mesh_from_args)))
        return mlir::failure();
    }
    if (!inferred_mesh_from_args) continue;

    std::string mesh_str = inferred_mesh_from_args->ToString();
    if (inferred_default_mesh->has_value() &&
        inferred_default_mesh->value().getValue().str() != mesh_str) {
      inferred_default_mesh->reset();
      return mlir::success();
    }

    inferred_default_mesh->emplace(builder->getStringAttr(std::move(mesh_str)));
  }
  return mlir::success();
}

// Annotates `tf._mesh` attribute to argument of `function` with
// string of `mesh`.
void AnnotateFunctionArgumentsWithMeshInformation(
    const Mesh& mesh,
    const llvm::SmallVector<mlir::OpOperand*, 8>& input_values_from_mesh,
    mlir::func::FuncOp function, mlir::OpBuilder* builder) {
  for (auto value : input_values_from_mesh) {
    function.setArgAttr(value->getOperandNumber(), kCustomDeviceMeshAttr,
                        builder->getStringAttr(mesh.ToString()));
  }
}

// Annotates return value attributes of `function_to_annotate` with mesh
// information parsed from usages of the function. `callsite_operation` is
// callable op whose function definition is `function_to_annotate`.
mlir::LogicalResult AnnotateFunctionReturnValuesWithMeshInformation(
    const llvm::SmallVector<mlir::OpOperand*, 8>& return_values_from_mesh,
    mlir::Operation* callsite_operation,
    mlir::func::FuncOp function_to_annotate, mlir::OpBuilder* builder) {
  for (auto value : return_values_from_mesh) {
    absl::optional<mlir::StringAttr> result_mesh_attribute;
    if (llvm::isa<mlir::func::ReturnOp>(value->getOwner())) {
      auto parent_function =
          callsite_operation->getParentOfType<mlir::func::FuncOp>();
      auto function_result_layout =
          parent_function.getResultAttrOfType<mlir::StringAttr>(
              value->getOperandNumber(), kCustomDefaultLayoutAttr);
      if (function_result_layout) {
        auto layout_or_status =
            Layout::FromString(function_result_layout.getValue().str());
        if (!layout_or_status.ok())
          return parent_function.emitOpError(
              layout_or_status.status().error_message());

        result_mesh_attribute.emplace(
            builder->getStringAttr(layout_or_status->mesh().ToString()));
      } else {
        auto function_result_mesh =
            parent_function.getResultAttrOfType<mlir::StringAttr>(
                value->getOperandNumber(), kCustomDeviceMeshAttr);
        if (function_result_mesh)
          result_mesh_attribute.emplace(function_result_mesh);
      }
    } else {
      auto op_mesh =
          value->getOwner()->getAttrOfType<mlir::StringAttr>(kMeshAttr);
      if (op_mesh) result_mesh_attribute.emplace(std::move(op_mesh));
    }

    if (result_mesh_attribute)
      function_to_annotate.setResultAttr(
          value->get().cast<mlir::OpResult>().getResultNumber(),
          kCustomDeviceMeshAttr, result_mesh_attribute.value());
  }
  return mlir::success();
}

// MLIR pass that propagates mesh information to tf_device.Cluster ops.
struct DTensorMeshPropagation
    : public impl::DTensorMeshPropagationBase<DTensorMeshPropagation> {
  void runOnOperation() override {
    mlir::MLIRContext& context = getContext();
    mlir::OpBuilder builder(&context);
    auto module = getOperation();
    mlir::func::FuncOp main_func =
        module.lookupSymbol<mlir::func::FuncOp>("main");
    if (!main_func) return;

    mlir::Dialect* tf_dialect =
        context.getLoadedDialect<mlir::TF::TensorFlowDialect>();

    // This maps from OpResults to a list of OpOperands that consume this.
    // Note that this will pass over/through
    // (Stateful)PartitionedCall and other control flow, directly connecting
    // producing ops to their consumers in the function. I.e. it presents
    // flattened/inlined view of the flow of data.
    llvm::DenseMap<mlir::Value, std::vector<mlir::OpOperand*>> consumers;
    // Maintain a reverse mapping. Note that for controlflow operations like
    // tf.If op, there may be multiple producers for a mlir::Value.
    llvm::DenseMap<mlir::OpOperand*, std::vector<mlir::Value>> producers;

    // Create consumers and producers maps.
    if (mlir::failed(
            PopulateConsumersFromModule(&module, tf_dialect, consumers)))
      return signalPassFailure();

    for (auto& consumer : consumers) {
      for (auto* operand : consumer.second) {
        producers[operand].emplace_back(consumer.first);
      }
    }

    bool mesh_changed = true;
    while (mesh_changed) {
      mesh_changed = false;
      if (mlir::failed(
              PropagateMesh(producers, main_func, &builder, &mesh_changed)))
        return signalPassFailure();
    }
  }

  // Propagates and sets `_mesh` attributes to all clusters inside `function` if
  // possible.
  mlir::LogicalResult PropagateMesh(
      const llvm::DenseMap<mlir::OpOperand*, std::vector<mlir::Value>>&
          producers,
      mlir::func::FuncOp, mlir::OpBuilder* builder, bool* mesh_changed);

  // Infers mesh of `cluster` from its input operations.
  mlir::LogicalResult PropagateMeshFromInputs(
      const llvm::DenseMap<mlir::OpOperand*, std::vector<mlir::Value>>&
          producers,
      mlir::tf_device::ClusterOp cluster, mlir::OpBuilder* builder,
      bool* mesh_changed);

  // Infers mesh of `cluster` from its consuming operations.
  mlir::LogicalResult PropagateMeshFromConsumers(
      const llvm::DenseMap<mlir::OpOperand*, std::vector<mlir::Value>>&
          producers,
      mlir::tf_device::ClusterOp cluster, mlir::OpBuilder* builder,
      bool* mesh_changed);

  // Assigns function default mesh to clusters with no mesh specified. Note that
  // function has default mesh if all its dtensor inputs/outputs are assigned to
  // a single mesh.
  mlir::LogicalResult PropagateDefaultMeshToUnAssignedClusters(
      const llvm::DenseMap<mlir::OpOperand*, std::vector<mlir::Value>>&
          producers,
      mlir::func::FuncOp, mlir::OpBuilder* builder, bool* mesh_changed);
};

mlir::LogicalResult
DTensorMeshPropagation::PropagateDefaultMeshToUnAssignedClusters(
    const llvm::DenseMap<mlir::OpOperand*, std::vector<mlir::Value>>& producers,
    mlir::func::FuncOp function, mlir::OpBuilder* builder, bool* mesh_changed) {
  absl::optional<mlir::StringAttr> mesh;
  if (mlir::failed(
          InferFunctionDefaultMesh(producers, function, builder, &mesh)))
    return mlir::failure();

  llvm::SmallVector<mlir::tf_device::ClusterOp, 4> clusters_without_mesh;
  auto walk_result = function.walk([&](mlir::tf_device::ClusterOp cluster) {
    if (llvm::isa<mlir::TF::CopyToMeshGradOp>(&cluster.GetBody().front()))
      return mlir::WalkResult::advance();

    auto mesh_or_status = ExtractDeviceMeshFromOp(cluster);
    if (!mesh_or_status.ok()) {
      cluster.GetBody().front().emitOpError(
          mesh_or_status.status().error_message());
      return mlir::WalkResult::interrupt();
    }

    const auto& mesh = mesh_or_status.value();
    if (mesh.has_value()) return mlir::WalkResult::advance();

    clusters_without_mesh.emplace_back(cluster);
    return mlir::WalkResult::advance();
  });

  if (walk_result.wasInterrupted()) return mlir::failure();

  if (!mesh.has_value()) return mlir::success();

  // Set function default mesh to cluster with unspecified mesh.
  for (auto cluster_without_mesh : clusters_without_mesh) {
    *mesh_changed = true;
    cluster_without_mesh->setAttr(kMeshAttr, mesh.value());
  }

  return mlir::success();
}

mlir::LogicalResult DTensorMeshPropagation::PropagateMeshFromInputs(
    const llvm::DenseMap<mlir::OpOperand*, std::vector<mlir::Value>>& producers,
    mlir::tf_device::ClusterOp cluster, mlir::OpBuilder* builder,
    bool* mesh_changed) {
  // If operation inside a mesh cluster is not a callable operation and
  // mesh is already specified on a cluster, do nothing.
  auto inner_func = MaybeFindFunction(&cluster.GetBody().front());
  auto cluster_mesh = cluster->getAttrOfType<mlir::StringAttr>(kMeshAttr);
  if (!inner_func && cluster_mesh) return mlir::success();

  // If mesh of `cluster` is not specified, infer mesh using inputs of mesh
  // cluster.
  absl::optional<Mesh> extracted_mesh;
  llvm::SmallVector<mlir::OpOperand*, 8> inputs_with_inferred_mesh;
  if (failed(InferMeshFromInputs(producers, cluster, &extracted_mesh,
                                 &inputs_with_inferred_mesh))) {
    return mlir::failure();
  }

  // If operation include 'cluster` is a function call, annotate input and
  // output mesh of `cluster` using function argument and return value
  // attributes, then recursively propagate mesh of the function definition.
  if (inner_func) {
    // All inputs to cluster must be from the same mesh. If input mesh to
    // callable operation is inferred, then annotated the input mesh to
    // function argument attribute so that this information can be used to
    // infer mesh of ops inside `inner_func`.
    if (extracted_mesh.has_value()) {
      AnnotateFunctionArgumentsWithMeshInformation(extracted_mesh.value(),
                                                   inputs_with_inferred_mesh,
                                                   inner_func.value(), builder);
    }

    // Recursively propagate mesh to clusters in function definition of
    // `inner_func`.
    if (mlir::failed(PropagateMesh(producers, inner_func.value(), builder,
                                   mesh_changed)))
      return mlir::failure();

    // Once all clusters inside `inner_func` callable has been set, now we can
    // infer mesh of `cluster`. That is, mesh of call site operation is equal
    // to mesh of return values of the function.
    absl::optional<mlir::StringAttr> function_mesh;
    if (mlir::failed(InferFunctionDefaultMesh(producers, inner_func.value(),
                                              builder, &function_mesh)))
      return mlir::failure();

    if (function_mesh && !cluster_mesh) {
      *mesh_changed = true;
      cluster->setAttr(kMeshAttr, function_mesh.value());
    }
  } else if (!cluster_mesh && extracted_mesh.has_value()) {
    *mesh_changed = true;
    cluster->setAttr(kMeshAttr,
                     builder->getStringAttr(extracted_mesh->ToString()));
  }
  return mlir::success();
}

// Set mesh of `cluster`, inferring mesh from consumer operations of `cluster`.
mlir::LogicalResult DTensorMeshPropagation::PropagateMeshFromConsumers(
    const llvm::DenseMap<mlir::OpOperand*, std::vector<mlir::Value>>& producers,
    mlir::tf_device::ClusterOp cluster, mlir::OpBuilder* builder,
    bool* mesh_changed) {
  mlir::Operation* op_inside_cluster = &cluster.GetBody().front();
  auto inner_func = MaybeFindFunction(op_inside_cluster);
  auto cluster_mesh = cluster->getAttrOfType<mlir::StringAttr>(kMeshAttr);
  // If mesh is already set, then do nothing.
  if (!inner_func && cluster_mesh) return mlir::success();

  // Infer mesh of `cluster` from its output usages.
  absl::optional<Mesh> extracted_mesh_from_consumers;
  llvm::SmallVector<mlir::OpOperand*, 8> consumers_with_mesh_information;
  if (failed(InferMeshFromConsumers(cluster, &extracted_mesh_from_consumers,
                                    &consumers_with_mesh_information)))
    return mlir::failure();

  // If operation inside mesh cluster is a function callsite operation,
  // then propagate mesh of the function recursively.
  if (inner_func) {
    if (mlir::failed(AnnotateFunctionReturnValuesWithMeshInformation(
            consumers_with_mesh_information, op_inside_cluster,
            inner_func.value(), builder)))
      return mlir::failure();

    if (mlir::failed(PropagateMesh(producers, inner_func.value(), builder,
                                   mesh_changed)))
      return mlir::failure();

    absl::optional<mlir::StringAttr> function_mesh;
    if (mlir::failed(InferFunctionDefaultMesh(producers, inner_func.value(),
                                              builder, &function_mesh)))
      return mlir::failure();

    if (function_mesh && !cluster_mesh) {
      *mesh_changed = true;
      cluster->setAttr(kMeshAttr, function_mesh.value());
    }
  } else if (extracted_mesh_from_consumers && !cluster_mesh) {
    *mesh_changed = true;
    cluster->setAttr(kMeshAttr, builder->getStringAttr(
                                    extracted_mesh_from_consumers->ToString()));
  }
  return mlir::success();
}

mlir::LogicalResult RewriteCopyToMeshGradOp(
    const llvm::DenseMap<mlir::OpOperand*, std::vector<mlir::Value>>& producers,
    mlir::tf_device::ClusterOp cluster, mlir::OpBuilder* builder,
    bool* mesh_changed) {
  auto backward_op = llvm::dyn_cast_or_null<mlir::TF::CopyToMeshGradOp>(
      &cluster.GetBody().front());
  if (!backward_op) {
    // No CopyToMeshGradOp is found. Either the cluster did not have one,
    // or it has been rewritten from previous iterations.
    return mlir::success();
  }

  if (cluster->getAttrOfType<mlir::StringAttr>(kMeshAttr)) {
    return backward_op.emitOpError(
        "A cluster with CopyToMeshGrad is already assigned a mesh. "
        "This indicates an internal error.");
  }

  std::optional<Mesh> mesh;
  mlir::OpOperand& operand = backward_op->getOpOperand(1);  // forward_input();
  // Gets mesh from the forward_input; if propagation has not reached to
  // forward_input, try again later.
  if (mlir::failed(ExtractMeshFromOperand(producers, &operand, &mesh))) {
    return mlir::success();
  }
  cluster->setAttr(kMeshAttr, builder->getStringAttr(mesh->ToString()));

  // Rewrites to CopyToMesh, by combining the sharding spec of the reference
  // layout with the mesh.
  // This assumes the CopyToMesh maintains the layout of the input and only
  // changes the mesh.
  builder->setInsertionPoint(backward_op);
  StatusOr<Layout> layout =
      Layout::FromString(backward_op.getReferenceLayout().str());
  if (!layout.ok()) {
    return backward_op.emitOpError("Failure passing layout: ")
           << backward_op.getReferenceLayout().str();
  }
  layout->set_mesh(mesh.value());

  auto op = builder->create<mlir::TF::CopyToMeshOp>(
      backward_op->getLoc(), backward_op->getResult(0).getType(),
      backward_op.getInput(), layout->ToString());

  backward_op->replaceAllUsesWith(op);
  backward_op->erase();
  *mesh_changed = true;
  return mlir::success();
}

// Propagates mesh information to all `tf_device.Cluster` ops in `function`. If
// `function` includes callable ops, then recursively traverse the function
// definition to propagate mesh information using input operands and consuming
// result ops. Note that at current stage of graph optimization,
// tf_device.cluster ops are enclosing a single operation.
mlir::LogicalResult DTensorMeshPropagation::PropagateMesh(
    const llvm::DenseMap<mlir::OpOperand*, std::vector<mlir::Value>>& producers,
    mlir::func::FuncOp function, mlir::OpBuilder* builder, bool* mesh_changed) {
  // Iterate clusters in topological order propagating mesh from operations'
  // inputs.
  llvm::SmallVector<mlir::tf_device::ClusterOp, 8> cluster_ops;
  for (auto cluster : function.getOps<mlir::tf_device::ClusterOp>()) {
    cluster_ops.emplace_back(cluster);

    if (mlir::failed(
            PropagateMeshFromInputs(producers, cluster, builder, mesh_changed)))
      return mlir::failure();
  }

  // Iterate clusters in reverse topological order and propagate mesh from
  // consumers.
  for (auto cluster : llvm::reverse(cluster_ops)) {
    if (mlir::failed(PropagateMeshFromConsumers(producers, cluster, builder,
                                                mesh_changed)))
      return mlir::failure();
  }

  if (mlir::failed(PropagateDefaultMeshToUnAssignedClusters(
          producers, function, builder, mesh_changed)))
    return mlir::failure();

  for (auto cluster : llvm::reverse(cluster_ops)) {
    if (mlir::failed(RewriteCopyToMeshGradOp(producers, cluster, builder,
                                             mesh_changed))) {
      return mlir::failure();
    }
  }
  return mlir::success();
}

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorMeshPropagationPass() {
  return std::make_unique<DTensorMeshPropagation>();
}

}  // namespace dtensor
}  // namespace tensorflow
