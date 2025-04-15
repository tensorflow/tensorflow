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

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_join.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"

namespace tensorflow {
namespace dtensor {

namespace {
#define GEN_PASS_DEF_DTENSORMESHPROPAGATION
#include "tensorflow/dtensor/mlir/dtensor_passes.h.inc"

// Extracts mesh of `block_arg` by parsing function argument attributes of it's
// enclosing function. Mesh is inferred either using `tf._layout` or `tf._mesh`
// attributes.
mlir::LogicalResult ExtractMeshFromBlockArgumentFunction(
    mlir::BlockArgument block_arg, mlir::func::FuncOp func_op,
    std::optional<Mesh>* out) {
  auto layout_or_status = ExtractLayoutFromOperand(block_arg);
  if (!layout_or_status.ok())
    return func_op.emitOpError(layout_or_status.status().message());

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

// Extracts mesh of `block_arg` which is an operand of while op.
mlir::LogicalResult ExtractMeshFromBlockArgumentWhile(
    mlir::BlockArgument block_arg, mlir::TF::WhileRegionOp while_op,
    std::optional<Mesh>* out) {
  auto while_op_operand = while_op.getOperand(block_arg.getArgNumber());
  auto defining_op = while_op_operand.getDefiningOp();
  if (defining_op) {
    // The while op operand is the result of another op, then follow the
    // defining op to get mesh.
    auto mesh = ExtractDeviceMeshFromOp(defining_op);
    if (!mesh.ok()) {
      return while_op.emitOpError(mesh.status().message());
    }
    if (mesh->has_value()) {
      *out = mesh->value();
    }
    return mlir::success();
  } else if (auto func_block_arg =
                 mlir::dyn_cast<mlir::BlockArgument>(while_op_operand)) {
    // The while op operand is a block argument of the function, then follow the
    // same routine of getting mesh from function argument.
    auto function_op = mlir::dyn_cast_or_null<mlir::func::FuncOp>(
        func_block_arg.getOwner()->getParentOp());
    if (!function_op) {
      return while_op.emitOpError(
          "Block argument must be enclosed by a function");
    }
    return ExtractMeshFromBlockArgumentFunction(func_block_arg, function_op,
                                                out);
  } else {
    return while_op.emitOpError("Can not resolve block argument of while op");
  }
}

// Extracts mesh of `block_arg`.
mlir::LogicalResult ExtractMeshFromBlockArgument(mlir::BlockArgument block_arg,
                                                 std::optional<Mesh>* out) {
  if (auto func_op = mlir::dyn_cast_or_null<mlir::func::FuncOp>(
          block_arg.getOwner()->getParentOp())) {
    return ExtractMeshFromBlockArgumentFunction(block_arg, func_op, out);
  }

  if (auto while_op = mlir::dyn_cast_or_null<mlir::TF::WhileRegionOp>(
          block_arg.getOwner()->getParentOp())) {
    return ExtractMeshFromBlockArgumentWhile(block_arg, while_op, out);
  }

  return block_arg.getOwner()->getParentOp()->emitOpError(
      "must be enclosed by a function of a while op");
}

// Extracts mesh of operation that produces `value`.
mlir::LogicalResult ExtractMeshFromOpOutput(mlir::Value value,
                                            std::optional<Mesh>* out) {
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
                      mesh_or_status.status().message()));

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
    mlir::OpOperand* operand, std::optional<Mesh>* out) {
  mlir::Value operand_value = operand->get();

  const auto check_and_assign_mesh =
      [](mlir::Location loc, std::optional<Mesh>& mesh,
         std::optional<Mesh>& operand_mesh) -> mlir::LogicalResult {
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
  if (auto block_arg = mlir::dyn_cast<mlir::BlockArgument>(operand_value)) {
    if (mlir::failed(ExtractMeshFromBlockArgument(block_arg, out)))
      return mlir::failure();

    if (!out->has_value()) {
      auto it = producers.find(operand);
      if (it != producers.end()) {
        auto producer_values = it->getSecond();
        std::optional<Mesh> operand_mesh;
        for (mlir::Value producer_value : producer_values) {
          if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(producer_value)) {
            std::optional<Mesh> mesh;
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
                mlir::cast<mlir::OpResult>(producer_value).getResultNumber());

            std::optional<Mesh> mesh;
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
    mlir::tf_device::ClusterOp cluster, std::optional<Mesh>* mesh) {
  llvm::SmallVector<mlir::OpOperand*, 8> inputs_with_inferred_mesh;
  auto result = mlir::success();

  mlir::visitUsedValuesDefinedAbove(
      cluster.getBody(), cluster.getBody(), [&](mlir::OpOperand* operand) {
        if (mlir::failed(result)) return;
        std::optional<Mesh> extracted_config;

        // If inputs to mesh is from DTensorLayout op, then use the mesh
        // extracted from the DTensorLayout op to infer the mesh of the cluster.
        if (auto layout_op =
                llvm::dyn_cast<mlir::TF::DTensorLayout>(operand->getOwner())) {
          extracted_config.emplace(layout_op.getLayout().mesh());
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

        inputs_with_inferred_mesh.emplace_back(operand);
        if (mesh->has_value() && extracted_config != mesh->value()) {
          llvm::SmallVector<std::string, 8> input_debug_strings;
          int index = 0;
          for (const auto& input : inputs_with_inferred_mesh) {
            input_debug_strings.push_back(
                llvm::formatv("Input Cluster {0}: {1}", index, input->get()));
            ++index;
          }
          result = cluster.emitOpError(
              llvm::formatv("failed during mesh propagation. All inputs to "
                            "`tf_device.Cluster` must have same mesh "
                            "configuration. List of found inputs:\n{0}",
                            absl::StrJoin(input_debug_strings, "\n")));
        }

        if (!mesh->has_value()) mesh->emplace(extracted_config.value());
      });

  return result;
}

// Extracts mesh from function return attributes. If `tf._default_layout`
// attribute exists, mesh from the default layout is used. If not, mesh from
// `tf._mesh` attribute is used.
StatusOr<std::optional<Mesh>> ExtractMeshFromFuctionOutput(
    const int output_index, mlir::func::FuncOp function) {
  std::optional<Mesh> function_mesh;
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
    mlir::tf_device::ClusterOp cluster, std::optional<Mesh>* mesh,
    llvm::SmallVector<mlir::OpOperand*, 8>* consumers_with_mesh) {
  for (auto& use_value : cluster.getOperation()->getUses()) {
    mlir::Operation* consumer = use_value.getOwner();

    // `tf.CopyToMesh` specifies that all operations following the
    // operation are executed on target device mesh cluster specified by
    // `tf.CopyToMesh`. Therefore, if `consumer` operation is `tf.CopyToMesh`
    // do not propagate mesh backwards to `cluster`.
    if (llvm::isa<mlir::TF::CopyToMeshOp>(consumer)) continue;
    if (llvm::isa<mlir::TF::RelayoutOp>(consumer)) continue;
    if (llvm::isa<mlir::TF::CopyToMeshGradOp>(&cluster.GetBody().front()))
      continue;
    if (llvm::isa<mlir::TF::RelayoutLikeOp>(&cluster.GetBody().front()))
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
        return cluster.emitOpError(mesh_or_status.status().message());

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
    mlir::StringAttr* inferred_default_mesh) {
  mlir::StringAttr inferred_mesh;
  auto terminator = function.getCallableRegion()->front().getTerminator();
  for (auto& result_value : terminator->getOpOperands()) {
    auto result_defining_op = result_value.get().getDefiningOp();
    if (!result_defining_op) continue;

    auto result_cluster =
        llvm::cast<mlir::tf_device::ClusterOp>(result_defining_op);
    auto result_mesh =
        result_cluster->getAttrOfType<mlir::StringAttr>(kMeshAttr);
    if (!result_mesh) continue;

    if (inferred_mesh && inferred_mesh != result_mesh) {
      return mlir::success();
    }
    inferred_mesh = result_mesh;
  }

  std::optional<Mesh> inferred_mesh_from_args;
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
    if (inferred_mesh && inferred_mesh.getValue().str() != mesh_str) {
      return mlir::success();
    }
    inferred_mesh = builder->getStringAttr(std::move(mesh_str));
  }
  // At this time, we are sure that all the inputs and outputs of a function
  // belong to the same mesh. Use this as the inferred default mesh.
  *inferred_default_mesh = inferred_mesh;
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

    if (mlir::failed(PropagateMesh(producers, main_func, &builder))) {
      return signalPassFailure();
    }

    mlir::StringAttr default_mesh;
    if (mlir::failed(InferFunctionDefaultMesh(producers, main_func, &builder,
                                              &default_mesh))) {
      return signalPassFailure();
    }
    if (!default_mesh) {
      default_mesh =
          module->getAttrOfType<mlir::StringAttr>(kCustomDefaultMeshAttr);
    }

    if (default_mesh) {
      if (mlir::failed(PropagateDefaultMeshToUnAssignedClusters(
              producers, main_func, default_mesh, &builder)))
        return signalPassFailure();
    }
  }

  // Propagates and sets `_mesh` attributes to all clusters inside `function` if
  // possible.
  mlir::LogicalResult PropagateMesh(
      const llvm::DenseMap<mlir::OpOperand*, std::vector<mlir::Value>>&
          producers,
      mlir::func::FuncOp, mlir::OpBuilder* builder);

  // Infers mesh of `cluster` from its input operations.
  mlir::LogicalResult PropagateMeshFromInputs(
      const llvm::DenseMap<mlir::OpOperand*, std::vector<mlir::Value>>&
          producers,
      mlir::tf_device::ClusterOp cluster, mlir::OpBuilder* builder);

  // Infers mesh of `cluster` from its consuming operations.
  mlir::LogicalResult PropagateMeshFromConsumers(
      const llvm::DenseMap<mlir::OpOperand*, std::vector<mlir::Value>>&
          producers,
      mlir::tf_device::ClusterOp cluster, mlir::OpBuilder* builder);

  // Assigns function default mesh to clusters with no mesh specified. Note that
  // function has default mesh if all its dtensor inputs/outputs are assigned to
  // a single mesh.
  mlir::LogicalResult PropagateDefaultMeshToUnAssignedClusters(
      const llvm::DenseMap<mlir::OpOperand*, std::vector<mlir::Value>>&
          producers,
      mlir::func::FuncOp, mlir::StringAttr mesh, mlir::OpBuilder* builder);
};

mlir::LogicalResult
DTensorMeshPropagation::PropagateDefaultMeshToUnAssignedClusters(
    const llvm::DenseMap<mlir::OpOperand*, std::vector<mlir::Value>>& producers,
    mlir::func::FuncOp function, mlir::StringAttr mesh,
    mlir::OpBuilder* builder) {
  llvm::SmallVector<mlir::tf_device::ClusterOp, 4> clusters_without_mesh;
  auto walk_result = function.walk([&](mlir::tf_device::ClusterOp cluster) {
    if (llvm::isa<mlir::TF::CopyToMeshGradOp>(&cluster.GetBody().front()))
      return mlir::WalkResult::advance();

    auto mesh_or_status = ExtractDeviceMeshFromOp(cluster);
    if (!mesh_or_status.ok()) {
      cluster.GetBody().front().emitOpError(mesh_or_status.status().message());
      return mlir::WalkResult::interrupt();
    }

    const auto& mesh = mesh_or_status.value();
    if (mesh.has_value()) return mlir::WalkResult::advance();

    clusters_without_mesh.emplace_back(cluster);
    return mlir::WalkResult::advance();
  });

  if (walk_result.wasInterrupted()) return mlir::failure();

  // Set function default mesh to cluster with unspecified mesh.
  for (auto cluster_without_mesh : clusters_without_mesh) {
    cluster_without_mesh->setAttr(kMeshAttr, mesh);
  }

  return mlir::success();
}

mlir::LogicalResult DTensorMeshPropagation::PropagateMeshFromInputs(
    const llvm::DenseMap<mlir::OpOperand*, std::vector<mlir::Value>>& producers,
    mlir::tf_device::ClusterOp cluster, mlir::OpBuilder* builder) {
  // If mesh is already specified on a cluster, do nothing.
  auto cluster_mesh = cluster->getAttrOfType<mlir::StringAttr>(kMeshAttr);
  if (cluster_mesh) return mlir::success();

  // If `cluster` wraps a `tf.CopyToMesh` op, do not infer mesh from it's
  // inputs. `tf.CopyToMesh` specifies that all operations following the
  // operation is executed on target device mesh cluster specified by
  // `tf.CopyToMesh`.
  if (llvm::isa<mlir::TF::CopyToMeshOp, mlir::TF::RelayoutOp,
                mlir::TF::CopyToMeshGradOp, mlir::TF::RelayoutLikeOp>(
          &cluster.GetBody().front())) {
    return mlir::success();
  }

  // If mesh of `cluster` is not specified, infer mesh using inputs of mesh
  // cluster.
  std::optional<Mesh> extracted_mesh;
  if (failed(InferMeshFromInputs(producers, cluster, &extracted_mesh))) {
    return mlir::failure();
  }
  if (extracted_mesh.has_value()) {
    cluster->setAttr(kMeshAttr,
                     builder->getStringAttr(extracted_mesh->ToString()));
  }
  return mlir::success();
}

// Set mesh of `cluster`, inferring mesh from consumer operations of `cluster`.
mlir::LogicalResult DTensorMeshPropagation::PropagateMeshFromConsumers(
    const llvm::DenseMap<mlir::OpOperand*, std::vector<mlir::Value>>& producers,
    mlir::tf_device::ClusterOp cluster, mlir::OpBuilder* builder) {
  auto cluster_mesh = cluster->getAttrOfType<mlir::StringAttr>(kMeshAttr);
  // If mesh is already set, then do nothing.
  if (cluster_mesh) return mlir::success();

  // Infer mesh of `cluster` from its output usages.
  std::optional<Mesh> extracted_mesh_from_consumers;
  llvm::SmallVector<mlir::OpOperand*, 8> consumers_with_mesh_information;
  if (failed(InferMeshFromConsumers(cluster, &extracted_mesh_from_consumers,
                                    &consumers_with_mesh_information)))
    return mlir::failure();

  if (extracted_mesh_from_consumers && !cluster_mesh) {
    cluster->setAttr(kMeshAttr, builder->getStringAttr(
                                    extracted_mesh_from_consumers->ToString()));
  }
  return mlir::success();
}

mlir::LogicalResult PropagateLikeMesh(
    const llvm::DenseMap<mlir::OpOperand*, std::vector<mlir::Value>>& producers,
    mlir::tf_device::ClusterOp cluster, mlir::OpBuilder* builder) {
  mlir::Operation* backward_op = &cluster.GetBody().front();

  if (!mlir::isa<mlir::TF::CopyToMeshGradOp>(backward_op) &&
      !mlir::isa<mlir::TF::RelayoutLikeOp>(backward_op)) {
    // No CopyToMeshGradOp is found. Either the cluster did not have one,
    // or it has been rewritten from previous iterations.
    return mlir::success();
  }

  auto old_mesh = cluster->getAttrOfType<mlir::StringAttr>(kMeshAttr);

  std::optional<Mesh> mesh;
  mlir::OpOperand& operand = backward_op->getOpOperand(1);  // forward_input();
  // Gets mesh from the forward_input; if propagation has not reached to
  // forward_input, try again later.
  if (mlir::failed(ExtractMeshFromOperand(producers, &operand, &mesh))) {
    return mlir::success();
  }
  if (old_mesh != nullptr) {
    if (old_mesh.getValue().str() == mesh->ToString()) {
      return mlir::success();
    }
  }

  cluster->setAttr(kMeshAttr, builder->getStringAttr(mesh->ToString()));

  return mlir::success();
}

// Propagates mesh information to all `tf_device.Cluster` ops in `function`. If
// `function` includes callable ops, then recursively traverse the function
// definition to propagate mesh information using input operands and consuming
// result ops. Note that at current stage of graph optimization,
// tf_device.cluster ops are enclosing a single operation.
mlir::LogicalResult DTensorMeshPropagation::PropagateMesh(
    const llvm::DenseMap<mlir::OpOperand*, std::vector<mlir::Value>>& producers,
    mlir::func::FuncOp function, mlir::OpBuilder* builder) {
  // Iterate clusters (including nested clusters) in topological order
  // propagating mesh from operations' inputs.
  llvm::SmallVector<mlir::tf_device::ClusterOp, 8> cluster_ops;
  llvm::SmallVector<mlir::tf_device::ClusterOp, 8> while_cluster_ops;
  auto walk_result = function.walk([&](mlir::tf_device::ClusterOp cluster)
                                       -> mlir::WalkResult {
    if (llvm::isa<mlir::TF::WhileRegionOp>(cluster.GetBody().front())) {
      while_cluster_ops.emplace_back(cluster);
    } else {
      cluster_ops.emplace_back(cluster);
      if (mlir::failed(PropagateMeshFromInputs(producers, cluster, builder))) {
        return mlir::WalkResult::interrupt();
      }
    }
    return mlir::WalkResult::advance();
  });
  if (walk_result.wasInterrupted()) {
    return mlir::failure();
  }

  // Iterate clusters in reverse topological order and propagate mesh from
  // consumers.
  for (auto cluster : llvm::reverse(cluster_ops)) {
    if (mlir::failed(PropagateMeshFromConsumers(producers, cluster, builder)))
      return mlir::failure();
  }

  for (auto cluster : llvm::reverse(cluster_ops)) {
    if (mlir::failed(PropagateLikeMesh(producers, cluster, builder))) {
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
