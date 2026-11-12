/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_TFRT_IFRT_IFRT_PROGRAM_H_
#define TENSORFLOW_CORE_TFRT_IFRT_IFRT_PROGRAM_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/ifrt_types.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/tf2hlo.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/layout.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/topology.h"
#include "xla/shape.h"
#include "xla/tsl/concurrency/future.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_persistent_compilation_cache.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_serving_core_selector.h"
#include "tensorflow/core/tfrt/ifrt/tf_host_callback.h"
#include "tsl/platform/protobuf.h"

namespace tensorflow {
namespace ifrt_serving {

// Marker prepended to the error returned when a new executable would have to be
// compiled while XLA compilation is disabled (see
// `TfToHloCompiler::IsXlaCompilationDisabled()`). It matches the marker emitted
// by the non-IFRT TPU runtimes so that model servers can uniformly detect
// compilation cache misses that happen while compilation is disabled (e.g. to
// count them, or to retry with compilation enabled in dry-run mode).
inline constexpr absl::string_view kXlaCompilationDisabledErrorMarker =
    "[TpuCompilationDisabled]";

// Encodes the layout of `xla_input_shapes` to the module.
absl::Status EncodeLayout(absl::Span<const xla::Shape> xla_input_shapes,
                          mlir::ModuleOp module);

// Returns a mask over the executable's `num_inputs` inputs that is true when
// the compiled program may donate that input, derived from the compiled HLO's
// input/output alias config and buffer-donor config. This is the
// post-compilation ground truth for both TF (tf.aliasing_output) and JAX
// (jax.buffer_donor) frontends, unlike
// LoadedExecutable::GetDonatableInputIndices which only reads the JAX MLIR
// attribute. Exposed for testing.
absl::StatusOr<std::vector<bool>> GetInputDonationMask(
    const xla::ifrt::LoadedExecutable& ifrt_executable, int num_inputs);

// A TPU program outlined from a model: what to run. It owns the program's MLIR
// module and compiles it into executables, one per set of input shapes.
//
// An `IfrtProgram` is independent of the variables its call sites bind: a
// variable argument is only a parameter of the compiled executable. Each call
// site is served by an `IfrtServingExecutable`, which binds the call site's
// variables to the executables compiled here.
//
// This class is thread safe.
class IfrtProgram {
 public:
  // Everything compilation produces for one set of input shapes. Immutable
  // once built.
  struct CompiledExecutable {
    xla::ifrt::LoadedExecutableRef ifrt_executable;
    tensorflow::tpu::TPUCompileMetadataProto compile_metadata;
    std::vector<std::unique_ptr<TfHostCallback>> host_callbacks;

    // Per-argument information. The index `i` in these vectors corresponds to
    // the i-th argument in the executable.
    std::vector<xla::ifrt::DType> ifrt_input_dtypes;
    // If populated, these are the input shapes and layouts that the
    // executable was compiled with. `xla_input_shapes` and `xla_input_layouts`
    // are either both populated or both empty and they will have the same size.
    std::vector<std::shared_ptr<const xla::Shape>> xla_input_shapes;
    std::vector<absl::InlinedVector<int64_t, 4>> byte_strides;
    std::vector<std::shared_ptr<const xla::ifrt::Shape>> ifrt_input_shapes;
    std::vector<xla::ifrt::LayoutRef> xla_input_layouts;
    // Input tensor shapes that matches the Tf2Hlo compiled shapes.
    std::vector<tensorflow::TensorShape> reshaped_input_tensors;
    // True for inputs the compiled program may donate (input/output aliasing
    // or buffer donors). Empty when donation info could not be determined.
    // Callers must treat a missing entry as "may be donated" and leave the
    // input's buffers donatable.
    std::vector<bool> input_may_be_donated;
    std::vector<xla::HloSharding> arg_hlo_shardings;
    std::vector<xla::ifrt::ShardingRef> arg_ifrt_shardings;
    // Only populated when portable execution is used, currently only single
    // device sharding is supported.
    absl::flat_hash_map<xla::ifrt::DeviceId,
                        std::shared_ptr<xla::ifrt::SingleDeviceSharding>>
        portable_single_device_shardings;

    std::vector<xla::HloSharding> retval_hlo_shardings;

    CompiledExecutable() = default;
    // Move only
    CompiledExecutable(CompiledExecutable&& other) = default;
    CompiledExecutable& operator=(CompiledExecutable&& other) = default;
    CompiledExecutable(const CompiledExecutable& other) = delete;
    CompiledExecutable& operator=(const CompiledExecutable& other) = delete;
  };

  // Cache key: the input shapes an executable is compiled for.
  struct Key {
    std::vector<tensorflow::TensorShape> input_shapes;
    template <typename H>
    friend H AbslHashValue(H h, const Key& key) {
      for (const auto& shape : key.input_shapes) {
        for (auto size : shape.dim_sizes()) {
          h = H::combine(std::move(h), size);
        }
      }
      return h;
    }

    friend bool operator==(const Key& x, const Key& y) {
      return x.input_shapes == y.input_shapes;
    }
  };

  // A view of the key. This is used to avoid copying the input shapes and
  // dtypes and shapes when looking up the cache.
  struct KeyView {
    absl::Span<const DtypeAndShape> dtypes_and_shapes;

    template <typename H>
    friend H AbslHashValue(H h, const KeyView& key) {
      for (const auto& dtype_and_shape : key.dtypes_and_shapes) {
        for (auto size : dtype_and_shape.GetShapeForCompilation().dim_sizes()) {
          h = H::combine(std::move(h), size);
        }
      }
      return h;
    }
  };

  // Hash function for the key.
  struct KeyHash {
    using is_transparent = void;

    size_t operator()(const Key& key) const { return absl::Hash<Key>()(key); }
    size_t operator()(const KeyView& key) const {
      return absl::Hash<KeyView>()(key);
    }
  };

  // Equality function for the key.
  struct KeyEq {
    using is_transparent = void;

    bool operator()(const Key& lhs, const Key& rhs) const { return lhs == rhs; }
    bool operator()(const Key& lhs, const KeyView& rhs) const {
      if (lhs.input_shapes.size() != rhs.dtypes_and_shapes.size()) {
        return false;
      }
      for (int i = 0; i < lhs.input_shapes.size(); ++i) {
        if (lhs.input_shapes[i] !=
            rhs.dtypes_and_shapes[i].GetShapeForCompilation()) {
          return false;
        }
      }
      return true;
    }
    bool operator()(const KeyView& lhs, const Key& rhs) const {
      return this->operator()(rhs, lhs);
    }
  };

  // Creates a program from `module`, whose entry function is `main`.
  // `signature_name` is only used for logging and dumps.
  static absl::StatusOr<std::unique_ptr<IfrtProgram>> Create(
      absl::string_view model_name, absl::string_view signature_name,
      mlir::OwningOpRef<mlir::ModuleOp> module,
      std::shared_ptr<xla::ifrt::Client> client,
      tensorflow::DeviceMgr* device_mgr,
      tensorflow::XlaHelpers::ShapeRepresentationFn shape_representation_fn,
      IfrtServingCoreSelector* ifrt_serving_core_selector,
      std::variant<tsl::protobuf::Message*,
                   xla::CompileOptions::EnvironmentOptionOverrides>
          compilation_env_or_overrides,
      TfToHloCompiler* tf_to_hlo_compiler,
      IfrtPersistentCompilationCache* persistent_compilation_cache,
      bool use_undonatable_buffer_converter = false);

  IfrtProgram(const IfrtProgram&) = delete;
  IfrtProgram& operator=(const IfrtProgram&) = delete;

  absl::string_view model_name() const { return model_name_; }
  absl::string_view signature_name() const { return signature_name_; }

  // Returns the executable compiled for `dtypes_and_shapes`, compiling it if
  // there is none. This is the only place that decides whether compilation is
  // allowed: returns an error if there is none and the program is frozen or
  // XLA compilation is disabled. The error is not cached, so a later call can
  // compile once allowed. A failed compilation is cached.
  absl::StatusOr<tsl::Future<std::shared_ptr<const CompiledExecutable>>>
  GetOrCompile(absl::Span<const DtypeAndShape> dtypes_and_shapes,
               absl::Span<const int> variable_arg_indices);

  // Freezes the program: releases the module and rejects compiling for new
  // input shapes from now on.
  void Freeze();

  // Returns the number of input shape sets compiled (or failed to compile) so
  // far.
  int num_compiled() const {
    absl::MutexLock lock(compile_mutex_);
    return compiled_executables_.size();
  }

  const tensorflow::tpu::TPUCompileMetadataProto& original_compile_metadata()
      const {
    return original_compile_metadata_;
  }
  const xla::ifrt::DeviceListRef& assigned_device_list() const {
    return assigned_device_list_;
  }
  const absl::flat_hash_map<size_t /*original_arg_idx*/,
                            size_t /*static_shape_arg_idx*/>&
  static_shape_arg_map() const {
    return static_shape_arg_map_;
  }

  // Whether the program is executed on a device selected at runtime (single
  // device program with core selection) rather than on its assigned devices.
  bool UsePortableExecution() const;

 private:
  friend class IfrtBackendCompilerTest;

  IfrtProgram(
      absl::string_view model_name, absl::string_view signature_name,
      mlir::OwningOpRef<mlir::ModuleOp> module,
      std::shared_ptr<xla::ifrt::Client> client,
      tensorflow::DeviceMgr* device_mgr,
      tensorflow::XlaHelpers::ShapeRepresentationFn shape_representation_fn,
      IfrtServingCoreSelector* ifrt_serving_core_selector,
      tensorflow::tpu::TPUCompileMetadataProto original_compile_metadata,
      xla::ifrt::DeviceListRef assigned_device_list,
      absl::flat_hash_map<size_t, size_t> static_shape_arg_map,
      std::variant<tsl::protobuf::Message*,
                   xla::CompileOptions::EnvironmentOptionOverrides>
          compilation_env_or_overrides,
      TfToHloCompiler* tf_to_hlo_compiler,
      IfrtPersistentCompilationCache* persistent_compilation_cache,
      bool use_undonatable_buffer_converter)
      : model_name_(std::string(model_name)),
        signature_name_(std::string(signature_name)),
        module_(std::move(module)),
        original_compile_metadata_(std::move(original_compile_metadata)),
        assigned_device_list_(std::move(assigned_device_list)),
        static_shape_arg_map_(std::move(static_shape_arg_map)),
        ifrt_client_(std::move(client)),
        device_mgr_(device_mgr),
        shape_representation_fn_(std::move(shape_representation_fn)),
        ifrt_serving_core_selector_(ifrt_serving_core_selector),
        compilation_env_or_overrides_(compilation_env_or_overrides),
        tf_to_hlo_compiler_(tf_to_hlo_compiler),
        persistent_compilation_cache_(persistent_compilation_cache),
        use_undonatable_buffer_converter_(use_undonatable_buffer_converter) {}

  // Creates an executable by calling tf2xla and xla compiler
  absl::StatusOr<std::shared_ptr<const CompiledExecutable>>
  CreateExecutableSynchronously(
      mlir::OwningOpRef<mlir::ModuleOp> module_copy,
      const tensorflow::tpu::TPUCompileMetadataProto& compile_metadata,
      absl::Span<const DtypeAndShape> dtypes_and_shapes,
      absl::Span<const int> variable_arg_indices);

  absl::Status PopulateInvariantMetadata(
      const Tf2HloResult& tf2hlo_result,
      xla::ifrt::LoadedExecutableRef ifrt_executable,
      std::vector<std::unique_ptr<TfHostCallback>> host_callbacks,
      const xla::ifrt::Topology* topology, CompiledExecutable& compiled);

  std::string model_name_;
  std::string signature_name_;

  mlir::OwningOpRef<mlir::ModuleOp> module_ ABSL_GUARDED_BY(compile_mutex_);
  // The original compile metadata. We need to keep it around to be able to
  // test portable execution condition even if the Module itself is already
  // released.
  tensorflow::tpu::TPUCompileMetadataProto original_compile_metadata_;
  const xla::ifrt::DeviceListRef assigned_device_list_;
  absl::flat_hash_map<size_t /*original_arg_idx*/,
                      size_t /*static_shape_arg_idx*/>
      static_shape_arg_map_;

  std::shared_ptr<xla::ifrt::Client> ifrt_client_;
  tensorflow::DeviceMgr* device_mgr_;  // Not owned. For host callback.
  tensorflow::XlaHelpers::ShapeRepresentationFn shape_representation_fn_;
  IfrtServingCoreSelector* ifrt_serving_core_selector_;  // May be nullptr.

  std::variant<tsl::protobuf::Message*,
               xla::CompileOptions::EnvironmentOptionOverrides>
      compilation_env_or_overrides_;  // proto is NOT OWNED. can be nullptr.

  // The tf_to_hlo_compiler_ is not owned by this program. It is expected to
  // be alive during the lifetime of the program.
  TfToHloCompiler* tf_to_hlo_compiler_;

  // The persistent compilation cache is a global cache and is not owned by
  // this program. When it is nullptr, the persistent compilation cache is
  // disabled at ifrt serving level.
  IfrtPersistentCompilationCache* persistent_compilation_cache_;

  bool use_undonatable_buffer_converter_ = false;

  // Executables compiled from `module_`, keyed by input shapes. Guards
  // compilation and freeze.
  mutable absl::Mutex compile_mutex_;
  absl::flat_hash_map<Key,
                      tsl::Future<std::shared_ptr<const CompiledExecutable>>,
                      KeyHash, KeyEq>
      compiled_executables_ ABSL_GUARDED_BY(compile_mutex_);
  bool is_frozen_ ABSL_GUARDED_BY(compile_mutex_) = false;
};

}  // namespace ifrt_serving
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_IFRT_IFRT_PROGRAM_H_
