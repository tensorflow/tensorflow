/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_CORE_API_OP_RESOLVER_H_
#define TENSORFLOW_LITE_CORE_API_OP_RESOLVER_H_

#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/compiler/mlir/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

#ifndef DOXYGEN_SKIP
class OpResolverInternal;  // For friend declaration below.
class Subgraph;            // For friend declaration below.

namespace internal {
class CommonOpaqueConversionUtil;  // For friend declaration below.
class OperatorsCache;              // Forward decl.
}  // namespace internal
#endif

/// Abstract interface that returns TfLiteRegistrations given op codes or custom
/// op names. This is the mechanism that ops being referenced in the flatbuffer
/// model are mapped to executable function pointers (TfLiteRegistrations).
///
/// The lifetime of the TfLiteRegistration object whose address is
/// returned by FindOp must exceed the lifetime of any InterpreterBuilder or
/// Interpreter created with this OpResolver.
/// Likewise the lifetime of the TfLiteOperator object referenced
/// from the TfLiteRegistration object, if any, must exceed the lifetime of
/// any InterpreterBuilder or Interpreter created with this OpResolver.
class OpResolver {
 public:
  /// Finds the op registration for a builtin operator by enum code.
  virtual const TfLiteRegistration* FindOp(tflite::BuiltinOperator op,
                                           int version) const = 0;
  /// Finds the op registration of a custom operator by op name.
  virtual const TfLiteRegistration* FindOp(const char* op,
                                           int version) const = 0;

  // Represents a sequence of delegates.
  using TfLiteDelegatePtrVector =
      std::vector<std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>>;

  // Returns optional delegates for resolving and handling ops in the flatbuffer
  // model. This may be used in addition to the standard TfLiteRegistration
  // lookup for graph resolution.
  // WARNING: This API is deprecated, GetDelegateCreators is preferred.
  virtual TfLiteDelegatePtrVector GetDelegates(int num_threads) const {
    return {};
  }

  // Represents a function that creates a TfLite delegate instance.
  using TfLiteDelegateCreator =
      std::function<std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
          TfLiteContext* /*context*/)>;

  // Represents a sequence of delegate creator functions.
  using TfLiteDelegateCreators = std::vector<TfLiteDelegateCreator>;

  // Returns a vector of delegate creators to create optional delegates for
  // resolving and handling ops in the flatbuffer model. This may be used in
  // addition to the standard TfLiteRegistration lookup for graph resolution.
  //
  // Note that this method is not used (will not be called) if you are using
  // TF Lite in Google Play Services; the GetOpaqueDelegateCreators method
  // (see below) is used for that case.
  virtual TfLiteDelegateCreators GetDelegateCreators() const { return {}; }

  // TODO(b/202712825): it would be nice if we could avoid the need for separate
  // "opaque" types & methods for use only with TF Lite in Google Play Services.

  // Represents an opaque delegate instance.
  // WARNING: Experimental interface, subject to change.
  using TfLiteOpaqueDelegatePtr =
      std::unique_ptr<TfLiteOpaqueDelegate, void (*)(TfLiteOpaqueDelegate*)>;

  // Represents a function that creates an opaque delegate instance.
  // WARNING: Experimental interface, subject to change.
  using TfLiteOpaqueDelegateCreator =
      std::function<TfLiteOpaqueDelegatePtr(int /*num_threads*/)>;

  // Represents a sequence of opaque delegate creator functions.
  // WARNING: Experimental interface, subject to change.
  using TfLiteOpaqueDelegateCreators = std::vector<TfLiteOpaqueDelegateCreator>;

  // Returns a vector of opaque delegate creators to create optional opaque
  // delegates for resolving and handling ops in the flatbuffer model. This may
  // be used in addition to the standard TfLiteRegistration lookup for graph
  // resolution.
  //
  // Note that this method will be called only if you are using TF Lite in
  // Google Play Services; if you are using regular TF Lite, GetDelegateCreators
  // (see above) is used instead.
  //
  // WARNING: Experimental interface, subject to change.
  virtual TfLiteOpaqueDelegateCreators GetOpaqueDelegateCreators() const {
    return {};
  }

  virtual ~OpResolver() = default;
  OpResolver() = default;
  OpResolver(const OpResolver& other) = default;

 private:
  /// Returns true if this OpResolver may contain any "user defined" ops.
  /// By "user defined" ops, we mean any op definitions other than those
  /// contained in tflite::ops::builtin::BuiltinOpResolver.
  ///
  /// If this method returns true, it doesn't necessarily mean that the
  /// OpResolver contains a user-defined op, just that the absence of
  /// user-defined ops can't be guaranteed.
  ///
  /// Note that "user-defined" ops are not the same as "custom" ops;
  /// BuiltinOpResolver may support certain "custom" ops, in addition to
  /// "builtin" ops, and may not support all of the "builtin" op enum values.
  virtual bool MayContainUserDefinedOps() const { return true; }

#ifndef DOXYGEN_SKIP
  friend class OpResolverInternal;
  friend class Subgraph;  // For OpId.
  friend class tflite::internal::CommonOpaqueConversionUtil;
  friend class tflite::internal::OperatorsCache;
#endif

  // This holds the identity of an operator.
  // Ths is used as the key for the OperatorsCache below.
  struct OpId {
    int builtin_code;
    const char* custom_name;
    int version;
    bool operator==(const OpId& other) const {
      return builtin_code == other.builtin_code &&
             custom_name == other.custom_name && version == other.version;
    }
    struct Hasher {
      size_t operator()(const OpId& op_id) const {
        size_t hash_builtin_code = std::hash<int>()(op_id.builtin_code);
        size_t hash_custom_name =
            op_id.custom_name != nullptr
                ? std::hash<std::string>()(std::string(op_id.custom_name))
                : 0;
        size_t hash_version = std::hash<int>()(op_id.version);
        return Combine(hash_builtin_code,
                       Combine(hash_custom_name, hash_version));
      }

     private:
      static size_t Combine(size_t hash1, size_t hash2) {
        constexpr int num_bits_to_rotate_left = 21;
        constexpr int num_bits_to_rotate_right =
            std::numeric_limits<size_t>::digits - num_bits_to_rotate_left;
        size_t hash1_rotated = (hash1 << num_bits_to_rotate_left) |
                               (hash1 >> num_bits_to_rotate_right);
        return hash1_rotated + hash2;
      }
    };
  };

  // A set of 'TfLiteOperator' objects whose lifetimes need to
  // last at least as long as the lifetime of the OpResolver.
  // We use shared_ptr rather than unique_ptr here, to allow the
  // OperatorsCache to be shared with other classes such as the
  // InterpreterBuilder and Interpreter. This is so that the
  // TfLiteOperator objects allocated by an OpResolver,
  // which may be referenced by a Subgraph in an Interpreter, can remain live
  // even if the OpResolver is destroyed, while also allowing the same
  // OpResolver to be used with multiple InterpreterBuilders and multiple
  // Interpreters.
  mutable std::shared_ptr<internal::OperatorsCache>
      registration_externals_cache_;
};

#ifndef DOXYGEN_SKIP
// Type for a set of owned 'TfLiteOperator' objects.
// This is needed when converting TfLiteRegistration to
// TfLiteOperator, to ensure that the number of
// TfLiteOperator objects that we allocate is bounded, and to
// ensure that those objects get deallocated at the appropriate time.
// We use a public class rather than a typedef or using declaration here,
// to ensure that the class can be forward-declared.
// WARNING: Experimental interface, subject to change.
namespace internal {
class OperatorsCache
    : private std::unordered_map<OpResolver::OpId,
                                 std::unique_ptr<TfLiteOperator>,
                                 OpResolver::OpId::Hasher> {
  friend class ::tflite::Subgraph;
  friend class ::tflite::internal::CommonOpaqueConversionUtil;
};
}  // namespace internal
#endif

// Handles the logic for converting between an OperatorCode structure extracted
// from a flatbuffer and information about a registered operator
// implementation.
TfLiteStatus GetRegistrationFromOpCode(const OperatorCode* opcode,
                                       const OpResolver& op_resolver,
                                       ErrorReporter* error_reporter,
                                       const TfLiteRegistration** registration);

}  // namespace tflite

#endif  // TENSORFLOW_LITE_CORE_API_OP_RESOLVER_H_
