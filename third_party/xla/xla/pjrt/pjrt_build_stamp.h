/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_PJRT_PJRT_BUILD_STAMP_H_
#define XLA_PJRT_PJRT_BUILD_STAMP_H_

#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "google/protobuf/timestamp.pb.h"
#include "absl/base/attributes.h"
#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/node_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler_variant.h"
#include "xla/pjrt/proto/pjrt_build_stamp.pb.h"
#include "xla/pjrt/proto/pjrt_value_type.pb.h"

namespace xla {

// Immutable value type representing the build provenance, binary hash,
// compile-time components, compatibility versions, and extensible attributes
// for a given PJRT plugin or compiler.
//
// `components` hold opaque source hashes of non-direct compile-time
// dependencies (e.g. FFI/custom-call handlers) folded into `full_source_hash`
// for cache invalidation, whereas `compatibility_versions` hold ordered
// `(major, minor)` numbers queried via `GetVersion()` for IR/ABI compatibility.
//
// Two PJRT plugins or compilers are equivalent if and only if they have the
// same `full_source_hash`. These two plugins/compilers have the same
// source_hash (or fallback value), same components, and same
// compatibility_versions, after deduping. They can have different build_label,
// attributes, and any lower-priority identity fields (vcs_revision or
// build_timestamp) not used as the fallback. Attributes are informational-only
// metadata and do not affect compilation or full_source_hash.
//
// The authoritative build stamp of a compiler is the one returned by
// `GetCompilerBuildStamp()`, which is produced by the factory the compiler
// registered. Consumers should read stamps from there rather than construct
// their own. A stamp built from a hand-constructed or modified proto does not
// describe any compiler. Compiler build stamp factories populate the proto
// using the helpers in `pjrt_build_stamp_writer.h`.
class PjRtBuildStamp final {
 public:
  PjRtBuildStamp(const PjRtBuildStamp&) = default;
  PjRtBuildStamp& operator=(const PjRtBuildStamp&) = default;
  PjRtBuildStamp(PjRtBuildStamp&&) = default;
  PjRtBuildStamp& operator=(PjRtBuildStamp&&) = default;

  // Validates invariants, canonicalizes components and compatibility_versions,
  // calculates a comprehensive source hash, and creates an immutable
  // PjRtBuildStamp from a proto.
  //
  // Intended for compiler build stamp factories and for deserializing a stamp
  // produced by one (e.g. across the PJRT C API), not for constructing or
  // modifying stamps elsewhere.
  //
  // Returns `kInvalidArgument` if `proto` violates an invariant.
  static absl::StatusOr<PjRtBuildStamp> FromProto(PjRtBuildStampProto proto);

  // Returns a validated proto.
  const PjRtBuildStampProto& ToProto() const ABSL_ATTRIBUTE_LIFETIME_BOUND;

  absl::string_view build_label() const ABSL_ATTRIBUTE_LIFETIME_BOUND;

  absl::Time build_timestamp() const;

  absl::string_view vcs_revision() const ABSL_ATTRIBUTE_LIFETIME_BOUND;

  absl::string_view source_hash() const ABSL_ATTRIBUTE_LIFETIME_BOUND;

  // Returns a deterministic 128-bit lowercase hex fingerprint combining
  // `source_hash` (or a fallback from `vcs_revision` or `build_timestamp`) with
  // sorted, deduplicated `components` and `compatibility_versions`. Does not
  // include `attributes`.
  absl::string_view full_source_hash() const ABSL_ATTRIBUTE_LIFETIME_BOUND;

  // Returns the compatibility version for the given version name if present.
  // For well-known versions, prefer the typed getters in
  // `pjrt_compatibility_versions.h` (e.g. `GetMosaicVersion()`). Returns
  // `kNotFound` if no version has the given name.
  absl::StatusOr<PjRtCompatibilityVersionProto> GetVersion(
      absl::string_view name) const;

  // Returns the value of the attribute with the given key if it exists and is
  // of type `T`. Returns `kNotFound` if the attribute is not set, and
  // `kInvalidArgument` if it has a different type.
  template <typename T>
  absl::StatusOr<T> GetAttribute(absl::string_view key) const {
    auto it = proto_.attributes().find(key);
    if (it == proto_.attributes().end()) {
      return absl::NotFoundError(
          absl::StrCat("Attribute '", key, "' not found."));
    }
    PjRtValueType val = PjRtValueTypeFromProto(it->second);
    if (T* typed_val = std::get_if<T>(&val)) {
      return std::move(*typed_val);
    }
    return absl::InvalidArgumentError(
        absl::StrCat("Attribute '", key, "' has type ",
                     PjRtValueTypeCaseName(it->second.value_case()),
                     ", not the requested type \"", PjRtTypeName<T>(), "\"."));
  }

  // Zero-copy lookup for a string attribute. Returns the same errors as
  // `GetAttribute<std::string>()`.
  absl::StatusOr<absl::string_view> GetStringAttribute(
      absl::string_view key) const ABSL_ATTRIBUTE_LIFETIME_BOUND;

 private:
  explicit PjRtBuildStamp(PjRtBuildStampProto proto,
                          std::string full_source_hash);

  // All state is derived from `proto_` in `FromProto()`.
  //
  // Not `const`: the class is already immutable (no mutators). `const` members
  // would make PjRtBuildStamp non-assignable (so it couldn't be assigned into
  // an existing `absl::StatusOr<PjRtBuildStamp>`) and would turn moves into
  // copies.
  PjRtBuildStampProto proto_;
  std::string full_source_hash_;
};

// Each of the following validators returns `kInvalidArgument` if validation
// fails.

// Validates that `vcs_revision` is non-empty and not an unknown-revision
// sentinel (`"<unknown>"` or `"unknown"`).
absl::Status ValidateVcsRevision(absl::string_view vcs_revision);

// Validates that `timestamp` has `nanos` in `[0, 999'999'999]` and `seconds` in
// `[1, 253'402'300'799]`. The upper bound (9999-12-31T23:59:59Z) matches
// `google.protobuf.Timestamp`; the lower bound is stricter (strictly after the
// Unix epoch) because unstamped builds report `seconds == 0`.
absl::Status ValidateBuildTimestamp(
    const google::protobuf::Timestamp& timestamp);

// Validates that `version` has a non-empty name and non-negative major and
// minor version numbers.
absl::Status ValidateCompatibilityVersion(
    const PjRtCompatibilityVersionProto& version);

// Validates that `key` is non-empty and `value` has a value set.
absl::Status ValidateAttribute(absl::string_view key,
                               const PjRtValueTypeProto& value);

// A thread-safe registry of build stamp factories, keyed by platform and
// compiler variant.
//
// Keyed by both `platform_id` and `variant` because different variants for the
// same platform (e.g. an in-process compiler vs. a prebuilt or remote
// compilation service) can execute compilers at different revisions and, thus,
// have different build stamps.
//
// The registry is append-only: a key can be registered at most once, and
// registrations are never removed or replaced. A factory runs on the first
// call to `GetBuildStamp()` for its key, and its stamp is memoized for the
// lifetime of the registry. Errors are not memoized, so a factory that fails
// (e.g. on a transient RPC error) runs again on the next lookup.
//
// Registration always names an explicit `PjRtCompilerVariant`. Lookups without
// a `variant` resolve the platform's active compiler variant through
// `PjRtCompilerVariantPickerRegistry` (defaulting to
// `LinkedCompilerVariantId()` if no picker is registered), matching the variant
// selection of `PjRtCompile()`.
class PjRtBuildStampRegistry {
 public:
  // Produces the build stamp of one compiler. A factory does not run at
  // registration; it runs lazily, on the thread of the first `GetBuildStamp()`
  // for its key, while holding a lock on that key only (so it may be slow, and
  // may do I/O or RPCs, without blocking other keys). It runs again on each
  // lookup until it succeeds once, so its failure path should be cheap. It must
  // not look up its own key, directly or transitively.
  using Factory = absl::AnyInvocable<absl::StatusOr<PjRtBuildStamp>()>;

  explicit PjRtBuildStampRegistry(
      const PjRtCompilerVariantPickerRegistry& variant_pickers
          ABSL_ATTRIBUTE_LIFETIME_BOUND =
              PjRtCompilerVariantPickerRegistry::Global())
      : variant_pickers_(variant_pickers) {}

  // Not copyable or movable.
  PjRtBuildStampRegistry(const PjRtBuildStampRegistry&) = delete;
  PjRtBuildStampRegistry& operator=(const PjRtBuildStampRegistry&) = delete;

  // Returns the process-wide registry.
  static PjRtBuildStampRegistry& Global();

  // Registers `factory` for `(platform_id, variant)`. Returns
  // `kInvalidArgument` if `factory` is null, or `kAlreadyExists` if a factory
  // is already registered for the key.
  absl::Status RegisterBuildStampFactory(PjRtPlatformId platform_id,
                                         PjRtCompilerVariant variant,
                                         Factory factory);

  // Returns the build stamp for the active compiler variant of `platform_id`
  // (selected via `variant_pickers_`). Runs the factory on each lookup until
  // one succeeds, then returns the memoized stamp. Concurrent lookups of a key
  // wait for a running factory rather than running it again. Returns
  // `kNotFound` if no factory is registered for the active key, or the error
  // from the variant picker or factory. The returned stamp is valid for the
  // lifetime of the registry.
  //
  // A factory must not look up its own key, directly or transitively; doing so
  // deadlocks.
  absl::StatusOr<const PjRtBuildStamp* absl_nonnull> GetBuildStamp(
      PjRtPlatformId platform_id) const ABSL_ATTRIBUTE_LIFETIME_BOUND;

 private:
  absl::StatusOr<const PjRtBuildStamp* absl_nonnull> GetBuildStamp(
      PjRtPlatformId platform_id,
      PjRtCompilerVariant variant) const ABSL_ATTRIBUTE_LIFETIME_BOUND;
  struct Key {
    PjRtPlatformId platform_id;
    PjRtCompilerVariant variant;

    template <typename H>
    friend H AbslHashValue(H h, const Key& key) {
      return H::combine(std::move(h), key.platform_id, key.variant);
    }

    friend bool operator==(const Key& a, const Key& b) {
      return a.platform_id == b.platform_id && a.variant == b.variant;
    }
  };

  struct Entry {
    explicit Entry(Factory factory) : factory(std::move(factory)) {}

    // Held while the factory runs, so that it runs at most once at a time.
    absl::Mutex mu;
    // Reset once `stamp` is set, to release any state the factory captured.
    Factory factory ABSL_GUARDED_BY(mu);
    // Set by the first successful run of `factory`, and never changed after.
    std::optional<PjRtBuildStamp> stamp ABSL_GUARDED_BY(mu);
  };

  const PjRtCompilerVariantPickerRegistry& variant_pickers_;

  mutable absl::Mutex mu_;
  // `node_hash_map` for pointer stability: `Entry` is not movable (because of
  // `absl::Mutex`), `GetBuildStamp()` uses `Entry*` outside of `mu_`, and
  // returns pointers into `Entry::stamp` that must survive rehashes.
  mutable absl::node_hash_map<Key, Entry> entries_ ABSL_GUARDED_BY(mu_);
};

// Registers `factory` in `registry`. CHECK-fails if `factory` is null or a
// factory is already registered for the key. Intended to be called from static
// initializers or module initializers.
void RegisterBuildStampFactory(
    PjRtPlatformId platform_id, PjRtCompilerVariant variant,
    PjRtBuildStampRegistry::Factory factory,
    PjRtBuildStampRegistry& registry = PjRtBuildStampRegistry::Global());

// Returns the build stamp for the active compiler variant of `platform_id`.
//
// Callers that only need the full hash of the compiler (e.g. for a compilation
// cache key) can call `GetCompilerFullSourceHash()` instead.
//
// Use `GetCompilerBuildStamp()` to get compatibility versions, build
// provenance, etc.
absl::StatusOr<const PjRtBuildStamp* absl_nonnull> GetCompilerBuildStamp(
    PjRtPlatformId platform_id,
    const PjRtBuildStampRegistry& registry ABSL_ATTRIBUTE_LIFETIME_BOUND =
        PjRtBuildStampRegistry::Global());

// Preferred entrypoint for callers needing a stable compiler version
// identifier (e.g. for compilation cache keys). Returns the
// `full_source_hash()` of `GetCompilerBuildStamp(platform_id, registry)` (for
// the platform's active compiler variant), with the same errors.
absl::StatusOr<absl::string_view> GetCompilerFullSourceHash(
    PjRtPlatformId platform_id,
    const PjRtBuildStampRegistry& registry ABSL_ATTRIBUTE_LIFETIME_BOUND =
        PjRtBuildStampRegistry::Global());

}  // namespace xla

#endif  // XLA_PJRT_PJRT_BUILD_STAMP_H_
