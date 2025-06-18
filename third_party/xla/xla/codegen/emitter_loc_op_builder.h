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

#ifndef XLA_CODEGEN_EMITTER_LOC_OP_BUILDER_H_
#define XLA_CODEGEN_EMITTER_LOC_OP_BUILDER_H_

#include <string>

#include "absl/strings/string_view.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "tsl/platform/platform.h"

#if defined(PLATFORM_GOOGLE)
// The source_location.h is not available in open source.
#include "absl/types/source_location.h"
#endif

namespace xla {

// The builder that could add the NameLoc attribute to the newly created
// operations and fills this attribute with the SourceLocation(file:line) of the
// create<OpTy>(...) calls. The location info will be added to the current_loc_
// location that the builder got through the constructor. The copy constructor
// also remembers the source location where the copy was created.
//
// Why: it is useful for tracking up the emitter file and line from the
// generated MLIR.
//
// How:
// 1. create<OpTy>(...) functions have absl::SourceLocation as the last
// argument with the default value of SourceLocation::current(). Every time they
// construct a new NameLoc attribute that contains the string from the
// current_loc_ and file:line from the source location parameter.
//
// 2. The copy constructor also gets the source location as the argument and
// remembers it in the current_loc_ as a join of the original current_loc_ and
// the place where the copy was created.
class EmitterLocOpBuilder : public mlir::ImplicitLocOpBuilder {
 public:
  // TODO(b/382419919): Remove ifdefs once we have absl::SourceLocation in absl
  // OSS builds.
#if defined(PLATFORM_GOOGLE)
  using SourceLocation = absl::SourceLocation;
  constexpr static bool kSourceLocationSupported = true;
#else
  // Mimicking absl::SourceLocation and doing nothing.
  class FakeSourceLocation {
   public:
    static FakeSourceLocation current() { return FakeSourceLocation(); }
    absl::string_view file_name() const { return ""; }
    int line() const { return 0; }
  };
  using SourceLocation = FakeSourceLocation;
  constexpr static bool kSourceLocationSupported = false;
#endif

  // Constructor that takes the op builder and a flag indicating whether to
  // annotate the location of the operations.
  EmitterLocOpBuilder(mlir::ImplicitLocOpBuilder& op_builder, bool annotate_loc)
      : mlir::ImplicitLocOpBuilder(op_builder),
        annotate_loc_(annotate_loc),
        current_loc_(op_builder.getLoc()) {}

  // A few constructors below that could be used when we replace the
  // mlir::ImplicitLocOpBuilder and mlir::OpBuilder one by one.
  // The intent is to use EmitterLocOpBuilder everywhere in the emitters.

  // The constructor that should be used instead of mlir::ImplicitLocOpBuilder.
  EmitterLocOpBuilder(mlir::Location loc, mlir::OpBuilder& op_builder,
                      bool annotate_loc = false)
      : mlir::ImplicitLocOpBuilder(loc, op_builder),

        annotate_loc_(annotate_loc),
        current_loc_(loc) {}

  // The constructor that should be used instead of mlir::ImplicitLocOpBuilder.
  EmitterLocOpBuilder(mlir::Location loc, mlir::MLIRContext* mlir_context,
                      bool annotate_loc = false)
      : mlir::ImplicitLocOpBuilder(loc, mlir_context),
        annotate_loc_(annotate_loc),
        current_loc_(loc) {}

  EmitterLocOpBuilder& operator=(const EmitterLocOpBuilder&) = delete;

  // Copy constructor that also remembers the source location where the copy
  // was created. If the helper functions that gets the builder as the argument
  // receives the argument by value then the current location points to the
  // place where the copy was created.
  EmitterLocOpBuilder(const EmitterLocOpBuilder& builder,
                      SourceLocation location = SourceLocation::current())
      : mlir::ImplicitLocOpBuilder(builder),
        annotate_loc_(builder.annotate_loc_),
        current_loc_(builder.Loc(location)) {}

  // Formats the MLIR IR with annotations to make it easier to read.
  static std::string FormatTritonIrWithAnnotations(absl::string_view mlir_ir);

  // Below is the set of create() methods that are used to create operations.
  // These are all templated to allow for the creation of operations with
  // different numbers of arguments.
  //
  // For some reason the version of create that accepts the variadic arguments
  // and a source location with the default value does not work.

  template <typename OpTy>
  OpTy create(SourceLocation location = SourceLocation::current()) {
    return OpBuilder::create<OpTy>(Loc(location));
  }

  // Creates an operation with the given type and one argument.
  template <typename OpTy, typename Arg0>
  OpTy create(Arg0&& arg, SourceLocation location = SourceLocation::current()) {
    return OpBuilder::create<OpTy>(Loc(location), std::forward<Arg0>(arg));
  }

  template <typename OpTy, typename Arg0, typename Arg1>
  OpTy create(Arg0&& arg0, Arg1&& arg1,
              SourceLocation location = SourceLocation::current()) {
    return OpBuilder::create<OpTy>(Loc(location), std::forward<Arg0>(arg0),
                                   std::forward<Arg1>(arg1));
  }

  template <typename OpTy, typename Arg0, typename Arg1, typename Arg2>
  OpTy create(Arg0&& arg0, Arg1&& arg1, Arg2&& arg2,
              SourceLocation location = SourceLocation::current()) {
    return OpBuilder::create<OpTy>(Loc(location), std::forward<Arg0>(arg0),
                                   std::forward<Arg1>(arg1),
                                   std::forward<Arg2>(arg2));
  }

  template <typename OpTy, typename Arg0, typename Arg1, typename Arg2,
            typename Arg3>
  OpTy create(Arg0&& arg0, Arg1&& arg1, Arg2&& arg2, Arg3&& arg3,
              SourceLocation location = SourceLocation::current()) {
    return OpBuilder::create<OpTy>(
        Loc(location), std::forward<Arg0>(arg0), std::forward<Arg1>(arg1),
        std::forward<Arg2>(arg2), std::forward<Arg3>(arg3));
  }

  template <typename OpTy, typename Arg0, typename Arg1, typename Arg2,
            typename Arg3, typename Arg4>
  OpTy create(Arg0&& arg0, Arg1&& arg1, Arg2&& arg2, Arg3&& arg3, Arg4&& arg4,
              SourceLocation location = SourceLocation::current()) {
    return OpBuilder::create<OpTy>(
        Loc(location), std::forward<Arg0>(arg0), std::forward<Arg1>(arg1),
        std::forward<Arg2>(arg2), std::forward<Arg3>(arg3),
        std::forward<Arg4>(arg4));
  }

  template <typename OpTy, typename Arg0, typename Arg1, typename Arg2,
            typename Arg3, typename Arg4, typename Arg5>
  OpTy create(Arg0&& arg0, Arg1&& arg1, Arg2&& arg2, Arg3&& arg3, Arg4&& arg4,
              Arg5&& arg5,
              SourceLocation location = SourceLocation::current()) {
    return OpBuilder::create<OpTy>(
        Loc(location), std::forward<Arg0>(arg0), std::forward<Arg1>(arg1),
        std::forward<Arg2>(arg2), std::forward<Arg3>(arg3),
        std::forward<Arg4>(arg4), std::forward<Arg5>(arg5));
  }

  template <typename OpTy, typename Arg0, typename Arg1, typename Arg2,
            typename Arg3, typename Arg4, typename Arg5, typename Arg6>
  OpTy create(Arg0&& arg0, Arg1&& arg1, Arg2&& arg2, Arg3&& arg3, Arg4&& arg4,
              Arg5&& arg5, Arg6&& arg6,
              SourceLocation location = SourceLocation::current()) {
    return OpBuilder::create<OpTy>(
        Loc(location), std::forward<Arg0>(arg0), std::forward<Arg1>(arg1),
        std::forward<Arg2>(arg2), std::forward<Arg3>(arg3),
        std::forward<Arg4>(arg4), std::forward<Arg5>(arg5),
        std::forward<Arg6>(arg6));
  }

  mlir::Location current_loc() const { return current_loc_; }

  bool annotate_loc() const { return annotate_loc_; }

 private:
  // Helper function to create a location from a source location.
  mlir::Location Loc(SourceLocation location) const;

  // Keep the current location of the builder and use it for annotating the
  // newly created operations.
  const bool annotate_loc_;
  const mlir::Location current_loc_;
};

}  // namespace xla

#endif  // XLA_CODEGEN_EMITTER_LOC_OP_BUILDER_H_
