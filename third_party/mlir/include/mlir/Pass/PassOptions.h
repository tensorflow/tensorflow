//===- PassOptions.h - Pass Option Utilities --------------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains utilities for registering options with compiler passes and
// pipelines.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PASS_PASSOPTIONS_H_
#define MLIR_PASS_PASSOPTIONS_H_

#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include <memory>

namespace mlir {
namespace detail {
/// Base class for PassOptions<T> that holds all of the non-CRTP features.
class PassOptionsBase : protected llvm::cl::SubCommand {
public:
  /// This class represents a specific pass option, with a provided data type.
  template <typename DataType> struct Option : public llvm::cl::opt<DataType> {
    template <typename... Args>
    Option(PassOptionsBase &parent, StringRef arg, Args &&... args)
        : llvm::cl::opt<DataType>(arg, llvm::cl::sub(parent),
                                  std::forward<Args>(args)...) {
      assert(!this->isPositional() && !this->isSink() &&
             "sink and positional options are not supported");
    }
  };

  /// This class represents a specific pass option that contains a list of
  /// values of the provided data type.
  template <typename DataType> struct List : public llvm::cl::list<DataType> {
    template <typename... Args>
    List(PassOptionsBase &parent, StringRef arg, Args &&... args)
        : llvm::cl::list<DataType>(arg, llvm::cl::sub(parent),
                                   std::forward<Args>(args)...) {
      assert(!this->isPositional() && !this->isSink() &&
             "sink and positional options are not supported");
    }
  };

  /// Parse options out as key=value pairs that can then be handed off to the
  /// `llvm::cl` command line passing infrastructure. Everything is space
  /// separated.
  LogicalResult parseFromString(StringRef options);
};
} // end namespace detail

/// Subclasses of PassOptions provide a set of options that can be used to
/// initialize a pass instance. See PassRegistration for usage details.
///
/// Usage:
///
/// struct MyPassOptions : PassOptions<MyPassOptions> {
///   List<int> someListFlag{
///        *this, "flag-name", llvm::cl::MiscFlags::CommaSeparated,
///        llvm::cl::desc("...")};
/// };
template <typename T> class PassOptions : public detail::PassOptionsBase {
public:
  /// Factory that parses the provided options and returns a unique_ptr to the
  /// struct.
  static std::unique_ptr<T> createFromString(StringRef options) {
    auto result = std::make_unique<T>();
    if (failed(result->parseFromString(options)))
      return nullptr;
    return result;
  }
};

/// A default empty option struct to be used for passes that do not need to take
/// any options.
struct EmptyPassOptions : public PassOptions<EmptyPassOptions> {};

} // end namespace mlir

#endif // MLIR_PASS_PASSOPTIONS_H_
