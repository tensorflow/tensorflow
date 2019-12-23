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
/// Base container class and manager for all pass options.
class PassOptions : protected llvm::cl::SubCommand {
private:
  /// This is the type-erased option base class. This provides some additional
  /// hooks into the options that are not available via llvm::cl::Option.
  class OptionBase {
  public:
    virtual ~OptionBase() = default;

    /// Out of line virtual function to provide home for the class.
    virtual void anchor();

    /// Print the name and value of this option to the given stream.
    virtual void print(raw_ostream &os) = 0;

    /// Return the argument string of this option.
    StringRef getArgStr() const { return getOption()->ArgStr; }

  protected:
    /// Return the main option instance.
    virtual const llvm::cl::Option *getOption() const = 0;

    /// Copy the value from the given option into this one.
    virtual void copyValueFrom(const OptionBase &other) = 0;

    /// Allow access to private methods.
    friend PassOptions;
  };

  /// This is the parser that is used by pass options that use literal options.
  /// This is a thin wrapper around the llvm::cl::parser, that exposes some
  /// additional methods.
  template <typename DataType>
  struct GenericOptionParser : public llvm::cl::parser<DataType> {
    using llvm::cl::parser<DataType>::parser;

    /// Returns an argument name that maps to the specified value.
    Optional<StringRef> findArgStrForValue(const DataType &value) {
      for (auto &it : this->Values)
        if (it.V.compare(value))
          return it.Name;
      return llvm::None;
    }
  };

  /// The specific parser to use depending on llvm::cl parser used. This is only
  /// necessary because we need to provide additional methods for certain data
  /// type parsers.
  /// TODO(riverriddle) We should upstream the methods in GenericOptionParser to
  /// avoid the need to do this.
  template <typename DataType>
  using OptionParser =
      std::conditional_t<std::is_base_of<llvm::cl::generic_parser_base,
                                         llvm::cl::parser<DataType>>::value,
                         GenericOptionParser<DataType>,
                         llvm::cl::parser<DataType>>;

  /// Utility methods for printing option values.
  template <typename DataT>
  static void printOptionValue(raw_ostream &os,
                               GenericOptionParser<DataT> &parser,
                               const DataT &value) {
    if (Optional<StringRef> argStr = parser.findArgStrForValue(value))
      os << argStr;
    else
      llvm_unreachable("unknown data value for option");
  }
  template <typename DataT, typename ParserT>
  static void printOptionValue(raw_ostream &os, ParserT &parser,
                               const DataT &value) {
    os << value;
  }
  template <typename ParserT>
  static void printOptionValue(raw_ostream &os, ParserT &parser,
                               const bool &value) {
    os << (value ? StringRef("true") : StringRef("false"));
  }

public:
  /// This class represents a specific pass option, with a provided data type.
  template <typename DataType>
  class Option : public llvm::cl::opt<DataType, /*ExternalStorage=*/false,
                                      OptionParser<DataType>>,
                 public OptionBase {
  public:
    template <typename... Args>
    Option(PassOptions &parent, StringRef arg, Args &&... args)
        : llvm::cl::opt<DataType, /*ExternalStorage=*/false,
                        OptionParser<DataType>>(arg, llvm::cl::sub(parent),
                                                std::forward<Args>(args)...) {
      assert(!this->isPositional() && !this->isSink() &&
             "sink and positional options are not supported");
      parent.options.push_back(this);
    }
    using llvm::cl::opt<DataType, /*ExternalStorage=*/false,
                        OptionParser<DataType>>::operator=;
    ~Option() override = default;

  private:
    /// Return the main option instance.
    const llvm::cl::Option *getOption() const final { return this; }

    /// Print the name and value of this option to the given stream.
    void print(raw_ostream &os) final {
      os << this->ArgStr << '=';
      printOptionValue(os, this->getParser(), this->getValue());
    }

    /// Copy the value from the given option into this one.
    void copyValueFrom(const OptionBase &other) final {
      this->setValue(static_cast<const Option<DataType> &>(other).getValue());
    }
  };

  /// This class represents a specific pass option that contains a list of
  /// values of the provided data type.
  template <typename DataType>
  class ListOption : public llvm::cl::list<DataType, /*StorageClass=*/bool,
                                           OptionParser<DataType>>,
                     public OptionBase {
  public:
    template <typename... Args>
    ListOption(PassOptions &parent, StringRef arg, Args &&... args)
        : llvm::cl::list<DataType, /*StorageClass=*/bool,
                         OptionParser<DataType>>(arg, llvm::cl::sub(parent),
                                                 std::forward<Args>(args)...) {
      assert(!this->isPositional() && !this->isSink() &&
             "sink and positional options are not supported");
      parent.options.push_back(this);
    }
    ~ListOption() override = default;

    /// Allow assigning from an ArrayRef.
    ListOption<DataType> &operator=(ArrayRef<DataType> values) {
      (*this)->assign(values.begin(), values.end());
      return *this;
    }

    std::vector<DataType> *operator->() { return &*this; }

  private:
    /// Return the main option instance.
    const llvm::cl::Option *getOption() const final { return this; }

    /// Print the name and value of this option to the given stream.
    void print(raw_ostream &os) final {
      os << this->ArgStr << '=';
      auto printElementFn = [&](const DataType &value) {
        printOptionValue(os, this->getParser(), value);
      };
      interleave(*this, os, printElementFn, ",");
    }

    /// Copy the value from the given option into this one.
    void copyValueFrom(const OptionBase &other) final {
      (*this) = ArrayRef<DataType>((ListOption<DataType> &)other);
    }
  };

  PassOptions() = default;

  /// Copy the option values from 'other' into 'this', where 'other' has the
  /// same options as 'this'.
  void copyOptionValuesFrom(const PassOptions &other);

  /// Parse options out as key=value pairs that can then be handed off to the
  /// `llvm::cl` command line passing infrastructure. Everything is space
  /// separated.
  LogicalResult parseFromString(StringRef options);

  /// Print the options held by this struct in a form that can be parsed via
  /// 'parseFromString'.
  void print(raw_ostream &os);

private:
  /// A list of all of the opaque options.
  std::vector<OptionBase *> options;
};
} // end namespace detail

//===----------------------------------------------------------------------===//
// PassPipelineOptions
//===----------------------------------------------------------------------===//

/// Subclasses of PassPipelineOptions provide a set of options that can be used
/// to initialize a pass pipeline. See PassPipelineRegistration for usage
/// details.
///
/// Usage:
///
/// struct MyPipelineOptions : PassPipelineOptions<MyPassOptions> {
///   ListOption<int> someListFlag{
///        *this, "flag-name", llvm::cl::MiscFlags::CommaSeparated,
///        llvm::cl::desc("...")};
/// };
template <typename T> class PassPipelineOptions : public detail::PassOptions {
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
struct EmptyPipelineOptions : public PassPipelineOptions<EmptyPipelineOptions> {
};

} // end namespace mlir

#endif // MLIR_PASS_PASSOPTIONS_H_
