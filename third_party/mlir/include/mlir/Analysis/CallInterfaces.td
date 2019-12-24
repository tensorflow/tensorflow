//===- CallInterfaces.td - Call Interfaces for ops -*- tablegen ---------*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a set of interfaces that can be used to define information
// related to call-like and callable operations. Each of which are defined along
// with the respective interface below.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CALLINTERFACES
#define MLIR_CALLINTERFACES

include "mlir/IR/OpBase.td"

// `CallInterfaceCallable`: This is a type used to represent a single callable
// region. A callable is either a symbol, or an SSA value, that is referenced by
// a call-like operation. This represents the destination of the call.

/// Interface for call-like operations.
def CallOpInterface : OpInterface<"CallOpInterface"> {
  let description = [{
    A call-like operation is one that transfers control from one sub-routine to
    another. These operations may be traditional direct calls `call @foo`, or
    indirect calls to other operations `call_indirect %foo`. An operation that
    uses this interface, must *not* also provide the `CallableOpInterface`.
  }];

  let methods = [
    InterfaceMethod<[{
        Returns the callee of this call-like operation. A `callee` is either a
        reference to a symbol, via SymbolRefAttr, or a reference to a defined
        SSA value.
      }],
      "CallInterfaceCallable", "getCallableForCallee"
    >,
    InterfaceMethod<[{
        Returns the operands within this call that are used as arguments to the
        callee.
      }],
      "Operation::operand_range", "getArgOperands"
    >,
  ];
}

/// Interface for callable operations.
def CallableOpInterface : OpInterface<"CallableOpInterface"> {
  let description = [{
    A callable operation is one who represents a potential sub-routine, and may
    be a target for a call-like operation (those providing the CallOpInterface
    above). These operations may be traditional functional operation
    `func @foo(...)`, as well as function producing operations
    `%foo = dialect.create_function(...)`. These operations may produce multiple
    callable regions, or subroutines.
  }];

  let methods = [
    InterfaceMethod<[{
        Returns a region on the current operation that the given callable refers
        to. This may return null in the case of an external callable object,
        e.g. an external function.
      }],
      "Region *", "getCallableRegion", (ins "CallInterfaceCallable":$callable)
    >,
    InterfaceMethod<[{
        Returns all of the callable regions of this operation.
      }],
      "void", "getCallableRegions",
              (ins "SmallVectorImpl<Region *> &":$callables)
    >,
    InterfaceMethod<[{
        Returns the results types that the given callable region produces when
        executed.
      }],
      "ArrayRef<Type>", "getCallableResults", (ins "Region *":$callable)
    >,
  ];
}

#endif // MLIR_CALLINTERFACES
