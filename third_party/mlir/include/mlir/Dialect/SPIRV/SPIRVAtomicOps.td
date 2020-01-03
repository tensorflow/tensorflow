//===-- SPIRVAtomicOps.td - MLIR SPIR-V Atomic Ops ---------*- tablegen -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains atomic ops for the SPIR-V dialect. It corresponds to
// "3.32.18. Atomic Instructions" of the SPIR-V specification.
//
//===----------------------------------------------------------------------===//

#ifndef SPIRV_ATOMIC_OPS
#define SPIRV_ATOMIC_OPS

class SPV_AtomicUpdateOp<string mnemonic, list<OpTrait> traits = []> :
  SPV_Op<mnemonic, traits> {
  let parser = [{ return ::parseAtomicUpdateOp(parser, result, false); }];
  let printer = [{ return ::printAtomicUpdateOp(getOperation(), p); }];
  let verifier = [{ return ::verifyAtomicUpdateOp(getOperation()); }];

  let arguments = (ins
    SPV_AnyPtr:$pointer,
    SPV_ScopeAttr:$memory_scope,
    SPV_MemorySemanticsAttr:$semantics
  );
  let results = (outs
    SPV_Integer:$result
  );
}

class SPV_AtomicUpdateWithValueOp<string mnemonic, list<OpTrait> traits = []> :
  SPV_Op<mnemonic, traits> {
  let parser = [{ return ::parseAtomicUpdateOp(parser, result, true); }];
  let printer = [{ return ::printAtomicUpdateOp(getOperation(), p); }];
  let verifier = [{ return ::verifyAtomicUpdateOp(getOperation()); }];

  let arguments = (ins
    SPV_AnyPtr:$pointer,
    SPV_ScopeAttr:$memory_scope,
    SPV_MemorySemanticsAttr:$semantics,
    SPV_Integer:$value
  );
  let results = (outs
    SPV_Integer:$result
  );
}

// -----

def SPV_AtomicAndOp : SPV_AtomicUpdateWithValueOp<"AtomicAnd", []> {
  let summary = [{
    Perform the following steps atomically with respect to any other atomic
    accesses within Scope to the same location:
  }];

  let description = [{
    1) load through Pointer to get an Original Value,

    2) get a New Value by the bitwise AND of Original Value and Value, and

    3) store the New Value back through Pointer.

    The instruction’s result is the Original Value.

    Result Type must be an integer type scalar.

     The type of Value must be the same as Result Type.  The type of the
    value pointed to by Pointer must be the same as Result Type.

    Memory must be a valid memory Scope.

    ### Custom assembly form

    ```
    scope ::= `"CrossDevice"` | `"Device"` | `"Workgroup"` | ...

    memory-semantics ::= `"None"` | `"Acquire"` | "Release"` | ...

    atomic-and-op ::=
        `spv.AtomicAnd` scope memory-semantics
                        ssa-use `,` ssa-use `:` spv-pointer-type
    ```

    For example:

    ```
    %0 = spv.AtomicAnd "Device" "None" %pointer, %value :
                       !spv.ptr<i32, StorageBuffer>
    ```
  }];
}

// -----

def SPV_AtomicCompareExchangeWeakOp : SPV_Op<"AtomicCompareExchangeWeak", []> {
  let summary = "Deprecated (use OpAtomicCompareExchange).";

  let description = [{
    Has the same semantics as OpAtomicCompareExchange.

    Memory must be a valid memory Scope.

    ### Custom assembly form

    ```
    atomic-compare-exchange-weak-op ::=
        `spv.AtomicCompareExchangeWeak` scope memory-semantics memory-semantics
                                        ssa-use `,` ssa-use `,` ssa-use
                                        `:` spv-pointer-type
    ```

    For example:

    ```
    %0 = spv.AtomicCompareExchangeWeak "Workgroup" "Acquire" "None"
                                       %pointer, %value, %comparator
                                       : !spv.ptr<i32, WorkGroup>
    ```
  }];

  let arguments = (ins
    SPV_AnyPtr:$pointer,
    SPV_ScopeAttr:$memory_scope,
    SPV_MemorySemanticsAttr:$equal_semantics,
    SPV_MemorySemanticsAttr:$unequal_semantics,
    SPV_Integer:$value,
    SPV_Integer:$comparator
  );

  let results = (outs
    SPV_Integer:$result
  );
}

// -----

def SPV_AtomicIAddOp : SPV_AtomicUpdateWithValueOp<"AtomicIAdd", []> {
  let summary = [{
    Perform the following steps atomically with respect to any other atomic
    accesses within Scope to the same location:
  }];

  let description = [{
    1) load through Pointer to get an Original Value,

    2) get a New Value by integer addition of Original Value and Value, and

    3) store the New Value back through Pointer.

    The instruction’s result is the Original Value.

    Result Type must be an integer type scalar.

     The type of Value must be the same as Result Type.  The type of the
    value pointed to by Pointer must be the same as Result Type.

    Memory must be a valid memory Scope.

    ### Custom assembly form

    ```
    atomic-iadd-op ::=
        `spv.AtomicIAdd` scope memory-semantics
                         ssa-use `,` ssa-use `:` spv-pointer-type
    ```

    For example:

    ```
    %0 = spv.AtomicIAdd "Device" "None" %pointer, %value :
                        !spv.ptr<i32, StorageBuffer>
    ```
  }];
}

// -----

def SPV_AtomicIDecrementOp : SPV_AtomicUpdateOp<"AtomicIDecrement", []> {
  let summary = [{
    Perform the following steps atomically with respect to any other atomic
    accesses within Scope to the same location:
  }];

  let description = [{
    1) load through Pointer to get an Original Value,

    2) get a New Value through integer subtraction of 1 from Original Value,
    and

    3) store the New Value back through Pointer.

    The instruction’s result is the Original Value.

    Result Type must be an integer type scalar.  The type of the value
    pointed to by Pointer must be the same as Result Type.

    Memory must be a valid memory Scope.

    ### Custom assembly form

    ```
    atomic-idecrement-op ::=
        `spv.AtomicIDecrement` scope memory-semantics ssa-use
                               `:` spv-pointer-type
    ```

    For example:

    ```
    %0 = spv.AtomicIDecrement "Device" "None" %pointer :
                              !spv.ptr<i32, StorageBuffer>
    ```
  }];
}

// -----

def SPV_AtomicIIncrementOp : SPV_AtomicUpdateOp<"AtomicIIncrement", []> {
  let summary = [{
    Perform the following steps atomically with respect to any other atomic
    accesses within Scope to the same location:
  }];

  let description = [{
    1) load through Pointer to get an Original Value,

    2) get a New Value through integer addition of 1 to Original Value, and

    3) store the New Value back through Pointer.

    The instruction’s result is the Original Value.

    Result Type must be an integer type scalar.  The type of the value
    pointed to by Pointer must be the same as Result Type.

    Memory must be a valid memory Scope.

    ### Custom assembly form

    ```
    atomic-iincrement-op ::=
        `spv.AtomicIIncrement` scope memory-semantics ssa-use
                               `:` spv-pointer-type
    ```

    For example:

    ```
    %0 = spv.AtomicIncrement "Device" "None" %pointer :
                             !spv.ptr<i32, StorageBuffer>
    ```
  }];
}

// -----

def SPV_AtomicISubOp : SPV_AtomicUpdateWithValueOp<"AtomicISub", []> {
  let summary = [{
    Perform the following steps atomically with respect to any other atomic
    accesses within Scope to the same location:
  }];

  let description = [{
    1) load through Pointer to get an Original Value,

    2) get a New Value by integer subtraction of Value from Original Value,
    and

    3) store the New Value back through Pointer.

    The instruction’s result is the Original Value.

    Result Type must be an integer type scalar.

     The type of Value must be the same as Result Type.  The type of the
    value pointed to by Pointer must be the same as Result Type.

    Memory must be a valid memory Scope.

    ### Custom assembly form

    ```
    atomic-isub-op ::=
        `spv.AtomicISub` scope memory-semantics
                         ssa-use `,` ssa-use `:` spv-pointer-type
    ```

    For example:

    ```
    %0 = spv.AtomicISub "Device" "None" %pointer, %value :
                        !spv.ptr<i32, StorageBuffer>
    ```
  }];
}

// -----

def SPV_AtomicOrOp : SPV_AtomicUpdateWithValueOp<"AtomicOr", []> {
  let summary = [{
    Perform the following steps atomically with respect to any other atomic
    accesses within Scope to the same location:
  }];

  let description = [{
    1) load through Pointer to get an Original Value,

    2) get a New Value by the bitwise OR of Original Value and Value, and

    3) store the New Value back through Pointer.

    The instruction’s result is the Original Value.

    Result Type must be an integer type scalar.

     The type of Value must be the same as Result Type.  The type of the
    value pointed to by Pointer must be the same as Result Type.

    Memory must be a valid memory Scope.

    ### Custom assembly form

    ```
    atomic-or-op ::=
        `spv.AtomicOr` scope memory-semantics
                       ssa-use `,` ssa-use `:` spv-pointer-type
    ```

    For example:

    ```
    %0 = spv.AtomicOr "Device" "None" %pointer, %value :
                      !spv.ptr<i32, StorageBuffer>
    ```
  }];
}

// -----

def SPV_AtomicSMaxOp : SPV_AtomicUpdateWithValueOp<"AtomicSMax", []> {
  let summary = [{
    Perform the following steps atomically with respect to any other atomic
    accesses within Scope to the same location:
  }];

  let description = [{
    1) load through Pointer to get an Original Value,

    2) get a New Value by finding the largest signed integer of Original
    Value and Value, and

    3) store the New Value back through Pointer.

    The instruction’s result is the Original Value.

    Result Type must be an integer type scalar.

     The type of Value must be the same as Result Type.  The type of the
    value pointed to by Pointer must be the same as Result Type.

    Memory must be a valid memory Scope.

    ### Custom assembly form

    ```
    atomic-smax-op ::=
        `spv.AtomicSMax` scope memory-semantics
                         ssa-use `,` ssa-use `:` spv-pointer-type
    ```

    For example:

    ```
    %0 = spv.AtomicSMax "Device" "None" %pointer, %value :
                        !spv.ptr<i32, StorageBuffer>
    ```
  }];
}

// -----

def SPV_AtomicSMinOp : SPV_AtomicUpdateWithValueOp<"AtomicSMin", []> {
  let summary = [{
    Perform the following steps atomically with respect to any other atomic
    accesses within Scope to the same location:
  }];

  let description = [{
    1) load through Pointer to get an Original Value,

    2) get a New Value by finding the smallest signed integer of Original
    Value and Value, and

    3) store the New Value back through Pointer.

    The instruction’s result is the Original Value.

    Result Type must be an integer type scalar.

     The type of Value must be the same as Result Type.  The type of the
    value pointed to by Pointer must be the same as Result Type.

    Memory must be a valid memory Scope.

    ### Custom assembly form

    ```
    atomic-smin-op ::=
        `spv.AtomicSMin` scope memory-semantics
                         ssa-use `,` ssa-use `:` spv-pointer-type
    ```

    For example:

    ```
    %0 = spv.AtomicSMin "Device" "None" %pointer, %value :
                        !spv.ptr<i32, StorageBuffer>
    ```
  }];
}

// -----

def SPV_AtomicUMaxOp : SPV_AtomicUpdateWithValueOp<"AtomicUMax", []> {
  let summary = [{
    Perform the following steps atomically with respect to any other atomic
    accesses within Scope to the same location:
  }];

  let description = [{
    1) load through Pointer to get an Original Value,

    2) get a New Value by finding the largest unsigned integer of Original
    Value and Value, and

    3) store the New Value back through Pointer.

    The instruction’s result is the Original Value.

    Result Type must be an integer type scalar.

     The type of Value must be the same as Result Type.  The type of the
    value pointed to by Pointer must be the same as Result Type.

    Memory must be a valid memory Scope.

    ### Custom assembly form

    ```
    atomic-umax-op ::=
        `spv.AtomicUMax` scope memory-semantics
                         ssa-use `,` ssa-use `:` spv-pointer-type
    ```

    For example:

    ```
    %0 = spv.AtomicUMax "Device" "None" %pointer, %value :
                        !spv.ptr<i32, StorageBuffer>
    ```
  }];
}

// -----

def SPV_AtomicUMinOp : SPV_AtomicUpdateWithValueOp<"AtomicUMin", []> {
  let summary = [{
    Perform the following steps atomically with respect to any other atomic
    accesses within Scope to the same location:
  }];

  let description = [{
    1) load through Pointer to get an Original Value,

    2) get a New Value by finding the smallest unsigned integer of Original
    Value and Value, and

    3) store the New Value back through Pointer.

    The instruction’s result is the Original Value.

    Result Type must be an integer type scalar.

     The type of Value must be the same as Result Type.  The type of the
    value pointed to by Pointer must be the same as Result Type.

    Memory must be a valid memory Scope.

    ### Custom assembly form

    ```
    atomic-umin-op ::=
        `spv.AtomicUMin` scope memory-semantics
                         ssa-use `,` ssa-use `:` spv-pointer-type
    ```

    For example:

    ```
    %0 = spv.AtomicUMin "Device" "None" %pointer, %value :
                        !spv.ptr<i32, StorageBuffer>
    ```
  }];
}

// -----

def SPV_AtomicXorOp : SPV_AtomicUpdateWithValueOp<"AtomicXor", []> {
  let summary = [{
    Perform the following steps atomically with respect to any other atomic
    accesses within Scope to the same location:
  }];

  let description = [{
    1) load through Pointer to get an Original Value,

    2) get a New Value by the bitwise exclusive OR of Original Value and
    Value, and

    3) store the New Value back through Pointer.

    The instruction’s result is the Original Value.

    Result Type must be an integer type scalar.

     The type of Value must be the same as Result Type.  The type of the
    value pointed to by Pointer must be the same as Result Type.

    Memory must be a valid memory Scope.

    ### Custom assembly form

    ```
    atomic-xor-op ::=
        `spv.AtomicXor` scope memory-semantics
                        ssa-use `,` ssa-use `:` spv-pointer-type
    ```

    For example:

    ```
    %0 = spv.AtomicXor "Device" "None" %pointer, %value :
                       !spv.ptr<i32, StorageBuffer>
    ```
  }];
}

// -----

#endif // SPIRV_ATOMIC_OPS
