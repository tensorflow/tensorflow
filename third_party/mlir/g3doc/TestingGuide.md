# Testing Guide

Testing is an integral part of any software infrastructure. In general, all
commits to the MLIR repository should include an accompanying test of some form.
Commits that include no functional changes, such as API changes like symbol
renaming, should be tagged with NFC(no functional changes). This signals to the
reviewer why the change doesn't/shouldn't include a test.

MLIR generally separates testing into two main categories, [Check](#check-tests)
tests and [Unit](#unit-tests) tests.

## Check tests

Check tests are tests that verify that some set of string tags appear in the
output of some program. These tests generally encompass anything related to the
state of the IR (and more); analysis, parsing, transformation, verification,
etc. They are written utilizing several different tools:

### FileCheck tests

[FileCheck](https://llvm.org/docs/CommandGuide/FileCheck.html) is a utility tool
that "reads two files (one from standard input, and one specified on the command
line) and uses one to verify the other." Essentially, one file contains a set of
tags that are expected to appear in the output file. MLIR utilizes FileCheck, in
combination with [lit](https://llvm.org/docs/CommandGuide/lit.html), to verify
different aspects of the IR - such as the output of a transformation pass.

An example FileCheck test is shown below:

```mlir {.mlir}
// RUN: mlir-opt %s -cse | FileCheck %s

// CHECK-LABEL: func @simple_constant
func @simple_constant() -> (i32, i32) {
  // CHECK-NEXT: %[[RESULT:.*]] = constant 1
  // CHECK-NEXT: return %[[RESULT]], %[[RESULT]]

  %0 = constant 1 : i32
  %1 = constant 1 : i32
  return %0, %1 : i32, i32
}
```

The above test performs a check that after running Common Sub-Expression
elimination, only one constant remains in the IR.

#### FileCheck best practices

FileCheck is an extremely useful utility, it allows for easily matching various
parts of the output. This ease of use means that it becomes easy to write
brittle tests that are essentially `diff` tests. FileCheck tests should be as
self-contained as possible and focus on testing the minimal set of
functionalities needed. Let's see an example:

```mlir {.mlir}
// RUN: mlir-opt %s -cse | FileCheck %s

// CHECK-LABEL: func @simple_constant() -> (i32, i32)
func @simple_constant() -> (i32, i32) {
  // CHECK-NEXT: %result = constant 1 : i32
  // CHECK-NEXT: return %result, %result : i32, i32
  // CHECK-NEXT: }

  %0 = constant 1 : i32
  %1 = constant 1 : i32
  return %0, %1 : i32, i32
}
```

The above example is another way to write the original example shown in the main
[FileCheck tests](#filecheck-tests) section. There are a few problems with this
test; below is a breakdown of the no-nos of this test to specifically highlight
best practices.

*   Tests should be self-contained.

This means that tests should not test lines or sections outside of what is
intended. In the above example, we see lines such as `CHECK-NEXT: }`. This line
in particular is testing pieces of the Parser/Printer of FuncOp, which is
outside of the realm of concern for the CSE pass. This line should be removed.

*   Tests should be minimal, and only check what is absolutely necessary.

This means that anything in the output that is not core to the functionality
that you are testing should *not* be present in a CHECK line. This is a separate
bullet just to highlight the importance of it, especially when checking against
IR output.

If we naively remove the unrelated `CHECK` lines in our source file, we may end
up with:

```mlir {.mlir}
// CHECK-LABEL: func @simple_constant
func @simple_constant() -> (i32, i32) {
  // CHECK-NEXT: %result = constant 1 : i32
  // CHECK-NEXT: return %result, %result : i32, i32

  %0 = constant 1 : i32
  %1 = constant 1 : i32
  return %0, %1 : i32, i32
}
```

It may seem like this is a minimal test case, but it still checks several
aspects of the output that are unrelated to the CSE transformation. Namely the
result types of the `constant` and `return` operations, as well the actual SSA
value names that are produced. FileCheck `CHECK` lines may contain
[regex statements](https://llvm.org/docs/CommandGuide/FileCheck.html#filecheck-regex-matching-syntax)
as well as named
[string substitution blocks](https://llvm.org/docs/CommandGuide/FileCheck.html#filecheck-string-substitution-blocks).
Utilizing the above, we end up with the example shown in the main
[FileCheck tests](#filecheck-tests) section.

```mlir {.mlir}
// CHECK-LABEL: func @simple_constant
func @simple_constant() -> (i32, i32) {
  /// Here we use a substitution variable as the output of the constant is
  /// useful for the test, but we omit as much as possible of everything else.
  // CHECK-NEXT: %[[RESULT:.*]] = constant 1
  // CHECK-NEXT: return %[[RESULT]], %[[RESULT]]

  %0 = constant 1 : i32
  %1 = constant 1 : i32
  return %0, %1 : i32, i32
}
```

### Diagnostic verification tests

MLIR provides rich source location tracking that can be used to emit errors,
warnings, etc. easily from anywhere throughout the codebase. Certain classes of
tests are written to check that certain diagnostics are emitted for a given
input program, such as an MLIR file. These tests are useful in that they allow
checking specific invariants of the IR without transforming or changing
anything. Some examples of tests in this category are: those that verify
invariants of operations, or check the expected results of an analysis.
Diagnostic verification tests are written utilizing the
[source manager verifier handler](Diagnostics.md#sourcemgr-diagnostic-verifier-handler),
accessible via the `verify-diagnostics` flag in mlir-opt.

An example .mlir test running under `mlir-opt` is shown below:

```mlir {.mlir}
// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// Expect an error on the same line.
func @bad_branch() {
  br ^missing  // expected-error {{reference to an undefined block}}
}

// -----

// Expect an error on an adjacent line.
func @foo(%a : f32) {
  // expected-error@+1 {{unknown comparison predicate "foo"}}
  %result = cmpf "foo", %a, %a : f32
  return
}
```

## Unit tests

Unit tests are written using
[Google Test](https://github.com/google/googletest/blob/master/googletest/docs/primer.md)
and are located in the unittests/ directory. Tests of these form *should* be
limited to API tests that cannot be reasonably written as [Check](#check-tests)
tests, e.g. those for data structures. It is important to keep in mind that the
C++ APIs are not stable, and evolve over time. As such, directly testing the C++
IR interfaces makes the tests more fragile as those C++ APIs evolve over time.
This makes future API refactorings, which may happen frequently, much more
cumbersome as the number of tests scale.
