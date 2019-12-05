# Introduction and Usage Guide to MLIR's Diagnostics Infrastructure

[TOC]

This document presents an introduction to using and interfacing with MLIR's
diagnostics infrastructure.

See [MLIR specification](LangRef.md) for more information about MLIR, the
structure of the IR, operations, etc.

## Source Locations

Source location information is extremely important for any compiler, because it
provides a baseline for debuggability and error-reporting. MLIR provides several
different location types depending on the situational need.

### CallSite Location

``` {.ebnf}
callsite-location ::= 'callsite' '(' location 'at' location ')'
```

An instance of this location allows for representing a directed stack of
location usages. This connects a location of a `callee` with the location of a
`caller`.

### FileLineCol Location

``` {.ebnf}
filelinecol-location ::= string-literal ':' integer-literal ':' integer-literal
```

An instance of this location represents a tuple of file, line number, and column
number. This is similar to the type of location that you get from most source
languages.

### Fused Location

``` {.ebnf}
fused-location ::= `fused` fusion-metadata? '[' location (location ',')* ']'
fusion-metadata ::= '<' attribute-value '>'
```

An instance of a `fused` location represents a grouping of several other source
locations, with optional metadata that describes the context of the fusion.
There are many places within a compiler in which several constructs may be fused
together, e.g. pattern rewriting, that normally result partial or even total
loss of location information. With `fused` locations, this is a non-issue.

### Name Location

``` {.ebnf}
name-location ::= string-literal ('(' location ')')?
```

An instance of this location allows for attaching a name to a child location.
This can be useful for representing the locations of variable, or node,
definitions.

### Opaque Location

An instance of this location essentially contains a pointer to some data
structure that is external to MLIR and an optional location that can be used if
the first one is not suitable. Since it contains an external structure, only the
optional location is used during serialization.

### Unknown Location

``` {.ebnf}
unknown-location ::= `unknown`
```

Source location information is an extremely integral part of the MLIR
infrastructure. As such, location information is always present in the IR, and
must explicitly be set to unknown. Thus an instance of the `unknown` location,
represents an unspecified source location.

## Diagnostic Engine

The `DiagnosticEngine` acts as the main interface for diagnostics in MLIR. It
manages the registration of diagnostic handlers, as well as the core API for
diagnostic emission. Handlers generally take the form of
`LogicalResult(Diagnostic &)`. If the result is `success`, it signals that the
diagnostic has been fully processed and consumed. If `failure`, it signals that
the diagnostic should be propagated to any previously registered handlers. It
can be interfaced with via an `MLIRContext` instance.

```c++
DiagnosticEngine engine = ctx->getDiagEngine();

/// Handle the reported diagnostic.
// Return success to signal that the diagnostic has either been fully processed,
// or failure if the diagnostic should be propagated to the previous handlers.
DiagnosticEngine::HandlerID id = engine.registerHandler(
    [](Diagnostic &diag) -> LogicalResult {
  bool should_propage_diagnostic = ...;
  return failure(should_propage_diagnostic);
});


// We can also elide the return value completely, in which the engine assumes
// that all diagnostics are consumed(i.e. a success() result).
DiagnosticEngine::HandlerID id = engine.registerHandler([](Diagnostic &diag) {
  return;
});

// Unregister this handler when we are done.
engine.eraseHandler(id);
```

### Constructing a Diagnostic

As stated above, the `DiagnosticEngine` holds the core API for diagnostic
emission. A new diagnostic can be emitted with the engine via `emit`. This
method returns an [InFlightDiagnostic](#inflight-diagnostic) that can be
modified further.

```c++
InFlightDiagnostic emit(Location loc, DiagnosticSeverity severity);
```

Using the `DiagnosticEngine`, though, is generally not the preferred way to emit
diagnostics in MLIR. [`operation`](LangRef.md#operations) provides utility
methods for emitting diagnostics:

```c++
// `emit` methods available in the mlir namespace.
InFlightDiagnostic emitError/Remark/Warning(Location);

// These methods use the location attached to the operation.
InFlightDiagnostic Operation::emitError/Remark/Warning();

// This method creates a diagnostic prefixed with "'op-name' op ".
InFlightDiagnostic Operation::emitOpError();
```

## Diagnostic

A `Diagnostic` in MLIR contains all of the necessary information for reporting a
message to the user. A `Diagnostic` essentially boils down to three main
components:

*   [Source Location](#source-locations)
*   Severity Level
    -   Error, Note, Remark, Warning
*   Diagnostic Arguments
    -   The diagnostic arguments are used when constructing the output message.

### Appending arguments

One a diagnostic has been constructed, the user can start composing it. The
output message of a diagnostic is composed of a set of diagnostic arguments that
have been attached to it. New arguments can be attached to a diagnostic in a few
different ways:

```c++
// A few interesting things to use when composing a diagnostic.
Attribute fooAttr;
Type fooType;
SmallVector<int> fooInts;

// Diagnostics can be composed via the streaming operators.
op->emitError() << "Compose an interesting error: " << fooAttr << ", " << fooType
                << ", (" << fooInts << ')';

// This could generate something like (FuncAttr:@foo, IntegerType:i32, {0,1,2}):
"Compose an interesting error: @foo, i32, (0, 1, 2)"
```

### Attaching notes

Unlike many other compiler frameworks, notes in MLIR cannot be emitted directly.
They must be explicitly attached to another diagnostic non-note diagnostic. When
emitting a diagnostic, notes can be directly attached via `attachNote`. When
attaching a note, if the user does not provide an explicit source location the
note will inherit the location of the parent diagnostic.

```c++
// Emit a note with an explicit source location.
op->emitError("...").attachNote(noteLoc) << "...";

// Emit a note that inherits the parent location.
op->emitError("...").attachNote() << "...";
```

## InFlight Diagnostic

Now that [Diagnostics](#diagnostic) have been explained, we introduce the
`InFlightDiagnostic`. is an RAII wrapper around a diagnostic that is set to be
reported. This allows for modifying a diagnostic while it is still in flight. If
it is not reported directly by the user it will automatically report when
destroyed.

```c++
{
  InFlightDiagnostic diag = op->emitError() << "...";
}  // The diagnostic is automatically reported here.
```

## Common Diagnostic Handlers

To interface with the diagnostics infrastructure, users will need to register a
diagnostic handler with the [`DiagnosticEngine`](#diagnostic-engine).
Recognizing the many users will want the same handler functionality, MLIR
provides several common diagnostic handlers for immediate use.

### Scoped Diagnostic Handler

This diagnostic handler is a simple RAII class that registers and unregisters a
given diagnostic handler. This class can be either be used directly, or in
conjunction with a derived diagnostic handler.

```c++
// Construct the handler directly.
MLIRContext context;
ScopedDiagnosticHandler scopedHandler(&context, [](Diagnostic &diag) {
  ...
});

// Use this handler in conjunction with another.
class MyDerivedHandler : public ScopedDiagnosticHandler {
  MyDerivedHandler(MLIRContext *ctx) : ScopedDiagnosticHandler(ctx) {
    // Set the handler that should be RAII managed.
    setHandler([&](Diagnostic diag) {
      ...
    });
  }
};
```

### SourceMgr Diagnostic Handler

This diagnostic handler is a wrapper around an llvm::SourceMgr instance. It
provides support for displaying diagnostic messages inline with a line of a
respective source file. This handler will also automatically load newly seen
source files into the SourceMgr when attempting to display the source line of a
diagnostic. Example usage of this handler can be seen in the `mlir-opt` tool.

```shell
$ mlir-opt foo.mlir

/tmp/test.mlir:6:24: error: expected non-function type
func @foo() -> (index, ind) {
                       ^
```

To use this handler in your tool, add the following:

```c++
SourceMgr sourceMgr;
MLIRContext context;
SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
```

### SourceMgr Diagnostic Verifier Handler

This handler is a wrapper around a llvm::SourceMgr that is used to verify that
certain diagnostics have been emitted to the context. To use this handler,
annotate your source file with expected diagnostics in the form of:

*   `expected-(error|note|remark|warning) {{ message }}`

A few examples are shown below:

```mlir {.mlir}
// Expect an error on the same line.
func @bad_branch() {
  br ^missing  // expected-error {{reference to an undefined block}}
}

// Expect an error on an adjacent line.
func @foo(%a : f32) {
  // expected-error@+1 {{unknown comparison predicate "foo"}}
  %result = cmpf "foo", %a, %a : f32
  return
}

// Expect an error on the next line that does not contain a designator.
// expected-remark@below {{remark on function below}}
// expected-remark@below {{another remark on function below}}
func @bar(%a : f32)

// Expect an error on the previous line that does not contain a designator.
func @baz(%a : f32)
// expected-remark@above {{remark on function above}}
// expected-remark@above {{another remark on function above}}

```

The handler will report an error if any unexpected diagnostics were seen, or if
any expected diagnostics weren't.

```shell
$ mlir-opt foo.mlir

/tmp/test.mlir:6:24: error: unexpected error: expected non-function type
func @foo() -> (index, ind) {
                       ^

/tmp/test.mlir:15:4: error: expected remark "expected some remark" was not produced
// expected-remark {{expected some remark}}
   ^~~~~~~~~~~~~~~~~~~~~~~~~~
```

Similarly to the [SourceMgr Diagnostic Handler](#sourcemgr-diagnostic-handler),
this handler can be added to any tool via the following:

```c++
SourceMgr sourceMgr;
MLIRContext context;
SourceMgrDiagnosticVerifierHandler sourceMgrHandler(sourceMgr, &context);
```

### Parallel Diagnostic Handler

MLIR is designed from the ground up to be multi-threaded. One important to thing
to keep in mind when multi-threading is determinism. This means that the
behavior seen when operating on multiple threads is the same as when operating
on a single thread. For diagnostics, this means that the ordering of the
diagnostics is the same regardless of the amount of threads being operated on.
The ParallelDiagnosticHandler is introduced to solve this problem.

After creating a handler of this type, the only remaining step is to ensure that
each thread that will be emitting diagnostics to the handler sets a respective
'orderID'. The orderID corresponds to the order in which diagnostics would be
emitted when executing synchronously. For example, if we were processing a list
of operations [a, b, c] on a single-thread. Diagnostics emitted while processing
operation 'a' would be emitted before those for 'b' or 'c'. This corresponds 1-1
with the 'orderID'. The thread that is processing 'a' should set the orderID to
'0'; the thread processing 'b' should set it to '1'; and so on and so forth.
This provides a way for the handler to deterministically order the diagnostics
that it receives given the thread that it is receiving on.

A simple example is shown below:

```c++
MLIRContext *context = ...;
ParallelDiagnosticHandler handler(context);

// Process a list of operations in parallel.
std::vector<Operation *> opsToProcess = ...;
llvm::for_each_n(llvm::parallel::par, 0, opsToProcess.size(),
                 [&](size_t i) {
  // Notify the handler that we are processing the i'th operation.
  handler.setOrderIDForThread(i);
  auto *op = opsToProcess[i];
  ...

  // Notify the handler that we are finished processing diagnostics on this
  // thread.
  handler.eraseOrderIDForThread();
});
```
