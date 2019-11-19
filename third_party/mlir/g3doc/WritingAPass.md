# Writing a Pass

[TOC]

Passes represent the basic infrastructure for transformation and optimization.
This document provides a quickstart to the pass infrastructure in MLIR and how
to use it.

See [MLIR specification](LangRef.md) for more information about MLIR and its
core aspects, such as the IR structure and operations.

See [MLIR Rewrites](QuickstartRewrites.md) for a quick start on graph rewriting
in MLIR. If your transformation involves pattern matching operation DAGs, this
is a great place to start.

## Operation Pass

In MLIR, the main unit of abstraction and transformation is an
[operation](LangRef.md#operations). As such, the pass manager is designed to
work on instances of operations at different levels of nesting. The structure of
the [pass manager](#pass-manager), and the concept of nesting, is detailed
further below. All passes in MLIR derive from `OperationPass` and adhere to the
following restrictions; any noncompliance will lead to problematic behavior in
multithreaded and other advanced scenarios:

*   Modify anything within the parent block/region/operation/etc, outside of the
    current operation being operated on. This includes adding or removing
    operations from the parent block.
*   Maintain pass state across invocations of `runOnOperation`. A pass may be
    run on several different operations with no guarantee of execution order.
    *   When multithreading, a specific pass instance may not even execute on
        all operations within the module. As such, a pass should not rely on
        running on all operations.
*   Modify the state of another operation not nested within the current
    operation being operated on.
    *   Other threads may be operating on different operations within the module
        simultaneously.
*   Maintain any global mutable state, e.g. static variables within the source
    file. All mutable state should be maintained by an instance of the pass.
*   Must be copy-constructible, multiple instances of the pass may be created by
    the pass manager to process operations in parallel.
*   Inspect the IR of sibling operations. Other threads may be modifying these
    operations in parallel.

When creating an operation pass, there are two different types to choose from
depending on the usage scenario:

### OperationPass : Op-Specific

An `op-specific` operation pass operates explicitly on a given operation type.
This operation type must adhere to the restrictions set by the pass manager for
pass execution.

To define an op-specific operation pass, a derived class must adhere to the
following:

*   Inherit from the CRTP class `OperationPass` and provide the operation type
    as an additional template parameter.
*   Override the virtual `void runOnOperation()` method.

A simple pass may look like:

```c++
namespace {
struct MyFunctionPass : public OperationPass<MyFunctionPass, FuncOp> {
  void runOnOperation() override {
    // Get the current FuncOp operation being operated on.
    FuncOp f = getOperation();

    // Walk the operations within the function.
    f.walk([](Operation *inst) {
      ....
    });
  }
};
} // end anonymous namespace

// Register this pass to make it accessible to utilities like mlir-opt.
// (Pass registration is discussed more below)
static PassRegistration<MyFunctionPass> pass(
    "flag-name-to-invoke-pass-via-mlir-opt", "Pass description here");
```

### OperationPass : Op-Agnostic

An `op-agnostic` pass operates on the operation type of the pass manager that it
is added to. This means that a pass that operates on several different operation
types in the same way only needs one implementation.

To create an operation pass, a derived class must adhere to the following:

*   Inherit from the CRTP class `OperationPass`.
*   Override the virtual `void runOnOperation()` method.

A simple pass may look like:

```c++
struct MyOperationPass : public OperationPass<MyOperationPass> {
  void runOnOperation() override {
    // Get the current operation being operated on.
    Operation *op = getOperation();
    ...
  }
};
```

## Analysis Management

An important concept, along with transformation passes, are analyses. These are
conceptually similar to transformation passes, except that they compute
information on a specific operation without modifying it. In MLIR, analyses are
not passes but free-standing classes that are computed lazily on-demand and
cached to avoid unnecessary recomputation. An analysis in MLIR must adhere to
the following:

*   Provide a valid constructor taking an `Operation*`.
*   Must not modify the given operation.

The base `OperationPass` class provide utilities for querying and preserving
analyses for the current operation being processed. Using the example passes
defined above, let's see some examples:

### Querying Analyses

*   OperationPass automatically provides the following utilities for querying
    analyses:
    *   `getAnalysis<>`
        -   Get an analysis for the current operation, constructing it if
            necessary.
    *   `getCachedAnalysis<>`
        -   Get an analysis for the current operation, if it already exists.
    *   `getCachedParentAnalysis<>`
        -   Get an analysis for a given parent operation, if it exists.
    *   `getCachedChildAnalysis<>`
        -   Get an analysis for a given child operation, if it exists.
    *   `getChildAnalysis<>`
        -   Get an analysis for a given child operation, constructing it if
            necessary.

A few example usages are shown below:

```c++
/// An interesting analysis.
struct MyOperationAnalysis {
  // Compute this analysis with the provided operation.
  MyOperationAnalysis(Operation *op);
};

void MyOperationPass::runOnOperation() {
  // Query MyOperationAnalysis for the current operation.
  MyOperationAnalysis &myAnalysis = getAnalysis<MyOperationAnalysis>();

  // Query a cached instance of MyOperationAnalysis for the current operation.
  // It will not be computed if it doesn't exist.
  auto optionalAnalysis = getCachedAnalysis<MyOperationAnalysis>();
  if (optionalAnalysis)
    ...

  // Query a cached instance of MyOperationAnalysis for the parent operation of
  // the current operation. It will not be computed if it doesn't exist.
  auto optionalAnalysis = getCachedParentAnalysis<MyOperationAnalysis>();
  if (optionalAnalysis)
    ...
}
```

### Preserving Analyses

Analyses that are constructed after being queried by a pass are cached to avoid
unnecessary computation if they are requested again later. To avoid stale
analyses, all analyses are assumed to be invalidated by a pass. To avoid
invalidation, a pass must specifically mark analyses that are known to be
preserved.

*   All Pass classes automatically provide the following utilities for
    preserving analyses:
    *   `markAllAnalysesPreserved`
    *   `markAnalysesPreserved<>`

```c++
void MyOperationPass::runOnOperation() {
  // Mark all analyses as preserved. This is useful if a pass can guarantee
  // that no transformation was performed.
  markAllAnalysesPreserved();

  // Mark specific analyses as preserved. This is used if some transformation
  // was performed, but some analyses were either unaffected or explicitly
  // preserved.
  markAnalysesPreserved<MyAnalysis, MyAnalyses...>();
}
```

## Pass Failure

Passes in MLIR are allowed to gracefully fail. This may happen if some invariant
of the pass was broken, potentially leaving the IR in some invalid state. If
such a situation occurs, the pass can directly signal a failure to the pass
manager. If a pass signaled a failure when executing, no other passes in the
pipeline will execute and the `PassManager::run` will return failure. Failure
signaling is provided in the form of a `signalPassFailure` method.

```c++
void MyPass::runOnOperation() {
  // Signal failure on a broken invariant.
  if (some_broken_invariant) {
    signalPassFailure();
    return;
  }
}
```

## Pass Manager

Above we introduced the different types of passes and their constraints. Now
that we have our pass we need to be able to run it over a specific module. This
is where the pass manager comes into play. The `PassManager` class is used to
configure and run a pipeline. The `OpPassManager` class is used to schedule
passes to run at a specific level of nesting.

### OpPassManager

An `OpPassManager` is essentially a collection of passes to execute on an
operation of a given type. This operation type must adhere to the following
requirement:

*   Must be registered and marked `IsolatedFromAbove`.

    *   Passes are expected to not modify operations at or above the current
        operation being processed. If the operation is not isolated, it may
        inadvertently modify the use-list of an operation it is not supposed to
        modify.

Passes can be added to a pass manager via `addPass`. The pass must either be an
`op-specific` pass operating on the same operation type as `OpPassManager`, or
an `op-agnostic` pass.

An `OpPassManager` cannot be created directly, but must be explicitly nested
within another `OpPassManager` via the `nest<>` method. This method takes the
operation type that the nested pass manager will operate on. At the top-level, a
`PassManager` acts as an `OpPassManager` that operates on the
[`module`](LangRef.md#module) operation. Nesting in this sense, corresponds to
the structural nesting within [Regions](LangRef.md#regions) of the IR.

For example, the following `.mlir`:

```
module {
  spv.module "Logical" "GLSL450" {
    func @foo() {
      ...
    }
  }
}
```

Has the nesting structure of:

```
`module`
  `spv.module`
    `function`
```

Below is an example of constructing a pipeline that operates on the above
structure:

```c++
PassManager pm(ctx);

// Add a pass on the top-level module operation.
pm.addPass(std::make_unique<MyModulePass>());

// Nest a pass manager that operates on spirv module operations nested directly
// under the top-level module.
OpPassManager &nestedModulePM = pm.nest<spirv::ModuleOp>();
nestedModulePM.addPass(std::make_unique<MySPIRVModulePass>());

// Nest a pass manager that operates on functions within the nested SPIRV
// module.
OpPassManager &nestedFunctionPM = nestedModulePM.nest<FuncOp>();
nestedFunctionPM.addPass(std::make_unique<MyFunctionPass>());

// Run the pass manager on the top-level module.
Module m = ...;
if (failed(pm.run(m)))
    ... // One of the passes signaled a failure.
```

The above pass manager would contain the following pipeline structure:

```
OpPassManager<ModuleOp>
  MyModulePass
  OpPassManager<spirv::ModuleOp>
    MySPIRVModulePass
    OpPassManager<FuncOp>
      MyFunctionPass
```

These pipelines are then run over a single operation at a time. This means that,
for example, given a series of consecutive passes on FuncOp, it will execute all
on the first function, then all on the second function, etc. until the entire
program has been run through the passes. This provides several benefits:

*   This improves the cache behavior of the compiler, because it is only
    touching a single function at a time, instead of traversing the entire
    program.
*   This improves multi-threading performance by reducing the number of jobs
    that need to be scheduled, as well as increasing the efficiency of each job.
    An entire function pipeline can be run on each function asynchronously.

## Pass Registration

Briefly shown in the example definitions of the various
pass types is the `PassRegistration` class. This is a utility to
register derived pass classes so that they may be created, and inspected, by
utilities like mlir-opt. Registering a pass class takes the form:

```c++
static PassRegistration<MyPass> pass("command-line-arg", "description");
```

*   `MyPass` is the name of the derived pass class.
*   "command-line-arg" is the argument to use on the command line to invoke the
    pass from `mlir-opt`.
*   "description" is a description of the pass.

For passes that cannot be default-constructed, `PassRegistration` accepts an
optional third argument that takes a callback to create the pass:

```c++
static PassRegistration<MyParametricPass> pass(
    "command-line-arg", "description",
    []() -> std::unique_ptr<Pass> {
      std::unique_ptr<Pass> p = std::make_unique<MyParametricPass>(/*options*/);
      /*... non-trivial-logic to configure the pass ...*/;
      return p;
    });
```

This variant of registration can be used, for example, to accept the
configuration of a pass from command-line arguments and pass it over to the pass
constructor. Make sure that the pass is copy-constructible in a way that does
not share data as the [pass manager](#pass-manager) may create copies of the
pass to run in parallel.

### Pass Pipeline Registration

Described above is the mechanism used for registering a specific derived pass
class. On top of that, MLIR allows for registering custom pass pipelines in a
similar fashion. This allows for custom pipelines to be available to tools like
mlir-opt in the same way that passes are, which is useful for encapsulating
common pipelines like the "-O1" series of passes. Pipelines are registered via a
similar mechanism to passes in the form of `PassPipelineRegistration`. Compared
to `PassRegistration`, this class takes an additional parameter in the form of a
pipeline builder that modifies a provided `OpPassManager`.

```c++
void pipelineBuilder(OpPassManager &pm) {
  pm.addPass(std::make_unique<MyPass>());
  pm.addPass(std::make_unique<MyOtherPass>());
}

// Register an existing pipeline builder function.
static PassPipelineRegistration<> pipeline(
  "command-line-arg", "description", pipelineBuilder);

// Register an inline pipeline builder.
static PassPipelineRegistration<> pipeline(
  "command-line-arg", "description", [](OpPassManager &pm) {
    pm.addPass(std::make_unique<MyPass>());
    pm.addPass(std::make_unique<MyOtherPass>());
  });
```

Pipeline registration also allows for simplified registration of
specifializations for existing passes:

```c++
static PassPipelineRegistration<> foo10(
    "foo-10", "Foo Pass 10", [] { return std::make_unique<FooPass>(10); } );
```

### Textual Pass Pipeline Specification

In the previous sections, we showed how to register passes and pass pipelines
with a specific argument and description. Once registered, these can be used on
the command line to configure a pass manager. The limitation of using these
arguments directly is that they cannot build a nested pipeline. For example, if
our module has another module nested underneath, with just `-my-module-pass`
there is no way to specify that this pass should run on the nested module and
not the top-level module. This is due to the flattened nature of the command
line.

To circumvent this limitation, MLIR also supports a textual description of a
pass pipeline. This allows for explicitly specifying the structure of the
pipeline to add to the pass manager. This includes the nesting structure, as
well as the passes and pass pipelines to run. A textual pipeline is defined as a
series of names, each of which may in itself recursively contain a nested
pipeline description. The syntax for this specification is as follows:

```ebnf
pipeline          ::= op-name `(` pipeline-element (`,` pipeline-element)* `)`
pipeline-element  ::= pipeline | (pass-name | pass-pipeline-name) options?
options           ::= '{' (key ('=' value)?)+ '}'
```

*   `op-name`
    *   This corresponds to the mnemonic name of an operation to run passes on,
        e.g. `func` or `module`.
*   `pass-name` | `pass-pipeline-name`
    *   This corresponds to the command-line argument of a registered pass or
        pass pipeline, e.g. `cse` or `canonicalize`.
*   `options`
    *   Options are pass specific key value pairs that are handled as described
        in the instance specific pass options section.

For example, the following pipeline:

```shell
$ mlir-opt foo.mlir -cse -canonicalize -convert-std-to-llvm
```

Can also be specified as (via the `-pass-pipeline` flag):

```shell
$ mlir-opt foo.mlir -pass-pipeline='func(cse, canonicalize), convert-std-to-llvm'
```

In order to support round-tripping your pass to the textual representation using
`OpPassManager::printAsTextualPipeline(raw_ostream&)`, override
`Pass::printAsTextualPipeline(raw_ostream&)` to format your pass-name and
options in the format described above.

### Instance Specific Pass Options

Options may be specified for a parametric pass. Individual options are defined
using `llvm::cl::opt` flag definition rules. These options will then be parsed
at pass construction time independently for each instance of the pass. The
`PassRegistration` and `PassPipelineRegistration` templates take an additional
optional template parameter that is the Option struct definition to be used for
that pass. To use pass specific options, create a class that inherits from
`mlir::PassOptions` and then add a new constructor that takes `const
MyPassOptions&` and constructs the pass. When using `PassPipelineRegistration`,
the constructor now takes a function with the signature `void (OpPassManager
&pm, const MyPassOptions&)` which should construct the passes from the options
and pass them to the pm. The user code will look like the following:

```c++
class MyPass ... {
public:
  MyPass(const MyPassOptions& options) ...
};

struct MyPassOptions : public PassOptions<MyPassOptions> {
  // These just forward onto llvm::cl::list and llvm::cl::opt respectively.
  Option<int> exampleOption{*this, "flag-name", llvm::cl::desc("...")};
  List<int> exampleListOption{*this, "list-flag-name", llvm::cl::desc("...")};
};

static PassRegistration<MyPass, MyPassOptions> pass("my-pass", "description");
```

## Pass Instrumentation

MLIR provides a customizable framework to instrument pass execution and analysis
computation. This is provided via the `PassInstrumentation` class. This class
provides hooks into the PassManager that observe various pass events:

*   `runBeforePipeline`
    *   This callback is run just before a pass pipeline, i.e. pass manager, is
        executed.
*   `runAfterPipeline`
    *   This callback is run right after a pass pipeline has been executed,
        successfully or not.
*   `runBeforePass`
    *   This callback is run just before a pass is executed.
*   `runAfterPass`
    *   This callback is run right after a pass has been successfully executed.
        If this hook is executed, runAfterPassFailed will not be.
*   `runAfterPassFailed`
    *   This callback is run right after a pass execution fails. If this hook is
        executed, runAfterPass will not be.
*   `runBeforeAnalysis`
    *   This callback is run just before an analysis is computed.
*   `runAfterAnalysis`
    *   This callback is run right after an analysis is computed.

PassInstrumentation objects can be registered directly with a
[PassManager](#pass-manager) instance via the `addInstrumentation` method.
Instrumentations added to the PassManager are run in a stack like fashion, i.e.
the last instrumentation to execute a `runBefore*` hook will be the first to
execute the respective `runAfter*` hook. Below in an example instrumentation
that counts the number of times DominanceInfo is computed:

```c++
struct DominanceCounterInstrumentation : public PassInstrumentation {
  unsigned &count;

  DominanceCounterInstrumentation(unsigned &count) : count(count) {}
  void runAfterAnalysis(llvm::StringRef, AnalysisID *id, Operation *) override {
    if (id == AnalysisID::getID<DominanceInfo>())
      ++count;
  }
};

MLIRContext *ctx = ...;
PassManager pm(ctx);

// Add the instrumentation to the pass manager.
unsigned domInfoCount;
pm.addInstrumentation(
    std::make_unique<DominanceCounterInstrumentation>(domInfoCount));

// Run the pass manager on a module operation.
ModuleOp m = ...;
if (failed(pm.run(m)))
    ...

llvm::errs() << "DominanceInfo was computed " << domInfoCount << " times!\n";
```

### Standard Instrumentations

MLIR utilizes the pass instrumentation framework to provide a few useful
developer tools and utilities. Each of these instrumentations are immediately
available to all users of the MLIR pass framework.

#### Pass Timing

The PassTiming instrumentation provides timing information about the execution
of passes and computation of analyses. This provides a quick glimpse into what
passes are taking the most time to execute, as well as how much of an effect
your pass has on the total execution time of the pipeline. Users can enable this
instrumentation directly on the PassManager via `enableTiming`. This
instrumentation is also made available in mlir-opt via the `-pass-timing` flag.
The PassTiming instrumentation provides several different display modes for the
timing results, each of which is described below:

##### List Display Mode

In this mode, the results are displayed in a list sorted by total time with each
pass/analysis instance aggregated into one unique result. This view is useful
for getting an overview of what analyses/passes are taking the most time in a
pipeline. This display mode is available in mlir-opt via
`-pass-timing-display=list`.

```shell
$ mlir-opt foo.mlir -disable-pass-threading -cse -canonicalize -convert-std-to-llvm -pass-timing -pass-timing-display=list

===-------------------------------------------------------------------------===
                      ... Pass execution timing report ...
===-------------------------------------------------------------------------===
  Total Execution Time: 0.0203 seconds

   ---Wall Time---  --- Name ---
   0.0047 ( 55.9%)  Canonicalizer
   0.0019 ( 22.2%)  VerifierPass
   0.0016 ( 18.5%)  LLVMLoweringPass
   0.0003 (  3.4%)  CSE
   0.0002 (  1.9%)  (A) DominanceInfo
   0.0084 (100.0%)  Total
```

##### Pipeline Display Mode

In this mode, the results are displayed in a nested pipeline view that mirrors
the internal pass pipeline that is being executed in the pass manager. This view
is useful for understanding specifically which parts of the pipeline are taking
the most time, and can also be used to identify when analyses are being
invalidated and recomputed. This is the default display mode.

```shell
$ mlir-opt foo.mlir -disable-pass-threading -cse -canonicalize -convert-std-to-llvm -pass-timing

===-------------------------------------------------------------------------===
                      ... Pass execution timing report ...
===-------------------------------------------------------------------------===
  Total Execution Time: 0.0249 seconds

   ---Wall Time---  --- Name ---
   0.0058 ( 70.8%)  'func' Pipeline
   0.0004 (  4.3%)    CSE
   0.0002 (  2.6%)      (A) DominanceInfo
   0.0004 (  4.8%)    VerifierPass
   0.0046 ( 55.4%)    Canonicalizer
   0.0005 (  6.2%)    VerifierPass
   0.0005 (  5.8%)  VerifierPass
   0.0014 ( 17.2%)  LLVMLoweringPass
   0.0005 (  6.2%)  VerifierPass
   0.0082 (100.0%)  Total
```

##### Multi-threaded Pass Timing

When multi-threading is enabled in the pass manager the meaning of the display
slightly changes. First, a new timing column is added, `User Time`, that
displays the total time spent across all threads. Secondly, the `Wall Time`
column displays the longest individual time spent amongst all of the threads.
This means that the `Wall Time` column will continue to give an indicator on the
perceived time, or clock time, whereas the `User Time` will display the total
cpu time.

```shell
$ mlir-opt foo.mlir -cse -canonicalize -convert-std-to-llvm -pass-timing

===-------------------------------------------------------------------------===
                      ... Pass execution timing report ...
===-------------------------------------------------------------------------===
  Total Execution Time: 0.0078 seconds

   ---User Time---   ---Wall Time---  --- Name ---
   0.0177 ( 88.5%)     0.0057 ( 71.3%)  'func' Pipeline
   0.0044 ( 22.0%)     0.0015 ( 18.9%)    CSE
   0.0029 ( 14.5%)     0.0012 ( 15.2%)      (A) DominanceInfo
   0.0038 ( 18.9%)     0.0015 ( 18.7%)    VerifierPass
   0.0089 ( 44.6%)     0.0025 ( 31.1%)    Canonicalizer
   0.0006 (  3.0%)     0.0002 (  2.6%)    VerifierPass
   0.0004 (  2.2%)     0.0004 (  5.4%)  VerifierPass
   0.0013 (  6.5%)     0.0013 ( 16.3%)  LLVMLoweringPass
   0.0006 (  2.8%)     0.0006 (  7.0%)  VerifierPass
   0.0200 (100.0%)     0.0081 (100.0%)  Total
```

#### IR Printing

When debugging it is often useful to dump the IR at various stages of a pass
pipeline. This is where the IR printing instrumentation comes into play. This
instrumentation allows for conditionally printing the IR before and after pass
execution by optionally filtering on the pass being executed. This
instrumentation can be added directly to the PassManager via the
`enableIRPrinting` method. `mlir-opt` provides a few useful flags for utilizing
this instrumentation:

*   `print-ir-before=(comma-separated-pass-list)`
    *   Print the IR before each of the passes provided within the pass list.
*   `print-ir-before-all`
    *   Print the IR before every pass in the pipeline.

```shell
$ mlir-opt foo.mlir -cse -print-ir-before=cse

*** IR Dump Before CSE ***
func @simple_constant() -> (i32, i32) {
  %c1_i32 = constant 1 : i32
  %c1_i32_0 = constant 1 : i32
  return %c1_i32, %c1_i32_0 : i32, i32
}
```

*   `print-ir-after=(comma-separated-pass-list)`
    *   Print the IR after each of the passes provided within the pass list.
*   `print-ir-after-all`
    *   Print the IR after every pass in the pipeline.

```shell
$ mlir-opt foo.mlir -cse -print-ir-after=cse

*** IR Dump After CSE ***
func @simple_constant() -> (i32, i32) {
  %c1_i32 = constant 1 : i32
  return %c1_i32, %c1_i32 : i32, i32
}
```

*   `print-ir-module-scope`
    *   Always print the top-level module operation, regardless of pass type or
        operation nesting level.
    *   Note: Printing at module scope should only be used when multi-threading
        is disabled(`-disable-pass-threading`)

```shell
$ mlir-opt foo.mlir -disable-pass-threading -cse -print-ir-after=cse -print-ir-module-scope

*** IR Dump After CSE ***  ('func' operation: @bar)
func @bar(%arg0: f32, %arg1: f32) -> f32 {
  ...
}

func @simple_constant() -> (i32, i32) {
  %c1_i32 = constant 1 : i32
  %c1_i32_0 = constant 1 : i32
  return %c1_i32, %c1_i32_0 : i32, i32
}

*** IR Dump After CSE ***  ('func' operation: @simple_constant)
func @bar(%arg0: f32, %arg1: f32) -> f32 {
  ...
}

func @simple_constant() -> (i32, i32) {
  %c1_i32 = constant 1 : i32
  return %c1_i32, %c1_i32 : i32, i32
}
```

## Crash and Failure Reproduction

The [pass manager](#pass-manager) in MLIR contains a builtin mechanism to
generate reproducibles in the even of a crash, or a
[pass failure](#pass-failure). This functionality can be enabled via
`PassManager::enableCrashReproducerGeneration` or via the command line flag
`pass-pipeline-crash-reproducer`. In either case, an argument is provided that
corresponds to the output `.mlir` file name that the reproducible should be
written to. The reproducible contains the configuration of the pass manager that
was executing, as well as the initial IR before any passes were run. A potential
reproducible may have the form:

```mlir
// configuration: -pass-pipeline='func(cse, canonicalize), inline'
// note: verifyPasses=false

module {
  func @foo() {
    ...
  }
}
```
