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

## Pass Types {#pass-types}

MLIR provides different pass classes for several different granularities of
transformation. Depending on the granularity of the transformation being
performed, a pass may derive from [FunctionPass](#function-pass) or
[ModulePass](#module-pass); with each requiring a different set of constraints.

### FunctionPass {#function-pass}

A function pass operates on a per-function granularity, executing on
non-external functions within a module in no particular order. Function passes
have the following restrictions, and any noncompliance will lead to problematic
behavior in multithreaded and other advanced scenarios:

*   Modify anything within the parent module, outside of the current function
    being operated on. This includes adding or removing functions from the
    module.
*   Maintain pass state across invocations of runOnFunction. A pass may be run
    on several different functions with no guarantee of execution order.
    *   When multithreading, a specific pass instance may not even execute on
        all functions within the module. As such, a function pass should not
        rely on running on all functions within a module.
*   Access, or modify, the state of another function within the module.
    *   Other threads may be operating on different functions within the module.
*   Maintain any global mutable state, e.g. static variables within the source
    file. All mutable state should be maintained by an instance of the pass.

To create a function pass, a derived class must adhere to the following:

*   Inherit from the CRTP class `FunctionPass`.
*   Override the virtual `void runOnFunction()` method.
*   Must be copy-constructible, multiple instances of the pass may be created by
    the pass manager to process functions in parallel.

A simple function pass may look like:

```c++
namespace {
struct MyFunctionPass : public FunctionPass<MyFunctionPass> {
  void runOnFunction() override {
    // Get the current function being operated on.
    Function *f = getFunction();

    // Operate on the instructions within the function.
    f->walk([](Instruction *inst) {
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

### ModulePass {#module-pass}

A module pass operates on a per-module granularity, executing on the entire
program as a unit. As such, module passes are able to add/remove/modify any
functions freely. Module passes have the following restrictions, and any
noncompliance will lead to problematic behavior in multithreaded and other
advanced scenarios:

*   Maintain pass state across invocations of runOnModule.
*   Maintain any global mutable state, e.g. static variables within the source
    file. All mutable state should be maintained by an instance of the pass.

To create a module pass, a derived class must adhere to the following:

*   Inherit from the CRTP class `ModulePass`.
*   Override the virtual `void runOnModule()` method.

A simple module pass may look like:

```c++
namespace {
struct MyModulePass : public ModulePass<MyModulePass> {
  void runOnModule() override {
    // Get the current module being operated on.
    Module *m = getModule();

    // Operate on the functions within the module.
    for (auto &func : *m) {
      ....
    }
  }
};
} // end anonymous namespace

// Register this pass to make it accessible to utilities like mlir-opt.
// (Pass registration is discussed more below)
static PassRegistration<MyModulePass> pass(
    "flag-name-to-invoke-pass-via-mlir-opt", "Pass description here");
```

## Analysis Management

An important concept, along with transformation passes, are analyses. These are
conceptually similar to transformation passes, except that they compute
information on a specific Function, or Module, without modifying it. In MLIR,
analyses are not passes but free standing classes that are computed lazily
on-demand and cached to avoid unnecessary recomputation. An analysis in MLIR
must adhere to the following:

*   Provide a valid constructor for a given IR unit.
*   Must not modify the given IR unit.

Each of the base Pass classes provide utilities for querying and preserving
analyses for the current IR being processed. Using the example passes defined
above, let's see some examples:

### Querying Analyses

*   FunctionPass automatically provides the following utilities for querying
    analyses:
    *   `getAnalysis<>`
    *   `getCachedAnalysis<>`
    *   `getCachedModuleAnalysis<>`
*   ModulePass automatically provides the following utilities:
    *   `getAnalysis<>`
    *   `getCachedAnalysis<>`
    *   `getFunctionAnalysis<>`
    *   `getCachedFunctionAnalysis<>`

```c++
/// An interesting function analysis.
struct MyFunctionAnalysis {
  // Compute this analysis with the provided function.
  MyFunctionAnalysis(Function *function);
};

/// An interesting module analysis.
struct MyModuleAnalysis {
  // Compute this analysis with the provided module.
  MyModuleAnalysis(Module *module);
};

void MyFunctionPass::runOnFunction() {
  // Query MyFunctionAnalysis for the current function.
  MyFunctionAnalysis &myAnalysis = getAnalysis<MyFunctionAnalysis>();

  // Query a cached instance of MyFunctionAnalysis for the current function. It
  // will not be computed if it doesn't exist.
  auto optionalAnalysis = getCachedAnalysis<MyFunctionAnalysis>();
  if (optionalAnalysis)
    ...

  // Query a cached instance of MyModuleAnalysis for the parent module of the
  // current function. It will not be computed if it doesn't exist.
  auto optionalAnalysis = getCachedModuleAnalysis<MyModuleAnalysis>();
  if (optionalAnalysis)
    ...
}

void MyModulePass::runOnModule() {
  // Query MyModuleAnalysis for the current module.
  MyModuleAnalysis &myAnalysis = getAnalysis<MyModuleAnalysis>();

  // Query a cached instance of MyModuleAnalysis for the current module. It
  // will not be computed if it doesn't exist.
  auto optionalAnalysis = getCachedAnalysis<MyModuleAnalysis>();
  if (optionalAnalysis)
    ...

  // Query MyFunctionAnalysis for a child function of the current module. It
  // will be computed if it doesn't exist.
  auto *fn = &*getModule().begin();
  MyFunctionAnalysis &myAnalysis = getFunctionAnalysis<MyFunctionAnalysis>(fn);
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
void MyPass::runOn*() {
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
void MyPass::runOn*() {
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
is where the pass manager comes into play. The PassManager is the interface used
for scheduling passes to run over a module. The pass manager provides simple
interfaces for adding passes, of any kind, and running them over a given module.
A simple example is shown below:

```c++
PassManager pm;

// Add a module pass.
pm.addPass(new MyModulePass());

// Add a few function passes.
pm.addPass(new MyFunctionPass());
pm.addPass(new MyFunctionPass2());
pm.addPass(new MyFunctionPass3());

// Add another module pass.
pm.addPass(new MyModulePass2());

// Run the pass manager on a module.
Module *m = ...;
if (failed(pm.run(m)))
    ... // One of the passes signaled a failure.
```

The pass manager automatically structures added passes into nested pipelines on
specific IR units. These pipelines are then run over a single IR unit at a time.
This means that, for example, given a series of consecutive function passes, it
will execute all on the first function, then all on the second function, etc.
until the entire program has been run through the passes. This provides several
benefits:

*   This improves the cache behavior of the compiler, because it is only
    touching a single function at a time, instead of traversing the entire
    program.
*   This improves multi-threading performance by reducing the number of jobs
    that need to be scheduled, as well as increasing the efficency of each job.
    An entire function pipeline can be run on each function asynchronously.

As an example, the above pass manager would contain the following pipeline
structure:

```c++
MyModulePass
Function Pipeline
   MyFunctionPass
   MyFunctionPass2
   MyFunctionPass3
MyModulePass2
```

## Pass Registration {pass-registration}

Briefly shown in the example definitions of the various
[pass types](#pass-types) is the `PassRegistration` class. This is a utility to
register derived pass classes so that they may be created, and inspected, by
utilities like mlir-opt. Registering a pass class takes the form:

```c++
static PassRegistration<MyPass> pass("command-line-arg", "description");
```

*   `MyPass` is the name of the derived pass class.
*   "command-line-arg" is the argument to use on the command line to invoke the
    pass from `mlir-opt`.
*   "description" is a description of the pass.

### Pass Pipeline Registration

Described above is the mechanism used for registering a specific derived pass
class. On top of that, MLIR allows for registering custom pass pipelines in a
similar fashion. This allows for custom pipelines to be available to tools like
mlir-opt in the same way that passes are, which is useful for encapsulating
common pipelines like the "-O1" series of passes. Pipelines are registered via a
similar mechanism to passes in the form of `PassPipelineRegistration`. Compared
to `PassRegistration`, this class takes an additional parameter in the form of a
pipeline builder that modifies a provided PassManager.

```c++
void pipelineBuilder(PassManager &pm) {
  pm.addPass(new MyPass());
  pm.addPass(new MyOtherPass());
}

// Register an existing pipeline builder function.
static PassPipelineRegistration pipeline(
  "command-line-arg", "description", pipelineBuilder);

// Register an inline pipeline builder.
static PassPipelineRegistration pipeline(
  "command-line-arg", "description", [](PassManager &pm) {
    pm.addPass(new MyPass());
    pm.addPass(new MyOtherPass());
  });
```

Pipeline registration also allows for simplified registration of
specifializations for existing passes:

```c++
static PassPipelineRegistration foo10(
    "foo-10", "Foo Pass 10", [] { return new FooPass(10); } );
```
