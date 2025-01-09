# XLA HLO Pass Framework

This folder provides the core components of the XLA HLO pass framework, which is
responsible for optimizing and transforming HLO computations within XLA
compiler.

## Key APIs and Files

### `hlo_pass_interface.h`

Defines the foundational classes for HLO passes:

* `HloPassInterface`: Abstract base class for all HLO passes.
* `HloModulePass`: Subclass for passes that operate on individual HloModules.
* `HloModuleGroupPass`: Subclass for passes that operate on HloModuleGroups
(collections of modules).

Provides core methods like `Run`, `RunOnModuleGroup`, and
`RunOnChangedComputations` that passes must implement to perform their
transformations.

### `hlo_pass_fix.h`

Introduces the `HloPassFix` template class. Allows running an HLO pass
repeatedly until a fixed point is reached (no further changes occur in the HLO).
Useful for passes that may trigger further optimizations when applied
iteratively.

### `hlo_pass_pipeline.h`

Defines the `HloPassPipeline` class. Organizes a sequence of HLO passes into a
pipeline for sequential execution.  Provides methods to add passes (`AddPass`)
and invariant checkers (`AddInvariantChecker`) to the pipeline. `Run` method
executes the entire pipeline on an HloModule or HloModuleGroup.

## Example Usage

```C++
// Create a pipeline
HloPassPipeline pipeline("my_pipeline");

// Add passes to the pipeline
pipeline.AddPass<SomeOptimizationPass>(/* pass arguments */);
pipeline.AddPass<HloPassFix<AnotherOptimizationPass>>(/* pass arguments */);

// Run the pipeline on an HloModule
HloModule module(/* ... */);
auto status = pipeline.Run(&module);
```

## Important Considerations

When creating custom HLO passes, inherit from either `HloModulePass` or
`HloModuleGroupPass` depending on the scope of your transformation.  Implement
the required virtual methods (e.g., `Run`) to define the pass's behavior.
Utilize `HloPassFix` when your pass's transformations may trigger further
optimizations upon repeated application. Construct `HloPassPipelines` to
orchestrate the execution of multiple passes in a defined sequence.
