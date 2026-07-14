# StableHLO C++ Reference Library

The goal of this library is to provide a C++ reference implementation of
StableHLO kernels.

## Contributing

Please review the [Tensorflow Contributing Guide] for the repository's
contributing guidelines.

The code makes use of C++17 and is built using Bazel.

Unless specified, the [Google style guide] should be followed. Clang-format with
`google` style should be used for automatic code formatting.

To keep familiarity for people who are used to working with StableHLO, the data
structures try to follow the naming and hierarchy that are found in the
[StableHLO specification][stablehlo]

While the library does not strive for performance, we try to avoid unnecessary
performance penalties. This means avoiding dynamic allocation when possible of
moving use cases to the `Create` or `Prepare` functions (in order of
preference).

### Adding an Operation

Refer to the [specification][stablehlo-op] for the naming of an operation, its
attributes and its inputs.

#### API

An operation is defined using a state structure and three functions.

-   `ExampleOp` is the class/structure that keeps the operation state. It
    defines a public (possibly empty) `Attributes` structure that holds the
    attributes described in the operation specification.

    > Tip: Search for `Input attributes` in the [specification][stablehlo] for
    > more information about attributes.

    > Tip: When reading the specification, the difference between input
    > attributes and input values is not immediately apparent. Check out the
    > examples that are given to distinguish them. The definitive authority is
    > the [StableHLO dialect definition][stablehlo-dialect]: check out the
    > operations `arguments` declaration for `*Attr` input types.

    ```cpp
    // Operation data.
    class ExampleOp {
     public:
      // The attributes are a direct mapping of the StableHLO spec.
      struct Attributes {
        int64_t attribute_one;
        float attribute_two;
      };
    };
    ```

-   `Create` initialises the operation data using its attributes as passed
    through the Attributes structure.

    ```cpp
    ExampleOp Create(const ExampleOp::Attributes&);
    ```

-   `Prepare` sets up data and pre-computations that should be reused between
    evaluations. **In case of dynamic tensors, this step also computes the
    output tensor dimensions** and should set them.

    -   Preconditions:
        -   Input tensor shapes are known.
    -   Postconditions:
        -   Output tensor shapes are set and valid.

    ```cpp
    // When an unknown number of tensors can be passed.
    Status Prepare(ExampleOp& op, const absl::Span<Tensor>& inputs, absl::Span<Tensor>& outputs);
    // When the number of input/output tensors is known at compile time we can provide an overload
    Status Prepare(ExampleOp& op, const Tensor& lhs, const Tensor& rhs, Tensor& output);
    ```

-   `Evaluate` computes the operation result.

    -   Preconditions:
        -   Input tensor shapes are the same as what was passed to Prepare.
        -   Input tensor data is known.
        -   Output tensor shape is known.
        -   Output tensor buffer is set and allocated.
    -   Postconditions:
        -   Output tensor buffers are filled with the operation result.

    ```cpp
    // When an unknown number of tensors can be passed.
    Status Eval(ExampleOp& op, const absl::Span<Tensor>& inputs, absl::Span<Tensor>& outputs);
    // When the number of input/output tensors is known at compile time.
    Status Eval(ExampleOp& op, const Tensor& lhs, const Tensor& rhs, Tensor& output);
    ```

Specific operations may define extra functions for implementation configuration
or tweaks.

#### Bazel

Each operation should be defined in a separate library with the associated tests
and benchmarks. The code should live in the `ops` folder.

-   The library name should be the name of the operation in `snake_case`.
-   The implementation and header files should be the name of the library with
    the `h/cc` extension.

```bzl
cc_library(
  name = "op_name",
  srcs = [ "op_name.cc" ],
  hdrs = [ "op_name.h" ],
  deps = [
    # ...
  ]
)
```

#### Testing

Testing is done with [GoogleTest]. Each operation should be fully tested for
result correctness and robustness.

-   The test name should be the name of the library with the `_test` suffix.
-   Use the result matchers to check for results.

```bzl
cc_test(
  name = "op_name_test",
  srcs = [ "op_name_test.cc" ],
  hdrs = [ "op_name_test.h" ], # Generally not needed.
  deps = [
    # ...
  ]
)
```

#### Benchmarking

Testing is done with [Google Benchmark]. Each operation should be fully tested
for result correctness and robustness.

-   The benchmark name should be the name of the library with the `_bench`
    suffix.

```bzl
cc_test(
  name = "op_name_bench",
  srcs = [ "op_name_bench.cc" ],
  hdrs = [ "op_name_bench.h" ], # Generally not needed.
  deps = [
    # ...
  ]
)
```

### Running Tests and Benchmarks

This section is a short introduction to running a binary on device.

#### Useful Flags

The following bazel flags may be useful when benchmarking and debugging.

-   `-c dbg`: Compile in debug mode.
-   `-c opt`: Compile in optimized mode.
-   `-gmlt`: Adds line and function name debug information to optimised builds.

#### x86

##### Tests

```sh
bazel test -c opt --dynamic_mode=off ops:op_name_test
```

> Note: it is often useful to run test in optimized **and** in debug mode.

##### Benchmarks

```sh
bazel run -c opt --dynamic_mode=off ops:op_name_bench
```

#### Android

```sh
bazel build -c opt --dynamic_mode=off --config=android_arm64 --copt=-DGOOGLE_COMMANDLINEFLAGS_FULL_API=1 ops:op_name_test
```

Bazel should print the location of the build binary. It should resemble
`shlo/ops/op_name_test`.

You can then push the binary to the device `/data/local/tmp` folder and run it
using ADB.

```sh
adb push shlo/ops/op_name_test /data/local/tmp
adb shell /data/local/tmp/op_name_test
```

#### iOS

##### Prerequisites

Follow the instructions for setting up the iOS development environment in the
TensorFlow Lite [Build for iOS] guide. The `configure` script must be run and
you must opt-in to iOS development.

##### Building

```
bazel build -c opt --config=ios_arm64 ops:op_name_test
```

##### Testing

TODO:

[stablehlo]: https://github.com/openxla/stablehlo/blob/main/docs/spec.md
[stablehlo-op]: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#operations
[stablehlo-dialect]: https://github.com/openxla/stablehlo/blob/main/stablehlo/dialect/StablehloOps.td
[GoogleTest]: https://github.com/google/googletest
[Google Benchmark]: https://github.com/google/benchmark
[Google style guide]: https://google.github.io/styleguide/cppguide.html
[Tensorflow Contributing Guide]: https://github.com/tensorflow/tensorflow/blob/master/CONTRIBUTING.md
[Build for iOS]: https://www.tensorflow.org/lite/guide/build_ios
