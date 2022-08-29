# StableHLO Specification Draft

## Types

Following are the supported element types in StableHLO:

  * **Integer types**

    * Signed integer with two’s complement representation. Referred to in the
    document as `si<N>`, where the bit-width N ∊ {4, 8, 16, 32, 64}
    * Unsigned integer referred to in the document as `ui<N>`, where the
    bit-width N ∊ {4, 8, 16, 32, 64}
  * **Boolean types** referred to in the document as `pred`
  * **Floating-point types**
    * Single precision `f32`, double precision `f64` and half precision `f16`
    floating-points complying with [IEEE 754
    format](https://ieeexplore.ieee.org/document/8766229).
    * Bfloat16 `bf16` floating-point complying with [Brain Floating-Point Format](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus).
    Provides the same number of exponent bits as `f32`, so that it matches its
    dynamic range, but with greatly reduced precision.
 * **Complex types** represents a pair of floating-point types. Supported ones
 are `c64` (represents paired `f32`) and `c128` (represents paired `f64`).

StableHLO supports a Tensor type `tensor`, to model the type of a n-dimensional
array, represented in the opset as `tensor<SxE>` such that

  * Shape `S` is a list of number of elements in each of the
  dimensions and represented, in increasing order of the corresponding dimension
  number, as an array of values of type `ui64`. A 0-value represents missing or
  empty data in the corresponding dimension.
  * Element type `E` is any one of the supported element types mentioned above.

## Programs

StableHLO programs consist of functions. Each function has arguments and results
of supported types and a list of ops in static single-assignment (SSA) form
which is terminated by a return op which produces the results of the function.
StableHLO ops take operands and produce results.

```
ml_program.func @example_func(%arg: tensor<4x16xf32>) -> tensor<4x16xf32> {
 %1 = stablehlo.floor %arg : tensor<4x16xf32>
 %2 = stablehlo.ceil %arg : tensor<4x16xf32>
 %3 = stablehlo.add %1, %2 : tensor<4x16xf32>
 ml_program.return %3 : tensor<4x16xf32>
}
```

A program is executed by passing argument values to a given function and
computing result values. Result values of a function are computed by evaluating
the graph of ops rooted in the corresponding return op. The evaluation order is
implementation-defined, as long as ops are evaluated before their uses. Possible
execution orders of the above example program are `%1` → `%2` → `%3` → `return`
or `%2` → `%1` → `%3` → `return`.

## Constants

The section describes the constants supported in StableHLO along with their
syntax.

  * **Integer Constants** Standard integers, e.g. `123`, are constants of the
  integer type (signed or unsigned). Negative numbers can be used with signed
  integer types.
  * **Boolean Constants** `true` and `false` are both valid constants of the
  `pred` type.
  * **Floating-point Constants** Floating-point constants use standard decimal
  notation, e.g. `123.421`, exponential notation, e.g. `1.23421e+2`, or a more
  precise hexadecimal notation, e.g. `0x42f6d78d`.
  * **Complex Constants** Complex constants are represented as a pair of real
  and imaginary values of `f32` or `f64` types, e.g. `(12.34, 56,78)`.

## Structure of an Op’s Specification

The specification of an op comprises of the following components (in the order
    described below)

  * **Syntax** Operation mnemonic and its signature.
  * **Semantics** Semantics of the operation.
  * **Operands** Meaning of operand(s) and their type(s).
  * **Results** Meaning of the result(s) and the type(s).
  * **Constraints** Type constraints on the operand(s), result(s).
  * **Examples** Examples demonstrating the working of the op.

## stablehlo.add

`stablehlo.add(lhs, rhs) -> result`

### Semantics

Performs element-wise addition of two tensors `lhs` and `rhs` and produces a
`result` tensor. For integer element types, if the element-wise sum has an
unsigned/signed overflow/underflow, the result is implementation defined and one
of the followings:

  * mathematical result modulo $2^n$, where n is the bit width of the result.
  * saturation to $2^{n-1} - 1$ (or $-2^{n-1}$) for signed overflow (or signed
      underflow) and saturation to $2^n - 1$ (or $0$) for unsigned overflow (or
        unsigned underflow).

For floating-point element types, corner cases are defined by the IEEE-754
specification.

### Arguments

| Operand Name(s) | Type |
|-|-|
| `lhs` | `tensor` of Integer, Floating-point and Complex element types |
| `rhs` | `tensor` of Integer, Floating-point and Complex element types |

### Results

| Result Name | Type |
|-|-|
| `result` | `tensor` of Integer, Floating-point, and Complex element types |

### Constraints

  * `lhs`, `rhs` have the same type.
  * Supported shapes: all static shapes.
  * `result` must have the same type as that of `lhs` (or `rhs`).

### Examples

```c
// %x: [[1, 2], [3, 4]]
// %y: [[5, 6], [7, 8]]
%z = stablehlo.add %x, %y : tensor<2x2xf32>
// %z: [[6, 8], [10, 12]]
```
