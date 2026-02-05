# Error code: E2001

**Category:** Compile Time: Unsupported RHS DataType on Hardware

This error occurs when the data type used for the **Right-Hand Side (RHS)**
operand in a matrix multiplication (e.g., `jax.lax.dot_general`, `jax.lax.conv`,
`jax.numpy.matmul`, or the `@` operator) is not natively supported by the
specific TPU generation being used.

**Sample Error Messages:**

```
INTERNAL: Mosaic failed to compile TPU kernel: Unsupported matmul RHS type on target: 'vector<256x256xi8>'
...

The MLIR operation involved:
%13440 = "tpu.matmul"(%13435, %13437, %13439) <dimension_numbers = #tpu.dot_dimension_numbers<...>
```

**XLA Backends:** TPU

## Overview

The TPU's Matrix Multiply Unit (MXU) natively supports `Float32` operations on
all hardware generations.

However, native support for `BFloat16` and other quantized data types
(e.g., Int4, Int8, or Float8) varies by hardware generation. This error is
triggered when your kernel attempts to map a matrix multiplication to the MXU
using a data type that your specific TPU generation does not have the physical
circuitry to execute.

This error typically indicates that the compiler's **Canonicalization** pass—
which attempts to automatically convert unsupported types into supported ones
(e.g., via software emulation)—was unable to find a valid conversion rule or
was prevented from doing so because **Compatibility Mode** was disabled.

## Debugging

To resolve this error, you must align your data types with the capabilities of
your hardware. You have following options:

### 1. Cast to Native Types
The most reliable fix is to manually cast your operands to a hardware-supported
datatype (like `Float32` or `BFloat16` on TPU v4+) inside your kernel before the
matmul operation.

* **Why:** `Float32` is the universal data type supported natively by the
MXU on all TPU generations.
* **Trade-off:** This comes with a VPU (Vector Processing Unit) cost - the
cycles required to perform the cast, but it guarantees your kernel
will run on the current hardware.

### 2. Check Compatibility Mode
Typically the compiler can automatically handle these type mismatch issues in
**Compatibility Mode** which is enabled by default. Double check XLA configs
to make sure `--xla_mosaic_compat_mode` is not set to false.

This acts as a "polyfill," injecting software emulation sequences for operations
your hardware does not natively support.

**What Compatibility Mode enables:**

* **Mixed-Precision MatMuls:** Allows mixing Integer operands with Float
accumulators by automatically inserting cast operations (e.g., extending
integers to `Float32` before the matmul).
* **Low-Precision Emulation:** On certain hardware generations, emulates
unsupported types like `4-bit` floating point (`4E2M1FN`) or `8-bit` floating
point (`8E4M3FN`) by extending them to supported types like `BFloat16` or
`Float32` before execution.

Note that this mode prioritizes compatibility over peak performance as emulation
requires additional instructions to convert data formats before the MXU can
operate on them.

### 3. Upgrade Hardware or Request Support

If your algorithm strictly requires native performance for types like `Int4` or
`Float8` without the overhead of casting or emulation, you will need to run on
a newer TPU generation with native support.

**Feature Request:** If you believe your hardware supports this operation, or
if the compiler is missing a valid emulation path even in Compatibility Mode,
please file a feature request. We usually guarantee that operations are forward
compatible. So if your kernel runs on a TPU generation then it should run on all
future generations. But it is not guaranteed to have emulation for older
generations (for some of which the casts would be very expensive).
