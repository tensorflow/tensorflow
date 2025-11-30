# Fix Summary: Issue #105367 - Segmentation Fault with Complex Variable Operations

## Issue Description
A segmentation fault occurred when performing complex number operations involving:
- Complex64/complex128 variables
- `tf.raw_ops.Conj` operation
- `Variable.assign_add()` method

## Reproduction Code
```python
import tensorflow as tf
input_data = tf.constant([1 + 2j, 3 + 4j], dtype=tf.complex64)
var = tf.Variable(input_data, dtype=tf.complex64)
conj_result = tf.raw_ops.Conj(input=input_data)
assign_add_op = var.assign_add(conj_result)
# Segmentation fault (core dumped)
```

## Root Cause Analysis

The segmentation fault was caused by **missing complex type support** in two critical locations:

1. **GPU DenseUpdate Functor Instantiations** (`dense_update_functor_gpu.cu.cc`)
   - The template instantiations for `DenseUpdate<GPUDevice, T, ADD>` and `DenseUpdate<GPUDevice, T, SUB>` only included `TF_CALL_GPU_NUMBER_TYPES` and `TF_CALL_INTEGRAL_TYPES`
   - `TF_CALL_GPU_NUMBER_TYPES` = {half, bfloat16, float, double} - **does NOT include complex types**
   - `TF_CALL_COMPLEX_TYPES` = {complex64, complex128} - **was missing**

2. **GPU Kernel Registrations** (`resource_variable_ops.cc`)
   - The GPU kernel registrations for `AssignAddVariableOp` and `AssignSubVariableOp` similarly only included `TF_CALL_GPU_NUMBER_TYPES` and `TF_CALL_INTEGRAL_TYPES_NO_INT32`
   - Complex types were not registered for GPU execution

When users attempted to use `assign_add` on complex variables (especially after operations like `tf.raw_ops.Conj`), the kernel was not properly instantiated for complex types on GPU, leading to undefined behavior and segmentation faults.

## Solution

### Files Modified

1. **tensorflow/core/kernels/dense_update_functor_gpu.cu.cc**
   ```cpp
   // Added complex type support
   #define DEFINE_GPU_KERNELS(T)                              \
     template struct functor::DenseUpdate<GPUDevice, T, ADD>; \
     template struct functor::DenseUpdate<GPUDevice, T, SUB>;
   TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_KERNELS);
   TF_CALL_INTEGRAL_TYPES(DEFINE_GPU_KERNELS);
   TF_CALL_COMPLEX_TYPES(DEFINE_GPU_KERNELS);  // <-- ADDED
   TF_CALL_float8_e5m2(DEFINE_GPU_KERNELS);
   TF_CALL_float8_e4m3fn(DEFINE_GPU_KERNELS);
   ```

2. **tensorflow/core/kernels/resource_variable_ops.cc**
   ```cpp
   // Added complex type support to GPU kernel registrations
   TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS);
   TF_CALL_INTEGRAL_TYPES_NO_INT32(REGISTER_GPU_KERNELS);
   TF_CALL_COMPLEX_TYPES(REGISTER_GPU_KERNELS);  // <-- ADDED
   ```

3. **tensorflow/python/kernel_tests/variables/resource_variable_ops_test.py**
   - Added `testComplexVariableAssignAddWithConj()` - Tests GPU execution with Conj operation
   - Added `testComplexVariableAssignAddCPU()` - Tests CPU execution with complex types
   - Both tests cover complex64 and complex128 data types

## Testing

The fix has been validated with:
- ✅ The original reproduction case from issue #105367
- ✅ New unit tests covering both complex64 and complex128 types
- ✅ Tests for both CPU and GPU execution paths
- ✅ Tests with `tf.raw_ops.Conj` operation combined with `assign_add`

## Impact

This fix enables:
- Proper support for complex number arithmetic in resource variables on GPU
- Safe usage of `assign_add` and `assign_sub` with complex variables
- Compatibility with operations that produce complex results (like `Conj`, `FFT`, etc.)

## Pull Request

- **Branch**: `fix-complex-variable-conj-segfault`
- **PR URL**: https://github.com/CodersAcademy006/tensorflow/pull/9
- **Fixes**: #105367

## Technical Details

### Type Macro Definitions
- `TF_CALL_GPU_NUMBER_TYPES`: half, bfloat16, float, double
- `TF_CALL_COMPLEX_TYPES`: complex64, complex128
- `TF_CALL_NUMBER_TYPES`: TF_CALL_REAL_NUMBER_TYPES + TF_CALL_COMPLEX_TYPES

### Why This Worked on CPU but Failed on GPU
- CPU implementations use generic templates defined in header files
- GPU implementations require explicit template instantiations in `.cu.cc` files
- CPU kernel registrations already included `TF_CALL_NUMBER_TYPES` (which includes complex types)
- GPU kernel registrations only included `TF_CALL_GPU_NUMBER_TYPES` (which excludes complex types)

This asymmetry caused the issue to only manifest on GPU execution paths.
