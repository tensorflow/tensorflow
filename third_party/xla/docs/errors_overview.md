# XLA Errors Overview

XLA errors are categorized into different XLA error sources. Each source has a
list of an additional context other than the error message, which will be
attached to each error within the category.

🚧 Note that this standarization effort is a work in progress so not all error
messages will have an attached error code yet.

An example error log might look like:

```
XlaRuntimeError: RESOURCE_EXHAUSTED: XLA:TPU compile permanent error. Ran out of memory in memory space hbm. Used 49.34G of 32.00G hbm. Exceeded hbm capacity by 17.34G. Total hbm usage >= 49.34G: reserved 3.12M program unknown size arguments 49.34G

JaxRuntimeError: RESOURCE_EXHAUSTED: Ran out of memory in memory space vmem while allocating on stack for %ragged_latency_optimized_all_gather_lhs_contracting_gated_matmul_kernel.18 = bf16[2048,4096]{1,0:T(8,128)(2,1)} custom-call(%get-tuple-element.18273, %get-tuple-element.18274, %get-tuple-element.18275, %get-tuple-element.18276, %get-tuple-element.18277, /*index=5*/%bitcast.8695, %get-tuple-element.19201, %get-tuple-element.19202, %get-tuple-element.19203, %get-tuple-element.19204), custom_call_target=""
```

## Statuses and CHECK failures

In general, in XLA we can flag corrupted execution with two mechanisms: statuses
and CHECK macro failures.

Statuses are meant for non-fatal, recoverable errors. The assumption is that the
function returns, and execution continues down the path where the caller
explicitly checks the returned Status object. It's useful for handling invalid
user input or expected resource constraints.

On the other hand, CHECK failures cover programmer's errors or violations of
invariants that should never happen if the code is correct. In case of an
activated CHECK the program will log the error message and immediately
terminate. It could ensure internal consistency, such as checking that a pointer
is non-null before dereferencing it.

## Error codes

Here is an index list with all [error codes](error_codes.md).
