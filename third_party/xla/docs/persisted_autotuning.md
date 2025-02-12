# Persisted autotuning (GPU only)

We use OpenAI Triton for generating some of the GPU kernels. Triton allows
generating fast GPU kernels for certain fusions, but we have to tune some
parameters for each such fusion.

This can take a long time if there are many fusions, so we provide a way to load
those autotuning results, while still running the other compilation steps
normally. Autotuning caches are still useful if we make a few changes: the
fusions that are present in the cache will use the cache, and the other ones
will be autotuned normally.

## Recommended: Cache directory

```
--xla_gpu_per_fusion_autotune_cache_dir=your/directory
```

Use and maintain a per-fusion autotune cache in the given directory. There will
be one file per distinct fusion.

The main advantage of this approach is that you can use the same cache directory
for multiple XLA runs (of different models) and your cache will grow with each
new fusion encountered - speeding up subsequent runs. There is also basic
support for running multiple XLA instances with the same cache directory
concurrently.

XLA will read existing results when they are needed and write new results after
they are determined.

-   The directory must exist before running XLA and it must be writable.
-   Cache invalidation has to be handled by the user:
    -   Please use an empty directory if you want to start with an empty cache.
-   XLA version checks must be done by the user:
    -   If you want to use separate caches for different versions of XLA, please
        use different directories.

The cache is turned off by default (when you don't provide the parameter).

Limitation: This is not guaranteed to work well in combination with the other
caching method described below.

## Alternative: Loading or dumping all results from a given HLO to one file

The autotuning results can be dumped/loaded using these parameters:

```
--xla_gpu_dump_autotune_results_to=
--xla_gpu_load_autotune_results_from=
```

If we specify a .txt or .textproto file, then the cache will be dumped in
textproto format, otherwise in binary protobuf format.

## In tests

Persisted autotuning can also be used in tests. It is recommended to use it if
the tests are very big, especially if the performance of the test environment is
limited.

It only works well if the autotune cache contains results generated on the same
type of GPU where the tests are being run.

### Making a test use persisted autotuning

For now let's assume that the test in question always uses the same GPU type.

1.  We have to export the autotune results from the test, for example by
    specifying these parameters to the test command:

    ```
    --test_env=XLA_FLAGS=--xla_gpu_dump_autotune_results_to=TEST_UNDECLARED_OUTPUTS_DIR/autotune_cache.textproto
    --test_sharding_strategy=disabled
    ```

    Sharding must be disabled to correctly get a single autotune cache for all
    tests.

2.  Then we have to upload that cache to our code repository.

3.  Then we have to add the cache to the data dependencies of our test target,
    and load it using an environment variable.

    ```
    data = ["test_autotune_cache.textproto"],
    env = {"XLA_FLAGS": "--xla_gpu_load_autotune_results_from=" +
                        "$(execpath test_autotune_cache.textproto)"},
    ```

    (It is OK to use sharding in tests that load autotune results.)

Please also see the example tests in
[xla/service/gpu/tests/BUILD](https://github.com/openxla/xla/blob/main/xla/service/gpu/tests/BUILD):

-   load_autotune_results_using_execpath_test
-   load_autotune_results_from_test_workspace_test
-   dump_autotune_results_to_test_outputs_test

### Cache obsolescence

If many changes are made to a model, it is possible that the cache will no
longer contain all fusions, so the test will become slower. In this case we
would have to regenerate the autotuning cache.

If we start using a new type of GPU for running the tests, the same applies.

The cache may also become obsolete if the XLA compiler evolves and generates
different fusions.
