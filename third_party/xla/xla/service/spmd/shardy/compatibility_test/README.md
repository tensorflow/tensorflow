# Compatibility test workflow

For each new change in XLA Shardy round tripping attributes we follow the
process:

1. Copy the latest `compatibility_test.{Month}_{Day}_{Year}.mlir` file and
 create a new one with new version number in the filename
2. Append tests for new functionality to the file
3. Generate `bc` file for this test:

```sh
# change the version to the correct one
bazel run @shardy//shardy/tools:sdy_translate -- --strip-debuginfo --serialize --target-version=X.Y.Z < xla/service/spmd/shardy/compatibility_test/compatibility_test_MM_DD_YYYY.mlir > xla/service/spmd/shardy/compatibility_test/compatibility_test_MM_DD_YYYY.mlir.bc
```

Note: **forward compatibility** is supported by keeping old .bc files and
running checks that compiler can still generate old .bc files (when asked for
certain target_version) and the test works as expected

Note: **backward compatibility** is supported by running old tests and being
able to deserialize with new compiler and test works as expected
