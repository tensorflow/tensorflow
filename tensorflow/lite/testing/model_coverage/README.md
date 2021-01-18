# TensorFlow Lite model coverage tests

Various conversion tests on popular mobile models.


## Golden values

Some tests rely on pre-computed golden values. The main goal is to detect
changes affecting unintended parts of TFLite.

Should a golden value test fail after an intended change, the golden values can
be updated with the following command:

```
bazel run //third_party/tensorflow/lite/testing/model_coverage:<target> --test_output=all -- --update_goldens
```

Notice `bazel run` instead of `bazel test` and the addition of the
`--update_golden` flag.

The updated golden data files must then be included in the change list.
