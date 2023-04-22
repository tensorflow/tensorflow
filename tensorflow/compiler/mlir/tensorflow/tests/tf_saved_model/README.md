# SavedModel importer FileCheck tests.

## Debugging tests

While debugging tests, the following commands are handy.

Run FileCheck test:

```
bazel run :foo.py.test
```

Run just the Python file and look at the output:

```
bazel run :foo
```

Generate saved model to inspect proto:

```
bazel run :foo -- --save_model_path=/tmp/my.saved_model
# Inspect /tmp/my.saved_model/saved_model.pb
```

## Rationale for Python-based tests

For a SavedModel importer, the natural place to start is to feed in the
SavedModel format directly and test the output MLIR. We don't do that though.

The SavedModel format is a directory structure which contains a SavedModel proto
and some other stuff (mostly binary files of some sort) in it. That makes it not
suitable for use as a test input, since it is not human-readable. Even just the
text proto for the SavedModel proto is difficult to use as a test input, since a
small piece of Python code (e.g. just a tf.Add) generates thousands of lines of
text proto.

That points to a solution though: write our tests starting from the Python API's
that generate the SavedModel. That leads to very compact test inputs.

As the SavedModel work progresses, it's likely to be of interest to find a
shortcut between the Python `tf.Module` and the SavedModel MLIR representation
that doesn't involve serializing a SavedModel to disk and reading it back.

## Potential improvements

The test iteration cycle for these tests is very long (usually over a minute).
We need to find a way to improve this in the future.
