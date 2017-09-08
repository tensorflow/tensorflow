TensorFlow API backwards compatibility test
This test ensures all changes to the public API of TensorFlow are intended.

If this test fails, it means a change has been made to the public API. Backwards
incompatible changes are not allowed. You can run the test as follows to update
test goldens and package them with your change.

    $ bazel build tensorflow/tools/api/tests:api_compatibility_test
    $ bazel-bin/tensorflow/tools/api/tests/api_compatibility_test \
          --update_goldens True

You will need an API approval to make changes to the public TensorFlow API. This
includes additions to the API.
