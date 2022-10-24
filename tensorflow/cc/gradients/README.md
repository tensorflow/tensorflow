# C++ gradients

Gradients are currently being ported from
[python](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/ops)
to C++ (in this directory).

Contributions are welcome and much appreciated; please follow the instructions
below.

1.  Create the op gradient function in `foo_grad.cc` corresponding to the
    `foo_grad.py` file where the op originated (i.e. `array_grad.py` op
    gradients should be written in `array_grad.cc`).

2.  Write the op gradient with the following naming scheme:

    ```
    Status OpNameGrad(const Scope& scope, const Operation& op,
                      const std::vector<Output>& grad_inputs,
                      std::vector<Output>* grad_outputs) {
      ...
      return scope.status();
    }
    REGISTER_GRADIENT_OP("OpName", OpNameGrad);
    ```

3.  Ops gradients are implemented by using the
    [C++ API](https://www.tensorflow.org/api_docs/cc/).

4.  Tests should be included in `foo_grad_test.cc`. Please see
    [`array_grad_test.cc`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/cc/gradients/array_grad_test.cc)
    for many examples. Tests are as simple as, creating a placeholder input for
    the op's inputs and calling `RunTest` (`RunTest` uses a
    [gradient checker](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/cc/framework/gradient_checker.cc)
    to verify that the theoretical gradient matches the numeric gradient). For
    example:

    ```
    TEST_F(ArrayGradTest, IdentityGrad) {
      TensorShape shape({5, 2});
      auto x = Placeholder(scope_, DT_FLOAT, Placeholder::Shape(shape));
      auto y = Identity(scope_, x);
      RunTest(x, shape, y, shape);
    }
    ```

NOTE: There are some ops that require features from the C++ API that are not yet
implemented.

*   Ops that require PartialTensorShape information cannot yet be implemented.

*   Ops that require SparseTensor or IndexSlices (currently only in python)
    cannot yet be implemented.

*   Maybe more.

For questions: Please create an issue assigned to suharshs.
