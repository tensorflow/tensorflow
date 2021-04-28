# Porting Reference Ops from Lite to Micro

This is a guide to porting reference ops from Lite to Micro. It explains,
step-by-step, the recommended code changes and the process for submitting them
for review and acceptance.  The process results in multiple pull requests, or
PRs. Multiple, [small PRs][] are easier for the project to review and merge.

[small PRs]: https://google.github.io/eng-practices/review/developer/small-cls.html

The [Micro Contributing Guidelines][] are prerequisite reading. They cover
general code health, maintainability, style, and submission, as well as how to
setup a development environment. This guide contains step-by-step instructions
for the specific task of porting reference ops from Lite to Micro.

[Micro Contributing Guidelines]: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/CONTRIBUTING.md

## 1. Look for a port already in progress

Begin by searching the TensorFlow GitHub repository for issues containing the
name of the op under consideration to ensure someone isn't already working on a
port.

## 2. Open a GitHub issue to track the port
    
Open a GitHub issue to announce your intent to port the op, and to begin a
record of your work. Document the entire process of porting the op in this
issue. Link constituent PRs to this issue. See the article [Providing
Context][] for background on documenting your work via bug reports.

[Providing Context]: https://testing.googleblog.com/2017/09/code-health-providing-context-with.html

A good example is [issue #45306: micro: port op FILL from lite][#45306].

## 3. Extract Lite's code for parsing op parameters to a function (PR1)

Now we begin changing, testing, and submitting code. This step will result in
the first pull request, PR1.

1.  Extract the code for parsing op parameters out of the switch statement in
    [`ParseOpDataTfLite()`][] in `lite/core/api/flatbuffer_conversions.cc` into
    a standalone function, and call that function from the switch statement.
    This standalone function is now available to be called by the Micro op
    resolver, which also needs to parse the op parameters, in a future change.
    A simple example is [PR #45307][], and a more complicated example is [PR
    #46021][].

    [`ParseOpDataTfLite()`]: https://github.com/tensorflow/tensorflow/blob/d8394a6d774f5e3c02d97f1fc18ff445199db598/tensorflow/lite/core/api/flatbuffer_conversions.cc#L135
    [PR #45307]: https://github.com/tensorflow/tensorflow/pull/45307
    [PR #46021]: https://github.com/tensorflow/tensorflow/pull/46021

1.  Use `clang-format` to make sure the code is properly formatted.

        clang-format --style=google -i $(git ls-files -m | grep -E '\.cc|\.h')

1.  Make sure your code is lint-free.

        cpplint.py $(git ls-files -m)

1.  Create a single commit containing the change. Observe the guidelines for
    good commit log messages found in the article [Providing Context][].
    A good example is commit [0664214](https://github.com/tensorflow/tensorflow/pull/45307/commits/0664214792ad2357f6224e7002661894775cb512).

1.  Since this change modifies the op's implementation in Lite, test the change
    with the relevant Lite unit tests.

        bazel test tensorflow/lite/kernels:all

1.  Create and submit the PR. Write a [good PR description][], and be sure to
    link to the GitHub issue created to document the port. A good example is
    [PR #45307][].

    [good PR description]: https://google.github.io/eng-practices/review/developer/cl-descriptions.html

## 4. Extract the reference for the op to a standalone header (PR2)

Move the reference implementation of the op in [reference_ops.h][] to a standalone header so that
Micro can include it without including unrelated dependencies via
reference_ops.h.

A good example is [PR #45311][].

1.  Copy an existing header from
    *tensorflow/lite/kernels/internal/reference/* to
    *tensorflow/lite/kernels/internal/reference/<new_op.h>* to create the
    boilerplate.

1.  Move the implementation from
    *tensorflow/lite/kernels/internal/reference/reference_ops.h* to
    *tensorflow/lite/kernels/internal/reference/<new_op.h>*.

1.  Add the new header to the build in
    *tensorflow/lite/kernels/internal/BUILD* under *reference_base* and
    *legacy_reference_base*. E.g., [for FILL][].

    [for FILL]: https://github.com/tensorflow/tensorflow/pull/45311/commits/92f459e6b917fa5099ef5317d14c5100d33a86f0#diff-0b0fc9e1affece3c5a141ee9326f882876b6b958bc8b12a7c01d7540dc04983e

1.  Use `clang-format` to make sure the code is properly formatted.

        clang-format --style=google -i $(git ls-files -m | grep -E '\.cc|\.h')

    Do not clang-format existing code in *BUILD* or *reference_ops.h*.

1.  Make sure your code is lint-free.

        cpplint.py $(git ls-files -m)

    Do not modify code in *BUILD* or *reference_ops.h* to satisfy cpplint.py.

1.  Create a single commit containing the change. Observe the guidelines for
    good commit log messages found in the article [Providing Context][].
    A good example is commit [92f459e](https://github.com/tensorflow/tensorflow/commit/92f459e6b917fa5099ef5317d14c5100d33a86f0).

1.  Since this change modifies the op's implementation in Lite, test the change
    with the relevant Lite unit tests.

        bazel test tensorflow/lite/kernels:all

1.  Create and submit the PR. Write a [good PR description][], and be sure to
    link to the GitHub issue created to document the port. A good example is
    [PR #45311][].

## 5. Port the op from Lite to Micro (PR3)

1.  Copy the kernel and test from Lite to Micro.
    
    In the first commit of this PR, copy the kernel and test from Lite to Micro
    without making any modifications and without adding them to the build.
    
    A good example is commit [a2ca1fd](https://github.com/tensorflow/tensorflow/commit/a2ca1fd7a174438f736c0435dd3e4e618612fdee).

    This copy action is in its own commit in order to create readable, reviewable diffs
    when modifications are made in later commits. If the files were copied and
    modified in one step, the modifications would not appear as a diff of the Lite
    version. Instead, the files would simply appear at the destination path in
    their final form.


1.  Remove Lite-specific code from copies

    In the second commit of this PR, remove the bulk of Lite-specific code from
    the files copied to micro in the previous step.
    
    A good example is commit [a5a87b4](https://github.com/tensorflow/tensorflow/commit/a5a87b420b87a1f832e241db3a5b724207ea700a).
    
    This bulk-delete action is in its own commit for reasons similar to
    those given in the step above: to produce a more readable, reviewable diff in this
    step and in the next. Because the files are not yet added to the build, they
    need not (and obviously won't) compiler or function. What to delete now as
    opposed to deleting in the next commit is somewhat subjective, but make
    deletes in order to:

    -   Flatten the namespace down to `tflite`.
    -   Stop resizing output tensors.
    -   Remove input and output types other than int8 and float32.
    -   Stop using gmock and gtest.
    -   etc.

1.  Port the op and the test

    Make the necessary changes to the micro kernel, header, and test to make the op
    implementation suitable for micro. Include these in the build.

    This step requires the most creativity, and may receive the most feedback
    during review. Maintain good atomicity in your commits. Considering its
    scope, this step will consist of more than one commit. A good example is
    the changes made in [PR #45647][]. 

1.  Use `clang-format` to make sure the code is properly formatted.

        clang-format --style=google -i $(git ls-files -m | grep -E '\.cc|\.h')

    Do not clang-format existing code in *BUILD* or *reference_ops.h*.

1.  Make sure the code is lint-free.

        cpplint.py $(git ls-files -m)

    Do not modify code in *BUILD* or *reference_ops.h* to satisfy cpplint.py.

1.  Make sure the port passes all applicable tests.

        bazel test tensorflow/lite/micro/kernels:${op}_test
        bazel test tensorflow/lite/micro/kernels:all
        make -f tensorflow/lite/micro/tools/make/Makefile test_kernel_${op}_test
        make -f tensorflow/lite/micro/tools/make/Makefile test

    See the general [Micro Contributing Guidelines][] for other testing ideas,
    including the use of address sanitizers.

1.  Create and submit the PR. Write a [good PR description][], and be sure to
    link to the GitHub issue created to document the port. A good example is
    [PR #45647][].

[#45306]: https://github.com/tensorflow/tensorflow/issues/45306
[PR #45311]: https://github.com/tensorflow/tensorflow/pull/45311
[PR #45457]: https://github.com/tensorflow/tensorflow/pull/45457
[PR #45646]: https://github.com/tensorflow/tensorflow/pull/45646
[PR #45647]: https://github.com/tensorflow/tensorflow/pull/45647
[pre-submit checklist]: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/CONTRIBUTING.md#before-submitting-your-pr
[reference_ops.h]: https://github.com/tensorflow/tensorflow/blob/92f459e6b917fa5099ef5317d14c5100d33a86f0/tensorflow/lite/kernels/internal/reference/reference_ops.h
[general porting guidelines]: #general-porting-guidelines

## General Guidelines

### Check each commit for formatting, lint, and unit-test passage

Check each commit against the [pre-submit checklist][] in the micro
Contributing Guidelines. Specifically, make sure your code:

1.  Is formatted with clang-format.
1.  Passes a lint check.
1.  Passes all unit tests.

    make -s -j8 -f tensorflow/lite/micro/tools/make/Makefile test

CI runs these checks on all PRs, and will hold up your PR if any of these checks fail.

### Add yourself as an asignee to PRs

Feel free to add yourself as an additional assignee to PRs which you submit.
Other assignees may be set by the project's various bots.

### Maintain a 1:1 correspondence between Micro and Lite versions of unit tests

To the extent possible, maintain a 1:1 correspondence between Micro and Lite
versions of unit tests. Avoid cleanup of merely stylistic issues, e.g., by
replacing the hardcoded literal `3.40282e+038` with
`std::numeric_limits<float>::max()`. Any changes between the Micro and Lite
versions of a test put a burden on future maintainers to figure out whether the
differences are actually significant or just stylistic.

### Sometimes CI checks on PRs are flakey and fail

Sometimes CI checks on PRs don't fail because of the PRs contents, but because
of some problem with the test infrastructure. Marking issues with the label
`kokoro:force-run` causes the checks to be rerun.

## Notes

*   There was discussion of commits vs. PRs in [#45387][].

*   On Debian, running bazel required installing package bazel-3.1.0.

*   If you have permission, add the label *comp:micro* to these PRs.

*   If you have permission, the label *kokoro:force-run* can be applied to
    manually trigger the CI builds.

*   [TensorFlow Lite 8-bit quantization specification](https://www.tensorflow.org/lite/performance/quantization_spec)

[#45387]: https://github.com/tensorflow/tensorflow/issues/45387
