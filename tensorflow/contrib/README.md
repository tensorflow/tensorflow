# TensorFlow contrib

Any code in this directory is not officially supported, and may change or be
removed at any time without notice.

The contrib directory contains project directories, each of which has designated
owners. It is meant to contain features and contributions that eventually should
get merged into core TensorFlow, but whose interfaces may still change, or which
require some testing to see whether they can find broader acceptance. We are
trying to keep dupliction within contrib to a minimum, so you may be asked to
refactor code in contrib to use some feature inside core or in another project
in contrib rather than reimplementing the feature.

When adding a project, please stick to the following directory structure:
Create a project directory in `contrib/`, and mirror the portions of the
TensorFlow tree that your project requires underneath `contrib/my_project/`.

For example, let's say you create foo ops in two files: `foo_ops.py` and
`foo_ops_test.py`. If you were to merge those files directly into TensorFlow,
they would live in `tensorflow/python/ops/foo_ops.py` and
`tensorflow/python/kernel_tests/foo_ops_test.py`. In `contrib/`, they are part
of project `foo`, and their full paths are `contrib/foo/python/ops/foo_ops.py`
and `contrib/foo/python/kernel_tests/foo_ops_test.py`.
