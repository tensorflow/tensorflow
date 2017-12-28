# Contributing guidelines

## How to become a contributor and submit your own code

### Contributor License Agreements

We'd love to accept your patches! Before we can take them, we have to jump a couple of legal hurdles.

Please fill out either the individual or corporate Contributor License Agreement (CLA).

  * If you are an individual writing original source code and you're sure you own the intellectual property, then you'll need to sign an [individual CLA](https://code.google.com/legal/individual-cla-v1.0.html).
  * If you work for a company that wants to allow you to contribute your work, then you'll need to sign a [corporate CLA](https://code.google.com/legal/corporate-cla-v1.0.html).

Follow either of the two links above to access the appropriate CLA and instructions for how to sign and return it. Once we receive it, we'll be able to accept your pull requests.

***NOTE***: Only original source code from you and other people that have signed the CLA can be accepted into the main repository.

### Contributing code

If you have improvements to TensorFlow, send us your pull requests! For those
just getting started, Github has a [howto](https://help.github.com/articles/using-pull-requests/).

TensorFlow team members will be assigned to review your pull requests. Once the pull requests are approved and pass continuous integration checks, we will merge the pull requests.
For some pull requests, we will apply the patch for each pull request to our internal version control system first, and export the change out as a new commit later, at which point the original pull request will be closed. The commits in the pull request will be squashed into a single commit with the pull request creator as the author. These pull requests will be labeled as pending merge internally.

If you want to contribute but you're not sure where to start, take a look at the
[issues with the "contributions welcome" label](https://github.com/tensorflow/tensorflow/labels/stat%3Acontributions%20welcome).
These are issues that we believe are particularly well suited for outside
contributions, often because we probably won't get to them right now. If you
decide to start on an issue, leave a comment so that other people know that
you're working on it. If you want to help out, but not alone, use the issue
comment thread to coordinate.

### Contribution guidelines and standards

Before sending your pull request for
[review](https://github.com/tensorflow/tensorflow/pulls),
make sure your changes are consistent with the guidelines and follow the
TensorFlow coding style.

#### General guidelines and philosophy for contribution

* Include unit tests when you contribute new features, as they help to
  a) prove that your code works correctly, b) guard against future breaking
  changes to lower the maintenance cost.
* Bug fixes also generally require unit tests, because the presence of bugs
  usually indicates insufficient test coverage.
* Keep API compatibility in mind when you change code in core TensorFlow,
  e.g., code in [tensorflow/core](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core) and  [tensorflow/python](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python).
  TensorFlow has reached version 1 and hence cannot make
  non-backward-compatible API changes without a major release. Reviewers of your
  pull request will comment on any API compatibility issues.
* When you contribute a new feature to TensorFlow, the maintenance burden is (by
  default) transferred to the TensorFlow team. This means that benefit of
  contribution must be compared against the cost of maintaining the feature.
* Full new features (e.g., a new op implementing a cutting-edge algorithm)
  typically will live in
  [tensorflow/contrib](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib)
  to get some airtime before decision is made regarding whether they are to be
  migrated to the core.

#### License

Include a license at the top of new files.

* [C/C++ license example](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/op.cc#L1)
* [Python license example](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/nn.py#L1)
* [Java license example](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/java/src/main/java/org/tensorflow/Graph.java#L1)
* [Go license example](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/go/operation.go#L1)
* [Bash license example](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/ci_build/ci_sanity.sh#L2)
* [HTML license example](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tensorboard/dist/index.html#L2)
* [JavaScript/TypeScript license example](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tensorboard/components/tf_backend/backend.ts#L1)

Bazel BUILD files also need to include a license section, e.g.,
[BUILD example](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/BUILD#L61).

#### C++ coding style

Changes to TensorFlow C++ code should conform to
[Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).

Use `clang-tidy` to check your C/C++ changes. To install clang-tidy on ubuntu:16.04, do:

```bash
apt-get install -y clang-tidy
```

You can check a C/C++ file by doing:


```bash
clang-format <my_cc_file> --style=google > /tmp/my_cc_file.cc
diff <my_cc_file> /tmp/my_cc_file.cc
```

#### Python coding style

Changes to TensorFlow Python code should conform to
[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)

Use `pylint` to check your Python changes. To install `pylint` and
retrieve TensorFlow's custom style definition:

```bash
pip install pylint
wget -O /tmp/pylintrc https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/tools/ci_build/pylintrc
```

To check a file with `pylint`:

```bash
pylint --rcfile=/tmp/pylintrc myfile.py
```

#### Coding style for other languages

* [Google Java Style Guide](https://google.github.io/styleguide/javaguide.html)
* [Google JavaScript Style Guide](https://google.github.io/styleguide/jsguide.html)
* [Google Shell Style Guide](https://google.github.io/styleguide/shell.xml)
* [Google Objective-C Style Guide](https://google.github.io/styleguide/objcguide.html)

#### Running sanity check

If you have Docker installed on your system, you can perform a sanity check on
your changes by running the command:

```bash
tensorflow/tools/ci_build/ci_build.sh CPU tensorflow/tools/ci_build/ci_sanity.sh
```

This will catch most license, Python coding style and BUILD file issues that
may exist in your changes.

#### Running unit tests

There are two ways to run TensorFlow unit tests.

1. Using tools and libraries installed directly on your system.

   Refer to the
   [CPU-only developer Dockerfile](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/docker/Dockerfile.devel) and
   [GPU developer Dockerfile](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/docker/Dockerfile.devel-gpu)
   for the required packages. Alternatively, use the said
   [Docker images](https://hub.docker.com/r/tensorflow/tensorflow/tags/), e.g.,
   `tensorflow/tensorflow:nightly-devel` and `tensorflow/tensorflow:nightly-devel-gpu`
   for development to avoid installing the packages directly on your system.

   Once you have the packages installed, you can run a specific unit test in
   bazel by doing as follows:

   If the tests are to be run on GPU, add CUDA paths to LD_LIBRARY_PATH and add
   the `cuda` option flag

   ```bash
   export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH"

   export flags="--config=opt --config=cuda -k"
   ```

   For example, to run all tests under tensorflow/python, do:

   ```bash
   bazel test ${flags} //tensorflow/python/...
   ```

2. Using [Docker](www.docker.com) and TensorFlow's CI scripts.

   ```bash
   # Install Docker first, then this will build and run cpu tests
   tensorflow/tools/ci_build/ci_build.sh CPU bazel test //tensorflow/...
   ```

   See
   [TensorFlow Builds](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/ci_build) for details.

