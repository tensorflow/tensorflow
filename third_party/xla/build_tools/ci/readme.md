This folder contains scripts and commands executed by GitHub actions in OpenXLA,
TensorFlow and JAX OSS repos. The GitHub actions replaced
KokoroPresubmit-tensorflow* workflows. They run under
`Copybara_XLA{Presubmit,Submit}` presubmit chip (category) at Google internally.
The tests run on several target OS configurations/GCP containers such as Linux
x86 with GPU or Linux ARM64. They assure that XLA/Tensor Flow/JAX compiles and
runs on those platforms. The tests uses released OSS c++ clang compiler which
has some differences in supporting c++ standards compared Google's internal
version.

#### How it works

Repo specific GitHub actions call `build.py --build="build_name"`. E.g. OpenXLA
uses https://github.com/openxla/xla/blob/main/.github/workflows/ci.yml

The build here is a set of shell script commands executing the test targets or
doing compile only testing. Each GitHub action call translates into compile only
test:

1.  dryrun `bazel build --nobuild ... test_targets`
1.  actual compile `bazel build ... test_targets`
1.  analyse results `bazel analyze-profile ...`

or compile and run test commands:

1.  dry run `bazel build --nobuild ... test_targets`
1.  actual test `bazel test ... test_targets`
1.  analyse results `bazel analyze-profile profile.json.gz`

Checking in changes to `build.py` regenerates `golden_commands.txt` which lets
us see how commands are changing. `golden_commands.txt` are not called as part
of the continuous integration process.

#### When does it run?

The GitHub actions are automatically called as part of Google presubmit. The
GitHub actions are automatically called on GitHub PR commit if the committer is
part of OpenXLA GitHub org. Committer which are not part of OpenXLA org need an
approval from OpenXLA org member to run the actions.
