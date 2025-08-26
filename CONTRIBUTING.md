# Contributing Guidelines

## Pull Request Checklist

Before submitting your pull request, please ensure you have completed the following:

* Read the [Contributing Guidelines](CONTRIBUTING.md).
* Read the [Code of Conduct](CODE_OF_CONDUCT.md).
* Ensure you have signed the [Contributor License Agreement (CLA)](https://cla.developers.google.com/).
* Check that your changes are consistent with the [General Guidelines and Philosophy for Contribution](#general-guidelines-and-philosophy-for-contribution).
* Verify that your changes follow the [Coding Style](#c-coding-style).
* Run the [Unit Tests](#running-unit-tests).

## How to Become a Contributor and Submit Your Code

![Screen Shot 2022-08-30 at 7 27 04 PM](https://user-images.githubusercontent.com/42785357/187579207-9924eb32-da31-47bb-99f9-d8bf1aa238ad.png)

### Typical Pull Request Workflow

**1. New PR**

* As a contributor, you submit a new pull request (PR) on GitHub.
* We inspect every incoming PR and add certain labels such as `size:` or `comp:`.
* At this stage, we check whether the PR meets basic quality requirements. For example, we verify that the CLA is signed, the PR includes a sufficient description, unit tests are added (if applicable), and the contribution is meaningful (i.e., not a single-line cosmetic fix).

**2. Validation**

* If the PR passes quality checks, a reviewer will be assigned.
* If the PR does not meet the validation criteria, we will request additional changes. In rare cases, the PR may be rejected.

**3. Review**

* For a valid PR, a reviewer familiar with the code or functionality will check whether the changes look good or require modifications.
* If changes are requested, you will need to make the updates and resubmit.
* This cycle repeats until the PR is approved.
* *Note: If a PR is awaiting your response for more than two weeks, we may reach out as a friendly reminder.*

**4. Approval**

* Once the PR is approved, the `kokoro:force-run` label is applied, which triggers CI/CD tests.
* If these tests fail, we cannot proceed until the issues are resolved. You may be asked to update your PR to address the failures.
* After the tests pass, the code is imported into the internal codebase using a tool called **Copybara**.

**5. Integration with Google’s Internal Codebase**

* After the PR is imported, additional internal tests are run to ensure compatibility with dependencies and the rest of the system.
* In rare cases, if these tests fail, the code cannot be merged. We may contact you for adjustments, or we may fix the issue internally. Please be patient while we resolve such cases.
* Once internal tests pass, the code is merged both internally and externally on GitHub.

The lifecycle of a PR looks like this:

![image](https://github.com/tensorflow/tensorflow/assets/52792999/3eea4ca5-daa0-4570-b0b5-2a2b03a724a3)

---

### Contributor License Agreements

We’d love to accept your contributions! Before we can, you must complete a Contributor License Agreement (CLA):

* **Individual CLA**: If you are writing original source code and you own the intellectual property, sign the [individual CLA](https://code.google.com/legal/individual-cla-v1.0.html).
* **Corporate CLA**: If you are contributing on behalf of your employer, sign the [corporate CLA](https://code.google.com/legal/corporate-cla-v1.0.html).

Once we receive the signed CLA, we can accept your contributions.

> **Note:** Only original source code from you (or other CLA signers) can be accepted into the repository.

---

### Contributing Code

If you have improvements to TensorFlow, submit a pull request!
For beginners, GitHub provides a helpful [how-to guide on pull requests](https://help.github.com/articles/using-pull-requests/).

* TensorFlow team members will review your pull requests.
* Once approved and after passing CI checks, a team member will apply the `ready to pull` label.
* This signals that your change is being submitted to the internal repository.
* Once merged internally, your PR will be automatically merged on GitHub.

To get started:

* Explore the [issues tab](https://github.com/tensorflow/tensorflow/issues).
* Try a smaller issue marked with [“good first issue”](https://github.com/tensorflow/tensorflow/labels/good%20first%20issue).
* Consider tackling issues labeled [“contributions welcome”](https://github.com/tensorflow/tensorflow/labels/stat%3Acontributions%20welcome).

If you start working on an issue, leave a comment to let others know. If you’d like to collaborate, use the issue thread to coordinate.

---

### Contribution Guidelines and Standards

Before submitting your PR for [review](https://github.com/tensorflow/tensorflow/pulls), ensure your changes align with TensorFlow’s guidelines and coding style.

#### General Guidelines

* **Always include unit tests** for new features and bug fixes. Tests validate correctness and protect against regressions.
* **Maintain API compatibility** in core TensorFlow (e.g., `tensorflow/core` and `tensorflow/python`). TensorFlow cannot make non-backward-compatible API changes outside of a major release. Reviewers will flag API compatibility issues.
* **Consider maintenance costs** — contributions become TensorFlow team responsibilities. A feature’s benefit must outweigh its long-term maintenance burden.
* **Large new features** often first live in [tensorflow/addons](https://github.com/tensorflow/addons) before being considered for core.
* **Avoid trivial PRs** (e.g., fixing one typo). Instead, address all similar issues in a file at once.
* **Follow testing best practices** ([guide here](https://www.tensorflow.org/community/contribute/tests)).

#### License Headers

New files must include a license header:

* [C/C++ example](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/op.cc#L1)
* [Python example](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/nn.py#L1)
* [Java example](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/java/src/main/java/org/tensorflow/Graph.java#L1)
* [Go example](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/go/operation.go#L1)
* [Bash example](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/ci_build/ci_build.sh#L2)
* [JavaScript/TypeScript example](https://github.com/tensorflow/tensorboard/blob/master/tensorboard/components/tf_backend/backend.ts#L1)

Bazel `BUILD` files also need a license section (see [example](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/BUILD#L61)).

#### Coding Style

* **C++**: Follow the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html). Use `clang-tidy` to check changes.
* **Python**: Follow the [Google Python Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md). Use `pylint` with TensorFlow’s config (\`tensorflow/tools/ci\_build/p
