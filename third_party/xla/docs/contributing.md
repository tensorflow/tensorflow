# Contributing to OpenXLA

Everyone can contribute to OpenXLA, and we value everyone’s contributions. There
are several ways to contribute, including:

*   Answering questions on OpenXLA’s discussions forums (openxla-discuss)

*   Improving or expanding OpenXLA’s documentation

*   Contributing to OpenXLA’s code-base

*   Contributing in any of the above ways to the broader ecosystem of libraries
built on OpenXLA

The OpenXLA project follows
[Google’s Open Source Community Guidelines](https://opensource.google/conduct/).

## Before you begin

### Sign the Contributor License Agreement

Contributions to this project must be accompanied by a
[Contributor License Agreement](https://cla.developers.google.com/about) (CLA).
You (or your employer) retain the copyright to your contribution; this simply
gives us permission to use and redistribute your contributions as part of the
project.

If you or your current employer have already signed the Google CLA (even if it
was for a different project), you probably don't need to do it again.

Visit <https://cla.developers.google.com/> to see your current agreements or to
sign a new one.

### Review the Code of Conduct

This project follows
[Tensorflow's Code of Conduct](https://github.com/tensorflow/tensorflow/blob/master/CODE_OF_CONDUCT.md).

## Contribution process

### Developer Guide

For a guide on how to setup a development environment for OpenXLA, including getting
code, building it, running tests and submitting changes, please refer to the
[Developer guide](docs/developer_guide.md).

### Code standards

*   *Coding style*: We follow [Google's code style guide](https://google.github.io/styleguide/).
    Specifically see the [C/C++](https://google.github.io/styleguide/cppguide.html) and [Python](https://google.github.io/styleguide/pyguide.html) guides. All
    code submitted must strictly conform to these style guides.

*   *Compact changes*: We follow
    [Google's engineering practices](https://google.github.io/eng-practices/).
    In particular, please observe the
    [guide on writing compact changes](https://google.github.io/eng-practices/review/developer/small-cls.html).
    Doing so will greatly increase the speed at which you can get your code
    merged due to improve reviewability, and reducing the likelihood of
    unintentional side effects of change. Even if you have a large change, there
    are many strategies for breaking it up into more incremental changes.

*   *Test Coverage*: All changes should include appropriate unit tests. Unit
    tests should not be dependent on specific hardware (CPU, GPU, etc.) timings,
    and should make liberal use of mocks and fakes in order to make
    deterministic and focused tests. Changes seeking to extend existing code
    that’s currently hard to test should make appropriate improvements to
    testability.

    All changes should include appropriate benchmark results as well in the
    change title to ensure the benefits are clearly understood.

*   When in doubt as to conventions within the code, it is always a good idea to
    examine pre-existing code and to try to follow the patterns already in place
    in OpenXLA.


### Review Process

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

*   Code must follow all standards listed above prior to review. These are not
    optional and it is critical that the submitter ensure their code conforms
    before requesting review in order to assure timely acceptance of changes.

*   *All tests must pass*. If you find that a test is broken and the issue is not
    related to your build environment or otherwise your changes, please contact
    the maintainers.

*   Try to avoid scope creep during the review process. This is the
    responsibility of both the submitter and the reviewer. If a change starts to
    get too large, consider breaking it up into multiple changes.

*   Before a change is merged, it will undergo internal testing that uses code
    internal to Google and other hardware vendors. This can potentially add extra
    steps to the review process if there are failures on internal tests that our
    public CI doesn't catch. The Googler reviewing your change will communicate
    any internal test failures and describe what needs to be fixed.


## Frequently asked questions (FAQ)

### "This infrastructure change is not related to my PR. Why should I do it?"

The XLA team doesn't have a dedicated infrastructure team, so it's up to us all
to build helper libraries and avoid technical debt. We consider it to be a
regular part of making changes to XLA, and everyone is expected to participate.
We generally build infrastructure as needed when writing code.

XLA reviewers may ask you to build some infrastructure (or otherwise make a
large change to a PR) along with a PR that you've written. This request may seem
unnecessary or orthogonal to the change you're trying to make. This is likely
because of a mismatch between your expectations about how much infra you need to
build and your reviewer's expectations for the same.

A mismatch in expectations is okay! That's expected when you're new to a project
(and it sometimes even happens to us old hats). It's likely that projects you've
worked on in the past have different expectations. That's also okay and
expected! It doesn't mean either one of these projects has the wrong approach;
they're just different. We invite you to take infra requests alongside all other
review comments as an opportunity to learn what we expect on this project.

### "Can I address your comment in a future PR?"

A frequent question with respect to infrastructure requests (or other large
requests) in PRs is whether or not the change must be made in the original PR,
or whether it can be done as a follow-up in a future PR.

In general, XLA does not allow PR authors to address review comments with a
follow-up PR. When a reviewer decides that something needs to be addressed in a
given PR, we generally expect authors to address it in that PR, even if what's
requested is a large change. This standard applies externally and also
internally within Google.

There are a few reasons that XLA takes this approach.

*   *Trust:* Having earned the reviewer's trust is a key component. In an
    open-source project, contributors can appear or disappear at will. After we
    approve a PR, reviewers have no way to ensure that any promised follow-ups
    actually get done.

*   *Impact on other developers:* If you have sent a PR touching a particular
    part of XLA, there's a good chance other people are looking at the same
    part. If we accept technical debt in your PR, then everyone who's looking at
    this file will be impacted by this debt until the follow-up is submitted.

*   *Reviewer bandwidth:* Deferring a change to a follow-up imposes multiple
    costs on our already overloaded reviewers. Reviewers will probably forget
    what the first PR was about while waiting for the follow-up, making the next
    review more difficult. Also, reviewers will have to keep track of expected
    follow-ups, making sure that they actually happen. If the change can be made
    such that it is truly orthogonal to the original PR so that some other
    reviewer could review it, bandwidth would be less of a problem. In our
    experience, this is rarely the case.
