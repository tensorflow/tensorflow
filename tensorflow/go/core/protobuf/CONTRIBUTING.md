# Contributing to Go Protocol Buffers

Go protocol buffers is an open source project and accepts contributions.
The source of truth for this repository is at
[go.googlesource.com/protobuf](https://go.googlesource.com/protobuf).
The code review tool used is
[Gerrit Code Review](https://www.gerritcodereview.com/).
At this time, we are unfortunately unable to accept GitHub pull requests.


## Becoming a contributor

The first step is to configure your environment.
Please follow the steps outlined in
["Becoming a contributor" (golang.org)](https://golang.org/doc/contribute.html#contributor)
as the setup for contributing to the `protobuf` project is identical
to that for contributing to the `go` project.


## Before contributing code

The project welcomes submissions, but to make sure things are well coordinated
we ask that contributors discuss any significant changes before starting work.
Best practice is to connect your work to the
[issue tracker](https://github.com/golang/protobuf/issues),
either by filing a new issue or by claiming an existing issue.


## Sending a change via Gerrit

The `protobuf` project performs development in Gerrit.
Below are the steps to send a change using Gerrit.


**Step 1:** Clone the Go source code:
```
$ git clone https://go.googlesource.com/protobuf
```

**Step 2:** Setup a Git hook:
Setup a hook to run the tests prior to submitting changes to Gerrit:
```
$ (cd protobuf/.git/hooks && echo -e '#!/bin/bash\n./test.bash' > pre-push && chmod a+x pre-push)
```

**Step 3:** Prepare changes in a new branch, created from the `master` branch.
To commit the changes, use `git codereview change`;
that will create or amend a single commit in the branch.

```
$ git checkout -b mybranch
$ [edit files...]
$ git add [files...]
$ git codereview change   # create commit in the branch
$ [edit again...]
$ git add [files...]
$ git codereview change   # amend the existing commit with new changes
$ [etc.]
```

**Step 4:** Send the changes for review to Gerrit using `git codereview mail`.
```
$ git codereview mail     # send changes to Gerrit
```

**Step 5:** After a review, there may be changes that are required.
Do so by applying changes to the same commit and mail them to Gerrit again:
```
$ [edit files...]
$ git add [files...]
$ git codereview change   # update same commit
$ git codereview mail     # send to Gerrit again
```

When calling `git codereview mail`, it will call `git push` under the hood,
which will trigger the test hook that was setup in step 2.

The [Contribution Guidelines](https://golang.org/doc/contribute.html) for the
Go project provides additional details that are also relevant to
contributing to the Go `protobuf` project.
