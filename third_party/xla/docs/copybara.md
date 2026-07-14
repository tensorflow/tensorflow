# Copybara quirks

The purpose of this document is to describe oddities you might see while
contributing due to the tool that manages copying source back and forth from
Google's internal repository. This tool is called [Copybara](https://github.com/google/copybara).

## Internal source of truth

Because the source of truth for the code in this repository is Google's internal
repo, Copybara does transformations to the code whenever the code is imported
and exported. This means that sometimes seemingly normal changes can break
internally in surprising ways.

## PR merge status and diff inconsistencies

Since the source of truth is internal, PRs are not merged directly, they are
imported to the Google internal repo where they undergo additional testing,
and then that internal change is submitted, and attributed to the PR author.
Because of the transformations that Copybara applies, there's no guarantee that
the diff will be identical (for example, Copybara applies formatting on import).

For this reason, Copybara won't mark the PR as merged, it will close the PR and
separately apply a commit that should map very closely to the PR.

## Dependency on TSL by copy

As implemented currently, to prevent any temporary broken commits, XLA
depends on TSL not by downloading a copy by using Bazel's `http_archive`, but by
having Copybara copy TSL into XLA's `third_party` directory.
