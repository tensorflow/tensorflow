add-status-macros.patch adds `ASSERT_OK`, `EXPECT_OK`, `ASSERT_OK_AND_ASSIGN`
to gmock.h so that the header's provided functionality matches internal gmock.

What other things have we tried?

1. Introducing a custom header to be used in OSS instead of `gmock/gmock.h`.

   The export-to-OSS process imposes a few restrictions. Notably, header
   rewrite has to be reversible, so we need a 1:1 mapping between headers used
   internally and in OSS.

   If we introduced a custom header to be used in OSS instead of gmock, it
   would have to take the place of current rewrite of internal gmock to
   `gmock/gmock.h`. This means, any use of `gmock/gmock.h` in OSS XLA code can
   no longer map to internal gmock. We'd have to ban the header.

   Therefore, updating OSS `gmock/gmock.h` seems necessary.

2. Patching in the extra macros to `gmock/gmock.h` by including
   `absl/status/status_macros.h`.

   This introduces a circular dependency between absl and gmock which makes
   bazel strongly opposed to the idea.

3. Introducing a googletest bazel module wrapper.

   This would be a module would proxy all `gmock/gmock.h` within XLA without
   additional patching of googletest. However, having multiple sources of the
   same gmock/gmock.h header path only works *sometimes*. The order of include
   paths emitted by bazel depends on the target definition and ordering of
   dependencies, so it ends up working in some case and not in others.

4. Expanding 3. by renaming googletest's `gmock.h` to `gmock.upstream.h` to
   avoid header name conflicts.

   `gmock/gmock.h` is also included by googletest itself, so redirecting it to
   `gmock/gmock.upstream.h` is needed. That boils down to even more brittle
   patching.

Overall, the add-status-macros.patch change is the least invasive one that
works.
