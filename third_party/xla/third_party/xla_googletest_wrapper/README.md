A gtest wrapper that adds ASSERT_OK, EXPECT_OK, ASSERT_OK_AND_ASSIGN to gmock.h
so that the header's provided functionality matches internal gmock.

The repo contains a minimal set of reexports necessary to build XLA with this as
a drop-in replacement for googletest.