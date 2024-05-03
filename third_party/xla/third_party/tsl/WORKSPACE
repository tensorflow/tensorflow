# buildifier: disable=load-on-top
workspace(name = "tsl")

# Initialize the TSL repository and all dependencies.
#
# The cascade of load() statements and tsl_workspace?() calls works around the
# restriction that load() statements need to be at the top of .bzl files.
# E.g. we can not retrieve a new repository with http_archive and then load()
# a macro from that repository in the same file.

# buildifier: disable=load-on-top

load(":workspace3.bzl", "tsl_workspace3")

tsl_workspace3()

load(":workspace2.bzl", "tsl_workspace2")

tsl_workspace2()

load(":workspace1.bzl", "tsl_workspace1")

tsl_workspace1()

load(":workspace0.bzl", "tsl_workspace0")

tsl_workspace0()
