# buildifier: disable=load-on-top
workspace(name = "xla")

# Initialize the XLA repository and all dependencies.
#
# The cascade of load() statements and xla_workspace?() calls works around the
# restriction that load() statements need to be at the top of .bzl files.
# E.g. we can not retrieve a new repository with http_archive and then load()
# a macro from that repository in the same file.
load(":workspace4.bzl", "xla_workspace4")

xla_workspace4()

load(":workspace3.bzl", "xla_workspace3")

xla_workspace3()

load(":workspace2.bzl", "xla_workspace2")

xla_workspace2()

load(":workspace1.bzl", "xla_workspace1")

xla_workspace1()

load(":workspace0.bzl", "xla_workspace0")

xla_workspace0()
