workspace(name = "org_tensorflow")

# Initialize the TensorFlow repository and all dependencies.
#
# The cascade of load() statements and workspace() calls works around the
# restriction that load() statements need to be at the top of .bzl files.
# E.g. we can not retrieve a new repository with http_archive and then load()
# a macro from that repository in the same file.
load("@//tensorflow:workspace3.bzl", "workspace")
workspace()
load("@//tensorflow:workspace2.bzl", "workspace")
workspace()
load("@//tensorflow:workspace1.bzl", "workspace")
workspace()
load("@//tensorflow:workspace0.bzl", "workspace")
workspace()




