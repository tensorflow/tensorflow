# buildifier: disable=load-on-top

workspace(name = "org_tensorflow")

# buildifier: disable=load-on-top

# Initialize the TensorFlow repository and all dependencies.
#
# The cascade of load() statements and tf_workspace?() calls works around the
# restriction that load() statements need to be at the top of .bzl files.
# E.g. we can not retrieve a new repository with http_archive and then load()
# a macro from that repository in the same file.
load("@//tensorflow:workspace3.bzl", "tf_workspace3")

# Initialize TensorFlow workspace
tf_workspace3()

# Initialize hermetic Python
load("@local_xla//third_party/py:python_init_rules.bzl", "python_init_rules")

python_init_rules()

load("@local_xla//third_party/py:python_init_repositories.bzl", "python_init_repositories")

python_init_repositories(
    default_python_version = "system",
    requirements = {
        "3.9": "//:requirements_lock_3_9.txt",
        "3.10": "//:requirements_lock_3_10.txt",
        "3.11": "//:requirements_lock_3_11.txt",
        "3.12": "//:requirements_lock_3_12.txt",
    },
)

load("@local_xla//third_party/py:python_init_toolchains.bzl", "python_init_toolchains")

python_init_toolchains()

load("@local_xla//third_party/py:python_init_pip.bzl", "python_init_pip")

python_init_pip()

load("@pypi//:requirements.bzl", "install_deps")

# Install dependencies
install_deps()

# Documentation
# Add comments to explain the purpose of each section or step in the code
# For example:
# This section initializes the TensorFlow workspace and dependencies

# Error Handling
# Include error handling mechanisms to gracefully handle any failures that may occur during the initialization process

# Dependency Updates
# Regularly check for updates to the dependencies like rules_java and TensorFlow, and update the versions accordingly

# Testing
# Define test targets to ensure the setup is working correctly and dependencies are properly configured

# Build Targets
# Define build targets for your specific project within this workspace to build and run TensorFlow applications

# Environment Configuration
# Set up specific environment variables or configurations needed for TensorFlow

# Version Control
# Ensure that this workspace configuration is properly version-controlled to track changes and facilitate collaboration

# Optimizations
# Look for opportunities to optimize the build process, such as caching dependencies or parallelizing build tasks
