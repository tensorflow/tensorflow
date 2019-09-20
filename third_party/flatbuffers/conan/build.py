#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import re
import subprocess
from cpt.packager import ConanMultiPackager


def set_appveyor_environment():
    if os.getenv("APPVEYOR") is not None:
        compiler_version = os.getenv("CMAKE_VS_VERSION").split(" ")[0].replace('"', '')
        os.environ["CONAN_VISUAL_VERSIONS"] = compiler_version
        os.environ["CONAN_STABLE_BRANCH_PATTERN"] = "master"
        ci_platform = os.getenv("Platform").replace('"', '')
        ci_platform = "x86" if ci_platform == "x86" else "x86_64"
        os.environ["CONAN_ARCHS"] = ci_platform
        os.environ["CONAN_BUILD_TYPES"] = os.getenv("Configuration").replace('"', '')


def get_branch():
    try:
        for line in subprocess.check_output("git branch", shell=True).decode().splitlines():
            line = line.strip()
            if line.startswith("*") and " (HEAD detached" not in line:
                return line.replace("*", "", 1).strip()
        return ""
    except Exception:
        pass
    return ""


def get_version():
    version = get_branch()
    if os.getenv("TRAVIS", False):
        version = os.getenv("TRAVIS_BRANCH")

    if os.getenv("APPVEYOR", False):
        version = os.getenv("APPVEYOR_REPO_BRANCH")
        if os.getenv("APPVEYOR_REPO_TAG") == "true":
            version = os.getenv("APPVEYOR_REPO_TAG_NAME")

    match = re.search(r"v(\d+\.\d+\.\d+.*)", version)
    if match:
        return match.group(1)
    return version


def get_reference(username):
    return "flatbuffers/{}@google/stable".format(get_version())


if __name__ == "__main__":
    login_username = os.getenv("CONAN_LOGIN_USERNAME", "aardappel")
    username = os.getenv("CONAN_USERNAME", "google")
    upload = os.getenv("CONAN_UPLOAD", "https://api.bintray.com/conan/aardappel/flatbuffers")
    stable_branch_pattern = os.getenv("CONAN_STABLE_BRANCH_PATTERN", r"v\d+\.\d+\.\d+.*")
    test_folder = os.getenv("CPT_TEST_FOLDER", os.path.join("conan", "test_package"))
    upload_only_when_stable = os.getenv("CONAN_UPLOAD_ONLY_WHEN_STABLE", True)
    set_appveyor_environment()

    builder = ConanMultiPackager(reference=get_reference(username),
                                 username=username,
                                 login_username=login_username,
                                 upload=upload,
                                 stable_branch_pattern=stable_branch_pattern,
                                 upload_only_when_stable=upload_only_when_stable,
                                 test_folder=test_folder)
    builder.add_common_builds(pure_c=False)
    builder.run()
