#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

if os.getenv("APPVEYOR_REPO_TAG") != "true":
    print("Skip build step. It's not TAG")
else:
    os.system("python conan/build.py")
