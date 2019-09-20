#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

if os.getenv("APPVEYOR_REPO_TAG") != "true":
    print("Skip step. It's not TAG")
else:
    os.system("pip install conan conan-package-tools")
