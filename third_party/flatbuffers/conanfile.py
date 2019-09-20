#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Conan recipe package for Google FlatBuffers
"""
import os
import shutil
from conans import ConanFile, CMake, tools


class FlatbuffersConan(ConanFile):
    name = "flatbuffers"
    license = "Apache-2.0"
    url = "https://github.com/google/flatbuffers"
    homepage = "http://google.github.io/flatbuffers/"
    author = "Wouter van Oortmerssen"
    topics = ("conan", "flatbuffers", "serialization", "rpc", "json-parser")
    description = "Memory Efficient Serialization Library"
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False], "fPIC": [True, False]}
    default_options = {"shared": False, "fPIC": True}
    generators = "cmake"
    exports = "LICENSE.txt"
    exports_sources = ["CMake/*", "include/*", "src/*", "grpc/*", "CMakeLists.txt", "conan/CMakeLists.txt"]

    def source(self):
        """Wrap the original CMake file to call conan_basic_setup
        """
        shutil.move("CMakeLists.txt", "CMakeListsOriginal.txt")
        shutil.move(os.path.join("conan", "CMakeLists.txt"), "CMakeLists.txt")

    def config_options(self):
        """Remove fPIC option on Windows platform
        """
        if self.settings.os == "Windows":
            self.options.remove("fPIC")

    def configure_cmake(self):
        """Create CMake instance and execute configure step
        """
        cmake = CMake(self)
        cmake.definitions["FLATBUFFERS_BUILD_TESTS"] = False
        cmake.definitions["FLATBUFFERS_BUILD_SHAREDLIB"] = self.options.shared
        cmake.definitions["FLATBUFFERS_BUILD_FLATLIB"] = not self.options.shared
        cmake.configure()
        return cmake

    def build(self):
        """Configure, build and install FlatBuffers using CMake.
        """
        cmake = self.configure_cmake()
        cmake.build()

    def package(self):
        """Copy Flatbuffers' artifacts to package folder
        """
        cmake = self.configure_cmake()
        cmake.install()
        self.copy(pattern="LICENSE.txt", dst="licenses")
        self.copy(pattern="FindFlatBuffers.cmake", dst=os.path.join("lib", "cmake", "flatbuffers"), src="CMake")
        self.copy(pattern="flathash*", dst="bin", src="bin")
        self.copy(pattern="flatc*", dst="bin", src="bin")
        if self.settings.os == "Windows" and self.options.shared:
            if self.settings.compiler == "Visual Studio":
                shutil.move(os.path.join(self.package_folder, "lib", "%s.dll" % self.name),
                            os.path.join(self.package_folder, "bin", "%s.dll" % self.name))
            elif self.settings.compiler == "gcc":
                shutil.move(os.path.join(self.package_folder, "lib", "lib%s.dll" % self.name),
                            os.path.join(self.package_folder, "bin", "lib%s.dll" % self.name))

    def package_info(self):
        """Collect built libraries names and solve flatc path.
        """
        self.cpp_info.libs = tools.collect_libs(self)
        self.user_info.flatc = os.path.join(self.package_folder, "bin", "flatc")
