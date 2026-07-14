#!/bin/bash
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# A script to merge Mach-O object files into a single object file and hide
# their internal symbols. Only allowed symbols will be visible in the
# symbol table after this script.

# To run this script, you must set several variables:
#   INPUT_FRAMEWORK: a zip file containing the iOS static framework.
#   BUNDLE_NAME: the pod/bundle name of the iOS static framework.
#   ALLOWLIST_FILE_PATH: contains the allowed symbols.
#   EXTRACT_SCRIPT_PATH: path to the extract_object_files script.
#   OUTPUT: the output zip file.

# Halt on any error or any unknown variable.
set -ue

# mktemp from coreutils has different flags. Make sure we get the iOS one.
MKTEMP=/usr/bin/mktemp

LD_DEBUGGABLE_FLAGS="-x"
# Uncomment the below to get debuggable output. This can only be done for one
# library at a time.
# LD_DEBUGGABLE_FLAGS="-d"

# Exits if C++ symbols are found in the allowlist.
if grep -q "^__Z" "${ALLOWLIST_FILE_PATH}"; then
  echo "ERROR: Failed in symbol hiding. This rule does not permit hiding of" \
       "C++ symbols due to possible serious problems mixing symbol hiding," \
       "shared libraries and the C++ runtime." \
       "More info can be found in go/ios-symbols-hiding." \
       "Please recheck the allowlist and remove C++ symbols:"
  echo "$(grep "^__Z" "${ALLOWLIST_FILE_PATH}")"
  exit 1 # terminate and indicate error
fi
# Unzips the framework zip file into a temp workspace.
framework=$($MKTEMP -t framework -d)
unzip "${INPUT_FRAMEWORK}" -d "${framework}"/

# Executable file in the framework.
executable_file="${BUNDLE_NAME}.framework/${BUNDLE_NAME}"

# Extracts architectures from the framework binary.
archs_str=$(xcrun lipo -info "${framework}/${executable_file}" |
sed -En -e 's/^(Non-|Architectures in the )fat file: .+( is architecture| are): (.*)$/\3/p')

IFS=' ' read -r -a archs <<< "${archs_str}"

merge_cmd=(xcrun lipo)

# Merges object files and hide symbols for each architecture.
for arch in "${archs[@]}"; do
    archdir=$($MKTEMP -t "${arch}" -d)
    arch_file="${archdir}/${arch}"

    # Handles the binary differently if they are fat or thin.
    if [[ "${#archs[@]}" -gt 1 ]]; then
       xcrun lipo "${framework}/${executable_file}" -thin "${arch}" -output "${arch_file}"
    else
       mv "${framework}/${executable_file}" "${arch_file}"
    fi
    if [[ "$arch" == "armv7" ]]; then
      # Check that there are no thread local variables in the input, as they get broken.
      # See b/124533863.
      thread_locals=$(xcrun nm -m -g "${arch_file}" | awk '/__DATA,__thread_vars/ { print $5 }' | c++filt)
      if [[ -n "${thread_locals}" ]]; then
         echo
         echo "WARNING: This symbol hiding script breaks thread local variables on 32-bit arm, you had:"
         echo "${thread_locals}"
         echo
         echo "Your build will crash if these variables are actually used at runtime."
         echo
      fi
    fi
    if [[ ! -z "${EXTRACT_SCRIPT_PATH}" ]]; then
      "${EXTRACT_SCRIPT_PATH}" "${arch_file}" "${archdir}"
    else
      # ar tool extracts the objects in the current working directory. Since the
      # default working directory for a genrule is always the same, there can be
      # a race condition when this script is called for multiple targets
      # simultaneously.
      pushd "${archdir}" > /dev/null
      xcrun ar -x "${arch_file}"
      popd > /dev/null
    fi

    objects_file_list=$($MKTEMP)
    # Hides the symbols except the allowed ones.
    find "${archdir}" -name "*.o" >> "${objects_file_list}"

    # Checks whether bitcode is enabled in the framework.
    all_objects_have_bitcode=true
    for object_file in $(cat "$objects_file_list"); do
      if otool -arch "${arch}" -l "${object_file}" | grep -q __LLVM; then
        : # Do nothing
      else
        echo "The ${arch} in ${object_file} is NOT bitcode-enabled."
        all_objects_have_bitcode=false
        break
      fi
    done
    if [[ "$all_objects_have_bitcode" = "true" ]]; then
      echo "The ${arch} in ${executable_file} is fully bitcode-enabled."
      xcrun ld -r -bitcode_bundle -exported_symbols_list \
        "${ALLOWLIST_FILE_PATH}" \
        $LD_DEBUGGABLE_FLAGS \
        -filelist "${objects_file_list}" -o "${arch_file}_processed.o"
    else
      echo "The ${arch} in ${executable_file} is NOT fully bitcode-enabled."
      xcrun ld -r -exported_symbols_list \
        "${ALLOWLIST_FILE_PATH}" \
        $LD_DEBUGGABLE_FLAGS \
        -filelist "${objects_file_list}" -o "${arch_file}_processed.o"
    fi

    output_object="${framework}/${arch}"

    mv "${arch_file}_processed.o" "${output_object}"
    rm -rf "${archdir}"
    rm "${objects_file_list}"
    merge_cmd+=(-arch "${arch}" "${output_object}")
done

# Repackages the processed object files.
unzip "${INPUT_FRAMEWORK}"
merge_cmd+=(-create -output "${BUNDLE_NAME}")
"${merge_cmd[@]}"

chmod +x "${BUNDLE_NAME}"
rm "${executable_file}"
mv "${BUNDLE_NAME}" "${executable_file}"
( TZ=UTC find "${BUNDLE_NAME}.framework/" -exec touch -h -t 198001010000 {} \+ )
zip --compression-method store --symlinks --recurse-paths --quiet "${OUTPUT}" "${BUNDLE_NAME}.framework/"
