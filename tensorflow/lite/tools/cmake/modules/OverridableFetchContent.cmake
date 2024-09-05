#
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

include(FetchContent)

# Pairs of regex --> replacement strings that map Git repositories to archive
# URLs. GIT_COMMIT is replaced with the hash of the commit.
set(OVERRIDABLE_FETCH_CONTENT_GITHUB_MATCH
  "^https?://github.com/([^/]+)/([^/.]+)(\\.git)?\$"
)
set(OVERRIDABLE_FETCH_CONTENT_GITHUB_REPLACE
  "https://github.com/\\1/\\2/archive/GIT_COMMIT.zip"
)
set(OVERRIDABLE_FETCH_CONTENT_GITLAB_MATCH
  "^https?://gitlab.com/([^/]+)/([^/.]+)(\\.git)?"
)
set(OVERRIDABLE_FETCH_CONTENT_GITLAB_REPLACE
  "https://gitlab.com/\\1/\\2/-/archive/GIT_COMMIT/\\2-GIT_COMMIT.tar.gz"
)
set(OVERRIDABLE_FETCH_CONTENT_GOOGLESOURCE_MATCH
  "^(https?://[^.]+\\.googlesource\\.com/.*)"
)
set(OVERRIDABLE_FETCH_CONTENT_GOOGLESOURCE_REPLACE
  "\\1/+archive/GIT_COMMIT.tar.gz"
)
# List of prefixes for regex match and replacement variables that map Git
# repositories to archive URLs.
list(APPEND OVERRIDABLE_FETCH_CONTENT_GIT_TRANSFORMS
  OVERRIDABLE_FETCH_CONTENT_GITHUB
  OVERRIDABLE_FETCH_CONTENT_GITLAB
  OVERRIDABLE_FETCH_CONTENT_GOOGLESOURCE
)

# Pairs of regex --> replacement strings that map Git repositories to raw file
# URLs.
set(OVERRIDABLE_FETCH_CONTENT_GITHUB_FILE_MATCH
  "${OVERRIDABLE_FETCH_CONTENT_GITHUB_MATCH}"
)
set(OVERRIDABLE_FETCH_CONTENT_GITHUB_FILE_REPLACE
  "https://raw.githubusercontent.com/\\1/\\2/GIT_COMMIT/FILE_PATH"
)
set(OVERRIDABLE_FETCH_CONTENT_GITLAB_FILE_MATCH
  "${OVERRIDABLE_FETCH_CONTENT_GITLAB_MATCH}"
)
set(OVERRIDABLE_FETCH_CONTENT_GITLAB_FILE_REPLACE
  "https://gitlab.com/\\1/\\2/-/raw/GIT_COMMIT/FILE_PATH"
)
set(OVERRIDABLE_FETCH_CONTENT_GOOGLESOURCE_FILE_MATCH
  "${OVERRIDABLE_FETCH_CONTENT_GOOGLESOURCE_MATCH}"
)
# This isn't the raw file, gitiles doesn't support raw file download without
# decoding the file from base64.
set(OVERRIDABLE_FETCH_CONTENT_GOOGLESOURCE_FILE_REPLACE
  "\\1/+/GIT_COMMIT/FILE_PATH"
)

# List of prefixes for regex match and replacement variables that map Git
# repositories to archive URLs.
list(APPEND OVERRIDABLE_FETCH_CONTENT_GIT_FILE_TRANSFORMS
  OVERRIDABLE_FETCH_CONTENT_GITHUB_FILE
  OVERRIDABLE_FETCH_CONTENT_GITLAB_FILE
  OVERRIDABLE_FETCH_CONTENT_GOOGLESOURCE_FILE
)

# Try applying replacements to string.
#
# TRANSFORMS: List which contains prefixes for  _MATCH / _REPLACE replacements
# to try. For example, given the list "FOO" this will try to apply a regex
# replacement with the value of FOO_MATCH and FOO_REPLACE.
# TO_REPLACE: String to apply replacements to.
# OUTPUT_VAR: Name of the variable to store the URL if successful. If
# conversion fails this variable will be empty.
function(_ApplyReplacements TRANSFORMS TO_REPLACE OUTPUT_VAR)
  foreach(PREFIX ${TRANSFORMS})
    message(VERBOSE "Try converting ${GIT_REPOSITORY} with ${${PREFIX}_MATCH}")
    set(MATCH "${${PREFIX}_MATCH}")
    set(REPLACE "${${PREFIX}_REPLACE}")
    if(MATCH AND REPLACE)
      string(REGEX REPLACE
        "${MATCH}"
        "${REPLACE}"
        REPLACED
        "${TO_REPLACE}"
      )
      if(NOT "${REPLACED}" STREQUAL "${TO_REPLACE}")
        set(${OUTPUT_VAR} "${REPLACED}" PARENT_SCOPE)
      endif()
    endif()
  endforeach()
endfunction()


# Try to convert a Git repository to an archive URL.
#
# GIT_REPOSITORY: Repository URL to convert.
# GIT_COMMIT: Commit hash or tag to convert.
# REPORT_WARNING: Whether to report a warning if conversion fails.
# OUTPUT_VAR: Name of the variable to store the URL if successful. If
# conversion fails this variable will be empty.
function(_GitRepoArchiveUrl GIT_REPOSITORY GIT_COMMIT REPORT_WARNING OUTPUT_VAR)
  list(REMOVE_DUPLICATES OVERRIDABLE_FETCH_CONTENT_GIT_TRANSFORMS)
  _ApplyReplacements(
    "${OVERRIDABLE_FETCH_CONTENT_GIT_TRANSFORMS}"
    "${GIT_REPOSITORY}"
    REPLACED
  )
  if(REPLACED)
    string(REPLACE "GIT_COMMIT" "${GIT_COMMIT}" WITH_COMMIT "${REPLACED}")
    message(VERBOSE "${GIT_REPOSITORY} / ${GIT_COMMIT} --> ${WITH_COMMIT}")
    set(${OUTPUT_VAR} "${WITH_COMMIT}" PARENT_SCOPE)
  elseif(REPORT_WARNING)
    message(WARNING
      "Unable to map ${GIT_REPOSITORY} / ${GIT_COMMIT} to an archive URL"
    )
  endif()
endfunction()


# Try to convert a Git repository, commit and relative path to a link to the
# file.
#
# GIT_REPOSITORY: Repository URL to convert.
# GIT_COMMIT: Commit hash or tag to convert.
# FILE_PATH: Path to the file.
# OUTPUT_VAR: Name of the variable to store the URL if successful. If
# conversion fails this variable will be empty.
function(_GitRepoFileUrl GIT_REPOSITORY GIT_COMMIT FILE_PATH OUTPUT_VAR)
  list(REMOVE_DUPLICATES OVERRIDABLE_FETCH_CONTENT_GIT_FILE_TRANSFORMS)
  _ApplyReplacements(
    "${OVERRIDABLE_FETCH_CONTENT_GIT_FILE_TRANSFORMS}"
    "${GIT_REPOSITORY}"
    REPLACED
  )
  if(REPLACED)
    string(REPLACE "GIT_COMMIT" "${GIT_COMMIT}" WITH_COMMIT "${REPLACED}")
    string(REPLACE "FILE_PATH" "${FILE_PATH}" WITH_FILE "${WITH_COMMIT}")
    message(VERBOSE
      "${GIT_REPOSITORY} / ${GIT_COMMIT} / ${FILE_PATH} --> ${WITH_FILE}"
    )
    set(${OUTPUT_VAR} "${WITH_FILE}" PARENT_SCOPE)
  else()
    message(WARNING
      "Unable to map ${GIT_REPOSITORY} / ${GIT_COMMIT} / ${FILE_PATH} to a URL"
    )
  endif()
endfunction()


# Try to determine the license URL from a path within the content and
# cache LICENSE_FILE and LICENSE_URL properties.
#
# CONTENT_NAME: Name of the content that hosts the license.
# LICENSE_FILE: Relative path in the archive.
# OUTPUT_VAR: Name of variable to store / retrieve the license URL.
function(_LicenseFileToUrl CONTENT_NAME LICENSE_FILE OUTPUT_VAR)
  foreach(PROPERTY GIT_REPOSITORY GIT_COMMIT LICENSE_URL)
    _OverridableFetchContent_GetProperty(
      "${CONTENT_NAME}"
      "${PROPERTY}"
      "${PROPERTY}"
    )
  endforeach()
  _OverridableFetchContent_SetProperty(
    "${CONTENT_NAME}"
    LICENSE_FILE
    "License for ${CONTENT_NAME}"
    "${LICENSE_FILE}"
  )
  if(NOT LICENSE_URL)
    if(GIT_REPOSITORY AND GIT_COMMIT)
      # Try to synthesize the license URL from the repo path.
      _GitRepoFileUrl(
        "${GIT_REPOSITORY}"
        "${GIT_COMMIT}"
        "${LICENSE_FILE}"
        LICENSE_URL
      )
      if(LICENSE_URL)
        _OverridableFetchContent_SetProperty(
          "${CONTENT_NAME}"
          LICENSE_URL
          "License URL for ${CONTENT_NAME}"
          "${LICENSE_URL}"
        )
        set(${OUTPUT_VAR} "${LICENSE_URL}" PARENT_SCOPE)
      endif()
    endif()
  endif()
endfunction()


# Replacement for FetchContent_Declare() that allows the user to override the
# download URL for Git and URL sources and also favor fetching via URL vs.
# a Git repo using variables external to this method.
#
# See FetchContent_Declare() and ExternalProject_Add() for the arguments
# supported by this method.
#
# In addition to FetchContent_Declare() and ExternalProject_Add() arguments,
# this method supports LICENSE_FILE that enables the caller to specify the
# relative path of the license in the downloaded archive which disables the
# search for a license in OverridableFetchContent_Populate().
# LICENSE_URL can be specified to override the URL of the LICENSE_FILE if
# a direct link to the URL can't be formed from the download path.
#
# It's possible to override, GIT_REPOSITORY, GIT_TAG, URL and URL_HASH for
# a target by setting
# OVERRIDABLE_FETCH_CONTENT_<contentName>_<variable> where <contentName> is the
# CONTENT_NAME argument content provided to this method and <variable> is the
# argument of this method to override. For example, given CONTENT_NAME = foo
# the GIT_REPOSITORY can be overridden by setting foo_GIT_REPOSITORY to the
# value to use instead.
#
# To convert a GIT_REPOSITORY / GIT_TAG reference to a URL,
# set OVERRIDABLE_FETCH_CONTENT_GIT_REPOSITORY_AND_TAG_TO_URL_<contentName>
# to ON for one repository or
# OVERRIDABLE_FETCH_CONTENT_GIT_REPOSITORY_AND_TAG_TO_URL to ON for all
# repositories. This will, where possible, convert a GIT_REPOSITORY / GIT_TAG
# reference to a URL to download instead which is much faster to copy than
# cloning a git repo.
#
# If OVERRIDABLE_FETCH_CONTENT_USE_GIT is ON, when a GIT_REPOSITORY and a
# download URL are specified this method will clone the GIT_REPOSITORY. When
# OVERRIDABLE_FETCH_CONTENT_USE_GIT is OFF or not set and both GIT_REPOSITORY
# and download URL are specified the download URL is used instead.
#
# To override the archive URL before it's passed to FetchContent_Declare()
# set OVERRIDABLE_FETCH_CONTENT_<contentName>_MATCH to a regular expression
# to match the archive URL and OVERRIDABLE_FETCH_CONTENT_<contentName>_REPLACE
# with the string to replace the archive URL.
#
# All content names passed to this method are added to the global property
# OVERRIDABLE_FETCH_CONTENT_LIST.
function(OverridableFetchContent_Declare CONTENT_NAME)
  set(OVERRIDABLE_ARGS
    GIT_REPOSITORY
    GIT_TAG
    URL
    URL_HASH
    URL_MD5
  )
  set(ALL_VALUE_ARGS LICENSE_FILE LICENSE_URL ${OVERRIDABLE_ARGS})
  cmake_parse_arguments(ARGS
    ""
    "${ALL_VALUE_ARGS}"
    ""
    ${ARGN}
  )
  # Optionally override parsed arguments with values from variables in the form
  # ${CONTENT_NAME}_${OVERRIDABLE_ARG}.
  foreach(OVERRIDABLE_ARG in ${OVERRIDABLE_ARGS})
    set(OVERRIDE_VALUE
      ${OVERRIDABLE_FETCH_CONTENT_${CONTENT_NAME}_${OVERRIDABLE_ARG}}
    )
    if(NOT "${OVERRIDE_VALUE}" STREQUAL "")
      set(ARGS_${OVERRIDABLE_ARG} "${OVERRIDE_VALUE}")
      message(VERBOSE "Overriding ${OVERRIDABLE_ARG} of content "
        "${CONTENT_NAME} with '${OVERRIDE_VALUE}'"
      )
    endif()
  endforeach()

  # If specified, save the source URL so it's possible to synthesize a link to
  # the license when the content is populated.
  if(ARGS_GIT_REPOSITORY AND ARGS_GIT_TAG)
    _OverridableFetchContent_SetProperty(
      "${CONTENT_NAME}"
      GIT_REPOSITORY
      "Git repo for ${CONTENT_NAME}"
      "${ARGS_GIT_REPOSITORY}"
    )
    _OverridableFetchContent_SetProperty(
      "${CONTENT_NAME}"
      GIT_COMMIT
      "Git commit for ${CONTENT_NAME}"
      "${ARGS_GIT_TAG}"
    )
  endif()

  # Set the license file and URL properties.
  if(ARGS_LICENSE_URL)
    _OverridableFetchContent_SetProperty(
      "${CONTENT_NAME}"
      LICENSE_URL
      "License URL for ${CONTENT_NAME}"
      "${ARGS_LICENSE_URL}"
    )
  endif()
  if(ARGS_LICENSE_FILE)
    _LicenseFileToUrl(
      "${CONTENT_NAME}"
      "${ARGS_LICENSE_FILE}"
      ARGS_LICENSE_URL
    )
  endif()

  # Try mapping to an archive URL.
  set(ARCHIVE_URL "")
  if(ARGS_GIT_REPOSITORY AND ARGS_GIT_TAG)
    _GitRepoArchiveUrl(
      "${ARGS_GIT_REPOSITORY}"
      "${ARGS_GIT_TAG}"
      OFF
      ARCHIVE_URL
    )
    # If conversion from git repository to archive URL is enabled.
    if(OVERRIDABLE_FETCH_CONTENT_GIT_REPOSITORY_AND_TAG_TO_URL_${CONTENT_NAME}
       OR OVERRIDABLE_FETCH_CONTENT_GIT_REPOSITORY_AND_TAG_TO_URL)
      # Try converting to an archive URL.
      if(NOT ARGS_URL)
        _GitRepoArchiveUrl(
          "${ARGS_GIT_REPOSITORY}"
          "${ARGS_GIT_TAG}"
          ON
          ARGS_URL
        )
        set(ARCHIVE_URL "${ARGS_URL}")
      endif()
    endif()
  endif()

  # If a download URL and git repository with tag are specified either use
  # the git repo or the download URL.
  if(ARGS_URL AND ARGS_GIT_REPOSITORY)
    if(OVERRIDABLE_FETCH_CONTENT_USE_GIT)
      unset(ARGS_URL)
      unset(ARGS_URL_HASH)
      unset(ARGS_URL_MD5)
    else()
      unset(ARGS_GIT_REPOSITORY)
      unset(ARGS_GIT_TAG)
    endif()
  endif()

  # Optionally map the archive URL to a mirror.
  if(ARGS_URL)
    _ApplyReplacements(
      "OVERRIDABLE_FETCH_CONTENT_${CONTENT_NAME}"
      "${ARGS_URL}"
      REPLACED
    )
    if(REPLACED)
      set(ARGS_URL "${REPLACED}")
    endif()
  endif()

  # Save the archive URL.
  if(ARGS_URL)
    set(ARCHIVE_URL "${ARGS_URL}")
  endif()
  if(ARCHIVE_URL)
    _OverridableFetchContent_SetProperty(
      "${CONTENT_NAME}"
      ARCHIVE_URL
      "Archive URL for ${CONTENT_NAME}"
      "${ARCHIVE_URL}"
    )
  endif()

  # Build the list of arguments to pass to FetchContent_Declare() starting with
  # the overridable arguments.
  set(OUTPUT_ARGS "")
  foreach(OVERRIDABLE_ARG ${OVERRIDABLE_ARGS})
    set(OVERRIDABLE_ARG_VALUE "${ARGS_${OVERRIDABLE_ARG}}")
    if(OVERRIDABLE_ARG_VALUE)
      list(APPEND OUTPUT_ARGS ${OVERRIDABLE_ARG} "${OVERRIDABLE_ARG_VALUE}")
    endif()
  endforeach()
  list(APPEND OUTPUT_ARGS ${ARGS_UNPARSED_ARGUMENTS})

  # Add all defined packages to a global property.
  get_property(OVERRIDABLE_FETCH_CONTENT_LIST GLOBAL PROPERTY
    OVERRIDABLE_FETCH_CONTENT_LIST
  )
  set(DOCUMENTATION "List of all fetched content")
  define_property(GLOBAL PROPERTY OVERRIDABLE_FETCH_CONTENT_LIST
    BRIEF_DOCS "${DOCUMENTATION}"
    FULL_DOCS "${DOCUMENTATION}"
  )
  list(APPEND OVERRIDABLE_FETCH_CONTENT_LIST "${CONTENT_NAME}")
  set_property(GLOBAL PROPERTY OVERRIDABLE_FETCH_CONTENT_LIST
    "${OVERRIDABLE_FETCH_CONTENT_LIST}"
  )

  message(VERBOSE "FetchContent_Declare(${CONTENT_NAME} ${OUTPUT_ARGS}")
  FetchContent_Declare("${CONTENT_NAME}" ${OUTPUT_ARGS})
endfunction()


# Get a property name for this module.
# CONTENT_NAME: Name of the content associated with the FetchContent function.
# PROPERTY_NAME: Name of the property.
# OUTPUT_VAR: Variable to store the name in.
function(_OverridableFetchContent_GetPropertyName CONTENT_NAME PROPERTY_NAME
    OUTPUT_VAR)
  # The implementation of FetchContent_GetProperties() uses the lower case
  # content name to prefix property names so follow the same pattern here.
  string(TOLOWER "${CONTENT_NAME}" CONTENT_NAME_LOWER)
  set(${OUTPUT_VAR}
    "_OverridableFetchContent_${CONTENT_NAME_LOWER}_${PROPERTY_NAME}"
    PARENT_SCOPE
  )
endfunction()


# Set a global property for this module.
# CONTENT_NAME: Name of the content associated with the FetchContent function.
# PROPERTY_NAME: Name of the property to set.
# DOCUMENTATION: Documentation string for the property.
# PROPERTY_VALUE: Value to set the property to.
function(_OverridableFetchContent_SetProperty CONTENT_NAME PROPERTY_NAME
    DOCUMENTATION PROPERTY_VALUE)
  _OverridableFetchContent_GetPropertyName(
    "${CONTENT_NAME}"
    "${PROPERTY_NAME}"
    GLOBAL_PROPERTY_NAME
  )
  define_property(GLOBAL PROPERTY "${GLOBAL_PROPERTY_NAME}"
    BRIEF_DOCS "${DOCUMENTATION}"
    FULL_DOCS "${DOCUMENTATION}"
  )
set_property(
  GLOBAL PROPERTY "${GLOBAL_PROPERTY_NAME}"
  "${PROPERTY_VALUE}"
)
endfunction()


# Get a global property for this module.
# CONTENT_NAME: Name of the content associated with the FetchContent function.
# PROPERTY_NAME: Name of the property to get.
# OUTPUT_VAR: Variable to store the value in.
function(_OverridableFetchContent_GetProperty CONTENT_NAME PROPERTY_NAME
    OUTPUT_VAR)
  _OverridableFetchContent_GetPropertyName(
    "${CONTENT_NAME}"
    "${PROPERTY_NAME}"
    GLOBAL_PROPERTY_NAME
  )
  get_property(VALUE GLOBAL PROPERTY "${GLOBAL_PROPERTY_NAME}")
  if(VALUE)
    set(${OUTPUT_VAR} "${VALUE}" PARENT_SCOPE)
  endif()
endfunction()


# Export a list of variables to the parent scope of the caller function.
macro(_OverridableFetchContent_ExportToParentScope)
  # Export requested variables to the parent scope.
  foreach(VARIABLE_NAME ${ARGN})
    if(${VARIABLE_NAME})
      message(DEBUG "Export ${VARIABLE_NAME} ${${VARIABLE_NAME}}")
      set(${VARIABLE_NAME} "${${VARIABLE_NAME}}" PARENT_SCOPE)
    endif()
  endforeach()
endmacro()


# Wrapper around FetchContent_GetProperties().
#
# Sets the same variables as FetchContent_GetProperties() in addition to:
# * <contentName>_LICENSE_FILE: License file relative to
#   <contentName>_SOURCE_DIR if found.
# * <contentName>_LICENSE_URL: License URL if the file is found.
# * <contentName_ARCHIVE_URL: URL to the source package.
function(OverridableFetchContent_GetProperties CONTENT_NAME)
  set(EXPORT_VARIABLE_ARGS SOURCE_DIR BINARY_DIR POPULATED)
  cmake_parse_arguments(ARGS
    ""
    "${EXPORT_VARIABLE_ARGS}"
    ""
    ${ARGN}
  )

  # The implementation of FetchContent_Populate() uses the lower case
  # content name to prefix returned variable names.
  string(TOLOWER "${CONTENT_NAME}" CONTENT_NAME_LOWER)
  # Get the names of the variables to export to the parent scope.
  set(EXPORT_VARIABLES "")
  set(OUTPUT_ARGS "")
  foreach(ARG_NAME ${EXPORT_VARIABLE_ARGS})
    set(ARG_VARIABLE_NAME "ARGS_${ARG_NAME}")
    set(ARG_VARIABLE_VALUE "${${ARG_VARIABLE_NAME}}")
    list(APPEND EXPORT_VARIABLES "${CONTENT_NAME_LOWER}_${ARG_NAME}")
    if(ARG_VARIABLE_VALUE)
      list(APPEND EXPORT_VARIABLES "${ARG_VARIABLE_VALUE}")
      list(APPEND OUTPUT_ARGS "${ARG_NAME}" "${ARG_VARIABLE_VALUE}")
    endif()
  endforeach()
  list(APPEND OUTPUT_ARGS ${ARGS_UNPARSED_ARGUMENTS})

  foreach(EXPORT_PROPERTY LICENSE_FILE LICENSE_URL ARCHIVE_URL)
    _OverridableFetchContent_GetProperty("${CONTENT_NAME}"
      "${EXPORT_PROPERTY}"
      "${EXPORT_PROPERTY}"
    )
    set(PROPERTY_VALUE "${${EXPORT_PROPERTY}}")
    if(PROPERTY_VALUE)
      set(${CONTENT_NAME_LOWER}_${EXPORT_PROPERTY}
        "${PROPERTY_VALUE}"
        PARENT_SCOPE
      )
    endif()
  endforeach()
  FetchContent_GetProperties("${CONTENT_NAME}" ${OUTPUT_ARGS})
  _OverridableFetchContent_ExportToParentScope(${EXPORT_VARIABLES})
endfunction()


# Replacement for FetchContent_Populate() that searches a newly cloned
# repository for a top level license file and provides it to the caller
# via the <contentName>_LICENSE_FILE and <contentName>_LICENSE_URL variables
# where <contentName> is the value passed as the CONTENT_NAME argument of this
# method.
#
# To ensure a fetched repo has a license file and URL
# OVERRIDABLE_FETCH_CONTENT_LICENSE_CHECK_<contentName> to ON for one
# repository or OVERRIDABLE_FETCH_CONTENT_LICENSE_CHECK to ON for all
# repositories.
function(OverridableFetchContent_Populate CONTENT_NAME)
  # The implementation of FetchContent_Populate() uses the lower case
  # content name to prefix returned variable names.
  string(TOLOWER "${CONTENT_NAME}" CONTENT_NAME_LOWER)

  FetchContent_Populate("${CONTENT_NAME}")
  OverridableFetchContent_GetProperties("${CONTENT_NAME}")

  # If a license file isn't cached try finding it in the repo.
  set(LICENSE_FILE "${${CONTENT_NAME_LOWER}_LICENSE_FILE}")
  set(LICENSE_URL "${${CONTENT_NAME_LOWER}_LICENSE_URL}")
  set(SOURCE_DIR "${${CONTENT_NAME_LOWER}_SOURCE_DIR}")
  if(${CONTENT_NAME}_POPULATED AND NOT LICENSE_FILE)
    find_file(_${CONTENT_NAME_LOWER}_LICENSE_FILE
      NAMES LICENSE LICENSE.md LICENSE.txt NOTICE COPYING
      PATHS "${SOURCE_DIR}"
      DOC "${CONTENT_NAME} license file"
      NO_DEFAULT_PATH
      NO_CMAKE_FIND_ROOT_PATH
    )
    set(LICENSE_FILE "${_${CONTENT_NAME_LOWER}_LICENSE_FILE}")
    if(LICENSE_FILE)
      file(RELATIVE_PATH LICENSE_FILE "${SOURCE_DIR}" "${LICENSE_FILE}")
      file(TO_CMAKE_PATH "${LICENSE_FILE}" LICENSE_FILE)
    endif()
  endif()
  # If LICENSE_FILE was not found but a URL was specified then try downloading
  # the license.
  set(LICENSE_FILE_FULL_PATH "${SOURCE_DIR}/${LICENSE_FILE}")
  if(NOT EXISTS "${LICENSE_FILE_FULL_PATH}" AND LICENSE_URL)
    set(LICENSE_FILE_DOWNLOAD "${SOURCE_DIR}/${CONTENT_NAME}_LICENSE.txt")
    if(NOT EXISTS "${LICENSE_FILE_DOWNLOAD}")
      message(STATUS
        "${CONTENT_NAME} '${LICENSE_FILE_FULL_PATH}' does not exist "
        "downloading ${LICENSE_URL} --> ${LICENSE_FILE_DOWNLOAD}")
      file(DOWNLOAD "${LICENSE_URL}" "${LICENSE_FILE_DOWNLOAD}" STATUS RESULT)
      list(GET RESULT 0 RESULT)
      if(NOT RESULT EQUAL 0)
        message(
          FATAL_ERROR
          "Failed to download ${LICENSE_URL} for ${CONTENT_NAME} to "
          "${LICENSE_FILE_DOWNLOAD}"
        )
      endif()
    endif()
    file(RELATIVE_PATH LICENSE_FILE "${SOURCE_DIR}" "${LICENSE_FILE_DOWNLOAD}")
    _OverridableFetchContent_SetProperty(
      "${CONTENT_NAME}"
      LICENSE_FILE
      "License for ${CONTENT_NAME}"
      "${LICENSE_FILE}"
    )
  endif()
  # If a LICENSE_FILE was found populate the URL.
  if(LICENSE_FILE AND NOT LICENSE_URL)
    _LicenseFileToUrl(
      "${CONTENT_NAME}"
      "${LICENSE_FILE}"
      LICENSE_URL
    )
  endif()

  # If enabled, check for source licenses.
  if(OVERRIDABLE_FETCH_CONTENT_LICENSE_CHECK OR
     OVERRIDABLE_FETCH_CONTENT_LICENSE_CHECK_${CONTENT_NAME})
    message(DEBUG "LICENSE_FILE: ${LICENSE_FILE}, LICENSE_URL: ${LICENSE_URL}")
    if(NOT LICENSE_FILE)
      message(FATAL_ERROR
        "Required license file not found for ${CONTENT_NAME}"
      )
    endif()
    if(NOT LICENSE_URL)
      message(FATAL_ERROR
        "Required license URL not found for ${CONTENT_NAME}"
      )
    endif()
  endif()

  # Export return values to the parent scope.
  set(EXPORT_VARIABLES "")
  foreach(VARIABLE_POSTFIX SOURCE_DIR BINARY_DIR POPULATED)
    list(APPEND EXPORT_VARIABLES "${CONTENT_NAME_LOWER}_${VARIABLE_POSTFIX}")
  endforeach()
  _OverridableFetchContent_ExportToParentScope(${EXPORT_VARIABLES})
endfunction()
