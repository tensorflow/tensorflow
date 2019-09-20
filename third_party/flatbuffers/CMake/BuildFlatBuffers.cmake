# Copyright 2015 Google Inc. All rights reserved.
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

# General function to create FlatBuffer build rules for the given list of
# schemas.
#
# flatbuffers_schemas: A list of flatbuffer schema files to process.
#
# schema_include_dirs: A list of schema file include directories, which will be
# passed to flatc via the -I parameter.
#
# custom_target_name: The generated files will be added as dependencies for a
# new custom target with this name. You should add that target as a dependency
# for your main target to ensure these files are built. You can also retrieve
# various properties from this target, such as GENERATED_INCLUDES_DIR,
# BINARY_SCHEMAS_DIR, and COPY_TEXT_SCHEMAS_DIR.
#
# additional_dependencies: A list of additional dependencies that you'd like
# all generated files to depend on. Pass in a blank string if you have none.
#
# generated_includes_dir: Where to generate the C++ header files for these
# schemas. The generated includes directory will automatically be added to
# CMake's include directories, and will be where generated header files are
# placed. This parameter is optional; pass in empty string if you don't want to
# generate include files for these schemas.
#
# binary_schemas_dir: If you specify an optional binary schema directory, binary
# schemas will be generated for these schemas as well, and placed into the given
# directory.
#
# copy_text_schemas_dir: If you want all text schemas (including schemas from
# all schema include directories) copied into a directory (for example, if you
# need them within your project to build JSON files), you can specify that
# folder here. All text schemas will be copied to that folder.
#
# IMPORTANT: Make sure you quote all list arguments you pass to this function!
# Otherwise CMake will only pass in the first element.
# Example: build_flatbuffers("${fb_files}" "${include_dirs}" target_name ...)
function(build_flatbuffers flatbuffers_schemas
                           schema_include_dirs
                           custom_target_name
                           additional_dependencies
                           generated_includes_dir
                           binary_schemas_dir
                           copy_text_schemas_dir)

  # Test if including from FindFlatBuffers
  if(FLATBUFFERS_FLATC_EXECUTABLE)
    set(FLATC_TARGET "")
    set(FLATC ${FLATBUFFERS_FLATC_EXECUTABLE})
  else()
    set(FLATC_TARGET flatc)
    set(FLATC flatc)
  endif()
  set(FLATC_SCHEMA_ARGS --gen-mutable)
  if(FLATBUFFERS_FLATC_SCHEMA_EXTRA_ARGS)
    set(FLATC_SCHEMA_ARGS
      ${FLATBUFFERS_FLATC_SCHEMA_EXTRA_ARGS}
      ${FLATC_SCHEMA_ARGS}
      )
  endif()

  set(working_dir "${CMAKE_CURRENT_SOURCE_DIR}")

  set(schema_glob "*.fbs")
  # Generate the include files parameters.
  set(include_params "")
  set(all_generated_files "")
  foreach (include_dir ${schema_include_dirs})
    set(include_params -I ${include_dir} ${include_params})
    if (NOT ${copy_text_schemas_dir} STREQUAL "")
      # Copy text schemas from dependent folders.
      file(GLOB_RECURSE dependent_schemas ${include_dir}/${schema_glob})
      foreach (dependent_schema ${dependent_schemas})
        file(COPY ${dependent_schema} DESTINATION ${copy_text_schemas_dir})
      endforeach()
    endif()
  endforeach()

  foreach(schema ${flatbuffers_schemas})
    get_filename_component(filename ${schema} NAME_WE)
    # For each schema, do the things we requested.
    if (NOT ${generated_includes_dir} STREQUAL "")
      set(generated_include ${generated_includes_dir}/${filename}_generated.h)
      add_custom_command(
        OUTPUT ${generated_include}
        COMMAND ${FLATC} ${FLATC_SCHEMA_ARGS}
        -o ${generated_includes_dir}
        ${include_params}
        -c ${schema}
        DEPENDS ${FLATC_TARGET} ${schema} ${additional_dependencies}
        WORKING_DIRECTORY "${working_dir}")
      list(APPEND all_generated_files ${generated_include})
    endif()

    if (NOT ${binary_schemas_dir} STREQUAL "")
      set(binary_schema ${binary_schemas_dir}/${filename}.bfbs)
      add_custom_command(
        OUTPUT ${binary_schema}
        COMMAND ${FLATC} -b --schema
        -o ${binary_schemas_dir}
        ${include_params}
        ${schema}
        DEPENDS ${FLATC_TARGET} ${schema} ${additional_dependencies}
        WORKING_DIRECTORY "${working_dir}")
      list(APPEND all_generated_files ${binary_schema})
    endif()

    if (NOT ${copy_text_schemas_dir} STREQUAL "")
      file(COPY ${schema} DESTINATION ${copy_text_schemas_dir})
    endif()
  endforeach()

  # Create a custom target that depends on all the generated files.
  # This is the target that you can depend on to trigger all these
  # to be built.
  add_custom_target(${custom_target_name}
                    DEPENDS ${all_generated_files} ${additional_dependencies})

  # Register the include directory we are using.
  if (NOT ${generated_includes_dir} STREQUAL "")
    include_directories(${generated_includes_dir})
    set_property(TARGET ${custom_target_name}
      PROPERTY GENERATED_INCLUDES_DIR
      ${generated_includes_dir})
  endif()

  # Register the binary schemas dir we are using.
  if (NOT ${binary_schemas_dir} STREQUAL "")
    set_property(TARGET ${custom_target_name}
      PROPERTY BINARY_SCHEMAS_DIR
      ${binary_schemas_dir})
  endif()

  # Register the text schema copy dir we are using.
  if (NOT ${copy_text_schemas_dir} STREQUAL "")
    set_property(TARGET ${custom_target_name}
      PROPERTY COPY_TEXT_SCHEMAS_DIR
      ${copy_text_schemas_dir})
  endif()
endfunction()
