# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.22)

project(OpenvinoGnaPlugin)

find_package(OpenVINODeveloperPackage REQUIRED)

set(GNA_LIB_VERSION "03.05.00.2116" CACHE STRING "GNA Library version number")
set(GNA_ROOT_DIR ${OpenVINO_SOURCE_DIR}/temp/gna_${GNA_LIB_VERSION})

set(GNA_PATH ${GNA_ROOT_DIR}/win64/x64)
set(GNA_EXT_DIR ${GNA_ROOT_DIR})

set(GNA_PLUGIN_OUTPUT_DIR ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${GNA_PLUGIN_OUTPUT_DIR}/${BIN_FOLDER})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${GNA_PLUGIN_OUTPUT_DIR}/${BIN_FOLDER})
set(CMAKE_COMPILE_PDB_OUTPUT_DIRECTORY ${GNA_PLUGIN_OUTPUT_DIR}/${BIN_FOLDER})
set(CMAKE_PDB_OUTPUT_DIRECTORY ${GNA_PLUGIN_OUTPUT_DIR}/${BIN_FOLDER})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${GNA_PLUGIN_OUTPUT_DIR}/${BIN_FOLDER})

message("OpenVINO source directory = ${OpenVINO_SOURCE_DIR}")
message("GNA package directory = ${GNA_ROOT_DIR}")
message("Artifacts directory = ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")

enable_testing()

#deal with setup of legacy projects first...
add_subdirectory(legacy)

set(TARGET_NAME "openvino_intel_gna_plugin")

if (ENABLE_INTEL_GNA_DEBUG)
    add_compile_definitions(GNA_DEBUG)
endif()

file(GLOB_RECURSE SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

file(GLOB_RECURSE HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp)

find_package(libGNA REQUIRED
             CONFIG
             PATHS "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
             NO_DEFAULT_PATH
             NO_CMAKE_FIND_ROOT_PATH)

#
# Shared plugin library
#

ov_add_plugin(NAME ${TARGET_NAME}
              DEVICE_NAME "GNA"
              SOURCES ${SOURCES} ${HEADERS}
              VERSION_DEFINES_FOR src/gna_plugin_entry_points.cpp
              ADD_CLANG_FORMAT)

# Enable support of CC for the plugin
ie_mark_target_as_cc(${TARGET_NAME})

target_link_libraries(${TARGET_NAME} PRIVATE inference_engine_legacy
        Threads::Threads libGNA)
target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_compile_definitions(${TARGET_NAME}
    PRIVATE
        _NO_MKL_
    )

# must be called after all target_link_libraries
ie_add_api_validator_post_build_step(TARGET ${TARGET_NAME})

#
# Static version for tests
#

add_library(${TARGET_NAME}_test_static STATIC EXCLUDE_FROM_ALL ${SOURCES} ${HEADERS})

ov_add_version_defines(src/gna_plugin_entry_points.cpp ${TARGET_NAME}_test_static)

target_compile_definitions(${TARGET_NAME}_test_static
        PRIVATE
            _NO_MKL_
            IMPLEMENT_INFERENCE_ENGINE_PLUGIN
        PUBLIC
            INTEGER_LOW_P
            USE_STATIC_IE)


target_link_libraries(${TARGET_NAME}_test_static
    PUBLIC debug inference_engine_sd optimized inference_engine_s
    PUBLIC libGNA::API)
target_include_directories(${TARGET_NAME}_test_static
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        $<TARGET_PROPERTY:inference_engine_legacy,INTERFACE_INCLUDE_DIRECTORIES>
    PRIVATE
        $<TARGET_PROPERTY:openvino::conditional_compilation,INTERFACE_INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:openvino::runtime::dev,INTERFACE_INCLUDE_DIRECTORIES>)

set_target_properties(${TARGET_NAME} ${TARGET_NAME}_test_static
                      PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

# install

file(GLOB_RECURSE gna_libraries "${libGNA_LIBRARIES_BASE_PATH}/*${CMAKE_SHARED_LIBRARY_SUFFIX}*")
set(gna_library_installed OFF)

foreach(gna_lib IN LISTS gna_libraries)
    # install only versioned files like libgna.so.N or libgna.so.N.M
    # and don't install global symlink libgna.so. Exceptions:
    # 1. On Windows we don't have libraries versioning and need to install all .dlls
    # 2. For case of static libraries we also need to install .so since it's required for linkage with final application
    if(gna_lib MATCHES "^.*\.${CMAKE_SHARED_LIBRARY_SUFFIX}(\.[0-9]+)+$" OR WIN32 OR NOT BUILD_SHARED_LIBS)
        install(FILES ${gna_lib}
                DESTINATION ${OV_CPACK_RUNTIMEDIR}
                COMPONENT gna)
        set(gna_library_installed ON)
    endif()
endforeach()

if(NOT gna_library_installed)
    message(FATAL_ERROR "Failed to install GNA library (available libraries: ${gna_libraries})")
endif()

# additional install rules for case of static libraries

if(NOT BUILD_SHARED_LIBS)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/libGNAConfig.cmake ${CMAKE_BINARY_DIR} COPYONLY)
    install(FILES "${CMAKE_BINARY_DIR}/libGNAConfig.cmake"
            DESTINATION ${OV_CPACK_IE_CMAKEDIR}
            COMPONENT gna)

    # install .lib file on Windows
    if(WIN32)
        file(GLOB_RECURSE gna_libraries  "${libGNA_LIBRARIES_BASE_PATH}/*${CMAKE_STATIC_LIBRARY_SUFFIX}")
        install(FILES ${gna_libraries}
                DESTINATION ${OV_CPACK_LIBRARYDIR}
                COMPONENT gna)
    endif()
endif()

add_subdirectory(tests)
