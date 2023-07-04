# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    ie_add_compiler_flags(-Wno-literal-conversion)
endif()

set(TARGET_NAME ov_gna_func_tests)

file(GLOB_RECURSE MODULE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
file(GLOB_RECURSE MODULE_HEADERS ${CMAKE_CURRENT_SOURCE_DIR} *.hpp)

add_executable(${TARGET_NAME} ${MODULE_SOURCES} ${MODULE_HEADERS})
target_link_libraries(${TARGET_NAME}
    openvino::gmock
    openvino::gtest
    openvino::gtest_main
    openvino::funcTestUtils
    openvino::funcSharedTests
    openvino::sharedTestClasses
)

set_target_properties(${TARGET_NAME} PROPERTIES VS_DEBUGGER_ENVIRONMENT "PATH=%PATH%;${OpenVINO_SOURCE_DIR}/temp/tbb/bin;${OpenVINO_SOURCE_DIR}/${BIN_FOLDER}/$<CONFIG>;${GNA_PATH}")

add_test(NAME ${TARGET_NAME}_run COMMAND ${TARGET_NAME})
set_tests_properties(${TARGET_NAME}_run PROPERTIES ENVIRONMENT "PATH=${OpenVINO_SOURCE_DIR}/temp/tbb/bin")
set_tests_properties(${TARGET_NAME}_run PROPERTIES ENVIRONMENT_MODIFICATION "PATH=path_list_append:${OpenVINO_SOURCE_DIR}/${BIN_FOLDER}/$<CONFIG>;PATH=path_list_append:${GNA_PATH}")
