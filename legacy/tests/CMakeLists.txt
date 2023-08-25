# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(CMAKE_COMPILER_IS_GNUCXX)
    ie_add_compiler_flags(-Wno-missing-declarations)
    ie_add_compiler_flags(-Wno-bool-compare)
endif()

set(TARGET_NAME ov_legacy_transformations_tests)

file(GLOB MODULE_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
)

add_executable(${TARGET_NAME} ${MODULE_SOURCES})

target_include_directories(${TARGET_NAME} PRIVATE
    $<TARGET_PROPERTY:inference_engine_legacy_obj,INTERFACE_INCLUDE_DIRECTORIES>
    ${OV_ROOT_DIR}/src/inference/src #For CNNNetworkNGraphImpl. ptomasz1: Dirty hack, but inference_engine_obj is not exported in developer package
)

target_link_libraries(${TARGET_NAME}
    openvino::gmock
    openvino::gtest
    openvino::gtest_main
    openvino::ngraphFunctions
    openvino::commonTestUtils
    openvino::func_test_utils
    inference_engine_legacy
)

if(WIN32)
    string(REPLACE "/Zi" "/Z7" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
    string(REPLACE "/Zi" "/Z7" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
endif()

set_target_properties(${TARGET_NAME} PROPERTIES VS_DEBUGGER_ENVIRONMENT "PATH=%PATH%;${OpenVINO_SOURCE_DIR}/temp/tbb/bin;${OpenVINO_SOURCE_DIR}/${BIN_FOLDER}/$<CONFIG>")

add_test(NAME ${TARGET_NAME}_run COMMAND ${TARGET_NAME})
set_tests_properties(${TARGET_NAME}_run PROPERTIES ENVIRONMENT "PATH=${OpenVINO_SOURCE_DIR}/temp/tbb/bin")
set_tests_properties(${TARGET_NAME}_run PROPERTIES ENVIRONMENT_MODIFICATION "PATH=path_list_append:${OpenVINO_SOURCE_DIR}/${BIN_FOLDER}/$<CONFIG>")

