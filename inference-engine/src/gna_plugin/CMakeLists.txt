# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(TARGET_NAME "GNAPlugin")


file(GLOB_RECURSE SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

file(GLOB_RECURSE HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/descriptions/*.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/layers/*.hpp)

addVersionDefines(gna_plugin_entry_points.cpp CI_BUILD_NUMBER)

find_package(libGNA)

set(TARGET_INCLUDE_DIRS
        ${IE_MAIN_SOURCE_DIR}/src/inference_engine
        ${IE_MAIN_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR})

ie_add_plugin(NAME ${TARGET_NAME}
              DEVICE_NAME "GNA"
              SOURCES ${SOURCES} ${HEADERS})

if(GNA_LIBRARY_VERSION STREQUAL "GNA2")
    SET(GNA_LIBRARY_VERSION_NUMBER 2)
else()
    SET(GNA_LIBRARY_VERSION_NUMBER 1)
endif()

#saving rpath to GNA shared library be used by CI
log_rpath_from_dir(GNA ${libGNA_LIBRARIES_BASE_PATH})

target_link_libraries(${TARGET_NAME} PRIVATE inference_engine ${INTEL_ITT_LIBS} Threads::Threads libGNA)
target_include_directories(${TARGET_NAME} PUBLIC ${TARGET_INCLUDE_DIRS})
target_compile_definitions(${TARGET_NAME}
    PRIVATE
        -D_NO_MKL_
    PUBLIC
        GNA_LIB_VER=${GNA_LIBRARY_VERSION_NUMBER})

set(TEST_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/gna_plugin.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/gna_pass_manager.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/quantization/quantization.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/dnn.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/dnn_components.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/gna_device.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/pwl_design.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/floatmath.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/dnn_memory.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/util.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/gna_model_serial.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/gna_plugin_query_api.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/gna2_model_debug_log.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/gna2_model_helper.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/gna2_model_export_helper.cpp")

add_library(${TARGET_NAME}_test_static STATIC ${TEST_SOURCES} ${HEADERS})
target_compile_definitions(${TARGET_NAME}_test_static
        PRIVATE
            -D_NO_MKL_
        PUBLIC
            -DINTEGER_LOW_P
            -DUSE_STATIC_IE)
target_compile_definitions(${TARGET_NAME}_test_static PUBLIC GNA_LIB_VER=${GNA_LIBRARY_VERSION_NUMBER})
target_link_libraries(${TARGET_NAME}_test_static PUBLIC inference_engine_preproc_s libGNA::API)
target_include_directories(${TARGET_NAME}_test_static PUBLIC ${TARGET_INCLUDE_DIRS})

set_target_properties(${TARGET_NAME}_test_static PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME}_test_static)

if(WIN32)
    # Correct 'jnl' macro/jit issue
    target_compile_options(${TARGET_NAME} PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/bigobj> )
    target_compile_options(${TARGET_NAME}_test_static PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/bigobj> )
endif()
