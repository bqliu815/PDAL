# -----------------------------------------------------------------------------
# ROOT CMakeLists.txt for RHR-Lin-CP-AL
# Modified for the RHR-Lin-CP-AL release by Benqi Liu, 2026.
# -----------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.24)

project(rhr_lin_cp_al LANGUAGES C CXX CUDA)

set(CP_AL_VERSION_MAJOR 0)
set(CP_AL_VERSION_MINOR 1)
set(CP_AL_VERSION_PATCH 3)

set(CP_AL_VERSION "${CP_AL_VERSION_MAJOR}.${CP_AL_VERSION_MINOR}.${CP_AL_VERSION_PATCH}")
add_compile_definitions(CP_AL_VERSION="${CP_AL_VERSION}")

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

# C/C++ standards
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# Set default build type to Release if not specified (Optimized builds)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.23")
    set(CMAKE_CUDA_ARCHITECTURES all)
else()
    set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75 80 86 89 90 90-virtual)
endif()

# Global Compile flags (corresponding to CFLAGS/NVCCFLAGS)
add_compile_options(-fPIC -O3 -Wall -Wextra -g)

# Windows compatibility
if (WIN32)
    add_definitions(-Dstrtok_r=strtok_s)
endif()

# CUDA standards and RDC (Relocatable Device Code)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# -----------------------------------------------------------------------------
# CONTROL OPTIONS
# -----------------------------------------------------------------------------
include(CMakeDependentOption)

option(CP_AL_BUILD_STATIC_LIB "Build the CP_AL static library" ON)
option(CP_AL_BUILD_SHARED_LIB "Build the CP_AL shared library" ON)

cmake_dependent_option(CP_AL_BUILD_CLI "Build the CP_AL command-line executable" ON
                       "CP_AL_BUILD_STATIC_LIB" OFF)

cmake_dependent_option(CP_AL_BUILD_TESTS "Build the CP_AL test suite" OFF
                       "CP_AL_BUILD_STATIC_LIB" OFF)

# -----------------------------------------------------------------------------
# FIND DEPENDENCIES
# -----------------------------------------------------------------------------
# Core dependencies
find_package(CUDAToolkit REQUIRED)
find_package(ZLIB REQUIRED)

include(FetchContent)

# -----------------------------------------------------------------------------
# SOURCE DISCOVERY & TARGET DEFINITION
# -----------------------------------------------------------------------------
# Using file(GLOB) for convenience, but explicit lists are recommended for robust builds
file(GLOB C_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
)
file(GLOB CU_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
)
# Exclude cli.c from library builds
list(REMOVE_ITEM C_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/cli.c")

# Set common include directories for the core libraries
set(CORE_INCLUDE_DIRS
  PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include     # Public API headers
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/internal    # Internal implementation headers
)

# Set common link libraries
set(CORE_LINK_LIBS PUBLIC
  CUDA::cudart
  CUDA::cublas
  CUDA::cusparse
  ZLIB::ZLIB
)

# -----------------------------------------------------------------------------
# 1. Core STATIC Library (cp_al_core)
# -----------------------------------------------------------------------------
if(CP_AL_BUILD_STATIC_LIB)
    add_library(cp_al_core STATIC
      ${C_SOURCES}
      ${CU_SOURCES}
    )
    target_include_directories(cp_al_core ${CORE_INCLUDE_DIRS})
    target_link_libraries(cp_al_core ${CORE_LINK_LIBS})

    set_target_properties(cp_al_core PROPERTIES
      POSITION_INDEPENDENT_CODE ON
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()

# -----------------------------------------------------------------------------
# 2. Shared Library (libCP_AL.so)
# -----------------------------------------------------------------------------
if(CP_AL_BUILD_SHARED_LIB)
    add_library(CP_AL_shared SHARED
      ${C_SOURCES}
      ${CU_SOURCES}
    )
    target_include_directories(CP_AL_shared ${CORE_INCLUDE_DIRS})
    target_link_libraries(CP_AL_shared ${CORE_LINK_LIBS})

    # Shared library must resolve device symbols as it is a final link point
    set_target_properties(CP_AL_shared PROPERTIES
        OUTPUT_NAME "rhr_lin_cp_al"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()

# -----------------------------------------------------------------------------
# 3. CLI Executable (cp_al)
# -----------------------------------------------------------------------------
if(CP_AL_BUILD_CLI)
    if(NOT TARGET cp_al_core)
        message(FATAL_ERROR "CP_AL_BUILD_CLI=ON requires CP_AL_BUILD_STATIC_LIB=ON.")
    endif()

    add_executable(CP_AL_cli src/cli.c)

    target_include_directories(CP_AL_cli PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/internal
    )

    # Link CLI to the static core library
    target_link_libraries(CP_AL_cli PRIVATE cp_al_core)

    # CLI is a final executable, it must resolve device symbols
    set_target_properties(CP_AL_cli PROPERTIES
        OUTPUT_NAME "rhr_lin_cp_al"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()

# -----------------------------------------------------------------------------
# 4. Tests (CTest Integration)
# -----------------------------------------------------------------------------
if(CP_AL_BUILD_TESTS)
    if(NOT TARGET cp_al_core)
        message(FATAL_ERROR "CP_AL_BUILD_TESTS=ON requires CP_AL_BUILD_STATIC_LIB=ON.")
    endif()

    enable_testing()
    file(GLOB TEST_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/test/*.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/test/*.cu"
    )

    foreach(TEST_SRC ${TEST_SOURCES})
        get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)

        add_executable(${TEST_NAME} ${TEST_SRC})

        # Link tests to the core static library
        target_link_libraries(${TEST_NAME} PRIVATE cp_al_core)

        # Set up test includes
        target_include_directories(${TEST_NAME}
          PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
          PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/internal
        )

        # Tests are final executables, they must resolve device symbols
        set_target_properties(${TEST_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/tests"
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
        )

        # Register with CTest
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})

    endforeach()
endif()

# -----------------------------------------------------------------------------
# 5. Install Targets
# -----------------------------------------------------------------------------
if(TARGET cp_al_core)
    install(TARGETS cp_al_core
        ARCHIVE DESTINATION lib
    )
endif()

if(TARGET CP_AL_shared)
    install(TARGETS CP_AL_shared
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
    )
endif()

if(TARGET CP_AL_cli)
    install(TARGETS CP_AL_cli
        RUNTIME DESTINATION bin
    )
endif()

install(DIRECTORY include/
    DESTINATION include/
    FILES_MATCHING PATTERN "*.h"
)
