# -----------------------------------------------------------------------------
# ROOT CMakeLists.txt for CP_AL (Unified Build System)
# -----------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.20)

# Project config
project(cp_al LANGUAGES C CXX CUDA)

set(CP_AL_VERSION_MAJOR 0)
set(CP_AL_VERSION_MINOR 2)
set(CP_AL_VERSION_PATCH 5)

set(CP_AL_VERSION "${CP_AL_VERSION_MAJOR}.${CP_AL_VERSION_MINOR}.${CP_AL_VERSION_PATCH}")
add_compile_definitions(CP_AL_VERSION="${CP_AL_VERSION}")

if (WIN32)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

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

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86 89 90)
endif()

# Global Compile flags (corresponding to CFLAGS/NVCCFLAGS)
if(MSVC)
    add_compile_options(/O2 /W4 /Zi)
else()
    add_compile_options(-O3 -Wall -Wextra -g)
endif()

if (NOT WIN32)
  add_compile_options(-fPIC)
endif()

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

# format: cmake_dependent_option(OPTION "docstring" DEFAULT_VALUE "DEPENDENCY_VARIABLE" FORCE_OFF_VALUE)
cmake_dependent_option(CP_AL_BUILD_PYTHON "Build the CP_AL Python bindings" OFF
                       "CP_AL_BUILD_STATIC_LIB" OFF)

cmake_dependent_option(CP_AL_BUILD_CLI "Build the CP_AL command-line executable" ON
                       "CP_AL_BUILD_STATIC_LIB" OFF)

cmake_dependent_option(CP_AL_BUILD_TESTS "Build the CP_AL test suite" OFF
                       "CP_AL_BUILD_STATIC_LIB" OFF)

# -----------------------------------------------------------------------------
# FIND DEPENDENCIES
# -----------------------------------------------------------------------------
# Core dependencies (required for Julia/Yggdrasil and Python)
find_package(CUDAToolkit REQUIRED)

include(FetchContent)
# 1. Try to find ZLIB in the system first (Standard for Linux/Mac)
find_package(ZLIB QUIET)

if(ZLIB_FOUND)
    message(STATUS "Found System ZLIB: ${ZLIB_LIBRARIES}")
else()
    # 2. If not found (Common on Windows), fetch from source
    message(STATUS "ZLIB not found in system. Fetching from source...")

    FetchContent_Declare(
        zlib
        GIT_REPOSITORY https://github.com/madler/zlib.git
        GIT_TAG        v1.3
    )
    FetchContent_MakeAvailable(zlib)

    # 3. Create alias ZLIB::ZLIB to match your CORE_LINK_LIBS usage
    if(NOT TARGET ZLIB::ZLIB)
        if(TARGET zlibstatic)
            # Prefer zlibstatic to ensure static linking (avoids DLL issues on Windows)
            add_library(ZLIB::ZLIB ALIAS zlibstatic)
            # Expose include directories so zlib.h can be found
            target_include_directories(zlibstatic INTERFACE ${zlib_SOURCE_DIR} ${zlib_BINARY_DIR})
        else()
            # Fallback if the target is simply named 'zlib'
            add_library(ZLIB::ZLIB ALIAS zlib)
            target_include_directories(zlib INTERFACE ${zlib_SOURCE_DIR} ${zlib_BINARY_DIR})
        endif()
    endif()
endif()

if (CP_AL_BUILD_PYTHON)
    # Dependencies required only for Python bindings
    find_package(pybind11 CONFIG REQUIRED)
    find_package(Python3 COMPONENTS Interpreter REQUIRED) # For versioning script and pybind11
endif()

set(PSLP_VERSION_TAG "v0.0.4")

FetchContent_Declare(
  pslp
  GIT_REPOSITORY https://github.com/dance858/PSLP.git
  GIT_TAG        ${PSLP_VERSION_TAG}
)

FetchContent_MakeAvailable(pslp)
include_directories(${pslp_SOURCE_DIR}/PSLP)
add_compile_definitions(PSLP_VERSION=\"${PSLP_VERSION_TAG}\")

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
  PSLP
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
# 2. Shared Library (libcp_al.so)
# -----------------------------------------------------------------------------
if(CP_AL_BUILD_SHARED_LIB)
    add_library(cp_al_shared SHARED
      ${C_SOURCES}
      ${CU_SOURCES}
    )
    target_include_directories(cp_al_shared ${CORE_INCLUDE_DIRS})
    target_link_libraries(cp_al_shared ${CORE_LINK_LIBS})

    # Shared library must resolve device symbols as it is a final link point
    set_target_properties(cp_al_shared PROPERTIES
        OUTPUT_NAME "cp_al"
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
    
    add_executable(cp_al_cli src/cli.c)

    target_include_directories(cp_al_cli PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/internal
    )

    # Link CLI to the static core library
    target_link_libraries(cp_al_cli PRIVATE cp_al_core)

    # CLI is a final executable, it must resolve device symbols
    set_target_properties(cp_al_cli PROPERTIES
        OUTPUT_NAME "cp_al"
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
# 5. Python Bindings (Conditional)
# -----------------------------------------------------------------------------
if (CP_AL_BUILD_PYTHON)
    if(NOT TARGET cp_al_core)
        message(FATAL_ERROR "CP_AL_BUILD_PYTHON=ON requires CP_AL_BUILD_STATIC_LIB=ON.")
    endif()
    
    add_subdirectory(python_bindings)
endif()

# -----------------------------------------------------------------------------
# 6. Install Targets
# -----------------------------------------------------------------------------

if (CP_AL_BUILD_PYTHON)
    install(DIRECTORY include/
        DESTINATION include/
        FILES_MATCHING PATTERN "*.h"
    )

else()
    if(TARGET cp_al_core)
        install(TARGETS cp_al_core
            ARCHIVE DESTINATION lib
        )
    endif()
    
    if(TARGET cp_al_shared)
        install(TARGETS cp_al_shared
            LIBRARY DESTINATION lib
            RUNTIME DESTINATION bin # 'bin' for DLLs on Windows, 'lib' for .so on Linux
        )
    endif()
    
    if(TARGET cp_al_cli)
        install(TARGETS cp_al_cli
            RUNTIME DESTINATION bin
        )
    endif()

    install(DIRECTORY include/
        DESTINATION include/
        FILES_MATCHING PATTERN "*.h"
    )
endif()