# python_bindings/CMakeLists.txt

cmake_minimum_required(VERSION 3.20)

# sources
set(PYBIND_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/_core_bindings.cpp)

# build the pybind11 module
pybind11_add_module(_fa_cp_core MODULE ${PYBIND_SOURCES})

target_link_libraries(_fa_cp_core PRIVATE 
    fa_cp_core
)

# set rpath so that the module can find the shared libraries at runtime
set_target_properties(_fa_cp_core PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH "$ORIGIN/../lib"
  BUILD_RPATH   "$ORIGIN/../lib"
)

# better error messages from pybind11
target_compile_definitions(_fa_cp_core PRIVATE PYBIND11_DETAILED_ERROR_MESSAGES=1)

install(TARGETS _fa_cp_core
  LIBRARY DESTINATION fa_cp
  RUNTIME DESTINATION fa_cp
  ARCHIVE DESTINATION fa_cp
)
