cmake_minimum_required(VERSION 3.16)
project(fastnb VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- OpenMP (optional) ---
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
else()
    message(STATUS "OpenMP not found — single-threaded mode")
endif()

# --- SIMD detection ---
include(CheckCXXCompilerFlag)
if(NOT MSVC)
    check_cxx_compiler_flag("-mavx2" HAS_AVX2)
    check_cxx_compiler_flag("-msse4.2" HAS_SSE42)
    if(HAS_AVX2)
        add_compile_options(-mavx2)
        message(STATUS "AVX2 enabled")
    elseif(HAS_SSE42)
        add_compile_options(-msse4.2)
        message(STATUS "SSE4.2 enabled")
    endif()
endif()

# --- Core library ---
add_library(fastnb_core STATIC src/nb_classifier.cpp)
target_include_directories(fastnb_core PUBLIC src/)
if(OpenMP_CXX_FOUND)
    target_link_libraries(fastnb_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# --- pybind11 Python module ---
find_package(pybind11 QUIET)
if(pybind11_FOUND)
    pybind11_add_module(fastnb_cpp python/bindings.cpp)
    target_link_libraries(fastnb_cpp PRIVATE fastnb_core)
    message(STATUS "pybind11 found — building Python module")
else()
    message(STATUS "pybind11 not found — skipping Python module")
endif()

# --- Tests ---
enable_testing()
add_executable(test_nb_classifier tests/cpp/test_nb_classifier.cpp)
target_link_libraries(test_nb_classifier PRIVATE fastnb_core)
add_test(NAME test_nb_classifier COMMAND test_nb_classifier)
