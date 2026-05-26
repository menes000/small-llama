cmake_minimum_required(VERSION 3.14)
project(llama-minimal C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

# llama_add_compile_flags() — used by src/CMakeLists.txt
include(common)

# BUILD_NUMBER from git/file
include(build-info)

set(LLAMA_BUILD_NUMBER     ${BUILD_NUMBER})
set(LLAMA_INSTALL_VERSION  0.0.${LLAMA_BUILD_NUMBER})
set(GGML_BUILD_NUMBER      ${LLAMA_BUILD_NUMBER})

# Apple Silicon — Metal default ON, BLAS off (we did not ship ggml-blas/)
if (APPLE)
    set(GGML_METAL ON CACHE BOOL "" FORCE)
    set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
    set(GGML_BLAS OFF CACHE BOOL "" FORCE)
    set(GGML_ACCELERATE OFF CACHE BOOL "" FORCE)
endif()

add_subdirectory(ggml)
add_subdirectory(src)
add_subdirectory(examples/simple)
add_subdirectory(examples/spec)
add_subdirectory(examples/chat)
