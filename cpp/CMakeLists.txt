cmake_minimum_required(VERSION 3.16)
project(tubaspis LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- OpenMP ---
find_package(OpenMP REQUIRED)

# --- Compiler flags ---
if(MSVC)
    add_compile_options(/O2 /fp:fast /EHsc)
else()
    add_compile_options(-O3 -march=native -ffast-math)
endif()

# --- Executable ---
add_executable(tubaspis src/main.cpp)

target_include_directories(tubaspis PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/third_party
)

target_link_libraries(tubaspis PRIVATE OpenMP::OpenMP_CXX)
