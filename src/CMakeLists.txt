cmake_minimum_required(VERSION 3.18)
project(NLUPOMCP VERSION 1.0 LANGUAGES CXX)

include(FetchContent)
FetchContent_Declare(
  or-tools
  GIT_REPOSITORY https://github.com/google/or-tools.git
  GIT_TAG        main
)

# After the following call, the CMake targets defined by OR-Tools
# will be defined and available to the rest of the build
FetchContent_MakeAvailable(or-tools)
find_package(Boost 1.40 COMPONENTS program_options REQUIRED)

file(GLOB SOURCES
    ./*.h
    ./*.cpp
)

add_executable(pomcp ${SOURCES})
target_link_libraries(pomcp ortools::ortools)
target_link_libraries(pomcp Boost::program_options)