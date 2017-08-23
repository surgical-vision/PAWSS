cmake_minimum_required(VERSION 2.8)

project("PAWSS")
string(TOLOWER ${CMAKE_PROJECT_NAME} PROJECT_NAME_LOW)
string(TOUPPER ${CMAKE_PROJECT_NAME} PROJECT_NAME_CAP)

set(${PROJECT_NAME_CAP}_VERSION_MAJOR 0)
set(${PROJECT_NAME_CAP}_VERSION_MINOR 1)
set(${PROJECT_NAME_CAP}_VERSION ${${PROJECT_NAME_CAP}_VERSION_MAJOR}.${${PROJECT_NAME_CAP}_VERSION_MINOR})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules/")

################################################################################
# Add local path for finding packages, set the local version first
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules" ${CMAKE_MODULE_PATH})

set(CMAKE_VERBOSE_MAKEFILE ON)

# Overide with cmake -DCMAKE_BUILD_TYPE=Debug {dir}
if (NOT CMAKE_BUILD_TYPE AND NOT _WIN_)
    message("Build type not set (defaults to release)")
    message("-DCMAKE_BUILD_TYPE=Debug for debug")
    set(CMAKE_BUILD_TYPE Release)
endif ()

# Platform configuration vars
include(SetPlatformVars)

set(CPACK_GENERATOR "DEB")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Xiaofei Du")
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_NAME_CAP}_VERSION_MAJOR)
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_NAME_CAP}_VERSION_MINOR)
set(CPACK_PACKAGE_VERSION_PATCH "0")
include(CPack)


option(BUILD_SHARED_LIBS "Build Shared Library" ON)

set(CMAKE_CXX_STANDARD 11)
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
    set(CMAKE_CXX_FLAGS "-Wall -Werror -Wextra -Wno-deprecated-declarations -Wno-deprecated-register")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
    set(CMAKE_CXX_FLAGS "-Wall -Werror -Wextra -std=c++11 -Wno-deprecated-declarations -ftemplate-backtrace-limit=0")
    set(CMAKE_CXX_FLAGS_COVERAGE "${CMAKE_CXX_FLAGS_DEBUG} --coverage -fno-inline -fno-inline-small-functions -fno-default-inline")
    set(CMAKE_EXE_LINKER_FLAGS_COVERAGE "${CMAKE_EXE_LINKER_FLAGS_DEBUG} --coverage")
    set(CMAKE_SHARED_LINKER_FLAGS_COVERAGE "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} --coverage")
endif ()

if (_OSX_)
    set(CMAKE_MACOSX_RPATH ON)
endif ()
#set(CMAKE_MACOSX_RPATH 1)

set(LIBRARY_NAME ${PROJECT_NAME_LOW})

# make an uninstall target
configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
        IMMEDIATE @ONLY)


add_custom_target(uninstall
        "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake")
		
add_subdirectory("src")
add_subdirectory("demo")

# try to find TraX header and library ...
#FIND_FILE(TRAX_HEADER NAMES trax.h PATHS header/votUtils/native)
#IF(EXISTS ${TRAX_HEADER})
#    GET_FILENAME_COMPONENT(TRAX_DIR ${TRAX_HEADER} DIRECTORY)
#    ADD_DEFINITIONS(-DTRAX) # Added to enable TraX protocol support
#    LINK_DIRECTORIES(${TRAX_DIR}) # Add library directory
#    INCLUDE_DIRECTORIES(${TRAX_DIR}) # Add includes directory
#    MESSAGE(${TRAX_HEADER})
#ENDIF()



