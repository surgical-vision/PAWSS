cmake_minimum_required(VERSION 2.8)
project(PAWSS)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()

# try to find TraX header and library ...
FIND_FILE(TRAX_HEADER NAMES trax.h PATHS header/votUtils/native)
IF(EXISTS ${TRAX_HEADER})
    GET_FILENAME_COMPONENT(TRAX_DIR ${TRAX_HEADER} DIRECTORY)
    ADD_DEFINITIONS(-DTRAX) # Added to enable TraX protocol support
    LINK_DIRECTORIES(${TRAX_DIR}) # Add library directory
    INCLUDE_DIRECTORIES(${TRAX_DIR}) # Add includes directory
    MESSAGE(${TRAX_HEADER})
ENDIF()

set(OpenCV_DIR "/usr/local/opt/opencv3/share/OpenCV")
find_package(OpenCV REQUIRED)
set(CMAKE_MODULE_PATH ${CMAKE_HOME_DIRECTORY}/cmake ${CMAKE_MODULE_PATH})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(Eigen REQUIRED)

find_package(Boost REQUIRED COMPONENTS filesystem system)

include_directories(
    header
    src
    ${EIGEN_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
)

# add all .cpp and .h files in src
file(GLOB_RECURSE CONFIG RELATIVE ${PROJECT_SOURCE_DIR} *.txt)
file(GLOB_RECURSE SRC RELATIVE ${PROJECT_SOURCE_DIR} src/*.cpp)
file(GLOB_RECURSE HEADERS RELATIVE ${PROJECT_SOURCE_DIR} header/*.h)


add_executable(PAWSS ${HEADERS} ${SRC} ${CONFIG})

IF(EXISTS ${TRAX_HEADER})
    target_link_libraries(PAWSS
        ${OpenCV_LIBS}
        ${Boost_FILESYSTEM_LIBRARY}
        ${Boost_SYSTEM_LIBRARY}
        trax)
ELSE()
    target_link_libraries(PAWSS
        ${OpenCV_LIBS}
        ${Boost_FILESYSTEM_LIBRARY}
        ${Boost_SYSTEM_LIBRARY})
ENDIF()

