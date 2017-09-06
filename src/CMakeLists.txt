# Project headers trump everything (including any potentially installed PAWSS)
list(APPEND LIB_INC_DIR "${PROJECT_SOURCE_DIR}/include;${CMAKE_CURRENT_BINARY_DIR}/include")

# Head only includes
set(INCDIR ${PROJECT_SOURCE_DIR}/include/PAWSS)
file(GLOB HEADERS
	${INCDIR}/*.h*
	${INCDIR}/Features/*.h*
	${INCDIR}/GraphUtils/*.h*)

## Store list of source lists
file(GLOB SOURCES 
	${PROJECT_SOURCE_DIR}/src/*.cpp
	${PROJECT_SOURCE_DIR}/src/Features/*.cpp
	${PROJECT_SOURCE_DIR}/src/GraphUtils/*.cpp)


# message(STATUS ${INCDIR})
# message(STATUS ${PROJECT_SOURCE_DIR})
 message(STATUS ${HEADERS})
# message(STATUE ${SOURCES})

set(OpenCV_DIR "/usr/local/opt/opencv3/share/OpenCV")
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    set(HAVE_OPENCV 1)
    list(APPEND INTERNAL_INC ${OpenCV_INCLUDE_DIRS})
    list(APPEND LINK_LIBS ${OpenCV_LIBS})
    message(STATUS "OpenCV Found and Enabled.")
endif()

find_package(Eigen3 REQUIRED)
if(EIGEN3_FOUND)
	set(HAVE_EIGEN 1)
	list(APPEND INTERNAL_INC ${EIGEN3_INCLUDE_DIRS})
	message(STATUS "Eigen Found and Enabled.")
endif()

find_package(Boost REQUIRED COMPONENTS filesystem system)
if(Boost_FOUND)
    set(HAVE_BOOST 1)
    list(APPEND INTERNAL_INC ${Boost_INCLUDE_DIRS})
    list(APPEND LINK_LIBS ${Boost_FILESYSTEM_LIBRARY})
    list(APPEND LINK_LIBS ${Boost_SYSTEM_LIBRARY})
    message(STATUS "Boost Found and Enabled.")
endif()

include_directories(${LIB_INC_DIR})
include_directories(${INTERNAL_INC})

add_library(${LIBRARY_NAME} ${SOURCES} ${HEADERS})
target_link_libraries(${LIBRARY_NAME} ${LINK_LIBS})

#######################################################
## Create config.h file for inclusion in library
configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/config.h.in"
        "${CMAKE_CURRENT_BINARY_DIR}/include/${CMAKE_PROJECT_NAME}/config.h")

######################################################
# This relative path allows installed files to be relocatable.
set(CMAKECONFIG_INSTALL_DIR lib/cmake/${PROJECT_NAME})
file(RELATIVE_PATH REL_INCLUDE_DIR
        "${CMAKE_INSTALL_PREFIX}/${CMAKECONFIG_INSTALL_DIR}"
        "${CMAKE_INSTALL_PREFIX}/include")

# Export library for easy inclusion from other cmake projects. APPEND allows
# call to function even as subdirectory of larger project.
FILE(REMOVE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake")
export(TARGETS ${LIBRARY_NAME}
        APPEND FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake")

# Version information
configure_file(${PROJECT_NAME}ConfigVersion.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake" @ONLY)

# Build tree config
set(EXPORT_LIB_INC_DIR ${LIB_INC_DIR})
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}Config.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake @ONLY IMMEDIATE)

# Install tree config
set(EXPORT_LIB_INC_DIR ${PROJECT_CMAKE_DIR}/${REL_INCLUDE_DIR})
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}Config.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${PROJECT_NAME}Config.cmake @ONLY)


# Add package to CMake package registery for use from the build tree
option(EXPORT_${PROJECT_NAME}
        "Should the ${PROJECT_NAME} package be exported for use by other software" ON)

if (EXPORT_${PROJECT_NAME})
    export(PACKAGE ${PROJECT_NAME})
endif ()

#######################################################
## Install headers / targets
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/${LIBRARY_NAME}/config.h"
        DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${LIBRARY_NAME})
install(DIRECTORY ${INCDIR}
        DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
install(TARGETS ${LIBRARY_NAME}
        EXPORT ${PROJECT_NAME}Targets
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

#######################################################
## Install CMake config
install(FILES "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${PROJECT_NAME}Config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
        DESTINATION ${CMAKECONFIG_INSTALL_DIR})
install(EXPORT ${PROJECT_NAME}Targets DESTINATION ${CMAKECONFIG_INSTALL_DIR})

message(STATUS ${PROJECT_CMAKE_DIR})
message(STATUS ${CMAKECONFIG_INSTALL_DIR})
