find_package(PAWSS REQUIRED)
if(PAWSS_FOUND)
    message(STATUS "PAWSS Found and Enabled.")
endif()

include_directories(${PAWSS_INCLUDE_DIRS})
MESSAGE(STATUS "WA")
MESSAGE(STATUS ${PAWSS_INCLUDE_DIRS})

add_executable(MOTracking main.cpp)
target_link_libraries(MOTracking ${PAWSS_LIBRARIES})
