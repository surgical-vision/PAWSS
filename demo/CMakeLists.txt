find_package(PAWSS REQUIRED)
if(PAWSS_FOUND)
    message(STATUS "PAWSS Found and Enabled.")
endif()

include_directories(${PAWSS_INCLUDE_DIRS})

add_executable(Tracking main.cpp)
target_link_libraries(Tracking ${PAWSS_LIBRARIES})
