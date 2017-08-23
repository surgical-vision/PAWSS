## Compiler configuration
if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_COMPILER_IS_GNUCC)
    set(_GCC_ 1)
endif ()

if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    set(_CLANG_ 1)
endif ()

if (MSVC)
    set(_MSVC_ 1)
endif ()

## Platform configuration
if (WIN32 OR WIN64)
    set(_WIN_ 1)
endif ()

if (UNIX)
    set(_UNIX_ 1)
endif ()

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(_OSX_ 1)
endif ()

if (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    set(_LINUX_ 1)
endif ()

if (ANDROID)
    set(_ANDROID_ 1)
endif ()

if (IOS)
    set(_APPLE_IOS_ 1)
endif ()

## Default search paths
if (_WIN_)
    if (${CMAKE_CL_64})
        LIST(APPEND CMAKE_INCLUDE_PATH "c:/dev/sysroot64/usr/include")
        LIST(APPEND CMAKE_LIBRARY_PATH "c:/dev/sysroot64/usr/lib")
        LIST(APPEND CMAKE_LIBRARY_PATH "c:/dev/sysroot64/usr/bin")
        set(PROGRAM_FILES "$ENV{PROGRAMW6432}")
    else ()
        LIST(APPEND CMAKE_INCLUDE_PATH "c:/dev/sysroot32/usr/include")
        LIST(APPEND CMAKE_LIBRARY_PATH "c:/dev/sysroot32/usr/lib")
        LIST(APPEND CMAKE_LIBRARY_PATH "c:/dev/sysroot32/usr/bin")
        set(PROGRAM_FILES "$ENV{PROGRAMFILES}")
    endif ()
endif ()
