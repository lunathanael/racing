include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.14.0
)
FetchContent_MakeAvailable(Catch2)

set(TEST_TARGET test_all)
set(TEST_SOURCES
    test/main.cpp
    test/test_matmul.cpp
)

add_executable(${TEST_TARGET} ${TEST_SOURCES})
target_link_libraries(${TEST_TARGET} PRIVATE ${TARGET_LIB} Catch2::Catch2WithMain)
target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_SOURCE_DIR})

include(CTest)
include(Catch)
catch_discover_tests(${TEST_TARGET})

add_custom_target(validation
    COMMENT "Build test executable"
    DEPENDS ${TEST_TARGET}
)