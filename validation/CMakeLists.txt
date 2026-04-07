include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.14.0
)
FetchContent_MakeAvailable(Catch2)

set(TEST_TARGET test_all)
set(BENCHMARK_TARGET benchmark_all)


set(TEST_SOURCES
    test/main.cpp
    test/test_matmul.cpp
)

add_executable(${TEST_TARGET} ${TEST_SOURCES})
target_link_libraries(${TEST_TARGET} PRIVATE ${TARGET_LIB} Catch2::Catch2WithMain)
target_compile_features(${TEST_TARGET} PRIVATE cxx_std_23)

include(CTest)
include(Catch)
catch_discover_tests(${TEST_TARGET})

FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.5
)
set(BENCHMARK_ENABLE_TESTING OFF)
FetchContent_MakeAvailable(benchmark)

set(BENCHMARK_SOURCES
    benchmark/main.cpp
    benchmark/matmul.cpp
)

add_executable(${BENCHMARK_TARGET} ${BENCHMARK_SOURCES})
target_link_libraries(${BENCHMARK_TARGET} PRIVATE ${TARGET_LIB} benchmark::benchmark)
target_compile_features(${BENCHMARK_TARGET} PRIVATE cxx_std_23)

add_custom_target(validation
    COMMENT "Build all test and benchmark executables"
    DEPENDS ${TEST_TARGET} ${BENCHMARK_TARGET}
)