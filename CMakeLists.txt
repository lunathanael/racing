cmake_minimum_required(VERSION 3.16)
project(racer LANGUAGES CXX)

set(TARGET_LIB racer_lib)
set(TARGET_RENDER_LIB racer_render_lib)
set(TARGET racer)
set(TARGET_RL racer_rl)
set(TARGET_AP racer_ap)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(
    -Wall -Wextra -Wpedantic -Wno-unused-parameter
    -O3
    -pipe
    -flto
)

# TARGET LIB

set(GAME_SOURCES
    game/car.cpp
    game/maps.cpp
    game/race.cpp
    game/track.cpp
)

set(GAME_HEADERS
    game/car.h
    game/common.h
    game/maps.h
    game/race.h
    game/track.h
)

add_library(${TARGET_LIB} STATIC ${GAME_SOURCES} ${GAME_HEADERS})
target_include_directories(${TARGET_LIB} PRIVATE ${CMAKE_SOURCE_DIR})

set(RENDER_SOURCES
    render/car.cpp
    render/drift.cpp
    render/hud.cpp
    render/track.cpp
)

set(RENDER_HEADERS
    render/car.h
    render/common.h
    render/drift.h
    render/hud.h
    render/track.h
)

add_library(${TARGET_RENDER_LIB} STATIC ${RENDER_SOURCES} ${RENDER_HEADERS})
target_include_directories(${TARGET_RENDER_LIB} PRIVATE ${CMAKE_SOURCE_DIR})

# TARGET RENDER

if(EMSCRIPTEN)
    find_package(SDL3 REQUIRED CONFIG REQUIRED COMPONENTS SDL3-static)
else()
    find_package(SDL3 REQUIRED CONFIG REQUIRED COMPONENTS SDL3-shared)
endif()

add_executable(${TARGET} main_game.cpp)
target_link_libraries(${TARGET} PRIVATE ${TARGET_LIB} ${TARGET_RENDER_LIB} SDL3::SDL3)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})

# TARGET AP
set(AP_SOURCES
    main_ap.cpp
    rl/env.cpp
)

set(AP_HEADERS
    rl/env.h
)

add_executable(${TARGET_AP} ${AP_SOURCES} ${AP_HEADERS})
target_link_libraries(${TARGET_AP} PRIVATE ${TARGET_LIB} ${TARGET_RENDER_LIB}  SDL3::SDL3)
target_include_directories(${TARGET_AP} PRIVATE ${CMAKE_SOURCE_DIR})

# TARGET RL
set(RL_SOURCES
    main_rl.cpp
    rl/env.cpp
)

set(RL_HEADERS
    rl/env.h
    rl/mutate.h
)

add_executable(${TARGET_RL} ${RL_SOURCES} ${RL_HEADERS})
target_link_libraries(${TARGET_RL} PRIVATE ${TARGET_LIB} ${TARGET_RENDER_LIB}  SDL3::SDL3)
target_include_directories(${TARGET_RL} PRIVATE ${CMAKE_SOURCE_DIR})

# Generate Track Points

if(NOT DEFINED PYTHON_EXECUTABLE)
    set(VENV_DIR "${CMAKE_SOURCE_DIR}/.venv")
    find_package(Python3 COMPONENTS Interpreter REQUIRED)

    if(NOT EXISTS "${VENV_DIR}")
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -m venv ${VENV_DIR}
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        )
        execute_process(
            COMMAND ${VENV_DIR}/bin/python -m pip install -r requirements.txt
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        )
    endif()

    set(PYTHON_EXECUTABLE "${VENV_DIR}/bin/python")
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(TRACK_IMGS_DIR ${CMAKE_SOURCE_DIR}/track_imgs)
set(TRACK_LABELS ${CMAKE_SOURCE_DIR}/track_labels.json)
set(TRACK_DATA_DIR_RAW track_data)
set(TRACK_DATA_DIR ${CMAKE_SOURCE_DIR}/${TRACK_DATA_DIR_RAW})

# Track Data
file(GLOB TRACK_IMAGES ${TRACK_IMGS_DIR}/*.png)

set(TRACK_DATA)
file(MAKE_DIRECTORY ${TRACK_DATA_DIR})

foreach(INPUT ${TRACK_IMAGES})
    get_filename_component(NAME ${INPUT} NAME_WE)

    set(OUT ${TRACK_DATA_DIR}/${NAME}.txt)

    add_custom_command(
        OUTPUT ${OUT}
        COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/utils/process.py ${INPUT} ${TRACK_LABELS} ${OUT}
        DEPENDS ${INPUT} ${CMAKE_SOURCE_DIR}/utils/process.py
        COMMENT "Processing ${INPUT}"
        VERBATIM
    )

    list(APPEND TRACK_DATA ${OUT})
endforeach()

add_custom_target(track_data DEPENDS ${TRACK_DATA})

add_custom_command(
    OUTPUT ${TRACK_DATA_DIR}/index
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/utils/generate_index.py ${TRACK_DATA_DIR}/index ${TRACK_DATA}
    DEPENDS ${TRACK_DATA} ${CMAKE_SOURCE_DIR}/utils/generate_index.py
    COMMENT "Generating index"
)

add_custom_target(track_data_index ALL DEPENDS track_data ${TRACK_DATA_DIR}/index)

add_dependencies(${TARGET_LIB} track_data_index)

if(EMSCRIPTEN)
    set_target_properties(${TARGET} PROPERTIES SUFFIX ".html")
    set_target_properties(${TARGET_RL} PROPERTIES SUFFIX ".html")
    set_target_properties(${TARGET_AP} PROPERTIES SUFFIX ".html")

    target_link_options(${TARGET} PRIVATE
        -sASYNCIFY
        -sALLOW_MEMORY_GROWTH
        --preload-file ${TRACK_DATA_DIR}@/${TRACK_DATA_DIR_RAW}
    )
    target_link_options(${TARGET_RL} PRIVATE
        -sASYNCIFY
        -sALLOW_MEMORY_GROWTH
        --preload-file ${TRACK_DATA_DIR}@/${TRACK_DATA_DIR_RAW}
    )
    target_link_options(${TARGET_AP} PRIVATE
        -sASYNCIFY
        -sALLOW_MEMORY_GROWTH
        --preload-file ${TRACK_DATA_DIR}@/${TRACK_DATA_DIR_RAW}
    )

    target_link_libraries(${TARGET} PRIVATE SDL3::SDL3)
    target_link_libraries(${TARGET_RL} PRIVATE SDL3::SDL3)
    target_link_libraries(${TARGET_AP} PRIVATE SDL3::SDL3)
endif()

target_compile_features(${TARGET_LIB} PUBLIC cxx_std_23)
target_compile_features(${TARGET_RENDER_LIB} PUBLIC cxx_std_23)

target_compile_features(${TARGET} PRIVATE cxx_std_23)
target_compile_features(${TARGET_RL} PRIVATE cxx_std_23)
target_compile_features(${TARGET_AP} PRIVATE cxx_std_23)