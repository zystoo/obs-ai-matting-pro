cmake_minimum_required(VERSION 3.16)

project(obs-ai-matting-pro VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ─── Options ───────────────────────────────────────────────────────
option(ENABLE_FRONTEND_API "Use obs-frontend-api for UI functionality" OFF)

# ─── Plugin target ─────────────────────────────────────────────────
add_library(${CMAKE_PROJECT_NAME} MODULE)

# ─── Source files ───────────────────────────────────────────────────
target_sources(
    ${CMAKE_PROJECT_NAME}
    PRIVATE
    src/plugin-main.cpp
    src/rvm-engine.cpp
)

# ─── OBS SDK (system-installed or manual) ──────────────────────────
# Linux: apt install libobs-dev
# macOS: -DOBS_INCLUDE_DIR=/path/to/obs/libobs -DOBS_LIB=/path/to/libobs.dylib
# Windows: -DOBS_INCLUDE_DIR=... -DOBS_LIB=...
set(OBS_INCLUDE_DIR "" CACHE PATH "Path to OBS headers (containing obs-module.h)")
set(OBS_LIB "" CACHE FILEPATH "Path to OBS library (libobs.so / libobs.dylib / obs.lib)")

if(NOT OBS_INCLUDE_DIR AND NOT OBS_LIB)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(OBS QUIET obs libobs)
    endif()
endif()

if(OBS_INCLUDE_DIR AND OBS_LIB)
    # Manual paths provided
    target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${OBS_INCLUDE_DIR})
    target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE ${OBS_LIB})
elseif(OBS_FOUND)
    target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${OBS_INCLUDE_DIRS})
    target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE ${OBS_LIBRARIES})
else()
    # Fallback: try common paths
    find_path(_OBS_INCLUDE_DIR obs-module.h
        PATHS /usr/include/obs /usr/local/include/obs
              "/Library/Application Support/obs-studio/include"
              "$ENV{OBS_PREFIX}/include"
    )
    find_library(_OBS_LIB NAMES obs libobs
        PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu
              "$ENV{OBS_PREFIX}/lib"
    )
    if(_OBS_INCLUDE_DIR)
        target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${_OBS_INCLUDE_DIR})
    else()
        message(FATAL_ERROR "OBS SDK not found. Set OBS_INCLUDE_DIR and OBS_LIB manually.")
    endif()
    if(_OBS_LIB)
        target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE ${_OBS_LIB})
    else()
        message(FATAL_ERROR "OBS library not found. Set OBS_LIB manually.")
    endif()
endif()

if(ENABLE_FRONTEND_API)
    find_package(obs-frontend-api REQUIRED)
    target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE OBS::obs-frontend-api)
endif()

# ─── ONNX Runtime (cross-platform) ─────────────────────────────────
set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to ONNX Runtime installation")

if(ONNXRUNTIME_ROOT)
    target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE "${ONNXRUNTIME_ROOT}/include")
    if(APPLE)
        target_link_directories(${CMAKE_PROJECT_NAME} PRIVATE "${ONNXRUNTIME_ROOT}/lib")
        target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE "-L${ONNXRUNTIME_ROOT}/lib" "-lonnxruntime")
    elseif(WIN32)
        target_link_directories(${CMAKE_PROJECT_NAME} PRIVATE "${ONNXRUNTIME_ROOT}/lib")
        target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")
    else()
        target_link_directories(${CMAKE_PROJECT_NAME} PRIVATE "${ONNXRUNTIME_ROOT}/lib")
        target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE "-L${ONNXRUNTIME_ROOT}/lib" "-lonnxruntime")
    endif()
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE HAS_ONNXRUNTIME=1)
else()
    find_package(onnxruntime CONFIG QUIET)
    if(NOT onnxruntime_FOUND)
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(PC_onnxruntime QUIET onnxruntime IMPORTED_TARGET)
        endif()
    endif()

    if(onnxruntime_FOUND)
        target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE onnxruntime::onnxruntime)
        target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE HAS_ONNXRUNTIME=1)
    elseif(PC_onnxruntime_FOUND)
        target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE PkgConfig::onnxruntime)
        target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${PC_onnxruntime_INCLUDE_DIRS})
        target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE HAS_ONNXRUNTIME=1)
    else()
        message(WARNING "ONNX Runtime not found. Set ONNXRUNTIME_ROOT manually.")
    endif()
endif()

# ─── Platform-specific linking ─────────────────────────────────────
if(APPLE)
    target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE
        "-framework Accelerate"
        "-framework Foundation"
        "-framework CoreML"
    )
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE PLATFORM_MACOS=1)
elseif(WIN32)
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE PLATFORM_WINDOWS=1)
    target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE d3d11 dxgi)
else()
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE PLATFORM_LINUX=1)
endif()

# ─── OBS config defines (normally generated by CMake) ─────────────
target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE OBS_INSTALL_PREFIX="")

# ─── Output name ───────────────────────────────────────────────────
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES
    OUTPUT_NAME obs-ai-matting-pro
    PREFIX ""  # Remove "lib" prefix on Linux
)

# ─── Install ────────────────────────────────────────────────────────
if(APPLE)
    install(TARGETS ${CMAKE_PROJECT_NAME}
        LIBRARY DESTINATION "obs-plugins"
        RUNTIME DESTINATION "obs-plugins"
    )
elseif(WIN32)
    install(TARGETS ${CMAKE_PROJECT_NAME}
        LIBRARY DESTINATION "obs-plugins/64bit"
        RUNTIME DESTINATION "obs-plugins/64bit"
    )
else()
    install(TARGETS ${CMAKE_PROJECT_NAME}
        LIBRARY DESTINATION "lib/obs-plugins"
    )
endif()

# ─── Install model files ────────────────────────────────────────────
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/data/models/rvm_mobilenetv3_fp32.onnx"
    DESTINATION "share/obs-plugins/${CMAKE_PROJECT_NAME}/models"
    OPTIONAL
)

# ─── Install locale ─────────────────────────────────────────────────
install(DIRECTORY
    "${CMAKE_CURRENT_SOURCE_DIR}/data/locale/"
    DESTINATION "share/obs-plugins/${CMAKE_PROJECT_NAME}/locale"
    OPTIONAL
)
