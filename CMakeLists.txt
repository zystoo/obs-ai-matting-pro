cmake_minimum_required(VERSION 3.28...3.30)

# ─── Bootstrap OBS plugin build system ─────────────────────────────
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/common/bootstrap.cmake" NO_POLICY_SCOPE)

# ─── Project definition ────────────────────────────────────────────
project(obs-ai-matting-pro VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ─── Options ───────────────────────────────────────────────────────
option(ENABLE_FRONTEND_API "Use obs-frontend-api for UI functionality" OFF)
option(ENABLE_QT "Use Qt functionality" OFF)
option(ENABLE_COREML "Enable CoreML execution provider on macOS" ON)
option(ENABLE_DIRECTML "Enable DirectML execution provider on Windows" ON)
option(ENABLE_CUDA "Enable CUDA execution provider on Linux" ON)

include(compilerconfig)
include(defaults)
include(helpers)

# ─── Plugin target ─────────────────────────────────────────────────
add_library(${CMAKE_PROJECT_NAME} MODULE)

# ─── Source files ───────────────────────────────────────────────────
target_sources(
    ${CMAKE_PROJECT_NAME}
    PRIVATE
    src/plugin-main.cpp
    src/rvm-engine.cpp
)

# ─── OBS SDK ────────────────────────────────────────────────────────
find_package(libobs REQUIRED)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE OBS::libobs)

if(ENABLE_FRONTEND_API)
    find_package(obs-frontend-api REQUIRED)
    target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE OBS::obs-frontend-api)
endif()

if(ENABLE_QT)
    find_package(Qt6 COMPONENTS Widgets Core)
    target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE Qt6::Core Qt6::Widgets)
    set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES AUTOMOC ON AUTOUIC ON AUTORCC ON)
endif()

# ─── ONNX Runtime (cross-platform) ─────────────────────────────────
set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to ONNX Runtime installation")

if(ONNXRUNTIME_ROOT)
    # Use manual path when ONNXRUNTIME_ROOT is set (avoids CMake config path issues)
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
    # Try find_package, then pkg_check_modules
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
else()
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE PLATFORM_LINUX=1)
endif()

# ─── OpenCV (for image processing) ──────────────────────────────────
find_package(OpenCV QUIET COMPONENTS core imgproc)
if(OpenCV_FOUND)
    target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE ${OpenCV_LIBS})
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE HAS_OPENCV=1)
else()
    message(STATUS "OpenCV not found - using built-in image processing")
endif()

# ─── Plugin output name ────────────────────────────────────────────
set_target_properties_plugin(${CMAKE_PROJECT_NAME} PROPERTIES OUTPUT_NAME obs-ai-matting-pro)

# ─── Install model files ────────────────────────────────────────────
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/data/models/rvm_mobilenetv3_fp32.onnx"
    DESTINATION data/obs-plugins/${CMAKE_PROJECT_NAME}/models
    OPTIONAL
)

# ─── Install locale ─────────────────────────────────────────────────
install(DIRECTORY
    "${CMAKE_CURRENT_SOURCE_DIR}/data/locale/"
    DESTINATION data/obs-plugins/${CMAKE_PROJECT_NAME}/locale
    OPTIONAL
)
