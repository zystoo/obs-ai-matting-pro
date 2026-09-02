cmake_minimum_required(VERSION 3.16)

project(obs-ai-matting-pro VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(ENABLE_FRONTEND_API "Use obs-frontend-api for UI functionality" OFF)

add_library(${CMAKE_PROJECT_NAME} MODULE)

target_sources(
    ${CMAKE_PROJECT_NAME}
    PRIVATE
    src/plugin-main.cpp
    src/rvm-engine.cpp
)

set(OBS_INCLUDE_DIR "" CACHE PATH "Path to OBS headers")
set(OBS_LIB "" CACHE FILEPATH "Path to OBS library")

if(NOT OBS_INCLUDE_DIR AND NOT OBS_LIB)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(OBS QUIET obs libobs)
    endif()
endif()

if(OBS_INCLUDE_DIR AND OBS_LIB)
    target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${OBS_INCLUDE_DIR})
    target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE ${OBS_LIB})
elseif(OBS_FOUND)
    target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${OBS_INCLUDE_DIRS})
    target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE ${OBS_LIBRARIES})
else()
    find_path(_OBS_INCLUDE_DIR obs-module.h
        PATHS /usr/include/obs /usr/local/include/obs
              "/Library/Application Support/obs-studio/include"
              "$ENV{OBS_PREFIX}/include")
    find_library(_OBS_LIB NAMES obs libobs
        PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu
              "$ENV{OBS_PREFIX}/lib")
    if(_OBS_INCLUDE_DIR)
        target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${_OBS_INCLUDE_DIR})
    else()
        message(FATAL_ERROR "OBS SDK not found.")
    endif()
    if(_OBS_LIB)
        target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE ${_OBS_LIB})
    else()
        message(FATAL_ERROR "OBS library not found.")
    endif()
endif()

if(ENABLE_FRONTEND_API)
    find_package(obs-frontend-api REQUIRED)
    target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE OBS::obs-frontend-api)
endif()

set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to ONNX Runtime")

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
        message(WARNING "ONNX Runtime not found.")
    endif()
endif()

if(APPLE)
    target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE "-framework Accelerate" "-framework Foundation" "-framework CoreML")
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE PLATFORM_MACOS=1)
elseif(WIN32)
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE PLATFORM_WINDOWS=1)
    target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE d3d11 dxgi)
else()
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE PLATFORM_LINUX=1)
endif()

find_package(OpenCV QUIET COMPONENTS core imgproc)
if(OpenCV_FOUND)
    target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE ${OpenCV_LIBS})
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE HAS_OPENCV=1)
endif()

set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES OUTPUT_NAME obs-ai-matting-pro PREFIX "")

if(APPLE)
    install(TARGETS ${CMAKE_PROJECT_NAME} LIBRARY DESTINATION "obs-plugins" RUNTIME DESTINATION "obs-plugins")
elseif(WIN32)
    install(TARGETS ${CMAKE_PROJECT_NAME} LIBRARY DESTINATION "obs-plugins/64bit" RUNTIME DESTINATION "obs-plugins/64bit")
else()
    install(TARGETS ${CMAKE_PROJECT_NAME} LIBRARY DESTINATION "lib/obs-plugins")
endif()
