cmake_minimum_required(VERSION 3.20)

# Use a no-space CMake project name; we'll still display the friendly name elsewhere.
project(FirstExcerciseVulkan LANGUAGES CXX)

# Friendly name used in logs/messages
set(PROJECT_DISPLAY_NAME "Vulkan Feature Extraction")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Help CMake find Vulkan using your environment
# Make sure VULKAN_SDK is set to: /Users/olehoffmann/VulkanSDK/vulkan/macOS
# (VS Code will also inject this via .vscode/settings.json)
if(NOT DEFINED ENV{VULKAN_SDK})
  message(WARNING "VULKAN_SDK is not set. Please set it to /Users/olehoffmann/VulkanSDK/vulkan/macOS")
else()
  message(STATUS "Using VULKAN_SDK at: $ENV{VULKAN_SDK}")
  # Some find modules check CMAKE_PREFIX_PATH
  list(APPEND CMAKE_PREFIX_PATH "$ENV{VULKAN_SDK}")
endif()

find_package(Vulkan REQUIRED)

find_package(OpenCV REQUIRED)


# ---- App target (skeleton; does not run compute yet) ----
add_executable(vulkan_feature_extraction
  src/main.cpp
  src/GPUFrontend.cpp
)

target_include_directories(vulkan_feature_extraction PRIVATE 
  ${CMAKE_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(vulkan_feature_extraction PRIVATE 
  Vulkan::Vulkan
  ${OpenCV_LIBS}
)

# Warnings (nice during development)
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
  target_compile_options(vulkan_feature_extraction PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ---- Shader build step (placeholder) ----
# Compiles shaders/comp.glsl -> build/shaders/comp.spv using glslangValidator
add_custom_target(shaders
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/shaders
  COMMAND glslangValidator -V ${CMAKE_SOURCE_DIR}/shaders/comp.comp.glsl -o ${CMAKE_BINARY_DIR}/shaders/comp.spv
  BYPRODUCTS ${CMAKE_BINARY_DIR}/shaders/comp.spv
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMENT "Compiling GLSL compute shader to SPIR-V"
)
add_dependencies(vulkan_feature_extraction shaders)
# ----------------------------------------

message(STATUS "Project: ${PROJECT_DISPLAY_NAME}")
message(STATUS "Vulkan include dirs: ${Vulkan_INCLUDE_DIRS}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
