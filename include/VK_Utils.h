// VKUtils.h
//General Utility/Helper functions
#pragma once
#include <vulkan/vulkan.h>
#include <stdexcept>
#include <string>
#include <fstream>

inline void VK_CHECK(VkResult r, const char* where) {
    if (r != VK_SUCCESS) throw std::runtime_error(std::string("Vulkan error at ") + where + " (" + std::to_string(r) + ")");
}

static uint32_t findMemoryType(uint32_t typeFilter, VkPhysicalDevice physDevice, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physDevice, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("Failed to find suitable memory type!");
}

static std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("failed to open file: " + filename);
    }
    size_t fileSize = (size_t) file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    return buffer;
}