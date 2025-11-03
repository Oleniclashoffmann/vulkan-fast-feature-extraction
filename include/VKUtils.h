// VKUtils.h
//General Utility/Helper functions
#pragma once
#include <vulkan/vulkan.h>
#include <stdexcept>
#include <string>

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

