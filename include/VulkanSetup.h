#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdint>

class VulkanSetup {
public:
    struct Options {
        const char* appName = "ComputeExample";
        uint32_t apiVersion = VK_API_VERSION_1_0;
        bool enableValidation = false; 
    };

    VulkanSetup(const Options& opts);
    ~VulkanSetup();

    // Non-copyable, movable
    VulkanSetup(const VulkanSetup&) = delete;
    VulkanSetup& operator=(const VulkanSetup&) = delete;
    VulkanSetup(VulkanSetup&& other) noexcept;
    VulkanSetup& operator=(VulkanSetup&& other) noexcept;

    // Accessors
    VkInstance           instance()           const { return instance_; }
    VkPhysicalDevice     physicalDevice()     const { return physicalDevice_; }
    VkDevice             device()             const { return device_; }
    VkQueue              computeQueue()       const { return computeQueue_; }
    uint32_t             computeQueueFamily() const { return computeQueueFamilyIndex_; }

private:
    void createInstance(const Options& opts);
    void pickPhysicalDeviceAndQueue();
    void createDeviceAndQueue();

    // Utility
    static bool hasExtension(const std::vector<VkExtensionProperties>& list, const char* name);
    static bool hasLayer(const std::vector<VkLayerProperties>& list, const char* name);

private:
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    uint32_t computeQueueFamilyIndex_ = 0;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue computeQueue_ = VK_NULL_HANDLE;

    // For cleanup 
    bool portabilityEnabled_ = false;
};
