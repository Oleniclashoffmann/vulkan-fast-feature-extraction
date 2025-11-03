#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>

class VulkanImage {
public:
    VulkanImage(VkDevice device, VkPhysicalDevice physicalDevice, 
                uint32_t width, uint32_t height, 
                VkFormat format = VK_FORMAT_R8G8B8A8_UNORM,
                VkImageUsageFlags usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    
    ~VulkanImage();
    
    // Move semantics (für Performance)
    VulkanImage(VulkanImage&& other) noexcept;
    VulkanImage& operator=(VulkanImage&& other) noexcept;
    
    // Accessors
    VkImage image() const { return image_; }
    VkDeviceMemory memory() const { return memory_; }
    VkImageView view() const { return view_; }
    VkDevice device() const { return device_; }
    
    uint32_t width() const { return width_; }
    uint32_t height() const { return height_; }
    VkFormat format() const { return format_; }
    VkDeviceSize size() const { return width_ * height_ * 4; }
    
private:
    void createImage();
    void allocateMemory();
    void createImageView();
    void cleanup();
    
private:
    VkDevice device_;
    VkPhysicalDevice physicalDevice_;
    
    VkImage image_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkImageView view_ = VK_NULL_HANDLE;
    
    uint32_t width_;
    uint32_t height_;
    VkFormat format_;
    VkImageUsageFlags usage_;
    
    // Non-copyable
    VulkanImage(const VulkanImage&) = delete;
    VulkanImage& operator=(const VulkanImage&) = delete;
};