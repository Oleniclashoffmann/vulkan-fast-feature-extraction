#pragma once
#include <vulkan/vulkan.h>
#include <opencv2/opencv.hpp>
#include <cstdint>

class VulkanBuffer {
public:
    VulkanBuffer(VkDevice device, VkPhysicalDevice physicalDevice, 
                 VkDeviceSize size, 
                 VkBufferUsageFlags usage,
                 VkMemoryPropertyFlags properties);
    
    ~VulkanBuffer();
    
    // Move semantics
    VulkanBuffer(VulkanBuffer&& other) noexcept;
    VulkanBuffer& operator=(VulkanBuffer&& other) noexcept;
    
    // Accessors
    VkBuffer buffer() const { return buffer_; }
    VkDeviceMemory memory() const { return memory_; }
    VkDevice device() const { return device_; }
    VkDeviceSize size() const { return size_; }
    
    // Upload data to buffer (for staging buffers)
    void uploadData(const void* data, VkDeviceSize size, VkDeviceSize offset = 0);
    
    // Map memory for direct access
    void* mapMemory(VkDeviceSize size = VK_WHOLE_SIZE, VkDeviceSize offset = 0);
    void unmapMemory();
    
    // Convenience method for OpenCV Mat upload
    void uploadImage(const cv::Mat& image);
    
private:
    void createBuffer();
    void allocateMemory();
    void cleanup();
    
private:
    VkDevice device_;
    VkPhysicalDevice physicalDevice_;
    
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    
    VkDeviceSize size_;
    VkBufferUsageFlags usage_;
    VkMemoryPropertyFlags properties_;
    
    // Non-copyable
    VulkanBuffer(const VulkanBuffer&) = delete;
    VulkanBuffer& operator=(const VulkanBuffer&) = delete;
};

// Utility class for staging operations
class StagingBuffer : public VulkanBuffer {
public:
    StagingBuffer(VkDevice device, VkPhysicalDevice physicalDevice, VkDeviceSize size)
        : VulkanBuffer(device, physicalDevice, size, 
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) {}
    
    // Upload and transfer to image in one call
    void uploadToImage(const cv::Mat& image, VkImage targetImage, 
                      VkCommandBuffer cmd, uint32_t width, uint32_t height);
};