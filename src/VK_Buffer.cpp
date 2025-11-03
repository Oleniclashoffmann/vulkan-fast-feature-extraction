#include "VK_Buffer.h"
#include "VK_Utils.h"
#include <stdexcept>
#include <cstring>

VulkanBuffer::VulkanBuffer(VkDevice device, VkPhysicalDevice physicalDevice, 
                          VkDeviceSize size, VkBufferUsageFlags usage,
                          VkMemoryPropertyFlags properties)
    : device_(device), physicalDevice_(physicalDevice),
      size_(size), usage_(usage), properties_(properties) {
    
    createBuffer();
    allocateMemory();
}

VulkanBuffer::~VulkanBuffer() {
    cleanup();
}

VulkanBuffer::VulkanBuffer(VulkanBuffer&& other) noexcept 
    : device_(other.device_), physicalDevice_(other.physicalDevice_),
      buffer_(other.buffer_), memory_(other.memory_),
      size_(other.size_), usage_(other.usage_), properties_(other.properties_) {
    
    // Reset other object
    other.buffer_ = VK_NULL_HANDLE;
    other.memory_ = VK_NULL_HANDLE;
}

VulkanBuffer& VulkanBuffer::operator=(VulkanBuffer&& other) noexcept {
    if (this != &other) {
        cleanup();
        
        device_ = other.device_;
        physicalDevice_ = other.physicalDevice_;
        buffer_ = other.buffer_;
        memory_ = other.memory_;
        size_ = other.size_;
        usage_ = other.usage_;
        properties_ = other.properties_;
        
        // Reset other object
        other.buffer_ = VK_NULL_HANDLE;
        other.memory_ = VK_NULL_HANDLE;
    }
    return *this;
}

void VulkanBuffer::createBuffer() {
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size_;
    bi.usage = usage_;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    VK_CHECK(vkCreateBuffer(device_, &bi, nullptr, &buffer_), "vkCreateBuffer");
}

void VulkanBuffer::allocateMemory() {
    VkMemoryRequirements mr{};
    vkGetBufferMemoryRequirements(device_, buffer_, &mr);
    
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = findMemoryType(mr.memoryTypeBits, physicalDevice_, properties_);
    
    VK_CHECK(vkAllocateMemory(device_, &mai, nullptr, &memory_), "vkAllocateMemory");
    VK_CHECK(vkBindBufferMemory(device_, buffer_, memory_, 0), "vkBindBufferMemory");
}

void VulkanBuffer::uploadData(const void* data, VkDeviceSize size, VkDeviceSize offset) {
    void* mapped = mapMemory(size, offset);
    std::memcpy(mapped, data, size);
    unmapMemory();
}

void* VulkanBuffer::mapMemory(VkDeviceSize size, VkDeviceSize offset) {
    void* mapped = nullptr;
    VK_CHECK(vkMapMemory(device_, memory_, offset, size, 0, &mapped), "vkMapMemory");
    return mapped;
}

void VulkanBuffer::unmapMemory() {
    vkUnmapMemory(device_, memory_);
}

void VulkanBuffer::uploadImage(const cv::Mat& image) {
    VkDeviceSize imageSize = image.cols * image.rows * image.channels();
    uploadData(image.data, imageSize);
}

void VulkanBuffer::cleanup() {
    if (buffer_ != VK_NULL_HANDLE) {
        vkDestroyBuffer(device_, buffer_, nullptr);
        buffer_ = VK_NULL_HANDLE;
    }
    if (memory_ != VK_NULL_HANDLE) {
        vkFreeMemory(device_, memory_, nullptr);
        memory_ = VK_NULL_HANDLE;
    }
}

// StagingBuffer implementation
void StagingBuffer::uploadToImage(const cv::Mat& image, VkImage targetImage, 
                                 VkCommandBuffer cmd, uint32_t width, uint32_t height) {
    // Upload image data to staging buffer
    uploadImage(image);
    
    // Copy from staging buffer to target image
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {width, height, 1};
    
    vkCmdCopyBufferToImage(cmd, buffer(), targetImage, 
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
}