#include "VK_Image.h"
#include "VK_Utils.h"
#include <stdexcept>

VulkanImage::VulkanImage(VkDevice device, VkPhysicalDevice physicalDevice, 
                         uint32_t width, uint32_t height, 
                         VkFormat format, VkImageUsageFlags usage)
    : device_(device), physicalDevice_(physicalDevice),
      width_(width), height_(height), format_(format), usage_(usage) {
    
    createImage();
    allocateMemory();
    createImageView();
}

VulkanImage::~VulkanImage() {
    cleanup();
}

VulkanImage::VulkanImage(VulkanImage&& other) noexcept 
    : device_(other.device_), physicalDevice_(other.physicalDevice_),
      image_(other.image_), memory_(other.memory_), view_(other.view_),
      width_(other.width_), height_(other.height_), 
      format_(other.format_), usage_(other.usage_) {
    
    // Reset other object
    other.image_ = VK_NULL_HANDLE;
    other.memory_ = VK_NULL_HANDLE;
    other.view_ = VK_NULL_HANDLE;
}

VulkanImage& VulkanImage::operator=(VulkanImage&& other) noexcept {
    if (this != &other) {
        cleanup();
        
        device_ = other.device_;
        physicalDevice_ = other.physicalDevice_;
        image_ = other.image_;
        memory_ = other.memory_;
        view_ = other.view_;
        width_ = other.width_;
        height_ = other.height_;
        format_ = other.format_;
        usage_ = other.usage_;
        
        // Reset other object
        other.image_ = VK_NULL_HANDLE;
        other.memory_ = VK_NULL_HANDLE;
        other.view_ = VK_NULL_HANDLE;
    }
    return *this;
}

void VulkanImage::createImage() {
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.extent = {width_, height_, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.format = format_;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ici.usage = usage_;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    VK_CHECK(vkCreateImage(device_, &ici, nullptr, &image_), "vkCreateImage");
}

void VulkanImage::allocateMemory() {
    VkMemoryRequirements mr{};
    vkGetImageMemoryRequirements(device_, image_, &mr);
    
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = findMemoryType(mr.memoryTypeBits, physicalDevice_, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    
    VK_CHECK(vkAllocateMemory(device_, &mai, nullptr, &memory_), "vkAllocateMemory(image)");
    VK_CHECK(vkBindImageMemory(device_, image_, memory_, 0), "vkBindImageMemory");
}

void VulkanImage::createImageView() {
    VkImageViewCreateInfo iv{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    iv.image = image_;
    iv.viewType = VK_IMAGE_VIEW_TYPE_2D;
    iv.format = format_;
    iv.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    
    VK_CHECK(vkCreateImageView(device_, &iv, nullptr, &view_), "vkCreateImageView");
}

void VulkanImage::cleanup() {
    if (view_ != VK_NULL_HANDLE) {
        vkDestroyImageView(device_, view_, nullptr);
        view_ = VK_NULL_HANDLE;
    }
    if (image_ != VK_NULL_HANDLE) {
        vkDestroyImage(device_, image_, nullptr);
        image_ = VK_NULL_HANDLE;
    }
    if (memory_ != VK_NULL_HANDLE) {
        vkFreeMemory(device_, memory_, nullptr);
        memory_ = VK_NULL_HANDLE;
    }
}