#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <cstdint>

class VulkanDescriptor {
public:
    struct ImageBinding {
        uint32_t binding;
        VkImageView imageView;
        VkImageLayout layout;
        VkDescriptorType type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    };
    
    struct BufferBinding {
        uint32_t binding;
        VkBuffer buffer;
        VkDeviceSize offset = 0;
        VkDeviceSize range = VK_WHOLE_SIZE;
        VkDescriptorType type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    };

public:
    VulkanDescriptor(VkDevice device, VkShaderStageFlags stageFlags = VK_SHADER_STAGE_COMPUTE_BIT);
    ~VulkanDescriptor();
    
    // Move semantics
    VulkanDescriptor(VulkanDescriptor&& other) noexcept;
    VulkanDescriptor& operator=(VulkanDescriptor&& other) noexcept;
    
    // Add bindings (call before finalize())
    void addImageBinding(uint32_t binding, VkDescriptorType type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    void addBufferBinding(uint32_t binding, VkDescriptorType type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    
    // Finalize layout and create pool/set
    void finalize(uint32_t maxSets = 1);
    
    // Update descriptor set with actual resources
    void updateImageBinding(uint32_t binding, VkImageView imageView, 
                           VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL);
    void updateBufferBinding(uint32_t binding, VkBuffer buffer, 
                            VkDeviceSize offset = 0, VkDeviceSize range = VK_WHOLE_SIZE);
    
    // Convenience method for multiple image updates
    void updateImages(const std::vector<ImageBinding>& imageBindings);
    void updateBuffers(const std::vector<BufferBinding>& bufferBindings);
    
    // Accessors
    VkDescriptorSetLayout layout() const { return layout_; }
    VkDescriptorSet set() const { return set_; }
    VkDescriptorPool pool() const { return pool_; }
    VkDevice device() const { return device_; }
    
private:
    void createLayout();
    void createPool(uint32_t maxSets);
    void allocateSet();
    void cleanup();
    
private:
    VkDevice device_;
    VkShaderStageFlags stageFlags_;
    
    std::vector<VkDescriptorSetLayoutBinding> bindings_;
    std::vector<VkDescriptorPoolSize> poolSizes_;
    
    VkDescriptorSetLayout layout_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    VkDescriptorSet set_ = VK_NULL_HANDLE;
    
    bool finalized_ = false;
    
    // Non-copyable
    VulkanDescriptor(const VulkanDescriptor&) = delete;
    VulkanDescriptor& operator=(const VulkanDescriptor&) = delete;
};

// Convenience class for common compute shader setups
class ComputeDescriptor : public VulkanDescriptor {
public:
    ComputeDescriptor(VkDevice device) : VulkanDescriptor(device, VK_SHADER_STAGE_COMPUTE_BIT) {}
    
    // Common setup for input/output images
    void setupImageInputOutput() {
        addImageBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);  // Input
        addImageBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);  // Output
        finalize();
    }
    
    void updateInputOutput(VkImageView input, VkImageView output) {
        updateImageBinding(0, input);
        updateImageBinding(1, output);
    }
};