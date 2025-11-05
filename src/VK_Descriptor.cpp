#include "VK_Descriptor.h"
#include "VK_Utils.h"
#include <stdexcept>
#include <algorithm>

VulkanDescriptor::VulkanDescriptor(VkDevice device, VkShaderStageFlags stageFlags)
    : device_(device), stageFlags_(stageFlags) {
}

VulkanDescriptor::~VulkanDescriptor() {
    cleanup();
}

VulkanDescriptor::VulkanDescriptor(VulkanDescriptor&& other) noexcept 
    : device_(other.device_), stageFlags_(other.stageFlags_),
      bindings_(std::move(other.bindings_)), poolSizes_(std::move(other.poolSizes_)),
      layout_(other.layout_), pool_(other.pool_), set_(other.set_),
      finalized_(other.finalized_) {
    
    // Reset other object
    other.layout_ = VK_NULL_HANDLE;
    other.pool_ = VK_NULL_HANDLE;
    other.set_ = VK_NULL_HANDLE;
    other.finalized_ = false;
}

VulkanDescriptor& VulkanDescriptor::operator=(VulkanDescriptor&& other) noexcept {
    if (this != &other) {
        cleanup();
        
        device_ = other.device_;
        stageFlags_ = other.stageFlags_;
        bindings_ = std::move(other.bindings_);
        poolSizes_ = std::move(other.poolSizes_);
        layout_ = other.layout_;
        pool_ = other.pool_;
        set_ = other.set_;
        finalized_ = other.finalized_;
        
        // Reset other object
        other.layout_ = VK_NULL_HANDLE;
        other.pool_ = VK_NULL_HANDLE;
        other.set_ = VK_NULL_HANDLE;
        other.finalized_ = false;
    }
    return *this;
}

void VulkanDescriptor::addImageBinding(uint32_t binding, VkDescriptorType type) {
    if (finalized_) {
        throw std::runtime_error("Cannot add bindings after finalize()");
    }
    
    VkDescriptorSetLayoutBinding layoutBinding{};
    layoutBinding.binding = binding;
    layoutBinding.descriptorType = type;
    layoutBinding.descriptorCount = 1;
    layoutBinding.stageFlags = stageFlags_;
    
    bindings_.push_back(layoutBinding);
    
    // Update pool sizes
    auto it = std::find_if(poolSizes_.begin(), poolSizes_.end(),
        [type](const VkDescriptorPoolSize& size) { return size.type == type; });
    
    if (it != poolSizes_.end()) {
        it->descriptorCount++;
    } else {
        poolSizes_.push_back({type, 1});
    }
}

void VulkanDescriptor::addBufferBinding(uint32_t binding, VkDescriptorType type) {
    if (finalized_) {
        throw std::runtime_error("Cannot add bindings after finalize()");
    }
    
    VkDescriptorSetLayoutBinding layoutBinding{};
    layoutBinding.binding = binding;
    layoutBinding.descriptorType = type;
    layoutBinding.descriptorCount = 1;
    layoutBinding.stageFlags = stageFlags_;
    
    bindings_.push_back(layoutBinding);
    
    // Update pool sizes
    auto it = std::find_if(poolSizes_.begin(), poolSizes_.end(),
        [type](const VkDescriptorPoolSize& size) { return size.type == type; });
    
    if (it != poolSizes_.end()) {
        it->descriptorCount++;
    } else {
        poolSizes_.push_back({type, 1});
    }
}

void VulkanDescriptor::finalize(uint32_t maxSets) {
    if (finalized_) {
        throw std::runtime_error("Already finalized");
    }
    
    createLayout();
    createPool(maxSets);
    allocateSet();
    
    finalized_ = true;
}

void VulkanDescriptor::createLayout() {
    VkDescriptorSetLayoutCreateInfo createInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    createInfo.bindingCount = static_cast<uint32_t>(bindings_.size());
    createInfo.pBindings = bindings_.data();
    
    VK_CHECK(vkCreateDescriptorSetLayout(device_, &createInfo, nullptr, &layout_), 
             "vkCreateDescriptorSetLayout");
}

void VulkanDescriptor::createPool(uint32_t maxSets) {
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes_.size());
    poolInfo.pPoolSizes = poolSizes_.data();
    poolInfo.maxSets = maxSets;
    
    VK_CHECK(vkCreateDescriptorPool(device_, &poolInfo, nullptr, &pool_), 
             "vkCreateDescriptorPool");
}

void VulkanDescriptor::allocateSet() {
    VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.descriptorPool = pool_;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &layout_;
    
    VK_CHECK(vkAllocateDescriptorSets(device_, &allocInfo, &set_), 
             "vkAllocateDescriptorSets");
}

void VulkanDescriptor::updateImageBinding(uint32_t binding, VkImageView imageView, 
                                         VkImageLayout layout) {
    if (!finalized_) {
        throw std::runtime_error("Must call finalize() before updating bindings");
    }
    
    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageView = imageView;
    imageInfo.imageLayout = layout;
    
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = set_;
    write.dstBinding = binding;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;  // Could be made dynamic
    write.descriptorCount = 1;
    write.pImageInfo = &imageInfo;
    
    vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
}

void VulkanDescriptor::updateBufferBinding(uint32_t binding, VkBuffer buffer, 
                                          VkDeviceSize offset, VkDeviceSize range) {
    if (!finalized_) {
        throw std::runtime_error("Must call finalize() before updating bindings");
    }
    
    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = buffer;
    bufferInfo.offset = offset;
    bufferInfo.range = range;
    
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = set_;
    write.dstBinding = binding;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;  // Could be made dynamic
    write.descriptorCount = 1;
    write.pBufferInfo = &bufferInfo;
    
    vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
}

void VulkanDescriptor::updateImages(const std::vector<ImageBinding>& imageBindings) {
    std::vector<VkWriteDescriptorSet> writes;
    std::vector<VkDescriptorImageInfo> imageInfos;
    
    writes.reserve(imageBindings.size());
    imageInfos.reserve(imageBindings.size());
    
    for (const auto& binding : imageBindings) {
        VkDescriptorImageInfo& imageInfo = imageInfos.emplace_back();
        imageInfo.imageView = binding.imageView;
        imageInfo.imageLayout = binding.layout;
        
        VkWriteDescriptorSet& write = writes.emplace_back();
        write = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = set_;
        write.dstBinding = binding.binding;
        write.descriptorType = binding.type;
        write.descriptorCount = 1;
        write.pImageInfo = &imageInfo;
    }
    
    vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()), 
                          writes.data(), 0, nullptr);
}

void VulkanDescriptor::updateBuffers(const std::vector<BufferBinding>& bufferBindings) {
    std::vector<VkWriteDescriptorSet> writes;
    std::vector<VkDescriptorBufferInfo> bufferInfos;
    
    writes.reserve(bufferBindings.size());
    bufferInfos.reserve(bufferBindings.size());
    
    for (const auto& binding : bufferBindings) {
        VkDescriptorBufferInfo& bufferInfo = bufferInfos.emplace_back();
        bufferInfo.buffer = binding.buffer;
        bufferInfo.offset = binding.offset;
        bufferInfo.range = binding.range;
        
        VkWriteDescriptorSet& write = writes.emplace_back();
        write = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = set_;
        write.dstBinding = binding.binding;
        write.descriptorType = binding.type;
        write.descriptorCount = 1;
        write.pBufferInfo = &bufferInfo;
    }
    
    vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()), 
                          writes.data(), 0, nullptr);
}

void VulkanDescriptor::cleanup() {
    if (pool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device_, pool_, nullptr);
        pool_ = VK_NULL_HANDLE;
        set_ = VK_NULL_HANDLE;  // Sets werden automatisch mit Pool zerstört
    }
    if (layout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, layout_, nullptr);
        layout_ = VK_NULL_HANDLE;
    }
}