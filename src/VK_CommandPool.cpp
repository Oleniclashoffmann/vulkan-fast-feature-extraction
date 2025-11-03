#include "VK_CommandPool.h"
#include "VK_Utils.h" // für VK_CHECK
#include <stdexcept>

CommandPool::CommandPool(VkDevice dev, uint32_t queueFamily, VkCommandPoolCreateFlags flags)
    : dev_(dev) {
    
    VkCommandPoolCreateInfo poolCI{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolCI.queueFamilyIndex = queueFamily;
    poolCI.flags = flags;
    
    VK_CHECK(vkCreateCommandPool(dev_, &poolCI, nullptr, &pool_), "vkCreateCommandPool");
}

CommandPool::~CommandPool() {
    if (pool_ != VK_NULL_HANDLE && dev_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(dev_, pool_, nullptr);
    }
}

CommandPool::CommandPool(CommandPool&& other) noexcept 
    : dev_(other.dev_), pool_(other.pool_) {
    other.dev_ = VK_NULL_HANDLE;
    other.pool_ = VK_NULL_HANDLE;
}

CommandPool& CommandPool::operator=(CommandPool&& other) noexcept {
    if (this != &other) {
        // Cleanup current resources
        if (pool_ != VK_NULL_HANDLE && dev_ != VK_NULL_HANDLE) {
            vkDestroyCommandPool(dev_, pool_, nullptr);
        }
        
        // Move from other
        dev_ = other.dev_;
        pool_ = other.pool_;
        
        // Reset other
        other.dev_ = VK_NULL_HANDLE;
        other.pool_ = VK_NULL_HANDLE;
    }
    return *this;
}


//optional
VkCommandBuffer CommandPool::allocateCommandBuffer(VkCommandBufferLevel level) const {
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = pool_;
    ai.level = level;
    ai.commandBufferCount = 1;
    
    VkCommandBuffer cmd{};
    VK_CHECK(vkAllocateCommandBuffers(dev_, &ai, &cmd), "vkAllocateCommandBuffers");
    return cmd;
}

void CommandPool::freeCommandBuffer(VkCommandBuffer buffer) const {
    vkFreeCommandBuffers(dev_, pool_, 1, &buffer);
}