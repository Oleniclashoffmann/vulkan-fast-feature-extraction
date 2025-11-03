//Command Pool
#pragma once
#include <vulkan/vulkan.h>

class CommandPool {
public:
    CommandPool(VkDevice dev, uint32_t queueFamily, 
                VkCommandPoolCreateFlags flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
    ~CommandPool();
    
    // Move semantics
    CommandPool(CommandPool&&) noexcept; 
    CommandPool& operator=(CommandPool&&) noexcept;
    
    // Accessors
    VkCommandPool get() const { return pool_; }
    VkDevice device() const { return dev_; }
    
    // Convenience methods
    VkCommandBuffer allocateCommandBuffer(VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY) const;
    void freeCommandBuffer(VkCommandBuffer buffer) const;
    
private:
    VkDevice dev_{VK_NULL_HANDLE};
    VkCommandPool pool_{VK_NULL_HANDLE};
    
    // Non-copyable
    CommandPool(const CommandPool&) = delete; 
    CommandPool& operator=(const CommandPool&) = delete;
}; 
