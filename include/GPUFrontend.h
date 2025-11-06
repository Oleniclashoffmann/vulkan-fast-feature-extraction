#pragma once
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_metal.h>  
#include <opencv2/opencv.hpp>
#include <fstream>
#include <stdexcept>

class GPUFrontend {
public:
    GPUFrontend();
    ~GPUFrontend();
    
    void Setup();
    void stagingBufferImage(cv::Mat image, int width, int height, int pixelbytes);
    void createComputePipeline(const std::string& shaderPath);
    void createDescriptorResources(); 
    

private:

    //Member Variables for General Setup on Apple
    vk::Instance m_instance = nullptr; 
    vk::PhysicalDevice m_physicalDevice = nullptr;
    std::vector<vk::QueueFamilyProperties> m_queueFamilyProperties{};
    vk::Device m_device = nullptr;
    float m_queuePriority = 0.0f;
    uint32_t m_computeQueueFamilyIndex = 0;
    vk::CommandPool m_commandPool = nullptr;
    vk::CommandBuffer m_commandBuffer = nullptr;
    vk::DeviceSize m_deviceSize = 0;

    //Member for Pipeline 
    vk::ShaderModule m_shaderModule = nullptr;
    vk::DescriptorSetLayout m_descriptorSetLayout = nullptr;
    vk::PipelineLayout m_pipelineLayout = nullptr;
    vk::PipelineCache m_pipelineCache = nullptr;
    vk::Pipeline m_computePipeline = nullptr;

    //Member for Buffer 
    vk::Buffer m_InBuffer = nullptr;
    vk::Buffer m_OutBuffer = nullptr;
    vk::DeviceMemory m_InBufferMemory = nullptr;
    vk::DeviceMemory m_OutBufferMemory = nullptr;


    //functions for General Setup on Apple
    void createInstanceForPlatform();
    void createPhysicalDevice();
    void queueFamilyProperties();
    void queueFamilyIndex();
    void createLogicalDevice(); 
    void createCommandPool();
    void createCommandBuffer();

    //Pipeline helper functions 
    std::vector<char> readShaderFile(const std::string& filename);
    void createShaderModule(const std::vector<char>& code);
    void createDescriptorSetLayout();
    void createPipelineLayout();

    //Member for Descriptors
    vk::DescriptorPool m_descriptorPool = nullptr;
    vk::DescriptorSet m_descriptorSet = nullptr;

    //Descriptor helper functions 
    void createDescriptorPool();
    void allocateDescriptorSet();
    void updateDescriptorSet();

};
