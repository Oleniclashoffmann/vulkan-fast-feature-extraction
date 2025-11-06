#include "GPUFrontend.h"


//GPUFrontend methods -------------------------------------
GPUFrontend::GPUFrontend() = default;

GPUFrontend::~GPUFrontend() {
    if (m_device) {
        // Clean up pipeline resources
        if (m_computePipeline) {
            m_device.destroyPipeline(m_computePipeline);
        }
        if (m_pipelineCache) {
            m_device.destroyPipelineCache(m_pipelineCache);
        }
        if (m_pipelineLayout) {
            m_device.destroyPipelineLayout(m_pipelineLayout);
        }
        if (m_descriptorSetLayout) {
            m_device.destroyDescriptorSetLayout(m_descriptorSetLayout);
        }
        if (m_shaderModule) {
            m_device.destroyShaderModule(m_shaderModule);
        }
        
        // Clean up buffers
        if (m_InBuffer) {
            m_device.destroyBuffer(m_InBuffer);
        }
        if (m_OutBuffer) {
            m_device.destroyBuffer(m_OutBuffer);
        }
        if (m_InBufferMemory) {
            m_device.freeMemory(m_InBufferMemory);
        }
        if (m_OutBufferMemory) {
            m_device.freeMemory(m_OutBufferMemory);
        }

        // Clean up command pool
        if (m_commandPool) {
            m_device.destroyCommandPool(m_commandPool);
        }
        m_device.destroy();

        // Clean up descriptor resources ← NEU
        if (m_descriptorPool) {
            m_device.destroyDescriptorPool(m_descriptorPool);
        }
    }
    if (m_instance) {
        m_instance.destroy();
    }
} 

void GPUFrontend::Setup() 
{

    // Vulkan Instance und Device erstellen
    createInstanceForPlatform();
    createPhysicalDevice();
    queueFamilyProperties();
    queueFamilyIndex();
    createLogicalDevice();
    createCommandPool();
    createCommandBuffer();
}

void GPUFrontend::createInstanceForPlatform() 
{
    vk::ApplicationInfo appInfo("MyApp", 1, "MyEngine", 1, VK_API_VERSION_1_3);

#ifdef __APPLE__
    std::vector<const char*> exts = {
        VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME,
        VK_KHR_SURFACE_EXTENSION_NAME,
        VK_EXT_METAL_SURFACE_EXTENSION_NAME
    };
    vk::InstanceCreateInfo ci(vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR, &appInfo, 0, nullptr, static_cast<uint32_t>(exts.size()), exts.data());

#else
    std::vector<const char*> exts = { VK_KHR_SURFACE_EXTENSION_NAME };
    vk::InstanceCreateInfo ci{
        {}, &appInfo, 0, nullptr,
        static_cast<uint32_t>(exts.size()), exts.data()
    };
#endif

    m_instance = vk::createInstance(ci);
}

void GPUFrontend::createPhysicalDevice() 
{
    auto physicalDevices = m_instance.enumeratePhysicalDevices();
    if (physicalDevices.empty()) {
        throw std::runtime_error("No Vulkan-compatible GPU found.");
    }
    m_physicalDevice = physicalDevices.front(); 
}

void GPUFrontend::queueFamilyProperties() 
{
    m_queueFamilyProperties = m_physicalDevice.getQueueFamilyProperties();
}

void GPUFrontend::queueFamilyIndex()
{
    auto properties = m_physicalDevice.getQueueFamilyProperties();

    for (uint32_t i = 0; i < properties.size(); ++i) {
        if (properties[i].queueFlags & vk::QueueFlagBits::eGraphics) {
            m_computeQueueFamilyIndex = i;
            std::cout << "Graphics queue family index: " << i << std::endl;
            return;
        }
    }
    
    throw std::runtime_error("No graphics queue family found!");
}

void GPUFrontend::createLogicalDevice()
{
    vk::DeviceQueueCreateInfo deviceQueueCreateInfo( vk::DeviceQueueCreateFlags(), static_cast<uint32_t>( m_computeQueueFamilyIndex ), 1, &m_queuePriority );
    m_device = m_physicalDevice.createDevice( vk::DeviceCreateInfo( vk::DeviceCreateFlags(), deviceQueueCreateInfo ) );
}

void GPUFrontend::createCommandPool()
{
    m_commandPool = m_device.createCommandPool( vk::CommandPoolCreateInfo( vk::CommandPoolCreateFlags(), m_computeQueueFamilyIndex ) );
}

void GPUFrontend::createCommandBuffer()
{
    m_commandBuffer = m_device.allocateCommandBuffers( vk::CommandBufferAllocateInfo( m_commandPool, vk::CommandBufferLevel::ePrimary, 1 ) ).front();
}

void GPUFrontend::stagingBufferImage(cv::Mat image, int width, int height, int pixelbytes)
{
    m_deviceSize = width * height * pixelbytes;

    vk::BufferCreateInfo BufferCreateInfo{
		vk::BufferCreateFlags(),                    // Flags
		m_deviceSize,                                 // Size
		vk::BufferUsageFlagBits::eStorageBuffer,    // Usage
		vk::SharingMode::eExclusive,                // Sharing mode
		1,                                          // Number of queue family indices
		&m_computeQueueFamilyIndex                    // List of queue family indices
	};

    m_InBuffer = m_device.createBuffer(BufferCreateInfo);
	m_OutBuffer = m_device.createBuffer(BufferCreateInfo);

    vk::MemoryRequirements InBufferMemoryRequirements = m_device.getBufferMemoryRequirements(m_InBuffer);
	vk::MemoryRequirements OutBufferMemoryRequirements = m_device.getBufferMemoryRequirements(m_OutBuffer);

    vk::PhysicalDeviceMemoryProperties MemoryProperties = m_physicalDevice.getMemoryProperties();

    uint32_t MemoryTypeIndex = uint32_t(~0);
    vk::DeviceSize MemoryHeapSize = uint32_t(~0);
    
    for (uint32_t CurrentMemoryTypeIndex = 0; CurrentMemoryTypeIndex < MemoryProperties.memoryTypeCount; ++CurrentMemoryTypeIndex)
    {
        vk::MemoryType MemoryType = MemoryProperties.memoryTypes[CurrentMemoryTypeIndex];

        if ((vk::MemoryPropertyFlagBits::eHostVisible & MemoryType.propertyFlags) &&
            (vk::MemoryPropertyFlagBits::eHostCoherent & MemoryType.propertyFlags))
        {
            MemoryHeapSize = MemoryProperties.memoryHeaps[MemoryType.heapIndex].size;
            MemoryTypeIndex = CurrentMemoryTypeIndex;
            break;
        }
    }

    if (MemoryTypeIndex == uint32_t(~0)) {
        throw std::runtime_error("Could not find suitable memory type for buffers!");
    }

    vk::MemoryAllocateInfo InBufferMemoryAllocateInfo(InBufferMemoryRequirements.size, MemoryTypeIndex);
    vk::MemoryAllocateInfo OutBufferMemoryAllocateInfo(OutBufferMemoryRequirements.size, MemoryTypeIndex);
    
    m_InBufferMemory = m_device.allocateMemory(InBufferMemoryAllocateInfo);
    m_OutBufferMemory = m_device.allocateMemory(OutBufferMemoryAllocateInfo);

    void* InBufferPtr = m_device.mapMemory(m_InBufferMemory, 0, m_deviceSize);
    
    if (image.isContinuous()) {
        std::memcpy(InBufferPtr, image.data, m_deviceSize);
    } else {
        uint8_t* dstPtr = static_cast<uint8_t*>(InBufferPtr);
        for (int y = 0; y < height; ++y) {
            std::memcpy(dstPtr + y * width * pixelbytes, 
                       image.ptr(y), 
                       width * pixelbytes);
        }
    }
    
    m_device.unmapMemory(m_InBufferMemory);

    m_device.bindBufferMemory(m_InBuffer, m_InBufferMemory, 0);
    m_device.bindBufferMemory(m_OutBuffer, m_OutBufferMemory, 0);
}


std::vector<char> GPUFrontend::readShaderFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open shader file: " + filename);
    }
    
    size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(fileSize);
    
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    
    return buffer;
}

void GPUFrontend::createShaderModule(const std::vector<char>& code) 
{
    vk::ShaderModuleCreateInfo createInfo(
        vk::ShaderModuleCreateFlags(),
        code.size(),
        reinterpret_cast<const uint32_t*>(code.data())
    );
    
    m_shaderModule = m_device.createShaderModule(createInfo);
}

void GPUFrontend::createDescriptorSetLayout() 
{
    const std::vector<vk::DescriptorSetLayoutBinding> bindings = {
        {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}
    };
    
    vk::DescriptorSetLayoutCreateInfo layoutInfo(
        vk::DescriptorSetLayoutCreateFlags(),
        bindings
    );
    
    m_descriptorSetLayout = m_device.createDescriptorSetLayout(layoutInfo);
}

void GPUFrontend::createPipelineLayout() 
{
    vk::PipelineLayoutCreateInfo layoutInfo(
        vk::PipelineLayoutCreateFlags(),
        m_descriptorSetLayout
    );
    
    m_pipelineLayout = m_device.createPipelineLayout(layoutInfo);
    m_pipelineCache = m_device.createPipelineCache(vk::PipelineCacheCreateInfo());
}

void GPUFrontend::createComputePipeline(const std::string& shaderPath) 
{
    std::vector<char> shaderCode = readShaderFile(shaderPath);
    createShaderModule(shaderCode);

    createDescriptorSetLayout();
    
    createPipelineLayout();
    
    vk::PipelineShaderStageCreateInfo shaderStageInfo(
        vk::PipelineShaderStageCreateFlags(),
        vk::ShaderStageFlagBits::eCompute,
        m_shaderModule,
        "main"  
    );
    
    vk::ComputePipelineCreateInfo pipelineInfo(
        vk::PipelineCreateFlags(),
        shaderStageInfo,
        m_pipelineLayout
    );
    
    auto result = m_device.createComputePipeline(m_pipelineCache, pipelineInfo);
    
    if (result.result != vk::Result::eSuccess) {
        throw std::runtime_error("Failed to create compute pipeline!");
    }
    
    m_computePipeline = result.value;
}

void GPUFrontend::createDescriptorPool() 
{
    vk::DescriptorPoolSize poolSize(
        vk::DescriptorType::eStorageBuffer, 
        2  
    );
    
    vk::DescriptorPoolCreateInfo poolInfo(
        vk::DescriptorPoolCreateFlags(), 
        1,         
        poolSize   
    );
    
    m_descriptorPool = m_device.createDescriptorPool(poolInfo);
}

void GPUFrontend::allocateDescriptorSet() 
{
    vk::DescriptorSetAllocateInfo allocInfo(
        m_descriptorPool,
        1,                       
        &m_descriptorSetLayout   
    );
    
    const std::vector<vk::DescriptorSet> descriptorSets = 
        m_device.allocateDescriptorSets(allocInfo);
    
    m_descriptorSet = descriptorSets.front();
}

void GPUFrontend::updateDescriptorSet() 
{
    vk::DescriptorBufferInfo inBufferInfo(
        m_InBuffer, 
        0,             
        m_deviceSize  
    );
    
    vk::DescriptorBufferInfo outBufferInfo(
        m_OutBuffer, 
        0,            
        m_deviceSize    
    );
    
    const std::vector<vk::WriteDescriptorSet> writeDescriptorSets = {
        {
            m_descriptorSet,                        
            0,                                      
            0,                                      
            1,                                      
            vk::DescriptorType::eStorageBuffer,    
            nullptr,                                
            &inBufferInfo                           
        },

        {
            m_descriptorSet,                       
            1,                                      
            0,                                      
            1,                                      
            vk::DescriptorType::eStorageBuffer,    
            nullptr,                                
            &outBufferInfo
        }
    };
    
    m_device.updateDescriptorSets(writeDescriptorSets, {});
}

void GPUFrontend::createDescriptorResources() 
{
    createDescriptorPool();
    allocateDescriptorSet();
    updateDescriptorSet();
}