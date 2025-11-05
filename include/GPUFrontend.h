#pragma once
#include <vulkan/vulkan.h>
#include <opencv2/opencv.hpp>

class GPUFrontend {
public:
    GPUFrontend();
    ~GPUFrontend();
    
    void Setup();

private:

    vk::Instance instance = nullptr; 



};
