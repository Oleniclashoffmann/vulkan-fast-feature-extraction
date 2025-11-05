#include "GPUFrontend.h"

void GPUFrontend::Setup() {
    // Implementation of setup logic goes here

    instance = vk::createInstance(vk::InstanceCreateInfo());
    
}