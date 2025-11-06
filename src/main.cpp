#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <vulkan/vulkan.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <chrono> 


#include "GPUFrontend.h"





int main() {

    GPUFrontend gpuFrontend;
    gpuFrontend.Setup();
    
    //Get the image
    cv::Mat image = cv::imread("/Users/olehoffmann/Documents/TUM/Studium/4. Semester/Guided Research/Coding/Vulkan Feature Extraction/Images/church.jpg", cv::IMREAD_COLOR);
    std::cout << "Image loaded with size: " << image.cols << "x" << image.rows << std::endl;

    int W = image.cols; 
    int H = image.rows;

    cv::Mat rgba; 
    cv::cvtColor(image, rgba, cv::COLOR_BGR2RGBA);

    gpuFrontend.stagingBufferImage(rgba, rgba.cols, rgba.rows, 4);

    gpuFrontend.createComputePipeline( "/Users/olehoffmann/Documents/TUM/Studium/4. Semester/Guided Research/Coding/Vulkan Feature Extraction/build/shaders/comp.spv");

    gpuFrontend.createDescriptorResources();

    

    return 0;
}