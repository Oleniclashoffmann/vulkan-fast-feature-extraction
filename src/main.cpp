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

    const uint32_t workgroupSizeX = 16;
    const uint32_t workgroupSizeY = 16;
    uint32_t groupCountX = (W + workgroupSizeX - 1) / workgroupSizeX;
    uint32_t groupCountY = (H + workgroupSizeY - 1) / workgroupSizeY;
        
    gpuFrontend.executeComputeShader(groupCountX, groupCountY, 1);
        
    cv::Mat outputImage = gpuFrontend.readbackOutputBuffer(W, H, 3); 
        
    cv::imwrite("output.png", outputImage);
    std::cout << "✓ Output saved to output.png" << std::endl;

    
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    const int fastThresh = int(0.3f * 255.0f); // Shader THRESH=0.3 → ~77
    bool nonmax = true;
    cv::Ptr<cv::FastFeatureDetector> fast = cv::FastFeatureDetector::create(
        fastThresh, nonmax, cv::FastFeatureDetector::TYPE_9_16);

    std::vector<cv::KeyPoint> kps;
    fast->detect(gray, kps);

    cv::Mat cpuVis = image.clone();
    cv::drawKeypoints(image, kps, cpuVis, cv::Scalar(0,0,255), cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
    cv::imwrite("output_cpu.png", cpuVis);
    std::cout << "✓ CPU FAST saved to output_cpu.png (" << kps.size() << " keypoints)" << std::endl;


    return 0;
}