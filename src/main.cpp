#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <vulkan/vulkan.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <chrono> 
#include <unordered_set>
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

    //Staging the image to GPU
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

    //Corner locations
    auto cornersFlat = gpuFrontend.readbackCorners();
    
    //CPU FAST for comparison
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    const int fastThresh = int(0.3f * 255.0f); 
    bool nonmax = true;
    cv::Ptr<cv::FastFeatureDetector> fast = cv::FastFeatureDetector::create(
        fastThresh, nonmax, cv::FastFeatureDetector::TYPE_9_16);
    std::vector<cv::KeyPoint> kps;
    fast->detect(gray, kps);
    cv::Mat cpuVis = image.clone();
    cv::drawKeypoints(image, kps, cpuVis, cv::Scalar(0,0,255), cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
    cv::imwrite("output_cpu.png", cpuVis);
    std::cout << "✓ CPU FAST saved to output_cpu.png (" << kps.size() << " keypoints)" << std::endl;

    //Compare the results of the two: 
    {
        std::unordered_set<uint64_t> gpuSet;
        gpuSet.reserve(cornersFlat.size()/2);
        for (size_t i=0; i+1<cornersFlat.size(); i+=2) {
            uint32_t x = cornersFlat[i];
            uint32_t y = cornersFlat[i+1];
            gpuSet.insert((uint64_t(x) << 32) | uint64_t(y));
        }

        size_t gpuCount = gpuSet.size();
        size_t cpuCount = kps.size();
        size_t exactMatches = 0;

        // CPU Keypoints exakt vergleichen (auf ganze Pixel runden)
        for (const auto& kp : kps) {
            int x = int(std::round(kp.pt.x));
            int y = int(std::round(kp.pt.y));
            if (x < 0 || y < 0 || x >= W || y >= H) continue;
            uint64_t key = (uint64_t(x) << 32) | uint64_t(y);
            if (gpuSet.find(key) != gpuSet.end()) {
                exactMatches++;
            }
        }

        size_t gpuOnly = gpuCount - exactMatches;
        size_t cpuOnly = cpuCount - exactMatches;

        std::cout << "Exact match stats:\n"
                  << "GPU corners: " << gpuCount << "\n"
                  << "CPU corners: " << cpuCount << "\n"
                  << "Exact overlaps: " << exactMatches << "\n"
                  << "GPU only: " << gpuOnly << "\n"
                  << "CPU only: " << cpuOnly << "\n";
    }

    return 0;
}