#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <vulkan/vulkan.h>
#include <iostream>
#include <vector>
#include <cassert>
#include "VulkanSetup.h"
#include <chrono> 
#include "VK_Utils.h"
#include "VK_CommandPool.h"
#include "VK_Image.h"
#include "VK_Buffer.h"
#include "VK_Descriptor.h"



void GetImage(const std::string& filename, cv::Mat& image)
{
    image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Fehler beim Laden des Bildes!" << std::endl;
    }
}

int main() {
    auto beginOneShot = [&](VkDevice dev, VkCommandPool pool) -> VkCommandBuffer {
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = pool; 
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; 
        ai.commandBufferCount = 1;
        VkCommandBuffer cmd{};
        VK_CHECK(vkAllocateCommandBuffers(dev, &ai, &cmd), "vkAllocateCommandBuffers");
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(cmd, &bi), "vkBeginCommandBuffer");
        return cmd;
    };
    auto endOneShot = [&](VkDevice dev, VkQueue q, VkCommandPool pool, VkCommandBuffer cmd) {
        VK_CHECK(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; 
        si.commandBufferCount = 1; 
        si.pCommandBuffers = &cmd;
        VK_CHECK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE), "vkQueueSubmit");
        VK_CHECK(vkQueueWaitIdle(q), "vkQueueWaitIdle");
        vkFreeCommandBuffers(dev, pool, 1, &cmd);
    };
    auto transitionImage = [&](VkCommandBuffer cmd, VkImage img,
                               VkImageLayout oldL, VkImageLayout newL,
                               VkAccessFlags srcAccess, VkAccessFlags dstAccess,
                               VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage) {
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.oldLayout = oldL; 
        b.newLayout = newL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = img;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        b.srcAccessMask = srcAccess; 
        b.dstAccessMask = dstAccess;
        vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &b);
    };

    // Schritt 1: VulkanSetup initialisieren
    VulkanSetup::Options opts;
    opts.appName = "ComputeShaderExample";
    opts.apiVersion = VK_API_VERSION_1_2;
    opts.enableValidation = true; 
    VulkanSetup vk(opts);

    // Schritt 2: Command pool
    CommandPool cmdPool(vk.device(), vk.computeQueueFamily());

    //Schritt 3: Get Image 
    cv::Mat image;
    GetImage("/Users/olehoffmann/Documents/TUM/Studium/4. Semester/Guided Research/Coding/Vulkan Feature Extraction/Images/church.jpg", image);
    cv::Mat gray = image;
    if(image.channels() == 3) 
    {
        cv::cvtColor(image, image, cv::COLOR_BGR2RGBA);
    }
    else if(image.channels() == 1) 
    {
        cv::cvtColor(image, image, cv::COLOR_GRAY2RGBA);
    }

    //Schritt 4: Create Images
    uint32_t width = image.cols;
    uint32_t height = image.rows;
    VkDeviceSize imageSize = width * height * 4;

    VulkanImage inImage(vk.device(), vk.physicalDevice(), width, height);
    VulkanImage outImage(vk.device(), vk.physicalDevice(), width, height);

    //5. Staging buffer & upload to inImage
    StagingBuffer stagingBuffer(vk.device(), vk.physicalDevice(), imageSize);

    stagingBuffer.uploadImage(image); //Upload Image to staging Buffer
    // Transfer to GPU images
    {
        VkCommandBuffer cmd = beginOneShot(vk.device(), cmdPool.get());
    
        // Transition input image for transfer
        transitionImage(cmd, inImage.image(),
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            0, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

        // Copy staging to input image
        stagingBuffer.uploadToImage(image, inImage.image(), cmd, width, height);

        // Transition input image for compute
        transitionImage(cmd, inImage.image(),
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        // Prepare output image for compute writes
        transitionImage(cmd, outImage.image(),
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            0, VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        endOneShot(vk.device(), vk.computeQueue(), cmdPool.get(), cmd);
    }


    //8) Descriptors mit VK_Descriptor.h
    ComputeDescriptor descriptor(vk.device());
    descriptor.setupImageInputOutput();  // Erstellt bindings 0 und 1 für Images
    descriptor.updateInputOutput(inImage.view(), outImage.view());

    //9) Pipeline Layout
    VkDescriptorSetLayout descriptorLayout = descriptor.layout();
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &descriptorLayout;  // ← Verwende descriptorLayout
    VkPipelineLayout layout{};
    VK_CHECK(vkCreatePipelineLayout(vk.device(), &plci, nullptr, &layout), "vkCreatePipelineLayout");

    // ----- 9) Pipeline (load comp.spv) -----
    auto code = readFile("/Users/olehoffmann/Documents/TUM/Studium/4. Semester/Guided Research/Coding/Vulkan Feature Extraction/build/shaders/comp.spv");
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = code.size();
    smci.pCode = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule sm{};
    VK_CHECK(vkCreateShaderModule(vk.device(), &smci, nullptr, &sm), "vkCreateShaderModule");

    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                  VK_SHADER_STAGE_COMPUTE_BIT, sm, "main"};
    cpci.layout = layout;
    VkPipeline pipeline{};
    VK_CHECK(vkCreateComputePipelines(vk.device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeline), "vkCreateComputePipelines");

    // ----- 10) Dispatch -----
    VkCommandBufferAllocateInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbi.commandPool = cmdPool.get();
    cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; 
    cbi.commandBufferCount = 1;
    VkCommandBuffer cmdBuf{};
    VK_CHECK(vkAllocateCommandBuffers(vk.device(), &cbi, &cmdBuf), "vkAllocateCommandBuffers");

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VK_CHECK(vkBeginCommandBuffer(cmdBuf, &bi), "vkBeginCommandBuffer");

    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    VkDescriptorSet descriptorSet = descriptor.set();
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &descriptorSet, 0, nullptr);

    const uint32_t lsx = 16, lsy = 16; // match shader local_size
    uint32_t gx = (width  + lsx - 1) / lsx;
    uint32_t gy = (height + lsy - 1) / lsy;
    vkCmdDispatch(cmdBuf, gx, gy, 1);

    VK_CHECK(vkEndCommandBuffer(cmdBuf), "vkEndCommandBuffer");
    {
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cmdBuf;
        VK_CHECK(vkQueueSubmit(vk.computeQueue(), 1, &si, VK_NULL_HANDLE), "vkQueueSubmit");
        VK_CHECK(vkQueueWaitIdle(vk.computeQueue()), "vkQueueWaitIdle");
    }

    // ----- 11) Readback outImage -> buffer -> save as out.png -----
    VulkanBuffer readbackBuffer(vk.device(), vk.physicalDevice(), imageSize,
                           VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    {
        VkCommandBuffer cmd = beginOneShot(vk.device(), cmdPool.get());
        transitionImage(cmd, outImage.image(),
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

        VkBufferImageCopy r{};
        r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        r.imageOffset = {0,0,0};
        r.imageExtent = {width, height, 1};
        vkCmdCopyImageToBuffer(cmd, outImage.image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, 
                          readbackBuffer.buffer(), 1, &r);

        endOneShot(vk.device(), vk.computeQueue(), cmdPool.get(), cmd);
    }

    {
        void* mapped = readbackBuffer.mapMemory(imageSize);
        cv::Mat out(height, width, CV_8UC4, mapped); // RGBA
        cv::Mat outBGR; cv::cvtColor(out, outBGR, cv::COLOR_RGBA2BGR);
        cv::imwrite("out.png", outBGR);
        readbackBuffer.unmapMemory();
        std::cout << "Wrote out.png\n";
    }

    // Schritt 12: Ressourcen freigeben (Aufräumen)
    vkDeviceWaitIdle(vk.device());

    vkDestroyPipeline(vk.device(), pipeline, nullptr);
    vkDestroyPipelineLayout(vk.device(), layout, nullptr);
    vkDestroyShaderModule(vk.device(), sm, nullptr);

    return 0;
}