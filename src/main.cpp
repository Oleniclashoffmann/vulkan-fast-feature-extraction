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
    VkBuffer staging{}; 
    VkDeviceMemory stagingMem{};
    {
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = imageSize; 
        bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        VK_CHECK(vkCreateBuffer(vk.device(), &bi, nullptr, &staging), "vkCreateBuffer(staging)");
        VkMemoryRequirements mr{}; 
        vkGetBufferMemoryRequirements(vk.device(), staging, &mr);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = mr.size;
        mai.memoryTypeIndex = findMemoryType(mr.memoryTypeBits, vk.physicalDevice(),
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VK_CHECK(vkAllocateMemory(vk.device(), &mai, nullptr, &stagingMem), "vkAllocateMemory(staging)");
        VK_CHECK(vkBindBufferMemory(vk.device(), staging, stagingMem, 0), "vkBindBufferMemory(staging)");
        void* mapped = nullptr;
        VK_CHECK(vkMapMemory(vk.device(), stagingMem, 0, imageSize, 0, &mapped), "vkMapMemory");
        std::memcpy(mapped, image.data, (size_t)imageSize);
        vkUnmapMemory(vk.device(), stagingMem);
    }

    {
        VkCommandBuffer cmd = beginOneShot(vk.device(), cmdPool.get());
        transitionImage(cmd, inImage.image(),
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            0, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageOffset = {0,0,0};
        region.imageExtent = {width, height, 1};
        vkCmdCopyBufferToImage(cmd, staging, inImage.image(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        transitionImage(cmd, inImage.image(),
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        // outImage ready for compute writes
        transitionImage(cmd, outImage.image(),
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            0, VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        endOneShot(vk.device(), vk.computeQueue(), cmdPool.get(), cmd);
    }

    // ----- 8) Descriptors: binding 0 = in, 1 = out -----
    VkDescriptorSetLayoutBinding b0{}; 
    b0.binding = 0; 
    b0.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; 
    b0.descriptorCount = 1; 
    b0.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutBinding b1{}; 
    b1.binding = 1; 
    b1.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; 
    b1.descriptorCount = 1; 
    b1.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutBinding bindings[2] = {b0, b1};

    VkDescriptorSetLayoutCreateInfo dlci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dlci.bindingCount = 2; 
    dlci.pBindings = bindings;
    VkDescriptorSetLayout dsl{};
    VK_CHECK(vkCreateDescriptorSetLayout(vk.device(), &dlci, nullptr, &dsl), "vkCreateDescriptorSetLayout");

    VkDescriptorPoolSize poolSizes[1] = {{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2}};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.poolSizeCount = 1; 
    dpci.pPoolSizes = poolSizes; 
    dpci.maxSets = 1;
    VkDescriptorPool dpool{};
    VK_CHECK(vkCreateDescriptorPool(vk.device(), &dpci, nullptr, &dpool), "vkCreateDescriptorPool");

    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = dpool; 
    dsai.descriptorSetCount = 1; 
    dsai.pSetLayouts = &dsl;
    VkDescriptorSet dset{};
    VK_CHECK(vkAllocateDescriptorSets(vk.device(), &dsai, &dset), "vkAllocateDescriptorSets");

    VkDescriptorImageInfo inInfo{};  
    inInfo.imageView = inImage.view();   
    inInfo.imageLayout  = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorImageInfo outInfo{}; 
    outInfo.imageView = outImage.view(); 
    outInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet writes[2]{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = dset; 
    writes[0].dstBinding = 0; 
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; 
    writes[0].descriptorCount = 1; writes[0].pImageInfo = &inInfo;
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = dset; 
    writes[1].dstBinding = 1; 
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; 
    writes[1].descriptorCount = 1; 
    writes[1].pImageInfo = &outInfo;
    vkUpdateDescriptorSets(vk.device(), 2, writes, 0, nullptr);

        // ----- 9) Pipeline (load comp.spv) -----
    auto code = readFile("/Users/olehoffmann/Documents/TUM/Studium/4. Semester/Guided Research/Coding/Vulkan Feature Extraction/build/shaders/comp.spv");
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = code.size();
    smci.pCode = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule sm{};
    VK_CHECK(vkCreateShaderModule(vk.device(), &smci, nullptr, &sm), "vkCreateShaderModule");

    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; 
    plci.pSetLayouts = &dsl;
    VkPipelineLayout layout{};
    VK_CHECK(vkCreatePipelineLayout(vk.device(), &plci, nullptr, &layout), "vkCreatePipelineLayout");

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
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &dset, 0, nullptr);

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
    VkBuffer readback{};
    VkDeviceMemory readMem{};
    {
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = imageSize; bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VK_CHECK(vkCreateBuffer(vk.device(), &bi, nullptr, &readback), "vkCreateBuffer(readback)");
        VkMemoryRequirements mr{}; 
        vkGetBufferMemoryRequirements(vk.device(), readback, &mr);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = mr.size;
        mai.memoryTypeIndex = findMemoryType(mr.memoryTypeBits, vk.physicalDevice(),
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VK_CHECK(vkAllocateMemory(vk.device(), &mai, nullptr, &readMem), "vkAllocateMemory(readback)");
        VK_CHECK(vkBindBufferMemory(vk.device(), readback, readMem, 0), "vkBindBufferMemory(readback)");
    }
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
        vkCmdCopyImageToBuffer(cmd, outImage.image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback, 1, &r);

        endOneShot(vk.device(), vk.computeQueue(), cmdPool.get(), cmd);
    }
    {
        void* mapped = nullptr;
        VK_CHECK(vkMapMemory(vk.device(), readMem, 0, imageSize, 0, &mapped), "vkMapMemory(readback)");
        cv::Mat out(height, width, CV_8UC4, mapped); // RGBA
        cv::Mat outBGR; cv::cvtColor(out, outBGR, cv::COLOR_RGBA2BGR);
        cv::imwrite("out.png", outBGR);
        vkUnmapMemory(vk.device(), readMem);
        std::cout << "Wrote out.png\n";
    }

    // Schritt 12: Ressourcen freigeben (Aufräumen)
    vkDeviceWaitIdle(vk.device());

    vkDestroyBuffer(vk.device(), readback, nullptr);
    vkFreeMemory(vk.device(), readMem, nullptr);

    vkDestroyPipeline(vk.device(), pipeline, nullptr);
    vkDestroyPipelineLayout(vk.device(), layout, nullptr);
    vkDestroyShaderModule(vk.device(), sm, nullptr);

    vkDestroyDescriptorPool(vk.device(), dpool, nullptr);
    vkDestroyDescriptorSetLayout(vk.device(), dsl, nullptr);


    vkDestroyBuffer(vk.device(), staging, nullptr);
    vkFreeMemory(vk.device(), stagingMem, nullptr);

    return 0;
}