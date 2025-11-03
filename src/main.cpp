#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <vulkan/vulkan.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <fstream>
#include "VulkanSetup.h"
#include <chrono> 

static void VK_CHECK(VkResult r, const char* where) {
    if (r != VK_SUCCESS) throw std::runtime_error(std::string("Vulkan error at ") + where + " (" + std::to_string(r) + ")");
}
// Fehlende Hilfsfunktionen hinzufügen:
static uint32_t findMemoryType(uint32_t typeFilter, VkPhysicalDevice physDevice, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physDevice, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("Failed to find suitable memory type!");
}

static std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("failed to open file: " + filename);
    }
    size_t fileSize = (size_t) file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    return buffer;
}


int main() {
    auto beginOneShot = [&](VkDevice dev, VkCommandPool pool) -> VkCommandBuffer {
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = pool; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
        VkCommandBuffer cmd{};
        VK_CHECK(vkAllocateCommandBuffers(dev, &ai, &cmd), "vkAllocateCommandBuffers");
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(cmd, &bi), "vkBeginCommandBuffer");
        return cmd;
    };
    auto endOneShot = [&](VkDevice dev, VkQueue q, VkCommandPool pool, VkCommandBuffer cmd) {
        VK_CHECK(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
        VK_CHECK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE), "vkQueueSubmit");
        VK_CHECK(vkQueueWaitIdle(q), "vkQueueWaitIdle");
        vkFreeCommandBuffers(dev, pool, 1, &cmd);
    };
    auto transitionImage = [&](VkCommandBuffer cmd, VkImage img,
                               VkImageLayout oldL, VkImageLayout newL,
                               VkAccessFlags srcAccess, VkAccessFlags dstAccess,
                               VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage) {
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.oldLayout = oldL; b.newLayout = newL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = img;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        b.srcAccessMask = srcAccess; b.dstAccessMask = dstAccess;
        vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &b);
    };

    // Schritt 1: VulkanSetup initialisieren
    VulkanSetup::Options opts;
    opts.appName = "ComputeShaderExample";
    opts.apiVersion = VK_API_VERSION_1_2;
    opts.enableValidation = true; 
    VulkanSetup vk(opts);

    // Command pool
    VkCommandPoolCreateInfo poolCI{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolCI.queueFamilyIndex = vk.computeQueueFamily();
    poolCI.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool cmdPool{};
    VK_CHECK(vkCreateCommandPool(vk.device(), &poolCI, nullptr, &cmdPool), "vkCreateCommandPool");

    //Get local image to perform FAST
    cv::Mat image = cv::imread("/Users/olehoffmann/Documents/TUM/Studium/4. Semester/Guided Research/Coding/Vulkan Feature Extraction/Images/church.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat gray = image;
    if (image.empty()) {
        std::cerr << "Fehler beim Laden des Bildes!" << std::endl;
        return -1;
    }
    std::cout << "Bildgröße: " << image.cols << "x" << image.rows << std::endl;
    if(image.channels() == 3) cv::cvtColor(image, image, cv::COLOR_BGR2RGBA);
    else if(image.channels() == 1) cv::cvtColor(image, image, cv::COLOR_GRAY2RGBA);

    //Run FAST with CPU 
    int threshold = 76;
    bool nonmaxSuppression = false;
    std::vector<cv::KeyPoint> kps;
    cv::FAST(gray, kps, threshold, nonmaxSuppression, cv::FastFeatureDetector::TYPE_9_16);
    
    cv::Mat color;
    cv::cvtColor(gray, color, cv::COLOR_GRAY2BGR);
    const int radius = 6;
    const cv::Scalar red(0, 0, 255); // BGR
    for (const auto& kp : kps) {
        cv::Point center(cvRound(kp.pt.x), cvRound(kp.pt.y));
        cv::circle(color, center, radius, red, 2, cv::LINE_AA); // thickness=2 ring
    }
    if (!cv::imwrite("out_CPU.png", color)) {
        std::cerr << "Konnte out_CPU.png nicht speichern!" << std::endl;
        return -1;
    }


    uint32_t width = image.cols;
    uint32_t height = image.rows;
    VkDeviceSize imageSize = width * height * 4; 
    VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;

    auto createImage = [&](VkImage& image, VkDeviceMemory& mem) {
        VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.extent = {width, height, 1};
        ici.mipLevels = 1; ici.arrayLayers = 1;
        ici.format = format;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        ici.samples = VK_SAMPLE_COUNT_1_BIT; ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VK_CHECK(vkCreateImage(vk.device(), &ici, nullptr, &image), "vkCreateImage");
        VkMemoryRequirements mr{}; vkGetImageMemoryRequirements(vk.device(), image, &mr);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = mr.size;
        mai.memoryTypeIndex = findMemoryType(mr.memoryTypeBits, vk.physicalDevice(), VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VK_CHECK(vkAllocateMemory(vk.device(), &mai, nullptr, &mem), "vkAllocateMemory(image)");
        VK_CHECK(vkBindImageMemory(vk.device(), image, mem, 0), "vkBindImageMemory");
    };

    auto makeView = [&](VkImage img)->VkImageView{
        VkImageViewCreateInfo iv{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        iv.image = img; iv.viewType = VK_IMAGE_VIEW_TYPE_2D; iv.format = format;
        iv.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0,1,0,1};
        VkImageView v{}; VK_CHECK(vkCreateImageView(vk.device(), &iv, nullptr, &v), "vkCreateImageView");
        return v;
    };

    //4. Create input and output images
    VkImage inImage{}, outImage{};
    VkDeviceMemory inMem{}, outMem{};
    createImage(inImage, inMem);
    createImage(outImage, outMem);
    VkImageView inView = makeView(inImage);
    VkImageView outView = makeView(outImage);

    //5. Staging buffer & upload to inImage
    VkBuffer staging{}; 
    VkDeviceMemory stagingMem{};
    {
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = imageSize; 
        bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        VK_CHECK(vkCreateBuffer(vk.device(), &bi, nullptr, &staging), "vkCreateBuffer(staging)");
        VkMemoryRequirements mr{}; vkGetBufferMemoryRequirements(vk.device(), staging, &mr);
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
        VkCommandBuffer cmd = beginOneShot(vk.device(), cmdPool);
        transitionImage(cmd, inImage,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            0, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageOffset = {0,0,0};
        region.imageExtent = {width, height, 1};
        vkCmdCopyBufferToImage(cmd, staging, inImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        transitionImage(cmd, inImage,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        // outImage ready for compute writes
        transitionImage(cmd, outImage,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            0, VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        endOneShot(vk.device(), vk.computeQueue(), cmdPool, cmd);
    }

    // ----- 8) Descriptors: binding 0 = in, 1 = out -----
    VkDescriptorSetLayoutBinding b0{}; b0.binding = 0; b0.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; b0.descriptorCount = 1; b0.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutBinding b1{}; b1.binding = 1; b1.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; b1.descriptorCount = 1; b1.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutBinding bindings[2] = {b0, b1};

    VkDescriptorSetLayoutCreateInfo dlci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dlci.bindingCount = 2; dlci.pBindings = bindings;
    VkDescriptorSetLayout dsl{};
    VK_CHECK(vkCreateDescriptorSetLayout(vk.device(), &dlci, nullptr, &dsl), "vkCreateDescriptorSetLayout");

    VkDescriptorPoolSize poolSizes[1] = {{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2}};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.poolSizeCount = 1; dpci.pPoolSizes = poolSizes; dpci.maxSets = 1;
    VkDescriptorPool dpool{};
    VK_CHECK(vkCreateDescriptorPool(vk.device(), &dpci, nullptr, &dpool), "vkCreateDescriptorPool");

    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = dpool; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &dsl;
    VkDescriptorSet dset{};
    VK_CHECK(vkAllocateDescriptorSets(vk.device(), &dsai, &dset), "vkAllocateDescriptorSets");

    VkDescriptorImageInfo inInfo{};  inInfo.imageView = inView;   inInfo.imageLayout  = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorImageInfo outInfo{}; outInfo.imageView = outView; outInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet writes[2]{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = dset; writes[0].dstBinding = 0; writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; writes[0].descriptorCount = 1; writes[0].pImageInfo = &inInfo;
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = dset; writes[1].dstBinding = 1; writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; writes[1].descriptorCount = 1; writes[1].pImageInfo = &outInfo;
    vkUpdateDescriptorSets(vk.device(), 2, writes, 0, nullptr);

        // ----- 9) Pipeline (load comp.spv) -----
    auto code = readFile("/Users/olehoffmann/Documents/TUM/Studium/4. Semester/Guided Research/Coding/Vulkan Feature Extraction/build/shaders/comp.spv");
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = code.size();
    smci.pCode = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule sm{};
    VK_CHECK(vkCreateShaderModule(vk.device(), &smci, nullptr, &sm), "vkCreateShaderModule");

    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &dsl;
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
    cbi.commandPool = cmdPool; cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbi.commandBufferCount = 1;
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
        VkMemoryRequirements mr{}; vkGetBufferMemoryRequirements(vk.device(), readback, &mr);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = mr.size;
        mai.memoryTypeIndex = findMemoryType(mr.memoryTypeBits, vk.physicalDevice(),
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VK_CHECK(vkAllocateMemory(vk.device(), &mai, nullptr, &readMem), "vkAllocateMemory(readback)");
        VK_CHECK(vkBindBufferMemory(vk.device(), readback, readMem, 0), "vkBindBufferMemory(readback)");
    }
    {
        VkCommandBuffer cmd = beginOneShot(vk.device(), cmdPool);
        transitionImage(cmd, outImage,
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

        VkBufferImageCopy r{};
        r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        r.imageOffset = {0,0,0};
        r.imageExtent = {width, height, 1};
        vkCmdCopyImageToBuffer(cmd, outImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback, 1, &r);

        endOneShot(vk.device(), vk.computeQueue(), cmdPool, cmd);
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

    vkDestroyImageView(vk.device(), inView, nullptr);
    vkDestroyImageView(vk.device(), outView, nullptr);
    vkDestroyImage(vk.device(), inImage, nullptr);
    vkDestroyImage(vk.device(), outImage, nullptr);
    vkFreeMemory(vk.device(), inMem, nullptr);
    vkFreeMemory(vk.device(), outMem, nullptr);

    vkDestroyBuffer(vk.device(), staging, nullptr);
    vkFreeMemory(vk.device(), stagingMem, nullptr);

    vkDestroyCommandPool(vk.device(), cmdPool, nullptr);

    return 0;
}