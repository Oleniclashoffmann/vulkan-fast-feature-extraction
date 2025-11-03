#include "VulkanSetup.h"
#include <iostream>
#include <cstring>

#if defined(__APPLE__)
#define VK_USE_PLATFORM_MACOS_MVK
#endif

#ifndef VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME
#define VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME "VK_KHR_portability_subset"
#endif

// --- Helpers ---------------------------------------------------------------

static void vkCheck(VkResult r, const char* where) {
    if (r != VK_SUCCESS) {
        throw std::runtime_error(std::string("Vulkan error at ") + where + " (VkResult=" + std::to_string(r) + ")");
    }
}

bool VulkanSetup::hasExtension(const std::vector<VkExtensionProperties>& list, const char* name) {
    for (const auto& e : list) {
        if (std::strcmp(e.extensionName, name) == 0) return true;
    }
    return false;
}

bool VulkanSetup::hasLayer(const std::vector<VkLayerProperties>& list, const char* name) {
    for (const auto& l : list) {
        if (std::strcmp(l.layerName, name) == 0) return true;
    }
    return false;
}

// --- Lifecycle -------------------------------------------------------------

VulkanSetup::VulkanSetup(const Options& opts) {
    createInstance(opts);
    pickPhysicalDeviceAndQueue();
    createDeviceAndQueue();
}

VulkanSetup::~VulkanSetup() {
    if (device_ != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device_);
        vkDestroyDevice(device_, nullptr);
    }
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
    }
}

VulkanSetup::VulkanSetup(VulkanSetup&& o) noexcept {
    *this = std::move(o);
}

VulkanSetup& VulkanSetup::operator=(VulkanSetup&& o) noexcept {
    if (this == &o) return *this;
    // Clean current
    this->~VulkanSetup();
    // Move
    instance_ = o.instance_;
    physicalDevice_ = o.physicalDevice_;
    computeQueueFamilyIndex_ = o.computeQueueFamilyIndex_;
    device_ = o.device_;
    computeQueue_ = o.computeQueue_;
    portabilityEnabled_ = o.portabilityEnabled_;

    // Null out source
    o.instance_ = VK_NULL_HANDLE;
    o.physicalDevice_ = VK_NULL_HANDLE;
    o.device_ = VK_NULL_HANDLE;
    o.computeQueue_ = VK_NULL_HANDLE;
    return *this;
}

// --- Steps -----------------------------------------------------------------

void VulkanSetup::createInstance(const Options& opts) {
    // Instance extensions
    uint32_t extCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> availExts(extCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &extCount, availExts.data());

    std::vector<const char*> enabledExts;
    // KHR portability enumeration is recommended for MoltenVK/portability
    if (hasExtension(availExts, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
        enabledExts.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    }

    // (Optional) Debug utils if validation enabled and available
    const char* debugExt = "VK_EXT_debug_utils";
    if (opts.enableValidation && hasExtension(availExts, debugExt)) {
        enabledExts.push_back(debugExt);
    }

    // Layers (optional)
    std::vector<const char*> enabledLayers;
#ifndef NDEBUG
    if (opts.enableValidation) {
        uint32_t layerCount = 0;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
        std::vector<VkLayerProperties> layers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, layers.data());
        if (hasLayer(layers, "VK_LAYER_KHRONOS_validation")) {
            enabledLayers.push_back("VK_LAYER_KHRONOS_validation");
        }
    }
#endif

    VkApplicationInfo appInfo{ };
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = opts.appName;
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "NoEngine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = opts.apiVersion;

    VkInstanceCreateInfo ci{ };
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &appInfo;
    if (hasExtension(availExts, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
        ci.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        portabilityEnabled_ = true;
    }
    ci.enabledExtensionCount = static_cast<uint32_t>(enabledExts.size());
    ci.ppEnabledExtensionNames = enabledExts.empty() ? nullptr : enabledExts.data();
    ci.enabledLayerCount = static_cast<uint32_t>(enabledLayers.size());
    ci.ppEnabledLayerNames = enabledLayers.empty() ? nullptr : enabledLayers.data();

    vkCheck(vkCreateInstance(&ci, nullptr, &instance_), "vkCreateInstance");
}

void VulkanSetup::pickPhysicalDeviceAndQueue() {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance_, &deviceCount, nullptr);
    if (deviceCount == 0) throw std::runtime_error("No Vulkan-capable devices found.");

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance_, &deviceCount, devices.data());

    for (auto dev : devices) {
        // Find a compute queue family
        uint32_t qCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &qCount, nullptr);
        std::vector<VkQueueFamilyProperties> qprops(qCount);
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &qCount, qprops.data());

        for (uint32_t i = 0; i < qCount; ++i) {
            if (qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                physicalDevice_ = dev;
                computeQueueFamilyIndex_ = i;
                return;
            }
        }
    }
    throw std::runtime_error("No device with a compute-capable queue family found.");
}

void VulkanSetup::createDeviceAndQueue() {
    // Device extensions (portability subset is required by portability drivers like MoltenVK)
    uint32_t devExtCount = 0;
    vkEnumerateDeviceExtensionProperties(physicalDevice_, nullptr, &devExtCount, nullptr);
    std::vector<VkExtensionProperties> devExts(devExtCount);
    vkEnumerateDeviceExtensionProperties(physicalDevice_, nullptr, &devExtCount, devExts.data());

    std::vector<const char*> enabledDevExts;
    if (hasExtension(devExts, "VK_KHR_portability_subset")) {
        enabledDevExts.push_back("VK_KHR_portability_subset");
    }

    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{ };
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = computeQueueFamilyIndex_;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;

    VkPhysicalDeviceFeatures features{ }; // default; add features if needed

    VkDeviceCreateInfo dci{ };
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.pEnabledFeatures = &features;
    dci.enabledExtensionCount = static_cast<uint32_t>(enabledDevExts.size());
    dci.ppEnabledExtensionNames = enabledDevExts.empty() ? nullptr : enabledDevExts.data();

    vkCheck(vkCreateDevice(physicalDevice_, &dci, nullptr, &device_), "vkCreateDevice");
    vkGetDeviceQueue(device_, computeQueueFamilyIndex_, 0, &computeQueue_);
}
