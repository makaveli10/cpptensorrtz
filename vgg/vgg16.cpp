#include <string>
#include <vector>
#include <fstream>
#include <map>
#include <chrono>

#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)


using namespace nvinfer1;

static Logger gLogger;

/*
 The Vggtrt params required to create tensorrt engine.
*/
struct Vgg16TrtParams{
    int32_t batchSize{1};              // Number of inputs in a batch
    bool int8{false};                  // Allow runnning the network in Int8 mode.
    bool fp16{false};                  // Allow running the network in FP16 mode.
    const char* inputTensorName = "data";
    const char* outputTensorName = "prob";

    int inputW;                // The input width of the network.
    int inputH;                // The input height of the the network.
    int outputSize;           // THe output size of the network.
    std::string weightsFile;   // Weights file filename.
    std::string trtEngineFile; // trt engine file name
};


class Vgg16Trt{
public:
    Vgg16Trt(const Vgg16TrtParams params)
    : mParams(params)
    , mEngine(nullptr)
    , mContext(nullptr)
    {
    }

    // Function that builds the Tensorrt network engine.
    bool build();

    // Runs the Tensorrt network inference engine on a sample.
    void doInference(float* input, float* output, int batchSize);

    // Cleans up any state created in the VggTrt class.
    bool cleanUp();

private:
    Vgg16TrtParams mParams;   // The parameters for Vgg.
    
    std::map<std::string, Weights> weightMap; // The weight value map.

    ICudaEngine* mEngine;  // The tensorrt engine used to run the network.

    IExecutionContext* mContext; // The TensorRT execution context to run inference.

    // Uses the API to create the network.
    bool createEngine(IBuilder* builder, IBuilderConfig* config);

    // Loads weights from weights file
    std::map<std::string, Weights> loadWeights(const std::string& file);

    bool deserialize();
}


bool Vgg16Trt::build()
{
    // load weights
    weightMap = loadWeights(mParams.weightsFile);

    // create builder
    IBuilder* builder = createInferBuilder(&gLogger);
    assert(builder != nullptr);

    // create builder config
    IBuilderConfig* config = builder -> createBuilderConfig();
    assert(config != nullptr);

    // create engine
    bool created = createEngine(builder, config);
    if (!created){
        std::cout << "Engine creation failed. Check logs." << std::endl;
        return false;
    }

    // serilaize engine
    assert(mEngine != nullptr);
    IHostMemory* modelStream{nullptr};

    std::cout << "Serilaizing model to stream ..." << std::endl;
    modelStream = mEngine -> serialize();
    assert(modelStream != nullptr);
    
    // destroy
    config -> destroy();
    builder -> destroy();

    // write serialized engine to file
    std::ofstream trtFile(mParams.trtEngineFile);
    if(!trtFile){
        std::cerr << "Unable to open engine file." << std::endl;
        return false;
    }

    // write serialized engine to file
    trtFile.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
    std::cout << "Engine serialized and saved." << std::endl;

    // clean up
    modelStream -> destroy();

    return true;
}


// VGG(
//   (features): Sequential(
//     (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
//     (1): ReLU(inplace=True)
//     (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
//     (3): ReLU(inplace=True)
//     (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
//     (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
//     (6): ReLU(inplace=True)
//     (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
//     (8): ReLU(inplace=True)
//     (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
//     (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
//     (11): ReLU(inplace=True)
//     (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
//     (13): ReLU(inplace=True)
//     (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
//     (15): ReLU(inplace=True)
//     (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
//     (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
//     (18): ReLU(inplace=True)
//     (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
//     (20): ReLU(inplace=True)
//     (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
//     (22): ReLU(inplace=True)
//     (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
//     (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
//     (25): ReLU(inplace=True)
//     (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
//     (27): ReLU(inplace=True)
//     (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
//     (29): ReLU(inplace=True)
//     (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
//   )
//   (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
//   (classifier): Sequential(
//     (0): Linear(in_features=25088, out_features=4096, bias=True)
//     (1): ReLU(inplace=True)
//     (2): Dropout(p=0.5, inplace=False)
//     (3): Linear(in_features=4096, out_features=4096, bias=True)
//     (4): ReLU(inplace=True)
//     (5): Dropout(p=0.5, inplace=False)
//     (6): Linear(in_features=4096, out_features=1000, bias=True)
//   )
// )

/**
 * Cleans up any state created in the VggTrt class
**/
bool Vgg16Trt::cleanup()
{
    if (mEngine != nullptr)
    {
        mEngine -> destroy();
        mEngine = nullptr;
    }

    if (mCOntext != nullptr)
    {
        mContext -> destroy();
        mContext = nullptr;
    }

    return true;
}


std::map<std::string, Weights> Vgg16Trt::loadWeights(const std::string& file)
{
    std::cout<<"Loading weights ..." << std::endl;
    std::map<std::string, Weights> weightMap;

    // open weight file
    std::ifstream input(file, std::ios::binary);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weights file.");

    // Loop through and create weight map
    while(count--)
    {
        // Initialize weight with Datatype and nullptr
        Weight wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and size(decimal format) of blob
        std::string name;
        input >> name >> std::dec >> size;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t i=0; i<size; ++i){
            input >> std::hex >> val[i];
        }

        // assign weights and size to WeightMap
        wt.values = val;
        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}