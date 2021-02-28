#include <string>
#include <vector>
#include <fstream>

#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include "common.h"

using namespace nvinfer1;

static Logger gLogger;

/*
 The Alextrt params required to create tensorrt engine.
*/
struct AlexnetTrtParams{
    int32_t batchSize{1};              // Number of inputs in a batch
    bool int8{false};                  // Allow runnning the network in Int8 mode.
    bool fp16{false};                  // Allow running the network in FP16 mode.
    const char* inputTensorName = "data";
    const char* outputTensorName = "prob";

    int inputW;              // The input width of the network.
    int inputH;              // The input height of the the network.
    int output_size;         // THe output size of the network.
    std::string weightsFile; // Weights file filename.
};


class AlexnetTrt{
public:
    AlexnetTrt(const AlexnetTrtParams &params)
    : mParams(&params)
    , mEngine(nullptr)
    , modelStream(nullptr)
    {
    }

    // Function that builds the Tensorrt network engine.
    bool build();

    // Runs the Tensorrt network inference engine on a sample.
    void doInference();

    // Cleans up any state created in the AlexnetTrt class.
    bool cleanUp();

private:
    AlexnetTrtParams mParams;   // The parameters for Alexnet.

    IHostMemory* modelStream;   // A IHostMemory object that will store the serialized engine.
    
    std::map<std::string, Weights> weightMap; // The weight value map.

    std::vector<IHostMemory> weightsMem;  // Host weights memory holder.

    ICudaEngine* mEngine;  // The tensorrt engine used to run the network.

    // Uses the API to create the network.
    bool createEngine(IBuilder* builder, IBuilderConfig* config);

    // Loads weights from weights file
    std::map<std::string, Weights> loadWeights(const std::string& file);
}


/**
 * Creates the network, configures the builder and creates the network engine.
 * 
 * This function creates the Alexnet network by using the API to create a model and builds
 * the engine that will be used to run Alexnet (mEngine)
 * 
 * Returns true if the engine was created successfully and false otherwise
**/
bool AlexnetTrt::build(){
    // load weights
    weightMap = loadWeights(mParams.weightsFile);

    // Create the builder
    IBuilder* builder = createInferBuilder(&gLogger);
    assert(builder != nullptr); 

    // Create builder config
    IBuilderConfig* builderConfig = builder.createBuilderConfig();
    assert(builderConfig != nullptr);

    // create cuda engine
    bool created = createEngine(builder, config)

    if (!created){
        std::cout << "Engine creation failed. Check logs." << std::endl;
        return false;
    }

    assert(mEngine != nullptr)

    // Serialize the engine
    modelStream = mEngine -> serialize();
    
    // destroy everything
    mEngine -> destroy();
    config -> destroy();
    builder -> destroy();
    
    return true;
}


// AlexNet(
//   (features): Sequential(
//     (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
//     (1): ReLU(inplace=True)
//     (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
//     (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
//     (4): ReLU(inplace=True)
//     (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
//     (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
//     (7): ReLU(inplace=True)
//     (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
//     (9): ReLU(inplace=True)
//     (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
//     (11): ReLU(inplace=True)
//     (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
//   )
//   (avgpool): AdaptiveAvgPool2d(output_size=(6, 6))
//   (classifier): Sequential(
//     (0): Dropout(p=0.5, inplace=False)
//     (1): Linear(in_features=9216, out_features=4096, bias=True)
//     (2): ReLU(inplace=True)
//     (3): Dropout(p=0.5, inplace=False)
//     (4): Linear(in_features=4096, out_features=4096, bias=True)
//     (5): ReLU(inplace=True)
//     (6): Linear(in_features=4096, out_features=1000, bias=True)
//   )
// )

/**
 * Uses the TensorRT API to create the network engine.
**/
bool AlexnetTrt::createEngine(IBuilder* builder, IBuilderConfig* config){
    // Initialize the network
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Add the Input layer to the network, with the input dimensions
    ITensor* dataInput = network->addInput(mParams.inputTensorName, DataType::kFLOAT, Dims3{3, mParams.inputH, mParams.inputW});
    assert(dataInput);

    // Add 2d conv with 64 11x11 kernels, strid 4 and padding 2
    IConvolutionLayer* conv1 = network -> addConvolutionNd(dataInput, 64, DimsHW{11,11}, weightMap["features.0.weight"], weightMap["features.0.bias"])
    assert(conv1);
    conv1 -> setStrideNd(DimsHW{4,4});
    conv1 -> setPaddingNd(DimsHW{2,2});
    // Add relu activation function
    IActivationLayer* relu1 = network -> addActivation(*conv1->getOutput(0), ActivationType::kRELU);
    assert(relu1);
    // Add pooling with stride 2, kernel ize 3x3
    IPoolingLayer* pool1 = network -> addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{3,3});
    assert(pool1);
    pool1 -> setStrideNd(DimsHW{2,2});

    IConvolutionLayer* conv2 = network -> addConvolutionNd(*pool1->getOutput(0), 192, DimsHW{5,5}, weightMap["features.3.weight"], weightMap["features.3.bias"])
    assert(conv2);
    conv2 -> setPaddingNd(DimsHW{2,2});
    IActivationLayer* relu2 = network -> addActivation(*conv2->getOutput(0), ActivationType::kRELU);
    assert(relu2);
    IPoolingLayer* pool2 = network -> addPoolingNd(*relu2->getOutput(0), PoolingType::kMAX, DimsHW{3,3});
    assert(pool2);
    pool2 -> setStrideNd(DimsHW{2,2});

    IConvolutionLayer* conv3 = network -> addConvolutionNd(*pool2->getOutput(0), 384, DimsHW{3,3}, weightMap["features.6.weight"], weightMap["features.6.bias"]);
    assert(conv3);
    IActivationLayer* relu3 = nework -> addActivation(*conv3->getOutput(0), ActivationType::kRELU);
    assert(relu3);

    IConvolutionLayer* conv4 = network -> addConvolutionNd(*relu3->getOutput(0), 256, DimsHW{3,3}, weightMap["features.8.weight"], weightMap["features.8.bias"]);
    assert(conv4);
    IActivationLayer* relu4 = nework -> addActivation(*conv4->getOutput(0), ActivationType::kRELU);
    assert(relu4);

    IConvolutionLayer* conv5 = network -> addConvolutionNd(*relu4->getOutput(0), 256, DimsHW{3,3}, weightMap["features.10.weight"], weightMap["features.10.bias"]);
    assert(conv5);
    IActivationLayer* relu5 = nework -> addActivation(*conv5->getOutput(0), ActivationType::kRELU);
    assert(relu5);
    IPoolingLayer* pool3 = network -> addPoolingNd(*relu5->getOutput(0), PoolingType::kMAX, DimsHW{3,3});
    assert(pool3);
    pool3 -> setStrideNd(DimsHW{2,2});

    // Add Fully Connected
    IFullyConnectedLayer* fc1 = network -> addFullyConnected(*pool3->getOutput(0), 4096, weightMap["classifier.1.weight"], weightMap["classifier.1.bias"]);
    assert(fc1);
    IActivationLayer* relu6 = nework -> addActivation(*fc1->getOutput(0), ActivationType::kRELU);
    assert(relu6);

    IFullyConnectedLayer* fc2 = network -> addFullyConnected(*relu6->getOutput(0), 4096, weightMap["classifier.4.weight"], weightMap["classifier.4.bias"]);
    assert(fc2);
    IActivationLayer* relu7 = nework -> addActivation(*fc2->getOutput(0), ActivationType::kRELU);
    assert(relu7);
    
    IFullyConnectedLayer* fc3 = network -> addFullyConnected(*relu7->getOutput(0), 1000, weightMap["classifier.6.weight"], weightMap["classifier.6.bias"]);
    assert(fc3);
    
    // set ouput blob name
    fc3->getOutput(0)->setName(mParams.outputTensorName);

    // mark the output
    network->markOutput(*fc3->getOutput(0));

    // set batchsize and workspace size
    builder->setMaxBatchSize(mParams.batchSize);
    config->setMaxWorkspaceSize(16 * 1<<20);    //16__MiB
    mEngine = builder->buildEngineWithConfig(network, config);

    std::cout << "engine built." << std::endl;

    // destroy network
    network -> destroy();

    // Release host memory
    for(auto& mem: weightMap)
    {
        free((void*)(mem.second.values));
    }
    
    if (engine == nullptr) return false;
    return true;
}


void AlexnetTrt::doInference(){

}

/**
 * Cleans up any state created in the sample class
**/
bool AlexnetTrt::cleanUp(){
    return true;
}


/**
 * Loads weights from Weights file.
 * Tensorrt weights file have a simple space delimited format:
 * [type] [size] <data x size in hex>
 * 
 * Returns weightMap
**/
std::map<std::string, Weights> AlexnetTrt::loadWeights(const std::string& file){
    std::cout << "Loading weights ..." << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file in binary mode
    std::ifstream input(file, std::ios::binary);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weights file.");

    // Loop through all the blobs and write weightMap
    while(count--){
        // Initialize weight with Datatype and nullptr
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and size(decimal format) of blob
        std::string name;
        input >> name >> std::dec >> size;
        // wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for(uint32_t x = 0; x < size; ++x){
            // Read weights in hex format
            input >> std::hex >> val[x];
        }

        // assign weights and size to WeightMap
        wt.values = val;
        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

int main(){
}
