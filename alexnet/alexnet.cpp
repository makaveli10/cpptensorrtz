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
 The Alextrt params required to create tensorrt engine.
*/
struct AlexnetTrtParams{
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


class AlexnetTrt{
public:
    AlexnetTrt(const AlexnetTrtParams &params)
    : mParams(params)
    , mEngine(nullptr)
    , mContext(nullptr)
    {
    }

    // Function that builds the Tensorrt network engine.
    bool build();

    // Runs the Tensorrt network inference engine on a sample.
    void doInference(float* input, float* output, int batchSize);

    // Cleans up any state created in the AlexnetTrt class.
    bool cleanUp();

private:
    AlexnetTrtParams mParams;   // The parameters for Alexnet.
    
    std::map<std::string, Weights> weightMap; // The weight value map.

    ICudaEngine* mEngine;  // The tensorrt engine used to run the network.

    IExecutionContext* mContext; // The TensorRT execution context to run inference.

    // Uses the API to create the network.
    bool createEngine(IBuilder* builder, IBuilderConfig* config);

    // Loads weights from weights file
    std::map<std::string, Weights> loadWeights(const std::string& file);

    bool deserialize();
};


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
    IBuilder* builder = createInferBuilder(gLogger);
    assert(builder != nullptr); 

    // Create builder config
    IBuilderConfig* builderConfig = builder -> createBuilderConfig();
    assert(builderConfig != nullptr);

    // create cuda engine
    bool created = createEngine(builder, builderConfig);

    if (!created){
        std::cout << "Engine creation failed. Check logs." << std::endl;
        return false;
    }

    assert(mEngine != nullptr);
    IHostMemory* modelStream{nullptr};

    std::cout << "Serializing engine to stream ..." << std::endl;
    // Serialize the engine
    modelStream = mEngine -> serialize();
    assert(modelStream != nullptr);

    // destroy everything
    builderConfig -> destroy();
    builder -> destroy();

    // open file in write mode
    std::ofstream trtfile(mParams.trtEngineFile);
    if(!trtfile){
        std::cerr << "Unable to open engine file." << std::endl;
        return false;
    }

    // write serialized engine to file
    trtfile.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
    std::cout << "Engine serialized and saved." << std::endl;

    // clean up
    modelStream -> destroy();

    return true;
}


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
    IConvolutionLayer* conv1 = network -> addConvolutionNd(*dataInput, 64, DimsHW{11,11}, weightMap["features.0.weight"], weightMap["features.0.bias"]);
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

    IConvolutionLayer* conv2 = network -> addConvolutionNd(*pool1->getOutput(0), 192, DimsHW{5,5}, weightMap["features.3.weight"], weightMap["features.3.bias"]);
    assert(conv2);
    conv2 -> setPaddingNd(DimsHW{2,2});
    IActivationLayer* relu2 = network -> addActivation(*conv2->getOutput(0), ActivationType::kRELU);
    assert(relu2);
    IPoolingLayer* pool2 = network -> addPoolingNd(*relu2->getOutput(0), PoolingType::kMAX, DimsHW{3,3});
    assert(pool2);
    pool2 -> setStrideNd(DimsHW{2,2});

    IConvolutionLayer* conv3 = network -> addConvolutionNd(*pool2->getOutput(0), 384, DimsHW{3,3}, weightMap["features.6.weight"], weightMap["features.6.bias"]);
    assert(conv3);
    conv3->setPaddingNd(DimsHW{1, 1});
    IActivationLayer* relu3 = network -> addActivation(*conv3->getOutput(0), ActivationType::kRELU);
    assert(relu3);

    IConvolutionLayer* conv4 = network -> addConvolutionNd(*relu3->getOutput(0), 256, DimsHW{3,3}, weightMap["features.8.weight"], weightMap["features.8.bias"]);
    assert(conv4);
    conv4->setPaddingNd(DimsHW{1, 1});
    IActivationLayer* relu4 = network -> addActivation(*conv4->getOutput(0), ActivationType::kRELU);
    assert(relu4);

    IConvolutionLayer* conv5 = network -> addConvolutionNd(*relu4->getOutput(0), 256, DimsHW{3,3}, weightMap["features.10.weight"], weightMap["features.10.bias"]);
    assert(conv5);
    conv5->setPaddingNd(DimsHW{1, 1});
    IActivationLayer* relu5 = network -> addActivation(*conv5->getOutput(0), ActivationType::kRELU);
    assert(relu5);
    IPoolingLayer* pool3 = network -> addPoolingNd(*relu5->getOutput(0), PoolingType::kMAX, DimsHW{3,3});
    assert(pool3);
    pool3 -> setStrideNd(DimsHW{2,2});

    // Add Fully Connected
    IFullyConnectedLayer* fc1 = network -> addFullyConnected(*pool3->getOutput(0), 4096, weightMap["classifier.1.weight"], weightMap["classifier.1.bias"]);
    assert(fc1);
    IActivationLayer* relu6 = network -> addActivation(*fc1->getOutput(0), ActivationType::kRELU);
    assert(relu6);

    IFullyConnectedLayer* fc2 = network -> addFullyConnected(*relu6->getOutput(0), 4096, weightMap["classifier.4.weight"], weightMap["classifier.4.bias"]);
    assert(fc2);
    IActivationLayer* relu7 = network -> addActivation(*fc2->getOutput(0), ActivationType::kRELU);
    assert(relu7);
    
    IFullyConnectedLayer* fc3 = network -> addFullyConnected(*relu7->getOutput(0), 1000, weightMap["classifier.6.weight"], weightMap["classifier.6.bias"]);
    assert(fc3);
    
    // set ouput blob name
    fc3 -> getOutput(0)->setName(mParams.outputTensorName);

    // mark the output
    network -> markOutput(*fc3->getOutput(0));

    // set batchsize and workspace size
    builder -> setMaxBatchSize(mParams.batchSize);
    config -> setMaxWorkspaceSize(1<<28);    //16__MiB
    mEngine = builder->buildEngineWithConfig(*network, *config);

    std::cout << "engine built." << std::endl;

    // destroy network
    network -> destroy();

    // Release host memory
    for(auto& mem: weightMap)
    {
        free((void*)(mem.second.values));
    }
    
    if (mEngine == nullptr) return false;
    return true;
}


/**
 * Performs inference on the given input and 
 * writes the output from device to host memory.
**/
void AlexnetTrt::doInference(float* input, float* output, int batchSize){
    // check if context is null
    if (mContext == nullptr){
        // create execution context
        if (mEngine == nullptr){
            // deserialize engine
            if (!deserialize()){
                std::cerr << "Unable to deserialize the cuda engine" << std::endl;
                return;
            }
        }
        
        std::cout << "deserialized engine successfully." << std::endl;

        // create execution context
        mContext = mEngine -> createExecutionContext();
        assert(mContext != nullptr);
    }

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(mEngine->getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = mEngine -> getBindingIndex(mParams.inputTensorName);
    const int outputIndex = mEngine -> getBindingIndex(mParams.outputTensorName);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * mParams.inputH * mParams.inputW * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * mParams.outputSize * sizeof(float)));

    // create cuda stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * mParams.inputH * mParams.inputW * sizeof(float), cudaMemcpyHostToDevice));
    mContext -> enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * mParams.outputSize * sizeof(float), cudaMemcpyDeviceToHost));
    cudaStreamSynchronize(stream);

    // release stream 
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}


/**
 * Uses the serialized engine file and reads
 * data into a stream to deserialize the cuda 
 * engine from the stream. 
**/
bool AlexnetTrt::deserialize(){
    char* trt_model_stream{nullptr};
    size_t size{0};

    // open file in binary read mode
    std::ifstream s(mParams.trtEngineFile, std::ios::binary);

    if (s.good())
    {
        // get length of file:
        s.seekg (0, s.end);
        size = s.tellg();
        s.seekg(0, s.beg);

        trt_model_stream = new char[size];

        // read data as a block
        s.read(trt_model_stream, size);
        s.close();
    }

    if (trt_model_stream==nullptr)
    {
        return false;
    }

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);

    mEngine = runtime -> deserializeCudaEngine(trt_model_stream, size, nullptr);
    assert(mEngine != nullptr);

    runtime -> destroy();
    return true;
}

/**
 * Cleans up any state created in the sample class
**/
bool AlexnetTrt::cleanUp(){
    if (mContext != nullptr)
        mContext -> destroy();
    
    if (mEngine != nullptr)
        mEngine -> destroy();
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


/**
 * Initializes AlexnetTrt class params in the 
 * AlexnetTrtParams structure.
**/
AlexnetTrtParams initializeParams()
{
    AlexnetTrtParams params;

    params.batchSize = 1;
    params.fp16 = true;

    params.inputH = 224;
    params.inputW = 224;
    params.outputSize = 1000;

    // change weights file name here
    params.weightsFile = "../alexnet.wts";

    // change engine file name here
    params.trtEngineFile = "alexnet.engine";
    return params;
}

int main(int argc, char** argv){
    if (argc != 2){
        std::cerr << "Incorrect args. Please check." << std::endl;
        std::cerr <<"./alexnet -r  // create engine" << std::endl;
    }

    AlexnetTrtParams params = initializeParams(); 
    AlexnetTrt alex(params);

    std::ifstream f{params.trtEngineFile};

    if (!f.good())
    {
        std::cout << "Building network ..." << std::endl;
        f.close();
        // serialize engine file
        alex.build();

    }
    else
    {
        std::cout << "Engine already exists. Deserializing ..." << std::endl;
    }

    
    float data[3 * params.inputH * params.inputW];
    for (int i = 0; i < 3 * params.inputH * params.inputW; i++)
        data[i] = 1;

    // Run inference
    float prob[params.outputSize];
    for (int i = 0; i < 100; i++) {
        auto start = std::chrono::system_clock::now();
        alex.doInference(data, prob, 1);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }

    // cleanup
    bool cleaned = alex.cleanUp();

    // Print histogram of the output distribution
    std::cout << "\nOutput:\n\n";
    for (unsigned int i = 0; i < 1000; i++)
    {
        std::cout << prob[i] << ", ";
        if (i % 10 == 0) std::cout << i / 10 << std::endl;
    }
    std::cout << std::endl;

    return 0;
}
