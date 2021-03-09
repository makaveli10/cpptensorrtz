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
};


bool Vgg16Trt::build()
{
    // load weights
    weightMap = loadWeights(mParams.weightsFile);

    // create builder
    IBuilder* builder = createInferBuilder(gLogger);
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


/**
 * Uses the TensorRT API to create the network engine.  
**/
bool Vgg16Trt::createEngine(IBuilder* builder, IBuilderConfig* config)
{
    // Initialize NetworkDefinition
    INetworkDefinition* network = builder -> createNetworkV2(0U);

    // add input 
    auto data = network -> addInput(mParams.inputTensorName, DataType::kFLOAT, Dims3{3, mParams.inputH, mParams.inputW});
    assert(data);

    // add conv, relu blocks
    auto conv1 = network -> addConvolutionNd(*data, 64, DimsHW{3, 3}, weightMap["features.0.weight"], weightMap["features.0.bias"]);
    assert(conv1);
    conv1 -> setPaddingNd(DimsHW{1, 1});
    auto relu1 = network -> addActivation(*conv1 -> getOutput(0), ActivationType::kRELU);
    assert(relu1);

    auto conv2 = network -> addConvolutionNd(*relu1 -> getOutput(0), 64, DimsHW{3, 3}, weightMap["features.2.weight"], weightMap["features.2.bias"]);
    assert(conv2);
    conv2 -> setPaddingNd(DimsHW{1, 1});
    auto relu2 = network -> addActivation(*conv2 -> getOutput(0), ActivationType::kRELU);
    assert(relu2);

    // add pooling
    auto pool1 = network -> addPoolingNd(*relu2->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    assert(pool1);
    pool1 -> setStrideNd(DimsHW{2,2});

    auto conv3 = network -> addConvolutionNd(*pool1 -> getOutput(0), 128, DimsHW{3, 3}, weightMap["features.5.weight"], weightMap["features.5.bias"]);
    assert(conv3);
    conv3 -> setPaddingNd(DimsHW{1, 1});
    auto relu3 = network -> addActivation(*conv3 -> getOutput(0), ActivationType::kRELU);
    assert(relu3);

    auto conv4 = network -> addConvolutionNd(*relu3 -> getOutput(0), 128, DimsHW{3, 3}, weightMap["features.7.weight"], weightMap["features.7.bias"]);
    assert(conv4);
    conv4 -> setPaddingNd(DimsHW{1, 1});
    auto relu4 = network -> addActivation(*conv4 -> getOutput(0), ActivationType::kRELU);
    assert(relu4);

    // add pooling
    auto pool2 = network -> addPoolingNd(*relu4->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    assert(pool2);
    pool2 -> setStrideNd(DimsHW{2,2});

    auto conv5 = network -> addConvolutionNd(*pool2 -> getOutput(0), 256, DimsHW{3, 3}, weightMap["features.10.weight"], weightMap["features.10.bias"]);
    assert(conv5);
    conv5 -> setPaddingNd(DimsHW{1, 1});
    auto relu5 = network -> addActivation(*conv5 -> getOutput(0), ActivationType::kRELU);
    assert(relu5);

    auto conv6 = network -> addConvolutionNd(*relu5 -> getOutput(0), 256, DimsHW{3, 3}, weightMap["features.12.weight"], weightMap["features.12.bias"]);
    assert(conv6);
    conv6 -> setPaddingNd(DimsHW{1, 1});
    auto relu6 = network -> addActivation(*conv6 -> getOutput(0), ActivationType::kRELU);
    assert(relu6);

    auto conv7 = network -> addConvolutionNd(*relu6 -> getOutput(0), 256, DimsHW{3, 3}, weightMap["features.14.weight"], weightMap["features.14.bias"]);
    assert(conv7);
    conv7 -> setPaddingNd(DimsHW{1, 1});
    auto relu7 = network -> addActivation(*conv7 -> getOutput(0), ActivationType::kRELU);
    assert(relu7);

    // add pooling
    auto pool3 = network -> addPoolingNd(*relu7->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    assert(pool3);
    pool3 -> setStrideNd(DimsHW{2,2});

    auto conv8 = network -> addConvolutionNd(*pool3 -> getOutput(0), 512, DimsHW{3, 3}, weightMap["features.17.weight"], weightMap["features.17.bias"]);
    assert(conv8);
    conv8 -> setPaddingNd(DimsHW{1, 1});
    auto relu8 = network -> addActivation(*conv8 -> getOutput(0), ActivationType::kRELU);
    assert(relu8);

    auto conv9 = network -> addConvolutionNd(*relu8 -> getOutput(0), 512, DimsHW{3, 3}, weightMap["features.19.weight"], weightMap["features.19.bias"]);
    assert(conv9);
    conv9 -> setPaddingNd(DimsHW{1, 1});
    auto relu9 = network -> addActivation(*conv9 -> getOutput(0), ActivationType::kRELU);
    assert(relu9);

    auto conv10 = network -> addConvolutionNd(*relu9 -> getOutput(0), 512, DimsHW{3, 3}, weightMap["features.21.weight"], weightMap["features.21.bias"]);
    assert(conv10);
    conv10 -> setPaddingNd(DimsHW{1, 1});
    auto relu10 = network -> addActivation(*conv10 -> getOutput(0), ActivationType::kRELU);
    assert(relu10);

    // add pooling
    auto pool4 = network -> addPoolingNd(*relu10->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    assert(pool4);
    pool4 -> setStrideNd(DimsHW{2,2});

    auto conv11 = network -> addConvolutionNd(*pool4 -> getOutput(0), 512, DimsHW{3, 3}, weightMap["features.24.weight"], weightMap["features.24.bias"]);
    assert(conv11);
    conv11 -> setPaddingNd(DimsHW{1, 1});
    auto relu11 = network -> addActivation(*conv11 -> getOutput(0), ActivationType::kRELU);
    assert(relu11);

    auto conv12 = network -> addConvolutionNd(*relu11 -> getOutput(0), 512, DimsHW{3, 3}, weightMap["features.26.weight"], weightMap["features.26.bias"]);
    assert(conv12);
    conv12 -> setPaddingNd(DimsHW{1, 1});
    auto relu12 = network -> addActivation(*conv12 -> getOutput(0), ActivationType::kRELU);
    assert(relu12);

    auto conv13 = network -> addConvolutionNd(*relu12 -> getOutput(0), 512, DimsHW{3, 3}, weightMap["features.28.weight"], weightMap["features.28.bias"]);
    assert(conv13);
    conv13 -> setPaddingNd(DimsHW{1, 1});
    auto relu13 = network -> addActivation(*conv13 -> getOutput(0), ActivationType::kRELU);
    assert(relu13);

    // add pooling
    auto pool5 = network -> addPoolingNd(*relu13->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    assert(pool5);
    pool5 -> setStrideNd(DimsHW{2,2});

    // add fully connected layers
    auto fc1 = network-> addFullyConnected(*pool5 -> getOutput(0), 4096, weightMap["classifier.0.weight"], weightMap["classifier.0.bias"]);
    assert(fc1);
    auto relu14 = network -> addActivation(*fc1 -> getOutput(0), ActivationType::kRELU);
    assert(relu14);

    auto fc2 = network-> addFullyConnected(*relu14 -> getOutput(0), 4096, weightMap["classifier.3.weight"], weightMap["classifier.3.bias"]);
    assert(fc2);
    auto relu15 = network -> addActivation(*fc2 -> getOutput(0), ActivationType::kRELU);
    assert(relu15);

    auto fc3 = network-> addFullyConnected(*relu15 -> getOutput(0), 1000, weightMap["classifier.6.weight"], weightMap["classifier.6.bias"]);
    
    // set ouput blob name
    fc3 -> getOutput(0) -> setName(mParams.outputTensorName);

    // mark the output
    network -> markOutput(*fc3 -> getOutput(0));

    // set batchsize and workspace size
    builder -> setMaxBatchSize(mParams.batchSize);
    config -> setMaxWorkspaceSize(1 << 28); // 256 MiB

    // build engine
    mEngine = builder -> buildEngineWithConfig(*network, *config);

    std::cout << "engine built." << std::endl;
    
    // destroy
    network -> destroy();

    // fere host mem
    for(auto& mem: weightMap)
    {
        free((void*)(mem.second.values));
    }

    if (mEngine == nullptr) return false;
    return true;
}


/**
 * Cleans up any state created in the VggTrt class
**/
bool Vgg16Trt::cleanUp()
{
    if (mContext != nullptr)
        mContext -> destroy();
    
    if (mEngine != nullptr)
        mEngine -> destroy();

    return true;
}


/**
 * Performs inference on the given input and 
 * writes the output from device to host memory.
**/
void Vgg16Trt::doInference(float* input, float* output, int batchSize)
{
    if (mContext==nullptr || mEngine==nullptr)
    {    
        std::cout << "deserializing.." << std::endl;
        if(!deserialize()) return;
    }
    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers
    assert(mEngine -> getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = mEngine -> getBindingIndex(mParams.inputTensorName);
    const int outputIndex = mEngine -> getBindingIndex(mParams.outputTensorName);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * mParams.inputH * mParams.inputW * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * mParams.outputSize * sizeof(float)));

    // create cuda stream
    cudaStream_t cudaStream;
    CHECK(cudaStreamCreate(&cudaStream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * mParams.inputH * mParams.inputW * sizeof(float), cudaMemcpyHostToDevice));
    mContext -> enqueue(batchSize, buffers, cudaStream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * mParams.outputSize * sizeof(float), cudaMemcpyDeviceToHost));
    cudaStreamSynchronize(cudaStream);

    // release stream 
    cudaStreamDestroy(cudaStream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}


/**
 * Uses the serialized engine file and reads
 * data into a stream to deserialize the cuda 
 * engine from the stream. 
**/
bool Vgg16Trt::deserialize(){
    if (mContext != nullptr && mEngine != nullptr)
    {
        return true;
    }

    if (mEngine == nullptr)
    {
        char* trtModelStream{nullptr};
        size_t size{0};

        // open file
        std::ifstream f(mParams.trtEngineFile, std::ios::binary);

        if (f.good())
        {
            // get size
            f.seekg(0, f.end);
            size = f.tellg();
            f.seekg(0, f.beg);

            trtModelStream = new char[size];

            // read data as a block
            f.read(trtModelStream, size);
            f.close();
        }

        if (trtModelStream == nullptr)
        {
            return false;
        }

        // deserialize
        IRuntime* runtime = createInferRuntime(gLogger);
        assert(runtime);

        mEngine = runtime -> deserializeCudaEngine(trtModelStream, size, 0);
        assert(mEngine != nullptr);

        // clean up
        runtime -> destroy();
        delete[] trtModelStream;

    }

    std::cout << "deserialized engine successfully." << std::endl;

    // create execution context
    mContext = mEngine -> createExecutionContext();
    assert(mContext != nullptr);

    return true;
}


/**
 * Loads weights from Weights file.
 * Tensorrt weights file have a simple space delimited format:
 * [type] [size] <data x size in hex>
 * 
 * Returns weightMap
**/
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
        Weights wt{DataType::kFLOAT, nullptr, 0};
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


/**
 * Initializes AlexnetTrt class params in the 
 * AlexnetTrtParams structure.
**/
Vgg16TrtParams initializeParams()
{
    Vgg16TrtParams params;

    params.batchSize = 1;
    params.fp16 = true;

    params.inputH = 224;
    params.inputW = 224;
    params.outputSize = 1000;

    // change weights file name here
    params.weightsFile = "../vgg16.wts";

    // change engine file name here
    params.trtEngineFile = "vgg16.engine";
    return params;
}


int main(int argc, char** argv){
    if (argc != 2)
    {
        std::cerr << "Invalid args. please check." << std::endl;
        std::cerr <<"./alexnet -r  // create engine" << std::endl;
        return 0;
    }

    Vgg16TrtParams params = initializeParams();
    Vgg16Trt vgg16(params);

    // check if engine exists already
    std::ifstream f(params.trtEngineFile, std::ios::binary);
    
    // if engine does not exists build, serialize and save
    if(!f.good())
    {
        std::cout << "Building network ..." << std::endl;
        f.close();
        vgg16.build();
    }
    else
    {
        // deserialize
        std::cout << "engine already exists ..." << std::endl;
        // vgg16.deserialize();
    }

    // create data
    float data[3 * params.inputH * params.inputW];
    for(int i=0; i<3*params.inputH*params.inputW; i++)
    {
        data[i] = 1;
    }
    
    // run inference
    float prob[params.outputSize];
    for(int i=0; i<100; i++)
    {
        auto start = std::chrono::system_clock::now();
        vgg16.doInference(data, prob, 1);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }

    // cleanup
    bool cleaned = vgg16.cleanUp();
    
    std::cout << "\nOutput:\n\n";
    for (unsigned int i = 0; i < params.outputSize; i++)
    {
        std::cout << prob[i] << ", ";
        if (i % 10 == 0) std::cout << i / 10 << std::endl;
    }
    std::cout << std::endl;

    return 0;
}