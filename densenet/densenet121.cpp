#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"

#include <string>
#include <vector>
#include <fstream>
#include <map>
#include <chrono>
#include <cmath>

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
 Densenet TensorRT parameters used to create engine.
*/
struct DenseNet121Params
{
    /* data */
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

class DenseNet121{
public:
    DenseNet121(const DenseNet121Params params)
    : mParams(params)
    , mEngine(nullptr)
    , mContext(nullptr)
    {
    }

    // Function that builds the DenseNet121 TensorRT network engine.
    bool build();

    // Runs the Tensorrt network inference engine on a sample.
    void doInference(float* input, float* output, int batchSize);

    // Cleans up any state created in the densenet121 class.
    bool cleanUp();

private:
    DenseNet121Params mParams;   // The parameters for densenet121.
    
    std::map<std::string, Weights> weightMap; // The weight value map.

    ICudaEngine* mEngine;  // The tensorrt engine used to run the network.

    IExecutionContext* mContext; // The TensorRT execution context to run inference.

    // adds Dense Block to the network definition
    IConcatenationLayer* addDenseBlock(INetworkDefinition* network, ITensor* input, int numDenseLayers, std::string lname, float eps);

    // adds a denselayer to the network def
    IConvolutionLayer* addDenseLayer(INetworkDefinition* network, ITensor* input, std::string lname, float eps);

    IPoolingLayer* addTransition(INetworkDefinition* network, ITensor& input, int outch,std::string lname, float eps);

    // Creates a batchnorm layer
    IScaleLayer* addBatchNorm2d(INetworkDefinition* network, ITensor& input, std::string lname, float eps);

    // Uses the API to create the network.
    bool createEngine(IBuilder* builder, IBuilderConfig* config);

    // Loads weights from weights file
    std::map<std::string, Weights> loadWeights(const std::string& file);

    bool deserialize();
};


/**
 * Builds the tensorrt engine and serializes it.
**/
bool DenseNet121::build()
{
    // load weights
    weightMap = loadWeights(mParams.weightsFile);

    // create builder
    IBuilder* builder = createInferBuilder(gLogger);
    assert(builder);

    // create builder config
    IBuilderConfig* config = builder -> createBuilderConfig();
    assert(config);

    // create engine
    bool created = createEngine(builder, config);
    if(!created)
    {
        std::cerr << "Engine creation failed. Check logs." << std::endl;
        return false;
    }

    // serilaize engine
    assert(mEngine != nullptr);
    IHostMemory* modelStream{nullptr};
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

    trtFile.write(reinterpret_cast<const char*>(modelStream -> data()), modelStream -> size());
    std::cout << "Engine serialized and saved." << std::endl;

    // clean
    modelStream -> destroy();

    return true;
}


/**
 * Cleans up any state created in the DenseNetTrt class
**/
bool DenseNet121::cleanUp()
{
    if (mContext != nullptr)
        mContext -> destroy();
    
    if (mEngine != nullptr)
        mEngine -> destroy();

    return true;
}


IConcatenationLayer* DenseNet121::addDenseBlock(INetworkDefinition* network, ITensor* input, int numDenseLayers, std::string lname, float eps)
{
    IConvolutionLayer* c{nullptr};
    IConcatenationLayer* concat{nullptr};
    ITensor* inputTensors[numDenseLayers+1];
    inputTensors[0] = input;

    c = addDenseLayer(network, input, lname + ".denselayer" + std::to_string(1), eps);
    int i;
    for(i=1; i<numDenseLayers; i++)
    {
        // inch += 32;
        inputTensors[i] = c -> getOutput(0);
        concat = network -> addConcatenation(inputTensors, i+1);
        assert(concat);
        c = addDenseLayer(network, concat->getOutput(0), lname + ".denselayer" + std::to_string(i+1), eps);
    }
    inputTensors[numDenseLayers] = c -> getOutput(0);
    concat = network -> addConcatenation(inputTensors, numDenseLayers+1);
    assert(concat);
    return concat;
}


IConvolutionLayer* DenseNet121::addDenseLayer(INetworkDefinition* network, ITensor* input, std::string lname, float eps)
{
    // add Batchnorm
    IScaleLayer* bn1 = addBatchNorm2d(network, *input, lname + ".norm1", eps);

    // add relu
    IActivationLayer* relu1 = network -> addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    // add conv
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv1 = network -> addConvolutionNd(*relu1->getOutput(0), 128, DimsHW{1, 1}, weightMap[lname + ".conv1.weight"], emptywts);
    assert(conv1);
    conv1 -> setStrideNd(DimsHW{1, 1});

    // add Batchnorm
    IScaleLayer* bn2 = addBatchNorm2d(network, *conv1 -> getOutput(0), lname + ".norm2", eps);

    // add relu
    IActivationLayer* relu2 = network -> addActivation(*bn2->getOutput(0), ActivationType::kRELU);
    assert(relu2);

    // add conv
    IConvolutionLayer* conv2 = network -> addConvolutionNd(*relu2->getOutput(0), 32, DimsHW{3, 3}, weightMap[lname + ".conv2.weight"], emptywts);
    assert(conv2);
    conv2 -> setStrideNd(DimsHW{1, 1});
    conv2 -> setPaddingNd(DimsHW{1, 1});
    return conv2;
}


IPoolingLayer* DenseNet121::addTransition(INetworkDefinition* network, ITensor& input, int outch, std::string lname, float eps)
{
    // add batch norm
    IScaleLayer* bn1 = addBatchNorm2d(network, input, lname + ".norm", eps);

    // add relu activation
    IActivationLayer* relu1 = network -> addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    // add convolution layer
    // empty weights for no bias
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv1 = network -> addConvolutionNd(*relu1->getOutput(0), outch, DimsHW{1, 1}, weightMap[lname + ".conv.weight"], emptywts);
    assert(conv1);
    conv1 -> setStrideNd(DimsHW{1, 1});

    // add pooling
    IPoolingLayer* pool1 = network->addPoolingNd(*conv1->getOutput(0), PoolingType::kAVERAGE, DimsHW{2, 2});
    assert(pool1);
    pool1 -> setStrideNd(DimsHW{2, 2});
    pool1 -> setPaddingNd(DimsHW{0,0});
    return pool1;
}


/**
 * Adds a batch norm scale layer to the network definition. 
**/
IScaleLayer* DenseNet121::addBatchNorm2d(INetworkDefinition* network, ITensor& input, std::string lname, float eps)
{
    // get weights from weight map
    float* gamma = (float*)weightMap[lname + ".weight"].values;
    float* beta = (float*)weightMap[lname + ".bias"].values;
    float* mean = (float*)weightMap[lname + ".running_mean"].values;
    float* var = (float*)weightMap[lname + ".running_var"].values;

    int len = weightMap[lname + ".running_var"].count;
    std::cout << "len " << len << std::endl;

    // compute scale value
    float* scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for(int i=0; i<len; i++)
    {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};

    float* shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for(int i=0; i<len; i++)
    {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float* pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for(int i=0; i<len; i++)
    {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    // add scale, shift and power weights to weight map
    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;

    // add scale batchnorm to network
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

/**
 * Uses the TensorRT API to create the network engine.  
**/
bool DenseNet121::createEngine(IBuilder* builder, IBuilderConfig* config)
{
    // Initialize NetworkDefinition
    INetworkDefinition* network = builder -> createNetworkV2(0U);

    auto data = network -> addInput(mParams.inputTensorName, DataType::kFLOAT, Dims3{3, mParams.inputW, mParams.inputW});
    assert(data);

    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    auto conv0 = network -> addConvolutionNd(*data, 64, DimsHW{7, 7}, weightMap["features.conv0.weight"], emptywts);
    assert(conv0);
    conv0 -> setStrideNd(DimsHW{2, 2});
    conv0 -> setPaddingNd(DimsHW{3, 3});

    auto norm0 = addBatchNorm2d(network, *conv0 -> getOutput(0), "features.norm0", 1e-5);

    auto relu0 = network -> addActivation(*norm0 -> getOutput(0), ActivationType::kRELU);
    assert(relu0);

    auto pool0 = network -> addPoolingNd(*relu0 -> getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    assert(pool0);
    pool0 -> setStrideNd(DimsHW{2, 2});
    pool0 -> setPaddingNd(DimsHW{1, 1});
    
    auto dense1 = addDenseBlock(network, pool0 -> getOutput(0), 6, "features.denseblock1", 1e-5);
    auto transition1 = addTransition(network, *dense1 -> getOutput(0), 128, "features.transition1", 1e-5);

    auto dense2 = addDenseBlock(network, transition1 -> getOutput(0), 12, "features.denseblock2", 1e-5);
    auto transition2 = addTransition(network, *dense2 -> getOutput(0), 256, "features.transition2", 1e-5);

    auto dense3 = addDenseBlock(network, transition2 -> getOutput(0), 24, "features.denseblock3", 1e-5);
    auto transition3 = addTransition(network, *dense3 -> getOutput(0), 512, "features.transition3", 1e-5);

    auto dense4 = addDenseBlock(network, transition3 -> getOutput(0), 16, "features.denseblock4", 1e-5);

    auto bn5 = addBatchNorm2d(network, *dense4 -> getOutput(0), "features.norm5", 1e-5);
    auto relu5 = network -> addActivation(*bn5 -> getOutput(0), ActivationType::kRELU);

    // adaptive average pool => pytorch (F.adaptive_avg_pool2d(input, (1, 1)))
    auto pool5 = network -> addPoolingNd(*relu5 -> getOutput(0), PoolingType::kAVERAGE, DimsHW{7,7});

    auto fc1 = network -> addFullyConnected(*pool5 -> getOutput(0), 1000, weightMap["classifier.weight"], weightMap["classifier.bias"]);
    assert(fc1);

    // set ouput blob name
    fc1 -> getOutput(0) -> setName(mParams.outputTensorName);

    // mark the output
    network -> markOutput(*fc1 -> getOutput(0));

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
 * Loads weights from Weights file.
 * Tensorrt weights file have a simple space delimited format:
 * [type] [size] <data x size in hex>
 * 
 * Returns weightMap 
**/
std::map<std::string, Weights> DenseNet121::loadWeights(const std::string& weightsFile)
{
    std::cout << "Loading Weights ..." << std::endl;
    std::map<std::string, Weights> weightMap;

    // open weight file
    std::ifstream file(weightsFile);
    assert(file.is_open() && "Unable to open file.");

    // Read number of weight blobs
    int32_t count;
    file >> count;
    assert(count > 0 && "Invalid weights file.");

    // Loop through and create weight map
    while(count--)
    {
        // Initialize weight with Datatype and nullptr
        Weights wt{DataType::kFLOAT, nullptr, 0};

        // Read name and size(decimal format) of blob
        uint32_t size;
        std::string name;
        file >> name >> std::dec >> size;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for(uint32_t i=0; i<size; i++)
        {
            file >> std::hex >> val[i];
        }

        // assign weights and size to WeightMap
        wt.values = val;
        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}


/**
 * Performs inference on the given input and 
 * writes the output from device to host memory.
**/
void DenseNet121::doInference(float* input, float* output, int batchSize)
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
bool DenseNet121::deserialize(){
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
 * Initializes DenseNet class params in the 
 * DenseNetTrtParams structure.
**/
DenseNet121Params initializeParams()
{
    DenseNet121Params params;

    params.batchSize = 1;
    params.fp16 = false;

    params.inputH = 224;
    params.inputW = 224;
    params.outputSize = 1000;

    // change weights file name here
    params.weightsFile = "../densenet121.wts";

    // change engine file name here
    params.trtEngineFile = "densenet121.engine";
    return params;
}


int main(int argc, char** argv){
    if (argc != 2)
    {
        std::cerr << "Invalid args. please check." << std::endl;
        std::cerr <<"./densenet -r  // create engine" << std::endl;
        return 0;
    }

    DenseNet121Params params = initializeParams();
    DenseNet121 densenet121(params);

    // check if engine exists already
    std::ifstream f(params.trtEngineFile, std::ios::binary);
    
    // if engine does not exists build, serialize and save
    if(!f.good())
    {
        std::cout << "Building network ..." << std::endl;
        f.close();
        densenet121.build();
    }
    else
    {
        // deserialize
        std::cout << "engine already exists ..." << std::endl;
        // densenet121.deserialize();
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
        densenet121.doInference(data, prob, 1);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }

    // cleanup
    bool cleaned = densenet121.cleanUp();
    
    std::cout << "\nOutput:\n\n";
    for (unsigned int i = 0; i < params.outputSize; i++)
    {
        std::cout << prob[i] << ", ";
        if (i % 10 == 0) std::cout << i / 10 << std::endl;
    }
    std::cout << std::endl;

    return 0;
}