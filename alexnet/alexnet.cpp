#include <string>
#include <vector>

#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include "common.h"


/*
 The Alextrt params required to create tensorrt engine.
*/
struct AlexnetTrtParams{
    int32_t batchSize{1};              // Number of inputs in a batch
    int32_t dlaCore{-1};               // Specify the DLA core to run network on.
    bool int8{false};                  // Allow runnning the network in Int8 mode.
    bool fp16{false};                  // Allow running the network in FP16 mode.
    std::vector<std::string> dataDirs; // Directory paths where sample data files are stored
    std::vector<std::string> inputTensorNames;
    std::vector<std::string> outputTensorNames;

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
    
    std::map<std::string, nvinfer1::Weights> weightMap; // The weight value map.

    std::vector<nvinfer1::IHostMemory> weightsMem;  // Host weights memory holder.

    nvinfer1::ICudaEngine* mEngine;  // The tensorrt engine used to run the network.

    // Uses the API to create the network.
    bool createEngine(nvinfer1::IBuilder* builder, 
        nvinfer::INetworkDefinition* network, nvinfer1::IBuilderConfig* config);
    
    // Serializes the network engine
    void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream);

    // Loads weights from weights file
    std::map<std::string, nvinfer1::Weights> loadWeights(const std::string& file);
}



int main(){

}
