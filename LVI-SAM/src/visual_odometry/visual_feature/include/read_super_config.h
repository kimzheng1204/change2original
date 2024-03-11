#ifndef READ_CONFIGS_H_
#define READ_CONFIGS_H_

#include <iostream>
#include <yaml-cpp/yaml.h>
#include "utils.h"



struct  SuperPointConfig
{
    int max_keypoints;
    double keypoint_threshold;
    int remove_borders;
    int dla_core;

    std::vector<std::string> input_tensor_names;
    std::vector<std::string> output_tensor_names;

    std::string onnx_file;
    std::string engine_file; //指定训练好的加速模型的位置
};




struct Configs{
    

    std::string model_dir;
    
    SuperPointConfig superpoint_config;

    Configs(const std::string& config_file, const std::string& model_dir){

        std::cout << "config_file = " << config_file << std::endl;

        YAML::Node file_node = YAML::LoadFile(config_file); 

        YAML::Node superpoint_node = file_node["superpoint"];

        superpoint_config.max_keypoints = superpoint_node["max_keypoints"].as<int>();
        superpoint_config.keypoint_threshold = superpoint_node["keypoint_threshold"].as<double>();
        superpoint_config.remove_borders = superpoint_node["remove_borders"].as<int>();
        superpoint_config.dla_core = superpoint_node["dla_core"].as<int>();

        YAML::Node superpoint_input_tensor_names_node = superpoint_node["input_tensor_names"];
        size_t superpoint_num_input_tensor_names = superpoint_input_tensor_names_node.size();
        for(size_t i = 0; i < superpoint_num_input_tensor_names; i++){
            superpoint_config.input_tensor_names.push_back(superpoint_input_tensor_names_node[i].as<std::string>());
            }

        
        YAML::Node superpoint_output_tensor_names_node = superpoint_node["output_tensor_names"];
        size_t superpoint_num_output_tensor_names = superpoint_output_tensor_names_node.size();
        for(size_t i = 0; i < superpoint_num_output_tensor_names; i++){
            superpoint_config.output_tensor_names.push_back(superpoint_output_tensor_names_node[i].as<std::string>());
            }

        std::string superpoint_onnx_file = superpoint_node["onnx_file"].as<std::string>();
        std::string superpoint_engine_file = superpoint_node["engine_file"].as<std::string>();

        superpoint_config.onnx_file = ConcatenateFolderAndFileName(model_dir,superpoint_onnx_file);
        superpoint_config.engine_file = ConcatenateFolderAndFileName(model_dir,superpoint_engine_file);


    }
};




#endif  // READ_CONFIGS_H_
