//
// Created by haoyuefan on 2021/9/22.
//
#include "super_point.h"
#include <utility>
#include <unordered_map>
#include <opencv2/opencv.hpp>


//来自Thirdparty的部分
using namespace tensorrt_common;
using namespace tensorrt_log;
using namespace tensorrt_buffer;



SuperPoint::SuperPoint(const SuperPointConfig &super_point_config)
        : super_point_config_(super_point_config), engine_(nullptr) {
    setReportableSeverity(Logger::Severity::kINTERNAL_ERROR);
}



/**
 * @brief 构建一个基于TensorRT的SuperPoint模型
*/
bool SuperPoint::build() {

    //如果已经成功反序列化了.engine文件，则返回true
    //就是首先尝试反序列化.engine文件，如果失败则从ONNX文件构建TensorRT模型
    //然后设置TensorRT的优化配置，构建并保存TensorRT引擎，并最终验证网络的输入和输出维度
    if(deserialize_engine()){
        return true;
    }
    
    //创建TensorRT的 builder 实例
    auto builder = TensorRTUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder) {
        return false;
    }

    //创建TensorRT网络
    const auto explicit_batch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = TensorRTUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicit_batch));
    if (!network) {
        return false;
    }

    //创建TensorRT的BuilderConfig
    auto config = TensorRTUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        return false;
    }

    //创建ONNX解析器并解析网络结构
    auto parser = TensorRTUniquePtr<nvonnxparser::IParser>(
            nvonnxparser::createParser(*network, gLogger.getTRTLogger()));
    if (!parser) {
        return false;
    }
    
    //创建优化设置
    auto profile = builder->createOptimizationProfile();
    if (!profile) {
        return false;
    }

    //设置输入张量的优化设置
    profile->setDimensions(super_point_config_.input_tensor_names[0].c_str(),
                           OptProfileSelector::kMIN, Dims4(1, 1, 100, 100));
    profile->setDimensions(super_point_config_.input_tensor_names[0].c_str(),
                           OptProfileSelector::kOPT, Dims4(1, 1, 500, 500));
    profile->setDimensions(super_point_config_.input_tensor_names[0].c_str(),
                           OptProfileSelector::kMAX, Dims4(1, 1, 1500, 1500));
    config->addOptimizationProfile(profile);
    
    //构建网络
    auto constructed = construct_network(builder, network, config, parser);
    if (!constructed) {
        return false;
    }

    //创建CUDA流
    auto profile_stream = makeCudaStream();
    if (!profile_stream) {
        return false;
    }


    //设置优化配置流
    config->setProfileStream(*profile_stream);
    
    //构建序列化网络
    TensorRTUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan) {
        return false;
    }

    //创建TensorRT运行时
    TensorRTUniquePtr<IRuntime> runtime{createInferRuntime(gLogger.getTRTLogger())};
    if (!runtime) {
        return false;
    }

    //反序列化engine并保存
    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()));
    if (!engine_) {
        return false;
    }
    save_engine();

    //断言网络的输入和输出维度
    ASSERT(network->getNbInputs() == 1);
    input_dims_ = network->getInput(0)->getDimensions();
    ASSERT(input_dims_.nbDims == 4);
    ASSERT(network->getNbOutputs() == 2);
    semi_dims_ = network->getOutput(0)->getDimensions();
    ASSERT(semi_dims_.nbDims == 3);
    desc_dims_ = network->getOutput(1)->getDimensions();
    ASSERT(desc_dims_.nbDims == 4);
    return true;
}



/**
 * @brief ONNX --> engine
*/
bool SuperPoint::construct_network(TensorRTUniquePtr<nvinfer1::IBuilder> &builder,
                                   TensorRTUniquePtr<nvinfer1::INetworkDefinition> &network,
                                   TensorRTUniquePtr<nvinfer1::IBuilderConfig> &config,
                                   TensorRTUniquePtr<nvonnxparser::IParser> &parser) const {
    
    //解析ONNX文件并构建网络
    auto parsed = parser->parseFromFile(super_point_config_.onnx_file.c_str(),
                                        static_cast<int>(gLogger.getReportableSeverity()));
    if (!parsed) {
        return false;
    }

    //设置最大工作空间大小和优化设置
    config->setMaxWorkspaceSize(512_MiB);
    config->setFlag(BuilderFlag::kFP16);
    enableDLA(builder.get(), config.get(), super_point_config_.dla_core);
    return true;
}



/**
 * @brief 执行superpoint模型的推理
*/
bool SuperPoint::infer(const cv::Mat &image, Eigen::Matrix<double, 259, Eigen::Dynamic> &features) {
    
    //如果上下文为空，创建执行上下文
    if (!context_) {
        context_ = TensorRTUniquePtr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
        if (!context_) {
            return false;
        }
    }
    
    
    //断言绑定的数量是否为3
    assert(engine_->getNbBindings() == 3);

    
    //获取输入索引并设置输入维度
    const int input_index = engine_->getBindingIndex(super_point_config_.input_tensor_names[0].c_str());
    context_->setBindingDimensions(input_index, Dims4(1, 1, image.rows, image.cols));

    //创建缓冲管理器并处理输入
    BufferManager buffers(engine_, 0, context_.get());
    
    ASSERT(super_point_config_.input_tensor_names.size() == 1);
    if (!process_input(buffers, image)) {
        return false;
    }
    buffers.copyInputToDevice();

    
    //执行推理
    bool status = context_->executeV2(buffers.getDeviceBindings().data());
    if (!status) {
        return false;
    }


    //处理输出并拷贝到主机内存
    buffers.copyOutputToHost();
    if (!process_output(buffers, features)) {
        return false;
    }
    return true;
}


/**
 * @brief 将输入图像归一化并填充到TensorRT的输入缓冲区
*/
bool SuperPoint::process_input(const BufferManager &buffers, const cv::Mat &image) {
    
    //设置输入维度
    input_dims_.d[2] = image.rows;
    input_dims_.d[3] = image.cols;
    
    //设置半特征图的维度
    semi_dims_.d[1] = image.rows;
    semi_dims_.d[2] = image.cols;
    
    //设置描述子的维度
    desc_dims_.d[1] = 256;
    desc_dims_.d[2] = image.rows / 8;
    desc_dims_.d[3] = image.cols / 8;
    
    //获取输入数据缓冲区
    auto *host_data_buffer = static_cast<float *>(buffers.getHostBuffer(super_point_config_.input_tensor_names[0]));
    
    //将图像数据归一化并填充到输入缓冲区中
    for (int row = 0; row < image.rows; ++row) {
        for (int col = 0; col < image.cols; ++col) {
            host_data_buffer[row * image.cols + col] = float(image.at<unsigned char>(row, col)) / 255.0;
        }
    }
    return true;
}



/**
 * @brief 找到分数高于阈值的关键点索引
*/
void SuperPoint::find_high_score_index(std::vector<float> &scores, std::vector<std::vector<int>> &keypoints,
                                       int h, int w, double threshold) {
    
    //在分数中找到高于阈值的索引
    std::vector<float> new_scores;
    for (int i = 0; i < scores.size(); ++i) {
        if (scores[i] > threshold) {
            
            //计算索引对应的位置
            std::vector<int> location = {int(i / w), i % w};
            
            //将位置添加到关键点列表中
            keypoints.emplace_back(location);
            
            //符合条件的分数添加到新的分数数组中
            new_scores.push_back(scores[i]);
        }
    }
    //更新scores数组为新的分数数组
    scores.swap(new_scores);
}



/**
 * @brief 移除位于边界内的关键点
*/
void SuperPoint::remove_borders(std::vector<std::vector<int>> &keypoints, std::vector<float> &scores, int border,
                                int height,
                                int width) {
    
    std::vector<std::vector<int>> keypoints_selected;
    std::vector<float> scores_selected;
    for (int i = 0; i < keypoints.size(); ++i) {
        bool flag_h = (keypoints[i][0] >= border) && (keypoints[i][0] < (height - border));
        bool flag_w = (keypoints[i][1] >= border) && (keypoints[i][1] < (width - border));
        if (flag_h && flag_w) {
            
            //将位于边界内的关键点和对应的分数保存
            keypoints_selected.push_back(std::vector<int>{keypoints[i][1], keypoints[i][0]});
            scores_selected.push_back(scores[i]);
        }
    }

    //更新关键点和分数数组
    keypoints.swap(keypoints_selected);
    scores.swap(scores_selected);
}



/**
 * @brief 对关键点的分数进行排序，以便后续按照得分高低选择关键点
 * @param[in] data 浮点数向量
 * @return  按照该向量值从大到小排序的索引向量
*/
std::vector<size_t> SuperPoint::sort_indexes(std::vector<float> &data) {
    
    //创建大小为数据大小的索引向量
    std::vector<size_t> indexes(data.size());
    
    //用iota函数填充索引向量，以便之后对其排序
    iota(indexes.begin(), indexes.end(), 0);
    
    //对索引向量进行排序，排序规则是根据数值的从小到达进行排序
    sort(indexes.begin(), indexes.end(), [&data](size_t i1, size_t i2) { return data[i1] > data[i2]; });
    
    //返回排序后的索引向量
    return indexes;
}



/**
 * @brief 选取分数最高的k个关键点
 * @param[in]  keypoints 的坐标 & 分数 & 整数k
 * @param[out] top_k 分数最高的前k个点，以及对应分数；并更新关键点和分数向量
*/
void SuperPoint::top_k_keypoints(std::vector<std::vector<int>> &keypoints, std::vector<float> &scores, int k) {
    
    //如果要求的最高分的关键点数小于关键点的数量，并且k不等于-1（表示不限制数量）
    if (k < keypoints.size() && k != -1) {
        
        //创建用于存储前k个关键点的新向量
        std::vector<std::vector<int>> keypoints_top_k;
        
        //创建用于存储前k个关键点得分的新向量
        std::vector<float> scores_top_k;
        
        //获取按得分从高到底排序的索引向量
        std::vector<size_t> indexes = sort_indexes(scores);
        
        //将前k个关键点和对应得分存储到新向量中
        for (int i = 0; i < k; ++i) {
            keypoints_top_k.push_back(keypoints[indexes[i]]);
            scores_top_k.push_back(scores[indexes[i]]);
        }

        //用新的关键点和得分向量替换原始的向量
        keypoints.swap(keypoints_top_k);
        scores.swap(scores_top_k);
    }
}



/**
 * @brief 将原始的关键点坐标转换成相对于图像尺寸和尺度的归一化坐标
 * @param[in] keypoints,h,w,s 原始关键点坐标 & 归一化后关键点坐标 & 图像的宽、高、尺度
 * @details 这个归一化的公式咋决定的？
*/
void
normalize_keypoints(const std::vector<std::vector<int>> &keypoints, std::vector<std::vector<double>> &keypoints_norm,
                    int h, int w, int s) {
    
    //对输入的关键点进行归一化处理，并将结果存储到指定的向量中
    for (auto &keypoint : keypoints) {
        std::vector<double> kp = {keypoint[0] - s / 2 + 0.5, keypoint[1] - s / 2 + 0.5};
        kp[0] = kp[0] / (w * s - s / 2 - 0.5);
        kp[1] = kp[1] / (h * s - s / 2 - 0.5);
        kp[0] = kp[0] * 2 - 1;
        kp[1] = kp[1] * 2 - 1;
        
        //将归一化后的关键点坐标存储到结果向量中
        keypoints_norm.push_back(kp);
    }
}


/**
 * @brief 确保采样位置在图像范围内
 * @details 对给定值进行截断，确保其在指定的范围内
*/
int clip(int val, int max) {
    if (val < 0) return 0;
    return std::min(val, max - 1);
}



/**
 * @brief 根据归一化的关键点坐标从输入数据中采样生成描述符，以供后续处理使用
 * @param[in] input,grid,output 输入网络、网格采样位置向量、输出向量
 * @details 首先根据归一化的插值点坐标计算出插值点周围四个像素的坐标，然后计算出这四个像素的权重，并利用
 *          双现行插值方法对输入数据进行插值，生成相应的输出数据
*/
void grid_sample(const float *input, std::vector<std::vector<double>> &grid,
                 std::vector<std::vector<double>> &output, int dim, int h, int w) {
    // descriptors 1x256x60x106
    // keypoints 1x1xnumberx2
    // out 1x256x1xnumber
    
    
    for (auto &g : grid) {
        
        //计算在输入上的采样位置
        double ix = ((g[0] + 1) / 2) * (w - 1);
        double iy = ((g[1] + 1) / 2) * (h - 1);

        //计算四个相邻像素的坐标
        int ix_nw = clip(std::floor(ix), w);
        int iy_nw = clip(std::floor(iy), h);

        int ix_ne = clip(ix_nw + 1, w);
        int iy_ne = clip(iy_nw, h);

        int ix_sw = clip(ix_nw, w);
        int iy_sw = clip(iy_nw + 1, h);

        int ix_se = clip(ix_nw + 1, w);
        int iy_se = clip(iy_nw + 1, h);

        //计算四个相邻像素的权重
        double nw = (ix_se - ix) * (iy_se - iy);
        double ne = (ix - ix_sw) * (iy_sw - iy);
        double sw = (ix_ne - ix) * (iy - iy_ne);
        double se = (ix - ix_nw) * (iy - iy_nw);

        //对每个维度进行网格采样
        std::vector<double> descriptor;
        for (int i = 0; i < dim; ++i) {
            // 256x60x106 whd
            // x * Height * Depth + y * Depth + z
            
            //使用双线性插值计算采样值
            float nw_val = input[i * h * w + iy_nw * w + ix_nw];
            float ne_val = input[i * h * w + iy_ne * w + ix_ne];
            float sw_val = input[i * h * w + iy_sw * w + ix_sw];
            float se_val = input[i * h * w + iy_se * w + ix_se];
            descriptor.push_back(nw_val * nw + ne_val * ne + sw_val * sw + se_val * se);
        }
        output.push_back(descriptor);
    }
}


/**
 * @brief 计算输入迭代器范围内的向量的范数，使用内积计算，然后返回其平方根
*/
template<typename Iter_T>
double vector_normalize(Iter_T first, Iter_T last) {
    return sqrt(inner_product(first, last, first, 0.0));
}


/**
 * @brief 对输入的目标描述符向量进行归一化处理
 * @details 首先计算每个描述符向量的逆范数，然后将每个元素除以逆范数以完成归一化
*/
void normalize_descriptors(std::vector<std::vector<double>> &dest_descriptors) {
    for (auto &descriptor : dest_descriptors) {
        
        //计算描述符向量的逆范数
        double norm_inv = 1.0 / vector_normalize(descriptor.begin(), descriptor.end());
        
        //将描述符向量归一化
        std::transform(descriptor.begin(), descriptor.end(), descriptor.begin(),
                       std::bind1st(std::multiplies<double>(), norm_inv));
    }
}


/**
 * @brief 对给定的关键点坐标进行归一化处理，然后对描述符进行网格采样，并对采样得到的描述符向量进行归一化处理
*/
void SuperPoint::sample_descriptors(std::vector<std::vector<int>> &keypoints, float *descriptors,
                                    std::vector<std::vector<double>> &dest_descriptors, int dim, int h, int w, int s) {
    
    //对关键点坐标进行归一化
    std::vector<std::vector<double>> keypoints_norm;
    normalize_keypoints(keypoints, keypoints_norm, h, w, s);
    
    //对描述符进行网格采样
    grid_sample(descriptors, keypoints_norm, dest_descriptors, dim, h, w);
    
    //对采样得到的描述符向量进行归一化处理
    normalize_descriptors(dest_descriptors);
}



/**
 * @brief 处理模型的输出，并生成特征矩阵
 * @details 首先从缓冲区获取模型输出的得分和描述符数据；
 *          然后根据得分找到高分关键点，移除边界上的关键点，并保留前k个最高分的关键点；
 *          接着，调整特征矩阵的大小，并对关键点进行描述符采样；
 *          最后，将得分、关键点坐标和描述符放入特征矩阵中，完成特征矩阵的生成
*/
bool SuperPoint::process_output(const BufferManager &buffers, Eigen::Matrix<double, 259, Eigen::Dynamic> &features) {
    
    //清空之前存储的关键点和描述符
    keypoints_.clear();
    descriptors_.clear();
    
    //获取模型输出的得分和描述符数据
    auto *output_score = static_cast<float *>(buffers.getHostBuffer(super_point_config_.output_tensor_names[0]));
    auto *output_desc = static_cast<float *>(buffers.getHostBuffer(super_point_config_.output_tensor_names[1]));
    
    //获取半尺寸特征图的尺寸
    int semi_feature_map_h = semi_dims_.d[1];
    int semi_feature_map_w = semi_dims_.d[2];
    
    //将分数数据转换为向量
    std::vector<float> scores_vec(output_score, output_score + semi_feature_map_h * semi_feature_map_w);
    
    //根据得分找到高分关键点
    find_high_score_index(scores_vec, keypoints_, semi_feature_map_h, semi_feature_map_w,
                          super_point_config_.keypoint_threshold);
    
    //移除边界上的关键点
    remove_borders(keypoints_, scores_vec, super_point_config_.remove_borders, semi_feature_map_h, semi_feature_map_w);
    
    //保留前k个最高分的关键点
    top_k_keypoints(keypoints_, scores_vec, super_point_config_.max_keypoints);
    // std::cout << "super point number is " << std::to_string(scores_vec.size()) << std::endl;
    
    //调整特征矩阵大小
    features.resize(259, scores_vec.size());
    
    //获取描述符的特征维度和特征图尺寸
    int desc_feature_dim = desc_dims_.d[1];
    int desc_feature_map_h = desc_dims_.d[2];
    int desc_feature_map_w = desc_dims_.d[3];
    
    //对关键点进行描述符采样
    sample_descriptors(keypoints_, output_desc, descriptors_, desc_feature_dim, desc_feature_map_h, desc_feature_map_w);
    
    
    //将得分和关键点坐标放入特征矩阵的前三行
    for (int i = 0; i < scores_vec.size(); i++){
        features(0, i) = scores_vec[i];
    }

    for (int i = 1; i < 3; ++i) {
        for (int j = 0; j < keypoints_.size(); ++j) {
            features(i, j) = keypoints_[j][i-1];
        }
    }

    //将描述符放入特征矩阵的剩余行
    for (int m = 3; m < 259; ++m) {
        for (int n = 0; n < descriptors_.size(); ++n) {
            features(m, n) = descriptors_[n][m-3];
        }
    }
    return true;
}




/**
 * @brief 在输入的图像上绘制关键点，并将带有关键点的图像保存为JPEG文件
*/
void SuperPoint::visualization(const std::string &image_name, const cv::Mat &image) {
    
    //创建一个用于显示的图像副本
    cv::Mat image_display;
    
    //如果输入图像是单通道的灰度图像，则将其转换为三通道的BGR图像
    if(image.channels() == 1)
        cv::cvtColor(image, image_display, cv::COLOR_GRAY2BGR);
    else
        image_display = image.clone();

    //在图像上绘制关键点
    for (auto &keypoint : keypoints_) {
        cv::circle(image_display, cv::Point(int(keypoint[0]), int(keypoint[1])), 1, cv::Scalar(255, 0, 0), -1, 16);
    }

    //将带有关键点的图像保存为JPEG文件
    cv::imwrite(image_name + ".jpg", image_display);
}



/**
 * @brief 将序列化后的引擎保存到文件中
*/
void SuperPoint::save_engine() {
    
    //如果未指定引擎文件名，则直接返回
    if (super_point_config_.engine_file.empty()) return;
    
    //如果存在已序列化的引擎，则将其保存到文件中
    if (engine_ != nullptr) {
        nvinfer1::IHostMemory *data = engine_->serialize();
        std::ofstream file(super_point_config_.engine_file, std::ios::binary);
        
        //如果无法打开文件，则直接返回
        if (!file) return;

        //将序列化后的引擎写入文件
        file.write(reinterpret_cast<const char *>(data->data()), data->size());
    }
}



/**
 * @brief 反序列化引擎模型
 * @details 首先，打开引擎文件并读取其内容；
 *          然后分配内存来存储模型数据，并使用次数据创建运行时的环境；
 *          接着使用运行时环境中的方法反序列化模型数据
 * @return 反序列化是否成功的布尔值
*/
bool SuperPoint::deserialize_engine() {
    
    //打开引擎文件
    std::ifstream file(super_point_config_.engine_file.c_str(), std::ios::binary);
    
    //如果文件成功打开
    if (file.is_open()) {
        
        //获取文件大小
        file.seekg(0, std::ifstream::end);
        size_t size = file.tellg();
        file.seekg(0, std::ifstream::beg);
        
        //为引擎模型分配内存并读取文件内容
        char *model_stream = new char[size];
        file.read(model_stream, size);
        file.close();
        
        //创建运行时环境
        IRuntime *runtime = createInferRuntime(gLogger);
        
        //如果无法创建运行时环境，则释放内存并返回失败
        if (runtime == nullptr) {
            delete[] model_stream;
            return false;
        }

        //反序列化引擎模型
        engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(model_stream, size));
        delete[] model_stream;
        
        //如果引擎模型为空，则返回失败，否则返回成功
        if (engine_ == nullptr) return false;
        return true;
    }
    return false;
}

