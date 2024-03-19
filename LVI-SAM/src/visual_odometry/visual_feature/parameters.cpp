#include "parameters.h"

std::string IMAGE_TOPIC;
std::string IMU_TOPIC;
std::string POINT_CLOUD_TOPIC;
std::string PROJECT_NAME;

std::vector<std::string> CAM_NAMES;
std::string FISHEYE_MASK;
int MAX_CNT;
int MIN_DIST;
int WINDOW_SIZE;
int FREQ;
double F_THRESHOLD;
int SHOW_TRACK;
int STEREO_TRACK;
int EQUALIZE;
int ROW;
int COL;
int FOCAL_LENGTH;
int FISHEYE;
bool PUB_THIS_FRAME;

double L_C_TX;
double L_C_TY;
double L_C_TZ;
double L_C_RX;
double L_C_RY;
double L_C_RZ;

int USE_LIDAR;
int LIDAR_SKIP;


//superpoint_config_inset
int max_keypoints;
double keypoints_threshold;
int remove_borders;
int dla_core;

std::vector<std::string> input_tensor_names;
std::vector<std::string> output_tensor_names;

std::string onnx_file;
std::string engine_file;



void readParameters(ros::NodeHandle &n)
{
    std::string config_file;
    n.getParam("vins_config_file", config_file);
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }


    //用多级的方式打开
    cv::FileNode file_node;

    // project name
    fsSettings["project_name"] >> PROJECT_NAME;
    std::string pkg_path = ros::package::getPath(PROJECT_NAME);

    // sensor topics
    fsSettings["image_topic"]       >> IMAGE_TOPIC;
    fsSettings["imu_topic"]         >> IMU_TOPIC;
    fsSettings["point_cloud_topic"] >> POINT_CLOUD_TOPIC;

    // lidar configurations
    fsSettings["use_lidar"] >> USE_LIDAR;
    fsSettings["lidar_skip"] >> LIDAR_SKIP;

    // feature and image settings
    MAX_CNT = fsSettings["max_cnt"];
    MIN_DIST = fsSettings["min_dist"];
    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    FREQ = fsSettings["freq"];
    F_THRESHOLD = fsSettings["F_threshold"];
    SHOW_TRACK = fsSettings["show_track"];
    EQUALIZE = fsSettings["equalize"];

    L_C_TX = fsSettings["lidar_to_cam_tx"];
    L_C_TY = fsSettings["lidar_to_cam_ty"];
    L_C_TZ = fsSettings["lidar_to_cam_tz"];
    L_C_RX = fsSettings["lidar_to_cam_rx"];
    L_C_RY = fsSettings["lidar_to_cam_ry"];
    L_C_RZ = fsSettings["lidar_to_cam_rz"];

    // fisheye mask
    FISHEYE = fsSettings["fisheye"];
    if (FISHEYE == 1)
    {
        std::string mask_name;
        fsSettings["fisheye_mask"] >> mask_name;
        FISHEYE_MASK = pkg_path + mask_name;
    }


    /*****************读取superpoint特征点参数*********************************/
    //int类型
    file_node = fsSettings["superpoint"]["max_keypoints"];
    if(!file_node.isNone() && !file_node.empty()){
        file_node >> max_keypoints;
    }
    else{
        return;
    }

    file_node = fsSettings["superpoint"]["keypoint_threshold"];
    file_node >> keypoints_threshold;

    file_node = fsSettings["superpoint"]["remove_borders"];
    file_node >> remove_borders;

    file_node = fsSettings["superpoint"]["dla_core"];
    file_node >> dla_core;

    //string类型
    file_node = fsSettings["superpoint"]["onnx_file"];
    file_node >> onnx_file;

    file_node = fsSettings["superpoint"]["engine_file"];
    file_node >> engine_file;


    //vector类型
    cv::FileNode input = fsSettings["superpoint"]["input_tensor_names"];
    size_t input_nums = input.size();
    for(size_t i = 0; i < input_nums ; i++){
        input_tensor_names.push_back(input[i].operator std::string());
    }

    cv::FileNode output = fsSettings["superpoint"]["output_tensor_names"];
    size_t output_nums = output.size();
    for(size_t i = 0; i < output_nums ; i++){
        output_tensor_names.push_back(output[i].operator std::string());
    }
    /*************************************************************************/
    
    // camera config
    CAM_NAMES.push_back(config_file);

    WINDOW_SIZE = 20;
    STEREO_TRACK = false;
    FOCAL_LENGTH = 460;
    PUB_THIS_FRAME = false;

    if (FREQ == 0)
        FREQ = 100;

    fsSettings.release();
    usleep(100);
}

float pointDistance(PointType p)
{
    return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
}

float pointDistance(PointType p1, PointType p2)
{
    return sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y) + (p1.z-p2.z)*(p1.z-p2.z));
}

void publishCloud(ros::Publisher *thisPub, pcl::PointCloud<PointType>::Ptr thisCloud, ros::Time thisStamp, std::string thisFrame)
{
    if (thisPub->getNumSubscribers() == 0)
        return;
    sensor_msgs::PointCloud2 tempCloud;
    pcl::toROSMsg(*thisCloud, tempCloud);
    tempCloud.header.stamp = thisStamp;
    tempCloud.header.frame_id = thisFrame;
    thisPub->publish(tempCloud); 
}