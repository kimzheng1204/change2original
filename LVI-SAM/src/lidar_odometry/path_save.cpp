#include"ros/ros.h"
#include "utility.h"
#include "lvi_sam/cloud_info.h"


void path_save(nav_msgs::Odometry odomAftMapped){
    //保存轨迹，path_save是文件目录，txt提前建好，//home/fz/LoFTR-LIV/localization-baseline/Result-Record/RECORD/gate_01.txt
    std::ofstream pose1("/workspace/Loftr/LoFTR/output/street_real.txt",std::ios::app);
    pose1.setf(std::ios::scientific,std::ios::floatfield);
    pose1.precision(9);

    //static double timeStart = odomAftMapped.header.stamp.toSec();
    //auto T1 = ros::Time().fromSec(timeStart);
    //这个写入的顺序要咋弄啊
    pose1 << odomAftMapped.header.stamp << " "
    << odomAftMapped.pose.pose.position.x << " "
    << odomAftMapped.pose.pose.position.y << " "
    << odomAftMapped.pose.pose.position.z << " "
    << odomAftMapped.pose.pose.orientation.x << " "
    << odomAftMapped.pose.pose.orientation.y << " "
    << odomAftMapped.pose.pose.orientation.z << " "
    << odomAftMapped.pose.pose.orientation.w << std::endl;
    pose1.close();

}

int main(int argc, char **argv){
    ros::init(argc,argv,"path_save");
    ros::NodeHandle nh;
    ros::Subscriber save_path = nh.subscribe<nav_msgs::Odometry>("/lvi_sam/lidar/mapping/odometry",100,path_save);

    ros::spin();
}