#include "utility.h"
#include "lvi_sam/cloud_info.h"

// Velodyne
struct PointXYZIRT
{
    PCL_ADD_POINT4D //分别有float类型的 xyz还有一个对齐变量
    PCL_ADD_INTENSITY;//float类型的密集度
    uint16_t ring;//总的线圈数
    float time;//时间戳
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW //保证在内存中是对齐的状态
} EIGEN_ALIGN16; //定义结构体的别名

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRT,  
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint16_t, ring, ring) (float, time, time)
)

// Ouster
// struct PointXYZIRT {
//     PCL_ADD_POINT4D;
//     float intensity;
//     uint32_t t;
//     uint16_t reflectivity;
//     uint8_t ring;
//     uint16_t noise;
//     uint32_t range;
//     EIGEN_MAKE_ALIGNED_OPERATOR_NEW
// }EIGEN_ALIGN16;

// POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRT,
//     (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
//     (uint32_t, t, t) (uint16_t, reflectivity, reflectivity)
//     (uint8_t, ring, ring) (uint16_t, noise, noise) (uint32_t, range, range)
// )

const int queueLength = 500;

class ImageProjection : public ParamServer
{
private:

    //为什么要给这俩加锁呢？
    std::mutex imuLock;//imu锁
    std::mutex odoLock;//里程计锁，这里的里程计是视觉里程计啊

    ros::Subscriber subLaserCloud;
    ros::Publisher  pubLaserCloud;
    
    ros::Publisher pubExtractedCloud;
    ros::Publisher pubLaserCloudInfo;

    ros::Subscriber subImu;
    std::deque<sensor_msgs::Imu> imuQueue;//imu队列

    ros::Subscriber subOdom;
    std::deque<nav_msgs::Odometry> odomQueue;//里程计队列

    std::deque<sensor_msgs::PointCloud2> cloudQueue;//点云队列
    sensor_msgs::PointCloud2 currentCloudMsg; //当前的点云信息
    
    double *imuTime = new double[queueLength];
    double *imuRotX = new double[queueLength];
    double *imuRotY = new double[queueLength];
    double *imuRotZ = new double[queueLength];

    int imuPointerCur; //点云对应的imu的索引
    bool firstPointFlag;
    Eigen::Affine3f transStartInverse; //初始位姿

    pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;
    
    //PointType是在utility.h里面重定义的 pcl::PointXYZI
    pcl::PointCloud<PointType>::Ptr   fullCloud;
    pcl::PointCloud<PointType>::Ptr   extractedCloud;

    int deskewFlag;
    cv::Mat rangeMat;

    bool odomDeskewFlag;
    float odomIncreX;
    float odomIncreY;
    float odomIncreZ;

    lvi_sam::cloud_info cloudInfo;
    double timeScanCur;
    double timeScanNext;
    std_msgs::Header cloudHeader;


public:
    ImageProjection():
    deskewFlag(0)//函数后面加冒号来赋值，是c++写法的特性
    {
        subImu        = nh.subscribe<sensor_msgs::Imu>        (imuTopic, 2000, &ImageProjection::imuHandler, this, ros::TransportHints().tcpNoDelay());
        subOdom       = nh.subscribe<nav_msgs::Odometry>      (PROJECT_NAME + "/vins/odometry/imu_propagate_ros", 2000, &ImageProjection::odometryHandler, this, ros::TransportHints().tcpNoDelay());
        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 5, &ImageProjection::cloudHandler, this, ros::TransportHints().tcpNoDelay());

        pubExtractedCloud = nh.advertise<sensor_msgs::PointCloud2> (PROJECT_NAME + "/lidar/deskew/cloud_deskewed", 5);
        pubLaserCloudInfo = nh.advertise<lvi_sam::cloud_info>      (PROJECT_NAME + "/lidar/deskew/cloud_info", 5);

        //怎么还有分配存储记忆和重置参数
        allocateMemory();
        resetParameters();

        pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    }

    void allocateMemory()
    {
        laserCloudIn.reset(new pcl::PointCloud<PointXYZIRT>());
        fullCloud.reset(new pcl::PointCloud<PointType>());
        extractedCloud.reset(new pcl::PointCloud<PointType>());

        //将所有点云数据安装行列按照从小到大的顺序存储在fullCloud中
        fullCloud->points.resize(N_SCAN*Horizon_SCAN);

        cloudInfo.startRingIndex.assign(N_SCAN, 0);
        cloudInfo.endRingIndex.assign(N_SCAN, 0);

        cloudInfo.pointColInd.assign(N_SCAN*Horizon_SCAN, 0);
        cloudInfo.pointRange.assign(N_SCAN*Horizon_SCAN, 0);

        resetParameters();
    }

    //对每个获取的lidar message进行参数重置
    void resetParameters()
    {
        laserCloudIn->clear();
        extractedCloud->clear();
        // reset range matrix for range image projection
        //雷达深度图 range image
        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));

        imuPointerCur = 0;
        firstPointFlag = true;
        odomDeskewFlag = false;

        for (int i = 0; i < queueLength; ++i)
        {
            imuTime[i] = 0;
            imuRotX[i] = 0;
            imuRotY[i] = 0;
            imuRotZ[i] = 0;
        }
    }

    ~ImageProjection(){}

    void imuHandler(const sensor_msgs::Imu::ConstPtr& imuMsg)
    {
        //imu原始数据转换到lidar系，数据类型还是不变
        sensor_msgs::Imu thisImu = imuConverter(*imuMsg);
        //将imu的三个轴的线加速度和角加速度的信息旋转到以LiDAR为中心的坐标系
        std::lock_guard<std::mutex> lock1(imuLock);//出作用域自动解锁
        imuQueue.push_back(thisImu);
    }

    void odometryHandler(const nav_msgs::Odometry::ConstPtr& odometryMsg)
    {
        std::lock_guard<std::mutex> lock2(odoLock);
        odomQueue.push_back(*odometryMsg);
    }

    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
    {
        //1 检查队列里面的点云数量是否满足要求 并做一些前置操作
        if (!cachePointCloud(laserCloudMsg))
            return;

        
        //2 对IMU和视觉里程计去畸变
        if (!deskewInfo())
            return;

        //3 获取雷达深度图
        projectPointCloud();

        //4 点云提取
        cloudExtraction();

        //5 发布点云
        publishClouds();

        //6 重置参数
        resetParameters();
    }

    bool cachePointCloud(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
    {
        // cache point cloud
        //缓存点云
        cloudQueue.push_back(*laserCloudMsg);

        //就是队列里的点云到达一定数目才执行这些操作
        if (cloudQueue.size() <= 2)
            return false;
        else
        {
            currentCloudMsg = cloudQueue.front();
            cloudQueue.pop_front(); //弹出队列头的点云，是因为pop了这个所以下一个就是下一帧的时间戳吗？

            cloudHeader = currentCloudMsg.header;
            timeScanCur = cloudHeader.stamp.toSec(); //toSec()是转成double类型，当前点云扫描的起始时间戳
            timeScanNext = cloudQueue.front().header.stamp.toSec();//下一帧点云的时间戳
        }

        // convert cloud
        pcl::fromROSMsg(currentCloudMsg, *laserCloudIn);

        // check dense flag
        if (laserCloudIn->is_dense == false)//表示点云里面没有去除无效点(NaN)
        {
            ROS_ERROR("Point cloud is not in dense format, please remove NaN points first!");
            ros::shutdown();
        }

        // check ring channel
        //检查点云是否包含ring通道
        //该部分主要用来计算rowIdn
        //就是检查点云的质量，希望它是合格的点云，包含ring通道信息
        static int ringFlag = 0;
        if (ringFlag == 0)
        {
            ringFlag = -1;
            for (int i = 0; i < (int)currentCloudMsg.fields.size(); ++i)
            {
                if (currentCloudMsg.fields[i].name == "ring")
                {
                    ringFlag = 1;
                    break;
                }
            }
            if (ringFlag == -1)
            {
                ROS_ERROR("Point cloud ring channel not available, please configure your point cloud data!");
                ros::shutdown();
            }
        }     

        // check point time
        //也是确保点云信息的质量，希望它是包含时间戳的合格的点云
        //yaml文件中 timeField:"time" #point timestamp field, Velodyne - "time", Ouster - "t"
        if (deskewFlag == 0)
        {
            deskewFlag = -1;
            for (int i = 0; i < (int)currentCloudMsg.fields.size(); ++i)
            {
                //表示当前信息具有时间戳
                if (currentCloudMsg.fields[i].name == timeField)
                {
                    deskewFlag = 1;
                    break;
                }
            }
            if (deskewFlag == -1)
                ROS_WARN("Point cloud timestamp not available, deskew function disabled, system will drift significantly!");
        }

        return true;
    }

    bool deskewInfo()
    {
        //怎么又要互斥锁
        std::lock_guard<std::mutex> lock1(imuLock);
        std::lock_guard<std::mutex> lock2(odoLock);

        // make sure IMU data available for the scan
        //就是imu队列里有数据，imu的时间要吞掉lidar的
        if (imuQueue.empty() || imuQueue.front().header.stamp.toSec() > timeScanCur || imuQueue.back().header.stamp.toSec() < timeScanNext)
        {
            ROS_DEBUG("Waiting for IMU data ...");
            return false;
        }

        //分别是imu和VO去畸变，都是void，没有返回值的函数，在最开始把flag置为false，最后改成true
        imuDeskewInfo();

        odomDeskewInfo();

        return true;
    }

    /**
     * 计算当前激光时间戳前，每个时刻imu累积出的角速度，并判定imu数据是否可用（数量够）
    */
    void imuDeskewInfo()
    {
        cloudInfo.imuAvailable = false;

        //直到imu的时间戳到当前scan时间戳的前0.01s内
        while (!imuQueue.empty())
        {
            //再次确保时间戳
            if (imuQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
                imuQueue.pop_front();
            else
                break;
        }

        if (imuQueue.empty())
            return;

        imuPointerCur = 0;

        
        for (int i = 0; i < (int)imuQueue.size(); ++i)
        {
            sensor_msgs::Imu thisImuMsg = imuQueue[i];
            double currentImuTime = thisImuMsg.header.stamp.toSec();

            //当前imu时间戳小于当前点云时间戳转换，大于下一帧+0.01s退出
            // get roll, pitch, and yaw estimation for this scan
            if (currentImuTime <= timeScanCur)
                //将message里面的IMU消息转为tf类型的数据
                //前者只是个float类型的结构体，后者则是一个类，封装了很多函数
                //将imu的朝向赋值给点云，若非九轴imu在此处会不准
                imuRPY2rosRPY(&thisImuMsg, &cloudInfo.imuRollInit, &cloudInfo.imuPitchInit, &cloudInfo.imuYawInit);

            //当前的imu时间戳比lidar时间戳大过0.01s --- 是大太多了吗？是的
            if (currentImuTime > timeScanNext + 0.01)
                break;

            //就是存储对应时间戳的IMU的pose
            //赋值0
            if (imuPointerCur == 0){
                imuRotX[0] = 0;
                imuRotY[0] = 0;
                imuRotZ[0] = 0;
                imuTime[0] = currentImuTime; //存储和lidar时间戳接近的imu时间戳
                ++imuPointerCur;
                continue;
            }

            // get angular velocity
            double angular_x, angular_y, angular_z;
            imuAngular2rosAngular(&thisImuMsg, &angular_x, &angular_y, &angular_z);

            // integrate rotation 用axt累加
            //计算当前时间差 当前的imu时间戳减去上一时刻的imu时间戳得到时间增量
            double timeDiff = currentImuTime - imuTime[imuPointerCur-1];
            //当前时刻的旋转等于上一时刻的旋转加上上一时刻角速度乘上时间增量
            imuRotX[imuPointerCur] = imuRotX[imuPointerCur-1] + angular_x * timeDiff;
            imuRotY[imuPointerCur] = imuRotY[imuPointerCur-1] + angular_y * timeDiff;
            imuRotZ[imuPointerCur] = imuRotZ[imuPointerCur-1] + angular_z * timeDiff;
            imuTime[imuPointerCur] = currentImuTime;
            ++imuPointerCur;
        }

        /*******为什么--呢？******/
        //把赋值0后面的自加减回去
        --imuPointerCur;

        //如果数量为0直接返回，不为0，改imu属性为可用
        if (imuPointerCur <= 0)
            return;

        cloudInfo.imuAvailable = true;//开始，结束的标志，在完成去畸变之前，这个信息是不可用的
    }

    void odomDeskewInfo()
    {
        //是有哪儿加了这个flag要是true才能用的判断吗
        cloudInfo.odomAvailable = false;

        while (!odomQueue.empty())
        {
            if (odomQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
                odomQueue.pop_front();
            else
                break;
        }

        if (odomQueue.empty())
            return;

        /*****这个return了之后会在哪儿报错吗*********/
        if (odomQueue.front().header.stamp.toSec() > timeScanCur)
            return;

        // get start odometry at the beinning of the scan
        nav_msgs::Odometry startOdomMsg;

        //里程计开始的时间要在这个lidar前面
        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            startOdomMsg = odomQueue[i];

            if (ROS_TIME(&startOdomMsg) < timeScanCur)
                continue;
            else
                break;
        }

        tf::Quaternion orientation;
        tf::quaternionMsgToTF(startOdomMsg.pose.pose.orientation, orientation);

        double roll, pitch, yaw;
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

        // Initial guess used in mapOptimization
        //将点云信息中的位姿设置为视觉里程计的位姿
        cloudInfo.odomX = startOdomMsg.pose.pose.position.x;
        cloudInfo.odomY = startOdomMsg.pose.pose.position.y;
        cloudInfo.odomZ = startOdomMsg.pose.pose.position.z;
        cloudInfo.odomRoll  = roll;
        cloudInfo.odomPitch = pitch;
        cloudInfo.odomYaw   = yaw;
        cloudInfo.odomResetId = (int)round(startOdomMsg.pose.covariance[0]);

        cloudInfo.odomAvailable = true;//表示此时视觉里程计可用

        // get end odometry at the end of the scan
        //检查视觉里程计队列末尾的值
        odomDeskewFlag = false;

        //如果视觉里程计队列末尾值的时间戳小于两帧lidar的时间戳，说明视觉里程计频率过低
        if (odomQueue.back().header.stamp.toSec() < timeScanNext)
            return;

        nav_msgs::Odometry endOdomMsg;

        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            endOdomMsg = odomQueue[i];

            if (ROS_TIME(&endOdomMsg) < timeScanNext)
                continue;
            else
                break;
        }

        //visual_estimator/utility/visualization.cpp中发布给LIO的最后一项为failureCount
        //初始值设为 -1，每次clearState都会导致++failureCount
        //如果前后的failureCount不一致，说明在lidar当前帧内，VO至少重启了一次，跟踪失败，则值不准确
        //因此不对当前VO去畸变
        if (int(round(startOdomMsg.pose.covariance[0])) != int(round(endOdomMsg.pose.covariance[0])))
            return;

        Eigen::Affine3f transBegin = pcl::getTransformation(startOdomMsg.pose.pose.position.x, startOdomMsg.pose.pose.position.y, startOdomMsg.pose.pose.position.z, roll, pitch, yaw);

        tf::quaternionMsgToTF(endOdomMsg.pose.pose.orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        Eigen::Affine3f transEnd = pcl::getTransformation(endOdomMsg.pose.pose.position.x, endOdomMsg.pose.pose.position.y, endOdomMsg.pose.pose.position.z, roll, pitch, yaw);

        Eigen::Affine3f transBt = transBegin.inverse() * transEnd;

        float rollIncre, pitchIncre, yawIncre;
        pcl::getTranslationAndEulerAngles(transBt, odomIncreX, odomIncreY, odomIncreZ, rollIncre, pitchIncre, yawIncre);

        //在后续的findPosition中使用
        odomDeskewFlag = true;
    }

    //根据点云中某点的时间戳赋予其对应插值得到的旋转量
    void findRotation(double pointTime, float *rotXCur, float *rotYCur, float *rotZCur)
    {
        *rotXCur = 0; *rotYCur = 0; *rotZCur = 0;

        int imuPointerFront = 0;
        //找到点云时间戳前最近的一个imu时间戳
        while (imuPointerFront < imuPointerCur)
        {
            //点云时间比imu时间队列中最靠前的都小，则直接返回
            if (pointTime < imuTime[imuPointerFront])
                break;
            ++imuPointerFront;
        }

        //函数到这时imuPointerFront = imuPointerCur
        if (pointTime > imuTime[imuPointerFront] || imuPointerFront == 0)
        {
            //此时点云的姿态等于IMU的姿态
            *rotXCur = imuRotX[imuPointerFront];
            *rotYCur = imuRotY[imuPointerFront];
            *rotZCur = imuRotZ[imuPointerFront];
        } else {
            //如果点云时间还要靠前，则用差值的方法计算姿态
            int imuPointerBack = imuPointerFront - 1;
            //算一下该点时间戳在本组imu中的位置
            double ratioFront = (pointTime - imuTime[imuPointerBack]) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            double ratioBack = (imuTime[imuPointerFront] - pointTime) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            *rotXCur = imuRotX[imuPointerFront] * ratioFront + imuRotX[imuPointerBack] * ratioBack;
            *rotYCur = imuRotY[imuPointerFront] * ratioFront + imuRotY[imuPointerBack] * ratioBack;
            *rotZCur = imuRotZ[imuPointerFront] * ratioFront + imuRotZ[imuPointerBack] * ratioBack;
        }
    }

    //对位置信息去畸变，原版说速度比较慢是不起太大作用
    void findPosition(double relTime, float *posXCur, float *posYCur, float *posZCur)
    {
        *posXCur = 0; *posYCur = 0; *posZCur = 0;

        // if (cloudInfo.odomAvailable == false || odomDeskewFlag == false)
        //     return;

        //同样是按照差值的方式计算此时的点云
        // float ratio = relTime / (timeScanNext - timeScanCur);

        // *posXCur = ratio * odomIncreX;
        // *posYCur = ratio * odomIncreY;
        // *posZCur = ratio * odomIncreZ;
    }

    
    //点云信息去畸变
    PointType deskewPoint(PointType *point, double relTime)
    {
        //来源于上文的时间戳通道和imu可用判断，没有或是不可用则返回点
        if (deskewFlag == -1 || cloudInfo.imuAvailable == false)
            return *point;

        //点的时间等于scan时间加relTime （后文的laserCloudIn->points[i].time）
        double pointTime = timeScanCur + relTime;

        //根据时间戳插值获取imu计算的旋转量与位置量
        float rotXCur, rotYCur, rotZCur;
        findRotation(pointTime, &rotXCur, &rotYCur, &rotZCur);

        float posXCur, posYCur, posZCur;
        findPosition(relTime, &posXCur, &posYCur, &posZCur);

        
        
        
        //这里的firstPointFlag来源于resetParameters函数，而resetParameters函数每次ros调用cloudHandler都会启动
        //也就是求扫描第一个点时lidar的世界坐标系下变换矩阵的逆
        if (firstPointFlag == true)
        {
            transStartInverse = (pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur)).inverse();
            firstPointFlag = false;
        }

        // transform points to start
        //扫描当前点时lidar的世界坐标系下变换矩阵
        Eigen::Affine3f transFinal = pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur);
        //扫描该点相对扫描本次scan第一个点的lidar变换矩阵=第一个点时lidar世界坐标系下变换矩阵的逆×当前点时lidar世界坐标系下变换矩阵
        Eigen::Affine3f transBt = transStartInverse * transFinal;

        //根据lidar位姿变幻，修正点云位置
        PointType newPoint;
        newPoint.x = transBt(0,0) * point->x + transBt(0,1) * point->y + transBt(0,2) * point->z + transBt(0,3);
        newPoint.y = transBt(1,0) * point->x + transBt(1,1) * point->y + transBt(1,2) * point->z + transBt(1,3);
        newPoint.z = transBt(2,0) * point->x + transBt(2,1) * point->y + transBt(2,2) * point->z + transBt(2,3);
        newPoint.intensity = point->intensity;

        return newPoint;
    }

    //将深度西悉尼投影到RangeImgae上
    void projectPointCloud()
    {
        int cloudSize = (int)laserCloudIn->points.size();
        // range image projection
        for (int i = 0; i < cloudSize; ++i)
        {
            PointType thisPoint;
            thisPoint.x = laserCloudIn->points[i].x;
            thisPoint.y = laserCloudIn->points[i].y;
            thisPoint.z = laserCloudIn->points[i].z;
            thisPoint.intensity = laserCloudIn->points[i].intensity;

            int rowIdn = laserCloudIn->points[i].ring;
            
            //0--N_SCAN内，且是整数
            if (rowIdn < 0 || rowIdn >= N_SCAN)
                continue;

            if (rowIdn % downsampleRate != 0)
                continue;

            //行---方位角，4象限反正切
            //在rangeimage上面算的
            float horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;

            
            //Horizon_SCAN就是一条线一周扫描多少次
            //水平扫描角度步长，例如一周扫描1800次，则两次扫描间隔角度0.2度
            static float ang_res_x = 360.0/float(Horizon_SCAN);
            int columnIdn = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN/2;
            if (columnIdn >= Horizon_SCAN)
                columnIdn -= Horizon_SCAN;

            //如果线数不正确
            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
                continue;

            float range = pointDistance(thisPoint);
            
            if (range < 1.0)
                continue;

            if (rangeMat.at<float>(rowIdn, columnIdn) != FLT_MAX)
                continue;

            // for the amsterdam dataset
            // if (range < 6.0 && rowIdn <= 7 && (columnIdn >= 1600 || columnIdn <= 200))
            //     continue;
            // if (thisPoint.z < -2.0)
            //     continue;

            ////为啥先存储距离再去畸变
            rangeMat.at<float>(rowIdn, columnIdn) = range;

            //对点云去畸变
            thisPoint = deskewPoint(&thisPoint, laserCloudIn->points[i].time); // Velodyne
            // thisPoint = deskewPoint(&thisPoint, (float)laserCloudIn->points[i].t / 1000000000.0); // Ouster

            int index = columnIdn  + rowIdn * Horizon_SCAN;
            fullCloud->points[index] = thisPoint;
        }
    }

    //点云RangeIMage
    void cloudExtraction()
    {
        int count = 0;
        // extract segmented cloud for lidar odometry
        for (int i = 0; i < N_SCAN; ++i)//竖直方向
        {
            //从第一个scan开始
            //记录每根扫描线起始第5个激光点在一维数组中的索引
            cloudInfo.startRingIndex[i] = count - 1 + 5;//最开始的5个不考虑

            for (int j = 0; j < Horizon_SCAN; ++j)//水平方向
            {
                if (rangeMat.at<float>(i,j) != FLT_MAX)
                {
                    // mark the points' column index for marking occlusion later
                    //记录激光点对应的Horizon_SCAN方向上的索引
                    cloudInfo.pointColInd[count] = j;
                    // save range info
                    //激光电距离
                    cloudInfo.pointRange[count] = rangeMat.at<float>(i,j);
                    // save extracted cloud
                    //加入有效激光电
                    extractedCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                    // size of extracted cloud
                    ++count;
                }
            }
            cloudInfo.endRingIndex[i] = count -1 - 5;//最末尾的5个不考虑
        }
    }
    
    //发布当前矫正后点云，有效点
    void publishClouds()
    {
        cloudInfo.header = cloudHeader; //点云的头部信息，包含时间戳、坐标系信息
        cloudInfo.cloud_deskewed  = publishCloud(&pubExtractedCloud, extractedCloud, cloudHeader.stamp, "base_link");
        pubLaserCloudInfo.publish(cloudInfo);
    }
};

int main(int argc, char** argv)
{
    //节点初始化，反正它的名字后面会被换掉，那这个地方叫什么是不是也无所谓的？---是的，已经测试过了，无所谓的
    ros::init(argc, argv, "lidar");

    ImageProjection IP;
    
    ROS_INFO("\033[1;32m----> Lidar Cloud Deskew Started.\033[0m");

    ros::MultiThreadedSpinner spinner(3);
    spinner.spin();
    
    return 0;
}