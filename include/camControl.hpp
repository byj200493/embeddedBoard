/*
 * @Author: your name
 * @Date: 2020-09-17 11:37:08
 * @LastEditTime: 2020-09-25 15:28:39
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /libsniris/examples/three_camControl/camControl.hpp
 */
#include <iostream>
#include <sys/time.h>
#include <pcl/filters/passthrough.h>
#include <libsniris/sniris.hpp>
#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <condition_variable>


using namespace sniris;
using namespace std;
using namespace cv;

#define CAM_NUM 3

enum NETCOM_TYPE
{
    NETCOM_REQ_BASEINFO = 0,
    NETCOM_REQ_CANDIINFO,
    NETCOM_SEND_STATUSINFO
};

enum SEGMETHOD
{
    YOLOV4_M1 = 0,
    YOLOV4_M2,
    CLUSTER_M
};

struct s_rectPointsInfors
{
	cv::Point3f maxPoint3D;
	cv::Point3f minPoint3D;
	cv::Point2i maxPoint2D;
	cv::Point2i minPoint2D;
	cv::Rect objRect;
};

struct s_camParams
{
    double cfx;
    double cfy;
    double cppx;
    double cppy;

    double dfx;
    double dfy;
    double dppx;
    double dppy;

    Mat d2cR;
    Vec3d d2cT;

    Mat rectify_map1;
    Mat rectify_map2;   
};

struct s_captureInfos
{
    cv::Mat colorMat;
    cv::Mat depthMat;
    bool update_color;
    bool update_depth;
    struct timeval color_sys_tv;
    struct timeval depth_sys_tv;

    uint64_t color_fpga_tv;
    uint64_t depth_fpga_tv;
};


class camControl
{

public:
    camControl();
    ~camControl();

private:

    int m_width;
    int m_height;

    std::string m_left_sn;
    std::string m_right_sn;
    std::string m_mid_sn;

    std::vector<cv::Mat> m_cam_rotMat;
    std::vector<cv::Mat> m_cam_tVec;

    int m_midcam_id;
    
public:

    void init(int wodth, int height, float thresh, std::string xmlPath);

    void get_cam_params(sniris::color_stream* _color_stream, sniris::depth_stream* _depth_stream, int id);

    void lr_img_thead(infrared_stream* stream);
    void color_img_thread(color_stream* stream, int id);
    void depth_img_thread(depth_stream* stream, int id);

    void capture_image_data();// 

private:
    
    void depth2colorAligned(cv::Mat &depth, cv::Mat &aligned_depth, int id);
    bool checkUpdateLabel();



//行李位置检测
//深度分析
    std::string m_configPath;


};
