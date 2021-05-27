#include "camControl.hpp"

#ifdef USE_TENSORRT
// prepare input data ---------------------------
static float g_tensorrt_data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
static float g_tensorrt_prob[BATCH_SIZE * OUTPUT_SIZE];
#endif

mutex show_mutex;
int _key = -1;
s_camParams camParams[CAM_NUM];//相机参数
s_captureInfos captureInfos[CAM_NUM];//每个相机的图像数据
s_captureInfos mergeInfos[CAM_NUM];
std::mutex mtx_capture[CAM_NUM*2];
std::mutex mtx_proc;
std::condition_variable cond_proc;

int g_moduleType = 0;
long g_timeStamp = 1000;

cv::Mat backgroud_depth;
int depthPointNumThresh = 5000;



camControl::camControl(/* args */)
{

    m_midcam_id = -1;
    m_width = 1280;
    m_height = 800;
 
}

camControl::~camControl()
{
    if (m_cam_rotMat.size()>0)
    {
        m_cam_rotMat.clear();
    }

    if (m_cam_tVec.size()>0)
    {
        m_cam_tVec.clear();
    }
}


void camControl::init(int width,int height, float thresh, std::string configPath)
{
    m_configPath = configPath;

    m_width = width;
    m_height = height;


    for (size_t i = 0; i < CAM_NUM; i++)
    {
        captureInfos[i].colorMat = cv::Mat(height, width, CV_8UC3);
        captureInfos[i].depthMat = cv::Mat(height, width, CV_16UC1);
        captureInfos[i].update_color = false;
        captureInfos[i].update_depth = false;

        mergeInfos[i].colorMat = cv::Mat(height, width, CV_8UC3);
        mergeInfos[i].depthMat = cv::Mat(height, width, CV_16UC1);
        mergeInfos[i].update_color = false;
        mergeInfos[i].update_depth = false;
    }


}


void camControl::get_cam_params(sniris::color_stream* _color_stream, sniris::depth_stream* _depth_stream, int id)
{
    Mat _leftCamera=cv::Mat::eye(3,3,CV_64F);
    Mat _rgbCamera=cv::Mat::eye(3,3,CV_64F);
    Mat _leftCoeffs=cv::Mat::zeros(1,5,CV_64F);
    Mat _rgbCoeffs=cv::Mat::zeros(1,5,CV_64F);

    auto color_intrinsic = _color_stream->m_intrinsics;
    _rgbCamera.at<double>(0,0) = color_intrinsic.fx;
    _rgbCamera.at<double>(1,1) = color_intrinsic.fy;
    _rgbCamera.at<double>(0,2) = color_intrinsic.ppx;
    _rgbCamera.at<double>(1,2) = color_intrinsic.ppy;
    _rgbCoeffs.ptr<double>(0)[0] = color_intrinsic.coeffs[0];
    _rgbCoeffs.ptr<double>(0)[1] = color_intrinsic.coeffs[1];
    _rgbCoeffs.ptr<double>(0)[2] = color_intrinsic.coeffs[2];
    _rgbCoeffs.ptr<double>(0)[3] = color_intrinsic.coeffs[3];

    // 去畸变得到映射表
    initUndistortRectifyMap(_rgbCamera,_rgbCoeffs,cv::Mat::eye(3,3,CV_32F),_rgbCamera,Size(1280,800),CV_32F, camParams[id].rectify_map1, camParams[id].rectify_map2);
    _rgbCoeffs.ptr<double>(0)[0]=0.0;
    _rgbCoeffs.ptr<double>(0)[1]=0.0;
    _rgbCoeffs.ptr<double>(0)[2]=0.0;
    _rgbCoeffs.ptr<double>(0)[3]=0.0;
    _rgbCoeffs.ptr<double>(0)[4]=0.0;

    auto depth_intrinsic = _depth_stream->m_intrinsics;


    auto depth_trans = _depth_stream->m_extrinsics;

    //
    camParams[id].cfx = color_intrinsic.fx;
    camParams[id].cfy = color_intrinsic.fy;
    camParams[id].cppx = color_intrinsic.ppx;
    camParams[id].cppy = color_intrinsic.ppy;

    camParams[id].dfx = depth_intrinsic.fx;
    camParams[id].dfy = depth_intrinsic.fy;
    camParams[id].dppx = depth_intrinsic.ppx;
    camParams[id].dppy = depth_intrinsic.ppy;

    camParams[id].d2cR = cv::Mat::zeros(3,3,CV_64F);
    memcpy(camParams[id].d2cR.data,depth_trans.rotation,sizeof(double)*9);

    camParams[id].d2cT = cv::Vec3d(0,0,0);
    camParams[id].d2cT << depth_trans.translation[0], depth_trans.translation[1], depth_trans.translation[2];

    std::cout<<camParams[id].d2cR<<std::endl;
}

void camControl::depth2colorAligned(cv::Mat &depth, cv::Mat &aligned_depth, int id)
{
    int height = depth.rows;
    int width = depth.cols;
    aligned_depth=cv::Mat(height,width,CV_16UC1,Scalar(0xffff));

    auto alignedDepthData = aligned_depth.ptr<ushort>();

    // depth 焦距和光心
    auto dfx = camParams[id].dfx;
    auto dfy = camParams[id].dfy;
    auto dppx = camParams[id].dppx;
    auto dppy = camParams[id].dppy;
    // 深度到彩色 T
    auto rotationData = camParams[id].d2cR.ptr<double>();
    // 彩色图内参畸变
    auto cfx = camParams[id].cfx;
    auto cfy = camParams[id].cfy;
    auto cppx = camParams[id].cppx;
    auto cppy = camParams[id].cppy;

    for (int j = 0; j < height; ++j)
    {
        auto row = depth.ptr<ushort>(j);
        for (int i = 0; i < width; ++i)
        {
            auto depthVal = row[i];
            if (depthVal <= 0xffff)
            {
                // 先转换成depth的3维坐标
                auto dx = ((i - dppx) / dfx)*depthVal;
                auto dy = ((j - dppy) / dfy)*depthVal;
                auto dz = depthVal;

                // 彩色图的3维坐标
                auto cx = rotationData[0] * dx + rotationData[1] * dy + rotationData[2] * dz + camParams[id].d2cT[0];
                auto cy = rotationData[3] * dx + rotationData[4] * dy + rotationData[5] * dz + camParams[id].d2cT[1];
                auto cz = rotationData[6] * dx + rotationData[7] * dy + rotationData[8] * dz + camParams[id].d2cT[2];

                // 彩色图3维转2d
                auto cxx = cx / cz;
                auto cyy = cy / cz;

                auto cPointx = int(cxx * cfx + cppx + 0.5);
                auto cPointy = int(cyy * cfy + cppy + 0.5);
                int cindex = cPointy*width + cPointx;
                // 将对应的点填入深度值
                if (cPointx >= 0 && cPointx <= width - 1 && cPointy >= 0 && cPointy <= height - 1)
                {
                    alignedDepthData[cindex] = depthVal;
                }
            }
        }
    }
    cv::medianBlur(aligned_depth,aligned_depth,5);
}
// 判断color和depth是否都已经更新
bool camControl::checkUpdateLabel()
{
    bool ret = true;
    for (size_t i = 0; i < CAM_NUM; i++)
    {
        ret = ret && captureInfos[i].update_color && captureInfos[i].update_depth;
    }

    return ret;
}




void camControl::lr_img_thead(infrared_stream* stream)
{
    Mat ir_img = Mat(Size(2560, 800), CV_8UC1);
    while (1)
    {
        if (stream->wait_for_frame(ir_img.data))
        {
            std::lock_guard<mutex> lk(show_mutex);
            Mat left = ir_img.colRange(0, 1280);
            imshow("infrared left and right", ir_img);
            imshow("left", left);
            _key = waitKey(5);
            if (_key == 'q')
            {
                break;
            }
        }
    }
}

void camControl::color_img_thread(color_stream* stream, int id)
{
    Mat color_yuv2 = Mat(Size(1280, 800), CV_8UC2);
    Mat color_bgr = Mat(Size(1280, 800), CV_8UC3);
    Mat color_dist = Mat(Size(1280, 800), CV_8UC3);
    while(1)
    {
        if (stream->wait_for_frame(color_yuv2.data))
        {
            uint64_t fpga_timeStamp = stream->get_timestamp();
            struct timeval sys_tv;
            gettimeofday(&sys_tv, NULL);
            //printf("color thread : %d\t%d\n", sys_tv.tv_sec, sys_tv.tv_usec);

            cvtColor(color_yuv2, color_bgr, COLOR_YUV2BGR_YUY2);
            //rgb图畸变矫正
            remap(color_bgr,color_dist,camParams[id].rectify_map1,camParams[id].rectify_map2,cv::INTER_LINEAR);

            //
            std::lock_guard<mutex> lk(mtx_capture[id*2]);
            std::memcpy(captureInfos[id].colorMat.data, color_dist.data, m_height*m_width*3);
            captureInfos[id].update_color = true; // 图片已更新，置为true
            captureInfos[id].color_sys_tv.tv_sec = sys_tv.tv_sec;
            captureInfos[id].color_sys_tv.tv_usec = sys_tv.tv_usec;
            captureInfos[id].color_fpga_tv = fpga_timeStamp;
            cond_proc.notify_one();
            // std::lock_guard<mutex> lk(show_mutex);
            // imshow(std::to_string(id)+"_color", color_dist);
            // _key = waitKey(5);

            // if (_key == 'q')
            // {
            //     break;
            // }
        }
    }
}

void camControl::depth_img_thread(depth_stream* stream, int id)
{
    Mat depth_img = Mat(Size(1280, 800), CV_16UC1);
    Mat aligned_depth_img;
    while (1)
    {
        if (stream->wait_for_frame(depth_img.data))
        {
            uint64_t fpga_timeStamp = stream->get_timestamp();
            struct timeval sys_tv;
            gettimeofday(&sys_tv, NULL);
            //printf("depth thread : %d\t%d\n", sys_tv.tv_sec, sys_tv.tv_usec);

            depth2colorAligned(depth_img, aligned_depth_img, id);

            std::lock_guard<mutex> lk(mtx_capture[id*2+1]);
            std::memcpy(captureInfos[id].depthMat.data, aligned_depth_img.data, m_height*m_width*2);
            captureInfos[id].update_depth = true; // 图片已更新，置为true
            captureInfos[id].depth_sys_tv = sys_tv;
            captureInfos[id].depth_fpga_tv = fpga_timeStamp;
            cond_proc.notify_one();
            // std::lock_guard<mutex> lk(show_mutex);
            // imshow(std::to_string(id)+"_depth", aligned_depth_img);
            // //waitKey(5);
            // if (_key == 'q')
            // {
            //     break;
            // }
        }
    }
}



// Thread function
//
void camControl::capture_image_data()
{
    if(m_midcam_id == -1)
        m_midcam_id = 0;

    cout << "middle device id : " << m_midcam_id << endl;


    char buf[100];
    char buffer[100];
    int cnt=0;



    // 计数在roi窗口内多于25帧就认为传送带停止了
    int checkCount = 0;
    int nFrameCount = 0;
    while (1)
    {
        std::unique_lock<std::mutex> lk(mtx_proc);
        // 等待，直到color和depth数据都更新才做后面的处理
        cond_proc.wait(lk, [&] {return (checkUpdateLabel());});

        for (size_t i = 0; i < CAM_NUM; i++)
        {
            captureInfos[i].update_color = false;
            captureInfos[i].update_depth = false;
            std::memcpy(mergeInfos[i].colorMat.data, captureInfos[i].colorMat.data, m_width*m_height*3);
            std::memcpy(mergeInfos[i].depthMat.data, captureInfos[i].depthMat.data, m_width*m_height*2);
        }

        struct timeval curr_sys_tv;
        curr_sys_tv.tv_sec = captureInfos[m_midcam_id].color_sys_tv.tv_sec;
        curr_sys_tv.tv_usec = captureInfos[m_midcam_id].color_sys_tv.tv_usec;

        nFrameCount++;

        lk.unlock();
        

        //图像数据保存
        char szPathDepth[100];
        char szPathColor[100];
        for (size_t i = 0; i < CAM_NUM; i++)
        {
            sprintf(szPathColor,"./threeCamCap/%03d/%03d_color.png",i,nFrameCount);
            imwrite(szPathColor,mergeInfos[i].colorMat);

            sprintf(szPathDepth,"./threeCamCap/%03d/%03d_depth.png",i,nFrameCount);
            imwrite(szPathDepth,mergeInfos[i].depthMat);

            std::cout<<"Save camara " << i + 1 << "image data" << std::endl;

        }

        // cv::imshow("color",mergeInfos[0].colorMat);
        // cv::waitKey(30);
        /*
        cv::imshow("depth", mergeInfos[0].depthMat);
        //imwrite("depth.png", mergeInfos[0].depthMat);
        auto _key = cv::waitKey(5);
        if(_key == 's')
        {
            imwrite("/home/supernode/test/bg_depth.png", mergeInfos[0].depthMat);
            cout << "save background depth success..." << endl;
        }
        */

    }
}

