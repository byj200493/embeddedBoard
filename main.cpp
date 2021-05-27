/*
 * @Author: your name
 * @Date: 2020-09-17 09:48:22
 * @LastEditTime: 2020-10-29 00:29:45
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /libsniris/examples/three_camControl/main.cpp
 */
#include <iostream>
#include "camControl.hpp"
#include "version_soft.hpp"

using namespace sniris;
using namespace std;
using namespace cv;
using namespace SN;

int main(int argc, char const* argv[])
{
    std::cout << "use threeCamScan version : " << THREECAMSCAN_VERSION_STR << endl;

    camControl c_camControl;
    string path = "../config/0602";
    c_camControl.init(1280, 800, 1.1f, path);

#ifdef _WIN32
    // 令控制台显示utf-8编码
    system("chcp 65001");
#endif // _WIN32
    cout << "use libsniris version : " << IRIS_VERSION_STR << endl;
    // 输出调试信息到控制台
    // log_to_console(log_severity::LOG_SEVERITY_DEBUG);
    //log_to_file(log_severity::LOG_SEVERITY_DEBUG, "./error.log", 1);

    // 创建搜索设备的配置参数类
    auto ctx = creat_preconfig(IRIS_VERSION);
    // 获取符合设备的列表
    auto devLists = query_devices(ctx);

    std::vector<shared_ptr<device>> device_query;
    device_query.clear();
    if (devLists->empty())
        return 0;

    int cam_count = CAM_NUM;

    for(int i=0; i<cam_count; i++)
    {
        auto dev = (*devLists)[i];
        auto dev_mode_name = dev->get_device_model_name();
        cout << "device model name : " << dev_mode_name << endl;

        auto stream_bayerColor = dev->is_streams_supports(stream_model_type::COLORBAYER_COLOR);
        auto stream_remapColor = dev->is_streams_supports(stream_model_type::REMAP_COLOR);

        auto status = dev->set_streams(stream_model_type::COLOR_DEPTH, depth_resolution::VALID_RES_1280x800, 10);
        status = dev->set_laser_power(80);
        dev->set_ir_exposure_average(45);
        dev->start();
        std::string dev_sn = dev->get_device_sn();
        std::cout<<"dev_sn : " << dev_sn << std::endl;
        //c_camControl.camExtrinsicParams(i, dev_sn);
        device_query.push_back(dev);
    }

    std::vector<thread> vecThreads;
    vecThreads.clear();
    for(int i=0; i<cam_count; i++)
    {
        auto color_stream = device_query[i]->get_color_stream();
        auto depth_stream = device_query[i]->get_depth_stream();
        c_camControl.get_cam_params(color_stream, depth_stream, i);
        vecThreads.push_back(thread(&camControl::color_img_thread, c_camControl, color_stream, i));
        vecThreads.push_back(thread(&camControl::depth_img_thread, c_camControl, depth_stream, i));
    }


    vecThreads.push_back(thread(&camControl::capture_image_data, c_camControl));



    for (auto &t : vecThreads)
    {
        t.join();
    }

    return 0;
}
