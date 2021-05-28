#include "utility.h"
#include "opencv2/opencv.hpp"
//#include <pcl/point_cloud.h>
//#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
//#include <pcl/visualization/pcl_visualizer.h>
#include <QElapsedTimer>
#include "Simplify.h"
#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include "pcl/filters/extract_indices.h"
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/pca.h>

#include "pcl_alignment.h"
/*
created at October 14 2020
*/
void _fillingHoles(cv::Mat &depthMat, int holeSize)
{
    int width = depthMat.cols;
    int height = depthMat.rows;
    cv::Mat labelMat(depthMat.size(), CV_16UC1);
    for (int iy = 0; iy < depthMat.rows; ++iy)
    {
        for (int ix = 0; ix < depthMat.cols; ++ix)
        {
            if (depthMat.at<unsigned short>(iy, ix) > 0)
                labelMat.at<unsigned short>(iy, ix) = 1;
            else
                labelMat.at<unsigned short>(iy, ix) = 0;
        }
    }
    //extract holes
    int label = 1;
    std::vector<REG> regs;
    for (int iy = 1; iy < height-1; ++iy)
    {
        for (int ix = 1; ix < width-1; ++ix)
        {
            if (labelMat.at<unsigned short>(iy, ix) > 0)
                continue;
            ++label;
            std::vector<cv::Point> reg;
            unsigned long count = 0;
            cv::Point p(ix, iy);
            reg.push_back(p);
            while (reg.size() > count)
            {
                p = reg[count];
                ++count;
                if ((p.x + 1) < (width - 1))
                {
                    cv::Point r;
                    r.x = p.x + 1;
                    r.y = p.y;
                    if (labelMat.at<unsigned short>(r.y, r.x) == 0)
                    {
                        reg.push_back(r);
                        labelMat.at<unsigned short>(r.y, r.x) = label;
                    }
                }
                if ((p.x - 1) > 0)
                {
                    cv::Point r;
                    r.x = p.x - 1;
                    r.y = p.y;
                    if (labelMat.at<unsigned short>(r.y, r.x) == 0)
                    {
                        reg.push_back(r);
                        labelMat.at<unsigned short>(r.y, r.x) = label;
                    }
                }
                if ((p.y - 1) > 0)
                {
                    cv::Point r;
                    r.x = p.x;
                    r.y = p.y-1;
                    if (labelMat.at<unsigned short>(r.y, r.x) == 0)
                    {
                        reg.push_back(r);
                        labelMat.at<unsigned short>(r.y, r.x) = label;
                    }
                }
                if ((p.y + 1) < (height-1))
                {
                    cv::Point r;
                    r.x = p.x;
                    r.y = p.y+1;
                    if (labelMat.at<unsigned short>(r.y, r.x) == 0)
                    {
                        reg.push_back(r);
                        labelMat.at<unsigned short>(r.y, r.x) = label;
                    }
                }
            }
            regs.push_back(reg);
        }
    }
    std::vector<REG> holes;
    for (unsigned long i = 0; i < regs.size(); ++i)
    {
        if (regs[i].size() < (unsigned long)holeSize)
        {
            holes.push_back(regs[i]);
        }
    }
    //filling holes
    for (unsigned long i = 0; i < holes.size(); ++i)
    {
        /*
        extract boundary of hole and fill it.
        repeat this process until filling hole fully.
        */
        while (holes[i].size() > 0)
        {
            //extracting boundary of hole
            std::vector<cv::Point> boundary;
            std::vector<unsigned short> depths;
            for (unsigned long j = 0; j < holes[i].size(); ++j)
            {
                cv::Point p = holes[i][j];
                if (depthMat.at<unsigned short>(p.y, p.x + 1) > 0)
                {
                    boundary.push_back(p);
                    depths.push_back(depthMat.at<unsigned short>(p.y, p.x + 1));
                    holes[i].erase(holes[i].begin() + j);
                    continue;
                }
                if (depthMat.at<unsigned short>(p.y, p.x - 1) > 0)
                {
                    boundary.push_back(p);
                    depths.push_back(depthMat.at<unsigned short>(p.y, p.x - 1));
                    holes[i].erase(holes[i].begin() + j);
                    continue;
                }
                if (depthMat.at<unsigned short>(p.y-1, p.x) > 0)
                {
                    boundary.push_back(p);
                    depths.push_back(depthMat.at<unsigned short>(p.y-1, p.x));
                    holes[i].erase(holes[i].begin() + j);
                    continue;
                }
                if (depthMat.at<unsigned short>(p.y+1, p.x) > 0)
                {
                    boundary.push_back(p);
                    depths.push_back(depthMat.at<unsigned short>(p.y+1, p.x));
                    holes[i].erase(holes[i].begin() + j);
                    continue;
                }
            }
            if (boundary.size() == 0)
                break;
            //filling boundary of hole
            for (unsigned long j = 0; j < boundary.size(); ++j)
            {
                cv::Point p = boundary[j];
                labelMat.at<unsigned short>(p.y, p.x) = 1;
                depthMat.at<unsigned short>(p.y, p.x) = depths[j];
            }
            /*for (int j = 0; j < holes[i].size(); ++j)
            {
                cv::Point p = holes[i][j];
                if (depthMat.at<unsigned short>(p.y, p.x + 1) > 0)
                {
                    depthMat.at<unsigned short>(p.y, p.x) = depthMat.at<unsigned short>(p.y, p.x + 1);
                    holes[i].erase(holes[i].begin() + j);
                    continue;
                }
                if (depthMat.at<unsigned short>(p.y-1, p.x) > 0)
                {
                    depthMat.at<unsigned short>(p.y, p.x) = depthMat.at<unsigned short>(p.y-1, p.x);
                    holes[i].erase(holes[i].begin() + j);
                    continue;
                }
                if (depthMat.at<unsigned short>(p.y, p.x - 1) > 0)
                {
                    depthMat.at<unsigned short>(p.y, p.x) = depthMat.at<unsigned short>(p.y, p.x - 1);
                    holes[i].erase(holes[i].begin() + j);
                    continue;
                }
                if (depthMat.at<unsigned short>(p.y+1, p.x) > 0)
                {
                    depthMat.at<unsigned short>(p.y, p.x) = depthMat.at<unsigned short>(p.y+1, p.x);
                    holes[i].erase(holes[i].begin() + j);
                    continue;
                }
            }*/
        }
    }
}
/*
replaces zero depth with max depth among neighbours's depths.
This is like closing of morphological operations.
*/
void fillingHoles(cv::Mat &depthMat, int nLoop)
{
    int loop = 0;
    while (loop < nLoop)
    {
        cv::Mat tmp;
        depthMat.copyTo(tmp);
        unsigned short ne[9];
        int count = 0;
        for (int iy = 1; iy < depthMat.rows - 1; ++iy)
        {
            for (int ix = 1; ix < depthMat.cols - 1; ++ix)
            {
                unsigned short depth = depthMat.at<unsigned short>(iy, ix);
                if (depth > 0)
                    continue;
                ne[0] = tmp.at<unsigned short>(iy, ix);
                ne[1] = tmp.at<unsigned short>(iy, ix + 1);
                ne[2] = tmp.at<unsigned short>(iy - 1, ix + 1);
                ne[3] = tmp.at<unsigned short>(iy - 1, ix);
                ne[4] = tmp.at<unsigned short>(iy - 1, ix - 1);
                ne[5] = tmp.at<unsigned short>(iy, ix - 1);
                ne[6] = tmp.at<unsigned short>(iy + 1, ix - 1);
                ne[7] = tmp.at<unsigned short>(iy + 1, ix);
                ne[8] = tmp.at<unsigned short>(iy + 1, ix + 1);
                unsigned short max = 0;
                for (int i = 0; i < 9; ++i)
                {
                    if (ne[i] > max)
                        max = ne[i];
                }
                if (max > 0)
                {
                    depthMat.at<unsigned short>(iy, ix) = max;
                    ++count;
                }
            }
        }
        ++loop;
    }
    loop = 0;
    while (loop < nLoop)
    {
        cv::Mat tmp;
        depthMat.copyTo(tmp);
        for (int iy = 1; iy < depthMat.rows - 1; ++iy)
        {
            for (int ix = 1; ix < depthMat.cols - 1; ++ix)
            {
                if (tmp.at<unsigned short>(iy, ix) == 0)
                    continue;
                unsigned short ne[9];
                ne[0] = tmp.at<unsigned short>(iy, ix + 1);
                ne[1] = tmp.at<unsigned short>(iy - 1, ix + 1);
                ne[2] = tmp.at<unsigned short>(iy - 1, ix);
                ne[3] = tmp.at<unsigned short>(iy - 1, ix - 1);
                ne[4] = tmp.at<unsigned short>(iy, ix - 1);
                ne[5] = tmp.at<unsigned short>(iy + 1, ix - 1);
                ne[6] = tmp.at<unsigned short>(iy + 1, ix);
                ne[7] = tmp.at<unsigned short>(iy + 1, ix + 1);
                for (int i = 0; i < 8; ++i)
                {
                    if (ne[i] == 0)
                    {
                        depthMat.at<unsigned short>(iy, ix) = 0;
                        break;
                    }
                }
            }
        }
        ++loop;
    }
}
/*
segment depthmap in to smooth regions.
smooth regions are extracted by tolerance.
sorts them by their sizes.
*/
void segmentDepthmap(cv::Mat &depthMap, float maxDepth, float minDepth, float tolerance, cv::Mat &labelOut, std::vector<PAIR> &labels)
{
    int width = depthMap.cols;
    int height = depthMap.rows;
    cv::Mat labelMat(depthMap.size(), CV_16UC1, cv::Scalar(0));
    for (int iy = 0; iy < height; ++iy)
    {
        for (int ix = 0; ix < width; ++ix)
        {
            if (depthMap.at<unsigned short>(iy, ix) > maxDepth)
                depthMap.at<unsigned short>(iy, ix) = 0;
            if (depthMap.at<unsigned short>(iy, ix) < minDepth)
                depthMap.at<unsigned short>(iy, ix) = 0;
        }
    }
    int label = 1;
    for (int iy = 0; iy < height; ++iy)
    {
        for (int ix = 0; ix < width; ++ix)
        {
            if (depthMap.at<unsigned short>(iy, ix) == 0)
                labelMat.at<unsigned short>(iy, ix) = label;
        }
    }
    for (int iy = 1; iy < height - 1; ++iy)
    {
        for (int ix = 1; ix < width - 1; ++ix)
        {
            if (labelMat.at<unsigned short>(iy, ix) > 0)
                continue;
            ++label;
            std::vector<cv::Point> cluster;
            cluster.resize(1);
            cluster[0].x = ix; cluster[0].y = iy;
            labelMat.at<unsigned short>(iy, ix) = label;
            unsigned long count = 0;
            std::pair<int, int> pair;
            while (count < cluster.size())
            {
                std::vector<cv::Point> tmp;
                if ((cluster[count].x - 1) > 0)
                {
                    cv::Point p = cluster[count];
                    p.x--;
                    tmp.push_back(p);
                }
                if ((cluster[count].x + 1) < width)
                {
                    cv::Point p = cluster[count];
                    p.x++;
                    tmp.push_back(p);
                }
                if ((cluster[count].y - 1) > 0)
                {
                    cv::Point p = cluster[count];
                    p.y--;
                    tmp.push_back(p);
                }
                if ((cluster[count].y + 1) < height)
                {
                    cv::Point p = cluster[count];
                    p.y++;
                    tmp.push_back(p);
                }
                int depth = depthMap.at<unsigned short>(cluster[count].y, cluster[count].x);
                for (unsigned long i = 0; i < tmp.size(); ++i)
                {
                    int diff = abs(depth - (int)depthMap.at<unsigned short>(tmp[i].y, tmp[i].x));
                    if (diff < tolerance && labelMat.at<unsigned short>(tmp[i].y, tmp[i].x) == 0)
                    {
                        labelMat.at<unsigned short>(tmp[i].y, tmp[i].x) = label;
                        cluster.push_back(tmp[i]);
                    }
                }
                ++count;
            }
            pair.first = label;
            pair.second = count;
            labels.push_back(pair);
        }
    }
    //sort labels
    bool loop = true;
    while (loop == true)
    {
        loop = false;
        for (unsigned long i = 0; i < labels.size()-1; ++i)
        {
            std::pair<int, int> p;
            p = labels[i];
            if (labels[i].second < labels[i + 1].second)
            {
                labels[i] = labels[i + 1];
                labels[i + 1] = p;
                loop = true;
            }
        }
    }
    labelMat.copyTo(labelOut);
}

void extractObject(cv::Mat &depthMap, int maxDepth, int minDepth, int cWidth, int cHeight, float tolerance)
{
    int width = depthMap.cols;
    int height = depthMap.rows;
    cv::Mat labelMat(depthMap.size(), CV_16UC1, cv::Scalar(0));
    for (int iy = 0; iy < height; ++iy)
    {
        for (int ix = 0; ix < width; ++ix)
        {
            if (depthMap.at<unsigned short>(iy, ix) > maxDepth)
                depthMap.at<unsigned short>(iy, ix) = 0;
            if (depthMap.at<unsigned short>(iy, ix) < minDepth)
                depthMap.at<unsigned short>(iy, ix) = 0;
        }
    }
    int cx = width/2;
    int cy = height/2;
    int localMinDepth = 5000;
    int sx = 0;
    int sy = 0;
    for(int iy=cy-cHeight; iy < cy+cHeight; ++iy)
    {
        for(int ix=cx-cWidth; ix < cx+cWidth; ++ix)
        {
            unsigned short d = depthMap.at<unsigned short>(iy, ix);
            if (d == 0)
                continue;
            if (d < localMinDepth)
            {
                localMinDepth = d;
                sx = ix;
                sy = iy;
            }
        }
    }
    int label = 1;
    for (int iy = 0; iy < height; ++iy)
    {
        for (int ix = 0; ix < width; ++ix)
        {
            if (depthMap.at<unsigned short>(iy, ix) == 0)
                labelMat.at<unsigned short>(iy, ix) = label;
        }
    }
    ++label;
    std::vector<cv::Point> cluster;
    cluster.resize(1);
    cluster[0].x = sx;
    cluster[0].y = sy;
    int count = 0;
    while(cluster.size() > count)
    {
        //labelMat.at<unsigned short>(cluster[count].y, cluster[count].x) = label;
        std::vector<cv::Point> tmp;
        if ((cluster[count].x - 1) > 0)
        {
            cv::Point p = cluster[count];
            p.x--;
            if(labelMat.at<unsigned short>(p.y,p.x)==0)
                tmp.push_back(p);
        }
        if ((cluster[count].x + 1) < width)
        {
            cv::Point p = cluster[count];
            p.x++;
            if(labelMat.at<unsigned short>(p.y,p.x)==0)
                tmp.push_back(p);
        }
        if ((cluster[count].y - 1) > 0)
        {
            cv::Point p = cluster[count];
            p.y--;
            if(labelMat.at<unsigned short>(p.y,p.x)==0)
                tmp.push_back(p);
        }
        if ((cluster[count].y + 1) < height)
        {
            cv::Point p = cluster[count];
            p.y++;
            if(labelMat.at<unsigned short>(p.y,p.x)==0)
                tmp.push_back(p);
        }
        int depth = depthMap.at<unsigned short>(cluster[count].y, cluster[count].x);
        for (unsigned long i = 0; i < tmp.size(); ++i)
        {
            int diff = abs(depth - (int)depthMap.at<unsigned short>(tmp[i].y, tmp[i].x));
            if (diff < tolerance)
            {
                labelMat.at<unsigned short>(tmp[i].y, tmp[i].x) = label;
                cluster.push_back(tmp[i]);
            }
        }
        ++count;
    }
    for(int iy=0;iy < height; ++iy)
    {
        for(int ix=0; ix < width; ++ix)
        {
            if(labelMat.at<unsigned short>(iy,ix) < 2)
                depthMap.at<unsigned short>(iy,ix) = 0;
        }
    }
}

/*
Computes normals at every pixel.
*/
void computeNormal(cv::Mat &depthMat, cv::Mat &normalMat)
{
    cv::Mat normals(depthMat.size(), CV_32FC3);
    for (int iy = 1; iy < depthMat.rows - 1; ++iy)
    {
        for (int ix = 1; ix < depthMat.cols - 1; ++ix)
        {
            if (depthMat.at<unsigned short>(iy, ix) == 0)
                continue;
            Eigen::Vector3f normal;
            normal[0] = (float)(depthMat.at<unsigned short>(iy, ix - 1) - depthMat.at<unsigned short>(iy, ix + 1)) / 2;
            normal[1] = (float)(depthMat.at<unsigned short>(iy - 1, ix) - depthMat.at<unsigned short>(iy + 1, ix)) / 2;
            normal[2] = 1.0f;
            normal.normalized();
            normals.at<cv::Vec3f>(iy, ix) = cv::Vec3f(normal[0], normal[1], normal[2]);
        }
    }
    normals.copyTo(normalMat);
}
/*
    Replaces depth with average of neighbours's depths.
    pixel whose depth is zero or difference with interest pixel is more than 30 is excluded from averaging.
*/
void smoothingDepthMap(cv::Mat &depthMap, int Nx, int Ny)
{
    cv::Mat normalMat;
    computeNormal(depthMap, normalMat);
    cv::Mat tmp(depthMap.size(), CV_16UC1, cv::Scalar(0));
    for (int iy = Ny; iy < depthMap.rows-Ny; ++iy)
    {
        for (int ix = Nx; ix < depthMap.cols-Nx; ++ix)
        {
            float d1 = depthMap.at<unsigned short>(iy, ix);
            if ( d1 == 0)
                continue;
            float depth = 0.0f;
            int count = 0;
            cv::Vec3f normal0 = normalMat.at<cv::Vec3f>(iy, ix);
            for (int y = iy - Ny; y < iy + Ny; ++y)
            {
                for (int x = ix - Nx; x < ix + Nx; ++x)
                {
                    cv::Vec3f normal1 = normalMat.at<cv::Vec3f>(y, x);
                    if (fabs(normal0.dot(normal1)) < 0.5f)
                        continue;
                    float d2 = depthMap.at<unsigned short>(y, x);
                    if (d2 == 0)
                        continue;
                    float diff = fabs(d1 - d2);
                    if (diff > 30)
                        continue;
                    depth = (count*depth + (float)d2) / (count + 1);
                    ++count;
                }
            }
            tmp.at<unsigned short>(iy, ix) = (unsigned short)(depth + 0.5f);
        }
    }
    tmp.copyTo(depthMap);
}
/*
choose the biggest segment as foreground object.
fills holes of area for foreground object and smooth it.
*/
void extractForeground(cv::Mat &depthMat, float dist_floor, float tolerance)
{
    for (int iy = 0; iy < depthMat.rows; ++iy)
    {
        for (int ix = 0; ix < depthMat.cols; ++ix)
        {
            float depth = depthMat.at<unsigned short>(iy, ix);
            if (depth > dist_floor)
            {
                depthMat.at<unsigned short>(iy, ix) = 0;
            }
        }
    }
    cv::Mat labelOut;
    std::vector<PAIR> labels;
    segmentDepthmap(depthMat, dist_floor, 100, tolerance, labelOut, labels);
    for (int iy = 0; iy < depthMat.rows; ++iy)
    {
        for (int ix = 0; ix < depthMat.cols; ++ix)
        {
            if (labelOut.at<unsigned short>(iy, ix) != labels[0].first)
            {
                depthMat.at<unsigned short>(iy, ix) = 0;
            }
        }
    }
    //fillingHoles(depthMat, 3);
    _fillingHoles(depthMat, 800);
    for (int i = 0; i < 10; ++i)
    {
        smoothingDepthMap(depthMat, 3, 3);
    }
}
/*
load calibration data from xml file.
*/
void getCalibration(std::string &calibFilePath, Eigen::Vector4f &intr_vec, Eigen::Affine3f &pose)
{
    cv::FileStorage fs(calibFilePath, cv::FileStorage::READ);
    cv::Mat cam, R, T;
    if (fs.isOpened())
    {
        fs["cameraMatrix"] >> cam;
        fs["Rotation"] >> R;
        fs["Translation"] >> T;
        fs.release();
    }
    double camfx = cam.at<double>(0, 0);
    double camfy = cam.at<double>(1, 1);
    double camu = cam.at<double>(0, 2);
    double camv = cam.at<double>(1, 2);
    intr_vec[0] = camfx;
    intr_vec[1] = camfy;
    intr_vec[2] = camu;
    intr_vec[3] = camv;
    Eigen::Matrix3f rotMat;
    Eigen::Vector3f tvec;
    rotMat << R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2),
              R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2),
              R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2);
    tvec << T.at<double>(0,0)/1000, T.at<double>(1,0)/1000, T.at<double>(2,0)/1000;
    pose.linear() = rotMat;
    pose.translation() = 1000*tvec;
}
/*
generates color point cloud from depth map, color image and intrinsic parameters.
*/
void fromDepthToColorPointCloud(cv::Mat &depthMat, cv::Mat &colorMat, float fx, float fy, float cx, float cy,
                                pcl::PointCloud<pcl::PointXYZRGB>::Ptr pCloud)
{
    int width = depthMat.cols;
    int height = depthMat.rows;
    for(int iy=0; iy < height; ++iy)
    {
        for(int ix=0; ix < width; ++ix)
        {
            double d = depthMat.at<unsigned short>(iy,ix);//depthRow[ix];
            if(d > 1500)
                continue;
            if(d == 0xffff || d == 0)
            {
            }
            else
            {
                double x =d*(ix-cx)/fx;
                double y = d*(iy-cy)/fy;
                pcl::PointXYZRGB p;
                p.x = (double)x/1000;
                p.y = (double)y/1000;
                p.z = (double)d/1000;
                p.r = colorMat.at<cv::Vec3b>(iy,ix)[2];
                p.g = colorMat.at<cv::Vec3b>(iy,ix)[1];
                p.b = colorMat.at<cv::Vec3b>(iy,ix)[0];
                pCloud->push_back(p);
            }
        }
    }
}
/*
generates color point cloud from dataset.
*/
void fromDatasetToPointCloud(std::string &root, float maxDepth, float minDepth, float smooth_tolerance, int filtering,
    pcl::PointCloud<pcl::PointXYZRGB> &cloud_merged)
{
    std::vector<std::string> view_paths;
    view_paths.resize(3);
    view_paths[0] = root + "/view00";
    view_paths[1] = root + "/view01";
    view_paths[2] = root + "/view02";
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_final(new pcl::PointCloud<pcl::PointXYZRGB>());
    QElapsedTimer timer;
    timer.start();
    for (unsigned long i = 0; i < view_paths.size(); ++i)
    {
        std::string depth_path = view_paths[i] + "/depth_Image001.png";
        std::string color_path = view_paths[i] + "/gray_Image001.png";
        std::string calib_path = view_paths[i] + "/calibration.xml";
        Eigen::Vector4f intr_vec;
        Eigen::Affine3f pose;
        cv::Mat depthMat = cv::imread(depth_path, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
        cv::Mat colorMat = cv::imread(color_path, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
        //if (filtering == 1)
        //    extractForeground(depthMat, maxDepth, smooth_tolerance);
        extractObject(depthMat, maxDepth, minDepth, 100, 80, smooth_tolerance);
        getCalibration(calib_path, intr_vec, pose);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>());
        fromDepthToColorPointCloud(depthMat, colorMat, intr_vec[0], intr_vec[1], intr_vec[2], intr_vec[3], cloud_ptr);
        pcl::transformPointCloud(*cloud_ptr, *cloud_ptr, pose);
        *cloud_final += *cloud_ptr;
    }
    cloud_merged = *cloud_final;
    qint64 elapsedTime = timer.elapsed();
    std::cout << "elapsed time for constructing point cloud:" << (float)elapsedTime <<"ms"<< std::endl;
}

void scalingPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ptr, float s)
{
    for(unsigned long i=0; i < cloud_ptr->points.size(); ++i)
    {
        cloud_ptr->points[i].x *= s;
        cloud_ptr->points[i].y *= s;
        cloud_ptr->points[i].z *= s;
    }
}

Eigen::Vector3f getCenterOfPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ptr, pcl::PointXYZ &center)
{
    Eigen::Vector3f vCenter(0.0f,0.0f,0.0f);
    for(unsigned long i=0; i < cloud_ptr->points.size(); ++i)
    {
        vCenter = (i*vCenter + cloud_ptr->points[i].getVector3fMap())/(i+1);
    }
    center.x = vCenter[0];
    center.y = vCenter[1];
    center.z = vCenter[2];
    return vCenter;
}

Eigen::Vector3f getCenter(std::vector<Eigen::Vector3f> &vertices)
{
    Eigen::Vector3f vCenter(0.0f,0.0f,0.0f);
    for(unsigned long i=0; i < vertices.size(); ++i)
    {
        vCenter = (i*vCenter + vertices[i])/(i+1);
    }
    return vCenter;
}
bool isInClippedArea(Eigen::Vector3f &v, std::vector<Eigen::Hyperplane<float, 3>> &clipPlanes)
{
    for(unsigned long i=0; i < clipPlanes.size(); ++i)
    {
        //if(clipPlanes[4].signedDistance(v) > 0.0f)
        if(clipPlanes[i].signedDistance(v) > 0.0f)
            return false;
    }
    return true;
}

void extractColors(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ptr, std::vector<Eigen::Vector3f> &colors)
{
    for(unsigned long i=0; i < cloud_ptr->points.size(); ++i)
    {
        Eigen::Vector3f c;
        c[0] = (float)cloud_ptr->points[i].r/255;
        c[1] = (float)cloud_ptr->points[i].g/255;
        c[2] = (float)cloud_ptr->points[i].b/255;
        colors.push_back(c);
    }
}

boost::shared_ptr<pcl::PolygonMesh> convertToMesh(const pcl::gpu::DeviceArray<pcl::PointXYZ>& triangles,
                                                  std::vector<Eigen::Hyperplane<float, 3>> &clipPlanes)
{
  if (triangles.empty())
      return boost::shared_ptr<pcl::PolygonMesh>();
  pcl::PointCloud<pcl::PointXYZ> cloud, cloudClipped;
  cloud.width  = (int)triangles.size();
  cloud.height = 1;
  triangles.download(cloud.points);
  pcl::PointXYZ points[3];
  Eigen::Vector3f vecs[3], vDist;
  int nFaces = cloud.points.size()/3;
  std::vector<Eigen::Vector3f> vertices;
  boost::shared_ptr<pcl::PolygonMesh> mesh_ptr( new pcl::PolygonMesh() );
  for(int t=0; t < nFaces; ++t)
  {
      int l = 3*t;
      bool flg = true;
      for(int i=0; i < 3; ++i)
      {
          Eigen::Vector3f v = cloud.points[l].getVector3fMap();
          /*vDist = v - vCenter;
          if(vDist.norm() > fRadius)
          {
              flg = false;
              break;
          }
          float dist = plane.signedDistance(v);
          if(dist < 0.0f)
          {
              flg = false;
              break;
          }*/
          if(isInClippedArea(v, clipPlanes)==false)
          {
              flg = false;
              break;
          }
          points[i].x = cloud.points[l].x;
          points[i].y = cloud.points[l].y;
          points[i].z = cloud.points[l].z;
          ++l;
      }
      if(flg==false)
          continue;
      for(int i=0; i < 3; ++i)
      {
         cloudClipped.points.push_back(points[i]);
      }
  }
  //16 January 2020
  std::vector<int> indices;
  indices.resize(cloudClipped.points.size());
  for(unsigned long i=0; i < cloudClipped.points.size(); ++i)
  {
      indices[i] = -1;
  }
  int nb_points = 0;
  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr1(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::copyPointCloud(cloudClipped, *cloud_ptr1);
  kdtree.setInputCloud (cloud_ptr1);
  pcl::PointCloud<pcl::PointXYZ> cloud_final;
  for(unsigned long i=0; i < cloudClipped.points.size(); ++i)
  {
      if(indices[i] > -1)
          continue;
      pcl::PointXYZ p1 = cloudClipped.points[i];
      std::vector<int> k_indices;
      std::vector<float> k_dists;
      kdtree.nearestKSearch(p1, 50, k_indices, k_dists);
      for(unsigned long j=0; j < k_indices.size(); ++j)
      {
         if(k_dists[j]==0)
         {
            indices[k_indices[j]] = nb_points;
         }
      }
      cloud_final.points.push_back(p1);
      ++nb_points;
  }
  pcl::toPCLPointCloud2(cloud_final, mesh_ptr->cloud);
  unsigned long nb_facets = indices.size()/3;
  for(unsigned long i=0; i < nb_facets; ++i)
  {
      if(indices[3*i]==indices[3*i+1])
          continue;
      if(indices[3*i]==indices[3*i+2])
          continue;
      if(indices[3*i+1]==indices[3*i+2])
          continue;
      pcl::Vertices vertices;
      vertices.vertices.resize(3);
      vertices.vertices[0] = indices[3*i];
      vertices.vertices[1] = indices[3*i+2];
      vertices.vertices[2] = indices[3*i+1];
      mesh_ptr->polygons.push_back(vertices);
  }
  return mesh_ptr;
}

boost::shared_ptr<pcl::PolygonMesh> convertToMesh(const pcl::gpu::DeviceArray<pcl::PointXYZ>& triangles)
{
  if (triangles.empty())
      return boost::shared_ptr<pcl::PolygonMesh>();
  pcl::PointCloud<pcl::PointXYZ> cloud;
  cloud.width  = (int)triangles.size();
  cloud.height = 1;
  triangles.download(cloud.points);
  pcl::PointXYZ points[3];
  Eigen::Vector3f vecs[3], vDist;
  unsigned long nFaces = cloud.points.size()/3;
  std::vector<int> indices;
  indices.resize(3*nFaces);
  for(unsigned long i=0; i < cloud.points.size(); ++i)
  {
      indices[i] = -1;
  }
  int nb_points = 0;
  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr1(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::copyPointCloud(cloud, *cloud_ptr1);
  kdtree.setInputCloud (cloud_ptr1);
  pcl::PointCloud<pcl::PointXYZ> cloud_final;
  for(unsigned long i=0; i < cloud.points.size(); ++i)
  {
      if(indices[i] > -1)
          continue;
      pcl::PointXYZ p1 = cloud.points[i];
      std::vector<int> k_indices;
      std::vector<float> k_dists;
      kdtree.nearestKSearch(p1, 50, k_indices, k_dists);
      for(unsigned long j=0; j < k_indices.size(); ++j)
      {
         if(k_dists[j]==0)
         {
            indices[k_indices[j]] = nb_points;
         }
      }
      cloud_final.points.push_back(p1);
      ++nb_points;
  }
  boost::shared_ptr<pcl::PolygonMesh> mesh_ptr( new pcl::PolygonMesh() );
  pcl::toPCLPointCloud2(cloud_final, mesh_ptr->cloud);
  unsigned long nb_facets = indices.size()/3;
  for(unsigned long i=0; i < nb_facets; ++i)
  {
      if(indices[3*i]==indices[3*i+1])
          continue;
      if(indices[3*i]==indices[3*i+2])
          continue;
      if(indices[3*i+1]==indices[3*i+2])
          continue;
      pcl::Vertices vertices;
      vertices.vertices.resize(3);
      vertices.vertices[0] = indices[3*i];
      vertices.vertices[1] = indices[3*i+2];
      vertices.vertices[2] = indices[3*i+1];
      mesh_ptr->polygons.push_back(vertices);
  }
  return mesh_ptr;
}

void decimateMesh(pcl::PolygonMesh& mesh_in, pcl::PolygonMesh& out_mesh)
{
    //PolygonMesh mesh_refined;
    //creatingTriangleMesh(mesh_in, mesh_refined);
    //refiningMesh(mesh_in, mesh_refined);
    pcl::PointCloud<pcl::PointXYZ> cloud_clipped;
    pcl::fromPCLPointCloud2(mesh_in.cloud, cloud_clipped);
    std::vector<Eigen::Vector3f> vertices_clipped;
    for(unsigned long i=0; i < cloud_clipped.size(); ++i)
    {
        Eigen::Vector3f v = cloud_clipped.at(i).getVector3fMap();
        vertices_clipped.push_back(v);
    }
    std::vector<int> indices_clipped;
    //for(int i=0; i < mesh_refined.polygons.size(); ++i)
    for(unsigned long i=0; i < mesh_in.polygons.size(); ++i)
    {
        pcl::Vertices vertices = mesh_in.polygons[i];//mesh_refined.polygons[i];
        int id = vertices.vertices[0];
        indices_clipped.push_back(id);
        id = vertices.vertices[1];
        indices_clipped.push_back(id);
        id = vertices.vertices[2];
        indices_clipped.push_back(id);
    }
    Simplify::importMesh(vertices_clipped, indices_clipped);
    int target_count = mesh_in.polygons.size()/5;//mesh_refined.polygons.size()/5;//10
    double agressiveness = 7;
    Simplify::simplify_mesh(target_count, agressiveness, true);
    std::vector<Eigen::Vector3f> vertices_decimated;
    std::vector<int> indices_decimated;
    Simplify::exportMesh(vertices_decimated, indices_decimated);
    pcl::PointCloud<pcl::PointXYZ> cloud_decimated;
    for(unsigned long i=0; i < vertices_decimated.size(); ++i)
    {
        pcl::PointXYZ p;
        p.x = vertices_decimated[i][0];
        p.y = vertices_decimated[i][1];
        p.z = vertices_decimated[i][2];
        cloud_decimated.push_back(p);
    }
    pcl::toPCLPointCloud2(cloud_decimated, out_mesh.cloud);
    for(unsigned long i=0; i < indices_decimated.size()/3; ++i)
    {
        pcl::Vertices vertices;
        vertices.vertices.resize(3);
        vertices.vertices[0] = indices_decimated[3*i];
        vertices.vertices[1] = indices_decimated[3*i+1];
        vertices.vertices[2] = indices_decimated[3*i+2];
        out_mesh.polygons.push_back(vertices);
    }
}

void writePolygonMeshFile (const pcl::PolygonMesh& mesh, std::string &meshFileName)
{
    pcl::io::savePLYFile(meshFileName, mesh);
}

inline bool check_path (const std::string& name) {
    struct stat buffer;
    return (stat (name.c_str(), &buffer) == 0);
}

void getDataPaths(std::string &root, std::vector<std::string> &depthPaths, std::vector<std::string> &colorPaths,
                  std::vector<std::string> &backPaths, std::vector<std::string> &calibPaths)
{
    char buf[256];
    int count = 0;
    while(1)
    {
       sprintf(buf, "/view%0.2d", count);
       std::string depthPath = root + buf + "/depth_Image001.png";
       if(!check_path(depthPath))
           break;
       depthPaths.push_back(depthPath);
       std::string colorPath = root + buf + "/gray_Image001.png";
       if(!check_path(colorPath))
           break;
       colorPaths.push_back(colorPath);
       std::string backPath = root + buf + "/gray_BackgroundImg.png";
       if(!check_path(backPath))
           break;
       backPaths.push_back(backPath);
       std::string calibPath = root + buf + "/calibration.xml";
       if(!check_path(calibPath))
           break;
       calibPaths.push_back(calibPath);
       ++count;
    }
}

void mkdir(std::string &dir_name)
{
    QString qdir_name = QString::fromUtf8(dir_name.c_str());
    QDir dir;
    dir.mkdir(qdir_name);
}

void fromDepthToPointCloud(cv::Mat &depthMat, Eigen::Vector4f &intr_vec, int maxDepth, pcl::PointCloud<pcl::PointXYZ>::Ptr pCloud)
{
    float fx = intr_vec[0];
    float fy = intr_vec[1];
    float cx = intr_vec[2];
    float cy = intr_vec[3];
    int width = depthMat.cols;
    int height = depthMat.rows;
    for(int iy=0; iy < height; ++iy)
    {
        for(int ix=0; ix < width; ++ix)
        {
            double d = depthMat.at<unsigned short>(iy,ix);//depthRow[ix];
            if(d > maxDepth)
                continue;
            if(d == 0xffff || d == 0)
            {
            }
            else
            {
                double x =d*(ix-cx)/fx;
                double y = d*(iy-cy)/fy;
                pcl::PointXYZ p;
                p.x = (double)x/1000;
                p.y = (double)y/1000;
                p.z = (double)d/1000;
                pCloud->push_back(p);
            }
        }
    }
}

pcl::visualization::PCLVisualizer::Ptr createViewer(Eigen::Affine3f &pose, std::string &viewerName)
{
    pcl::visualization::PCLVisualizer::Ptr viewer = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer(viewerName));
    viewer->setBackgroundColor(0, 80, 200);
    viewer->addCoordinateSystem(1.0,"global");
    viewer->initCameraParameters();
    viewer->setPosition(0,0);
    viewer->setSize(640,480);
    viewer->setCameraClipDistances(0.1, 10.0);
    Eigen::Vector3f vEyePos = pose.translation();//pose*Eigen::Vector3f(0.0f,0.0f,0.0f);
    Eigen::Vector3f vLookAt = vEyePos + pose.rotation()*Eigen::Vector3f(0.0f, 0.0f, 1.0f);
    Eigen::Vector3f vUp = pose.rotation()*Eigen::Vector3f(0.0,-1.0f,0.0f);
    viewer->setCameraPosition(vEyePos[0], vEyePos[1], vEyePos[2], vLookAt[0], vLookAt[1], vLookAt[2], vUp[0], vUp[1], vUp[2]);
    return viewer;
}

void showPointCloud(Eigen::Affine3f &camPose, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ptr)
{
    std::string caption = "cloud_viewer";
    pcl::visualization::PCLVisualizer::Ptr viewer = createViewer(camPose, caption);
    viewer->removeAllPointClouds();
    viewer->removeAllShapes();
    viewer->addPointCloud<pcl::PointXYZRGB>(cloud_ptr);
    while(1)
    {
        viewer->spinOnce((false));
    }
}

void centeringOfCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ptr)
{
    pcl::PointXYZ center;
   getCenterOfPointCloud(cloud_ptr, center);
   for(unsigned long i=0; i < cloud_ptr->points.size(); ++i)
   {
      cloud_ptr->points[i].x -= center.x;
      cloud_ptr->points[i].y -= center.y;
      cloud_ptr->points[i].z -= center.z;
   }
}

void centeringOfMesh(std::vector<std::vector<Eigen::Vector3f>> &vertices_parts)
{
    Eigen::Vector3f vCenter(0.0f,0.0f,0.0f);
    int count = 0;
    for(unsigned long i=0; i < vertices_parts.size(); ++i)
    {
        for(unsigned long j=0; j < vertices_parts[i].size(); ++j)
        {
            vCenter = (count*vCenter + vertices_parts[i][j])/(count+1);
        }
    }
    for(unsigned long i=0; i < vertices_parts.size(); ++i)
    {
        for(unsigned long j=0; j < vertices_parts[i].size(); ++j)
        {
            vertices_parts[i][j] -= vCenter;
        }
    }
}

Eigen::Vector3f centering(std::vector<Eigen::Vector3f> &vertices)
{
    Eigen::Vector3f vCenter(0.0f,0.0f,0.0f);
    int count = 0;
    for(unsigned long i=0; i < vertices.size(); ++i)
    {
        vCenter = (count*vCenter + vertices[i])/(count+1);
        ++count;
    }
    for(unsigned long i=0; i < vertices.size(); ++i)
    {
        vertices[i] -= vCenter;
    }
    return vCenter;
}

void downSamplingCloud(float leaf_size, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ptr)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_tmp(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::VoxelGrid<pcl::PointXYZRGB> grid;
    grid.setLeafSize(leaf_size, leaf_size, leaf_size);
    grid.setInputCloud(cloud_ptr);
    grid.filter(*cloud_tmp);
    cloud_ptr->swap(*cloud_tmp);
}
/*
segment depthmap in to smooth regions.
smooth regions are extracted by tolerance.
sorts them by their sizes.
depthMap: depth image with object
tolerance: threshold to ignore difference between two adjacent pixels's depths.
lableOut: presents segments
labels: id and size of each segment
*/
void segmentDepthmap(cv::Mat &depthMap, int tolerance, cv::Mat &labelOut, std::vector<std::pair<int, int>> &labels)
{
    int width = depthMap.cols;
    int height = depthMap.rows;
    cv::Mat labelMat(depthMap.size(), CV_16UC1, cv::Scalar(0));
    int label = 1;
    for (int iy = 0; iy < height; ++iy)
    {
        for (int ix = 0; ix < width; ++ix)
        {
            if (depthMap.at<unsigned short>(iy, ix) == 0)
                labelMat.at<unsigned short>(iy, ix) = label;
        }
    }
    for (int iy = 1; iy < height - 1; ++iy)
    {
        for (int ix = 1; ix < width - 1; ++ix)
        {
            if (labelMat.at<unsigned short>(iy, ix) > 0)
                continue;
            ++label;
            std::vector<cv::Point> cluster;
            cluster.resize(1);
            cluster[0].x = ix; cluster[0].y = iy;
            labelMat.at<unsigned short>(iy, ix) = label;
            unsigned long count = 0;
            std::pair<int, int> pair;
            while (count < cluster.size())//extract region connected
            {
                std::vector<cv::Point> tmp;
                if ((cluster[count].x - 1) > 0)
                {
                    cv::Point p = cluster[count];
                    p.x--;
                    tmp.push_back(p);
                }
                if ((cluster[count].x + 1) < width)
                {
                    cv::Point p = cluster[count];
                    p.x++;
                    tmp.push_back(p);
                }
                if ((cluster[count].y - 1) > 0)
                {
                    cv::Point p = cluster[count];
                    p.y--;
                    tmp.push_back(p);
                }
                if ((cluster[count].y + 1) < height)
                {
                    cv::Point p = cluster[count];
                    p.y++;
                    tmp.push_back(p);
                }
                int depth = depthMap.at<unsigned short>(cluster[count].y, cluster[count].x);
                for (unsigned long i = 0; i < tmp.size(); ++i)
                {
                    int diff = abs(depth - (int)depthMap.at<unsigned short>(tmp[i].y, tmp[i].x));
                    if (diff < tolerance && labelMat.at<unsigned short>(tmp[i].y, tmp[i].x) == 0)
                    {
                        labelMat.at<unsigned short>(tmp[i].y, tmp[i].x) = label;
                        cluster.push_back(tmp[i]);
                    }
                }
                ++count;
            }
            pair.first = label;
            pair.second = count;
            labels.push_back(pair);
        }
    }
    //sort labels
    bool loop = true;
    while (loop == true)
    {
        loop = false;
        for (unsigned long i = 0; i < labels.size()-1; ++i)
        {
            std::pair<int, int> p;
            p = labels[i];
            if (labels[i].second < labels[i + 1].second)
            {
                labels[i] = labels[i + 1];
                labels[i + 1] = p;
                loop = true;
            }
        }
    }
    labelMat.copyTo(labelOut);
}
void fromDepthToPointCloud(cv::Mat &depthMat, Eigen::Vector4f &intr_vec, pcl::PointCloud<pcl::PointXYZ>::Ptr pCloud)
{
    float fx = intr_vec[0];
    float fy = intr_vec[1];
    float cx = intr_vec[2];
    float cy = intr_vec[3];
    int width = depthMat.cols;
    int height = depthMat.rows;
    for(int iy=0; iy < height; ++iy)
    {
        auto depthRow = depthMat.ptr<uint16_t>(iy);
        for(int ix=0; ix < width; ++ix)
        {
            //auto d = depthMat.at<unsigned short>(iy,ix);//depthRow[ix];
            auto d = depthRow[ix];
            if(d > 1500)
                continue;
            if(d == 0xffff || d == 0)
            {
            }
            else
            {
                auto x =d*(ix-cx)/fx;
                auto y = d*(iy-cy)/fy;
                pcl::PointXYZ p;
                p.x = (double)x/1000;
                p.y = (double)y/1000;
                p.z = (double)d/1000;
                pCloud->push_back(p);
            }
        }
    }
}
/*
created at December 23, 2020
*/
void estimateVirtualFenceParams(cv::Mat &depthMat, Eigen::Vector4f &intr_vec, float &minY, float &maxY)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr pCloud(new pcl::PointCloud<pcl::PointXYZ>());
    fromDepthToPointCloud(depthMat, intr_vec, pCloud);
    minY = 1000.0f;
    maxY = -1000.0f;
    for(unsigned long i=0; i < pCloud->points.size(); ++i)
    {
        if(pCloud->points[i].y < minY)
            minY = pCloud->points[i].y;
        if(pCloud->points[i].y > maxY)
            maxY = pCloud->points[i].y;
    }
}

float estimateHeightOfTopCam(cv::Mat &depthMat)
{
    cv::Mat labelMat;
    std::vector<std::pair<int, int>> labels;
    segmentDepthmap(depthMat, 10, labelMat, labels);
    float height = 0.0f;
    int count = 0;
    for(int iy = 0; iy < depthMat.rows; ++iy)
    {
        for(int ix = 0; ix < depthMat.cols; ++ix)
        {
            if(labelMat.at<unsigned short>(iy, ix)==labels[0].first)
            {
                float d = (float)depthMat.at<unsigned short>(iy, ix)/1000;
                height = (count*height + d)/(count + 1);
                ++count;
            }
        }
    }
    return height;
}

void removeFloor(cv::Mat &depthMat, unsigned short threshold, cv::Mat &colorMat)
{
    for(int iy = 0; iy < depthMat.rows; ++iy)
    {
        for(int ix = 0; ix < depthMat.cols; ++ix)
        {
            if(depthMat.at<unsigned short>(iy, ix) > threshold)
            {
                depthMat.at<unsigned short>(iy, ix) = 0;
                colorMat.at< cv::Vec3b>(iy, ix) = cv::Vec3b(0,0,0);
            }
        }
    }
}

void estimateParams(cv::Mat &depthMat, cv::Mat &colorMat, Eigen::Vector4f &intr_vec, float &camHeight, float &minY, float &maxY)
{
    camHeight = estimateHeightOfTopCam(depthMat);
    unsigned short threshold = 1000*(camHeight-0.02f);
    removeFloor(depthMat, threshold, colorMat);
    estimateVirtualFenceParams(depthMat, intr_vec, minY, maxY);
    std::cout << "camHeight:" << camHeight << std::endl;
    std::cout << "minY:" << minY << std::endl;
    std::cout << "maxY:" << maxY << std::endl;
}

void writeParams(std::string &fileName, float &camHeight, float &minY, float &maxY)
{
    cv::FileStorage fs(fileName, cv::FileStorage::WRITE);
    cv::Mat paramMat(cv::Size(3,3), CV_32FC1, cv::Scalar(0));
    paramMat.at<float>(0,0) = camHeight;
    paramMat.at<float>(1,1) = minY;
    paramMat.at<float>(2,2) = maxY;
    fs << "paramMat" << paramMat;
    fs.release();
}

void readParams(std::string &fileName, float &camHeight, float &minY, float &maxY)
{
    cv::FileStorage fs(fileName, cv::FileStorage::READ);
    cv::Mat paramMat;
    fs["paramMat"] >> paramMat;
    fs.release();
    camHeight = paramMat.at<float>(0,0);
    minY = paramMat.at<float>(1,1);
    maxY = paramMat.at<float>(2,2);
}

void extractingObjects(std::vector<cv::Mat> &depthMats, int maxDepth, int minDepth, int smooth_tolerance)
{
    for(unsigned long i=0; i < depthMats.size(); ++i)
    {
        extractObject(depthMats[i], maxDepth, minDepth, 100, 100, smooth_tolerance);
    }
}

void getData(std::string &dataset, std::vector<cv::Mat> &depthMats, std::vector<cv::Mat> &colorMats,
             std::vector<cv::Mat> &backMats, std::vector<Eigen::Vector4f> &intr_vecs, std::vector<Eigen::Affine3f> &poses)
{
    std::vector<std::string> depthPaths, colorPaths, backPaths, calibPaths;
    getDataPaths(dataset, depthPaths, colorPaths, backPaths, calibPaths);
    std::cout << "depth path count:" << depthPaths.size() << std::endl;
    for(unsigned long i=0; i < depthPaths.size(); ++i)
    {
       auto depthMat = cv::imread(depthPaths[i], cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
       auto colorMat = cv::imread(colorPaths[i], cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
       auto backMat = cv::imread(backPaths[i], cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
       depthMats.push_back(depthMat);
       colorMats.push_back(colorMat);
       backMats.push_back(backMat);
       Eigen::Vector4f intr_vec;
       Eigen::Affine3f pose;
       getCalibration(calibPaths[i], intr_vec, pose);
       intr_vecs.push_back(intr_vec);
       poses.push_back(pose);
    }
}

void constructPointCloud(std::vector<cv::Mat> &depthMats, std::vector<cv::Mat> &colorMats,
                         std::vector<Eigen::Vector4f> &intr_vecs, std::vector<Eigen::Affine3f> &poses,
                         pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_out)
{
    for(unsigned long i=0; i < depthMats.size(); ++i)
    {
        float fx = intr_vecs[i][0];
        float fy = intr_vecs[i][1];
        float cx = intr_vecs[i][2];
        float cy = intr_vecs[i][3];
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>());
        fromDepthToColorPointCloud(depthMats[i], colorMats[i], fx, fy, cx, cy, cloud_ptr);
        pcl::transformPointCloud(*cloud_ptr, *cloud_ptr, poses[i]);
        *cloud_out += *cloud_ptr;
    }
//    downSamplingCloud(0.002f, cloud_out);
}

void downsamplingData(std::vector<cv::Mat> &depthMats, std::vector<cv::Mat> &colorMats, std::vector<Eigen::Vector4f> &intr_vecs, float sample,
                      std::vector<cv::Mat> &depthMats_out, std::vector<cv::Mat> &colorMats_out, std::vector<Eigen::Vector4f> &intr_vecs_out)
{
    int width = depthMats[0].cols/sample;
    int height = depthMats[0].rows/sample;
    for(unsigned long i=0; i < depthMats.size(); ++i)
    {
       cv::resize(depthMats[i], depthMats_out[i], cv::Size(width, height));
       cv::resize(colorMats[i], colorMats_out[i], cv::Size(width, height));
       intr_vecs_out[i] = intr_vecs[i]/sample;
    }
}
/*
created at Febrary 09, 2021
*/
void extractForeground(cv::Mat &depthMat, cv::Mat &colorMat)
{
    for(int iy=0; iy < depthMat.rows; ++iy)
    {
        for(int ix=0; ix < depthMat.cols; ++ix)
        {
            if(depthMat.at<unsigned short>(iy,ix)==0)
            {
                colorMat.at<cv::Vec3b>(iy,ix) *= 0;
            }
        }
    }
}
/*
created at March 19, 2021
*/
void createClipPlanes(std::vector<Eigen::Affine3f> &poses, std::vector<Eigen::Hyperplane<float,3>> &planes)
{
    std::vector<Eigen::Vector3f> vertices;
    for(unsigned long i=0; i < poses.size(); ++i)
    {
        vertices.push_back(poses[i].translation());
    }
    //plane0
    Eigen::Vector3f vec0 = vertices[1] - vertices[0];
    Eigen::Vector3f vec1 = vertices[4] - vertices[0];
    Eigen::Vector3f normal = vec0.cross(vec1);
    normal.normalize();
    Eigen::Hyperplane<float, 3> plane0(normal, vertices[0]);
    planes.push_back(plane0);
    //plane1
    vec0 = vertices[2] - vertices[1];
    vec1 = vertices[5] - vertices[1];
    normal = vec0.cross(vec1);
    normal.normalize();
    Eigen::Hyperplane<float, 3> plane1(normal, vertices[1]);
    planes.push_back(plane1);
    //plane2
    vec0 = vertices[3] - vertices[2];
    vec1 = vertices[6] - vertices[2];
    normal = vec0.cross(vec1);
    normal.normalize();
    Eigen::Hyperplane<float, 3> plane2(normal, vertices[2]);
    planes.push_back(plane2);
    //plane3
    vec0 = vertices[0] - vertices[3];
    vec1 = vertices[7] - vertices[3];
    normal = vec0.cross(vec1);
    normal.normalize();
    Eigen::Hyperplane<float, 3> plane3(normal, vertices[3]);
    planes.push_back(plane3);
    //plane4
    vec0 = vertices[5] - vertices[4];
    vec1 = vertices[7] - vertices[4];
    normal = vec0.cross(vec1);
    normal.normalize();
    Eigen::Hyperplane<float, 3> plane4(normal, vertices[4]);
    planes.push_back(plane4);
}
/*
created at March 19, 2021
*/
void constructDepthmap(Eigen::Vector4f &intr_vec, Eigen::Affine3f &pose, std::vector<Eigen::Hyperplane<float,3>> &planes_in,
                       int n1, int n2, int n3, cv::Mat &depthMat, cv::Mat &colorMat)
{
    float fx, fy, cx, cy;
    fx = intr_vec[0];
    fy = intr_vec[1];
    cx = intr_vec[2];
    cy = intr_vec[3];
    std::vector<Eigen::Hyperplane<float,3>> planes;
    planes.push_back(planes_in[n1]);
    planes.push_back(planes_in[n2]);
    planes.push_back(planes_in[n3]);
    Eigen::Vector3f vEye = pose.translation();
    cv::Vec3b colors[3];
    colors[0] = cv::Vec3b(255,0,0);
    colors[1] = cv::Vec3b(0,255,0);
    colors[2] = cv::Vec3b(0,0,255);
    for(int iy=0; iy < depthMat.rows; ++iy)
    {
       for(int ix=0; ix < depthMat.cols; ++ix)
       {
            Eigen::Vector3f vTarg = Eigen::Vector3f((float)(ix-cx)/fx, (float)(iy-cy)/fy, 1.0f);
            vTarg = pose*vTarg;
            Eigen::ParametrizedLine<float,3> line = Eigen::ParametrizedLine<float,3>::Through(vEye,vTarg);
            float depth = std::numeric_limits<float>::max();
            for(int i=0; i < 3; ++i)
            {
                float intersection = line.intersection(planes[i]);
                Eigen::Vector3f intersectPos = intersection*((vTarg-vEye).normalized()) + vEye;
                /*Eigen::Vector3f vec1 = intersectPos - vEye;
                Eigen::Vector3f vec2 = vTarg - vEye;
                if(vec2.dot(vec1) < 0.0f)
                    continue;*/
                intersectPos = pose.inverse()*intersectPos;
                if(intersectPos[2] < 0.0f)
                    continue;
                intersection = fabs(intersection);
                if(depth > intersection)
                {
                    depth = intersection;
                    depthMat.at<unsigned short>(iy, ix) = 1000*depth;
                    colorMat.at<cv::Vec3b>(iy,ix) = colors[i];
                    //std::cout << depthMat.at<unsigned short>(iy,ix) << "," << std::endl;
                }
            }
       }
   }
}
/*
created at March 19, 2021
*/
void constructDepthmaps(std::vector<Eigen::Vector4f> &intr_vecs, std::vector<Eigen::Affine3f> &poses,
                        std::vector<cv::Mat> &depthMats, std::vector<cv::Mat> &colorMats)
{
    std::vector<Eigen::Hyperplane<float,3>> planes;
    createClipPlanes(poses, planes);
    constructDepthmap(intr_vecs[0], poses[0], planes, 1, 2, 4, depthMats[0], colorMats[0]);
    constructDepthmap(intr_vecs[1], poses[1], planes, 2, 3, 4, depthMats[1], colorMats[1]);
    constructDepthmap(intr_vecs[2], poses[2], planes, 0, 3, 4, depthMats[2], colorMats[2]);
    constructDepthmap(intr_vecs[3], poses[3], planes, 0, 1, 4, depthMats[3], colorMats[3]);
    //bottom
    Eigen::Vector3f upVec = poses[0].translation() - poses[4].translation();
    upVec.normalize();
    Eigen::Vector3f p = poses[4].translation() - 0.05f*upVec;
    Eigen::Hyperplane<float,3> floor_plane(upVec, p);
    planes.push_back(floor_plane);
    constructDepthmap(intr_vecs[4], poses[4], planes, 1, 2, 5, depthMats[4], colorMats[4]);
    constructDepthmap(intr_vecs[5], poses[5], planes, 2, 3, 5, depthMats[5], colorMats[5]);
    constructDepthmap(intr_vecs[6], poses[6], planes, 0, 3, 5, depthMats[6], colorMats[6]);
    constructDepthmap(intr_vecs[7], poses[7], planes, 0, 1, 5, depthMats[7], colorMats[7]);
}
/*
created at March 19, 2021
modified at April 27, 2021
*/
void extractObject(cv::Mat &depthMat, cv::Mat &backMat, float threshold, cv::Mat &colorMat)
{
    for(int iy=0; iy < depthMat.rows; ++iy)
    {
        for(int ix=0; ix < depthMat.cols; ++ix)
        {
            float depth = depthMat.at<unsigned short>(iy,ix);
            float backDepth = backMat.at<unsigned short>(iy,ix);
            if(depth==0 || backDepth==0)
            {
                depthMat.at<unsigned short>(iy,ix) = 0;
                colorMat.at<cv::Vec3b>(iy,ix) = cv::Vec3b(0,0,0);
                continue;
            }
            /*float diff = depth - backDepth;
            diff = fabs(diff);
            if(diff < threshold)
            {
                depthMat.at<unsigned short>(iy,ix) = 0;
                colorMat.at<cv::Vec3b>(iy,ix) = cv::Vec3b(0,0,0);
            }*/
            if(backDepth < (depth+threshold))
            {
                depthMat.at<unsigned short>(iy,ix) = 0;
                colorMat.at<cv::Vec3b>(iy,ix) = cv::Vec3b(0,0,0);
            }
        }
    }
}

void getDataPaths(std::string &root, std::vector<std::string> &depthPaths, std::vector<std::string> &colorPaths,
                  std::vector<std::string> &calibPaths)
{
    char buf[256];
    int count = 0;
    while(1)
    {
       sprintf(buf, "/view%0.2d", count);
       std::string depthPath = root + buf + "/depth_Image001.png";
       if(!check_path(depthPath))
           break;
       depthPaths.push_back(depthPath);
       std::string colorPath = root + buf + "/gray_Image001.png";
       if(!check_path(colorPath))
           break;
       colorPaths.push_back(colorPath);
       std::string calibPath = root + buf + "/calibration.xml";
       if(!check_path(calibPath))
           break;
       calibPaths.push_back(calibPath);
       ++count;
    }
}

void getData(std::string &dataset, std::vector<cv::Mat> &depthMats, std::vector<cv::Mat> &colorMats,
             std::vector<Eigen::Vector4f> &intr_vecs, std::vector<Eigen::Affine3f> &poses)
{
    std::vector<std::string> depthPaths, colorPaths, calibPaths;
    getDataPaths(dataset, depthPaths, colorPaths, calibPaths);
    std::cout << "depth path count:" << depthPaths.size() << std::endl;
    for(unsigned long i=0; i < depthPaths.size(); ++i)
    {
       auto depthMat = cv::imread(depthPaths[i], cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
       auto colorMat = cv::imread(colorPaths[i], cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
       depthMats.push_back(depthMat);
       colorMats.push_back(colorMat);
       Eigen::Vector4f intr_vec;
       Eigen::Affine3f pose;
       getCalibration(calibPaths[i], intr_vec, pose);
       intr_vecs.push_back(intr_vec);
       poses.push_back(pose);
    }
}

void creatingClipPlanesForBelt(float fDist, float beltWidth, std::vector<Eigen::Hyperplane<float, 3>> &planes)
{

}
/*
created at April 24, 2021
*/
void extractPlanes(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in, std::vector<Eigen::Hyperplane<float, 3>> &planes,
                   pcl::PointCloud<pcl::PointXYZRGB> &cloud_result)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_f(new pcl::PointCloud<pcl::PointXYZRGB>()), cloud_p(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::copyPointCloud(*cloud_in, *cloud_filtered);
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
     // Create the segmentation object
    pcl::SACSegmentation<pcl::PointXYZRGB> seg;
    // Optional
    seg.setOptimizeCoefficients (true);
    // Mandatory
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setDistanceThreshold (0.01);
    //create the filtering object
    pcl::ExtractIndices<pcl::PointXYZRGB> extract;
    std::vector<pcl::ModelCoefficients::Ptr> coeffs_vec;
    int nloop = 0;
    //While 30% of the original cloud is still there
    //while(cloud_filtered->points.size() > 0.3f*nr_points)
    while(nloop < 4)
    {
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
        seg.setInputCloud (cloud_filtered);
        seg.segment (*inliers, *coefficients);
        coeffs_vec.push_back(coefficients);
        if(inliers->indices.size()==0)
        {
            std::cerr << "Could not estimate a planar model" << std::endl;
        }
        //Extract the inliers
        extract.setInputCloud(cloud_filtered);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*cloud_p);
        cloud_result += *cloud_p;
        extract.setNegative(true);
        extract.filter(*cloud_f);
        cloud_filtered.swap(cloud_f);
        ++nloop;
    }
    Eigen::Vector3f zDir(0.0f, 0.0f, 1.0f);
    std::vector<Eigen::Hyperplane<float, 3>> horizen_planes, vertical_planes;
    for(unsigned long i=0; i < coeffs_vec.size(); ++i)
    {
        Eigen::Vector3f normal(coeffs_vec[i]->values[0], coeffs_vec[i]->values[1], coeffs_vec[i]->values[2]);
        //normal.normalize();
        Eigen::Hyperplane<float, 3> plane(normal, coeffs_vec[i]->values[3]);
        float dot = zDir.dot(normal.normalized());
        dot = fabs(dot);
        if(dot > 0.8f)
            horizen_planes.push_back(plane);
        if(dot < 0.3f)
        {
            planes.push_back(plane);
        }
    }
    Eigen::Hyperplane<float, 3> bottom;
    Eigen::Vector3f vEyePos(0.0f,0.0f,0.0f);
    float maxDist = 0.0f;
    for(unsigned long i=0; i < horizen_planes.size(); ++i)
    {
        float dist = horizen_planes[i].absDistance(vEyePos);
        if(dist > maxDist)
        {
            maxDist = dist;
            bottom = horizen_planes[i];
        }
    }
    planes.push_back(bottom);
}
/*
created at April 24, 2021
*/
void removeBottom(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in, float delta, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_out)
{
    std::vector<Eigen::Hyperplane<float, 3>> planes;
    pcl::PointCloud<pcl::PointXYZRGB> cloud_result;
    extractPlanes(cloud_in, planes, cloud_result);
    Eigen::Hyperplane<float, 3> bottom = planes[planes.size()-1];
    for(unsigned long i=0; i < cloud_in->points.size(); ++i)
    {
        Eigen::Vector3f v = cloud_in->points[i].getVector3fMap();
        if(bottom.absDistance(v) < delta)
            continue;
        cloud_out->points.push_back(cloud_in->points[i]);
    }
}
/*
created at April 24, 2021
modified at April 26, 2021
*/
void extractClipPlanes(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in, float belt_width, float dist_to_belt, float delta,
                       pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_out, std::vector<Eigen::Hyperplane<float, 3>> &clipPlanes)
{
    std::vector<Eigen::Hyperplane<float, 3>> planes;
    pcl::PointCloud<pcl::PointXYZRGB> cloud_result;
    extractPlanes(cloud_in, planes, cloud_result);
    Eigen::Hyperplane<float, 3> bottom = planes[planes.size()-1];//(-Eigen::Vector3f::UnitZ(), dist_to_belt); //planes[planes.size()-1];
    Eigen::Vector3f bottom_normal = Eigen::Vector3f::UnitZ();//bottom.normal();
    Eigen::Vector3f normal_y = bottom_normal.cross(Eigen::Vector3f::UnitX());
    Eigen::Vector3f vCenter = bottom.signedDistance(Eigen::Vector3f(0.0,0.0,0.0))*bottom_normal;
    Eigen::Vector3f v1 = vCenter + 0.5f*belt_width*normal_y;
    Eigen::Vector3f v2 = vCenter - 0.5f*belt_width*normal_y;
    Eigen::Hyperplane<float, 3> plane1(normal_y, v1);
    Eigen::Hyperplane<float, 3> plane2(-normal_y, v2);
    clipPlanes.push_back(plane1);
    clipPlanes.push_back(plane2);
    clipPlanes.push_back(bottom);
    for(unsigned long i=0; i < cloud_in->points.size(); ++i)
    {
        bool remove = false;
        Eigen::Vector3f v = cloud_in->points[i].getVector3fMap();
        for(int j=0; j < 3; ++j)
        {
            if(clipPlanes[j].absDistance(v) < delta)
            {
                remove = true;
                break;
            }
        }
        if(remove==true)
            continue;
        cloud_out->points.push_back(cloud_in->points[i]);
    }
}
/*
created at April 26, 2021
*/
void constructFenceDepthmap(Eigen::Vector4f &intr_vec, Eigen::Affine3f &pose, std::vector<Eigen::Hyperplane<float,3>> &planes,
                            float fence_height, cv::Mat &depthMat, cv::Mat &colorMat)
{
    float fx, fy, cx, cy;
    fx = intr_vec[0];
    fy = intr_vec[1];
    cx = intr_vec[2];
    cy = intr_vec[3];
    Eigen::Vector3f vEye = pose.translation();
    cv::Vec3b colors[3];
    colors[0] = cv::Vec3b(255,0,0);
    colors[1] = cv::Vec3b(0,255,0);
    colors[2] = cv::Vec3b(0,0,255);
    Eigen::Affine3f inv = pose.inverse();
    for(int iy=0; iy < depthMat.rows; ++iy)
    {
       for(int ix=0; ix < depthMat.cols; ++ix)
       {
            Eigen::Vector3f vTarg = Eigen::Vector3f((float)(ix-cx)/fx, (float)(iy-cy)/fy, 1.0f);
            vTarg = pose*vTarg;
            Eigen::ParametrizedLine<float,3> line = Eigen::ParametrizedLine<float,3>::Through(vEye,vTarg);
            float depth = std::numeric_limits<float>::max();
            int plane_no = 0;
            for(unsigned long i=0; i < planes.size(); ++i)
            {
                Eigen::Vector3f intersectPoint = line.intersectionPoint(planes[i]);
                Eigen::Vector3f intersectPoint_cam = inv*intersectPoint;
                //if(intersectPoint_cam[2] < 0.0f)
                  //  continue;
                if(planes[2].absDistance(intersectPoint) > fence_height)
                    continue;
                if(intersectPoint_cam[2] < depth)
                {
                    depth = intersectPoint_cam[2];
                    plane_no = i;
                }
            }
            depthMat.at<unsigned short>(iy,ix) = 1000*depth;
            colorMat.at<cv::Vec3b>(iy,ix) = colors[plane_no];
       }
   }
   std::vector<cv::Point> seg;
   seg.push_back(cv::Point(colorMat.cols/2, colorMat.rows/2));
   colorMat.at<cv::Vec3b>(seg[0].y, seg[0].x) = cv::Vec3b(0,0,0);
   int width = colorMat.cols;
   int height = colorMat.rows;
   int count = 0;
   while(count < seg.size())
   {
       cv::Point p = seg[count];
       std::vector<cv::Point> tmp;
       if((p.x+1) < width)
           tmp.push_back(cv::Point(p.x+1, p.y));
       if((p.x-1) >= 0)
       {
           tmp.push_back(cv::Point(p.x-1, p.y));
       }
       if((p.y+1) < height)
       {
           tmp.push_back(cv::Point(p.x, p.y+1));
       }
       if((p.y-1) >= 0)
       {
           tmp.push_back(cv::Point(p.x, p.y-1));
       }
       for(unsigned long i=0; i < tmp.size(); ++i)
       {
           if(colorMat.at<cv::Vec3b>(tmp[i].y, tmp[i].x)==colors[2])
           {
               seg.push_back(tmp[i]);
               colorMat.at<cv::Vec3b>(tmp[i].y, tmp[i].x) = cv::Vec3b(0,0,0);
           }
       }
       ++count;
   }
   for(int iy=0; iy < depthMat.rows; ++iy)
   {
       for(int ix=0; ix < depthMat.cols; ++ix)
       {
           if(colorMat.at<cv::Vec3b>(iy,ix)==colors[2])
           {
               depthMat.at<unsigned short>(iy,ix) = 0;
           }
       }
   }
}
/*
created at April 24, 2021
*/
void constructDepthmapsForClipPlanes(std::vector<Eigen::Hyperplane<float, 3>> &clipPlanes, std::vector<Eigen::Vector4f> &intr_vecs, std::vector<Eigen::Affine3f> &poses,
                                     float fence_height, std::vector<cv::Mat> &depthMats, std::vector<cv::Mat> &colorMats)
{
    for(int i=0; i < 3; ++i)
    {
        constructFenceDepthmap(intr_vecs[i], poses[i], clipPlanes, fence_height, depthMats[i], colorMats[i]);
    }
}
/*
created at April 27, 2021
*/
void extractObjects(std::vector<cv::Mat> &depthMats, std::vector<cv::Mat> &colorMats, std::vector<cv::Mat> &backDepthMats, float threshold)
{
    for(unsigned long i=0; i < depthMats.size(); ++i)
    {
        extractObject(depthMats[i], backDepthMats[i], threshold, colorMats[i]);
    }
}
/*
created at April 28, 2021
*/
void morphology(cv::Mat &depthMat)
{
    int morph_elem = cv::MORPH_RECT;//cv::MORPH_ELLIPSE;//cv::MORPH_CROSS;//cv::MORPH_RECT;
    int morph_size = 3;
    /*int morph_operator = 0;
    int const max_operator = 4;
    int const max_elem = 2;
    int const max_kernel_size = 21;
    // Since MORPH_X : 2,3,4,5 and 6
     int operation = morph_operator + 2;*/
     cv::Mat element = cv::getStructuringElement( morph_elem, cv::Size( 2*morph_size + 1, 2*morph_size+1 ), cv::Point( morph_size, morph_size ) );
     //cv::morphologyEx( depthMat, depthMat, cv::MORPH_CLOSE, element );
     int loop = 0;
     while(loop < 1)
     {
         cv::morphologyEx( depthMat, depthMat, cv::MORPH_DILATE, element );
         ++loop;
     }
     loop = 0;
     morph_size = 5;
     element = cv::getStructuringElement( morph_elem, cv::Size( 2*morph_size + 1, 2*morph_size+1 ), cv::Point( morph_size, morph_size ) );
     while(loop < 2)
     {
         cv::morphologyEx( depthMat, depthMat, cv::MORPH_ERODE, element );
         ++loop;
     }
}
/*
created at April 30
*/
void extractingOBB(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_in, pcl::PointXYZ &minPoint, pcl::PointXYZ &maxPoint,
                   pcl::PointXYZ &OBB_position, Eigen::Matrix3f &rotation_mat)
{
    Eigen::Vector4f pcaCentroid;
    pcl::compute3DCentroid(*cloud_in, pcaCentroid);
    Eigen::Matrix3f covariance;
    pcl::computeCovarianceMatrixNormalized(*cloud_in, pcaCentroid, covariance);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
    Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();
    eigenVectorsPCA.col(2) = eigenVectorsPCA.col(0).cross(eigenVectorsPCA.col(1));
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPCAprojection (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PCA<pcl::PointXYZ> pca;
    pca.setInputCloud(cloud_in);
    pca.project(*cloud_in, *cloudPCAprojection);
    std::cerr << std::endl << "EigenVectors: " << pca.getEigenVectors() << std::endl;
    std::cerr << std::endl << "EigenValues: " << pca.getEigenValues() << std::endl;

    Eigen::Matrix4f projectionTransform(Eigen::Matrix4f::Identity());
    projectionTransform.block<3,3>(0,0) = eigenVectorsPCA.transpose();
    projectionTransform.block<3,1>(0,3) = -1.f * (projectionTransform.block<3,3>(0,0) * pcaCentroid.head<3>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPointsProjected (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*cloud_in, *cloudPointsProjected, projectionTransform);
    // Get the minimum and maximum points of the transformed cloud.
    //pcl::PointXYZ minPoint, maxPoint;
    pcl::getMinMax3D(*cloudPointsProjected, minPoint, maxPoint);
    const Eigen::Vector3f meanDiagonal = 0.5f*(maxPoint.getVector3fMap() + minPoint.getVector3fMap());
    rotation_mat = eigenVectorsPCA;
    Eigen::Vector3f position = eigenVectorsPCA * meanDiagonal + pcaCentroid.head<3>();
    OBB_position.x = position[0];
    OBB_position.y = position[1];
    OBB_position.z = position[2];
    //const Eigen::Quaternionf bboxQuaternion(eigenVectorsPCA);
    //const Eigen::Vector3f bboxTransform = eigenVectorsPCA * meanDiagonal + pcaCentroid.head<3>();
}
/*
created at May 01, 2021
*/
void getOBB(pcl::PointCloud<pcl::PointXYZ>::Ptr pCloud,	Eigen::Vector3f &major_vector, Eigen::Vector3f &middle_vector, Eigen::Vector3f &minor_vector,
            pcl::PointXYZ &max_point_OBB, pcl::PointXYZ &min_point_OBB, pcl::PointXYZ &position_OBB, Eigen::Matrix3f &rotational_matrix_OBB, Eigen::Vector3f &mass_center)
{
    pcl::MomentOfInertiaEstimation <pcl::PointXYZ> feature_extractor;
    feature_extractor.setInputCloud (pCloud);
    feature_extractor.compute ();
    std::vector <float> moment_of_inertia;
    std::vector <float> eccentricity;
    pcl::PointXYZ min_point_AABB;
    pcl::PointXYZ max_point_AABB;
    feature_extractor.getMomentOfInertia (moment_of_inertia);
    feature_extractor.getEccentricity (eccentricity);
    feature_extractor.getAABB (min_point_AABB, max_point_AABB);
    feature_extractor.getOBB (min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);
    feature_extractor.getEigenVectors (major_vector, middle_vector, minor_vector);
    feature_extractor.getMassCenter(mass_center);
    major_vector.normalize();
    middle_vector.normalize();
    minor_vector.normalize();
}
/*
created at April 30, 2021
modified at May 06, 2021
*/
void estimateOBB_Vertices(cv::Mat &depthMat, float distToFloor, Eigen::Affine3f &pose, Eigen::Vector4f &intr_vec,
                 std::vector<Eigen::Vector3f> &vertices)
{
    float fx = intr_vec[0];
    float fy = intr_vec[1];
    float cx = intr_vec[2];
    float cy = intr_vec[3];
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>());
    for(int iy=0; iy < depthMat.rows; ++iy)
    {
        iy+=4;
        for(int ix=0; ix < depthMat.cols; ++ix)
        {
            ix+=4;
            if(depthMat.at<unsigned short>(iy,ix)==0)
                continue;
            pcl::PointXYZ p1, p2;
            p1.x = ix;
            p1.y = iy;
            p1.z = (float)depthMat.at<unsigned short>(iy,ix)/1000;
            cloud_ptr->points.push_back(p1);
        }
    }
    pcl::PointXYZ minPoint, maxPoint, position_OBB;
    Eigen::Matrix3f rotation_mat;
    Eigen::Vector3f major_vector, middle_vector, minor_vector, mass_center;
    getOBB(cloud_ptr, major_vector, middle_vector, minor_vector, maxPoint, minPoint, position_OBB, rotation_mat, mass_center);
    Eigen::Vector3f mv1 = rotation_mat*minPoint.getVector3fMap() + position_OBB.getVector3fMap();
    float minZ = mv1[2];
    std::cout << "minZ:" << minZ << std::endl;
    vertices.resize(8);
    vertices[0][0] = minPoint.x;
    vertices[0][1] = minPoint.y;
    vertices[0][2] = minZ;

    vertices[1][0] = minPoint.x;
    vertices[1][1] = maxPoint.y;
    vertices[1][2] = minZ;

    vertices[2][0] = maxPoint.x;
    vertices[2][1] = maxPoint.y;
    vertices[2][2] = minZ;

    vertices[3][0] = maxPoint.x;
    vertices[3][1] = minPoint.y;
    vertices[3][2] = minZ;

    std::vector<cv::Point> obb_corners;
    for(int i=0; i < 4; ++i)
    {
        vertices[i] = rotation_mat*vertices[i] + position_OBB.getVector3fMap();
        vertices[i][2] = minZ;
        vertices[4+i] = vertices[i];
        vertices[4+i][2] = distToFloor;
        cv::Point p;
        p.x = vertices[i][0];
        p.y = vertices[i][1];
        obb_corners.push_back(p);
    }

    for(int i=0; i < 4; ++i)
    {
        vertices[i][0] = vertices[i][2]*(vertices[i][0]-cx)/fx;
        vertices[i][1] = vertices[i][2]*(vertices[i][1]-cy)/fy;
        vertices[i] = pose*vertices[i];
    }
    for(int i=4; i < 8; ++i)
    {
        vertices[i][2] = distToFloor;
        vertices[i][0] = vertices[i][2]*(vertices[i][0]-cx)/fx;
        vertices[i][1] = vertices[i][2]*(vertices[i][1]-cy)/fy;
        vertices[i] = pose*vertices[i];
    }
}
/*
created at May 06, 2021
*/
void constructOBBData(std::vector<Eigen::Vector3f> &vertices, std::vector<std::vector<int>> &indices, std::vector<Eigen::Vector3f> &normals)
{
    indices.resize(6);
    for(int i=0; i < 6; ++i)
    {
        indices[i].resize(4);
    }
    //face 1
    indices[0][0] = 0; indices[0][1] = 1; indices[0][2] = 2; indices[0][3] = 3;
    //face 2
    indices[1][0] = 4; indices[1][1] = 5; indices[1][2] = 6; indices[1][3] = 7;
    //face 3
    indices[2][0] = 0; indices[2][1] = 1; indices[2][2] = 5; indices[2][3] = 4;
    //face 4
    indices[3][0] = 1; indices[3][1] = 2; indices[3][2] = 6; indices[3][3] = 5;
    //face 5
    indices[4][0] = 2; indices[4][1] = 3; indices[4][2] = 7; indices[4][3] = 6;
    //face 6
    indices[5][0] = 3; indices[5][1] = 0; indices[5][2] = 4; indices[5][3] = 7;

    Eigen::Vector3f vCenter(0.0f,0.0f,0.0f);
    for(size_t i=0; i < vertices.size(); ++i)
    {
        vCenter = (i*vCenter + vertices[i])/(i+1);
    }
    for(size_t i=0; i < indices.size(); ++i)
    {
        Eigen::Vector3f v0 = vertices[indices[i][0]];
        Eigen::Vector3f v1 = vertices[indices[i][1]];
        Eigen::Vector3f v2 = vertices[indices[i][2]];
        Eigen::Vector3f vec0 = v1 - v0;
        Eigen::Vector3f vec1 = v2 - v0;
        Eigen::Vector3f vec2 = v0 - vCenter;
        Eigen::Vector3f normal = vec1.cross(vec0);
        if(normal.dot(vec2) < 0.0f)
        {
            normal = -normal;
        }
        normal.normalize();
        normals.push_back(normal);
    }
}
/*
created at May 07, 2021
*/
void detectVisibleFaces(Eigen::Affine3f &pose, std::vector<Eigen::Vector3f> &vertices,
                        std::vector<std::vector<int>> &indices, std::vector<Eigen::Vector3f> &normals, std::vector<int> &faceIds)
{
    for(size_t i=0; i < normals.size(); ++i)
    {
        Eigen::Vector3f vCenter(0.0f,0.0f,0.0f);
        for(size_t j=0; j < indices[i].size(); ++j)
        {
            vCenter = (j*vCenter + vertices[indices[i][j]])/(j+1);
        }
        Eigen::Vector3f vec = pose.translation() - vCenter;
        if(vec.dot(normals[i]) > 0.01f)
            faceIds.push_back(i);
    }
}
/*
  created at May 06, 2021
  modified at May 07, 2021
*/
void drawOBB(Eigen::Vector4f &intr_vec, Eigen::Affine3f &pose, std::vector<Eigen::Vector3f> &vertices,
             std::vector<std::vector<int>> &indices, std::vector<Eigen::Vector3f> &normals,
             cv::Mat &colorMat)
{
    std::cout << "draw obb" << std::endl;
    std::vector<int> faceIds;
    detectVisibleFaces(pose, vertices, indices, normals, faceIds);
    std::cout << "face count:" << faceIds.size() << std::endl;
    float fx = intr_vec[0];
    float fy = intr_vec[1];
    float cx = intr_vec[2];
    float cy = intr_vec[3];
    std::vector<std::vector<cv::Point>> corners;
    corners.resize(faceIds.size());
    for(size_t i=0; i < faceIds.size(); ++i)
    {
        for(size_t j=0; j < indices[faceIds[i]].size(); ++j)
        {
            Eigen::Vector3f v = vertices[indices[faceIds[i]][j]];
            v = pose.inverse()*v;
            cv::Point p;
            p.x = fx*v[0]/v[2] + cx;
            p.y = fy*v[1]/v[2] + cy;
            corners[i].push_back(p);
            //corners_out.push_back(p);
        }
    }
    for(size_t i=0; i < corners.size(); ++i)
    {
        for(size_t j=0; j < 3; ++j)
        {
            //cv::line(mask, corners[i][j], corners[i][j+1], cv::Scalar(1), 1, 8);
            cv::line(colorMat, corners[i][j], corners[i][j+1], cv::Scalar(0,255,0), 2, 8);
        }
        //cv::line(mask, corners[i][3], corners[i][0], cv::Scalar(1), 1, 8);
        cv::line(colorMat, corners[i][3], corners[i][0], cv::Scalar(0,255,0), 1, 8);
    }
    cv::imshow("obb", colorMat);
    while(1)
    {
        cv::waitKey(10);
    }
}
/*
created at May 07, 2021
*/
void createMask(Eigen::Vector4f &intr_vec, Eigen::Affine3f &pose, std::vector<Eigen::Vector3f> &vertices,
                std::vector<Eigen::Vector3f> &normals, std::vector<std::vector<int>> &indices, cv::Mat &mask)
{
    std::vector<cv::Point> corners;
    //drawOBB(intr_vec, pose, vertices, indices, normals, corners, mask);
    for(int iy=0; iy < mask.rows; ++iy)
    {
        for(int ix=0; ix < mask.cols; ++ix)
        {
            if(mask.at<unsigned char>(iy, ix)==1)
                break;
            mask.at<unsigned char>(iy ,ix) = 0;
        }
        for(int ix=mask.cols-1; ix > -1; --ix)
        {
            if(mask.at<unsigned char>(iy, ix)==1)
                break;
            mask.at<unsigned char>(iy ,ix) = 0;
        }
    }
    /*cv::Point leftTop, rightBottom;
    int minX, maxX, minY, maxY;
    minX = minY = 50000;
    maxX = maxY = 0;
    for(size_t i=0; i < corners.size(); ++i)
    {
        if(corners[i].x < minX)
            minX = corners[i].x;
        if(corners[i].x > maxX)
            maxX = corners[i].x;
        if(corners[i].y < minY)
            minY = corners[i].y;
        if(corners[i].y > maxY)
            maxY = corners[i].y;
    }
    for(int iy=minY; iy < maxY; ++iy)
    {
        for(int ix=minX; ix < maxX; ++ix)
        {
            if(mask.at<unsigned char>(iy,ix)==1)
            {
                std::vector<cv::Point> pixes;
                while(ix < maxX)
                {
                    pixes.push_back(cv::Point(ix,iy));
                    ++ix;
                    if(mask.at<unsigned char>(iy,ix)==1)
                    {
                        for(size_t i=0; i<pixes.size(); ++i)
                        {
                            mask.at<unsigned char>(pixes[i].y, pixes[i].x) = 1;
                        }
                        pixes.resize(0);
                    }
                }
            }
        }
    }*/
}
/*
created at May 07, 2021
*/
void maskingDepth(cv::Mat &depthMat, cv::Mat &maskMat)
{
    depthMat = depthMat*maskMat;
}
/*
created at May 08, 2021
*/
void fromOBBToMesh(std::vector<Eigen::Vector3f> &vertices, std::vector<Eigen::Vector3f> &normals, std::vector<Eigen::Vector3i> &vertex_indices)
{
    std::vector<std::vector<int>> obb_indices;
    std::vector<Eigen::Vector3f> obb_normals;
    constructOBBData(vertices, obb_indices, obb_normals);
    for(size_t i=0; i < obb_indices.size(); ++i)
    {
        Eigen::Vector3i ids;
        ids[0] = obb_indices[i][0];
        ids[1] = obb_indices[i][1];
        ids[2] = obb_indices[i][2];
        vertex_indices.push_back(ids);
        normals.push_back(obb_normals[i]);
        ids[0] = obb_indices[i][2];
        ids[1] = obb_indices[i][3];
        ids[2] = obb_indices[i][0];
        vertex_indices.push_back(ids);
        normals.push_back(obb_normals[i]);
    }
}
/*
created at May 11, 2021
*/
void detectVisibleFaces(std::vector<Eigen::Affine3f> &poses, std::vector<Eigen::Vector3f> &vertices,
                          std::vector<Eigen::Vector3f> &normals, std::vector<Eigen::Vector3i> &indices)
{
    std::cout<<"whole face count:"<< indices.size() << std::endl;
    std::vector<Eigen::Vector3i> newIndices;
    std::vector<Eigen::Vector3f> newNormals;
    for(size_t i=0; i < indices.size(); ++i)
    {
        Eigen::Vector3f vCenter(0.0f,0.0f,0.0f);
        for(int j=0; j < 3; ++j)
        {
            vCenter = (j*vCenter + vertices[indices[i][j]])/(j+1);
        }
        for(size_t j=0; j < poses.size(); ++j)
        {
            Eigen::Vector3f vec = poses[j].translation() - vCenter;
            if(vec.dot(normals[i]) > 0.1f)
            {
                newIndices.push_back(indices[i]);
                newNormals.push_back(normals[i]);
                break;
            }
        }
   }
   indices.swap(newIndices);
   normals.swap(newNormals);
   std::cout<<"visible face count:"<< indices.size() << std::endl;
}
/*
created at May 13, 2021
*/
void createTextureData(int face_num, std::vector<Eigen::Vector2f> &texCoords, std::vector<Eigen::Vector3i> &texIndices)
{
    texCoords.resize(3*face_num);
    texIndices.resize(face_num);
    for(size_t i=0; i < face_num; ++i)
    {
        texIndices[i][0] = 3*i;
        texIndices[i][1] = 3*i+1;
        texIndices[i][2] = 3*i+2;
    }
}
/*
created at May 10, 2021
*/
void creatingPatches(std::vector<Eigen::Vector3f> &vertices, std::vector<Eigen::Vector3i> &vertex_indices,
                     std::vector<Eigen::Affine3f> &poses, std::vector<Eigen::Vector3f> &face_normals,
                     std::vector<std::vector<int>> &patches)
{
    for(size_t i=0; i < face_normals.size(); ++i)
    {
        Eigen::Vector3f vCenter(0.0f,0.0f,0.0f);
        for(int j=0; j < 3; ++j)
        {
            vCenter = (j*vCenter + vertices[vertex_indices[i][j]])/(j+1);
        }
        float maxDot = 0.0f;
        int goodView = 0;
        for(size_t j=0; j < poses.size(); ++j)
        {
            Eigen::Vector3f vec = poses[j].translation() - vCenter;
            vec.normalize();
            float dot = vec.dot(face_normals[i]);
            if(dot > maxDot)
            {
                maxDot = dot;
                goodView = j;
            }
        }
        //if(maxDot > 0.0f)
        //{
            patches[goodView].push_back(i);
        //}
    }
    std::vector<Eigen::Vector3i> newIndices;
    std::vector<Eigen::Vector3f> newNormals;
    std::vector<std::vector<int>> newPatches;
    newPatches.resize(patches.size());
    int face_count = 0;
    for(size_t i=0; i < patches.size(); ++i)
    {
        newPatches[i].resize(patches[i].size());
        for(size_t j=0; j < patches[i].size(); ++j)
        {
            Eigen::Vector3i t = vertex_indices[patches[i][j]];
            newIndices.push_back(t);
            Eigen::Vector3f normal = face_normals[patches[i][j]];
            newNormals.push_back(normal);
            newPatches[i][j] = face_count;
            ++face_count;
        }
    }
    vertex_indices.swap(newIndices);
    face_normals.swap(newNormals);
    for(size_t i=0; i < newPatches.size(); ++i)
    {
        for(size_t j=0; j < newPatches[i].size(); ++j)
        {
            patches[i][j] = newPatches[i][j];
        }
    }
    std::cout << "face count:" << face_count << std::endl;
}
/*
created at May 10, 2021
*/
void computeLocalPatchTexCoords(Eigen::Vector4f &intr_vec, Eigen::Affine3f &pose,
                                std::vector<Eigen::Vector3f> &vertices, std::vector<Eigen::Vector3i> &vertex_indices,
                                std::vector<Eigen::Vector2f> &tex_coords, std::vector<Eigen::Vector3i> &tex_indices,
                                std::vector<int> &patch_indices)
{
    float fx = intr_vec[0];
    float fy = intr_vec[1];
    float cx = intr_vec[2];
    float cy = intr_vec[3];
    Eigen::Affine3f inv = pose.inverse();
    for(size_t i = 0; i < patch_indices.size(); ++i)
    {
        int t = patch_indices[i];
        Eigen::Vector3i f = vertex_indices[t];
        for(int j=0; j < 3; ++j)
        {
            Eigen::Vector3f vert3D = inv*vertices[f[j]];
            Eigen::Vector2f texCoord;
            texCoord[0] = fx*vert3D[0]/vert3D[2] + cx;
            texCoord[1] = fy*vert3D[1]/vert3D[2] + cy;
            tex_coords[tex_indices[t][j]] = texCoord;
        }
    }
}
/*
created at May 12, 2021
modified at May 13, 2021
*/
void mappingTexturePatch(cv::Mat &targMat, cv::Mat &srcMat, Eigen::Vector2f &targOrigin,
                         Eigen::Vector2f &srcOrigin, std::vector<Eigen::Vector2f> &texCoords)
{
    for(size_t i=0; i < texCoords.size()/3; ++i)
    {
        Eigen::Vector2f vec1 = texCoords[3*i+1] - texCoords[3*i];
        Eigen::Vector2f vec2 = texCoords[3*i+2] - texCoords[3*i];
        int N1 = vec1.norm();
        int N2 = vec2.norm();
        for(int k=0; k < N2; ++k)
        {
            for(int j=0; j < N1; ++j)
            {
                float u = (float)j/N1;
                float v = (float)k/N2;
                if((u+v) > 1.0f)
                    continue;
                Eigen::Vector2f src_p = u*vec1 + v*vec2 + texCoords[3*i];
                Eigen::Vector2f targ_p = targOrigin + src_p - srcOrigin;
                targMat.at<cv::Vec3b>(targ_p[1], targ_p[0]) = srcMat.at<cv::Vec3b>(src_p[1], src_p[0]);
            }
        }
    }
}
/*
created at May 10, 2021
modified at May 12, 2021
*/
void creatingTexturePatches(std::vector<Eigen::Vector4f> &intr_vecs, std::vector<Eigen::Affine3f> &poses,
                            std::vector<Eigen::Vector3f> &vertices, std::vector<Eigen::Vector3f> face_normals,
                            std::vector<Eigen::Vector3i> &vertex_indices,
                            std::vector<Eigen::Vector2f> &tex_coords, std::vector<Eigen::Vector3i> &tex_indices,
                            std::vector<cv::Mat> &imgs, cv::Mat &texImg)
{
    size_t nViews = poses.size();
    std::vector<std::vector<int>> patches;
    patches.resize(nViews);
    creatingPatches(vertices, vertex_indices, poses, face_normals, patches);
    size_t face_num = vertex_indices.size();
    createTextureData(face_num, tex_coords, tex_indices);
    cv::Point leftTop(0,0);
    int bottomY = 0;
    for(size_t i=0; i < nViews; ++i)
    {
       if(patches[i].size()==0)
           continue;
       computeLocalPatchTexCoords(intr_vecs[i], poses[i], vertices, vertex_indices, tex_coords, tex_indices, patches[i]);
       float minX, minY, maxX, maxY;
       minX = minY = 50000.0f;
       maxX = maxY = 0.0f;
       std::vector<Eigen::Vector2f> patch_tex_coords;
       for(size_t j=0; j < patches[i].size(); ++j)
       {
           Eigen::Vector3i t = tex_indices[patches[i][j]];
           for(size_t k=0; k < 3; ++k)
           {
               if(minX > tex_coords[t[k]][0])
                    minX = tex_coords[t[k]][0];
               if(maxX < tex_coords[t[k]][0])
                    maxX = tex_coords[t[k]][0];
               if(minY > tex_coords[t[k]][1])
                    minY = tex_coords[t[k]][1];
               if(maxY < tex_coords[t[k]][1])
                    maxY = tex_coords[t[k]][1];
               patch_tex_coords.push_back(tex_coords[t[k]]);
           }
       }
       int local_width = maxX - minX;
       int local_height = maxY - minY;
       if((leftTop.x + local_width) > texImg.cols)
       {
          leftTop.x = 0;
          leftTop.y = bottomY;
       }
       for(int iy=0;iy < local_height; ++iy)
       {
           for(int ix=0; ix < local_width; ++ix)
           {
              texImg.at<cv::Vec3b>(leftTop.y+iy, leftTop.x+ix) = imgs[i].at<cv::Vec3b>(minY+iy,minX+ix);
           }
       }
       /*Eigen::Vector2f targOrigin(leftTop.x,leftTop.y);
       Eigen::Vector2f srcOrigin(minX, minY);
       mappingTexturePatch(texImg, imgs[i], targOrigin, srcOrigin, patch_tex_coords);*/
       for(size_t j=0; j < patches[i].size(); ++j)
       {
           Eigen::Vector3i t = tex_indices[patches[i][j]];
           for(int k=0; k < 3; ++k)
           {
              tex_coords[t[k]][0] = (float)(leftTop.x + tex_coords[t[k]][0] - minX)/texImg.cols;
              tex_coords[t[k]][1] = (float)(leftTop.y + tex_coords[t[k]][1] - minY)/texImg.rows;
           }
       }
       leftTop.x += local_width;
       if(bottomY < (leftTop.y + local_height))
           bottomY = leftTop.y + local_height;
    }
    /*cv::imshow("texture", texImg);
    while(1)
    {
        cv::waitKey(10);
    }*/
}
/*
created at May 14, 2021
*/
void extractObject(Eigen::Vector4f &intr_vec, Eigen::Affine3f &pose, cv::Mat &depthMat, std::vector<Eigen::Vector3f> &obb_vertices)
{
    float minX = 50000.0f;
    float maxX = 0.0f;
    float minY = 50000.0f;
    float maxY = 0.0f;
    float fx = intr_vec[0];
    float fy = intr_vec[1];
    float cx = intr_vec[2];
    float cy = intr_vec[3];
    Eigen::Affine3f inv = pose.inverse();
    for(size_t i =0; i < obb_vertices.size(); ++i)
    {
        Eigen::Vector3f vert_cam =inv*obb_vertices[i];
        float x = fx*vert_cam[0]/vert_cam[2] + cx;
        float y = fy*vert_cam[1]/vert_cam[2] + cy;
        if(minX > x)
            minX = x;
        if(maxX < x)
            maxX = x;
        if(minY > y)
            minY = y;
        if(maxY < y)
            maxY = y;
    }
    for(int iy = 0; iy < depthMat.rows; ++iy)
    {
        for(int ix = 0; ix < depthMat.cols; ++ix)
        {
            if(ix < minX)
            {
                depthMat.at<unsigned short>(iy,ix) = 0;
                continue;
            }
            if(iy < minY)
            {
                depthMat.at<unsigned short>(iy,ix) = 0;
                continue;
            }
            if(ix > maxX)
            {
                depthMat.at<unsigned short>(iy,ix) = 0;
                continue;
            }
            if(iy > maxY)
            {
                depthMat.at<unsigned short>(iy,ix) = 0;
                continue;
            }
            if(depthMat.at<unsigned short>(iy,ix) > 1200)
                depthMat.at<unsigned short>(iy,ix) = 0;
            if(depthMat.at<unsigned short>(iy,ix) < 100)
                depthMat.at<unsigned short>(iy,ix) = 0;
        }
    }
}
/*
created at May 14, 2021
*/
void estimateFaceNormals(std::vector<Eigen::Vector3f> &vertices, std::vector<Eigen::Vector3i> &vertex_indices,
                         std::vector<Eigen::Vector3f> &normals)
{
    Eigen::Vector3f vCenter(0.0f,0.0f,0.0f);
    for(size_t i=0; i < vertices.size(); ++i)
    {
        vCenter = (i*vCenter + vertices[i])/(i+1);
    }
    for(size_t i=0; i < vertex_indices.size(); ++i)
    {
        Eigen::Vector3i f = vertex_indices[i];
        Eigen::Vector3f v = (vertices[f[0]] + vertices[f[1]] + vertices[f[2]])/3;
        Eigen::Vector3f vec0 = v - vCenter;
        Eigen::Vector3f vec1 = vertices[f[1]] - vertices[f[0]];
        Eigen::Vector3f vec2 = vertices[f[2]] - vertices[f[0]];
        Eigen::Vector3f normal = vec1.cross(vec2);
        normal.normalize();
        if(normal.dot(vec0) < 0.0f)
            normal = -normal;
        normals.push_back(normal);
    }
}
/*
created at May 17, 2021
modified at May 18, 2021
*/
void fromFaceToDepth(std::vector<Eigen::Vector3f> &vertices, std::vector<int> &indices,
                     Eigen::Vector4f &intr_vec, Eigen::Affine3f &pose, cv::Mat &depthMat, cv::Mat &labelMat)
{
    float fx = intr_vec[0];
    float fy = intr_vec[1];
    float cx = intr_vec[2];
    float cy = intr_vec[3];
    std::vector<Eigen::Vector3f> vertices_cam;
    Eigen::Affine3f inv = pose.inverse();
    vertices_cam.resize(indices.size());
    for(size_t i=0; i < indices.size(); ++i)
    {
        vertices_cam[i] = inv*vertices[indices[i]];
    }
    Eigen::Vector2f pixels[4];
    for(size_t i=0; i < 4; ++i)
    {
        pixels[i][0] = fx*vertices_cam[i][0]/vertices_cam[i][2] + cx;
        pixels[i][1] = fy*vertices_cam[i][1]/vertices_cam[i][2] + cy;
    }
    Eigen::Vector2f vec1 = pixels[1] - pixels[0];
    Eigen::Vector2f vec2 = pixels[3] - pixels[0];
    int N1 = std::max(std::fabs(vec1[0]), std::fabs(vec1[1]));
    int N2 = std::max(std::fabs(vec2[0]), std::fabs(vec2[1]));
    cv::Mat tmp(depthMat.size(), CV_16UC1, cvScalar(0));
    float err = 0.0f;
    int count = 0;
    Eigen::Vector3f vec1_3d = vertices_cam[1] - vertices_cam[0];
    Eigen::Vector3f vec2_3d = vertices_cam[3] - vertices_cam[0];
    for(int j=0; j < N2; ++j)
    {
        for(int i=0;i < N1; ++i)
        {
            float a = (float)i/N1;
            float b = (float)j/N2;
            Eigen::Vector3f v = vertices_cam[0] + a*vec1_3d + b*vec2_3d;
            int x = fx*v[0]/v[2] + cx;
            int y = fy*v[1]/v[2] + cy;
            if(x > (depthMat.cols-1))
                continue;
            if(y > (depthMat.rows-1))
                continue;
            labelMat.at<unsigned char>(y,x) = 1;
            if(depthMat.at<unsigned short>(y,x)==0)
            {
                tmp.at<unsigned short>(y,x) = 1000*v[2];
            }
            else
            {
                float e = depthMat.at<unsigned short>(y,x) - 1000*v[2];
                err = (count*err + e)/(count + 1);
                ++count;
            }
        }
    }
    for(int iy=0; iy < depthMat.rows; ++iy)
    {
        for(int ix=0; ix < depthMat.cols; ++ix)
        {
           if(tmp.at<unsigned short>(iy,ix) > 0)
               depthMat.at<unsigned short>(iy,ix) = tmp.at<unsigned short>(iy,ix) - err;
        }
    }
}
/*
created at May 17, 2021
*/
void estimateDepth(cv::Mat &depthMat, cv::Mat &labelMat, Eigen::Vector4f &intr_vec, Eigen::Affine3f &pose,
                   std::vector<Eigen::Vector3f> &vertices, std::vector<std::vector<int>> &indices, std::vector<int> &faceIds)
{
    for(size_t i=0; i < faceIds.size(); ++i)
    {
       fromFaceToDepth(vertices, indices[faceIds[i]], intr_vec, pose, depthMat, labelMat);
    }
}
/*
created at May 19, 2021
*/
void groupingFaces(std::vector<Eigen::Vector3f> &vertices, std::vector<std::vector<int>> &indices, std::vector<Eigen::Vector3f> &normals,
                   std::vector<Eigen::Affine3f> &poses, std::vector<Eigen::Vector4f> &intr_vecs, std::vector<std::vector<int>> &faces)
{
    faces.resize(poses.size());
    for(size_t i=0; i < indices.size(); ++i)
    {
        Eigen::Vector3f vCenter(0.0f,0.0f,0.0f);
        for(size_t j=0; j < indices[i].size(); ++j)
        {
            vCenter = (j*vCenter + vertices[indices[i][j]])/(j+1);
        }
        float maxDot = 0.0f;
        int goodView = 0;
        for(size_t j=0; j < poses.size(); ++j)
        {
            Eigen::Vector3f vDir = poses[j].translation() - vCenter;
            vDir.normalize();
            float dot = vDir.dot(normals[j]);
            if(dot > maxDot)
            {
                maxDot = dot;
                goodView = j;
            }
        }
        if(maxDot > 0.0f)
        {
            faces[goodView].push_back(i);
        }
    }
}
/*
created at May 20, 2021
*/
void getFaceTightPlane(cv::Mat &depthMat, std::vector<Eigen::Vector3f> &vertices, std::vector<std::vector<int>> &indices,
                       int faceId, Eigen::Affine3f &pose, Eigen::Vector4f &intr_vec, Eigen::Hyperplane<float, 3> &plane)
{
    float fx = intr_vec[0];
    float fy = intr_vec[1];
    float cx = intr_vec[2];
    float cy = intr_vec[3];
    std::vector<Eigen::Vector2f> points;
    Eigen::Affine3f inv = pose.inverse();
    Eigen::Vector2f vCenter(0.0f,0.0f);
    for(size_t i=0; i < indices[faceId].size(); ++i)
    {
        Eigen::Vector3f vertex = vertices[indices[faceId][i]];
        vertex = inv*vertex;
        Eigen::Vector2f point;
        point[0] = fx*vertex[0]/vertex[2] + cx;
        point[1] = fy*vertex[1]/vertex[2] + cy;
        points.push_back(point);
        vCenter = (i*vCenter + point)/(i+1);
    }
    for(size_t i=0; i < points.size(); ++i)
    {
        points[i] = vCenter + 0.8*(points[i] - vCenter);
    }
    Eigen::Vector2f vec1 = points[1] - points[0];
    Eigen::Vector2f vec2 = points[3] - points[0];
    int N1 = std::max(vec1[0], vec1[1]);
    int N2 = std::max(vec2[0], vec2[1]);
    for(int j=0; j < N2; ++j)
    {
        for(int i=0; i < N1; ++i)
        {
            float a = (float)i/N1;
            float b = (float)j/N2;
            Eigen::Vector2f p = points[0] + a*vec1 + b*vec2;
        }
    }
}
