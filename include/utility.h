#ifndef UTILITY_H
#define UTILITY_H

#endif // UTILITY_H
#include "opencv/cv.h"
#include "opencv2/opencv.hpp"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
//#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/io/ply_io.h>
//#include <pcl/features/fpfh.h>
//#include <pcl/visualization/pcl_visualizer.h>
// Visualization Toolkit (VTK)
#include <vtkRenderWindow.h>
#include "qdir.h"
#include "GL/freeglut.h"
#include "GL/gl.h"
typedef std::vector<cv::Point> REG;
typedef std::pair<int, int> PAIR;
void extractForeground(cv::Mat &depthMat, float dist_floor, float tolerance);
void fromDatasetToPointCloud(std::string &root, float maxDepth, float minDepth, float smooth_tolerance, int filtering,
                              pcl::PointCloud<pcl::PointXYZRGB> &cloud_merged);
void scalingPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ptr, float s);
Eigen::Vector3f getCenterOfPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ptr, pcl::PointXYZ &center);
void extractColors(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ptr, std::vector<Eigen::Vector3f> &colors);
void decimateMesh(pcl::PolygonMesh& mesh_in, pcl::PolygonMesh& out_mesh);
void writePolygonMeshFile (const pcl::PolygonMesh& mesh, std::string &meshFileName);
void getDataPaths(std::string &root, std::vector<std::string> &depthPaths, std::vector<std::string> &colorPaths, std::vector<std::string> &backPaths, std::vector<std::string> &calibPaths);
void mkdir(std::string &dir_name);
void getCalibration(std::string &calibFilePath, Eigen::Vector4f &intr_vec, Eigen::Affine3f &pose);
void fromDepthToPointCloud(cv::Mat &depthMat, Eigen::Vector4f &intr_vec, int maxDepth, pcl::PointCloud<pcl::PointXYZ>::Ptr pCloud);
void showPointCloud(Eigen::Affine3f &camPose, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ptr);
void centeringOfCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ptr);
void downSamplingCloud(float leaf_size, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ptr);
void estimateParams(cv::Mat &depthMat, cv::Mat &colorMat, Eigen::Vector4f &intr_vec, float &camHeight, float &minY, float &maxY);
void writeParams(std::string &fileName, float &camHeight, float &minY, float &maxY);
void readParams(std::string &fileName, float &camHeight, float &minY, float &maxY);
void extractObject(cv::Mat &depthMap, int maxDepth, int minDepth, int cWidth, int cHeight, float tolerance);
void getData(std::string &dataset, std::vector<cv::Mat> &depthMats, std::vector<cv::Mat> &colorMats,
             std::vector<cv::Mat> &backColorMats, std::vector<Eigen::Vector4f> &intr_vecs, std::vector<Eigen::Affine3f> &poses);
void extractingObjects(std::vector<cv::Mat> &depthMats, int maxDepth, int minDepth, int smooth_tolerance);
void constructPointCloud(std::vector<cv::Mat> &depthMats, std::vector<cv::Mat> &colorMats,
                         std::vector<Eigen::Vector4f> &intr_vecs, std::vector<Eigen::Affine3f> &poses,
                         pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ptr);
void downsamplingData(std::vector<cv::Mat> &depthMats, std::vector<cv::Mat> &colorMats, std::vector<Eigen::Vector4f> &intr_vecs, float sample,
                      std::vector<cv::Mat> &depthMats_out, std::vector<cv::Mat> &colorMats_out, std::vector<Eigen::Vector4f> &intr_vecs_out);
void centeringOfMesh(std::vector<std::vector<Eigen::Vector3f>> &vertices_parts);
Eigen::Vector3f centering(std::vector<Eigen::Vector3f> &vertices);
void extractForeground(cv::Mat &depthMat, cv::Mat &colorMat);
void constructDepthmaps(std::vector<Eigen::Vector4f> &intr_vecs, std::vector<Eigen::Affine3f> &poses,
                        std::vector<cv::Mat> &depthMats, std::vector<cv::Mat> &colorMats);
void extractObject(cv::Mat &depthMat, cv::Mat &backMat, cv::Mat &colorMat);
void getData(std::string &dataset, std::vector<cv::Mat> &depthMats, std::vector<cv::Mat> &colorMats,
             std::vector<Eigen::Vector4f> &intr_vecs, std::vector<Eigen::Affine3f> &poses);
Eigen::Vector3f getCenter(std::vector<Eigen::Vector3f> &vertices);
void extractPlanes(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in, std::vector<Eigen::Hyperplane<float, 3>> &planes,
                   pcl::PointCloud<pcl::PointXYZRGB> &cloud_result);
void removeBottom(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in, float delta, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_out);
void extractClipPlanes(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in, float belt_width, float dist_to_belt, float delta,
                       pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_out, std::vector<Eigen::Hyperplane<float, 3>> &clipPlanes);
void constructDepthmapsForClipPlanes(std::vector<Eigen::Hyperplane<float, 3>> &clipPlanes, std::vector<Eigen::Vector4f> &intr_vecs, std::vector<Eigen::Affine3f> &poses,
                                     float fence_height, std::vector<cv::Mat> &depthMats, std::vector<cv::Mat> &colorMats);
void extractObjects(std::vector<cv::Mat> &depthMats, std::vector<cv::Mat> &colorMats, std::vector<cv::Mat> &backDepthMats, float threshold);
void morphology(cv::Mat &depthMat);
void estimateOBB_Vertices(cv::Mat &depthMat, float distToFloor, Eigen::Affine3f &pose, Eigen::Vector4f &intr_vec,
                 std::vector<Eigen::Vector3f> &vertices);
void constructOBBData(std::vector<Eigen::Vector3f> &vertices, std::vector<std::vector<int>> &indices, std::vector<Eigen::Vector3f> &normals);
void drawOBB(Eigen::Vector4f &intr_vec, Eigen::Affine3f &pose, std::vector<Eigen::Vector3f> &vertices,
             std::vector<std::vector<int>> &indices, std::vector<Eigen::Vector3f> &normals, cv::Mat &colorMat);
void createMask(Eigen::Vector4f &intr_vec, Eigen::Affine3f &pose, std::vector<Eigen::Vector3f> &vertices,
                std::vector<Eigen::Vector3f> &normals, std::vector<std::vector<int>> &indices, cv::Mat &mask);
void fromOBBToMesh(std::vector<Eigen::Vector3f> &vertices, std::vector<Eigen::Vector3f> &normals, std::vector<Eigen::Vector3i> &vertex_indices);
void creatingTexturePatches(std::vector<Eigen::Vector4f> &intr_vecs, std::vector<Eigen::Affine3f> &poses,
                            std::vector<Eigen::Vector3f> &vertices, std::vector<Eigen::Vector3f> face_normals, std::vector<Eigen::Vector3i> &vertex_indices,
                            std::vector<Eigen::Vector2f> &tex_coords, std::vector<Eigen::Vector3i> &tex_indices,
                            std::vector<cv::Mat> &imgs, cv::Mat &texImg);
void detectVisibleFaces(std::vector<Eigen::Affine3f> &poses, std::vector<Eigen::Vector3f> &vertices,
                          std::vector<Eigen::Vector3f> &normals, std::vector<Eigen::Vector3i> &indices);
void detectVisibleFaces(Eigen::Affine3f &pose, std::vector<Eigen::Vector3f> &vertices,
                        std::vector<std::vector<int>> &indices, std::vector<Eigen::Vector3f> &normals, std::vector<int> &faceIds);
void extractObject(Eigen::Vector4f &intr_vec, Eigen::Affine3f &pose, cv::Mat &depthMat, std::vector<Eigen::Vector3f> &obb_vertices);
void estimateFaceNormals(std::vector<Eigen::Vector3f> &vertices, std::vector<Eigen::Vector3i> &vertex_indices, std::vector<Eigen::Vector3f> &normals);
void estimateDepth(cv::Mat &depthMat, cv::Mat &labelMat, Eigen::Vector4f &intr_vec, Eigen::Affine3f &pose, std::vector<Eigen::Vector3f> &vertices,
                   std::vector<std::vector<int>> &indices, std::vector<int> &faceIds);
