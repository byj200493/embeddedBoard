#ifndef RENDER_H
#define RENDER_H
#include "GL/freeglut.h"
#include "GL/gl.h"

void rotatingVertices(std::vector<Eigen::Vector3f> &vertices, float rotX, float rotY, float rotZ);
GLuint setTexture(std::vector<unsigned char> &texels, int width, int height);
GLuint setViewToTexture(cv::Mat &colorMat);
void createTexturesForViews(std::vector<cv::Mat> &imgMats, std::vector<GLuint> &textures);
void drawTexturedMesh(GLuint texture,
                      std::vector<Eigen::Vector3f> &vertices,
                      std::vector<Eigen::Vector3f> &normals,
                      std::vector<Eigen::Vector2f> &tex_coords,
                      std::vector<Eigen::Vector3i> &vertex_indices, std::vector<Eigen::Vector3i> &normal_indices,
                      std::vector<Eigen::Vector3i> &tex_indices, Eigen::Affine3f &pose, float fScale);
void drawPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ptr, Eigen::Affine3f &pose, float scale);
void drawImage(GLuint texture, int width, int height);
void rotatingVertices(std::vector<Eigen::Vector3f> &vertices1, std::vector<Eigen::Vector3f> &vertices2, Eigen::Affine3f &pose);
void rotatingVertices(std::vector<Eigen::Vector3f> &vertices1, std::vector<Eigen::Vector3f> &vertices2, Eigen::Vector3f &ea);
GLuint setColorMatToTexture(cv::Mat &colorMat);
#endif // RENDER_H
