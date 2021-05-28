#include "render.h"

GLuint setTexture(std::vector<unsigned char> &texels, int width, int height)
{
    GLuint id;
    glGenTextures(1, &id);
    glBindTexture(GL_TEXTURE_2D, id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, texels.data());
    return id;
}
/*
created at March 05, 2021
*/
GLuint setViewToTexture(cv::Mat &colorMat)
{
    cv::Mat objMat;
    colorMat.copyTo(objMat);
    cv::resize(objMat, objMat, cv::Size(80, 80));
    std::vector<unsigned char> pixels;
    int width = objMat.cols;
    int height = objMat.cols;
    for(int iy=height-1; iy >= 0; --iy)
    {
        for(int ix=0; ix < width; ++ix)
        {
            cv::Vec3b color = objMat.at<cv::Vec3b>(iy,ix);
            pixels.push_back(color[2]);
            pixels.push_back(color[1]);
            pixels.push_back(color[0]);
        }
    }
    return setTexture(pixels, width, height);
}
/*
created at March 05, 2021
*/
void createTexturesForViews(std::vector<cv::Mat> &imgMats, std::vector<GLuint> &textures)
{
    for(unsigned long i=0; i < imgMats.size(); ++i)
    {
       cv::Mat img;
       cv::resize(imgMats[i], img, cv::Size(imgMats[i].cols/2, imgMats[i].rows/2));
       GLuint id = setViewToTexture(img);
       textures.push_back(id);
    }
}

void rotatingVertices(std::vector<Eigen::Vector3f> &vertices, float rotX, float rotY, float rotZ)
{
    Eigen::Affine3f tr = Eigen::Affine3f::Identity();
    Eigen::Matrix3f m;
    m = Eigen::AngleAxisf(rotX*M_PI/180, Eigen::Vector3f::UnitX())
      * Eigen::AngleAxisf(rotY*M_PI/180,  Eigen::Vector3f::UnitY())
      * Eigen::AngleAxisf(rotZ*M_PI/180, Eigen::Vector3f::UnitZ());
    tr.rotate(m);
    for(unsigned long i=0; i < vertices.size(); ++i)
    {
        vertices[i] = tr*vertices[i];
    }
    glutPostRedisplay();
}
/*
created at January 30, 2021
modified at March 02, 2021
modified at March 15, 2021
*/
void drawTexturedMesh(GLuint texture,
                      std::vector<Eigen::Vector3f> &vertices_in,
                      std::vector<Eigen::Vector3f> &normals,
                      std::vector<Eigen::Vector2f> &tex_coords,
                      std::vector<Eigen::Vector3i> &vertex_indices, std::vector<Eigen::Vector3i> &normal_indices,
                      std::vector<Eigen::Vector3i> &tex_indices, Eigen::Affine3f &pose, float fScale)
{
    Eigen::Affine3f inv = pose.inverse();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(0.0,1.0,0.0,0.0);
    glClearDepth(1.0f);
    glEnable(GL_DEPTH_TEST);
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluPerspective(45, 1.2f, 0.01f, 3.0f);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glRotatef(180, 1.0, 0.0, 0.0);
    //glRotatef(rotX, 1.0, 0.0, 0.0);
    //glRotatef(rotY, 0.0, 1.0, 0.0);
    glScalef(fScale, fScale, fScale);
    //glTranslatef(t[0],t[1],t[2]);
    glEnable(GL_TEXTURE_2D);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
    glBindTexture(GL_TEXTURE_2D, texture);
    glBegin(GL_TRIANGLES);
    //unsigned long face_num = std::min(vertex_indices.size(), tex_indices.size());
    //face_num = std::min(face_num, normal_indices.size());
    unsigned long face_num = vertex_indices.size();
    for(unsigned long i=0; i < face_num; ++i)
    {
        for(int j=0; j < 3; ++j)
        {
            glTexCoord2f(tex_coords[tex_indices[i][j]][0], tex_coords[tex_indices[i][j]][1]);
            Eigen::Vector3f v = inv*vertices_in[vertex_indices[i][j]];
            glVertex3f(v[0], v[1], v[2]);
            //glNormal3f(normals[normal_indices[i][j]][0], normals[normal_indices[i][j]][1], normals[normal_indices[i][j]][2]);
        }
    }
    glEnd();
    glPopMatrix();
    glDisable(GL_TEXTURE_2D);
    glDisable(GL_DEPTH_TEST);
    glFlush();

}

void orthogonalStart(int width, int height)
{
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(-width/2, width/2, -height/2, height/2);
    glMatrixMode(GL_MODELVIEW);
}

void orthogonalEnd()
{
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}

void drawImage(GLuint texture, int width, int height)
{
    glEnable(GL_TEXTURE_2D);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
    glBindTexture( GL_TEXTURE_2D, texture );
    orthogonalStart(width, height);
    // texture width/height
    glPushMatrix();
    glTranslatef( -width/2, -height/2, 0 );
    glBegin(GL_QUADS);
    glTexCoord2i(0,0); glVertex2i(0, 0);
    glTexCoord2i(1,0); glVertex2i(width, 0);
    glTexCoord2i(1,1); glVertex2i(width, height);
    glTexCoord2i(0,1); glVertex2i(0, height);
    glEnd();
    glPopMatrix();
    orthogonalEnd();
    //glutSwapBuffers();
    //glFlush();
    //glDisable(GL_TEXTURE_2D);
}

void rotatingVertices(std::vector<Eigen::Vector3f> &vertices1, std::vector<Eigen::Vector3f> &vertices2, Eigen::Affine3f &pose)
{
    Eigen::Matrix3f m;
    m = pose.rotation();
    for(unsigned long i=0; i < vertices1.size(); ++i)
    {
        vertices2[i] = m*vertices1[i];
    }
}

void rotatingVertices(std::vector<Eigen::Vector3f> &vertices1, std::vector<Eigen::Vector3f> &vertices2, Eigen::Vector3f &ea)
{
    Eigen::Matrix3f m;
    m = Eigen::AngleAxisf(ea[0], Eigen::Vector3f::UnitX())
            * Eigen::AngleAxisf(ea[1], Eigen::Vector3f::UnitY())
            * Eigen::AngleAxisf(ea[2], Eigen::Vector3f::UnitZ());
    for(unsigned long i=0;  i < vertices1.size(); ++i)
    {
        vertices2[i] = m*vertices1[i];
    }
}

void rotatingPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ptr, float setaX, float setaY, float setaZ)
{
    Eigen::Affine3f tr = Eigen::Affine3f::Identity();
    Eigen::Matrix3f m;
    m = Eigen::AngleAxisf(setaX*M_PI/180, Eigen::Vector3f::UnitX())
      * Eigen::AngleAxisf(setaY*M_PI/180, Eigen::Vector3f::UnitY())
      * Eigen::AngleAxisf(setaZ*M_PI/180, Eigen::Vector3f::UnitZ());
    tr.rotate(m);
    pcl::transformPointCloud(*cloud_ptr, *cloud_ptr, tr);
}
/*
created at March 09, 2021
*/
Eigen::Vector3f angleBetweenTwoVectors(Eigen::Vector3f &vec1, Eigen::Vector3f &vec2)
{
    Eigen::Matrix3f R;
    R = Eigen::Quaternionf().setFromTwoVectors(vec1, vec2);
    return R.eulerAngles(0,1,2);
}
/*
modified at Febrary 27, 2021
modified at March 02, 2021
*/
void drawPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ptr, Eigen::Affine3f &pose, float fScale)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(0.0,1.0,0.0,0.0);
    glClearDepth(1.0f);
    glEnable(GL_DEPTH_TEST);
    Eigen::Affine3f inv = pose.inverse();
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluPerspective(45, 1.8f, 0.01f, 3.0f);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glRotatef(180, 1.0, 0.0, 0.0);
    //glRotatef(rotX, 1.0, 0.0, 0.0);
    //glRotatef(rotY, 0.0, 1.0, 0.0);
    //glRotatef(zRot, 0.0, 0.0, 1.0);*/
    glScalef(fScale, fScale, fScale);
    glBegin(GL_POINTS);
    for(unsigned long i=0; i < cloud_ptr->points.size(); ++i)
    {
        float r = (float)cloud_ptr->points[i].r/255;
        float g = (float)cloud_ptr->points[i].g/255;
        float b = (float)cloud_ptr->points[i].b/255;
        glColor3f(r, g, b);
        Eigen::Vector3f v = cloud_ptr->points[i].getVector3fMap();
        v = inv*v;
        glVertex3f(v[0], v[1], v[2]);
    }
    glEnd();
    glPopMatrix();
    glDisable(GL_DEPTH_TEST);
    glFlush();
}

GLuint setColorMatToTexture(cv::Mat &colorMat)
{
    std::vector<unsigned char> pixels;
    int width = colorMat.cols;
    int height = colorMat.cols;
    for(int iy=0; iy < height; ++iy)
    {
        for(int ix=0; ix < width; ++ix)
        {
            cv::Vec3b color = colorMat.at<cv::Vec3b>(iy,ix);
            pixels.push_back(color[2]);
            pixels.push_back(color[1]);
            pixels.push_back(color[0]);
        }
    }
    return setTexture(pixels, width, height);
}
