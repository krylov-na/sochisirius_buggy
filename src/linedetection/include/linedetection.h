#ifndef LINEDETECTION_H
#define LINEDETECTION_H

#define DEBUG

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Odometry.h>
#include <tf/LinearMath/Quaternion.h>
#include <tf/transform_datatypes.h>

#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
#include <queue>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/opencv.hpp>

typedef std::vector<std::vector<cv::Point> > point_cloud;
typedef std::vector<cv::Point2f> point_vector;
typedef boost::shared_ptr<const nav_msgs::OccupancyGrid> grid_ptr;
typedef std::vector<cv::Vec4i> line;

namespace enc = sensor_msgs::image_encodings;

class LineDetection
{
private:
    ros::NodeHandle nodeLine;
    image_transport::ImageTransport imgTransp;
    image_transport::Subscriber imageSub;
    image_transport::Publisher imagePub;
    ros::Subscriber gridSub;
    ros::Subscriber odometrySub;
    ros::Publisher gridPub;

    boost::shared_ptr<const nav_msgs::Odometry> odometry;

    std::string imageTopic = "/rgb/image_rect_color";
    std::string gridTopic = "/rtabmap/grid_map";
    std::string odometryTopic = "/zed/odom";
    std::string outputTopic = "/linedetection/map";
    std::string imageOutTopic = "/linedetection/image";

    int frequency = 30;

    cv::Mat source;
    cv::Mat matrP;

    int xMat = 267; //100 + 167
    int yMat = 322;
    int widthMat = 861; // 1028 - 167
    int heightMat = 217;
    int lengthReform = 215;

    float gamma = 0.9;
    int whiteStill = 235;

    int minSizeContours = 10;
    float minAngleLine = 25;

    std::mutex imageMutex;
    std::mutex gridMutex;

    std::queue<point_vector> pointQueue;
    std::queue<grid_ptr> gridQueue;

    int maxGrid = 5;
    int maxPoint = 5;

    float realWidth = 2;
    float realHeight = 2;

    void sendLines();
    void gammaCorrection(const cv::Mat& src, cv::Mat& dst, float fGamma);
    void binarization(const cv::Mat& src, cv::Mat& dst);
    void findContoursMat(const cv::Mat& src, cv::Mat& dst);
    void findLines(const cv::Mat& src, cv::Mat& dst);
    point_vector findPointCloud(const cv::Mat& src);
    void setPerspective(const cv::Mat& src, cv::Mat& dst);

    void lineDetect(const sensor_msgs::ImageConstPtr& msg);
    void odometryHost(const nav_msgs::Odometry::ConstPtr& odometry_);
    void occupancyHost(const nav_msgs::OccupancyGrid::ConstPtr& grid);
    cv::Point2f imgToLocalSpace(const cv::Point2f& point);
    cv::Point2f rotateVector(const cv::Point2f& v, double r);

public:
    LineDetection();
};

#endif // LINEDETECTION_H
