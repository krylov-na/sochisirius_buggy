#include <ros/ros.h>

#include "../include/linedetection.h"

int main(int argc, char** argv)
{
    ros::init(argc, argv, "linedetection");
    LineDetection li;
    ros::spin();
    return 0;
}
