#include "../include/linedetection.h"

LineDetection::LineDetection() : imgTransp(nodeLine)
{
    nodeLine.param("/linedetection/imageTopic", imageTopic, imageTopic);
    nodeLine.param("/linedetection/gridTopic", gridTopic, gridTopic);
    nodeLine.param("/linedetection/odometryTopic", odometryTopic, odometryTopic);
    nodeLine.param("/linedetection/outputTopic", outputTopic, outputTopic);
    nodeLine.param("/linedetection/imageOutTopic", imageOutTopic, imageOutTopic);
    
    nodeLine.param("/linedetection/realWidth", realWidth, realWidth);
    nodeLine.param("/linedetection/realHeight", realHeight, realHeight);
    nodeLine.param("/linedetection/frequency", frequency, frequency);
    nodeLine.param("/linedetection/xMat", xMat, xMat);
    nodeLine.param("/linedetection/yMat", yMat, yMat);
    nodeLine.param("/linedetection/widthMat", widthMat, widthMat);
    nodeLine.param("/linedetection/heightMat", heightMat, heightMat);
    nodeLine.param("/linedetection/lengthReform", lengthReform, lengthReform);
    nodeLine.param("/linedetection/gamma", gamma, gamma);
    nodeLine.param("/linedetection/whiteStill", whiteStill, whiteStill);
    nodeLine.param("/linedetection/minSizeContours", minSizeContours, minSizeContours);
    nodeLine.param("/linedetection/minAngleLine", minAngleLine, minAngleLine);

    imageSub = imgTransp.subscribe(imageTopic, 1, &LineDetection::lineDetect, this);
    gridSub = nodeLine.subscribe<nav_msgs::OccupancyGrid>(gridTopic, 1, &LineDetection::occupancyHost, this);
    odometrySub = nodeLine.subscribe<nav_msgs::Odometry>(odometryTopic, 1, &LineDetection::odometryHost, this);

    gridPub = nodeLine.advertise<nav_msgs::OccupancyGrid>(outputTopic, 1);
    imagePub = imgTransp.advertise(imageOutTopic, 1);

    point_vector srcPoint, dstPoint;

    srcPoint.emplace_back(cv::Point2f(0, heightMat));
    srcPoint.emplace_back(cv::Point2f(lengthReform, 0));
    srcPoint.emplace_back(cv::Point2f(widthMat - lengthReform, 0));
    srcPoint.emplace_back(cv::Point2f(widthMat, heightMat));

    dstPoint.emplace_back(cv::Point2f(0, heightMat));
    dstPoint.emplace_back(cv::Point2f(0, 0));
    dstPoint.emplace_back(cv::Point2f(widthMat, 0));
    dstPoint.emplace_back(cv::Point2f(widthMat, heightMat));

    matrP = cv::findHomography(srcPoint, dstPoint);

    std::cout << "StartClass" << std::endl;

    std::thread th(&LineDetection::sendLines, this);
    th.detach();
}

void LineDetection::gammaCorrection(const cv::Mat& src, cv::Mat& dst, float fGamma)
{
    unsigned char lut[256];
    for (int i = 0; i < 256; i++)
    {
        lut[i] = cv::saturate_cast<uchar>(pow((float)(i / 255.0), fGamma) * 255.0f);
    }

    dst = src.clone();
    const int channels = dst.channels();
    switch (channels)
    {
        case 1:
        {
            cv::MatIterator_<uchar> it, end;
            for (it = dst.begin<uchar>(), end = dst.end<uchar>(); it != end; ++it)
            *it = lut[(*it)];
            break;
        }
        case 3:
        {
            cv::MatIterator_<cv::Vec3b> it, end;
            for (it = dst.begin<cv::Vec3b>(), end = dst.end<cv::Vec3b>(); it != end; ++it)
            {
            (*it)[0] = lut[((*it)[0])];
            (*it)[1] = lut[((*it)[1])];
            (*it)[2] = lut[((*it)[2])];
            }
            break;
        }
    }
}

void LineDetection::binarization(const cv::Mat& src, cv::Mat& dst)
{
    cv::Mat hsvMat;
    cv::cvtColor(src, hsvMat, cv::COLOR_RGB2HSV);
    std::vector<cv::Mat> channels;
    cv::split(hsvMat, channels);
    cv::Mat hMat = channels[2];

    cv::Mat gammaMat;
    gammaCorrection(hMat, gammaMat, gamma);

    cv::inRange(gammaMat, whiteStill, 255, dst);
}

void LineDetection::findContoursMat(const cv::Mat& src, cv::Mat& dst)
{
    cv::Mat cannyMat = cv::Mat::zeros(src.size(), CV_8UC1);
    dst = cv::Mat::zeros(src.size(), CV_8UC1);

    cv::Canny(src, cannyMat, 10, 100, 3);

    point_cloud contours;
    line hierarchy;
    cv::findContours(cannyMat, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
    for(int i = 0; i < contours.size(); ++i)
    {
        if(cv::contourArea(contours[i]) < minSizeContours)
        {
            contours.erase(contours.begin() + i);
            continue;
        }
        cv::drawContours(dst, contours, i, cv::Scalar(256, 256, 256), 1, 8, hierarchy, 0, cv::Point());
    }
}

void LineDetection::findLines(const cv::Mat& src, cv::Mat& dst)
{
    dst = cv::Mat::zeros(src.size(), CV_8UC1);

    line lines;
    cv::HoughLinesP(src, lines, 1, CV_PI/180, 35, 50, 100);

    for(size_t i = 0; i < lines.size(); ++i)
    {
        if(atan(abs((lines[i][1] - lines[i][3]) * 1.0 / (lines[i][0] - lines[i][2]))) * 180 / CV_PI < minAngleLine)
        {
            lines.erase(lines.begin() + i);
            continue;
        }
        cv::line(dst, cv::Point(lines[i][0], lines[i][1]), cv::Point(lines[i][2], lines[i][3]), cv::Scalar(255), 8, CV_AA);
    }
}

void LineDetection::setPerspective(const cv::Mat &src, cv::Mat &dst)
{
    dst = cv::Mat::zeros(src.size(), CV_8UC1);
    cv::warpPerspective(src, dst, matrP, src.size());

}

point_vector LineDetection::findPointCloud(const cv::Mat& src)
{
    point_cloud cloud;
    line graph;
    point_vector points;
    cv::findContours(src, cloud, graph, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
    for(int i = 0; i < cloud.size(); ++i)
        for(int j = 0; j < cloud[i].size(); ++j)
            points.push_back(cloud[i][j]);
    return points;

}

void LineDetection::sendLines()
{
    std::cout << "inSendLines" << std::endl;
    ros::Rate rate(frequency);

    point_vector pointVec;
    grid_ptr grid;

    while(nodeLine.ok())
    {
        rate.sleep();

        int pointSize = 0;
        int gridSize = 0;

        { //Lock both queues
            std::lock_guard<std::mutex> image_lock(imageMutex);
            std::lock_guard<std::mutex> grid_lock(gridMutex);

            if(pointQueue.empty() || gridQueue.empty()) {
                continue;
            }

            pointSize = pointQueue.size();
            gridSize = gridQueue.size();

            pointVec = pointQueue.front();
            grid = gridQueue.front();
            pointQueue.pop();
            gridQueue.pop();
        }

        nav_msgs::OccupancyGrid msg;

        const int gw = grid->info.width;
        const int gh = grid->info.height;
        const float resolution = grid->info.resolution;

        msg.info.height = gh;
        msg.info.width = gw;
        msg.info.resolution = resolution;
        msg.info.origin = grid->info.origin;
        msg.header = grid->header;
        msg.data = grid->data;

        for(const auto &point : pointVec){
            cv::Point2f position = imgToLocalSpace(point);

            double yaw = tf::getYaw(odometry->pose.pose.orientation);

            position = rotateVector(position, yaw);

            position += cv::Point2f(odometry->pose.pose.position.x, odometry->pose.pose.position.y);

            position += cv::Point2f(msg.info.origin.position.x, msg.info.origin.position.y);

            int py = static_cast<int>(std::round(position.y / resolution));

            int px = static_cast<int>(std::round(position.x / resolution));

            if (py < 0 || py >= gw || px < 0 || px >= gh)
                continue;

            msg.data[py + px * gw] = 100;
        }
        gridPub.publish(msg);
        std::cout << "LinesPublic" << std::endl;
    }

    
}

void LineDetection::lineDetect(const sensor_msgs::ImageConstPtr& msg)
{
    cv_bridge::CvImageConstPtr cvImg;
    try
    {
        if (enc::isColor(msg->encoding))
            cvImg = cv_bridge::toCvShare(msg, enc::BGR8);
        else
            cvImg = cv_bridge::toCvShare(msg, enc::MONO8);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cvImg->image.copyTo(source);
    cv::Mat cutMat = cv::Mat(source, cv::Rect(xMat, yMat, widthMat, heightMat)).clone();

    cv::Mat binMat;
    binarization(cutMat, binMat);

    cv::Mat contoursMat;
    findContoursMat(binMat, contoursMat);

    cv::Mat linesMat;
    findLines(contoursMat, linesMat);

    cv::Mat perspectMat;
    setPerspective(linesMat, perspectMat);

    point_vector points;
    points = findPointCloud(perspectMat);

    { // Lock image queue
        std::lock_guard<std::mutex> lock(imageMutex);
        pointQueue.emplace(std::move(points));

        if(pointQueue.size() > maxPoint)
            pointQueue.pop();
    }

    sensor_msgs::ImagePtr imgMsg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", perspectMat).toImageMsg();
    imagePub.publish(imgMsg);

#ifdef DEBUG
    cv::imshow("SOURCE", source);
    cv::imshow("CUTMAT", cutMat);
    cv::imshow("BINMAT", contoursMat);
    cv::imshow("LINES", linesMat);
    cv::imshow("PERSPECTIVE", perspectMat);
    cv::waitKey(1);
#endif
}

void LineDetection::occupancyHost(const nav_msgs::OccupancyGrid::ConstPtr& grid){
    std::lock_guard<std::mutex> lock(gridMutex);
    gridQueue.push(grid);

    if(gridQueue.size() > maxGrid)
        gridQueue.pop();
}

void LineDetection::odometryHost(const nav_msgs::Odometry::ConstPtr& odometry_){
    odometry = odometry_;
}

cv::Point2f LineDetection::imgToLocalSpace(const cv::Point2f& point) {
    float x = (heightMat - point.y) / heightMat * realHeight;
    float y = (point.x - (widthMat / 2)) / widthMat * realWidth;
    return cv::Point2f(x, y);
}

cv::Point2f LineDetection::rotateVector(const cv::Point2f& v, double r){
    double ca = cos(r);
    double sa = sin(r);
    return cv::Point2f(ca*v.x - sa*v.y, sa*v.x + ca*v.y);
}
