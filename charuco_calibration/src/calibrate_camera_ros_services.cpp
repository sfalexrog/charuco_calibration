#include "calibrator.h"
#include "utils.h"

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/SetCameraInfo.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <std_srvs/Trigger.h>

#include <opencv2/imgcodecs.hpp>

#include <charuco_calibration/CalibratorCommand.h>

#include <stdlib.h>
#include <iomanip>
#include <sstream>

#include <sys/stat.h>
#include <sys/types.h>

#include <vector>
#include <iostream>
#include <ctime>
#include <fstream>

using namespace cv;

cv::Mat lastImage;
bool hasImage = false;
int calibratorCommand = 0;

void imageCallback(const sensor_msgs::ImageConstPtr &img)
{
    cv_bridge::CvImagePtr cv_img = cv_bridge::toCvCopy(img);
    lastImage = cv_img->image;
    hasImage = true;
}

bool commandCallback(charuco_calibration::CalibratorCommandRequest& request,
                     charuco_calibration::CalibratorCommandResponse& response)
{
    calibratorCommand = request.command;
    response.success = true;
    return true;
}

int waitCommand(ros::Duration timeout)
{
    ros::Time start = ros::Time::now();
    while (ros::Time::now() < start + timeout)
    {
        if (calibratorCommand != 0)
        {
            return calibratorCommand;
        }
        ros::spinOnce();
    }
    return calibratorCommand;
}

static void logFunction(charuco_calibration::LogLevel logLevel, const std::string& message)
{
    ros::console::levels::Level rosLogLevel;
    switch(logLevel)
    {
        case charuco_calibration::LogLevel::DEBUG:
            rosLogLevel = ros::console::levels::Level::Debug;
            break;
        case charuco_calibration::LogLevel::INFO:
            rosLogLevel = ros::console::levels::Level::Info;
            break;
        case charuco_calibration::LogLevel::WARN:
            rosLogLevel = ros::console::levels::Level::Warn;
            break;
        case charuco_calibration::LogLevel::ERROR:
            rosLogLevel = ros::console::levels::Level::Error;
            break;
        case charuco_calibration::LogLevel::FATAL:
            rosLogLevel = ros::console::levels::Level::Fatal;
            break;
        default:
            // This should never happen, but if it does, everything went south
            rosLogLevel = ros::console::levels::Level::Fatal;
    }
    ROS_LOG(rosLogLevel, ROSCONSOLE_DEFAULT_NAME, "%s", message.c_str());
}

/**
 */
int main(int argc, char *argv[]) {
    using namespace charuco_calibration;
    ros::init(argc, argv, "cv_calib");

    std::string imgPath;

    // Get current datetime
    auto t = time(nullptr);
    auto tm = *localtime(&t);
    std::ostringstream oss;
    oss << "calibration_" << std::put_time(&tm, "%Y%m%d_%H%M%S");
    auto datetime = oss.str();

    ros::NodeHandle nh;
    ros::NodeHandle nhPriv("~");

    charuco_calibration::Calibrator calibrator;
    readCalibratorParams(nhPriv, calibrator);

    bool saveCalibrationImages = nhPriv.param<bool>("save_images", true);
    std::string outputFile = nhPriv.param<std::string>("output_file", "calibration.yaml");
    
    // Make folder with timedate name
    mkdir(datetime.c_str(), 0775);

    // Get output filepath
    oss << "/" << outputFile;
    auto outputFilePath = oss.str();

    ros::NodeHandle nh_detector("~detector_parameters");
    readDetectorParameters(nh_detector, calibrator.arucoDetectorParams);

    //image_transport::TransportHints hints("compressed", ros::TransportHints());
    image_transport::ImageTransport it(nh);
    image_transport::ImageTransport itPriv(nhPriv);

    auto setCameraInfo = nh.serviceClient<sensor_msgs::SetCameraInfo>("set_camera_info");

    ROS_INFO("Waiting for set_camera_info service to become available...");
    if (!setCameraInfo.waitForExistence(ros::Duration(60.0)))
    {
        ROS_ERROR("set_camera_info unavailable after 60 seconds, is camera running?");
        return 1;
    }

    int waitTime = 10;
    ROS_INFO("Subscribing to image topic");
    auto sub = it.subscribe("image", 1, imageCallback /*, hints */);
    ROS_INFO("Advertising charuco board image");
    auto boardPub = nhPriv.advertise<sensor_msgs::Image>("board", 1, true);
    ROS_INFO("Advertising feedback image");
    auto feedbackPub = itPriv.advertise("feedback", 1);
    ROS_INFO("Advertising command service");
    auto commandSrv = nhPriv.advertiseService("calib_command", commandCallback);

    int boardImgWidth = nhPriv.param("board_image_width", 1536);
    int boardImgHeight = nhPriv.param("board_image_height", 2048);
    int boardImgBorder = nhPriv.param("board_image_border", 100);

    auto boardImg = calibrator.getBoardImage(boardImgWidth, boardImgHeight, boardImgBorder);

    cv::imwrite("board.png", boardImg);
    cv_bridge::CvImage boardImgBridge;
    boardImgBridge.image = boardImg;
    boardImgBridge.encoding = sensor_msgs::image_encodings::MONO8;
    boardPub.publish(boardImgBridge.toImageMsg());

    while(!hasImage)
    {
        ros::spinOnce();
    }

    int imgCounter = 1;

    calibrator.setLogger(logFunction);

    while(hasImage) {
        Mat image;
        image = lastImage.clone();

        auto detectionResult = calibrator.processImage(image);
        cv_bridge::CvImage displayedImage;
        displayedImage.image = calibrator.drawDetectionResults(detectionResult);
        displayedImage.encoding = sensor_msgs::image_encodings::BGR8;

        feedbackPub.publish(displayedImage.toImageMsg());

        int command = waitCommand(ros::Duration(0.016));
        calibratorCommand = 0;

        if(command == charuco_calibration::CalibratorCommandRequest::COMMAND_CALIBRATE) break;
        if(command == charuco_calibration::CalibratorCommandRequest::COMMAND_CAPTURE) {
            if (detectionResult.isValid()) {
                std::cout << "Frame " << imgCounter << " captured and saved" << std::endl;
                calibrator.addToCalibrationList(detectionResult);
                if (saveCalibrationImages) {
                    imgPath = datetime + "/" + std::to_string(imgCounter) + ".png";
                    imwrite(imgPath.c_str(), image);
                }
                imgCounter++;
            }
            else {
                std::cout << "Frame rejected" << std::endl;
            }
        }
        ros::spinOnce();
        if (ros::isShuttingDown())
        {
            return 0;
        }
    }

    std::cout << "Calibrating..." << std::endl;

    auto calibResult = calibrator.performCalibration();

    if (calibResult.isValid)
    {
        std::ofstream outFile(outputFilePath);
        outFile << "# File generated by charuco_calibration" << std::endl;
        saveCameraInfo(outFile, calibResult);
        if (!outFile)
        {
            std::cerr << "Encountered an error while writing result" << std::endl;
        }
        else
        {
            std::cout << "Calibration saved to " << outputFilePath << std::endl;
            std::cout << "Check undistorted images from camera. Press esc to exit." << std::endl;
        }
    }

    while(hasImage) {
        Mat image, imageUndistorted;
        image = lastImage.clone();
        imageUndistorted = image.clone();

        undistort(image, imageUndistorted, calibResult.cameraMatrix, calibResult.distCoeffs);
        cv_bridge::CvImage imageMsg;
        imageMsg.image = imageUndistorted;
        imageMsg.encoding = sensor_msgs::image_encodings::BGR8;

        feedbackPub.publish(imageMsg.toImageMsg());

        ros::spinOnce();
        if (ros::isShuttingDown())
        {
            return 0;
        }
    }

    return 0;
}
