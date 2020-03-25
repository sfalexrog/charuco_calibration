#include "calibrator.h"
#include "utils.h"

#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/SetCameraInfo.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

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

#include <memory>
#include <thread>
#include <atomic>
#include <mutex>

using namespace cv;

cv::Mat lastImage;
bool hasImage = false;
int calibratorCommand = 0;

namespace charuco_calibration
{

class CalibrationServices : public nodelet::Nodelet
{
private:

    enum class CalibratorState
    {
        // Acquiring images for calibration
        Acquiring,
        // Performing calibration
        Calibrating,
        // Showing calibration preview
        Previewing
    };

    std::atomic<CalibratorState> calibratorState;

    bool saveImages = false;
    std::string imgPath;
    std::atomic<int> imgCounter;

    CalibratorDetectionResult lastDetectResult;
    CalibrationResult calibResult;
    std::unique_ptr<Calibrator> calibrator;

    image_transport::Publisher feedbackPub;
    image_transport::Subscriber sourceSub;
    ros::ServiceServer commandServer;
    ros::Publisher boardPub;

    std::string loggerMessage;
    std::mutex loggerMessageMutex;

    // Image saving function (should be executed in a separate thread)
    void saveImage(cv::Mat image)
    {
        int currentImageNum = imgCounter.fetch_add(1, std::memory_order::memory_order_relaxed);
        std::string imagePath = imgPath + "/" + std::to_string(currentImageNum) + ".png";
        NODELET_INFO("Saving image %s", imagePath.c_str());
        cv::imwrite(imagePath, image);
    }

    // Calibration thread
    void calibThread()
    {
        calibratorState = CalibratorState::Calibrating;
        calibResult = calibrator->performCalibration();
        calibratorState = CalibratorState::Previewing;
    }

    void logFunction(charuco_calibration::LogLevel logLevel, const std::string& message)
    {
        if (loggerMessageMutex.try_lock())
        {
            loggerMessage = message;
            loggerMessageMutex.unlock();
        }
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
        ROS_LOG(rosLogLevel, std::string(ROSCONSOLE_NAME_PREFIX) + getName(), "%s", message.c_str());
    }

    bool commandCallback(charuco_calibration::CalibratorCommandRequest& request,
                         charuco_calibration::CalibratorCommandResponse& response)
    {
        if (calibratorState != CalibratorState::Acquiring)
        {
            response.success = false;
            response.message = "Invalid state for calibrator command";
            return true;
        }
        switch(request.command)
        {
            case charuco_calibration::CalibratorCommandRequest::COMMAND_CAPTURE:
               if (lastDetectResult.isValid())
                {
                    calibrator->addToCalibrationList(lastDetectResult);
                    if (saveImages)
                    {
                        std::thread(&CalibrationServices::saveImage, 
                            this, lastDetectResult.sourceImage.clone()).detach();
                    }
                    response.success = true;
                    response.message = "Added image #" + std::to_string(calibrator->getStoredCalibrationImages().size());
                }
                else
                {
                    response.success = false;
                    response.message = "Refused to add invalid image to calibration set";
                }
            break;
            case charuco_calibration::CalibratorCommandRequest::COMMAND_CALIBRATE:
                {
                    std::thread(&CalibrationServices::calibThread, this).detach();
                    response.success = true;
                    response.message = "Starting calibration";
                }
            break;
            default:
                response.success = false;
                response.message = "Incorrect command: " + std::to_string(request.command);
        }
        return true;
    }

    void publishProgress(const cv_bridge::CvImageConstPtr &img)
    {
        const int FONT_HEIGHT = 16;
        std::string message = " ";
        if (loggerMessageMutex.try_lock())
        {
            message = loggerMessage;
            loggerMessageMutex.unlock();
        }
        int baseline = 0;
        auto textSize = cv::getTextSize(message, cv::FONT_HERSHEY_TRIPLEX, 1.0, 1.0, &baseline);
        cv::Mat processImage = img->image.clone();
        cv::putText(processImage, "Calibration in progress",
            {0, processImage.size().height - textSize.height},
            cv::FONT_HERSHEY_TRIPLEX,
            1.0, 
            {0, 255, 0});
        cv::putText(processImage, loggerMessage,
            {0, processImage.size().height}, cv::FONT_HERSHEY_TRIPLEX,
            1.0,
            {0, 255, 0});
        sensor_msgs::ImagePtr feedbackMsg = cv_bridge::CvImage(
            img->header, "bgr8",
            processImage).toImageMsg();
        feedbackPub.publish(feedbackMsg);
    }

    cv::Mat undistortMapX, undistortMapY;

    void publishPreview(const cv_bridge::CvImageConstPtr& img)
    {
        if (undistortMapX.size().width == 0 || undistortMapY.size().width == 0 ||
            undistortMapY.size().height == 0 || undistortMapY.size().height == 0)
        {
            cv::Mat newCameraMatrix = cv::getOptimalNewCameraMatrix(
                calibResult.cameraMatrix, calibResult.distCoeffs,
                img->image.size(), 1.0, img->image.size());
            cv::initUndistortRectifyMap(calibResult.cameraMatrix,
                calibResult.distCoeffs, cv::Mat(), newCameraMatrix,
                img->image.size(), CV_32FC1, undistortMapX, undistortMapY);
        }
        cv::Mat output;
        cv::remap(img->image, output, undistortMapX, undistortMapY, cv::INTER_LINEAR);
        sensor_msgs::ImagePtr feedbackMsg = cv_bridge::CvImage(
            img->header, "bgr8",
            output).toImageMsg();
        feedbackPub.publish(feedbackMsg);
    }

    void imageCallback(const sensor_msgs::ImageConstPtr &img)
    {
        cv_bridge::CvImageConstPtr cvImg = cv_bridge::toCvShare(img, "bgr8");
        bool hasSubscribers = feedbackPub.getNumSubscribers() > 0;
        switch (calibratorState)
        {
            case CalibratorState::Acquiring:
                lastDetectResult = calibrator->processImage(cvImg->image);
                if (hasSubscribers) {
                    sensor_msgs::ImagePtr feedbackMsg = cv_bridge::CvImage(
                        img->header, "bgr8",
                        calibrator->drawDetectionResults(lastDetectResult)).toImageMsg();
                    feedbackPub.publish(feedbackMsg);
                }
                break;
            case CalibratorState::Calibrating:
                publishProgress(cvImg);
                break;
            case CalibratorState::Previewing:
                publishPreview(cvImg);
            break;
        }
    }

    void initSavePaths()
    {
        auto t = time(nullptr);
        auto tm = *localtime(&t);
        std::ostringstream oss;
        oss << "calibration_" << std::put_time(&tm, "%Y%m%d_%H%M%S");
        auto datetime = oss.str();
        mkdir(datetime.c_str(), 0775);
        imgPath = datetime;
    }

    void onInit() override
    {
        NODELET_INFO("Starting charuco calibration nodelet");
        ros::NodeHandle& nh = getNodeHandle();
        ros::NodeHandle& nhPriv = getPrivateNodeHandle();
        ros::NodeHandle nhDetector = ros::NodeHandle(nhPriv, "detector_parameters");
        calibrator.reset(new Calibrator());
        calibrator->setLogger(std::bind(&CalibrationServices::logFunction,
            this, std::placeholders::_1, std::placeholders::_2));
        readCalibratorParams(nhPriv, *calibrator);
        readDetectorParameters(nhDetector, calibrator->arucoDetectorParams);

        saveImages = nhPriv.param<bool>("save_images", true);
        initSavePaths();

        boardPub = nhPriv.advertise<sensor_msgs::Image>("board", 1, true);
        int boardImgWidth = nhPriv.param("board_image_width", 1536);
        int boardImgHeight = nhPriv.param("board_image_height", 2048);
        int boardImgBorder = nhPriv.param("board_image_border", 100);

        auto boardImg = calibrator->getBoardImage(boardImgWidth, boardImgHeight, boardImgBorder);

        cv::imwrite("board.png", boardImg);
        cv_bridge::CvImage boardImgBridge;
        boardImgBridge.image = boardImg;
        boardImgBridge.encoding = sensor_msgs::image_encodings::MONO8;
        boardPub.publish(boardImgBridge.toImageMsg());

        image_transport::ImageTransport it(nh), itPriv(nhPriv);

        feedbackPub = itPriv.advertise("feedback", 1);
        imgCounter = 1;

        calibratorState = CalibratorState::Acquiring;

        sourceSub = it.subscribe("image", 1, &CalibrationServices::imageCallback, this);
        commandServer = nhPriv.advertiseService("calib_command", &CalibrationServices::commandCallback, this);
    }

public:

};

}

PLUGINLIB_EXPORT_CLASS(charuco_calibration::CalibrationServices, nodelet::Nodelet);
