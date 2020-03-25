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


namespace charuco_calibration
{

class CalibrationServices : public nodelet::Nodelet
{
private:

    // Current calibrator state
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

    // Image saving properties
    bool saveImages = false;
    std::string imgPath;
    std::atomic<int> imgCounter;

    // Latest data from ChArUco detector
    CalibratorDetectionResult lastDetectResult;
    // Calibration result
    CalibrationResult calibResult;
    // Calibrator object
    std::unique_ptr<Calibrator> calibrator;

    // Publisher for "feedback" images - current detection status, current calibration process, etc
    image_transport::Publisher feedbackPub;
    // Image source
    image_transport::Subscriber sourceSub;
    // Command service manager
    ros::ServiceServer commandServer;
    // Publisher for board image
    ros::Publisher boardPub;

    // Last logged message (that we are aware of)
    std::string loggerMessage;
    // Mutex protecting loggerMessage
    std::mutex loggerMessageMutex;

    /**
     * Save passed image to current image path.
     * 
     * @param image OpenCV image to be saved.
     * @note This function is supposed to be executed in a separate thread;
     *       make sure you're passing a copy of the image that won't be accessed
     *       anywhere else or otherwise synchronize your access to the saved image!
     */
    void saveImage(cv::Mat image)
    {
        // We use atomic counter to avoid problems with several threads accessing
        // the image number.
        int currentImageNum = imgCounter.fetch_add(1, std::memory_order::memory_order_relaxed);
        std::string imagePath = imgPath + "/" + std::to_string(currentImageNum) + ".png";
        NODELET_INFO("Saving image %s", imagePath.c_str());
        cv::imwrite(imagePath, image);
    }

    /**
     * Calibration function. Should be executed on a separate thread.
     */
    void calibThread()
    {
        calibratorState = CalibratorState::Calibrating;
        calibResult = calibrator->performCalibration();
        calibratorState = CalibratorState::Previewing;
    }

    /**
     * Log function that is passed to the calibrator. Calibrator will call this function
     * to report its current progress.
     * 
     * @param logLevel Severity of the log message.
     * @param message A human-readable message.
     * @note This function will also attempt to set the "loggerMessage" field; this field
     *       contains the last message passed to logFunction. loggerMessage won't be updated
     *       if it's currently accessed by the image callback, though; since loggerMessage
     *       is only used to draw text on a feedback image, we don't consider it a big problem.
     */
    void logFunction(charuco_calibration::LogLevel logLevel, const std::string& message)
    {
        // We try to update the message but don't complain much if we can't
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
        // NODELET_INFO(ARGS) is a macro that expands to ROS_INFO(getName(), ARGS), which
        // itself expands to ROS_LOG(ros::...::Info, std::string(ROSCONSOLE_NAME_PREFIX) + getName(), ARGS).
        // We basically expand the macros ourselves.
        ROS_LOG(rosLogLevel, std::string(ROSCONSOLE_NAME_PREFIX) + getName(), "%s", message.c_str());
    }

    /**
     * Calibrator command handler.
     * 
     * @param request A CalibratorCommandRequest containing the command.
     * @param response Our response to the command.
     * @note This function always returns true, since its value corresponds to
     *       "whether or not we managed to somehow handle the request". Actual
     *       success flag is contained within the response, along with a human-readable
     *       message about our reaction.
     */
    bool commandCallback(charuco_calibration::CalibratorCommandRequest& request,
                         charuco_calibration::CalibratorCommandResponse& response)
    {
        // We don't want to handle any commands while calibration is performed;
        // we *might* want to do more after the calibration, though.
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
                        // We really want to pass a copy of the image to the thread,
                        // otherwise our images are going to be corrupted
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
            // We don't have enum classes in ROS messages, so we just complain about unknown
            // command if any goes our way.
            default:
                response.success = false;
                response.message = "Incorrect command: " + std::to_string(request.command);
        }
        return true;
    }

    /**
     * Publish image with current calibration process.
     * 
     * @param img A cv_bridge-based image.
     * @note We try to access loggerMessage here, but don't really complain if we can't;
     *       the worst that's going to happen is an empty message instead of the current
     *       progress.
     */
    void publishProgress(const cv_bridge::CvImageConstPtr &img)
    {
        // Make sure there's *something* for OpenCV to draw
        std::string message = " ";
        if (loggerMessageMutex.try_lock())
        {
            message = loggerMessage;
            loggerMessageMutex.unlock();
        }
        int baseline = 0;
        auto textSize = cv::getTextSize(message, cv::FONT_HERSHEY_TRIPLEX, 1.0, 1.0, &baseline);
        // We don't want to change the original image
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

    // Undistortion maps
    cv::Mat undistortMapX, undistortMapY;

    /**
     * Publish preview (undistorted) image.
     * 
     * @param img A cv_bridge-based image.
     * @note The first call to this function may take some time
     *       (if undistortion maps are not initialized). Next calls
     *       will be performed much faster.
     * @note Do not call this function before calibration is done!
     *       This will lead to crashes.
     */
    void publishPreview(const cv_bridge::CvImageConstPtr& img)
    {
        // Initialize undistortion maps before first use
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

    /**
     * Image callback. Should only be called by ROS.
     * 
     * @param img ROS message containing image data.
     * @note This function may do several different things based on the
     *       current state.
     */
    void imageCallback(const sensor_msgs::ImageConstPtr &img)
    {
        cv_bridge::CvImageConstPtr cvImg = cv_bridge::toCvShare(img, "bgr8");
        // FIXME: Maybe we don't really want to do anything if we don't have
        // any subscribers?
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

    /**
     * Initialize save paths. Create directory for calibration images and calibration file.
     * Should only be called once.
     */
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
        // We kind of need to create our calibrator late, since its constructor
        // is not very trivial.
        calibrator.reset(new Calibrator());
        // Our log function is a member function, so we can't pass it as-is; instead, we
        // have to rely on partial function application.
        calibrator->setLogger(std::bind(&CalibrationServices::logFunction,
            this, std::placeholders::_1, std::placeholders::_2));
        readCalibratorParams(nhPriv, *calibrator);
        readDetectorParameters(nhDetector, calibrator->arucoDetectorParams);

        // We initialize our save paths anyway, since the calibration file may end up
        // in the same place where images are supposed to be.
        saveImages = nhPriv.param<bool>("save_images", true);
        initSavePaths();

        // image_transport does not work for latched images for some reason,
        // so we have to work around that.
        boardPub = nhPriv.advertise<sensor_msgs::Image>("board", 1, true);

        int boardImgWidth = nhPriv.param("board_image_width", 1536);
        int boardImgHeight = nhPriv.param("board_image_height", 2048);
        int boardImgBorder = nhPriv.param("board_image_border", 100);

        // FIXME: This should probably be done in a thread (then again, why would your 
        // board creation and writing take up a non-trivial amount of time?)
        auto boardImg = calibrator->getBoardImage(boardImgWidth, boardImgHeight, boardImgBorder);

        cv::imwrite("board.png", boardImg);
        cv_bridge::CvImage boardImgBridge;
        boardImgBridge.image = boardImg;
        // We ignore all of the header stuff and hope other ROS tools won't mind
        boardImgBridge.encoding = sensor_msgs::image_encodings::MONO8;
        boardPub.publish(boardImgBridge.toImageMsg());

        image_transport::ImageTransport it(nh), itPriv(nhPriv);

        feedbackPub = itPriv.advertise("feedback", 1);
        imgCounter = 1;

        calibratorState = CalibratorState::Acquiring;

        sourceSub = it.subscribe("image", 1, &CalibrationServices::imageCallback, this);
        commandServer = nhPriv.advertiseService("calib_command", &CalibrationServices::commandCallback, this);
    }
};

}

PLUGINLIB_EXPORT_CLASS(charuco_calibration::CalibrationServices, nodelet::Nodelet);
