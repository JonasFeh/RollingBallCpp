#include<opencv2/imgproc.hpp>

namespace jonascv
{
    enum class BackgroundLevel
    {
        Middle = 0,
        Bottom = -1,
        Top = 1,
    };

    void rollingBall(cv::Mat& inputImage, cv::Mat& backgroundImage, cv::Mat& foregroundImage, int ballRadius, BackgroundLevel backgroundlevel = BackgroundLevel::Top);
}