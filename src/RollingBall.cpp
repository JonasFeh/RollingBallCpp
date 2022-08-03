#include "../include/RollingBall.h"
#include <algorithm>
#include <map>
#include <limits>
#include <set>

namespace jonascv
{

    struct Coordinate
    {
        int X, Y;
    };

    struct RollingBallMapEntry
    {
        float BallHeight;
        float DistanceToCenterSquared;
    };

    struct less_than_key
    {
        inline bool operator() (const std::pair<float, uchar>& leftpair, const std::pair<float, uchar>& rightpair)
        {
            return (leftpair.first < rightpair.first);
        }
    };

    uchar getBackgroundValue(cv::Mat& inputImage, int u, int v, int ballRadius, std::map<Coordinate, RollingBallMapEntry>& ballHeightMap, BackgroundLevel backgroundlevel)
    {
        int startIndexX = std::max(0, u - ballRadius - 1);
        int endIndexX = std::min(inputImage.cols, u + ballRadius - 1);

        int startIndexY = std::max(0, v - ballRadius - 1);
        int endIndexY = std::min(inputImage.rows, v + ballRadius - 1);

        uchar backgroundValueMin = UCHAR_MAX;

        std::vector<std::pair<float, uchar>> intensitySortedByDistance = std::vector<std::pair<float, uchar>>();

        for (int y = startIndexY; y < endIndexY; ++y)
        {
            for (int x = startIndexX; x < endIndexX; ++x)
            {

                auto distanceToCenterSquared = pow(u - x, 2) + pow(v - y, 2);
                auto ballheight = sqrt(pow(ballRadius, 2) - distanceToCenterSquared);
                auto coordinateTuple = std::tuple<int, int>(x, y);
                if (isnan(ballheight))
                {
                    continue;
                }
                auto grayScaleValueDouble = static_cast<double>(inputImage.at<uchar>(x, y)) - ballheight + (double)ballRadius * static_cast<int>(backgroundlevel);

                uchar grayScaleValueChar = static_cast<uchar>(std::clamp(grayScaleValueDouble, 0.0, 255.0));
                backgroundValueMin = std::min(backgroundValueMin, grayScaleValueChar);

                if (backgroundValueMin == 0)
                {
                    return backgroundValueMin;
                }
            }
        }

        return backgroundValueMin;
    }

    std::map<Coordinate, RollingBallMapEntry> calculateRollingBallMapEntry(int ballRadius)
    {
        std::map<Coordinate, RollingBallMapEntry> rollingBallMap = std::map<Coordinate, RollingBallMapEntry>();
        for (size_t y = 0; y < ballRadius + 1; y++)
        {
            for (size_t x = 0; x < ballRadius + 1; x++)
            {
                auto distanceToCenterSquared = std::pow(x, 2) + std::pow(y, 2);

                if (distanceToCenterSquared > std::pow(ballRadius, 2))
                {
                    continue;
                }
                auto coordinate = Coordinate(x, y);
                auto rollingBallMapEntry = RollingBallMapEntry(distanceToCenterSquared, sqrt(pow(ballRadius, 2) - distanceToCenterSquared));
                rollingBallMap.emplace(coordinate, rollingBallMapEntry);
            }
        }

        return rollingBallMap;
    }


    void rollingBall(cv::Mat& inputImage, cv::Mat& backgroundImage, cv::Mat& foregroundImage, int ballRadius, BackgroundLevel backgroundlevel)
    {
        backgroundImage = cv::Mat(inputImage.rows, inputImage.cols, CV_8UC1);
        foregroundImage = cv::Mat(inputImage.rows, inputImage.cols, CV_8UC1);

        auto ballHeightMap = calculateRollingBallMapEntry(ballRadius);
        for (int v = 0; v < inputImage.rows; ++v)
        {
            for (int u = 0; u < inputImage.cols; ++u)
            {
                auto backgroundValue = getBackgroundValue(inputImage, u, v, ballRadius, ballHeightMap, backgroundlevel);
                backgroundImage.at<uchar>(u, v) = backgroundValue;
                foregroundImage.at<uchar>(u, v) = std::clamp((uchar)(inputImage.at<uchar>(u, v) - backgroundValue), (uchar)0, (uchar)255);
            }
        }
    }
}
