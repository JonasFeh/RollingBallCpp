#include <iostream>
#include <cstdarg>

#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>

#include "../include/RollingBall.h"

void printProcessingTime(std::chrono::duration<long long, std::nano> elapsed)
{
    std::cout << std::setprecision(12) << std::scientific << static_cast<long double>(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()) / 1000000000 << " s" << std::endl;
}

void showManyImages(std::string title, int nArgs, ...)
{
    int size;
    int i;
    int m, n;
    int x, y;

    // w - Maximum number of images in a row
    // h - Maximum number of images in a column
    int w, h;

    // scale - How much we have to resize the image
    float scale;
    int max;

    // If the number of arguments is lesser than 0 or greater than 12
    // return without displaying
    if (nArgs <= 0) {
        printf("Number of arguments too small....\n");
        return;
    }
    else if (nArgs > 14) {
        printf("Number of arguments too large, can only handle maximally 12 images at a time ...\n");
        return;
    }
    // Determine the size of the image,
    // and the number of rows/cols
    // from number of arguments
    else if (nArgs == 1) {
        w = h = 1;
        size = 300;
    }
    else if (nArgs == 2) {
        w = 2; h = 1;
        size = 300;
    }
    else if (nArgs == 3) {
        w = 3; h = 1;
        size = 300;
    }
    else if (nArgs == 4) {
        w = 2; h = 2;
        size = 300;
    }
    else if (nArgs == 5 || nArgs == 6) {
        w = 3; h = 2;
        size = 200;
    }
    else if (nArgs == 7 || nArgs == 8) {
        w = 4; h = 2;
        size = 200;
    }
    else {
        w = 4; h = 3;
        size = 150;
    }

    // Create a new 3 channel image
    cv::Mat DispImage = cv::Mat::zeros(cv::Size(100 + size * w, 60 + size * h), CV_8UC1);

    // Used to get the arguments passed
    va_list args;
    va_start(args, nArgs);

    // Loop for nArgs number of arguments
    for (i = 0, m = 20, n = 20; i < nArgs; i++, m += (20 + size)) {
        // Get the Pointer to the IplImage
        cv::Mat img = va_arg(args, cv::Mat);

        // Check whether it is NULL or not
        // If it is NULL, release the image, and return
        if (img.empty()) {
            printf("Invalid arguments");
            return;
        }

        // Find the width and height of the image
        x = img.cols;
        y = img.rows;

        // Find whether height or width is greater in order to resize the image
        max = (x > y) ? x : y;

        // Find the scaling factor to resize the image
        scale = (float)((float)max / size);

        // Used to Align the images
        if (i % w == 0 && m != 20) {
            m = 20;
            n += 20 + size;
        }

        // Set the image ROI to display the current image
        // Resize the input image and copy the it to the Single Big Image
        cv::Rect ROI(m, n, (int)(x / scale), (int)(y / scale));
        cv::Mat temp; resize(img, temp, cv::Size(ROI.width, ROI.height));
        temp.copyTo(DispImage(ROI));
    }
    // Create a new window, and show the Single Big Image
    cv::namedWindow(title, 1);
    imshow(title, DispImage);
    cv::waitKey();

    // End the number of arguments
    va_end(args);
}

cv::Mat getImage(std::string theFileName)
{
    auto sourceImage = cv::imread(theFileName, cv::IMREAD_UNCHANGED);
    cv::Mat grayScaleImage;

    if (sourceImage.rows == 0 || sourceImage.cols == 0)
    {
        std::cout << "Unable to read image!" << std::endl;
    }

    cv::cvtColor(sourceImage, grayScaleImage, cv::COLOR_BGRA2GRAY);
    return grayScaleImage;
}

int main(int argc, char* argv[])
{
    std::string fileName = "../../../res/RollingBallTestImage.png";
    auto grayScaleImage = getImage(fileName);
    cv::bitwise_not(grayScaleImage, grayScaleImage);
    cv::Mat foreground, background;
    auto start = std::chrono::high_resolution_clock::now();
    jonascv::rollingBall(grayScaleImage, background, foreground, 5);
    auto elapsed = std::chrono::high_resolution_clock::now() - start;

    printProcessingTime(elapsed);

    showManyImages("Source - Background - Foreground", 3, grayScaleImage, background, foreground);
}