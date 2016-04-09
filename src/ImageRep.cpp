#include "ImageRep.h"

ImageRep::ImageRep(const cv::Mat &image, bool hsv_flag, bool color) :
    mRect(0, 0, image.cols, image.rows),
    mChannels(color ? 3:1)
{
    mBaseImg = image;
    if(color)
    {
        assert(image.channels() == 3);
        mColorImg = cv::Mat(image.rows, image.cols, CV_8UC3);
        for(int i=0; i<mChannels; ++i)
            mImgs.push_back(cv::Mat(image.rows, image.cols, CV_8UC1));
        mGrayImg = cv::Mat(image.rows, image.cols, CV_8UC1);

        cv::cvtColor(image, mGrayImg, CV_RGB2GRAY);

        if(hsv_flag)
            cv::cvtColor(image, mColorImg, CV_RGB2HSV);
        else
            mColorImg = image;

        cv::split(mColorImg, mImgs);
    }
    else
    {
        assert(image.channels() == 3 || image.channels() == 3);
        mGrayImg = cv::Mat(image.rows, image.cols, CV_8UC1);
        if(image.channels() == 3)
        {
            mColorImg = image;
            cv::cvtColor(image, mGrayImg, CV_RGB2GRAY);
        }
        else {
            mGrayImg = image; }
    }
}

