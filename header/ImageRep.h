#ifndef IMAGE_REF_H
#define IMAGE_REF_H
#include <opencv2/opencv.hpp>
#include "Rect.h"


class ImageRep
{
public:
    ImageRep(const cv::Mat &image, bool hsv_flag,
             bool color=true);

    inline const cv::Mat& GetImage(int channel = 0) const { return mImgs[channel]; }
    inline const cv::Mat& GetBaseImage() const { return mBaseImg; }
    inline const cv::Mat& GetGrayImage() const { return mGrayImg; }
    inline const cv::Mat& GetColorImage() const { return mColorImg; }
    inline const IntRect& GetRect() const { return mRect; }

private:
    cv::Mat mBaseImg;
    cv::Mat mColorImg;
    cv::Mat mGrayImg;
    std::vector<cv::Mat> mImgs;
    int mChannels;
    IntRect mRect;


};

#endif
