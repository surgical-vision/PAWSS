#ifndef SCALE_ESTIMATOR_H
#define SCALE_ESTIMATOR_H

#include <opencv2/opencv.hpp>

class ScaleEstimator
{
public:
    ~ScaleEstimator();

    void initialize(const cv::Mat& prevImg, const std::vector<cv::Point2f>& prevPts);
    void update(const cv::Mat& prevImg, const std::vector<cv::Point2f>& prevPts);
    float estimateScale(const cv::Mat& currImg);
    inline void setPrevPts(const std::vector<cv::Point2f>& pts) { mPrevPts = pts; }
    inline void setPrevImg(const cv::Mat& img) { mPrevImg = img.clone(); }
    inline int getPtNum() const { return mPrevPts.size(); }

private:
    void trackPts(const cv::Mat& currImg);
    cv::Mat mPrevImg;
    std::vector<cv::Point2f> mPrevPts;
    std::vector<cv::Point2f> mCurrPts;
    std::vector<uchar> mStatus;

};

#endif
