#ifndef MOT_MODEL
#define MOT_MODEL
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <PAWSS/Rect.h>
#include <PAWSS/ImageRep.h>
#include <PAWSS/mUtils.h>
#include <PAWSS/Features/MotFeature.h>

class motModel{
public:
    motModel();
    ~motModel();

    void setPrevImg(ImageRep &img) { mPrevImg = img.GetGrayImage(); }
    bool hasSetPrevImg() {return !mPrevImg.empty(); }
    void showMotPropMap(const cv::Mat& propMap);

    void evalMotion(const cv::Mat &curr, const IntRect &rect, cv::Mat &flow);
    void getValidMotion(const cv::Vec2f &mot, const IntRect &rect, const cv::Mat &flow, cv::Mat &propMap);

    void getValidMotion1(const cv::Vec2f &mot, const IntRect &rect, const cv::Mat &flow, cv::Mat &propMap);

    void drawOptFlowMap(const cv::Mat& flow, cv::Mat& cflowmap, int step);
    void getHistogram(const cv::Mat& curr, const IntRect &rect);

    void evalMotionT(const cv::Mat &curr);
    void setTrueMotion(const FloatRect& rect);
    void setDenseMotion(const cv::Mat& flow);
    void getValidMotionT(const IntRect& rect, cv::Mat &propMap);

private:
    cv::Mat mPrevImg;


    // test histogram
    MotFeature mMotFeature;

    // test trajectory
    std::vector<cv::Mat> mBFlows;
    std::vector<cv::Point2f> mTrueMots;
    std::vector<FloatRect> mTrueBbs;



};

#endif
