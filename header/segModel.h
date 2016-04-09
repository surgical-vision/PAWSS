#ifndef SEG_MODEL
#define SEG_MODEL
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include "Rect.h"

class segModel{
public:
    segModel(int numBin);
    ~segModel();

    void updateWProb(const cv::Mat& bin_map, const Eigen::VectorXd& weights,
                     const std::vector<IntRect>& patch_rects,
                     const IntRect& bound_rect, const IntRect &outer_rect);

    inline double getForeProb(const int binIdx) const {return mProb_c1_y[binIdx]; }
    inline double getBackProb(const int binIdx) const {return mProb_c0_y[binIdx]; }

    void getProbImg(const cv::Mat &binImg, const IntRect &rect, cv::Mat &probImg);
    void getProbImg(const cv::Mat& img, const IntRect& rect, cv::Mat& binImg, cv::Mat& probImg);

    bool mInitialized;
    int mNumBin;
    std::vector<double> mProb_c0_y;
    std::vector<double> mProb_c1_y;
    std::vector<double> mProb_y_c0;
    std::vector<double> mProb_y_c1;
};

#endif
