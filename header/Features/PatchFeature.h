#ifndef PATCH_FEATURE_H
#define PATCH_FEATURE_H
#include <Eigen/Core>
#include <vector>
#include <opencv2/opencv.hpp>
#include "Features/Feature.h"

class ImageRep;


static const float kAlpha = 0.1;

class PatchFeature : public Feature
{
public:

    inline const Eigen::VectorXd& Eval(const Sample& s) const
    {
        const_cast<PatchFeature*>(this)->UpdateFeatureVector(s);
        return mFeatVec;
    }
    virtual void Eval(const multiSample& samples, std::vector<Eigen::VectorXd>& featVecs);
    virtual void UpdateWeightModel(const Sample& s)=0;
    virtual void setPrevImg(ImageRep& img) { }

    cv::Mat getPatchWeightImg(const FloatRect &bb);
    cv::Mat getPatchWeightFrame(const cv::Mat& frame, const FloatRect& bb);
    void showPatchWeightFrame(const cv::Mat& frame, const FloatRect& bb);
    void showPatchWeightImg(const int frame_idx, const FloatRect &bb);

    void extractPatchPts(const FloatRect& bb, const int ptNumPerPatch, std::vector<cv::Point2f>& pts);

protected:
    virtual void PrepEval(const multiSample& samples){}
    void setPatchRect(const cv::Size& sample_size);
    virtual void UpdateFeatureVector(const Sample& s)=0;

    int mPatchNumX;
    int mPatchNumY;
    std::vector<IntRect> mPatchRects;
    cv::Size mSampleSize;
//    std::vector<double> mPatchWeights;
    Eigen::VectorXd mPatchWeights;



};

#endif
