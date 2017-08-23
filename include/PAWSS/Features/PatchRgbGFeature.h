#ifndef PATCH_RGBG_FEATURE_H
#define PATCH_RGBG_FEATURE_H
#include <PAWSS/Features/PatchFeature.h>
#include <PAWSS/Features/RgbFeature.h>
#include <PAWSS/Features/GradFeature.h>
#include <PAWSS/Config.h>
#include <PAWSS/segModel.h>

class PatchRgbGFeature : public PatchFeature
{
public:
    PatchRgbGFeature(const Config& conf);
    void UpdateWeightModel(const Sample &s);
    cv::Mat getWeightImg(const cv::Mat &frame, const FloatRect &bb);

private:
    void UpdateFeatureVector(const Sample& s);
    void PrepEval(const multiSample& samples);

    int mRgbBinNum;
    int mGradBinNum;
    int mBinNum;
    Config::kernelType mKernelType;

    RgbFeature mRgbFeature;
    GradFeature mGradFeature;

    segModel mWeightModel;

    std::vector<cv::Mat> mIntegs;
    cv::Mat mWeightInteg;

    double mColorWeight;
    double mGradWeight;

    bool mPatchWeightInitialized;

};

#endif
