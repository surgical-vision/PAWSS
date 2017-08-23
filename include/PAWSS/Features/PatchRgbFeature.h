#ifndef PATCH_RGB_FEATURE_H
#define PATCH_RGB_FEATURE_H
#include <PAWSS/Features/PatchFeature.h>
#include <PAWSS/Features/RgbFeature.h>
#include <PAWSS/Config.h>
#include <PAWSS/segModel.h>

class PatchRgbFeature : public PatchFeature
{
public:
    PatchRgbFeature(const Config& conf);
    void UpdateWeightModel(const Sample &s);
    cv::Mat getWeightImg(const cv::Mat& frame, const FloatRect& bb);

private:
    void UpdateFeatureVector(const Sample& s);
    void PrepEval(const multiSample& samples);

    int mBinNum;
    Config::kernelType mKernelType;

    RgbFeature mRgbFeature;
    segModel mWeightModel;

    std::vector<cv::Mat> mIntegs;
    cv::Mat mWeightInteg;

    bool mPatchWeightInitialized;

};

#endif
