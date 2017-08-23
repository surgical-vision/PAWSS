#ifndef TRACKER_H
#define TRACKER_H
#include <opencv/cv.h>
#include <opencv2/highgui.hpp>
#include <PAWSS/Rect.h>

class Config;
class ImageRep;
class PatchFeature;
class Kernel;
class structuredSVM;
class Sample;
class ScaleEstimator;
class motModel;
class pixelSim;


class Tracker
{
public:
    Tracker(const Config& conf);
    ~Tracker();

    void Initialise(const cv::Mat& frame, const FloatRect& bb);
    void Reset();
    void Debug(const cv::Mat frame);
    void Track(const cv::Mat &frame);
    void UpdateClassifier(const ImageRep& image);
    void UpdateWeightModel(const Sample& s);
    void UpdateDebugImage(const std::vector<FloatRect>& samples, const FloatRect& centre, const std::vector<double>& scores);

    inline bool isInitialised() const {return mInitialised; }
    inline const FloatRect& getBB() const { return mBb; }
    inline float getScale() const { return mScale; }
private:

    void genOneScaleBBs(const ImageRep& img, const FloatRect& centre, std::vector<FloatRect>& keptRects);
    void genGradualScaleBBs(const ImageRep& img, const FloatRect& centre, std::vector<FloatRect>& rects);
    void genAllScaleBBs(const ImageRep& img, const FloatRect& centre, const float scale, std::vector<FloatRect>& rects);

    void getBestBB(const std::vector<double>& scores, int &bestInd);
    void getBestBB(const std::vector<FloatRect>& rects, const std::vector<double>& scores, int &bestInd);

    bool mInitialised;
    bool mNeedColor;
    bool mNeedHsv;

    FloatRect mInitBb;
    FloatRect mBb;
    float mScale;
    std::vector<float> mScales;

    const Config& mConfig;
    structuredSVM* mClassifier;
    PatchFeature* mFeature;
    Kernel* mKernel;
    ScaleEstimator* mScaleEstimator;
    cv::Mat mDebugImage;

    motModel* mMotEstimator;
    pixelSim* mPixelSim;

};

#endif