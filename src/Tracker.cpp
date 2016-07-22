#include "Tracker.h"
#include "mUtils.h"
#include "Sample.h"
#include "Kernels.h"
#include "Features/Feature.h"
#include "structuredSVM.h"
#include "Config.h"
#include "ImageRep.h"
#include "Features/PatchGradFeature.h"
#include "Features/PatchGrayFeature.h"
#include "Features/PatchGrayGFeature.h"
#include "Features/PatchHsvFeature.h"
#include "Features/PatchHsvGFeature.h"
#include "Features/PatchRgbFeature.h"
#include "Features/PatchRgbGFeature.h"
#include "Features/PatchMotFeature.h"
#include "Features/PatchRgbMFeature.h"
#include "Features/PatchHsvMFeature.h"
#include "scaleEstimator.h"
#include "motModel.h"
#include "pixelComp.h"

static const int kGradualScaleNum = 9;
static const int kAbruptScaleNum = 11;
static const float kScaleBase = 1.003;
static const double kUpdateSimilarity = 0.3;

Tracker::Tracker(const Config& conf) :
    mConfig(conf),
    mClassifier(0), mFeature(0), mKernel(0), mScaleEstimator(0), mMotEstimator(0), mPixelSim(0),
    mDebugImage(2*(int)conf.mSearchRadius+1, 2*(int)conf.mSearchRadius+1, CV_32FC1)
{
    Reset();
}

void Tracker::Reset()
{
    mInitialised = false;
    mDebugImage.setTo(0);

    if(mClassifier) delete mClassifier;
    if(mFeature) delete mFeature;
    if(mKernel) delete mKernel;
    if(mScaleEstimator) delete mScaleEstimator;
    if(mMotEstimator) delete mMotEstimator;
    if(mPixelSim) delete mPixelSim;


    mNeedColor = false;
    mNeedHsv = false;

    // scale
    mScale = 1.0f;
    mScales.clear();
    mScales.push_back(mScale);
    for(int i=1; i<=kGradualScaleNum/2; ++i)
    {
        mScales.push_back(pow(kScaleBase, -i));
        mScales.push_back(pow(kScaleBase, i));
    }

    // feature and kernel
    switch(mConfig.mFeatureKernelPair.mFeature) {
    case Config::kFeatureTypePatchGrad:
        mFeature = new PatchGradFeature(mConfig);
        break;
    case Config::kFeatureTypePatchGray:
        mFeature = new PatchGrayFeature(mConfig);
        break;
    case Config::kFeatureTypePatchGrayG:
        mFeature = new PatchGrayGFeature(mConfig);
        break;
    case Config::kFeatureTypePatchHsv:
        mFeature = new PatchHsvFeature(mConfig);
        mNeedColor = true;
        mNeedHsv = true;
        break;
    case Config::kFeatureTypePatchHsvG:
        mFeature = new PatchHsvGFeature(mConfig);
        mNeedColor = true;
        mNeedHsv = true;
        break;
    case Config::kFeatureTypePatchRgb:
        mFeature = new PatchRgbFeature(mConfig);
        mNeedColor = true;
        break;
    case Config::kFeatureTypePatchRgbG:
        mFeature = new PatchRgbGFeature(mConfig);
        mNeedColor = true;
        break;
    case Config::kFeatureTypePatchMot:
        mFeature = new PatchMotFeature(mConfig);
        break;
    case Config::kFeatureTypePatchRgbM:
        mFeature = new PatchRgbMFeature(mConfig);
        mNeedColor = true;
        break;
    case Config::kFeatureTypePatchHsvM:
        mFeature = new PatchHsvMFeature(mConfig);
        mNeedColor = true;
        mNeedHsv = true;
        break;
    default:
        break;
    }
    switch (mConfig.mFeatureKernelPair.mKernel) {
    case Config::kKernelTypeLinear:
        mKernel = new LinearKernel();
        break;
//    case Config::kKernelTypeGaussian:
//        mKernel = new GaussianKernel(mConfig.mFeatureKernelPair.mParams[0]);
//        break;
    case Config::kKernelTypeIntersection:
        mKernel = new IntersectionKernel();
        break;
//    case Config::kKernelTypeChi2:
//        mKernel = new Chi2Kernel();
//        break;
    default:
        break;
    }

    mClassifier = new structuredSVM(mConfig, *mFeature, *mKernel);
    mScaleEstimator = new ScaleEstimator();
    mMotEstimator = new motModel();
    mPixelSim = new pixelSim();


}

Tracker::~Tracker()
{
    delete mClassifier;
    delete mFeature;
    delete mKernel;
    delete mScaleEstimator;
    delete mMotEstimator;
    delete mPixelSim;

    mScales.clear();
}

void Tracker::Debug(const cv::Mat frame)
{
    // show score image
    cv::imshow("tracker", mDebugImage);
    // show classifier debug image
    mClassifier->Debug();
    // show Weightimage
    mFeature->showPatchWeightFrame(frame, mBb);
}

void Tracker::Initialise(const cv::Mat &frame, const FloatRect &bb)
{
    mBb = FloatRect(bb);
    mInitBb = FloatRect(bb);
    ImageRep image(frame, mNeedHsv, mNeedColor);

    // initialize the previous image
    mFeature->setPrevImg(image);
//    mMotEstimator->setPrevImg(image);
//    mMotEstimator->setTrueMotion(mBb);


    UpdateClassifier(image);
    Sample s(image, mBb);
    UpdateWeightModel(s);

    // initialize the scale estimator
    std::vector<cv::Point2f> pts;
    mFeature->extractPatchPts(mBb, 5, pts);
    mScaleEstimator->initialize(image.GetGrayImage(), pts);


    mInitialised = true;
}


void Tracker::Track(const cv::Mat& frame)
{
    assert(mInitialised);

    // save the bounding box center
    FloatRect prevBb = mBb;
    cv::Vec2f prevBbCenter = cv::Vec2f(prevBb.XCentre(), prevBb.YCentre());

    ImageRep image(frame, mNeedHsv, mNeedColor);
    std::vector<FloatRect> keptRects;
    genOneScaleBBs(image, mBb, keptRects);
    multiSample samples(image, keptRects);

    std::vector<double> scores;
    // evaluate all samples
    // extract feature vector for all the samples
    std::vector<Eigen::VectorXd> featVecs;
    const_cast<PatchFeature*>(mFeature)->Eval(samples, featVecs);
    mClassifier->EvalMultiSamples(featVecs, scores);


    UpdateDebugImage(keptRects, mBb, scores);

    int bestInd = -1;
    getBestBB(scores, bestInd);
//    if(bestInd == -1)
//    {
//        std::cout<<"error: evaluate best sample return -1"<< std::endl;
//        return;
//    }
    assert(bestInd != -1);

    // update classifier and weight model
    if(mConfig.mScaleType == Config::kScaleTypeOne)
    {
        mBb = keptRects[bestInd];
        // update classifier and weight model
        //   extract feature vector for the true sample
        const Eigen::VectorXd& tfv = const_cast<PatchFeature*>(mFeature)->Eval(samples.getSample(bestInd));
        double similarity = mClassifier->EvalTrueSample(tfv);
        if(similarity > kUpdateSimilarity)
        {
            UpdateClassifier(image);
            UpdateWeightModel(samples.getSample(bestInd));
        }
        return;
    }

    // multi-scale samples
    std::vector<FloatRect> scaleRects;
    if(mConfig.mScaleType == Config::kScaleTypeGradual)
    {
        genGradualScaleBBs(image, keptRects[bestInd], scaleRects);
    }
    else if (mConfig.mScaleType == Config::kScaleTypeAll)
    {
        // scaler estimation
        float scale = mScaleEstimator->estimateScale(image.GetGrayImage());
        genAllScaleBBs(image, keptRects[bestInd], scale, scaleRects);
    }
    multiSample scaleSamples(image, scaleRects);
    scores.clear();
    // extract feature vector for all the samples
    const_cast<PatchFeature*>(mFeature)->Eval(scaleSamples, featVecs);
    mClassifier->EvalMultiSamples(featVecs, scores);
    getBestBB(scaleRects, scores, bestInd);
//    if(bestInd == -1)
//    {
//        std::cout<<"error: evaluate best sample return -1"<< std::endl;
//        return;
//    }
    assert(bestInd != -1);
    mBb = scaleRects[bestInd];
    float wr = 1.0 * mBb.Width() / mInitBb.Width();
    float hr = 1.0 * mBb.Height() / mInitBb.Height();
    mScale = std::max(wr, hr);

    // update the scale estimator
    std::vector<cv::Point2f> pts;
    mFeature->extractPatchPts(mBb, 5, pts);
    mScaleEstimator->update(image.GetGrayImage(), pts);

    // update classifier and weight model
    //  extract feature vector for the true sample
    const Eigen::VectorXd& tfv = const_cast<PatchFeature*>(mFeature)->Eval(scaleSamples.getSample(bestInd));
    double similarity = mClassifier->EvalTrueSample(tfv);
#if VERBOSE
    std::cout<<"similarity: "<<similarity<<std::endl;
#endif
    if(similarity > kUpdateSimilarity)
    {
        UpdateClassifier(image);
        UpdateWeightModel(scaleSamples.getSample(bestInd));
    }
    else
    {
#if VERBOSE
        std::cout<<"similarity: "<<similarity<<" too low, no update!"<<std::endl;
#endif
    }

    if(!mMotEstimator->hasSetPrevImg())
    {
        mMotEstimator->setPrevImg(image);
        mMotEstimator->setTrueMotion(mBb);
    }
    else
    {
        // the motion estimator
    //    cv::Mat flow;
    //    mMotEstimator->evalMotion(image.GetGrayImage(), mBb, flow);
        mMotEstimator->setTrueMotion(mBb);
        mMotEstimator->evalMotionT(image.GetGrayImage());
        cv::Mat motPropMap = cv::Mat(image.GetRect().Height(), image.GetRect().Width(), CV_32FC1);
        // todo: the true motion is simplified here.
    //    cv::Vec2f trueMot = cv::Vec2f(mBb.XCentre(), mBb.YCentre()) - prevBbCenter;
        // be careful:
        mMotEstimator->getValidMotionT(mBb, motPropMap);
        // get the map 2times bigger
        float zoom_fac = 2;
        float x_min = fmax(0, mBb.XCentre() - zoom_fac * mBb.Width()/2);
        float y_min = fmax(0, mBb.YCentre() - zoom_fac * mBb.Height()/2);
        float x_max = fmin(image.GetRect().Width(), mBb.XCentre() + zoom_fac * mBb.Width()/2);
        float y_max = fmin(image.GetRect().Height(), mBb.YCentre() + zoom_fac * mBb.Height()/2);
        FloatRect biggerBB(x_min, y_min, x_max-x_min, y_max-y_min);
        mMotEstimator->getValidMotionT(biggerBB, motPropMap);
        // show the MotPropMap
        mMotEstimator->showMotPropMap(motPropMap);

        // test: show obj/back histogram
        mMotEstimator->getHistogram(image.GetGrayImage(), prevBb);
        mMotEstimator->setPrevImg(image);

        // test: similarity
        cv::Mat simMap = cv::Mat(image.GetRect().Height(), image.GetRect().Width(), CV_32FC1);
        mPixelSim->evalSimMap(image.GetBaseImage(), mBb, simMap);
        mPixelSim->showSimPropMap(simMap);
    }


}

void Tracker::UpdateClassifier(const ImageRep &image)
{
    std::vector<FloatRect> rects, keptRects;
//    RadialSamples(mBb, mConfig.mSearchRadius, 5, 16, rects);
    RadialSamples(mBb, 2*mConfig.mSearchRadius*mScale, 5, 16, rects);

    keptRects.push_back(rects[0]);  // the true sample
    for(int i=1; i<(int)rects.size(); ++i)
    {
        if (!rects[i].IsInside(image.GetRect())) continue;
        keptRects.push_back(rects[i]);
    }

# if VERBOSE
    std::cout << keptSamples.size() << std::endl;
#endif

    multiSample samples(image, keptRects);
    // extract feature vector for the samples
    std::vector<Eigen::VectorXd> fvs;
    const_cast<PatchFeature*>(mFeature)->Eval(samples, fvs);
    mClassifier->Update(samples, fvs, 0);

}

void Tracker::UpdateWeightModel(const Sample &s)
{
    const_cast<PatchFeature*>(mFeature)->UpdateWeightModel(s);
}


void Tracker::UpdateDebugImage(const std::vector<FloatRect>& samples, const FloatRect& centre, const std::vector<double>& scores)
{
    double mn = Eigen::VectorXd::Map(&scores[0], scores.size()).minCoeff();
    double mx = Eigen::VectorXd::Map(&scores[0], scores.size()).maxCoeff();
    mDebugImage.setTo(0);
    for(int i=0; i < (int) samples.size(); ++i)
    {
        int x = (int)(samples[i].XMin() - centre.XMin());
        int y = (int)(samples[i].YMin() - centre.YMin());
        mDebugImage.at<float>(mConfig.mSearchRadius+y, mConfig.mSearchRadius+x) = (float)((scores[i] - mn)/(mx-mn));
    }
}

void Tracker::genOneScaleBBs(const ImageRep &img, const FloatRect &centre, std::vector<FloatRect> &keptRects)
{
    std::vector<FloatRect> rects;
    keptRects.clear();
    float radius = mConfig.mSearchRadius;
    PixelSamples(centre, radius, true, rects);
    for(int i=0; i<(int)rects.size(); ++i)
    {
        if (!rects[i].IsInside(img.GetRect())) continue;
        keptRects.push_back(rects[i]);
    }
//    std::cout<<"one scale sample number: "<<keptRects.size() << std::endl;
}

void Tracker::genGradualScaleBBs(const ImageRep &img, const FloatRect &centre, std::vector<FloatRect> &rects)
{
    float w = centre.Width();
    float h = centre.Height();
    float xc = centre.XCentre();
    float yc = centre.YCentre();

    FloatRect r;
    for(int i = 0; i< mScales.size(); ++i)
    {
        r.SetWidth(w * mScales[i]);
        r.SetHeight(h * mScales[i]);
        r.SetXMin(xc - r.Width()/2);
        r.SetYMin(yc - r.Height()/2);
        if(!r.IsInside(img.GetRect()) || r.Width() < mInitBb.Width() * 0.1 || r.Height() < mInitBb.Height() * 0.1)
            continue;
        rects.push_back(r);
    }
    // put radius = 5
    int radius = 5;
    int r2 = radius * radius;
    for(int iy = -radius; iy<=radius; ++iy)
    {
        for(int ix=-radius; ix<=radius; ++ix)
        {
            if(ix*ix+iy*iy > r2) continue;
            if(ix==0 && iy==0) continue;
            for(int i=0; i<mScales.size(); ++i)
            {
                r.SetWidth(w * mScales[i]);
                r.SetHeight(h * mScales[i]);
                r.SetXMin(xc + ix - r.Width()/2);
                r.SetYMin(yc + iy - r.Height()/2);
                if(!r.IsInside(img.GetRect()) || r.Width() < mInitBb.Width() * 0.1 || r.Height() < mInitBb.Height() * 0.1)
                    continue;
                rects.push_back(r);
            }
        }
    }
}

void Tracker::genAllScaleBBs(const ImageRep &img, const FloatRect &centre, const float scale, std::vector<FloatRect> &rects)
{

    genGradualScaleBBs(img, centre, rects);

    if(scale == 1.0)
        return;

    float w = centre.Width();
    float h = centre.Height();
    float xc = centre.XCentre();
    float yc = centre.YCentre();

    // todo
    double scales[kAbruptScaleNum];
    double scale_step = (scale-1.0) / double(kAbruptScaleNum);
    for(int i=0; i<kAbruptScaleNum; ++i) {
        scales[i] = 1.0 + i * scale_step;
    }
    FloatRect r;
    for(int i=0; i<kAbruptScaleNum; ++i)
    {
        r.SetWidth(w * scales[i]);
        r.SetHeight(h * scales[i]);
        r.SetXMin(xc - r.Width()/2);
        r.SetYMin(yc - r.Height()/2);
        if(!r.IsInside(img.GetRect()) || r.Width() < mInitBb.Width() * 0.1 || r.Height() < mInitBb.Height() * 0.1)
            continue;
        rects.push_back(r);
    }
    // put radius = 5
    int radius = 5;
    int r2 = radius * radius;
    for(int iy = -radius; iy<=radius; ++iy)
    {
        for(int ix=-radius; ix<=radius; ++ix)
        {
            if(ix*ix+iy*iy > r2) continue;
            if(ix==0 && iy==0) continue;
            for(int i=0; i<kAbruptScaleNum; ++i)
            {
                r.SetWidth(w * scales[i]);
                r.SetHeight(h * scales[i]);
                r.SetXMin(xc + ix - r.Width()/2);
                r.SetYMin(yc + iy - r.Height()/2);
                if(!r.IsInside(img.GetRect()) || r.Width() < mInitBb.Width() * 0.1 || r.Height() < mInitBb.Height() * 0.1)
                    continue;
                rects.push_back(r);
            }
        }
    }

//     std::cout<<"all scale sample number: "<<rects.size() << std::endl;
}
void Tracker::getBestBB(const std::vector<double> &scores, int& bestInd)
{
    double bestScore = -DBL_MAX;
    for(int i=0; i<(int)scores.size(); ++i) {
        if(scores[i] > bestScore)
        {
            bestScore = scores[i];
            bestInd = i;
        }
    }
}

void Tracker::getBestBB(const std::vector<FloatRect>& rects, const std::vector<double>& scores, int& bestInd)
{
    double bestScore = -DBL_MAX;
    float scale_diff = -1;
    for(int i=0; i< (int)rects.size(); ++i)
    {
        if(scores[i] > bestScore)
        {
            float rect_scale = std::max(1.0 * rects[i].Width()/ mInitBb.Width(), 1.0 * rects[i].Height()/ mInitBb.Height());
            scale_diff = fabs(rect_scale - 1.0);
            bestScore = scores[i];
            bestInd = i;
        }
        else if(scores[i] == bestScore)
        {
            float rect_scale = std::max(1.0 * rects[i].Width()/ mInitBb.Width(), 1.0 * rects[i].Height()/ mInitBb.Height());
            float curr_scale_diff = fabs(rect_scale - 1.0);
            if(curr_scale_diff < scale_diff)
            {
                bestScore = scores[i];
                bestInd = i;
            }
        }
    }
}




