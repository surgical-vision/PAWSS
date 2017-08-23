#include <PAWSS/scaleEstimator.h>
#include <PAWSS/mUtils.h>


ScaleEstimator::~ScaleEstimator()
{
    mPrevPts.clear();
    mStatus.clear();
}

void ScaleEstimator::initialize(const cv::Mat &prevImg, const std::vector<cv::Point2f> &prevPts)
{
    assert(prevImg.channels() == 1);
    setPrevImg(prevImg);
    setPrevPts(prevPts);
    mCurrPts.reserve(mPrevPts.size());
    mStatus.reserve(mPrevPts.size());
}

void ScaleEstimator::update(const cv::Mat &prevImg, const std::vector<cv::Point2f> &prevPts)
{
    initialize(prevImg, prevPts);
}



float ScaleEstimator::estimateScale(const cv::Mat &currImg)
{
    trackPts(currImg);

    int trackedNum = 0;
    for(size_t i=0; i<mStatus.size(); ++i) {
        if(mStatus[i] != 0) {
            trackedNum++; } }
    if(float(trackedNum)/mStatus.size() < 0.5)
        return 1.0;

    float prevDist, currDist, ratio;
    std::vector<float> pairRatios;
    for(size_t i=0; i<mStatus.size(); ++i)
    {
        if(mStatus[i] == 0)
            continue;

        for(size_t j=i+1; j<mStatus.size(); ++j)
        {
            if(mStatus[j] == 0)
                continue;
            prevDist =  getPtDist(mPrevPts[i], mPrevPts[j]);
            currDist =  getPtDist(mCurrPts[i], mCurrPts[j]);
            if(prevDist!=0)
                ratio = currDist/prevDist;
            else if(currDist == 0)
                ratio = 1.0;
            else
                continue;
            pairRatios.push_back(ratio);
        }
    }
    // get the middle ratio
    std::sort(pairRatios.begin(), pairRatios.end());
    return pairRatios[pairRatios.size()/2];
}

void ScaleEstimator::trackPts(const cv::Mat &currImg)
{
    assert(currImg.channels() == 1);
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(mPrevImg, currImg, mPrevPts, mCurrPts, mStatus, err);
}
