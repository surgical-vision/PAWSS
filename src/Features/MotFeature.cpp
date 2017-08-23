#include <assert.h>
#include <PAWSS/Features/MotFeature.h>
#include <PAWSS/mUtils.h>
#include <PAWSS/macros.h>


static const int kNumBin = 16;

MotFeature::MotFeature()
{
    mBinNum = kNumBin;
    mBinStep = 2*M_PI / mBinNum;
}

int MotFeature::compBinIdx(const float orientation) const
{
//    assert(orientation<2*M_PI);
    float ori = mod(orientation, 2*M_PI);
    return int(ori/mBinStep);
}

void MotFeature::compMotion(const cv::Mat &prev, const cv::Mat &curr, const IntRect &rect, cv::Mat &orientation, cv::Mat &mag) const
{
    // get the image ready
    PAWSS_UNUSED(rect);
    assert(prev.channels() == 1 || prev.channels() == 3 || curr.channels() == 1 || curr.channels() == 3);
    cv::Mat grayPrev, grayCurr;
    if(prev.channels() == 3)
    {
        grayPrev = cv::Mat(prev.rows, prev.cols, CV_8UC1);
        cv::cvtColor(prev, grayPrev, CV_RGB2GRAY);
    }
    else
    {
        grayPrev = prev;
    }
    if(curr.channels() == 3)
    {
        grayCurr = cv::Mat(curr.rows, curr.cols, CV_8UC1);
        cv::cvtColor(curr, grayCurr, CV_RGB2GRAY);
    }
    else
    {
        grayCurr = curr;
    }

    // eval dense motion
    cv::Mat flow;
    double pyr_scale = 0.5;
    int levels = 3;
    int  winsize = 15;
    int iterations = 3;
    int poly_n = 5;
    double poly_sigma = 1.1;
    int flags = 0;
    cv::calcOpticalFlowFarneback(grayPrev, grayCurr, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags);

    std::vector<cv::Mat> flows;
    for(int i=0; i<flow.channels(); ++i)
        flows.push_back(cv::Mat(flow.rows, flow.cols, CV_32FC1));

    cv::split(flow, flows);

    // transform notion to mag and angle
    cv::cartToPolar(flows[0], flows[1], mag, orientation, false);

//    double min, max;
//    cv::minMaxLoc(orientation, &min, &max);

}



