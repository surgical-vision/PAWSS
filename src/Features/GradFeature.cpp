#include <math.h>
#include <assert.h>
#include "Features/GradFeature.h"
#include "mUtils.h"

static const int kNumGrad = 16;
static bool kUnsigned = false;

GradFeature::GradFeature()
{
    mBinNum = kNumGrad;
    if(kUnsigned) {
        mBinStep = M_PI / mBinNum; }
    else {
        mBinStep = 2*M_PI / mBinNum; }
}

int GradFeature::compBinIdx(const float orientation) const
{

    if(kUnsigned) {
        assert(orientation<M_PI); }
    else {
        assert(orientation<2*M_PI); }

    //    int i = 0;
    //    for(i=0; i<kNumGrad; ++i)
    //    {
    //        if(orientation <= m_bin_step * (i+1))
    //            break;
    //    }

    return int(orientation / mBinStep);
}

void GradFeature::compGrad(const cv::Mat &img, const IntRect &rect, cv::Mat &orientation, cv::Mat &mag) const
{
    assert(img.channels() ==1 || img.channels() == 3);
    cv::Mat grayImg;
    if(img.channels() == 3)
    {
        grayImg = cv::Mat(img.rows, img.cols, CV_8UC1);
        cv::cvtColor(img, grayImg, CV_RGB2GRAY);
    }
    else
    {
        grayImg = img;
    }

    cv::Mat x_sobel, y_sobel;
    cv::Sobel(grayImg, x_sobel, CV_32FC1, 1, 0);
    cv::Sobel(grayImg, y_sobel, CV_32FC1, 0, 1);

    float *xp, *yp, *op;
    double *mp;
    for(int iy=rect.YMin(); iy < rect.YMax(); ++iy)
    {
        xp = x_sobel.ptr<float>(iy);
        yp = y_sobel.ptr<float>(iy);
        op = orientation.ptr<float>(iy);
        mp = mag.ptr<double>(iy);

        if(kUnsigned)
        {
            for(int ix=rect.XMin(); ix < rect.XMax(); ++ix)
            {
                float x_grad = xp[ix];
                float y_grad = yp[ix];
                if(x_grad == 0)
                    x_grad += FLT_EPSILON;
                op[ix] = mod(std::atanf(y_grad/x_grad) + M_PI_2, M_PI);
                mp[ix] = std::sqrtf(x_grad*x_grad + y_grad*y_grad);
            }
        }
        else
        {
            for(int ix=rect.XMin(); ix < rect.XMax(); ++ix)
            {
                float x_grad = xp[ix];
                float y_grad = yp[ix];
                if(x_grad == 0)
                    x_grad += FLT_EPSILON;
                op[ix] = mod(std::atan2f(y_grad, x_grad) + M_PI, 2*M_PI);
                mp[ix] = std::sqrtf(x_grad*x_grad + y_grad*y_grad);
            }
        }
    }

}
