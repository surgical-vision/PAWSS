#include <numeric>
#include "segModel.h"

static const float kProb_c10 = 0.4;
static const float kProb_c11 = 0.6;
static const float kProb_c01 = 0.4;
static const float kProb_c00 = 0.6;
static const float kAlpha = 0.1;

segModel::segModel(int numBin)
{
    mNumBin = numBin;

    mProb_c0_y = std::vector<double>(mNumBin, 0.5);
    mProb_c1_y = std::vector<double>(mNumBin, 0.5);
    mProb_y_c0 = std::vector<double>(mNumBin, 0.0);
    mProb_y_c1 = std::vector<double>(mNumBin, 0.0);

    mInitialized = false;
}

segModel::~segModel()
{
    mProb_c0_y.clear();
    mProb_c1_y.clear();
    mProb_y_c0.clear();
    mProb_y_c1.clear();
}

void segModel::updateWProb(const cv::Mat& bin_map, const Eigen::VectorXd &weights,
                           const std::vector<IntRect>& patch_rects,
                           const IntRect& bound_rect, const IntRect &outer_rect)
{
    std::vector<double> pixelnum_c0(mNumBin, 0);
    std::vector<double> pixelnum_c1(mNumBin, 0);

    assert(weights.size() == patch_rects.size());
    const int *hp;
    int hsv_bin_idx;

    // foreground
    IntRect rect;
    float weight;
    for(int id=0; id<patch_rects.size(); ++id)
    {
        rect = patch_rects[id];
        weight = weights[id];
        for(int iy = rect.YMin(); iy<rect.YMax(); ++iy)
        {
            hp = bin_map.ptr<int>(iy);
            for(int ix = rect.XMin(); ix<rect.XMax(); ++ix)
            {
                hsv_bin_idx = hp[ix];
                pixelnum_c1[hsv_bin_idx] += weight;
            }
        }
    }

    // background
    for(int iy=outer_rect.YMin(); iy<outer_rect.YMax(); ++iy)
    {
        hp = bin_map.ptr<int>(iy);

        for(int ix=outer_rect.XMin(); ix<outer_rect.XMax(); ++ix)
        {
            hsv_bin_idx = hp[ix];

            if( (iy<bound_rect.YMin()) || (iy>=bound_rect.YMax()) ||
                (ix<bound_rect.XMin()) || (ix>=bound_rect.XMax()))     // outside boundary rect
                pixelnum_c0[hsv_bin_idx] += 1;
        }
    }

    auto num_pixel_c0 = std::accumulate(pixelnum_c0.begin(), pixelnum_c0.end(), 0);
    auto num_pixel_c1 = std::accumulate(pixelnum_c1.begin(), pixelnum_c1.end(), 0);
    if(num_pixel_c0 == 0 || num_pixel_c1 == 0)
        return;

    std::vector<double> prob_c0_y(mNumBin, 0.0);
    std::vector<double> prob_c1_y(mNumBin, 0.0);
    if(!mInitialized)    // initialization
    {
        for(int idx=0; idx<mNumBin; ++idx)
        {
            mProb_y_c0[idx] = 1.0 * pixelnum_c0[idx] / num_pixel_c0;
            mProb_y_c1[idx] = 1.0 * pixelnum_c1[idx] / num_pixel_c1;

            prob_c0_y[idx] = mProb_y_c0[idx] * (kProb_c01 * mProb_c1_y[idx] + kProb_c00 * mProb_c0_y[idx]);
            prob_c1_y[idx] = mProb_y_c1[idx] * (kProb_c11 * mProb_c1_y[idx] + kProb_c10 * mProb_c0_y[idx]);
            auto pixelnum_bin = prob_c0_y[idx] + prob_c1_y[idx];
            if(pixelnum_bin != 0)
            {
                mProb_c0_y[idx] = 1.0 * prob_c0_y[idx] / pixelnum_bin;
                mProb_c1_y[idx] = 1.0 * prob_c1_y[idx] / pixelnum_bin;
            }
        }
        mInitialized = true;
    }
    else                // update
    {
        for(int idx=0; idx<mNumBin; ++idx)
        {
            mProb_y_c0[idx] = kAlpha * pixelnum_c0[idx] / num_pixel_c0 + (1-kAlpha) * mProb_y_c0[idx];
            mProb_y_c1[idx] = kAlpha * pixelnum_c1[idx] / num_pixel_c1 + (1-kAlpha) * mProb_y_c1[idx];

            prob_c0_y[idx] = mProb_y_c0[idx] * (kProb_c01 * mProb_c1_y[idx] + kProb_c00 * mProb_c0_y[idx]);
            prob_c1_y[idx] = mProb_y_c1[idx] * (kProb_c11 * mProb_c1_y[idx] + kProb_c10 * mProb_c0_y[idx]);
            auto pixelnum_bin = prob_c0_y[idx] + prob_c1_y[idx];
            if(pixelnum_bin != 0)
            {
                mProb_c0_y[idx] = 1.0 * prob_c0_y[idx] / pixelnum_bin;
                mProb_c1_y[idx] = 1.0 * prob_c1_y[idx] / pixelnum_bin;
            }
        }
    }
}

void segModel::getProbImg(const cv::Mat &binImg, const IntRect &rect, cv::Mat &probImg)
{
    assert(binImg.rows==probImg.rows && binImg.cols==probImg.cols);
    probImg.setTo(0.0);

    const int *bp;
    float *pp;
    for(int iy=rect.YMin(); iy<rect.YMax(); ++iy)
    {
        bp = binImg.ptr<int>(iy);
        pp = probImg.ptr<float>(iy);
        for(int ix=rect.XMin(); ix<rect.XMax(); ++ix) {
            pp[ix] = getForeProb(bp[ix]); }
    }
}

