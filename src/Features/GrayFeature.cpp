#include <PAWSS/Features/GrayFeature.h>

static const int kBinNum = 8;

GrayFeature::GrayFeature()
{
    mBinNum = kBinNum;
    mStep = 256.0 / mBinNum;
}

void GrayFeature::getBinImg(const cv::Mat &img, const IntRect &rect, cv::Mat &binImg) const
{
    assert(img.rows == binImg.rows && img.cols == binImg.cols);

    const uchar *p;
    int *bp;
    int idx;
    for(int iy=rect.YMin(); iy<rect.YMax(); ++iy)
    {
        p = img.ptr<uchar>(iy);
        bp = binImg.ptr<int>(iy);
        for(int ix=rect.XMin(); ix<rect.XMax(); ++ix)
        {
            compBinIdx(p[ix], idx);
            // todo:
            bp[ix] = idx;
        }
    }
}
