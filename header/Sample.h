#ifndef SAMPLE_H
#define SAMPLE_H
#include <vector>
#include <assert.h>
#include "Rect.h"

class ImageRep;

class Sample
{
public:
    Sample(const ImageRep& image, const FloatRect& rect) :
        mImage(image), mBb(rect)
    {   }
    inline const FloatRect& getRect() const {return mBb; }
    inline const ImageRep& getImage() const {return mImage; }

private:
    const ImageRep& mImage;
    FloatRect mBb;
};

class multiSample
{
public:
    multiSample(const ImageRep& image, const std::vector<FloatRect>& rects) :
        mImage(image),
        mBbs(rects)
    {   }
    inline const ImageRep& getImage() const { return mImage; }
    inline const std::vector<FloatRect>& getRects() const { return mBbs; }
    inline Sample getSample(int i) const { assert(i<mBbs.size()); return Sample(mImage, mBbs[i]);}


private:
    const ImageRep& mImage;
    std::vector<FloatRect> mBbs;
};

#endif
