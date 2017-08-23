#ifndef FEATURE_H
#define FEATURE_H

#include <Eigen/Core>
#include <vector>
#include <PAWSS/Rect.h>

class Sample;
class multiSample;

class Feature
{
public:
    Feature();
    virtual ~Feature() {}
    inline int GetCount() const { return mFeatureCount; }
    inline const Eigen::VectorXd& Eval(const Sample& s) const
    {
        const_cast<Feature*>(this)->UpdateFeatureVector(s);
        return mFeatVec;
    }
    virtual void Eval(const multiSample& samples, std::vector<Eigen::VectorXd>& featVecs);

protected:
    int mFeatureCount;
    Eigen::VectorXd mFeatVec;

    void SetCount(int c);
    virtual void UpdateFeatureVector(const Sample& s)=0;

};

#endif
