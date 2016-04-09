#include "Features/Feature.h"
#include "Sample.h"


Feature::Feature() :
    mFeatureCount(0)
{

}

void Feature::Eval(const multiSample &samples, std::vector<Eigen::VectorXd> &featVecs)
{
    featVecs.resize(samples.getRects().size());
    for(int i=0; i<(int)featVecs.size(); ++i)
        featVecs[i] = Eval(samples.getSample(i));
}

void Feature::SetCount(int c)
{
    mFeatureCount = c;
    mFeatVec = Eigen::VectorXd::Zero(c);
}
