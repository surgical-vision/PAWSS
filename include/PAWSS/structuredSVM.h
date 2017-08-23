#ifndef STRUCTUREDSVM_H
#define STRUCTUREDSVM_H

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <PAWSS/Rect.h>

class Config;
class Feature;
class Kernel;
class Sample;
class multiSample;

struct SupportPattern
{
    std::vector<Eigen::VectorXd> x;
    std::vector<FloatRect> yv;
    std::vector<cv::Mat> images;
    int y;
    int refCount;
};

struct SupportVector
{
    SupportPattern* x;
    int y;
    double b;
    double g;
    cv::Mat image;
};

class structuredSVM
{
public:
    structuredSVM(const Config& conf, const Feature &feature, const Kernel& kernel);
    ~structuredSVM();
    void Update(const multiSample& samples, const std::vector<Eigen::VectorXd>& featVecs, int y);
    void Debug();
    void EvalMultiSamples(const std::vector<Eigen::VectorXd>& fvs, std::vector<double>& scores) const;
    double EvalTrueSample(const Eigen::VectorXd& tfv) const;

//    inline const Feature& getFeature() const {return mFeature; }

private:
    inline double Loss(const FloatRect& y1, const FloatRect& y2) const {return 1.0 - y1.Overlap(y2); }

    void SMOStep(int ipos, int ineg);
    std::pair<int, double> MinGradient(int ind);
    void ProcessNew(int ind);
    void Reprocess();
    void ProcessOld();
    void Optimize();

    int AddSupportVector(SupportPattern* x, int y, double g);
    void RemoveSupportVector(int ind);
    void SwapSupportVectors(int ind1, int ind2);

    void BudgetMaintenance();
    void BudgetMaintenanceRemove();

    double Evaluate(const Eigen::VectorXd& x) const;
    double Test(const Eigen::VectorXd& x) const;

    void UpdateDebugImage();

    std::vector<SupportPattern*> mSps;
    std::vector<SupportVector*> mSvs;
    cv::Mat mDebugImage;
    Eigen::MatrixXd mK;
    Eigen::VectorXd mW;
    double mC;

    const Config& mConfig;
//    const Feature& mFeature;
    const Kernel& mKernel;



};



#endif
