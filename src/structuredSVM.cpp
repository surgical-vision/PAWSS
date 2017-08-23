#include <opencv2/highgui.hpp>
#include <PAWSS/structuredSVM.h>
#include <PAWSS/Config.h>
#include <PAWSS/Kernels.h>
#include <PAWSS/Features/PatchFeature.h>
#include <PAWSS/Sample.h>
#include <PAWSS/ImageRep.h>
#include <PAWSS/GraphUtils/GraphUtils.h>

static const int kTileSize = 30;
static const int kMaxSVs =  2000;  // TODO (only used when no budget)

structuredSVM::structuredSVM(const Config &conf, const Feature& feature, const Kernel &kernel) :
    mC(conf.mSvmC),
    mConfig(conf),
    mKernel(kernel)
{
    int N = conf.mSvmBudgetSize > 0 ? conf.mSvmBudgetSize+2 : kMaxSVs;
    mK = Eigen::MatrixXd::Zero(N, N);
    mW = Eigen::VectorXd(feature.GetCount());
    mDebugImage = cv::Mat(800, 600, CV_8UC3);
}

structuredSVM::~structuredSVM()
{

}

void structuredSVM::Update(const multiSample &samples, const std::vector<Eigen::VectorXd> &featVecs, int y)
{
    // add new support pattern
    SupportPattern* sp = new SupportPattern;
    const std::vector<FloatRect>& rects = samples.getRects();
    FloatRect centre = rects[y];
    for(size_t i=0; i<rects.size(); ++i)
    {
        // express r in coord frame of centre sample
        FloatRect r = rects[i];
        r.Translate(-centre.XMin(), -centre.YMin());
        sp->yv.push_back(r);
        if (!mConfig.mQuietMode && mConfig.mDebugMode)
        {
            // store a thumbnail for each sample
            cv::Mat im(kTileSize, kTileSize, CV_8UC1);
            IntRect rect = rects[i];
            cv::Rect roi(rect.XMin(), rect.YMin(), rect.Width(), rect.Height());
            const cv::Mat& grayimg = samples.getImage().GetGrayImage();
            cv::resize(grayimg(roi), im, im.size());
            sp->images.push_back(im);
        }
    }
    // evaluate feature for each sample
    sp->x.resize(rects.size());
    sp->x = featVecs;
//    const_cast<Feature&>(mFeature).Eval(samples, sp->x);
    sp->y = y;
    sp->refCount = 0;
    mSps.push_back(sp);

    ProcessNew((int)mSps.size()-1);
    BudgetMaintenance();

    for (int i = 0; i < 10; ++i)
    {
        Reprocess();
        BudgetMaintenance();
    }

    // update decision boundary
    mW.setZero();
    for(size_t i=0; i<mSvs.size(); ++i)
    {
        const SupportVector& sv = *mSvs[i];
        mW += sv.b * sv.x->x[sv.y];
    }

}

void structuredSVM::Debug()
{
//    std::cout<<mSps.size()<<"/"<<mSvs.size()<<" support patterns/vectors "<< std::endl;
    UpdateDebugImage();
    cv::imshow("classifier", mDebugImage);
}

void structuredSVM::EvalMultiSamples(const std::vector<Eigen::VectorXd> &fvs, std::vector<double> &scores) const
{

//    const_cast<Feature&>(mFeature).Eval(samples, fvs);

    scores.resize(fvs.size());
    if(mConfig.mFeatureKernelPair.mKernel == Config::kKernelTypeLinear)
    {
        for(int i=0; i<(int)fvs.size(); ++i) {
            scores[i] = Test(fvs[i]); }
    }
    else
    {
        for(int i=0; i<(int)fvs.size(); ++i) {
            scores[i] = Evaluate(fvs[i]); }
    }
}


double structuredSVM::EvalTrueSample(const Eigen::VectorXd &tfv) const
{
    double sim = 0.0;
    int numPsv = 0;
//    const Eigen::VectorXd tfv = const_cast<Feature&>(mFeature).Eval(s);
    for(size_t i=0; i<mSvs.size(); ++i)
    {
        const SupportVector& sv = *mSvs[i];
        if(sv.b > 0)
        {
            ++numPsv;
            sim += mKernel.Eval(tfv, sv.x->x[sv.y]);
        }
    }
    sim /= numPsv;

#if VERBOSE
    std::cout<<"true example similarity: " << sim << std::endl;
#endif
    return sim;
}


void structuredSVM::SMOStep(int ipos, int ineg)
{
    if (ipos == ineg) return;
    SupportVector* svp = mSvs[ipos];
    SupportVector* svn =  mSvs[ineg];
    assert(svp->x == svn->x);
    SupportPattern* sp = svp->x;
#if VERBOSE
    std::cout<<"SMO: gpos: " << svp->g << " gneg: "<<svn->g<<std::endl;
#endif
    if ((svp->g - svn->g) < 1e-5)
    {
#if VERBOSE
        std::cout<< "SMO: skipping" <<std::endl;
#endif
    }
    else
    {
        double kii = mK(ipos, ipos) + mK(ineg, ineg) - 2*mK(ipos, ineg);
        double lu = (svp->g - svn->g)/kii;
        // no need to clamp against 0 since we'd have skiiped in that case
        double l = cv::min(lu, mC*(int)(svp->y == sp->y) - svp->b);

//        // foo test
//        std::cout<<"SMOStep: "<<l<<std::endl;

        svp->b += l;
        svn->b -= l;

        // update gradients
        for(int i=0; i<(int)mSvs.size(); ++i)
        {
            SupportVector* svi = mSvs[i];
            svi->g -= l*(mK(i, ipos) - mK(i, ineg));
        }
#if VERBOSE
        std::cout<< "SMO: "<< ipos <<","<<ineg<<" -- "<< svp->b << "," << svn->b << "(" << l << ")" << std::endl;
#endif
    }

    // check if we should remove either sv now
    if (fabs(svp->b) < 1e-8)
    {
        RemoveSupportVector(ipos);
        if (ineg == (int)mSvs.size())
        {
            // ineg and ipos will have been swapped during sv removel
            ineg = ipos;
        }
    }
    if (fabs(svn->b) < 1e-8)
    {
        RemoveSupportVector(ineg);
    }
}

std::pair<int, double> structuredSVM::MinGradient(int ind)
{
    const SupportPattern* sp = mSps[ind];
    std::pair<int, double> minGrad(-1, DBL_MAX);
    for(int i=0; i<(int)sp->yv.size(); ++i)
    {
        double grad = -Loss(sp->yv[i], sp->yv[sp->y]) - Evaluate(sp->x[i]);
        if (grad < minGrad.second)
        {
            minGrad.first = i;
            minGrad.second = grad;
        }

    }

    return minGrad;
}

void structuredSVM::ProcessNew(int ind)
{
    // gradient is -f(x,y) since loss=0
    int ipos = AddSupportVector(mSps[ind], mSps[ind]->y, -Evaluate(mSps[ind]->x[mSps[ind]->y]));


    std::pair<int, double> minGrad = MinGradient(ind);
    int ineg = AddSupportVector(mSps[ind], minGrad.first, minGrad.second);

    SMOStep(ipos, ineg);

}



void structuredSVM::Reprocess()
{
    ProcessOld();
    for(int i=0; i< 10; ++i)
        Optimize();
}



void structuredSVM::ProcessOld()
{
    if ( mSps.size() == 0) return;
    // choose pattern to process
    int ind = rand() % mSps.size();

    // find existing sv with largest grad and nonzero beta
    int ip = -1;
    double maxGrad = -DBL_MAX;
    for(int i=0; i<(int)mSvs.size(); ++i)
    {
        if(mSvs[i]->x != mSps[ind]) continue;

        const SupportVector* svi = mSvs[i];
        if(svi->g > maxGrad && svi->b < mC *(int)(svi->y == mSps[ind]->y))
        {
            ip = i;
            maxGrad = svi->g;
        }
    }
    assert(ip != -1);
    if(ip == -1) return;


    // find potentially new sv with smallest grad
    std::pair<int, double> minGrad = MinGradient(ind);
    int in = -1;
    for(int i=0; i<(int)mSvs.size(); ++i)
    {
        if(mSvs[i]->x != mSps[ind]) continue;

        if(mSvs[i]->y == minGrad.first)
        {
            in = i;
            break;
        }
    }
    if (in == -1)
    {
        // add new sv
        in = AddSupportVector(mSps[ind], minGrad.first, minGrad.second);

    }
//    std::cout<<"ProcessOld->SMOStep----"<<ip<<" "<<in<<std::endl;
    SMOStep(ip, in);
}

void structuredSVM::Optimize()
{
    if(mSps.size() == 0) return;
    // choose pattern to optimize
    int ind = rand() % mSps.size();
    int ip = -1;
    int in = -1;
    double maxGrad = -DBL_MAX;
    double minGrad = DBL_MAX;
    for(int i=0; i< (int)mSvs.size(); ++i)
    {
        if (mSvs[i]->x != mSps[ind]) continue;
        const SupportVector* svi = mSvs[i];
        if (svi->g > maxGrad && svi->b < mC*(int)(svi->y == mSps[ind]->y))
        {
            ip = i;
            maxGrad = svi->g;
        }
        if (svi->g < minGrad)
        {
            in = i;
            minGrad = svi->g;
        }
    }
    assert(ip != -1 && in != -1);
    if (ip == -1 || in == -1)
    {
        // this should not happen
        std::cout<<"!!!!!!!!!!!!!!!!!!!!"<< std::endl;
        return;
    }
//    std::cout<<"Optimize->SMOStep----"<<ip<<" "<<in<<std::endl;
    SMOStep(ip, in);
}


int structuredSVM::AddSupportVector(SupportPattern *x, int y, double g)
{
    SupportVector* sv = new SupportVector;
    sv->b = 0.0;
    sv->x = x;
    sv->y = y;
    sv->g = g;

    int ind = (int)mSvs.size();
    mSvs.push_back(sv);
    x->refCount++;
#if VERBOSE
    std::cout<< "Add in SV: "<<ind<<std::endl;
#endif

    if(mConfig.mFeatureKernelPair.mKernel == Config::kKernelTypeLinear)
    {

        // update kernel matrix
        for(int i=0; i<ind; ++i)
        {
            mK(i, ind) = mKernel.Eval(mSvs[i]->x->x[mSvs[i]->y], x->x[y]);
            mK(ind, i) = mK(i, ind);
        }
        mK(ind, ind) = mKernel.Eval(x->x[y]);
    }

    return ind;
}

void structuredSVM::RemoveSupportVector(int ind)
{
#if VERBOSE
    std::cout<<"Removing SV: "<< ind << std::endl;
#endif
    mSvs[ind]->x->refCount--;
    if(mSvs[ind]->x->refCount == 0)
    {
        // also remove the support pattern
        for(int i=0; i<(int) mSps.size(); ++i)
            if(mSps[i] == mSvs[ind]->x)
            {
                delete mSps[i];
                mSps.erase(mSps.begin()+i);
                break;
            }
    }
    // make sure the support vector is at the back, this lets us keep the kernel matrix cached and valid
    if(ind<(int)mSvs.size()-1)
    {
        SwapSupportVectors(ind, (int)mSvs.size()-1);
        ind = (int)mSvs.size()-1;
    }
    delete mSvs[ind];
    mSvs.pop_back();
}

void structuredSVM::SwapSupportVectors(int ind1, int ind2)
{
    SupportVector* tmp = mSvs[ind1];
    mSvs[ind1] = mSvs[ind2];
    mSvs[ind2] = tmp;

    Eigen::VectorXd row1 = mK.row(ind1);
    mK.row(ind1) = mK.row(ind2);
    mK.row(ind2) = row1;

    Eigen::VectorXd col1 = mK.col(ind1);
    mK.col(ind1) = mK.col(ind2);
    mK.col(ind2) = col1;
}

void structuredSVM::BudgetMaintenanceRemove()
{
    // find negative sv with smallest effect on discriminant function if removed
    double minVal = DBL_MAX;
    int in = -1;
    int ip = -1;
    for (int i = 0; i < (int)mSvs.size(); ++i)
    {
        if (mSvs[i]->b < 0.0)
        {
            // find corresponding positive sv
            int j = -1;
            for (int k = 0; k < (int)mSvs.size(); ++k)
            {
                if (mSvs[k]->b > 0.0 && mSvs[k]->x == mSvs[i]->x)
                {
                    j = k;
                    break;
                }
            }
            double val = mSvs[i]->b * mSvs[i]->b * (mK(i,i) + mK(j,j) - 2.0*mK(i,j));
            if (val < minVal)
            {
                minVal = val;
                in = i;
                ip = j;
            }
        }
    }

    // adjust weight of positive sv to compensate for removal of negative
    mSvs[ip]->b += mSvs[in]->b;

    // remove negative sv
    RemoveSupportVector(in);
    if (ip == (int)mSvs.size())
    {
        // ip and in will have been swapped during support vector removal
        ip = in;
    }

    if (mSvs[ip]->b < 1e-8)
    {
        // also remove positive sv
        RemoveSupportVector(ip);
    }

    // update gradients
    // TODO: this could be made cheaper by just adjusting incrementally rather than recomputing
    for (int i = 0; i < (int)mSvs.size(); ++i)
    {
        SupportVector& svi = *mSvs[i];
        svi.g = -Loss(svi.x->yv[svi.y],svi.x->yv[svi.x->y]) - Evaluate(svi.x->x[svi.y]);
    }

}

void structuredSVM::BudgetMaintenance()
{
    if(mConfig.mSvmBudgetSize > 0)
    {
        while((int)mSvs.size() > mConfig.mSvmBudgetSize)
            BudgetMaintenanceRemove();
    }
}


// used for computing grad
double structuredSVM::Evaluate(const Eigen::VectorXd &x) const
{
    double f = 0.0;
    for(int i=0; i<(int)mSvs.size(); ++i)
    {
        const SupportVector& sv = *mSvs[i];
        f += sv.b*mKernel.Eval(x, sv.x->x[sv.y]);
    }

    return f;
}

double structuredSVM::Test(const Eigen::VectorXd &x) const
{
    return mW.dot(x);
}

void structuredSVM::UpdateDebugImage()
{
    mDebugImage.setTo(0);
    int n = (int)mSvs.size();

    if (n == 0) return;

    const int kCanvasSize = 600;
    int gridSize = (int)sqrtf((float)(n-1)) + 1;
    int tileSize = (int)((float)kCanvasSize/gridSize);

    if (tileSize < 5)
    {
        std::cout << "too many support vectors to display" << std::endl;
        return;
    }

    cv::Mat temp(tileSize, tileSize, CV_8UC1);
    int x = 0;
    int y = 0;
    int ind = 0;
    float vals[kMaxSVs];
    memset(vals, 0, sizeof(float)*n);
    int drawOrder[kMaxSVs];

    for (int set = 0; set < 2; ++set)
    {
        for (int i = 0; i < n; ++i)
        {
            if (((set == 0) ? 1 : -1)*mSvs[i]->b < 0.0) continue;

            drawOrder[ind] = i;
            vals[ind] = (float)mSvs[i]->b;
            ++ind;

            cv::Mat I = mDebugImage(cv::Rect(x, y, tileSize, tileSize));
            cv::resize(mSvs[i]->x->images[mSvs[i]->y], temp, temp.size());
            cv::cvtColor(temp, I, CV_GRAY2RGB);
            double w = 1.0;
            cv::rectangle(I, cv::Point(0, 0), cv::Point(tileSize-1, tileSize-1), (mSvs[i]->b > 0.0) ? CV_RGB(0, (uchar)(255*w), 0) : CV_RGB((uchar)(255*w), 0, 0), 3);
            x += tileSize;
            if ((x+tileSize) > kCanvasSize)
            {
                y += tileSize;
                x = 0;
            }
        }
    }

    const int kKernelPixelSize = 2;
    int kernelSize = kKernelPixelSize*n;

    double kmin = mK.minCoeff();
    double kmax = mK.maxCoeff();

    if (kernelSize < mDebugImage.cols && kernelSize < mDebugImage.rows)
    {
        cv::Mat K = mDebugImage(cv::Rect(mDebugImage.cols-kernelSize, mDebugImage.rows-kernelSize, kernelSize, kernelSize));
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                cv::Mat Kij = K(cv::Rect(j*kKernelPixelSize, i*kKernelPixelSize, kKernelPixelSize, kKernelPixelSize));
                uchar v = (uchar)(255*(mK(drawOrder[i], drawOrder[j])-kmin)/(kmax-kmin));
                Kij.setTo(cv::Scalar(v, v, v));
            }
        }
    }
    else
    {
        kernelSize = 0;
    }


    cv::Mat I = mDebugImage(cv::Rect(0, mDebugImage.rows - 200, mDebugImage.cols-kernelSize, 200));
    I.setTo(cv::Scalar(255,255,255));
    IplImage II = I;
    setGraphColor(0);
    drawFloatGraph(vals, n, &II, 0.f, 0.f, I.cols, I.rows);
}

