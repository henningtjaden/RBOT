/**
 *   #, #,         CCCCCC  VV    VV MM      MM RRRRRRR
 *  %  %(  #%%#   CC    CC VV    VV MMM    MMM RR    RR
 *  %    %## #    CC        V    V  MM M  M MM RR    RR
 *   ,%      %    CC        VV  VV  MM  MM  MM RRRRRR
 *   (%      %,   CC    CC   VVVV   MM      MM RR   RR
 *     #%    %*    CCCCCC     VV    MM      MM RR    RR
 *    .%    %/
 *       (%.      Computer Vision & Mixed Reality Group
 *                For more information see <http://cvmr.info>
 *
 * This file is part of RBOT.
 *
 *  @copyright:   RheinMain University of Applied Sciences
 *                Wiesbaden Rüsselsheim
 *                Germany
 *     @author:   Henning Tjaden
 *                <henning dot tjaden at gmail dot com>
 *    @version:   1.0
 *       @date:   30.08.2018
 *
 * RBOT is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * RBOT is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with RBOT. If not, see <http://www.gnu.org/licenses/>.
 */

#include "optimization_engine.h"

using namespace std;
using namespace cv;


OptimizationEngine::OptimizationEngine(int width, int height)
{
    renderingEngine = RenderingEngine::Instance();
    
    SDT2D = new SignedDistanceTransform2D(8.0f);
    
    this->width = width;
    this->height = height;
}

OptimizationEngine::~OptimizationEngine()
{
    delete SDT2D;
}


void OptimizationEngine::minimize(vector<Mat>& imagePyramid, vector<Object3D*>& objects, int runs)
{
    // OPTIMIZATION ITERATIONS
    
    // level 2
    for(int iter = 0; iter < runs*4; iter++)
    {
        runIteration(objects, imagePyramid, 2);
    }
    
    // level 1
    for(int iter = 0; iter < runs*2; iter++)
    {
        runIteration(objects, imagePyramid, 1);
    }
    
    // level 0
    for(int iter = 0; iter < runs*1; iter++)
    {
        runIteration(objects, imagePyramid, 0);
    }
}



void OptimizationEngine::runIteration(vector<Object3D*>& objects, const vector<Mat>& imagePyramid, int level)
{
    Rect roi;
    Mat mask, depth, depthInv, sdt, xyPos;
    Mat croppedMask, croppedDepth, croppedDepthInv;
    
    renderingEngine->setLevel(level);
    
    int numInitialized = 0;
    
    // increase the image pyramid level until the area of the 2D bounding box
    // of every object is greater than 3000 pixels in the image
    for(int o = 0; o < objects.size(); o++)
    {
        if(objects[o]->isInitialized())
        {
            roi = compute2DROI(objects[o], Size(width/pow(2, level), height/pow(2, level)), 8);
            
            if(roi.area() != 0)
            {
                while(roi.area() < 3000 && level > 0)
                {
                    level--;
                    renderingEngine->setLevel(level);
                    roi = compute2DROI(objects[o], Size(width/pow(2, level), height/pow(2, level)), 8);
                }
            }
            numInitialized++;
        }
    }
    
    // render the common silhouette mask
    renderingEngine->setLevel(level);
    renderingEngine->renderSilhouette(vector<Model*>(objects.begin(), objects.end()), GL_FILL);
    
    // download the depth buffer
    depth = renderingEngine->downloadFrame(RenderingEngine::DEPTH);
    
    // if more than one object is initialized, download the common silhouette
    // mask required for occlusion detection
    if(numInitialized > 1)
    {
        mask = renderingEngine->downloadFrame(RenderingEngine::MASK);
    }
    else // otherwise for a single object the mask is equal to the depth buffer
    {
        mask = depth;
    }
    
    for(int o = 0; o < objects.size(); o++)
    {
        if(objects[o]->isInitialized())
        {
            // compute the 2D region of interest containing the silhouette of the current object
            roi = compute2DROI(objects[o], Size(width/pow(2, level), height/pow(2, level)), 8);
            
            if(roi.area() == 0)
            {
                continue;
            }
            
            // render the individual inverse depth buffer per object
            renderingEngine->renderSilhouette(objects[o], GL_FILL, true);
            depthInv = renderingEngine->downloadFrame(RenderingEngine::DEPTH);
            
            // crop the images wrt to the 2D roi
            croppedMask = mask(roi).clone();
            croppedDepth = depth(roi).clone();
            croppedDepthInv = depthInv(roi).clone();
            
            int m_id = (numInitialized <= 1) ? -1 : objects[o]->getModelID();
            
            // compute the 2D signed distance transform of the silhouette
            SDT2D->computeTransform(croppedMask, sdt, xyPos, 8, m_id);
            
            // the hessian approximation
            Matx66f wJTJ;
            // the gradient
            Matx61f JT;
            
            // compute the Jacobian terms (i.e. the gradient and the hessian approx.) needed for the Gauss-Newton step
            parallel_computeJacobians(objects[o], imagePyramid[level], croppedDepth, croppedDepthInv, sdt, xyPos, roi, croppedMask, m_id, level, wJTJ, JT, roi.height);
            
            // update the pose by computing the Gauss-Newton step
            applyStepGaussNewton(objects[o], wJTJ, JT);
        }
    }
}


void OptimizationEngine::parallel_computeJacobians(Object3D* object, const Mat& frame, const Mat& depth, const Mat& depthInv, const Mat& sdt, const Mat& xyPos, const Rect& roi, const cv::Mat& mask, int m_id, int level, Matx66f& wJTJ, Matx61f &JT, int threads)
{
    float zNear = renderingEngine->getZNear();
    float zFar = renderingEngine->getZFar();
    Matx33f K = renderingEngine->getCalibrationMatrix().get_minor<3, 3>(0, 0);
    
    JT = Matx61f::zeros();
    wJTJ = Matx66f::zeros();
    
    vector<Matx61f> JTCollection(threads);
    vector<Matx66f> wJTJCollection(threads);
    
    parallel_for_(cv::Range(0, threads), Parallel_For_computeJacobiansGN(object->getTCLCHistograms(), frame, sdt, xyPos, depth, depthInv, K, zNear, zFar, roi, mask, m_id, level, wJTJCollection, JTCollection, threads));
    
    for(int i = 0; i < threads; i++)
    {
        JT += JTCollection[i];
        wJTJ += wJTJCollection[i];
    }
    
    // copy the top right triangular matrix into the bottom left triangle
    for(int i = 0; i < wJTJ.rows; i++)
    {
        for(int j = i+1; j < wJTJ.cols; j++)
        {
            wJTJ(j, i) = wJTJ(i, j);
        }
    }
}

Rect OptimizationEngine::compute2DROI(Object3D* object, const cv::Size& maxSize, int offset)
{
    // PROJECT THE 3D BOUNDING BOX AS 2D ROI
    Rect boundingRect;
    vector<Point2f> projections;
    
    renderingEngine->projectBoundingBox(object, projections, boundingRect);
    
    if(boundingRect.x >= maxSize.width || boundingRect.y >= maxSize.height
       || boundingRect.x + boundingRect.width <= 0 || boundingRect.y + boundingRect.height <= 0)
    {
        return Rect(0, 0, 0, 0);
    }
    
    // CROP THE ROI AROUND THE SILHOUETTE
    Rect roi = Rect(boundingRect.x - offset, boundingRect.y - offset, boundingRect.width + 2*offset, boundingRect.height + 2*offset);
    
    if(roi.x < 0)
    {
        roi.width += roi.x;
        roi.x = 0;
    }
    
    if(roi.y < 0)
    {
        roi.height += roi.y;
        roi.y = 0;
    }
    
    if(roi.x+roi.width > maxSize.width) roi.width = maxSize.width - roi.x;
    if(roi.y+roi.height > maxSize.height) roi.height = maxSize.height - roi.y;
    
    return roi;
}

void OptimizationEngine::applyStepGaussNewton(Object3D* object, const Matx66f& wJTJ, const Matx61f& JT)
{
    // Gauss-Newton step in se3
    Matx61f delta_xi = -wJTJ.inv(DECOMP_CHOLESKY)*JT;
    
    // get the current pose
    Matx44f T_cm = object->getPose();
    
    // apply the update step in SE3
    T_cm = Transformations::exp(delta_xi)*T_cm;
    
    // set the updated pose
    object->setPose(T_cm);
}

