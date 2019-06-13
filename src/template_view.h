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

#ifndef TEMPLATE_VIEW_H
#define TEMPLATE_VIEW_H

#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "rendering_engine.h"
#include "object3d.h"
#include "tclc_histograms.h"
#include "signed_distance_transform2d.h"

/**
 *  The template view data per pixel.
 */
struct PixelData
{
    // The original 2D pixel location.
    int x;
    int y;
    
    // The Heaviside value.
    float hsVal;
    
    // The number of tclc-histograms the pixel lies within.
    int ids_size;
    
    // The IDs of all tclc-histograms the pixel lies within.
    int* ids;
};

/**
 *  A class representing a single template view at multiple image scales
 *  for region-based object pose detection using tclc-histograms.
 */
class TemplateView {
    
public:
    /**
     *  Constructor for the template view at a given object rotation and
     *  distance to the camera.
     *
     *  @param  object The 3D object for which the template view is to be created.
     *  @param  alpha The Euler angle of the object's rotation around the x-axis (in degrees).
     *  @param  beta The Euler angle of the object's rotation around the y-axis (in degrees).
     *  @param  gamma The Euler angle of the object's rotation around the z-axis (in degrees).
     *  @param  distance The object's distance to the camera to be used.
     *  @param  numLevels Number of template pyramid levels to be created with a downscale factor of 2.
     *  @param  generateNeighbors A flag telling whether neighboring templates should also be created or not.
     */
    TemplateView(Object3D *object, float alpha, float beta, float gamma, float distance, int numLevels, bool generateNeighbors);
    
    ~TemplateView();
    
    /**
     *  Returns the 6DOF object pose coresponding to the
     *  template view in form of a 4x4 float matrix
     *  T_cm = [r11 r12 r13 tx]
     *         [r21 r22 r23 ty]
     *         [r31 r32 r33 tz]
     *         [  0   0   0  1],
     *  describing the transformation from model coordinates X_m
     *  into camera coordinates X_c.
     *
     *  @return  The 6DOF object pose within the template.
     */
    cv::Matx44f getPose();
    
    /**
     *  Returns the Euler angle of the object's rotation around the
     *  x-axis coresponding to the template.
     *
     *  @return  The object's rotation around the x-axis within the template (in degrees).
     */
    float getAlpha();
    
    /**
     *  Returns the Euler angle of the object's rotation around the
     *  y-axis coresponding to the template.
     *
     *  @return  The object's rotation around the y-axis within the template (in degrees).
     */
    float getBeta();
    
    /**
     *  Returns the Euler angle of the object's rotation around the
     *  z-axis coresponding to the template.
     *
     *  @return  The object's rotation around the z-axis within the template (in degrees).
     */
    float getGamma();
    
    /**
     *  Returns the object's distance to the camera coresponding to
     *  the template.
     *
     *  @return  The object's distance to the camera within the template.
     */
    float getDistance();
    
    /**
     *  Returns the total number of pixels in the object region at a given
     *  pyramid level.
     *
     *  @param level The pyramid level to be used.
     *  @return  The total number of pixels in the object region within the template.
     */
    int getEtaF(int level);
    
    /**
     *  Returns the binary mask image of the template at a given pyramid
     *  level.
     *
     *  @param level The pyramid level to be used.
     *  @return  The binary mask image of the template.
     */
    cv::Mat getMask(int level);
    
    /**
     *  Returns the 2D signed distance transform of the binary mask of the
     *  template at a given pyramid level.
     *
     *  @param level The pyramid level to be used.
     *  @return  The 2D signed distance transform of the binary mask of the template.
     */
    cv::Mat getSDT(int level);
    
    /**
     *  Returns the smoothed Heaviside representation of the 2D signed distance
     *  transform of the template at a given pyramid level.
     *
     *  @param level The pyramid level to be used.
     *  @return  The smoothed Heaviside representation of the 2D signed distance transform of the template.
     */
    cv::Mat getHeaviside(int level);
    
    /**
     *  Returns the 2D region of interest around the object in the template at
     *  a given pyramid level.
     *
     *  @param level The pyramid level to be used.
     *  @return  The 2D region of interest around the object in the template.
     */
    cv::Rect getROI(int level);
    
    /**
     *  Returns the 2D (x, y) offset at which the template currently matches
     *  best with the corresponding matching score at a given pyramid level.
     *
     *  @param level The pyramid level to be used.
     *  @return  The 2D offset with the matching score (x, y, score).
     */
    cv::Point3f getCurrentOffset(int level);
    
    /**
     *  Sets the 2D (x, y) offset at which the template currently matches
     *  best with the corresponding matching score at a given pyramid level.
     *
     *  @param offset The 2D offset with the matching score (x, y, score).
     *  @param level The pyramid level to be used.
     */
    void setCurrentOffset(cv::Point3f &offset, int level);
    
    /**
     *  Returns the 2D centers and IDs of all tclc-histograms in the
     *  template at a given pyramid level.
     *
     *  @param level The pyramid level to be used.
     *  @return  The 2D centers and IDs of all tclc-histograms in the template [(x_0, y_0, id_0), (x_1, y_1, id_1), ...].
     */
    std::vector<cv::Point3i> getCentersAndIDs(int level);
    
    /**
     *  Returns a linearized representation of the template at a given pyramid
     *  level.
     *
     *  @param level The pyramid level to be used.
     *  @return  The linearized representation of the template.
     */
    std::vector<PixelData> &getCompressedPixelData(int level);
    
    /**
     *  Adds a neighboring template view to this template.
     *
     *  @param kv The neighboring template view to be added.
     */
    void addNeighborTemplate(TemplateView *kv);
    
    /**
     *  Returns the set of all neighboring templates of this template.
     *
     *  @return  The set of all neighboring templates of this template.
     */
    std::vector<TemplateView*> getNeighborTemplates();
    
private:
    RenderingEngine *renderingEngine;
    
    cv::Matx44f T_cm;
    
    std::vector<int> etaFPyramid;
    std::vector<cv::Mat> maskPyramid;
    std::vector<cv::Mat> sdtPyramid;
    std::vector<cv::Mat> heavisidePyramid;
    
    std::vector<cv::Rect> roiPyramid;
    
    std::vector<std::vector<cv::Point3i> > centersIDsPyramid;
    
    std::vector<std::vector<PixelData> > pixelDataPyramid;
    
    cv::Point3f currentOffset;
    
    float _alpha;
    float _beta;
    float _gamma;
    
    float _distance;
    
    int _numLevels;
    
    std::vector<TemplateView*> neighbors;
    
    void compressTemplateData(const std::vector<cv::Point3i> &centersIDs, const cv::Mat &heaviside, const cv::Rect &roi, int radius, int level);
    
    cv::Rect computeBoundingBox(const std::vector<cv::Point3i> &centersIDs, int offset, int level, const cv::Size &maxSize);
};


/**
 *  This class extends the OpenCV ParallelLoopBody for efficiently parallelized
 *  computations. Within the corresponding for loop, every pixel of a given 2D
 *  signed distance transform is mapped to its correspoding value in the
 *  smoothed Heaviside function representation.
 */
class Parallel_For_convertToHeaviside: public cv::ParallelLoopBody
{
private:
    cv::Mat _sdt, _heaviside;
    
    float *sdtData, *hsData;
    
    int _threads;
    
public:
    Parallel_For_convertToHeaviside(const cv::Mat &sdt, cv::Mat &heaviside, int threads)
    {
        _sdt = sdt;
        
        heaviside.create(_sdt.rows, _sdt.cols, CV_32FC1);
        _heaviside = heaviside;
        
        sdtData = _sdt.ptr<float>();
        hsData = _heaviside.ptr<float>();
        
        _threads = threads;
    }
    
    virtual void operator()( const cv::Range &r ) const
    {
        int range = _sdt.rows/_threads;
        
        int yEnd = r.end*range;
        if(r.end == _threads)
        {
            yEnd = _sdt.rows;
        }
        
        float s = 1.2f;
        
        for(int y = r.start*range; y < yEnd; y++)
        {
            float* sdtRow = sdtData + y*_sdt.cols;
            float* hsRow = hsData + y*_heaviside.cols;
            
            for(int x = 0; x < _sdt.cols; x++)
            {
                float dist = sdtRow[x];
                hsRow[x] = (fabs(dist) <= 8.0f) ? 1.0f/float(CV_PI)*(-atan(dist*s)) + 0.5f : -1.0f;
            }
        }
    }
};


#endif /* TEMPLATE_VIEW_H */
