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

#ifndef OBJECT3D_H
#define OBJECT3D_H

#include "model.h"

class TCLCHistograms;
class TemplateView;

/**
 *  A representation of a 3D object that provides all nessecary information
 *  for region-based pose estimation using tclc-histograms. It extends the
 *  basic model class by including a set of tclc-histograms and a list of
 *  all corresponding templates used for pose detection.
 */
class Object3D : public Model
{
public:
    /**
     *  Constructor creating a 3D object class from a specified initial 6DOF pose, a
     *  scaling factor, a tracking quality threshhold and a set of distances to the
     *  camera for template generation used within pose detection. Here, also the set
     *  of n tclc-histograms is initialized, with n being the total number of 3D model
     *  vertices.
     *
     *  @param objFilename  The relative path to an OBJ/PLY file describing the model.
     *  @param tx  The models initial translation in X-direction relative to the camera.
     *  @param ty  The models initial translation in Y-direction relative to the camera.
     *  @param tz  The models initial translation in Z-direction relative to the camera.
     *  @param alpha  The models initial Euler angle rotation about X-axis of the camera.
     *  @param beta  The models initial Euler angle rotation about Y-axis of the camera.
     *  @param gamma  The models initial Euler angle rotation about Z-axis of the camera.
     *  @param scale  A scaling factor applied to the model in order change its size independent of the original data.
     *  @param qualityThreshold  The individual quality tracking quality threshold used to decide whether tracking and detection have been successful (should be within [0.5,0.6]).
     *  @param templateDistances  A vector of absolute Z-distance values to be used for template generation (typically 3 values: a close, an intermediate and a far distance)
     */
    Object3D(const std::string objFilename, float tx, float ty, float tz, float alpha, float beta, float gamma, float scale, float qualityThreshold, std::vector<float> &templateDistances);
    
    ~Object3D();
    
    /**
     *  Tells whether the tracking has been lost, i.e. the quality was below
     *  the prescribed threshhold.
     *
     *  @return  True if it has been lost and false otherwise.
     */
    bool isTrackingLost();
    
    /**
     *  Sets the tracking state of the object.
     *
     *  @param val True  if the tracking has been lost, false if it is currently being tracked successfully.
     */
    void setTrackingLost(bool val);
    
    /**
     *  Returns the tracking quality threshold for this object used to decide
     *  whether tracking and detection have been successful.
     *
     *  @return  The tracking quality threshold.
     */
    float getQualityThreshold();

    /**
     *  Returns the set of tclc-histograms associated with this object.
     *
     *  @return  The set of tclc-histograms associated with this object.
     */
    TCLCHistograms *getTCLCHistograms();
    
    /**
     *  Generates all base and neighboring templates required for
     *  the pose detection algorithm after a tracking loss.
     *  Must be called after the rendering buffers of the
     *  corresponding 3D model have been initialized and while
     *  the offscreen rendering OpenGL context is active.
     */
    void generateTemplates();
    
    /**
     *  Returns the set of all pre-generated base and neighboring template views
     *  of this object used during pose detection.
     *
     *  @return  The set of all template views for this object.
     */
    std::vector<TemplateView*> getTemplateViews();
    
    /**
     *  Returns the number of Z-distances used during template view generation
     *  for this object.
     *
     *  @return  The number of Z-distances of the template views for this object.
     */
    int getNumDistances();
    
    /**
     *  Clears all tclc-histograms and resets the pose of the object to the initial
     *  configuration.
     */
    void reset();
    
private:
    bool trackingLost;
    
    float qualityThreshold;
    
    int numDistances;
    
    std::vector<float> templateDistances;
    
    std::vector<cv::Vec3f> baseIcosahedron;
    std::vector<cv::Vec3f> subdivIcosahedron;
    
    TCLCHistograms *tclcHistograms;
    
    std::vector<TemplateView*> baseTemplates;
    std::vector<TemplateView*> neighboringTemplates;
    
};


#endif /* OBJECT3D_H */
