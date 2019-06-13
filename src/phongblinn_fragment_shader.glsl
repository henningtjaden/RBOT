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

#version 330

uniform vec3 uColor;
uniform float uAlpha;
uniform float uShininess;

in vec3 vPosition;
in vec3 vLightPosition1;
in vec3 vLightPosition2;
in vec3 vLightPosition3;
in vec3 vNormal;

const vec3 ambientColor = vec3(0.0, 0.0, 0.0);
const vec3 specColor = vec3(0.5, 0.5, 0.5);

layout(location = 0) out vec4 fragColor;

void main()
{
	vec3 normal = normalize(vNormal);
    
    vec3 lightDir1 = normalize(vLightPosition1);
    vec3 lightDir2 = normalize(vLightPosition2);
    vec3 lightDir3 = normalize(vLightPosition3);
	
    float lambertian1 = max(dot(lightDir1, normal), 0.0);
    float lambertian2 = max(dot(lightDir2, normal), 0.0);
    float lambertian3 = max(dot(lightDir3, normal), 0.0);
    
    float specular1 = 0.0;
    float specular2 = 0.0;
    float specular3 = 0.0;
    
    vec3 viewDir = normalize(-vPosition);
    
    vec3 halfDir1 = normalize(lightDir1 + viewDir);
    vec3 halfDir2 = normalize(lightDir2 + viewDir);
    vec3 halfDir3 = normalize(lightDir3 + viewDir);
    
    float specAngle1 = max(dot(halfDir1, normal), 0.0);
    float specAngle2 = max(dot(halfDir2, normal), 0.0);
    float specAngle3 = max(dot(halfDir3, normal), 0.0);
    
    specular1 = pow(specAngle1, uShininess);
    specular2 = pow(specAngle2, uShininess);
    specular3 = pow(specAngle3, uShininess);
    
    float lambertian = (lambertian1+lambertian2+lambertian3)/1.0;
    float specular = (specular1+specular2+specular3)/1.0;
    
    fragColor = vec4(ambientColor + lambertian*uColor + specular*specColor, uAlpha);
}
