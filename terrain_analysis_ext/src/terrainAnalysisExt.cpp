// Copyright 2024 Hongbiao Zhu
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Original work based on sensor_scan_generation package by Hongbiao Zhu.

#include <math.h>
#include <queue>

#include "nav_msgs/msg/odometry.hpp"
#include "pcl/filters/voxel_grid.h"
#include "pcl/kdtree/kdtree_flann.h"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
#include "pcl_conversions/pcl_conversions.h"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joy.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "std_msgs/msg/float32.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2_ros/transform_broadcaster.h"

double scanVoxelSize = 0.1;
double decayTime = 10.0;
double noDecayDis = 0;
double clearingDis = 30.0;
bool clearingCloud = false;
bool useSorting = false;
double quantileZ = 0.25;
double vehicleHeight = 1.5;
int voxelPointUpdateThre = 100;
double voxelTimeUpdateThre = 2.0;
double lowerBoundZ = -1.5;
double upperBoundZ = 1.0;
double disRatioZ = 0.1;
bool checkTerrainConn = true;
double terrainUnderVehicle = -0.75;
double terrainConnThre = 0.5;
double ceilingFilteringThre = 2.0;
double localTerrainMapRadius = 4.0;

// terrain voxel parameters
float terrainVoxelSize = 2.0;
int terrainVoxelShiftX = 0;
int terrainVoxelShiftY = 0;
const int terrainVoxelWidth = 41;
int terrainVoxelHalfWidth = (terrainVoxelWidth - 1) / 2;
constexpr int kTerrainVoxelNum = terrainVoxelWidth * terrainVoxelWidth;

// planar voxel parameters
float planarVoxelSize = 0.4;
const int planarVoxelWidth = 101;
int planarVoxelHalfWidth = (planarVoxelWidth - 1) / 2;
constexpr int kPlanarVoxelNum = planarVoxelWidth * planarVoxelWidth;

pcl::PointCloud<pcl::PointXYZI>::Ptr
    laserCloud(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr
    laserCloudCrop(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr
    laserCloudDwz(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr
    terrainCloud(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr
    terrainCloudElev(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr
    terrainCloudLocal(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr terrainVoxelCloud[kTerrainVoxelNum];

int terrainVoxelUpdateNum[kTerrainVoxelNum] = {0};
float terrainVoxelUpdateTime[kTerrainVoxelNum] = {0};
float planarVoxelElev[kPlanarVoxelNum] = {0};
int planarVoxelConn[kPlanarVoxelNum] = {0};
std::vector<float> planarPointElev[kPlanarVoxelNum];
std::queue<int> planarVoxelQueue;

double laserCloudTime = 0;
bool newlaserCloud = false;

double systemInitTime = 0;
bool systemInited = false;

float vehicleRoll = 0, vehiclePitch = 0, vehicleYaw = 0;
float vehicleX = 0, vehicleY = 0, vehicleZ = 0;

pcl::VoxelGrid<pcl::PointXYZI> downSizeFilter;
pcl::KdTreeFLANN<pcl::PointXYZI> kdtree;

// state estimation callback function
void odometryHandler(const nav_msgs::msg::Odometry::ConstSharedPtr odom) {
  double roll, pitch, yaw;
  geometry_msgs::msg::Quaternion geoQuat = odom->pose.pose.orientation;
  tf2::Matrix3x3(tf2::Quaternion(geoQuat.x, geoQuat.y, geoQuat.z, geoQuat.w))
      .getRPY(roll, pitch, yaw);

  vehicleRoll = roll;
  vehiclePitch = pitch;
  vehicleYaw = yaw;
  vehicleX = odom->pose.pose.position.x;
  vehicleY = odom->pose.pose.position.y;
  vehicleZ = odom->pose.pose.position.z;
}

// registered laser scan callback function
void laserCloudHandler(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr laserCloud2) {
  laserCloudTime = rclcpp::Time(laserCloud2->header.stamp).seconds();

  if (!systemInited) {
    systemInitTime = laserCloudTime;
    systemInited = true;
  }

  laserCloud->clear();
  pcl::fromROSMsg(*laserCloud2, *laserCloud);

  pcl::PointXYZI point;
  laserCloudCrop->clear();
  int laserCloudSize = laserCloud->points.size();
  for (int i = 0; i < laserCloudSize; i++) {
    point = laserCloud->points[i];

    float pointX = point.x;
    float pointY = point.y;
    float pointZ = point.z;

    float dis = sqrt((pointX - vehicleX) * (pointX - vehicleX) +
                     (pointY - vehicleY) * (pointY - vehicleY));
    if (pointZ - vehicleZ > lowerBoundZ - disRatioZ * dis &&
        pointZ - vehicleZ < upperBoundZ + disRatioZ * dis &&
        dis < terrainVoxelSize * (terrainVoxelHalfWidth + 1)) {
      point.x = pointX;
      point.y = pointY;
      point.z = pointZ;
      point.intensity = laserCloudTime - systemInitTime;
      laserCloudCrop->push_back(point);
    }
  }

  newlaserCloud = true;
}

// local terrain cloud callback function
void terrainCloudLocalHandler(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr terrainCloudLocal2) {
  terrainCloudLocal->clear();
  pcl::fromROSMsg(*terrainCloudLocal2, *terrainCloudLocal);
}

// joystick callback function
void joystickHandler(const sensor_msgs::msg::Joy::ConstSharedPtr joy) {
  if (joy->buttons[5] > 0.5) {
    clearingCloud = true;
  }
}

// cloud clearing callback function
void clearingHandler(const std_msgs::msg::Float32::ConstSharedPtr dis) {
  clearingDis = dis->data;
  clearingCloud = true;
}

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto nh = rclcpp::Node::make_shared("terrainAnalysisExt");

  nh->declare_parameter<double>("scanVoxelSize", scanVoxelSize);
  nh->declare_parameter<double>("decayTime", decayTime);
  nh->declare_parameter<double>("noDecayDis", noDecayDis);
  nh->declare_parameter<double>("clearingDis", clearingDis);
  nh->declare_parameter<bool>("useSorting", useSorting);
  nh->declare_parameter<double>("quantileZ", quantileZ);
  nh->declare_parameter<double>("vehicleHeight", vehicleHeight);
  nh->declare_parameter<int>("voxelPointUpdateThre", voxelPointUpdateThre);
  nh->declare_parameter<double>("voxelTimeUpdateThre", voxelTimeUpdateThre);
  nh->declare_parameter<double>("lowerBoundZ", lowerBoundZ);
  nh->declare_parameter<double>("upperBoundZ", upperBoundZ);
  nh->declare_parameter<double>("disRatioZ", disRatioZ);
  nh->declare_parameter<bool>("checkTerrainConn", checkTerrainConn);
  nh->declare_parameter<double>("terrainUnderVehicle", terrainUnderVehicle);
  nh->declare_parameter<double>("terrainConnThre", terrainConnThre);
  nh->declare_parameter<double>("ceilingFilteringThre", ceilingFilteringThre);
  nh->declare_parameter<double>("localTerrainMapRadius", localTerrainMapRadius);

  nh->get_parameter("scanVoxelSize", scanVoxelSize);
  nh->get_parameter("decayTime", decayTime);
  nh->get_parameter("noDecayDis", noDecayDis);
  nh->get_parameter("clearingDis", clearingDis);
  nh->get_parameter("useSorting", useSorting);
  nh->get_parameter("quantileZ", quantileZ);
  nh->get_parameter("vehicleHeight", vehicleHeight);
  nh->get_parameter("voxelPointUpdateThre", voxelPointUpdateThre);
  nh->get_parameter("voxelTimeUpdateThre", voxelTimeUpdateThre);
  nh->get_parameter("lowerBoundZ", lowerBoundZ);
  nh->get_parameter("upperBoundZ", upperBoundZ);
  nh->get_parameter("disRatioZ", disRatioZ);
  nh->get_parameter("checkTerrainConn", checkTerrainConn);
  nh->get_parameter("terrainUnderVehicle", terrainUnderVehicle);
  nh->get_parameter("terrainConnThre", terrainConnThre);
  nh->get_parameter("ceilingFilteringThre", ceilingFilteringThre);
  nh->get_parameter("localTerrainMapRadius", localTerrainMapRadius);

  auto subOdometry = nh->create_subscription<nav_msgs::msg::Odometry>(
      "lidar_odometry", 5, odometryHandler);

  auto subLaserCloud = nh->create_subscription<sensor_msgs::msg::PointCloud2>(
      "registered_scan", 5, laserCloudHandler);

  auto subJoystick =
      nh->create_subscription<sensor_msgs::msg::Joy>("joy", 5, joystickHandler);

  auto subClearing = nh->create_subscription<std_msgs::msg::Float32>(
      "cloud_clearing", 5, clearingHandler);

  auto subTerrainCloudLocal =
      nh->create_subscription<sensor_msgs::msg::PointCloud2>(
          "terrain_map", 2, terrainCloudLocalHandler);

  auto pubTerrainCloud =
      nh->create_publisher<sensor_msgs::msg::PointCloud2>("terrain_map_ext", 2);

  for (int i = 0; i < kTerrainVoxelNum; i++) {
    terrainVoxelCloud[i].reset(new pcl::PointCloud<pcl::PointXYZI>());
  }

  downSizeFilter.setLeafSize(scanVoxelSize, scanVoxelSize, scanVoxelSize);

  std::vector<int> pointIdxNKNSearch;
  std::vector<float> pointNKNSquaredDistance;

  rclcpp::Rate rate(100);
  bool status = rclcpp::ok();
  while (status) {
    rclcpp::spin_some(nh);

    if (newlaserCloud) {
      newlaserCloud = false;

      // terrain voxel roll over
      float terrainVoxelCenX = terrainVoxelSize * terrainVoxelShiftX;
      float terrainVoxelCenY = terrainVoxelSize * terrainVoxelShiftY;

      while (vehicleX - terrainVoxelCenX < -terrainVoxelSize) {
        for (int indY = 0; indY < terrainVoxelWidth; indY++) {
          pcl::PointCloud<pcl::PointXYZI>::Ptr terrainVoxelCloudPtr =
              terrainVoxelCloud[terrainVoxelWidth * (terrainVoxelWidth - 1) +
                                indY];
          for (int index = terrainVoxelWidth - 1; index >= 1; index--) {
            terrainVoxelCloud[terrainVoxelWidth * index + indY] =
                terrainVoxelCloud[terrainVoxelWidth * (index - 1) + indY];
          }
          terrainVoxelCloud[indY] = terrainVoxelCloudPtr;
          terrainVoxelCloud[indY]->clear();
        }
        terrainVoxelShiftX--;
        terrainVoxelCenX = terrainVoxelSize * terrainVoxelShiftX;
      }

      while (vehicleX - terrainVoxelCenX > terrainVoxelSize) {
        for (int indY = 0; indY < terrainVoxelWidth; indY++) {
          pcl::PointCloud<pcl::PointXYZI>::Ptr terrainVoxelCloudPtr =
              terrainVoxelCloud[indY];
          for (int index = 0; index < terrainVoxelWidth - 1; index++) {
            terrainVoxelCloud[terrainVoxelWidth * index + indY] =
                terrainVoxelCloud[terrainVoxelWidth * (index + 1) + indY];
          }
          terrainVoxelCloud[terrainVoxelWidth * (terrainVoxelWidth - 1) +
                            indY] = terrainVoxelCloudPtr;
          terrainVoxelCloud[terrainVoxelWidth * (terrainVoxelWidth - 1) + indY]
              ->clear();
        }
        terrainVoxelShiftX++;
        terrainVoxelCenX = terrainVoxelSize * terrainVoxelShiftX;
      }

      while (vehicleY - terrainVoxelCenY < -terrainVoxelSize) {
        for (int index = 0; index < terrainVoxelWidth; index++) {
          pcl::PointCloud<pcl::PointXYZI>::Ptr terrainVoxelCloudPtr =
              terrainVoxelCloud[terrainVoxelWidth * index +
                                (terrainVoxelWidth - 1)];
          for (int indY = terrainVoxelWidth - 1; indY >= 1; indY--) {
            terrainVoxelCloud[terrainVoxelWidth * index + indY] =
                terrainVoxelCloud[terrainVoxelWidth * index + (indY - 1)];
          }
          terrainVoxelCloud[terrainVoxelWidth * index] = terrainVoxelCloudPtr;
          terrainVoxelCloud[terrainVoxelWidth * index]->clear();
        }
        terrainVoxelShiftY--;
        terrainVoxelCenY = terrainVoxelSize * terrainVoxelShiftY;
      }

      while (vehicleY - terrainVoxelCenY > terrainVoxelSize) {
        for (int index = 0; index < terrainVoxelWidth; index++) {
          pcl::PointCloud<pcl::PointXYZI>::Ptr terrainVoxelCloudPtr =
              terrainVoxelCloud[terrainVoxelWidth * index];
          for (int indY = 0; indY < terrainVoxelWidth - 1; indY++) {
            terrainVoxelCloud[terrainVoxelWidth * index + indY] =
                terrainVoxelCloud[terrainVoxelWidth * index + (indY + 1)];
          }
          terrainVoxelCloud[terrainVoxelWidth * index +
                            (terrainVoxelWidth - 1)] = terrainVoxelCloudPtr;
          terrainVoxelCloud[terrainVoxelWidth * index + (terrainVoxelWidth - 1)]
              ->clear();
        }
        terrainVoxelShiftY++;
        terrainVoxelCenY = terrainVoxelSize * terrainVoxelShiftY;
      }

      // stack registered laser scans
      pcl::PointXYZI point;
      int laserCloudCropSize = laserCloudCrop->points.size();
      for (int i = 0; i < laserCloudCropSize; i++) {
        point = laserCloudCrop->points[i];

        int index =
            static_cast<int>((point.x - vehicleX + terrainVoxelSize / 2) /
                             terrainVoxelSize) +
            terrainVoxelHalfWidth;
        int indY =
            static_cast<int>((point.y - vehicleY + terrainVoxelSize / 2) /
                             terrainVoxelSize) +
            terrainVoxelHalfWidth;

        if (point.x - vehicleX + terrainVoxelSize / 2 < 0)
          indX--;
        if (point.y - vehicleY + terrainVoxelSize / 2 < 0)
          indY--;

        if (index >= 0 && index < terrainVoxelWidth && indY >= 0 &&
            indY < terrainVoxelWidth) {
          terrainVoxelCloud[terrainVoxelWidth * index + indY]->push_back(point);
          terrainVoxelUpdateNum[terrainVoxelWidth * index + indY]++;
        }
      }

      for (int ind = 0; ind < kTerrainVoxelNum; ind++) {
        if (terrainVoxelUpdateNum[ind] >= voxelPointUpdateThre ||
            laserCloudTime - systemInitTime - terrainVoxelUpdateTime[ind] >=
                voxelTimeUpdateThre ||
            clearingCloud) {
          pcl::PointCloud<pcl::PointXYZI>::Ptr terrainVoxelCloudPtr =
              terrainVoxelCloud[ind];

          laserCloudDwz->clear();
          downSizeFilter.setInputCloud(terrainVoxelCloudPtr);
          downSizeFilter.filter(*laserCloudDwz);

          terrainVoxelCloudPtr->clear();
          int laserCloudDwzSize = laserCloudDwz->points.size();
          for (int i = 0; i < laserCloudDwzSize; i++) {
            point = laserCloudDwz->points[i];
            float dis = sqrt((point.x - vehicleX) * (point.x - vehicleX) +
                             (point.y - vehicleY) * (point.y - vehicleY));
            if (point.z - vehicleZ > lowerBoundZ - disRatioZ * dis &&
                point.z - vehicleZ < upperBoundZ + disRatioZ * dis &&
                (laserCloudTime - systemInitTime - point.intensity <
                     decayTime ||
                 dis < noDecayDis) &&
                !(dis < clearingDis && clearingCloud)) {
              terrainVoxelCloudPtr->push_back(point);
            }
          }

          terrainVoxelUpdateNum[ind] = 0;
          terrainVoxelUpdateTime[ind] = laserCloudTime - systemInitTime;
        }
      }

      terrainCloud->clear();
      for (int index = terrainVoxelHalfWidth - 10;
           index <= terrainVoxelHalfWidth + 10; index++) {
        for (int indY = terrainVoxelHalfWidth - 10;
             indY <= terrainVoxelHalfWidth + 10; indY++) {
          *terrainCloud += *terrainVoxelCloud[terrainVoxelWidth * index + indY];
        }
      }

      // estimate ground and compute elevation for each point
      for (int i = 0; i < kPlanarVoxelNum; i++) {
        planarVoxelElev[i] = 0;
        planarVoxelConn[i] = 0;
        planarPointElev[i].clear();
      }

      int terrainCloudSize = terrainCloud->points.size();
      for (int i = 0; i < terrainCloudSize; i++) {
        point = terrainCloud->points[i];
        float dis = sqrt((point.x - vehicleX) * (point.x - vehicleX) +
                         (point.y - vehicleY) * (point.y - vehicleY));
        if (point.z - vehicleZ > lowerBoundZ - disRatioZ * dis &&
            point.z - vehicleZ < upperBoundZ + disRatioZ * dis) {
          int index =
              static_cast<int>((point.x - vehicleX + planarVoxelSize / 2) /
                               planarVoxelSize) +
              planarVoxelHalfWidth;
          int indY =
              static_cast<int>((point.y - vehicleY + planarVoxelSize / 2) /
                               planarVoxelSize) +
              planarVoxelHalfWidth;

          if (point.x - vehicleX + planarVoxelSize / 2 < 0)
            indX--;
          if (point.y - vehicleY + planarVoxelSize / 2 < 0)
            indY--;

          for (int dX = -1; dX <= 1; dX++) {
            for (int dY = -1; dY <= 1; dY++) {
              if (index + dX >= 0 && index + dX < planarVoxelWidth &&
                  indY + dY >= 0 && indY + dY < planarVoxelWidth) {
                planarPointElev[planarVoxelWidth * (index + dX) + indY + dY]
                    .push_back(point.z);
              }
            }
          }
        }
      }

      if (useSorting) {
        for (int i = 0; i < kPlanarVoxelNum; i++) {
          int planarPointElevSize = planarPointElev[i].size();
          if (planarPointElevSize > 0) {
            sort(planarPointElev[i].begin(), planarPointElev[i].end());

            int quantileID = static_cast<int>(quantileZ * planarPointElevSize);
            if (quantileID < 0)
              quantileID = 0;
            else if (quantileID >= planarPointElevSize)
              quantileID = planarPointElevSize - 1;

            planarVoxelElev[i] = planarPointElev[i][quantileID];
          }
        }
      } else {
        for (int i = 0; i < kPlanarVoxelNum; i++) {
          int planarPointElevSize = planarPointElev[i].size();
          if (planarPointElevSize > 0) {
            float minZ = 1000.0;
            int minID = -1;
            for (int j = 0; j < planarPointElevSize; j++) {
              if (planarPointElev[i][j] < minZ) {
                minZ = planarPointElev[i][j];
                minID = j;
              }
            }

            if (minID != -1) {
              planarVoxelElev[i] = planarPointElev[i][minID];
            }
          }
        }
      }

      // check terrain connectivity to remove ceiling
      if (checkTerrainConn) {
        int ind =
            planarVoxelWidth * planarVoxelHalfWidth + planarVoxelHalfWidth;
        if (planarPointElev[ind].size() == 0)
          planarVoxelElev[ind] = vehicleZ + terrainUnderVehicle;

        planarVoxelQueue.push(ind);
        planarVoxelConn[ind] = 1;
        while (!planarVoxelQueue.empty()) {
          int front = planarVoxelQueue.front();
          planarVoxelConn[front] = 2;
          planarVoxelQueue.pop();

          int index = static_cast<int>(front / planarVoxelWidth);
          int indY = front % planarVoxelWidth;
          for (int dX = -10; dX <= 10; dX++) {
            for (int dY = -10; dY <= 10; dY++) {
              if (index + dX >= 0 && index + dX < planarVoxelWidth &&
                  indY + dY >= 0 && indY + dY < planarVoxelWidth) {
                ind = planarVoxelWidth * (index + dX) + indY + dY;
                if (planarVoxelConn[ind] == 0 &&
                    planarPointElev[ind].size() > 0) {
                  if (fabs(planarVoxelElev[front] - planarVoxelElev[ind]) <
                      terrainConnThre) {
                    planarVoxelQueue.push(ind);
                    planarVoxelConn[ind] = 1;
                  } else if (fabs(planarVoxelElev[front] -
                                  planarVoxelElev[ind]) >
                             ceilingFilteringThre) {
                    planarVoxelConn[ind] = -1;
                  }
                }
              }
            }
          }
        }
      }

      // compute terrain map beyond localTerrainMapRadius
      terrainCloudElev->clear();
      int terrainCloudElevSize = 0;
      for (int i = 0; i < terrainCloudSize; i++) {
        point = terrainCloud->points[i];
        float dis = sqrt((point.x - vehicleX) * (point.x - vehicleX) +
                         (point.y - vehicleY) * (point.y - vehicleY));
        if (point.z - vehicleZ > lowerBoundZ - disRatioZ * dis &&
            point.z - vehicleZ < upperBoundZ + disRatioZ * dis &&
            dis > localTerrainMapRadius) {
          int index =
              static_cast<int>((point.x - vehicleX + planarVoxelSize / 2) /
                               planarVoxelSize) +
              planarVoxelHalfWidth;
          int indY =
              static_cast<int>((point.y - vehicleY + planarVoxelSize / 2) /
                               planarVoxelSize) +
              planarVoxelHalfWidth;

          if (point.x - vehicleX + planarVoxelSize / 2 < 0)
            indX--;
          if (point.y - vehicleY + planarVoxelSize / 2 < 0)
            indY--;

          if (index >= 0 && index < planarVoxelWidth && indY >= 0 &&
              indY < planarVoxelWidth) {
            int ind = planarVoxelWidth * index + indY;
            float disZ = fabs(point.z - planarVoxelElev[ind]);
            if (disZ < vehicleHeight &&
                (planarVoxelConn[ind] == 2 || !checkTerrainConn)) {
              terrainCloudElev->push_back(point);
              terrainCloudElev->points[terrainCloudElevSize].x = point.x;
              terrainCloudElev->points[terrainCloudElevSize].y = point.y;
              terrainCloudElev->points[terrainCloudElevSize].z = point.z;
              terrainCloudElev->points[terrainCloudElevSize].intensity = disZ;

              terrainCloudElevSize++;
            }
          }
        }
      }

      // merge in local terrain map within localTerrainMapRadius
      int terrainCloudLocalSize = terrainCloudLocal->points.size();
      for (int i = 0; i < terrainCloudLocalSize; i++) {
        point = terrainCloudLocal->points[i];
        float dis = sqrt((point.x - vehicleX) * (point.x - vehicleX) +
                         (point.y - vehicleY) * (point.y - vehicleY));
        if (dis <= localTerrainMapRadius) {
          terrainCloudElev->push_back(point);
        }
      }

      clearingCloud = false;

      // publish points with elevation
      sensor_msgs::msg::PointCloud2 terrainCloud2;
      pcl::toROSMsg(*terrainCloudElev, terrainCloud2);
      terrainCloud2.header.stamp =
          rclcpp::Time(static_cast<uint64_t>(laserCloudTime * 1e9));
      terrainCloud2.header.frame_id = "odom";
      pubTerrainCloud->publish(terrainCloud2);
    }

    status = rclcpp::ok();
    rate.sleep();
  }

  return 0;
}
