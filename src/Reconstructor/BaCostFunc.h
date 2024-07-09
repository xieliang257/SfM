#pragma once
#include <ceres/ceres.h>

namespace sfm {
ceres::CostFunction* CreateCostFunction(const int intrinsicSize, const double& xObserve, const double& yObserve);
}