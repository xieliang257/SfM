#include <Reconstructor/BaCostFunc.h>
#include <ceres/rotation.h>

namespace sfm {

/**
* @brief A struct to model a basic pinhole camera for projection without distortion.
*
* This structure represents a basic pinhole camera model, utilizing focal length and principal
* point coordinates to project 3D points onto the 2D image plane.
*/
struct PinholeModel {
    explicit PinholeModel(const double& xObserve, const double& yObserve)
        :xObserve_(xObserve), yObserve_(yObserve) {}

    template <typename T>
    bool operator()(const T* const intrinsics, const T* const extrinsics, const T* const object, T* residuals) const {
        const T* camR = extrinsics;
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> camT(&extrinsics[3]);
        Eigen::Matrix<T, 3, 1> transformedPoint;
        ceres::AngleAxisRotatePoint(camR, object, transformedPoint.data());
        transformedPoint += camT;

        const Eigen::Matrix<T, 2, 1> projectedPoint = transformedPoint.hnormalized();

        const T& f = intrinsics[0];
        const T& cx = intrinsics[1];
        const T& cy = intrinsics[2];
        residuals[0] = cx + projectedPoint.x() * f - (T)xObserve_;
        residuals[1] = cy + projectedPoint.y() * f - (T)yObserve_;
        return true;
    }

    static ceres::CostFunction* Create(const double& xObserve, const double& yObserve) {
        // <2,3,6,3>
        // 2: dimension of the residuals
        // 3: intrinsic data block(f, cx, cy)
        // 6: extrinsic data block(rx, ry, rz, tx, ty, tz)
        // 3: object point data block(wx, wy, wz)
        return (new ceres::AutoDiffCostFunction<PinholeModel, 2, 3, 6, 3>(new PinholeModel(xObserve, yObserve)));
    }

    // The 2D observation
    const double xObserve_;
    const double yObserve_;
};

struct PinholeK1Model {
    explicit PinholeK1Model(const double& xObserve, const double& yObserve)
        :xObserve_(xObserve), yObserve_(yObserve){}

    template <typename T>
    bool operator()(const T* const intrinsics, const T* const extrinsics, const T* const pt3d, T* residuals) const {
        const T* camR = extrinsics;
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> camT(&extrinsics[3]);
        Eigen::Matrix<T, 3, 1> transformedPoint;
        ceres::AngleAxisRotatePoint(camR, pt3d, transformedPoint.data());
        transformedPoint += camT;

        const Eigen::Matrix<T, 2, 1> projectedPoint = transformedPoint.hnormalized();

        const T& f = intrinsics[0];
        const T& cx = intrinsics[1];
        const T& cy = intrinsics[2];
        const T& k1 = intrinsics[3];

        const T r2 = projectedPoint.squaredNorm();
        const T rCoeff = 1.0 + k1 * r2;

        residuals[0] = cx + (projectedPoint.x() * rCoeff) * f - xObserve_;
        residuals[1] = cy + (projectedPoint.y() * rCoeff) * f - yObserve_;
        return true;
    }

    static ceres::CostFunction* Create(const double& xObserve, const double& yObserve) {
        // <2,4,6,3>
        // 2: dimension of the residuals
        // 4: intrinsic data block(f, cx, cy, k1)
        // 6: extrinsic data block(rx, ry, rz, tx, ty, tz)
        // 3: object point data block(wx, wy, wz)
        return(new ceres::AutoDiffCostFunction<PinholeK1Model, 2, 4, 6, 3>(new PinholeK1Model(xObserve, yObserve)));
    }

    // observations
    const double xObserve_;
    const double yObserve_;
};

ceres::CostFunction* CreateCostFunction(const int intrinsicSize, const double& xObserve, const double& yObserve) {
    if (intrinsicSize == 3) {
        return PinholeModel::Create(xObserve, yObserve);
    }
    else if (intrinsicSize == 4) {
        return PinholeK1Model::Create(xObserve, yObserve);
    }else{
        return nullptr;
    }
}
}