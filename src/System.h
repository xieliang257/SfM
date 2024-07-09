#pragma once
#include <memory>
#include "ImageProcessor/ImageProcessor.h"
#include "Reconstructor/Reconstructor.h"

namespace sfm {
/**
* @brief Manages the overall operation of a Structure from Motion (SfM) system.
*
* This class serves as the central coordination point for a Structure from Motion process.
* It handles configuration loading, directs image processing, runs 3D reconstruction, and
* manages the output of the reconstructed point cloud. It simplifies the interaction between
* the user and the complex operations of the SfM process.
*/
class System {
public:
	/**
	* @brief Constructor for the System class.
	*
	* Initializes the system with necessary paths and sets up the required components for SfM.
	*
	* @param configFile Path to the configuration file that includes settings for reconstruction.
	* @param workDir Directory where intermediate and final outputs will be stored.
	*/
	System(const std::string& configFile, const std::string& workDir);

	/**
	 * @brief Executes the complete Structure from Motion (SfM) process.
	 *
	 * This method drives the entire SfM pipeline. It starts by processing images to extract and match features.
	 * It then reconstructs the 3D scene using these features and finally outputs the resulting point cloud
	 * and camera positions to a PLY file. Timing for each major step is recorded and output to give insights
	 * into the performance of each stage.
	 *
	 * @param imageDir Directory containing the images to be processed.
	 */
	void RunSFM(const std::string& imageDir);

	/**
	 * @brief Exports the 3D points and camera positions to a PLY file.
	 *
	 * This function gathers 3D points from SFM features and their associated color information,
	 * along with the camera frustum positions. It filters points based on visibility and the number
	 * of features. The results are then written to a PLY file for visualization or further processing.
	 *
	 * @param dirPath The directory path where the output PLY file will be saved.
	 * @param pSfmFeatures Shared pointer to a map containing SFM features. Each feature includes
	 *                     3D coordinates, color, and associated image features.
	 */
	void OutputPointCloud(const std::string& dirPath, const std::shared_ptr<std::map<size_t, SFMFeature>>& pSfmFeatures);

private:
	// Directory for storing output and intermediate files.
	std::string workDir_;

	// Handles image loading, feature extraction, and matching.
	std::shared_ptr<ImageProcessor> imageProcessor_;

	// Handles the 3D reconstruction process.
	std::shared_ptr<Reconstructor> reconstructor_;
};
}
