
# Visualization Samples

This directory is intended for user-provided RGB image samples for visualizing StructFlow optical flow and depth/disparity predictions.

Download the corresponding datasets from their official sources and place the required images in the folders described below.

## Sample Directory Structure

| Dataset | Sample Directory | Required Images |
|---|---|---|
| KITTI 2012 | `kitti2012/` | Matching left-camera image pairs (`_10`, `_11`) for optical flow, with corresponding right-camera images for disparity. |
| KITTI 2015 | `kitti2015/benchmark/` | Matching benchmark image pairs, including selected scene IDs, and corresponding right-camera images for disparity. |
| KITTI Raw | `kitti2015/raw_drive/`, `kitti2015/raw_long_drive/` | Consecutive frames from the same driving sequence, with synchronized stereo images for disparity and video generation. |
| MPI Sintel | `sintel/alley_1/`, `sintel/temple_2/`, `sintel/bamboo_1/` | Consecutive RGB frames from the selected sequences, with matching stereo frames when required. |
| FlyingThings3D | `flyingthings3d/C_0001/` | Consecutive RGB frames from the same scene, including matching left and right camera images. |
| Spring | `spring/0001/` | Consecutive RGB frames from the same sequence, with corresponding right-camera images. |

## Preparing the Images

1. Download the images from the official dataset sources.
2. Place the images in their corresponding dataset folders.
3. For optical flow, provide at least two consecutive frames from the same scene.
4. For disparity, provide synchronized left and right images when required by the model.
5. Preserve the original filenames, resolution, and frame ordering.
6. Do not combine unrelated scenes or sequences.
7. You may replace the samples with any other compatible frames from the same dataset.

