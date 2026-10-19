# INS3155 Computer Vision - Midterm Project (Code and Results)

Traditional image processing only (no deep learning): image filtering (Part A), 3D reconstruction from stereo (Part B) and multi-view image stitching (Part C).

## 1. Files

| File | Description |
|---|---|
| `INS3155_Midterm_CV.ipynb` | Main notebook for Google Colab (all three parts) |
| `INS3155_Midterm_CV.py` | The same code as a plain Python script (cells are separated by `# %%`) |
| `README.md` | This file |

Input images are not bundled with the code: add your own images with the file names given in Section 3 (or edit the names in the CONFIG cell).

## 2. Environment

- Python 3 with: `opencv-python` (4.x), `numpy`, `scipy`, `pandas`, `matplotlib`, `scikit-image`. All are available on Google Colab.
- **OpenCV must be 4.x.** AKAZE is not available in OpenCV 5.0.0 (`opencv-python`). The first cell ("OPENCV CHECK") installs `opencv-python==4.10.0.84` automatically if AKAZE is missing. Run that cell before any other cell in a fresh session. If `cv2` was already imported, restart the runtime and run all cells again.
- SURF is not used (patented, not in the standard OpenCV build). Part C compares SIFT, ORB and AKAZE.
- If a detector is not available in the installed OpenCV build, it is skipped automatically (at least SIFT and ORB are required).

## 3. How to run (Google Colab)

1. Open `INS3155_Midterm_CV.ipynb` in Google Colab.
2. Upload the input images with the file browser (folder icon on the left), or mount Google Drive and change the paths.
3. Edit the **CONFIG** cell if your file names differ:

| Variable | Default | Meaning |
|---|---|---|
| `PATH_A` | `part_a.jpg` | One clean image for Part A |
| `PATH_B_LEFT`, `PATH_B_RIGHT` | `part_b_left.png`, `part_b_right.png` | Rectified stereo pair for Part B |
| `PATH_B_GT` | `None` | Ground-truth disparity of the left image (optional) |
| `GT_SCALE` | `4.0` | Ground truth = disparity x 4 (Middlebury 2003 `disp2.png`) |
| `FOCAL_PX`, `BASELINE` | `None`, `1.0` | Camera parameters for the point cloud (see Section 6) |
| `RECTIFY_UNCALIBRATED` | `False` | Set `True` if the stereo pair is not rectified |
| `PATHS_C` | `part_c_1.jpg` ... `part_c_4.jpg` | Four panorama images ordered left to right |
| `REF_IDX` | `1` | Reference image index (0..3) of the panorama |
| `SEED` | `42` | Random seed for the noise (reproducible results) |

4. Run all cells (Runtime -> Run all). The last cell creates `results_outputs.zip` and downloads it.

Suggested Part B data: the Teddy or Cones pair from the Middlebury stereo dataset (`https://vision.middlebury.edu/stereo/data/`, 2003 scenes): `im2.png` (left), `im6.png` (right), `disp2.png` (ground truth).

To run the script locally instead: `python INS3155_Midterm_CV.py` from a folder containing the input images (use a non-interactive Matplotlib backend if there is no display, e.g. `MPLBACKEND=Agg`).

## 4. What each part does

**Part A - Image filtering and enhancement**
- Noise models: Gaussian (sigma = 25) and Salt-and-Pepper (5 percent).
- Filters: Mean (box), Gaussian (sigma = 1.0 and 2.0), Median, Laplacian sharpening (c = 0.5). Kernel sizes 3x3, 5x5, 7x7.
- Metrics: PSNR (dB), SSIM, MSE, execution time (ms).
- The noise level, sigma values, `LAP_C` and the reference kernel size of the grids (`GRID_K`, `GRID_SIGMA`) can be changed in the Part A cells.

**Part B - 3D reconstruction from stereo**
- SIFT matches and Lowe ratio test, Fundamental matrix with RANSAC, epipolar lines, Sampson and epipolar errors.
- Disparity with Block Matching (BM) and Semi-Global Block Matching (SGBM) for several `blockSize` and `numDisparities` values.
- Metrics: valid disparity ratio, Sampson error, runtime, and RMSE / bad-pixel rate if a ground truth is given.
- 3D point cloud with Z = fB / d, saved as a `.ply` file (can be opened in MeshLab, CloudCompare or Open3D).

**Part C - Multi-view image stitching**
- Detectors: SIFT, ORB, AKAZE. KNN matching with Lowe ratio test, homography with RANSAC (thresholds 1, 3, 5 px).
- Warping onto a common canvas and blending: feathering (main results) and Laplacian-pyramid multi-band blending (bonus).
- Metrics: keypoints, initial matches, inliers, inlier ratio, mean reprojection RMSE, total stitching time.

## 5. Outputs

Everything is written to `outputs/` (also zipped as `results_outputs.zip`).

| Path | Content |
|---|---|
| `outputs/partA/partA_grid_*.png` | Grids: Original, Noisy, Denoised, Sharpened (one per noise type) |
| `outputs/partA/partA_kernel_comparison_*.png` | Zoomed edge region for 3x3, 5x5, 7x7 and sigma values |
| `outputs/partA/partA_laplacian_sharpening_clean_image.png` | Halo / ringing of Laplacian sharpening on the clean image |
| `outputs/partA/partA_metrics_vs_kernel_size.png` | PSNR and SSIM versus kernel size |
| `outputs/partA/all_results/` | Full-frame result of every filter setting, one folder per noise type |
| `outputs/partB/B_epipolar_lines.png` | Epipolar lines in both images |
| `outputs/partB/B_disparity_bm_vs_sgbm.png` | Colorized disparity maps (best BM and best SGBM, ground truth if given) |
| `outputs/partB/B_pointcloud_views.png`, `pointcloud_sgbm.ply` | Point cloud views and the point cloud file |
| `outputs/partC/C_keypoints_and_matches.png` | Keypoints and inlier matches for each detector |
| `outputs/partC/C_panoramas.png`, `panorama_*.jpg` | Final panoramas |
| `outputs/partC/C_blending_bonus.png` | Feathering versus multi-band blending (zoom) |
| `outputs/tables/*.csv` | All quantitative tables of Parts A, B and C |
| `outputs/summary_charts.png` | Summary chart for the comparative analysis |

## 6. Notes and limitations

- Part B: without real calibration the point cloud has the correct shape but an arbitrary scale. Set `FOCAL_PX` and `BASELINE` (for example from the Middlebury `calib.txt`) to get metric depth.
- For an already rectified stereo pair the Sampson error of the disparity matches is close to zero by construction.
- Part C: the pipeline assumes the four images overlap and are ordered left to right. Strong parallax or a very wide field of view may cause perspective distortion because a single homography per pair is used.
- The code was tested with OpenCV 4.10 on synthetic test images (a standard test photo, a generated stereo pair with known disparity and four overlapping crops). It has not been verified on the real data used for the report.
