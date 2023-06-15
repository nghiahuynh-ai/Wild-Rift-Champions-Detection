# Wild Drift Champion Classifier

Preprocessing steps:

* Dectect circles in test images. A circle can be described using three parameters - row index, column index, and radius. The row and column indices represent the location of the circle's center.

  ![1686768517846](image/README/1686768517846.png)
* Select the left most circle in the detected list (the circle that has the smallest column index).
* Crop the image precisely to the diameter of the selected circle's edge.
* Resize the cropped images to 42 x 42 pixels to match the ground truth's size.

  `Test image:` 		![1686768796531](image/README/1686768796531.png)		`Ground truth:` 			![1686803011753](image/README/1686803011753.png)

- Mask the pixels in the ground truth and test image that lie beyond the boundary of the detected circle.
  `Cropped test image:` 	![1686824950768](image/README/1686824950768.png)		`Cropped ground truth:` 	![1686824974165](image/README/1686824974165.png)

Compare test images to groundtruths:

* Distances: Structural Similarity Index Measure (SSIM), Peak Signal-to-Noise Ratio (PSNR).
* For each test image, iterate through the ground truth set and calculate SSIM and PSNR.
* The ouput of a test image is the label of the reference image that has the highest score (SSIM/PSNR) when compared to the test image.

Accuracy:

* Without masking:
  * SSIM Measure: 52.041%
  * PSNR Measure: 40.816%
* With masking:
  * SSIM Measure: 59.184%
  * PSNR Measure: 53.061%

Run:

```
py main.py <path/to/refrence> <path/to/test_data>
```
