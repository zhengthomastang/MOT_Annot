# MOT_Annot

This repo contains the annotation tools in Python for labeling multiple object tracking in a single camera and across multiple cameras, as well as randomly generating re-identification datasets. It was used in *CityFlow: A City-Scale Benchmark for Multi-Target Multi-Camera Vehicle Tracking and Re-Identification*, CVPR 2019.

[[Paper](https://arxiv.org/abs/1903.09254)] [[Presentation](https://youtu.be/fzJe8M2y1s0)] [[Slides](http://zhengthomastang.github.io/files/CityFlow_slides.pdf)] [[Poster](http://zhengthomastang.github.io/files/CityFlow_poster.pdf)]

## Introduction

This package is designed for semi-automatic annotation of multiple object tracking in a single camera and across multiple cameras. It requires baseline detection and single-camera tracking results. Then the user can create manual labels and incorporate them to the tracking results to generate the ground truths. When multi-camera tracking labels are available, there is also a script for randomly generating a re-identification dataset. 

## Getting Started

### Environment

The code was developed and tested with Python 2.7.15 on Windows 10. Other platforms may work but are not fully tested.

### How to Use

We highly recommend to create a virtual environment for the following steps. For example, an introduction to Conda environments can be found [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). 

1. Clone the repo, and change the current working directory to `MOT_Annot`, which will be referred to as `${ANNOT_ROOT}`:
   ```
   cd ${ANNOT_ROOT}
   ```

2. Add the videos from various data splits, scenes, and cameras to be annotated. Your directory tree may look like this:
   ```
   ${ANNOT_ROOT}
    |-- LICENSE
    |-- README.md
    |-- src
    |-- train
        |-- S01
            |-- c001
                |-- vdo.avi
                |-- roi.jpg
                |-- ...
            |-- ...
        |-- ...
    |-- validation
        |-- ...
    `-- test
        |-- ...

   ```

3. Extract frame images from input video files (frame indices starting from 1 by default): 
   ```
   python src/extract_vdo_frms.py --data-root train/S01
   ```

   ![Extracted frame images](figures/frame_images.jpg)

4. Use a baseline object detection method, e.g., [Detectron (Mask/Faster R-CNN)](ode.amazon.com/packages/OrvilleEmpennageInference/trees/mainline), to output detection (and segmentation) results. The results need to be converted to the [MOTChallenge format](https://motchallenge.net/instructions/). An example script for the conversion is given: 
   ```
   python src/convert_to_motchallenge_det.py --data-root train/S01
   ```

5. Plot the detection results to visualize the performance and confirm that it is satisfactory:
   ```
   python src/plot_det_results.py --data-root train/S01
   ```

   ![Plotted detection results](figures/det_results.jpg)

6. Use a baseline single-camera tracking method, e.g., [TrackletNet](https://github.com/GaoangW/TNT/tree/master/AIC19), to output multi-target single-camera (MTSC) tracking results. The results need to be converted to the [MOTChallenge format](https://motchallenge.net/instructions/). An example script for the conversion is given: 
   ```
   python src/convert_to_motchallenge_mtsc.py --data-root train/S01
   ```

7. Plot the MTSC tracking results to visualize the performance and confirm that it is satisfactory:
   ```
   python src/plot_mtsc_results.py --data-root train/S01
   ```

   ![Plotted single-camera tracking results](figures/mtsc_results.jpg)

8. By checking the plotted baseline MTSC results frame by frame, manually create an annotation file, e.g. `annotation.txt`. There are 3 types of operations that can be entered in the annotation file at each row (the order doesn't matter): 
   - Assign a global ID to a vehicle trajectory: `assign,<local_ID>,<global_ID>`
      - The vehicle(s) that are not assigned will be ignored. 
   - Insert an instance to replace an existing one or fill in a missing one: `insert,<frame_index>,<local_ID>,<bbox_x>,<bbox_y>,<bbox_wid>,<bbox_hei>`
      - Use a tool like [IrfanView](https://www.irfanview.com/) to draw/adjust bounding boxes and read the coordinates. 

        ![Annotation by IrfanView](figures/annotation.jpg)

      - The missing instance(s) in a continuous trajectory will be interpolated linearly, so there is no need to insert at every frame index.
      - The instance(s) occluded by more than 50% will be automatically detected and removed. 
   - Remove a range of instance(s): `remove,<frame_range>,<local_ID>`
      - The `<frame_range>` can be represented as `<frame_index>`, `<frm_index_start>-`, `-<frm_index_end>`, or `<frame_index_start>-<frame_index_end>` (inclusive).

9. Incorporate the annotations to the baseline MTSC results and generate the ground truths:
   ```
   python src/generate_ground_truths.py --data-root train/S01
   ``` 

10. Plot the ground truths using the above script for plotting MTSC results (change the input and output paths accordingly) to confirm that the annotations are accurate. If not, modify the corresponding lines in `annotation.txt` and repeat steps 8 and 9 again. 

11. Plot the ground truth crop of each global ID at each frame for further validation: 
    ```
    python src/plot_gt_crops.py --data-root train/S01
    ``` 

    ![Plotted ground truths](figures/ground_truth.jpg)

12. Generate the labels of ground truths for the evaluation system: 
    ```
    python src/generate_ground_truths_eval_system.py --data-root train/S01
    ``` 

13. Generate a random dataset for re-identification (according to the format of the [VeRi dataset](https://vehiclereid.github.io/VeRi/)): 
    ```
    python src/generate_reid_dataset.py --data-root ./ --output-root reid
    ``` 

## References

Please cite these papers if you use this code in your research:

    @inproceedings{Tang19CityFlow,
      author = {Zheng Tang and Milind Naphade and Ming-Yu Liu and Xiaodong Yang and Stan Birchfield and Shuo Wang and Ratnesh Kumar and David Anastasiu and Jenq-Neng Hwang},
      title = {City{F}low: {A} city-scale benchmark for multi-target multi-camera vehicle tracking and re-identification},
      booktitle = {Proc. of the Conference on Computer Vision and Pattern Recognition (CVPR)},
      pages = {8797–-8806},
      address = {Long Beach, CA, USA},
      month = Jun,
      year = 2019
    }

    @inproceedings{Naphade19AIC19,
      author = {Milind Naphade and Zheng Tang and Ming-Ching Chang and David C. Anastasiu and Anuj Sharma and Rama Chellappa and Shuo Wang and Pranamesh Chakraborty and Tingting Huang and Jenq-Neng Hwang and Siwei Lyu},
      title = {The 2019 {AI} {C}ity {C}hallenge},
      booktitle = {Proc. of the Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
      pages = {452--460},
      address = {Long Beach, CA, USA},
      month = Jun,
      year = 2019
    }

## License

Code in the repository, unless otherwise specified, is licensed under the [MIT License](LICENSE).

## Contact

For any questions please contact [Zheng (Thomas) Tang](https://github.com/zhengthomastang).
