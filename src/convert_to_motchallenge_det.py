import os
import cv2
import argparse


def main(args):
    expand_ratio = 0.1
    bbox_area_min_threshold = 1000
    bbox_area_max_threshold = 0.50
    seq_list = os.listdir(args.data_root)

    for seq_name in seq_list:
        path_data = os.path.join(args.data_root, seq_name)
        path_image = os.path.join(path_data, 'img1/000001.jpg')
        image = cv2.imread(path_image)
        height, width, channels = image.shape

        in_det_file_path = os.path.join(args.data_root, 'det/det_mask_rcnn_orig.txt')
        in_segm_file_path = os.path.join(args.data_root, 'segm/segm_mask_rcnn_orig.txt')
        out_det_file_path = os.path.join(args.data_root, 'det/det_mask_rcnn.txt')
        out_segm_file_path = os.path.join(args.data_root, 'segm/segm_mask_rcnn.txt')

        frame_area = width * height
        out_det_file = open(out_det_file_path, 'w+')
        out_segm_file = open(out_segm_file_path, 'w+')

        with open(in_det_file_path) as in_det_file, open(in_segm_file_path) as in_segm_file:
            for det, segm in zip(in_det_file, in_segm_file):
                det_list = det.split(',')
                class_name = det_list.pop().rstrip()
                if class_name in ('car', 'truck', 'bus'):
                    # change frame index from 0-based to 1-based
                    det_list[0] = str(int(det_list[0]) + 1)
                    # expand bounding boxes
                    x_min = float(det_list[2])
                    y_min = float(det_list[3])
                    bbox_width = float(det_list[4])
                    bbox_height = float(det_list[5])
                    x_min_expand = x_min - (bbox_width * expand_ratio / 2)
                    y_min_expand = y_min - (bbox_height * expand_ratio / 2)
                    bbox_width_expand = bbox_width * (1 + expand_ratio)
                    bbox_height_expand = bbox_height  * (1 + expand_ratio)
                    if x_min_expand < 0:
                        x_min_expand = 0
                    if y_min_expand < 0:
                        y_min_expand = 0
                    if x_min_expand + bbox_width_expand > width:
                        bbox_width_expand = width - x_min_expand
                    if y_min_expand + bbox_height_expand > height:
                        bbox_height_expand = height - y_min_expand
                    bbox_area_expand = bbox_width_expand * bbox_height_expand
                    if bbox_area_expand > bbox_area_min_threshold and bbox_area_expand < frame_area * bbox_area_max_threshold: 
                        det_list[2] = '%.3f' % x_min_expand
                        det_list[3] = '%.3f' % y_min_expand
                        det_list[4] = '%.3f' % bbox_width_expand
                        det_list[5] = '%.3f' % bbox_height_expand
                        # output detection and segmentation results
                        det_out = ','.join(det_list)
                        segm_out = det_out + ',' + segm.rstrip()
                        out_det_file.write(det_out + '\n')
                        out_segm_file.write(segm_out + '\n')

        out_det_file.close()
        out_segm_file.close()


if __name__ == '__main__':
    print("Loading parameters...")
    parser = argparse.ArgumentParser(description='Convert object detection results to the MOTChallenge format')
    parser.add_argument('--data-root', dest='data_root', default='train/S01',
                        help='dataset root path')

    args = parser.parse_args()

    main(args)
