import os
import cv2
import argparse


def check_and_create(folder_path):
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
    return folder_path


def main(args):
    seq_list = os.listdir(args.data_root)

    for seq_name in seq_list:
        path_data = os.path.join(args.data_root, seq_name)
        det_file_path = os.path.join(path_data, 'det/det_mask_rcnn.txt')
        in_img_dir_path = os.path.join(path_data, 'img1')
        out_img_dir_path = os.path.join(path_data, 'det/det_mask_rcnn')
        check_and_create(out_img_dir_path)

        with open(det_file_path) as det_file:
            for det in det_file:
                det_list = det.split(',')
                frame_index = int(det_list[0])
                out_img_path = os.path.join(out_img_dir_path, '%06d.jpg' % frame_index)
                if not os.path.exists(out_img_path):
                    print('Frame Index: %06d' % frame_index)
                    in_img_path  = os.path.join(in_img_dir_path, '%06d.jpg' % frame_index)
                    frame_img = cv2.imread(in_img_path)
                    cv2.imwrite(out_img_path, frame_img)
                frame_img = cv2.imread(out_img_path)
                x_min = float(det_list[2])
                y_min = float(det_list[3])
                bbox_width = float(det_list[4])
                bbox_height = float(det_list[5])
                cv2.rectangle(frame_img, (int(round(x_min)), int(round(y_min))),
                    (int(round(x_min + bbox_width)), int(round(y_min + bbox_height))), 
                    (0,0,255), 6)
                cv2.imwrite(out_img_path, frame_img)


if __name__ == '__main__':
    print("Loading parameters...")
    parser = argparse.ArgumentParser(description='Plot object detection results')
    parser.add_argument('--data-root', dest='data_root', default='train/S01',
                        help='dataset root path')

    args = parser.parse_args()

    main(args)
