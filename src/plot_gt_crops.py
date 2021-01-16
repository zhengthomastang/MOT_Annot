import os
import csv
import cv2
import argparse


def main(args):
    crop_size_min = 1000

    gt_dict = {}
    
    seq_list = os.listdir(args.data_root)

    for seq_name in seq_list:
        gt_path = os.path.join(args.data_root, seq_name, 'gt/gt.txt')

        with open(gt_path) as gt_file:
            gt_reader = csv.reader(gt_file, delimiter=',')
            for row in gt_reader:
                if int(row[1]) not in gt_dict:
                    gt_dict[int(row[1])] = {}
                if seq_name not in gt_dict[int(row[1])]:
                    gt_dict[int(row[1])][seq_name] = {}
                gt_dict[int(row[1])][seq_name][int(row[0])] = (int(row[2]), int(row[3]), int(row[4]), int(row[5]))

    crop_root_dir = os.path.join(args.data_root, 'crops')
    if not os.path.exists(crop_root_dir):
        os.makedirs(crop_root_dir)

    for global_id in sorted(gt_dict.keys()):
        print('%04d' % global_id)
        crop_dir = os.path.join(crop_root_dir, '%04d' % global_id)
        os.makedirs(crop_dir)
        if len(gt_dict[global_id].keys()) < 2:
            print('ERROR: Less than 2 camera IDs for global ID %d' % global_id)
        for seq_name in sorted(gt_dict[global_id].keys()):
            img_dir = os.path.join(args.data_root, seq_name, 'img1')
            for frm_idx in sorted(gt_dict[global_id][seq_name].keys()):
                img_frm_path = os.path.join(img_dir, '%06d.jpg' % frm_idx)
                img_frm = cv2.imread(img_frm_path)
                bbox = gt_dict[global_id][seq_name][frm_idx]
                img_crop = img_frm[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
                crop_size = bbox[2] * bbox[3]
                crop_flag = True
                if crop_size < crop_size_min:
                    crop_flag = False
                img_crop_path = os.path.join(crop_dir, '%s_%06d_%s.jpg' % (seq_name, frm_idx, crop_flag))
                cv2.imwrite(img_crop_path, img_crop)


if __name__ == '__main__':
    print("Loading parameters...")
    parser = argparse.ArgumentParser(description='Plot vehicle crops')
    parser.add_argument('--data-root', dest='data_root', default='train/S01',
                        help='dataset root path')

    args = parser.parse_args()

    main(args)
