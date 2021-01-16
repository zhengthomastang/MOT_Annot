import os
import csv
import argparse


def main(args):
    bbox_area_limit = 1000
    seq_list = os.listdir(args.data_root)

    for seq_name in seq_list:
        path_data = os.path.join(args.data_root, seq_name)
        mtsc_baseline_path = os.path.join(path_data, 'mtsc/mtsc_tnt_mask_rcnn_orig.txt')
        mtsc_baseline_output_path = os.path.join(path_data, 'mtsc/mtsc_tnt_mask_rcnn.txt')

        mtsc_baseline_dict = {}

        with open(mtsc_baseline_path) as mtsc_baseline_file:
            mtsc_baseline_reader = csv.reader(mtsc_baseline_file, delimiter=',')
            for row in mtsc_baseline_reader:
                if int(row[0]) not in mtsc_baseline_dict:
                    mtsc_baseline_dict[int(row[0])] = {}
                if int(row[4]) * int(row[5]) > bbox_area_limit:
                    mtsc_baseline_dict[int(row[0])][int(row[1])] = (int(row[2]), int(row[3]), int(row[4]), int(row[5]))

        mtsc_file = open(mtsc_baseline_output_path, 'w')
        for frm_index in sorted(mtsc_baseline_dict.keys()):
            for global_id in sorted(mtsc_baseline_dict[frm_index].keys()):
                bbox = mtsc_baseline_dict[frm_index][global_id]
                mtsc_file.write('%d,%d,%d,%d,%d,%d,1,-1,-1,-1\n' % (frm_index, global_id, bbox[0], bbox[1], bbox[2], bbox[3]))
        mtsc_file.close()


if __name__ == '__main__':
    print("Loading parameters...")
    parser = argparse.ArgumentParser(description='Convert single-camera tracking results to the MOTChallenge format')
    parser.add_argument('--data-root', dest='data_root', default='train/S01',
                        help='dataset root path')

    args = parser.parse_args()

    main(args)
