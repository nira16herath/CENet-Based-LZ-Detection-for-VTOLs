import argparse
import yaml
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import confusion_matrix

def plot_projection(scan, labels, save_path, filename):
    img_size = (300, 300)
    projection = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
    
    x, y = scan[:, 0], scan[:, 1]
    x = ((x - x.min()) / (x.max() - x.min()) * (img_size[0] - 1)).astype(int)
    y = ((y - y.min()) / (y.max() - y.min()) * (img_size[1] - 1)).astype(int)
    
    for i in range(len(x)):
        projection[y[i], x[i]] = label_colors.get(labels[i], [0, 0, 0])
    
    plt.figure(figsize=(5, 5))
    plt.imshow(projection)
    plt.axis('off')
    plt.savefig(os.path.join(save_path, filename))
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', '-p', type=str, required=True, help='Directory of the predictions.')
    parser.add_argument('--gt_path', '-g', type=str, required=True, help='Directory of the ground truth data')
    parser.add_argument('--model', '-m', type=str, required=True, help='Directory to get the trained model.')
    FLAGS = parser.parse_args()

    try:
        DATA = yaml.safe_load(open(os.path.join(FLAGS.model, "data_cfg.yaml"), 'r'))
    except Exception as e:
        print(e)
        quit()

    results = []

    label_colors = DATA["color_map"]
    label_names = DATA["labels"]
    
    
    for seq in DATA["split"]["test"]:
        seq = '{0:02d}'.format(int(seq))
        print(f"Processing sequence {seq}")

        pred_path = os.path.join(FLAGS.pred_path, "sequences", seq, 'predictions')
        gt_path = os.path.join(FLAGS.gt_path, "sequences", seq, 'labels')

        images_save_path = os.path.join(FLAGS.pred_path, "sequences", seq, 'images')

        os.makedirs(images_save_path, exist_ok=True)
   
        for file in sorted(os.listdir(pred_path)):
            if file.endswith('.label'):
                pred_file = os.path.join(pred_path, file)
                gt_file = os.path.join(gt_path, file)
                scan_file = os.path.join(FLAGS.gt_path, "sequences", seq, "velodyne", file.replace(".label", ".bin"))
                
                if not os.path.exists(gt_file) or not os.path.exists(scan_file):
                    continue

                pred = np.fromfile(pred_file, dtype=np.int32).reshape(-1)
                gt = np.fromfile(gt_file, dtype=np.int32).reshape(-1)
                scan = np.fromfile(scan_file, dtype=np.float32).reshape(-1, 4)
                
                plot_projection(scan, pred, images_save_path, file.replace(".label", "_pred.png"))
                plot_projection(scan, gt, images_save_path, file.replace(".label", "_gt.png"))
                print("saved:",images_save_path+'/'+file.replace(".label", "_pred.png"))
                
    print("Completed! results saved to:",images_save_path)
