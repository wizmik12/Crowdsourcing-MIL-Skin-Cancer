"""
Script to generate the file all_patches.csv containing each path filename as a row.
"""
import os, glob
import pandas as pd
import argparse

def get_relevant_regions(csv_dir):
    train = pd.read_csv(csv_dir + 'train_crowdsourcing_MV.csv', delimiter=';')
    train = train.drop(train[train.Patient=='HUSC_4021'].index)
    val = pd.read_csv(csv_dir + 'validation_crowdsourcing.csv', delimiter=';')
    regs = pd.read_csv(csv_dir + 'HUSC_HCUV_RI.csv')
    regs['Patient'] = regs.images.apply(lambda x: x[:9])
    total = pd.concat([train, val]).drop('Malignant', axis=1)
    regs = pd.merge(left=total, right=regs, left_on='Patient', right_on='Patient')
    return regs # columns: Patient (name) | images (file)

def join_path_regions(path_df, regs):
    path_df['images'] = path_df[0].apply(lambda x: x.split('/')[-1])
    res = pd.merge(regs, path_df)
    return res.drop(['Patient', 'images'], axis=1)

def generate_csv(csv_dir, png_dir):
    regs = get_relevant_regions(csv_dir)

    path_temp = os.path.join(png_dir, '*', '*.jpg')
    patch_path = glob.glob(path_temp) # /folder/*.jpg
    path_temp = os.path.join(png_dir, '*', '*.png')
    patch_path.extend(glob.glob(path_temp))
    path_df = pd.DataFrame(patch_path)

    df = join_path_regions(path_df, regs)
    df.to_csv('all_patches.csv', index=False)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_dir', type=str, default='/home/jose/crowdsourcing/data/',
            help='Path to folder containing HUSC_HCUV_RI.csv, train_crowdsourcing_MV.csv, validation_crowdsourcing.csv')
    parser.add_argument('--png_dir', type=str, default='/data/BasesDeDatos/AI4SkIN_data/',
            help='Path to folder containing the folders that contain the images.')
    args = parser.parse_args()
    generate_csv(args.csv_dir, args.png_dir)