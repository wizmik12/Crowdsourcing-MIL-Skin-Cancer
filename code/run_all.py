import argparse
import json
import pandas as pd
import numpy as np
import svgp as svgp 
import SVGPCR_tf2_utils as SVGPCR_tf2_utils
import pickle
from join_results import join
import tensorflow as tf
import os
import warnings
warnings.filterwarnings('ignore')


N_EPOCHS=1000
N_REPS=10
MB_SIZE=172
FLOAT_TYPE = tf.float64
K=2


def process_data(X_tr, X_ts):
    ''' Normalize '''
    X_mean = np.mean(X_tr,axis=0)
    X_std = np.std(X_tr,axis=0)
    X_std[X_std<=0] = 1
    X_tr_nor = (X_tr-X_mean)/X_std
    X_ts_nor = (X_ts-X_mean)/X_std
    N, x_dim = X_tr_nor.shape
    return X_tr_nor, X_ts_nor, N


def read_features(dir_path, feats_name):
    feats_path = dir_path + 'Features_' + feats_name + '/'
    with open(feats_path+'X_train.pickle', 'rb') as f:
        X_tr = pickle.load(f)
    with open(feats_path+'X_val.pickle', 'rb') as f:
        X_val = pickle.load(f)
    X_tr = np.concatenate([X_tr, X_val])
    with open(feats_path + 'X_test.pickle', 'rb') as f:
        X_ts = pickle.load(f)
    return process_data(X_tr, X_ts)


def read_labels(labels_path):
    with open(labels_path+'mv_train.pickle', 'rb') as f:
        mv_tr = pickle.load(f).astype('float').astype('int')
    with open(labels_path+'mv_val.pickle', 'rb') as f:
        mv_val = pickle.load(f).astype('float').astype('int')
    Y_mv = np.concatenate([mv_tr, mv_val])
    with open(labels_path+'mv_test.pickle', 'rb') as f:
        mv_test = pickle.load(f).astype('float').astype('int')

    with open(labels_path+'y_train.pickle', 'rb') as f:
        tmp = pickle.load(f)
    Y_cr = tmp["Y"]
    Y_mask = tmp["mask"]
    A = tmp["A"]
    with open(labels_path+'y_val.pickle', 'rb') as f:
        tmp = pickle.load(f)
    Y_val = tmp["Y"]
    Y_val_mask = tmp["mask"]
    Y_cr = np.concatenate([Y_cr, Y_val])
    Y_cr[Y_cr == -1] = 0
    Y_mask = np.concatenate([Y_mask, Y_val_mask])
    

    with open(labels_path+'z_train.pickle', 'rb') as f:
        z_tr = pickle.load(f).astype('float').astype('int')
    with open(labels_path+'z_val.pickle', 'rb') as f:
        z_val = pickle.load(f).astype('float').astype('int')
    Y_exp = np.concatenate([z_tr, z_val])

    with open(labels_path+'z_test.pickle', 'rb') as f:
        z_test = pickle.load(f).astype('float').astype('int')
    return Y_mv, Y_exp, Y_cr, Y_mask, A, z_test, mv_test
    

def run_experiment_normal(experiment, X_tr_nor, X_ts_nor, Y_tr, Y_mv, Y_exp, mv_test, z_test, N, logs=False):
    Y_tr = tf.convert_to_tensor(Y_tr,dtype=FLOAT_TYPE)
    data = (X_tr_nor, Y_tr) # caracteristicas y etiquetas
    train_dataset = tf.data.Dataset.from_tensor_slices(data).repeat().shuffle(N)
    train_iter = iter(train_dataset.batch(MB_SIZE))

    model, optimizer = svgp.create_setup(X_tr_nor, experiment['lengthscale'], experiment['variance'], experiment['learning_rate'], experiment['num_inducing_points'], K, N)
    train_metrics_mv, train_metrics_exp, test_metrics_mv, test_metrics_exp = svgp.run_adam(train_iter, model, optimizer, N_EPOCHS, logs, X_tr_nor, Y_mv, Y_exp, X_ts_nor, mv_test, z_test)

    z_test = tf.convert_to_tensor(z_test,dtype=FLOAT_TYPE)
    f1,acc,auc,prec,rec = svgp.evaluate(model,X_ts_nor,z_test)
    final_result = {"F1 Score": f1, "Accuracy": acc, "ROC_AUC": auc, "Precision": prec, "Recall": rec} 
    if logs:
        return final_result, train_metrics_mv, train_metrics_exp, test_metrics_mv, test_metrics_exp
    return final_result


def run_experiment_crowd(experiment, Y_cr, Y_mask, A, z_test, X_tr_nor, X_ts_nor, N):
    # Initialization
    alpha_tilde_raw_init, alpha, q_raw_init = SVGPCR_tf2_utils.alpha_init(Y_cr, Y_mask, A, K)
    q_raw, alpha_tilde_raw, model, trainable_variables, optimizer = \
        SVGPCR_tf2_utils.create_setup(q_raw_init, alpha_tilde_raw_init, 
                experiment['lengthscale'], experiment['variance'], experiment['learning_rate'], K, X_tr_nor, 
                experiment['num_inducing_points'], N)
    Y_cr = tf.convert_to_tensor(Y_cr)
    Y_mask = tf.convert_to_tensor(Y_mask)
    # Main loop
    optimization_step = SVGPCR_tf2_utils.build_optimizer(q_raw, alpha_tilde_raw, alpha, model, trainable_variables, optimizer, N, K)  
    for ep in range(N_EPOCHS):
        print('    Epoch: {:2d} / {}'.format(ep+1, N_EPOCHS), end='\r')
        idxs = np.random.permutation(N)
        idxs_iter = iter(tf.data.Dataset.from_tensor_slices(idxs).batch(MB_SIZE))
        for idxs_mb in idxs_iter:
            X_tr_nor_mb = tf.gather(X_tr_nor,idxs_mb)     # N_mb,D
            Y_cr_mb = tf.cast(tf.gather(Y_cr,idxs_mb),tf.int32)            # N_mb,S,2
            Y_mask_mb = tf.gather(Y_mask,idxs_mb)       # N_mb,S
            ELBO = optimization_step(X_tr_nor_mb,Y_cr_mb,Y_mask_mb,idxs_mb)

    f1,acc,auc,prec,rec = SVGPCR_tf2_utils.evaluate(model,X_ts_nor,z_test)
    return {"F1 Score": f1, "Accuracy": acc, "ROC_AUC": auc, "Precision": prec, "Recall": rec} 


def log2tensorboard(log_dir, tag, train_metrics_mv, train_metrics_exp, test_metrics_mv, test_metrics_exp):
    # Tensorboard logging directories
    train_mv_log_dir = log_dir + 'train_mv'
    train_exp_log_dir = log_dir + 'train_exp'
    test_mv_log_dir = log_dir + 'test_mv'
    test_exp_log_dir = log_dir + 'test_exp'
    train_mv_summary_writer = tf.summary.create_file_writer(train_mv_log_dir)
    train_exp_summary_writer = tf.summary.create_file_writer(train_exp_log_dir)
    test_mv_summary_writer = tf.summary.create_file_writer(test_mv_log_dir)
    test_exp_summary_writer = tf.summary.create_file_writer(test_exp_log_dir)
    # Saving to tensorboard
    for key in train_metrics_mv.keys():
        with train_mv_summary_writer.as_default():
            for step in range(0, N_EPOCHS, 20):
                tf.summary.scalar(key + '/' + tag, train_metrics_mv[key][step // 20], step=step)
        with train_exp_summary_writer.as_default():
            for step in range(0, N_EPOCHS, 20):
                tf.summary.scalar(key + '/' + tag, train_metrics_exp[key][step // 20], step=step)
        with test_mv_summary_writer.as_default():
            for step in range(0, N_EPOCHS, 20):
                tf.summary.scalar(key + '/' + tag, test_metrics_mv[key][step // 20], step=step)
        with test_exp_summary_writer.as_default():
            for step in range(0, N_EPOCHS, 20):
                tf.summary.scalar(key + '/' + tag, test_metrics_exp[key][step // 20], step=step)


def run_experiment(which, experiment, save_path, name, X_tr_nor, X_ts_nor, N, Y_mv, Y_exp, Y_cr, Y_mask, A, z_test, mv_test, log_dir):
    result_agg = {"F1 Score": [], "Accuracy": [], "ROC_AUC": [], "Precision": [], "Recall": []}
    train_metrics_mv = {"F1 Score": np.zeros((N_EPOCHS // 20,)), "Accuracy": np.zeros((N_EPOCHS // 20,)), "ROC_AUC": np.zeros((N_EPOCHS // 20,)), "Precision": np.zeros((N_EPOCHS // 20,)), "Recall": np.zeros((N_EPOCHS // 20,))}   
    train_metrics_exp = {"F1 Score": np.zeros((N_EPOCHS // 20,)), "Accuracy": np.zeros((N_EPOCHS // 20,)), "ROC_AUC": np.zeros((N_EPOCHS // 20,)), "Precision": np.zeros((N_EPOCHS // 20,)), "Recall": np.zeros((N_EPOCHS // 20,))}   
    test_metrics_mv = {"F1 Score": np.zeros((N_EPOCHS // 20,)), "Accuracy": np.zeros((N_EPOCHS // 20,)), "ROC_AUC": np.zeros((N_EPOCHS // 20,)), "Precision": np.zeros((N_EPOCHS // 20,)), "Recall": np.zeros((N_EPOCHS // 20,))} 
    test_metrics_exp = {"F1 Score": np.zeros((N_EPOCHS // 20,)), "Accuracy": np.zeros((N_EPOCHS // 20,)), "ROC_AUC": np.zeros((N_EPOCHS // 20,)), "Precision": np.zeros((N_EPOCHS // 20,)), "Recall": np.zeros((N_EPOCHS // 20,))} 
    for kk in range(N_REPS):
        print('  Run: {:2d} / {}'.format(kk+1, N_REPS), end='')
        if which == 'mv':
            print(' [MV]')
            result, train_metrics_step_mv, train_metrics_step_exp, test_metrics_step_mv, test_metrics_step_exp = run_experiment_normal(experiment, X_tr_nor, X_ts_nor, Y_mv, Y_mv, Y_exp, mv_test, z_test, N, logs=True)
        elif which == 'exp':
            print(' [experts]')
            result, train_metrics_step_mv, train_metrics_step_exp, test_metrics_step_mv, test_metrics_step_exp = run_experiment_normal(experiment, X_tr_nor, X_ts_nor, Y_exp, Y_mv, Y_exp, mv_test, z_test, N, logs=True)
        elif which == 'crowd':
            print(' [crowd]')
            result = run_experiment_crowd(experiment, Y_cr, Y_mask, A, z_test, X_tr_nor, X_ts_nor, N)
        else:
            assert False, 'which must be either mv, exp or crowd.'
        if which == 'mv' or which == 'exp':
            for key in train_metrics_mv.keys():
                train_metrics_mv[key] += np.array(train_metrics_step_mv[key])
                train_metrics_exp[key] += np.array(train_metrics_step_exp[key])
                test_metrics_mv[key] += np.array(test_metrics_step_mv[key])
                test_metrics_exp[key] += np.array(test_metrics_step_exp[key])
        print()
        for key in result_agg.keys():
            result_agg[key].append(result[key])
    if which == 'mv' or which == 'exp':
        # Taking the average series
        for key in train_metrics_mv.keys():
            train_metrics_mv[key] /= N_REPS
            train_metrics_exp[key] /= N_REPS
            test_metrics_mv[key] /= N_REPS
            test_metrics_exp[key] /= N_REPS
        # Name for saving logs
        tag = str(experiment['lengthscale']) + '-' + str(experiment['variance']) + '-' \
                + str(experiment['learning_rate']) + '-' + str(experiment['num_inducing_points']) 
        # Log to tensorboard
        log2tensorboard(log_dir + '/' + which, tag, train_metrics_mv, train_metrics_exp, test_metrics_mv, test_metrics_exp)
    result_df = pd.DataFrame(result_agg)
    mean = result_df.apply(np.mean)
    mean = pd.DataFrame(mean).transpose().set_index([pd.Series(['mean'])])
    std = result_df.apply(np.std)
    std = pd.DataFrame(std).transpose().set_index([pd.Series(['std'])])
    result_df = pd.concat([result_df, mean, std])
    save_path = save_path + which + '_' + name + '/'
    save_result(result_df, save_path, experiment)


def create_output_folders(save_path, name):
    for prefix in ['mv_', 'crowd_', 'exp_']:
        dir_results = save_path + prefix + name
        os.makedirs(dir_results, exist_ok=True)


def save_result(result_df, save_path, experiment):
    name = str(experiment['lengthscale']) + '-' + str(experiment['variance']) + '-' \
            + str(experiment['learning_rate']) + '-' + str(experiment['num_inducing_points']) + '.csv'
    result_df.to_csv(save_path + name)


def join_all(save_path, name):
    for prefix in ['mv_', 'crowd_', 'exp_']:
        dir_results = save_path + prefix + name + '/'
        join(dir_results)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config file (json).', default='./Features/config.json')
    parser.add_argument('--features', type=str, help='Path to features folders.', default='./Features/')
    parser.add_argument('--labels', type=str, help='Path to labels folders.', default='./Features/labels/')
    parser.add_argument('--save', type=str, help='Path to folder where results are saved.', default='./results/metrics/')
    parser.add_argument('--log-dir', type=str, help='Path to folder where tensorboard logs are saved.', default='./results/log/')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    for feature_name in config['features']:
        print('Features:', feature_name)
        create_output_folders(args.save, feature_name)

        # Tensorboard log directories
        log_dir = args.log_dir + feature_name

        X_tr_nor, X_ts_nor, N = read_features(args.features, feature_name)
        X_tr_nor = tf.convert_to_tensor(X_tr_nor,dtype=FLOAT_TYPE)
        X_ts_nor = tf.convert_to_tensor(X_ts_nor,dtype=FLOAT_TYPE)
        Y_mv, Y_exp, Y_cr, Y_mask, A, z_test, mv_test = read_labels(args.labels)
        for k, experiment in enumerate(config['experiments']):
            print('Experiment: {:2d} / {}'.format(k+1, len(config['experiments'])))
            run_experiment('mv', experiment, args.save, feature_name, X_tr_nor, X_ts_nor, N, Y_mv, Y_exp, Y_cr, Y_mask, A, z_test, mv_test, log_dir)
            run_experiment('crowd', experiment, args.save, feature_name, X_tr_nor, X_ts_nor, N, Y_mv, Y_exp, Y_cr, Y_mask, A, z_test, mv_test, log_dir)
            run_experiment('exp', experiment, args.save, feature_name, X_tr_nor, X_ts_nor, N, Y_mv, Y_exp, Y_cr, Y_mask, A, z_test, mv_test, log_dir)
        join_all(args.save, feature_name)
            
        
