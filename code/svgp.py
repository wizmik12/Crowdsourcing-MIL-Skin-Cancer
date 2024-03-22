import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import tensorflow as tf
os.environ['AUTOGRAPH_VERBOSITY'] = '0'
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import gpflow
from scipy.cluster.vq import kmeans2
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def create_setup(X_tr_nor, lengthscale, variance, lr, num_inducing, K, N):
    model = gpflow.models.SVGP(kernel=gpflow.kernels.SquaredExponential(lengthscales=lengthscale, variance=variance),
                            likelihood=gpflow.likelihoods.MultiClass(K),
                            inducing_variable=kmeans2(X_tr_nor,num_inducing,minit='points')[0],
                            num_latent_gps=K,
                            num_data=N)
    optimizer = tf.optimizers.Adam(learning_rate=lr)
    return model, optimizer


def evaluate_and_format(model,X,Y):
    f1,acc,auc,prec,rec = evaluate(model,X,Y)
    metrics_step = {"F1 Score": f1, "Accuracy": acc, "ROC_AUC": auc, "Precision": prec, "Recall": rec}
    return metrics_step


def run_adam(train_iter, model, optimizer, iterations, logs=False, X_tr_nor=None, Y_mv=None, Y_exp=None, X_ts_nor=None, mv_test=None, z_test=None):
    """
    Utility function running the Adam optimizer

    :param model: GPflow model
    :param interations: number of iterations
    """
    # Create an Adam Optimizer action
    training_loss = model.training_loss_closure(train_iter, compile=True)

    @tf.function
    def optimization_step():
        optimizer.minimize(training_loss, model.trainable_variables)

    train_metrics_mv = {"F1 Score": [], "Accuracy": [], "ROC_AUC": [], "Precision": [], "Recall": []}   
    train_metrics_exp = {"F1 Score": [], "Accuracy": [], "ROC_AUC": [], "Precision": [], "Recall": []}   
    test_metrics_mv = {"F1 Score": [], "Accuracy": [], "ROC_AUC": [], "Precision": [], "Recall": []} 
    test_metrics_exp = {"F1 Score": [], "Accuracy": [], "ROC_AUC": [], "Precision": [], "Recall": []} 
    for step in range(iterations):
        print('    Epoch: {:2d} / {}'.format(step+1, iterations), end='\r')
        optimization_step()
        if logs and step % 20 == 0:
            train_metrics_step_mv = evaluate_and_format(model,X_tr_nor,Y_mv)
            train_metrics_step_exp = evaluate_and_format(model,X_tr_nor,Y_exp)
            test_metrics_step_mv = evaluate_and_format(model,X_ts_nor,mv_test)
            test_metrics_step_exp = evaluate_and_format(model,X_ts_nor,z_test)
            for key in test_metrics_step_mv.keys():
                train_metrics_mv[key].append(train_metrics_step_mv[key])
                train_metrics_exp[key].append(train_metrics_step_exp[key])
                test_metrics_mv[key].append(test_metrics_step_mv[key])
                test_metrics_exp[key].append(test_metrics_step_exp[key])
    return train_metrics_mv, train_metrics_exp, test_metrics_mv, test_metrics_exp


####### Functions to predict and evaluate
@tf.function
def predict(model,X_ts_nor):
    y_pred,_ = model.predict_y(X_ts_nor)  # y_pred is N_ts,K (test probabilities, adds 1 by rows)
    return y_pred


def evaluate(model,X_ts_nor,z_test):
    y_pred = predict(model,X_ts_nor)
    y_pred = y_pred.numpy()
    y_bin = np.argmax(y_pred,axis=1)
    acc = np.mean(y_bin==z_test)
    f1 = f1_score(z_test,y_bin)
    prec = precision_score(z_test, y_bin)
    rec = recall_score(z_test, y_bin)
    auc = roc_auc_score(z_test, y_pred[:,1])
    return f1,acc,auc,prec,rec