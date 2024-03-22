import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import tensorflow as tf
os.environ['AUTOGRAPH_VERBOSITY'] = '0'
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import gpflow
from scipy.cluster.vq import kmeans2
from math import ceil
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


FLOAT_TYPE = tf.float64


def _init_q_raw(Y_cr,Y_mask,K):
    ##### Initializations of q_raw and alpha_tilde_raw and alpha (they are _raw because these values must be constrained)
    counts_init = np.array([np.bincount(y[m==1,1], minlength=K) for y,m in zip(Y_cr,Y_mask)])
    counts_init = counts_init + np.ones(counts_init.shape)
    q_raw_init = np.log(counts_init/np.sum(counts_init,axis=1,keepdims=True))
    #q_raw_init = np.log(np.exp(counts_init/np.sum(counts_init,axis=1,keepdims=True))-1.0)
    return q_raw_init


def _init_behaviors(probs, Y_cr, Y_mask, A, K):
    alpha_tilde = np.ones((A,K,K))/K
    counts = np.ones((A,K))
    for n in range(len(Y_cr)):
        for a,c,m in zip(Y_cr[n][:,0], Y_cr[n][:,1], Y_mask[n]):
            if m==1:
                alpha_tilde[a,c,:] += probs[n,:]
                counts[a,c] += 1
    alpha_tilde=alpha_tilde/counts[:,:,None]
    alpha_tilde = (counts/np.sum(counts,axis=1,keepdims=True))[:,:,None]*alpha_tilde
    return alpha_tilde/np.sum(alpha_tilde,axis=1,keepdims=True)


def alpha_init(Y_cr, Y_mask, A, K):
    q_raw_init = _init_q_raw(Y_cr,Y_mask,K)
    # alpha_tilde_raw_init = np.load(path_data+"alpha_tilde_raw_init.npy")  # The initialization of alpha_tilde is as in the TPAMI code (I just saved it in a .npy file to import it easily)
    alpha_tilde_raw_init = _init_behaviors(q_raw_init, Y_cr, Y_mask, A, K)
    #alpha_tilde_raw_init_2 = np.log(np.exp(np.exp(alpha_tilde_raw_init))-1.0)
    alpha = tf.ones((A,K,K),dtype=FLOAT_TYPE)
    return alpha_tilde_raw_init, alpha, q_raw_init


def create_setup(q_raw_init, alpha_tilde_raw_init, lengthscale, variance, lr, K, X_tr_nor, num_inducing, N):
    ###### Variables that will be optimized
    q_raw = tf.Variable(q_raw_init,dtype=FLOAT_TYPE)  # N,K
    alpha_tilde_raw = tf.Variable(alpha_tilde_raw_init,dtype=FLOAT_TYPE) # A,K,K
    #alpha_tilde_raw = tf.Variable(alpha_tilde_raw_init_2,dtype=FLOAT_TYPE) # A,K,K
    model = gpflow.models.SVGP(kernel=gpflow.kernels.SquaredExponential(lengthscales=lengthscale, variance=variance),
                            likelihood=gpflow.likelihoods.MultiClass(K),
                            inducing_variable=kmeans2(X_tr_nor,num_inducing,minit='points')[0],
                            num_latent_gps=K,
                            num_data=N)
    trainable_variables = (alpha_tilde_raw,q_raw)+model.trainable_variables # (the latter contains Z,q_mu,q_sqrt and both kernel params)
    optimizer = tf.optimizers.Adam(learning_rate=lr)
    return q_raw, alpha_tilde_raw, model, trainable_variables, optimizer


###### This is the optimization step. Basic idea:
######  * We compute the ELBO (the five terms in eq.15 TPAMI), obtain the gradient, and give one step
######  * Gradients are recorded for the computations inside the tf.GradientTape block
######  * The @tf.function decorator makes things faster, since the function is implemented internally as a graph
def build_optimizer(q_raw, alpha_tilde_raw, alpha, model, trainable_variables, optimizer, N, K):
    @tf.function
    def optimization_step(X_tr_nor_mb,Y_cr_mb,Y_mask_mb,idxs_mb):
        scale = N/tf.cast(X_tr_nor_mb.shape[0],FLOAT_TYPE) # scale to take into account the minibatch size
        with tf.GradientTape() as tape:   # tf.GradientTape() records the gradients. All the computations for which gradients are required must go in such a block
            q_mb = tf.nn.softmax(tf.gather(q_raw,idxs_mb)) # N_mb,K (constraint: positive and adds to 1 by rows). Gather selects the minibatch
            # q_mb = tf.math.softplus(tf.gather(q_raw,idxs_mb)) # N_mb,K (adds to 1 and positive by rows)
            # q_mb = q_mb/tf.reduce_sum(q_mb,axis=1,keepdims=True)
            alpha_tilde = tf.exp(alpha_tilde_raw)  # A,K,K  (cosntraint: positive)
            #alpha_tilde = tf.math.softplus(alpha_tilde_raw)  # A,K,K
            # Annotations term (term 1 in eq.15 TPAMI)
            expect_log = tf.math.digamma(alpha_tilde)-tf.math.digamma(tf.reduce_sum(alpha_tilde,1,keepdims=True)) # A,K,K
            tnsr_expCrow = tf.gather_nd(expect_log,Y_cr_mb)*tf.cast(Y_mask_mb[:,:,None],FLOAT_TYPE) # (N_mb,S,K) = (N_mb,S,K)*(N_mb,S,1)
            annot_term = tf.reduce_sum(tnsr_expCrow*q_mb[:,None,:])*scale # scalar
            # SVGP likelihood term (terms 2 in eq.15 TPAMI)
            f_mean, f_var = model.predict_f(X_tr_nor_mb, full_cov=False, full_output_cov=False) # N,K ; N,K
            liks = [model.likelihood.variational_expectations(f_mean,f_var,
                                                                c*tf.ones((f_mean.shape[0],1),dtype=tf.int32))
                    for c in np.arange(K)] # [(N_mb),....,(N_mb)]
            lik_term = scale*tf.reduce_sum(q_mb*tf.stack(liks,axis=1))  # 1 <- reduce_sum[(N_mb,K)*(N_mb,K)]
            # Entropy term (term 3 in eq.15 TPAMI)
            entropy_term = -tf.reduce_sum(q_mb*tf.math.log(q_mb))*scale  #scalar
            # KL SVGP term (term 4 in eq.15 TPAMI)
            KL_svgp_term = model.prior_kl()
            # KL annot term (term 5 in eq.15 TPAMI)
            alpha_diff = alpha_tilde-alpha
            KL_annot_term=(tf.reduce_sum(alpha_diff*tf.math.digamma(alpha_tilde))-
                    tf.reduce_sum(tf.math.digamma(tf.reduce_sum(alpha_tilde,1))*tf.reduce_sum(alpha_diff,1))+
                    tf.reduce_sum(tf.math.lbeta(tf.linalg.matrix_transpose(alpha))-tf.math.lbeta(tf.linalg.matrix_transpose(alpha_tilde))))
            negELBO = -(annot_term + lik_term + entropy_term - KL_svgp_term - KL_annot_term)
        grads = tape.gradient(negELBO,trainable_variables)          # The gradients are obtained
        optimizer.apply_gradients(zip(grads,trainable_variables))   # The gradients are applied with the optimizer
        return -negELBO            # The ELBO is returned just to have access to it and print it in the main program (but the minimization takes place in the line above; here we could just return nothing)
    return optimization_step


####### Functions to predict and evaluate
#@tf.function
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