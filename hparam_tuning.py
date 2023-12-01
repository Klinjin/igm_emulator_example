import jax
import optuna
from optuna.samplers import TPESampler
import sys
import os
sys.path.append(os.path.expanduser('~') + '/igm_emulator/igm_emulator/emulator')
import haiku as hk
from emulator_trainer import TrainerModule
import h5py
import numpy as np
from jax.config import config
import jax.numpy as jnp
import jax
import dill
config.update("jax_enable_x64", True)

'''
Set redshift and data bin size
'''
redshift = 5.4  # choose redshift from [5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0]
small_bin_bool = True  # True: small bins n=59; False: large bins n=276

'''
Load datasets
'''
# get the appropriate string and pathlength for chosen redshift
zs = np.array([5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0])
z_idx = np.argmin(np.abs(zs - redshift))
z_strings = ['z54', 'z55', 'z56', 'z57', 'z58', 'z59', 'z6']
z_string = z_strings[z_idx]
dir_lhs = os.path.expanduser('~') + '/igm_emulator/igm_emulator/emulator/GRID/'

if small_bin_bool == True:
    train_num = '_training_768_bin59'
    test_num = '_test_89_bin59'
    vali_num = '_vali_358_bin59'
    n_path = 20  # 17->20
    n_covar = 500000
    bin_label = '_set_bins_3'
    in_path = f'/mnt/quasar2/mawolfson/correlation_funct/temp_gamma/final_135/{z_string}/'
    out_tag = f'{z_string}{train_num}'
else:
    train_num = '_training_768'
    test_num = '_test_89'
    vali_num = '_vali_358'
    n_path = 17
    n_covar = 500000
    bin_label = '_set_bins_4'
    in_path = f'/mnt/quasar2/mawolfson/correlation_funct/temp_gamma/final/{z_string}/final_135/'
    out_tag = f'{z_string}{train_num}_bin276'

#get the fixed covariance dictionary for likelihood
T0_idx = 8  # 0-14
g_idx = 4  # 0-8
f_idx = 4  # 0-8
like_name = f'likelihood_dicts_R_30000_nf_9_T{T0_idx}_G{g_idx}_SNR0_F{f_idx}_ncovar_{n_covar}_P{n_path}{bin_label}.p'
like_dict = dill.load(open(in_path + like_name, 'rb'))

# load the training, test and validation data
X = dill.load(open(dir_lhs + f'{z_string}_param{train_num}.p',
                   'rb'))  # load normalized cosmological parameters from grab_models.py
X_test = dill.load(open(dir_lhs + f'{z_string}_param{test_num}.p', 'rb'))
X_vali = dill.load(open(dir_lhs + f'{z_string}_param{vali_num}.p', 'rb'))
meanX = X.mean(axis=0)
stdX = X.std(axis=0)
X_train = (X - meanX) / stdX
X_test = (X_test - meanX) / stdX
X_vali = (X_vali - meanX) / stdX

Y = dill.load(open(dir_lhs + f'{z_string}_model{train_num}.p', 'rb'))
Y_test = dill.load(open(dir_lhs + f'{z_string}_model{test_num}.p', 'rb'))
Y_vali = dill.load(open(dir_lhs + f'{z_string}_model{vali_num}.p', 'rb'))
meanY = Y.mean(axis=0)
stdY = Y.std(axis=0)
Y_train = (Y - meanY) / stdY
Y_test = (Y_test - meanY) / stdY
Y_vali = (Y_vali - meanY) / stdY

if __name__ == '__main__':
    def objective(trial):
        layer_sizes_tune = trial.suggest_categorical('layer_sizes', [(100, 100, 100, 59), (100, 100, 59), (100, 59)])
        activation_tune = trial.suggest_categorical('activation', ['jax.nn.leaky_relu', 'jax.nn.relu', 'jax.nn.sigmoid', 'jax.nn.tanh'])
        dropout_rate_tune = trial.suggest_categorical('dropout_rate', [None, 0.05, 0.1])
        max_grad_norm_tune = trial.suggest_float('max_grad_norm', 0, 0.5, step=0.1)
        lr_tune = trial.suggest_float('lr', 1e-5,1e-3, log=True)
        decay_tune = trial.suggest_float('decay', 1e-4, 5e-3, log=True)
        l2_tune = trial.suggest_categorical('l2', [0, 1e-5, 1e-4, 1e-3])
        c_loss_tune = trial.suggest_float('c_loss', 1e-3, 1, log=True)
        percent_loss_tune = trial.suggest_categorical('percent', [True, False])
        n_epochs_tune = trial.suggest_categorical('n_epochs', [500, 1000, 2000])
        loss_str_tune = trial.suggest_categorical('loss_str', ['chi_one_covariance', 'mse', 'mse+fft', 'huber', 'mae'])
        trainer = TrainerModule(X_train, Y_train, X_test, Y_test, X_vali, Y_vali, meanX, stdX, meanY, stdY,
                                layer_sizes= layer_sizes_tune,
                                activation=eval(activation_tune),
                                dropout_rate=dropout_rate_tune,
                                optimizer_hparams=[max_grad_norm_tune, lr_tune, decay_tune],
                                loss_str=loss_str_tune,
                                loss_weights=[l2_tune,c_loss_tune,percent_loss_tune],
                                like_dict=like_dict,
                                init_rng=42,
                                n_epochs=n_epochs_tune,
                                pv=100,
                                out_tag=out_tag)

        best_vali_loss = trainer.train_loop(False)[1]
        del trainer
        return best_vali_loss

    def save_best_param_objective(trial):
        layer_sizes_tune = trial.suggest_categorical('layer_sizes', [(100, 100, 100, 59), (100, 100, 59), (100, 59)])
        activation_tune = trial.suggest_categorical('activation', ['jax.nn.leaky_relu', 'jax.nn.relu', 'jax.nn.sigmoid', 'jax.nn.tanh'])
        dropout_rate_tune = trial.suggest_categorical('dropout_rate', [None, 0.05, 0.1])
        max_grad_norm_tune = trial.suggest_float('max_grad_norm', 0, 0.5, step=0.1)
        lr_tune = trial.suggest_float('lr', 1e-5,1e-3, log=True)
        decay_tune = trial.suggest_float('decay', 1e-4, 5e-3, log=True)
        l2_tune = trial.suggest_categorical('l2', [0, 1e-5, 1e-4, 1e-3])
        c_loss_tune = trial.suggest_float('c_loss', 1e-3, 1, log=True)
        percent_loss_tune = trial.suggest_categorical('percent', [True, False])
        n_epochs_tune = trial.suggest_categorical('n_epochs', [500, 1000, 2000])
        loss_str_tune = trial.suggest_categorical('loss_str', ['chi_one_covariance', 'mse', 'mse+fft', 'huber', 'mae'])
        trainer = TrainerModule(X_train, Y_train, X_test, Y_test, X_vali, Y_vali, meanX, stdX, meanY, stdY,
                                layer_sizes= layer_sizes_tune,
                                activation=eval(activation_tune),
                                dropout_rate=dropout_rate_tune,
                                optimizer_hparams=[max_grad_norm_tune, lr_tune, decay_tune],
                                loss_str=loss_str_tune,
                                loss_weights=[l2_tune,c_loss_tune,percent_loss_tune],
                                like_dict=like_dict,
                                init_rng=42,
                                n_epochs=n_epochs_tune,
                                pv=100,
                                out_tag=out_tag)

        best_vali_loss = trainer.train_loop(False)[1]
        trainer.save_training_info(redshift)
        del trainer

    print('*** Running the hyperparameter tuning ***')

    # create the study
    number_of_trials = 50
    sampler = TPESampler(seed=10)  # 10
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=number_of_trials, gc_after_trial=True)

    trial = study.best_trial
    print(f'\nBest Validation Loss: {trial.value}')
    print(f'Best Params:')
    
    for key, value in trial.params.items():
        print(f'-> {key}: {value}')
    dill.dump(trial.params, open(f'/mnt/quasar2/zhenyujin/igm_emulator/emulator/best_params/{out_tag}_hparams_tuned.p', 'wb'))
    print('Best hyperparameters saved to /mnt/quasar2/zhenyujin/igm_emulator/emulator/best_params')
    save_best_param_objective(trial)
    print('Best params for optuna tuned emulator saved to /mnt/quasar2/zhenyujin/igm_emulator/emulator/best_params')