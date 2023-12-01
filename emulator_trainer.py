import dill
import os
import numpy as np
import haiku as hk
import jax.numpy as jnp
import jax
from typing import Any, Sequence, Optional, Tuple, Iterator, Dict, Callable, Union
import optax
from tqdm import trange
from jax.config import config
from jax import jit
from jax.scipy.stats.multivariate_normal import logpdf
from functools import partial
from sklearn.metrics import r2_score
import sys
import os
sys.path.append(os.path.expanduser('~') + '/igm_emulator/igm_emulator/emulator')
from haiku_custom_forward import schedule_lr, loss_fn, accuracy, update, MyModuleCustom
from utils_plot import *
sys.path.append(os.path.expanduser('~') + '/igm_emulator/igm_emulator/scripts')
import h5py
import IPython
config.update("jax_enable_x64", True)
dtype=jnp.float64

'''
Training Loop Module + Visualization of params
'''
class TrainerModule:

    def __init__(self,
                X_train: Any,
                Y_train: Any,
                X_test: Any,
                Y_test: Any,
                X_vali: Any,
                Y_vali: Any,
                meanX: Any,
                stdX: Any,
                meanY: Any,
                stdY: Any,
                layer_sizes: Sequence[int],
                activation: Callable[[jnp.ndarray], jnp.ndarray],
                dropout_rate: float,
                optimizer_hparams: Sequence[Any],
                loss_str: str,
                loss_weights: Sequence[Any],
                like_dict: dict,
                out_tag: str,
                init_rng=42,
                n_epochs=1000,
                pv=100):

        super().__init__()
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.X_vali = X_vali
        self.Y_vali = Y_vali
        self.meanX = meanX
        self.stdX = stdX
        self.meanY = meanY
        self.stdY = stdY
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.optimizer_hparams = optimizer_hparams
        self.loss_str = loss_str
        self.l2_weight, self.c_loss, self.percent_loss = loss_weights
        self.like_dict = like_dict
        self.out_tag = out_tag
        self.var_tag =f'{loss_str}_l2_{self.l2_weight}_perc_{self.percent_loss}_activation_{activation.__name__}'
        self.init_rng = init_rng
        self.n_epochs = n_epochs
        self.pv = pv
        def _custom_forward_fn(x):
            module = MyModuleCustom(output_size=self.layer_sizes, activation=self.activation,
                                    dropout_rate=self.dropout_rate)
            return module(x)
        self.custom_forward = hk.without_apply_rng(hk.transform(_custom_forward_fn))

    @partial(jit, static_argnums=(0,))
    def loss_fn(self, params, X, Y):
        _loss_fn = jax.tree_util.Partial(loss_fn, like_dict=self.like_dict, custom_forward=self.custom_forward, l2=self.l2_weight, c_loss=self.c_loss, loss_str=self.loss_str, percent=self.percent_loss)
        return _loss_fn(params, X, Y)

    @partial(jit, static_argnums=(0,5))
    def update(self, params, opt_state, X_train, Y_train, optimizer):
        _update = jax.tree_util.Partial(update, like_dict=self.like_dict, custom_forward=self.custom_forward, l2=self.l2_weight, c_loss=self.c_loss, loss_str=self.loss_str, percent=self.percent_loss)
        return _update(params, opt_state, X_train, Y_train, optimizer)


    def train_loop(self, plot=True):
        '''
        Training loop for the neural network
        Parameters
        ----------
        plot

        Returns
        -------

        '''
        custom_forward = self.custom_forward
        params = custom_forward.init(rng=next(hk.PRNGSequence(jax.random.PRNGKey(self.init_rng))), x=self.X_train)

        n_samples = self.X_train.shape[0]
        total_steps = self.n_epochs*n_samples + self.n_epochs
        max_grad_norm, lr, decay = self.optimizer_hparams
        optimizer = optax.chain(optax.clip_by_global_norm(max_grad_norm),
                                optax.adamw(learning_rate=schedule_lr(lr,total_steps),weight_decay=decay)
                                )

        opt_state = optimizer.init(params)
        early_stopping_counter = 0
        best_loss = np.inf
        validation_loss = []
        training_loss = []
        print(f'***Training Loop Start***')
        print(f'MLP info: {self.var_tag}')
        with trange(self.n_epochs) as t:
            for step in t:
                # optimizing loss by update function
                params, opt_state, batch_loss, grads = self.update(params, opt_state, self.X_train, self.Y_train, optimizer)

                # compute training & validation loss at the end of the epoch
                l = self.loss_fn(params, self.X_vali, self.Y_vali)
                training_loss.append(batch_loss)
                validation_loss.append(l)

                # update the progressbar
                t.set_postfix(loss=validation_loss[-1])

                # early stopping condition
                if l <= best_loss:
                    best_loss = l
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                if early_stopping_counter >= self.pv:
                    break

        self.best_params = params
        vali_preds = custom_forward.apply(self.best_params, self.X_vali)
        self.best_chi_loss = jnp.mean(jnp.abs((vali_preds - self.Y_vali) * self.stdY ) / jnp.sqrt(jnp.diagonal(self.like_dict['covariance'])))
        self.best_chi_2_loss = -logpdf(x=custom_forward.apply(self.best_params, self.X_vali) * self.stdY, mean=self.Y_vali * self.stdY, cov=self.like_dict['covariance'])
        print(f'Reached max number of epochs in this batch. Validation loss ={best_loss}. Training loss ={batch_loss}')
        print(f'early_stopping_counter: {early_stopping_counter}')
        print(f'Test Loss: {self.loss_fn(params, self.X_test, self.Y_test)}')

        #Metrics
        self.batch_loss = batch_loss
        test_preds = custom_forward.apply(self.best_params, self.X_test)
        test_accuracy = (self.Y_test*self.stdY-test_preds*self.stdY)/(self.Y_test*self.stdY+self.meanY)
        self.RelativeError = test_accuracy
        print(f'Test accuracy: {jnp.sqrt(jnp.mean(jnp.square(test_accuracy)))}')

        self.test_loss = self.loss_fn(params, self.X_test, self.Y_test)
        self.test_R2 = r2_score(test_preds.squeeze(), self.Y_test)
        print('Test R^2 Score: {}\n'.format(self.test_R2))  # R^2 score: ranging 0~1, 1 is good model
        preds = custom_forward.apply(self.best_params, self.X_train)

        if plot:
            #Prediction overplots: Training And Test
            plt.plot(range(len(validation_loss)), validation_loss, label=f'vali loss:{best_loss:.4f}')  # plot validation loss
            plt.plot(range(len(training_loss)), training_loss, label=f'train loss:{batch_loss: .4f}')  # plot training loss
            plt.legend()
            print(f'***Result Plots saved {dir_exp}***')
            train_overplot(preds, self.X_train, self.Y_train, self.meanY, self.stdY, self.out_tag, self.var_tag)
            test_overplot(test_preds, self.Y_test, self.X_test,self.meanX,self.stdX,self.meanY,self.stdY, self.out_tag, self.var_tag)

            #Accuracy + Results Plots
            plot_residue(self.RelativeError,self.out_tag, self.var_tag)
            #bad_learned_plots(self.RelativeError,self.X_test,self.Y_test,test_preds,self.meanY,self.stdY, self.out_tag, self.var_tag)
            plot_error_distribution(self.RelativeError,self.out_tag,self.var_tag)

        return self.best_params, self.best_chi_loss

    def save_training_info(self, redshift):
            zs = np.array([5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0])
            z_idx = np.argmin(np.abs(zs - redshift))
            z_strings = ['z54', 'z55', 'z56', 'z57', 'z58', 'z59', 'z6']
            z_string = z_strings[z_idx]
            max_grad_norm, lr, decay = self.optimizer_hparams
            #Save best emulated parameter

            print(f'***Saving training info & best parameters***')

            f = h5py.File(os.path.expanduser('~') + f'/igm_emulator/igm_emulator/emulator/best_params/{self.out_tag}_{self.var_tag}_savefile.hdf5', 'a')
            group1 = f.create_group('haiku_nn')
            group1.attrs['redshift'] = redshift
            group1.attrs['adamw_decay'] = decay
            group1.attrs['epochs'] = self.n_epochs
            group1.create_dataset('layers', data = self.layer_sizes)
            group1.attrs['activation_function'] = self.activation.__name__
            group1.attrs['learning_rate'] = lr
            group1.attrs['L2_weight'] = self.l2_weight
            group1.attrs['loss_fn'] = self.loss_str

            group2 = f.create_group('data')
            group2.attrs['train_dir'] = dir_lhs + f'{z_string}_param{train_num}.p'
            group2.attrs['test_dir'] = dir_lhs + f'{z_string}_param{test_num}.p'
            group2.attrs['vali_dir'] = dir_lhs + f'{z_string}_param{vali_num}.p'
            group2.create_dataset('test_data', data = self.X_test)
            group2.create_dataset('train_data', data = self.X_train)
            group2.create_dataset('vali_data', data = self.X_vali)
            group2.create_dataset('meanX', data=self.meanX)
            group2.create_dataset('stdX', data=self.stdX)
            group2.create_dataset('meanY', data=self.meanY)
            group2.create_dataset('stdY', data=self.stdY)
            #IPython.embed()
            group3 = f.create_group('performance')
            group3.attrs['R2'] = self.test_R2
            group3.attrs['test_loss'] = self.test_loss
            group3.attrs['train_loss'] = self.batch_loss
            group3.attrs['vali_loss'] = self.best_chi_loss
            group3.attrs['residuals_results'] = f'{jnp.mean(self.RelativeError)*100}% +/- {jnp.std(self.RelativeError) * 100}%'
            group3.create_dataset('residuals', data=self.RelativeError)
            f.close()
            print("training directories and hyperparameters saved")

            dir = os.path.expanduser('~') + '/igm_emulator/igm_emulator/emulator/best_params'
            dir2 = '/mnt/quasar2/zhenyujin/igm_emulator/emulator/best_params'
            dill.dump(self.best_params, open(os.path.join(dir, f'{self.out_tag}_{self.var_tag}_best_param.p'), 'wb'))
            dill.dump(self.best_params, open(os.path.join(dir2, f'{self.out_tag}_{self.var_tag}_best_param.p'), 'wb'))
            print("trained parameters saved")

'''
#Initiate training module
'''
max_grad_norm = 0.1
lr = 1e-3
#beta = 1e-3 #BNN
decay = 5e-3
l2 =0.0001
if __name__ == '__main__':
    trainer = TrainerModule(X_train,Y_train,X_test,Y_test,X_vali,Y_vali,meanX,stdX,meanY,stdY,
                            layer_sizes=[100,100,100,59],
                            activation= jax.nn.leaky_relu,
                            dropout_rate=None,
                            optimizer_hparams=[max_grad_norm, lr, decay],
                            loss_str='mse',
                            loss_weights=[l2,0,False],
                            like_dict=like_dict,
                            init_rng=42,
                            n_epochs=1000,
                            pv=100,
                            out_tag=out_tag)
    trainer.train_loop()
    trainer.save_training_info(5.4)

    trainer_optuna = TrainerModule(X_train,Y_train,X_test,Y_test,X_vali,Y_vali,meanX,stdX,meanY,stdY,
                            layer_sizes=[100,100,59],
                            activation= jax.nn.tanh,
                            dropout_rate=None,
                            optimizer_hparams=[0.30000000000000004, 0.0005946616649768666, 0.00013552715097890048],
                            loss_str='huber',
                            loss_weights=[1e-05,0.0033025697025815485,True],
                            like_dict=like_dict,
                            init_rng=42,
                            n_epochs=1000,
                            pv=100,
                            out_tag=out_tag)
    trainer_optuna.train_loop()
    trainer_optuna.save_training_info(5.4)   
    IPython.embed()

