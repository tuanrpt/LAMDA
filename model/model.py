#  Copyright (c) 2021, Tuan Nguyen.
#  All rights reserved.

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.framework import add_arg_scope
from tensorbayes.layers import dense, conv2d, batch_norm, instance_norm
from tensorflow.python.ops.nn_impl import sigmoid_cross_entropy_with_logits as sigmoid_x_entropy
from tensorbayes.tfutils import softmax_cross_entropy_with_two_logits as softmax_x_entropy_two

from generic_utils import random_seed

from layers import leaky_relu
import os
from generic_utils import model_dir
import numpy as np
import tensorbayes as tb
from layers import batch_ema_acc


def build_block(input_layer, layout, info=1):
    x = input_layer
    for i in range(0, len(layout)):
        with tf.variable_scope('l{:d}'.format(i)):
            f, f_args, f_kwargs = layout[i]
            x = f(x, *f_args, **f_kwargs)
            if info > 1:
                print(x)
    return x


@add_arg_scope
def normalize_perturbation(d, scope=None):
    with tf.name_scope(scope, 'norm_pert'):
        output = tf.nn.l2_normalize(d, axis=np.arange(1, len(d.shape)))
    return output


def build_encode_template(
        input_layer, training_phase, scope, encode_layout,
        reuse=None, internal_update=False, getter=None, inorm=True, cnn_size='large'):
    with tf.variable_scope(scope, reuse=reuse, custom_getter=getter):
        with arg_scope([leaky_relu], a=0.1), \
             arg_scope([conv2d, dense], activation=leaky_relu, bn=True, phase=training_phase), \
             arg_scope([batch_norm], internal_update=internal_update):

            preprocess = instance_norm if inorm else tf.identity

            layout = encode_layout(preprocess=preprocess, training_phase=training_phase, cnn_size=cnn_size)
            output_layer = build_block(input_layer, layout)

    return output_layer


def build_decode_template(
        input_layer, training_phase, scope, decode_layout,
        reuse=None, internal_update=False, getter=None, inorm=False, cnn_size='large'):
    with tf.variable_scope(scope, reuse=reuse, custom_getter=getter):
        with arg_scope([leaky_relu], a=0.1), \
             arg_scope([conv2d, dense], activation=leaky_relu, bn=True, phase=training_phase), \
             arg_scope([batch_norm], internal_update=internal_update):
            layout = decode_layout(training_phase=training_phase)
            output_layer = build_block(input_layer, layout)

    return output_layer


def build_class_discriminator_template(
        input_layer, training_phase, scope, num_classes, class_discriminator_layout,
        reuse=None, internal_update=False, getter=None, cnn_size='large'):
    with tf.variable_scope(scope, reuse=reuse, custom_getter=getter):
        with arg_scope([leaky_relu], a=0.1), \
             arg_scope([conv2d, dense], activation=leaky_relu, bn=True, phase=training_phase), \
             arg_scope([batch_norm], internal_update=internal_update):
            layout = class_discriminator_layout(num_classes=num_classes, global_pool=True, activation=None,
                                                cnn_size=cnn_size)
            output_layer = build_block(input_layer, layout)

    return output_layer


def build_domain_discriminator_template(x, domain_layout, c=1, reuse=None):
    with tf.variable_scope('domain_disc', reuse=reuse):
        with arg_scope([dense], activation=tf.nn.relu):
            layout = domain_layout(c=c)
            output_layer = build_block(x, layout)

    return output_layer


def get_default_config():
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.log_device_placement = False
    tf_config.allow_soft_placement = True
    return tf_config


class LAMDA():
    def __init__(self,
                 model_name="LAMDA-results",
                 learning_rate=0.001,
                 batch_size=128,
                 num_iters=80000,
                 summary_freq=400,
                 src_class_trade_off=1.0,
                 src_vat_trade_off=1.0,
                 trg_trade_off=1.0,
                 domain_trade_off=1.0,
                 adapt_domain_trade_off=False,
                 encode_layout=None,
                 decode_layout=None,
                 classify_layout=None,
                 domain_layout=None,
                 freq_calc_metrics=10,
                 init_calc_metrics=2,
                 current_time='',
                 inorm=True,
                 m_on_D_trade_off=1.0,
                 m_plus_1_on_D_trade_off=1.0,
                 m_plus_1_on_G_trade_off=1.0,
                 m_on_G_trade_off=0.1,
                 lamda_model_id='',
                 save_grads=False,
                 only_save_final_model=True,
                 cnn_size='large',
                 update_target_loss=True,
                 sample_size=50,
                 src_recons_trade_off=0.1,
                 **kwargs):
        self.model_name = model_name
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_iters = num_iters
        self.summary_freq = summary_freq
        self.src_class_trade_off = src_class_trade_off
        self.src_vat_trade_off = src_vat_trade_off
        self.trg_trade_off = trg_trade_off
        self.domain_trade_off = domain_trade_off
        self.adapt_domain_trade_off = adapt_domain_trade_off

        self.encode_layout = encode_layout
        self.decode_layout = decode_layout
        self.classify_layout = classify_layout
        self.domain_layout = domain_layout

        self.freq_calc_metrics = freq_calc_metrics
        self.init_calc_metrics = init_calc_metrics

        self.current_time = current_time
        self.inorm = inorm

        self.m_on_D_trade_off = m_on_D_trade_off
        self.m_plus_1_on_D_trade_off = m_plus_1_on_D_trade_off
        self.m_plus_1_on_G_trade_off = m_plus_1_on_G_trade_off
        self.m_on_G_trade_off = m_on_G_trade_off

        self.lamda_model_id = lamda_model_id

        self.save_grads = save_grads
        self.only_save_final_model = only_save_final_model

        self.cnn_size = cnn_size
        self.update_target_loss = update_target_loss

        self.sample_size = sample_size
        self.src_recons_trade_off = src_recons_trade_off


    def _init(self, data_loader):
        np.random.seed(random_seed())
        tf.set_random_seed(random_seed())
        tf.reset_default_graph()

        self.tf_graph = tf.get_default_graph()
        self.tf_config = get_default_config()
        self.tf_session = tf.Session(config=self.tf_config, graph=self.tf_graph)

        self.data_loader = data_loader
        self.num_classes = self.data_loader.num_class
        self.batch_size_src = self.sample_size*self.num_classes

    def _get_variables(self, list_scopes):
        variables = []
        for scope_name in list_scopes:
            variables.append(tf.get_collection('trainable_variables', scope_name))
        return variables

    def convert_one_hot(self, y):
        y_idx = y.reshape(-1).astype(int) if y is not None else None
        y = np.eye(self.num_classes)[y_idx] if y is not None else None
        return y

    def _get_scope(self, part_name, side_name, same_network=True):
        suffix = ''
        if not same_network:
            suffix = '/' + side_name
        return part_name + suffix

    def _get_primary_scopes(self):
        return ['generator', 'classifier', 'decode']

    def _get_secondary_scopes(self):
        return ['domain_disc']

    def _build_source_middle(self, x_src):
        scope_name = self._get_scope('generator', 'src')
        return build_encode_template(x_src, encode_layout=self.encode_layout,
                                     scope=scope_name, training_phase=self.is_training, inorm=self.inorm, cnn_size=self.cnn_size)

    def _build_middle_source(self, x_src_mid):
        scope_name = self._get_scope('decode', 'src')
        return build_decode_template(
            x_src_mid, decode_layout=self.decode_layout, scope=scope_name, training_phase=self.is_training, inorm=self.inorm, cnn_size=self.cnn_size
        )

    def _build_target_middle(self, x_trg):
        scope_name = self._get_scope('generator', 'trg')
        return build_encode_template(
            x_trg, encode_layout=self.encode_layout,
            scope=scope_name, training_phase=self.is_training, inorm=self.inorm,
            reuse=True, internal_update=True, cnn_size=self.cnn_size
        )  # reuse the 'encode_layout'

    def _build_classifier(self, x, num_classes, ema=None, is_teacher=False):
        g_teacher_scope = self._get_scope('generator', 'teacher', same_network=False)
        g_x = build_encode_template(
            x, encode_layout=self.encode_layout,
            scope=g_teacher_scope if is_teacher else 'generator', training_phase=False, inorm=self.inorm,
            reuse=False if is_teacher else True, getter=None if is_teacher else tb.tfutils.get_getter(ema),
            cnn_size=self.cnn_size
        )

        h_teacher_scope = self._get_scope('classifier', 'teacher', same_network=False)
        h_g_x = build_class_discriminator_template(
            g_x, training_phase=False, scope=h_teacher_scope if is_teacher else 'classifier', num_classes=num_classes,
            reuse=False if is_teacher else True, class_discriminator_layout=self.classify_layout,
            getter=None if is_teacher else tb.tfutils.get_getter(ema), cnn_size=self.cnn_size
        )
        return h_g_x

    def _build_domain_discriminator(self, x_mid, reuse=False):
        return build_domain_discriminator_template(x_mid, domain_layout=self.domain_layout, c=self.num_classes+1, reuse=reuse)

    def _build_class_src_discriminator(self, x_src, num_src_classes):
        return build_class_discriminator_template(
            self.x_src_mid, training_phase=self.is_training, scope='classifier', num_classes=num_src_classes,
            class_discriminator_layout=self.classify_layout, cnn_size=self.cnn_size
        )

    def _build_class_trg_discriminator(self, x_trg, num_trg_classes):
        return build_class_discriminator_template(
            self.x_trg_mid, training_phase=self.is_training, scope='classifier', num_classes=num_trg_classes,
            reuse=True, internal_update=True, class_discriminator_layout=self.classify_layout, cnn_size=self.cnn_size
        )

    def perturb_image(self, x, p, num_classes, class_discriminator_layout, encode_layout,
                      pert='vat', scope=None, radius=3.5, scope_classify=None, scope_encode=None, training_phase=None):
        with tf.name_scope(scope, 'perturb_image'):
            eps = 1e-6 * normalize_perturbation(tf.random_normal(shape=tf.shape(x)))

            # Predict on randomly perturbed image
            x_eps_mid = build_encode_template(
                x + eps, encode_layout=encode_layout, scope=scope_encode, training_phase=training_phase, reuse=True,
                inorm=self.inorm, cnn_size=self.cnn_size)
            x_eps_pred = build_class_discriminator_template(
                x_eps_mid, class_discriminator_layout=class_discriminator_layout,
                training_phase=training_phase, scope=scope_classify, reuse=True, num_classes=num_classes,
                cnn_size=self.cnn_size
            )
            # eps_p = classifier(x + eps, phase=True, reuse=True)
            loss = softmax_x_entropy_two(labels=p, logits=x_eps_pred)

            # Based on perturbed image, get direction of greatest error
            eps_adv = tf.gradients(loss, [eps], aggregation_method=2)[0]

            # Use that direction as adversarial perturbation
            eps_adv = normalize_perturbation(eps_adv)
            x_adv = tf.stop_gradient(x + radius * eps_adv)

        return x_adv

    def vat_loss(self, x, p, num_classes, class_discriminator_layout, encode_layout,
                 scope=None, scope_classify=None, scope_encode=None, training_phase=None):

        with tf.name_scope(scope, 'smoothing_loss'):
            x_adv = self.perturb_image(
                x, p, num_classes, class_discriminator_layout=class_discriminator_layout, encode_layout=encode_layout,
                scope_classify=scope_classify, scope_encode=scope_encode, training_phase=training_phase)

            x_adv_mid = build_encode_template(
                x_adv, encode_layout=encode_layout, scope=scope_encode, training_phase=training_phase, inorm=self.inorm,
                reuse=True, cnn_size=self.cnn_size)
            x_adv_pred = build_class_discriminator_template(
                x_adv_mid, training_phase=training_phase, scope=scope_classify, reuse=True, num_classes=num_classes,
                class_discriminator_layout=class_discriminator_layout, cnn_size=self.cnn_size
            )
            # p_adv = classifier(x_adv, phase=True, reuse=True)
            loss = tf.reduce_mean(softmax_x_entropy_two(labels=tf.stop_gradient(p), logits=x_adv_pred))

        return loss

    def _build_vat_loss(self, x, p, num_classes, scope=None, scope_classify=None, scope_encode=None):
        return self.vat_loss(  # compute the divergence between C(x) and C(G(x+r))
            x, p, num_classes,
            class_discriminator_layout=self.classify_layout,
            encode_layout=self.encode_layout,
            scope=scope, scope_classify=scope_classify, scope_encode=scope_encode,
            training_phase=self.is_training
        )

    def _build_model(self):
        self.x_src = tf.placeholder(dtype=tf.float32, shape=(None, 2048))
        self.x_trg = tf.placeholder(dtype=tf.float32, shape=(None, 2048))

        self.y_src = tf.placeholder(dtype=tf.float32, shape=(None, self.num_classes))
        self.y_trg = tf.placeholder(dtype=tf.float32, shape=(None, self.num_classes))

        T = tb.utils.TensorDict(dict(
            x_tmp=tf.placeholder(dtype=tf.float32, shape=(None, 2048)),
            y_tmp=tf.placeholder(dtype=tf.float32, shape=(None, self.num_classes))
        ))

        self.is_training = tf.placeholder(tf.bool, shape=(), name='is_training')

        self.x_src_mid = self._build_source_middle(self.x_src)
        self.x_src_prime = self._build_middle_source(self.x_src_mid)
        self.x_trg_mid = self._build_target_middle(self.x_trg)

        self.x_fr_src = self._build_domain_discriminator(self.x_src_mid)
        self.x_fr_trg = self._build_domain_discriminator(self.x_trg_mid, reuse=True)

        # use m units of D(G(x_s)) for classification on joint space
        self.m_src_on_D_logit = tf.gather(self.x_fr_src, tf.range(0, self.num_classes, dtype=tf.int32), axis=1)
        self.loss_m_src_on_D = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_src,
            logits=self.m_src_on_D_logit))

        # maximize log likelihood of target data and minimize that of source data on 11th class
        self.m_plus_1_src_logit_on_D = tf.gather(self.x_fr_src, tf.range(self.num_classes, self.num_classes + 1,
                                                                         dtype=tf.int32), axis=1)
        self.m_plus_1_trg_logit_on_D = tf.gather(self.x_fr_trg, tf.range(self.num_classes, self.num_classes + 1,
                                                                         dtype=tf.int32), axis=1)

        self.loss_m_plus_1_on_D = 0.5 * tf.reduce_mean(sigmoid_x_entropy(
            labels=tf.ones_like(self.m_plus_1_trg_logit_on_D), logits=self.m_plus_1_trg_logit_on_D) + \
                                                             sigmoid_x_entropy(
                                                                 labels=tf.zeros_like(self.m_plus_1_src_logit_on_D),
                                                                 logits=self.m_plus_1_src_logit_on_D))

        self.loss_disc = self.m_on_D_trade_off*self.loss_m_src_on_D + self.m_plus_1_on_D_trade_off*self.loss_m_plus_1_on_D

        self.y_src_logit = self._build_class_src_discriminator(self.x_src_mid, self.num_classes)
        self.y_trg_logit = self._build_class_trg_discriminator(self.x_trg_mid, self.num_classes)

        self.y_src_pred = tf.argmax(self.y_src_logit, 1, output_type=tf.int32)
        self.y_trg_pred = tf.argmax(self.y_trg_logit, 1, output_type=tf.int32)
        self.y_src_sparse = tf.argmax(self.y_src, 1, output_type=tf.int32)
        self.y_trg_sparse = tf.argmax(self.y_trg, 1, output_type=tf.int32)

        ###############################
        # classification loss
        self.src_loss_class_detail = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.y_src_logit, labels=self.y_src)  # (batch_size,)
        self.src_loss_class = tf.reduce_mean(self.src_loss_class_detail)  # real number

        self.trg_loss_class_detail = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.y_trg_logit, labels=self.y_trg)
        self.trg_loss_class = tf.reduce_mean(self.trg_loss_class_detail)  # just use for testing

        self.src_accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y_src_sparse, self.y_src_pred), 'float32'))
        self.trg_accuracy_batch = tf.cast(tf.equal(self.y_trg_sparse, self.y_trg_pred), 'float32')
        self.trg_accuracy = tf.reduce_mean(self.trg_accuracy_batch)

        #############################
        # generator loss
        self.loss_m_plus_1_on_G = 0.5 * tf.reduce_mean(sigmoid_x_entropy(
            labels=tf.zeros_like(self.m_plus_1_trg_logit_on_D), logits=self.m_plus_1_trg_logit_on_D) + \
                                                             sigmoid_x_entropy(
                                                                 labels=tf.ones_like(self.m_plus_1_src_logit_on_D),
                                                                 logits=self.m_plus_1_src_logit_on_D))

        self.A_m = self.y_trg_logit
        self.m_trg_on_D_logit = tf.gather(self.x_fr_trg, tf.range(0, self.num_classes, dtype=tf.int32), axis=1)

        self.loss_m_trg_on_G = tf.reduce_mean(
            softmax_x_entropy_two(logits=self.m_trg_on_D_logit, labels=self.A_m))

        self.loss_generator = self.m_plus_1_on_G_trade_off * self.loss_m_plus_1_on_G + \
                              self.m_on_G_trade_off * self.loss_m_trg_on_G

        #############################
        # vat loss
        self.src_loss_vat = self._build_vat_loss(
            self.x_src, self.y_src_logit, self.num_classes,
            scope_encode=self._get_scope('generator', 'src'), scope_classify='classifier'
        )
        self.trg_loss_vat = self._build_vat_loss(
            self.x_trg, self.y_trg_logit, self.num_classes,
            scope_encode=self._get_scope('generator', 'trg'), scope_classify='classifier'
        )

        #############################
        # conditional entropy loss w.r.t. target distribution
        self.trg_loss_cond_entropy = tf.reduce_mean(softmax_x_entropy_two(labels=self.y_trg_logit,
                                                                   logits=self.y_trg_logit))

        #############################
        # reconstruct loss
        # self.src_reconstruct_loss = tf.reduce_mean(tf.pow(tf.norm(self.x_src - self.x_src_prime, axis=1, ord=2), 2)) / 1000.0
        #############################
        # construct primary loss
        if self.adapt_domain_trade_off:
            self.domain_trade_off_ph = tf.placeholder(dtype=tf.float32)
        lst_primary_losses = [
            (self.src_class_trade_off, self.src_loss_class),
            (self.domain_trade_off, self.loss_generator),
            (self.src_vat_trade_off, self.src_loss_vat),
            (self.trg_trade_off, self.trg_loss_vat),
            (self.trg_trade_off, self.trg_loss_cond_entropy)
            # (self.src_recons_trade_off, self.src_reconstruct_loss)
        ]
        self.primary_loss = tf.constant(0.0)
        for trade_off, loss in lst_primary_losses:
            if trade_off != 0:
                self.primary_loss += trade_off * loss

        primary_variables = self._get_variables(self._get_primary_scopes())

        # Evaluation (EMA)
        ema = tf.train.ExponentialMovingAverage(decay=0.998)
        var_list_for_ema = primary_variables[0] + primary_variables[1]
        ema_op = ema.apply(var_list=var_list_for_ema)
        self.ema_p = self._build_classifier(T.x_tmp, self.num_classes, ema)

        # Accuracies
        self.batch_ema_acc = batch_ema_acc(T.y_tmp, self.ema_p)
        self.fn_batch_ema_acc = tb.function(self.tf_session, [T.x_tmp, T.y_tmp], self.batch_ema_acc)

        self.train_main = \
            tf.train.AdamOptimizer(self.learning_rate, 0.5).minimize(self.primary_loss, var_list=primary_variables)

        self.primary_train_op = tf.group(self.train_main, ema_op)
        # self.primary_train_op = tf.group(self.train_main)

        if self.save_grads:
            self.grads_wrt_primary_loss = tf.train.AdamOptimizer(self.learning_rate, 0.5).compute_gradients(
                self.primary_loss, var_list=primary_variables)
        #############################
        # construct secondary loss
        secondary_variables = self._get_variables(self._get_secondary_scopes())
        self.secondary_train_op = \
            tf.train.AdamOptimizer(self.learning_rate, 0.5).minimize(self.loss_disc,
                                                                   var_list=secondary_variables)
        #############################
        #  construct one more target loss
        if self.update_target_loss:
            self.target_loss = self.trg_trade_off * (self.trg_loss_vat + self.trg_loss_cond_entropy)

            self.target_train_op = \
                tf.train.AdamOptimizer(self.learning_rate, 0.5).minimize(self.target_loss,
                                                                         var_list=primary_variables)

        if self.save_grads:
            self.grads_wrt_secondary_loss = tf.train.AdamOptimizer(self.learning_rate, 0.5).compute_gradients(
                self.loss_disc, var_list=secondary_variables)
        ############################
        # summaries
        tf.summary.scalar('domain/loss_disc', self.loss_disc)
        tf.summary.scalar('domain/loss_disc/loss_m_src_on_D', self.loss_m_src_on_D)
        tf.summary.scalar('domain/loss_disc/loss_m_plus_1_on_D', self.loss_m_plus_1_on_D)

        tf.summary.scalar('primary_loss/src_loss_class', self.src_loss_class)
        tf.summary.scalar('primary_loss/loss_generator', self.loss_generator)
        tf.summary.scalar('primary_loss/loss_generator/loss_m_plus_1_on_G', self.loss_m_plus_1_on_G)
        tf.summary.scalar('primary_loss/loss_generator/loss_m_trg_on_G', self.loss_m_trg_on_G)

        tf.summary.scalar('acc/src_acc', self.src_accuracy)
        tf.summary.scalar('acc/trg_acc', self.trg_accuracy)

        tf.summary.scalar('hyperparameters/learning_rate', self.learning_rate)
        tf.summary.scalar('hyperparameters/src_class_trade_off', self.src_class_trade_off)
        tf.summary.scalar('hyperparameters/domain_trade_off',
                          self.domain_trade_off_ph if self.adapt_domain_trade_off
                          else self.domain_trade_off)

        self.tf_merged_summaries = tf.summary.merge_all()

        if self.save_grads:
            with tf.name_scope("visualize"):
                for var in tf.trainable_variables():
                    tf.summary.histogram(var.op.name + '/values', var)
                for grad, var in self.grads_wrt_primary_loss:
                    if grad is not None:
                        tf.summary.histogram(var.op.name + '/grads_wrt_primary_loss', grad)
                for grad, var in self.grads_wrt_secondary_loss:
                    if grad is not None:
                        tf.summary.histogram(var.op.name + '/grads_wrt_secondary_loss', grad)

    def _fit_loop(self):
        print('Start training', 'LAMDA at', os.path.basename(__file__))
        print('============ LOG-ID: %s ============' % self.current_time)

        self.tf_session.run(tf.global_variables_initializer())

        num_src_samples = self.data_loader.src_train[0][2].shape[0]
        num_trg_samples = self.data_loader.trg_train[0][2].shape[0]

        with self.tf_graph.as_default():
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)

        self.checkpoint_path = os.path.join(model_dir(), self.model_name, "saved-model", "{}".format(self.lamda_model_id))
        check_point = tf.train.get_checkpoint_state(self.checkpoint_path)

        if check_point and tf.train.checkpoint_exists(check_point.model_checkpoint_path):
            print("Load model parameters from %s\n" % check_point.model_checkpoint_path)
            saver.restore(self.tf_session, check_point.model_checkpoint_path)

        for it in range(self.num_iters):
            idx_src_samples = np.random.permutation(num_src_samples)[:self.batch_size]
            idx_trg_samples = np.random.permutation(num_trg_samples)[:self.batch_size]

            feed_data = dict()
            feed_data[self.x_src] = self.data_loader.src_train[0][1][idx_src_samples, :]
            feed_data[self.y_src] = self.data_loader.src_train[0][2][idx_src_samples]
            feed_data[self.y_src] = feed_data[self.y_src]

            feed_data[self.x_trg] = self.data_loader.trg_train[0][1][idx_trg_samples, :]
            feed_data[self.y_trg] = self.data_loader.trg_train[0][2][idx_trg_samples]
            feed_data[self.y_trg] = feed_data[self.y_trg]
            feed_data[self.is_training] = True

            _, loss_disc = \
                self.tf_session.run(
                    [self.secondary_train_op, self.loss_disc],
                    feed_dict=feed_data
                )

            _, src_loss_class, loss_generator, trg_loss_class, src_acc, trg_acc = \
                self.tf_session.run(
                    [self.primary_train_op, self.src_loss_class, self.loss_generator,
                     self.trg_loss_class, self.src_accuracy, self.trg_accuracy],
                    feed_dict=feed_data
                )

            if it == 0 or (it + 1) % self.summary_freq == 0:
                print("iter %d/%d loss_disc %.3f; src_loss_class %.5f; loss_generator %.3f\n"
                "src_acc %.2f" % (it + 1, self.num_iters, loss_disc, src_loss_class, loss_generator, src_acc * 100))

            if (it + 1) % self.summary_freq == 0:
                if not self.only_save_final_model:
                    self.save_trained_model(saver, it + 1)
                elif it + 1 == self.num_iters:
                    self.save_trained_model(saver, it + 1)

                # Save acc values
                self.save_value(step=it + 1)

    def save_trained_model(self, saver, step):
        # Save model
        checkpoint_path = os.path.join(model_dir(), self.model_name, "saved-model",
                                       "{}".format(self.current_time))
        checkpoint_path = os.path.join(checkpoint_path, "lamda_" + self.current_time + ".ckpt")

        directory = os.path.dirname(checkpoint_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        saver.save(self.tf_session, checkpoint_path, global_step=step)

    def save_value(self, step):
        # Save ema accuracy
        acc_trg_test_ema, summary_trg_test_ema = self.compute_value(self.fn_batch_ema_acc, 'test/trg_test_ema',
                                                                    x_full=self.data_loader.trg_test[0][1],
                                                                    y=self.data_loader.trg_test[0][2], labeler=None)
        print_list = ['trg_test_ema', round(acc_trg_test_ema * 100, 2)]
        print(print_list)

    def compute_value(self, fn_batch_ema_acc, tag, x_full, y, labeler, full=True):

        with tb.nputils.FixedSeed(0):
            shuffle = np.random.permutation(len(x_full))

        xs = x_full[shuffle]
        ys = y[shuffle] if y is not None else None

        if not full:
            xs = xs[:1000]
            ys = ys[:1000] if ys is not None else None

        n = len(xs)
        bs = 200

        acc_full = np.ones(n, dtype=float)

        for i in range(0, n, bs):
            x = xs[i:i + bs]
            y = ys[i:i + bs] if ys is not None else labeler(x)
            acc_batch = fn_batch_ema_acc(x, y)
            acc_full[i:i + bs] = acc_batch

        acc = np.mean(acc_full)

        summary = tf.Summary.Value(tag=tag, simple_value=acc)
        summary = tf.Summary(value=[summary])
        return acc, summary
