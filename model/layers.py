#  Copyright (c) 2021, Tuan Nguyen.
#  All rights reserved.

import tensorflow as tf
from tensorflow.contrib.framework import add_arg_scope
# from tensorbayes.tfutils import softmax_cross_entropy_with_two_logits as softmax_x_entropy_two

@add_arg_scope
def noise(x, std, phase, scope=None, reuse=None):
    with tf.name_scope(scope, 'noise'):
        eps = tf.random_normal(tf.shape(x), 0.0, std)
        output = tf.where(phase, x + eps, x)
    return output


@add_arg_scope
def leaky_relu(x, a=0.2, name=None):
    with tf.name_scope(name, 'leaky_relu'):
        return tf.maximum(x, a * x)

@add_arg_scope
def basic_accuracy(a, b, scope=None):
    with tf.name_scope(scope, 'basic_acc'):
        a = tf.argmax(a, 1)
        b = tf.argmax(b, 1)
        eq = tf.cast(tf.equal(a, b), 'float32')
        output = tf.reduce_mean(eq)
    return output

@add_arg_scope
def batch_ema_acc(a, b, scope=None):
    with tf.name_scope(scope, 'basic_acc'):
        a = tf.argmax(a, 1)
        b = tf.argmax(b, 1)
        output = tf.cast(tf.equal(a, b), 'float32')
    return output

@add_arg_scope
def batch_teac_stud_avg_acc(y_trg_true, y_trg_logit, y_trg_teacher,  scope=None):
    with tf.name_scope(scope, 'average_acc'):
        y_trg_prob = tf.nn.softmax(y_trg_logit)
        y_pred_avg = (y_trg_prob + y_trg_teacher) / 2.0

        y_trg_true = tf.argmax(y_trg_true, 1)
        y_pred_avg = tf.argmax(y_pred_avg, 1)
        output = tf.cast(tf.equal(y_trg_true, y_pred_avg), 'float32')
    return output

@add_arg_scope
def batch_teac_stud_ent_acc(y_trg_true, y_trg_logit, y_trg_teacher,  scope=None):
    with tf.name_scope(scope, 'entropy_acc'):
        y_trg_prob = tf.nn.softmax(y_trg_logit)
        # compute entropy
        y_trg_student_ent = -tf.reduce_sum(y_trg_prob * tf.log(y_trg_prob), axis=-1)
        y_trg_teacher_ent = -tf.reduce_sum(y_trg_teacher * tf.log(y_trg_teacher), axis=-1)
        min_entropy = tf.argmin(tf.stack([y_trg_student_ent, y_trg_teacher_ent]), axis=0)

        y_trg_pred_sparse = tf.argmax(y_trg_logit, 1, output_type=tf.int32)
        y_trg_teacher_sparse = tf.argmax(y_trg_teacher, 1, output_type=tf.int32)
        student_teacher_concat = tf.stack([y_trg_pred_sparse, y_trg_teacher_sparse], axis=1)

        y_pred_entropy_voting = tf.reduce_max(student_teacher_concat * tf.one_hot(min_entropy, 2, dtype=tf.int32),
                                              axis=1)

        y_trg_true = tf.argmax(y_trg_true, 1, output_type=tf.int32)
        output = tf.cast(tf.equal(y_trg_true, y_pred_entropy_voting), 'float32')
    return output