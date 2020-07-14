import os
import glob

import tensorflow as tf
import neuralgym as ng

from inpaint_model import InpaintCAModel


def multigpu_graph_def(model, FLAGS, data, gpu_id=0, loss_type='g'):
    with tf.device('/cpu:0'):
        batch_data = data.data_pipeline(FLAGS.batch_size)
    if gpu_id == 0 and loss_type == 'g':
        _, _, losses = model.build_graph_with_losses(
            FLAGS, batch_data, FLAGS, summary=True, reuse=True)
    else:
        _, _, losses = model.build_graph_with_losses(
            FLAGS, batch_data, FLAGS, reuse=True)
    if loss_type == 'g':
        return losses['g_loss']
    elif loss_type == 'd':
        return losses['d_loss']
    else:
        raise ValueError('loss type is not supported.')


if __name__ == "__main__":
    # training data
    FLAGS = ng.Config('inpaint.yml')
    img_shapes = FLAGS.img_shapes
    
    # Read flist input and mask image
    with open(FLAGS.data_flist[FLAGS.dataset][0]) as f:
        fnames_input = f.read().splitlines()
    train_fnames = [(fname, fname[:-4]+'mask.png') for fname in fnames_input]
    data = ng.data.DataFromFNames(
        train_fnames, img_shapes, random_crop=FLAGS.random_crop,
        nthreads=FLAGS.num_cpus_per_job)
    batch_data = data.data_pipeline(FLAGS.batch_size)
    # main model
    model = InpaintCAModel()
    g_vars, d_vars, losses = model.build_graph_with_losses(FLAGS, batch_data)
    # validation images
    with open(FLAGS.data_flist[FLAGS.dataset][1]) as f:
        val_fnames = f.read().splitlines()
    val_fnames = [(fname, fname[:-4] + 'mask.png') for fname in val_fnames]
    batch_val = ng.data.DataFromFNames(
            val_fnames, img_shapes, nthreads=FLAGS.num_cpus_per_job,
            random_crop = FLAGS.random_crop).data_pipeline(FLAGS.batch_size)
    batch_val_complete = model.build_infer_graph(FLAGS, batch_val)
    num_iters_val = len(val_fnames)//FLAGS.batch_size        
    # training settings
    lr = tf.get_variable(
        'lr', shape=[], trainable=False,
        initializer=tf.constant_initializer(1e-4))
    d_optimizer = tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.999)
    g_optimizer = d_optimizer
    # train discriminator with secondary trainer, should initialize before
    # primary trainer.
#    discriminator_training_callback = ng.callbacks.SecondaryTrainer(
    discriminator_training_callback = ng.callbacks.SecondaryMultiGPUTrainer(
        num_gpus=FLAGS.num_gpus_per_job,
        pstep=1,
        optimizer=d_optimizer,
        var_list=d_vars,
        max_iters=1,
        grads_summary=False,
        graph_def=multigpu_graph_def,
        graph_def_kwargs={
            'model': model, 'FLAGS': FLAGS, 'data': data, 'loss_type': 'd'},
    )
    # train generator with primary trainer
#    trainer = ng.train.Trainer(
    trainer = ng.train.MultiGPUTrainer(
        num_gpus=FLAGS.num_gpus_per_job,
        optimizer=g_optimizer,
        var_list=g_vars,
        max_iters=FLAGS.max_iters,
        graph_def=multigpu_graph_def,
        grads_summary=False,
        gradient_processor=None,
        graph_def_kwargs={
            'model': model, 'FLAGS': FLAGS, 'data': data, 'loss_type': 'g'},
        spe=FLAGS.train_spe,
        log_dir=FLAGS.log_dir,
    )
    # add all callbacks
    trainer.add_callbacks([
        discriminator_training_callback,
        ng.callbacks.WeightsViewer(),
        ng.callbacks.ModelRestorer(trainer.context['saver'], dump_prefix=FLAGS.model_restore+'/snap', optimistic=True),
        ng.callbacks.ModelSaver(FLAGS.train_spe, trainer.context['saver'], FLAGS.log_dir+'/snap'),
        ng.callbacks.SummaryWriter((FLAGS.val_psteps//1), trainer.context['summary_writer'], tf.summary.merge_all()),
        #ng.callbacks.EarlyStopper(FLAGS.train_spe, model, batch_val, num_iters_val, FLAGS),
    ])
    # launch training
    trainer.train()
