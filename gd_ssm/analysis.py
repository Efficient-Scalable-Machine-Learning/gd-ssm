import numpy as np
import wandb
import PIL.Image, PIL.ImageDraw, PIL.ImageFont
import jax
from functools import partial
from jax import vmap,jit
from jax import numpy as jnp
from jax import jacfwd, jacrev,random
from gd_ssm.gd_ssm import discretize_zoh
from gd_ssm.model_init import model_init
from gd_ssm.train_helpers import linear_warmup, cosine_annealing, constant_lr, train_epoch,\
        validate,get_prediction
from gd_ssm.data import create_constructed_reg_data,create_reg_data_classic_token,\
    create_vec_reg_data_classic_token, create_reg_data_sin_classic
from IPython.display import Image, HTML, clear_output
import matplotlib.pylab as pl
import matplotlib.colors as mcolors
import os
colors = pl.colormaps['Dark2'] 


def scan_lrs(args,rng,lin_diag,bs):
    args.discretization = None
    lr_scan_range = jnp.arange(0.001, 25, 0.1)
    eval_rng,_ =  random.split(rng, num=2)
    if args.dataset in ["normal_token_scalar"]:
        seq_len = (args.dataset_size *2) + 1
        if args.regression == "non-linear":
            data_creator = vmap(create_reg_data_sin_classic,
                            in_axes=(0, None, None, None, None, None),
                            out_axes=0)
        else:
          data_creator = vmap(create_reg_data_classic_token,
                              in_axes=(0, None, None, None, None, None),
                              out_axes=0)
    elif args.dataset in ["normal_token_vector"]:
        seq_len = (args.dataset_size *2) + 1
        data_creator = vmap(create_vec_reg_data_classic_token,
                            in_axes=(0, None, None, None, None, None),
                            out_axes=0)
    else:
        pass
    data = data_creator(random.split(eval_rng, num=bs),
                      args.input_size,
                      args.dataset_size,
                      args.size_distract,
                      args.input_range,
                      args.weight_scale)
    
    losses_lr = []
    for lr in lr_scan_range:
        if args.n_layers ==1:
            gd_model_cls,gd_state = model_init(args,rng,gd_params=True,gd_lr=lr)
        else:
            gd_model_cls,gd_state = model_init(args,rng,gd_params=False,gd_lr=lr)
                ## initialising LSA weights
            gd_state.params['layers_0']['seq']['w_q'] = jnp.outer(jnp.array([1.0,0,0]), jnp.array([0,1.0,0])) 
            gd_state.params['layers_1']['seq']['w_q'] = jnp.outer(jnp.array([1.0,0,0]), jnp.array([1.0,0,0]))
            gd_state.params['layers_2']['seq']['w_q'] = jnp.outer(jnp.array([1.0,0,0]), jnp.array([0,1.0,0]))
            ## skipweight
            gd_state.params['layers_0']['seq']['D'] = jnp.array([0, 0, 1])
            gd_state.params['layers_1']['seq']['D'] = jnp.array([0, 0, 1])
            gd_state.params['layers_2']['seq']['D'] = jnp.array([0, 0, 1]) 
            ## Recurrent weights
            gd_state.params['layers_0']['seq']['Lambda_re'] = jnp.ones((args.ssm_size_base,args.ssm_size_base))
            gd_state.params['layers_1']['seq']['Lambda_re'] = jnp.ones((args.ssm_size_base,args.ssm_size_base))
            gd_state.params['layers_2']['seq']['Lambda_re'] = jnp.ones((args.ssm_size_base,args.ssm_size_base))          
            ## Learning rate parameters
            gd_state.params['layers_0']['seq']['C'] = jnp.eye(args.ssm_size_base) * (-lr/args.ssm_size_base)
            gd_state.params['layers_1']['seq']['C'] = jnp.eye(args.ssm_size_base) * (-lr/args.ssm_size_base)
            gd_state.params['layers_2']['seq']['C'] = jnp.eye(args.ssm_size_base)* (-lr/args.ssm_size_base)
            
        
        val_loss = validate(gd_state,
                            gd_model_cls,
                            data,
                            seq_len,
                            args.input_size,
                            args.batchnorm,
                            args.dataset)
        losses_lr.append(val_loss)
    losses_lr = jnp.array(losses_lr)
    lr_min_i = jnp.argmin(losses_lr)
    min_loss = jnp.min(losses_lr)
    gd_lr = lr_scan_range[lr_min_i]
    if args.n_layers ==1:
      gd_model_cls,gd_state = model_init(args,rng,gd_params=True,gd_lr=gd_lr)
    else:
      gd_model_cls,gd_state = model_init(args,rng,gd_params=False,gd_lr=lr)
          ## initialising LSA weights
      gd_state.params['layers_0']['seq']['w_q'] = jnp.outer(jnp.array([1.0,0,0]), jnp.array([0,1.0,0]))
      gd_state.params['layers_0']['seq']['w_q'] = gd_state.params['layers_0']['seq']['w_q'].at[-1,-1].set(1)
      gd_state.params['layers_1']['seq']['w_q'] = jnp.outer(jnp.array([1.0,0,0]), jnp.array([1.0,0,0]))
      gd_state.params['layers_2']['seq']['w_q'] = jnp.outer(jnp.array([1.0,0,0]), jnp.array([0,1.0,0]))
      gd_state.params['layers_2']['seq']['w_q'] = gd_state.params['layers_2']['seq']['w_q'].at[-1,-1].set(1)
      
      ## skipweight
      gd_state.params['layers_2']['seq']['D'] = jnp.array([0, 0, 1]) *1.0
      ## Recurrent weights
      gd_state.params['layers_0']['seq']['Lambda_re'] = jnp.ones((args.ssm_size_base,args.ssm_size_base))
      gd_state.params['layers_1']['seq']['Lambda_re'] = jnp.ones((args.ssm_size_base,args.ssm_size_base))
      gd_state.params['layers_2']['seq']['Lambda_re'] = jnp.ones((args.ssm_size_base,args.ssm_size_base))          
      ## Learning rate parameters
      gd_state.params['layers_0']['seq']['C'] = jnp.eye(args.ssm_size_base) * (-gd_lr/args.ssm_size_base)
      gd_state.params['layers_1']['seq']['C'] = jnp.eye(args.ssm_size_base) * (-gd_lr/args.ssm_size_base)
      gd_state.params['layers_2']['seq']['C'] = jnp.eye(args.ssm_size_base)* (-gd_lr/args.ssm_size_base)
    return gd_lr,min_loss,gd_model_cls,gd_state
      


def np2pil(a):
  if a.dtype in [np.float32, np.float64]:
    a = np.uint8(np.clip(a, 0, 1)*255)
  return PIL.Image.fromarray(a)

def imwrite(f, a, fmt=None):
  a = np.asarray(a)
  if isinstance(f, str):
    fmt = f.rsplit('.', 1)[-1].lower()
    if fmt == 'jpg':
      fmt = 'jpeg'
    f = open(f, 'wb') #GFile.open(f, 'wb')
  np2pil(a).save(f, fmt, quality=95)

def imencode(a, fmt='jpeg'):
  a = np.asarray(a)
  if len(a.shape) == 3 and a.shape[-1] == 4:
    fmt = 'png'
  f = io.BytesIO()
  imwrite(f, a, fmt)
  return f.getvalue()

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def grab_plot(close=True):
  """Return the current Matplotlib figure as an image."""
  fig = pl.gcf()
  fig.canvas.draw()
  img = np.array(fig.canvas.renderer._renderer)
  a = np.float32(img[..., 3:]/255.0)
  img = np.uint8(255*(1.0-a) + img[...,:3] * a)  # alpha
  if close:
    pl.close()
  return img

def display_learning(training_steps,train, test=None, gt=None, inter=None, title="train", 
                     title1="Trained TF", title2="Test", 
                     title3='Gradient descent', title4='Interpolated',
                     y_label1 = 'L2 Norm', y_label2 = 'Cosine sim',
                     y_lim_l=0,  y_lim_u=1, single_seeds= False,
                     plot_title = None,
                     y_lim_u2= 1., y_lim_l2=0.,  x_label = 'Training steps',   
                     second_axis=False, color_add=0, rw=10, num_iter_os=None, 
                     allow_download=False, plot_num=1, two_plots=False, 
                     loc_first = 'upper left', label_title="Loss",
                     loc_sec='upper left', yscale_log=False, line="-",
                     color_axis=True, 
                     height=3.5, width = 4, ax1=None, ax2=None):
  
  """Update learning curve image."""
  pl.rcParams.update({'font.size': 12})
  pl.rc('axes', labelsize=14)
  pl.rcParams.update({
        "text.usetex": False,
    })

  train_list = train
  train = np.array(train)
  num_seeds_train = train.shape[0]
  train_std = np.std(train, axis=0)
  train = np.mean(train, axis=0)
  training_step = training_steps
  
  if test is not None:
    test_list = test
    test_std = np.std(test, axis=0)
    test = np.mean(test, axis=0)

  if gt is not None:
    gt_list = gt
    gt_std = np.std(gt, axis=0)
    gt = np.mean(gt, axis=0)

  if inter is not None:
    inter_list = inter
    inter_std = np.std(inter, axis=0)
    inter = np.mean(inter, axis=0)

  if plot_num == 1:
    fig, ax1 = pl.subplots()
    ax1.set_xlabel(x_label)
    fig.set_size_inches(width, height)

  
  if test is not None and not second_axis:
    #training_step = np.arange(0, num_iter_os, int(num_iter_os/len(test)))
    training_step = training_steps #FIXME : change training_steps later
    if len(test_list) > 1:
      if single_seeds:
        for s in test_list:
          ax1.plot(training_step, s, color=colors(0.1+color_add), alpha=0.2, label=title2,linewidth='2')
      else:
        ax1.fill_between(training_step, test-test_std, test+test_std ,alpha=0.2, facecolor=colors(0.1+color_add))
    ax1.plot(training_step, test, color=colors(0.1+color_add), label=title2,linewidth='3')
    #test_avg = moving_average(test, rw)
    #ax1.plot(training_step[:len(test_avg)], test_avg, color=colors(0.1+color_add), label=title2)
      
  if gt is not None:
    if not second_axis:
      #training_step = np.arange(0, num_iter_os, int(num_iter_os/len(gt)))
      #ax1.plot(training_step[:len(gt[:-rw])], gt[:-rw], color=colors(0.2+color_add), alpha=0.3)
      #gt_avg = moving_average(gt, rw)
      ax1.plot(training_step, gt, color=colors(0.2+color_add), label=title3,linewidth='3')
      if len(gt_list) > 1:
        if single_seeds:
          for s in gt_list:
            ax1.plot(training_step, s, color=colors(0.2+color_add), alpha=0.2, linewidth='2', zorder=0)
        else:
          ax1.fill_between(training_step, gt-gt_std, gt+gt_std,alpha=0.2, facecolor=colors(0.2+color_add))
    else:
      #training_step = np.arange(0, num_iter_os, int(num_iter_os/len(gt)))
      ax1.plot(training_step, gt, color=colors(0.6+color_add), label=title3,linewidth='3')
      if len(gt_list) > 1:
        if single_seeds:
          for s in gt_list:
            ax1.plot(training_step, s, color=colors(0.6+color_add), alpha=0.3, linewidth='2', zorder=0)
        else:
          ax1.fill_between(training_step, gt-gt_std, gt+gt_std ,alpha=0.2, facecolor=colors(0.6+color_add))

  if test is not None and second_axis:
    #training_step = np.arange(0, num_iter_os, int(num_iter_os/len(test)))
    #training_step = training_steps
    ax1.plot(training_step, test, color=colors(0.5+color_add), label=title2,linewidth='3')
    #test_avg = moving_average(test, rw)
    #ax1.plot(training_step[:len(test_avg)],test_avg, color=colors(0.5+color_add))
    if len(test_list) > 1:
      if single_seeds:
        for s in test_list:
          ax1.plot(training_step, s, color=colors(0.5+color_add), linewidth='2', alpha=0.3, zorder=0)
      else:
        ax1.fill_between(training_step, test-test_std, test+test_std ,alpha=0.2, facecolor=colors(0.5+color_add))

  if inter is not None and not second_axis:
    #training_step = np.arange(0, num_iter_os, int(num_iter_os/len(inter)))
    ax1.plot(training_step, inter, color=colors(0.4+color_add), label=title4, linewidth='3', zorder=10)
    if len(inter_list) > 1:
      if single_seeds:
        for s in inter_list:
          ax1.plot(training_step, s, color=colors(0.4+color_add), alpha=0.3, linewidth='2', zorder=0)
      else:
        ax1.fill_between(training_step, inter-inter_std, inter+inter_std ,alpha=0.2, facecolor=colors(0.4+color_add), zorder=1)
    #inter_avg = moving_average(inter, rw)
    #ax1.plot(training_step[:len(inter_avg)], inter_avg, color=colors(0.7+color_add), label=title4)


  if second_axis:
    if ax2 is None:
      ax2 = ax1.twinx()
    ax2.set_zorder(0)
    ax1.set_zorder(1)
    ax1.set_frame_on(False)
    #train_avg = moving_average(train, rw)
    #ax2.plot(train[:-rw], color=colors(0.1+color_add), alpha=0.3)
    ax2.plot(training_step, train, color=colors(0.4+color_add), label=title1, linewidth='3')
    ax2.plot(training_step, np.ones_like(train), "--", color="gray", linewidth='0.7')
    if len(train_list) > 1:
      if single_seeds:
        for s in train_list:
          print(training_step, s)
          ax1.plot(training_step, s, line, color=colors(0.4+color_add), alpha=0.3, linewidth='2', zorder=0)
      else:
        ax2.fill_between(training_step, train-train_std, train+train_std ,alpha=0.2, facecolor=colors(0.4+color_add))

    if color_axis:
      ax2.yaxis.label.set_color(colors(0.4+color_add))
    else:
      legend2 = ax2.legend(loc='upper right', framealpha=0.99, facecolor='white')
      legend2.set_zorder(100)
    ax2.spines['top'].set_visible(False)
  else:
    #train_avg = moving_average(train, rw)
    if line != "-":
      ax1.scatter(training_step, train, s=[100 for _ in training_step], 
                  marker="+", color=colors(0.3+color_add), alpha=1, label=title1, zorder=3, linewidths=3)
    else:
      ax1.plot(training_step, train, line, color=colors(0.3+color_add), label=title1, linewidth='3', zorder=11)
    #ax1.plot(training_step[:len(train_avg)], train_avg, line, color=colors(0.3+color_add), label=title1)
    if len(train_list) > 1:
      if single_seeds:
          for s in train_list:
            ax1.plot(training_step, s, line, color=colors(0.3+color_add), alpha=0.3, linewidth='2', zorder=0)
      else: 
        ax1.fill_between(training_step, train-train_std, train+train_std,
                       alpha=0.5, facecolor=colors(0.3+color_add))

    ax1.legend(loc='best', framealpha=1, facecolor='white')
    ax1.spines['right'].set_visible(False)
    legend = ax1.legend(loc='upper right', framealpha=0.99, facecolor='white')
    legend.set_zorder(100)
  
  legend1 = ax1.legend(loc=loc_first, framealpha=0.99, facecolor='white')
  legend1.set_zorder(100)
  if second_axis:
    ax2.set_ylabel(y_label2)
    ax1.set_ylabel(y_label1)
    ax1.set_ylim(y_lim_l, y_lim_u)
    legend1 = ax1.legend(loc=loc_sec, framealpha=0.99, facecolor='white')
    ax2.set_ylim(y_lim_l2, y_lim_u2)
    ax1.set_ylim(bottom=0)
  else:
    pl.ylabel(label_title)
    pl.ylim(y_lim_l, y_lim_u)
  ax1.spines['top'].set_visible(False)
  
  if plot_title is not None:
    pl.title(plot_title)
    
  if yscale_log:
    ax1.set_yscale("log")
  #pl.title(title)
  pl.tight_layout()

  if allow_download:
    if second_axis:
      pl.savefig("sim.pdf", format="pdf")
      #download_file sim.pdf
    else:
      pl.savefig("train.pdf", format="pdf")
      #download_file train.pdf
  else:
    img = grab_plot()
    display(Image(data=imencode(img, fmt='jpeg')), display_id=title)
    
def analyse(dataset,dataset_size,batchnorm,data,state,model_cls, gd_model_cls,gd_state):
    # Trained Transformer
    if dataset in ["normal_token_scalar"]:
        seq_len = (dataset_size *2) + 1
    elif dataset in ["normal_token_vector"]:
        seq_len = (dataset_size *2) + 1
    else:
        pass
    pred = lambda z: get_prediction(state,model_cls,z[None, ...],seq_len,10,batchnorm,dataset)
    predictions = vmap(pred)(data[0])
    if dataset in ['normal_token_vector']:
      grads = vmap(jacrev(pred))(data[0])[:, :,-1, :]
      grads_norm = jnp.linalg.norm(grads, axis=2)
    elif dataset in ['normal_token_scalar']:
      grads = vmap(jax.grad(pred))(data[0])[:, -1, :]
      grads_norm = jnp.linalg.norm(grads, axis=1)
    else:
      grads = vmap(jax.grad(pred))(data[0])[:, -1, :-1]
      grads_norm = jnp.linalg.norm(grads, axis=1)
    #grads = vmap(jax.grad(pred))(data[0])[:, -1, :-1]  #+ w_init
    
#    gd_model_cls,gd_state = model_init(args,rng,gd_params=True,gd_lr=gd_lr)
    
  # GD
    pred_c = lambda z: get_prediction(gd_state,gd_model_cls,z[None, ...],seq_len,10,batchnorm,dataset)
    if dataset in ['normal_token_vector']:
      grads_c = vmap(jacrev(pred_c))(data[0])[:, :,-1, :]
      grads_c_norm = jnp.linalg.norm(grads_c, axis=2)
      dot_products = jnp.einsum('ijk,ijk->ij', grads/(grads_norm[..., None] + 1e-8),
                                grads_c/(grads_c_norm[..., None]+ 1e-8))
      dot_products = jnp.mean(dot_products,axis=1)
    elif dataset in ['normal_token_scalar']:
      grads_c = vmap(jax.grad(pred_c))(data[0])[:, -1, :]
      grads_c_norm = jnp.linalg.norm(grads_c, axis=1)
      dot_products = jnp.einsum('ij,ij->i', grads/(grads_norm[..., None] + 1e-8),
                                grads_c/(grads_c_norm[..., None]+ 1e-8))
      #dot_products = jnp.mean(dot_products,axis=1)
    else:
      grads_c = vmap(jax.grad(pred_c))(data[0])[:, -1, :-1]
      grads_c_norm = jnp.linalg.norm(grads_c, axis=1)
      dot_products = jnp.einsum('ij,ij->i', grads/(grads_norm[..., None] + 1e-8),
                                grads_c/(grads_c_norm[..., None]+ 1e-8))
    #grads_c = vmap(jax.grad(pred_c))(data[0])[:, -1, :-1] 
    predictions_c = vmap(pred_c)(data[0])
    # Metrics
    dot = jnp.mean(dot_products)
    norm = jnp.mean(jnp.linalg.norm(grads-grads_c, axis=1))
    pred_norm = jnp.mean(jnp.linalg.norm(predictions[..., None]-
                                        predictions_c[..., None], axis=1))
    return dot, norm, pred_norm
  
def display_featuresize_expts(input_sizes, gd_losses, trained_ssm_loss):

    log = False
    every = 5
    leg_loc = 'upper left'
    pl.rcParams.update({'font.size': 12})

    fig, ax1 = pl.subplots()
    fig.set_size_inches(4, 3.5)
    pl.xlabel("Num datapoints / Input dim")
    pl.ylabel('Loss')
    if log:
      pl.yscale('log')
  
    ax1.scatter(input_sizes, gd_losses, s=[150 for _ in input_sizes], marker="v", color=colors(0.1),label="GD",linewidth=3, zorder=2)
    ax1.scatter(input_sizes, trained_ssm_loss, s=[230 for _ in input_sizes], marker="+", color=colors(0.3), alpha=1, label="Trained GD-SSM", zorder=5, linewidths=3)
    #pl.scatter(input_sizes, gd_losses, marker='v', color='green', label='GD', s=100)
    #pl.scatter(input_sizes, trained_ssm_losses, marker='+', color='blue', label='Trained GD-SSM', s=100)
    
    legend1 = ax1.legend(loc=leg_loc,framealpha=0.85, facecolor='white')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_yticks([0.1], minor= [0.1]) 
    
    # pl.xscale('log')
    # pl.xlabel('Num datapoints / Input dim')
    # pl.ylabel('Loss')
    #pl.xticks(input_sizes, input_sizes)
    ax1.set_xticks(input_sizes)
    
    #pl.ylim(0, np.max(np.concatenate([np.array(gd_losses), np.array(trained_ssm_loss)])) + 0.2)
    
    # pl.legend(loc='upper left')
    # ax = pl.gca()
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(True)
    # ax.spines['left'].set_visible(True)
    
    pl.tight_layout()
    pl.show()
    pl.savefig("nfeature.svg", format="svg")

  

def display_ood_data(ir_gd_list, ir_t_list):
    log = True
    every = 5
    l_bound, u_bound = 0.06, 10
    leg_loc = 'upper left'
    pl.rcParams.update({'font.size': 12})

    title = r'Test on larger inputs'

    x_axis = r'$\alpha$   where x $\sim$ U($-\alpha, \alpha$)'
    download = 'Input scale'
    ir_inter_list = np.ones(len(ir_t_list))
    ir_gd_trained_list = np.ones(len(ir_t_list))
    data = [ir_gd_list, ir_gd_trained_list, ir_t_list, ir_inter_list, np.arange(0.5, 2+0.03, 0.03)]
    training_x = np.ones([100])
    step = np.max(np.array(data[:1]))/100.
    training_y = np.arange(0, np.max(np.array(data[:1])) - step, step)
    
    
    fig, ax1 = pl.subplots()
    fig.set_size_inches(4, 3.5)
    pl.xlabel(x_axis)
    pl.ylabel('Loss')
    if log:
      pl.yscale('log')

    #print(training_x, training_y)
    pl.plot(training_x[:len(training_y)], training_y, "--", color="gray", linewidth='0.7',zorder=0)

    stretch = data[-1][0::every]
    gd_list = np.array(data[0])
    num_seeds = gd_list.shape[0]
    #ir_gd_std = np.std(gd_list, axis=0)[0][0::every]
    ir_gd = np.mean(gd_list, axis=0)[0][0::every]

    # ir_gd_trained_std = np.std(data[1], axis=0)[0][0::every]
    # ir_gd_trained = np.mean(data[1], axis=0)[0][0::every]

    ir_t_std = np.std(data[2], axis=0)[0][0::every]
    ir_t = np.mean(data[2], axis=0)[0][0::every]

    ax1.scatter(stretch, ir_gd, s=[150 for _ in stretch], marker="v", color=colors(0.1),label="GD",linewidth=3, zorder=2)
    ax1.plot(stretch, ir_gd, color=colors(0.1),linewidth='2', zorder=2)
    ax1.scatter(stretch, ir_t, s=[230 for _ in stretch], marker="+", color=colors(0.3), alpha=1, label="Trained GD-SSM", zorder=5, linewidths=3)
    legend1 = ax1.legend(loc=leg_loc,framealpha=0.85, facecolor='white')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_yticks([0.1], minor= [0.1])
    pl.title(title)
    if 'ood' in download: 
      pl.tight_layout()
    #pl.savefig("ood.pdf", format="pdf")
    #%download_file ood.pdf
    else:
      pl.tight_layout()
      pl.savefig("normal_multi.svg", format="svg")
    #%download_file normal.pdf
    pl.show()
    
def nfeature_analyse(args):
    key = random.PRNGKey(args.jax_seed)
    if args.USE_WANDB:
        # Make wandb config dictionary
        wandb.init(project=args.wandb_project, job_type='model_training', config=vars(args), entity=args.wandb_entity)
    dim_ls = [5, 10, 20, 35, 50]
    trained_ssm_loss = []
    gd_ssm_loss = []
    for in_dim in dim_ls:
    # Set randomness...
      args.input_size=in_dim
      args.d_model = in_dim
      args.ssm_size_base = in_dim
      if in_dim %2  !=0:
        args.blocks = 1
      key,_ = random.split(key,2)
      init_rng, train_rng , data_rng,eval_rng= random.split(key, num=4)
      if args.regression == "non-linear":
              args.input_size = 1
      model_cls,state = model_init(args,init_rng,gd_params=False,gd_lr=None)
      ssm_lr = args.ssm_lr_base
      lr = args.lr_factor * ssm_lr
      ### Data
      if (args.dataset !="normal_token_scalar") & (args.regression=='non-linear'):
          raise NotImplementedError("Non-linear regression only implemented for dataset 'normal_token_scalar'") 
      
      if args.dataset in ["normal_token_scalar"]:
          seq_len = (args.dataset_size *2) + 1
          if args.regression == "non-linear":
              data_creator = vmap(create_reg_data_sin_classic,
                              in_axes=(0, None, None, None, None, None),
                              out_axes=0)
          else:
              data_creator = vmap(create_reg_data_classic_token,
                              in_axes=(0, None, None, None, None, None),
                              out_axes=0)
      elif args.dataset in ["normal_token_vector"]:
          seq_len = (args.dataset_size *2) + 1
          data_creator = vmap(create_vec_reg_data_classic_token,
                              in_axes=(0, None, None, None, None, None),
                              out_axes=0)
      elif args.dataset in ["constructed_token"]:
          seq_len = args.dataset_size
          data_creator = vmap(create_constructed_reg_data,
                          in_axes=(0, None, None, None, None, None),
                          out_axes=0)
          
      else:
          raise NotImplementedError("dataset method {} not implemented".format(args.dataset))
          
      #Create eval data
      eval_data = data_creator(random.split(eval_rng, num=10000),
                                      in_dim, 
                                      args.dataset_size,
                                      args.size_distract,
                                      args.input_range,
                                      args.weight_scale)
      gd_lr,gd_min_loss,gd_model_cls,gd_state = scan_lrs(args,data_rng,False,10000)
      #gd_ssm_loss.append(gd_min_loss)
      # Training Loop over epochs
      ls_trainloss, ls_valloss= [],[]
      step = 0  # for per step learning rate decay
      train_size= args.bsz # FIXME - hardcode
      steps_per_epoch = int(train_size/args.bsz)
      for epoch in range(args.epochs):
          rng, data_rng = random.split(data_rng, 2)
          trainloader = data_creator(random.split(rng, num=args.bsz), 
                                      in_dim,
                                      args.dataset_size,
                                      args.size_distract,
                                      args.input_range,
                                      args.weight_scale)
          #print(f"[*] Starting Training Epoch {epoch + 1}...")

          if epoch < args.warmup_end:
              print("using linear warmup for epoch {}".format(epoch+1))
              decay_function = linear_warmup
              end_step = steps_per_epoch * args.warmup_end

          elif args.cosine_anneal:
              if (epoch+1) % 100 ==0:
                  print("using cosine annealing for epoch {}".format(epoch+1))
              decay_function = cosine_annealing
              #for per step learning rate decay
              end_step = steps_per_epoch * args.epochs - (steps_per_epoch * args.warmup_end)
          else:
              #print("using constant lr for epoch {}".format(epoch+1))
              decay_function = constant_lr
              end_step = None
          lr_params = (decay_function, ssm_lr, lr, step, end_step, args.opt_config, args.lr_min)
          train_rng, skey = random.split(train_rng)
          state, train_loss, step = train_epoch(state,
                                                  skey,
                                                  model_cls,
                                                  trainloader,
                                                  seq_len,
                                                  in_dim,
                                                  args.batchnorm,
                                                  args.dataset,
                                                  lr_params)
          val_loss = validate(state,
                              model_cls,
                              eval_data,
                              seq_len,
                              in_dim,
                              args.batchnorm,
                              args.dataset)
          ls_trainloss.append(train_loss)
          ls_valloss.append(val_loss)
          if (epoch<1000) & (args.n_layers !=1):
                  gd_state, gd_train_loss, step = train_epoch(gd_state,
                                                  skey,
                                                  gd_model_cls,
                                                  trainloader,
                                                  seq_len,
                                                  in_dim,
                                                  args.batchnorm,
                                                  args.dataset,
                                                  lr_params)
                  gd_val_loss = validate(gd_state,
                              gd_model_cls,
                              eval_data,
                              seq_len,
                              in_dim,
                              args.batchnorm,
                              args.dataset)
                  gd_min_loss = gd_val_loss
      best_loss = np.array(np.min(ls_valloss))
      trained_ssm_loss.append(best_loss)
      if args.USE_WANDB:
        wandb.log({"no_features": in_dim,"GD_loss":gd_min_loss,"trained_SSM_loss":best_loss})
      gd_ssm_loss.append(gd_min_loss)
    display_featuresize_expts(dim_ls,gd_ssm_loss,trained_ssm_loss)
      
def weight_visualisation(args,rng,training_state,gd_lr=None):
  if gd_lr is None:
    gd_lr,_,_,_ = scan_lrs(args,rng,False,10000)

  if args.n_layers==1:
      #GD parameters
      w_q_gd = np.outer(np.array([1,0,0]), np.array([0,1,0]))
      Lambda_bar_gd = np.ones((args.ssm_size_base,args.ssm_size_base))
      rec_param_gd = np.diag(np.mean(Lambda_bar_gd,axis=1))
      C_tilde_gd = (-gd_lr/10)*np.eye(args.ssm_size_base)
      D_gd = np.array([0,0,1])
      #Trained SSM parameters
      w_q= training_state.params["layers_0"]["seq"]["w_q"] #Local self attention weight
      D= training_state.params["layers_0"]["seq"]["D"] # Skip connection weight
      discretization = "zoh"
      step_rescale = 1.0
      lambda_bar_all = []
      Lambda_re = training_state.params["layers_0"]["seq"]["Lambda_re"]
      log_step = training_state.params["layers_0"]["seq"]["log_step"]
      for param_id in range(Lambda_re.shape[0]):
                  step = step_rescale * np.exp(log_step[:, 0])
                  # Discretize
                  if discretization in ["zoh"]:
                      Lambda_bar, _ = discretize_zoh(Lambda_re[param_id,:], training_state.params["layers_0"]["seq"]["B"], step)
                  else:
                      raise NotImplementedError("Discretization method {} not implemented".format(discretization))
                  lambda_bar_all.append(Lambda_bar)
      Lambda_bar = np.vstack(lambda_bar_all)     # Recurrent weight
      rec_param = np.diag(np.mean(Lambda_bar,axis=1))
      C = training_state.params["layers_0"]["seq"]["C"]
      fig, (ax1, ax2) = pl.subplots(figsize=(15, 5), ncols=2)
          
      vmin = min(np.min(w_q), np.min(w_q_gd))
      vmax = max(np.max(w_q), np.max(w_q_gd))
          
      pos = ax1.imshow(np.abs(w_q), cmap='RdBu', vmin=vmin, vmax=vmax)
      ax1.set_title('Trained')
      ax1.set_yticks(ticks=range(0, w_q.shape[1]))
      ax1.set_yticklabels(range(1, w_q.shape[1]+1))
      ax1.set_xticks(ticks=range(0, w_q.shape[0]))
      ax1.set_xticklabels(range(1, w_q.shape[0]+1))
      #fig.colorbar(pos, ax=ax1, shrink=0.6)

      pos = ax2.imshow(np.abs(w_q_gd), cmap='RdBu', vmin=vmin, vmax=vmax)
      ax2.set_title('Constructed')
      ax2.set_yticks(ticks=range(0, w_q_gd.shape[1]))
      ax2.set_yticklabels(range(1, w_q_gd.shape[1]+1))
      ax2.set_xticks(ticks=range(0, w_q_gd.shape[0]))
      ax2.set_xticklabels(range(1, w_q_gd.shape[0]+1))
      fig.colorbar(pos, ax=ax2, shrink=0.6)
      fig.suptitle("Local self attention weight")
      if not os.path.isdir(os.path.join('fig', args.regression,str(args.n_layers))):
            os.makedirs(os.path.join('fig', args.regression,str(args.n_layers)))
      fig.savefig('fig/'+args.regression+"/"+str(args.n_layers)+"/lsa_weights.svg")

      fig, (ax1, ax2) = pl.subplots(figsize=(15, 5), ncols=2)
          
      vmin = min(np.min(rec_param), np.min(rec_param_gd))
      vmax = max(np.max(rec_param), np.max(rec_param_gd))
          
      pos = ax1.imshow(np.abs(rec_param), cmap='RdBu', vmin=0.98, vmax=vmax)
      ax1.set_title('Trained')
      ax1.set_yticks(ticks=range(0, rec_param.shape[1]))
      ax1.set_yticklabels(range(1, rec_param.shape[1]+1))
      ax1.set_xticks(ticks=range(0, rec_param.shape[0]))
      ax1.set_xticklabels(range(1, rec_param.shape[0]+1))
      #fig.colorbar(pos, ax=ax1, shrink=0.6)

      pos = ax2.imshow(np.abs(rec_param_gd), cmap='RdBu', vmin=0.98, vmax=vmax)
      ax2.set_title('Constructed')
      ax2.set_yticks(ticks=range(0, rec_param_gd.shape[1]))
      ax2.set_yticklabels(range(1, rec_param_gd.shape[1]+1))
      ax2.set_xticks(ticks=range(0, rec_param_gd.shape[0]))
      ax2.set_xticklabels(range(1, rec_param_gd.shape[0]+1))
      fig.colorbar(pos, ax=ax2, shrink=0.6)
      fig.suptitle("Recurrent weight")
      fig.savefig('fig/'+args.regression+"/"+str(args.n_layers)+"/recurrent_weights.svg")

      fig, (ax1, ax2) = pl.subplots(figsize=(8, 5), ncols=2)
          
      vmin = min(np.min(D), np.min(D_gd))
      vmax = max(np.max(D), np.max(D_gd))
          
      pos = ax1.imshow(np.abs(D[:,None]), cmap='RdBu', vmin=vmin, vmax=vmax)
      ax1.set_title('Trained')
      ax1.set_yticks([0,1,2])
      ax1.set_yticklabels([1,2,3])
      ax1.set_xticks(ticks=[0])
      ax1.set_xticklabels([1])
      #fig.colorbar(pos, ax=ax1, shrink=0.6)

      pos = ax2.imshow(np.abs(D_gd[:,None]), cmap='RdBu', vmin=vmin, vmax=vmax)
      ax2.set_title('Constructed')
      ax2.set_yticks([0,1,2])
      ax2.set_yticklabels([1,2,3])
      ax2.set_xticks(ticks=[0])
      ax2.set_xticklabels([1])
      fig.colorbar(pos, ax=ax2, shrink=0.6)
      fig.suptitle("Input skip weights")
      fig.savefig('fig/'+args.regression+"/"+str(args.n_layers)+"/D.svg")



  if args.n_layers ==3:
    # Layer 1
    w_q_1= training_state.params["layers_0"]["seq"]["w_q"] #Local self attention weight
    D_1= training_state.params["layers_0"]["seq"]["D"] # Skip connection weight
    C1 = training_state.params["layers_0"]["seq"]["C"]
    discretization = "zoh"
    step_rescale = 1.0
    lambda_bar_all = []
    Lambda_re = training_state.params["layers_0"]["seq"]["Lambda_re"]
    log_step = training_state.params["layers_0"]["seq"]["log_step"]
    for param_id in range(Lambda_re.shape[0]):
                step = step_rescale * np.exp(log_step[:, 0])
                # Discretize
                if discretization in ["zoh"]:
                    Lambda_bar, _ = discretize_zoh(Lambda_re[param_id,:], training_state.params["layers_0"]["seq"]["B"], step)
                else:
                    raise NotImplementedError("Discretization method {} not implemented".format(discretization))
                lambda_bar_all.append(Lambda_bar)
    Lambda_bar_1 = np.vstack(lambda_bar_all)     # Recurrent weight
    rec_param_1 = np.diag(np.mean(Lambda_bar_1,axis=1))
    C_1 = training_state.params["layers_0"]["seq"]["C"]
    # Layer 2
    w_q_2= training_state.params["layers_1"]["seq"]["w_q"] #Local self attention weight
    D_2= training_state.params["layers_1"]["seq"]["D"] # Skip connection weight
    C_2 = training_state.params["layers_1"]["seq"]["C"]
    discretization = "zoh"
    step_rescale = 1.0
    lambda_bar_all = []
    Lambda_re = training_state.params["layers_1"]["seq"]["Lambda_re"]
    log_step = training_state.params["layers_1"]["seq"]["log_step"]
    for param_id in range(Lambda_re.shape[0]):
                step = step_rescale * np.exp(log_step[:, 0])
                # Discretize
                if discretization in ["zoh"]:
                    Lambda_bar, _ = discretize_zoh(Lambda_re[param_id,:], training_state.params["layers_1"]["seq"]["B"], step)
                else:
                    raise NotImplementedError("Discretization method {} not implemented".format(discretization))
                lambda_bar_all.append(Lambda_bar)
    Lambda_bar_2 = np.vstack(lambda_bar_all)     # Recurrent weight
    rec_param_2 = np.diag(np.mean(Lambda_bar_2,axis=1))
    C_2 = training_state.params["layers_1"]["seq"]["C"]
    # Layer 3
    w_q_3= training_state.params["layers_2"]["seq"]["w_q"] #Local self attention weight
    D_3= training_state.params["layers_2"]["seq"]["D"] # Skip connection weight
    C_3 = training_state.params["layers_2"]["seq"]["C"]
    discretization = "zoh"
    step_rescale = 1.0
    lambda_bar_all = []
    Lambda_re = training_state.params["layers_2"]["seq"]["Lambda_re"]
    log_step = training_state.params["layers_2"]["seq"]["log_step"]
    for param_id in range(Lambda_re.shape[0]):
                step = step_rescale * np.exp(log_step[:, 0])
                # Discretize
                if discretization in ["zoh"]:
                    Lambda_bar, _ = discretize_zoh(Lambda_re[param_id,:], training_state.params["layers_2"]["seq"]["B"], step)
                else:
                    raise NotImplementedError("Discretization method {} not implemented".format(discretization))
                lambda_bar_all.append(Lambda_bar)
    Lambda_bar_3 = np.vstack(lambda_bar_all)     # Recurrent weight
    rec_param_3 = np.diag(np.mean(Lambda_bar_3,axis=1))
    C_3 = training_state.params["layers_2"]["seq"]["C"]
    
    #GD-SSM based on gradient descent based construction
    w_q_gd_1 = jnp.outer(jnp.array([1.0,0,0]), jnp.array([0,1.0,0]))
    w_q_gd_2 = jnp.outer(jnp.array([1.0,0,0]), jnp.array([1.0,0,0]))
    w_q_gd_3 = jnp.outer(jnp.array([1.0,0,0]), jnp.array([0,1.0,0]))

    ## skipweight
    D_gd_3 = jnp.array([0, 0, 1])
            ## Recurrent weights
    Lambda_bar_gd_1 = jnp.ones((args.ssm_size_base,args.ssm_size_base))
    Lambda_bar_gd_2 = jnp.ones((args.ssm_size_base,args.ssm_size_base))
    Lambda_bar_gd_3 = jnp.ones((args.ssm_size_base,args.ssm_size_base))          
    ## Learning rate parameters
    C_gd1 = jnp.eye(args.ssm_size_base)*(-gd_lr/args.ssm_size_base)
    C_gd2 = jnp.eye(args.ssm_size_base)*(-gd_lr/args.ssm_size_base)
    C_gd3 = jnp.eye(args.ssm_size_base)*(-gd_lr/args.ssm_size_base)
    
    #Layer1
    #w_q_gd_1 = gd_state.params["layers_0"]["seq"]["w_q"]
    #Lambda_bar_gd_1 = gd_state.params["layers_0"]["seq"]["Lambda_re"]
    rec_param_gd_1 = np.diag(np.mean(Lambda_bar_gd_1,axis=1))
    #C_tilde_gd = (-gd_lr/10)*np.eye(10)

    #Layer 2
    # w_q_gd_2 = gd_state.params["layers_1"]["seq"]["w_q"]
    # Lambda_bar_gd_2 = np.ones((10,10))
    rec_param_gd_2 = np.diag(np.mean(Lambda_bar_gd_2,axis=1))
    #C_tilde_gd = (-gd_lr/10)*np.eye(10)

    #Layer 3
    #w_q_gd_3 = gd_state.params["layers_2"]["seq"]["w_q"]
    #Lambda_bar_gd_3 = np.ones((10,10))
    rec_param_gd_3 = np.diag(np.mean(Lambda_bar_gd_3,axis=1))
    #C_tilde_gd = (-gd_lr/10)*np.eye(10)
    #D_gd_3 = gd_state.params["layers_2"]["seq"]["D"]
    
    fig, (ax1, ax2) = pl.subplots(figsize=(15, 5), ncols=2)
        
    vmin = np.min(np.abs(w_q_1))
    vmax = np.max(np.abs(w_q_1))
        
    pos = ax1.imshow(np.abs(w_q_1), cmap='RdBu', vmin=vmin, vmax=vmax)
    ax1.set_title('Trained')
    ax1.set_yticks(ticks=range(0, w_q_1.shape[1]))
    ax1.set_yticklabels(range(1, w_q_1.shape[1]+1))
    ax1.set_xticks(ticks=range(0, w_q_1.shape[0]))
    ax1.set_xticklabels(range(1, w_q_1.shape[0]+1))
    fig.colorbar(pos, ax=ax1, shrink=0.6)

    pos = ax2.imshow(w_q_gd_1, cmap='RdBu', vmin=vmin, vmax=vmax)
    ax2.set_title('Constructed')
    ax2.set_yticks(ticks=range(0, w_q_gd_1.shape[1]))
    ax2.set_yticklabels(range(1, w_q_gd_1.shape[1]+1))
    ax2.set_xticks(ticks=range(0, w_q_gd_1.shape[0]))
    ax2.set_xticklabels(range(1, w_q_gd_1.shape[0]+1))
    fig.colorbar(pos, ax=ax2, shrink=0.6)
    fig.suptitle("Local self attention weight (Head 1)")
    if not os.path.isdir(os.path.join('fig', args.regression,str(args.n_layers))):
          os.makedirs(os.path.join('fig', args.regression,str(args.n_layers)))
    fig.savefig('fig/'+args.regression+"/"+str(args.n_layers)+"/lsa_weights_1.svg")

    #recurrent weights layer 1
    fig, (ax1, ax2) = pl.subplots(figsize=(15, 5), ncols=2)

    vmin = np.min(0.8)
    vmax = np.max(rec_param_1)

    pos = ax1.imshow(np.abs(rec_param_1), cmap='RdBu', vmin=0.98, vmax=vmax)
    ax1.set_title('Trained')
    ax1.set_yticks(ticks=range(0, rec_param_1.shape[1]))
    ax1.set_yticklabels(range(1, rec_param_1.shape[1]+1))
    ax1.set_xticks(ticks=range(0, rec_param_1.shape[0]))
    ax1.set_xticklabels(range(1, rec_param_1.shape[0]+1))

    pos = ax2.imshow(np.abs(rec_param_gd_1), cmap='RdBu', vmin=0.98, vmax=vmax)
    ax2.set_title('Constructed')
    ax2.set_yticks(ticks=range(0, rec_param_gd_1.shape[1]))
    ax2.set_yticklabels(range(1, rec_param_gd_1.shape[1]+1))
    ax2.set_xticks(ticks=range(0, rec_param_gd_1.shape[0]))
    ax2.set_xticklabels(range(1, rec_param_gd_1.shape[0]+1))
    fig.colorbar(pos, ax=ax2, shrink=0.6)
    fig.suptitle("Recurrent weight")
    fig.savefig('fig/'+args.regression+"/"+str(args.n_layers)+"/recurrent_weights_1.svg")

    #LSA weights layer2
    fig, (ax1, ax2) = pl.subplots(figsize=(15, 5), ncols=2)

    vmin = np.min(np.abs(w_q_2))
    vmax = np.max(np.abs(w_q_2))

    pos = ax1.imshow(np.abs(w_q_2), cmap='RdBu', vmin=vmin, vmax=1)
    ax1.set_title('Trained')
    ax1.set_yticks(ticks=range(0, w_q_2.shape[1]))
    ax1.set_yticklabels(range(1, w_q_2.shape[1]+1))
    ax1.set_xticks(ticks=range(0, w_q_2.shape[0]))
    ax1.set_xticklabels(range(1, w_q_2.shape[0]+1))
    #fig.colorbar(pos, ax=ax1, shrink=0.6)

    pos = ax2.imshow(w_q_gd_2, cmap='RdBu', vmin=vmin, vmax=1)
    ax2.set_title('Constructed')
    ax2.set_yticks(ticks=range(0, w_q_gd_1.shape[1]))
    ax2.set_yticklabels(range(1, w_q_gd_1.shape[1]+1))
    ax2.set_xticks(ticks=range(0, w_q_gd_1.shape[0]))
    ax2.set_xticklabels(range(1, w_q_gd_1.shape[0]+1))
    fig.colorbar(pos, ax=ax2, shrink=0.6)
    fig.suptitle("Local self attention weight")
    fig.savefig("fig/"+args.regression+"/"+str(args.n_layers)+"/lsa_weights_2.svg")
    
    #Recurrent weight head 2
    fig, (ax1, ax2) = pl.subplots(figsize=(15, 5), ncols=2)

    vmin = np.min(rec_param_2)
    vmax = np.max(rec_param_2)

    pos = ax1.imshow(np.abs(rec_param_2), cmap='RdBu', vmin=0.98, vmax=vmax)
    ax1.set_title('Trained')
    ax1.set_yticks(ticks=range(0, rec_param_2.shape[1]))
    ax1.set_yticklabels(range(1, rec_param_2.shape[1]+1))
    ax1.set_xticks(ticks=range(0, rec_param_2.shape[0]))
    ax1.set_xticklabels(range(1, rec_param_2.shape[0]+1))
    #fig.colorbar(pos, ax=ax1, shrink=0.6)

    pos = ax2.imshow(np.abs(rec_param_gd_2), cmap='RdBu', vmin=0.98, vmax=vmax)
    ax2.set_title('Constructed')
    ax2.set_yticks(ticks=range(0, rec_param_gd_2.shape[1]))
    ax2.set_yticklabels(range(1, rec_param_gd_2.shape[1]+1))
    ax2.set_xticks(ticks=range(0, rec_param_gd_2.shape[0]))
    ax2.set_xticklabels(range(1, rec_param_gd_2.shape[0]+1))
    fig.colorbar(pos, ax=ax2, shrink=0.6)
    fig.suptitle("Recurrent weight")
    fig.savefig("fig/"+args.regression+"/"+str(args.n_layers)+"/recurrent_weights_2.svg")
    
    # Local self attention layer 3
    fig, (ax1, ax2) = pl.subplots(figsize=(15, 5), ncols=2)

    vmin = np.min(w_q_3)
    vmax = np.max(w_q_3)

    pos = ax1.imshow(np.abs(w_q_3), cmap='RdBu', vmin=vmin, vmax=1)
    ax1.set_title('Trained')
    ax1.set_yticks(ticks=range(0, w_q_3.shape[1]))
    ax1.set_yticklabels(range(1, w_q_3.shape[1]+1))
    ax1.set_xticks(ticks=range(0, w_q_3.shape[0]))
    ax1.set_xticklabels(range(1, w_q_3.shape[0]+1))
    #fig.colorbar(pos, ax=ax1, shrink=0.6)

    pos = ax2.imshow(w_q_gd_3, cmap='RdBu', vmin=vmin, vmax=1)
    ax2.set_title('Constructed')
    ax2.set_yticks(ticks=range(0, w_q_gd_1.shape[1]))
    ax2.set_yticklabels(range(1, w_q_gd_1.shape[1]+1))
    ax2.set_xticks(ticks=range(0, w_q_gd_1.shape[0]))
    ax2.set_xticklabels(range(1, w_q_gd_1.shape[0]+1))
    fig.colorbar(pos, ax=ax2, shrink=0.6)
    fig.suptitle("Local self attention weight")
    fig.savefig("fig/"+args.regression+"/"+str(args.n_layers)+"/lsa_weights_3.svg")
    
    #Recurrent weights layer 3
    fig, (ax1, ax2) = pl.subplots(figsize=(15, 5), ncols=2)

    vmin = np.min(rec_param_3)
    vmax = np.max(rec_param_3)

    pos = ax1.imshow(rec_param_3, cmap='RdBu', vmin=0.8, vmax=vmax)
    ax1.set_title('Trained')
    ax1.set_yticks(ticks=range(0, rec_param_3.shape[1]))
    ax1.set_yticklabels(range(1, rec_param_3.shape[1]+1))
    ax1.set_xticks(ticks=range(0, rec_param_3.shape[0]))
    ax1.set_xticklabels(range(1, rec_param_3.shape[0]+1))
    #fig.colorbar(pos, ax=ax1, shrink=0.6)

    pos = ax2.imshow(rec_param_gd_3, cmap='RdBu', vmin=0.8, vmax=1)
    ax2.set_title('Constructed')
    ax2.set_yticks(ticks=range(0, rec_param_gd_3.shape[1]))
    ax2.set_yticklabels(range(1, rec_param_gd_3.shape[1]+1))
    ax2.set_xticks(ticks=range(0, rec_param_gd_3.shape[0]))
    ax2.set_xticklabels(range(1, rec_param_gd_3.shape[0]+1))
    fig.colorbar(pos, ax=ax2, shrink=0.6)
    fig.suptitle("Recurrent weight")
    fig.savefig("fig/"+args.regression+"/"+str(args.n_layers)+"/recurrent_weights_3.svg")
    
    #Output skip weight
    fig, (ax1, ax2) = pl.subplots(figsize=(8, 5), ncols=2)
    vmin = np.min(D_3)
    vmax = np.max(D_3)

    pos = ax1.imshow((D_3[:,None]), cmap='RdBu', vmin=vmin, vmax=vmax)
    ax1.set_title('Trained')
    ax1.set_yticks([0,1,2])
    ax1.set_yticklabels([1,2,3])
    ax1.set_xticks(ticks=[0])
    ax1.set_xticklabels([1])
    #fig.colorbar(pos, ax=ax1, shrink=0.6)

    pos = ax2.imshow(np.abs(D_gd_3[:,None]), cmap='RdBu', vmin=vmin, vmax=vmax)
    ax2.set_title('Constructed')
    ax2.set_yticks([0,1,2])
    ax2.set_yticklabels([1,2,3])
    ax2.set_xticks(ticks=[0])
    ax2.set_xticklabels([1])
    fig.colorbar(pos, ax=ax2, shrink=0.6)
    fig.suptitle("Input skip weights")
    fig.savefig("fig/"+args.regression+"/"+str(args.n_layers)+"/D_2_step.svg")

