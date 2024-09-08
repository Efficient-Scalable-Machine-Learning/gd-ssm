import PIL.Image, PIL.ImageDraw, PIL.ImageFont
import jax
from jax import vmap
from jax import numpy as np
from s5.model_init import model_init
from s5.train_helpers import validate,get_prediction
from transformer.src.config import config
from transformer.src.data import create_reg_data_classic_token,create_vec_reg_data_classic_token
from IPython.display import Image, HTML, clear_output
import matplotlib.pylab as pl
import matplotlib.colors as mcolors
colors = pl.colormaps['Dark2'] 


def scan_lrs(args,rng, lin_diag=False, bs=10000):
    eval_rng,_ =  jax.random.split(rng, num=2)
    if args.dataset in ["normal_token_scalar"]:
        seq_len = args.dataset_size
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
    data = data_creator(jax.random.split(eval_rng, num=bs),
                      10,
                      args.dataset_size,
                      config.size_distract,
                      config.input_range, #FIXME : add to arg
                      config.weight_scale)
    lr_scan_range = np.arange(0.001, 25, 0.1)
    losses_lr = []
    for lr in lr_scan_range:
        gd_model_cls,gd_state = model_init(args,rng,gd_params=True,gd_lr=lr)
        val_loss = validate(gd_state,
                            gd_model_cls,
                            data,
                            seq_len,
                            10,
                            args.batchnorm,
                            args.dataset)
        losses_lr.append(val_loss)
    losses_lr = np.array(losses_lr)
    lr_min_i = np.argmin(losses_lr)
    min_loss = np.min(losses_lr)
    return lr_scan_range[lr_min_i], min_loss

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

def display_learning(train, test=None, gt=None, inter=None, title="train", 
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
    x_range = np.arange(0, num_iter_os, int(num_iter_os/len(test)))
    if len(test_list) > 1:
      if single_seeds:
        for s in test_list:
          ax1.plot(x_range, s, color=colors(0.1+color_add), alpha=0.2, label=title2,linewidth='2')
      else:
        ax1.fill_between(x_range, test-test_std, test+test_std ,alpha=0.2, facecolor=colors(0.1+color_add))
    ax1.plot(x_range, test, color=colors(0.1+color_add), label=title2,linewidth='3')
    #test_avg = moving_average(test, rw)
    #ax1.plot(x_range[:len(test_avg)], test_avg, color=colors(0.1+color_add), label=title2)
      
  if gt is not None:
    if not second_axis:
      x_range = np.arange(0, num_iter_os, int(num_iter_os/len(gt)))
      #ax1.plot(x_range[:len(gt[:-rw])], gt[:-rw], color=colors(0.2+color_add), alpha=0.3)
      #gt_avg = moving_average(gt, rw)
      ax1.plot(x_range, gt, color=colors(0.2+color_add), label=title3,linewidth='3')
      if len(gt_list) > 1:
        if single_seeds:
          for s in gt_list:
            ax1.plot(x_range, s, color=colors(0.2+color_add), alpha=0.2, linewidth='2', zorder=0)
        else:
          ax1.fill_between(x_range, gt-gt_std, gt+gt_std,alpha=0.2, facecolor=colors(0.2+color_add))
    else:
      x_range = np.arange(0, num_iter_os, int(num_iter_os/len(gt)))
      ax1.plot(x_range, gt, color=colors(0.6+color_add), label=title3,linewidth='3')
      if len(gt_list) > 1:
        if single_seeds:
          for s in gt_list:
            ax1.plot(x_range, s, color=colors(0.6+color_add), alpha=0.3, linewidth='2', zorder=0)
        else:
          ax1.fill_between(x_range, gt-gt_std, gt+gt_std ,alpha=0.2, facecolor=colors(0.6+color_add))

  if test is not None and second_axis:
    x_range = np.arange(0, num_iter_os, int(num_iter_os/len(test)))
    ax1.plot(x_range, test, color=colors(0.5+color_add), label=title2,linewidth='3')
    #test_avg = moving_average(test, rw)
    #ax1.plot(x_range[:len(test_avg)],test_avg, color=colors(0.5+color_add))
    if len(test_list) > 1:
      if single_seeds:
        for s in test_list:
          ax1.plot(x_range, s, color=colors(0.5+color_add), linewidth='2', alpha=0.3, zorder=0)
      else:
        ax1.fill_between(x_range, test-test_std, test+test_std ,alpha=0.2, facecolor=colors(0.5+color_add))

  if inter is not None and not second_axis:
    x_range = np.arange(0, num_iter_os, int(num_iter_os/len(inter)))
    ax1.plot(x_range, inter, color=colors(0.4+color_add), label=title4, linewidth='3', zorder=10)
    if len(inter_list) > 1:
      if single_seeds:
        for s in inter_list:
          ax1.plot(x_range, s, color=colors(0.4+color_add), alpha=0.3, linewidth='2', zorder=0)
      else:
        ax1.fill_between(x_range, inter-inter_std, inter+inter_std ,alpha=0.2, facecolor=colors(0.4+color_add), zorder=1)
    #inter_avg = moving_average(inter, rw)
    #ax1.plot(x_range[:len(inter_avg)], inter_avg, color=colors(0.7+color_add), label=title4)


  if second_axis:
    if ax2 is None:
      ax2 = ax1.twinx()
    ax2.set_zorder(0)
    ax1.set_zorder(1)
    ax1.set_frame_on(False)
    #train_avg = moving_average(train, rw)
    #ax2.plot(train[:-rw], color=colors(0.1+color_add), alpha=0.3)
    ax2.plot(x_range, train, color=colors(0.4+color_add), label=title1, linewidth='3')
    ax2.plot(x_range, np.ones_like(train), "--", color="gray", linewidth='0.7')
    if len(train_list) > 1:
      if single_seeds:
        for s in train_list:
          print(x_range, s)
          ax1.plot(x_range, s, line, color=colors(0.4+color_add), alpha=0.3, linewidth='2', zorder=0)
      else:
        ax2.fill_between(x_range, train-train_std, train+train_std ,alpha=0.2, facecolor=colors(0.4+color_add))

    if color_axis:
      ax2.yaxis.label.set_color(colors(0.4+color_add))
    else:
      legend2 = ax2.legend(loc='upper right', framealpha=0.99, facecolor='white')
      legend2.set_zorder(100)
    ax2.spines['top'].set_visible(False)
  else:
    #train_avg = moving_average(train, rw)
    if line != "-":
      ax1.scatter(x_range, train, s=[100 for _ in x_range], 
                  marker="+", color=colors(0.3+color_add), alpha=1, label=title1, zorder=3, linewidths=3)
    else:
      ax1.plot(x_range, train, line, color=colors(0.3+color_add), label=title1, linewidth='3', zorder=11)
    #ax1.plot(x_range[:len(train_avg)], train_avg, line, color=colors(0.3+color_add), label=title1)
    if len(train_list) > 1:
      if single_seeds:
          for s in train_list:
            ax1.plot(x_range, s, line, color=colors(0.3+color_add), alpha=0.3, linewidth='2', zorder=0)
      else: 
        ax1.fill_between(x_range, train-train_std, train+train_std,
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
    
def analyse(args,data, state,model_cls, rng, gd_lr):
    
    # Trained Transformer
    if args.dataset in ["normal_token_scalar"]:
        seq_len = args.dataset_size
    elif args.dataset in ["normal_token_vector"]:
        seq_len = (args.dataset_size *2) + 1
    else:
        pass
    pred = lambda z: get_prediction(state,model_cls,z[None, ...],seq_len,10,args.batchnorm,args.dataset)
    predictions = vmap(pred)(data[0])
    grads = vmap(jax.grad(pred))(data[0])[:, -1, :-1]  #+ w_init
    grads_norm = np.linalg.norm(grads, axis=1)
    gd_model_cls,gd_state = model_init(args,rng,gd_params=True,gd_lr=gd_lr)
    
  # GD
    pred_c = lambda z: get_prediction(gd_state,gd_model_cls,z[None, ...],seq_len,10,args.batchnorm,args.dataset)
    grads_c = vmap(jax.grad(pred_c))(data[0])[:, -1, :-1] 
    predictions_c = vmap(pred_c)(data[0]) 
    grads_c_norm = np.linalg.norm(grads_c, axis=1)

    # Metrics
    dot_products = np.einsum('ij,ij->i', grads/(grads_norm[..., None] + 1e-8),
                                grads_c/(grads_c_norm[..., None]+ 1e-8))
    dot = np.mean(dot_products)
    norm = np.mean(np.linalg.norm(grads-grads_c, axis=1))
    pred_norm = np.mean(np.linalg.norm(predictions[..., None]-
                                        predictions_c[..., None], axis=1))
    return dot, norm, pred_norm