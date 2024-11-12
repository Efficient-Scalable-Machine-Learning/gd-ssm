import os
import sys
import wandb
from jax import random,vmap
from flax.training import checkpoints
from gd_ssm.model_init import model_init
from gd_ssm.train_helpers import linear_warmup, cosine_annealing, constant_lr, train_epoch,validate
from gd_ssm.analysis import analyse,scan_lrs,nfeature_analyse,display_learning,weight_visualisation
from gd_ssm.data import create_constructed_reg_data,create_reg_data_classic_token,\
    create_vec_reg_data_classic_token, create_reg_data_sin_classic
        
def train(args):
    """
    Main function to train over a certain number of epochs
    """
    if args.num_feature_analyse:
        nfeature_analyse(args)
        sys.exit()
    if args.USE_WANDB:
        # Make wandb config dictionary
        wandb.init(project=args.wandb_project, job_type='model_training', config=vars(args), entity=args.wandb_entity)
    # Set randomness...
    key = random.PRNGKey(args.jax_seed)
    init_rng, train_rng , data_rng,eval_rng= random.split(key, num=4)
    if args.regression == "non-linear":
            args.input_size = 1
    model_cls,state = model_init(args,init_rng,gd_params=False,gd_lr=None)
    in_dim = args.input_size # In-context regression data feature size
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
        data_creator =  vmap(create_vec_reg_data_classic_token,
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
    if args.n_layers ==1:
        if args.analyse or args.ood_analyse:
            gd_lr,min_loss,gd_model_cls,gd_state = scan_lrs(args,data_rng,False,10000)
            gd_model_cls,gd_state = model_init(args,init_rng,gd_params=True,gd_lr=gd_lr)
            print(f"Validation loss on the gradient-descent based construction:{min_loss} for the learning rate {gd_lr}")
            if args.USE_WANDB:
                wandb.log({"gd lr": gd_lr})
    else:
        if args.analyse:
            gd_lr,min_loss,gd_model_cls,gd_state = scan_lrs(args,data_rng,False,10000)
            print(f"Validation loss on the gradient-descent based construction:{min_loss} for the learning rate {gd_lr}")
    # Training Loop over epochs
    ls_trainloss, ls_valloss, loss_ssm_list, losses_gd_list, cos_sim_list, grad_norm_list, p_norm_list,training_steps= [],[],[],[],[],[],[],[]
    ir_t_list = [[]  for _ in range(args.epochs)]
    ws_t_list = [[]  for _ in range(args.epochs)]
    ir_gd_list = [[]  for _ in range(args.epochs)]
    ws_gd_list = [[]  for _ in range(args.epochs)]
    best_loss, best_acc, best_epoch = 100000000, -100000000.0, 0  # This best loss is val_loss
    count, best_val_loss = 0, 100000000  # This line is for early stopping purposes
    lr_count, opt_acc = 0, -100000000.0  # This line is for learning rate decay
    step = 0  # for per step learning rate decay
    train_size= args.bsz # FIXME - hardcode
    steps_per_epoch = int(train_size/args.bsz)
    #gd_model_cls,gd_state = model_cls,state
    #model_cls,state = gd_model_cls,gd_state
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

        # TODO: Switch to letting Optax handle this.
        #  Passing this around to manually handle per step learning rate decay.
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
        if (epoch<1000) & (args.n_layers !=1):
            # Train the constructed model a few more epoch
            if args.analyse:
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
                if (epoch+1) % 100 ==0:
                    print(f"\n=>> Epoch {epoch + 1} GD Metrics ===")
                    print(
                        f"\tTrain Loss: {gd_train_loss:.5f} -- Val Loss: {gd_val_loss:.5f}"
                        )
                if args.USE_WANDB:
                    wandb.log({"gd_train_loss": gd_train_loss, "gd_val_loss": gd_val_loss})
                if not os.path.isdir(os.path.join(args.dir_name, 'checkpoints_gd')): #save GD checkpoint
                    os.makedirs(os.path.join(args.dir_name, 'checkpoints_gd'))
                checkpoints.save_checkpoint(ckpt_dir=os.path.join(os.path.abspath(args.dir_name),'checkpoints_gd'), target=gd_state, step=0,overwrite=True)
            
        if args.analyse:
            if (epoch+1) % 1000 ==0:
                cos_sim, w_norm, p_norm = analyse(args.dataset,args.dataset_size,args.batchnorm,eval_data,state,model_cls, gd_model_cls,gd_state)
                cos_sim_list.append(cos_sim)
                grad_norm_list.append(w_norm)
                p_norm_list.append(p_norm)
                loss_ssm_list.append(train_loss)
                ls_trainloss.append(train_loss)
                ls_valloss.append(val_loss)
                losses_gd_list.append(min_loss)
                training_steps.append(epoch+1)
                if args.USE_WANDB:
                    wandb.log({"train_loss": train_loss, "val_loss": val_loss,"Model_cos":cos_sim,"Model_diff":w_norm,"Preds_diff":p_norm,"GD_loss":min_loss,"epoch":epoch})

        else:
            if args.USE_WANDB:
                wandb.log({"train_loss": train_loss, "val_loss": val_loss})

        if (epoch+1) % 100 ==0:
            print(f"\n=>> Epoch {epoch + 1} Metrics ===")
            print(
                f"\tTrain Loss: {train_loss:.5f} -- Val Loss: {val_loss:.5f}"
                )
        if args.ood_analyse:
            import numpy as np
            ### OOD Tasks - Analyse alignement between GD and trained SSM on OOD settings
            stretch = np.arange(0.5, 5+0.1, 0.1)
            stretch_i = np.arange(0.5, 2+0.03, 0.03)
            ir_t,ws_t, ir_gd, ws_gd = [],[],[],[]
            stretch = np.arange(0.5, 5+0.1, 0.1)
            stretch_i = np.arange(0.5, 2+0.03, 0.03)
            for ws in stretch:
                ood_data = data_creator(random.split(rng, num=args.bsz),
                        args.input_size,
                        args.dataset_size,
                        args.size_distract,
                        args.input_range,
                        ws)
                loss = validate(gd_state,
                            gd_model_cls,
                            ood_data,
                            seq_len,
                            in_dim,
                            args.batchnorm,
                            args.dataset)
                gd_loss = validate(state,
                            model_cls,
                            ood_data,
                            seq_len,
                            in_dim,
                            args.batchnorm,
                            args.dataset)
                ws_t.append(loss)
                ws_gd.append(gd_loss)
            for ir in stretch_i:
                ood_data = data_creator(random.split(rng, num=args.bsz),
                        args.input_size,
                        args.dataset_size,
                        args.size_distract,
                        ir,
                        args.weight_scale)
                loss = validate(gd_state,
                            gd_model_cls,
                            ood_data,
                            seq_len,
                            in_dim,
                            args.batchnorm,
                            args.dataset)
                gd_loss = validate(state,
                            model_cls,
                            ood_data,
                            seq_len,
                            in_dim,
                            args.batchnorm,
                            args.dataset)
                ir_t.append(loss)
                ir_gd.append(gd_loss)
            ir_t_list[epoch].append(np.array(ir_t))
            ws_t_list[epoch].append(ws_t)
            ir_gd_list[epoch].append(ir_gd)
            ws_gd_list[epoch].append(ws_gd)
 #save model checkpoint
    if not os.path.isdir(os.path.join(args.dir_name, 'checkpoints')):
        os.makedirs(os.path.join(args.dir_name, 'checkpoints'))
    checkpoints.save_checkpoint(ckpt_dir=os.path.join(os.path.abspath(args.dir_name),'checkpoints'), target=state, step=0,overwrite=True)    

### Analysis
    if args.analyse:
        import numpy as np
        weight_visualisation(args,data_rng,state,gd_lr)
        cosine_low = 0.0
        loss_ssm_list =[np.array(ls_valloss)]
        cos_sim_list = [np.array(cos_sim_list)]
        grad_norm_list = [np.array(grad_norm_list)]
        p_norm_list = [np.array(p_norm_list)]
        losses_gd_list = [np.array(losses_gd_list)]
        training_steps = np.array(training_steps)
        display_learning(training_steps,loss_ssm_list, test=[losses_gd_list[0]], y_lim_u=0.4, y_lim_l=0.2,
                            rw=1, title="train.pdf", allow_download=True,
                            single_seeds = True, label_title ="Loss",
                            title2='GD', title1='Trained SSM', 
                            title3='GD',  loc_first='upper right',
                            num_iter_os=len(loss_ssm_list[0]))

        display_learning(training_steps,cos_sim_list, grad_norm_list, p_norm_list, 
                            title1="Model cos",
                            title2="Model diff", y_lim_u=2,
                            title3="Preds diff", second_axis=True, color_add=0.2,
                            y_lim_u2=1.19, loc_sec='center right', single_seeds = False, 
                            y_lim_l2=cosine_low, color_axis=False, width= 5, y_label2 = 'Cosine sim',
                            rw=1, num_iter_os=len(cos_sim_list[0]), title="sim.pdf",
                            allow_download=True)
    else:
        weight_visualisation(args,data_rng,state)
                
    if args.ood_analyse:    
        ### Plot the result - #TODO - visualisations in another script
        from gd_ssm.analysis import display_ood_data
        display_ood_data(ir_gd_list,ir_t_list)

        



    