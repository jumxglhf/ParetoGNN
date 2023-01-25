from src.options import Options
import os
import torch
import dgl
from pathlib import Path
import numpy as np
import src.utils
from src.model import PretrainModule, BigModel, GCN
import json
import src.data
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import src.min_norm_solvers
from src.data_train import Graph_Dataset, Universal_Collator


def main(rank, world_size, opt):

    def train(model, optimizer, scheduler, step, opt, checkpoint_path):
        torch.manual_seed(opt.local_rank + opt.seed)
        tasks = opt.tasks
        collator = Universal_Collator(g, opt.use_saint, opt.per_gpu_batch_size, opt.device, tasks, \
                                            opt.minsg_der, opt.minsg_dfr, opt.batch_size_multiplier_minsg, opt.khop_minsg, \
                                            opt.khop_ming, opt.batch_size_multiplier_ming, \
                                            opt.sub_size, 0.15, 0.15, 0.15, \
                                            opt.lp_neg_ratio, \
                                            opt.decor_size, opt.decor_der, opt.decor_dfr, \
                                            0 if opt.dataset != 'products' else 200000, 0.5
                                            )
        dataset = Graph_Dataset(g)
        curr_losses = {}
        model.train()
        model.zero_grad()
        inner_step = 0
        if opt.not_use_pareto:
            logger.info('Not using Pareto MTL.')
        else:
            logger.info('Using Pareto MTL.')

        while step < opt.total_steps:
            dataloader = DataLoader(dataset=dataset, shuffle=False, prefetch_factor=1, persistent_workers=True, \
                                    collate_fn=collator, num_workers=opt.worker, pin_memory=True)

            for sample in dataloader:

                loss_data = {}
                grads = {}
                # -------------- Begin of Pareto Multi-Tasking Learning --------------
                if opt.not_use_pareto:
                    sol = {t:1. for t in tasks}
                else:
                    if 'p_link' in tasks:
                        sg, pos_u, pos_v, neg_u, neg_v = sample['p_link']
                        loss = model.p_link(sg, pos_u, pos_v, neg_u, neg_v)
                        grads['p_link'] = []
                        loss_data['p_link'] = loss.data
                        loss.backward()
                        for param in model.big_model.parameters():
                            if param.grad is not None:
                                grads['p_link'].append(param.grad.data.detach().cpu())
                        model.zero_grad()

                    if 'p_ming' in tasks:
                        ming_graph, ming_feat, ming_cor_feat = sample['p_ming']
                        loss = model.p_ming(ming_graph.to(opt.device), ming_feat.to(opt.device), ming_cor_feat.to(opt.device))
                        grads['p_ming'] = []
                        loss_data['p_ming'] = loss.data
                        loss.backward()
                        for param in model.big_model.parameters():
                            if param.grad is not None:
                                grads['p_ming'].append(param.grad.data.detach().cpu())
                        model.zero_grad()

                    if 'p_minsg' in tasks:
                        minsg_g1, minsg_f1, minsg_g2, minsg_f2 = sample['p_minsg']
                        loss = model.p_minsg(minsg_g1.to(opt.device), minsg_f1.to(opt.device),
                                            minsg_g2.to(opt.device), minsg_f2.to(opt.device), opt.temperature_minsg)
                        grads['p_minsg'] = []
                        loss_data['p_minsg'] = loss.data
                        loss.backward()
                        for param in model.big_model.parameters():
                            if param.grad is not None:
                                grads['p_minsg'].append(param.grad.data.detach().cpu())
                        model.zero_grad() 
                    
                    if 'p_decor' in tasks:
                        decor_g1, decor_g2 = sample['p_decor']
                        loss = model.p_decor(decor_g1.to(opt.device), decor_g2.to(opt.device), opt.decor_lamb)
                        grads['p_decor'] = []
                        loss_data['p_decor'] = loss.data
                        loss.backward()
                        for param in model.big_model.parameters():
                            if param.grad is not None:
                                grads['p_decor'].append(param.grad.data.detach().cpu())
                        model.zero_grad()

                    if 'p_recon' in tasks:
                        p_recon_g, p_recon_mask = sample['p_recon']
                        loss = model.p_recon(p_recon_g.to(opt.device), p_recon_mask)
                        grads['p_recon'] = []
                        loss_data['p_recon'] = loss.data
                        loss.backward()
                        for param in model.big_model.parameters():
                            if param.grad is not None:
                                grads['p_recon'].append(param.grad.data.detach().cpu())
                        model.zero_grad() 

                    if len(tasks) > 1:
                        gn = src.min_norm_solvers.gradient_normalizers(grads, loss_data, opt.grad_norm)
                        for t in loss_data:
                            for gr_i in range(len(grads[t])):
                                grads[t][gr_i] = grads[t][gr_i] / gn[t].to(grads[t][gr_i].device)
                        sol, _ = src.min_norm_solvers.MinNormSolver.find_min_norm_element_FW([grads[t] for t in tasks])
                        sol = {k:sol[i] for i, k in enumerate(tasks)}
                    else:
                        sol = {tasks[0]:1.}
                # -------------- End of Pareto Multi-Tasking Learning --------------
                model.zero_grad()
                train_loss = 0
                actual_loss = 0
                loss_dict = model(sample, opt)  

                for i, l in loss_dict.items():
                    train_loss += float(sol[i]) * l
                    actual_loss += l
                
                train_loss.backward()

                loss_dict['train_loss'] = actual_loss.detach()
                for k, v in sol.items():
                    loss_dict[k+'_weight'] = torch.tensor(float(v))
                    if k not in curr_losses:
                        curr_losses[k] = loss_dict[k].item()
                    else:
                        curr_losses[k] += loss_dict[k].item()
                if 'train_loss' not in curr_losses:
                    curr_losses['train_loss'] = loss_dict['train_loss']
                else:
                    curr_losses['train_loss'] += loss_dict['train_loss']
                if opt.wandb and opt.is_main:
                    wandb.log({k: v.item() for k, v in loss_dict.items()})

                inner_step += 1
                if inner_step == opt.accumulation_steps:
                    inner_step = 0
                    step += 1
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    if opt.is_main and step % opt.save_freq == 0:
                        if not opt.debug:
                            ckpt_name = f"step-{step}_lp" if opt.mask_edge else f"step-{step}_ssnc" 
                            src.utils.save(model, optimizer, scheduler, step,
                                            opt, checkpoint_path, ckpt_name)
                            model.eval()
                            with torch.no_grad():
                                if opt.no_self_loop:
                                    use_g = dgl.add_self_loop(g)
                                else:
                                    use_g = g
                                if opt.is_distributed:
                                    X = model.module.compute_representation(use_g.to(opt.device), g.ndata['feat'].to(opt.device))
                                else:
                                    X = model.compute_representation(use_g.to(opt.device), g.ndata['feat'].to(opt.device))
                                fp = opt.dataset+f'_{step}_lp' if opt.mask_edge else opt.dataset+f'_{step}_ssnc'
                                fp += str(opt.tasks)
                                fp += '_saint' if opt.use_saint else '_k-order'
                                fp += '_prelu_' if opt.use_prelu else '_relu'
                                fp += 'hid_dim_{}_'.format(str(opt.hid_dim))
                                fp += 'optim_{}_'.format(str(opt.optim))
                                fp += 'inter_dim_{}_'.format(str(opt.inter_dim))
                                fp += 'pred_dim_{}_'.format(str(opt.predictor_dim))
                                fp += 'lr_{}_'.format(str(opt.lr))
                                fp += 'decay_{}_'.format(str(opt.weight_decay))
                                fp += 'no_self_loop_' if opt.no_self_loop else ''
                                fp += opt.grad_norm + '_'
                                fp += 'no_pareto' if opt.not_use_pareto else ''
                                fp += '.pt'
                                fp = os.path.join(checkpoint_path, fp)
                                torch.save(X, fp)
                            model.train()
                        # evaluate(model, g, opt) # skipping the node evaluation for now 
                        log = f"{step} / {opt.total_steps} |"
                        log += f"train loss: {curr_losses['train_loss']/(opt.save_freq*opt.accumulation_steps):.3f} |"
                        for t in sample:
                            log += f"{t} loss: {curr_losses[t]/(opt.save_freq*opt.accumulation_steps):.3f} |"
                        log += f"lr: {scheduler.get_last_lr()[0]:.5f}"
                        logger.info(log)
                        for k in curr_losses:
                            curr_losses[k] = 0

                if step >= opt.total_steps:
                    step += 1
                    break

    if opt.is_distributed:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        opt.local_rank = rank
        if opt.local_rank == 0:
            opt.is_main = True
        else:
            opt.is_main = False
        opt.device = "cuda:{}".format(opt.local_rank)
        opt.world_size = world_size
        torch.distributed.init_process_group(backend="nccl", world_size=opt.world_size, rank=opt.local_rank)
        torch.cuda.set_device(opt.local_rank)
    else:
        opt.device = "cuda:0"
        opt.is_main = True
        opt.local_rank = 0
        opt.world_size = 1

    if opt.wandb and opt.is_main:
        import wandb
        name = '{}_{}_{}_{}_{}_{}_{}_{}'.format(opt.dataset, str(opt.tasks), str(opt.hid_dim), str(opt.n_layer), \
             str(opt.total_steps), 'saint' if opt.use_saint else 'k-order', \
             str(opt.lr), str(opt.weight_decay))
        if opt.mask_edge:
            name += '_{}'.format('mask_edge')
        wandb.init(project="ParetoGNN")
        wandb.config = opt
        
    np.random.seed(opt.seed+opt.local_rank)
    dgl.seed(opt.seed+opt.local_rank)
    torch.manual_seed(opt.seed+opt.local_rank)
    checkpoint_path = Path(opt.checkpoint_dir)/opt.name
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    logger = src.utils.init_logger(
        opt.is_main,
        opt.is_distributed, # is_distributed=
        checkpoint_path / 'run.log'
    )
    opt.checkpoint_path = checkpoint_path

    logger.info(f"Initializing Data..")

    g = src.data.load_data(opt.dataset, opt.pretrain_label_dir, opt.mask_edge, opt.tvt_addr, opt.split, hetero_graph_path=opt.hetero_graph_path)

    logger.info(f"Initializing Model..")
        
    node_module = GCN(g.ndata['feat'].shape[1], opt.hid_dim, opt.dropout, opt.norm, opt.use_prelu)

    bigM = BigModel(node_module, None, opt.inter_dim)
    ParetoGNN = PretrainModule(bigM, opt.predictor_dim).to(opt.device)
    ParetoGNN_config = {'input_dim':g.ndata['feat'].shape[1], 'hid_dim':opt.hid_dim, 
                'n_layer':len(opt.hid_dim), 'inter_dim':opt.inter_dim, 'dropout':opt.dropout}
    opt.ParetoGNN_config = ParetoGNN_config
    logger.info("ParetoGNN CONFIG: "+json.dumps(ParetoGNN_config, indent=2))
    model = ParetoGNN.to(opt.device)
    optimizer, scheduler = src.utils.set_optim(opt, model)
    step = 0

    if opt.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.local_rank], output_device=opt.local_rank, static_graph=True)

    logger.info("Start training")

    train(
        model,
        optimizer,
        scheduler,
        step,
        opt,
        checkpoint_path)


if __name__ == '__main__':
    options = Options()
    options.add_ParetoGNN_options()
    options.add_optim_options()
    opt = options.parse()
    world_size = opt.world_size
    if opt.is_distributed:
        mp.spawn(
            main,
            args=(world_size, opt),
            nprocs=world_size,
            start_method='spawn',
            join=True
        )
    else:
        main(0, 1, opt)