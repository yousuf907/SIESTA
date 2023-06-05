import argparse
import torch
import json
import os
import numpy as np
from SIESTAModel import SIESTAModel
from imagenet_base_init import *
torch.multiprocessing.set_sharing_strategy('file_system')

def get_data_loader(images_dir, label_dir, split, min_class, max_class, batch_size=128, return_item_ix=False):
    data_loader = utils_imagenet.get_imagenet_data_loader(images_dir + '/' + split, label_dir, split, batch_size=batch_size, 
                                    shuffle=False, min_class=min_class, max_class=max_class, return_item_ix=return_item_ix)
    return data_loader

def streaming(args, siesta):
    counter = utils.Counter()
    start_time=time.time()
    num_sleep = 0
    rotation = 0 # to alternate/ control wake/sleep cycle
    sleep_freq = 2 # for wake/sleep cycle one after another repeatedly

    if args.resume_full_path is not None:
        # load in previous model to continue training
        latent_dict, rehearsal_ixs, class_id_to_item_ix_dict, opq, pq = siesta.resume(args.base_init_classes, 
                   args.resume_full_path)
        # validate performance from previous increment
        print('Previous model loaded...computing previous accuracy as sanity check...')
        init_test_loader = get_data_loader(args.images_dir, args.label_dir, 'val', args.min_class, args.base_init_classes,
                                      batch_size=args.batch_size) #0-100
        print('\nComputing base accuracies...')
        probas_base, true_base = siesta.predict(init_test_loader, opq, pq)
        top1_base, top5_base = utils.accuracy(probas_base, true_base, topk=(1, 5))
        print('\nREMIND acc on BASE-INIT classes: top1=%0.2f%% -- top5=%0.2f%%' % (top1_base, top5_base))
        counter.count = len(rehearsal_ixs)
    else:
        print('\nPerforming base initialization...')
        feat_data, label_data, item_ix_data = extract_base_init_features(args.images_dir, args.label_dir,
                                    args.extract_features_from, args.classifier_ckpt, args.base_arch, 
                                    args.base_init_classes, args.num_channels, args.spatial_feat_dim)

        pq, opq, latent_dict, rehearsal_ixs, class_id_to_item_ix_dict = fit_opq(feat_data, label_data, item_ix_data,
                args.num_channels, args.spatial_feat_dim, args.num_codebooks, args.codebook_size, counter=counter)
        print('Counter after pq init.', counter.count)

        ################
        ## INITIALIZING REMIND (fc IN CLASSIFIER_F) WITH BASE INIT CLASSES
        start_time_init = time.time()
        print('\nTraining classes {}-{}.'.format(args.min_class, args.base_init_classes))
        
        ##### Joint base-init training (ONlY APPLICABLE FOR SSL PRETRAINED MODEL) #####
        init_test_loader = get_data_loader(args.images_dir, args.label_dir, 'val', min_class=args.min_class,
                                              max_class=args.base_init_classes)
        siesta.joint_train_base(opq, pq, latent_dict, rehearsal_ixs, init_test_loader, args.save_dir, args.ckpt_file)
        ### save siesta model out
        siesta.save(args.base_init_classes, args.save_dir, rehearsal_ixs, latent_dict, class_id_to_item_ix_dict, opq, pq)
        ################
        
        ###INFERENCE ON BASE-INIT CLASSES
        print('\nComputing base accuracies...')
        probas_base, true_base = siesta.predict(init_test_loader, opq, pq)
        top1_base, top5_base = utils.accuracy(probas_base, true_base, topk=(1, 5))
        print('\nDNN Accuracy on BASE-INIT Classes: top1=%0.2f%% -- top5=%0.2f%%' % (top1_base, top5_base))
        print('\nRuntime for current iteration (in mins): %0.3f' % ((time.time() - start_time_init)/60))
    
    #last_class = 2*args.base_init_classes # 200
    last_class=1000
    sleep_tot = int((last_class - args.base_init_classes) / (2 * args.class_increment)) ## 18 for inc=25
    print("Total number of sleeps for ImageNet-1K:", sleep_tot)
    print('\nBeginning SIESTA Training...')
    new_class_list = []
    num_recent_stuff = 0
    val_acc_all=[]
    val_acc_old=[]
    val_acc_new=[]
    val_acc_all = np.append(val_acc_all, top1_base)
    val_acc_old = np.append(val_acc_old, top1_base)
    val_acc_new = np.append(val_acc_new, top1_base)
    val_acc_all5=[]
    val_acc_old5=[]
    val_acc_new5=[]
    val_acc_all5 = np.append(val_acc_all5, top5_base)
    val_acc_old5 = np.append(val_acc_old5, top5_base)
    val_acc_new5 = np.append(val_acc_new5, top5_base)
    ###
    dnn_acc_all=[]
    dnn_acc_old=[]
    dnn_acc_new=[]
    dnn_acc_all = np.append(dnn_acc_all, top1_base)
    dnn_acc_old = np.append(dnn_acc_old, top1_base)
    dnn_acc_new = np.append(dnn_acc_new, top1_base)
    dnn_acc_all5=[]
    dnn_acc_old5=[]
    dnn_acc_new5=[]
    dnn_acc_all5 = np.append(dnn_acc_all5, top5_base)
    dnn_acc_old5 = np.append(dnn_acc_old5, top5_base)
    dnn_acc_new5 = np.append(dnn_acc_new5, top5_base)
    num_iter= int(20017 * (64 / args.sleep_batch_size)) #135000 #50000 # 38000 #11000 # 7000 for 25 inc and 11000 for 50 inc
    #num_iter=20017
    print("Number of fixed iterations per sleep:", num_iter)
    Wt=0
    St0=0
    St=0

    for class_ix in range(args.streaming_min_class, args.streaming_max_class, args.class_increment):
        max_class = class_ix + args.class_increment # 100-125, 125-150, 150-175, 175-200
        start_time_itr = time.time()
        rotation += 1
        print('\nTraining classes {}-{}.'.format(class_ix, max_class))

        train_loader_curr = get_data_loader(args.images_dir, args.label_dir, 'train', class_ix, max_class,
                    batch_size=args.batch_size, return_item_ix=True) # 100-125, 125-150, 150-175, 175-200
        
        ### store current data in memory buffer        
        print('Number of stored samples BEFORE buffer update:', len(rehearsal_ixs))      
        latent_dict, rehearsal_ixs, class_id_to_item_ix_dict, recent_class_list =siesta.update_buffer(opq, pq, 
            train_loader_curr, latent_dict, rehearsal_ixs, class_id_to_item_ix_dict, counter) # 100-125, 125-150
        print('Counter AFTER buffer update.', counter.count)
        new_class_list = np.append(new_class_list, recent_class_list)
        num_recent_stuff = len(train_loader_curr.dataset)
        
        #### AWAKE PHASE ####
        if rotation < sleep_freq: # 1 < 2
            print('\nSTATUS -> AWAKE ...')
            start_time_wake = time.time()
            ### ~~~~~~~~ WAKE ONLINE LEARNING ~~~~~~~~~~ ###
            siesta.update_new_nodes(recent_class_list)

            test_time_wake = time.time()
            
            ## All & New Classes
            test_loader_all = get_data_loader(args.images_dir, args.label_dir, 'val', args.min_class, max_class,
                                    batch_size=args.batch_size) # 0-125
            test_loader_new = get_data_loader(args.images_dir, args.label_dir, 'val', class_ix, max_class,
                                    batch_size=args.batch_size) # 100-125
            test_loader_old = get_data_loader(args.images_dir, args.label_dir, 'val', args.min_class, class_ix,
                                    batch_size=args.batch_size) # 0-100
            
            ### DNN ###
            probas_dnn, true_dnn = siesta.predict(test_loader_all, opq, pq)
            top1_wake, top5_wake = utils.accuracy(probas_dnn, true_dnn, topk=(1, 5))
            print('\nWake Online Performance [ALL CLASSES]: top1=%0.2f%% -- top5=%0.2f%%' % (top1_wake, top5_wake))

            ## OLD CLASSES
            probas_old, true_old = siesta.predict(test_loader_old, opq, pq)
            top1_old, top5_old = utils.accuracy(probas_old, true_old, topk=(1, 5))
            print('Wake Online Performance [OLD CLASSES]: top1=%0.2f%% -- top5=%0.2f%%' % (top1_old, top5_old))

            ## NEW CLASSES
            probas_new, true_new = siesta.predict(test_loader_new, opq, pq)
            top1_new, top5_new = utils.accuracy(probas_new, true_new, topk=(1, 5))
            print('Wake Online Performance [NEW CLASSES]: top1=%0.2f%% -- top5=%0.2f%%' % (top1_new, top5_new))
            #exit()

            dnn_acc_all = np.append(dnn_acc_all, top1_wake)
            dnn_acc_old = np.append(dnn_acc_old, top1_old)
            dnn_acc_new = np.append(dnn_acc_new, top1_new)
            dnn_acc_all5 = np.append(dnn_acc_all5, top5_wake)
            dnn_acc_old5 = np.append(dnn_acc_old5, top5_old)
            dnn_acc_new5 = np.append(dnn_acc_new5, top5_new)

            Wt += time.time() - test_time_wake

            print('\nTime Spent in Recent Wake (in MINs): %0.3f' % ((time.time() - start_time_wake)/60))
            #exit()
        
        #/////////////////////////////////////////////// SLEEP PHASE /////////////////////////////////////////////
        elif rotation >= sleep_freq: # 2 >= 2
            start_time_sleep = time.time()
            rotation = 0
            num_sleep += 1
            print('\nSTATUS -> SLEEP... Number of Sleeps =', num_sleep)
            old_class_ix = max_class - len(new_class_list)

            ## All classes after sleep
            test_loader_all = get_data_loader(args.images_dir, args.label_dir, 'val', args.min_class, max_class,
                                    batch_size=args.batch_size) # 0-150
            test_loader_old = get_data_loader(args.images_dir, args.label_dir, 'val', args.min_class, old_class_ix,
                                    batch_size=args.batch_size) # 0-100
            test_loader_new = get_data_loader(args.images_dir, args.label_dir, 'val', old_class_ix, max_class,
                                    batch_size=args.batch_size) # 100-150
            ### Feature init ###
            assert len(new_class_list) == 2 * args.class_increment

            ## online update
            siesta.update_new_nodes(recent_class_list)
            
            test_time_presleep = time.time()
            print("Here is Pre-Sleep Evaluations..")
            ### All Classes ###
            probas_dnn, true_dnn = siesta.predict(test_loader_all, opq, pq)
            top1_wake, top5_wake = utils.accuracy(probas_dnn, true_dnn, topk=(1, 5))
            print('\nPre-Sleep Online Performance [ALL CLASSES]: top1=%0.2f%% -- top5=%0.2f%%' % (top1_wake, top5_wake))

            ## OLD CLASSES
            probas_old, true_old = siesta.predict(test_loader_old, opq, pq)
            top1_old, top5_old = utils.accuracy(probas_old, true_old, topk=(1, 5))
            print('Pre-Sleep Online Performance [OLD CLASSES]: top1=%0.2f%% -- top5=%0.2f%%' % (top1_old, top5_old))
            old_class_list = np.unique(true_old)

            ## NEW CLASSES
            probas_new, true_new = siesta.predict(test_loader_new, opq, pq)
            top1_new, top5_new = utils.accuracy(probas_new, true_new, topk=(1, 5))
            print('Pre-Sleep Online Performance [NEW CLASSES]: top1=%0.2f%% -- top5=%0.2f%%' % (top1_new, top5_new))

            val_acc_all = np.append(val_acc_all, top1_wake)
            val_acc_old = np.append(val_acc_old, top1_old)
            val_acc_new = np.append(val_acc_new, top1_new)
            val_acc_all5 = np.append(val_acc_all5, top5_wake)
            val_acc_old5 = np.append(val_acc_old5, top5_old)
            val_acc_new5 = np.append(val_acc_new5, top5_new)

            St0 += time.time() - test_time_presleep
            
            ### Train DNN
            #print("\nRehearsal Policy is Prototypical")
            print("\nRehearsal Policy is Uniform Balanced..")
            siesta.memory_consolidate(opq, pq, latent_dict, rehearsal_ixs, new_class_list, num_iter)

            #siesta.memory_consolidate_aug(opq, pq, latent_dict, rehearsal_ixs, new_class_list, num_iter)

            #print("\nRehearsal Policy is Min Margin")
            #siesta.memory_consolidate_margin(opq, pq, latent_dict, rehearsal_ixs, num_iter)

            #print("\nRehearsal Policy is Max Loss")
            #siesta.memory_consolidate_ml(opq, pq, latent_dict, rehearsal_ixs, num_iter)

            #print("\nRehearsal Policy is Min Replay")
            #siesta.memory_consolidate_minrep(opq, pq, latent_dict, rehearsal_ixs, new_class_list, num_iter)

            #print("\nRehearsal Policy is Max Interference")
            #siesta.memory_consolidate_mir(opq, pq, latent_dict, rehearsal_ixs, new_class_list, num_iter)

            test_time_sleep = time.time()
            print("Here is Post-Sleep Evaluations..")
            # All Seen Classes
            probas_all, true_all = siesta.predict(test_loader_all, opq, pq)
            top1_all_dnn, top5_all_dnn = utils.accuracy(probas_all, true_all, topk=(1, 5))
            print('\nPost Sleep Accuracy [ALL CLASSES]: top1=%0.2f%% -- top5=%0.2f%%' % (top1_all_dnn, top5_all_dnn))

            ## OLD Classes
            probas_old, true_old = siesta.predict(test_loader_old, opq, pq)
            top1_old_dnn, top5_old_dnn = utils.accuracy(probas_old, true_old, topk=(1, 5))
            print("DNN's Top1 accuracy on OLD classes: %1.2f" % top1_old_dnn)
            ### NEW Classes
            probas_new, true_new = siesta.predict(test_loader_new, opq, pq)
            top1_new_dnn, top5_new_dnn = utils.accuracy(probas_new, true_new, topk=(1, 5))
            print("DNN's Top1 accuracy on NEW classes: %1.2f" % top1_new_dnn)

            dnn_acc_all = np.append(dnn_acc_all, top1_all_dnn)
            dnn_acc_old = np.append(dnn_acc_old, top1_old_dnn)
            dnn_acc_new = np.append(dnn_acc_new, top1_new_dnn)

            dnn_acc_all5 = np.append(dnn_acc_all5, top5_all_dnn)
            dnn_acc_old5 = np.append(dnn_acc_old5, top5_old_dnn)
            dnn_acc_new5 = np.append(dnn_acc_new5, top5_new_dnn)

            St += time.time() - test_time_sleep

            ##### Reset New Class List #####
            new_class_list = []
            num_recent_stuff = 0
            ## save siesta model
            siesta.save(max_class, args.save_dir, rehearsal_ixs, latent_dict, class_id_to_item_ix_dict, opq, pq)
            ####### /// #######
            print('\nTime Spent in Recent Sleep (in MINs): %0.3f' % ((time.time() - start_time_sleep)/60))
            exit()
            
        #if max_class == last_class:
        #    break

    exp_dir=args.save_dir
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    #np.save(os.path.join(exp_dir, 'sleep_all_top5.npy'), all5)
    np.save(os.path.join(exp_dir, 'val_all_top1.npy'), dnn_acc_all)
    np.save(os.path.join(exp_dir, 'val_old_top1.npy'), dnn_acc_old)
    np.save(os.path.join(exp_dir, 'val_new_top1.npy'), dnn_acc_new)
    np.save(os.path.join(exp_dir, 'val_all_top5.npy'), dnn_acc_all5)
    np.save(os.path.join(exp_dir, 'val_old_top5.npy'), dnn_acc_old5)
    np.save(os.path.join(exp_dir, 'val_new_top5.npy'), dnn_acc_new5)

    np.save(os.path.join(exp_dir, 'pre_sleep_all_top1.npy'), val_acc_all)
    np.save(os.path.join(exp_dir, 'pre_sleep_old_top1.npy'), val_acc_old)
    np.save(os.path.join(exp_dir, 'pre_sleep_new_top1.npy'), val_acc_new)
    np.save(os.path.join(exp_dir, 'pre_sleep_all_top5.npy'), val_acc_all5)
    np.save(os.path.join(exp_dir, 'pre_sleep_old_top5.npy'), val_acc_old5)
    np.save(os.path.join(exp_dir, 'pre_sleep_new_top5.npy'), val_acc_new5)

    print('\nTotal test time for wake (in mins): %0.3f' % (Wt/60))
    print('Total test time for pre-sleep (in mins): %0.3f' % (St0/60))
    print('Total test time for post-sleep (in mins): %0.3f' % (St/60))

    print('\nRuntime for entire experiment (in mins): %0.3f' % ((time.time() - start_time)/60))

    t = (Wt + St0 + St) / 60
    tt = ((time.time() - start_time) / 60) - t
    print("Total run time without test time:", tt)

    ## /// END /// ##


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # directories and names
    parser.add_argument('--expt_name', type=str)  # name of the experiment
    parser.add_argument('--label_dir', type=str, default=None)  # directory for numpy label files
    parser.add_argument('--images_dir', type=str, default=None)  # directory for ImageNet train/val folders
    parser.add_argument('--save_dir', type=str, required=False)  # directory for saving results
    parser.add_argument('--resume_full_path', type=str, default=None)  # directory of previous model to load
    parser.add_argument('--ckpt_file', type=str, default='MobileNet_modified_100c.pth')

    # network parameters
    parser.add_argument('--base_arch', type=str, default='MobNetClassifyAfterLayer8')  # architecture for G
    parser.add_argument('--classifier', type=str, default='MobNet_StartAt_Layer8')  # architecture for F
    parser.add_argument('--classifier_ckpt', type=str, required=True)  # base-init ckpt ///// Pretrain Checkpoint /////
    parser.add_argument('--extract_features_from', type=str,
                        default='model.features.7')  # name of the layer to extract features              
    parser.add_argument('--num_channels', type=int, default=80)  # number of channels where features are extracted
    parser.add_argument('--spatial_feat_dim', type=int, default=14)  # spatial dimension of features being extracted
    parser.add_argument('--penul_feat_dim', type=int, default=1280)  # number of channels where features are extracted
    
    parser.add_argument('--weight_decay', type=float, default=1e-5)  # weight decay for network
    parser.add_argument('--batch_size', type=int, default=128)  # testing batch size

    # pq parameters
    parser.add_argument('--num_codebooks', type=int, default=8)
    parser.add_argument('--codebook_size', type=int, default=256)

    # replay buffer parameters
    parser.add_argument('--max_buffer_size', type=int, default=None)  # maximum number of samples in buffer
    # augmentation parameters
    parser.add_argument('--mixup_alpha', type=float, default=0.1)

    # streaming setup
    parser.add_argument('--num_classes', type=int, default=1000)  # total number of classes
    parser.add_argument('--min_class', type=int, default=0)  # overall minimum class
    parser.add_argument('--base_init_classes', type=int, default=100)  # number of base init classes
    parser.add_argument('--class_increment', type=int, default=100)  # how often to evaluate
    parser.add_argument('--streaming_min_class', type=int, default=100)  # class to begin stream training
    parser.add_argument('--streaming_max_class', type=int, default=1000)  # class to end stream training
    parser.add_argument('--sleep_batch_size', type=int, default=128)  # training batch size during sleep
    parser.add_argument('--sup_epoch', type=int, default=50)  # training epoch during sleep
    parser.add_argument('--init_lr', type=float, default=1.6)  # starting lr for offline training during sleep

    parser.add_argument('--step_size', type=int, default=15) # LR step size for base init supervised fine tune
    parser.add_argument('--lr_gamma', type=float, default=0.1) # LR decay for base init supervised fine tune

    # get arguments and print them out and make any necessary directories
    args = parser.parse_args()
    
    if args.save_dir is None:
        args.save_dir = 'imagenet_experiments/' + args.expt_name

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print("Arguments {}".format(json.dumps(vars(args), indent=4, sort_keys=True)))

    # make model and begin stream training
    siesta = SIESTAModel(num_classes=args.num_classes, classifier_G=args.base_arch,
                        extract_features_from=args.extract_features_from, classifier_F=args.classifier,
                        classifier_ckpt=args.classifier_ckpt,
                        weight_decay=args.weight_decay, lr_gamma=args.lr_gamma, step_size=args.step_size,
                        mixup_alpha=args.mixup_alpha, num_channels=args.num_channels, 
                        num_feats=args.spatial_feat_dim, penul_feat_dim=args.penul_feat_dim,
                        num_codebooks=args.num_codebooks, codebook_size=args.codebook_size,
                        max_buffer_size=args.max_buffer_size, sleep_batch_size=args.sleep_batch_size, 
                        sup_epoch=args.sup_epoch, init_lr=args.init_lr)
    streaming(args, siesta)