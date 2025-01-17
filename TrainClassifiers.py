def main(kwargs):

    print(kwargs)

    import numpy as np
    import TrainClassifiersBase as TCB
    import warnings
    warnings.filterwarnings("ignore")


    ########################################
    # Configuration
    ########################################

    default_params = {        

        "model_name" : "NONE",
        
        "suffix" : "",

        # IO
        "input_path"  : "data/",
        "name_train"  : "train_imgs.hdf5",
        "name_test"   : "test_imgs.hdf5",
        "name_val"    : "test_imgs.hdf5",
        "output_path" : "outputs/", 

        # False: Train; True: read weights file 
        "read_from_file" : False,
        
        "inputs" : "2d",

        "n_classes" : 5,

        "signal_branch" : "class_new",

        # Parameters for constituent approach
        "n_constit" : 15,

        "n_features" : 4,


        # Parameters for 2d convolutional architecture    
        "n_blocks"        : 1,    
        "n_conv_layers"   : 5,        
        "conv_nfeat"      : 19,
        "conv_size"       : 3,
        "conv_batchnorm"  : 0,
        "pool_size"       : 1,
        "n_dense_layers"  : 2,
        "n_dense_nodes"   : 300,
        "dense_batchnorm" : 0,

        "conv_dropout"    : 0.2,
        "block_dropout"   : 0.0,
        "dense_dropout"   : 0.0,

        "pool_type"       : "max",

        # Parameters for LorentzLayer
        "train_poly"                     : False,
        "train_offset"                   : "full",
        "train_metric"                   : True,
        "train_minmax"                   : True,
        "lola_filters"                   : 1,
        "use_angular_dr"                 : True,
        "n_lolas"                        : 1,
        "do_mult_p"                      : 0,
        "mult_p"                         : 0,
        "regularize_weight"              : False,
        "train_regularize_weight"        : False,
        "train_regularize_weight_target" : False,
        "take_diff"                      : True, 

        # Image pre-processing
        "cutoff"          : 0.0,
        "scale"           : 1.0,
        "rnd_scale"       : 0.0,
        "sum2"            : 0,

        # Common parameters
        "batch_size"        : 512,
        "lr"                : 0.14313348571934608,
        "decay"             : 0.,
        "momentum"          : 0.,            
        "nb_epoch"          : 5,
        "samples_per_epoch" : None, # later filled from input files
    }


    name = "w_"
    for k,v in kwargs.items():
        name += "{0}_{1}_".format(k,v)
    default_params["model_name"]=name        
        
    colors = ['black', 'red','blue','green','orange','green','magenta']

    ########################################
    # Read in parameters
    ########################################

    params = {}
    for param in default_params.keys():

        if param in kwargs.keys():
            cls = default_params[param].__class__
            value = cls(kwargs[param])
            params[param] = value
        else:
            params[param] = default_params[param]

    print("Parameters are:")
    for k,v in params.items():
        print("{0}={1}".format(k,v))

    tot_pool = params["pool_size"]**params["n_blocks"]
 
    if tot_pool > 0:
        if not ((19 % tot_pool == 0) and (tot_pool <= 19)):
            print("Total pool of {0} is too large. Exiting.".format(tot_pool))
            return 10.

    brs = [params["signal_branch"]]

    pixel_brs = []

    if params["n_features"] == 4:
        feat_list =  ["E","PX","PY","PZ"] 
    elif params["n_features"] == 5:
        feat_list =  ["E","PX","PY","PZ","C"] 


    if params["inputs"] == "2d":
        pixel_brs += ["c{0}".format(i) for i in range(19*19)]
    elif params["inputs"] == "constit_fcn":
        pixel_brs += ["{0}_{1}".format(feature,constit) for feature in feat_list for constit in range(params["n_constit"])]
    elif params["inputs"] == "constit_lola":
        pixel_brs += ["{0}_{1}".format(feature,constit) for feature in feat_list for constit in range(params["n_constit"])]

    # Reading H5FS
    infname_train = TCB.os.path.join(params["input_path"], params["name_train"])
    infname_test  = TCB.os.path.join(params["input_path"], params["name_test"])
    infname_val   = TCB.os.path.join(params["input_path"], params["name_val"])


    ########################################
    # H5FS: Count effective training samples
    ########################################

    store_train = TCB.pandas.HDFStore(infname_train)
    n_train_samples = int((store_train.get_storer('table').nrows/params["batch_size"]))*params["batch_size"]

    store_test = TCB.pandas.HDFStore(infname_test)
    n_test_samples = int((store_test.get_storer('table').nrows/params["batch_size"]))*params["batch_size"]

    store_val = TCB.pandas.HDFStore(infname_val)
    n_val_samples = int((store_val.get_storer('table').nrows/params["batch_size"]))*params["batch_size"]

    print("Total number of training samples = ", n_train_samples)
    print("Total number of testing samples = ", n_test_samples)
    print("Total number of valing samples = ", n_val_samples)
    params["samples_per_epoch"] = n_train_samples/1
    params["samples_per_epoch_test"] = n_test_samples/1
    params["samples_per_epoch_val"] = n_val_samples/1


    ########################################
    # Prepare data and scalers
    ########################################

    print(n_train_samples)

    datagen_train_pixel = TCB.datagen_batch_h5(brs+pixel_brs, infname_train, batch_size=params["batch_size"])
    datagen_test_pixel  = TCB.datagen_batch_h5(brs+pixel_brs, infname_test, batch_size=params["batch_size"])
    datagen_val_pixel  = TCB.datagen_batch_h5(brs+pixel_brs, infname_val, batch_size=params["batch_size"])

    if params["inputs"] == "2d":
        the_model     = TCB.Models.model_2d
        the_image_fun = TCB.Models.to_image_2d
    elif params["inputs"] == "constit_fcn":
        the_model     = TCB.Models.model_fcn
        the_image_fun = lambda x: TCB.Models.to_constit(x,params["n_constit"], params["n_features"])
    elif params["inputs"] == "constit_lola":
        the_model     = TCB.Models.model_lola
        the_image_fun = lambda x: TCB.Models.to_constit(x,params["n_constit"], params["n_features"])
    
    class_names = {}
    for i in range(params["n_classes"]):
        class_names[i] = "c{0}".format(i)

    classifiers = [
        TCB.Classifier(params["model_name"],
                   "keras",
                   params,
                   params["read_from_file"],
                   datagen_train_pixel,                   
                   datagen_test_pixel,               
                   datagen_val_pixel,               
                   the_model(params),
                   image_fun = the_image_fun,           
                   class_names = class_names,
                   inpath = "",
               )
    ]

    ########################################
    # Train/Load classifiers and make ROCs
    ########################################

    # Returns best val loss for keras training
    for clf in classifiers:
	vloss=[]
        for i in range(1):
            print '###### CV ITERATION ###### : '+str(i)
            clf.prepare()
        
            if params["suffix"]:
                ret = TCB.eval_single(clf, params["suffix"])
            else:
                ret = TCB.eval_single(clf)
                #print 'RET : '+str(i), ret
            vloss=np.append(vloss,ret)

        print 'LogLoss vector : ', vloss
        mloss=np.mean(vloss)
	sloss=np.std(vloss)
        print '<LogLoss> = ', mloss
	print 'STD = ', sloss

	f=open('tunings/logloss_result07.txt', 'ar')
	f.write(str(mloss)+'+/-'+str(sloss)+'\n')
	f.close()
        return mloss
