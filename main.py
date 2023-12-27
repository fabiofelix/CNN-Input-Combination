import sys, os, argparse, random, glob, numpy as np, tensorflow as tf

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.backend.tensorflow_backend import set_session

from utils import get_expected, get_data_generator, clean_global_variables, save_results, get_steps, get_folds
import models

IMG_DIM = (256, 256)

def train_kfold_model(args, train_fold = "train"):
  files, labels, kf = get_folds(args.source_path, train_fold, args.path_aux)
  k = 0

  for train_index, val_index in kf.split(files, labels):    
    k += 1

    print("Fold {}: ".format(k))
    target = os.path.join(args.target_path, "fold" + str(k))

    if not os.path.isdir(target):
      os.mkdir(target)    

    train_filter = files[train_index]
    val_filter   = files[val_index]

    train_model(args, train_filter = train_filter, val_filter = val_filter)

def train_model(args, train_fold = "train", val_fold = "validation", train_csv = "train.csv", val_csv = "val.csv", train_filter = None, val_filter = None):
  hc_train_path = None if args.handcrafted_path is None else os.path.join(args.handcrafted_path, train_csv)
  hc_val_path = None if args.handcrafted_path is None else os.path.join(args.handcrafted_path, val_csv)

  train = get_data_generator(args.source_path, train_fold, args.batch_size, 
    path_aux = args.path_aux, 
    handcrafted_path = hc_train_path, 
    three_channel = args.comb in models.THREE_CHANNEL_TYPES, 
    three_input = args.comb in models.THREE_INPUT_TYPES,
    filter = train_filter)
  validation = get_data_generator(args.source_path, val_fold, args.batch_size, 
    path_aux = args.path_aux, 
    handcrafted_path = hc_val_path, 
    three_channel = args.comb in models.THREE_CHANNEL_TYPES, 
    three_input = args.comb in models.THREE_INPUT_TYPES,
    filter = val_filter)

  model = models.build_model(args.model, IMG_DIM, args.qt_label, comb_type = args.comb, quant = args.quant, show_summary = True)

  early  = EarlyStopping(monitor = "val_loss", min_delta = 0.0001, patience = 20, mode = "min", restore_best_weights = True, verbose = 1)  
  weight = ModelCheckpoint(os.path.join(args.target_path, 'weights.hdf5'), mode = "min", save_best_only = True, verbose = 1)

  model.fit_generator(train, 
                      epochs = args.qt_epoch, 
                      validation_data = validation, 
                      callbacks = [early, weight],
                      workers = 1,
                      max_queue_size = 15,
                      use_multiprocessing = False,
                      shuffle = False, 
                      verbose = 1,
                      steps_per_epoch  = get_steps(args.source_path, train_fold, args.path_aux, args.batch_size),
                      validation_steps = get_steps(args.source_path, val_fold,   args.path_aux, args.batch_size))  

def apply_model(args, test_fold = "test", test_csv = "test.csv"):
  hc_test_path = None if args.handcrafted_path is None else os.path.join(args.handcrafted_path, test_csv)
  generator = get_data_generator(args.source_path, test_fold, args.batch_size, 
    path_aux = args.path_aux, 
    handcrafted_path = hc_test_path, 
    three_channel = args.comb in models.THREE_CHANNEL_TYPES, 
    three_input = args.comb in models.THREE_INPUT_TYPES)

  model = models.my_load_model(args.target_path, args.model, IMG_DIM, args.qt_label, comb_type = args.comb, quant = args.quant, show_summary = True)

  predicted = model.predict_generator(generator, steps = get_steps(args.source_path, test_fold, args.path_aux, args.batch_size))
  save_results(get_expected(args.source_path, test_fold, args.path_aux), predicted, args.eval)
  

# https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
def options(args):
  if (os.path.isdir(args.source_path)):
    #======================SEED===============================#
    seed_value = 1030 #default

    random.seed(seed_value)
    np.random.seed(seed_value)

    tf.set_random_seed(seed_value)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(graph = tf.get_default_graph(), config = config))  
    #=========================================================#

    clean_global_variables()

    if args.action == "train":
      train_kfold_model(args)
    elif args.action == "apply":
      apply_model(args)
  else:
    raise Exception("Source path doesn't exist")  

def main(*args):
  parser = argparse.ArgumentParser(description="Audio classification")
  
  parser.add_argument("-a", help = "Action to be performed", dest = "action", choices = ["train", "apply"], required = True)
  parser.add_argument("-s", help = "Directory to load spectrogram images", dest = "source_path", required = True)     
  parser.add_argument("-m", help = "Model index (0) SimpleCNN, (1) BirdVox, (2) ResNet, (3) Inception ", dest = "model", required = True, 
                      type = int, choices=[ct.value for ct in models.Model_type])
  parser.add_argument("-l", help = "Amount of labels", dest = "qt_label", required = True, type = int)

  parser.add_argument("-t", help = "Directory to save/load model. Default = current directory", dest = "target_path", required = False)
  parser.add_argument("-b", help = "Batch size used for training, validation and test. Default = 80", dest = "batch_size", required = False, default = 80, 
                      type = int)
  parser.add_argument("-e", help = "Quantity of epochs. Default = 100", dest = "qt_epoch", required = False, default = 100, type = int)

  parser.add_argument("-quant", 
    help = """Add quantification loss function. 
              (1 or nothing) first weighting, 
              (2) second weighting, 
              (3) third weighting case.""", dest = "quant", default = None, const = 1, type = int, nargs = '?', choices=range(1, 4)) # um ou nenum argumento
  parser.add_argument("-comb", 
    help = """Especifies the type of input combination 
              (0) None,
              (1) Batch normalization layer,
              (2) Dense layer,
              (3) Batch normalization, dense layers,

              (10) Three inputs,
              (11) Three inputs + Batch normalization layer,
              (12) Three inputs + Dense layer,
              (13) Three inputs + Batch normalization, dense layers,
            
              (20) Three channels,
              (21) Three channels + Batch normalization layer,
              (22) Three channels + Dense layer,
              (23) Three channels + Batch normalization, dense layers""", dest = "comb", default = models.Combine_Type.NONE, required = False, type = int, choices=[ct.value for ct in models.Combine_Type])
  parser.add_argument("-hc", help = "Directory to load handcrafted features.", dest = "handcrafted_path", required = False)
  parser.add_argument("-aux", help = "Directory list to load three inputs/channels images. Concatenated with SOURCE_PATH", dest = "path_aux", required = False, 
                      default = None, nargs = 3)  #um ou mais argumentos  
  parser.add_argument("-eval", help = "Generate model evaluation", dest = "eval", default = False, action="store_true")  

  parser.set_defaults(func = options)
  ARGS = parser.parse_args()

  error_msg = ""

  if ARGS.comb in models.HAND_CRAFTED_TYPES and ARGS.handcrafted_path is None:
    error_msg += "When combining with handcrafted features define HANDCRAFTED_PATH"
  if (ARGS.comb in models.THREE_INPUT_TYPES or ARGS.comb in models.THREE_CHANNEL_TYPES) and ARGS.path_aux is None:    
    error_msg += ("" if error_msg == "" else "\n") + "When combining images in 3 inputs/channels define PATH_AUX (3 subdirs inside SOURCE_PATH)"
  if error_msg != "":
    parser.error(error_msg)

  ARGS.func(ARGS)  

if __name__ == '__main__':
  main(*sys.argv[1:])  

# PYTHONHASHSEED=1030 python main.py 
# -a train
# -s /home/sounddb/RioClaro_40000/amostra_12tags/amostra_audioset/amostra_completa/spec/mel_44k 
# -m 0 
# -l 15 

# -t /home/fabio/Documents/Pesquisa/Experimentos/resultados/RioClaro/novas_tags/amostra_12tags/amostra_audioset//amostra_completa/redes/44k/r50_mel/seed/1030/ 
# -comb 0
# -hc /home/sounddb/RioClaro_40000/amostra_12tags/amostra_audioset/amostra_completa/indices/44k/comb
# -aux 44k mel_44k pcen_44k
# -b 80 
# -e 100 
# -quant
# -eval


# PYTHONHASHSEED=1030 python main.py -l 15 -s /home/sounddb/RioClaro_40000/amostra_12tags/amostra_audioset/amostra_completa/spec/mel_44k -t /home/fabio/Documents/Pesquisa/Experimentos/resultados/RioClaro/novas_tags/amostra_12tags/INPUTS/teste/ -a train -m 0 

# PYTHONHASHSEED=1030 python main.py -l 15 -s /home/sounddb/RioClaro_40000/amostra_12tags/amostra_audioset/amostra_completa/spec/mel_44k/ -t /home/fabio/Documents/Pesquisa/Experimentos/resultados/RioClaro/novas_tags/amostra_12tags/INPUTS/teste/ -a apply -m 2 -b 30 -comb 1 -hc /home/sounddb/RioClaro_40000/amostra_12tags/amostra_audioset/amostra_completa/indices/44k/comb

# PYTHONHASHSEED=1030 python main.py -l 15 -s /home/sounddb/RioClaro_40000/amostra_12tags/amostra_audioset/amostra_completa/spec/ -aux 44k mel_44k pcen_44k  -t /home/fabio/Documents/Pesquisa/Experimentos/resultados/RioClaro/novas_tags/amostra_12tags/INPUTS/teste/ -a apply -m 0 -b 80 -comb 10 

# PYTHONHASHSEED=1030 python teste16_v2.py -f 44100 -t 60 -c 3 -lt 1 -e 100 -aug 580 -seed 1030 -rs 0 -l 15 -m 18 -b 30 -kfold 5 -p /home/sounddb/RioClaro_40000/amostra_12tags/amostra_audioset/amostra_completa/spec/mel_44k/ -o /home/fabio/Documents/Pesquisa/Experimentos/resultados/RioClaro/novas_tags/amostra_12tags/INPUTS/teste/ -as -a train

# PYTHONHASHSEED=1030 python main.py -l 15 -s /home/sounddb/RioClaro_40000/amostra_12tags/amostra_audioset/amostra_completa/spec/ -aux 44k mel_44k pcen_44k  -t /home/fabio/Documents/Pesquisa/Experimentos/resultados/RioClaro/novas_tags/amostra_12tags/INPUTS/teste/ -hc /home/sounddb/RioClaro_40000/amostra_12tags/amostra_audioset/amostra_completa/indices/44k/comb -a train -m 0 -b 80 -comb 22
