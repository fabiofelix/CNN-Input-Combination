import os, numpy as np, glob, pandas as pd, threading, pdb
from PIL import Image
from enum import IntEnum
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

class MyWheightType(IntEnum): 
  C1Q1 = 0
  C2Q1 = 1
  C1Q2 = 2

LOCK = None
IMGS_PREDICTED = []
LABEL_ENCODER  = None
ONEHOT_ENCODER = None

def clean_global_variables():
  global IMGS_PREDICTED, LABEL_ENCODER, ONEHOT_ENCODER, LOCK
  IMGS_PREDICTED = []
  LABEL_ENCODER  = None
  ONEHOT_ENCODER = None
  LOCK = threading.Lock()

def create_encoder(base):
  global LABEL_ENCODER, ONEHOT_ENCODER

  if LABEL_ENCODER is None or ONEHOT_ENCODER is None:
    LABEL_ENCODER = LabelEncoder()
    ONEHOT_ENCODER = OneHotEncoder(sparse = False, categories = "auto")

    data = LABEL_ENCODER.fit_transform(base)  
    data = data.reshape(len(data), 1)

    ONEHOT_ENCODER.fit_transform(data)  

def encode_labels(labels, base):
  global LABEL_ENCODER, ONEHOT_ENCODER

  # pdb.set_trace()  

  create_encoder(base)
  code = LABEL_ENCODER.transform(labels) 
  code = code.reshape(len(code), 1)
  code = ONEHOT_ENCODER.transform(code)

  return code[0] 

def decode_labels(codes):
  global LABEL_ENCODER, ONEHOT_ENCODER

  if LABEL_ENCODER is None or ONEHOT_ENCODER is None:
    base = ["basi_culi", "cycl_guja", "myio_leuc", "pita_sulp", "vire_chiv", "zono_cape",  #bird
            "aden_marm", "apla_leuc", "boan_albo", "dend_minu", "isch_guen", "phys_cuvi",  #anuran
            "animal", "human", "natural"],  #other
    create_encoder(base)

  codes  = ONEHOT_ENCODER.inverse_transform(codes)

  return LABEL_ENCODER.inverse_transform(codes.ravel())

def save_ids(id):
  LOCK.acquire()

  try:
    IMGS_PREDICTED.append(id)  
  finally:
    LOCK.release()        

def load_image(path):
  img = Image.open(path)
  img = np.asarray(img)

  if len(img.shape) == 3:
    img = img[:, :, 0] #just first channel

  return img.astype(np.float32) / 255.0 #never forget this line

def load_images(files, file_label, three_channel, three_input, path_aux = None):
  imgs   = []
  input2 = []
  input3 = []
  labels = []

  for f in files:
    save_ids(os.path.basename(f))
    img = load_image(f)

    if path_aux is not None:
      if three_channel:
        second_channel = load_image(f.replace(path_aux[0], path_aux[1]))
        third_channel  = load_image(f.replace(path_aux[0], path_aux[2]))
        img = np.dstack((img, second_channel, third_channel))
      if three_input:
        input2.append(load_image(f.replace(path_aux[0], path_aux[1])))  
        input3.append(load_image(f.replace(path_aux[0], path_aux[2])))  

    imgs.append(img)  

    label = file_label.loc[os.path.basename(f)]["label"]
    code = encode_labels([label], file_label["label"])
    labels.extend([code])

  imgs = np.array(imgs)

  if not three_channel:
    imgs = np.expand_dims(imgs, 3) #images with the needed shape
  if three_input:
    input2 = np.array(input2)
    input3 = np.array(input3)

    input2 = np.expand_dims(input2, 3) #images with the needed shape
    input3 = np.expand_dims(input3, 3) #images with the needed shape

    return [imgs, input2, input3], np.array(labels)

  return [imgs], np.array(labels)  

def load_handcrafted(files, handcrafted):
  label_column = 'class'

  if 'label' in handcrafted:      
    label_column = 'label'
  elif 'group' in handcrafted:      
    label_column = 'group'

  data = handcrafted.loc[files] 
  data.drop([label_column], axis = 'columns', inplace = True)
  
  return data.to_numpy()

def load_data(files, labels, path_aux, handcrafted, three_channel, three_input):
  data = load_images(files, labels, three_channel, three_input, path_aux)

  if handcrafted is not None:
    files_name = [ os.path.basename(f) for f in files  ]
    hc   = load_handcrafted(files_name, handcrafted)
    data[0].append(hc)

  return data  

def get_expected(main_path, path, path_aux):
  internal_path = os.path.join(main_path, "" if path_aux is None else path_aux[0], path)
  labels = glob.glob(os.path.join(internal_path, "*.csv"))

  if len(labels) > 0:
    return pd.read_csv(labels[0]) 
  else:
    raise Exception("There is no csv file with labels in {}".format(path))  

def get_steps(main_path, path, path_aux, batch_size, file_type = "png"):
  internal_path = os.path.join(main_path, "" if path_aux is None else path_aux[0], path)
  files    = glob.glob(os.path.join(internal_path, "*." + file_type))

  return int(np.ceil(len(files) / batch_size))  

def get_data_generator(main_path, path, batch_size, path_aux = None, handcrafted_path = None, three_channel = False, three_input = False, filter = None):
  internal_path = os.path.join(main_path, "" if path_aux is None else path_aux[0], path)
  imgs   = glob.glob(os.path.join(internal_path, "*.png"))
  labels = glob.glob(os.path.join(internal_path, "*.csv"))
  handcrafted = None

  if len(labels) > 0:
    labels = pd.read_csv(labels[0]) 
    labels.set_index("file", inplace = True)
  else:
    raise Exception("There is no csv file with labels in {}".format(internal_path))

  if handcrafted_path is not None:
    if os.path.isfile(handcrafted_path):
      handcrafted = pd.read_csv(handcrafted_path) 
      handcrafted.set_index("file", inplace = True)
    else:
      raise Exception("There is no {} file with handcrafted features".format(handcrafted_path)) 

  if filter is not None:
    imgs = [f for f in imgs if os.path.basename(f) in filter]

  # pdb.set_trace()
  # aux = load_data(imgs[0:80], labels, path_aux, handcrafted, three_channel, three_input)
  # pdb.set_trace()  
  # print("hello world")

  while True:
    start = 0
    end   = batch_size

    while start < len(imgs):
      yield load_data(imgs[start:end], labels, path_aux, handcrafted, three_channel, three_input)

      start += batch_size
      end   += batch_size

def get_folds(main_path, path, path_aux, folds = 5):
#  pdb.set_trace()
  internal_path = os.path.join(main_path, "" if path_aux is None else path_aux[0], path)
  imgs   = glob.glob(os.path.join(internal_path, "*.png"))
  labels = glob.glob(os.path.join(internal_path, "*.csv"))

  if len(labels) > 0:
    labels = pd.read_csv(labels[0]) 
  else:
    raise Exception("There is no csv file with labels in {}".format(internal_path))

  id_column = 'name' if 'name' in labels else 'file'
  label_column = 'class'

  if 'label' in labels:      
    label_column = 'label'
  elif 'group' in labels:      
    label_column = 'group'

  labels.set_index(id_column, inplace = True)   
  
  if len(imgs) > labels.shape[0]:
    raise Exception('Number [{}] of files [{}] is greater then the number [{}] of labels'.format( len(imgs), "png", labels.shape[0] ))

  imgs   = [os.path.basename(f) for f in imgs]
  labels = [labels.loc[f][label_column] for f in imgs]

  kf = StratifiedKFold(n_splits = folds)

  return np.array(imgs), np.array(labels), kf

def save_results(expected, predicted_codes, evaluate):
  files = []
  label_code = []

  for i, p in enumerate(predicted_codes):
    files.append(IMGS_PREDICTED[i])
    aux   = np.zeros(len(p))
    aux[np.argmax(p)] = 1
    label_code.append(aux)

  predicted = decode_labels(label_code)

  data = {"file": files, "predicted": predicted}
  data = pd.DataFrame(data)
  data.to_csv("predicted_labels.csv", index = False)

  if evaluate:
    expected = expected.sort_values(by = "file", ascending = False)["label"].to_numpy()
    predicted = data.sort_values(by = "file", ascending = False)["predicted"].to_numpy()

    file = open("model_metrics.txt", "w")

    try:
      file.write(classification_report(expected, predicted, zero_division = 0)) 
      file.write("\n\n")     
      file.write("Balanced accuracy: {:.2f}\n".format(balanced_accuracy_score(expected, predicted)))
    finally:
      file.close()          
