import argparse
import json
import logging
import os
import pickle
import random
import re
import sys

import faiss
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.data.processors.utils import (DataProcessor, InputExample,
                                                InputFeatures)

logger = logging.getLogger(__name__)


def set_seed(args):
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  #TODO multi gpu support
  # if args.n_gpu > 0:
  #   torch.cuda.manual_seed_all(args.seed)


def set_dump_path(args, output_dir=None, exp_name=None):
  if output_dir is None: output_dir = args.output_dir
  if exp_name is None: exp_name = args.exp_name
  chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
  while True:
    exp_id = ''.join(random.choice(chars) for _ in range(10))
    if not os.path.isdir(os.path.join(output_dir, exp_name, exp_id)):
      break
  args.exp_id = exp_id
  dump_path = os.path.join(output_dir, exp_name, exp_id)
  os.makedirs(dump_path)
  args.dump_path = dump_path


def init_exp(args):
  # dump parameters
  set_dump_path(args)
  pickle.dump(args, open(os.path.join(args.dump_path, 'params.pkl'), 'wb'))

  # get running command
  command = ["python", sys.argv[0]]
  for x in sys.argv[1:]:
    if x.startswith('--'):
      assert '"' not in x and "'" not in x
      command.append(x)
    else:
      assert "'" not in x
      if re.match('^[a-zA-Z0-9_]+$', x):
        command.append("%s" % x)
      else:
        command.append("'%s'" % x)
  command = ' '.join(command)
  args.command = command + ' --exp_id "%s"' % args.exp_id

  # check experiment name
  assert len(args.exp_name.strip()) > 0

  logging.basicConfig(
    format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt = '%m/%d/%Y %H:%M:%S',
    level = logging.INFO)
  logger = logging.getLogger(__name__)
  logger.info("\n".join(
    "%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
  logger.info("The experiment will be stored in %s\n" % args.dump_path)
  logger.info("Running command: %s" % command)
  logger.info("")


def load_and_cache_examples(args, langpair, lang, tokenizer, key="", prefix="tatoeba"):

  cache_dir = os.path.join(args.data_dir, "pequod_cache")
  os.makedirs(cache_dir, exist_ok=True)
  cache_filename = os.path.join(
    cache_dir, "cached_%s_%s_%s" % (langpair, lang, key))
  
  if os.path.exists(cache_filename) and not args.overwrite_cache:
    logger.info("Loading features from cached file %s" % cache_filename)
    features = torch.load(cache_filename)
  else:
    processer = TatoebaProcesser()
    logger.info("Creating features from dataset file at %s" % args.data_dir)
    examples = processer.get_examples(args.data_dir, langpair, lang, prefix)
    features = TatoebaProcesser.convert_examples_to_features(
      examples, tokenizer, args.max_seq_length, 0,
      pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],)
    logger.info("Saving features to cache file %s" % cache_filename)
    torch.save(features, cache_filename)
  
  all_input_ids = torch.tensor(
    [f.input_ids for f in features], dtype=torch.long)
  all_attention_mask = torch.tensor(
    [f.attention_mask for f in features], dtype=torch.long)
  all_token_type_ids = torch.tensor(
    [f.token_type_ids for f in features], dtype=torch.long)

  dataset = TensorDataset(
    all_input_ids, all_attention_mask, all_token_type_ids)

  return dataset


class TatoebaProcesser(DataProcessor):

  @classmethod
  def convert_examples_to_features(cls, examples, tokenizer, max_length, pad_token_segment_id, pad_token, mask_padding_with_zero=True):

    features = []
    for ex_index, example in enumerate(examples):
      inputs = tokenizer.encode_plus(
        example.text_a,
        None,
        add_special_tokens=True,
        max_length=max_length,
      )
      input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

      attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

      padding_length = max_length - len(input_ids)
      input_ids = input_ids + ([pad_token] * padding_length)
      attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
      token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
    
      assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
      assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
      assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)

      if ex_index < 3:
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
        logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
      
      features.append(InputFeatures(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        label=None,
      ))

    return features

  def get_examples(self, data_dir, langpair, lang, prefix="tatoeba"):
    examples = []
    fn = os.path.join(data_dir, "%s.%s.%s" % (prefix, langpair, lang))
    with open(fn) as fp:
      for i, line in enumerate(fp):
        line = line.strip()
        examples.append(InputExample(
          guid="%s-%s-%d" % (langpair, lang, i),
          text_a=line,
        ))
    return examples


def to_cuda(tup):
  return tuple(t.cuda() for t in tup)


class Evaluator(object):

  def __init__(self, args, model, tokenizer, **kwargs):
    self.args = args
    self.datasets = {}
    self.model = model
    self.tokenizer = tokenizer
  
  def _parse_batch(self, batch, has_label=True, **kwargs):
    _batch = to_cuda(batch)
    # _batch = batch
    ret = {"input_ids": _batch[0],
      "attention_mask": _batch[1],
      "token_type_ids": _batch[2] if self.args.model_type == "bert" else None,}
    if has_label: ret["labels"] = _batch[3]
    ret.update(**kwargs)
    return ret
  
  def run(self):
    raise NotImplementedError

  def get_dataset(self, *args, **kwargs):
    if args in self.datasets: return self.datasets[args]
    dataset = self.load_and_cache_examples(*args, **kwargs)
    self.datasets[args] = dataset
    return dataset
  
  def load_and_cache_examples(self, *args, **kwargs):
    raise NotImplementedError

  def get_dataloader(self, *args, **kwargs):
    logger.info("Getting dataloader - args: %s" % str(args))
    dataset = kwargs.pop("dataset", self.get_dataset(*args, **kwargs))
    dataloader = DataLoader(dataset, batch_size=self.args.eval_batch_size)
    return dataloader


def similarity_search(x, y, dim, normalize=False):
  num = x.shape[0]
  idx = faiss.IndexFlatL2(dim)
  if normalize:
    faiss.normalize_L2(x)
    faiss.normalize_L2(y)
  idx.add(x)
  scores, prediction = idx.search(y, 1)
  return prediction


class TatoebaEvaluator(Evaluator):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.model_langs = ["share_lang", "order"]
    self.proj_matrix_fast = kwargs.get("proj_matrix_fast", None)
    if self.proj_matrix_fast is not None:
      logger.info("proj_matrix_fast:" + str(self.proj_matrix_fast.size()))
      self.proj_matrix_fast = self.proj_matrix_fast[0].float().cuda()
    self.res = {}
    self.cache_key = kwargs.pop('cache_key', None)

  def get_mean_emb(self, layer_outputs, pool_mask):
    embs = (layer_outputs * pool_mask.unsqueeze(2).float()).sum(dim=1) / \
      pool_mask.sum(dim=1).view(-1, 1).float()
    return embs.cpu().numpy().astype(np.float32)
  
  def get_cxlm_emb(self, layer_outputs):
    if self.proj_matrix_fast is None:
      raise ValueError
    ret = torch.mm(layer_outputs[:,0,:], self.proj_matrix_fast)
    # ret = layer_outputs[:,0,:]
    return ret.cpu().numpy().astype(np.float32)
  
  def get_cls_emb(self, layer_outputs):
    return layer_outputs[:,0,:].cpu().numpy().astype(np.float32)

  def get_embeddings(self, batch, outputs, emb_type=None):
    if emb_type is None:
      emb_type = self.args.emb_type
    last_layer_outputs, first_token_outputs, all_layer_outputs = outputs

    if emb_type == "mean":
      ret = self.get_mean_emb(all_layer_outputs[self.args.mean_layer_id], batch["attention_mask"])
    elif emb_type == "cls":
      ret = self.get_cls_emb(all_layer_outputs[-1])
    elif emb_type == "cxlm":
      ret = self.get_cxlm_emb(all_layer_outputs[8])
    else: raise ValueError

    # ret = None
    del last_layer_outputs, first_token_outputs, all_layer_outputs
    torch.cuda.empty_cache()
    return ret
  
  def get_langpairs(self):
    args = self.args
    if args.data_prefix == "tatoeba":
      langs = ["ara", "bul", "deu", "ell", "spa", "fra", "hin", "rus", "swh", "tha", "tur", "urd", "vie", "cmn"]
      langpairs = ["%s-eng" % lang for lang in langs]
    elif args.data_prefix == "cxlm":
      langpairs = "ar-en bg-en de-en el-en en-es en-fr en-hi en-ru en-sw en-th en-tr en-ur en-vi en-zh".split()
    elif args.data_prefix == "tat15plus":
      args.data_prefix = "tatoeba"
      l15 = set(["ara", "bul", "deu", "ell", "spa", "fra", "hin", "rus", "swh", "tha", "tur", "urd", "vie", "cmn"])
      ld = {'ara':'ar', 'heb':'he', 'vie':'vi', 'ind':'id',
        'jav':'jv', 'tgl':'tl', 'eus':'eu', 'mal':'ml',
        'tel':'te', 'afr':'af', 'nld':'nl', 'deu':'de',
        'ell':'el', 'ben':'bn', 'hin':'hi', 'mar':'mr', 'urd':'ur',
        'tam':'ta', 'fra':'fr', 'ita':'it', 'por':'pt', 'spa':'es',
        'bul':'bg', 'rus':'ru', 'jpn':'ja', 'kat':'ka', 'kor':'ko',
        'tha':'th', 'swh':'sw', 'cmn':'zh', 'kaz':'kk', 'tur':'tr',
        'est':'et', 'fin':'fi', 'hun':'hu', 'pes':'fa'}
      langs = []
      for l in ld:
        if l in l15: continue
        langs.append(l)
      # langs = ["afr", "jpn", "kor", "kaz", "est", "fin", "hun", "pes"]
      langpairs = ["%s-eng" % lang for lang in langs]
    else: raise ValueError
    return langpairs
  
  def run(self):
    args = self.args
    self.model.eval()

    langpairs = self.get_langpairs()

    for langpair in langpairs:
      lang1, lang2 = langpair.split("-")
      logger.info("Eval langpair: %s" % langpair)
      dl1 = self.get_dataloader(langpair, lang1)
      dl2 = self.get_dataloader(langpair, lang2)

      all_emb1 = []
      all_emb2 = []
      for batch1, batch2 in zip(dl1, dl2):
        batch1 = self._parse_batch(batch1, has_label=False)
        batch2 = self._parse_batch(batch2, has_label=False)
        #forward
        with torch.no_grad():
          outputs1 = self.model(**batch1)
          all_emb1.append(self.get_embeddings(batch1, outputs1))
          outputs2 = self.model(**batch2)
          all_emb2.append(self.get_embeddings(batch2, outputs2))
      
      all_emb1 = np.concatenate(all_emb1)
      all_emb2 = np.concatenate(all_emb2)
      emb_sz = all_emb1.shape[-1]
      if args.reverse_eval:
        all_emb1, all_emb2 = all_emb2, all_emb1
      predictions = similarity_search(
        all_emb1, all_emb2, emb_sz, normalize=self.args.normalize)
      correct = tot = 0
      for i, pred in enumerate(predictions):
        if i == pred[0]: correct += 1
        tot += 1
      logger.info("langpair:%s acc:%.2f" % (langpair, 100*correct/tot))
      self.res[langpair] = 100*correct/tot

    output_fn = os.path.join(args.exp_results_dir, args.exp_name)
    if args.reverse_eval: output_fn += "-rev"
    with open(output_fn, "w") as fp:
      json.dump(self.res, fp)
      

  def load_and_cache_examples(self, langpair, lang, **kwargs):
    args = self.args
    if self.cache_key is None:
      cache_key = "%s-%s" % (args.model_key, args.model_type)
    else:
      cache_key = self.cache_key
    return load_and_cache_examples(
      args=args,
      langpair=langpair,
      lang=lang,
      tokenizer=self.tokenizer,
      key=cache_key,
      prefix=args.data_prefix,
    )


class GlueccIrEvaluator(TatoebaEvaluator):
  
  def get_langpairs(self):
    return ["zh-classical_zh"]
  
  def get_embeddings(self, batch, outputs, emb_type=None):
    if emb_type is None:
      emb_type = self.args.emb_type
    all_layer_outputs=outputs.hidden_states

    if emb_type == "mean":
      ret = self.get_mean_emb(all_layer_outputs[self.args.mean_layer_id], batch["attention_mask"])
    elif emb_type == "cls":
      ret = self.get_cls_emb(all_layer_outputs[-1])
    else: raise ValueError

    # ret = None
    del all_layer_outputs
    torch.cuda.empty_cache()
    return ret
  
  def _parse_batch(self, batch, has_label=True, **kwargs):
    _batch = to_cuda(batch)
    # _batch = batch
    ret = {"input_ids": _batch[0],
      "attention_mask": _batch[1],
      "token_type_ids": _batch[2] if self.args.model_type == "bert" else None,
      "output_hidden_states": True, }
    ret.update(**kwargs)
    return ret



def get_params():
  parser = argparse.ArgumentParser()

  # Required parameters
  parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
  parser.add_argument("--data_prefix", default="gluecc", type=str)
  parser.add_argument("--output_dir", default=None, type=str, required=True,
                      help="The output directory where the model predictions and checkpoints will be written.")
  parser.add_argument("--exp_name", default=None, type=str, required=True,
                      help="Experiment name.")
  parser.add_argument("--max_seq_length", default=256, type=int)
  parser.add_argument("--mean_layer_id", default=8, type=int)
  parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
  parser.add_argument("--reload", default="", type=str)
  parser.add_argument("--do_lower_case", action='store_true',
                      help="Set this flag if you are using an uncased model.")
  parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
  parser.add_argument("--eval_batch_size", default=128, type=int,
                      help="Batch size per GPU/CPU for evaluation.")
  parser.add_argument("--emb_type", default="mean", type=str)
  parser.add_argument("--normalize", action='store_true')
  parser.add_argument("--reverse_eval", action='store_true')
  parser.add_argument("--exp_results_dir", default="", type=str, required=True)
  parser.add_argument("--model_name", default="hfl/chinese-roberta-wwm-ext", type=str)


  return parser.parse_args()


def main():
  args = get_params()
  init_exp(args)
  set_seed(args)

#   args.model_name = "hfl/chinese-roberta-wwm-ext"
#   args.model_name = "ethanyt/guwenbert-base"


  args.device = torch.device('cuda')
  model = AutoModel.from_pretrained(args.model_name)
  config = AutoConfig.from_pretrained(args.model_name)
  args.model_type = config.model_type
  tokenizer = AutoTokenizer.from_pretrained(args.model_name)
  cache_key = args.model_name.replace("/", "-")
  model.to(args.device)

  GlueccIrEvaluator(args, model, tokenizer, cache_key=cache_key).run()

if __name__ == "__main__":
  main()

"""
python ./examples/retrieval/gluecc.py --data_dir ~/res/gluecc/ir --output_dir ./local-test --exp_name local-test --max_seq_length 512 --exp_results_dir ./local-res
"""
