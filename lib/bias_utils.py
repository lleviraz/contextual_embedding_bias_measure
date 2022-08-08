import torch
import numpy as np
import pandas as pd
from typing import *
from pathlib import Path
from itertools import product
import matplotlib.pyplot as plt
from collections import defaultdict

class BiasUtils:
  '''
  A utility class for Bias calculation utilities based on lib/bias_calculator.py
  ...
  bias_utils = BiasUtils(model,processor)
  ...
  '''

  def __init__(self, model,processor):
      self.model = model
      self.model.eval() # Important! Disable dropout
      self.processor = processor

  def get_logits(self, sentence: str) -> np.ndarray:
      return self.model(self.processor.to_bert_model_input(sentence))[0, :, :].cpu().detach().numpy()

  def softmax(self, arr, axis=1):
      e = np.exp(arr)
      return e / e.sum(axis=axis, keepdims=True)

  def get_mask_fill_logits(self, sentence: str, words: Iterable[str],
                          use_last_mask=False, apply_softmax=True) -> Dict[str, float]:
      mask_i = self.processor.get_index(sentence, "[MASK]", last=use_last_mask, accept_wordpiece=True)
      logits = defaultdict(list)
      out_logits = self.get_logits(sentence)
      if apply_softmax: 
          out_logits = self.softmax(out_logits)
      return {w: out_logits[mask_i, self.processor.token_to_index(w, accept_wordpiece=True)] for w in words}

  def bias_score(self, sentence: str, gender_words: Iterable[Iterable[str]], 
                word: str, gender_comes_first=True) -> Dict[str, float]:
      """
      Input a sentence of the form "GGG is XXX"
      XXX is a placeholder for the target word
      GGG is a placeholder for the gendered words (the subject)
      We will predict the bias when filling in the gendered words and 
      filling in the target word.
      
      gender_comes_first: GGG comes before XXX
      """
      # e.g. probability of filling [MASK] with "he" vs. "she" when target is "programmer"
      mwords, fwords = gender_words
      all_words = mwords + fwords
      subject_fill_logits = self.get_mask_fill_logits(
          sentence.replace("XXX", word).replace("GGG", "[MASK]"), 
          all_words, use_last_mask=not gender_comes_first,
      )
      subject_fill_bias = np.log(sum(subject_fill_logits[mw] for mw in mwords)) - \
                          np.log(sum(subject_fill_logits[fw] for fw in fwords))
      # male words are simply more likely than female words
      # correct for this by masking the target word and measuring the prior probabilities
      subject_fill_prior_logits = self.get_mask_fill_logits(
          sentence.replace("XXX", "[MASK]").replace("GGG", "[MASK]"), 
          all_words, use_last_mask=gender_comes_first,
      )
      subject_fill_bias_prior_correction = \
              np.log(sum(subject_fill_prior_logits[mw] for mw in mwords)) - \
              np.log(sum(subject_fill_prior_logits[fw] for fw in fwords))
      
      return {
              "stimulus": word,
              "bias": subject_fill_bias,
              "prior_correction": subject_fill_bias_prior_correction,
              "bias_prior_corrected": subject_fill_bias - subject_fill_bias_prior_correction,
            }
  
  def get_word_vector(self,sentence: str, word: str):
      idx = self.processor.get_index(sentence, word, accept_wordpiece=True)
      outputs = None
      with torch.no_grad():
          sequence_output, _ = self.model.bert(self.processor.to_bert_model_input(sentence),
                                          output_all_encoded_layers=False)
          sequence_output.squeeze_(0)
      return sequence_output.detach().cpu().numpy()[idx]

  def cosine_similarity(self,x, y):
      return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

  def get_effect_size(self,df1, df2, k="bias_prior_corrected"):
      diff = (df1[k].mean() - df2[k].mean())
      std_ = pd.concat([df1, df2], axis=0)[k].std() + 1e-8
      return diff / std_
  
  def exact_mc_perm_test(self,xs, ys, nmc=100000):
      n, k = len(xs), 0
      diff = np.abs(np.mean(xs) - np.mean(ys))
      zs = np.concatenate([xs, ys])
      for j in range(nmc):
          np.random.shuffle(zs)
          k += diff < np.abs(np.mean(zs[:n]) - np.mean(zs[n:]))
      return k / nmc







