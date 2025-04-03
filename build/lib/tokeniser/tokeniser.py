import math
import json
import regex
from tqdm import tqdm
import secrets
import warnings
import numpy as np
import torch
import importlib.resources as pkg_resources
import tokeniser.data as data
from typing import Optional, Union, Dict, List, Tuple

class Tokeniser:
  def __init__(self, ln: str ='1b', token_ordered: bool = False, max_len: Optional[Union[int, str]] = None, max_token_id: Optional[Union[int, str]] = None, data: Optional[Dict[str, int]] = None, data_counts: Optional[Dict[str, int]] = None, map_counts_default: bool = False, arg_based_max_len_map: bool = False, unknown_token_marker: str = '<|unknown_token|>', verbose: int = 1) -> None:
    """
    Function header definition:
    ```def __init__(
                    self, 
                    ln: str ='1b', 
                    token_ordered: bool = False, 
                    max_len: Optional[Union[int, str]] = None, 
                    max_token_id: Optional[Union[int, str]] = None, 
                    data: Optional[Dict[str, int]] = None, 
                    data_counts: Optional[Dict[str, int]] = None, 
                    map_counts_default: bool = False, 
                    arg_based_max_len_map: bool = False, 
                    unknown_token_marker: str = '<|unknown_token|>',
                    verbose: int = 1) -> None :```

    `Note`: `Passing in` or `pass in` terms assumes that an argument is being passed in and is not `None`.
    `Note`: Assuming token_ids are 0 indexed.

    Arguments (optional):
        1) `ln` (str: default `'1b'`): 
        - `'0.5b'` to use the tokeniser with val data as vocab (with 0.5b tokens).
        - `'1b'` to use the tokeniser with both val and test data as vocab (with 0.5b tokens).
        - `Note`: The dataset being referred to is the `SlimPajama` dataset.

        2) `Token ordered` (bool: default `False`): 
        - `True` to use the tokeniser, where tokens are ordered by occurence counts.
        - `False` indicates that the tokeniser is not ordered by occurence counts.

        3) `max_len` (int, str [Optional]: default `None`): 
        - Maximum length of a token. internally defaults to 26 for ln = `'1b'` and 27 for ln = `'0.5b'`.
        - It can accept integer values, None, or `"auto"`. `"auto"` leads to auto compute of the maximum token length in the given data and data_counts.
        - It is recommended to not define `max_len` seperately if using either default `data` (token map) or default `data_counts` (token count map).
        - It is recommended to set it to `None` or `auto`.
        - It is recommended to set it to `None` (or `auto`) along with `arg_based_max_len_map` to `True` if `data` is being passed in and `data_counts` is not being passed in (is None) and `map_counts_default` is set to `True` (thus using the default token count map).
        - `auto` forces computation of the max from the token maps (would thus take some extra initialisation time, which is quite small for a normal LLM token vocab (<263k tokens)).

        4) `max_token_id` (int, str [Optional]: default `None`): 
        - Maximum token id. internally defaults to `131_071`
        - It can accept integer values, None, or `"auto"`. `"auto"` leads to auto compute of the maximum token id in the given data and data_counts.
        - It is over-ridden if either the default `data` (token map) is being used.
        - It is recommended to set it to `None` or `auto`.
        - If it is None and default `data_counts` is being used then it is set to `131_071`.
        - If it is None otherwise it defaults to `"auto"`.

        5) `data` (Dict[str, int] [Optional]: default `None`): 
        - Use an external token map (token: token_ids) for tokenisation by passing it as `data`.
        - Passing in `data` overrides the `ln` and `token_ordered` arguments for the default `data` assignment method.
        - If it is not None and `max_len` is None, then max_len is set to `"auto"` (does not happen if `arg_based_max_len_map` is set to True).
        - It is recommended to always pass in `data_counts` if `data` is being passed in.

        6) `data_counts` (Dict[str, int] [Optional]: default `None`): 
        - Use an external token count map (token: token_counts_or_occurences) for tokenisation by passing it as `data_counts`.
        - Passing in `data_counts` overrides the `ln` and `token_ordered` arguments for the default `data_counts` assignment method.
        - It is recommended to always pass in `data_counts` if `data` is being passed in.
        - If `data` is passed in and `data_counts` is not, then `data_counts` is by default set to 1 for all the tokens (unless `map_counts_default` is set to `True`).
        - It is also suggested to not pass in `data_counts` if `data` is not being passed in, basically relying for an external token count map for the same token map.

        7) `map_counts_default` (bool: default `False`): 
        - It is only used for the specific case when `data` is passed and `data_counts` is not. Otherwise it is of no use.
        - It is not recommended to be set `True` but provided, if needed in any case.
        - If set to True, then instead of it `data_counts`being mapped to 1, it is mapped to the default counts for the specified `ln` and `token_ordered`.

        8) `arg_based_max_len_map` (bool: default `False`): 
        - It is only used for the specific case when `data` is passed and `max_len` is None. Otherwise it is of no use.
        - It over-rides `max_len` being set to `auto` for `data` being not None.
        - It is recommended to set it to be `True` if `map_counts_default` is True.

        9) `unknown_token_marker` (str: default `<|unknown_token|>`):
        - The token used for denoting unknown tokens.

        10) `verbose` (int: default `1`):
        - 1 means all checks (print statements and warnings are shown).
        - 0 means only warnings are shown.
        - -1 means nothing is shown, not even warnings.
        - Only accepts 3 values (1, 0, -1).
    """
    if ln not in ('1b', '0.5b'):
      raise Exception("`ln` must be either '1b' or '0.5b'")
    
    if isinstance(max_len,str) and max_len != "auto":
      raise Exception("`max_len` must be either 'auto', None or an integer")
    
    if verbose not in (-1, 0, 1):
      raise Exception("`verbose` must be either -1, 0 or 1")
    
    # Precompile regex patterns (avoiding repeated compilations).
    self.non_word_pattern = regex.compile(r'[^\p{L}\p{N}_\s]+')
    self.unknown_token_marker = unknown_token_marker
    
    if data is None:
      self.max_token_id = 131_071
      if ln == '1b':
        if data_counts is None:
          self.data_counts = self.load_file("data_counts_1b")
        else:
          if verbose == 1:
            print("Not a warning, error or an exception, this does not affect execution. Just a check. 'data' is set to None but 'data_counts' is not None, indicating a different token count map usage for the same token map. It is suggested to use the original 'data_counts' (token counts map) for the original 'data' (token map).")
          self.data_counts = data_counts
        
        if max_len is not None and max_len != "auto":
          if verbose >= 0:
            warnings.warn("A warning, not an error, this does not affect execution. It is advised not to use a different max_len value with the default token map.")
          self.max_len = max_len
        else:
          self.max_len = 26

        if token_ordered:
          self.data  = self.load_file("data_ord_1b")
        else:
          self.data  = self.load_file("data_unord_1b")

      elif ln == '0.5b':
        if data_counts is None:
          self.data_counts = self.load_file("data_counts_0_5b")
        else:
          if verbose == 1:
            print("Not a warning, error or an exception, this does not affect execution. Just a check. 'data' is set to None but 'data_counts' is not None, indicating a different token count map usage for the same token map. It is suggested to use the original 'data_counts' (token counts map) for the original 'data' (token map).")
          self.data_counts = data_counts
        
        if max_len is not None and max_len != "auto":
          if verbose >= 0:
            warnings.warn("A warning, not an error, this does not affect execution. It is advised not to use a different max_len value with the default token map.")
          self.max_len = max_len
        else:
          self.max_len = 27
        
        if token_ordered:
          self.data  = self.load_file("data_ord_0_5b")
        else:
          self.data  = self.load_file("data_unord_0_5b")
    
    elif data is not None:
      if max_len != "auto" and max_len is not None:
        if verbose >= 0 and (data_counts is None and map_counts_default == True):
          warnings.warn("A warning, not an error, this does not affect execution. It is advised not to use a different max_len value with the default token counts map.")
        self.max_len = max_len
      elif max_len is None and arg_based_max_len_map == True:
        if verbose >= 0 and not (data_counts is None and map_counts_default == True):
          warnings.warn(f"A warning, not an error, this does not affect execution. Argument 'arg_based_max_len_map' is passed as 'True' even when 'map_counts_default' is {map_counts_default} and 'data_counts' is {data_counts}. This is valid but not recommended to be done. It is recommended to pass 'arg_based_max_len_map' as 'True' only when 'data_counts' is None and 'map_counts_default' is True, unless you absolutely know what you are doing.", UserWarning)
        self.max_len = 26 if ln == '1b' else 27
      elif max_len is None:
        self.max_len = "auto"
      
      if data_counts is None:
        if verbose == 1:
          print("Not a warning, error or an exception, this does not affect execution. Just a check. data_counts is None but data is not.")
        if map_counts_default:
          if max_token_id is None:
            self.max_token_id = 131_071
          else:
            self.max_token_id = max_token_id
          if verbose >= 0:
            warnings.warn("A warning, not an error, this does not affect execution. Argument 'map_counts_default' is passed as 'True' and 'data_counts' is 'None'. This is valid but not recommended to be done. It is recommended to pass 'map_counts_default' as 'False' always, unless you absolutely know what you are doing.", UserWarning)
          if ln == '1b':
            self.data_counts = self.load_file("data_counts_1b")
          elif ln == '0.5b':
            self.data_counts = self.load_file("data_counts_0_5b")
        else:
          if max_token_id is None:
            max_token_id = "auto"
          else:
            self.max_token_id = max_token_id
          self.data_counts = {token:1 for token in data}
      else:
        if max_token_id is None:
          max_token_id = "auto"
        else:
          self.max_token_id = max_token_id
        self.data_counts = data_counts

    if data is not None or data_counts is not None:
      for token in self.data:
        if token not in self.data_counts:
          raise Exception("Token in data not present in data_counts")
      
      for token in self.data_counts:
        if token not in self.data:
          raise Exception("Token in data_counts not present in data")

    if max_len == "auto":
      self.max_len = max(len(token) for token in self.data)
    
    if max_token_id == "auto":
      self.max_token_id = max(self.data[token] for token in self.data)

    if self.unknown_token_marker not in self.data:
      if verbose == 1:
        print(f"Unknown token marker is not present in the 'data' (token map). Adding it there. Setting its token id to {self.max_token_id + 1}, token count to 0 and increasing 'max_token_id' to {self.max_token_id + 1}")
      self.data[self.unknown_token_marker] = self.max_token_id + 1
      self.max_token_id += 1
      self.data_counts[self.unknown_token_marker] = 0
  
  def load_file(self, file_path: str) -> Dict[str, int]:
    """
    Load the desired file for the tokeniser maps.
    """
    file_path_map = {
        "data_ord_1b": "ordered_tokenizer_1b_val_test_data",
        "data_unord_1b": "unordered_tokenizer_1b_val_test_data",
        "data_counts_1b": "count_tokenizer_1b_val_test_data",
        "data_ord_0_5b": "ordered_tokenizer_0.5b_val_data",
        "data_unord_0_5b": "unordered_tokenizer_0.5b_val_data",
        "data_counts_0_5b": "count_tokenizer_0.5b_val_data",
    }

    file_name = file_path_map[file_path]+".json"
    with pkg_resources.open_text(data, file_name, encoding="utf-8") as file:
      json_file_dict = json.load(file)
    return json_file_dict

  def segment_word_dp(self, word: str) -> Tuple[int, int, int, List[str]]:
      """
      Bottom-up DP segmentation for a given word.
      dp[i] holds a tuple:
        (min_token_count, total_log, total_count, segmentation)
      where segmentation is a list of tokens forming word[:i].
      Only substrings that exist in 'data' are considered.
      """
      n = len(word)
      dp = [None] * (n + 1)
      dp[0] = (0, 0.0, 0, [])
      # Cache log values for token lengths 1..max_len
      log_cache = {i: math.log(i) for i in range(1, self.max_len + 1)}
      for i in range(1, n + 1):
          best = None
          # Only look back at most max_len characters.
          for j in range(max(0, i - self.max_len), i):
              token = word[j:i]
              if token in self.data:
                  if dp[j] is not None:
                      candidate = (
                          dp[j][0] + 1,
                          dp[j][1] + log_cache[len(token)],
                          dp[j][2] + self.data_counts[token],
                          dp[j][3] + [token]
                      )
                      if (best is None or 
                          candidate[0] < best[0] or 
                          (candidate[0] == best[0] and candidate[1] > best[1]) or 
                          (candidate[0] == best[0] and candidate[1] == best[1] and candidate[2] > best[2])):
                          best = candidate
          dp[i] = best
      return dp[n]

  def clean_token(self, tok: str) -> str:
      """
      Cleans a token by stripping any leading and trailing characters that
      are not in the vocabulary.
      (Only the outer characters are removed.)
      """
      start = 0
      while start < len(tok) and tok[start] not in self.data:
          start += 1
      end = len(tok)
      while end > start and tok[end - 1] not in self.data:
          end -= 1
      return tok[start:end]

  def tokenise(self, text: str) -> Tuple[List[str], int]:
      """
      Splits the text into tokens using your original character-by-character
      splitting logic and then applies DP segmentation on each token.
      If a token exists exactly in the vocabulary, it is used as-is.
      Otherwise, the token is cleaned (dropping extraneous outer characters)
      and then segmented using bottom-up DP.

      Function header definition:
      ```tokenise(self, text: str) -> Tuple[List[str], int]:```

      Arguments:
      text (str): The text to be tokenised.
      """
      words = []
      word = ""
      tokens = []
      n_toks = 0
      random_number = secrets.randbelow(1000000)  # between 0 and 999999
      
      # Use original splitting logic.
      for i in text:
          if i not in self.data:
            if len(word) > 0:
              words.append(word)
            words.append(f"{self.unknown_token_marker}_{random_number}")
            word = ""
          elif i == '\n':
              if word:
                  words.append(word)
              words.append('\n')
              word = ""
          elif i == '\t':
              if word:
                  words.append(word)
              words.append('\t')
              word = ""
          elif (i == ' ' or bool(self.non_word_pattern.match(i))) and word and (word[0] != ' ' and not bool(self.non_word_pattern.match(word[0]))):
              words.append(word)
              word = i
          elif i != ' ' and word and word[0] == ' ':
              words.append(word)
              word = i
          elif (not bool(self.non_word_pattern.match(i))) and word and bool(self.non_word_pattern.match(word[0])):
              words.append(word)
              word = i
          else:
              word += i
      if word:
          words.append(word)
      
      # Process each word/token.
      for token in tqdm(words, desc="Tokenising words"):
          # Preserve newline and tab tokens.
          if token in ("\n", "\t"):
              tokens.append(token)
              n_toks += 1
              continue
          if token == f"{self.unknown_token_marker}_{random_number}":
              tokens.append(self.unknown_token_marker)
              n_toks += 1
              continue
          # If token exists exactly in vocab, use it.
          if token in self.data:
              tokens.append(token)
              n_toks += 1
              continue
          # Clean token (remove extraneous outer characters).
          cleaned = self.clean_token(token)
          if not cleaned:
              continue
          if cleaned in self.data:
              tokens.append(cleaned)
              n_toks += 1
              continue
          # Otherwise, segment the cleaned token using DP.
          seg = self.segment_word_dp(cleaned)
          if seg is not None:
              tokens.extend(seg[3])
              n_toks += seg[0]
          else:
              # Fallback: add individual characters (if in vocab).
              for ch in cleaned:
                  if ch in self.data:
                      tokens.append(ch)
                      n_toks += 1
      return tokens, n_toks
  
  def token_ids(self, tokens: List[str]) -> List[int]:
    """ Get the token ids of a given token list.
    
    Function header definition:
    ```token_ids(self, tokens: List[str]) -> List[int]:```

    Arguments:
    tokens (List[str]): The list of tokens to be converted to token ids.
    """
    token_ids = []
    for token in tokens:
      if token not in self.data:
        if token == '<|unknown_token|>':
          token_ids.append(self.max_token_id+1)
        else:
          raise Exception(f"Token {token} not in token map")
      token_ids.append(self.data[token])
    return token_ids
  
  def token_map(self) -> Dict[str, int]:
    """ Get the token map copy. Non-copy version directly available with `self.data`.
    
    Function header definition:
    ```token_map(self) -> Dict[str, int]:```

    Function code:
    ```return self.data.copy()```
    """
    return self.data.copy()
  
  def token_count_map(self) -> Dict[str, int]:
    """ Get the token count map copy. Non-copy version directly available with `self.data_counts`.
    
    Function header definition:
    ```token_count_map(self) -> Dict[str, int]:```

    Function code:
    ```return self.data_counts.copy()```
    """
    return self.data.copy()
  
  def max_token_length(self) -> int:
    """ Get the maximum token length. Also accessible with self.max_len
    
    Function header definition:
    ```max_token_length(self) -> int:```

    Function code:
    ```return self.max_len```
    """
    return self.max_len
  
  def one_hot_tokens(self, token_ids: List[int], op: str = 'np') -> Union[np.ndarray, torch.Tensor]:
    """
    Get the one hot matrix of the tokens.

    Function header definition:
    ```one_hot_tokens(self, token_ids: List[int], op: str ='np') -> Union[np.ndarray, torch.Tensor]:```

    Arguments:
    - token_ids (List[int]): The list of token ids to be converted to one hot vector. Use  `self.token_ids()` function to get the token ids from the token list.
    - op (str): Sets the type for the one hot matrix. Accepts values 'np' or 'torch'. 'np' for numpy array and 'torch' for torch tensor.
    """

    # We assume token ids are zero-indexed so the vector length is self.max_token_id.
    vocab_size = self.max_token_id+1

    if op == 'np':
      # Create a one-hot encoded NumPy array using an identity matrix and indexing.
      one_hot = np.eye(vocab_size, dtype=bool)[token_ids]
      return one_hot

    elif op == 'torch':
      # Convert token_ids to a torch tensor.
      token_ids_tensor = torch.tensor(token_ids, dtype=torch.long)
      # Create a one-hot encoded tensor using torch.nn.functional.one_hot.
      one_hot = torch.eye(vocab_size, dtype=torch.bool)[token_ids_tensor]
      return one_hot

    else:
      raise ValueError("Invalid op argument. Use 'np' for numpy or 'torch' for torch.")
  
  def visualise_tokens(self, tokens: List[str]) -> None:
    """
    Visualise the tokens.

    Function header definition:
    ```visualise_tokens(self, tokens: List[str]) -> None:```

    Arguments:
    - tokens (List[str]): The list of tokens to be visualised.
    """
    for id, tok in enumerate(tokens):
      if id < len(tokens) - 1:
        print(f"'{tok}',",end="")
      else:
        print(f"'{tok}'") 
  
  def visualise_token_ids(self, token_ids: List[int]) -> None:
    """
    Visualise the tokens.

    Function header definition:
    ```visualise_tokens(self, tokens: List[str]) -> None:```

    Arguments:
    - token_ids (List[str]): The list of token_ids to be visualised.
    """
    for id, tok_id in enumerate(token_ids):
      if id < len(token_ids) - 1:
        if id == 0:
          print("[", end = "")
        elif id%20 == 0:
          print("\n",end="")
        print(f"{tok_id}, ",end="")
      else:
        print(f"{tok_id}]") 