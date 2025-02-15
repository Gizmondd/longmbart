import os
import argparse
import random
import numpy as np
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
from transformers.optimization import get_linear_schedule_with_warmup, Adafactor
import nlp
from rouge_score import rouge_scorer
import sacrebleu

import pytorch_lightning as pl
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


from longformer import LongformerEncoderDecoderForConditionalGeneration, LongformerEncoderDecoderConfig
from longformer.sliding_chunks import pad_to_window_size

import logging
from transformers import MBartTokenizer
from transformers import MBartForConditionalGeneration
from transformers.models.mbart.modeling_mbart import shift_tokens_right
from longformer.longformer_encoder_decoder import LongformerSelfAttentionForBart
from longformer.longformer_encoder_decoder_mbart import MLongformerEncoderDecoderForConditionalGeneration, MLongformerEncoderDecoderConfig
import datasets
import collections

from . import simplification
from longformer.simplification import prepare_input, get_eval_scores

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SimplificationDatasetForInference(Dataset):
    def __init__(self, inputs, reference, name, tokenizer, max_input_len, max_output_len, src_lang, tgt_lang, tags_included, target_tags):
        self.inputs = inputs
        self.reference = reference
        self.name = name # train, val, test
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.tags_included = tags_included
        self.target_tags = target_tags

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        source = self.inputs[idx]['text']
        reference = None
        target_tags = None
        if self.reference is not None:
            reference = self.reference[idx]['text']
        if self.target_tags is not None:
            target_tags = self.target_tags[idx]['text']
        if self.tags_included is not None:
            sample = self.tokenizer.prepare_seq2seq_batch(src_texts=[source], tags_included=True , max_length=self.max_input_len, max_target_length=self.max_output_len, truncation=True, padding=False, return_tensors="pt")
        else:
            sample = self.tokenizer.prepare_seq2seq_batch(src_texts=[source], src_lang=self.src_lang, tgt_lang=self.tgt_lang , max_length=self.max_input_len, max_target_length=self.max_output_len, truncation=True, padding=False, return_tensors="pt") # TODO move this to _get_dataloader, preprocess everything at once?

        input_ids = sample['input_ids'].squeeze()
        input_ids = torch.cat([input_ids[1:], input_ids[:1]]) # move language tag to the end of the sequence in source
        return input_ids, reference, target_tags

    @staticmethod
    def collate_fn(batch):
        
        pad_token_id = 1
    
        input_ids, ref, target_tags = list(zip(*batch))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        return input_ids, ref, target_tags


class InferenceSimplifier(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.args = args
       
        self.src_lang = self.args.src_lang
        self.tgt_lang = self.args.tgt_lang
        self.tags_included = self.args.tags_included
        
        self.config = MLongformerEncoderDecoderConfig.from_pretrained(self.args.model_path)
        self.tokenizer = MBartTokenizer.from_pretrained(self.args.tokenizer, use_fast=True)
        
        self.max_input_len = self.args.max_input_len if self.args.max_input_len is not None else self.config.max_encoder_position_embeddings
        self.max_output_len = self.args.max_output_len if self.args.max_output_len is not None else self.config.max_decoder_position_embeddings 
        self.test_dataloader_object = None
    
    def test_step(self, batch, batch_nb):
        for p in self.model.parameters():
            p.requires_grad = False

        input_ids, ref, tags  = batch
        input_ids, attention_mask = prepare_input(input_ids, self.model, self.config.attention_mode, self.tokenizer.pad_token_id, self.args.global_attention_indices)

        if self.args.bad_words is not None:
            bad_words = [line.strip() for line in open(self.args.bad_words, 'r').readlines() if len(line.strip()) > 0]
            input_vocab = set([word for line in open(self.args.test_source, 'r') for word in line.strip().split()])
            bad_words = [bad_word for bad_word in bad_words if bad_word in input_vocab]
            if args.used_bad_words is not None:
                with open(self.args.used_bad_words, 'w') as bw_out: bw_out.write('\n'.join(bad_words))
            bad_words_bpe = [self.tokenizer.tokenize(bad_word) for bad_word in bad_words]
            bad_words_ids = [self.tokenizer.convert_tokens_to_ids(word) for word in bad_words_bpe]
            for i in range (len(bad_words_ids)):
                if len(bad_words_ids[i]) < 1: print(i, bad_words[i])
        else:
            bad_words_ids = None

        if self.tags_included:
            assert (ref[0] is not None or tags[0] is not None), "Need either reference with target labels or list of target labels with --tags_included (multilingual batches)"
            if  ref[0] is not None:
                tgt_ids = [self.tokenizer.lang_code_to_id[sample.split(' ')[0]]  for sample in ref ] # first token
            elif tags[0] is not None:
                # get decoder_start_token_ids from file in target_tags
                tgt_ids = [self.tokenizer.lang_code_to_id[sample.split(' ')[0]]  for sample in tags ]

            decoder_start_token_ids = torch.tensor(tgt_ids, dtype=input_ids.dtype, device=input_ids.device).unsqueeze(1)
            generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                            use_cache=True, max_length=self.args.max_output_len,
                                            num_beams=self.args.beam_size, pad_token_id=self.tokenizer.pad_token_id, decoder_start_token_ids=decoder_start_token_ids,
                                            do_sample=self.args.do_sample,
                                            temperature=self.args.temperature,
                                            top_k=self.args.top_k,
                                            top_p=self.args.top_p,
                                            repetition_penalty=self.args.repetition_penalty,
                                            length_penalty=self.args.length_penalty,
                                            num_return_sequences=self.args.num_return_sequences,
                                            output_scores=True if self.args.output_to_json else self.args.output_scores,
                                            return_dict_in_generate=True if self.args.output_to_json else self.args.return_dict_in_generate,
                                            bad_words_ids=bad_words_ids)
        else:
            generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                            use_cache=True, max_length=self.max_input_len,
                                            num_beams=self.args.beam_size, pad_token_id=self.tokenizer.pad_token_id, decoder_start_token_id=self.tokenizer.lang_code_to_id[self.tgt_lang],
                                            do_sample=self.args.do_sample,
                                            temperature=self.args.temperature,
                                            top_k=self.args.top_k,
                                            top_p=self.args.top_p,
                                            repetition_penalty=self.args.repetition_penalty,
                                            length_penalty=self.args.length_penalty,
                                            num_return_sequences=self.args.num_return_sequences,
                                            output_scores=True if self.args.output_to_json else self.args.output_scores,
                                            return_dict_in_generate=True if self.args.output_to_json else self.args.return_dict_in_generate,
                                            bad_words_ids=bad_words_ids)

        if not self.args.output_to_json:

            generated_strs = self.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=not self.args.keep_special_tokens)
            with open(self.args.translation, 'a') as f:
                for sample in generated_strs:
                    f.write(sample + "\n")
            
            if self.args.test_target is not None:
                return get_eval_scores(ref, generated_strs, self.tags_included)
            else:
                return {'decoded' : generated_strs}
        
        else:            
            # if running inference with self.args.batch_size
            # > 1, we need to make sure we pair the correct input sequence
            # with the correct returned hypotheses.
            batch_hyp_strs = self.tokenizer.batch_decode(generated_ids.sequences.tolist(), skip_special_tokens=not self.args.keep_special_tokens)
            batch_hyp_scores = generated_ids.sequences_scores.tolist()
            batch_source_strs = self.tokenizer.batch_decode(input_ids.tolist(), skip_special_tokens=not self.args.keep_special_tokens)

            generated_strs = []

            for batch_i in range(len(batch_source_strs)):
                src_str = batch_source_strs[batch_i]
                if self.args.test_target:
                    ref_str = ' '.join(ref[batch_i].split(' ')[1:]) if self.tags_included else ref[batch_i]
                else:
                    ref_str = None

                # subselect only those hyps/scores for the
                # relevant src string
                hyps = batch_hyp_strs[batch_i:batch_i+self.args.num_return_sequences]
                scores = batch_hyp_scores[batch_i:batch_i+self.args.num_return_sequences]

                output_dict = {
                    'src': src_str,
                    'ref': ref_str,
                    'hyps': [],
                    }
                # Ensure output hyps are sorted by
                # overall NLL probability scores (smaller = better).
                scored_hyps = {score: hyp for score, hyp in zip(scores, hyps)}
                for i, score in enumerate(sorted(scored_hyps.keys(), reverse=True)):
                    # add the 1-best hypothesis to generated_strs for evaluation             
                    if i == 0:
                        generated_strs.append(scored_hyps[score])
                    output_dict['hyps'].append({'score': score, 'hyp': scored_hyps[score]})
                
                json_line = json.dumps(output_dict, ensure_ascii=False)
                with open(self.args.translation, 'a') as f:
                    f.write(json_line+"\n")
        
            if self.args.test_target is not None:
                return self.get_eval_scores(ref, generated_strs, self.tags_included)

            else:
                return {'decoded' : generated_strs}


    def test_epoch_end(self, outputs):
        for p in self.model.parameters():
            p.requires_grad = True
    
        if self.args.test_target is not None:
            names = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'bleu']
            metrics = []
            for name in names:
                scores = [x[name] for x in outputs]
                metric = sum(scores)/len(scores)
                metrics.append(metric)
            logs = dict(zip(*[names, metrics]))
            print("Evaluation on provided reference [{}] ".format(self.args.test_target))
            print(logs)

    def forward(self):
        pass
    
    def _get_dataloader(self, current_dataloader, split_name, is_train):
        if current_dataloader is not None:
            return current_dataloader
        reference = None
        if self.args.test_target is not None:
            reference = self.datasets[split_name + "_target"]
        target_tags = None
        if self.args.target_tags is not None:
            target_tags = self.datasets["target_tags"]
        elif self.args.infer_target_tags:
            target_tags = self.datasets["target_tags"]
        dataset = SimplificationDatasetForInference(inputs=self.datasets[split_name + "_source"], reference=reference , name=split_name, tokenizer=self.tokenizer,
                                       max_input_len=self.max_input_len, max_output_len=self.max_output_len, src_lang=self.src_lang, tgt_lang=self.tgt_lang, tags_included=self.tags_included, target_tags=target_tags)
      
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=is_train) if self.trainer.use_ddp else None

        return DataLoader(dataset, batch_size=self.args.batch_size, shuffle=(sampler is None),
                          num_workers=self.args.num_workers, sampler=sampler,
                          collate_fn=SimplificationDatasetForInference.collate_fn)

    def test_dataloader(self):
        self.test_dataloader_object = self._get_dataloader(self.test_dataloader_object, 'test', is_train=False)
        return self.test_dataloader_object

    def configure_ddp(self, model, device_ids):
        model = LightningDistributedDataParallel(
            model,
            device_ids=device_ids,
            find_unused_parameters=False
        )
        return model


    @staticmethod
    def add_model_specific_args(parser, root_dir):
        parser.add_argument("--model_path", type=str, help="Path to the checkpoint directory or model name")
        parser.add_argument("--checkpoint_name", type=str, help="Checkpoint in model_path to use.")
        parser.add_argument("--tokenizer", type=str, help="Path to the tokenizer directory.")
        
        #data
        parser.add_argument("--test_source", type=str, default=None, help="Path to the source test file.")
        parser.add_argument("--test_target", type=str, default=None, help="Path to the target test file (optional, if given, will output rouge and bleu).")
        parser.add_argument("--target_tags", type=str, default=None, help="If test_target is not given: provide path to file with list of target tags (one per sample in test_source).")
        parser.add_argument("--src_lang", type=str, default=None, help="Source language tag (optional, for multilingual batches, preprocess text files to include language tags.")
        parser.add_argument("--tgt_lang", type=str, default=None, help="Target language tag (optional, for multilingual batches, preprocess text files to include language tags.")
        parser.add_argument("--tags_included", action='store_true', help="Text files already contain special tokens (language tags and </s>. Source:  src_tag seq, Target:  tgt_tag seq. Note: actual source sequence is seq src_tag </s>, will be changed internally after possibly clipping sequence to given max_length.")
        parser.add_argument("--infer_target_tags", action="store_true", default=False, help="If test_target is not given and target language tags can be inferred from the source language tags provided with --tags_included (e.g. de_DE -> de_DE). This save having a dedicated text file in which the tags are explicitly specified.")
        parser.add_argument("--max_input_len", type=int, default=256, help="maximum num of wordpieces, if unspecified, will use number of encoder positions from model config.")
        parser.add_argument("--max_output_len", type=int, default=512, help="maximum num of wordpieces, if unspecified, will use number of decoder positions from model config.")
        parser.add_argument("--bad_words", type=str, default=None, help="Path to file containing bad words.")
        parser.add_argument("--used_bad_words", type=str, default=None, help="Path to file where used bad words should be saved.")
        # TODO
        # parser.add_argument("--do_predict", action="store_true", default=False, help="If given, predictions are run using the `predict_step()` method rather than `test_step()`. Outputs are written to the specified output file without being evaluated!")
        
        parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
        parser.add_argument("--num_workers", type=int, default=0, help="Number of data loader workers")
        parser.add_argument("--gpus", type=int, default=-1, help="Number of gpus. 0 for CPU")
        
        ## inference params
        parser.add_argument("--translation", type=str, default='decoded.out', help="Output file to write decoded sequence to.")
        parser.add_argument("--beam_size", type=int, default=4, help="Beam size for inference when testing/validating. Default: 4.")
        parser.add_argument("--test_percent_check", default=1.00, type=float, help='Percent of test data used')
        parser.add_argument("--global_attention_indices", type=int, nargs='+', default=[-1], required=False, help="List of indices of positions with global attention for longformer attention. Supports negative indices (-1 == last non-padding token). Default: [-1] == last source token (==lang_id) .")
        parser.add_argument("--keep_special_tokens", default=False, action="store_true", help="If true, do not skip special tokens.")

        parser.add_argument("--output_to_json", default=False, action="store_true", help='If true, decoding output is a verbose JSONL containing, src, tgt, and scored model output hyps')
        
        # decoding strategy params (passed to model.generate() (in generation_utils.py))
        parser.add_argument("--do_sample", default=False, action="store_true", help='Whether or not to use sampling ; use greedy decoding otherwise.')
        parser.add_argument("--temperature", default=1.0, type=float, help='The value used to module the next token probabilities.')
        parser.add_argument("--top_k", default=50, type=int, help='The number of highest probability vocabulary tokens to keep for top-k-filtering.')
        parser.add_argument("--top_p", default=1.0, type=float, help='If set to float < 1, only the most probable tokens with probabilities that add up to :obj:`top_p` or higher are kept for generation.')
        parser.add_argument("--repetition_penalty", default=1.0, type=float, help='The parameter for repetition penalty. 1.0 means no penalty.')
        parser.add_argument("--length_penalty", default=1.0, type=float, help='Exponential penalty to the length. 1.0 means no penalty.')
        parser.add_argument("--output_scores", default=False, action="store_true", help='Whether or not to return the prediction scores.')
        parser.add_argument("--num_return_sequences", default=1, type=int, help='The number of independently computed returned sequences for each element in the batch, i.e. N-best')
        parser.add_argument("--return_dict_in_generate", default=False, action="store_true", help='Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.')
        
        #logging params
        parser.add_argument("--progress_bar_refresh_rate", type=int, default=0, help="How often to refresh progress bar (in steps). Value 0 disables progress bar.")
        parser.add_argument("--fp32", action='store_true', help="default is fp16. Use --fp32 to switch to fp32")
        parser.add_argument("--print_params", action='store_true', help="Print parameter names and shapes.")
        
        return parser


def main(args):

    if Path(args.translation).is_file():
        logging.info("Output file `{}` already exists and will be overwritten...".format(args.translation))
        Path(args.translation).unlink()

    checkpoint_path=os.path.join(args.model_path, args.checkpoint_name)
    simplifier = InferenceSimplifier(args)
    
    cp = torch.load(checkpoint_path)
    simplifier.model = MLongformerEncoderDecoderForConditionalGeneration.from_pretrained(args.model_path)
   
    simplifier.load_state_dict(cp["state_dict"])
    #simplifier.load_from_checkpoint(checkpoint_path, args) ## does not work ("unexpected keys")
     
    if args.print_params:
        for name, param in simplifier.named_parameters():
            if param.requires_grad:
                print(name + ":" + str(param.data.shape))
        exit(0)
    
    if args.test_target is not None:
        simplifier.datasets = datasets.load_dataset('text', data_files={'test_source': args.test_source, 'test_target': args.test_target })
    else:
        if args.tags_included and args.infer_target_tags:
            # source texts must start in with a single valid language tag,
            # e.g. de_DE, en_XX, etc.
            data_dict = datasets.load_dataset('text', data_files={'test_source': args.test_source})
            # datasets library allows loading from an
            # in-memory dict, so construct one from the source
            # text tags that can be loaded
            # NOTE: tags_included expects input sequences to
            # be prefixed with a single language tag, e.g. de_DE
            target_tags_dict = {'text': [text.split()[0] for text in data_dict['test_source']['text']]} 
            data_dict['target_tags'] = datasets.Dataset.from_dict(target_tags_dict)
            simplifier.datasets = data_dict
        elif args.target_tags is not None:
            simplifier.datasets = datasets.load_dataset('text', data_files={'test_source': args.test_source, 'target_tags': args.target_tags })
        else:
            simplifier.datasets = datasets.load_dataset('text', data_files={'test_source': args.test_source })

    logger = TestTubeLogger(
        save_dir=".",
        name="decode.log",
        version=0  # always use version=0
    )

    trainer = pl.Trainer(gpus=args.gpus, distributed_backend='ddp' if torch.cuda.is_available() else None,
                         replace_sampler_ddp=False,
                         limit_test_batches=args.test_percent_check,
                         logger=logger,
                         progress_bar_refresh_rate=args.progress_bar_refresh_rate,
                         precision=32 if args.fp32 else 16, amp_level='O2'
                         )
    
    trainer.test(simplifier)
    
    print("Decoded outputs written to {}".format(args.translation))
        

if __name__ == "__main__":
    main_arg_parser = argparse.ArgumentParser(description="simplification")
    parser = InferenceSimplifier.add_model_specific_args(main_arg_parser, os.getcwd())
    args = parser.parse_args()
    main(args)

