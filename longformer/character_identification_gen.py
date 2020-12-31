import os
import argparse
import random
import numpy as np
from tqdm import tqdm
import itertools
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
from transformers.optimization import get_linear_schedule_with_warmup, Adafactor
import nlp
from rouge_score import rouge_scorer
import logging
import pickle
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.logging import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel


from longformer import LongformerEncoderDecoderForConditionalGeneration, LongformerEncoderDecoderConfig
from longformer.sliding_chunks import pad_to_window_size

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)


# class EncoderDecoderTextDataset(Dataset):
#     def __init__(self, tokenizer, args, file_path, block_size=512):
#         assert os.path.isfile(file_path)
#         directory, filename = os.path.split(file_path)
#         model_name = args.model_path.split("/")[-1]
#         filename = f"{model_name}_cached_{block_size}_{filename}"
#         cached_features_file = os.path.join(directory, filename)
#
#         if os.path.exists(cached_features_file) and not args.overwrite_cache:
#             logger.info(f"Loading features from cached file {cached_features_file}")
#             with open(cached_features_file, "rb") as handle:
#                 self.examples = pickle.load(handle)
#         else:
#             logger.info("Converting to token IDs")
#             examples = load_data(file_path)
#             logger.info(examples[:5])
#
#             # Add prefix to the output so we can predict the first real token in the decoder
#             inputs = [
#                 tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ex[0]))
#                 for ex in examples
#             ]
#             outputs = [
#                 [inputs[i][-1]]
#                 + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ex[1]))
#                 for i, ex in enumerate(examples)
#             ]
#
#             # Pad
#             max_input_length = min(
#                 args.max_input_len, max([len(ex) for ex in inputs])
#             )
#             max_output_length = min(
#                 args.max_output_len, max([len(ex) for ex in outputs])
#             )
#
#             input_lengths = [min(len(ex), max_input_length) for ex in inputs]
#             output_lengths = [min(len(ex), max_output_length) for ex in outputs]
#
# #            print(len(inputs), inputs[0])
#
#             inputs = [tokenizer.encode(
#                 ex[0], add_special_tokens=False, max_length=max_input_length, truncation=True, pad_to_max_length=True)
#                 for ex in examples]
#
#             outputs = [tokenizer.encode(
#                 ex[1], add_special_tokens=False, max_length=max_output_length, truncation=True, pad_to_max_length=True)
#                 for ex in examples]
#
#             self.examples = {
#                 "inputs": inputs,
#                 "outputs": outputs,
#                 "input_lengths": input_lengths,
#                 "output_lengths": output_lengths,
#             }
#
#         logger.info(f"Saving features into cached file {cached_features_file}")
#         with open(cached_features_file, "wb") as handle:
#             pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
#     def __len__(self):
#         return len(self.examples["input_lengths"])
#
#     def __getitem__(self, item):
#         inputs = torch.tensor(self.examples["inputs"][item])
#         outputs = torch.tensor(self.examples["outputs"][item])
#
#         max_length = inputs.shape[0]
#         input_lengths = self.examples["input_lengths"][item]
#         input_mask = torch.tensor([1] * input_lengths + [0] * (max_length - input_lengths))
#
#         max_length = outputs.shape[0]
#         output_lengths = self.examples["output_lengths"][item]
#         output_mask = torch.tensor([1] * output_lengths + [0] * (max_length - output_lengths))
#
#         return {
#             "inputs": inputs,
#             "input_mask": input_mask,
#             "outputs": outputs,
#             "output_mask": output_mask,
#         }


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
        count = (~pad_mask).sum()
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
        count = nll_loss.numel()

    nll_loss = nll_loss.sum() / count
    smooth_loss = smooth_loss.sum() / count
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

def _truncate(process, data, max_length):
    source = []
    target = []
    for i in range(len(data)):
        if len(process(f"[sum] [desc] {data[i]['masked_description']} [name]")) > max_length:
            continue
        while True:
            input_seq = "[sum] " + data[i]['summary'] + " [desc] " + data[i]['masked_description']+ " [name]"
            ids = process(input_seq)
            if len(ids) <= max_length:
                source.append(input_seq)
                target.append(data[i]['character_name'] + " <eos>")
                break
            summ_ = lst1[i].split()
            data[i]['summary'] = " ".join(summ_[:-5])
        if i % 500 == 0:
            print(f"{i} processed from {len(data)}")
    return source, target

def encode_file(tokenizer, data_path, max_input_len, max_output_len, overwrite_cache=False, tok_name="", char_len=None):
    # cache_path = Path(f"{data_path}_{tok_name}{max_length}.pt")
    # if not overwrite_cache and cache_path.exists():
    #     try:
    #         examples = torch.load(str(cache_path))
    #         assert isinstance(examples, list)
    #         return examples
    #
    #     except Exception:
    #         print(f"failed to load from {cache_path}, retokenizing {data_path}")
    data_dir = Path(data_path)

    # lns = lmap(str.strip, data_dir.open().readlines())
    lns = [json.loads(line.strip()) for line in data_dir.open()]
    assert lns, f"found empty file at {data_dir}"

    # process = lambda s: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s))
    # src, trg = _truncate(process, lns, max_input_len)
    #
    # sources = []
    # for text in tqdm(src, desc=f"Tokenizing source in {data_dir.name}"):
    #     tokenized_src = tokenizer.encode(text, truncation=True, max_length=max_input_len)
    #     sources.append(tokenized_src)
    #
    # targets = []
    # for text in tqdm(trg, desc=f"Tokenizing target in {data_dir.name}"):
    #     tokenized_trg = tokenizer.encode(text, truncation=True, max_length=max_output_len)
    #     targets.append(tokenized_trg)

    sources = []
    targets = []
    for line in tqdm(lns, desc=f"Tokenizing {data_dir.name}"):
        source = f"[choices] {', '.join(line['multichoice']['choices'])} [desc] {trim_str(line['masked_description'], char_len)} " \
                 f"[sum] {line['coref_truncated_summary'] if 'coref_truncated_summary' in line else line['summary']} [name]"
        tokenized_src = tokenizer.encode(source, truncation=True, max_length=max_input_len)
        sources.append(tokenized_src)
        target = line["character_name"] + " <eos>"
        tokenized_trg = tokenizer.encode(target, truncation=True, max_length=max_output_len)
        targets.append(tokenized_trg)

    print(source, target)
    return sources, targets

def lmap(f, x):
    return list(map(f, x))

def flatten_list(summary_ids: List[List]):
    return [x for x in itertools.chain.from_iterable(summary_ids)]

def trim_str(txt, max_len):
    max_len = len(txt.split()) if max_len is None else max_len
    return " ".join(txt.split()[:max_len]).strip("Read an ")

class SummarizationDataset(Dataset):
    def __init__(
            self,
            tokenizer,
            data_dir,
            max_input_len=1024,
            max_output_len=75,
            type_path="train",
            overwrite_cache=False,
            char_len=None
    ):

        tok_name = tokenizer.__class__.__name__.lower().rstrip("tokenizer")
        self.tokenizer = tokenizer

        self.source, self.target = encode_file(
            tokenizer,
            os.path.join(data_dir, type_path + ".jsonl"),
            max_input_len,
            max_output_len,
            overwrite_cache=overwrite_cache,
            tok_name=tok_name,
            char_len=char_len,
        )


    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        # entry = self.source[idx]
        # input_ids = self.tokenizer.encode(entry[0], truncation=True, max_length=self.max_input_len)
        # output_ids = self.tokenizer.encode(entry[1], truncation=True, max_length=self.max_output_len)
        input_ids = self.source[idx]
        output_ids = self.target[idx]
        if self.tokenizer.bos_token_id is None:  # pegasus
            output_ids = [self.tokenizer.pad_token_id] + output_ids
        return torch.tensor(input_ids), torch.tensor(output_ids)

    @staticmethod
    def collate_fn(batch):
        # A hack to know if this is bart or pegasus. DDP doesn't like global variables nor class-level memebr variables
        if batch[0][0][-1].item() == 2:
            pad_token_id = 1  # AutoTokenizer.from_pretrained('facebook/bart-base').pad_token_id
        elif batch[0][0][-1].item() == 1:
            pad_token_id = 0  # AutoTokenizer.from_pretrained('google/pegasus-large').pad_token_id
        else:
            assert False

        input_ids, output_ids = list(zip(*batch))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        output_ids = torch.nn.utils.rnn.pad_sequence(output_ids, batch_first=True, padding_value=pad_token_id)
        return input_ids, output_ids



class Summarizer(pl.LightningModule):

    def __init__(self, args, pad_to_length=False):
        super().__init__()
        self.args = args
        self.hparams = args
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer, use_fast=True)

        if 'long' in self.args.model_path:
            config = LongformerEncoderDecoderConfig.from_pretrained(self.args.model_path)
            config.attention_dropout = self.args.attention_dropout
            config.gradient_checkpointing = self.args.grad_ckpt
            config.attention_mode = self.args.attention_mode
            config.attention_window = [self.args.attention_window] * config.encoder_layers
            self.model = LongformerEncoderDecoderForConditionalGeneration.from_pretrained(
                self.args.model_path, config=config)
            # self.model = LongformerEncoderDecoderForConditionalGeneration.from_config(config)
        else:
            config = AutoConfig.from_pretrained(self.args.model_path)
            config.attention_dropout = self.args.attention_dropout
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.args.model_path, config=config)
        self.train_dataloader_object = self.val_dataloader_object = self.test_dataloader_object = None

        special_tokens = ["[name]", "[sum]", "[desc]", "<eos>", "[MASK]"]
        self.tokenizer.add_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.tokenizer.save_pretrained(args.save_dir)
        config.to_json_file(os.path.join(args.save_dir, "config.json"))

        # faeze added
        self.dataset_kwargs: dict = dict(
            data_dir=self.hparams.data_dir,
            max_input_len=self.hparams.max_input_len,
            char_len=self.args.char_length
        )
        n_observations_per_split = {
            "train": self.hparams.n_train,
            "val": self.hparams.n_val,
            "test": self.hparams.n_test,
        }
        self.n_obs = {k: v if v >= 0 else None for k, v in n_observations_per_split.items()}

        self.target_lens = {
            "train": self.hparams.max_output_len,
            "val": self.hparams.max_output_len,
            "test": self.hparams.max_output_len,
        }
        assert self.target_lens["train"] <= self.target_lens["val"], f"target_lens: {self.target_lens}"
        assert self.target_lens["train"] <= self.target_lens["test"], f"target_lens: {self.target_lens}"


        self.pad_to_length = pad_to_length
        self.eval_beams = self.model.config.num_beams if self.hparams.eval_beams is None else self.hparams.eval_beams

        # up to here

    def _prepare_input(self, input_ids):
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
        attention_mask[input_ids == self.tokenizer.pad_token_id] = 0
        if isinstance(self.model, LongformerEncoderDecoderForConditionalGeneration):
            attention_mask[:, 0] = 2  # global attention on one token for all model params to be used, which is important for gradient checkpointing to work
            if self.args.attention_mode == 'sliding_chunks':
                half_padding_mod = self.model.config.attention_window[0]
            elif self.args.attention_mode == 'sliding_chunks_no_overlap':
                half_padding_mod = self.model.config.attention_window[0] / 2
            else:
                raise NotImplementedError
            input_ids, attention_mask = pad_to_window_size(  # ideally, should be moved inside the LongformerModel
                input_ids, attention_mask, half_padding_mod, self.tokenizer.pad_token_id)
        return input_ids, attention_mask

    def forward(self, input_ids, output_ids):
        input_ids, attention_mask = self._prepare_input(input_ids)
        decoder_input_ids = output_ids[:, :-1]
        decoder_attention_mask = (decoder_input_ids != self.tokenizer.pad_token_id)
        labels = output_ids[:, 1:].clone()
        outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                use_cache=False,)
        lm_logits = outputs[0]
        if self.args.label_smoothing == 0:
            # Same behavior as modeling_bart.py, besides ignoring pad_token_id
            ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            assert lm_logits.shape[-1] == self.model.config.vocab_size
            loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))
        else:
            lprobs = torch.nn.functional.log_softmax(lm_logits, dim=-1)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs, labels, self.args.label_smoothing, ignore_index=self.tokenizer.pad_token_id
            )
        return [loss]

    def training_step(self, batch, batch_nb):
        output = self.forward(*batch)
        loss = output[0]
        lr = loss.new_zeros(1) + self.trainer.optimizers[0].param_groups[0]['lr']
        tensorboard_logs = {'train_loss': loss, 'lr': lr,
                            'input_size': batch[0].numel(),
                            'output_size': batch[1].numel(),
                            'mem': torch.cuda.memory_allocated(loss.device) / 1024 ** 3 if torch.cuda.is_available() else 0}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        for p in self.model.parameters():
            p.requires_grad = False

        outputs = self.forward(*batch)
        vloss = outputs[0]
        input_ids, output_ids = batch
        input_ids, attention_mask = self._prepare_input(input_ids)
        generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                            use_cache=True, max_length=self.args.max_output_len,
                                            num_beams=self.eval_beams)
        generated_str = self.tokenizer.batch_decode(
            generated_ids.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        gold_str = self.tokenizer.batch_decode(output_ids.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
        trimmed_input  = self.tokenizer.batch_decode(input_ids.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
        scorer = rouge_scorer.RougeScorer(rouge_types=['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=False)
        rouge1 = rouge2 = rougel = rougelsum = 0.0
        for ref, pred in zip(gold_str, generated_str):
            score = scorer.score(ref, pred)
            rouge1 += score['rouge1'].fmeasure
            rouge2 += score['rouge2'].fmeasure
            rougel += score['rougeL'].fmeasure
            rougelsum += score['rougeLsum'].fmeasure
        rouge1 /= len(generated_str)
        rouge2 /= len(generated_str)
        rougel /= len(generated_str)
        rougelsum /= len(generated_str)

        return {'vloss': vloss,
                'rouge1': vloss.new_zeros(1) + rouge1,
                'rouge2': vloss.new_zeros(1) + rouge2,
                'rougeL': vloss.new_zeros(1) + rougel,
                'rougeLsum': vloss.new_zeros(1) + rougelsum,
                'trimmed_input': trimmed_input,
                'preds': generated_str,
                'target': gold_str}

    def validation_epoch_end(self, outputs, prefix="val"):
        for p in self.model.parameters():
            p.requires_grad = True

        names = ['vloss', 'rouge1', 'rouge2', 'rougeL', 'rougeLsum']
        metrics = []
        for name in names:
            metric = torch.stack([x[name] for x in outputs]).mean()
            if self.trainer.use_ddp:
                torch.distributed.all_reduce(metric, op=torch.distributed.ReduceOp.SUM)
                metric /= self.trainer.world_size
            metrics.append(metric)
        logs = dict(zip(*[names, metrics]))
        source = flatten_list([x["trimmed_input"] for x in outputs])
        preds = flatten_list([x["preds"] for x in outputs])
        target = flatten_list([x["target"] for x in outputs])
        # if prefix == "test":
        self.write_generation(source, preds, target, prefix)
        print(logs)
        return {'avg_val_loss': logs['vloss'], 'log': logs, 'progress_bar': logs, 'source': source, 'preds': preds, 'target': target}

    def write_generation(self, src_str, pred_str, gold_str, prefix):
        # generation_path = Path(self.args.save_dir) / f"{prefix}_prediction_beams{self.eval_beams}_maxlen_{self.args.max_output_len}.jsonl" #f"{prefix}_generation.txt"
        generation_path = os.path.join(self.args.save_dir, f"{prefix}_prediction_beams{self.eval_beams}_maxlen_{self.args.max_output_len}.jsonl")
        # Path(generation_path).mkdir(exist_ok=True, parents=True)
        with open(generation_path, 'w') as f_out:
            for inp, output, pred in zip(src_str, gold_str, pred_str):
                f_out.write(
                    json.dumps({"input": inp, "gold": output, "predictions": pred})
                    + "\n"
                )

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_epoch_end(self, outputs):
        result = self.validation_epoch_end(outputs, prefix="test")
        print(result['avg_val_loss'], result['log'], result['progress_bar'])

    def configure_optimizers(self):
        if self.args.adafactor:
            optimizer = Adafactor(self.model.parameters(), lr=self.args.learning_rate, scale_parameter=False, relative_step=False)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        if self.args.debug:
            return optimizer  # const LR
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        num_steps = self.args.dataset_size * self.args.epochs /  num_gpus / self.hparams.gradient_accumulation_steps / self.args.batch_size
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup, num_training_steps=num_steps
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    # Faeze added this
    def get_dataset(self, type_path) -> SummarizationDataset:
        # n_obs = self.n_obs[type_path]
        max_target_length = self.target_lens[type_path]
        dataset = SummarizationDataset(
            self.tokenizer,
            type_path=type_path,
            # n_obs=n_obs,
            max_output_len=max_target_length,
            **self.dataset_kwargs,
        )
        return dataset


    # def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:
    #     dataset = self.get_dataset(type_path)
    #     sampler = None
    #     if self.hparams.sortish_sampler and type_path == "train":
    #         assert self.hparams.gpus <= 1  # TODO: assert earlier
    #         sampler = dataset.make_sortish_sampler(batch_size)
    #         shuffle = False
    #
    #     dataloader = DataLoader(
    #         dataset,
    #         batch_size=batch_size,
    #         collate_fn=dataset.collate_fn,
    #         shuffle=shuffle,
    #         num_workers=self.args.num_workers,
    #         sampler=sampler,
    #     )
    #     return dataloader
    #
    # def train_dataloader(self) -> DataLoader:
    #     dataloader = self.get_dataloader("train", batch_size=self.hparams.batch_size, shuffle=True)
    #     t_total = (
    #         (len(dataloader.dataset) // (self.hparams.batch_size * max(1, self.hparams.gpus)))
    #         // self.hparams.gradient_accumulation_steps
    #         * float(self.hparams.epochs)
    #     )
    #     scheduler = get_linear_schedule_with_warmup(
    #         self.opt, num_warmup_steps=self.hparams.warmup, num_training_steps=t_total
    #     )
    #     self.lr_scheduler = scheduler
    #     return dataloader
    #
    # def val_dataloader(self) -> DataLoader:
    #     return self.get_dataloader("val", batch_size=self.hparams.eval_batch_size)
    #
    # def test_dataloader(self) -> DataLoader:
    #     return self.get_dataloader("test", batch_size=self.hparams.eval_batch_size)

    # up to here

    def _get_dataloader(self, current_dataloader, split_name, is_train):
        if current_dataloader is not None:
            return current_dataloader
        # dataset = SummarizationDataset(hf_dataset=self.hf_datasets[split_name], tokenizer=self.tokenizer,
        #                                max_input_len=self.args.max_input_len, max_output_len=self.args.max_output_len)
        dataset = self.get_dataset(split_name)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=is_train) if self.trainer.use_ddp else None
        return DataLoader(dataset, batch_size=self.args.batch_size, shuffle=(sampler is None),
                          num_workers=self.args.num_workers, sampler=sampler,
                          collate_fn=SummarizationDataset.collate_fn)

    @pl.data_loader
    def train_dataloader(self):
        self.train_dataloader_object = self._get_dataloader(self.train_dataloader_object, 'train', is_train=True)
        return self.train_dataloader_object

    @pl.data_loader
    def val_dataloader(self):
        self.val_dataloader_object = self._get_dataloader(self.val_dataloader_object, 'val', is_train=False)
        return self.val_dataloader_object

    @pl.data_loader
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
        parser.add_argument("--save_dir", type=str, default='summarization')
        parser.add_argument("--save_prefix", type=str, default='test')
        parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
        parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="number of gradient accumulation steps")
        parser.add_argument("--gpus", type=int, default=-1,
                            help="Number of gpus. 0 for CPU")
        parser.add_argument("--warmup", type=int, default=1000, help="Number of warmup steps")
        parser.add_argument("--learning_rate", type=float, default=0.00003, help="Maximum learning rate")
        parser.add_argument("--val_every", type=float, default=1.0, help="Number of training steps between validations")
        parser.add_argument("--val_percent_check", default=1.00, type=float, help='Percent of validation data used')
        parser.add_argument("--num_workers", type=int, default=0, help="Number of data loader workers")
        parser.add_argument("--seed", type=int, default=1234, help="Seed")
        parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
        parser.add_argument("--disable_checkpointing", action='store_true', help="No logging or checkpointing")
        parser.add_argument("--max_output_len", type=int, default=100,
                            help="maximum num of wordpieces/summary. Used for training and testing")
        parser.add_argument("--max_input_len", type=int, default=2048,
                            help="maximum num of wordpieces/summary. Used for training and testing")
        parser.add_argument("--test", action='store_true', help="Test only, no training")
        parser.add_argument("--model_path", type=str, default='facebook/bart-base',
                            help="Path to the checkpoint directory or model name")
        parser.add_argument("--tokenizer", type=str, default='facebook/bart-base')
        parser.add_argument("--no_progress_bar", action='store_true', help="no progress bar. Good for printing")
        parser.add_argument("--fp32", action='store_true', help="default is fp16. Use --fp32 to switch to fp32")
        parser.add_argument("--debug", action='store_true', help="debug run")
        parser.add_argument("--resume_ckpt", type=str, help="Path of a checkpoint to resume from")
        parser.add_argument('--grad_ckpt', action='store_true', help='Enable gradient checkpointing to save memory')
        parser.add_argument("--attention_dropout", type=float, default=0.1, help="attention dropout")
        parser.add_argument("--attention_mode", type=str, default='sliding_chunks', help="Longformer attention mode")
        parser.add_argument("--attention_window", type=int, default=512, help="Attention window")
        parser.add_argument("--label_smoothing", type=float, default=0.0, required=False)
        parser.add_argument("--adafactor", action='store_true', help="Use adafactor optimizer")
        parser.add_argument(
            "--overwrite_cache", action="store_true", help="Overwrite the cached data."
        )
        parser.add_argument(
            "--data_dir",
            type=str,
            required=True,
            help="The input data dir. Should contain train.source, train.target, val.source, val.target, test.source, test.target",
        )
        parser.add_argument(
            "--eval_batch_size", default=4, type=int, help="Batch size for evaluation."
        )
        parser.add_argument("--sortish_sampler", action="store_true", default=False)
        parser.add_argument("--n_train", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--n_val", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--n_test", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--eval_beams", type=int, default=5, required=False)
        parser.add_argument("--best_checkpoint", type=str, default=None, required=False)
        parser.add_argument("--char_length", type=int, default=None, help="max masked character description length")

        return parser


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = Summarizer(args, pad_to_length=False)
    # model.hf_datasets = nlp.load_dataset('scientific_papers', 'arxiv')

    logger = TestTubeLogger(
        save_dir=args.save_dir,
        name=args.save_prefix,
        version=0  # always use version=0
    )

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(args.save_dir, args.save_prefix, "checkpoints"),
        save_top_k=5,
        verbose=True,
        monitor='avg_val_loss',
        mode='min',
        period=-1,
        prefix=''
    )

    print(args)
    args.dataset_size = 7023


    trainer = pl.Trainer(gpus=args.gpus, distributed_backend='ddp' if torch.cuda.is_available() else None,
                         track_grad_norm=-1,
                         max_epochs=args.epochs if not args.debug else 100,
                         max_steps=None if not args.debug else 1,
                         replace_sampler_ddp=False,
                         accumulate_grad_batches=args.gradient_accumulation_steps,
                         val_check_interval=args.val_every if not args.debug else 1,
                         num_sanity_val_steps=2 if not args.debug else 0,
                         check_val_every_n_epoch=1 if not args.debug else 1,
                         val_percent_check=args.val_percent_check,
                         test_percent_check=args.val_percent_check,
                         logger=logger,
                         checkpoint_callback=checkpoint_callback if not args.disable_checkpointing else False,
                         show_progress_bar=not args.no_progress_bar,
                         use_amp=not args.fp32, amp_level='O2',
                         resume_from_checkpoint=args.resume_ckpt,
                         )

    if not args.test:
        trainer.fit(model) # Faeze: can have trainer.fit(model, train_dataloader, val_dataloader) if the model has predefined dataloader this will be skipped!

    if args.best_checkpoint:
#        trainer.test(model, ckpt_path=args.best_checkpoint)
        hparams = torch.load(args.best_checkpoint)['hyper_parameters']
        model = model.load_from_checkpoint(checkpoint_path=args.best_checkpoint, **hparams)
        model.eval()
        print(model.learning_rate)
    else:
        trainer.test(model)


if __name__ == "__main__":
    main_arg_parser = argparse.ArgumentParser(description="summarization")
    parser = Summarizer.add_model_specific_args(main_arg_parser, os.getcwd())
    args = parser.parse_args()
    main(args)
