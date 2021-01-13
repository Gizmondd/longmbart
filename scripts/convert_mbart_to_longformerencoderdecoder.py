import argparse
import logging
import os
import copy

from transformers import MBartTokenizer

from transformers import MBartForConditionalGeneration
from transformers.modeling_bart import shift_tokens_right
from longformer.longformer_encoder_decoder import LongformerSelfAttentionForBart
from longformer.longformer_encoder_decoder_mbart import MLongformerEncoderDecoderForConditionalGeneration, MLongformerEncoderDecoderConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_long_model(
    save_model_to,
    base_model,
    tokenizer_name_or_path,
    attention_window,
    max_pos,
    cache_dir
):
    model = MBartForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=base_model, cache_dir=cache_dir)
    tokenizer = MBartTokenizer.from_pretrained(tokenizer_name_or_path, model_max_length=max_pos, cache_dir=cache_dir)
    config = MLongformerEncoderDecoderConfig.from_pretrained(base_model, cache_dir=cache_dir)
    model.config = config

    # in BART attention_probs_dropout_prob is attention_dropout, but LongformerSelfAttention
    # expects attention_probs_dropout_prob, so set it here
    config.attention_probs_dropout_prob = config.attention_dropout
    config.architectures = ['MLongformerEncoderDecoderForConditionalGeneration', ]

    # extend position embeddings
    tokenizer.model_max_length = max_pos
    tokenizer.init_kwargs['model_max_length'] = max_pos
    current_max_pos, embed_size = model.model.encoder.embed_positions.weight.shape
    assert current_max_pos == config.max_position_embeddings + 2

    config.max_encoder_position_embeddings = max_pos
    config.max_decoder_position_embeddings = config.max_position_embeddings
    del config.max_position_embeddings
    max_pos += 2  # NOTE: BART has positions 0,1 reserved, so embedding size is max position + 2
    assert max_pos >= current_max_pos

    # allocate a larger position embedding matrix for the encoder
    new_encoder_pos_embed = model.model.encoder.embed_positions.weight.new_empty(max_pos, embed_size)
    # copy position embeddings over and over to initialize the new position embeddings
    k = 2
    step = current_max_pos - 2
    while k < max_pos - 1:
        new_encoder_pos_embed[k:(k + step)] = model.model.encoder.embed_positions.weight[2:]
        k += step
    model.model.encoder.embed_positions.weight.data = new_encoder_pos_embed

    # allocate a larger position embedding matrix for the decoder
    # new_decoder_pos_embed = model.model.decoder.embed_positions.weight.new_empty(max_pos, embed_size)
    # # copy position embeddings over and over to initialize the new position embeddings
    # k = 2
    # step = current_max_pos - 2
    # while k < max_pos - 1:
    #     new_decoder_pos_embed[k:(k + step)] = model.model.decoder.embed_positions.weight[2:]
    #     k += step
    # model.model.decoder.embed_positions.weight.data = new_decoder_pos_embed

    # replace the `modeling_bart.SelfAttention` object with `LongformerSelfAttention`
    config.attention_window = [attention_window] * config.num_hidden_layers
    config.attention_dilation = [1] * config.num_hidden_layers

    for i, layer in enumerate(model.model.encoder.layers):
        longformer_self_attn_for_bart = LongformerSelfAttentionForBart(config, layer_id=i)

        longformer_self_attn_for_bart.longformer_self_attn.query = layer.self_attn.q_proj
        longformer_self_attn_for_bart.longformer_self_attn.key = layer.self_attn.k_proj
        longformer_self_attn_for_bart.longformer_self_attn.value = layer.self_attn.v_proj

        longformer_self_attn_for_bart.longformer_self_attn.query_global = copy.deepcopy(layer.self_attn.q_proj)
        longformer_self_attn_for_bart.longformer_self_attn.key_global = copy.deepcopy(layer.self_attn.k_proj)
        longformer_self_attn_for_bart.longformer_self_attn.value_global = copy.deepcopy(layer.self_attn.v_proj)

        longformer_self_attn_for_bart.output = layer.self_attn.out_proj

        layer.self_attn = longformer_self_attn_for_bart
    logger.info(f'saving model to {save_model_to}')
    model.save_pretrained(save_model_to)
    tokenizer.save_pretrained(save_model_to)
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Convert BART to LongBART. Replaces BART encoder's SelfAttnetion with LongformerSelfAttention")
    parser.add_argument(
        '--base_model',
        type=str,
        default='facebook/mbart-large-cc25',
        help='The name or path of the base model you want to convert'
    )
    parser.add_argument(
        '--tokenizer_name_or_path',
        type=str,
        default='facebook/mbart-large-cc25',
        help='The name or path of the tokenizer'
    )
    parser.add_argument(
        '--save_model_to',
        type=str,
        required=True,
        help='The path to save the converted model'
    )
    parser.add_argument(
        '--attention_window',
        type=int,
        default=512,
        help='attention window size for longformer self attention (one sided)'
    )
    parser.add_argument(
        '--max_pos',
        type=int,
        default=4096 * 4,
        help='maximum encoder positions'
    )
    
    parser.add_argument(
        '--cache_dir',
        type=str,
        help='where to save original model'
    )

    args = parser.parse_args()

    if not os.path.exists(args.save_model_to):
        os.mkdir(args.save_model_to)

    create_long_model(
        save_model_to=args.save_model_to,
        base_model=args.base_model,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        attention_window=args.attention_window,
        max_pos=args.max_pos,
        cache_dir=args.cache_dir
    )

    tokenizer = MBartTokenizer.from_pretrained(args.save_model_to)
    #TXT = "My friends are <mask> but they eat too many carbs."
    #TXT = "My friends are fine but they eat too many carbs."
    TXT = "Das ist ein Test."
    #model = MLongformerEncoderDecoderForConditionalGeneration.from_pretrained(args.save_model_to)
    model.model.encoder.config.gradient_checkpointing = True
    model.model.decoder.config.gradient_checkpointing = True
    #data = tokenizer([TXT], return_tensors='pt', padding='max_length', max_length=2048)
    #input_ids = data['input_ids']
    #attention_mask = data['attention_mask']
    #decoder_input_ids = shift_tokens_right(input_ids[:, :5], tokenizer.pad_token_id)
    #logits = model(input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, use_cache=False)[0]
    #masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero(as_tuple=False).item()
    #probs = logits[0, masked_index].softmax(dim=0)
    #values, predictions = probs.topk(5)
    #print(tokenizer.convert_ids_to_tokens(predictions))
    
    #batch = tokenizer.prepare_seq2seq_batch(src_texts=[TXT], src_lang="en_XX", max_length=1024, truncation=False, padding="max_length")
    #translated_tokens = model.generate(**batch, decoder_start_token_id=tokenizer.lang_code_to_id["de_DE"])
    
    batch = tokenizer.prepare_seq2seq_batch(src_texts=[TXT], src_lang="de_DE", max_length=1024, truncation=False, padding="max_length")
    translated_tokens = model.generate(**batch, decoder_start_token_id=tokenizer.lang_code_to_id["en_XX"])
    translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    print(translation)


if __name__ == "__main__":
    main()
    
    
#{'input_ids': tensor([[  2646,  23902,    621,   5885,   1284,   1836,  73203,   5792,   5941,
         #111758,      7,      5,      2, 250004]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}
