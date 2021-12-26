"""
RoBERTaを用いたテキスト生成
"""
import sys
import argparse
import torch
from transformers import RobertaTokenizerFast,RobertaForMaskedLM        #Hugging Face

MASK = "[MASK]"
MASK_ATOM = "[MASK]"

# 前処理
def preprocess(tokens, tokenizer):
    tok_ids = tokenizer.convert_tokens_to_ids(tokens)   #トークン化
    tok_tensor = torch.tensor([tok_ids])                #Tensor型にする
    return tok_tensor

# モデル入力するために入力文を変形
def get_seed_sent(seed_sentence, tokenizer, n_append_mask=0):
    mask_ids=[]

    seed_sentence = seed_sentence.replace(MASK, MASK_ATOM)
    toks = tokenizer.tokenize(seed_sentence)  # トークン化       ※Ġ は「スペース」という意味

    for i, tok in enumerate(toks):
        if tok == MASK_ATOM:
            mask_ids.append(i)

    for mask_id in mask_ids:
        toks[mask_id] = MASK  

    mask_ids = sorted(list(set(mask_ids)))

    for _ in range(n_append_mask):
        mask_ids.append(len(toks))
        toks.append(MASK)                   # 入力文にマスクを追加
    mask_ids = sorted(list(set(mask_ids)))

    seg = [0] * len(toks)               # toks :入力文のワード数
    seg_tensor = torch.tensor([seg])    # Tensor型にする

    return toks, seg_tensor, mask_ids

#モデル読み込み
def load_model(version):
    model = RobertaForMaskedLM.from_pretrained(version)
    model.eval()
    return model


# 予測
def predict(model, tokenizer, tok_tensor, seg_tensor,decoding_strategy):
    preds = model(tok_tensor, seg_tensor).logits  #Ex) tok_tensor [1142,1110,170,5650,119]    seg_tensor [0,0,0,0,0]
    # preds  outputs : Tensor size [1,number of words,28996]

    pred_idxs = preds.argmax(dim=-1).tolist()[0]  #Ex) pred_idxs [119,1110,170,119,119]

    pred_toks = tokenizer.convert_ids_to_tokens(pred_idxs)  # 文字列化
    return pred_toks


def sequential_decoding(toks, seg_tensor, model, tokenizer,decoding_strategy):
    """ Decode from model one token at a time """
    for step_n in range(len(toks)):  # len(toks)  :　文字列に含まれる単語の数
        print("Iteration %d: %s" % (step_n, " ".join(toks)))
        tok_tensor = preprocess(toks,tokenizer)  # 前処理(ID化)  Ex)  ["this" "is" "a" "sentence" "."]   ->  tok_tensor : [1142,1110,170,5650,119]
        pred_toks = predict(model, tokenizer, tok_tensor, seg_tensor,decoding_strategy)  # 予測
        print("\tRoBERTa prediction: %s" % (" ".join(pred_toks)))
        toks[step_n] = pred_toks[step_n]
    return toks


def masked_decoding(toks, seg_tensor, masks, model, tokenizer,decoding_strategy):
    """ Decode from model by replacing masks """
    for step_n, mask_id in enumerate(masks):
        print("Iteration %d: %s" % (step_n, " ".join(toks)))
        tok_tensor = preprocess(toks, tokenizer)  # 前処理(ID化)
        pred_toks = predict(model, tokenizer, tok_tensor, seg_tensor,decoding_strategy)
        print("\tRoBERTa prediction: %s\n" % (" ".join(pred_toks)))
        toks[mask_id] = pred_toks[mask_id]
    return toks


# ユーザーが文章を入力する場合
def interact(args, model, tokenizer,decoding_strategy):
    while True:
        raw_str = input(">>> ")  # 「raw_str」がユーザーが入力した文章が入る
        if raw_str.startswith("CHANGE"):
            _, attr, val = raw_str.split()
            setattr(args, attr, val)
            continue

        toks, seg_tensor, mask_ids = get_seed_sent(raw_str, tokenizer, n_append_mask=0)
        if args.decoding_strategy == "sequential":
            toks = sequential_decoding(toks, seg_tensor, model, tokenizer,decoding_strategy)
        elif args.decoding_strategy == "masked":
            toks = masked_decoding(toks, seg_tensor, mask_ids, model, tokenizer,decoding_strategy)
        else:
            raise NotImplementedError("Decoding strategy %s not found!" % args.decoding_strategy)

        print("Final: %s" % (" ".join(toks)))


def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument("--interact", action="store_true")
    parser.add_argument("--roberta_version", default="roberta-base",
                        choices=["roberta-large", "roberta-large-mnli",
                                 "distilroberta-base", "roberta-base-openai-detector",
                                 "roberta-large-openai-detector"])
    # デフォルトの入力文章
    parser.add_argument("--seed_sentence", type=str, default="I have a pen.")

    # Decoding
    parser.add_argument("--decoding_strategy", type=str, default="sequential",
                        choices=["masked", "sequential"])
    parser.add_argument("--n_append_mask", type=int, default=1)     # 次に来る単語を予測する数(単語数)

    args = parser.parse_args(arguments)
    tokenizer = RobertaTokenizerFast.from_pretrained(args.roberta_version)  # 字句解析設定
    model = load_model(args.roberta_version)                                # モデル設定

    print("Decoding strategy %s, argmax at each step" % (args.decoding_strategy))

    if args.interact:
        # ユーザーが文章を入力する場合
        sys.exit(interact(args, model, tokenizer,args.decoding_strategy))
    else:
        # ユーザーが文章を入力しない場合（デフォルト文で動作）
        toks, seg_tensor, mask_ids = get_seed_sent(args.seed_sentence, tokenizer,n_append_mask=args.n_append_mask)

        #Ex) toks =  ["this" "is" "a" "sentence" "."]
        #Ex) seg_tensor  tensor([0,0,0,0,0])
        #Ex) mask_ids  list 0 []

        if args.decoding_strategy == "sequential":
            toks = sequential_decoding(toks, seg_tensor, model, tokenizer,args.decoding_strategy)
        elif args.decoding_strategy == "masked":
            toks = masked_decoding(toks, seg_tensor, mask_ids, model, tokenizer,args.decoding_strategy)
        else:
            raise NotImplementedError("Decoding strategy %s not found!" % args.decoding_strategy)

        print("Final: %s" % (" ".join(toks)))

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))