from transformers import BertConfig, BertModel


def bert_processor(args, logger, is_fast_token=False):
    logger.info("Loading Modeling")
    config = BertConfig.from_pretrained(args.bert_path)
    if is_fast_token:
        from transformers import BertTokenizerFast
        logger.info("使用BertTokenizerFast")
        tokenizer = BertTokenizerFast.from_pretrained(args.bert_path)
    else:
        from transformers import BertTokenizer
        logger.info("使用BertTokenizer")
        tokenizer = BertTokenizer.from_pretrained(args.bert_path)

    model = BertModel.from_pretrained(args.bert_path, config=config)
    return config, tokenizer, model
