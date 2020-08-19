import torch
import config


def tokenization(tokenizer, review):
    inputs = tokenizer.encode_plus(
            review,
            None,
            add_special_tokens=True,
            max_length=config.MAX_LEN,
            pad_to_max_length=True,
        )

    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]
    
    padding_length = config.MAX_LEN - len(ids)
    ids = ids + ([0] * padding_length)
    mask = mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)

    ids = torch.tensor(ids, dtype=torch.long)
    mask = torch.tensor(mask, dtype=torch.long)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
    
    return ids, mask, token_type_ids


class BERTDataset:
    def __init__(self, review, target):
        self.review = review
        self.target = target
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.review)

    def __getitem__(self, item):
        review = str(self.review[item])
        review = " ".join(review.split())

        ids, mask, token_type_ids = tokenization(self.tokenizer, review)

        return {
            "ids": ids,
            "mask": mask,
            "token_type_ids": token_type_ids,
            "targets": torch.tensor(self.target[item], dtype=torch.float),
        }
