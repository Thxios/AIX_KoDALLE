
import os
import numpy as np
# import pandas as pd
# import pytorch_lightning as pl
import torch
from torch import nn
# from pytorch_lightning import loggers as pl_loggers
# from torch.utils.data import DataLoader, Dataset
# from dataset import KobartSummaryModule
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from torchsummary import summary

if __name__ == '__main__':

    model: nn.Module = BartForConditionalGeneration.from_pretrained('gogamza/kobart-base-v1', cache_dir='../.cache')

    print(model.lm_head)
    model.lm_head = nn.Linear(768, 16400, bias=False)
    model.final_logits_bias = torch.zeros([1, 16400])
    print(model)
    print(model.final_logits_bias)
    print(model.final_logits_bias.size())
    input_id = torch.LongTensor([[1, 43, 64, 34, 6]])
    print(input_id.size())
    out = model(input_id)
    # print(out)
    # print()
    # summary(model, (10,), device='cpu')
    cnt = 0

    for param in model.parameters():
        param.requires_grad = False
    for param in model.lm_head.parameters():
        param.requires_grad = True

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    # for module in model.modules():
    #     print(module)
    print(params)

