import os
import datetime
import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.optim as optim
from omegaconf import DictConfig
import wandb
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel
from transformers import BertJapaneseTokenizer


# Dataset
class CreateDataset(Dataset):  # 文章のtokenize処理を行ってDataLoaderに渡す関数
    TEXT_COLUMN = "sentence"
    LABEL_COLUMN = "label"

    def __init__(self, data, tokenizer, max_token_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_row = self.data.iloc[index]
        text = data_row[self.TEXT_COLUMN]
        labels = data_row[self.LABEL_COLUMN]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return dict(
            text=text,
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            labels=torch.tensor(labels),
        )


# DataModule
class CreateDataModule(pl.LightningDataModule):
    """
    DataFrameからモデリング時に使用するDataModuleを作成
    """

    def __init__(
        self,
        train_df,
        valid_df,
        test_df,
        batch_size=16,
        max_token_len=512,
        pretrained_model="cl-tohoku/bert-base-japanese-char-whole-word-masking",
    ):
        super().__init__()
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.max_token_len = max_token_len
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(pretrained_model)

    # train,val,testデータのsplit
    def setup(self, stage=None):
        self.train_dataset = CreateDataset(
            self.train_df, self.tokenizer, self.max_token_len
        )
        self.vaild_dataset = CreateDataset(
            self.valid_df, self.tokenizer, self.max_token_len
        )
        self.test_dataset = CreateDataset(
            self.test_df, self.tokenizer, self.max_token_len
        )

    #
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=os.cpu_count(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.vaild_dataset, batch_size=self.batch_size, num_workers=os.cpu_count()
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=os.cpu_count()
        )


# Model
class TextClassifierModel(pl.LightningModule):
    def __init__(
        self,
        n_classes: int,  # クラスの数
        n_epochs=None,
        pretrained_model="cl-tohoku/bert-base-japanese-char-whole-word-masking",
    ):
        super().__init__()
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # モデルの構造
        self.bert = BertModel.from_pretrained(pretrained_model, return_dict=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.n_epochs = n_epochs
        self.criterion = nn.CrossEntropyLoss()

        # BertLayerモジュールの最後を勾配計算ありに変更
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.encoder.layer[-1].parameters():
            param.requires_grad = True

    # 順伝搬
    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        preds = self.classifier(output.pooler_output)
        loss = 0
        if labels is not None:
            loss = self.criterion(preds, labels)
        return loss, preds

    # trainのミニバッチに対して行う処理
    def training_step(self, batch, batch_idx):
        loss, preds = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        return {"loss": loss, "batch_preds": preds, "batch_labels": batch["labels"]}

    # validation、testでもtrain_stepと同じ処理を行う
    def validation_step(self, batch, batch_idx):
        loss, preds = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        self.validation_step_outputs.append(loss)
        return {"loss": loss, "batch_preds": preds, "batch_labels": batch["labels"]}

    def test_step(self, batch, batch_idx):
        loss, preds = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        self.test_step_outputs.append(loss)
        return {"loss": loss, "batch_preds": preds, "batch_labels": batch["labels"]}

    # epoch終了時にvalidationのlossとaccuracyを記録
    def on_validation_epoch_end(
        self, mode="val"
    ):  # https://github.com/Lightning-AI/lightning/pull/16520
        # loss計算
        """print(self.validation_step_outputs)
        epoch_preds = torch.stack(
            [x["batch_preds"] for x in self.validation_step_outputs], dim=0
        )
        epoch_labels = torch.stack(
            [x["batch_labels"] for x in self.validation_step_outputs], dim=0
        )
        epoch_loss = self.criterion(epoch_preds, epoch_labels)
        self.log(f"{mode}_loss", epoch_loss, logger=True)

        # accuracy計算
        num_correct = (epoch_preds.argmax(dim=1) == epoch_labels).sum().item()
        epoch_accuracy = num_correct / len(epoch_labels)
        self.log(f"{mode}_accuracy", epoch_accuracy, logger=True)

        self.validation_step_outputs.clear()"""
        epoch_average = torch.stack(self.validation_step_outputs).mean()
        self.validation_step_outputs.clear()  # free memory
        self.log(f"{mode}_accuracy", epoch_average, logger=True)

    # testデータのlossとaccuracyを算出（validationの使いまわし）
    def on_test_epoch_end(self):
        return self.on_validation_epoch_end(self.test_step_outputs, "test")

    # optimizerの設定
    def configure_optimizers(self):
        # pretrainされているbert最終層のlrは小さめ、pretrainされていない分類層のlrは大きめに設定
        optimizer = optim.Adam(
            [
                {"params": self.bert.encoder.layer[-1].parameters(), "lr": 5e-5},
                {"params": self.classifier.parameters(), "lr": 1e-4},
            ]
        )

        return [optimizer]


# モデルの保存と更新のための関数
def make_callbacks(min_delta, patience, checkpoint_path):
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename="{epoch}",
        # save_top_k=1,  #save_best_only
        verbose=True,
        monitor="val_loss",
        mode="min",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=min_delta, patience=patience, mode="min"
    )

    return [early_stop_callback, checkpoint_callback]


# Train Runner
@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    cwd = hydra.utils.get_original_cwd()
    current = (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime(
        "%Y%m%d_%H%M%S"
    )

    # wandbの初期化
    wandb.init(
        project=cfg.wandb.prroject,
        name=("exp_" + str(cfg.wandb.exp_num)),
        project=cfg.wandb.project,
        tags=cfg.wandb.tags,
    )

    wandb_logger = WandbLogger(
        log_model=False,
    )
    wandb_logger.watch(model, log="all")

    checkpoint_path = os.path.join(
        wandb_logger.experiment.dir, cfg.path.checkpoint_path
    )

    # dataModuleのインスタンス化
    train, val_test = train_test_split(
        pd.read_csv(cfg.path.data_file_name),
        train_size=cfg.training.val_test_size,
        random_state=cfg.training.seed,
    )
    val, test = train_test_split(
        val_test,
        train_size=cfg.training.test_size,
        random_state=cfg.training.seed,
    )
    data_module = CreateDataModule(
        train,
        val,
        test,
        # cfg.training.batch_size,
        # cfg.model.max_length,
    )
    data_module.setup()

    # callbackのインスタンス化
    call_backs = make_callbacks(
        cfg.callbacks.patience_min_delta, cfg.callbacks.patience, checkpoint_path
    )

    # modelのインスタンスの作成
    model = TextClassifierModel(
        n_classes=cfg.model.n_classes,
        n_epochs=cfg.training.n_epochs,
    )

    # Trainerの設定
    trainer = pl.Trainer(
        max_epochs=cfg.training.n_epochs,
        devices="auto",
        # progress_bar_refresh_rate=30,
        callbacks=call_backs,
        logger=wandb_logger,
        fast_dev_run=True,
    )
    trainer.fit(model, data_module)

    wandb.finish()


if __name__ == "__main__":
    main()
