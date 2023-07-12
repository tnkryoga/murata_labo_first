import os
import datetime
import pandas as pd
import numpy as np
import hydra
import wandb
from omegaconf import DictConfig
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizerFast as BertTokenizer,
    BertModel,
    AdamW,
    get_linear_schedule_with_warmup,
)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from transformers import BertModel
from transformers import BertJapaneseTokenizer


# Dataset
class CreateDataset(Dataset):  # 文章のtokenize処理を行ってDataLoaderに渡す関数
    TEXT_COLUMN = "sentence"
    LABEL_COLUMNS = Dataset.columns.tolist()[1:]  # ラベル名の取得

    def __init__(self, data, tokenizer, max_token_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_row = self.data.iloc[index]
        text = data_row[self.TEXT_COLUMN]
        labels = data_row[self.LABEL_COLUMNS]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return dict(
            text=text,
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            labels=torch.FloatTensor(labels),
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
        max_token_len=128,
        pretrained_model="cl-tohoku/bert-base-japanese-char-whole-word-masking",
    ):
        super().__init__()
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.max_token_len = max_token_len
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(pretrained_model)

    # train,val,testデータのsetupの定義
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
class MultiLabelClassifier(pl.LightningModule):
    def __init__(
        self,
        n_classes: int,
        n_epochs=None,
        n_training_steps=None,
        n_warmup_steps=None,
        pretrained_model="cl-tohoku/bert-base-japanese-char-whole-word-masking",
    ):
        super().__init__()
        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # モデルの構造
        self.bert = BertModel.from_pretrained(pretrained_model, return_dict=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.n_epochs = n_epochs
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.criterion = nn.BCELoss()

    # 順伝搬
    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        output = self.classifier(output.pooler_output)
        output = torch.sigmoid(output)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output

    # trainのミニバッチに対して行う処理
    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)  # forward?
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.train_step_outputs.append(loss)
        self.train_step_outputs.append(outputs)
        self.train_step_outputs.append(labels)
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.train_step_outputs.append(loss)
        self.train_step_outputs.append(outputs)
        self.train_step_outputs.append(labels)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        self.train_step_outputs.append(loss)
        self.train_step_outputs.append(outputs)
        self.train_step_outputs.append(labels)
        return loss

    # 1epoch終了時に実行/debugに使用
    def on_train_epoch_end(self):
        labels = []
        predictions = []
        for output in self.train_step_outputs:
            for out_labels in output[2].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output[1].detach().cpu():
                predictions.append(out_predictions)
        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)
        """for i, name in enumerate(LABEL_COLUMNS):
            class_roc_auc = auroc(predictions[:, i], labels[:, i])
            self.logger.experiment.add_scalar(
                f"{name}_roc_auc/Train", class_roc_auc, self.current_epoch
            )"""
        self.train_step_outputs.clear()

    def on_validation_epoch_end(self):
        labels = []
        predictions = []
        for output in self.train_step_outputs:
            for out_labels in output[2].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output[1].detach().cpu():
                predictions.append(out_predictions)
        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)
        """for i, name in enumerate(LABEL_COLUMNS):
            class_roc_auc = auroc(predictions[:, i], labels[:, i])
            self.logger.experiment.add_scalar(
                f"{name}_roc_auc/Train", class_roc_auc, self.current_epoch
            )"""
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        labels = []
        predictions = []
        for output in self.train_step_outputs:
            for out_labels in output[2].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output[1].detach().cpu():
                predictions.append(out_predictions)
        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)
        """for i, name in enumerate(LABEL_COLUMNS):
            class_roc_auc = auroc(predictions[:, i], labels[:, i])
            self.logger.experiment.add_scalar(
                f"{name}_roc_auc/Train", class_roc_auc, self.current_epoch
            )"""
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps,  # 正規化の際の学習率の調整に使用
            num_training_steps=self.n_training_steps,
        )
        return dict(
            optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval="step")
        )


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
        project=cfg.wandb.project,
        name=("exp_" + str(cfg.wandb.exp_num)),
        tags=cfg.wandb.tags,
    )

    wandb_logger = WandbLogger(
        log_model=False,
    )
    # wandb_logger.watch(model, log="all")

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

    # train_step,warmup_stepの計算
    steps_per_epoch = len(train) // cfg.training.batch_size
    total_training_steps = steps_per_epoch * cfg.training.n_epochs
    warmup_steps = total_training_steps // 5

    # modelのインスタンスの作成
    model = MultiLabelClassifier(
        n_classes=cfg.model.n_classes,
        n_epochs=cfg.training.n_epochs,
        n_training_steps=total_training_steps,
        n_warmup_steps=warmup_steps,
    )

    # Trainerの設定
    trainer = pl.Trainer(
        max_epochs=cfg.training.n_epochs,
        devices="auto",
        # progress_bar_refresh_rate=30,
        callbacks=call_backs,
        logger=wandb_logger,
        fast_dev_run=False,
    )
    trainer.fit(model, data_module)

    wandb.finish()


if __name__ == "__main__":
    main()
