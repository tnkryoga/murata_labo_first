import os
import datetime
import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchmetrics
import torch.optim as optim
from omegaconf import DictConfig
import wandb
from wandb import plot
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel
from transformers import BertJapaneseTokenizer


# Dataset
class CreateDataset(Dataset):  # 文章のtokenize処理を行ってDataLoaderに渡す関数
    TEXT_COLUMN = "chunk"
    LABEL_COLUMN = "binary"

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
        batch_size,
        max_token_len,
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
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.vaild_dataset,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            pin_memory=True,
        )


# Model
class BinaryClassifierModel(pl.LightningModule):
    THRESHOLD = 0.5  # 閾値

    def __init__(
        self,
        hidden_size,
        n_epochs=None,
        pretrained_model="cl-tohoku/bert-base-japanese-char-whole-word-masking",
    ):
        super().__init__()
        self.train_step_outputs_preds = []
        self.train_step_outputs_labels = []
        self.validation_step_outputs_preds = []
        self.validation_step_outputs_labels = []
        self.test_step_outputs_preds = []
        self.test_step_outputs_labels = []

        # モデルの構造
        self.bert = BertModel.from_pretrained(pretrained_model, return_dict=True)
        self.hidden_layer = nn.Linear(
            self.bert.config.hidden_size, hidden_size
        )  # 入力BERT層、出力hidden_sizeの全結合層
        self.layer = nn.Linear(hidden_size, 1)  # 二値分類
        self.n_epochs = n_epochs
        self.criterion = nn.BCELoss()
        self.metrics = torchmetrics.MetricCollection(
            [
                torchmetrics.Accuracy(task="binary", threshold=self.THRESHOLD),
                torchmetrics.Precision(task="binary", threshold=self.THRESHOLD),
                torchmetrics.Recall(task="binary", threshold=self.THRESHOLD),
                torchmetrics.F1Score(task="binary", threshold=self.THRESHOLD),
                torchmetrics.MatthewsCorrCoef(task="binary", threshold=self.THRESHOLD),
            ]
        )

        # BertLayerモジュールの最後を勾配計算ありに変更
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.encoder.layer[-1].parameters():
            param.requires_grad = True

    # 順伝搬
    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        outputs = torch.relu(self.hidden_layer(output.pooler_output))  # 活性化関数Relu
        preds = torch.sigmoid(self.layer(outputs))  # sigmoidによる確率化
        loss = 0
        if labels is not None:
            loss = self.criterion(
                preds.view(-1), labels.float()
            )  # predsの次元数を2次元から１次元にする/labelsをfloat型に変更する
        return loss, preds

    # trainのミニバッチに対して行う処理
    def training_step(self, batch, batch_idx):
        loss, preds = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        self.train_step_outputs_preds.append(preds)
        self.train_step_outputs_labels.append(batch["labels"])
        self.log("train/loss", loss, on_step=True, prog_bar=True)
        return {"loss": loss, "batch_preds": preds, "batch_labels": batch["labels"]}

    # validation、testでもtrain_stepと同じ処理を行う
    def validation_step(self, batch, batch_idx):
        loss, preds = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        self.validation_step_outputs_preds.append(preds)
        self.validation_step_outputs_labels.append(batch["labels"])
        # self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return {"loss": loss, "batch_preds": preds, "batch_labels": batch["labels"]}

    def test_step(self, batch, batch_idx):
        loss, preds = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        self.test_step_outputs_preds.append(preds)
        self.test_step_outputs_labels.append(batch["labels"])
        # self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        return {"loss": loss, "batch_preds": preds, "batch_labels": batch["labels"]}

    # epoch終了時にtrainのlossを記録
    def on_train_epoch_end(self, mode="train"):
        epoch_preds = torch.cat(self.train_step_outputs_preds)
        epoch_preds = epoch_preds.squeeze()
        epoch_labels = torch.cat(self.train_step_outputs_labels)
        epoch_labels = epoch_labels.squeeze()
        epoch_loss = self.criterion(epoch_preds, epoch_labels.float())
        self.log(f"{mode}_loss", epoch_loss, logger=True)

        metrics = self.metrics(epoch_preds, epoch_labels)
        for metric in metrics.keys():
            self.log(f"{mode}/{metric.lower()}", metrics[metric].item(), logger=True)

        self.train_step_outputs_preds.clear()  # free memory
        self.train_step_outputs_labels.clear()  # free memory

    # epoch終了時にvalidationのlossとaccuracyを記録
    def on_validation_epoch_end(
        self, mode="val"
    ):  # https://github.com/Lightning-AI/lightning/pull/16520
        # loss計算
        epoch_preds = torch.cat(self.validation_step_outputs_preds)
        epoch_preds = epoch_preds.squeeze()
        epoch_labels = torch.cat(self.validation_step_outputs_labels)
        epoch_labels = epoch_labels.squeeze()
        epoch_loss = self.criterion(epoch_preds, epoch_labels.float())
        self.log(f"{mode}_loss", epoch_loss, logger=True)

        metrics = self.metrics(epoch_preds, epoch_labels)
        for metric in metrics.keys():
            self.log(f"{mode}/{metric.lower()}", metrics[metric].item(), logger=True)

        self.validation_step_outputs_preds.clear()  # free memory
        self.validation_step_outputs_labels.clear()  # free memory

    # testデータのlossとaccuracyを算出
    def on_test_epoch_end(self, mode="test"):
        epoch_preds = torch.cat(self.test_step_outputs_preds)
        epoch_preds = epoch_preds.squeeze()
        epoch_labels = torch.cat(self.test_step_outputs_labels)
        epoch_labels = epoch_labels.squeeze()
        epoch_loss = self.criterion(epoch_preds, epoch_labels.float())
        self.log(f"{mode}_loss", epoch_loss, logger=True)

        print("実行済み")

        epoch_preds, epoch_labels = (
            epoch_preds.cpu().numpy(),  # cpu上に移動し、numpy配列に変換
            epoch_labels.cpu().numpy(),
        )
        preds_binary = np.where(epoch_preds > self.THRESHOLD, 1, 0)

        # 混同行列
        wandb.log(
            {
                "test/confusion_matrix": plot.confusion_matrix(
                    probs=None,
                    y_true=epoch_labels,
                    preds=preds_binary,
                    class_names=["応答なし", "応答あり"],
                ),
            }
        )

        # PR曲線
        wandb.log(
            {
                "test/pr": plot.pr_curve(
                    y_true=epoch_labels,
                    y_probas=self.score_to_complement_pairs(epoch_preds),
                    labels=["応答なし", "応答あり"],
                ),
            }
        )

        # ROC曲線
        wandb.log(
            {
                "test/lf/roc": plot.roc_curve(
                    y_true=epoch_labels,
                    y_probas=self.score_to_complement_pairs(epoch_preds),
                    labels=["応答なし", "応答あり"],
                ),
            }
        )

        self.test_step_outputs_preds.clear()
        self.test_step_outputs_labels.clear()  # free memory

    # optimizerの設定
    def configure_optimizers(self):
        # pretrainされているbert最終層のlrは小さめ、pretrainされていない分類層のlrは大きめに設定
        optimizer = optim.Adam(
            [
                {"params": self.bert.encoder.layer[-1].parameters(), "lr": 5e-5},
                {"params": self.layer.parameters(), "lr": 1e-4},
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
        monitor="train_loss",
        mode="min",
    )

    early_stop_callback = EarlyStopping(
        monitor="train_loss", min_delta=min_delta, patience=patience, mode="min"
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
        cfg.training.batch_size,
        cfg.model.max_length,
    )
    data_module.setup()

    # callbackのインスタンス化
    call_backs = make_callbacks(
        cfg.callbacks.patience_min_delta, cfg.callbacks.patience, checkpoint_path
    )

    # modelのインスタンスの作成
    model = BinaryClassifierModel(
        hidden_size=cfg.model.hidden_size,
        n_epochs=cfg.training.n_epochs,
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
